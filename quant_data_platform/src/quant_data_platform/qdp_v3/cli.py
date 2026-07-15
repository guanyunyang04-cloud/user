from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any, Callable

from quant_data_platform.core.json_io import json_safe, read_json
from quant_data_platform.qdp_v3.audit import audit_candidate
from quant_data_platform.qdp_v3.build import build_candidate
from quant_data_platform.qdp_v3.compatibility import capture_baostock_0_9_1_golden, run_baostock_0_9_3_compatibility_gate
from quant_data_platform.qdp_v3.compaction import compact_raw_domain
from quant_data_platform.qdp_v3.corporate_actions import ingest_mootdx_xdxr
from quant_data_platform.qdp_v3.external_quant_5m import import_external_quant_5m
from quant_data_platform.qdp_v3.freeze import freeze_v2, validate_v2_freeze_proof
from quant_data_platform.qdp_v3.gc import collect_garbage
from quant_data_platform.qdp_v3.historical import (
    ingest_tushare_proxy_intraday,
    ingest_tushare_proxy_reference,
    proxy_symbol_inventory,
)
from quant_data_platform.qdp_v3.ingest import (
    ingest_baostock_date_partitions,
    ingest_factor_symbol_histories,
    ingest_security_master,
    ingest_trading_calendar,
    resolve_trade_dates,
)
from quant_data_platform.qdp_v3.intraday import ingest_intraday_5m
from quant_data_platform.qdp_v3.manifest import active_manifest_sha256, dataset_manifest_for_id, manifest_sha256, read_dataset_manifest
from quant_data_platform.qdp_v3.mootdx_compatibility import run_mootdx_5m_compatibility_gate
from quant_data_platform.qdp_v3.paths import qdp_v3_paths
from quant_data_platform.qdp_v3.proxy_compatibility import lock_bootstrap_cutoff, run_tushare_proxy_compatibility_gate
from quant_data_platform.qdp_v3.release import (
    diff_candidate,
    publish_candidate,
    read_candidate,
    rollback_active,
    validate_published_active,
)
from quant_data_platform.qdp_v3.retirement import retire_v2_intraday
from quant_data_platform.qdp_v3.secondary import ingest_baostock_report_domain, ingest_baostock_snapshot_domain
from quant_data_platform.qdp_v3.update import plan_update, run_update


HELP_TEXT = """usage: qdp [--workspace-root ROOT] [--generation v3] <command> [options]

QDP v3 manifest-first data-base commands:
  status                         Show v3 active/candidate/raw state.
  list                           List v3 active or candidate datasets.
  describe <domain|dataset-id>   Read one v3 dataset manifest.
  check [--candidate ID]         Audit active by default, or one candidate.
  ingest --provider baostock --mode date-snapshot|date-events
                                 Fetch resumable date partitions.
  ingest --provider mootdx --mode corporate-actions
                                 Preserve per-security TDX xdxr evidence.
  ingest --provider tushare-proxy --mode historical --domain DOMAIN
                                 Capture resumable 2010+ historical raw facts.
  ingest --provider external-quant-archive --mode historical --domain intraday
                                 Import the user-supplied direct 5m archive.
  build candidate                Build immutable canonical datasets and a candidate.
  compact --raw-domain DOMAIN    Bundle verified raw partitions into large Parquet files.
  audit --candidate ID           Run quick, full, or semantic candidate gates.
  diff --candidate ID            Compare a candidate with v3 active.
  publish --candidate ID --expect-active-sha SHA
                                 Compare-and-swap publish after semantic audit.
  rollback --expect-active-sha SHA
                                 Restore the latest rollback manifest.
  retire-v2-intraday             Retire v2 minute shards after verified v3 publication.
  update                         Execute ingest -> build -> audit -> CAS publish.
  gc                             Traverse active/candidate/pin/rollback/audit lineage.
  freeze v2                      Freeze and pin the legacy v2 evidence base.
  compatibility capture-golden  Save 0.9.1 per-symbol golden fixtures.
  compatibility run             Prove 0.9.3 batch/legacy compatibility.
  compatibility run --provider tushare-proxy
                                 Prove proxy entitlement, APIs and reverse pagination.
  compatibility run --provider mootdx
                                 Prove dynamic-node discovery and recent complete 5m bars.

Human-readable output is the default. Use --json for stable machine output.
"""


def _common(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--workspace-root", default="")
    parser.add_argument("--json", action="store_true")


def _workspace(args: argparse.Namespace) -> str | None:
    return str(getattr(args, "workspace_root", "") or "") or None


def _emit(payload: dict[str, Any], *, as_json: bool) -> None:
    if as_json:
        print(json.dumps(json_safe(payload), ensure_ascii=False, indent=2))
        return
    for key, value in payload.items():
        if isinstance(value, (dict, list)):
            print(f"{key}: {json.dumps(json_safe(value), ensure_ascii=False)}")
        else:
            print(f"{key}: {value}")


def _status(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(prog="qdp status", description="Show QDP v3 state without scanning data payloads.")
    _common(parser)
    parser.add_argument("--run", default="", help="Show one resumable ingest/update job.")
    args = parser.parse_args(argv)
    paths = qdp_v3_paths(_workspace(args))
    active_validation = validate_published_active(_workspace(args))
    active = dict(active_validation["active"]) if active_validation["valid"] else {}
    candidates = sorted(paths.candidates.glob("*.json")) if paths.candidates.exists() else []
    raw_domains = sorted(path.name for path in paths.raw.iterdir() if path.is_dir()) if paths.raw.exists() else []
    freeze = read_json(paths.metadata / "v2_freeze_20260713.json")
    payload = {
        "status": "active" if active else "not_published",
        "generation": "v3",
        "root": str(paths.root.resolve()),
        "active_sha256": active_manifest_sha256(_workspace(args)),
        "active": active,
        "candidate_count": len(candidates),
        "latest_candidate_id": candidates[-1].stem if candidates else "",
        "raw_domains": raw_domains,
        "v2_freeze": {
            "status": freeze.get("status", "missing"),
            **validate_v2_freeze_proof(freeze),
        },
    }
    if not active_validation["valid"] and paths.active_manifest.exists():
        payload["invalid_active"] = {
            "path": str(paths.active_manifest.resolve()),
            "sha256": active_manifest_sha256(_workspace(args)),
            "blockers": list(active_validation["blockers"]),
        }
    if args.run:
        run_path = paths.jobs / f"{str(args.run)}.json"
        run_state = read_json(run_path)
        if not run_state:
            raise FileNotFoundError(f"qdp_v3_run_not_found:{args.run}")
        payload["run"] = run_state
        payload["run_path"] = str(run_path.resolve())
    _emit(payload, as_json=bool(args.json))
    return 0


def _list(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(prog="qdp list", description="List QDP v3 active or candidate datasets.")
    _common(parser)
    parser.add_argument("--candidate", default="")
    args = parser.parse_args(argv)
    paths = qdp_v3_paths(_workspace(args))
    if args.candidate:
        candidate = read_candidate(str(args.candidate), workspace_root=_workspace(args))
        datasets = candidate.datasets
        source = candidate.candidate_id
    else:
        validation = validate_published_active(_workspace(args))
        if not validation["valid"]:
            raise RuntimeError("qdp_v3_active_missing_or_invalid")
        active = dict(validation["active"])
        datasets = dict(active.get("datasets", {}) or {})
        source = "active"
    rows: list[dict[str, Any]] = []
    for domain, dataset_id in sorted(datasets.items()):
        path = dataset_manifest_for_id(paths.root, str(dataset_id), str(domain))
        if path is None:
            rows.append({"domain": domain, "dataset_id": dataset_id, "status": "manifest_missing"})
            continue
        manifest = read_dataset_manifest(path)
        rows.append({"domain": domain, "dataset_id": dataset_id, "quality_tier": manifest.quality_tier, "start_date": manifest.start_date, "end_date": manifest.end_date, "row_count": manifest.row_count, "manifest_sha256": manifest_sha256(path)})
    _emit({"status": "ok", "source": source, "dataset_count": len(rows), "datasets": rows}, as_json=bool(args.json))
    return 0


def _describe(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(prog="qdp describe", description="Describe one QDP v3 dataset by domain or stable dataset id.")
    _common(parser)
    parser.add_argument("target")
    parser.add_argument("--candidate", default="")
    args = parser.parse_args(argv)
    paths = qdp_v3_paths(_workspace(args))
    target = str(args.target)
    domain_hint = ""
    dataset_id = target
    if args.candidate:
        candidate = read_candidate(str(args.candidate), workspace_root=_workspace(args))
        if target in candidate.datasets:
            domain_hint = target
            dataset_id = candidate.datasets[target]
    else:
        validation = validate_published_active(_workspace(args))
        if not validation["valid"]:
            raise RuntimeError("qdp_v3_active_missing_or_invalid")
        active = dict(validation["active"])
        active_datasets = dict(active.get("datasets", {}) or {})
        if target in active_datasets:
            domain_hint = target
            dataset_id = str(active_datasets[target])
    path = dataset_manifest_for_id(paths.root, dataset_id, domain_hint)
    if path is None:
        raise FileNotFoundError(f"qdp_v3_dataset_not_found:{target}")
    payload = read_json(path)
    payload["manifest_sha256"] = manifest_sha256(path)
    payload["manifest_path"] = str(path.resolve())
    _emit(payload, as_json=bool(args.json))
    return 0


def _audit(argv: list[str], *, prog: str = "qdp audit") -> int:
    parser = argparse.ArgumentParser(prog=prog, description="Audit a QDP v3 candidate; semantic is the publish gate.")
    _common(parser)
    parser.add_argument("--candidate", required=True)
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument("--quick", action="store_true")
    modes.add_argument("--full", action="store_true")
    modes.add_argument("--semantic", action="store_true")
    parser.add_argument("--no-write-report", action="store_true")
    args = parser.parse_args(argv)
    mode = "semantic" if args.semantic else "full" if args.full else "quick"
    payload = audit_candidate(candidate_id=str(args.candidate), mode=mode, workspace_root=_workspace(args), write_report=not bool(args.no_write_report))
    _emit(payload, as_json=bool(args.json))
    return 0 if payload.get("status") == "passed" else 2


def _check(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(
        prog="qdp check",
        description="Audit the QDP v3 active data base, or an explicitly selected candidate.",
    )
    _common(parser)
    parser.add_argument("--candidate", default="")
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument("--quick", action="store_true")
    modes.add_argument("--full", action="store_true")
    modes.add_argument("--semantic", action="store_true")
    parser.add_argument("--no-write-report", action="store_true")
    args = parser.parse_args(argv)
    workspace = _workspace(args)
    target = "candidate"
    candidate_id = str(args.candidate or "")
    if not candidate_id:
        validation = validate_published_active(workspace)
        if not validation["valid"]:
            raise RuntimeError("qdp_v3_active_missing_or_invalid")
        active = dict(validation["active"])
        candidate_id = str(active.get("candidate_id", "") or "")
        if not candidate_id:
            raise RuntimeError("qdp_v3_active_candidate_id_missing")
        target = "active"
    mode = "semantic" if args.semantic else "full" if args.full else "quick"
    payload = audit_candidate(
        candidate_id=candidate_id,
        mode=mode,
        workspace_root=workspace,
        write_report=not bool(args.no_write_report),
    )
    payload["target"] = target
    if target == "active":
        payload["active_sha256"] = active_manifest_sha256(workspace)
    _emit(payload, as_json=bool(args.json))
    return 0 if payload.get("status") == "passed" else 2


def _ingest(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(prog="qdp ingest", description="Ingest immutable, resumable QDP v3 provider partitions.")
    _common(parser)
    parser.add_argument(
        "--provider",
        default="baostock",
        choices=("baostock", "mootdx", "tushare-proxy", "external-quant-archive"),
    )
    parser.add_argument("--mode", required=True, choices=("historical", "date-snapshot", "date-events", "factor-symbol-history", "calendar", "security-master", "intraday-5m", "corporate-actions", "financial-quarterly", "performance-forecast", "performance-express", "industry", "index-constituents"))
    parser.add_argument("--domain", default="", choices=("", "stock-basic", "trade-calendar", "daily", "daily-basic", "factor", "identity", "status", "dividend", "financial", "intraday"))
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--trade-date", action="append", default=[])
    parser.add_argument("--start-date", default="")
    parser.add_argument("--end-date", default="")
    parser.add_argument("--universe-kind", default="all_a", choices=("all_a", "etf"))
    parser.add_argument("--refresh", action="store_true")
    parser.add_argument(
        "--max-workers",
        type=int,
        choices=(1, 2, 3, 4),
        default=4,
        help=(
            "Provider worker count; local CPU-bound imports default to four, "
            "Tushare clamps to three, and BaoStock date ingestion caps internally."
        ),
    )
    parser.add_argument(
        "--source-path",
        action="append",
        default=[],
        help="Explicit local archive ZIP or 5m CSV directory; repeat for multiple non-overlapping source segments.",
    )
    parser.add_argument("--max-symbols", type=int, default=0, help="Limit local archive symbols for a smoke/benchmark run; zero means all.")
    parser.add_argument("--job-id", default="")
    parser.add_argument("--symbol", action="append", default=[])
    parser.add_argument("--symbols-file", default="")
    args = parser.parse_args(argv)
    workspace = _workspace(args)
    if args.mode == "historical":
        if not args.domain or not args.start_date or not args.end_date:
            parser.error("historical mode requires --domain, --start-date, and --end-date")
        if args.provider == "external-quant-archive":
            if args.domain != "intraday":
                parser.error("external-quant-archive historical mode only supports --domain intraday")
            if not args.source_path:
                parser.error("external-quant-archive historical mode requires at least one --source-path")
            symbols = list(args.symbol)
            if args.symbols_file:
                symbols.extend(
                    item.strip()
                    for item in Path(args.symbols_file).read_text(encoding="utf-8").splitlines()
                    if item.strip()
                )
            if not symbols:
                symbols, _ = proxy_symbol_inventory(workspace_root=workspace, mainboard_only=True)
            if not symbols:
                parser.error("external-quant-archive requires proxy stock-basic raw or --symbol/--symbols-file")
            result = import_external_quant_5m(
                source_paths=tuple(args.source_path),
                workspace_root=workspace,
                symbols=tuple(symbols),
                start_date=str(args.start_date),
                end_date=str(args.end_date),
                max_symbols=max(0, int(args.max_symbols)),
                workers=int(args.max_workers),
            )
            payload = result.to_dict() if hasattr(result, "to_dict") else asdict(result)
            _emit(payload, as_json=bool(args.json))
            return 0 if payload.get("status") == "completed" else 2
        if args.provider != "tushare-proxy":
            parser.error("historical mode requires --provider tushare-proxy or external-quant-archive")
        symbols = list(args.symbol)
        if args.symbols_file:
            symbols.extend(item.strip() for item in Path(args.symbols_file).read_text(encoding="utf-8").splitlines() if item.strip())
        inventory_symbols, lifecycle = proxy_symbol_inventory(
            workspace_root=workspace,
            mainboard_only=args.domain == "intraday",
        )
        if not symbols:
            symbols = inventory_symbols
        dates = sorted(set(str(item)[:10] for item in args.trade_date if str(item).strip()))
        if not dates and args.domain in {"daily", "daily-basic", "status"}:
            dates = resolve_trade_dates(
                start_date=str(args.start_date),
                end_date=str(args.end_date),
                workspace_root=workspace,
            )
        if args.domain == "intraday":
            if not symbols:
                parser.error("historical intraday requires proxy stock-basic raw or --symbol/--symbols-file")
            cutoff = lock_bootstrap_cutoff(
                requested_cutoff=str(args.end_date),
                workspace_root=workspace,
            )
            payload = ingest_tushare_proxy_intraday(
                symbols=symbols,
                start_date=str(args.start_date),
                end_date=str(cutoff["bootstrap_cutoff"]),
                workspace_root=workspace,
                lifecycle_ranges=lifecycle,
                resume=bool(args.resume),
                job_id=str(args.job_id),
                max_workers=int(args.max_workers),
            )
        else:
            if args.domain in {"factor", "identity", "dividend", "financial"} and not symbols:
                parser.error(f"historical {args.domain} requires proxy stock-basic raw or --symbol/--symbols-file")
            payload = ingest_tushare_proxy_reference(
                domain=str(args.domain),
                start_date=str(args.start_date),
                end_date=str(args.end_date),
                trade_dates=dates,
                symbols=symbols,
                workspace_root=workspace,
                resume=bool(args.resume),
                job_id=str(args.job_id),
                max_workers=int(args.max_workers),
            )
    elif args.mode in {"financial-quarterly", "performance-forecast", "performance-express"}:
        if args.provider != "baostock":
            parser.error("report-domain modes require --provider baostock")
        symbols = list(args.symbol)
        if args.symbols_file:
            symbols.extend(item.strip() for item in Path(args.symbols_file).read_text(encoding="utf-8").splitlines() if item.strip())
        if not symbols or not args.start_date or not args.end_date:
            parser.error("report-domain modes require --symbol/--symbols-file, --start-date, and --end-date")
        payload = ingest_baostock_report_domain(
            domain=str(args.mode).replace("-", "_"),
            symbols=symbols,
            start_date=str(args.start_date),
            end_date=str(args.end_date),
            workspace_root=workspace,
            refresh=bool(args.refresh),
            job_id=str(args.job_id),
        )
    elif args.mode in {"industry", "index-constituents"}:
        if args.provider != "baostock":
            parser.error("snapshot-domain modes require --provider baostock")
        as_of_date = str(args.end_date or (args.trade_date[-1] if args.trade_date else ""))
        if not as_of_date:
            parser.error("snapshot-domain modes require --end-date or --trade-date")
        payload = ingest_baostock_snapshot_domain(
            domain="industry_concept" if args.mode == "industry" else "index_constituents",
            as_of_date=as_of_date,
            workspace_root=workspace,
            refresh=bool(args.refresh),
        )
    elif args.mode == "corporate-actions":
        if args.provider != "mootdx":
            parser.error("corporate-actions mode requires --provider mootdx")
        symbols = list(args.symbol)
        if args.symbols_file:
            symbols.extend(item.strip() for item in Path(args.symbols_file).read_text(encoding="utf-8").splitlines() if item.strip())
        if not symbols:
            parser.error("corporate-actions mode requires --symbol or --symbols-file")
        payload = ingest_mootdx_xdxr(
            symbols=symbols,
            workspace_root=workspace,
            refresh=bool(args.refresh),
            job_id=str(args.job_id),
        )
    elif args.mode == "calendar":
        if args.provider != "baostock":
            parser.error("calendar mode requires --provider baostock")
        if not args.start_date or not args.end_date:
            parser.error("calendar mode requires --start-date and --end-date")
        frame, ref = ingest_trading_calendar(start_date=str(args.start_date), end_date=str(args.end_date), workspace_root=workspace, refresh=bool(args.refresh))
        payload = {"status": "completed", "mode": args.mode, "row_count": len(frame), "content_sha256": ref.content_sha256, "payload_path": str(ref.payload_path.resolve())}
    elif args.mode == "security-master":
        if args.provider != "baostock":
            parser.error("security-master mode requires --provider baostock")
        as_of_date = str(args.end_date or (args.trade_date[-1] if args.trade_date else ""))
        if not as_of_date:
            parser.error("security-master mode requires --end-date or --trade-date")
        ref = ingest_security_master(as_of_date=as_of_date, workspace_root=workspace, refresh=bool(args.refresh))
        payload = {"status": "completed", "mode": args.mode, "row_count": ref.row_count, "content_sha256": ref.content_sha256, "payload_path": str(ref.payload_path.resolve())}
    elif args.mode == "factor-symbol-history":
        if args.provider != "baostock":
            parser.error("factor-symbol-history mode requires --provider baostock")
        symbols = list(args.symbol)
        if args.symbols_file:
            symbols.extend(item.strip() for item in Path(args.symbols_file).read_text(encoding="utf-8").splitlines() if item.strip())
        if not symbols or not args.end_date:
            parser.error("factor-symbol-history mode requires --symbol/--symbols-file and --end-date")
        payload = ingest_factor_symbol_histories(
            symbols=symbols,
            start_date=str(args.start_date or "1990-01-01"),
            end_date=str(args.end_date),
            workspace_root=workspace,
            refresh=bool(args.refresh),
            job_id=str(args.job_id),
        )
    else:
        dates = sorted(set(str(item)[:10] for item in args.trade_date if str(item).strip()))
        if not dates:
            if not args.start_date or not args.end_date:
                parser.error("mode requires --trade-date or both --start-date and --end-date")
            dates = resolve_trade_dates(start_date=str(args.start_date), end_date=str(args.end_date), workspace_root=workspace)
        if args.mode == "intraday-5m":
            symbols = list(args.symbol)
            if args.symbols_file:
                symbols.extend(item.strip() for item in Path(args.symbols_file).read_text(encoding="utf-8").splitlines() if item.strip())
            if not symbols:
                parser.error("intraday-5m mode requires --symbol or --symbols-file")
            payload = ingest_intraday_5m(
                symbols=symbols,
                trade_dates=dates,
                workspace_root=workspace,
                refresh=bool(args.refresh),
            )
        else:
            if args.provider != "baostock":
                parser.error("date-snapshot and date-events require --provider baostock")
            payload = ingest_baostock_date_partitions(
                mode=str(args.mode),
                trade_dates=dates,
                universe_kind=str(args.universe_kind),
                workspace_root=workspace,
                refresh=bool(args.refresh),
                cross_check=False,
                job_id=str(args.job_id),
                max_workers=min(int(args.max_workers), 2),
            )
    _emit(payload, as_json=bool(args.json))
    return 0 if payload.get("status") in {"completed", "strict"} else 2


def _build(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(prog="qdp build candidate", description="Build immutable QDP v3 canonical datasets from raw partitions.")
    _common(parser)
    parser.add_argument("--start-date", default="2010-01-01")
    parser.add_argument("--end-date", default="")
    parser.add_argument("--identity-config", default="")
    parser.add_argument("--full", action="store_true", help="Emit the full candidate manifest including raw partition references.")
    args = parser.parse_args(argv)
    candidate = build_candidate(
        workspace_root=_workspace(args),
        start_date=str(args.start_date),
        end_date=str(args.end_date),
        identity_config=str(args.identity_config) or None,
        require_factor_dual_path=False,
    )
    payload = candidate.to_dict() if args.full else {
        "status": candidate.status,
        "candidate_id": candidate.candidate_id,
        "created_at": candidate.created_at,
        "dataset_count": len(candidate.datasets),
        "datasets": candidate.datasets,
        "quality_tiers": candidate.quality_tiers,
        "blocker_count": len(candidate.blockers),
        "blocker_codes": sorted({str(item.get("code", "")) for item in candidate.blockers}),
        "coverage": candidate.coverage,
        "quarantine_count": len(candidate.quarantine),
        "raw_partition_count": len(candidate.raw_partitions),
    }
    payload["candidate_path"] = str((qdp_v3_paths(_workspace(args)).candidates / f"{candidate.candidate_id}.json").resolve())
    _emit(payload, as_json=bool(args.json))
    return 0


def _diff(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(prog="qdp diff", description="Diff a QDP v3 candidate against active.")
    _common(parser)
    parser.add_argument("--candidate", required=True)
    parser.add_argument("--against", choices=("active",), default="active")
    args = parser.parse_args(argv)
    payload = diff_candidate(str(args.candidate), workspace_root=_workspace(args), against=str(args.against))
    _emit(payload, as_json=bool(args.json))
    return 0


def _publish(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(prog="qdp publish", description="CAS-publish a fully audited QDP v3 candidate.")
    _common(parser)
    parser.add_argument("--candidate", required=True)
    parser.add_argument("--expect-active-sha", required=True)
    args = parser.parse_args(argv)
    payload = publish_candidate(candidate_id=str(args.candidate), expect_active_sha=str(args.expect_active_sha), workspace_root=_workspace(args))
    _emit(payload, as_json=bool(args.json))
    return 0 if payload.get("status") == "published" else 2


def _rollback(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(prog="qdp rollback", description="CAS-restore the latest QDP v3 rollback manifest.")
    _common(parser)
    parser.add_argument("--expect-active-sha", required=True)
    args = parser.parse_args(argv)
    payload = rollback_active(expect_active_sha=str(args.expect_active_sha), workspace_root=_workspace(args))
    _emit(payload, as_json=bool(args.json))
    return 0


def _gc(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(prog="qdp gc", description="Transitive manifest-aware QDP v3 garbage collection.")
    _common(parser)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--apply", action="store_true")
    parser.add_argument("--yes", action="store_true")
    parser.add_argument("--min-age-days", type=int, default=30)
    parser.add_argument("--purge-trash", action="store_true")
    parser.add_argument("--trash-retention-days", type=int, default=7)
    args = parser.parse_args(argv)
    payload = collect_garbage(workspace_root=_workspace(args), apply=bool(args.apply), yes=bool(args.yes), min_age_days=int(args.min_age_days), purge_trash=bool(args.purge_trash), trash_retention_days=int(args.trash_retention_days))
    _emit(payload, as_json=bool(args.json))
    return 0


def _compact(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(
        prog="qdp compact",
        description="Verify and compact legacy immutable raw partitions into large Parquet bundles.",
    )
    _common(parser)
    parser.add_argument("--raw-domain", action="append", required=True)
    parser.add_argument("--export-index-dir", default="")
    parser.add_argument("--delete-source", action="store_true")
    parser.add_argument("--yes", action="store_true")
    args = parser.parse_args(argv)
    if args.delete_source and not args.yes:
        raise ValueError("raw_compaction_source_deletion_requires_yes")
    workspace = _workspace(args)
    export_root = Path(args.export_index_dir) if args.export_index_dir else qdp_v3_paths(workspace).metadata / "raw_index"
    results: list[dict[str, Any]] = []
    for raw_domain in sorted(set(str(item) for item in args.raw_domain if str(item))):
        result = compact_raw_domain(
            raw_domain,
            workspace_root=workspace,
            export_index_dir=export_root,
            delete_sources=bool(args.delete_source),
            yes=bool(args.yes),
        )
        results.append(asdict(result))
    payload = {
        "status": "completed",
        "raw_domain_count": len(results),
        "results": results,
        "sources_deleted": bool(args.delete_source),
    }
    _emit(payload, as_json=bool(args.json))
    return 0


def _retire_v2_intraday(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(
        prog="qdp retire-v2-intraday",
        description="Physically retire the v2 intraday chain after all v3 release guards pass.",
    )
    _common(parser)
    parser.add_argument("--expect-active-sha", required=True)
    parser.add_argument("--delete", action="store_true")
    parser.add_argument("--yes", action="store_true")
    args = parser.parse_args(argv)
    payload = retire_v2_intraday(
        expect_active_sha=str(args.expect_active_sha),
        workspace_root=_workspace(args),
        delete=bool(args.delete),
        yes=bool(args.yes),
    )
    _emit(payload, as_json=bool(args.json))
    return 0 if payload.get("status") in {"ready_to_delete", "retired"} else 2


def _freeze(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(prog="qdp freeze v2", description="Hash and pin the QDP v2 active/minute evidence base.")
    _common(parser)
    parser.add_argument("--metadata-only", action="store_true", help="Pin immediately without reading every shard byte; full hashing remains required before release.")
    parser.add_argument("--expect-active-sha", default="e56f72a6cba8bcf86055817f6a0ec5e7391271fb3c27b4d628c3abc62944051e")
    parser.add_argument("--accept-missing-lineage", action="store_true", help="Use an explicitly authorized substitute proof when ancestor manifests are irrecoverable.")
    parser.add_argument("--authorization-note", default="", help="Required audit note for --accept-missing-lineage.")
    args = parser.parse_args(argv)
    payload = freeze_v2(
        workspace_root=_workspace(args),
        hash_shards=not bool(args.metadata_only),
        expected_active_sha256=str(args.expect_active_sha),
        accept_missing_lineage=bool(args.accept_missing_lineage),
        authorization_note=str(args.authorization_note),
    )
    _emit(payload, as_json=bool(args.json))
    return 0


def _update(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(prog="qdp update", description="Run the QDP v3 ingest-build-audit-publish DAG.")
    _common(parser)
    parser.add_argument("--as-of-date", required=True)
    parser.add_argument("--start-date", default="")
    parser.add_argument("--bootstrap", action="store_true")
    parser.add_argument(
        "--historical-provider",
        default="",
        choices=("", "tushare-proxy", "external-quant-archive"),
    )
    parser.add_argument(
        "--source-path",
        action="append",
        default=[],
        help="Explicit local 5m ZIP or directory used by the external archive bootstrap route.",
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--no-publish", action="store_true")
    parser.add_argument("--hash-v2-shards", action="store_true")
    args = parser.parse_args(argv)
    kwargs = {
        "as_of_date": str(args.as_of_date),
        "workspace_root": _workspace(args),
        "bootstrap": bool(args.bootstrap),
        "start_date": str(args.start_date),
        "historical_provider": str(args.historical_provider),
        "source_paths": tuple(args.source_path),
    }
    if args.dry_run:
        payload = plan_update(**kwargs)
    else:
        payload = run_update(**kwargs, publish=not bool(args.no_publish), hash_v2_shards=bool(args.hash_v2_shards))
    _emit(payload, as_json=bool(args.json))
    return 0 if payload.get("status") in {"planned", "completed"} else 2


def _compatibility_capture(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(prog="qdp compatibility capture-golden", description="Capture BaoStock 0.9.1 golden fixtures before upgrading.")
    _common(parser)
    parser.add_argument("--end-date", required=True)
    parser.add_argument("--start-year", type=int, default=2010)
    parser.add_argument("--sample-count", type=int, default=64)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args(argv)
    payload = capture_baostock_0_9_1_golden(
        end_date=str(args.end_date),
        workspace_root=_workspace(args),
        start_year=int(args.start_year),
        sample_count=int(args.sample_count),
        smoke=bool(args.smoke),
    )
    _emit(payload, as_json=bool(args.json))
    return 0


def _compatibility_run(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(prog="qdp compatibility run", description="Run a provider compatibility and protocol gate.")
    _common(parser)
    parser.add_argument("--provider", default="baostock", choices=("baostock", "mootdx", "tushare-proxy"))
    parser.add_argument("--as-of-date", default="")
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args(argv)
    if args.provider == "tushare-proxy":
        payload = run_tushare_proxy_compatibility_gate(
            workspace_root=_workspace(args),
            as_of_date=str(args.as_of_date),
            smoke=bool(args.smoke),
        )
    elif args.provider == "mootdx":
        payload = run_mootdx_5m_compatibility_gate(
            workspace_root=_workspace(args),
            as_of_date=str(args.as_of_date),
        )
    else:
        payload = run_baostock_0_9_3_compatibility_gate(workspace_root=_workspace(args), smoke=bool(args.smoke))
    _emit(payload, as_json=bool(args.json))
    return 0 if payload.get("status") == "passed" else 2


COMMANDS: dict[tuple[str, ...], Callable[[list[str]], int]] = {
    ("status",): _status,
    ("list",): _list,
    ("describe",): _describe,
    ("check",): _check,
    ("ingest",): _ingest,
    ("compact",): _compact,
    ("build", "candidate"): _build,
    ("audit",): _audit,
    ("diff",): _diff,
    ("publish",): _publish,
    ("rollback",): _rollback,
    ("retire-v2-intraday",): _retire_v2_intraday,
    ("gc",): _gc,
    ("freeze", "v2"): _freeze,
    ("compatibility", "capture-golden"): _compatibility_capture,
    ("compatibility", "run"): _compatibility_run,
    ("update",): _update,
}


def dispatch(argv: list[str]) -> int | None:
    raw = list(argv or [])
    if not raw or raw in (["-h"], ["--help"]):
        print(HELP_TEXT)
        return 0
    for prefix, handler in sorted(COMMANDS.items(), key=lambda item: len(item[0]), reverse=True):
        if tuple(raw[: len(prefix)]) == prefix:
            return int(handler(raw[len(prefix) :]) or 0)
    return None


def main(argv: list[str] | None = None) -> int:
    result = dispatch(list(argv or []))
    if result is None:
        raise ValueError("unsupported_qdp_v3_command")
    return result
