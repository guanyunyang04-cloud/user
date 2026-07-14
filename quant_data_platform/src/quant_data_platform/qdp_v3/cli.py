from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Callable

from quant_data_platform.core.json_io import json_safe, read_json
from quant_data_platform.qdp_v3.audit import audit_candidate
from quant_data_platform.qdp_v3.build import build_candidate
from quant_data_platform.qdp_v3.compatibility import capture_baostock_0_9_1_golden, run_baostock_0_9_3_compatibility_gate
from quant_data_platform.qdp_v3.corporate_actions import ingest_mootdx_xdxr
from quant_data_platform.qdp_v3.freeze import freeze_v2, validate_v2_freeze_proof
from quant_data_platform.qdp_v3.gc import collect_garbage
from quant_data_platform.qdp_v3.ingest import (
    ingest_baostock_date_partitions,
    ingest_factor_symbol_histories,
    ingest_security_master,
    ingest_trading_calendar,
    resolve_trade_dates,
)
from quant_data_platform.qdp_v3.intraday import ingest_intraday_5m
from quant_data_platform.qdp_v3.manifest import active_manifest_sha256, dataset_manifest_for_id, manifest_sha256, read_dataset_manifest
from quant_data_platform.qdp_v3.paths import qdp_v3_paths
from quant_data_platform.qdp_v3.release import diff_candidate, publish_candidate, read_candidate, rollback_active
from quant_data_platform.qdp_v3.secondary import ingest_baostock_report_domain, ingest_baostock_snapshot_domain
from quant_data_platform.qdp_v3.update import plan_update, run_update


HELP_TEXT = """usage: qdp [--workspace-root ROOT] [--generation v3] <command> [options]

QDP v3 manifest-first data-base commands:
  status                         Show v3 active/candidate/raw state.
  list                           List v3 active or candidate datasets.
  describe <domain|dataset-id>   Read one v3 dataset manifest.
  check --candidate ID           Audit a v3 candidate.
  ingest --provider baostock --mode date-snapshot|date-events
                                 Fetch resumable date partitions.
  ingest --provider mootdx --mode corporate-actions
                                 Preserve per-security TDX xdxr evidence.
  build candidate                Build immutable canonical datasets and a candidate.
  audit --candidate ID           Run quick, full, or semantic candidate gates.
  diff --candidate ID            Compare a candidate with v3 active.
  publish --candidate ID --expect-active-sha SHA
                                 Compare-and-swap publish after semantic audit.
  rollback --expect-active-sha SHA
                                 Restore the latest rollback manifest.
  update                         Execute ingest -> build -> audit -> CAS publish.
  gc                             Traverse active/candidate/pin/rollback/audit lineage.
  freeze v2                      Freeze and pin the legacy v2 evidence base.
  compatibility capture-golden  Save 0.9.1 per-symbol golden fixtures.
  compatibility run             Prove 0.9.3 batch/legacy compatibility.

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
    args = parser.parse_args(argv)
    paths = qdp_v3_paths(_workspace(args))
    active = read_json(paths.active_manifest)
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
        active = read_json(paths.active_manifest)
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
        active = read_json(paths.active_manifest)
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


def _ingest(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(prog="qdp ingest", description="Ingest immutable, resumable QDP v3 provider partitions.")
    _common(parser)
    parser.add_argument("--provider", default="baostock", choices=("baostock", "mootdx"))
    parser.add_argument("--mode", required=True, choices=("date-snapshot", "date-events", "factor-symbol-history", "calendar", "security-master", "intraday-5m", "corporate-actions", "financial-quarterly", "performance-forecast", "performance-express", "industry", "index-constituents"))
    parser.add_argument("--trade-date", action="append", default=[])
    parser.add_argument("--start-date", default="")
    parser.add_argument("--end-date", default="")
    parser.add_argument("--universe-kind", default="all_a", choices=("all_a", "etf"))
    parser.add_argument("--refresh", action="store_true")
    parser.add_argument("--no-cross-check", action="store_true")
    parser.add_argument(
        "--max-workers",
        type=int,
        choices=(1, 2),
        default=1,
        help="Prefetch up to two dates while keeping one persistent BaoStock login and one in-flight network request.",
    )
    parser.add_argument("--job-id", default="")
    parser.add_argument("--symbol", action="append", default=[])
    parser.add_argument("--symbols-file", default="")
    args = parser.parse_args(argv)
    workspace = _workspace(args)
    if args.mode in {"financial-quarterly", "performance-forecast", "performance-express"}:
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
                cross_check=not bool(args.no_cross_check),
                job_id=str(args.job_id),
                max_workers=int(args.max_workers),
            )
    _emit(payload, as_json=bool(args.json))
    return 0 if payload.get("status") in {"completed", "strict"} else 2


def _build(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(prog="qdp build candidate", description="Build immutable QDP v3 canonical datasets from raw partitions.")
    _common(parser)
    parser.add_argument("--start-date", default="2010-01-01")
    parser.add_argument("--end-date", default="")
    parser.add_argument("--identity-config", default="")
    parser.add_argument("--allow-unverified-factor-dual-path", action="store_true")
    parser.add_argument("--full", action="store_true", help="Emit the full candidate manifest including raw partition references.")
    args = parser.parse_args(argv)
    candidate = build_candidate(
        workspace_root=_workspace(args),
        start_date=str(args.start_date),
        end_date=str(args.end_date),
        identity_config=str(args.identity_config) or None,
        require_factor_dual_path=not bool(args.allow_unverified_factor_dual_path),
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
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--no-5m", action="store_true")
    parser.add_argument("--no-secondary", action="store_true")
    parser.add_argument("--no-publish", action="store_true")
    parser.add_argument("--hash-v2-shards", action="store_true")
    args = parser.parse_args(argv)
    kwargs = {
        "as_of_date": str(args.as_of_date),
        "workspace_root": _workspace(args),
        "bootstrap": bool(args.bootstrap),
        "start_date": str(args.start_date),
        "include_5m": not bool(args.no_5m),
        "include_secondary": not bool(args.no_secondary),
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
    parser = argparse.ArgumentParser(prog="qdp compatibility run", description="Run the BaoStock 0.9.3 batch and legacy API compatibility gate.")
    _common(parser)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args(argv)
    payload = run_baostock_0_9_3_compatibility_gate(workspace_root=_workspace(args), smoke=bool(args.smoke))
    _emit(payload, as_json=bool(args.json))
    return 0 if payload.get("status") == "passed" else 2


COMMANDS: dict[tuple[str, ...], Callable[[list[str]], int]] = {
    ("status",): _status,
    ("list",): _list,
    ("describe",): _describe,
    ("check",): lambda argv: _audit(argv, prog="qdp check"),
    ("ingest",): _ingest,
    ("build", "candidate"): _build,
    ("audit",): _audit,
    ("diff",): _diff,
    ("publish",): _publish,
    ("rollback",): _rollback,
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
