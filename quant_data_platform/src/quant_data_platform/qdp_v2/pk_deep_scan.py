from __future__ import annotations

import argparse
import json
import shutil
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping

from quant_data_platform.core.json_io import json_safe
from quant_data_platform.qdp_v2.environment import runtime_environment
from quant_data_platform.qdp_v2.manifest import (
    DatasetManifest,
    atomic_write_json,
    dataset_manifest_for_id,
    qdp_v2_root,
    read_active_manifest,
    read_dataset_manifest,
    resolve_manifest_path,
    utc_now,
)
from quant_data_platform.qdp_v2.runtime import resolve_runtime_profile
from quant_data_platform.qdp_v2.status import active_dataset_map


PRIMARY_KEYS = {
    "market_intraday_1m": ["trade_date", "symbol", "bar_time"],
    "market_intraday_5m": ["trade_date", "symbol", "bar_time"],
}


def run_pk_deep_scan(
    *,
    workspace_root: str | Path | None = None,
    domains: Iterable[str] | None = None,
    runtime: str = "balanced",
    duckdb_memory_limit: str = "",
    threads: int = 0,
    batch_shards: int = 64,
    sample_limit: int = 20,
    progress_path: str | Path | None = None,
) -> dict[str, Any]:
    root = qdp_v2_root(workspace_root)
    active = read_active_manifest(root)
    selected = [str(item).strip() for item in list(domains or []) if str(item).strip()] or ["market_intraday_1m", "market_intraday_5m"]
    profile = resolve_runtime_profile(runtime)
    memory_limit = str(duckdb_memory_limit or "").strip() or profile.duckdb_memory_limit
    thread_count = max(1, int(threads or profile.duckdb_threads or 1))
    progress = Path(progress_path) if progress_path else root / "audits" / f"pk_deep_scan_progress_{_stamp()}.jsonl"
    reports: list[dict[str, Any]] = []
    findings: list[dict[str, Any]] = []
    for domain in selected:
        try:
            report = _scan_domain(
                root=root,
                active=active,
                domain=domain,
                memory_limit=memory_limit,
                threads=thread_count,
                batch_shards=max(1, int(batch_shards or 1)),
                sample_limit=max(1, int(sample_limit or 1)),
                progress_path=progress,
            )
            reports.append(report)
            if report.get("status") != "ok":
                findings.append(
                    {
                        "severity": "high",
                        "domain": domain,
                        "code": "primary_key_deep_scan_failed",
                        "evidence": {
                            "duplicate_rows_within_shards": report.get("duplicate_rows_within_shards", 0),
                            "key_null_rows": report.get("key_null_rows", 0),
                            "cross_shard_overlap_candidate_pairs": report.get("cross_shard_overlap_candidate_pairs", 0),
                            "row_count_matches_manifest": report.get("row_count_matches_manifest", False),
                        },
                    }
                )
        except Exception as exc:
            reports.append({"domain": domain, "status": "error", "error_type": type(exc).__name__, "message": str(exc)})
            findings.append({"severity": "high", "domain": domain, "code": "primary_key_deep_scan_error", "evidence": {"error": str(exc)}})
    status = "ok" if not findings else "needs_attention"
    payload = {
        "status": status,
        "audit_type": "primary_key_deep_scan",
        "audited_at": utc_now(),
        "active_as_of_date": str(active.get("active_as_of_date", "") or ""),
        "runtime": profile.name,
        "duckdb_memory_limit": memory_limit,
        "threads": thread_count,
        "batch_shards": max(1, int(batch_shards or 1)),
        "method": "per_shard_exact_key_uniqueness_plus_cross_shard_symbol_date_range_overlap_proof",
        "reports": reports,
        "findings": findings,
        "progress_path": str(progress.resolve()),
        "runtime_environment": runtime_environment(),
    }
    audit_path = root / "audits" / f"pk_deep_scan_{_stamp()}.json"
    atomic_write_json(audit_path, payload)
    payload["audit_path"] = str(audit_path.resolve())
    return payload


def _scan_domain(
    *,
    root: Path,
    active: Mapping[str, Any],
    domain: str,
    memory_limit: str,
    threads: int,
    batch_shards: int,
    sample_limit: int,
    progress_path: Path,
) -> dict[str, Any]:
    datasets = active_dataset_map(dict(active))
    dataset_id = str(datasets.get(domain, "") or "")
    if not dataset_id:
        return {"domain": domain, "status": "skipped", "reason": "not_active_domain"}
    manifest_path = dataset_manifest_for_id(root, dataset_id, domain)
    if manifest_path is None:
        return {"domain": domain, "status": "error", "reason": "dataset_manifest_missing", "dataset_id": dataset_id}
    manifest = read_dataset_manifest(manifest_path)
    paths = [resolve_manifest_path(shard.path, root=root) for shard in manifest.shards if resolve_manifest_path(shard.path, root=root).exists()]
    pk = PRIMARY_KEYS.get(domain, list(manifest.primary_key or []))
    if not paths or not pk:
        return {"domain": domain, "status": "error", "reason": "paths_or_primary_key_missing", "dataset_id": dataset_id}

    import duckdb  # type: ignore

    scanned_rows = 0
    key_null_rows = 0
    duplicate_rows_within = 0
    duplicate_examples: list[dict[str, Any]] = []
    bad_bar_count_examples: list[dict[str, Any]] = []
    bad_bar_count_symbol_days = 0
    per_shard_rows: list[dict[str, Any]] = []
    memberships: list[dict[str, Any]] = []
    batches = list(_batches(paths, batch_shards))
    close_warning = ""
    temp_dir = root / "audits" / ".duckdb_tmp" / f"pk_{domain}_{_stamp()}"
    temp_dir.mkdir(parents=True, exist_ok=True)
    con = duckdb.connect(":memory:")
    try:
        _configure(con, memory_limit=memory_limit, threads=threads)
        con.execute(f"set temp_directory='{str(temp_dir).replace(chr(92), '/').replace(chr(39), chr(39) + chr(39))}'")
        for batch_index, batch in enumerate(batches, start=1):
            paths_sql = _path_list_sql(batch)
            null_expr = " or ".join(f"{_q(col)} is null" for col in pk) or "false"
            per_shard = con.execute(
                f"""
                select
                  filename,
                  count(*) as row_count,
                  sum(case when {null_expr} then 1 else 0 end) as key_null_rows,
                  strftime(min(cast(trade_date as date)), '%Y-%m-%d') as start_date,
                  strftime(max(cast(trade_date as date)), '%Y-%m-%d') as end_date,
                  count(distinct symbol) as symbol_count,
                  count(distinct trade_date) as trade_date_count
                from read_parquet({paths_sql}, union_by_name=true, filename=true)
                group by filename
                order by filename
                """
            ).fetchdf().to_dict(orient="records")
            per_shard_rows.extend(dict(row) for row in per_shard)
            for row in per_shard:
                scanned_rows += int(row.get("row_count", 0) or 0)
                key_null_rows += int(row.get("key_null_rows", 0) or 0)
            expected_bars = 240 if domain == "market_intraday_1m" else 48 if domain == "market_intraday_5m" else 0
            bad_groups = con.execute(
                f"""
                with grouped as (
                  select
                    filename,
                    symbol,
                    trade_date,
                    count(*) as row_count,
                    count(distinct bar_time) as distinct_bar_time_count
                  from read_parquet({paths_sql}, union_by_name=true, filename=true)
                  group by filename, symbol, trade_date
                )
                select *
                from grouped
                where row_count != distinct_bar_time_count
                   or ({int(expected_bars)} > 0 and row_count != {int(expected_bars)})
                order by filename, symbol, trade_date
                limit {sample_limit}
                """
            ).fetchdf().to_dict(orient="records")
            if bad_groups:
                aggregate_bad = con.execute(
                    f"""
                    with grouped as (
                      select
                        filename,
                        symbol,
                        trade_date,
                        count(*) as row_count,
                        count(distinct bar_time) as distinct_bar_time_count
                      from read_parquet({paths_sql}, union_by_name=true, filename=true)
                      group by filename, symbol, trade_date
                    )
                    select
                      sum(case when row_count != distinct_bar_time_count then row_count - distinct_bar_time_count else 0 end) as duplicate_rows,
                      sum(case when {int(expected_bars)} > 0 and row_count != {int(expected_bars)} then 1 else 0 end) as bad_bar_count_symbol_days
                    from grouped
                    where row_count != distinct_bar_time_count
                       or ({int(expected_bars)} > 0 and row_count != {int(expected_bars)})
                    """
                ).fetchone()
                duplicate_rows_in_batch = int(aggregate_bad[0] or 0)
                bad_bar_count_in_batch = int(aggregate_bad[1] or 0)
                duplicate_rows_within += duplicate_rows_in_batch
                if duplicate_rows_in_batch and len(duplicate_examples) < sample_limit:
                    duplicate_examples.extend([dict(row) for row in bad_groups[: max(0, sample_limit - len(duplicate_examples))]])
                if bad_bar_count_in_batch and len(bad_bar_count_examples) < sample_limit:
                    bad_bar_count_examples.extend([dict(row) for row in bad_groups[: max(0, sample_limit - len(bad_bar_count_examples))]])
                bad_bar_count_symbol_days += bad_bar_count_in_batch
            else:
                duplicate_rows_in_batch = 0
            membership = con.execute(
                f"""
                select
                  filename,
                  symbol,
                  strftime(min(cast(trade_date as date)), '%Y-%m-%d') as start_date,
                  strftime(max(cast(trade_date as date)), '%Y-%m-%d') as end_date,
                  count(distinct trade_date) as trade_date_count,
                  count(*) as row_count
                from read_parquet({paths_sql}, union_by_name=true, filename=true)
                group by filename, symbol
                order by symbol, start_date, end_date, filename
                """
            ).fetchdf().to_dict(orient="records")
            memberships.extend(dict(row) for row in membership)
            _append_progress(
                progress_path,
                {
                    "event": "pk_deep_scan_batch_done",
                    "domain": domain,
                    "batch_index": batch_index,
                    "batch_count": len(batches),
                    "scanned_rows": scanned_rows,
                    "membership_rows": len(memberships),
                },
            )
    except Exception:
        try:
            con.close()
        except Exception:
            pass
        raise
    try:
        con.close()
    except Exception as exc:
        close_warning = str(exc)
    shutil.rmtree(temp_dir, ignore_errors=True)

    overlaps = _find_symbol_date_overlaps(memberships, sample_limit=sample_limit)
    row_count_matches = int(scanned_rows) == int(manifest.row_count or 0)
    status = "ok" if duplicate_rows_within == 0 and key_null_rows == 0 and bad_bar_count_symbol_days == 0 and not overlaps["candidate_pairs"] and row_count_matches else "needs_attention"
    return {
        "domain": domain,
        "status": status,
        "dataset_id": dataset_id,
        "manifest_row_count": int(manifest.row_count or 0),
        "scanned_rows": scanned_rows,
        "row_count_matches_manifest": row_count_matches,
        "shard_count": len(manifest.shards),
        "scanned_shard_count": len(paths),
        "primary_key": pk,
        "key_null_rows": key_null_rows,
        "duplicate_rows_within_shards": duplicate_rows_within,
        "within_shard_duplicate_examples": duplicate_examples,
        "bad_bar_count_symbol_days": bad_bar_count_symbol_days,
        "bad_bar_count_examples": bad_bar_count_examples,
        "cross_shard_overlap_candidate_pairs": overlaps["candidate_pairs"],
        "cross_shard_overlap_symbols": overlaps["symbols"],
        "cross_shard_overlap_examples": overlaps["examples"],
        "duckdb_close_warning": close_warning,
        "proof": "If within-shard duplicates are zero and cross-shard same-symbol date-range overlaps are zero, the full primary key is globally unique for this sharding contract.",
    }


def _find_symbol_date_overlaps(rows: list[dict[str, Any]], *, sample_limit: int) -> dict[str, Any]:
    by_symbol: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_symbol[str(row.get("symbol", "") or "")].append(row)
    examples: list[dict[str, Any]] = []
    candidate_pairs = 0
    symbols: set[str] = set()
    for symbol, items in by_symbol.items():
        ordered = sorted(items, key=lambda row: (str(row.get("start_date", "") or ""), str(row.get("end_date", "") or ""), str(row.get("filename", "") or "")))
        active: list[dict[str, Any]] = []
        for item in ordered:
            start = str(item.get("start_date", "") or "")
            active = [prev for prev in active if str(prev.get("end_date", "") or "") >= start]
            for prev in active:
                candidate_pairs += 1
                symbols.add(symbol)
                if len(examples) < sample_limit:
                    examples.append(
                        {
                            "symbol": symbol,
                            "left_file": str(prev.get("filename", "")),
                            "left_range": f"{prev.get('start_date', '')}..{prev.get('end_date', '')}",
                            "right_file": str(item.get("filename", "")),
                            "right_range": f"{item.get('start_date', '')}..{item.get('end_date', '')}",
                        }
                    )
            active.append(item)
    return {"candidate_pairs": candidate_pairs, "symbols": len(symbols), "examples": examples}


def _configure(con: Any, *, memory_limit: str, threads: int) -> None:
    safe_memory = str(memory_limit).replace("'", "")
    con.execute(f"set memory_limit='{safe_memory}'")
    con.execute(f"set threads={max(1, int(threads or 1))}")
    con.execute("set preserve_insertion_order=false")


def _path_list_sql(paths: Iterable[Path]) -> str:
    return "[" + ", ".join("'" + str(path).replace("\\", "/").replace("'", "''") + "'" for path in paths) + "]"


def _key_expr(cols: Iterable[str]) -> str:
    parts = [f"coalesce(cast({_q(col)} as varchar), '<NULL>')" for col in cols]
    return "concat_ws(chr(31), " + ", ".join(parts) + ")"


def _q(name: str) -> str:
    return '"' + str(name).replace('"', '""') + '"'


def _batches(items: list[Path], size: int) -> Iterable[list[Path]]:
    step = max(1, int(size or 1))
    for index in range(0, len(items), step):
        yield items[index : index + step]


def _append_progress(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(json_safe(dict(payload)), ensure_ascii=False) + "\n")


def _stamp() -> str:
    return utc_now().replace(":", "").replace("-", "")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="qdp check pk-deep", description="Deep primary-key proof for large active datasets.")
    parser.add_argument("--workspace-root", default="")
    parser.add_argument("--domains", default="market_intraday_1m,market_intraday_5m")
    parser.add_argument("--runtime", default="balanced", choices=("safe", "balanced", "fast"))
    parser.add_argument("--duckdb-memory-limit", default="")
    parser.add_argument("--threads", type=int, default=0)
    parser.add_argument("--batch-shards", type=int, default=64)
    parser.add_argument("--sample-limit", type=int, default=20)
    parser.add_argument("--progress-path", default="")
    parser.add_argument("--json", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    domains = [item.strip() for item in str(args.domains or "").split(",") if item.strip()]
    payload = run_pk_deep_scan(
        workspace_root=str(args.workspace_root or "") or None,
        domains=domains,
        runtime=str(args.runtime or "balanced"),
        duckdb_memory_limit=str(args.duckdb_memory_limit or ""),
        threads=int(args.threads or 0),
        batch_shards=int(args.batch_shards or 64),
        sample_limit=int(args.sample_limit or 20),
        progress_path=str(args.progress_path or "") or None,
    )
    if bool(args.json):
        print(json.dumps(json_safe(payload), ensure_ascii=False, indent=2))
    else:
        print("\n".join(f"{key}: {value}" for key, value in payload.items() if key in {"status", "audit_path", "progress_path"}))
    return 0 if str(payload.get("status", "")) in {"ok", "needs_attention"} else 2


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
