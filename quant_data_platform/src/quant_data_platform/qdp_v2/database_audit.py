from __future__ import annotations

"""Lean quality checks for the single mutable QDP data store.

The checks protect conversion and storage correctness.  They deliberately do
not compare provider prices or derive 5-minute bars from a retired 1-minute
table.
"""

import argparse
import json
import os
import tempfile
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import pyarrow.parquet as pq

from quant_data_platform.core.json_io import json_safe
from quant_data_platform.qdp_v2.audit import audit_active
from quant_data_platform.qdp_v2.duckdb_resources import open_guarded_duckdb
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
from quant_data_platform.qdp_v2.status import active_dataset_map


# Optional low-frequency tables remain useful, but their temporary staleness
# must not make the daily market store unusable.
REQUIRED_DOMAINS = {
    "trading_calendar",
    "security_identity",
    "symbol_history",
    "market_daily_raw",
    "security_status",
    "universe_snapshot",
    "adjust_factor",
    "market_intraday_5m",
}

REQUIRED_COLUMNS: dict[str, tuple[str, ...]] = {
    "trading_calendar": ("trade_date", "is_open", "exchange", "source"),
    "security_identity": (
        "security_id",
        "exchange",
        "list_date",
        "current_symbol",
    ),
    "symbol_history": (
        "security_id",
        "symbol",
        "effective_from",
        "effective_to",
    ),
    "market_daily_raw": (
        "symbol",
        "trade_date",
        "open",
        "high",
        "low",
        "close",
        "volume",
        "amount",
        "source",
        "adjusted_flag",
    ),
    "security_status": (
        "symbol",
        "trade_date",
        "is_st",
        "is_suspended",
        "is_delisted",
        "source",
    ),
    "universe_snapshot": (
        "symbol",
        "trade_date",
        "name",
        "exchange",
        "list_status",
        "source",
    ),
    "adjust_factor": (
        "symbol",
        "trade_date",
        "adjust_factor",
        "factor_provider",
        "factor_semantics",
        "source",
    ),
    "market_intraday_5m": (
        "symbol",
        "trade_date",
        "bar_time",
        "open",
        "high",
        "low",
        "close",
        "volume",
        "amount",
        "source",
        "adjusted_flag",
    ),
}

EXPECTED_BAR_TIMES = tuple(
    f"{hour:02d}{minute:02d}00000"
    for hour, minute in (
        *[(9, minute) for minute in range(35, 60, 5)],
        *[(10, minute) for minute in range(0, 60, 5)],
        *[(11, minute) for minute in range(0, 31, 5)],
        *[(13, minute) for minute in range(5, 60, 5)],
        *[(14, minute) for minute in range(0, 60, 5)],
        (15, 0),
    )
)


def audit_latest_keys(
    *, workspace_root: str | Path | None = None
) -> dict[str, Any]:
    """Check only the newest daily partition and overlapping 5m shards."""

    root = qdp_v2_root(workspace_root)
    active = read_active_manifest(root)
    datasets = active_dataset_map(active)
    available = REQUIRED_DOMAINS.intersection(datasets)
    required_for_keys = {
        "market_daily_raw",
        "adjust_factor",
        "universe_snapshot",
        "security_status",
        "market_intraday_5m",
    }
    missing = sorted(required_for_keys.difference(available))
    if missing:
        return {
            "status": "skipped",
            "reason": f"latest_key_domains_missing:{','.join(missing)}",
            "finding_count": 0,
        }

    manifests = {
        domain: _active_manifest(root, datasets[domain], domain)
        for domain in required_for_keys
    }
    trade_date = str(manifests["market_daily_raw"].end_date or "")
    if not trade_date:
        return {
            "status": "skipped",
            "reason": "market_daily_end_date_missing",
            "finding_count": 0,
        }
    paths = {
        domain: _paths_for_date(root, manifest, trade_date)
        for domain, manifest in manifests.items()
    }
    absent_paths = sorted(domain for domain, items in paths.items() if not items)
    if absent_paths:
        return {
            "status": "needs_attention",
            "trade_date": trade_date,
            "finding_count": len(absent_paths),
            "errors": [f"latest_partition_missing:{item}" for item in absent_paths],
        }

    workspace = Path(workspace_root or Path.cwd()).resolve()
    temp = workspace / "quant_data_platform" / "data" / "qdp_runtime" / "check_spill"
    with open_guarded_duckdb(temp_directory=temp, threads=4) as con:
        daily_factor_missing = _except_count(
            con,
            paths["market_daily_raw"],
            paths["adjust_factor"],
            trade_date,
        )
        factor_daily_extra = _except_count(
            con,
            paths["adjust_factor"],
            paths["market_daily_raw"],
            trade_date,
        )
        universe_status_missing = _except_count(
            con,
            paths["universe_snapshot"],
            paths["security_status"],
            trade_date,
        )
        status_universe_extra = _except_count(
            con,
            paths["security_status"],
            paths["universe_snapshot"],
            trade_date,
        )
        daily_universe_missing = _except_count(
            con,
            paths["market_daily_raw"],
            paths["universe_snapshot"],
            trade_date,
        )
        five = _latest_5m_stats(
            con,
            daily_paths=paths["market_daily_raw"],
            intraday_paths=paths["market_intraday_5m"],
            trade_date=trade_date,
        )

    errors = {
        "daily_missing_factor": daily_factor_missing,
        "factor_extra_vs_daily": factor_daily_extra,
        "universe_missing_status": universe_status_missing,
        "status_extra_vs_universe": status_universe_extra,
        "daily_missing_universe": daily_universe_missing,
        "invalid_5m_stock_days": int(five["invalid_day_count"]),
    }
    blocking = {key: value for key, value in errors.items() if int(value) > 0}
    coverage = float(five["coverage_ratio"])
    if coverage < 0.98:
        blocking["5m_complete_coverage_below_98_percent"] = int(
            five["missing_complete_day_count"]
        )
    if blocking:
        status = "needs_attention"
    elif coverage < 0.99:
        status = "warning"
    else:
        status = "ok"
    return {
        "status": status,
        "trade_date": trade_date,
        "finding_count": len(blocking),
        "key_checks": errors,
        "five_minute": five,
        "errors": [f"{key}:{value}" for key, value in blocking.items()],
        "warnings": (
            [f"5m_complete_coverage_between_98_and_99_percent:{coverage:.6f}"]
            if not blocking and coverage < 0.99
            else []
        ),
    }


def audit_database(
    *,
    workspace_root: str | Path | None = None,
    deep: bool = True,
    runtime: str = "balanced",
    duckdb_memory_limit: str = "",
    threads: int | None = None,
    max_shards: int = 0,
    sample_limit: int = 20,
    skip_global_primary_key: bool = False,
    global_uniqueness_max_shards: int = 0,
    skip_cross_frequency: bool = False,
    write: bool = False,
) -> dict[str, Any]:
    """Audit current tables without provider arbitration or 1m dependencies.

    Deprecated resource and cross-frequency arguments remain accepted so old
    local commands fail gently; memory is always governed by the physical
    0.5-GiB/2-second floor.
    """

    del duckdb_memory_limit, global_uniqueness_max_shards, skip_cross_frequency
    root = qdp_v2_root(workspace_root)
    active = read_active_manifest(root)
    if not active:
        return {
            "status": "needs_attention",
            "mode": "full",
            "deep": bool(deep),
            "dataset_count": 0,
            "finding_count": 1,
            "findings": [
                _finding("high", "manifest", "active", "active_manifest_missing", {})
            ],
        }

    active_audit = audit_active(
        workspace_root=workspace_root,
        write=False,
        verify_footers=True,
    )
    datasets = active_dataset_map(active)
    missing_required = sorted(REQUIRED_DOMAINS.difference(datasets))
    findings: list[dict[str, Any]] = [
        _finding(
            "high",
            "coverage",
            domain,
            "required_domain_missing",
            {},
        )
        for domain in missing_required
    ]
    for error in list(active_audit.get("errors", []) or []):
        findings.append(
            _finding(
                "high",
                "manifest",
                "active",
                "active_manifest_or_footer_error",
                {"error": str(error)},
            )
        )

    selected_threads = int(threads or {"safe": 2, "balanced": 4, "fast": 8}.get(runtime, 4))
    selected_threads = max(1, min(selected_threads, os.cpu_count() or 1))
    workspace = Path(workspace_root or Path.cwd()).resolve()
    reports: list[dict[str, Any]] = []
    with _audit_temp_directory(workspace) as temp:
        with open_guarded_duckdb(temp_directory=temp, threads=selected_threads) as con:
            for domain, dataset_id in sorted(datasets.items()):
                manifest = _active_manifest(root, dataset_id, domain)
                report, domain_findings = _audit_dataset(
                    con,
                    root=root,
                    manifest=manifest,
                    deep=bool(deep),
                    max_shards=max(0, int(max_shards)),
                    sample_limit=max(1, int(sample_limit)),
                    skip_primary_key=bool(skip_global_primary_key),
                )
                reports.append(report)
                findings.extend(domain_findings)

    latest = audit_latest_keys(workspace_root=workspace_root)
    if latest.get("status") == "needs_attention":
        findings.extend(
            _finding(
                "high",
                "latest_keys",
                "market_core",
                "latest_key_check_failed",
                {"error": error},
            )
            for error in list(latest.get("errors", []) or [])
        )
    elif latest.get("status") == "warning":
        findings.append(
            _finding(
                "medium",
                "latest_keys",
                "market_intraday_5m",
                "latest_5m_coverage_warning",
                {"five_minute": latest.get("five_minute", {})},
            )
        )

    blocking = [item for item in findings if item["severity"] in {"high", "critical"}]
    payload: dict[str, Any] = {
        "status": "needs_attention" if blocking else "ok",
        "mode": "full",
        "deep": bool(deep),
        "qdp_v2_root": str(root.resolve()),
        "audited_at": utc_now(),
        "dataset_count": len(reports),
        "required_domains": sorted(REQUIRED_DOMAINS),
        "missing_required_domains": missing_required,
        "datasets": reports,
        "latest_keys": latest,
        "cross_dataset_checks": {"latest_keys": latest},
        "finding_count": len(findings),
        "findings": findings,
        "warnings": [item for item in findings if item["severity"] == "medium"],
        "errors": blocking,
        "resource_policy": {
            "threads": selected_threads,
            "memory": "dynamic_available_ram_minus_0.5_gib_with_2_second_watchdog",
        },
    }
    if write:
        audit_id = f"database_audit_{utc_now().replace(':', '').replace('-', '')}"
        path = root / "audits" / f"{audit_id}.json"
        atomic_write_json(path, payload)
        payload["audit_path"] = str(path.resolve())
    return payload


def _audit_dataset(
    con: Any,
    *,
    root: Path,
    manifest: DatasetManifest,
    deep: bool,
    max_shards: int,
    sample_limit: int,
    skip_primary_key: bool,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    all_paths = [resolve_manifest_path(item.path, root=root) for item in manifest.shards]
    paths = all_paths[:max_shards] if max_shards else all_paths
    findings: list[dict[str, Any]] = []
    columns: list[str] = []
    if paths:
        columns = list(pq.read_schema(paths[0]).names)
    missing_columns = sorted(set(REQUIRED_COLUMNS.get(manifest.domain, ())).difference(columns))
    if missing_columns:
        findings.append(
            _finding(
                "high",
                "schema",
                manifest.domain,
                "required_columns_missing",
                {"columns": missing_columns},
            )
        )

    checks: dict[str, Any] = {
        "schema": {
            "status": "ok" if not missing_columns else "error",
            "columns": columns,
            "missing_columns": missing_columns,
        },
        "scanned_shards": len(paths),
        "total_shards": len(all_paths),
    }
    if deep and paths and not missing_columns:
        if manifest.primary_key and not skip_primary_key:
            pk = _primary_key_check(con, paths, manifest.primary_key, sample_limit)
            checks["primary_key"] = pk
            if int(pk["null_key_rows"]) > 0:
                findings.append(
                    _finding("high", "primary_key", manifest.domain, "primary_key_null_rows", pk)
                )
            if int(pk["duplicate_rows"]) > 0:
                findings.append(
                    _finding(
                        "high",
                        "primary_key",
                        manifest.domain,
                        "primary_key_duplicate_rows",
                        pk,
                    )
                )
        if manifest.domain in {"market_daily_raw", "market_intraday_5m"}:
            ohlc = _ohlc_check(con, paths, sample_limit)
            checks["ohlc"] = ohlc
            if int(ohlc["invalid_rows"]) > 0:
                findings.append(
                    _finding("high", "ohlc", manifest.domain, "invalid_ohlcv_rows", ohlc)
                )
        if manifest.domain == "market_intraday_5m":
            bars = _bar_day_check(con, paths, sample_limit)
            checks["bar_days"] = bars
            if int(bars["invalid_day_count"]) > 0:
                findings.append(
                    _finding("high", "bar_contract", manifest.domain, "invalid_48_bar_days", bars)
                )
        if manifest.domain == "adjust_factor":
            factor = _factor_check(con, paths, sample_limit)
            checks["factor"] = factor
            if int(factor["invalid_rows"]) > 0:
                findings.append(
                    _finding("high", "factor", manifest.domain, "invalid_adjust_factor_rows", factor)
                )
        if manifest.domain == "symbol_history":
            identity = _symbol_history_check(con, paths, sample_limit)
            checks["identity_intervals"] = identity
            if int(identity["overlap_count"]) > 0:
                findings.append(
                    _finding("high", "identity", manifest.domain, "symbol_history_interval_overlap", identity)
                )

    return (
        {
            "domain": manifest.domain,
            "dataset_id": manifest.dataset_id,
            "row_count": manifest.row_count,
            "shard_count": len(manifest.shards),
            "start_date": manifest.start_date,
            "end_date": manifest.end_date,
            "checks": checks,
        },
        findings,
    )


def _primary_key_check(
    con: Any,
    paths: Sequence[Path],
    primary_key: Sequence[str],
    sample_limit: int,
) -> dict[str, Any]:
    keys = ",".join(_q(item) for item in primary_key)
    nulls = " OR ".join(f"{_q(item)} IS NULL" for item in primary_key)
    null_count = int(
        con.execute(
            f"SELECT count(*) FROM read_parquet(?, union_by_name=true) WHERE {nulls}",
            [_path_texts(paths)],
        ).fetchone()[0]
    )
    duplicate_rows = int(
        con.execute(
            f"""
            SELECT coalesce(sum(n - 1), 0) FROM (
              SELECT count(*) AS n
              FROM read_parquet(?, union_by_name=true)
              GROUP BY {keys}
              HAVING count(*) > 1
            )
            """,
            [_path_texts(paths)],
        ).fetchone()[0]
    )
    examples = con.execute(
        f"""
        SELECT {keys}, count(*) AS rows
        FROM read_parquet(?, union_by_name=true)
        GROUP BY {keys}
        HAVING count(*) > 1
        LIMIT {int(sample_limit)}
        """,
        [_path_texts(paths)],
    ).fetchdf().to_dict("records") if duplicate_rows else []
    return {
        "status": "ok" if not null_count and not duplicate_rows else "error",
        "null_key_rows": null_count,
        "duplicate_rows": duplicate_rows,
        "examples": examples,
    }


def _ohlc_check(con: Any, paths: Sequence[Path], sample_limit: int) -> dict[str, Any]:
    predicate = """
      NOT isfinite(try_cast(open AS DOUBLE))
      OR NOT isfinite(try_cast(high AS DOUBLE))
      OR NOT isfinite(try_cast(low AS DOUBLE))
      OR NOT isfinite(try_cast(close AS DOUBLE))
      OR try_cast(open AS DOUBLE) <= 0
      OR try_cast(high AS DOUBLE) <= 0
      OR try_cast(low AS DOUBLE) <= 0
      OR try_cast(close AS DOUBLE) <= 0
      OR try_cast(high AS DOUBLE) < greatest(try_cast(open AS DOUBLE), try_cast(low AS DOUBLE), try_cast(close AS DOUBLE))
      OR try_cast(low AS DOUBLE) > least(try_cast(open AS DOUBLE), try_cast(high AS DOUBLE), try_cast(close AS DOUBLE))
      OR try_cast(volume AS DOUBLE) < 0
      OR try_cast(amount AS DOUBLE) < 0
    """
    invalid = int(
        con.execute(
            f"SELECT count(*) FROM read_parquet(?, union_by_name=true) WHERE {predicate}",
            [_path_texts(paths)],
        ).fetchone()[0]
    )
    examples = con.execute(
        f"""
        SELECT symbol, trade_date, open, high, low, close, volume, amount
        FROM read_parquet(?, union_by_name=true)
        WHERE {predicate}
        LIMIT {int(sample_limit)}
        """,
        [_path_texts(paths)],
    ).fetchdf().to_dict("records") if invalid else []
    return {"status": "ok" if not invalid else "error", "invalid_rows": invalid, "examples": examples}


def _bar_day_check(con: Any, paths: Sequence[Path], sample_limit: int) -> dict[str, Any]:
    times = ",".join(_sql_literal(item) for item in EXPECTED_BAR_TIMES)
    query = f"""
      SELECT symbol, trade_date,
             count(*) AS rows,
             count(DISTINCT cast(bar_time AS VARCHAR)) AS distinct_times,
             sum(CASE WHEN cast(bar_time AS VARCHAR) IN ({times}) THEN 0 ELSE 1 END) AS unexpected_times
      FROM read_parquet(?, union_by_name=true)
      GROUP BY symbol, trade_date
      HAVING rows <> 48 OR distinct_times <> 48 OR unexpected_times <> 0
    """
    invalid = 0
    examples: list[dict[str, Any]] = []
    # Complete stock-days are stored in one date-partitioned shard. Checking
    # each shard keeps the 48-bar aggregation bounded by one year instead of
    # creating an avoidable all-history hash table for 472M rows.
    for path in paths:
        params = [[str(path)]]
        shard_invalid = int(
            con.execute(f"SELECT count(*) FROM ({query})", params).fetchone()[0]
        )
        invalid += shard_invalid
        remaining = max(0, int(sample_limit) - len(examples))
        if shard_invalid and remaining:
            rows = con.execute(
                f"SELECT * FROM ({query}) LIMIT {remaining}",
                params,
            ).fetchdf().to_dict("records")
            for row in rows:
                row["shard"] = str(path)
            examples.extend(rows)
    return {"status": "ok" if not invalid else "error", "invalid_day_count": invalid, "examples": examples}


def _factor_check(con: Any, paths: Sequence[Path], sample_limit: int) -> dict[str, Any]:
    predicate = "NOT isfinite(try_cast(adjust_factor AS DOUBLE)) OR try_cast(adjust_factor AS DOUBLE) <= 0"
    invalid = int(
        con.execute(
            f"SELECT count(*) FROM read_parquet(?, union_by_name=true) WHERE {predicate}",
            [_path_texts(paths)],
        ).fetchone()[0]
    )
    examples = con.execute(
        f"SELECT symbol, trade_date, adjust_factor FROM read_parquet(?, union_by_name=true) WHERE {predicate} LIMIT {int(sample_limit)}",
        [_path_texts(paths)],
    ).fetchdf().to_dict("records") if invalid else []
    return {"status": "ok" if not invalid else "error", "invalid_rows": invalid, "examples": examples}


def _symbol_history_check(con: Any, paths: Sequence[Path], sample_limit: int) -> dict[str, Any]:
    overlap_query = """
      SELECT a.security_id, a.symbol AS left_symbol, b.symbol AS right_symbol,
             a.effective_from AS left_from, a.effective_to AS left_to,
             b.effective_from AS right_from, b.effective_to AS right_to
      FROM read_parquet(?, union_by_name=true) a
      JOIN read_parquet(?, union_by_name=true) b
        ON cast(a.security_id AS VARCHAR)=cast(b.security_id AS VARCHAR)
       AND (cast(a.symbol AS VARCHAR), cast(a.effective_from AS VARCHAR))
           < (cast(b.symbol AS VARCHAR), cast(b.effective_from AS VARCHAR))
       AND cast(a.effective_from AS VARCHAR) <= cast(b.effective_to AS VARCHAR)
       AND cast(b.effective_from AS VARCHAR) <= cast(a.effective_to AS VARCHAR)
    """
    params = [_path_texts(paths), _path_texts(paths)]
    count = int(con.execute(f"SELECT count(*) FROM ({overlap_query})", params).fetchone()[0])
    examples = con.execute(
        f"SELECT * FROM ({overlap_query}) LIMIT {int(sample_limit)}", params
    ).fetchdf().to_dict("records") if count else []
    return {"status": "ok" if not count else "error", "overlap_count": count, "examples": examples}


def _latest_5m_stats(
    con: Any,
    *,
    daily_paths: Sequence[Path],
    intraday_paths: Sequence[Path],
    trade_date: str,
) -> dict[str, Any]:
    times = ",".join(_sql_literal(item) for item in EXPECTED_BAR_TIMES)
    params = [_path_texts(intraday_paths), trade_date, _path_texts(daily_paths), trade_date]
    row = con.execute(
        f"""
        WITH five AS (
          SELECT cast(symbol AS VARCHAR) AS symbol,
                 count(*) AS rows,
                 count(DISTINCT cast(bar_time AS VARCHAR)) AS distinct_times,
                 sum(CASE WHEN cast(bar_time AS VARCHAR) IN ({times}) THEN 0 ELSE 1 END) AS unexpected
          FROM read_parquet(?, union_by_name=true)
          WHERE cast(trade_date AS VARCHAR)=?
          GROUP BY symbol
        ), complete AS (
          SELECT symbol FROM five WHERE rows=48 AND distinct_times=48 AND unexpected=0
        ), traded AS (
          SELECT DISTINCT cast(symbol AS VARCHAR) AS symbol
          FROM read_parquet(?, union_by_name=true)
          WHERE cast(trade_date AS VARCHAR)=? AND coalesce(try_cast(volume AS DOUBLE),0)>0
        )
        SELECT
          (SELECT count(*) FROM traded) AS expected_days,
          (SELECT count(*) FROM complete) AS complete_days,
          (SELECT count(*) FROM five WHERE NOT (rows=48 AND distinct_times=48 AND unexpected=0)) AS invalid_days,
          (SELECT count(*) FROM (SELECT symbol FROM traded EXCEPT SELECT symbol FROM complete)) AS missing_days,
          (SELECT count(*) FROM (SELECT symbol FROM complete EXCEPT SELECT symbol FROM traded)) AS extra_days
        """,
        params,
    ).fetchone()
    expected = int(row[0])
    complete = int(row[1])
    return {
        "expected_traded_day_count": expected,
        "complete_day_count": complete,
        "invalid_day_count": int(row[2]),
        "missing_complete_day_count": int(row[3]),
        "extra_complete_day_count": int(row[4]),
        "coverage_ratio": float(complete / expected) if expected else 1.0,
    }


def _except_count(
    con: Any,
    left_paths: Sequence[Path],
    right_paths: Sequence[Path],
    trade_date: str,
) -> int:
    return int(
        con.execute(
            """
            SELECT count(*) FROM (
              SELECT cast(symbol AS VARCHAR), cast(trade_date AS VARCHAR)
              FROM read_parquet(?, union_by_name=true)
              WHERE cast(trade_date AS VARCHAR)=?
              EXCEPT
              SELECT cast(symbol AS VARCHAR), cast(trade_date AS VARCHAR)
              FROM read_parquet(?, union_by_name=true)
              WHERE cast(trade_date AS VARCHAR)=?
            )
            """,
            [_path_texts(left_paths), trade_date, _path_texts(right_paths), trade_date],
        ).fetchone()[0]
    )


def _active_manifest(root: Path, dataset_id: str, domain: str) -> DatasetManifest:
    path = dataset_manifest_for_id(root, dataset_id, domain)
    if path is None:
        raise FileNotFoundError(f"active_dataset_manifest_missing:{domain}:{dataset_id}")
    return read_dataset_manifest(path)


def _paths_for_date(root: Path, manifest: DatasetManifest, trade_date: str) -> tuple[Path, ...]:
    return tuple(
        resolve_manifest_path(item.path, root=root)
        for item in manifest.shards
        if (
            (not item.start_date or str(item.start_date) <= trade_date)
            and (not item.end_date or str(item.end_date) >= trade_date)
        )
    )


def _audit_temp_directory(workspace: Path):
    runtime = workspace / "quant_data_platform" / "data" / "qdp_runtime"
    runtime.mkdir(parents=True, exist_ok=True)
    return tempfile.TemporaryDirectory(prefix="audit_spill_", dir=str(runtime))


def _path_texts(paths: Iterable[Path]) -> list[str]:
    return [str(item) for item in paths]


def _q(value: str) -> str:
    return '"' + str(value).replace('"', '""') + '"'


def _sql_literal(value: str) -> str:
    return "'" + str(value).replace("'", "''") + "'"


def _finding(
    severity: str,
    category: str,
    domain: str,
    code: str,
    evidence: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "severity": severity,
        "category": category,
        "domain": domain,
        "code": code,
        "evidence": dict(evidence),
    }


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="qdp check --full",
        description="Run structural and row-level checks on the current QDP tables.",
    )
    parser.add_argument("--workspace-root", default="")
    parser.add_argument("--runtime", default="balanced", choices=("safe", "balanced", "fast"))
    parser.add_argument("--threads", type=int, default=0)
    parser.add_argument("--max-shards", type=int, default=0)
    parser.add_argument("--sample-limit", type=int, default=20)
    parser.add_argument("--skip-global-primary-key", action="store_true")
    parser.add_argument("--write-audit", action="store_true")
    parser.add_argument("--json", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    payload = audit_database(
        workspace_root=str(args.workspace_root or "") or None,
        deep=True,
        runtime=str(args.runtime),
        threads=int(args.threads) or None,
        max_shards=int(args.max_shards),
        sample_limit=int(args.sample_limit),
        skip_global_primary_key=bool(args.skip_global_primary_key),
        write=bool(args.write_audit),
    )
    if args.json:
        print(json.dumps(json_safe(payload), ensure_ascii=False, indent=2))
    else:
        print(_format(payload))
    return 0 if payload.get("status") == "ok" else 2


def _format(payload: Mapping[str, Any]) -> str:
    return "\n".join(
        [
            f"status: {payload.get('status', '')}",
            f"dataset_count: {payload.get('dataset_count', 0)}",
            f"finding_count: {payload.get('finding_count', 0)}",
        ]
    )


__all__ = ["REQUIRED_DOMAINS", "audit_database", "audit_latest_keys"]


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
