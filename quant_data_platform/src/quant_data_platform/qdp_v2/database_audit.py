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

import pandas as pd
import pyarrow.parquet as pq

from quant_data_platform.core.json_io import json_safe
from quant_data_platform.qdp_v2.audit import VALUATION_REQUIRED_COLUMNS, audit_active
from quant_data_platform.qdp_v2.auxiliary_update import AUXILIARY_DOMAINS
from quant_data_platform.qdp_v2.duckdb_resources import open_guarded_duckdb
from quant_data_platform.qdp_v2.manifest import (
    DatasetManifest,
    EXPECTED_BAR_TIMES,
    ShardManifestEntry,
    atomic_write_json,
    dataset_manifest_for_id,
    qdp_v2_root,
    read_active_manifest,
    read_dataset_manifest,
    resolve_manifest_path,
    utc_now,
)
from quant_data_platform.qdp_v2.pit_history import (
    LIFECYCLE_NORMALIZE_DOMAINS,
    PitHistoryError,
    audit_symbol_lifecycle_effectivity,
)
from quant_data_platform.qdp_v2.repair import _sql_literal
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
    *AUXILIARY_DOMAINS,
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
    "industry_concept": (
        "symbol",
        "trade_date",
        "industry",
        "source",
        "industry_fill_method",
        "industry_source_date",
        "industry_standard",
    ),
    "share_capital": (
        "symbol",
        "trade_date",
        "total_share",
        "float_share",
        "restricted_share",
        "total_share_source_date",
        "float_share_source_date",
        "restricted_share_source_date",
        "share_fill_method",
        "source",
    ),
    "valuation": tuple(sorted(VALUATION_REQUIRED_COLUMNS)),
    "name_change": (
        "symbol",
        "trade_date",
        "old_name",
        "new_name",
        "change_type",
        "source",
    ),
    "corporate_actions": (
        "symbol",
        "trade_date",
        "announcement_date",
        "ex_date",
        "record_date",
        "dividend_pay_date",
        "action_type",
        "cash_dividend_per_10",
        "bonus_share_per_10",
        "transfer_share_per_10",
        "source",
    ),
    "index_constituents": (
        "index_symbol",
        "symbol",
        "trade_date",
        "index_name",
        "source",
        "source_snapshot_date",
    ),
}

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
        status_semantics = _status_daily_partition_check(
            con,
            status_paths=paths["security_status"],
            daily_paths=paths["market_daily_raw"],
            start_date=trade_date,
            end_date=trade_date,
            sample_limit=5,
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
        "status_daily_semantic_mismatch": int(
            status_semantics["semantic_mismatch_count"]
        ),
        "status_null_suspension_flags": int(
            status_semantics["null_suspension_flag_count"]
        ),
        "invalid_5m_stock_days": int(five["invalid_day_count"]),
    }
    blocking = {key: value for key, value in errors.items() if int(value) > 0}
    coverage = float(five["coverage_ratio"])
    if int(five["missing_complete_day_count"]) > 0:
        blocking["5m_complete_coverage_not_100_percent"] = int(
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
    manifests: dict[str, DatasetManifest] = {}
    cross_checks: dict[str, Any] = {}
    with _audit_temp_directory(workspace) as temp:
        with open_guarded_duckdb(temp_directory=temp, threads=selected_threads) as con:
            for domain, dataset_id in sorted(datasets.items()):
                manifest = _active_manifest(root, dataset_id, domain)
                manifests[domain] = manifest
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
            if deep and {"market_daily_raw", "market_intraday_5m"}.issubset(manifests):
                consistency = _daily_intraday_consistency_check(
                    con,
                    root=root,
                    daily=manifests["market_daily_raw"],
                    intraday=manifests["market_intraday_5m"],
                    history=manifests.get("symbol_history"),
                    sample_limit=max(1, int(sample_limit)),
                )
                cross_checks["daily_intraday_5m"] = consistency
                if int(consistency["invalid_day_count"]) > 0:
                    findings.append(
                        _finding(
                            "high",
                            "cross_frequency",
                            "market_intraday_5m",
                            "daily_intraday_5m_semantic_mismatch",
                            consistency,
                        )
                    )
                coverage = float(consistency["complete_coverage_ratio"])
                missing_days = int(consistency["missing_positive_daily_count"])
                intraday_history_complete = bool(
                    dict(active.get("scope", {}) or {}).get(
                        "intraday_5m_restored_for_historical_symbols",
                        True,
                    )
                )
                if missing_days:
                    consistency["coverage_contract"] = (
                        "full_daily_intraday_history"
                        if intraday_history_complete
                        else "daily_pit_history_with_explicit_intraday_gaps"
                    )
                    findings.append(
                        _finding(
                            "high" if intraday_history_complete else "medium",
                            "coverage",
                            "market_intraday_5m",
                            (
                                "historical_5m_coverage_not_100_percent"
                                if intraday_history_complete
                                else "historical_5m_unavailable_for_restored_daily_symbols"
                            ),
                            consistency,
                        )
                    )
            if deep and {"market_daily_raw", "adjust_factor"}.issubset(manifests):
                factor_semantics = _factor_semantic_check(
                    con,
                    root=root,
                    daily=manifests["market_daily_raw"],
                    factor=manifests["adjust_factor"],
                    sample_limit=max(1, int(sample_limit)),
                )
                cross_checks["factor_semantics"] = factor_semantics
                if (
                    int(factor_semantics["uncompensated_change_count"]) > 0
                    or int(factor_semantics["excessive_change_symbol_year_count"]) > 0
                ):
                    findings.append(
                        _finding(
                            "high",
                            "factor",
                            "adjust_factor",
                            "adjust_factor_change_not_explained_by_raw_price",
                            factor_semantics,
                        )
                    )
            if deep and {"market_daily_raw", "security_status"}.issubset(manifests):
                status_semantics = _status_daily_consistency_check(
                    con,
                    root=root,
                    daily=manifests["market_daily_raw"],
                    status=manifests["security_status"],
                    sample_limit=max(1, int(sample_limit)),
                )
                cross_checks["status_daily_semantics"] = status_semantics
                if (
                    int(status_semantics["semantic_mismatch_count"]) > 0
                    or int(status_semantics["null_suspension_flag_count"]) > 0
                ):
                    findings.append(
                        _finding(
                            "high",
                            "status",
                            "security_status",
                            "security_status_daily_semantic_mismatch",
                            status_semantics,
                        )
                    )
            if deep and {"security_identity", "symbol_history"}.issubset(manifests):
                identity_history = _identity_history_consistency_check(
                    con,
                    root=root,
                    identity=manifests["security_identity"],
                    history=manifests["symbol_history"],
                    sample_limit=max(1, int(sample_limit)),
                )
                cross_checks["identity_symbol_history"] = identity_history
                if identity_history["status"] != "ok":
                    findings.append(
                        _finding(
                            "high",
                            "identity",
                            "symbol_history",
                            "identity_symbol_history_inconsistent",
                            identity_history,
                        )
                    )
            if deep and {"security_status", "trading_calendar"}.issubset(manifests):
                continuity = _long_suspension_check(
                    con,
                    root=root,
                    status=manifests["security_status"],
                    calendar=manifests["trading_calendar"],
                    sample_limit=max(1, int(sample_limit)),
                )
                cross_checks["long_suspension_continuity"] = continuity
            if deep and {
                "market_daily_raw",
                "adjust_factor",
                "security_status",
                "trading_calendar",
                "security_identity",
            }.issubset(manifests):
                reopen = _reopen_discontinuity_check(
                    con,
                    root=root,
                    daily=manifests["market_daily_raw"],
                    factor=manifests["adjust_factor"],
                    status=manifests["security_status"],
                    calendar=manifests["trading_calendar"],
                    identity=manifests["security_identity"],
                    sample_limit=max(1, int(sample_limit)),
                )
                cross_checks["reopen_discontinuities"] = reopen
                if int(reopen["unexpected_discontinuity_count"]) > 0:
                    findings.append(
                        _finding(
                            "high",
                            "price_continuity",
                            "market_daily_raw",
                            "unexplained_adjacent_trade_price_discontinuity",
                            reopen,
                        )
                    )
            auxiliary_required = {
                "market_daily_raw",
                "universe_snapshot",
                "adjust_factor",
                *AUXILIARY_DOMAINS,
            }
            if deep and auxiliary_required.issubset(manifests):
                auxiliary = _auxiliary_semantics_check(
                    con,
                    root=root,
                    manifests=manifests,
                    sample_limit=max(1, int(sample_limit)),
                )
                cross_checks["auxiliary_domains"] = auxiliary
                if auxiliary["status"] != "ok":
                    findings.append(
                        _finding(
                            "high",
                            "auxiliary",
                            "auxiliary_domains",
                            "auxiliary_strict_pit_or_completeness_error",
                            auxiliary,
                        )
                    )

    lifecycle_domains = tuple(
        domain for domain in LIFECYCLE_NORMALIZE_DOMAINS if domain in datasets
    )
    if deep and lifecycle_domains and {
        "security_identity",
        "symbol_history",
    }.issubset(datasets):
        try:
            lifecycle = audit_symbol_lifecycle_effectivity(
                workspace_root=workspace,
                domains=lifecycle_domains,
                sample_limit=max(1, int(sample_limit)),
            )
        except PitHistoryError as exc:
            lifecycle = {
                "status": "error",
                "error": str(exc),
                "domains": list(lifecycle_domains),
            }
        cross_checks["symbol_lifecycle_effectivity"] = lifecycle
        if lifecycle["status"] != "ok":
            findings.append(
                _finding(
                    "high",
                    "identity",
                    "daily_research_domains",
                    "symbol_lifecycle_effective_interval_violation",
                    lifecycle,
                )
            )

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

    blocking = [item for item in findings if item["severity"] == "high"]
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
        "cross_dataset_checks": {"latest_keys": latest, **cross_checks},
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
    entries = list(manifest.shards[:max_shards] if max_shards else manifest.shards)
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
            pk = (
                _ordered_intraday_primary_key_check(
                    con,
                    paths,
                    entries,
                    manifest.primary_key,
                    sample_limit,
                )
                if manifest.domain == "market_intraday_5m"
                else _primary_key_check(con, paths, manifest.primary_key, sample_limit)
            )
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


def _ordered_intraday_primary_key_check(
    con: Any,
    paths: Sequence[Path],
    entries: Sequence[ShardManifestEntry],
    primary_key: Sequence[str],
    sample_limit: int,
) -> dict[str, Any]:
    if set(primary_key) != {"symbol", "trade_date", "bar_time"}:
        return {
            "status": "error",
            "null_key_rows": 0,
            "duplicate_rows": 1,
            "range_overlap_count": 0,
            "examples": [{"reason": "unexpected_intraday_primary_key", "primary_key": list(primary_key)}],
            "method": "partitioned_physical_order",
        }
    invalid_predicate = """
      prior_symbol IS NOT NULL AND (
        symbol < prior_symbol
        OR (symbol = prior_symbol AND trade_date < prior_date)
        OR (symbol = prior_symbol AND trade_date = prior_date AND bar_time <= prior_time)
      )
    """
    ordered_cte = """
      WITH ordered AS (
        SELECT cast(symbol AS VARCHAR) AS symbol,
               cast(trade_date AS VARCHAR) AS trade_date,
               cast(bar_time AS VARCHAR) AS bar_time,
               lag(cast(symbol AS VARCHAR)) OVER () AS prior_symbol,
               lag(cast(trade_date AS VARCHAR)) OVER () AS prior_date,
               lag(cast(bar_time AS VARCHAR)) OVER () AS prior_time
        FROM read_parquet(?, union_by_name=true)
      )
    """
    null_count = 0
    invalid_count = 0
    examples: list[dict[str, Any]] = []
    for path in paths:
        row = con.execute(
            ordered_cte
            + f"""
              SELECT coalesce(sum(CASE WHEN symbol IS NULL OR trade_date IS NULL OR bar_time IS NULL THEN 1 ELSE 0 END),0),
                     coalesce(sum(CASE WHEN {invalid_predicate} THEN 1 ELSE 0 END),0)
              FROM ordered
            """,
            [[str(path)]],
        ).fetchone()
        null_count += int(row[0] or 0)
        shard_invalid = int(row[1] or 0)
        invalid_count += shard_invalid
        remaining = max(0, int(sample_limit) - len(examples))
        if shard_invalid and remaining:
            rows = con.execute(
                ordered_cte
                + f"""
                  SELECT symbol,trade_date,bar_time,prior_symbol,prior_date,prior_time
                  FROM ordered WHERE {invalid_predicate}
                  LIMIT {remaining}
                """,
                [[str(path)]],
            ).fetchdf().to_dict("records")
            for item in rows:
                item["shard"] = str(path)
            examples.extend(rows)

    ranged = [
        (index, str(entry.start_date), str(entry.end_date), path)
        for index, (path, entry) in enumerate(zip(paths, entries, strict=True))
        if str(entry.start_date) and str(entry.end_date)
    ]
    overlapping_pairs = [
        (left, right)
        for position, left in enumerate(ranged)
        for right in ranged[position + 1 :]
        if left[1] <= right[2] and right[1] <= left[2]
    ]
    symbol_sets: dict[int, set[str]] = {}
    for item in {part[0]: part for pair in overlapping_pairs for part in pair}.values():
        symbol_sets[item[0]] = {
            str(row[0])
            for row in con.execute(
                "SELECT DISTINCT cast(symbol AS VARCHAR) "
                "FROM read_parquet(?, union_by_name=true)",
                [[str(item[3])]],
            ).fetchall()
        }
    cross_shard_duplicates = 0
    verified_pair_count = 0
    for left, right in overlapping_pairs:
        common_symbols = sorted(symbol_sets[left[0]].intersection(symbol_sets[right[0]]))
        if not common_symbols:
            continue
        verified_pair_count += 1
        overlap_start = max(left[1], right[1])
        overlap_end = min(left[2], right[2])
        con.register(
            "qdp_intraday_overlap_symbols",
            pd.DataFrame({"symbol": common_symbols}),
        )
        try:
            duplicate_count = int(
                con.execute(
                    """
                    SELECT count(*)
                    FROM read_parquet(?, union_by_name=true) l
                    JOIN qdp_intraday_overlap_symbols s
                      ON cast(l.symbol AS VARCHAR)=s.symbol
                    JOIN read_parquet(?, union_by_name=true) r
                      ON cast(l.symbol AS VARCHAR)=cast(r.symbol AS VARCHAR)
                     AND cast(l.trade_date AS VARCHAR)=cast(r.trade_date AS VARCHAR)
                     AND cast(l.bar_time AS VARCHAR)=cast(r.bar_time AS VARCHAR)
                    WHERE cast(l.trade_date AS VARCHAR) BETWEEN ? AND ?
                      AND cast(r.trade_date AS VARCHAR) BETWEEN ? AND ?
                    """,
                    [
                        [str(left[3])],
                        [str(right[3])],
                        overlap_start,
                        overlap_end,
                        overlap_start,
                        overlap_end,
                    ],
                ).fetchone()[0]
            )
            cross_shard_duplicates += duplicate_count
            remaining = max(0, int(sample_limit) - len(examples))
            if duplicate_count and remaining:
                rows = con.execute(
                    """
                    SELECT cast(l.symbol AS VARCHAR) AS symbol,
                           cast(l.trade_date AS VARCHAR) AS trade_date,
                           cast(l.bar_time AS VARCHAR) AS bar_time
                    FROM read_parquet(?, union_by_name=true) l
                    JOIN qdp_intraday_overlap_symbols s
                      ON cast(l.symbol AS VARCHAR)=s.symbol
                    JOIN read_parquet(?, union_by_name=true) r
                      ON cast(l.symbol AS VARCHAR)=cast(r.symbol AS VARCHAR)
                     AND cast(l.trade_date AS VARCHAR)=cast(r.trade_date AS VARCHAR)
                     AND cast(l.bar_time AS VARCHAR)=cast(r.bar_time AS VARCHAR)
                    WHERE cast(l.trade_date AS VARCHAR) BETWEEN ? AND ?
                      AND cast(r.trade_date AS VARCHAR) BETWEEN ? AND ?
                    LIMIT ?
                    """,
                    [
                        [str(left[3])],
                        [str(right[3])],
                        overlap_start,
                        overlap_end,
                        overlap_start,
                        overlap_end,
                        remaining,
                    ],
                ).fetchdf().to_dict("records")
                for row in rows:
                    row["left_shard"] = str(left[3])
                    row["right_shard"] = str(right[3])
                examples.extend(rows)
        finally:
            con.unregister("qdp_intraday_overlap_symbols")
    range_overlap_count = len(overlapping_pairs)
    duplicate_rows = invalid_count + cross_shard_duplicates
    return {
        "status": "ok" if not null_count and not duplicate_rows else "error",
        "null_key_rows": null_count,
        "duplicate_rows": duplicate_rows,
        "range_overlap_count": range_overlap_count,
        "verified_overlap_pair_count": verified_pair_count,
        "cross_shard_duplicate_rows": cross_shard_duplicates,
        "examples": examples,
        "method": "partitioned_physical_order+verified_cross_shard_keys",
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


def _status_daily_partition_check(
    con: Any,
    *,
    status_paths: Sequence[Path],
    daily_paths: Sequence[Path],
    start_date: str,
    end_date: str,
    sample_limit: int,
) -> dict[str, Any]:
    base = """
      WITH daily AS (
        SELECT cast(symbol AS VARCHAR) AS symbol,
               cast(trade_date AS VARCHAR) AS trade_date,
               max(try_cast(volume AS DOUBLE)) AS volume
        FROM read_parquet(?, union_by_name=true)
        WHERE cast(trade_date AS VARCHAR) BETWEEN ? AND ?
        GROUP BY symbol,trade_date
      ), joined AS (
        SELECT cast(s.symbol AS VARCHAR) AS symbol,
               cast(s.trade_date AS VARCHAR) AS trade_date,
               try_cast(s.is_suspended AS BOOLEAN) AS old_suspended,
               d.volume AS daily_volume,
               (d.symbol IS NULL OR coalesce(d.volume,0)<=0) AS expected_suspended
        FROM read_parquet(?, union_by_name=true) s
        LEFT JOIN daily d
          ON cast(s.symbol AS VARCHAR)=d.symbol
         AND cast(s.trade_date AS VARCHAR)=d.trade_date
        WHERE cast(s.trade_date AS VARCHAR) BETWEEN ? AND ?
      )
    """
    params = [
        [str(path) for path in daily_paths],
        str(start_date),
        str(end_date),
        [str(path) for path in status_paths],
        str(start_date),
        str(end_date),
    ]
    row = con.execute(
        base
        + """
          SELECT count(*) AS status_rows,
                 count(*) FILTER (WHERE old_suspended IS NULL) AS null_flags,
                 count(*) FILTER (
                   WHERE coalesce(old_suspended,false)<>expected_suspended
                 ) AS mismatches,
                 count(*) FILTER (
                   WHERE coalesce(old_suspended,false)=false AND expected_suspended
                 ) AS unsuspended_without_positive_daily,
                 count(*) FILTER (
                   WHERE old_suspended=true AND NOT expected_suspended
                 ) AS suspended_with_positive_daily
          FROM joined
        """,
        params,
    ).fetchone()
    remaining = max(0, int(sample_limit))
    examples: list[dict[str, Any]] = []
    if remaining and (int(row[1] or 0) or int(row[2] or 0)):
        examples = con.execute(
            base
            + f"""
              SELECT symbol,trade_date,old_suspended,expected_suspended,daily_volume,
                     CASE
                       WHEN old_suspended IS NULL THEN 'null_suspension_flag'
                       WHEN coalesce(old_suspended,false)=false AND expected_suspended
                         THEN 'unsuspended_without_positive_daily'
                       WHEN old_suspended=true AND NOT expected_suspended
                         THEN 'suspended_with_positive_daily'
                       ELSE 'other'
                     END AS reason
              FROM joined
              WHERE old_suspended IS NULL
                 OR coalesce(old_suspended,false)<>expected_suspended
              ORDER BY trade_date,symbol
              LIMIT {remaining}
            """,
            params,
        ).fetchdf().to_dict("records")
    return {
        "status_row_count": int(row[0] or 0),
        "null_suspension_flag_count": int(row[1] or 0),
        "semantic_mismatch_count": int(row[2] or 0),
        "unsuspended_without_positive_daily_count": int(row[3] or 0),
        "suspended_with_positive_daily_count": int(row[4] or 0),
        "examples": examples,
    }


def _status_daily_consistency_check(
    con: Any,
    *,
    root: Path,
    daily: DatasetManifest,
    status: DatasetManifest,
    sample_limit: int,
) -> dict[str, Any]:
    totals = {
        "status_row_count": 0,
        "null_suspension_flag_count": 0,
        "semantic_mismatch_count": 0,
        "unsuspended_without_positive_daily_count": 0,
        "suspended_with_positive_daily_count": 0,
    }
    examples: list[dict[str, Any]] = []
    daily_entries = [
        (resolve_manifest_path(item.path, root=root), item) for item in daily.shards
    ]
    for status_entry in status.shards:
        start_date = str(status_entry.start_date or "")
        end_date = str(status_entry.end_date or "")
        if not start_date or not end_date:
            raise ValueError("status_consistency_shard_range_missing")
        status_path = resolve_manifest_path(status_entry.path, root=root)
        daily_paths = [
            path
            for path, entry in daily_entries
            if (
                (not entry.start_date or str(entry.start_date) <= end_date)
                and (not entry.end_date or str(entry.end_date) >= start_date)
            )
        ]
        if not daily_paths:
            raise ValueError(f"status_consistency_daily_paths_missing:{start_date}")
        result = _status_daily_partition_check(
            con,
            status_paths=[status_path],
            daily_paths=daily_paths,
            start_date=start_date,
            end_date=end_date,
            sample_limit=max(0, int(sample_limit) - len(examples)),
        )
        for key in totals:
            totals[key] += int(result[key])
        examples.extend(result["examples"])
    return {
        "status": (
            "ok"
            if not totals["semantic_mismatch_count"]
            and not totals["null_suspension_flag_count"]
            else "error"
        ),
        **totals,
        "semantics": "is_suspended iff daily row is absent or volume<=0",
        "examples": examples,
    }


def _daily_intraday_consistency_check(
    con: Any,
    *,
    root: Path,
    daily: DatasetManifest,
    intraday: DatasetManifest,
    history: DatasetManifest | None = None,
    sample_limit: int,
) -> dict[str, Any]:
    totals = {
        "positive_daily_count": 0,
        "complete_positive_daily_count": 0,
        "missing_positive_daily_count": 0,
        "missing_daily_for_5m_count": 0,
        "suspended_nonzero_5m_count": 0,
        "price_mismatch_day_count": 0,
        "flow_mismatch_day_count": 0,
        "volume_100x_day_count": 0,
        "invalid_day_count": 0,
    }
    examples: list[dict[str, Any]] = []
    daily_entries = [
        (resolve_manifest_path(item.path, root=root), item) for item in daily.shards
    ]
    history_paths = (
        [str(resolve_manifest_path(item.path, root=root)) for item in history.shards]
        if history is not None
        else []
    )
    for intraday_entry in intraday.shards:
        start_date = str(intraday_entry.start_date or "")
        end_date = str(intraday_entry.end_date or "")
        if not start_date or not end_date:
            raise ValueError("intraday_consistency_shard_range_missing")
        intraday_path = resolve_manifest_path(intraday_entry.path, root=root)
        daily_paths = [
            path
            for path, entry in daily_entries
            if (
                (not entry.start_date or str(entry.start_date) <= end_date)
                and (not entry.end_date or str(entry.end_date) >= start_date)
            )
        ]
        if not daily_paths:
            raise ValueError(f"intraday_consistency_daily_paths_missing:{start_date}")
        five_source = """
          five_source AS (
            SELECT upper(cast(symbol AS VARCHAR)) AS source_symbol,
                   cast(trade_date AS VARCHAR) AS trade_date,
                   count(*) AS bars,
                   arg_min(try_cast(open AS DOUBLE),cast(bar_time AS VARCHAR)) AS open,
                   max(try_cast(high AS DOUBLE)) AS high,
                   min(try_cast(low AS DOUBLE)) AS low,
                   arg_max(try_cast(close AS DOUBLE),cast(bar_time AS VARCHAR)) AS close,
                   sum(try_cast(volume AS DOUBLE)) AS volume,
                   sum(try_cast(amount AS DOUBLE)) AS amount
            FROM read_parquet(?, union_by_name=true)
            GROUP BY source_symbol,trade_date
          )
        """
        if history_paths:
            five_tables = (
                """
                  lifecycle_history AS (
                    SELECT cast(security_id AS VARCHAR) AS security_id,
                           upper(cast(symbol AS VARCHAR)) AS symbol,
                           cast(effective_from AS VARCHAR) AS effective_from,
                           cast(effective_to AS VARCHAR) AS effective_to
                    FROM read_parquet(?, union_by_name=true)
                  ), lifecycle_multi_ids AS (
                    SELECT security_id FROM lifecycle_history
                    GROUP BY security_id HAVING count(DISTINCT symbol)>1
                  ), lifecycle AS (
                    SELECT h.* FROM lifecycle_history h
                    JOIN lifecycle_multi_ids m USING(security_id)
                  ),
                """
                + five_source
                + """,
                  five_mapped AS (
                    SELECT coalesce(target.symbol,f.source_symbol) AS canonical_symbol,
                           f.*
                    FROM five_source f
                    LEFT JOIN lifecycle source
                      ON f.source_symbol=source.symbol
                    LEFT JOIN lifecycle target
                      ON source.security_id=target.security_id
                     AND f.trade_date BETWEEN target.effective_from AND target.effective_to
                  ), five AS (
                    SELECT canonical_symbol AS symbol,trade_date,bars,open,high,low,
                           close,volume,amount
                    FROM five_mapped
                    QUALIFY row_number() OVER(
                      PARTITION BY canonical_symbol,trade_date
                      ORDER BY CASE WHEN source_symbol=canonical_symbol THEN 0 ELSE 1 END,
                               source_symbol
                    )=1
                  )
                """
            )
            five_params: list[Any] = [history_paths, [str(intraday_path)]]
        else:
            five_tables = (
                five_source
                + """,
                  five AS (
                    SELECT source_symbol AS symbol,trade_date,bars,open,high,low,
                           close,volume,amount
                    FROM five_source
                  )
                """
            )
            five_params = [[str(intraday_path)]]
        joined = f"""
          WITH {five_tables}, day AS (
            SELECT cast(symbol AS VARCHAR) AS symbol,
                   cast(trade_date AS VARCHAR) AS trade_date,
                   try_cast(open AS DOUBLE) AS open,
                   try_cast(high AS DOUBLE) AS high,
                   try_cast(low AS DOUBLE) AS low,
                   try_cast(close AS DOUBLE) AS close,
                   try_cast(volume AS DOUBLE) AS volume,
                   try_cast(amount AS DOUBLE) AS amount
            FROM read_parquet(?, union_by_name=true)
            WHERE cast(trade_date AS VARCHAR) BETWEEN ? AND ?
          ), joined AS (
            SELECT coalesce(d.symbol,f.symbol) AS symbol,
                   coalesce(d.trade_date,f.trade_date) AS trade_date,
                   d.open AS dopen,d.high AS dhigh,d.low AS dlow,d.close AS dclose,
                   d.volume AS dvolume,d.amount AS damount,
                   f.bars,f.open AS fopen,f.high AS fhigh,f.low AS flow,f.close AS fclose,
                   f.volume AS fvolume,f.amount AS famount,
                   CASE WHEN d.volume>0 THEN abs(f.volume-d.volume)/greatest(abs(d.volume),1) END AS volume_rel,
                   CASE WHEN d.amount>0 THEN abs(f.amount-d.amount)/greatest(abs(d.amount),1) END AS amount_rel,
                   CASE WHEN d.volume>0 THEN f.volume/d.volume END AS volume_ratio,
                   greatest(
                     abs(f.open-d.open)/greatest(abs(d.open),1),
                     abs(f.high-d.high)/greatest(abs(d.high),1),
                     abs(f.low-d.low)/greatest(abs(d.low),1),
                     abs(f.close-d.close)/greatest(abs(d.close),1)
                   ) AS price_rel
            FROM day d FULL OUTER JOIN five f USING(symbol,trade_date)
          )
        """
        params = [
            *five_params,
            [str(path) for path in daily_paths],
            start_date,
            end_date,
        ]
        row = con.execute(
            joined
            + """
              SELECT
                count(*) FILTER (WHERE dvolume>0) AS positive_daily,
                count(*) FILTER (WHERE dvolume>0 AND bars=48) AS complete_positive,
                count(*) FILTER (WHERE dvolume>0 AND bars IS NULL) AS missing_positive,
                count(*) FILTER (WHERE dvolume IS NULL AND bars IS NOT NULL) AS missing_daily,
                count(*) FILTER (WHERE coalesce(dvolume,0)<=0 AND bars IS NOT NULL) AS suspended_nonzero,
                count(*) FILTER (WHERE dvolume>0 AND bars IS NOT NULL AND price_rel>0.10) AS price_mismatch,
                count(*) FILTER (WHERE dvolume>0 AND bars IS NOT NULL AND volume_rel>0.05 AND amount_rel>0.05) AS flow_mismatch,
                count(*) FILTER (WHERE dvolume>0 AND bars IS NOT NULL AND (volume_ratio BETWEEN 95 AND 105 OR volume_ratio BETWEEN 0.0095 AND 0.0105)) AS volume_100x,
                count(*) FILTER (WHERE bars IS NOT NULL AND (
                  dvolume IS NULL OR coalesce(dvolume,0)<=0 OR price_rel>0.10
                  OR (volume_rel>0.05 AND amount_rel>0.05)
                  OR volume_ratio BETWEEN 95 AND 105 OR volume_ratio BETWEEN 0.0095 AND 0.0105
                )) AS invalid_days
              FROM joined
            """,
            params,
        ).fetchone()
        keys = list(totals)
        for key, value in zip(keys, row, strict=True):
            totals[key] += int(value or 0)
        remaining = max(0, int(sample_limit) - len(examples))
        if remaining and (int(row[2] or 0) or int(row[8] or 0)):
            sample = con.execute(
                joined
                + f"""
                  SELECT symbol,trade_date,bars,dvolume,fvolume,damount,famount,
                         price_rel,volume_rel,amount_rel,volume_ratio,
                         CASE
                           WHEN dvolume>0 AND bars IS NULL THEN 'missing_positive_daily'
                           WHEN dvolume IS NULL AND bars IS NOT NULL THEN 'missing_daily'
                           WHEN coalesce(dvolume,0)<=0 AND bars IS NOT NULL THEN 'suspended_nonzero_5m'
                           WHEN volume_ratio BETWEEN 95 AND 105 OR volume_ratio BETWEEN 0.0095 AND 0.0105 THEN 'volume_100x'
                           WHEN price_rel>0.10 THEN 'price_mismatch'
                           WHEN volume_rel>0.05 AND amount_rel>0.05 THEN 'flow_mismatch'
                           ELSE 'other'
                         END AS reason
                  FROM joined
                  WHERE (dvolume>0 AND bars IS NULL) OR (bars IS NOT NULL AND (
                    dvolume IS NULL OR coalesce(dvolume,0)<=0 OR price_rel>0.10
                    OR (volume_rel>0.05 AND amount_rel>0.05)
                    OR volume_ratio BETWEEN 95 AND 105 OR volume_ratio BETWEEN 0.0095 AND 0.0105
                  ))
                  LIMIT {remaining}
                """,
                params,
            ).fetchdf().to_dict("records")
            examples.extend(sample)
    positive = totals["positive_daily_count"]
    complete = totals["complete_positive_daily_count"]
    return {
        "status": "ok" if not totals["invalid_day_count"] else "error",
        **totals,
        "complete_coverage_ratio": float(complete / positive) if positive else 1.0,
        "examples": examples,
    }


def _factor_semantic_check(
    con: Any,
    *,
    root: Path,
    daily: DatasetManifest,
    factor: DatasetManifest,
    sample_limit: int,
) -> dict[str, Any]:
    daily_paths = [str(resolve_manifest_path(item.path, root=root)) for item in daily.shards]
    factor_paths = [str(resolve_manifest_path(item.path, root=root)) for item in factor.shards]
    base = """
      WITH factors AS (
        SELECT cast(symbol AS VARCHAR) AS symbol,
               cast(trade_date AS VARCHAR) AS trade_date,
               try_cast(adjust_factor AS DOUBLE) AS factor,
               lag(try_cast(adjust_factor AS DOUBLE)) OVER(
                 PARTITION BY symbol ORDER BY cast(trade_date AS VARCHAR)
               ) AS prior_factor
        FROM read_parquet(?, union_by_name=true)
      ), prices AS (
        SELECT cast(symbol AS VARCHAR) AS symbol,
               cast(trade_date AS VARCHAR) AS trade_date,
               try_cast(open AS DOUBLE) AS open,
               lag(try_cast(close AS DOUBLE)) OVER(
                 PARTITION BY symbol ORDER BY cast(trade_date AS VARCHAR)
               ) AS prior_close,
               lag(cast(trade_date AS VARCHAR)) OVER(
                 PARTITION BY symbol ORDER BY cast(trade_date AS VARCHAR)
               ) AS prior_date
        FROM read_parquet(?, union_by_name=true)
      ), changes AS (
        SELECT f.symbol,f.trade_date,f.factor/f.prior_factor AS factor_ratio,
               p.open/p.prior_close AS raw_open_ratio,
               (f.factor/f.prior_factor)*(p.open/p.prior_close) AS adjusted_open_ratio,
               date_diff('day',try_cast(p.prior_date AS DATE),try_cast(f.trade_date AS DATE)) AS gap_days
        FROM factors f JOIN prices p USING(symbol,trade_date)
        WHERE f.prior_factor>0 AND p.prior_close>0
          AND (f.factor/f.prior_factor<0.99 OR f.factor/f.prior_factor>1.01)
      )
    """
    params = [factor_paths, daily_paths]
    row = con.execute(
        base
        + """
          SELECT count(*) AS changes,
                 count(*) FILTER (WHERE (adjusted_open_ratio<0.70 OR adjusted_open_ratio>1.30)
                   AND gap_days<=10 AND raw_open_ratio BETWEEN 0.70 AND 1.30) AS factor_induced,
                 count(*) FILTER (WHERE (adjusted_open_ratio<0.70 OR adjusted_open_ratio>1.30)
                   AND NOT (gap_days<=10 AND raw_open_ratio BETWEEN 0.70 AND 1.30)) AS raw_discontinuity,
                 coalesce(max(abs(adjusted_open_ratio-1)),0) AS max_error,
                 count(*) FILTER (WHERE symbol='600076.SH' AND substr(trade_date,1,4)='2024') AS regression_changes,
                 coalesce(max(abs(adjusted_open_ratio-1)) FILTER (
                   WHERE symbol='600076.SH' AND substr(trade_date,1,4)='2024'
                 ),0) AS regression_max_error
          FROM changes
        """,
        params,
    ).fetchone()
    excessive = con.execute(
        base
        + """
          SELECT count(*) FROM (
            SELECT symbol,substr(trade_date,1,4) AS year,count(*) AS changes
            FROM changes GROUP BY symbol,year HAVING count(*)>12
          )
        """,
        params,
    ).fetchone()[0]
    examples = []
    if int(row[1] or 0):
        examples = con.execute(
            base
            + f"""
              SELECT symbol,trade_date,gap_days,factor_ratio,raw_open_ratio,adjusted_open_ratio
              FROM changes
              WHERE (adjusted_open_ratio<0.70 OR adjusted_open_ratio>1.30)
                AND gap_days<=10 AND raw_open_ratio BETWEEN 0.70 AND 1.30
              ORDER BY abs(adjusted_open_ratio-1) DESC
              LIMIT {int(sample_limit)}
            """,
            params,
        ).fetchdf().to_dict("records")
    raw_examples = []
    if int(row[2] or 0):
        raw_examples = con.execute(
            base
            + f"""
              SELECT symbol,trade_date,gap_days,factor_ratio,raw_open_ratio,adjusted_open_ratio
              FROM changes
              WHERE (adjusted_open_ratio<0.70 OR adjusted_open_ratio>1.30)
                AND NOT (gap_days<=10 AND raw_open_ratio BETWEEN 0.70 AND 1.30)
              ORDER BY abs(adjusted_open_ratio-1) DESC
              LIMIT {int(sample_limit)}
            """,
            params,
        ).fetchdf().to_dict("records")
    return {
        "status": "ok" if not int(row[1] or 0) and not int(excessive or 0) else "error",
        "factor_change_count": int(row[0] or 0),
        "uncompensated_change_count": int(row[1] or 0),
        "raw_discontinuity_count": int(row[2] or 0),
        "max_compensation_error": float(row[3] or 0),
        "excessive_change_symbol_year_count": int(excessive or 0),
        "regression_600076_2024_change_count": int(row[4] or 0),
        "regression_600076_2024_max_compensation_error": float(row[5] or 0),
        "examples": examples,
        "raw_discontinuity_examples": raw_examples,
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


def _identity_history_consistency_check(
    con: Any,
    *,
    root: Path,
    identity: DatasetManifest,
    history: DatasetManifest,
    sample_limit: int,
) -> dict[str, Any]:
    identity_paths = [
        str(resolve_manifest_path(item.path, root=root)) for item in identity.shards
    ]
    history_paths = [
        str(resolve_manifest_path(item.path, root=root)) for item in history.shards
    ]
    sql = """
      WITH identities AS (
        SELECT DISTINCT cast(security_id AS VARCHAR) AS security_id
        FROM read_parquet(?, union_by_name=true)
      ), histories AS (
        SELECT DISTINCT cast(security_id AS VARCHAR) AS security_id,
               upper(trim(cast(symbol AS VARCHAR))) AS symbol
        FROM read_parquet(?, union_by_name=true)
      )
    """
    row = con.execute(
        sql
        + """
        SELECT
          (SELECT count(*) FROM histories h ANTI JOIN identities i USING(security_id)),
          (SELECT count(*) FROM identities i ANTI JOIN histories h USING(security_id)),
          (SELECT count(*) FROM (
             SELECT symbol FROM histories GROUP BY symbol
             HAVING count(DISTINCT security_id)>1
           ) conflicts)
        """,
        [identity_paths, history_paths],
    ).fetchone()
    orphan_history = int(row[0] or 0)
    identity_without_history = int(row[1] or 0)
    symbol_conflicts = int(row[2] or 0)
    examples = con.execute(
        sql
        + f"""
        SELECT 'orphan_history' AS reason,h.security_id,h.symbol
        FROM histories h ANTI JOIN identities i USING(security_id)
        LIMIT {int(sample_limit)}
        """,
        [identity_paths, history_paths],
    ).fetchdf().to_dict("records") if orphan_history else []
    return {
        "status": (
            "ok"
            if not orphan_history and not identity_without_history and not symbol_conflicts
            else "error"
        ),
        "orphan_symbol_history_count": orphan_history,
        "identity_without_symbol_history_count": identity_without_history,
        "symbol_multiple_identity_count": symbol_conflicts,
        "examples": examples,
    }


def _long_suspension_check(
    con: Any,
    *,
    root: Path,
    status: DatasetManifest,
    calendar: DatasetManifest,
    sample_limit: int,
    minimum_open_days: int = 20,
) -> dict[str, Any]:
    status_paths = [
        str(resolve_manifest_path(item.path, root=root)) for item in status.shards
    ]
    calendar_paths = [
        str(resolve_manifest_path(item.path, root=root)) for item in calendar.shards
    ]
    base = """
      WITH open_dates AS (
        SELECT DISTINCT cast(trade_date AS VARCHAR) AS trade_date
        FROM read_parquet(?, union_by_name=true)
        WHERE coalesce(try_cast(is_open AS BOOLEAN), false)
      ), ordered AS (
        SELECT cast(s.symbol AS VARCHAR) AS symbol,
               cast(s.trade_date AS VARCHAR) AS trade_date,
               coalesce(try_cast(s.is_suspended AS BOOLEAN), false) AS is_suspended,
               lead(coalesce(try_cast(s.is_suspended AS BOOLEAN), false)) OVER(
                 PARTITION BY cast(s.symbol AS VARCHAR)
                 ORDER BY cast(s.trade_date AS VARCHAR)
               ) AS next_is_suspended,
               sum(CASE WHEN coalesce(try_cast(s.is_suspended AS BOOLEAN), false)
                        THEN 0 ELSE 1 END) OVER(
                 PARTITION BY cast(s.symbol AS VARCHAR)
                 ORDER BY cast(s.trade_date AS VARCHAR)
                 ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW
               ) AS segment_id
        FROM read_parquet(?, union_by_name=true) s
        JOIN open_dates d ON d.trade_date=cast(s.trade_date AS VARCHAR)
      ), runs AS (
        SELECT symbol,min(trade_date) AS start_date,max(trade_date) AS end_date,
               count(*) AS suspended_open_days,
               max(CASE WHEN next_is_suspended=false THEN 1 ELSE 0 END) AS has_reopen_break
        FROM ordered
        WHERE is_suspended
        GROUP BY symbol,segment_id
        HAVING count(*) >= ?
      )
    """
    params = [calendar_paths, status_paths, int(minimum_open_days)]
    row = con.execute(
        base
        + """
        SELECT count(*),count(DISTINCT symbol),
               coalesce(sum(has_reopen_break),0),
               coalesce(sum(suspended_open_days),0)
        FROM runs
        """,
        params,
    ).fetchone()
    examples = con.execute(
        base
        + f"""
        SELECT symbol,start_date,end_date,suspended_open_days,
               cast(has_reopen_break AS BOOLEAN) AS continuity_break
        FROM runs ORDER BY suspended_open_days DESC,symbol,start_date
        LIMIT {int(sample_limit)}
        """,
        params,
    ).fetchdf().to_dict("records")
    return {
        "status": "ok",
        "minimum_suspended_open_days": int(minimum_open_days),
        "long_suspension_interval_count": int(row[0] or 0),
        "affected_symbol_count": int(row[1] or 0),
        "continuity_break_count": int(row[2] or 0),
        "long_suspension_open_day_count": int(row[3] or 0),
        "research_rule": (
            "lookbacks, forward labels, and execution tails may not cross a "
            "qualifying reopen break"
        ),
        "examples": examples,
    }


def _reopen_discontinuity_check(
    con: Any,
    *,
    root: Path,
    daily: DatasetManifest,
    factor: DatasetManifest,
    status: DatasetManifest,
    calendar: DatasetManifest,
    identity: DatasetManifest,
    sample_limit: int,
) -> dict[str, Any]:
    daily_paths = [
        str(resolve_manifest_path(item.path, root=root)) for item in daily.shards
    ]
    factor_paths = [
        str(resolve_manifest_path(item.path, root=root)) for item in factor.shards
    ]
    status_paths = [
        str(resolve_manifest_path(item.path, root=root)) for item in status.shards
    ]
    calendar_paths = [
        str(resolve_manifest_path(item.path, root=root)) for item in calendar.shards
    ]
    identity_paths = [
        str(resolve_manifest_path(item.path, root=root)) for item in identity.shards
    ]
    base = """
      WITH prices AS (
        SELECT cast(symbol AS VARCHAR) AS symbol,
               cast(trade_date AS VARCHAR) AS trade_date,
               try_cast(open AS DOUBLE) AS open,
               lag(try_cast(close AS DOUBLE)) OVER(
                 PARTITION BY cast(symbol AS VARCHAR)
                 ORDER BY cast(trade_date AS VARCHAR)
               ) AS prior_close,
               lag(cast(trade_date AS VARCHAR)) OVER(
                 PARTITION BY cast(symbol AS VARCHAR)
                 ORDER BY cast(trade_date AS VARCHAR)
               ) AS prior_trade_date,
               row_number() OVER(
                 PARTITION BY cast(symbol AS VARCHAR)
                 ORDER BY cast(trade_date AS VARCHAR)
               ) AS observation_number,
               min(cast(trade_date AS VARCHAR)) OVER(
                 PARTITION BY cast(symbol AS VARCHAR)
               ) AS first_trade_date
        FROM read_parquet(?, union_by_name=true)
        WHERE try_cast(close AS DOUBLE)>0
      ), factors AS (
        SELECT cast(symbol AS VARCHAR) AS symbol,
               cast(trade_date AS VARCHAR) AS trade_date,
               try_cast(adjust_factor AS DOUBLE) AS factor,
               lag(try_cast(adjust_factor AS DOUBLE)) OVER(
                 PARTITION BY cast(symbol AS VARCHAR)
                 ORDER BY cast(trade_date AS VARCHAR)
               ) AS prior_factor
        FROM read_parquet(?, union_by_name=true)
      ), gaps AS (
        SELECT p.symbol,p.prior_trade_date,p.trade_date,
               p.observation_number,p.first_trade_date,i.list_date,
               p.open/p.prior_close AS raw_open_ratio,
               f.factor/f.prior_factor AS factor_ratio,
               (p.open/p.prior_close)*(f.factor/f.prior_factor) AS adjusted_open_ratio
        FROM prices p
        LEFT JOIN factors f USING(symbol,trade_date)
        LEFT JOIN (
          SELECT upper(trim(cast(current_symbol AS VARCHAR))) AS symbol,
                 cast(list_date AS VARCHAR) AS list_date
          FROM read_parquet(?, union_by_name=true)
        ) i USING(symbol)
        WHERE p.prior_close>0
          AND (p.open/p.prior_close<0.70 OR p.open/p.prior_close>1.30)
      ), open_dates AS (
        SELECT DISTINCT cast(trade_date AS VARCHAR) AS trade_date
        FROM read_parquet(?, union_by_name=true)
        WHERE coalesce(try_cast(is_open AS BOOLEAN), false)
      ), status_rows AS (
        SELECT cast(symbol AS VARCHAR) AS symbol,
               cast(trade_date AS VARCHAR) AS trade_date,
               coalesce(try_cast(is_suspended AS BOOLEAN), false) AS is_suspended
        FROM read_parquet(?, union_by_name=true)
      ), gap_status AS (
        SELECT g.symbol,g.prior_trade_date,g.trade_date,g.observation_number,
               g.first_trade_date,g.list_date,g.raw_open_ratio,
               g.factor_ratio,g.adjusted_open_ratio,
               count(d.trade_date) AS intervening_open_days,
               count(s.trade_date) AS covered_status_days,
               count(s.trade_date) FILTER (WHERE s.is_suspended) AS suspended_status_days,
               count(s.trade_date) FILTER (WHERE NOT s.is_suspended) AS tradable_status_days
        FROM gaps g
        LEFT JOIN open_dates d
          ON d.trade_date>g.prior_trade_date AND d.trade_date<g.trade_date
        LEFT JOIN status_rows s
          ON s.symbol=g.symbol AND s.trade_date=d.trade_date
        GROUP BY g.symbol,g.prior_trade_date,g.trade_date,g.observation_number,
                 g.first_trade_date,g.list_date,g.raw_open_ratio,
                 g.factor_ratio,g.adjusted_open_ratio
      ), classified AS (
        SELECT *,
          CASE
            WHEN first_trade_date=list_date AND observation_number<=5
              THEN 'listing_price_discovery'
            WHEN intervening_open_days>0
             AND covered_status_days=intervening_open_days
             AND suspended_status_days=intervening_open_days
              THEN 'reopen_discontinuity'
            WHEN factor_ratio IS NOT NULL
             AND (factor_ratio<0.99 OR factor_ratio>1.01)
             AND adjusted_open_ratio BETWEEN 0.70 AND 1.30
              THEN 'factor_compensated_corporate_action'
            ELSE 'unexpected_discontinuity'
          END AS classification
        FROM gap_status
      )
    """
    params = [daily_paths, factor_paths, identity_paths, calendar_paths, status_paths]
    row = con.execute(
        base
        + """
        SELECT count(*),
               count(*) FILTER (WHERE classification='reopen_discontinuity'),
               count(*) FILTER (WHERE classification='factor_compensated_corporate_action'),
               count(*) FILTER (WHERE classification='listing_price_discovery'),
               count(*) FILTER (WHERE classification='unexpected_discontinuity'),
               count(*) FILTER (WHERE intervening_open_days>covered_status_days),
               count(*) FILTER (WHERE tradable_status_days>0)
        FROM classified
        """,
        params,
    ).fetchone()
    examples = con.execute(
        base
        + f"""
        SELECT symbol,prior_trade_date,trade_date,raw_open_ratio,factor_ratio,
               adjusted_open_ratio,intervening_open_days,covered_status_days,
               suspended_status_days,classification
        FROM classified
        ORDER BY CASE WHEN classification='unexpected_discontinuity' THEN 0 ELSE 1 END,
                 symbol,trade_date
        LIMIT {int(sample_limit)}
        """,
        params,
    ).fetchdf().to_dict("records")
    return {
        "status": "ok" if not int(row[4] or 0) else "error",
        "raw_discontinuity_count": int(row[0] or 0),
        "reopen_discontinuity_count": int(row[1] or 0),
        "factor_compensated_corporate_action_count": int(row[2] or 0),
        "listing_price_discovery_count": int(row[3] or 0),
        "unexpected_discontinuity_count": int(row[4] or 0),
        "status_missing_interval_count": int(row[5] or 0),
        "tradable_day_inside_gap_count": int(row[6] or 0),
        "examples": examples,
    }


def _audit_temp_directory(workspace: Path):
    runtime = workspace / "quant_data_platform" / "data" / "qdp_runtime"
    runtime.mkdir(parents=True, exist_ok=True)
    return tempfile.TemporaryDirectory(prefix="audit_spill_", dir=str(runtime))


def _auxiliary_semantics_check(
    con: Any,
    *,
    root: Path,
    manifests: Mapping[str, DatasetManifest],
    sample_limit: int,
) -> dict[str, Any]:
    del sample_limit

    def register(domain: str) -> str:
        name = "aux_" + domain.replace("-", "_")
        paths = [
            resolve_manifest_path(item.path, root=root).resolve()
            for item in manifests[domain].shards
        ]
        literals = ",".join("'" + str(path).replace("'", "''") + "'" for path in paths)
        con.execute(
            f"CREATE OR REPLACE TEMP VIEW {_q(name)} AS "
            f"SELECT * FROM read_parquet([{literals}], union_by_name=true)"
        )
        return _q(name)

    tables = {
        domain: register(domain)
        for domain in (
            "market_daily_raw",
            "universe_snapshot",
            "adjust_factor",
            *AUXILIARY_DOMAINS,
        )
    }
    target = str(manifests["market_daily_raw"].end_date or "")
    daily = tables["market_daily_raw"]
    checks: dict[str, Any] = {}
    blocking = 0

    for domain in ("industry_concept", "share_capital", "valuation"):
        table = tables[domain]
        missing = int(
            con.execute(
                f"SELECT count(*) FROM (SELECT symbol,trade_date FROM {daily} "
                "WHERE trade_date<=? EXCEPT SELECT symbol,trade_date FROM "
                + table
                + ")",
                [target],
            ).fetchone()[0]
        )
        extra = int(
            con.execute(
                f"SELECT count(*) FROM (SELECT symbol,trade_date FROM {table} "
                "EXCEPT SELECT symbol,trade_date FROM "
                + daily
                + " WHERE trade_date<=?)",
                [target],
            ).fetchone()[0]
        )
        checks[f"{domain}_missing_daily_keys"] = missing
        checks[f"{domain}_extra_daily_keys"] = extra
        blocking += missing + extra

    industry = tables["industry_concept"]
    industry_invalid = int(
        con.execute(
            f"SELECT count(*) FROM {industry} WHERE "
            "coalesce(trim(industry),'')='' OR "
            "coalesce(trim(industry_standard),'')='' OR "
            "industry_fill_method NOT IN ('direct_snapshot','prior_ffill','unavailable') OR "
            "try_cast(industry_source_date AS DATE)>try_cast(trade_date AS DATE)"
        ).fetchone()[0]
    )
    checks["industry_invalid_or_future_count"] = industry_invalid
    blocking += industry_invalid

    share = tables["share_capital"]
    share_invalid = int(
        con.execute(
            f"SELECT count(*) FROM {share} WHERE total_share<=0 OR float_share<0 OR "
            "float_share>total_share OR restricted_share<0 OR "
            "abs(restricted_share-greatest(total_share-float_share,0))>1e-6 OR "
            "try_cast(total_share_source_date AS DATE)>try_cast(trade_date AS DATE) OR "
            "try_cast(float_share_source_date AS DATE)>try_cast(trade_date AS DATE) OR "
            "try_cast(restricted_share_source_date AS DATE)>try_cast(trade_date AS DATE)"
        ).fetchone()[0]
    )
    checks["share_capital_invalid_or_future_count"] = share_invalid
    blocking += share_invalid

    valuation = tables["valuation"]
    valuation_formula = int(
        con.execute(
            f"""
            SELECT count(*)
            FROM {valuation} v
            JOIN {daily} d USING(symbol,trade_date)
            JOIN {share} s USING(symbol,trade_date)
            WHERE abs(v.total_mv-d.close*s.total_share)>
                    greatest(abs(d.close*s.total_share)*1e-8,1e-6)
               OR abs(v.circ_mv-d.close*s.float_share)>
                    greatest(abs(d.close*s.float_share)*1e-8,1e-6)
            """
        ).fetchone()[0]
    )
    valuation_nulls = con.execute(
        f"SELECT count(*) FILTER(WHERE pe IS NULL), "
        "count(*) FILTER(WHERE pb IS NULL), "
        "count(*) FILTER(WHERE turnover_rate IS NULL) FROM " + valuation
    ).fetchone()
    checks["valuation_formula_error_count"] = valuation_formula
    checks["valuation_provider_null_counts"] = {
        "pe": int(valuation_nulls[0] or 0),
        "pb": int(valuation_nulls[1] or 0),
        "turnover_rate": int(valuation_nulls[2] or 0),
    }
    blocking += valuation_formula

    name_change = tables["name_change"]
    name_invalid = int(
        con.execute(
            f"SELECT count(*) FROM {name_change} WHERE "
            "coalesce(trim(old_name),'')='' OR coalesce(trim(new_name),'')='' OR "
            "old_name=new_name"
        ).fetchone()[0]
    )
    universe = tables["universe_snapshot"]
    name_latest_mismatch = int(
        con.execute(
            f"""
            WITH e AS (
              SELECT symbol,new_name FROM {name_change}
              QUALIFY row_number() OVER(PARTITION BY symbol ORDER BY trade_date DESC)=1
            ), u AS (
              SELECT symbol,trim(name) AS name FROM {universe} WHERE trade_date=?
            )
            SELECT count(*) FROM e JOIN u USING(symbol) WHERE e.new_name<>u.name
            """,
            [target],
        ).fetchone()[0]
    )
    checks["name_change_invalid_count"] = name_invalid
    checks["name_change_latest_name_mismatch_count"] = name_latest_mismatch
    blocking += name_invalid + name_latest_mismatch

    corporate = tables["corporate_actions"]
    corporate_invalid = int(
        con.execute(
            f"SELECT count(*) FROM {corporate} WHERE ex_date IS NULL OR "
            "trade_date<>ex_date OR announcement_date IS NULL OR "
            "try_cast(announcement_date AS DATE)>try_cast(ex_date AS DATE) OR "
            "(record_date IS NOT NULL AND try_cast(record_date AS DATE)>try_cast(ex_date AS DATE)) OR "
            "(dividend_pay_date IS NOT NULL AND try_cast(dividend_pay_date AS DATE)<try_cast(ex_date AS DATE)) OR "
            "coalesce(cash_dividend_per_10,0)+coalesce(bonus_share_per_10,0)+"
            "coalesce(transfer_share_per_10,0)<=0"
        ).fetchone()[0]
    )
    factor = tables["adjust_factor"]
    corporate_without_factor = int(
        con.execute(
            f"""
            WITH changes AS (
              SELECT symbol,trade_date,adjust_factor,
                     lag(adjust_factor) OVER(PARTITION BY symbol ORDER BY trade_date) AS prior
              FROM {factor}
            )
            SELECT count(*) FROM {corporate} c
            LEFT JOIN changes f USING(symbol,trade_date)
            WHERE f.prior IS NULL OR abs(f.adjust_factor/f.prior-1)<1e-12
            """
        ).fetchone()[0]
    )
    checks["corporate_action_invalid_count"] = corporate_invalid
    checks["corporate_action_without_factor_change_count"] = corporate_without_factor
    blocking += corporate_invalid

    index = tables["index_constituents"]
    index_row = con.execute(
        f"SELECT count(*) FILTER(WHERE try_cast(source_snapshot_date AS DATE)>"
        "try_cast(trade_date AS DATE)), min(trade_date), max(trade_date), "
        "max(source_snapshot_date), count(distinct index_symbol) FROM " + index
    ).fetchone()
    index_future = int(index_row[0] or 0)
    latest_snapshot = str(index_row[3] or "")
    snapshot_age = (
        (pd.Timestamp(target) - pd.Timestamp(latest_snapshot)).days
        if latest_snapshot
        else 999999
    )
    maximum_snapshot_gap = int(
        con.execute(
            f"""
            WITH snapshots AS (
              SELECT DISTINCT index_symbol, source_snapshot_date FROM {index}
            ), gaps AS (
              SELECT date_diff(
                       'day',
                       try_cast(source_snapshot_date AS DATE),
                       lead(try_cast(source_snapshot_date AS DATE)) OVER (
                         PARTITION BY index_symbol
                         ORDER BY try_cast(source_snapshot_date AS DATE)
                       )
                     ) AS gap_days
              FROM snapshots
            )
            SELECT coalesce(max(gap_days), 0) FROM gaps
            """
        ).fetchone()[0]
    )
    index_contract_errors = (
        index_future
        + int(str(index_row[1] or "") != "2010-01-04")
        + int(str(index_row[2] or "") != target)
        + int(int(index_row[4] or 0) != 3)
        + int(snapshot_age > 40)
        + int(maximum_snapshot_gap > 40)
    )
    checks["index_future_snapshot_count"] = index_future
    checks["index_latest_snapshot_age_days"] = snapshot_age
    checks["index_maximum_snapshot_gap_days"] = maximum_snapshot_gap
    checks["index_contract_error_count"] = index_contract_errors
    blocking += index_contract_errors

    watermark_errors: list[str] = []
    secondary_status_errors: list[str] = []
    secondary_mismatches = 0
    for domain in AUXILIARY_DOMAINS:
        source = dict(manifests[domain].source or {})
        checked = str(source.get("checked_through", "") or "")
        if checked != target:
            watermark_errors.append(f"{domain}:{checked or 'missing'}")
        validation_status = str(
            source.get("secondary_validation_status", "") or ""
        )
        if validation_status not in {"ok", "not_comparable"}:
            secondary_status_errors.append(
                f"{domain}:{validation_status or 'missing'}"
            )
        secondary_mismatches += int(
            source.get("secondary_material_mismatch_count", 0) or 0
        )
    checks["watermark_errors"] = watermark_errors
    checks["secondary_validation_status_errors"] = secondary_status_errors
    checks["secondary_material_mismatch_count"] = secondary_mismatches
    blocking += (
        len(watermark_errors)
        + len(secondary_status_errors)
        + secondary_mismatches
    )
    return {
        "status": "ok" if blocking == 0 else "needs_attention",
        "target_date": target,
        "blocking_error_count": int(blocking),
        "checks": checks,
    }


def _path_texts(paths: Iterable[Path]) -> list[str]:
    return [str(item) for item in paths]


def _q(value: str) -> str:
    return '"' + str(value).replace('"', '""') + '"'


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
