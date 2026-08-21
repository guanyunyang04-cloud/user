"""Database audit latest checks."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from quantlab.data.core.paths import qdp_paths
from quantlab.data.qdp_v2.duckdb_resources import open_guarded_duckdb
from quantlab.data.qdp_v2.manifest import (
    qdp_v2_root,
    read_active_manifest,
)
from quantlab.data.qdp_v2.status import active_dataset_map

from .config import (
    REQUIRED_DOMAINS,
)
from .identity import (
    _active_manifest,
    _except_count,
    _paths_for_date,
)
from .market import (
    _latest_5m_stats,
    _status_daily_partition_check,
)


@dataclass(frozen=True)
class _LatestPartitions:
    trade_date: str
    paths: dict[str, list[Path]]


def _resolve_latest_partitions(
    root: Path,
    datasets: dict[str, str],
) -> _LatestPartitions | dict[str, Any]:
    required = {
        "market_daily_raw",
        "adjust_factor",
        "universe_snapshot",
        "security_status",
        "market_intraday_5m",
    }
    missing = sorted(required.difference(REQUIRED_DOMAINS.intersection(datasets)))
    if missing:
        return {
            "status": "skipped",
            "reason": f"latest_key_domains_missing:{','.join(missing)}",
            "finding_count": 0,
        }

    manifests = {domain: _active_manifest(root, datasets[domain], domain) for domain in required}
    trade_date = str(manifests["market_daily_raw"].end_date or "")
    if not trade_date:
        return {
            "status": "skipped",
            "reason": "market_daily_end_date_missing",
            "finding_count": 0,
        }
    paths = {domain: _paths_for_date(root, manifest, trade_date) for domain, manifest in manifests.items()}
    absent = sorted(domain for domain, items in paths.items() if not items)
    if absent:
        return {
            "status": "needs_attention",
            "trade_date": trade_date,
            "finding_count": len(absent),
            "errors": [f"latest_partition_missing:{domain}" for domain in absent],
        }
    return _LatestPartitions(trade_date=trade_date, paths=paths)


def _latest_key_checks(
    workspace: Path,
    partitions: _LatestPartitions,
) -> tuple[dict[str, int], dict[str, Any]]:
    paths = partitions.paths
    trade_date = partitions.trade_date
    temp = qdp_paths(workspace).runtime_dir / "check_spill"
    with open_guarded_duckdb(temp_directory=temp, threads=4) as con:
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
        checks = {
            "daily_missing_factor": _except_count(
                con,
                paths["market_daily_raw"],
                paths["adjust_factor"],
                trade_date,
            ),
            "factor_extra_vs_daily": _except_count(
                con,
                paths["adjust_factor"],
                paths["market_daily_raw"],
                trade_date,
            ),
            "universe_missing_status": _except_count(
                con,
                paths["universe_snapshot"],
                paths["security_status"],
                trade_date,
            ),
            "status_extra_vs_universe": _except_count(
                con,
                paths["security_status"],
                paths["universe_snapshot"],
                trade_date,
            ),
            "daily_missing_universe": _except_count(
                con,
                paths["market_daily_raw"],
                paths["universe_snapshot"],
                trade_date,
            ),
            "status_daily_semantic_mismatch": int(status_semantics["semantic_mismatch_count"]),
            "status_null_suspension_flags": int(status_semantics["null_suspension_flag_count"]),
            "invalid_5m_stock_days": int(five["invalid_day_count"]),
        }
    return checks, five


def _latest_result(
    partitions: _LatestPartitions,
    checks: dict[str, int],
    five: dict[str, Any],
) -> dict[str, Any]:
    blocking = {key: value for key, value in checks.items() if int(value) > 0}
    missing_5m = int(five["missing_complete_day_count"])
    if missing_5m > 0:
        blocking["5m_complete_coverage_not_100_percent"] = missing_5m
    coverage = float(five["coverage_ratio"])
    status = "needs_attention" if blocking else "warning" if coverage < 0.99 else "ok"
    return {
        "status": status,
        "trade_date": partitions.trade_date,
        "finding_count": len(blocking),
        "key_checks": checks,
        "five_minute": five,
        "errors": [f"{key}:{value}" for key, value in blocking.items()],
        "warnings": (
            [f"5m_complete_coverage_between_98_and_99_percent:{coverage:.6f}"]
            if not blocking and coverage < 0.99
            else []
        ),
    }


def audit_latest_keys(*, workspace_root: str | Path | None = None) -> dict[str, Any]:
    """Check only the newest daily partition and overlapping 5m shards."""

    root = qdp_v2_root(workspace_root)
    datasets = active_dataset_map(read_active_manifest(root))
    partitions = _resolve_latest_partitions(root, datasets)
    if isinstance(partitions, dict):
        return partitions
    workspace = Path(workspace_root or Path.cwd()).resolve()
    checks, five = _latest_key_checks(workspace, partitions)
    return _latest_result(partitions, checks, five)
