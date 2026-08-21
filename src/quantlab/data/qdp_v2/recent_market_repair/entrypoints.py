"""Recent Market Repair: entrypoints responsibilities."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from quantlab.data.domains.contracts import DataDomain
from quantlab.data.qdp_v2.manifest import (
    atomic_write_json,
)
from quantlab.data.qdp_v2.repair import (
    update_active_manifest_metadata,
)

from .baostock import (
    _active_intraday_quality,
    _build_baostock_daily_reference,
    _download_baostock_buckets,
)
from .commit import (
    _commit_bundles,
)
from .config import (
    DAILY_DOMAIN,
    DEFAULT_END_DATE,
    DEFAULT_START_DATE,
    DEFAULT_WORKERS,
    INTRADAY_DOMAIN,
    _SystemMemoryGuard,
)
from .inventory import (
    _active_inputs,
    _missing_daily_symbols,
    _missing_intraday_pairs,
)
from .mootdx import (
    _download_buckets,
)
from .state import (
    _configure_workspace_runtime,
    _date_text,
    _load_or_initialize_state,
    _pairs_sha256,
    _run_directory,
    _runtime_root,
)


def run_recent_daily_repair(
    *,
    trade_date: str = DEFAULT_END_DATE,
    start_date: str | None = None,
    workspace_root: str | Path | None = None,
    workers: int = DEFAULT_WORKERS,
    resume: bool = True,
) -> dict[str, Any]:
    end = _date_text(trade_date)
    start = _date_text(start_date or end)
    if start > end:
        raise ValueError(f"recent_repair_invalid_range:{start}>{end}")
    workspace = Path(workspace_root or Path.cwd()).resolve()
    _configure_workspace_runtime(workspace)
    inputs = _active_inputs(
        workspace,
        (DAILY_DOMAIN, "universe_snapshot", "security_status", "security_identity"),
    )
    pairs = tuple(
        sorted(
            {
                (symbol, date.strftime("%Y-%m-%d"))
                for date in pd.bdate_range(start, end)
                for symbol in _missing_daily_symbols(
                    inputs,
                    trade_date=date.strftime("%Y-%m-%d"),
                    workspace=workspace,
                )
            }
        )
    )
    wanted: dict[str, tuple[str, ...]] = {}
    for symbol, date in pairs:
        wanted[symbol] = (*wanted.get(symbol, ()), date)
    runtime = _runtime_root(workspace) / _run_directory("mootdx_daily", start, end)
    state = _load_or_initialize_state(
        runtime,
        kind="daily",
        start_date=start,
        end_date=end,
        task_count=len(pairs),
        resume=resume,
    )
    if state.get("status") == "applied":
        return state
    with _SystemMemoryGuard() as guard:
        state = _download_buckets(
            runtime=runtime,
            state=state,
            wanted=wanted,
            provider_domain=DataDomain.MARKET_DAILY,
            output_domain=DAILY_DOMAIN,
            start_date=start,
            end_date=end,
            workers=workers,
            guard=guard,
        )
        guard.check("daily_before_commit")
        state = _commit_bundles(
            state,
            runtime=runtime,
            domain=DAILY_DOMAIN,
            reason=f"fill missing daily rows from healthy mootdx {start}..{end}",
            workspace=workspace,
        )
        state["minimum_available_bytes"] = guard.minimum_available_bytes
        atomic_write_json(runtime / "state.json", state)
    return state


def run_recent_intraday_repair(
    *,
    start_date: str = DEFAULT_START_DATE,
    end_date: str = DEFAULT_END_DATE,
    workspace_root: str | Path | None = None,
    workers: int = DEFAULT_WORKERS,
    resume: bool = True,
) -> dict[str, Any]:
    start = _date_text(start_date)
    end = _date_text(end_date)
    if start > end:
        raise ValueError(f"recent_repair_invalid_range:{start}>{end}")
    workspace = Path(workspace_root or Path.cwd()).resolve()
    _configure_workspace_runtime(workspace)
    domains = (DAILY_DOMAIN, INTRADAY_DOMAIN)
    inputs = _active_inputs(workspace, domains, start_date=start, end_date=end)
    pairs = _missing_intraday_pairs(
        inputs,
        start_date=start,
        end_date=end,
        workspace=workspace,
    )
    wanted: dict[str, tuple[str, ...]] = {}
    for symbol, trade_date in pairs:
        wanted.setdefault(symbol, ())
        wanted[symbol] = (*wanted[symbol], trade_date)
    runtime = _runtime_root(workspace) / _run_directory("mootdx_5m", start, end)
    state = _load_or_initialize_state(
        runtime,
        kind="intraday_5m",
        start_date=start,
        end_date=end,
        task_count=len(pairs),
        resume=resume,
    )
    if state.get("status") == "applied":
        return state
    with _SystemMemoryGuard() as guard:
        state = _download_buckets(
            runtime=runtime,
            state=state,
            wanted=wanted,
            provider_domain=DataDomain.MARKET_INTRADAY_5M,
            output_domain=INTRADAY_DOMAIN,
            start_date=start,
            end_date=end,
            workers=workers,
            guard=guard,
        )
        guard.check("intraday_before_commit")
        state = _commit_bundles(
            state,
            runtime=runtime,
            domain=INTRADAY_DOMAIN,
            reason=f"fill complete recent 5m stock-days from healthy mootdx {start}..{end}",
            workspace=workspace,
        )
        state["minimum_available_bytes"] = guard.minimum_available_bytes
        atomic_write_json(runtime / "state.json", state)
    return state


def _record_baostock_repair_metadata(
    workspace: Path,
    *,
    state: dict[str, Any],
    start: str,
    end: str,
) -> dict[str, Any]:
    current_quality = _active_intraday_quality(workspace)
    positive_daily = int(current_quality.get("canonical_positive_daily_count", 0) or 0)
    complete_before = int(current_quality.get("canonical_complete_positive_daily_count", 0) or 0)
    accepted_days = int(state.get("accepted_day_count", 0) or 0)
    unresolved_days = int(state.get("unresolved_day_count", 0) or 0)
    rejection_counts = dict(state.get("rejection_counts", {}) or {})
    complete_after = min(positive_daily, complete_before + accepted_days)
    return update_active_manifest_metadata(
        INTRADAY_DOMAIN,
        reason="record validated historical BaoStock intraday repair",
        workspace_root=workspace,
        source_updates={
            "historical_repair_provider": "baostock",
            "historical_repair_source": "baostock_history_daily_validated",
            "historical_repair_date_range": f"{start}..{end}",
            "historical_repair_rejection_counts": rejection_counts,
        },
        quality_updates={
            "scope": "point_in_time_historical_mainboard_with_partial_intraday_coverage",
            "permanent_exclusions": {
                "applied": False,
                "reason": "historical intraday availability is not an eligibility filter",
            },
            "missing_history_is_not_an_eligibility_filter": True,
            "intraday_5m_restored_for_historical_symbols": accepted_days,
            "historical_baostock_recovered_positive_days": accepted_days,
            "historical_baostock_residual_positive_days": unresolved_days,
            "historical_baostock_rejection_counts": rejection_counts,
            "canonical_complete_positive_daily_count": complete_after,
            "canonical_complete_coverage_ratio": float(complete_after / positive_daily) if positive_daily else 1.0,
            "historical_missing_positive_days_after": max(0, positive_daily - complete_after),
        },
    )


def run_baostock_intraday_repair(
    *,
    start_date: str = "2020-01-01",
    end_date: str = DEFAULT_END_DATE,
    workspace_root: str | Path | None = None,
    workers: int = DEFAULT_WORKERS,
    resume: bool = True,
) -> dict[str, Any]:
    """Fill remaining traded 5m stock-days directly from BaoStock.

    BaoStock minute responses become slow when many years are requested at
    once, so work is split by symbol and natural year.  Only stock-days absent
    from the single active table are retained; provider payloads are never
    persisted as a second dataset.
    """

    start = _date_text(start_date)
    end = _date_text(end_date)
    if start > end:
        raise ValueError(f"recent_repair_invalid_range:{start}>{end}")
    if int(workers) not in {1, 2, 3, 4}:
        raise ValueError("baostock_intraday_workers_must_be_between_1_and_4")
    workspace = Path(workspace_root or Path.cwd()).resolve()
    _configure_workspace_runtime(workspace)
    inputs = _active_inputs(
        workspace,
        (DAILY_DOMAIN, INTRADAY_DOMAIN),
        start_date=start,
        end_date=end,
    )
    pairs = _missing_intraday_pairs(
        inputs,
        start_date=start,
        end_date=end,
        workspace=workspace,
    )
    wanted: dict[str, tuple[str, ...]] = {}
    for symbol, trade_date in pairs:
        wanted[symbol] = (*wanted.get(symbol, ()), trade_date)
    runtime = _runtime_root(workspace) / _run_directory("baostock_5m", start, end)
    daily_reference_paths = _build_baostock_daily_reference(
        inputs,
        wanted=wanted,
        runtime=runtime,
    )
    state = _load_or_initialize_state(
        runtime,
        kind="baostock_intraday_5m",
        start_date=start,
        end_date=end,
        task_count=len(pairs),
        input_hash=_pairs_sha256(pairs),
        resume=resume,
    )
    state["input_dataset_ids"] = {key: value.dataset_id for key, value in inputs.items()}
    state["requested_pairs_sha256"] = _pairs_sha256(pairs)
    if state.get("status") == "applied":
        return state
    with _SystemMemoryGuard() as guard:
        state = _download_baostock_buckets(
            runtime=runtime,
            state=state,
            wanted=wanted,
            daily_reference_paths=daily_reference_paths,
            workers=int(workers),
            guard=guard,
        )
        guard.check("baostock_intraday_before_commit")
        state = _commit_bundles(
            state,
            runtime=runtime,
            domain=INTRADAY_DOMAIN,
            reason=f"fill complete missing 5m stock-days from BaoStock {start}..{end}",
            workspace=workspace,
        )
        state["manifest_metadata"] = _record_baostock_repair_metadata(
            workspace,
            state=state,
            start=start,
            end=end,
        )
        state["minimum_available_bytes"] = guard.minimum_available_bytes
        atomic_write_json(runtime / "state.json", state)
    return state
