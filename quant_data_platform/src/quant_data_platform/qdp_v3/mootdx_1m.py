from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable

import pandas as pd

from quant_data_platform.core.json_io import read_json
from quant_data_platform.domains.contracts import DataDomain, DomainFetchRequest
from quant_data_platform.providers import MootdxOnlineProvider
from quant_data_platform.qdp_v3.constants import (
    MOOTDX_VERSION,
    QUALITY_STRICT,
    RAW_DAILY_ASTOCK,
    RAW_INTRADAY_1M_MOOTDX,
)
from quant_data_platform.qdp_v3.identity import board_for_symbol, normalize_symbol
from quant_data_platform.qdp_v3.intraday_1m import is_complete_1m_day, normalize_provider_1m
from quant_data_platform.qdp_v3.manifest import atomic_write_json, stable_hash, utc_now
from quant_data_platform.qdp_v3.paths import ensure_qdp_v3_layout, qdp_v3_paths
from quant_data_platform.qdp_v3.storage import get_raw_partition, read_raw_partition, write_raw_partition


def _expected_mainboard_symbols(
    trade_date: str,
    *,
    workspace_root: str | Path | None,
) -> list[str]:
    ref = get_raw_partition(
        RAW_DAILY_ASTOCK,
        partition_field="query_date",
        partition_value=str(trade_date),
        workspace_root=workspace_root,
    )
    if ref is None:
        return []
    raw = read_raw_partition(ref)
    if raw.empty or "code" not in raw.columns:
        return []
    symbols = raw["code"].map(normalize_symbol)
    status = raw.get("tradestatus", pd.Series("", index=raw.index)).astype(str)
    return sorted(set(symbols.loc[status.eq("1") & symbols.map(board_for_symbol).eq("MainBoard")]))


def _job_path(job_id: str, *, workspace_root: str | Path | None) -> Path:
    return qdp_v3_paths(workspace_root).jobs / f"{job_id}.json"


def ingest_mootdx_intraday_1m(
    *,
    trade_dates: Iterable[str],
    workspace_root: str | Path | None = None,
    provider: MootdxOnlineProvider | None = None,
    refresh: bool = True,
    job_id: str = "",
) -> dict[str, Any]:
    """Atomically capture complete recent 1m market dates from mootdx."""

    ensure_qdp_v3_layout(workspace_root)
    dates = sorted({str(item)[:10] for item in trade_dates if str(item)})
    if not dates:
        return {"status": "completed", "task_count": 0, "completed_count": 0, "pending_count": 0}
    resolved_job_id = str(job_id or f"mootdx__intraday_1m__{stable_hash({'dates': dates}, length=20)}")
    path = _job_path(resolved_job_id, workspace_root=workspace_root)
    previous = read_json(path)
    if previous and list(previous.get("trade_dates", []) or []) != dates:
        raise RuntimeError(f"mootdx_1m_job_contract_conflict:{resolved_job_id}")
    tasks = dict(previous.get("tasks", {}) or {}) if previous else {
        date: {"status": "pending", "row_count": 0, "content_sha256": "", "error": ""}
        for date in dates
    }
    state: dict[str, Any] = {
        "job_id": resolved_job_id,
        "provider": "mootdx_online",
        "mode": "incremental_1m_whole_date",
        "trade_dates": dates,
        "tasks": tasks,
        "status": "running",
        "package_version": MOOTDX_VERSION,
        "updated_at": utc_now(),
    }
    atomic_write_json(path, state)
    # The provider starts at offset zero and stops as soon as the target date
    # is observed.  Its page estimator still expands automatically when an
    # older target is not present on the first page.
    source = provider or MootdxOnlineProvider(max_pages=1)
    try:
        for trade_date in dates:
            existing = get_raw_partition(
                RAW_INTRADAY_1M_MOOTDX,
                partition_field="trade_date",
                partition_value=trade_date,
                workspace_root=workspace_root,
            )
            if existing is not None and not refresh:
                tasks[trade_date] = {
                    "status": "completed",
                    "row_count": existing.row_count,
                    "content_sha256": existing.content_sha256,
                    "error": "",
                }
                continue
            symbols = _expected_mainboard_symbols(trade_date, workspace_root=workspace_root)
            if not symbols:
                tasks[trade_date] = {
                    "status": "pending",
                    "row_count": 0,
                    "content_sha256": "",
                    "error": "expected_mainboard_symbols_missing",
                }
                break
            result = source.fetch_domain(
                DomainFetchRequest(
                    domain=DataDomain.MARKET_INTRADAY_1M,
                    symbols=tuple(symbols),
                    start_date=trade_date,
                    end_date=trade_date,
                )
            )
            if result.error_report:
                tasks[trade_date] = {
                    "status": "pending",
                    "row_count": int(len(result.data)),
                    "content_sha256": "",
                    "error": f"provider_errors:{len(result.error_report)}",
                    "error_sample": list(result.error_report)[:5],
                }
                break
            normalized_frames: list[pd.DataFrame] = []
            invalid: list[dict[str, Any]] = []
            raw = result.data.copy()
            symbol_column = "symbol" if "symbol" in raw.columns else "provider_symbol" if "provider_symbol" in raw.columns else ""
            if not symbol_column:
                raise RuntimeError("mootdx_1m_symbol_column_missing")
            raw["provider_symbol"] = raw[symbol_column].map(normalize_symbol)
            for symbol in symbols:
                day = normalize_provider_1m(
                    raw.loc[raw["provider_symbol"].eq(symbol)],
                    provider_symbol=symbol,
                    source="mootdx_online",
                    merge_0930_into_0931=False,
                )
                if not is_complete_1m_day(day):
                    invalid.append({"provider_symbol": symbol, "bar_count": int(len(day))})
                else:
                    normalized_frames.append(day)
            if invalid:
                tasks[trade_date] = {
                    "status": "pending",
                    "row_count": int(sum(len(item) for item in normalized_frames)),
                    "content_sha256": "",
                    "error": f"incomplete_stock_days:{len(invalid)}",
                    "error_sample": invalid[:20],
                }
                break
            complete = pd.concat(normalized_frames, ignore_index=True)
            ref, _ = write_raw_partition(
                raw_domain=RAW_INTRADAY_1M_MOOTDX,
                partition_field="trade_date",
                partition_value=trade_date,
                frame=complete,
                receipt={
                    "provider": "mootdx_online",
                    "endpoint": "bars_frequency_8_1m",
                    "package_version": MOOTDX_VERSION,
                    "request": {"trade_date": trade_date, "symbol_count": len(symbols), "start": 0, "offset": 800},
                    "bar_count_contract": "mootdx_1m_240_without_0930",
                    "whole_date_atomic": True,
                    "quality_tier": QUALITY_STRICT,
                    "provider_errors": [],
                },
                workspace_root=workspace_root,
            )
            tasks[trade_date] = {
                "status": "completed",
                "row_count": ref.row_count,
                "content_sha256": ref.content_sha256,
                "error": "",
            }
            state["tasks"] = tasks
            state["updated_at"] = utc_now()
            atomic_write_json(path, state)
    finally:
        source.close()
    completed_dates = [date for date in dates if tasks.get(date, {}).get("status") == "completed"]
    pending_dates = [date for date in dates if tasks.get(date, {}).get("status") != "completed"]
    state.update(
        {
            "status": "completed" if not pending_dates else "pending_provider_unhealthy",
            "completed_count": len(completed_dates),
            "pending_count": len(pending_dates),
            "watermark": max(completed_dates) if completed_dates else "",
            "updated_at": utc_now(),
        }
    )
    atomic_write_json(path, state)
    return {**state, "job_path": str(path.resolve())}


def one_minute_lag_trading_days(
    *,
    daily_watermark: str,
    one_minute_watermark: str,
    open_trade_dates: Iterable[str],
) -> int:
    dates = sorted({str(item)[:10] for item in open_trade_dates if str(item)})
    if not one_minute_watermark:
        return len([date for date in dates if date <= str(daily_watermark)])
    return len([date for date in dates if str(one_minute_watermark) < date <= str(daily_watermark)])
