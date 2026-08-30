"""Auxiliary Tail Update: valuation responsibilities."""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any

import pandas as pd

from quantlab.data.domains.contracts.requests import DatePartitionFetchRequest
from quantlab.data.domains.contracts.schema import DataDomain
from quantlab.data.providers.baostock.provider import BaostockProvider
from quantlab.data.qdp_v2.auxiliary_update.context import AuxiliaryContext, _paths, _scan_sql
from quantlab.data.qdp_v2.duckdb_resources import open_guarded_duckdb

from .common import (
    _append_if_any,
    _mark_checked,
    _missing_daily_keys,
)
from .config import (
    AuxiliaryTailUpdateError,
)


def update_valuation_tail(
    ctx: AuxiliaryContext,
    *,
    baostock_valuation_cache_path: str | Path | None = None,
) -> dict[str, Any]:
    missing = _missing_daily_keys(ctx, "valuation")
    if missing.empty:
        metadata = _mark_checked(ctx, "valuation", missing_daily_keys=0)
        return {"status": "already_complete", "row_count": 0, "metadata": metadata}
    frames: list[pd.DataFrame] = []
    cache = Path(baostock_valuation_cache_path).resolve() if baostock_valuation_cache_path is not None else None
    if cache is not None and cache.is_file():
        cached = pd.read_parquet(cache)
        required = {"symbol", "trade_date", "pe", "pb", "turnover_rate"}
        if required.issubset(cached.columns):
            frames.append(cached.loc[:, sorted(required)].copy())
    cached_dates = set(pd.concat(frames, ignore_index=True)["trade_date"].astype(str)) if frames else set()
    missing_dates = sorted(set(missing["trade_date"].astype(str)).difference(cached_dates))
    provider = BaostockProvider()
    try:
        for trade_date in missing_dates:
            result = provider.fetch_date_partition(
                DatePartitionFetchRequest(
                    domain=DataDomain.VALUATION,
                    trade_date=trade_date,
                    universe_kind="all_a",
                    fetch_mode="date_snapshot",
                )
            )
            if result.error_report:
                raise AuxiliaryTailUpdateError(f"valuation_tail_provider_errors:{trade_date}")
            if not result.data.empty:
                normalized = result.data.rename(
                    columns={
                        "provider_symbol": "symbol",
                        "pe_ttm": "pe",
                        "pb_mrq": "pb",
                    }
                )
                frames.append(normalized.loc[:, ["symbol", "trade_date", "pe", "pb", "turnover_rate"]].copy())
    finally:
        provider.close()
    fetched = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    if fetched.empty:
        raise AuxiliaryTailUpdateError("valuation_tail_provider_empty")
    daily = _scan_sql(_paths(ctx, "market_daily_raw"))
    share = _scan_sql(_paths(ctx, "share_capital"))
    spill = ctx.runtime / "tail_valuation_build_spill"
    with open_guarded_duckdb(temp_directory=spill, threads=2) as con:
        con.register("tail_missing_keys", missing)
        con.register("tail_valuation_fetched", fetched)
        frame = con.execute(
            f"""
            SELECT m.symbol, m.trade_date,
                   d.close*s.total_share AS total_mv,
                   d.close*s.float_share AS circ_mv,
                   v.pe, v.pb, v.turnover_rate,
                   'baostock_tail_plus_qdp_market_cap_formula' AS source
            FROM tail_missing_keys m
            JOIN {daily} d USING(symbol, trade_date)
            JOIN {share} s USING(symbol, trade_date)
            LEFT JOIN tail_valuation_fetched v USING(symbol, trade_date)
            ORDER BY m.trade_date, m.symbol
            """
        ).fetchdf()
    shutil.rmtree(spill, ignore_errors=True)
    if len(frame) != len(missing) or frame[["total_mv", "circ_mv"]].isna().any().any():
        raise AuxiliaryTailUpdateError("valuation_tail_market_cap_unresolved")
    result = _append_if_any(ctx, "valuation", frame)
    remaining = _missing_daily_keys(ctx, "valuation")
    if not remaining.empty:
        raise AuxiliaryTailUpdateError(f"valuation_tail_post_commit_missing:{len(remaining)}")
    result["metadata"] = _mark_checked(ctx, "valuation", missing_daily_keys=0)
    return result
