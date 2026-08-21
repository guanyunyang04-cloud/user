"""Auxiliary update index operations."""

from __future__ import annotations

import shutil
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import pandas as pd

from quantlab.data.qdp_v2.duckdb_resources import open_guarded_duckdb

from .baostock import (
    _fetch_baostock_snapshots,
)
from .context import (
    AuxiliaryContext,
    AuxiliaryUpdateError,
    _copy_query,
    _open_dates,
    _paths,
    _replace_domain,
    _scan_sql,
    _snapshot_dates,
)


def _index_repair_sql(ctx: AuxiliaryContext, *, fetched_parts: Sequence[Path]) -> str:
    current = _scan_sql(_paths(ctx, "index_constituents"))
    fetched = _scan_sql(fetched_parts)
    calendar = _scan_sql(_paths(ctx, "trading_calendar"))
    identity = _scan_sql(_paths(ctx, "security_identity"))
    return f"""
    WITH fetched_dates AS (
      SELECT DISTINCT source_snapshot_date FROM {fetched}
    ), raw_snapshots AS (
      SELECT index_symbol, symbol, index_name, source,
             source_snapshot_date
      FROM {current}
      WHERE source_snapshot_date IS NOT NULL
        AND try_cast(source_snapshot_date AS DATE) <= try_cast(trade_date AS DATE)
        AND source_snapshot_date NOT IN (
          SELECT source_snapshot_date FROM fetched_dates
        )
      UNION ALL
      SELECT index_symbol, symbol, index_name, source,
             source_snapshot_date
      FROM {fetched}
      WHERE try_cast(source_snapshot_date AS DATE)
            <= try_cast(snapshot_query_date AS DATE)
    ), members AS (
      SELECT DISTINCT index_symbol, symbol, index_name, source,
             source_snapshot_date
      FROM raw_snapshots
      WHERE index_symbol IN ('000016.SH','000300.SH','000905.SH')
        AND symbol IN (SELECT current_symbol FROM {identity})
    ), snapshot_ranges AS (
      SELECT index_symbol, source_snapshot_date,
             lead(source_snapshot_date) OVER (
               PARTITION BY index_symbol ORDER BY source_snapshot_date
             ) AS next_snapshot_date
      FROM (SELECT DISTINCT index_symbol, source_snapshot_date FROM members)
    ), open_dates AS (
      SELECT DISTINCT trade_date
      FROM {calendar}
      WHERE exchange='SSE' AND is_open
        AND trade_date BETWEEN '2010-01-04' AND '{ctx.target_date}'
    )
    SELECT m.index_symbol, m.symbol, d.trade_date, m.index_name, m.source,
           m.source_snapshot_date
    FROM members m
    JOIN snapshot_ranges r USING(index_symbol, source_snapshot_date)
    JOIN open_dates d
      ON d.trade_date >= m.source_snapshot_date
     AND (r.next_snapshot_date IS NULL OR d.trade_date < r.next_snapshot_date)
    ORDER BY d.trade_date, m.index_symbol, m.symbol
    """


def _index_prepared_stats(ctx: AuxiliaryContext, prepared: Path) -> tuple[tuple[Any, ...], int]:
    produced = _scan_sql([prepared])
    spill = ctx.runtime / "index_validate_spill"
    try:
        with open_guarded_duckdb(temp_directory=spill, threads=2) as con:
            row = con.execute(
                "SELECT count(*) FILTER (WHERE "
                "try_cast(source_snapshot_date AS DATE)>try_cast(trade_date AS DATE)), "
                "min(trade_date), max(trade_date), max(source_snapshot_date), "
                "count(distinct index_symbol) FROM " + produced
            ).fetchone()
            max_gap = int(
                con.execute(
                    f"""
                    WITH snapshots AS (
                      SELECT DISTINCT index_symbol, source_snapshot_date FROM {produced}
                    ), gaps AS (
                      SELECT date_diff(
                               'day', try_cast(source_snapshot_date AS DATE),
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
    finally:
        shutil.rmtree(spill, ignore_errors=True)
    return row, max_gap


def _validate_index_prepared(ctx: AuxiliaryContext, row: tuple[Any, ...] | None, max_gap: int) -> int:
    if (
        row is None
        or int(row[0] or 0) != 0
        or str(row[1]) != "2010-01-04"
        or str(row[2]) != ctx.target_date
        or int(row[4] or 0) != 3
        or max_gap > 40
    ):
        raise AuxiliaryUpdateError(f"index_contract_failed:{row}")
    age = (pd.Timestamp(ctx.target_date) - pd.Timestamp(row[3])).days
    if age > 40:
        raise AuxiliaryUpdateError(f"index_snapshot_stale:{age}")
    return age


def repair_index_constituents(ctx: AuxiliaryContext) -> dict[str, Any]:
    dates = _snapshot_dates(_open_dates(ctx))
    parts = _fetch_baostock_snapshots(ctx, kind="index", dates=dates)
    prepared = ctx.runtime / "index_constituents.prepared.parquet"
    _copy_query(ctx, sql=_index_repair_sql(ctx, fetched_parts=parts), target=prepared)
    row, max_gap = _index_prepared_stats(ctx, prepared)
    age = _validate_index_prepared(ctx, row, max_gap)
    result = _replace_domain(
        ctx,
        domain="index_constituents",
        prepared=prepared,
        primary_key=("trade_date", "index_symbol", "symbol"),
        contract_version="qdp_v2_index_constituents_strict_pit_v2",
        source_contract="BaoStock first-open/month-end/latest snapshots with past-only expansion",
        validation={
            "secondary_compared_count": 0,
            "secondary_material_mismatch_count": 0,
            "secondary_validation_status": "pending_tushare_latest_snapshot",
        },
    )
    shutil.rmtree(ctx.runtime / "baostock_index_parts", ignore_errors=True)
    return {
        **result,
        "snapshot_date_count": len(dates),
        "latest_snapshot_age_days": age,
        "maximum_snapshot_gap_days": max_gap,
    }
