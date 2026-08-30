"""Auxiliary Tail Update: industry responsibilities."""

from __future__ import annotations

import shutil
from typing import Any

from quantlab.data.qdp_v2.auxiliary_update.baostock import _fetch_baostock_snapshots
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


def update_industry_tail(ctx: AuxiliaryContext) -> dict[str, Any]:
    missing = _missing_daily_keys(ctx, "industry_concept")
    if missing.empty:
        metadata = _mark_checked(ctx, "industry_concept", missing_daily_keys=0)
        return {"status": "already_complete", "row_count": 0, "metadata": metadata}
    dates = sorted(missing["trade_date"].astype(str).unique())
    parts = _fetch_baostock_snapshots(ctx, kind="industry", dates=dates)
    current = _scan_sql(_paths(ctx, "industry_concept"))
    fetched = _scan_sql(parts)
    spill = ctx.runtime / "tail_industry_build_spill"
    with open_guarded_duckdb(temp_directory=spill, threads=2) as con:
        con.register("tail_missing_keys", missing)
        frame = con.execute(
            f"""
            WITH candidates AS (
              SELECT symbol, industry,
                     coalesce(nullif(original_source, ''), source) AS source,
                     coalesce(nullif(industry_source_date, ''), trade_date)
                       AS source_date,
                     coalesce(nullif(industry_standard, ''), '证监会行业分类')
                       AS industry_standard,
                     1 AS priority
              FROM {current}
              WHERE coalesce(trim(industry), '')<>''
                AND industry_fill_method IN ('direct_snapshot','prior_ffill')
                AND try_cast(industry_source_date AS DATE)<=try_cast(trade_date AS DATE)
              UNION ALL
              SELECT symbol, industry, source, source_date,
                     coalesce(nullif(industry_standard, ''), '证监会行业分类'),
                     2 AS priority
              FROM {fetched}
              WHERE coalesce(trim(industry), '')<>''
                AND try_cast(source_date AS DATE)<=try_cast(snapshot_query_date AS DATE)
            ), dedup AS (
              SELECT * EXCLUDE(priority)
              FROM candidates
              QUALIFY row_number() OVER(
                PARTITION BY symbol, source_date ORDER BY priority DESC
              )=1
            )
            SELECT m.symbol, m.trade_date, c.industry, c.source,
                   c.industry AS original_industry,
                   c.source AS original_source,
                   CASE WHEN c.source_date=m.trade_date THEN 'direct_snapshot'
                        ELSE 'prior_ffill' END AS industry_fill_method,
                   c.source_date AS industry_source_date,
                   c.industry_standard
            FROM tail_missing_keys m
            ASOF LEFT JOIN dedup c
              ON m.symbol=c.symbol AND m.trade_date>=c.source_date
            ORDER BY m.trade_date, m.symbol
            """
        ).fetchdf()
    shutil.rmtree(spill, ignore_errors=True)
    if len(frame) != len(missing) or frame["industry"].isna().any():
        raise AuxiliaryTailUpdateError(f"industry_tail_unresolved:{len(missing) - frame['industry'].notna().sum()}")
    result = _append_if_any(ctx, "industry_concept", frame)
    shutil.rmtree(ctx.runtime / "baostock_industry_parts", ignore_errors=True)
    remaining = _missing_daily_keys(ctx, "industry_concept")
    if not remaining.empty:
        raise AuxiliaryTailUpdateError(f"industry_tail_post_commit_missing:{len(remaining)}")
    result["metadata"] = _mark_checked(ctx, "industry_concept", missing_daily_keys=0)
    return result
