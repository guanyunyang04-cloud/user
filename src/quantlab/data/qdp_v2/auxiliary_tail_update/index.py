"""Auxiliary Tail Update: index responsibilities."""

from __future__ import annotations

import shutil
from typing import Any

from quantlab.data.qdp_v2.auxiliary_update import (
    AuxiliaryContext,
    _fetch_baostock_snapshots,
    _paths,
    _scan_sql,
)
from quantlab.data.qdp_v2.duckdb_resources import open_guarded_duckdb

from .common import (
    _append_if_any,
    _mark_checked,
)
from .config import (
    AuxiliaryTailUpdateError,
)


def _index_tail_dates(ctx: AuxiliaryContext) -> list[str]:
    current = _scan_sql(_paths(ctx, "index_constituents"))
    calendar = _scan_sql(_paths(ctx, "trading_calendar"))
    spill = ctx.runtime / "tail_index_inventory_spill"
    with open_guarded_duckdb(temp_directory=spill, threads=1) as con:
        dates = [
            str(item[0])
            for item in con.execute(
                f"""
                SELECT DISTINCT c.trade_date
                FROM {calendar} c
                WHERE c.exchange='SSE' AND c.is_open
                  AND try_cast(c.trade_date AS DATE)>(
                    SELECT coalesce(
                      max(try_cast(trade_date AS DATE)),
                      DATE '2010-01-03'
                    )
                    FROM {current}
                  )
                  AND try_cast(c.trade_date AS DATE)<=try_cast(? AS DATE)
                ORDER BY c.trade_date
                """,
                [ctx.target_date],
            ).fetchall()
        ]
    shutil.rmtree(spill, ignore_errors=True)
    return dates


def update_index_tail(ctx: AuxiliaryContext) -> dict[str, Any]:
    dates = _index_tail_dates(ctx)
    if not dates:
        metadata = _mark_checked(ctx, "index_constituents")
        return {"status": "already_complete", "row_count": 0, "metadata": metadata}
    parts = _fetch_baostock_snapshots(ctx, kind="index", dates=dates)
    fetched = _scan_sql(parts)
    identity = _scan_sql(_paths(ctx, "security_identity"))
    spill = ctx.runtime / "tail_index_build_spill"
    with open_guarded_duckdb(temp_directory=spill, threads=2) as con:
        frame = con.execute(
            f"""
            SELECT index_symbol, symbol,
                   snapshot_query_date AS trade_date,
                   index_name, source, source_snapshot_date
            FROM {fetched}
            WHERE symbol IN (SELECT current_symbol FROM {identity})
            QUALIFY row_number() OVER(
              PARTITION BY index_symbol, symbol, snapshot_query_date
              ORDER BY source_snapshot_date DESC
            )=1
            ORDER BY trade_date, index_symbol, symbol
            """
        ).fetchdf()
    shutil.rmtree(spill, ignore_errors=True)
    coverage = frame.groupby("trade_date")["index_symbol"].nunique()
    if coverage.empty or int(coverage.min()) != 3 or int(coverage.max()) != 3:
        raise AuxiliaryTailUpdateError(f"index_tail_incomplete:{coverage.to_dict()}")
    result = _append_if_any(ctx, "index_constituents", frame)
    shutil.rmtree(ctx.runtime / "baostock_index_parts", ignore_errors=True)
    result["metadata"] = _mark_checked(ctx, "index_constituents")
    return result
