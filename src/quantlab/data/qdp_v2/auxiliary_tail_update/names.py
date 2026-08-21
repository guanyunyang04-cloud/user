"""Auxiliary Tail Update: names responsibilities."""

from __future__ import annotations

import shutil
from typing import Any

from quantlab.data.qdp_v2.auxiliary_update import (
    AuxiliaryContext,
    _paths,
    _scan_sql,
)
from quantlab.data.qdp_v2.duckdb_resources import open_guarded_duckdb

from .common import (
    _append_if_any,
    _mark_checked,
)


def update_name_change_tail(ctx: AuxiliaryContext) -> dict[str, Any]:
    universe = _scan_sql(_paths(ctx, "universe_snapshot"))
    current = _scan_sql(_paths(ctx, "name_change"))
    spill = ctx.runtime / "tail_name_change_spill"
    with open_guarded_duckdb(temp_directory=spill, threads=2) as con:
        frame = con.execute(
            f"""
            WITH normalized AS (
              SELECT symbol, trade_date, trim(name) AS name
              FROM {universe}
              WHERE trade_date<='{ctx.target_date}'
                AND coalesce(trim(name), '')<>''
            ), transitions AS (
              SELECT symbol, trade_date,
                     lag(name) OVER(PARTITION BY symbol ORDER BY trade_date)
                       AS old_name,
                     name AS new_name
              FROM normalized
            ), events AS (
              SELECT symbol, trade_date, old_name, new_name,
                     'short_name' AS change_type,
                     'qdp_universe_tail_transition' AS source
              FROM transitions
              WHERE old_name IS NOT NULL AND old_name<>new_name
            )
            SELECT e.* FROM events e
            ANTI JOIN {current} c USING(symbol, trade_date, change_type)
            ORDER BY e.trade_date, e.symbol
            """
        ).fetchdf()
    shutil.rmtree(spill, ignore_errors=True)
    result = _append_if_any(ctx, "name_change", frame)
    result["metadata"] = _mark_checked(ctx, "name_change")
    return result
