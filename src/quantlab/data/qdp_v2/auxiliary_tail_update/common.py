"""Auxiliary Tail Update: common responsibilities."""

from __future__ import annotations

import shutil
from collections.abc import Mapping
from typing import Any

import numpy as np
import pandas as pd

from quantlab.data.qdp_v2.auxiliary_update.context import AuxiliaryContext, _manifest, _paths, _scan_sql
from quantlab.data.qdp_v2.duckdb_resources import open_guarded_duckdb
from quantlab.data.qdp_v2.repair.mutation import (
    append_active_shard,
    update_active_manifest_metadata,
)


def _only_changed_share_events(frame: pd.DataFrame) -> pd.DataFrame:
    if frame is None or frame.empty:
        return frame.copy() if isinstance(frame, pd.DataFrame) else pd.DataFrame()
    data = frame.sort_values(["symbol", "trade_date"], kind="stable").copy()
    prior_total = data.groupby("symbol", sort=False)["total_share"].shift()
    prior_float = data.groupby("symbol", sort=False)["float_share"].shift()
    changed = prior_total.isna() | prior_float.isna()
    changed |= ~np.isclose(
        pd.to_numeric(data["total_share"], errors="coerce"),
        pd.to_numeric(prior_total, errors="coerce"),
        rtol=1e-12,
        atol=0.5,
        equal_nan=True,
    )
    changed |= ~np.isclose(
        pd.to_numeric(data["float_share"], errors="coerce"),
        pd.to_numeric(prior_float, errors="coerce"),
        rtol=1e-12,
        atol=0.5,
        equal_nan=True,
    )
    return data.loc[changed].reset_index(drop=True)


def _unconfirmed_share_detections(
    detected: pd.DataFrame,
    confirmed: pd.DataFrame,
) -> list[tuple[str, str]]:
    detected_keys = (
        set(
            zip(
                detected["symbol"].astype(str),
                detected["trade_date"].astype(str),
                strict=True,
            )
        )
        if not detected.empty
        else set()
    )
    confirmed_keys = (
        set(
            zip(
                confirmed["symbol"].astype(str),
                confirmed["variation_date"].astype(str),
                strict=True,
            )
        )
        if not confirmed.empty
        else set()
    )
    return sorted(detected_keys - confirmed_keys)


def _missing_daily_keys(ctx: AuxiliaryContext, domain: str) -> pd.DataFrame:
    daily = _scan_sql(_paths(ctx, "market_daily_raw"))
    current = _scan_sql(_paths(ctx, domain))
    spill = ctx.runtime / f"tail_{domain}_inventory_spill"
    with open_guarded_duckdb(temp_directory=spill, threads=2) as con:
        frame = con.execute(
            f"""
            SELECT d.symbol, d.trade_date
            FROM {daily} d
            ANTI JOIN {current} c USING(symbol, trade_date)
            WHERE d.trade_date<=?
            ORDER BY d.trade_date, d.symbol
            """,
            [ctx.target_date],
        ).fetchdf()
    shutil.rmtree(spill, ignore_errors=True)
    return frame


def _checked_through(ctx: AuxiliaryContext, domain: str) -> str:
    manifest = _manifest(ctx.root, ctx.datasets, domain)
    return str(dict(manifest.source or {}).get("checked_through", "") or manifest.end_date or "2010-01-03")


def _mark_checked(
    ctx: AuxiliaryContext,
    domain: str,
    *,
    missing_daily_keys: int | None = None,
    source_updates: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    updates: dict[str, Any] = {
        "checked_through": ctx.target_date,
        **dict(source_updates or {}),
    }
    if missing_daily_keys is not None:
        updates["missing_daily_keys"] = int(missing_daily_keys)
    return update_active_manifest_metadata(
        domain,
        reason=f"free-source auxiliary tail checked through {ctx.target_date}",
        workspace_root=ctx.workspace,
        source_updates=updates,
        quality_updates={"future_source_dates": 0},
    )


def _append_if_any(
    ctx: AuxiliaryContext,
    domain: str,
    frame: pd.DataFrame,
) -> dict[str, Any]:
    if frame.empty:
        return {"status": "already_complete", "row_count": 0}
    commit = append_active_shard(
        domain,
        frame,
        f"append free-source auxiliary tail through {ctx.target_date}",
        workspace_root=ctx.workspace,
    )
    return {"status": "updated", "row_count": int(len(frame)), "commit": commit}
