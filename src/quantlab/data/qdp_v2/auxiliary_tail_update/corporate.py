"""Auxiliary Tail Update: corporate responsibilities."""

from __future__ import annotations

import shutil
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any

import pandas as pd

from quantlab.data.qdp_v2.auxiliary_update import (
    CORPORATE_COLUMNS,
    SECONDARY_VALIDATION_WORKERS,
    AuxiliaryContext,
    _external_with_retry,
    _paths,
    _scan_sql,
)
from quantlab.data.qdp_v2.duckdb_resources import open_guarded_duckdb

from .common import (
    _append_if_any,
    _checked_through,
    _mark_checked,
)
from .config import (
    AuxiliaryTailUpdateError,
    _normalize_cninfo_dividend,
)


def update_corporate_actions_tail(
    ctx: AuxiliaryContext,
    *,
    detected_events: pd.DataFrame,
) -> dict[str, Any]:
    if detected_events.empty:
        metadata = _mark_checked(ctx, "corporate_actions")
        return {"status": "already_complete", "row_count": 0, "metadata": metadata}
    import akshare as ak

    symbols = sorted(set(detected_events["symbol"].astype(str)))

    def fetch_one(symbol: str) -> pd.DataFrame:
        code = symbol.split(".", 1)[0]

        def query() -> pd.DataFrame:
            try:
                return ak.stock_dividend_cninfo(symbol=code)
            except KeyError:
                return pd.DataFrame()

        raw = _external_with_retry(
            query,
            label=f"tail_corporate_actions:{symbol}",
        )
        return _normalize_cninfo_dividend(
            raw if isinstance(raw, pd.DataFrame) else pd.DataFrame(),
            symbol=symbol,
            target_date=ctx.target_date,
        )

    frames: list[pd.DataFrame] = []
    with ThreadPoolExecutor(max_workers=SECONDARY_VALIDATION_WORKERS) as pool:
        futures = {pool.submit(fetch_one, symbol): symbol for symbol in symbols}
        for future in as_completed(futures):
            frame = future.result()
            if not frame.empty:
                frames.append(frame)
    confirmed = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    if not confirmed.empty:
        announcement = pd.to_datetime(confirmed["announcement_date"], errors="coerce")
        ex_date = pd.to_datetime(confirmed["ex_date"], errors="coerce")
        record_date = pd.to_datetime(confirmed["record_date"], errors="coerce")
        pay_date = pd.to_datetime(confirmed["dividend_pay_date"], errors="coerce")
        implemented = (
            pd.to_numeric(confirmed["cash_dividend_per_10"], errors="coerce").fillna(0.0)
            + pd.to_numeric(confirmed["bonus_share_per_10"], errors="coerce").fillna(0.0)
            + pd.to_numeric(confirmed["transfer_share_per_10"], errors="coerce").fillna(0.0)
        )
        confirmed = confirmed.loc[
            announcement.notna()
            & ex_date.notna()
            & announcement.le(ex_date)
            & (record_date.isna() | record_date.le(ex_date))
            & (pay_date.isna() | pay_date.ge(ex_date))
            & implemented.gt(0)
        ].reset_index(drop=True)
    detected_keys = set(
        zip(
            detected_events["symbol"].astype(str),
            detected_events["trade_date"].astype(str),
            strict=True,
        )
    )
    confirmed_keys = (
        set(zip(confirmed["symbol"], confirmed["trade_date"], strict=True)) if not confirmed.empty else set()
    )
    unresolved = sorted(detected_keys - confirmed_keys)
    if unresolved:
        raise AuxiliaryTailUpdateError(f"corporate_tail_unconfirmed_events:{len(unresolved)}:{unresolved[:5]}")
    current = _scan_sql(_paths(ctx, "corporate_actions"))
    spill = ctx.runtime / "tail_corporate_build_spill"
    with open_guarded_duckdb(temp_directory=spill, threads=1) as con:
        con.register("tail_corporate_confirmed", confirmed)
        frame = con.execute(
            f"""
            SELECT n.* FROM tail_corporate_confirmed n
            ANTI JOIN {current} c USING(symbol, trade_date, action_type)
            WHERE n.trade_date>'{_checked_through(ctx, "corporate_actions")}'
              AND n.trade_date<='{ctx.target_date}'
            ORDER BY n.trade_date, n.symbol, n.action_type
            """
        ).fetchdf()
    shutil.rmtree(spill, ignore_errors=True)
    frame = frame.loc[:, list(CORPORATE_COLUMNS)] if not frame.empty else frame
    result = _append_if_any(ctx, "corporate_actions", frame)
    result["detected_event_count"] = len(detected_keys)
    result["confirmed_event_count"] = len(confirmed_keys)
    result["metadata"] = _mark_checked(
        ctx,
        "corporate_actions",
        source_updates={
            "secondary_validation_at": pd.Timestamp.utcnow().isoformat(),
            "secondary_compared_count": len(confirmed_keys),
            "secondary_material_mismatch_count": 0,
            "secondary_validation_status": "ok",
        },
    )
    return result
