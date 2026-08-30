"""Auxiliary Tail Update: shares responsibilities."""

from __future__ import annotations

import shutil
from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any

import pandas as pd

from quantlab.data.domains.contracts.schema import DataDomain
from quantlab.data.providers.mootdx.frames import _mootdx_xdxr_domain_frame
from quantlab.data.providers.mootdx.provider import MootdxOnlineProvider
from quantlab.data.qdp_v2.auxiliary_update.context import (
    SECONDARY_VALIDATION_WORKERS,
    AuxiliaryContext,
    _external_with_retry,
    _paths,
    _scan_sql,
)
from quantlab.data.qdp_v2.auxiliary_update.corporate import _current_symbols
from quantlab.data.qdp_v2.duckdb_resources import open_guarded_duckdb

from .common import (
    _append_if_any,
    _mark_checked,
    _missing_daily_keys,
    _only_changed_share_events,
    _unconfirmed_share_detections,
)
from .config import (
    AuxiliaryTailUpdateError,
    _normalize_cninfo_share_change,
)


def _fetch_mootdx_evidence(
    ctx: AuxiliaryContext,
    *,
    start_date: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    symbols = _current_symbols(ctx)
    provider = MootdxOnlineProvider()
    try:
        raw = provider.fetch_xdxr_raw(symbols)
    finally:
        provider.close()
    if raw.error_report:
        raise AuxiliaryTailUpdateError(f"mootdx_xdxr_symbol_errors:{len(raw.error_report)}")
    share = _mootdx_xdxr_domain_frame(
        raw.data,
        domain=DataDomain.SHARE_CAPITAL,
        source="mootdx_online",
        as_of_date=ctx.target_date,
    )
    corporate = _mootdx_xdxr_domain_frame(
        raw.data,
        domain=DataDomain.CORPORATE_ACTIONS,
        source="mootdx_online",
        as_of_date=ctx.target_date,
    )
    share = _only_changed_share_events(share)
    for frame in (share, corporate):
        if not frame.empty:
            frame.drop(
                frame.index[~frame["trade_date"].astype(str).between(start_date, ctx.target_date, inclusive="right")],
                inplace=True,
            )
            frame.reset_index(drop=True, inplace=True)
    return share, corporate


def _fetch_cninfo_share_events(
    ctx: AuxiliaryContext,
    symbols: Sequence[str],
) -> pd.DataFrame:
    if not symbols:
        return pd.DataFrame(
            columns=[
                "symbol",
                "variation_date",
                "source_date",
                "total_share",
                "float_share",
                "source",
            ]
        )
    import akshare as ak

    def fetch_one(symbol: str) -> pd.DataFrame:
        code = symbol.split(".", 1)[0]
        raw = _external_with_retry(
            lambda: ak.stock_share_change_cninfo(
                symbol=code,
                start_date="19900101",
                end_date=ctx.target_date.replace("-", ""),
            ),
            label=f"tail_share_change:{symbol}",
        )
        if not isinstance(raw, pd.DataFrame) or raw.empty:
            return pd.DataFrame()
        return _normalize_cninfo_share_change(
            raw,
            symbol=symbol,
            target_date=ctx.target_date,
            source="cninfo_confirmed_mootdx_xdxr",
        )

    frames: list[pd.DataFrame] = []
    with ThreadPoolExecutor(max_workers=SECONDARY_VALIDATION_WORKERS) as pool:
        futures = {pool.submit(fetch_one, symbol): symbol for symbol in symbols}
        for future in as_completed(futures):
            frame = future.result()
            if not frame.empty:
                frames.append(frame)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def _unseeded_share_symbols(
    ctx: AuxiliaryContext,
    *,
    missing: pd.DataFrame,
    current: str,
) -> set[str]:
    spill = ctx.runtime / "tail_share_prior_spill"
    with open_guarded_duckdb(temp_directory=spill, threads=1) as con:
        con.register("tail_missing_keys", missing)
        symbols = {
            str(item[0])
            for item in con.execute(
                f"""
                SELECT DISTINCT m.symbol
                FROM tail_missing_keys m
                WHERE NOT EXISTS (
                  SELECT 1 FROM {current} c
                  WHERE c.symbol=m.symbol AND c.trade_date<m.trade_date
                )
                """
            ).fetchall()
        }
    shutil.rmtree(spill, ignore_errors=True)
    return symbols


def _empty_confirmed_share_events() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "symbol": pd.Series(dtype="object"),
            "variation_date": pd.Series(dtype="object"),
            "source_date": pd.Series(dtype="object"),
            "total_share": pd.Series(dtype="float64"),
            "float_share": pd.Series(dtype="float64"),
            "source": pd.Series(dtype="object"),
        }
    )


def _build_share_tail_frame(
    ctx: AuxiliaryContext,
    *,
    missing: pd.DataFrame,
    current: str,
    confirmed: pd.DataFrame,
) -> pd.DataFrame:
    spill = ctx.runtime / "tail_share_build_spill"
    with open_guarded_duckdb(temp_directory=spill, threads=2) as con:
        con.register("tail_missing_keys", missing)
        con.register(
            "tail_confirmed_share",
            confirmed if not confirmed.empty else _empty_confirmed_share_events(),
        )
        frame = con.execute(
            f"""
            WITH candidates AS (
              SELECT symbol,
                     cast(greatest(
                       try_cast(total_share_source_date AS DATE),
                       try_cast(float_share_source_date AS DATE)
                     ) AS VARCHAR) AS source_date,
                     total_share, float_share, source, 1 AS priority
              FROM {current}
              WHERE total_share>0 AND float_share>=0 AND float_share<=total_share
                AND try_cast(total_share_source_date AS DATE)<=try_cast(trade_date AS DATE)
                AND try_cast(float_share_source_date AS DATE)<=try_cast(trade_date AS DATE)
              UNION ALL
              SELECT symbol, source_date, total_share, float_share, source,
                     2 AS priority
              FROM tail_confirmed_share
            ), dedup AS (
              SELECT symbol, source_date, total_share, float_share, source
              FROM candidates
              QUALIFY row_number() OVER(
                PARTITION BY symbol, source_date ORDER BY priority DESC
              )=1
            )
            SELECT m.symbol, m.trade_date, c.total_share, c.float_share,
                   greatest(c.total_share-c.float_share, 0.0) AS restricted_share,
                   c.source_date AS total_share_source_date,
                   c.source_date AS float_share_source_date,
                   c.source_date AS restricted_share_source_date,
                   CASE WHEN c.source_date=m.trade_date THEN 'direct_daily'
                        ELSE 'prior_ffill' END AS share_fill_method,
                   c.source
            FROM tail_missing_keys m
            ASOF LEFT JOIN dedup c
              ON m.symbol=c.symbol AND m.trade_date>=c.source_date
            ORDER BY m.trade_date, m.symbol
            """
        ).fetchdf()
    shutil.rmtree(spill, ignore_errors=True)
    return frame


def _unconfirmed_detection_records(
    detections: Sequence[tuple[str, str]],
) -> list[dict[str, str]]:
    return [{"symbol": symbol, "detector_date": trade_date} for symbol, trade_date in detections[:20]]


def update_share_capital_tail(
    ctx: AuxiliaryContext,
    *,
    detected_events: pd.DataFrame,
) -> dict[str, Any]:
    missing = _missing_daily_keys(ctx, "share_capital")
    if missing.empty:
        metadata = _mark_checked(ctx, "share_capital", missing_daily_keys=0)
        return {"status": "already_complete", "row_count": 0, "metadata": metadata}
    current = _scan_sql(_paths(ctx, "share_capital"))
    affected = _unseeded_share_symbols(ctx, missing=missing, current=current)
    if not detected_events.empty:
        affected.update(detected_events["symbol"].astype(str))
    confirmed = _fetch_cninfo_share_events(ctx, sorted(affected))
    if affected and confirmed.empty:
        raise AuxiliaryTailUpdateError(f"share_tail_cninfo_confirmation_empty:{len(affected)}")
    # Mootdx xdxr is a detector, not the PIT authority.  Its dates can be a
    # weekend publication/update date and the first row returned for a symbol
    # can simply repeat the already-known share count.  Requiring an exact
    # (symbol, date) match would therefore block a healthy tail or, worse,
    # tempt callers to apply an unconfirmed secondary-source value.  Only the
    # CNInfo rows below are eligible to change the share history; unmatched
    # detector rows are retained as audit evidence while the last officially
    # visible value is carried forward.
    unconfirmed_detections = _unconfirmed_share_detections(
        detected_events,
        confirmed,
    )
    frame = _build_share_tail_frame(
        ctx,
        missing=missing,
        current=current,
        confirmed=confirmed,
    )
    if len(frame) != len(missing) or frame["total_share"].isna().any():
        raise AuxiliaryTailUpdateError(f"share_tail_unresolved:{len(missing) - frame['total_share'].notna().sum()}")
    result = _append_if_any(ctx, "share_capital", frame)
    remaining = _missing_daily_keys(ctx, "share_capital")
    if not remaining.empty:
        raise AuxiliaryTailUpdateError(f"share_tail_post_commit_missing:{len(remaining)}")
    result["detected_event_count"] = int(len(detected_events))
    result["confirmed_symbol_count"] = len(affected)
    result["unconfirmed_detection_count"] = len(unconfirmed_detections)
    examples = _unconfirmed_detection_records(unconfirmed_detections)
    result["unconfirmed_detection_examples"] = examples
    result["metadata"] = _mark_checked(
        ctx,
        "share_capital",
        missing_daily_keys=0,
        source_updates={
            "tail_unconfirmed_detector_event_count": len(unconfirmed_detections),
            "tail_unconfirmed_detector_event_examples": examples,
        },
    )
    return result
