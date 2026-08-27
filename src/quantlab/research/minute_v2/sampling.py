"""Causal stock-attention gate evaluated at every decision minute."""

from __future__ import annotations

from typing import Any

import duckdb
import pandas as pd

from .contracts import KEY_COLUMNS, MinuteV2Config, MinuteV2Error

EVENT_FLAG_COLUMNS = (
    "event_extreme_return",
    "event_volume_shock",
    "event_industry_leadership",
    "event_vwap_cross",
    "event_liquidity_anchor",
    "event_background_control",
)

CANDIDATE_COLUMNS = (
    "candidate_attention",
    "candidate_background_control",
    "candidate_selected",
    "candidate_reason_mask",
)


def event_query(
    *,
    feature_view: str = "minute_features",
    config: MinuteV2Config,
) -> str:
    """Keep causal stock candidates while preserving every decision-time group."""

    config.validate()
    background_percent = int(config.candidate_background_percent)
    liquidity = float(config.minimum_daily_liquidity_rank)
    stock_floor = float(config.candidate_stock_rank_floor)
    industry_floor = float(config.candidate_industry_rank_floor)
    industry_stock_floor = float(config.candidate_industry_stock_rank_floor)
    volume_floor = float(config.candidate_volume_rank_floor)
    return f"""
        WITH flags AS (
            SELECT
                *,
                (
                    stock_return_rank >= {stock_floor}
                    OR stock_return_5m_rank >= {stock_floor}
                ) AS event_extreme_return,
                (
                    stock_volume_acceleration_rank >= {volume_floor}
                    AND stock_amount_rank >= 0.50
                    AND ABS(COALESCE(return_from_previous_close, 0.0)) >= 0.001
                ) AS event_volume_shock,
                (
                    industry_strength_rank >= {industry_floor}
                    AND industry_stock_return_rank >= {industry_stock_floor}
                    AND COALESCE(industry_return_mean, 0.0) > 0.0
                ) AS event_industry_leadership,
                (
                    previous_vwap_deviation IS NOT NULL
                    AND vwap_deviation IS NOT NULL
                    AND SIGN(previous_vwap_deviation) <> SIGN(vwap_deviation)
                    AND ABS(COALESCE(return_5m, 0.0)) >= 0.001
                ) AS event_vwap_cross,
                stock_amount_rank >= 0.90 AS event_liquidity_anchor,
                (HASH(symbol, trade_date, bar_time) % 100) < {background_percent}
                    AS event_background_control
            FROM {feature_view}
        ),
        candidates AS (
            SELECT
                *,
                (
                    daily_liquidity_rank >= {liquidity}
                    AND (
                        event_extreme_return
                        OR event_volume_shock
                        OR event_industry_leadership
                        OR event_vwap_cross
                        OR event_liquidity_anchor
                    )
                ) AS candidate_attention,
                (
                    daily_liquidity_rank >= {liquidity}
                    AND event_background_control
                ) AS candidate_background_control
            FROM flags
        ),
        encoded AS (
            SELECT
                *,
                candidate_attention OR candidate_background_control AS candidate_selected,
                CAST(event_extreme_return AS INTEGER)
                    + 2 * CAST(event_volume_shock AS INTEGER)
                    + 4 * CAST(event_industry_leadership AS INTEGER)
                    + 8 * CAST(event_vwap_cross AS INTEGER)
                    + 16 * CAST(event_liquidity_anchor AS INTEGER)
                    + 32 * CAST(event_background_control AS INTEGER)
                    AS candidate_reason_mask
            FROM candidates
        )
        SELECT *, candidate_reason_mask AS event_mask
        FROM encoded
        WHERE candidate_selected
        ORDER BY trade_date, bar_time, symbol
    """


def build_event_frame(
    features: pd.DataFrame,
    *,
    config: MinuteV2Config | None = None,
    connection: Any | None = None,
) -> pd.DataFrame:
    current = config or MinuteV2Config()
    missing = sorted(set(KEY_COLUMNS).difference(features.columns))
    if missing:
        raise MinuteV2Error(f"minute_v2_event_keys_missing:{','.join(missing)}")
    owned = connection is None
    con = duckdb.connect(":memory:") if owned else connection
    try:
        con.register("minute_features", features)
        result = con.execute(event_query(config=current)).fetchdf()
    finally:
        if owned:
            con.close()
    if result.empty:
        raise MinuteV2Error("minute_v2_candidate_gate_empty")
    if result.duplicated(list(KEY_COLUMNS)).any():
        raise MinuteV2Error("minute_v2_duplicate_event_keys")
    source_sizes = features.groupby(["trade_date", "bar_time"], sort=False).size()
    result_sizes = result.groupby(["trade_date", "bar_time"], sort=False).size()
    if len(result_sizes) != len(source_sizes):
        raise MinuteV2Error(
            f"minute_v2_candidate_time_groups_missing:{len(result_sizes)}:{len(source_sizes)}"
        )
    expected_sizes = source_sizes.reindex(result_sizes.index)
    if result_sizes.gt(expected_sizes).any():
        raise MinuteV2Error("minute_v2_candidate_group_exceeds_universe")
    return result


__all__ = [
    "CANDIDATE_COLUMNS",
    "EVENT_FLAG_COLUMNS",
    "build_event_frame",
    "event_query",
]
