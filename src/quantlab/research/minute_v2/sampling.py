"""Causal, deterministic event sampling for minute-v2 training."""

from __future__ import annotations

from typing import Any

import duckdb
import pandas as pd

from .contracts import KEY_COLUMNS, MinuteV2Config, MinuteV2Error


def event_query(*, feature_view: str = "minute_features", config: MinuteV2Config) -> str:
    config.validate()
    periodic = int(config.periodic_sample_every)
    random_percent = int(config.random_negative_percent)
    liquidity = float(config.minimum_daily_liquidity_rank)
    return f"""
        WITH flags AS (
            SELECT
                *,
                minute_index % {periodic} = 0 AS event_periodic,
                (
                    stock_return_5m_rank >= 0.97
                    OR stock_return_5m_rank <= 0.03
                ) AS event_extreme_return,
                (
                    stock_volume_acceleration_rank >= 0.97
                    AND ABS(COALESCE(return_from_previous_close, 0.0)) >= 0.003
                ) AS event_volume_shock,
                (
                    industry_strength_rank >= 0.90
                    AND industry_stock_return_rank >= 0.80
                    AND COALESCE(industry_return_mean, 0.0) > 0.0
                ) AS event_industry_leadership,
                (
                    previous_vwap_deviation IS NOT NULL
                    AND vwap_deviation IS NOT NULL
                    AND SIGN(previous_vwap_deviation) <> SIGN(vwap_deviation)
                    AND ABS(return_5m) >= 0.002
                ) AS event_vwap_cross,
                (
                    HASH(symbol, trade_date, bar_time) % 100
                ) < {random_percent} AS event_random_negative
            FROM {feature_view}
            WHERE daily_liquidity_rank >= {liquidity}
        )
        SELECT
            *,
            CAST(event_periodic AS INTEGER)
                + 2 * CAST(event_extreme_return AS INTEGER)
                + 4 * CAST(event_volume_shock AS INTEGER)
                + 8 * CAST(event_industry_leadership AS INTEGER)
                + 16 * CAST(event_vwap_cross AS INTEGER)
                + 32 * CAST(event_random_negative AS INTEGER) AS event_mask
        FROM flags
        WHERE
            event_periodic
            OR event_extreme_return
            OR event_volume_shock
            OR event_industry_leadership
            OR event_vwap_cross
            OR event_random_negative
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
    if result.duplicated(list(KEY_COLUMNS)).any():
        raise MinuteV2Error("minute_v2_duplicate_event_keys")
    return result


__all__ = ["build_event_frame", "event_query"]
