"""Causal minute feature SQL shared by local tests and production builds."""

from __future__ import annotations

from typing import Any

import duckdb
import pandas as pd

from .contracts import (
    AFTERNOON_DECISION_END,
    AFTERNOON_DECISION_START,
    BASE_FEATURE_COLUMNS,
    CROSS_SECTION_FEATURE_COLUMNS,
    EXPECTED_SESSION_BARS,
    KEY_COLUMNS,
    MORNING_DECISION_END,
    MORNING_DECISION_START,
    MinuteV2Error,
)

BAR_COLUMNS = (
    "symbol",
    "trade_date",
    "bar_time",
    "open",
    "high",
    "low",
    "close",
    "volume",
    "amount",
)

STOCK_DAY_COLUMNS = (
    "symbol",
    "trade_date",
    "industry_name",
    "adjust_factor",
    "previous_close",
    "auction_price",
    "auction_amount",
    "previous_return_1d",
    "previous_return_5d",
    "previous_return_20d",
    "previous_volatility_20d",
    "previous_amount_20d",
    "daily_liquidity_rank",
    "exclude_open",
    "exclude_high",
    "exclude_low",
    "exclude_close",
)


def feature_query(
    *,
    bars_view: str = "minute_bars",
    stock_days_view: str = "stock_days",
    expected_session_bars: int = EXPECTED_SESSION_BARS,
) -> str:
    """Return one relational statement whose rows only depend on bars at or before ``bar_time``."""

    expected = int(expected_session_bars)
    if expected <= 0:
        raise MinuteV2Error("minute_v2_expected_session_bars_invalid")
    return f"""
        WITH joined AS (
            SELECT
                b.symbol,
                b.trade_date,
                b.bar_time,
                CAST(b.open AS DOUBLE) AS open,
                CAST(b.high AS DOUBLE) AS high,
                CAST(b.low AS DOUBLE) AS low,
                CAST(b.close AS DOUBLE) AS close,
                CAST(b.volume AS DOUBLE) AS volume,
                CAST(b.amount AS DOUBLE) AS amount,
                COALESCE(NULLIF(s.industry_name, ''), 'UNKNOWN') AS industry_name,
                CAST(s.adjust_factor AS DOUBLE) AS adjust_factor,
                CAST(s.previous_close AS DOUBLE) AS previous_close,
                CAST(s.auction_price AS DOUBLE) AS auction_price,
                CAST(s.auction_amount AS DOUBLE) AS auction_amount,
                CAST(s.previous_return_1d AS DOUBLE) AS previous_return_1d,
                CAST(s.previous_return_5d AS DOUBLE) AS previous_return_5d,
                CAST(s.previous_return_20d AS DOUBLE) AS previous_return_20d,
                CAST(s.previous_volatility_20d AS DOUBLE) AS previous_volatility_20d,
                CAST(s.previous_amount_20d AS DOUBLE) AS previous_amount_20d,
                CAST(s.daily_liquidity_rank AS DOUBLE) AS daily_liquidity_rank,
                NOT COALESCE(s.exclude_open, FALSE) AS valid_open,
                NOT COALESCE(s.exclude_high, FALSE) AS valid_high,
                NOT COALESCE(s.exclude_low, FALSE) AS valid_low,
                NOT COALESCE(s.exclude_close, FALSE) AS valid_close,
                ROW_NUMBER() OVER w AS minute_index,
                COUNT(*) OVER (PARTITION BY b.symbol, b.trade_date) AS session_bar_count,
                LAG(CAST(b.close AS DOUBLE), 1) OVER w AS close_lag_1,
                LAG(CAST(b.close AS DOUBLE), 3) OVER w AS close_lag_3,
                LAG(CAST(b.close AS DOUBLE), 5) OVER w AS close_lag_5,
                LAG(CAST(b.close AS DOUBLE), 15) OVER w AS close_lag_15,
                LAG(CAST(b.close AS DOUBLE), 30) OVER w AS close_lag_30,
                LAG(CAST(b.close AS DOUBLE), 60) OVER w AS close_lag_60,
                FIRST_VALUE(CAST(b.open AS DOUBLE)) OVER wc AS day_open,
                MAX(CAST(b.high AS DOUBLE)) OVER wc AS cumulative_high,
                MIN(CAST(b.low AS DOUBLE)) OVER wc AS cumulative_low,
                SUM(GREATEST(CAST(b.volume AS DOUBLE), 0.0)) OVER wc AS cumulative_volume,
                SUM(GREATEST(CAST(b.amount AS DOUBLE), 0.0)) OVER wc AS cumulative_amount,
                AVG(CAST(b.volume AS DOUBLE)) OVER (
                    PARTITION BY b.symbol, b.trade_date ORDER BY b.bar_time
                    ROWS BETWEEN 4 PRECEDING AND CURRENT ROW
                ) AS recent_volume_5,
                AVG(CAST(b.volume AS DOUBLE)) OVER (
                    PARTITION BY b.symbol, b.trade_date ORDER BY b.bar_time
                    ROWS BETWEEN 24 PRECEDING AND 5 PRECEDING
                ) AS previous_volume_20,
                MAX(CAST(b.high AS DOUBLE)) OVER (
                    PARTITION BY b.symbol, b.trade_date ORDER BY b.bar_time
                    ROWS BETWEEN 20 PRECEDING AND 1 PRECEDING
                ) AS previous_high_20
            FROM {bars_view} b
            JOIN {stock_days_view} s USING(symbol, trade_date)
            WINDOW
                w AS (PARTITION BY b.symbol, b.trade_date ORDER BY b.bar_time),
                wc AS (
                    PARTITION BY b.symbol, b.trade_date ORDER BY b.bar_time
                    ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW
                )
        ),
        returns AS (
            SELECT
                *,
                CASE WHEN close > 0 AND close_lag_1 > 0 THEN LN(close / close_lag_1) END AS log_return_1m,
                CASE WHEN close > 0 AND close_lag_1 > 0 THEN close / close_lag_1 - 1.0 END AS raw_return_1m,
                CASE WHEN cumulative_volume > 0 THEN cumulative_amount / cumulative_volume END AS cumulative_vwap,
                CASE WHEN high > low THEN (close - low) / (high - low) END AS raw_bar_close_location,
                CASE WHEN cumulative_high > cumulative_low
                     THEN (close - cumulative_low) / (cumulative_high - cumulative_low)
                END AS raw_cumulative_close_location
            FROM joined
            WHERE session_bar_count = {expected}
        ),
        rolling AS (
            SELECT
                *,
                STDDEV_SAMP(log_return_1m) OVER (
                    PARTITION BY symbol, trade_date ORDER BY bar_time ROWS BETWEEN 4 PRECEDING AND CURRENT ROW
                ) AS raw_realized_volatility_5m,
                STDDEV_SAMP(log_return_1m) OVER (
                    PARTITION BY symbol, trade_date ORDER BY bar_time ROWS BETWEEN 14 PRECEDING AND CURRENT ROW
                ) AS raw_realized_volatility_15m,
                STDDEV_SAMP(log_return_1m) OVER (
                    PARTITION BY symbol, trade_date ORDER BY bar_time ROWS BETWEEN 29 PRECEDING AND CURRENT ROW
                ) AS raw_realized_volatility_30m,
                SQRT(AVG(CASE WHEN log_return_1m < 0 THEN log_return_1m * log_return_1m ELSE 0.0 END) OVER (
                    PARTITION BY symbol, trade_date ORDER BY bar_time ROWS BETWEEN 14 PRECEDING AND CURRENT ROW
                )) AS raw_downside_volatility_15m,
                SUM(ABS(log_return_1m)) OVER (
                    PARTITION BY symbol, trade_date ORDER BY bar_time
                    ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW
                ) AS cumulative_absolute_return
            FROM returns
        ),
        base AS (
            SELECT
                symbol,
                trade_date,
                bar_time,
                industry_name,
                adjust_factor,
                open,
                high,
                low,
                close,
                volume,
                amount,
                valid_open,
                valid_high,
                valid_low,
                valid_close,
                CAST(minute_index AS SMALLINT) AS minute_index,
                CAST(minute_index AS DOUBLE) / {expected}.0 AS minute_fraction,
                SIN(2.0 * PI() * CAST(minute_index AS DOUBLE) / {expected}.0) AS time_sin,
                COS(2.0 * PI() * CAST(minute_index AS DOUBLE) / {expected}.0) AS time_cos,
                CASE WHEN valid_close THEN raw_return_1m END AS return_1m,
                CASE WHEN valid_close AND close_lag_3 > 0 THEN close / close_lag_3 - 1.0 END AS return_3m,
                CASE WHEN valid_close AND close_lag_5 > 0 THEN close / close_lag_5 - 1.0 END AS return_5m,
                CASE WHEN valid_close AND close_lag_15 > 0 THEN close / close_lag_15 - 1.0 END AS return_15m,
                CASE WHEN valid_close AND close_lag_30 > 0 THEN close / close_lag_30 - 1.0 END AS return_30m,
                CASE WHEN valid_close AND close_lag_60 > 0 THEN close / close_lag_60 - 1.0 END AS return_60m,
                CASE WHEN valid_close AND previous_close > 0 THEN close / previous_close - 1.0 END
                    AS return_from_previous_close,
                CASE WHEN valid_close AND valid_open AND day_open > 0 THEN close / day_open - 1.0 END
                    AS return_from_open,
                CASE WHEN valid_high AND valid_low AND low > 0 THEN high / low - 1.0 END AS bar_range,
                CASE WHEN valid_high AND valid_low AND valid_close THEN raw_bar_close_location END
                    AS bar_close_location,
                CASE WHEN valid_high AND valid_low AND cumulative_low > 0
                     THEN cumulative_high / cumulative_low - 1.0 END AS cumulative_range,
                CASE WHEN valid_high AND valid_low AND valid_close THEN raw_cumulative_close_location END
                    AS cumulative_close_location,
                CASE WHEN valid_close AND cumulative_vwap > 0 THEN close / cumulative_vwap - 1.0 END
                    AS vwap_deviation,
                CASE WHEN valid_close THEN raw_realized_volatility_5m END AS realized_volatility_5m,
                CASE WHEN valid_close THEN raw_realized_volatility_15m END AS realized_volatility_15m,
                CASE WHEN valid_close THEN raw_realized_volatility_30m END AS realized_volatility_30m,
                CASE WHEN valid_close THEN raw_downside_volatility_15m END AS downside_volatility_15m,
                CASE WHEN previous_volume_20 > 0 THEN recent_volume_5 / previous_volume_20 - 1.0 END
                    AS volume_acceleration_5_20,
                CASE WHEN previous_amount_20d > 0 AND minute_index > 0
                     THEN cumulative_amount / (previous_amount_20d * minute_index / {expected}.0) - 1.0
                END AS amount_curve_surprise,
                LN(1.0 + GREATEST(amount, 0.0)) AS log_bar_amount,
                CASE WHEN amount > 0 AND raw_return_1m IS NOT NULL
                     THEN ABS(raw_return_1m) / (amount / 1000000.0 + 1.0)
                END AS price_impact_1m,
                CASE WHEN valid_close AND cumulative_absolute_return > 0 AND day_open > 0
                     THEN ABS(LN(close / day_open)) / cumulative_absolute_return
                END AS trend_efficiency,
                CASE WHEN valid_close AND valid_high AND previous_high_20 > 0
                     THEN close / previous_high_20 - 1.0 END AS breakout_20m,
                CASE WHEN valid_close AND valid_high AND cumulative_high > 0
                     THEN close / cumulative_high - 1.0 END AS drawdown_from_day_high,
                CASE WHEN valid_close AND valid_low AND cumulative_low > 0
                     THEN close / cumulative_low - 1.0 END AS rebound_from_day_low,
                CASE WHEN valid_open AND previous_close > 0 AND auction_price > 0
                     THEN auction_price / previous_close - 1.0 END AS auction_gap,
                CASE WHEN previous_amount_20d > 0 AND auction_amount > 0
                     THEN auction_amount / previous_amount_20d END AS auction_amount_to_daily20,
                previous_return_1d,
                previous_return_5d,
                previous_return_20d,
                previous_volatility_20d,
                LN(1.0 + GREATEST(previous_amount_20d, 0.0)) AS previous_log_amount_20d,
                daily_liquidity_rank
            FROM rolling
            WHERE
                (bar_time BETWEEN '{MORNING_DECISION_START}' AND '{MORNING_DECISION_END}')
                OR (bar_time BETWEEN '{AFTERNOON_DECISION_START}' AND '{AFTERNOON_DECISION_END}')
        ),
        market_stats AS (
            SELECT
                trade_date,
                bar_time,
                AVG(return_from_previous_close) AS market_return_mean,
                AVG(return_5m) AS market_return_5m_mean,
                AVG(CASE WHEN return_from_previous_close > 0 THEN 1.0 ELSE 0.0 END)
                    AS market_breadth_positive,
                STDDEV_SAMP(return_from_previous_close) AS market_return_dispersion,
                LN(1.0 + SUM(GREATEST(amount, 0.0))) AS market_log_total_amount,
                SUM(GREATEST(amount, 0.0)) AS market_total_amount
            FROM base
            GROUP BY trade_date, bar_time
        ),
        industry_stats AS (
            SELECT
                trade_date,
                bar_time,
                industry_name,
                COUNT(*) AS industry_member_count,
                AVG(return_from_previous_close) AS industry_return_mean,
                AVG(return_5m) AS industry_return_5m_mean,
                AVG(CASE WHEN return_from_previous_close > 0 THEN 1.0 ELSE 0.0 END)
                    AS industry_breadth_positive,
                STDDEV_SAMP(return_from_previous_close) AS industry_return_dispersion,
                SUM(GREATEST(amount, 0.0)) AS industry_total_amount
            FROM base
            GROUP BY trade_date, bar_time, industry_name
        ),
        industry_ranked AS (
            SELECT
                *,
                PERCENT_RANK() OVER (
                    PARTITION BY trade_date, bar_time ORDER BY industry_return_mean NULLS FIRST
                ) AS industry_strength_rank
            FROM industry_stats
        ),
        enriched AS (
            SELECT
                b.*,
                m.market_return_mean,
                m.market_return_5m_mean,
                m.market_breadth_positive,
                m.market_return_dispersion,
                m.market_log_total_amount,
                i.industry_member_count,
                i.industry_return_mean,
                i.industry_return_5m_mean,
                i.industry_breadth_positive,
                i.industry_return_dispersion,
                CASE WHEN m.market_total_amount > 0
                     THEN i.industry_total_amount / m.market_total_amount END AS industry_amount_share,
                i.industry_strength_rank,
                CASE WHEN b.return_from_previous_close IS NOT NULL THEN PERCENT_RANK() OVER (
                    PARTITION BY b.trade_date, b.bar_time
                    ORDER BY b.return_from_previous_close NULLS FIRST
                ) END AS stock_return_rank,
                CASE WHEN b.return_5m IS NOT NULL THEN PERCENT_RANK() OVER (
                    PARTITION BY b.trade_date, b.bar_time ORDER BY b.return_5m NULLS FIRST
                ) END AS stock_return_5m_rank,
                CASE WHEN b.volume_acceleration_5_20 IS NOT NULL THEN PERCENT_RANK() OVER (
                    PARTITION BY b.trade_date, b.bar_time
                    ORDER BY b.volume_acceleration_5_20 NULLS FIRST
                ) END AS stock_volume_acceleration_rank,
                PERCENT_RANK() OVER (
                    PARTITION BY b.trade_date, b.bar_time ORDER BY b.amount NULLS FIRST
                ) AS stock_amount_rank,
                CASE WHEN b.return_from_previous_close IS NOT NULL THEN PERCENT_RANK() OVER (
                    PARTITION BY b.trade_date, b.bar_time, b.industry_name
                    ORDER BY b.return_from_previous_close NULLS FIRST
                ) END AS industry_stock_return_rank,
                PERCENT_RANK() OVER (
                    PARTITION BY b.trade_date, b.bar_time, b.industry_name ORDER BY b.amount NULLS FIRST
                ) AS industry_stock_amount_rank
            FROM base b
            JOIN market_stats m USING(trade_date, bar_time)
            JOIN industry_ranked i USING(trade_date, bar_time, industry_name)
        )
        SELECT
            *,
            return_from_previous_close - market_return_mean AS market_residual_return,
            return_from_previous_close - industry_return_mean AS industry_residual_return,
            return_5m - industry_return_5m_mean AS industry_residual_return_5m,
            LAG(vwap_deviation) OVER (
                PARTITION BY symbol, trade_date ORDER BY bar_time
            ) AS previous_vwap_deviation
        FROM enriched
        ORDER BY trade_date, bar_time, symbol
    """


def build_feature_frame(
    bars: pd.DataFrame,
    stock_days: pd.DataFrame,
    *,
    expected_session_bars: int = EXPECTED_SESSION_BARS,
    connection: Any | None = None,
) -> pd.DataFrame:
    """Materialize features for fixtures or bounded local probes."""

    missing_bars = sorted(set(BAR_COLUMNS).difference(bars.columns))
    missing_days = sorted(set(STOCK_DAY_COLUMNS).difference(stock_days.columns))
    if missing_bars:
        raise MinuteV2Error(f"minute_v2_bar_columns_missing:{','.join(missing_bars)}")
    if missing_days:
        raise MinuteV2Error(f"minute_v2_stock_day_columns_missing:{','.join(missing_days)}")
    if bars.duplicated(list(KEY_COLUMNS)).any():
        raise MinuteV2Error("minute_v2_duplicate_bar_keys")
    if stock_days.duplicated(["symbol", "trade_date"]).any():
        raise MinuteV2Error("minute_v2_duplicate_stock_day_keys")
    owned = connection is None
    con = duckdb.connect(":memory:") if owned else connection
    try:
        con.register("minute_bars", bars)
        con.register("stock_days", stock_days)
        result = con.execute(
            feature_query(expected_session_bars=expected_session_bars)
        ).fetchdf()
    finally:
        if owned:
            con.close()
    expected_features = set(BASE_FEATURE_COLUMNS) | set(CROSS_SECTION_FEATURE_COLUMNS)
    missing_features = sorted(expected_features.difference(result.columns))
    if missing_features:
        raise MinuteV2Error(
            f"minute_v2_materialized_features_missing:{','.join(missing_features)}"
        )
    if result.duplicated(list(KEY_COLUMNS)).any():
        raise MinuteV2Error("minute_v2_duplicate_feature_keys")
    return result


__all__ = [
    "BAR_COLUMNS",
    "STOCK_DAY_COLUMNS",
    "build_feature_frame",
    "feature_query",
]
