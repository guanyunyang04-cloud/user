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
    DAILY_WINDOWS,
    EXPECTED_SESSION_BARS,
    KEY_COLUMNS,
    MINUTE_WINDOWS,
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
    "previous_adjust_factor",
    "auction_price",
    "auction_amount",
    "previous_return_1d",
    "previous_return_5d",
    "previous_return_10d",
    "previous_return_20d",
    "previous_return_30d",
    "previous_return_60d",
    "previous_return_120d",
    "previous_return_240d",
    "previous_close_to_sma_5d",
    "previous_close_to_sma_10d",
    "previous_close_to_sma_20d",
    "previous_close_to_sma_30d",
    "previous_close_to_sma_60d",
    "previous_close_to_sma_120d",
    "previous_close_to_sma_240d",
    "previous_volatility_5d",
    "previous_volatility_10d",
    "previous_volatility_20d",
    "previous_volatility_30d",
    "previous_volatility_60d",
    "previous_volatility_120d",
    "previous_volatility_240d",
    "previous_amount_ratio_5d",
    "previous_amount_ratio_10d",
    "previous_amount_ratio_20d",
    "previous_amount_ratio_30d",
    "previous_amount_ratio_60d",
    "previous_amount_ratio_120d",
    "previous_amount_ratio_240d",
    "previous_amount_20d",
    "history_120d_available",
    "history_240d_available",
    "previous_total_share",
    "previous_float_share",
    "previous_total_mv",
    "previous_circ_mv",
    "previous_turnover_rate",
    "previous_pe",
    "previous_pb",
    "corporate_action_today",
    "cash_dividend_per_10",
    "bonus_share_per_10",
    "transfer_share_per_10",
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
    minute_lags = ",\n                ".join(
        expression
        for window in MINUTE_WINDOWS
        for expression in (
            f"LAG(CAST(b.close AS DOUBLE), {window}) OVER w AS close_lag_{window}",
            f"LAG(CAST(b.open AS DOUBLE), {window - 1}) OVER w AS open_lag_{window - 1}",
        )
    )
    daily_joined = ",\n                ".join(
        f"CAST(s.{name} AS DOUBLE) AS {name}"
        for window in DAILY_WINDOWS
        for name in (
            f"previous_return_{window}d",
            f"previous_close_to_sma_{window}d",
            f"previous_volatility_{window}d",
            f"previous_amount_ratio_{window}d",
        )
    )
    rolling_windows = ",\n                ".join(
        expression
        for window in MINUTE_WINDOWS
        for expression in (
            f"AVG(close) OVER (PARTITION BY symbol, trade_date ORDER BY bar_time "
            f"ROWS BETWEEN {window - 1} PRECEDING AND CURRENT ROW) AS raw_moving_average_{window}",
            f"MAX(high) OVER (PARTITION BY symbol, trade_date ORDER BY bar_time "
            f"ROWS BETWEEN {window - 1} PRECEDING AND CURRENT ROW) AS raw_rolling_high_{window}",
            f"MIN(low) OVER (PARTITION BY symbol, trade_date ORDER BY bar_time "
            f"ROWS BETWEEN {window - 1} PRECEDING AND CURRENT ROW) AS raw_rolling_low_{window}",
            f"AVG(volume) OVER (PARTITION BY symbol, trade_date ORDER BY bar_time "
            f"ROWS BETWEEN {window - 1} PRECEDING AND CURRENT ROW) AS raw_average_volume_{window}",
            f"STDDEV_SAMP(log_return_1m) OVER (PARTITION BY symbol, trade_date ORDER BY bar_time "
            f"ROWS BETWEEN {window - 1} PRECEDING AND CURRENT ROW) AS raw_realized_volatility_{window}",
            f"SUM(GREATEST(amount, 0.0)) OVER (PARTITION BY symbol, trade_date ORDER BY bar_time "
            f"ROWS BETWEEN {window - 1} PRECEDING AND CURRENT ROW) AS raw_window_amount_{window}",
        )
    )
    boundary_windows = ",\n                ".join(
        expression
        for window in MINUTE_WINDOWS
        for expression in (
            f"CASE WHEN minute_index >= {window} AND minute_index % {window} = 0 "
            f"AND valid_open AND valid_close AND open_lag_{window - 1} > 0 "
            f"THEN close / open_lag_{window - 1} - 1.0 END AS boundary_return_{window}",
            f"CASE WHEN minute_index >= {window} AND minute_index % {window} = 0 "
            f"AND valid_high AND valid_low AND raw_rolling_low_{window} > 0 "
            f"THEN raw_rolling_high_{window} / raw_rolling_low_{window} - 1.0 "
            f"END AS boundary_range_{window}",
            f"CASE WHEN minute_index >= {window} AND minute_index % {window} = 0 "
            f"AND valid_high AND valid_low AND valid_close "
            f"AND raw_rolling_high_{window} > raw_rolling_low_{window} "
            f"THEN (close - raw_rolling_low_{window}) "
            f"/ (raw_rolling_high_{window} - raw_rolling_low_{window}) "
            f"END AS boundary_close_location_{window}",
            f"CASE WHEN minute_index >= {window} AND minute_index % {window} = 0 "
            f"THEN LN(1.0 + raw_window_amount_{window}) END AS boundary_log_amount_{window}",
        )
    )
    completed_windows = ",\n                ".join(
        f"LAST_VALUE(boundary_{suffix}_{window} IGNORE NULLS) OVER ("
        "PARTITION BY symbol, trade_date ORDER BY bar_time "
        "ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW) "
        f"AS completed_kline_{suffix}_{window}m"
        for window in MINUTE_WINDOWS
        for suffix in ("return", "range", "close_location", "log_amount")
    )
    minute_features = ",\n                ".join(
        expression
        for window in MINUTE_WINDOWS
        for expression in (
            f"CASE WHEN valid_close AND minute_index > {window} AND close_lag_{window} > 0 "
            f"THEN close / close_lag_{window} - 1.0 END AS return_{window}m",
            f"CASE WHEN valid_close AND minute_index >= {window} "
            f"AND raw_moving_average_{window} > 0 "
            f"THEN close / raw_moving_average_{window} - 1.0 "
            f"END AS moving_average_deviation_{window}m",
            f"CASE WHEN valid_high AND valid_low AND minute_index >= {window} "
            f"AND raw_rolling_low_{window} > 0 "
            f"THEN raw_rolling_high_{window} / raw_rolling_low_{window} - 1.0 "
            f"END AS rolling_range_{window}m",
            f"CASE WHEN valid_close AND minute_index > {window} "
            f"THEN raw_realized_volatility_{window} END AS realized_volatility_{window}m",
            f"CASE WHEN minute_index >= {window} AND raw_average_volume_{window} > 0 "
            f"THEN volume / raw_average_volume_{window} - 1.0 END AS volume_ratio_{window}m",
        )
    )
    daily_features = ",\n                ".join(
        name
        for window in DAILY_WINDOWS
        for name in (
            f"previous_return_{window}d",
            f"previous_close_to_sma_{window}d",
            f"previous_volatility_{window}d",
            f"previous_amount_ratio_{window}d",
        )
    )
    fixed_features = ",\n                ".join(
        f"completed_kline_{suffix}_{window}m"
        for window in MINUTE_WINDOWS
        for suffix in ("return", "range", "close_location", "log_amount")
    )
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
                CAST(s.previous_adjust_factor AS DOUBLE) AS previous_adjust_factor,
                CAST(s.auction_price AS DOUBLE) AS auction_price,
                CAST(s.auction_amount AS DOUBLE) AS auction_amount,
                CAST(s.previous_return_1d AS DOUBLE) AS previous_return_1d,
                {daily_joined},
                CAST(s.previous_amount_20d AS DOUBLE) AS previous_amount_20d,
                CAST(s.history_120d_available AS DOUBLE) AS history_120d_available,
                CAST(s.history_240d_available AS DOUBLE) AS history_240d_available,
                CAST(s.previous_total_share AS DOUBLE) AS previous_total_share,
                CAST(s.previous_float_share AS DOUBLE) AS previous_float_share,
                CAST(s.previous_total_mv AS DOUBLE) AS previous_total_mv,
                CAST(s.previous_circ_mv AS DOUBLE) AS previous_circ_mv,
                CAST(s.previous_turnover_rate AS DOUBLE) AS previous_turnover_rate,
                CAST(s.previous_pe AS DOUBLE) AS previous_pe,
                CAST(s.previous_pb AS DOUBLE) AS previous_pb,
                CAST(s.corporate_action_today AS DOUBLE) AS corporate_action_today,
                CAST(s.cash_dividend_per_10 AS DOUBLE) AS cash_dividend_per_10,
                CAST(s.bonus_share_per_10 AS DOUBLE) AS bonus_share_per_10,
                CAST(s.transfer_share_per_10 AS DOUBLE) AS transfer_share_per_10,
                CAST(s.daily_liquidity_rank AS DOUBLE) AS daily_liquidity_rank,
                NOT COALESCE(s.exclude_open, FALSE) AS valid_open,
                NOT COALESCE(s.exclude_high, FALSE) AS valid_high,
                NOT COALESCE(s.exclude_low, FALSE) AS valid_low,
                NOT COALESCE(s.exclude_close, FALSE) AS valid_close,
                ROW_NUMBER() OVER w AS minute_index,
                COUNT(*) OVER (PARTITION BY b.symbol, b.trade_date) AS session_bar_count,
                LAG(CAST(b.close AS DOUBLE), 1) OVER w AS close_lag_1,
                LAG(CAST(b.close AS DOUBLE), 3) OVER w AS close_lag_3,
                LAG(CAST(b.close AS DOUBLE), 15) OVER w AS close_lag_15,
                {minute_lags},
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
                    PARTITION BY symbol, trade_date ORDER BY bar_time ROWS BETWEEN 14 PRECEDING AND CURRENT ROW
                ) AS raw_realized_volatility_15m,
                SQRT(AVG(CASE WHEN log_return_1m < 0 THEN log_return_1m * log_return_1m ELSE 0.0 END) OVER (
                    PARTITION BY symbol, trade_date ORDER BY bar_time ROWS BETWEEN 14 PRECEDING AND CURRENT ROW
                )) AS raw_downside_volatility_15m,
                SUM(ABS(log_return_1m)) OVER (
                    PARTITION BY symbol, trade_date ORDER BY bar_time
                    ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW
                ) AS cumulative_absolute_return,
                {rolling_windows}
            FROM returns
        ),
        boundaries AS (
            SELECT
                *,
                {boundary_windows}
            FROM rolling
        ),
        completed AS (
            SELECT
                *,
                {completed_windows}
            FROM boundaries
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
                CASE WHEN valid_close AND close_lag_15 > 0 THEN close / close_lag_15 - 1.0 END AS return_15m,
                {minute_features},
                {fixed_features},
                CASE WHEN valid_close AND previous_close > 0 AND previous_adjust_factor > 0
                     THEN close * adjust_factor / (previous_close * previous_adjust_factor) - 1.0 END
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
                CASE WHEN valid_close THEN raw_realized_volatility_15m END AS realized_volatility_15m,
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
                CASE WHEN valid_open AND previous_close > 0 AND previous_adjust_factor > 0
                          AND auction_price > 0
                     THEN auction_price * adjust_factor / (previous_close * previous_adjust_factor) - 1.0
                END AS auction_gap,
                CASE WHEN previous_amount_20d > 0 AND auction_amount > 0
                     THEN auction_amount / previous_amount_20d END AS auction_amount_to_daily20,
                previous_return_1d,
                {daily_features},
                LN(1.0 + GREATEST(previous_amount_20d, 0.0)) AS previous_log_amount_20d,
                daily_liquidity_rank,
                history_120d_available,
                history_240d_available,
                LN(1.0 + GREATEST(previous_total_mv, 0.0)) AS previous_log_total_market_value,
                LN(1.0 + GREATEST(previous_circ_mv, 0.0))
                    AS previous_log_circulating_market_value,
                previous_turnover_rate,
                previous_pe,
                previous_pb,
                CASE WHEN previous_float_share > 0
                     THEN cumulative_volume / previous_float_share END AS intraday_float_turnover,
                CASE WHEN previous_float_share > 0 AND valid_close
                     THEN LN(1.0 + close * previous_float_share) END
                    AS intraday_log_circulating_market_value,
                corporate_action_today,
                cash_dividend_per_10,
                bonus_share_per_10,
                transfer_share_per_10,
                CASE WHEN valid_close AND minute_index > 20 AND close_lag_20 > 0
                     THEN (close / close_lag_20 - 1.0) * previous_return_20d END
                    AS minute_daily_momentum_interaction_20,
                CASE WHEN valid_close AND minute_index > 60 AND close_lag_60 > 0
                     THEN (close / close_lag_60 - 1.0) * previous_return_60d END
                    AS minute_daily_momentum_interaction_60,
                CASE WHEN valid_close AND minute_index > 60 AND close_lag_5 > 0 AND close_lag_60 > 0
                     THEN (close / close_lag_5 - 1.0) - (close / close_lag_60 - 1.0) END
                    AS short_long_momentum_spread
            FROM completed
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
