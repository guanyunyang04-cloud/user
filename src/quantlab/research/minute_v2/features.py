"""Causal multi-timeframe feature SQL shared by fixtures and production builds."""

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
    SIXTY_MINUTE_WINDOWS,
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


def _minute_lag_expressions() -> str:
    return ",\n                ".join(
        f"LAG(adjusted_close, {window}) OVER w_symbol AS adjusted_close_lag_{window}"
        for window in MINUTE_WINDOWS
    )


def _minute_rolling_expressions() -> str:
    values: list[str] = []
    for window in MINUTE_WINDOWS:
        frame = (
            "PARTITION BY symbol ORDER BY trade_date, bar_time "
            f"ROWS BETWEEN {window - 1} PRECEDING AND CURRENT ROW"
        )
        values.extend(
            [
                f"AVG(adjusted_close) OVER ({frame}) AS raw_m1_sma_{window}",
                f"COUNT(adjusted_close) OVER ({frame}) AS raw_m1_close_count_{window}",
                f"MAX(adjusted_high) OVER ({frame}) AS raw_m1_high_{window}",
                f"COUNT(adjusted_high) OVER ({frame}) AS raw_m1_high_count_{window}",
                f"MIN(adjusted_low) OVER ({frame}) AS raw_m1_low_{window}",
                f"COUNT(adjusted_low) OVER ({frame}) AS raw_m1_low_count_{window}",
                f"STDDEV_SAMP(log_return_1m) OVER ({frame}) AS raw_m1_volatility_{window}",
                f"COUNT(log_return_1m) OVER ({frame}) AS raw_m1_return_count_{window}",
                f"AVG(volume) OVER ({frame}) AS raw_m1_average_volume_{window}",
                f"COUNT(volume) OVER ({frame}) AS raw_m1_volume_count_{window}",
            ]
        )
    return ",\n                ".join(values)


def _minute_feature_expressions() -> str:
    values: list[str] = []
    for window in MINUTE_WINDOWS:
        values.extend(
            [
                f"CASE WHEN adjusted_close > 0 AND adjusted_close_lag_{window} > 0 "
                f"THEN adjusted_close / adjusted_close_lag_{window} - 1.0 END "
                f"AS return_{window}m",
                f"CASE WHEN raw_m1_close_count_{window} = {window} "
                f"AND raw_m1_sma_{window} > 0 THEN adjusted_close / raw_m1_sma_{window} - 1.0 "
                f"END AS moving_average_deviation_{window}m",
                f"CASE WHEN raw_m1_high_count_{window} = {window} "
                f"AND raw_m1_low_count_{window} = {window} AND raw_m1_low_{window} > 0 "
                f"THEN raw_m1_high_{window} / raw_m1_low_{window} - 1.0 "
                f"END AS rolling_range_{window}m",
                f"CASE WHEN raw_m1_return_count_{window} = {window} "
                f"THEN raw_m1_volatility_{window} END AS realized_volatility_{window}m",
                f"CASE WHEN raw_m1_volume_count_{window} = {window} "
                f"AND raw_m1_average_volume_{window} > 0 "
                f"THEN volume / raw_m1_average_volume_{window} - 1.0 "
                f"END AS volume_ratio_{window}m",
            ]
        )
    return ",\n                ".join(values)


def _sixty_lag_expressions() -> str:
    return ",\n                ".join(
        f"LAG(adjusted_close_60m, {window}) OVER w60 AS adjusted_close_60m_lag_{window}"
        for window in SIXTY_MINUTE_WINDOWS
    )


def _sixty_rolling_expressions() -> str:
    values: list[str] = []
    for window in SIXTY_MINUTE_WINDOWS:
        frame = (
            "PARTITION BY symbol ORDER BY trade_date, sixty_minute_bucket "
            f"ROWS BETWEEN {window - 1} PRECEDING AND CURRENT ROW"
        )
        values.extend(
            [
                f"AVG(adjusted_close_60m) OVER ({frame}) AS raw_m60_sma_{window}",
                f"COUNT(adjusted_close_60m) OVER ({frame}) AS raw_m60_close_count_{window}",
                f"MAX(adjusted_high_60m) OVER ({frame}) AS raw_m60_high_{window}",
                f"COUNT(adjusted_high_60m) OVER ({frame}) AS raw_m60_high_count_{window}",
                f"MIN(adjusted_low_60m) OVER ({frame}) AS raw_m60_low_{window}",
                f"COUNT(adjusted_low_60m) OVER ({frame}) AS raw_m60_low_count_{window}",
                f"STDDEV_SAMP(log_return_60m) OVER ({frame}) AS raw_m60_volatility_{window}",
                f"COUNT(log_return_60m) OVER ({frame}) AS raw_m60_return_count_{window}",
                f"AVG(amount_60m) OVER ({frame}) AS raw_m60_average_amount_{window}",
                f"COUNT(amount_60m) OVER ({frame}) AS raw_m60_amount_count_{window}",
            ]
        )
    return ",\n                ".join(values)


def _sixty_feature_expressions() -> str:
    values: list[str] = []
    for window in SIXTY_MINUTE_WINDOWS:
        values.extend(
            [
                f"CASE WHEN adjusted_close_60m > 0 AND adjusted_close_60m_lag_{window} > 0 "
                f"THEN adjusted_close_60m / adjusted_close_60m_lag_{window} - 1.0 END "
                f"AS m60_return_{window}bar",
                f"CASE WHEN raw_m60_close_count_{window} = {window} "
                f"AND raw_m60_sma_{window} > 0 "
                f"THEN adjusted_close_60m / raw_m60_sma_{window} - 1.0 END "
                f"AS m60_close_to_sma_{window}bar",
                f"CASE WHEN raw_m60_high_count_{window} = {window} "
                f"AND raw_m60_low_count_{window} = {window} AND raw_m60_low_{window} > 0 "
                f"THEN raw_m60_high_{window} / raw_m60_low_{window} - 1.0 END "
                f"AS m60_range_{window}bar",
                f"CASE WHEN raw_m60_return_count_{window} = {window} "
                f"THEN raw_m60_volatility_{window} END AS m60_volatility_{window}bar",
                f"CASE WHEN raw_m60_amount_count_{window} = {window} "
                f"AND raw_m60_average_amount_{window} > 0 "
                f"THEN amount_60m / raw_m60_average_amount_{window} - 1.0 END "
                f"AS m60_amount_ratio_{window}bar",
            ]
        )
    return ",\n                ".join(values)


def _daily_joined_expressions() -> str:
    return ",\n                ".join(
        f"CAST(s.{name} AS DOUBLE) AS {name}"
        for window in DAILY_WINDOWS
        for name in (
            f"previous_return_{window}d",
            f"previous_close_to_sma_{window}d",
            f"previous_volatility_{window}d",
            f"previous_amount_ratio_{window}d",
        )
    )


def _daily_projection() -> str:
    return ",\n                ".join(
        name
        for window in DAILY_WINDOWS
        for name in (
            f"previous_return_{window}d",
            f"previous_close_to_sma_{window}d",
            f"previous_volatility_{window}d",
            f"previous_amount_ratio_{window}d",
        )
    )


def _sixty_projection(alias: str = "s60") -> str:
    return ",\n                ".join(
        f"{alias}.{name}"
        for window in SIXTY_MINUTE_WINDOWS
        for name in (
            f"m60_return_{window}bar",
            f"m60_close_to_sma_{window}bar",
            f"m60_range_{window}bar",
            f"m60_volatility_{window}bar",
            f"m60_amount_ratio_{window}bar",
        )
    )


def sixty_state_query(
    *,
    bars_view: str = "minute_bars_history",
    stock_days_view: str = "stock_days",
    factor_view: str = "adjust_factor",
    quality_view: str = "minute_feature_exclusions",
    expected_session_bars: int = EXPECTED_SESSION_BARS,
) -> str:
    """Aggregate long 60-minute history once before target-minute feature construction."""

    expected = int(expected_session_bars)
    if expected <= 0 or expected % 60 != 0:
        raise MinuteV2Error("minute_v2_expected_session_bars_invalid")
    return f"""
        WITH target_symbols AS (
            SELECT DISTINCT symbol FROM {stock_days_view}
        ),
        factors AS (
            SELECT symbol, trade_date, ANY_VALUE(CAST(adjust_factor AS DOUBLE)) AS adjust_factor
            FROM {factor_view}
            GROUP BY symbol, trade_date
        ),
        quality AS (
            SELECT
                symbol,
                trade_date,
                BOOL_OR(COALESCE(exclude_open, FALSE)) AS exclude_open,
                BOOL_OR(COALESCE(exclude_high, FALSE)) AS exclude_high,
                BOOL_OR(COALESCE(exclude_low, FALSE)) AS exclude_low,
                BOOL_OR(COALESCE(exclude_close, FALSE)) AS exclude_close
            FROM {quality_view}
            GROUP BY symbol, trade_date
        ),
        history_joined AS (
            SELECT
                CAST(b.symbol AS VARCHAR) AS symbol,
                CAST(b.trade_date AS VARCHAR) AS trade_date,
                CAST(b.bar_time AS VARCHAR) AS bar_time,
                CAST(b.open AS DOUBLE) AS open,
                CAST(b.high AS DOUBLE) AS high,
                CAST(b.low AS DOUBLE) AS low,
                CAST(b.close AS DOUBLE) AS close,
                GREATEST(CAST(b.amount AS DOUBLE), 0.0) AS amount,
                f.adjust_factor,
                NOT COALESCE(q.exclude_open, FALSE) AS valid_open,
                NOT COALESCE(q.exclude_high, FALSE) AS valid_high,
                NOT COALESCE(q.exclude_low, FALSE) AS valid_low,
                NOT COALESCE(q.exclude_close, FALSE) AS valid_close,
                ROW_NUMBER() OVER w_day AS minute_index,
                COUNT(*) OVER (PARTITION BY b.symbol, b.trade_date) AS session_bar_count
            FROM {bars_view} b
            JOIN target_symbols t ON t.symbol = b.symbol
            JOIN factors f ON f.symbol = b.symbol AND f.trade_date = b.trade_date
            LEFT JOIN quality q ON q.symbol = b.symbol AND q.trade_date = b.trade_date
            WINDOW w_day AS (PARTITION BY b.symbol, b.trade_date ORDER BY b.bar_time)
        ),
        valid_sessions AS (
            SELECT
                *,
                CAST(FLOOR((minute_index - 1) / 60) + 1 AS TINYINT) AS sixty_minute_bucket,
                CASE WHEN valid_open THEN open * adjust_factor END AS adjusted_open,
                CASE WHEN valid_high THEN high * adjust_factor END AS adjusted_high,
                CASE WHEN valid_low THEN low * adjust_factor END AS adjusted_low,
                CASE WHEN valid_close THEN close * adjust_factor END AS adjusted_close
            FROM history_joined
            WHERE session_bar_count = {expected} AND adjust_factor > 0
        ),
        sixty_bars AS (
            SELECT
                symbol,
                trade_date,
                sixty_minute_bucket,
                COUNT(*) AS constituent_count,
                ARG_MIN(adjusted_open, bar_time) AS adjusted_open_60m,
                MAX(adjusted_high) AS adjusted_high_60m,
                MIN(adjusted_low) AS adjusted_low_60m,
                ARG_MAX(adjusted_close, bar_time) AS adjusted_close_60m,
                SUM(amount) AS amount_60m
            FROM valid_sessions
            GROUP BY symbol, trade_date, sixty_minute_bucket
            HAVING constituent_count = 60
        ),
        sixty_ordered AS (
            SELECT
                *,
                ROW_NUMBER() OVER w60 AS m60_history_bar_count,
                LAG(adjusted_close_60m) OVER w60 AS adjusted_close_60m_lag_1,
                {_sixty_lag_expressions()}
            FROM sixty_bars
            WINDOW w60 AS (PARTITION BY symbol ORDER BY trade_date, sixty_minute_bucket)
        ),
        sixty_returns AS (
            SELECT
                *,
                CASE WHEN adjusted_close_60m > 0 AND adjusted_close_60m_lag_1 > 0
                     THEN LN(adjusted_close_60m / adjusted_close_60m_lag_1) END
                    AS log_return_60m
            FROM sixty_ordered
        ),
        sixty_rolling AS (
            SELECT
                *,
                {_sixty_rolling_expressions()}
            FROM sixty_returns
        )
        SELECT
            symbol,
            trade_date,
            sixty_minute_bucket,
            m60_history_bar_count,
            {_sixty_feature_expressions()}
        FROM sixty_rolling
        ORDER BY symbol, trade_date, sixty_minute_bucket
    """


def feature_query(
    *,
    bars_view: str = "minute_bars_history",
    stock_days_view: str = "stock_days",
    factor_view: str = "adjust_factor",
    quality_view: str = "minute_feature_exclusions",
    calendar_view: str = "trading_calendar",
    sixty_state_view: str = "sixty_minute_states",
    expected_session_bars: int = EXPECTED_SESSION_BARS,
) -> str:
    """Return features based only on complete bars at or before each decision time."""

    expected = int(expected_session_bars)
    if expected <= 0 or expected % 60 != 0:
        raise MinuteV2Error("minute_v2_expected_session_bars_invalid")
    return f"""
        WITH target_symbols AS (
            SELECT DISTINCT symbol FROM {stock_days_view}
        ),
        factors AS (
            SELECT symbol, trade_date, ANY_VALUE(CAST(adjust_factor AS DOUBLE)) AS adjust_factor
            FROM {factor_view}
            GROUP BY symbol, trade_date
        ),
        quality AS (
            SELECT
                symbol,
                trade_date,
                BOOL_OR(COALESCE(exclude_open, FALSE)) AS exclude_open,
                BOOL_OR(COALESCE(exclude_high, FALSE)) AS exclude_high,
                BOOL_OR(COALESCE(exclude_low, FALSE)) AS exclude_low,
                BOOL_OR(COALESCE(exclude_close, FALSE)) AS exclude_close
            FROM {quality_view}
            GROUP BY symbol, trade_date
        ),
        market_calendar AS (
            SELECT
                trade_date,
                ROW_NUMBER() OVER (ORDER BY trade_date) AS market_day_index
            FROM (
                SELECT DISTINCT trade_date FROM {calendar_view} WHERE is_open
            )
        ),
        history_joined AS (
            SELECT
                CAST(b.symbol AS VARCHAR) AS symbol,
                CAST(b.trade_date AS VARCHAR) AS trade_date,
                CAST(b.bar_time AS VARCHAR) AS bar_time,
                CAST(b.open AS DOUBLE) AS open,
                CAST(b.high AS DOUBLE) AS high,
                CAST(b.low AS DOUBLE) AS low,
                CAST(b.close AS DOUBLE) AS close,
                GREATEST(CAST(b.volume AS DOUBLE), 0.0) AS volume,
                GREATEST(CAST(b.amount AS DOUBLE), 0.0) AS amount,
                f.adjust_factor,
                NOT COALESCE(q.exclude_open, FALSE) AS valid_open,
                NOT COALESCE(q.exclude_high, FALSE) AS valid_high,
                NOT COALESCE(q.exclude_low, FALSE) AS valid_low,
                NOT COALESCE(q.exclude_close, FALSE) AS valid_close,
                c.market_day_index,
                ROW_NUMBER() OVER w_day AS minute_index,
                COUNT(*) OVER (PARTITION BY b.symbol, b.trade_date) AS session_bar_count
            FROM {bars_view} b
            JOIN target_symbols t ON t.symbol = b.symbol
            JOIN factors f ON f.symbol = b.symbol AND f.trade_date = b.trade_date
            JOIN market_calendar c ON c.trade_date = b.trade_date
            LEFT JOIN quality q ON q.symbol = b.symbol AND q.trade_date = b.trade_date
            WINDOW w_day AS (PARTITION BY b.symbol, b.trade_date ORDER BY b.bar_time)
        ),
        valid_sessions AS (
            SELECT
                *,
                CAST((market_day_index - 1) * {expected} + minute_index AS BIGINT)
                    AS trading_minute_ordinal,
                CAST(FLOOR((minute_index - 1) / 60) + 1 AS TINYINT) AS sixty_minute_bucket,
                CASE WHEN valid_open THEN open * adjust_factor END AS adjusted_open,
                CASE WHEN valid_high THEN high * adjust_factor END AS adjusted_high,
                CASE WHEN valid_low THEN low * adjust_factor END AS adjusted_low,
                CASE WHEN valid_close THEN close * adjust_factor END AS adjusted_close
            FROM history_joined
            WHERE session_bar_count = {expected} AND adjust_factor > 0
        ),
        ordered AS (
            SELECT
                *,
                ROW_NUMBER() OVER w_symbol AS symbol_minute_sequence,
                LAG(trade_date) OVER w_symbol AS previous_bar_trade_date,
                LAG(bar_time) OVER w_symbol AS previous_bar_time,
                LAG(adjusted_close, 1) OVER w_symbol AS adjusted_close_lag_1,
                LAG(adjusted_close, 3) OVER w_symbol AS adjusted_close_lag_3,
                LAG(adjusted_close, 15) OVER w_symbol AS adjusted_close_lag_15,
                {_minute_lag_expressions()},
                FIRST_VALUE(adjusted_open IGNORE NULLS) OVER w_day_cumulative AS day_open_adjusted,
                MAX(adjusted_high) OVER w_day_cumulative AS cumulative_high_adjusted,
                MIN(adjusted_low) OVER w_day_cumulative AS cumulative_low_adjusted,
                SUM(volume) OVER w_day_cumulative AS cumulative_volume,
                SUM(amount) OVER w_day_cumulative AS cumulative_amount,
                FIRST_VALUE(adjusted_open IGNORE NULLS) OVER w60_partial AS partial_60m_open,
                MAX(adjusted_high) OVER w60_partial AS partial_60m_high,
                MIN(adjusted_low) OVER w60_partial AS partial_60m_low,
                SUM(amount) OVER w60_partial AS partial_60m_amount,
                AVG(volume) OVER (
                    PARTITION BY symbol, trade_date ORDER BY bar_time
                    ROWS BETWEEN 4 PRECEDING AND CURRENT ROW
                ) AS recent_volume_5,
                AVG(volume) OVER (
                    PARTITION BY symbol, trade_date ORDER BY bar_time
                    ROWS BETWEEN 24 PRECEDING AND 5 PRECEDING
                ) AS previous_volume_20,
                MAX(adjusted_high) OVER (
                    PARTITION BY symbol, trade_date ORDER BY bar_time
                    ROWS BETWEEN 20 PRECEDING AND 1 PRECEDING
                ) AS previous_high_20
            FROM valid_sessions
            WINDOW
                w_symbol AS (PARTITION BY symbol ORDER BY trade_date, bar_time),
                w_day_cumulative AS (
                    PARTITION BY symbol, trade_date ORDER BY bar_time
                    ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW
                ),
                w60_partial AS (
                    PARTITION BY symbol, trade_date, sixty_minute_bucket ORDER BY bar_time
                    ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW
                )
        ),
        returns AS (
            SELECT
                *,
                CASE WHEN adjusted_close > 0 AND adjusted_close_lag_1 > 0
                     THEN LN(adjusted_close / adjusted_close_lag_1) END AS log_return_1m,
                CASE WHEN cumulative_volume > 0
                     THEN cumulative_amount / cumulative_volume END AS cumulative_vwap_raw,
                CASE WHEN adjusted_high > adjusted_low
                     THEN (adjusted_close - adjusted_low) / (adjusted_high - adjusted_low) END
                    AS raw_bar_close_location,
                CASE WHEN cumulative_high_adjusted > cumulative_low_adjusted
                     THEN (adjusted_close - cumulative_low_adjusted)
                          / (cumulative_high_adjusted - cumulative_low_adjusted) END
                    AS raw_cumulative_close_location
            FROM ordered
        ),
        minute_rolling AS (
            SELECT
                *,
                STDDEV_SAMP(log_return_1m) OVER (
                    PARTITION BY symbol ORDER BY trade_date, bar_time
                    ROWS BETWEEN 14 PRECEDING AND CURRENT ROW
                ) AS raw_realized_volatility_15m,
                SQRT(AVG(CASE WHEN log_return_1m < 0
                              THEN log_return_1m * log_return_1m ELSE 0.0 END) OVER (
                    PARTITION BY symbol ORDER BY trade_date, bar_time
                    ROWS BETWEEN 14 PRECEDING AND CURRENT ROW
                )) AS raw_downside_volatility_15m,
                SUM(ABS(log_return_1m)) OVER (
                    PARTITION BY symbol, trade_date ORDER BY bar_time
                    ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW
                ) AS cumulative_absolute_return,
                {_minute_rolling_expressions()}
            FROM returns
        ),
        session_dates AS (
            SELECT
                symbol,
                trade_date,
                LAG(trade_date) OVER (PARTITION BY symbol ORDER BY trade_date)
                    AS previous_complete_trade_date
            FROM (
                SELECT DISTINCT symbol, trade_date FROM {sixty_state_view}
            )
        ),
        target_minutes AS (
            SELECT
                m.*,
                COALESCE(NULLIF(s.industry_name, ''), 'UNKNOWN') AS industry_name,
                CAST(s.previous_close AS DOUBLE) AS previous_close,
                CAST(s.previous_adjust_factor AS DOUBLE) AS previous_adjust_factor,
                CAST(s.auction_price AS DOUBLE) AS auction_price,
                CAST(s.auction_amount AS DOUBLE) AS auction_amount,
                CAST(s.previous_return_1d AS DOUBLE) AS previous_return_1d,
                {_daily_joined_expressions()},
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
                d.previous_complete_trade_date,
                CAST(FLOOR(m.minute_index / 60) AS TINYINT) AS last_completed_60m_bucket
            FROM minute_rolling m
            JOIN {stock_days_view} s USING(symbol, trade_date)
            JOIN session_dates d USING(symbol, trade_date)
        ),
        target_with_sixty AS (
            SELECT
                t.*,
                s60.m60_history_bar_count,
                {_sixty_projection()}
            FROM target_minutes t
            LEFT JOIN {sixty_state_view} s60
              ON s60.symbol = t.symbol
             AND s60.trade_date = CASE
                    WHEN t.last_completed_60m_bucket = 0
                    THEN t.previous_complete_trade_date ELSE t.trade_date END
             AND s60.sixty_minute_bucket = CASE
                    WHEN t.last_completed_60m_bucket = 0
                    THEN 4 ELSE t.last_completed_60m_bucket END
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
                trading_minute_ordinal,
                symbol_minute_sequence,
                CAST(minute_index AS DOUBLE) / {expected}.0 AS minute_fraction,
                SIN(2.0 * PI() * CAST(minute_index AS DOUBLE) / {expected}.0) AS time_sin,
                COS(2.0 * PI() * CAST(minute_index AS DOUBLE) / {expected}.0) AS time_cos,
                previous_bar_trade_date = trade_date AND previous_bar_time = '113000000'
                    AND bar_time = '130100000' AS crossed_lunch_from_previous_bar,
                previous_bar_trade_date IS NOT NULL AND previous_bar_trade_date <> trade_date
                    AS crossed_overnight_from_previous_bar,
                CASE WHEN previous_bar_trade_date IS NOT NULL
                     THEN DATE_DIFF('day', CAST(previous_bar_trade_date AS DATE),
                                           CAST(trade_date AS DATE)) END
                    AS calendar_days_from_previous_bar,
                CASE WHEN adjusted_close > 0 AND adjusted_close_lag_1 > 0
                     THEN adjusted_close / adjusted_close_lag_1 - 1.0 END AS return_1m,
                CASE WHEN adjusted_close > 0 AND adjusted_close_lag_3 > 0
                     THEN adjusted_close / adjusted_close_lag_3 - 1.0 END AS return_3m,
                CASE WHEN adjusted_close > 0 AND adjusted_close_lag_15 > 0
                     THEN adjusted_close / adjusted_close_lag_15 - 1.0 END AS return_15m,
                {_minute_feature_expressions()},
                {_sixty_projection(alias='target_with_sixty')},
                CASE WHEN adjusted_close > 0 AND previous_close > 0 AND previous_adjust_factor > 0
                     THEN adjusted_close / (previous_close * previous_adjust_factor) - 1.0 END
                    AS return_from_previous_close,
                CASE WHEN adjusted_close > 0 AND day_open_adjusted > 0
                     THEN adjusted_close / day_open_adjusted - 1.0 END AS return_from_open,
                CASE WHEN adjusted_high > 0 AND adjusted_low > 0
                     THEN adjusted_high / adjusted_low - 1.0 END AS bar_range,
                raw_bar_close_location AS bar_close_location,
                CASE WHEN cumulative_high_adjusted > 0 AND cumulative_low_adjusted > 0
                     THEN cumulative_high_adjusted / cumulative_low_adjusted - 1.0 END
                    AS cumulative_range,
                raw_cumulative_close_location AS cumulative_close_location,
                CASE WHEN adjusted_close > 0 AND cumulative_vwap_raw > 0
                     THEN close / cumulative_vwap_raw - 1.0 END AS vwap_deviation,
                raw_realized_volatility_15m AS realized_volatility_15m,
                raw_downside_volatility_15m AS downside_volatility_15m,
                CASE WHEN previous_volume_20 > 0
                     THEN recent_volume_5 / previous_volume_20 - 1.0 END
                    AS volume_acceleration_5_20,
                CASE WHEN previous_amount_20d > 0 AND minute_index > 0
                     THEN cumulative_amount / (previous_amount_20d * minute_index / {expected}.0) - 1.0
                END AS amount_curve_surprise,
                LN(1.0 + amount) AS log_bar_amount,
                CASE WHEN amount > 0 AND adjusted_close > 0 AND adjusted_close_lag_1 > 0
                     THEN ABS(adjusted_close / adjusted_close_lag_1 - 1.0)
                          / (amount / 1000000.0 + 1.0) END AS price_impact_1m,
                CASE WHEN adjusted_close > 0 AND cumulative_absolute_return > 0
                          AND day_open_adjusted > 0
                     THEN ABS(LN(adjusted_close / day_open_adjusted)) / cumulative_absolute_return
                END AS trend_efficiency,
                CASE WHEN adjusted_close > 0 AND previous_high_20 > 0
                     THEN adjusted_close / previous_high_20 - 1.0 END AS breakout_20m,
                CASE WHEN adjusted_close > 0 AND cumulative_high_adjusted > 0
                     THEN adjusted_close / cumulative_high_adjusted - 1.0 END
                    AS drawdown_from_day_high,
                CASE WHEN adjusted_close > 0 AND cumulative_low_adjusted > 0
                     THEN adjusted_close / cumulative_low_adjusted - 1.0 END
                    AS rebound_from_day_low,
                CASE WHEN auction_price > 0 AND previous_close > 0 AND previous_adjust_factor > 0
                     THEN auction_price * adjust_factor / (previous_close * previous_adjust_factor)
                          - 1.0 END AS auction_gap,
                CASE WHEN previous_amount_20d > 0 AND auction_amount > 0
                     THEN auction_amount / previous_amount_20d END AS auction_amount_to_daily20,
                previous_return_1d,
                {_daily_projection()},
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
                CASE WHEN previous_float_share > 0 AND close > 0
                     THEN LN(1.0 + close * previous_float_share) END
                    AS intraday_log_circulating_market_value,
                corporate_action_today,
                cash_dividend_per_10,
                bonus_share_per_10,
                transfer_share_per_10,
                CASE WHEN adjusted_close > 0 AND adjusted_close_lag_20 > 0
                     THEN (adjusted_close / adjusted_close_lag_20 - 1.0)
                          * previous_return_20d END AS minute_daily_momentum_interaction_20,
                CASE WHEN adjusted_close > 0 AND adjusted_close_lag_60 > 0
                     THEN (adjusted_close / adjusted_close_lag_60 - 1.0)
                          * previous_return_60d END AS minute_daily_momentum_interaction_60,
                CASE WHEN adjusted_close > 0 AND adjusted_close_lag_5 > 0
                          AND adjusted_close_lag_60 > 0
                     THEN (adjusted_close / adjusted_close_lag_5 - 1.0)
                          - (adjusted_close / adjusted_close_lag_60 - 1.0) END
                    AS short_long_momentum_spread,
                last_completed_60m_bucket,
                m60_history_bar_count,
                CASE WHEN adjusted_close > 0 AND partial_60m_open > 0
                     THEN adjusted_close / partial_60m_open - 1.0 END AS partial_60m_return,
                CASE WHEN partial_60m_high > 0 AND partial_60m_low > 0
                     THEN partial_60m_high / partial_60m_low - 1.0 END AS partial_60m_range,
                CASE WHEN partial_60m_high > partial_60m_low
                     THEN (adjusted_close - partial_60m_low)
                          / (partial_60m_high - partial_60m_low) END
                    AS partial_60m_close_location,
                LN(1.0 + partial_60m_amount) AS partial_60m_log_amount
            FROM target_with_sixty
            WHERE
                (bar_time BETWEEN '{MORNING_DECISION_START}' AND '{MORNING_DECISION_END}')
                OR (bar_time BETWEEN '{AFTERNOON_DECISION_START}' AND '{AFTERNOON_DECISION_END}')
        ),
        timeframe_enriched AS (
            SELECT
                *,
                SIGN(moving_average_deviation_20m) * SIGN(m60_close_to_sma_20bar)
                    AS m1_m60_trend_alignment,
                SIGN(m60_close_to_sma_20bar) * SIGN(previous_close_to_sma_20d)
                    AS m60_daily_trend_alignment,
                (
                    SIGN(moving_average_deviation_20m)
                    + SIGN(m60_close_to_sma_20bar)
                    + SIGN(previous_close_to_sma_20d)
                ) / 3.0 AS three_timeframe_trend_alignment,
                moving_average_deviation_20m - m60_close_to_sma_20bar
                    AS m1_m60_sma_spread_20,
                m60_close_to_sma_20bar - previous_close_to_sma_20d
                    AS m60_daily_sma_spread_20,
                LAG(vwap_deviation) OVER (
                    PARTITION BY symbol, trade_date ORDER BY bar_time
                ) AS previous_vwap_deviation
            FROM base
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
                LN(1.0 + SUM(amount)) AS market_log_total_amount,
                SUM(amount) AS market_total_amount
            FROM timeframe_enriched
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
                SUM(amount) AS industry_total_amount
            FROM timeframe_enriched
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
        cross_section AS (
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
                PERCENT_RANK() OVER (
                    PARTITION BY b.trade_date, b.bar_time
                    ORDER BY b.return_from_previous_close NULLS FIRST
                ) AS stock_return_rank,
                PERCENT_RANK() OVER (
                    PARTITION BY b.trade_date, b.bar_time ORDER BY b.return_5m NULLS FIRST
                ) AS stock_return_5m_rank,
                PERCENT_RANK() OVER (
                    PARTITION BY b.trade_date, b.bar_time
                    ORDER BY b.volume_acceleration_5_20 NULLS FIRST
                ) AS stock_volume_acceleration_rank,
                PERCENT_RANK() OVER (
                    PARTITION BY b.trade_date, b.bar_time ORDER BY b.amount NULLS FIRST
                ) AS stock_amount_rank,
                PERCENT_RANK() OVER (
                    PARTITION BY b.trade_date, b.bar_time, b.industry_name
                    ORDER BY b.return_from_previous_close NULLS FIRST
                ) AS industry_stock_return_rank,
                PERCENT_RANK() OVER (
                    PARTITION BY b.trade_date, b.bar_time, b.industry_name
                    ORDER BY b.amount NULLS FIRST
                ) AS industry_stock_amount_rank
            FROM timeframe_enriched b
            JOIN market_stats m USING(trade_date, bar_time)
            JOIN industry_ranked i USING(trade_date, bar_time, industry_name)
        )
        SELECT
            *,
            return_from_previous_close - market_return_mean AS market_residual_return,
            return_from_previous_close - industry_return_mean AS industry_residual_return,
            return_5m - industry_return_5m_mean AS industry_residual_return_5m
        FROM cross_section
        ORDER BY trade_date, bar_time, symbol
    """


def _fixture_context(
    bars: pd.DataFrame,
    stock_days: pd.DataFrame,
    bar_day_context: pd.DataFrame | None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    keys = bars.loc[:, ["symbol", "trade_date"]].drop_duplicates().copy()
    if bar_day_context is None:
        columns = [
            "symbol",
            "trade_date",
            "adjust_factor",
            "exclude_open",
            "exclude_high",
            "exclude_low",
            "exclude_close",
        ]
        context = keys.merge(stock_days.loc[:, columns], on=["symbol", "trade_date"], how="left")
    else:
        required = {
            "symbol",
            "trade_date",
            "adjust_factor",
            "exclude_open",
            "exclude_high",
            "exclude_low",
            "exclude_close",
        }
        missing = sorted(required.difference(bar_day_context.columns))
        if missing:
            raise MinuteV2Error(f"minute_v2_bar_day_context_missing:{','.join(missing)}")
        context = keys.merge(
            bar_day_context.loc[:, sorted(required)],
            on=["symbol", "trade_date"],
            how="left",
        )
    context["adjust_factor"] = pd.to_numeric(
        context["adjust_factor"], errors="coerce"
    ).fillna(1.0)
    for name in ("exclude_open", "exclude_high", "exclude_low", "exclude_close"):
        context[name] = context[name].astype("boolean").fillna(False).astype(bool)
    factors = context.loc[:, ["symbol", "trade_date", "adjust_factor"]].copy()
    quality = context.loc[
        :,
        [
            "symbol",
            "trade_date",
            "exclude_open",
            "exclude_high",
            "exclude_low",
            "exclude_close",
        ],
    ].copy()
    calendar = pd.DataFrame(
        {
            "trade_date": sorted(bars["trade_date"].astype(str).unique()),
            "is_open": True,
        }
    )
    return factors, quality, calendar


def build_feature_frame(
    bars: pd.DataFrame,
    stock_days: pd.DataFrame,
    *,
    bar_day_context: pd.DataFrame | None = None,
    expected_session_bars: int = EXPECTED_SESSION_BARS,
    connection: Any | None = None,
) -> pd.DataFrame:
    """Materialize multi-timeframe features for fixtures or bounded probes."""

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
    factors, quality, calendar = _fixture_context(bars, stock_days, bar_day_context)
    owned = connection is None
    con = duckdb.connect(":memory:") if owned else connection
    try:
        con.register("minute_bars_history", bars)
        con.register("stock_days", stock_days)
        con.register("adjust_factor", factors)
        con.register("minute_feature_exclusions", quality)
        con.register("trading_calendar", calendar)
        con.execute(
            "CREATE OR REPLACE TEMP TABLE sixty_minute_states AS "
            + sixty_state_query(expected_session_bars=expected_session_bars)
        )
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
    "sixty_state_query",
]
