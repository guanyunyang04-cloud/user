"""Physically separate decision-step, same-session, and holding labels.

The ``label_*m`` family advances on the 234-bar decision grid so a late-day
signal has a well-defined next decision on the following session. The
``label_session_*m`` family advances on raw bars within the same session and
therefore captures the executable 11:30/15:00 tail separately.
"""

from __future__ import annotations

from datetime import date, timedelta
from typing import Any

import duckdb
import pandas as pd

from .contracts import (
    AFTERNOON_DECISION_END,
    AFTERNOON_DECISION_START,
    EXIT_WINDOW_END,
    EXIT_WINDOW_START,
    KEY_COLUMNS,
    MAXIMUM_DAILY_LABEL_HORIZON,
    MORNING_DECISION_END,
    MORNING_DECISION_START,
    MinuteV2Config,
    MinuteV2Error,
)

MINUTE_LABEL_HORIZONS = (5, 15, 30, 60)
DAILY_LABEL_HORIZONS = (1, 3, 5, MAXIMUM_DAILY_LABEL_HORIZON)


def _session_minute_ordinal_sql(alias: str = "b") -> str:
    """Map continuous-trading bars to a gap-free ordinal within one session."""

    value = f"CAST({alias}.bar_time AS VARCHAR)"
    minute_of_day = (
        f"CAST(SUBSTR({value}, 1, 2) AS INTEGER) * 60 + "
        f"CAST(SUBSTR({value}, 3, 2) AS INTEGER)"
    )
    return (
        "CASE "
        f"WHEN {value} BETWEEN '{MORNING_DECISION_START}' AND '113000000' "
        f"THEN ({minute_of_day} - (9 * 60 + 31)) "
        f"WHEN {value} BETWEEN '{AFTERNOON_DECISION_START}' AND '150000000' "
        f"THEN (120 + {minute_of_day} - (13 * 60 + 1)) "
        "END"
    )

LEGACY_LABEL_COLUMNS = (
    "symbol",
    "trade_date",
    "bar_time",
    "planned_exit_date",
    "entry_bar_time",
    "entry_price",
    "entry_volume",
    "entry_amount",
    "entry_executable",
    "entry_unfilled_reason",
    "actual_exit_date",
    "exit_price",
    "exit_volume",
    "exit_amount",
    "delayed_exit_days",
    "label_gross_return",
    "label_net_return",
    "label_adverse_return",
    "label_favorable_return",
    "label_observed",
)

MINUTE_HORIZON_LABEL_COLUMNS = tuple(
    name
    for horizon in MINUTE_LABEL_HORIZONS
    for name in (
        f"label_end_date_{horizon}m",
        f"label_end_bar_time_{horizon}m",
        f"label_return_{horizon}m",
        f"label_mfe_{horizon}m",
        f"label_mae_{horizon}m",
        f"label_{horizon}m_observed",
        f"label_{horizon}m_crossed_overnight",
        f"label_{horizon}m_crossed_lunch",
        f"label_{horizon}m_elapsed_calendar_days",
        f"label_{horizon}m_invalid_reason",
        f"label_{horizon}m_mfe_invalid_reason",
        f"label_{horizon}m_mae_invalid_reason",
    )
)

SESSION_HORIZON_LABEL_COLUMNS = tuple(
    name
    for horizon in MINUTE_LABEL_HORIZONS
    for name in (
        f"label_session_end_bar_time_{horizon}m",
        f"label_session_return_{horizon}m",
        f"label_session_mfe_{horizon}m",
        f"label_session_mae_{horizon}m",
        f"label_session_{horizon}m_observed",
        f"label_session_{horizon}m_invalid_reason",
    )
)

DAILY_HORIZON_LABEL_COLUMNS = tuple(
    name
    for horizon in DAILY_LABEL_HORIZONS
    for name in (
        f"label_end_date_{horizon}d",
        f"label_return_{horizon}d",
        f"label_mfe_{horizon}d",
        f"label_mae_{horizon}d",
        f"label_action_count_{horizon}d",
        f"label_{horizon}d_observed",
        f"label_{horizon}d_invalid_reason",
    )
)

LABEL_COLUMNS = (
    LEGACY_LABEL_COLUMNS
    + MINUTE_HORIZON_LABEL_COLUMNS
    + SESSION_HORIZON_LABEL_COLUMNS
    + DAILY_HORIZON_LABEL_COLUMNS
)


def _literal(value: str) -> str:
    return "'" + date.fromisoformat(str(value)).isoformat() + "'"


def label_stock_day_query(*, start_date: str, end_date: str) -> str:
    """Return daily outcome support with field-specific quality state."""

    warmup = (date.fromisoformat(str(start_date)) - timedelta(days=30)).isoformat()
    return f"""
        WITH factors AS (
            SELECT symbol, trade_date, ANY_VALUE(adjust_factor) AS adjust_factor
            FROM adjust_factor
            WHERE trade_date BETWEEN {_literal(warmup)} AND {_literal(end_date)}
            GROUP BY symbol, trade_date
        ),
        daily AS (
            SELECT
                d.symbol,
                d.trade_date,
                CAST(d.high AS DOUBLE) AS high,
                CAST(d.low AS DOUBLE) AS low,
                CAST(d.close AS DOUBLE) AS close,
                CAST(f.adjust_factor AS DOUBLE) AS adjust_factor,
                LAG(CAST(d.close AS DOUBLE)) OVER w AS previous_close,
                LAG(CAST(f.adjust_factor AS DOUBLE)) OVER w AS previous_adjust_factor
            FROM daily_raw d
            JOIN factors f USING(symbol, trade_date)
            WHERE d.trade_date BETWEEN {_literal(warmup)} AND {_literal(end_date)}
            WINDOW w AS (PARTITION BY d.symbol ORDER BY d.trade_date)
        ),
        actions AS (
            SELECT
                symbol,
                COALESCE(NULLIF(ex_date, ''), trade_date) AS action_date,
                COUNT(*) AS corporate_action_count
            FROM corporate_actions
            WHERE COALESCE(NULLIF(ex_date, ''), trade_date)
                  BETWEEN {_literal(start_date)} AND {_literal(end_date)}
            GROUP BY symbol, action_date
        )
        SELECT
            d.symbol,
            d.trade_date,
            d.high,
            d.low,
            d.close,
            d.adjust_factor,
            d.previous_close,
            d.previous_adjust_factor,
            st.is_suspended,
            st.is_delisted,
            COALESCE(q.exclude_high, FALSE) AS exclude_high,
            COALESCE(q.exclude_low, FALSE) AS exclude_low,
            COALESCE(q.exclude_close, FALSE) AS exclude_close,
            COALESCE(a.corporate_action_count, 0) AS corporate_action_count
        FROM daily d
        JOIN security_status st USING(symbol, trade_date)
        LEFT JOIN minute_feature_exclusions q USING(symbol, trade_date)
        LEFT JOIN actions a ON a.symbol = d.symbol AND a.action_date = d.trade_date
        WHERE d.trade_date BETWEEN {_literal(start_date)} AND {_literal(end_date)}
    """


def calendar_query(*, start_date: str, end_date: str) -> str:
    return f"""
        SELECT
            trade_date,
            ROW_NUMBER() OVER (ORDER BY trade_date) AS calendar_index,
            LEAD(trade_date) OVER (ORDER BY trade_date) AS next_trade_date
        FROM (
            SELECT DISTINCT trade_date
            FROM trading_calendar
            WHERE is_open AND trade_date BETWEEN {_literal(start_date)} AND {_literal(end_date)}
        ) dates
        ORDER BY trade_date
    """


def _decision_ordered_expressions() -> str:
    values: list[str] = []
    for horizon in MINUTE_LABEL_HORIZONS:
        frame = (
            "PARTITION BY symbol ORDER BY trade_date, bar_time "
            f"ROWS BETWEEN 1 FOLLOWING AND {horizon} FOLLOWING"
        )
        values.extend(
            [
                f"LEAD(trade_date, {horizon}) OVER w AS continuous_end_date_{horizon}m",
                f"LEAD(bar_time, {horizon}) OVER w AS continuous_end_time_{horizon}m",
                f"LEAD(decision_ordinal, {horizon}) OVER w AS continuous_end_ordinal_{horizon}m",
                f"LEAD(adjusted_close, {horizon}) OVER w AS continuous_end_close_{horizon}m",
                f"MAX(adjusted_high) OVER ({frame}) AS continuous_high_{horizon}m",
                f"MIN(adjusted_low) OVER ({frame}) AS continuous_low_{horizon}m",
                f"COUNT(*) OVER ({frame}) AS continuous_count_{horizon}m",
                f"COUNT(adjusted_high) OVER ({frame}) AS continuous_high_count_{horizon}m",
                f"COUNT(adjusted_low) OVER ({frame}) AS continuous_low_count_{horizon}m",
                f"COUNT(*) FILTER (WHERE high IS NULL "
                f"OR NOT isfinite(CAST(high AS DOUBLE)) "
                f"OR high <= 0 "
                f"OR NOT isfinite(CAST(adjust_factor AS DOUBLE)) "
                f"OR adjust_factor <= 0 "
                f"OR (isfinite(CAST(low AS DOUBLE)) AND high < low)) OVER ({frame}) "
                f"AS continuous_high_invalid_count_{horizon}m",
                f"COUNT(*) FILTER (WHERE low IS NULL "
                f"OR NOT isfinite(CAST(low AS DOUBLE)) "
                f"OR low <= 0 "
                f"OR NOT isfinite(CAST(adjust_factor AS DOUBLE)) "
                f"OR adjust_factor <= 0 "
                f"OR (isfinite(CAST(high AS DOUBLE)) AND high < low)) OVER ({frame}) "
                f"AS continuous_low_invalid_count_{horizon}m",
            ]
        )
    return ",\n                ".join(values)


def _session_ordered_expressions() -> str:
    values: list[str] = []
    for horizon in MINUTE_LABEL_HORIZONS:
        frame = (
            "PARTITION BY symbol, trade_date ORDER BY bar_time "
            f"ROWS BETWEEN 1 FOLLOWING AND {horizon} FOLLOWING"
        )
        values.extend(
            [
                f"LEAD(bar_time, {horizon}) OVER w AS session_end_time_{horizon}m",
                f"LEAD(session_minute_ordinal, {horizon}) OVER w AS session_end_ordinal_{horizon}m",
                f"LEAD(adjusted_close, {horizon}) OVER w AS session_end_close_{horizon}m",
                f"MAX(adjusted_high) OVER ({frame}) AS session_high_{horizon}m",
                f"MIN(adjusted_low) OVER ({frame}) AS session_low_{horizon}m",
                f"COUNT(*) OVER ({frame}) AS session_count_{horizon}m",
                f"COUNT(DISTINCT session_minute_ordinal) OVER ({frame}) "
                f"AS session_distinct_ordinal_count_{horizon}m",
                f"COUNT(adjusted_high) OVER ({frame}) AS session_high_count_{horizon}m",
                f"COUNT(adjusted_low) OVER ({frame}) AS session_low_count_{horizon}m",
                f"COUNT(*) FILTER (WHERE high IS NULL "
                f"OR NOT isfinite(CAST(high AS DOUBLE)) "
                f"OR high <= 0 "
                f"OR NOT isfinite(CAST(adjust_factor AS DOUBLE)) "
                f"OR adjust_factor <= 0 "
                f"OR (isfinite(CAST(low AS DOUBLE)) AND high < low)) OVER ({frame}) "
                f"AS session_high_invalid_count_{horizon}m",
                f"COUNT(*) FILTER (WHERE low IS NULL "
                f"OR NOT isfinite(CAST(low AS DOUBLE)) "
                f"OR low <= 0 "
                f"OR NOT isfinite(CAST(adjust_factor AS DOUBLE)) "
                f"OR adjust_factor <= 0 "
                f"OR (isfinite(CAST(high AS DOUBLE)) AND high < low)) OVER ({frame}) "
                f"AS session_low_invalid_count_{horizon}m",
            ]
        )
    return ",\n                ".join(values)


def _daily_aggregate_expressions() -> str:
    values: list[str] = []
    for horizon in DAILY_LABEL_HORIZONS:
        values.extend(
            [
                f"MAX(CASE WHEN x.calendar_index = e.signal_calendar_index + {horizon} "
                f"THEN x.trade_date END) AS endpoint_date_{horizon}d",
                f"MAX(CASE WHEN x.calendar_index = e.signal_calendar_index + {horizon} "
                "AND NOT s.is_suspended AND NOT s.is_delisted AND NOT s.exclude_close "
                "AND s.close > 0 AND s.adjust_factor > 0 "
                "AND isfinite(CAST(s.close AS DOUBLE)) "
                "AND isfinite(CAST(s.adjust_factor AS DOUBLE)) "
                f"THEN s.close * s.adjust_factor END) AS endpoint_price_{horizon}d",
                f"COUNT(*) FILTER (WHERE x.calendar_index <= e.signal_calendar_index + {horizon}) "
                f"AS future_day_count_{horizon}d",
                f"COUNT(*) FILTER (WHERE x.calendar_index <= e.signal_calendar_index + {horizon} "
                f"AND s.exclude_high) AS excluded_high_count_{horizon}d",
                f"COUNT(*) FILTER (WHERE x.calendar_index <= e.signal_calendar_index + {horizon} "
                f"AND s.exclude_low) AS excluded_low_count_{horizon}d",
                 f"COUNT(*) FILTER (WHERE x.calendar_index <= e.signal_calendar_index + {horizon} "
                 "AND (s.exclude_high OR s.high IS NULL "
                 "OR NOT isfinite(CAST(s.high AS DOUBLE)) "
                 "OR s.high <= 0 "
                 "OR s.adjust_factor IS NULL "
                 "OR NOT isfinite(CAST(s.adjust_factor AS DOUBLE)) "
                 "OR s.adjust_factor <= 0 "
                 "OR (isfinite(CAST(s.low AS DOUBLE)) AND s.high < s.low))) "
                 f"AS invalid_high_count_{horizon}d",
                 f"COUNT(*) FILTER (WHERE x.calendar_index <= e.signal_calendar_index + {horizon} "
                 "AND (s.exclude_low OR s.low IS NULL "
                 "OR NOT isfinite(CAST(s.low AS DOUBLE)) "
                 "OR s.low <= 0 "
                 "OR s.adjust_factor IS NULL "
                 "OR NOT isfinite(CAST(s.adjust_factor AS DOUBLE)) "
                 "OR s.adjust_factor <= 0 "
                 "OR (isfinite(CAST(s.high AS DOUBLE)) AND s.high < s.low))) "
                 f"AS invalid_low_count_{horizon}d",
                f"MAX(CASE WHEN x.calendar_index <= e.signal_calendar_index + {horizon} "
                 "AND NOT s.exclude_high "
                 "AND isfinite(CAST(s.high AS DOUBLE)) "
                 "AND s.high > 0 "
                 "AND isfinite(CAST(s.adjust_factor AS DOUBLE)) "
                 "AND s.adjust_factor > 0 "
                 "AND (s.low IS NULL OR NOT isfinite(CAST(s.low AS DOUBLE)) OR s.high >= s.low) "
                 f"THEN s.high * s.adjust_factor END) AS future_high_{horizon}d",
                f"MIN(CASE WHEN x.calendar_index <= e.signal_calendar_index + {horizon} "
                 "AND NOT s.exclude_low "
                 "AND isfinite(CAST(s.low AS DOUBLE)) "
                 "AND s.low > 0 "
                 "AND isfinite(CAST(s.adjust_factor AS DOUBLE)) "
                 "AND s.adjust_factor > 0 "
                 "AND (s.high IS NULL OR NOT isfinite(CAST(s.high AS DOUBLE)) OR s.high >= s.low) "
                 f"THEN s.low * s.adjust_factor END) AS future_low_{horizon}d",
                f"SUM(CASE WHEN x.calendar_index <= e.signal_calendar_index + {horizon} "
                f"THEN s.corporate_action_count ELSE 0 END) AS action_count_{horizon}d",
            ]
        )
    return ",\n                ".join(values)


def _continuous_final_expressions() -> str:
    values: list[str] = []
    for horizon in MINUTE_LABEL_HORIZONS:
        observed = (
            "COALESCE(("
            "COALESCE(continuous_entry_executable, FALSE) "
            "AND continuous_entry_ordinal = signal_decision_ordinal + 1 "
            f"AND continuous_end_ordinal_{horizon}m = signal_decision_ordinal + {horizon} "
            f"AND continuous_count_{horizon}m = {horizon} "
            f"AND isfinite(CAST(continuous_end_close_{horizon}m AS DOUBLE)) "
            f"AND continuous_end_close_{horizon}m > 0 "
            "AND isfinite(CAST(continuous_entry_adjusted_price AS DOUBLE)) "
            "AND continuous_entry_adjusted_price > 0 "
            f"AND isfinite(CAST(continuous_end_close_{horizon}m "
            "/ continuous_entry_adjusted_price - 1.0 AS DOUBLE))"
            "), FALSE)"
        )
        reason = (
            "COALESCE(CASE "
            "WHEN continuous_entry_ordinal IS NULL "
            "OR continuous_entry_ordinal <> signal_decision_ordinal + 1 "
            "THEN 'next_decision_bar_missing_or_untradable' "
            "WHEN NOT COALESCE(continuous_entry_executable, FALSE) "
            "THEN COALESCE(continuous_entry_unfilled_reason, 'continuous_entry_unexecutable') "
            f"WHEN continuous_end_ordinal_{horizon}m IS NULL "
            f"OR continuous_end_ordinal_{horizon}m <> signal_decision_ordinal + {horizon} "
            "THEN 'future_decision_bar_missing_or_untradable' "
            f"WHEN continuous_count_{horizon}m <> {horizon} THEN 'future_window_incomplete' "
            f"WHEN continuous_end_close_{horizon}m IS NULL "
            f"OR NOT isfinite(CAST(continuous_end_close_{horizon}m AS DOUBLE)) "
            "THEN 'endpoint_close_excluded' ELSE '' END, 'continuous_label_unavailable')"
        )
        values.extend(
            [
                f"continuous_end_date_{horizon}m AS label_end_date_{horizon}m",
                f"continuous_end_time_{horizon}m AS label_end_bar_time_{horizon}m",
                f"CASE WHEN {observed} THEN continuous_end_close_{horizon}m "
                f"/ continuous_entry_adjusted_price - 1.0 END AS label_return_{horizon}m",
                 f"CASE WHEN {observed} AND continuous_high_count_{horizon}m = {horizon} "
                 f"AND continuous_high_invalid_count_{horizon}m = 0 "
                 f"AND isfinite(CAST(continuous_high_{horizon}m "
                "/ continuous_entry_adjusted_price - 1.0 AS DOUBLE)) "
                f"THEN continuous_high_{horizon}m / continuous_entry_adjusted_price - 1.0 END "
                f"AS label_mfe_{horizon}m",
                 f"CASE WHEN {observed} AND continuous_low_count_{horizon}m = {horizon} "
                 f"AND continuous_low_invalid_count_{horizon}m = 0 "
                 f"AND isfinite(CAST(continuous_low_{horizon}m "
                "/ continuous_entry_adjusted_price - 1.0 AS DOUBLE)) "
                f"THEN continuous_low_{horizon}m / continuous_entry_adjusted_price - 1.0 END "
                f"AS label_mae_{horizon}m",
                f"({observed}) AS label_{horizon}m_observed",
                f"COALESCE(continuous_end_date_{horizon}m <> trade_date, FALSE) "
                f"AS label_{horizon}m_crossed_overnight",
                f"COALESCE(continuous_end_date_{horizon}m = trade_date "
                f"AND bar_time <= '{MORNING_DECISION_END}' "
                f"AND continuous_end_time_{horizon}m >= '{AFTERNOON_DECISION_START}', FALSE) "
                f"AS label_{horizon}m_crossed_lunch",
                f"CASE WHEN continuous_end_date_{horizon}m IS NOT NULL "
                f"THEN DATE_DIFF('day', CAST(trade_date AS DATE), "
                f"CAST(continuous_end_date_{horizon}m AS DATE)) END "
                f"AS label_{horizon}m_elapsed_calendar_days",
                f"{reason} AS label_{horizon}m_invalid_reason",
                f"CASE WHEN NOT ({observed}) THEN {reason} "
                 f"WHEN continuous_high_count_{horizon}m <> {horizon} "
                 f"OR continuous_high_invalid_count_{horizon}m > 0 "
                 f"THEN CASE WHEN continuous_high_invalid_count_{horizon}m > 0 "
                "THEN 'future_high_nonfinite' ELSE 'future_high_excluded' END "
                f"ELSE '' END AS label_{horizon}m_mfe_invalid_reason",
                f"CASE WHEN NOT ({observed}) THEN {reason} "
                 f"WHEN continuous_low_count_{horizon}m <> {horizon} "
                 f"OR continuous_low_invalid_count_{horizon}m > 0 "
                 f"THEN CASE WHEN continuous_low_invalid_count_{horizon}m > 0 "
                "THEN 'future_low_nonfinite' ELSE 'future_low_excluded' END "
                f"ELSE '' END AS label_{horizon}m_mae_invalid_reason",
            ]
        )
    return ",\n            ".join(values)


def _session_final_expressions() -> str:
    values: list[str] = []
    for horizon in MINUTE_LABEL_HORIZONS:
        observed = (
            "COALESCE(("
            "COALESCE(entry_executable, FALSE) "
            "AND session_minute_ordinal IS NOT NULL "
            f"AND session_end_ordinal_{horizon}m = session_minute_ordinal + {horizon} "
            f"AND session_count_{horizon}m = {horizon} "
            f"AND session_distinct_ordinal_count_{horizon}m = {horizon} "
            f"AND isfinite(CAST(session_end_close_{horizon}m AS DOUBLE)) "
            f"AND session_end_close_{horizon}m > 0 "
            "AND isfinite(CAST(entry_adjusted_price AS DOUBLE)) "
            "AND entry_adjusted_price > 0 "
            f"AND isfinite(CAST(session_end_close_{horizon}m "
            "/ entry_adjusted_price - 1.0 AS DOUBLE))"
            "), FALSE)"
        )
        reason = (
            "COALESCE(CASE WHEN NOT COALESCE(entry_executable, FALSE) "
            "THEN COALESCE(entry_unfilled_reason, 'entry_unexecutable') "
            f"WHEN session_minute_ordinal IS NULL OR session_end_ordinal_{horizon}m "
            f"     <> session_minute_ordinal + {horizon} THEN 'same_session_window_incomplete' "
            f"WHEN session_count_{horizon}m <> {horizon} THEN 'same_session_window_incomplete' "
            f"WHEN session_distinct_ordinal_count_{horizon}m <> {horizon} "
            "THEN 'same_session_window_incomplete' "
            f"WHEN session_end_close_{horizon}m IS NULL "
            f"OR NOT isfinite(CAST(session_end_close_{horizon}m AS DOUBLE)) "
            "THEN 'endpoint_close_excluded' "
            "ELSE '' END, 'session_label_unavailable')"
        )
        values.extend(
            [
                f"session_end_time_{horizon}m AS label_session_end_bar_time_{horizon}m",
                f"CASE WHEN {observed} THEN session_end_close_{horizon}m / entry_adjusted_price "
                f"- 1.0 END AS label_session_return_{horizon}m",
                 f"CASE WHEN {observed} AND session_high_count_{horizon}m = {horizon} "
                 f"AND session_high_invalid_count_{horizon}m = 0 "
                 f"AND isfinite(CAST(session_high_{horizon}m "
                "/ entry_adjusted_price - 1.0 AS DOUBLE)) "
                f"THEN session_high_{horizon}m / entry_adjusted_price - 1.0 END "
                f"AS label_session_mfe_{horizon}m",
                 f"CASE WHEN {observed} AND session_low_count_{horizon}m = {horizon} "
                 f"AND session_low_invalid_count_{horizon}m = 0 "
                 f"AND isfinite(CAST(session_low_{horizon}m "
                "/ entry_adjusted_price - 1.0 AS DOUBLE)) "
                f"THEN session_low_{horizon}m / entry_adjusted_price - 1.0 END "
                f"AS label_session_mae_{horizon}m",
                f"({observed}) AS label_session_{horizon}m_observed",
                f"{reason} AS label_session_{horizon}m_invalid_reason",
            ]
        )
    return ",\n            ".join(values)


def _daily_final_expressions() -> str:
    values: list[str] = []
    for horizon in DAILY_LABEL_HORIZONS:
        observed = (
            "COALESCE(("
            f"COALESCE(entry_executable, FALSE) AND endpoint_date_{horizon}d IS NOT NULL "
            f"AND isfinite(CAST(endpoint_price_{horizon}d AS DOUBLE)) "
            f"AND endpoint_price_{horizon}d > 0 "
            "AND isfinite(CAST(entry_adjusted_price AS DOUBLE)) "
            "AND entry_adjusted_price > 0 "
            f"AND isfinite(CAST(endpoint_price_{horizon}d "
            "/ entry_adjusted_price - 1.0 AS DOUBLE))"
            "), FALSE)"
        )
        values.extend(
            [
                f"endpoint_date_{horizon}d AS label_end_date_{horizon}d",
                f"CASE WHEN {observed} THEN endpoint_price_{horizon}d / entry_adjusted_price - 1.0 "
                f"END AS label_return_{horizon}d",
                f"CASE WHEN {observed} AND future_day_count_{horizon}d = {horizon} "
                f"AND invalid_high_count_{horizon}d = 0 "
                f"AND isfinite(CAST(future_high_{horizon}d "
                "/ entry_adjusted_price - 1.0 AS DOUBLE)) "
                f"THEN future_high_{horizon}d / entry_adjusted_price - 1.0 "
                f"END AS label_mfe_{horizon}d",
                f"CASE WHEN {observed} AND future_day_count_{horizon}d = {horizon} "
                f"AND invalid_low_count_{horizon}d = 0 "
                f"AND isfinite(CAST(future_low_{horizon}d "
                "/ entry_adjusted_price - 1.0 AS DOUBLE)) "
                f"THEN future_low_{horizon}d / entry_adjusted_price - 1.0 "
                f"END AS label_mae_{horizon}d",
                f"action_count_{horizon}d AS label_action_count_{horizon}d",
                f"({observed}) AS label_{horizon}d_observed",
                f"COALESCE(CASE WHEN NOT COALESCE(entry_executable, FALSE) "
                 f"THEN COALESCE(entry_unfilled_reason, 'entry_unexecutable') "
                 f"WHEN endpoint_date_{horizon}d IS NULL THEN 'future_market_day_missing' "
                 f"WHEN endpoint_price_{horizon}d IS NULL "
                 f"OR NOT isfinite(CAST(endpoint_price_{horizon}d AS DOUBLE)) "
                 f"THEN 'future_close_missing_or_untradable' "
                f"ELSE '' END, 'daily_label_unavailable') AS label_{horizon}d_invalid_reason",
            ]
        )
    return ",\n            ".join(values)


def label_query(
    *,
    event_view: str = "minute_events",
    target_bars_view: str = "minute_bars",
    extended_bars_view: str = "minute_bars_extended",
    stock_days_view: str = "label_stock_days",
    calendar_view: str = "calendar_dates",
    factor_view: str = "adjust_factor",
    quality_view: str = "minute_feature_exclusions",
    config: MinuteV2Config,
) -> str:
    del target_bars_view  # the event keys bound the target period; extended bars supply future paths
    config.validate()
    delayed = int(config.maximum_delayed_exit_days)
    buy_cost = (
        float(config.commission_bps)
        + float(config.transfer_fee_bps)
        + float(config.slippage_bps)
    ) / 10_000.0
    sell_common = buy_cost
    stamp_before = float(config.stamp_tax_bps_before_20230828) / 10_000.0
    stamp_after = float(config.stamp_tax_bps_after_20230828) / 10_000.0
    decision_filter = (
        f"(bar_time BETWEEN '{MORNING_DECISION_START}' AND '{MORNING_DECISION_END}') "
        f"OR (bar_time BETWEEN '{AFTERNOON_DECISION_START}' AND '{AFTERNOON_DECISION_END}')"
    )
    return f"""
        WITH factors AS (
            SELECT
                symbol,
                trade_date,
                ANY_VALUE(
                    CASE WHEN isfinite(CAST(adjust_factor AS DOUBLE))
                               AND CAST(adjust_factor AS DOUBLE) > 0
                         THEN CAST(adjust_factor AS DOUBLE) END
                ) AS adjust_factor
            FROM {factor_view}
            GROUP BY symbol, trade_date
        ),
        quality AS (
            SELECT
                symbol,
                trade_date,
                BOOL_OR(COALESCE(exclude_high, FALSE)) AS exclude_high,
                BOOL_OR(COALESCE(exclude_low, FALSE)) AS exclude_low,
                BOOL_OR(COALESCE(exclude_close, FALSE)) AS exclude_close
            FROM {quality_view}
            GROUP BY symbol, trade_date
        ),
        bar_context AS (
            SELECT
                CAST(b.symbol AS VARCHAR) AS symbol,
                CAST(b.trade_date AS VARCHAR) AS trade_date,
                CAST(b.bar_time AS VARCHAR) AS bar_time,
                CAST(b.high AS DOUBLE) AS high,
                CAST(b.low AS DOUBLE) AS low,
                CAST(b.close AS DOUBLE) AS close,
                GREATEST(CAST(b.volume AS DOUBLE), 0.0) AS volume,
                GREATEST(CAST(b.amount AS DOUBLE), 0.0) AS amount,
                {_session_minute_ordinal_sql('b')} AS session_minute_ordinal,
                f.adjust_factor,
                 CASE WHEN NOT COALESCE(q.exclude_high, FALSE)
                           AND isfinite(CAST(b.high AS DOUBLE))
                           AND CAST(b.high AS DOUBLE) > 0
                           AND isfinite(CAST(f.adjust_factor AS DOUBLE))
                           AND f.adjust_factor > 0
                           AND (b.low IS NULL
                                OR NOT isfinite(CAST(b.low AS DOUBLE))
                                OR CAST(b.high AS DOUBLE) >= CAST(b.low AS DOUBLE))
                      THEN CAST(b.high AS DOUBLE) * f.adjust_factor END AS adjusted_high,
                 CASE WHEN NOT COALESCE(q.exclude_low, FALSE)
                           AND isfinite(CAST(b.low AS DOUBLE))
                           AND CAST(b.low AS DOUBLE) > 0
                           AND isfinite(CAST(f.adjust_factor AS DOUBLE))
                           AND f.adjust_factor > 0
                           AND (b.high IS NULL
                                OR NOT isfinite(CAST(b.high AS DOUBLE))
                                OR CAST(b.high AS DOUBLE) >= CAST(b.low AS DOUBLE))
                      THEN CAST(b.low AS DOUBLE) * f.adjust_factor END AS adjusted_low,
                CASE WHEN NOT COALESCE(q.exclude_close, FALSE)
                          AND isfinite(CAST(b.close AS DOUBLE))
                          AND isfinite(CAST(f.adjust_factor AS DOUBLE))
                          AND f.adjust_factor > 0
                     THEN CAST(b.close AS DOUBLE) * f.adjust_factor END AS adjusted_close
            FROM {extended_bars_view} b
            JOIN factors f USING(symbol, trade_date)
            LEFT JOIN quality q USING(symbol, trade_date)
        ),
        market_decision_grid AS (
            SELECT
                trade_date,
                bar_time,
                ROW_NUMBER() OVER (ORDER BY trade_date, bar_time) AS decision_ordinal
            FROM (
                SELECT DISTINCT trade_date, bar_time
                FROM bar_context
                WHERE {decision_filter}
            )
        ),
        decision_bars AS (
            SELECT
                b.*, g.decision_ordinal
            FROM bar_context b
            JOIN market_decision_grid g USING(trade_date, bar_time)
        ),
        decision_ordered AS (
            SELECT
                *,
                LEAD(trade_date) OVER w AS continuous_entry_date,
                LEAD(bar_time) OVER w AS continuous_entry_time,
                LEAD(decision_ordinal) OVER w AS continuous_entry_ordinal,
                LEAD(high) OVER w AS continuous_entry_high,
                LEAD(low) OVER w AS continuous_entry_low,
                LEAD(volume) OVER w AS continuous_entry_volume,
                LEAD(amount) OVER w AS continuous_entry_amount,
                LEAD(adjust_factor) OVER w AS continuous_entry_factor,
                {_decision_ordered_expressions()}
            FROM decision_bars
            WINDOW w AS (PARTITION BY symbol ORDER BY trade_date, bar_time)
        ),
        session_ordered AS (
            SELECT
                *,
                LEAD(bar_time) OVER w AS raw_entry_time,
                LEAD(high) OVER w AS raw_entry_high,
                LEAD(low) OVER w AS raw_entry_low,
                LEAD(volume) OVER w AS raw_entry_volume,
                LEAD(amount) OVER w AS raw_entry_amount,
                LEAD(adjust_factor) OVER w AS raw_entry_factor,
                {_session_ordered_expressions()}
            FROM bar_context
            WINDOW w AS (PARTITION BY symbol, trade_date ORDER BY bar_time)
        ),
        entries_raw AS (
            SELECT
                e.symbol,
                CAST(e.trade_date AS VARCHAR) AS trade_date,
                CAST(e.bar_time AS VARCHAR) AS bar_time,
                c.next_trade_date AS planned_exit_date,
                c.calendar_index AS signal_calendar_index,
                d.decision_ordinal AS signal_decision_ordinal,
                r.raw_entry_time AS entry_bar_time,
                r.raw_entry_volume AS entry_volume,
                r.raw_entry_amount AS entry_amount,
                r.raw_entry_factor AS entry_adjust_factor,
                CASE WHEN r.raw_entry_volume > 0
                     THEN r.raw_entry_amount / r.raw_entry_volume END AS entry_price,
                d.continuous_entry_date,
                d.continuous_entry_time,
                d.continuous_entry_ordinal,
                d.continuous_entry_volume,
                d.continuous_entry_amount,
                d.continuous_entry_factor,
                CASE WHEN d.continuous_entry_volume > 0
                     THEN d.continuous_entry_amount / d.continuous_entry_volume END
                    AS continuous_entry_price,
                r.raw_entry_high,
                r.raw_entry_low,
                d.continuous_entry_high,
                d.continuous_entry_low,
                d.* EXCLUDE(
                    symbol, trade_date, bar_time, decision_ordinal,
                    continuous_entry_date, continuous_entry_time, continuous_entry_ordinal,
                    continuous_entry_high, continuous_entry_low, continuous_entry_volume,
                    continuous_entry_amount, continuous_entry_factor
                ),
                r.* EXCLUDE(
                    symbol, trade_date, bar_time, high, low, close, volume, amount,
                    adjust_factor, adjusted_high, adjusted_low, adjusted_close,
                    raw_entry_time, raw_entry_high, raw_entry_low, raw_entry_volume,
                    raw_entry_amount, raw_entry_factor, session_minute_ordinal
                )
            FROM {event_view} e
            JOIN decision_ordered d USING(symbol, trade_date, bar_time)
            JOIN session_ordered r USING(symbol, trade_date, bar_time)
            JOIN {calendar_view} c USING(trade_date)
        ),
        entries AS (
            SELECT
                *,
                COALESCE((
                     entry_bar_time IS NOT NULL
                    AND isfinite(CAST(entry_price AS DOUBLE))
                    AND entry_price > 0
                    AND isfinite(CAST(entry_adjust_factor AS DOUBLE))
                    AND entry_adjust_factor > 0
                    AND isfinite(CAST(entry_volume AS DOUBLE))
                    AND entry_volume > 0
                    AND isfinite(CAST(entry_amount AS DOUBLE))
                    AND entry_amount > 0
                     AND isfinite(CAST(raw_entry_high AS DOUBLE))
                     AND isfinite(CAST(raw_entry_low AS DOUBLE))
                     AND raw_entry_high > 0
                     AND raw_entry_low > 0
                     AND raw_entry_high >= raw_entry_low
                    AND NOT (ABS(raw_entry_high - raw_entry_low) <= 1.0e-12)
                ), FALSE) AS entry_executable,
                CASE
                    WHEN entry_bar_time IS NULL THEN 'next_raw_bar_missing'
                    WHEN entry_volume IS NULL OR NOT isfinite(CAST(entry_volume AS DOUBLE))
                         OR entry_amount IS NULL OR NOT isfinite(CAST(entry_amount AS DOUBLE))
                         OR entry_volume <= 0 OR entry_amount <= 0 THEN 'zero_flow'
                    WHEN entry_price IS NULL OR NOT isfinite(CAST(entry_price AS DOUBLE))
                         OR entry_price <= 0
                         OR entry_adjust_factor IS NULL
                         OR NOT isfinite(CAST(entry_adjust_factor AS DOUBLE))
                         OR entry_adjust_factor <= 0 THEN 'entry_price_invalid'
                     WHEN raw_entry_high IS NULL OR raw_entry_low IS NULL
                          OR NOT isfinite(CAST(raw_entry_high AS DOUBLE))
                          OR NOT isfinite(CAST(raw_entry_low AS DOUBLE))
                          OR raw_entry_high <= 0 OR raw_entry_low <= 0
                          OR raw_entry_high < raw_entry_low THEN 'entry_high_low_invalid'
                    WHEN ABS(raw_entry_high - raw_entry_low) <= 1.0e-12 THEN 'one_price'
                    ELSE ''
                END AS entry_unfilled_reason,
                COALESCE((
                    continuous_entry_time IS NOT NULL
                    AND isfinite(CAST(continuous_entry_price AS DOUBLE))
                    AND continuous_entry_price > 0
                    AND isfinite(CAST(continuous_entry_factor AS DOUBLE))
                    AND continuous_entry_factor > 0
                    AND isfinite(CAST(continuous_entry_volume AS DOUBLE))
                    AND continuous_entry_volume > 0
                    AND isfinite(CAST(continuous_entry_amount AS DOUBLE))
                    AND continuous_entry_amount > 0
                    AND isfinite(CAST(continuous_entry_high AS DOUBLE))
                    AND isfinite(CAST(continuous_entry_low AS DOUBLE))
                    AND continuous_entry_high > 0
                    AND continuous_entry_low > 0
                    AND continuous_entry_high >= continuous_entry_low
                    AND NOT (ABS(continuous_entry_high - continuous_entry_low) <= 1.0e-12)
                ), FALSE) AS continuous_entry_executable,
                CASE
                    WHEN continuous_entry_time IS NULL THEN 'next_decision_bar_missing'
                    WHEN continuous_entry_volume IS NULL
                         OR NOT isfinite(CAST(continuous_entry_volume AS DOUBLE))
                         OR continuous_entry_amount IS NULL
                         OR NOT isfinite(CAST(continuous_entry_amount AS DOUBLE))
                         OR continuous_entry_volume <= 0 OR continuous_entry_amount <= 0
                        THEN 'continuous_zero_flow'
                    WHEN continuous_entry_price IS NULL
                         OR NOT isfinite(CAST(continuous_entry_price AS DOUBLE))
                         OR continuous_entry_price <= 0
                         OR continuous_entry_factor IS NULL
                         OR NOT isfinite(CAST(continuous_entry_factor AS DOUBLE))
                         OR continuous_entry_factor <= 0 THEN 'continuous_entry_price_invalid'
                     WHEN continuous_entry_high IS NULL OR continuous_entry_low IS NULL
                          OR NOT isfinite(CAST(continuous_entry_high AS DOUBLE))
                          OR NOT isfinite(CAST(continuous_entry_low AS DOUBLE))
                          OR continuous_entry_high <= 0 OR continuous_entry_low <= 0
                          OR continuous_entry_high < continuous_entry_low
                        THEN 'continuous_entry_high_low_invalid'
                    WHEN ABS(continuous_entry_high - continuous_entry_low) <= 1.0e-12
                        THEN 'continuous_one_price'
                    ELSE ''
                END AS continuous_entry_unfilled_reason,
                entry_price * entry_adjust_factor AS entry_adjusted_price,
                continuous_entry_price * continuous_entry_factor
                    AS continuous_entry_adjusted_price
            FROM entries_raw
        ),
        exit_windows AS (
            SELECT
                b.symbol,
                b.trade_date,
                c.calendar_index,
                COUNT(*) AS exit_bar_count,
                 MIN(CASE WHEN isfinite(CAST(b.low AS DOUBLE))
                               AND CAST(b.low AS DOUBLE) > 0
                         THEN CAST(b.low AS DOUBLE) END) AS exit_low,
                 MAX(CASE WHEN isfinite(CAST(b.high AS DOUBLE))
                               AND CAST(b.high AS DOUBLE) > 0
                         THEN CAST(b.high AS DOUBLE) END) AS exit_high,
                 COUNT(*) FILTER (WHERE isfinite(CAST(b.high AS DOUBLE))
                                       AND isfinite(CAST(b.low AS DOUBLE))
                                       AND CAST(b.high AS DOUBLE) > 0
                                       AND CAST(b.low AS DOUBLE) > 0
                                       AND CAST(b.high AS DOUBLE) >= CAST(b.low AS DOUBLE))
                    AS exit_valid_range_count,
                COUNT(*) FILTER (WHERE isfinite(CAST(b.volume AS DOUBLE))
                                      AND CAST(b.volume AS DOUBLE) >= 0)
                    AS exit_valid_volume_count,
                COUNT(*) FILTER (WHERE isfinite(CAST(b.amount AS DOUBLE))
                                      AND CAST(b.amount AS DOUBLE) >= 0)
                    AS exit_valid_amount_count,
                SUM(CASE WHEN isfinite(CAST(b.volume AS DOUBLE))
                              AND CAST(b.volume AS DOUBLE) >= 0
                         THEN CAST(b.volume AS DOUBLE) ELSE 0.0 END) AS exit_volume,
                SUM(CASE WHEN isfinite(CAST(b.amount AS DOUBLE))
                              AND CAST(b.amount AS DOUBLE) >= 0
                         THEN CAST(b.amount AS DOUBLE) ELSE 0.0 END) AS exit_amount,
                CASE WHEN SUM(CASE WHEN isfinite(CAST(b.volume AS DOUBLE))
                                        AND CAST(b.volume AS DOUBLE) >= 0
                                   THEN CAST(b.volume AS DOUBLE) ELSE 0.0 END) > 0
                     THEN SUM(CASE WHEN isfinite(CAST(b.amount AS DOUBLE))
                                        AND CAST(b.amount AS DOUBLE) >= 0
                                   THEN CAST(b.amount AS DOUBLE) ELSE 0.0 END)
                          / SUM(CASE WHEN isfinite(CAST(b.volume AS DOUBLE))
                                        AND CAST(b.volume AS DOUBLE) >= 0
                                   THEN CAST(b.volume AS DOUBLE) ELSE 0.0 END) END AS exit_price,
                s.adjust_factor AS exit_adjust_factor,
                s.previous_close,
                s.previous_adjust_factor,
                s.is_suspended,
                s.is_delisted,
                s.exclude_high,
                s.exclude_low
            FROM {extended_bars_view} b
            JOIN {stock_days_view} s USING(symbol, trade_date)
            JOIN {calendar_view} c USING(trade_date)
            WHERE b.bar_time BETWEEN '{EXIT_WINDOW_START}' AND '{EXIT_WINDOW_END}'
            GROUP BY ALL
        ),
        executable_windows AS (
            SELECT
                *,
                (
                    exit_bar_count = 26
                    AND exit_valid_range_count = 26
                    AND exit_valid_volume_count = 26
                    AND exit_valid_amount_count = 26
                    AND exit_volume > 0
                    AND exit_amount > 0
                    AND isfinite(CAST(exit_price AS DOUBLE))
                    AND exit_adjust_factor > 0
                    AND isfinite(CAST(exit_adjust_factor AS DOUBLE))
                    AND NOT is_suspended
                    AND NOT is_delisted
                    AND NOT exclude_high
                    AND NOT exclude_low
                    AND NOT (
                        ABS(exit_high - exit_low) <= 1.0e-12
                        AND isfinite(CAST(previous_close AS DOUBLE))
                        AND previous_close > 0
                        AND isfinite(CAST(previous_adjust_factor AS DOUBLE))
                        AND previous_adjust_factor > 0
                        AND exit_price < previous_close * previous_adjust_factor / exit_adjust_factor
                    )
                ) AS exit_executable
            FROM exit_windows
        ),
        first_exit AS (
            SELECT
                e.symbol,
                e.trade_date,
                e.bar_time,
                x.trade_date AS actual_exit_date,
                x.calendar_index AS exit_calendar_index,
                x.exit_price,
                x.exit_low,
                x.exit_high,
                x.exit_volume,
                x.exit_amount,
                x.exit_adjust_factor
            FROM entries e
            JOIN executable_windows x
              ON x.symbol = e.symbol
             AND x.calendar_index BETWEEN e.signal_calendar_index + 1
                                      AND e.signal_calendar_index + {delayed + 1}
             AND x.exit_executable
            QUALIFY ROW_NUMBER() OVER (
                PARTITION BY e.symbol, e.trade_date, e.bar_time ORDER BY x.calendar_index
            ) = 1
        ),
        future_daily AS (
            SELECT
                e.symbol,
                e.trade_date,
                e.bar_time,
                {_daily_aggregate_expressions()}
            FROM entries e
            JOIN {calendar_view} x
              ON x.calendar_index BETWEEN e.signal_calendar_index + 1
                                      AND e.signal_calendar_index + {MAXIMUM_DAILY_LABEL_HORIZON}
            JOIN {stock_days_view} s
              ON s.symbol = e.symbol AND s.trade_date = x.trade_date
            GROUP BY e.symbol, e.trade_date, e.bar_time
        ),
        attached AS (
            SELECT
                e.*,
                x.actual_exit_date,
                x.exit_calendar_index,
                x.exit_price,
                x.exit_low,
                x.exit_high,
                x.exit_volume,
                x.exit_amount,
                x.exit_adjust_factor,
                CASE WHEN isfinite(CAST(x.exit_price AS DOUBLE))
                          AND isfinite(CAST(x.exit_adjust_factor AS DOUBLE))
                          AND x.exit_adjust_factor > 0
                     THEN x.exit_price * x.exit_adjust_factor END AS exit_adjusted_price,
                CASE WHEN isfinite(CAST(x.exit_low AS DOUBLE))
                          AND isfinite(CAST(x.exit_adjust_factor AS DOUBLE))
                          AND x.exit_adjust_factor > 0
                     THEN x.exit_low * x.exit_adjust_factor END AS exit_adjusted_low,
                CASE WHEN isfinite(CAST(x.exit_high AS DOUBLE))
                          AND isfinite(CAST(x.exit_adjust_factor AS DOUBLE))
                          AND x.exit_adjust_factor > 0
                     THEN x.exit_high * x.exit_adjust_factor END AS exit_adjusted_high,
                d.* EXCLUDE(symbol, trade_date, bar_time)
            FROM entries e
            LEFT JOIN first_exit x USING(symbol, trade_date, bar_time)
            LEFT JOIN future_daily d USING(symbol, trade_date, bar_time)
        )
        SELECT
            symbol,
            trade_date,
            bar_time,
            planned_exit_date,
            entry_bar_time,
            entry_price,
            entry_volume,
            entry_amount,
            entry_executable,
            entry_unfilled_reason,
            actual_exit_date,
            exit_price,
            exit_volume,
            exit_amount,
            CASE WHEN actual_exit_date IS NOT NULL
                 THEN exit_calendar_index - signal_calendar_index - 1 END AS delayed_exit_days,
            CASE WHEN entry_executable
                      AND isfinite(CAST(exit_adjusted_price AS DOUBLE))
                      AND exit_adjusted_price > 0
                      AND isfinite(CAST(entry_adjusted_price AS DOUBLE))
                      AND entry_adjusted_price > 0
                 THEN exit_adjusted_price / entry_adjusted_price - 1.0 END AS label_gross_return,
            CASE WHEN entry_executable
                      AND isfinite(CAST(exit_adjusted_price AS DOUBLE))
                      AND exit_adjusted_price > 0
                      AND isfinite(CAST(entry_adjusted_price AS DOUBLE))
                      AND entry_adjusted_price > 0
                 THEN (
                     exit_adjusted_price * (
                         1.0 - {sell_common} - CASE
                             WHEN actual_exit_date >= '2023-08-28' THEN {stamp_after}
                             ELSE {stamp_before}
                         END
                     )
                 ) / (entry_adjusted_price * (1.0 + {buy_cost})) - 1.0 END AS label_net_return,
            CASE WHEN entry_executable
                      AND isfinite(CAST(exit_adjusted_low AS DOUBLE))
                      AND exit_adjusted_low > 0
                      AND isfinite(CAST(entry_adjusted_price AS DOUBLE))
                      AND entry_adjusted_price > 0
                 THEN exit_adjusted_low / entry_adjusted_price - 1.0 END AS label_adverse_return,
            CASE WHEN entry_executable
                      AND isfinite(CAST(exit_adjusted_high AS DOUBLE))
                      AND exit_adjusted_high > 0
                      AND isfinite(CAST(entry_adjusted_price AS DOUBLE))
                      AND entry_adjusted_price > 0
                 THEN exit_adjusted_high / entry_adjusted_price - 1.0 END AS label_favorable_return,
            COALESCE((
                entry_executable
                AND actual_exit_date IS NOT NULL
                AND isfinite(CAST(entry_adjusted_price AS DOUBLE))
                AND entry_adjusted_price > 0
                AND isfinite(CAST(exit_adjusted_price AS DOUBLE))
                AND exit_adjusted_price > 0
                AND isfinite(CAST(exit_adjusted_price / entry_adjusted_price - 1.0 AS DOUBLE))
            ), FALSE)
                AS label_observed,
            {_continuous_final_expressions()},
            {_session_final_expressions()},
            {_daily_final_expressions()}
        FROM attached
        ORDER BY trade_date, bar_time, symbol
    """


def _fixture_factor_quality(
    events: pd.DataFrame,
    extended_bars: pd.DataFrame,
    label_stock_days: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    keys = extended_bars.loc[:, ["symbol", "trade_date"]].drop_duplicates().copy()
    context_keys = ["symbol", "trade_date"]

    # Keep presence separate from the numeric value.  A missing row is a
    # synthetic-fixture convenience, but a present row with a missing or bad
    # factor is an input error and must not be silently replaced with 1.0.
    if set(context_keys).issubset(label_stock_days.columns):
        daily_factor = label_stock_days.loc[
            :, context_keys + (["adjust_factor"] if "adjust_factor" in label_stock_days else [])
        ].drop_duplicates(context_keys)
        if "adjust_factor" not in daily_factor:
            daily_factor = daily_factor.assign(adjust_factor=pd.NA)
        daily_factor["_daily_context_present"] = True
    else:
        daily_factor = pd.DataFrame(
            columns=[*context_keys, "adjust_factor", "_daily_context_present"]
        )
    if set([*context_keys, "adjust_factor"]).issubset(events.columns):
        event_context = events.loc[:, [*context_keys, "adjust_factor"]].drop_duplicates(
            context_keys
        )
    elif set(context_keys).issubset(events.columns):
        event_context = events.loc[:, context_keys].drop_duplicates(context_keys)
        event_context = event_context.assign(adjust_factor=pd.NA)
    else:
        event_context = pd.DataFrame(columns=[*context_keys, "adjust_factor"])
    if not event_context.empty:
        event_context["_event_context_present"] = True
    else:
        event_context = event_context.assign(_event_context_present=pd.Series(dtype=bool))

    factors = keys.merge(daily_factor, how="left", on=context_keys)
    factors = factors.merge(
        event_context.rename(columns={"adjust_factor": "event_adjust_factor"}),
        how="left",
        on=context_keys,
    )
    daily_values = pd.to_numeric(factors["adjust_factor"], errors="coerce")
    event_values = pd.to_numeric(factors["event_adjust_factor"], errors="coerce")
    daily_present = factors["_daily_context_present"].astype("boolean").fillna(False)
    event_present = factors["_event_context_present"].astype("boolean").fillna(False)
    daily_valid = daily_values.notna() & daily_values.between(
        0.0, float("inf"), inclusive="neither"
    )
    event_valid = event_values.notna() & event_values.between(
        0.0, float("inf"), inclusive="neither"
    )
    if ((daily_present & ~daily_valid) | (event_present & ~event_valid)).any():
        raise MinuteV2Error("minute_v2_bar_day_context_adjust_factor_invalid")
    chosen = daily_values.where(daily_values.notna(), event_values)
    has_context = daily_present | event_present
    factors["adjust_factor"] = chosen.where(has_context, 1.0)
    factors = factors.loc[:, context_keys + ["adjust_factor"]]
    quality_columns = ["exclude_high", "exclude_low", "exclude_close"]
    daily_quality = (
        label_stock_days.loc[:, ["symbol", "trade_date", *quality_columns]]
        if set(quality_columns).issubset(label_stock_days.columns)
        else pd.DataFrame(columns=["symbol", "trade_date", *quality_columns])
    )
    quality = keys.merge(daily_quality, how="left", on=["symbol", "trade_date"])
    event_quality = events.loc[:, ["symbol", "trade_date", "valid_high", "valid_low", "valid_close"]].drop_duplicates(
        ["symbol", "trade_date"]
    )
    quality = quality.merge(event_quality, how="left", on=["symbol", "trade_date"])
    for name in ("high", "low", "close"):
        excluded = f"exclude_{name}"
        valid = f"valid_{name}"
        valid_values = quality[valid].astype("boolean").fillna(True)
        quality[excluded] = (
            quality[excluded]
            .astype("boolean")
            .fillna(~valid_values)
            .astype(bool)
        )
    return factors, quality.loc[:, ["symbol", "trade_date", *quality_columns]]


def build_label_frame(
    events: pd.DataFrame,
    target_bars: pd.DataFrame,
    extended_bars: pd.DataFrame,
    label_stock_days: pd.DataFrame,
    calendar_dates: pd.DataFrame,
    *,
    config: MinuteV2Config | None = None,
    connection: Any | None = None,
) -> pd.DataFrame:
    current = config or MinuteV2Config()
    factors, quality = _fixture_factor_quality(events, extended_bars, label_stock_days)
    owned = connection is None
    con = duckdb.connect(":memory:") if owned else connection
    try:
        con.register("minute_events", events)
        con.register("minute_bars", target_bars)
        con.register("minute_bars_extended", extended_bars)
        con.register("label_stock_days", label_stock_days)
        con.register("calendar_dates", calendar_dates)
        con.register("adjust_factor", factors)
        con.register("minute_feature_exclusions", quality)
        result = con.execute(label_query(config=current)).fetchdf()
    finally:
        if owned:
            con.close()
    missing = sorted(set(LABEL_COLUMNS).difference(result.columns))
    if missing:
        raise MinuteV2Error(f"minute_v2_label_columns_missing:{','.join(missing)}")
    if result.duplicated(list(KEY_COLUMNS)).any():
        raise MinuteV2Error("minute_v2_duplicate_label_keys")
    return result


__all__ = [
    "DAILY_LABEL_HORIZONS",
    "DAILY_HORIZON_LABEL_COLUMNS",
    "LABEL_COLUMNS",
    "MINUTE_HORIZON_LABEL_COLUMNS",
    "MINUTE_LABEL_HORIZONS",
    "SESSION_HORIZON_LABEL_COLUMNS",
    "build_label_frame",
    "calendar_query",
    "label_query",
    "label_stock_day_query",
]
