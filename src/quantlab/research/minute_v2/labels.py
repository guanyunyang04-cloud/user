"""Future outcomes kept physically separate from causal minute features."""

from __future__ import annotations

from datetime import date, timedelta
from typing import Any

import duckdb
import pandas as pd

from .contracts import (
    EXIT_WINDOW_END,
    EXIT_WINDOW_START,
    KEY_COLUMNS,
    MAXIMUM_DAILY_LABEL_HORIZON,
    MinuteV2Config,
    MinuteV2Error,
)

MINUTE_LABEL_HORIZONS = (5, 15, 30, 60)
DAILY_LABEL_HORIZONS = (1, 3, 5, MAXIMUM_DAILY_LABEL_HORIZON)

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
        f"label_end_bar_time_{horizon}m",
        f"label_return_{horizon}m",
        f"label_mfe_{horizon}m",
        f"label_mae_{horizon}m",
        f"label_{horizon}m_observed",
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
    )
)

LABEL_COLUMNS = LEGACY_LABEL_COLUMNS + MINUTE_HORIZON_LABEL_COLUMNS + DAILY_HORIZON_LABEL_COLUMNS


def _literal(value: str) -> str:
    return "'" + date.fromisoformat(str(value)).isoformat() + "'"


def label_stock_day_query(*, start_date: str, end_date: str) -> str:
    """Return future daily support; adjustment factors make outcomes continuous."""

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


def _minute_ordered_expressions() -> str:
    values: list[str] = []
    for horizon in MINUTE_LABEL_HORIZONS:
        values.extend(
            [
                f"LEAD(bar_time, {horizon}) OVER w AS endpoint_bar_time_{horizon}m",
                f"LEAD(CAST(close AS DOUBLE), {horizon}) OVER w AS endpoint_close_{horizon}m",
                f"MAX(CAST(high AS DOUBLE)) OVER (PARTITION BY symbol, trade_date ORDER BY bar_time "
                f"ROWS BETWEEN 1 FOLLOWING AND {horizon} FOLLOWING) AS future_high_{horizon}m",
                f"MIN(CAST(low AS DOUBLE)) OVER (PARTITION BY symbol, trade_date ORDER BY bar_time "
                f"ROWS BETWEEN 1 FOLLOWING AND {horizon} FOLLOWING) AS future_low_{horizon}m",
                f"COUNT(*) OVER (PARTITION BY symbol, trade_date ORDER BY bar_time "
                f"ROWS BETWEEN 1 FOLLOWING AND {horizon} FOLLOWING) AS future_count_{horizon}m",
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
                f"THEN s.close * s.adjust_factor END) AS endpoint_price_{horizon}d",
                f"COUNT(*) FILTER (WHERE x.calendar_index <= e.signal_calendar_index + {horizon}) "
                f"AS future_day_count_{horizon}d",
                f"COUNT(*) FILTER (WHERE x.calendar_index <= e.signal_calendar_index + {horizon} "
                f"AND s.exclude_high) AS excluded_high_count_{horizon}d",
                f"COUNT(*) FILTER (WHERE x.calendar_index <= e.signal_calendar_index + {horizon} "
                f"AND s.exclude_low) AS excluded_low_count_{horizon}d",
                f"MAX(CASE WHEN x.calendar_index <= e.signal_calendar_index + {horizon} "
                f"THEN s.high * s.adjust_factor END) AS future_high_{horizon}d",
                f"MIN(CASE WHEN x.calendar_index <= e.signal_calendar_index + {horizon} "
                f"THEN s.low * s.adjust_factor END) AS future_low_{horizon}d",
                f"SUM(CASE WHEN x.calendar_index <= e.signal_calendar_index + {horizon} "
                f"THEN s.corporate_action_count ELSE 0 END) AS action_count_{horizon}d",
            ]
        )
    return ",\n                ".join(values)


def _minute_final_expressions() -> str:
    values: list[str] = []
    for horizon in MINUTE_LABEL_HORIZONS:
        observed = (
            f"entry_executable AND valid_close AND future_count_{horizon}m = {horizon} "
            f"AND endpoint_bar_time_{horizon}m IS NOT NULL AND endpoint_close_{horizon}m > 0"
        )
        values.extend(
            [
                f"endpoint_bar_time_{horizon}m AS label_end_bar_time_{horizon}m",
                f"CASE WHEN {observed} THEN endpoint_close_{horizon}m / entry_price - 1.0 "
                f"END AS label_return_{horizon}m",
                f"CASE WHEN {observed} AND valid_high THEN future_high_{horizon}m / entry_price - 1.0 "
                f"END AS label_mfe_{horizon}m",
                f"CASE WHEN {observed} AND valid_low THEN future_low_{horizon}m / entry_price - 1.0 "
                f"END AS label_mae_{horizon}m",
                f"({observed}) AS label_{horizon}m_observed",
            ]
        )
    return ",\n            ".join(values)


def _daily_final_expressions() -> str:
    values: list[str] = []
    for horizon in DAILY_LABEL_HORIZONS:
        observed = (
            f"entry_executable AND endpoint_date_{horizon}d IS NOT NULL "
            f"AND endpoint_price_{horizon}d > 0 AND entry_adjusted_price > 0"
        )
        values.extend(
            [
                f"endpoint_date_{horizon}d AS label_end_date_{horizon}d",
                f"CASE WHEN {observed} THEN endpoint_price_{horizon}d / entry_adjusted_price - 1.0 "
                f"END AS label_return_{horizon}d",
                f"CASE WHEN {observed} AND future_day_count_{horizon}d = {horizon} "
                f"AND excluded_high_count_{horizon}d = 0 "
                f"THEN future_high_{horizon}d / entry_adjusted_price - 1.0 "
                f"END AS label_mfe_{horizon}d",
                f"CASE WHEN {observed} AND future_day_count_{horizon}d = {horizon} "
                f"AND excluded_low_count_{horizon}d = 0 "
                f"THEN future_low_{horizon}d / entry_adjusted_price - 1.0 "
                f"END AS label_mae_{horizon}d",
                f"action_count_{horizon}d AS label_action_count_{horizon}d",
                f"({observed}) AS label_{horizon}d_observed",
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
    config: MinuteV2Config,
) -> str:
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
    return f"""
        WITH ordered_entries AS (
            SELECT
                symbol,
                trade_date,
                bar_time,
                LEAD(bar_time) OVER w AS entry_bar_time,
                LEAD(CAST(high AS DOUBLE)) OVER w AS entry_high,
                LEAD(CAST(low AS DOUBLE)) OVER w AS entry_low,
                LEAD(CAST(volume AS DOUBLE)) OVER w AS entry_volume,
                LEAD(CAST(amount AS DOUBLE)) OVER w AS entry_amount,
                {_minute_ordered_expressions()}
            FROM {target_bars_view}
            WINDOW w AS (PARTITION BY symbol, trade_date ORDER BY bar_time)
        ),
        entries AS (
            SELECT
                e.symbol,
                e.trade_date,
                e.bar_time,
                e.adjust_factor AS entry_adjust_factor,
                e.valid_high,
                e.valid_low,
                e.valid_close,
                c.next_trade_date AS planned_exit_date,
                c.calendar_index AS signal_calendar_index,
                o.* EXCLUDE(symbol, trade_date, bar_time),
                CASE WHEN o.entry_volume > 0 THEN o.entry_amount / o.entry_volume END AS entry_price,
                (
                    o.entry_bar_time IS NOT NULL
                    AND o.entry_volume > 0
                    AND o.entry_amount > 0
                    AND NOT (
                        e.valid_high AND e.valid_low
                        AND ABS(o.entry_high - o.entry_low) <= 1.0e-12
                    )
                ) AS entry_executable,
                CASE
                    WHEN o.entry_bar_time IS NULL THEN 'next_bar_missing'
                    WHEN o.entry_volume <= 0 OR o.entry_amount <= 0 THEN 'zero_flow'
                    WHEN e.valid_high AND e.valid_low
                         AND ABS(o.entry_high - o.entry_low) <= 1.0e-12 THEN 'one_price'
                    ELSE ''
                END AS entry_unfilled_reason
            FROM {event_view} e
            JOIN ordered_entries o USING(symbol, trade_date, bar_time)
            JOIN {calendar_view} c USING(trade_date)
        ),
        exit_windows AS (
            SELECT
                b.symbol,
                b.trade_date,
                c.calendar_index,
                COUNT(*) AS exit_bar_count,
                MIN(CAST(b.low AS DOUBLE)) AS exit_low,
                MAX(CAST(b.high AS DOUBLE)) AS exit_high,
                SUM(GREATEST(CAST(b.volume AS DOUBLE), 0.0)) AS exit_volume,
                SUM(GREATEST(CAST(b.amount AS DOUBLE), 0.0)) AS exit_amount,
                CASE WHEN SUM(GREATEST(CAST(b.volume AS DOUBLE), 0.0)) > 0
                     THEN SUM(GREATEST(CAST(b.amount AS DOUBLE), 0.0))
                          / SUM(GREATEST(CAST(b.volume AS DOUBLE), 0.0)) END AS exit_price,
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
                    AND exit_volume > 0
                    AND exit_amount > 0
                    AND exit_adjust_factor > 0
                    AND NOT is_suspended
                    AND NOT is_delisted
                    AND NOT exclude_high
                    AND NOT exclude_low
                    AND NOT (
                        ABS(exit_high - exit_low) <= 1.0e-12
                        AND previous_close > 0
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
                e.entry_price * e.entry_adjust_factor AS entry_adjusted_price,
                x.exit_price * x.exit_adjust_factor AS exit_adjusted_price,
                x.exit_low * x.exit_adjust_factor AS exit_adjusted_low,
                x.exit_high * x.exit_adjust_factor AS exit_adjusted_high,
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
            CASE WHEN entry_executable AND exit_adjusted_price > 0 AND entry_adjusted_price > 0
                 THEN exit_adjusted_price / entry_adjusted_price - 1.0 END AS label_gross_return,
            CASE WHEN entry_executable AND exit_adjusted_price > 0 AND entry_adjusted_price > 0
                 THEN (
                     exit_adjusted_price * (
                         1.0 - {sell_common} - CASE
                             WHEN actual_exit_date >= '2023-08-28' THEN {stamp_after}
                             ELSE {stamp_before}
                         END
                     )
                 ) / (entry_adjusted_price * (1.0 + {buy_cost})) - 1.0 END AS label_net_return,
            CASE WHEN entry_executable AND exit_adjusted_low > 0 AND entry_adjusted_price > 0
                 THEN exit_adjusted_low / entry_adjusted_price - 1.0 END AS label_adverse_return,
            CASE WHEN entry_executable AND exit_adjusted_high > 0 AND entry_adjusted_price > 0
                 THEN exit_adjusted_high / entry_adjusted_price - 1.0 END AS label_favorable_return,
            entry_executable AND actual_exit_date IS NOT NULL AS label_observed,
            {_minute_final_expressions()},
            {_daily_final_expressions()}
        FROM attached
        ORDER BY trade_date, bar_time, symbol
    """


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
    owned = connection is None
    con = duckdb.connect(":memory:") if owned else connection
    try:
        con.register("minute_events", events)
        con.register("minute_bars", target_bars)
        con.register("minute_bars_extended", extended_bars)
        con.register("label_stock_days", label_stock_days)
        con.register("calendar_dates", calendar_dates)
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
    "LABEL_COLUMNS",
    "MINUTE_LABEL_HORIZONS",
    "build_label_frame",
    "calendar_query",
    "label_query",
    "label_stock_day_query",
]
