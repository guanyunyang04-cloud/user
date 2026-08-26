"""Future outcomes kept physically separate from causal minute features."""

from __future__ import annotations

from datetime import date, timedelta
from typing import Any

import duckdb
import pandas as pd

from .contracts import EXIT_WINDOW_END, EXIT_WINDOW_START, KEY_COLUMNS, MinuteV2Config, MinuteV2Error

LABEL_COLUMNS = (
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


def label_stock_day_query(*, start_date: str, end_date: str) -> str:
    start = date.fromisoformat(str(start_date))
    warmup = (start - timedelta(days=10)).isoformat()

    def literal(value: str) -> str:
        return "'" + date.fromisoformat(value).isoformat() + "'"

    return f"""
        WITH factors AS (
            SELECT symbol, trade_date, ANY_VALUE(adjust_factor) AS adjust_factor
            FROM adjust_factor
            WHERE trade_date BETWEEN {literal(warmup)} AND {literal(end_date)}
            GROUP BY symbol, trade_date
        ),
        daily AS (
            SELECT
                d.symbol,
                d.trade_date,
                CAST(d.close AS DOUBLE) AS close,
                LAG(CAST(d.close AS DOUBLE)) OVER (
                    PARTITION BY d.symbol ORDER BY d.trade_date
                ) AS previous_close
            FROM daily_raw d
            WHERE d.trade_date BETWEEN {literal(warmup)} AND {literal(end_date)}
        )
        SELECT
            d.symbol,
            d.trade_date,
            CAST(f.adjust_factor AS DOUBLE) AS adjust_factor,
            d.previous_close,
            st.is_suspended,
            st.is_delisted,
            COALESCE(q.exclude_high, FALSE) AS exclude_high,
            COALESCE(q.exclude_low, FALSE) AS exclude_low
        FROM daily d
        JOIN factors f USING(symbol, trade_date)
        JOIN security_status st USING(symbol, trade_date)
        LEFT JOIN minute_feature_exclusions q USING(symbol, trade_date)
        WHERE d.trade_date BETWEEN {literal(start_date)} AND {literal(end_date)}
    """


def calendar_query(*, start_date: str, end_date: str) -> str:
    start = "'" + date.fromisoformat(str(start_date)).isoformat() + "'"
    end = "'" + date.fromisoformat(str(end_date)).isoformat() + "'"
    return f"""
        SELECT
            trade_date,
            ROW_NUMBER() OVER (ORDER BY trade_date) AS calendar_index,
            LEAD(trade_date) OVER (ORDER BY trade_date) AS next_trade_date
        FROM (
            SELECT DISTINCT trade_date
            FROM trading_calendar
            WHERE is_open AND trade_date BETWEEN {start} AND {end}
        ) dates
        ORDER BY trade_date
    """


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
    sell_common = (
        float(config.commission_bps)
        + float(config.transfer_fee_bps)
        + float(config.slippage_bps)
    ) / 10_000.0
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
                LEAD(CAST(amount AS DOUBLE)) OVER w AS entry_amount
            FROM {target_bars_view}
            WINDOW w AS (PARTITION BY symbol, trade_date ORDER BY bar_time)
        ),
        entries AS (
            SELECT
                e.symbol,
                e.trade_date,
                e.bar_time,
                e.adjust_factor AS entry_adjust_factor,
                c.next_trade_date AS planned_exit_date,
                c.calendar_index AS signal_calendar_index,
                o.entry_bar_time,
                o.entry_high,
                o.entry_low,
                o.entry_volume,
                o.entry_amount,
                CASE WHEN o.entry_volume > 0 THEN o.entry_amount / o.entry_volume END AS entry_price,
                (
                    o.entry_bar_time IS NOT NULL
                    AND o.entry_volume > 0
                    AND o.entry_amount > 0
                    AND o.entry_high > o.entry_low
                    AND e.valid_high
                    AND e.valid_low
                ) AS entry_executable,
                CASE
                    WHEN o.entry_bar_time IS NULL THEN 'next_bar_missing'
                    WHEN NOT e.valid_high OR NOT e.valid_low THEN 'quality_masked'
                    WHEN o.entry_volume <= 0 OR o.entry_amount <= 0 THEN 'zero_flow'
                    WHEN o.entry_high <= o.entry_low THEN 'one_price'
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
                          / SUM(GREATEST(CAST(b.volume AS DOUBLE), 0.0))
                END AS exit_price,
                s.adjust_factor AS exit_adjust_factor,
                s.previous_close,
                s.is_suspended,
                s.is_delisted,
                s.exclude_high,
                s.exclude_low
            FROM {extended_bars_view} b
            JOIN {stock_days_view} s USING(symbol, trade_date)
            JOIN {calendar_view} c USING(trade_date)
            WHERE b.bar_time BETWEEN '{EXIT_WINDOW_START}' AND '{EXIT_WINDOW_END}'
            GROUP BY
                b.symbol,
                b.trade_date,
                c.calendar_index,
                s.adjust_factor,
                s.previous_close,
                s.is_suspended,
                s.is_delisted,
                s.exclude_high,
                s.exclude_low
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
                        AND exit_price < previous_close
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
                x.exit_high * x.exit_adjust_factor AS exit_adjusted_high
            FROM entries e
            LEFT JOIN first_exit x USING(symbol, trade_date, bar_time)
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
                 ) / (entry_adjusted_price * (1.0 + {buy_cost})) - 1.0
            END AS label_net_return,
            CASE WHEN entry_executable AND exit_adjusted_low > 0 AND entry_adjusted_price > 0
                 THEN exit_adjusted_low / entry_adjusted_price - 1.0 END AS label_adverse_return,
            CASE WHEN entry_executable AND exit_adjusted_high > 0 AND entry_adjusted_price > 0
                 THEN exit_adjusted_high / entry_adjusted_price - 1.0 END AS label_favorable_return,
            entry_executable AND actual_exit_date IS NOT NULL AS label_observed
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
    "LABEL_COLUMNS",
    "build_label_frame",
    "calendar_query",
    "label_query",
    "label_stock_day_query",
]
