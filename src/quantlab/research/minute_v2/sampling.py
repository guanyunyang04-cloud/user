"""Deterministic sampling of complete minute cross-sections for training."""

from __future__ import annotations

import math
from typing import Any

import duckdb
import pandas as pd

from .contracts import EXPECTED_DECISION_BARS, KEY_COLUMNS, MinuteV2Config, MinuteV2Error

EVENT_FLAG_COLUMNS = (
    "event_periodic",
    "event_extreme_return",
    "event_volume_shock",
    "event_industry_leadership",
    "event_vwap_cross",
    "event_random_negative",
)


def calibrate_groups_per_day(
    *,
    base_rows: int,
    complete_group_count: int,
    selected_trading_days: int,
    uncompressed_bytes: int,
    total_memory_bytes: int,
    memory_floor_bytes: int,
    memory_fraction: float,
    maximum_groups_per_day: int = EXPECTED_DECISION_BARS,
) -> dict[str, float | int | str]:
    """Choose a group count from measured row width and a machine memory budget."""

    if min(base_rows, complete_group_count, selected_trading_days, uncompressed_bytes) <= 0:
        raise MinuteV2Error("minute_v2_sampling_calibration_input_invalid")
    usable_memory = max(int(total_memory_bytes) - int(memory_floor_bytes), 1)
    target_bytes = max(1, int(usable_memory * float(memory_fraction)))
    bytes_per_row = float(uncompressed_bytes) / int(base_rows)
    rows_per_group = float(base_rows) / int(complete_group_count)
    bytes_per_group = bytes_per_row * rows_per_group
    raw_groups = math.floor(target_bytes / (bytes_per_group * int(selected_trading_days)))
    groups_per_day = max(1, min(int(maximum_groups_per_day), int(raw_groups)))
    expected_rows = round(rows_per_group * groups_per_day * int(selected_trading_days))
    return {
        "policy": "measured complete-cross-section groups under a fixed fraction of usable RAM",
        "base_rows": int(base_rows),
        "complete_group_count": int(complete_group_count),
        "selected_trading_days": int(selected_trading_days),
        "uncompressed_bytes": int(uncompressed_bytes),
        "bytes_per_row": bytes_per_row,
        "mean_rows_per_group": rows_per_group,
        "estimated_uncompressed_bytes_per_group": bytes_per_group,
        "total_memory_bytes": int(total_memory_bytes),
        "memory_floor_bytes": int(memory_floor_bytes),
        "memory_fraction": float(memory_fraction),
        "target_sample_bytes": target_bytes,
        "groups_per_day": groups_per_day,
        "expected_sample_rows": int(expected_rows),
    }


def event_query(
    *,
    feature_view: str = "minute_features",
    config: MinuteV2Config,
    groups_per_day: int | None = None,
) -> str:
    """Keep whole market snapshots; event ideas remain features instead of row gates."""

    config.validate()
    periodic = int(config.periodic_sample_every)
    random_percent = int(config.random_negative_percent)
    group_limit = EXPECTED_DECISION_BARS if groups_per_day is None else int(groups_per_day)
    if not 1 <= group_limit <= EXPECTED_DECISION_BARS:
        raise MinuteV2Error("minute_v2_groups_per_day_invalid")
    if group_limit == 1:
        session_expression = "0"
        morning_groups = 1
        afternoon_groups = 1
    else:
        session_expression = "CASE WHEN bar_time < '130000000' THEN 0 ELSE 1 END"
        morning_groups = round(group_limit * 119 / EXPECTED_DECISION_BARS)
        morning_groups = max(1, min(119, morning_groups))
        morning_groups = max(morning_groups, group_limit - 115)
        afternoon_groups = group_limit - morning_groups
    return f"""
        WITH flags AS (
            SELECT
                *,
                minute_index % {periodic} = 0 AS event_periodic,
                (stock_return_5m_rank >= 0.97 OR stock_return_5m_rank <= 0.03)
                    AS event_extreme_return,
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
                (HASH(symbol, trade_date, bar_time) % 100) < {random_percent}
                    AS event_random_negative
            FROM {feature_view}
        ),
        group_sessions AS (
            SELECT
                trade_date,
                bar_time,
                {session_expression} AS sampling_session
            FROM (SELECT DISTINCT trade_date, bar_time FROM flags)
        ),
        stratified_groups AS (
            SELECT
                trade_date,
                bar_time,
                sampling_session,
                NTILE(CASE WHEN sampling_session = 0 THEN {morning_groups}
                           ELSE {afternoon_groups} END) OVER (
                    PARTITION BY trade_date, sampling_session ORDER BY bar_time
                )
                    AS time_stratum
            FROM group_sessions
        ),
        ranked_groups AS (
            SELECT
                *,
                ROW_NUMBER() OVER (
                    PARTITION BY trade_date, sampling_session, time_stratum
                    ORDER BY HASH(CAST(trade_date AS VARCHAR), CAST(bar_time AS VARCHAR)), bar_time
                ) AS stratum_rank
            FROM stratified_groups
        ),
        selected_groups AS (
            SELECT
                trade_date,
                bar_time,
                time_stratum + CASE WHEN sampling_session = 0 THEN 0 ELSE {morning_groups} END
                    AS sampled_group_rank
            FROM ranked_groups
            WHERE stratum_rank = 1
        )
        SELECT
            f.*,
            CAST(event_periodic AS INTEGER)
                + 2 * CAST(event_extreme_return AS INTEGER)
                + 4 * CAST(event_volume_shock AS INTEGER)
                + 8 * CAST(event_industry_leadership AS INTEGER)
                + 16 * CAST(event_vwap_cross AS INTEGER)
                + 32 * CAST(event_random_negative AS INTEGER) AS event_mask,
            g.sampled_group_rank
        FROM flags f
        JOIN selected_groups g USING(trade_date, bar_time)
        ORDER BY trade_date, bar_time, symbol
    """


def build_event_frame(
    features: pd.DataFrame,
    *,
    config: MinuteV2Config | None = None,
    groups_per_day: int | None = None,
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
        result = con.execute(
            event_query(config=current, groups_per_day=groups_per_day)
        ).fetchdf()
    finally:
        if owned:
            con.close()
    if result.duplicated(list(KEY_COLUMNS)).any():
        raise MinuteV2Error("minute_v2_duplicate_event_keys")
    source_sizes = features.groupby(["trade_date", "bar_time"], sort=False).size()
    result_sizes = result.groupby(["trade_date", "bar_time"], sort=False).size()
    expected_sizes = source_sizes.reindex(result_sizes.index)
    if not result_sizes.equals(expected_sizes):
        raise MinuteV2Error("minute_v2_incomplete_cross_section_sample")
    return result


__all__ = [
    "EVENT_FLAG_COLUMNS",
    "build_event_frame",
    "calibrate_groups_per_day",
    "event_query",
]
