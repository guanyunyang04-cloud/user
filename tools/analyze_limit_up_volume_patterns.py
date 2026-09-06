"""Research two main-board limit-up volume patterns from active QDP data.

The tool is deliberately event based.  It reads the active point-in-time QDP
daily facts, reconstructs ordinary/ST upper limits, identifies two causal
patterns, and writes compact event-level and grouped statistics.  It never
mutates QDP datasets.
"""

from __future__ import annotations

import argparse
import gc
import json
import math
import time
from pathlib import Path
from typing import Any

import duckdb
import numpy as np
import pandas as pd
import psutil

GIB = 1024**3
START_DATE = "2020-01-01"
END_DATE = "2025-12-31"
LOAD_START_DATE = "2019-12-01"
LOAD_END_DATE = "2026-01-31"
HORIZONS = (1, 2, 3, 5)
PRICE_TOLERANCE = 0.0051
IPO_NO_LIMIT_OBSERVATIONS = 5


def _json_default(value: Any) -> Any:
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return None if not np.isfinite(value) else float(value)
    if isinstance(value, (pd.Timestamp,)):
        return value.isoformat()
    raise TypeError(type(value).__name__)


def _dump(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, default=_json_default) + "\n",
        encoding="utf-8",
    )


def _csv(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False, encoding="utf-8-sig")


def _scan(paths: list[Path]) -> str:
    return "read_parquet(?, union_by_name=true)"


def _active_paths(workspace: Path, domain: str) -> list[Path]:
    from quantlab.data.qdp_v2.active import resolve_active_domain

    active = resolve_active_domain(domain, workspace_root=workspace)
    return [(active.root / shard.path).resolve() for shard in active.manifest.shards]


def _check_memory(floor_gib: float) -> float:
    available = psutil.virtual_memory().available / GIB
    if available < floor_gib:
        raise RuntimeError(f"memory_floor_breached:{available:.3f}<{floor_gib:.3f}")
    return float(available)


def _configure_connection(workspace: Path, memory_limit: str, threads: int) -> duckdb.DuckDBPyConnection:
    connection = duckdb.connect()
    connection.execute(f"SET threads={max(1, int(threads))}")
    connection.execute(f"SET memory_limit='{memory_limit}'")
    temp_dir = workspace / "tmp" / "limit_up_volume_patterns_duckdb"
    temp_dir.mkdir(parents=True, exist_ok=True)
    connection.execute("SET temp_directory=?", [str(temp_dir)])
    connection.execute("SET preserve_insertion_order=false")
    return connection


def _build_base(connection: duckdb.DuckDBPyConnection, paths: dict[str, list[Path]]) -> None:
    daily_paths = [str(path) for path in paths["market_daily_raw"]]
    status_paths = [str(path) for path in paths["security_status"]]
    universe_paths = [str(path) for path in paths["universe_snapshot"]]
    industry_paths = [str(path) for path in paths["industry_concept"]]
    calendar_paths = [str(path) for path in paths["trading_calendar"]]

    connection.execute(
        f"""
        CREATE OR REPLACE TEMP TABLE calendar AS
        WITH selected AS (
            SELECT DISTINCT TRY_CAST(trade_date AS DATE) AS trade_date
            FROM {_scan(calendar_paths)}
            WHERE COALESCE(TRY_CAST(is_open AS BOOLEAN), FALSE)
              AND TRY_CAST(trade_date AS DATE) BETWEEN CAST(? AS DATE) AND CAST(? AS DATE)
        )
        SELECT trade_date,
               CAST(ROW_NUMBER() OVER (ORDER BY trade_date) - 1 AS BIGINT) AS trade_idx,
               LAG(trade_date) OVER (ORDER BY trade_date) AS prev_trade_date,
               LEAD(trade_date) OVER (ORDER BY trade_date) AS next_trade_date
        FROM selected
        ORDER BY trade_date
        """,
        [calendar_paths, LOAD_START_DATE, LOAD_END_DATE],
    )

    connection.execute(
        f"""
        CREATE OR REPLACE TEMP TABLE daily_raw AS
        SELECT CAST(symbol AS VARCHAR) AS symbol,
               TRY_CAST(trade_date AS DATE) AS trade_date,
               TRY_CAST(open AS DOUBLE) AS open,
               TRY_CAST(high AS DOUBLE) AS high,
               TRY_CAST(low AS DOUBLE) AS low,
               TRY_CAST(close AS DOUBLE) AS close,
               TRY_CAST(volume AS DOUBLE) AS volume,
               TRY_CAST(amount AS DOUBLE) AS amount
        FROM {_scan(daily_paths)}
        WHERE TRY_CAST(trade_date AS DATE) BETWEEN CAST(? AS DATE) AND CAST(? AS DATE)
          AND TRY_CAST(volume AS DOUBLE) > 0
          AND TRY_CAST(close AS DOUBLE) > 0
        """,
        [daily_paths, LOAD_START_DATE, LOAD_END_DATE],
    )
    connection.execute(
        f"""
        CREATE OR REPLACE TEMP TABLE status AS
        SELECT CAST(symbol AS VARCHAR) AS symbol,
               TRY_CAST(trade_date AS DATE) AS trade_date,
               BOOL_OR(COALESCE(TRY_CAST(is_st AS BOOLEAN), FALSE)) AS is_st
        FROM {_scan(status_paths)}
        WHERE TRY_CAST(trade_date AS DATE) BETWEEN CAST(? AS DATE) AND CAST(? AS DATE)
        GROUP BY 1, 2
        """,
        [status_paths, LOAD_START_DATE, LOAD_END_DATE],
    )
    connection.execute(
        f"""
        CREATE OR REPLACE TEMP TABLE universe AS
        SELECT CAST(symbol AS VARCHAR) AS symbol,
               TRY_CAST(trade_date AS DATE) AS trade_date,
               ANY_VALUE(NULLIF(TRIM(CAST(board AS VARCHAR)), '')) AS board,
               ANY_VALUE(NULLIF(TRIM(CAST(exchange AS VARCHAR)), '')) AS exchange,
               ANY_VALUE(TRY_CAST(list_date AS DATE)) AS list_date,
               ANY_VALUE(NULLIF(TRIM(CAST(name AS VARCHAR)), '')) AS name
        FROM {_scan(universe_paths)}
        WHERE TRY_CAST(trade_date AS DATE) BETWEEN CAST(? AS DATE) AND CAST(? AS DATE)
        GROUP BY 1, 2
        """,
        [universe_paths, LOAD_START_DATE, LOAD_END_DATE],
    )
    connection.execute(
        f"""
        CREATE OR REPLACE TEMP TABLE industry_raw AS
        SELECT CAST(symbol AS VARCHAR) AS symbol,
               TRY_CAST(trade_date AS DATE) AS trade_date,
               NULLIF(TRIM(CAST(industry_code AS VARCHAR)), '') AS industry_code_raw,
               NULLIF(TRIM(CAST(industry_name AS VARCHAR)), '') AS industry_name,
               NULLIF(TRIM(CAST(industry_standard AS VARCHAR)), '') AS industry_standard
        FROM {_scan(industry_paths)}
        WHERE TRY_CAST(trade_date AS DATE) BETWEEN CAST(? AS DATE) AND CAST(? AS DATE)
        """,
        [industry_paths, LOAD_START_DATE, LOAD_END_DATE],
    )
    connection.execute(
        """
        CREATE OR REPLACE TEMP TABLE industry AS
        WITH same_day_name_code AS (
            SELECT trade_date,
                   industry_name,
                   MIN(industry_code_raw) AS mapped_industry_code
            FROM industry_raw
            WHERE industry_name IS NOT NULL
              AND industry_code_raw IS NOT NULL
            GROUP BY trade_date, industry_name
            HAVING COUNT(DISTINCT industry_code_raw) = 1
        )
        SELECT r.symbol,
               r.trade_date,
               r.industry_code_raw,
               COALESCE(
                   r.industry_code_raw,
                   m.mapped_industry_code,
                   CASE WHEN r.industry_name IS NOT NULL
                        THEN 'NAME:' || r.industry_name END
               ) AS industry_code,
               r.industry_name,
               r.industry_standard,
               CASE WHEN r.industry_code_raw IS NOT NULL THEN 'source_code'
                    WHEN m.mapped_industry_code IS NOT NULL THEN 'same_day_name_code'
                    WHEN r.industry_name IS NOT NULL THEN 'name_fallback'
                    ELSE 'missing' END AS industry_join_method
        FROM industry_raw r
        LEFT JOIN same_day_name_code m
          ON m.trade_date = r.trade_date
         AND m.industry_name = r.industry_name
        """,
    )
    connection.execute(
        """
        CREATE OR REPLACE TEMP TABLE base0 AS
        SELECT d.*,
               c.trade_idx,
               c.prev_trade_date,
               c.next_trade_date,
               COALESCE(s.is_st, FALSE) AS is_st,
               u.board,
               u.exchange,
               u.list_date,
               u.name,
               i.industry_code,
               i.industry_name,
               i.industry_standard,
               i.industry_join_method
        FROM daily_raw d
        LEFT JOIN calendar c USING (trade_date)
        LEFT JOIN status s USING (symbol, trade_date)
        LEFT JOIN universe u USING (symbol, trade_date)
        LEFT JOIN industry i USING (symbol, trade_date)
        WHERE c.trade_idx IS NOT NULL
          AND COALESCE(u.board, 'main') = 'main'
        """,
    )
    connection.execute(
        """
        CREATE OR REPLACE TEMP TABLE marked0 AS
        SELECT b.*,
               LAG(trade_date) OVER (PARTITION BY symbol ORDER BY trade_date) AS symbol_prev_date,
               LAG(close) OVER (PARTITION BY symbol ORDER BY trade_date) AS prev_close,
               LAG(volume) OVER (PARTITION BY symbol ORDER BY trade_date) AS prev_volume,
               AVG(volume) OVER (
                   PARTITION BY symbol ORDER BY trade_date
                   ROWS BETWEEN 10 PRECEDING AND 1 PRECEDING
               ) AS prior_10d_avg_volume,
               ROW_NUMBER() OVER (PARTITION BY symbol ORDER BY trade_date) AS symbol_observation_no
        FROM base0 b
        """,
    )
    connection.execute(
        f"""
        CREATE OR REPLACE TEMP TABLE marked AS
        SELECT m.*,
               (symbol_prev_date = prev_trade_date) AS continuous_prev,
               ROUND(prev_close * (1.0 + CASE WHEN is_st THEN 0.05 ELSE 0.10 END), 2) AS up_limit,
               CASE WHEN symbol_observation_no > {IPO_NO_LIMIT_OBSERVATIONS}
                          AND prev_close > 0
                          AND ABS(close - ROUND(prev_close * (1.0 + CASE WHEN is_st THEN 0.05 ELSE 0.10 END), 2)) <= {PRICE_TOLERANCE}
                    THEN TRUE ELSE FALSE END AS is_limit_up_close,
               CASE WHEN symbol_observation_no > {IPO_NO_LIMIT_OBSERVATIONS}
                          AND prev_close > 0
                          AND high >= ROUND(prev_close * (1.0 + CASE WHEN is_st THEN 0.05 ELSE 0.10 END), 2) - {PRICE_TOLERANCE}
                    THEN TRUE ELSE FALSE END AS touched_limit_up
        FROM marked0 m
        """,
    )
    connection.execute(
        """
        CREATE OR REPLACE TEMP TABLE streak_groups AS
        SELECT m.*,
               COALESCE(
                   SUM(CASE WHEN is_limit_up_close AND continuous_prev THEN 0 ELSE 1 END)
                   OVER (PARTITION BY symbol ORDER BY trade_date
                         ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING),
                   0
               ) AS streak_group
        FROM marked m
        """,
    )
    connection.execute(
        """
        CREATE OR REPLACE TEMP TABLE streaks AS
        SELECT g.*,
               CASE WHEN is_limit_up_close THEN
                   ROW_NUMBER() OVER (PARTITION BY symbol, streak_group ORDER BY trade_date)
                    ELSE 0 END AS board_no
        FROM streak_groups g
        """,
    )
    connection.execute(
        """
        CREATE OR REPLACE TEMP TABLE base AS
        WITH flags AS (
            SELECT s.*,
                   (is_limit_up_close
                    AND ABS(open - up_limit) <= 0.0051
                    AND ABS(high - up_limit) <= 0.0051
                    AND ABS(low - up_limit) <= 0.0051
                    AND ABS(close - up_limit) <= 0.0051) AS is_one_word
            FROM streaks s
        )
        SELECT f.*,
               CASE WHEN prev_close > 0 THEN close / prev_close - 1.0 ELSE NULL END AS daily_return,
               CASE WHEN prev_volume > 0 THEN volume / prev_volume ELSE NULL END AS volume_ratio_prev,
               CASE WHEN prior_10d_avg_volume > 0 THEN volume / prior_10d_avg_volume ELSE NULL END AS volume_ratio_10d,
               CASE WHEN prior_10d_avg_volume > 0 AND volume > 1.5 * prior_10d_avg_volume THEN TRUE ELSE FALSE END AS volume_gt_1_5x_10d,
               CASE WHEN prev_volume > 0 AND volume > prev_volume THEN TRUE ELSE FALSE END AS volume_up_prev
        FROM flags f
        """,
    )


def _build_events(connection: duckdb.DuckDBPyConnection) -> None:
    connection.execute(
        """
        CREATE OR REPLACE TEMP TABLE event_a AS
        SELECT 'A' AS pattern,
               b1.symbol,
               b1.is_st,
               b2.industry_code,
               b2.industry_name,
               b2.industry_join_method,
               b1.trade_date AS first_board_date,
               b2.trade_date AS second_board_date,
               CAST(NULL AS DATE) AS shrink_start_date,
               b2.trade_date AS event_date,
               b1.trade_idx AS first_board_idx,
               b2.trade_idx AS event_idx,
               b2.board_no AS event_board_no,
               b2.close AS event_close,
               b2.open AS event_open,
               b2.high AS event_high,
               b2.low AS event_low,
               b2.volume AS event_volume,
               b1.volume AS first_board_volume,
               b1.prev_volume AS pre_first_board_volume,
               b1.volume_ratio_prev AS first_volume_ratio_prev,
               b1.volume_ratio_10d AS first_volume_ratio_10d,
               b2.volume_ratio_prev AS second_volume_ratio_prev,
               b2.volume_ratio_10d AS second_volume_ratio_10d,
               CAST(NULL AS BIGINT) AS shrink_run_len,
               CAST(NULL AS BIGINT) AS reexpansion_gap,
               b2.is_limit_up_close AS event_is_limit_up,
               b2.is_one_word AS event_is_one_word,
               CAST(NULL AS DOUBLE) AS reexpansion_volume_ratio_prev,
               CAST(NULL AS DOUBLE) AS reexpansion_volume_ratio_10d
        FROM base b1
        JOIN base b2
          ON b2.symbol = b1.symbol
         AND b2.trade_idx = b1.trade_idx + 1
        WHERE b1.board_no = 1
          AND b2.board_no = 2
          AND b1.prev_volume > 0
          AND b1.volume > b1.prev_volume
          AND b1.prior_10d_avg_volume > 0
          AND b1.volume > 1.5 * b1.prior_10d_avg_volume
          AND b2.volume > b1.volume
        """,
    )
    connection.execute(
        """
        CREATE OR REPLACE TEMP TABLE b_runs AS
        WITH RECURSIVE run AS (
            SELECT b1.symbol,
                   b1.is_st,
                   b1.industry_code,
                   b1.industry_name,
                   b1.trade_date AS first_board_date,
                   b1.trade_idx AS first_board_idx,
                   b1.volume AS first_board_volume,
                   b1.prev_volume AS pre_first_board_volume,
                   b1.volume_ratio_prev AS first_volume_ratio_prev,
                   b1.volume_ratio_10d AS first_volume_ratio_10d,
                   sh.trade_idx AS current_idx,
                   1::BIGINT AS run_len
            FROM base b1
            JOIN base sh
              ON sh.symbol = b1.symbol
             AND sh.trade_idx = b1.trade_idx + 1
            WHERE b1.board_no = 1
              AND b1.prev_volume > 0
              AND b1.volume > b1.prev_volume
              AND b1.prior_10d_avg_volume > 0
              AND b1.volume > 1.5 * b1.prior_10d_avg_volume
              AND sh.is_one_word
              AND sh.volume < b1.volume
              AND sh.prior_10d_avg_volume > 0
              AND sh.volume < sh.prior_10d_avg_volume
            UNION ALL
            SELECT r.symbol,
                   r.is_st,
                   r.industry_code,
                   r.industry_name,
                   r.first_board_date,
                   r.first_board_idx,
                   r.first_board_volume,
                   r.pre_first_board_volume,
                   r.first_volume_ratio_prev,
                   r.first_volume_ratio_10d,
                   n.trade_idx AS current_idx,
                   r.run_len + 1
            FROM run r
            JOIN base n
              ON n.symbol = r.symbol
             AND n.trade_idx = r.current_idx + 1
            WHERE n.is_one_word
              AND n.volume < r.first_board_volume
              AND n.prior_10d_avg_volume > 0
              AND n.volume < n.prior_10d_avg_volume
        )
        , aggregated AS (
            SELECT r.symbol,
                   r.is_st,
                   r.industry_code,
                   r.industry_name,
                   r.first_board_date,
                   r.first_board_idx,
                   r.first_board_volume,
                   r.pre_first_board_volume,
                   r.first_volume_ratio_prev,
                   r.first_volume_ratio_10d,
                   MIN(r.current_idx) AS shrink_start_idx,
                   MAX(r.run_len) AS shrink_run_len,
                   MAX(r.current_idx) AS run_end_idx
            FROM run r
            GROUP BY ALL
        )
        SELECT a.*,
               sh.trade_date AS shrink_start_date
        FROM aggregated a
        LEFT JOIN base sh
          ON sh.symbol = a.symbol AND sh.trade_idx = a.shrink_start_idx
        """,
    )
    connection.execute(
        """
        CREATE OR REPLACE TEMP TABLE event_b AS
        WITH candidates AS (
            SELECT r.*,
                   x.trade_date AS event_date,
                   x.trade_idx AS event_idx,
                   x.board_no AS event_board_no,
                   x.close AS event_close,
                   x.open AS event_open,
                   x.high AS event_high,
                   x.low AS event_low,
                   x.volume AS event_volume,
                   x.industry_code AS event_industry_code,
                   x.industry_name AS event_industry_name,
                   x.industry_join_method AS event_industry_join_method,
                   x.volume_ratio_prev AS reexpansion_volume_ratio_prev,
                   x.volume_ratio_10d AS reexpansion_volume_ratio_10d,
                   x.is_limit_up_close AS event_is_limit_up,
                   x.is_one_word AS event_is_one_word,
                   x.trade_idx - r.run_end_idx AS reexpansion_gap,
                   ROW_NUMBER() OVER (
                       PARTITION BY r.symbol, r.first_board_idx
                       ORDER BY x.trade_idx
                   ) AS reexpansion_rank
            FROM b_runs r
            JOIN base x
              ON x.symbol = r.symbol
             AND x.trade_idx > r.run_end_idx
             AND x.trade_idx <= r.run_end_idx + 5
             AND x.prior_10d_avg_volume > 0
             AND x.volume > 1.5 * x.prior_10d_avg_volume
        )
        SELECT 'B' AS pattern,
               symbol,
               is_st,
               event_industry_code AS industry_code,
               event_industry_name AS industry_name,
               event_industry_join_method AS industry_join_method,
               first_board_date,
               CAST(NULL AS DATE) AS second_board_date,
               shrink_start_date,
               event_date,
               first_board_idx,
               event_idx,
               event_board_no,
               event_close,
               event_open,
               event_high,
               event_low,
               event_volume,
               first_board_volume,
               pre_first_board_volume,
               first_volume_ratio_prev,
               first_volume_ratio_10d,
               CAST(NULL AS DOUBLE) AS second_volume_ratio_prev,
               CAST(NULL AS DOUBLE) AS second_volume_ratio_10d,
               shrink_run_len,
               reexpansion_gap,
               event_is_limit_up,
               event_is_one_word,
               reexpansion_volume_ratio_prev,
               reexpansion_volume_ratio_10d
        FROM candidates
        WHERE reexpansion_rank = 1
        """,
    )
    connection.execute(
        """
        CREATE OR REPLACE TEMP TABLE events AS
        SELECT a.*
        FROM event_a a
        UNION ALL
        SELECT b.*
        FROM event_b b
        """,
    )


def _attach_outcomes(connection: duckdb.DuckDBPyConnection) -> None:
    outcome_columns: list[str] = []
    joins: list[str] = []
    for horizon in HORIZONS:
        alias = f"d{horizon}"
        joins.append(
            f"LEFT JOIN base {alias} ON {alias}.symbol=e.symbol AND {alias}.trade_idx=e.event_idx+{horizon}"
        )
        for field in ("open", "high", "low", "close", "volume", "board_no", "is_limit_up_close"):
            outcome_columns.append(f"{alias}.{field} AS d{horizon}_{field}")
        outcome_columns.append(
            f"CASE WHEN {alias}.close IS NOT NULL AND e.event_close > 0 THEN {alias}.close / e.event_close - 1.0 ELSE NULL END AS d{horizon}_close_return"
        )
        outcome_columns.append(
            f"CASE WHEN {alias}.open IS NOT NULL AND e.event_close > 0 THEN {alias}.open / e.event_close - 1.0 ELSE NULL END AS d{horizon}_open_return"
        )
        outcome_columns.append(
            f"CASE WHEN {alias}.high IS NOT NULL AND e.event_close > 0 THEN {alias}.high / e.event_close - 1.0 ELSE NULL END AS d{horizon}_high_return"
        )
    connection.execute(
        """
        CREATE OR REPLACE TEMP TABLE event_outcomes AS
        SELECT e.*,
               """
        + ",\n               ".join(outcome_columns)
        + """,
               CASE WHEN e.pattern='A' AND d1.board_no=3 AND d1.is_limit_up_close THEN TRUE
                    WHEN e.pattern='A' AND d1.board_no IS NOT NULL THEN FALSE
                    ELSE NULL END AS success_to_3board,
               CASE WHEN d1.close IS NOT NULL THEN TRUE ELSE FALSE END AS d1_observed,
               CASE WHEN d2.close IS NOT NULL THEN TRUE ELSE FALSE END AS d2_observed,
               CASE WHEN d3.close IS NOT NULL THEN TRUE ELSE FALSE END AS d3_observed,
               CASE WHEN d5.close IS NOT NULL THEN TRUE ELSE FALSE END AS d5_observed
        FROM events e
        """
        + "\n".join(joins),
    )


def _attach_resonance(connection: duckdb.DuckDBPyConnection) -> None:
    connection.execute(
        """
        CREATE OR REPLACE TEMP TABLE industry_day AS
        SELECT trade_date,
               is_st,
               industry_code,
               ANY_VALUE(industry_name) AS industry_name,
               COUNT(*) AS industry_member_count,
               SUM(CASE WHEN is_limit_up_close THEN 1 ELSE 0 END) AS industry_limit_up_count,
               SUM(CASE WHEN board_no >= 2 THEN 1 ELSE 0 END) AS industry_board2_count,
               SUM(CASE WHEN is_one_word THEN 1 ELSE 0 END) AS industry_one_word_count,
               SUM(CASE WHEN daily_return > 0 THEN 1 ELSE 0 END) AS industry_positive_count,
               AVG(daily_return) AS industry_mean_return
        FROM base
        WHERE industry_code IS NOT NULL
        GROUP BY trade_date, is_st, industry_code
        """,
    )
    connection.execute(
        """
        CREATE OR REPLACE TEMP TABLE market_day AS
        SELECT trade_date,
               is_st,
               COUNT(*) AS market_member_count,
               SUM(CASE WHEN is_limit_up_close THEN 1 ELSE 0 END) AS market_limit_up_count,
               SUM(CASE WHEN board_no >= 2 THEN 1 ELSE 0 END) AS market_board2_count,
               SUM(CASE WHEN daily_return > 0 THEN 1 ELSE 0 END) AS market_positive_count
        FROM base
        GROUP BY trade_date, is_st
        """,
    )
    connection.execute(
        """
        CREATE OR REPLACE TEMP TABLE enriched_events AS
        SELECT e.*,
               i.industry_member_count,
               i.industry_limit_up_count,
               i.industry_board2_count,
               i.industry_one_word_count,
               i.industry_positive_count,
               i.industry_mean_return,
               CASE WHEN e.event_is_limit_up THEN i.industry_limit_up_count - 1 ELSE i.industry_limit_up_count END AS other_industry_limit_up_count,
               CASE WHEN e.event_board_no >= 2 THEN i.industry_board2_count - 1 ELSE i.industry_board2_count END AS other_industry_board2_count,
               CASE WHEN i.industry_member_count > 1 THEN
                   (CASE WHEN e.event_is_limit_up THEN i.industry_limit_up_count - 1 ELSE i.industry_limit_up_count END)::DOUBLE
                   / NULLIF(i.industry_member_count - 1, 0)
                    ELSE NULL END AS other_industry_limit_up_share,
               m.market_member_count,
               m.market_limit_up_count,
               m.market_board2_count,
               m.market_positive_count,
               CASE WHEN e.event_is_limit_up THEN m.market_limit_up_count - 1 ELSE m.market_limit_up_count END AS other_market_limit_up_count,
               CASE WHEN m.market_member_count > 1 THEN
                   (CASE WHEN e.event_is_limit_up THEN m.market_limit_up_count - 1 ELSE m.market_limit_up_count END)::DOUBLE
                   / NULLIF(m.market_member_count - 1, 0)
                    ELSE NULL END AS other_market_limit_up_share
        FROM event_outcomes e
        LEFT JOIN industry_day i
          ON i.trade_date=e.event_date AND i.is_st=e.is_st AND i.industry_code=e.industry_code
        LEFT JOIN market_day m
          ON m.trade_date=e.event_date AND m.is_st=e.is_st
        """,
    )


def _read_events(connection: duckdb.DuckDBPyConnection) -> pd.DataFrame:
    frame = connection.execute(
        """
        SELECT *,
               CASE WHEN other_industry_limit_up_count=0 THEN '0'
                    WHEN other_industry_limit_up_count=1 THEN '1'
                    WHEN other_industry_limit_up_count=2 THEN '2'
                    ELSE '3+' END AS industry_resonance_bin,
               CASE WHEN other_market_limit_up_count=0 THEN '0'
                    WHEN other_market_limit_up_count <= 5 THEN '1-5'
                    WHEN other_market_limit_up_count <= 20 THEN '6-20'
                    WHEN other_market_limit_up_count <= 50 THEN '21-50'
                    ELSE '51+' END AS market_resonance_bin,
               CASE WHEN pattern='B' AND reexpansion_gap=1 THEN 'immediate'
                    WHEN pattern='B' AND reexpansion_gap <= 5 THEN 'within_5_days'
                    ELSE 'other' END AS reexpansion_scope,
               CASE WHEN pattern='B' AND shrink_run_len=1 THEN '1'
                    WHEN pattern='B' AND shrink_run_len=2 THEN '2'
                    WHEN pattern='B' AND shrink_run_len>=3 THEN '3+'
                    ELSE NULL END AS shrink_run_bucket,
               EXTRACT(YEAR FROM event_date)::INTEGER AS event_year
        FROM enriched_events
        WHERE event_date BETWEEN CAST(? AS DATE) AND CAST(? AS DATE)
        ORDER BY event_date, symbol, pattern
        """,
        [START_DATE, END_DATE],
    ).fetchdf()
    return frame


def _metric_row(group: pd.DataFrame, keys: dict[str, Any]) -> dict[str, Any]:
    row = dict(keys)
    row["event_count"] = int(len(group))
    row["unique_symbol_count"] = int(group["symbol"].nunique())
    row["unique_event_date_count"] = int(group["event_date"].nunique())
    for name in ("d1", "d2", "d3", "d5"):
        values = pd.to_numeric(group[f"{name}_close_return"], errors="coerce")
        observed = values.dropna()
        row[f"{name}_observed_count"] = int(observed.size)
        row[f"{name}_missing_count"] = int(values.isna().sum())
        row[f"{name}_close_mean"] = float(observed.mean()) if len(observed) else math.nan
        row[f"{name}_close_median"] = float(observed.median()) if len(observed) else math.nan
        row[f"{name}_close_positive_probability"] = float((observed > 0).mean()) if len(observed) else math.nan
        row[f"{name}_close_ge_1pct_probability"] = float((observed >= 0.01).mean()) if len(observed) else math.nan
        row[f"{name}_close_ge_3pct_probability"] = float((observed >= 0.03).mean()) if len(observed) else math.nan
        high_values = pd.to_numeric(group[f"{name}_high_return"], errors="coerce").dropna()
        row[f"{name}_high_mean"] = float(high_values.mean()) if len(high_values) else math.nan
        row[f"{name}_high_positive_probability"] = float((high_values > 0).mean()) if len(high_values) else math.nan
        open_values = pd.to_numeric(group[f"{name}_open_return"], errors="coerce").dropna()
        row[f"{name}_open_mean"] = float(open_values.mean()) if len(open_values) else math.nan
        row[f"{name}_open_positive_probability"] = float((open_values > 0).mean()) if len(open_values) else math.nan
    success = group["success_to_3board"].dropna()
    row["success_to_3board_observed_count"] = int(len(success))
    row["success_to_3board_count"] = int(success.sum()) if len(success) else 0
    row["success_to_3board_rate"] = float(success.mean()) if len(success) else math.nan
    row["industry_joined_count"] = int(group["industry_member_count"].notna().sum())
    row["industry_join_rate"] = float(group["industry_member_count"].notna().mean()) if len(group) else math.nan
    return row


def _summarize(frame: pd.DataFrame, group_columns: list[str]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    if frame.empty:
        return pd.DataFrame()
    grouped = frame.groupby(group_columns, dropna=False, sort=True)
    for keys, group in grouped:
        if not isinstance(keys, tuple):
            keys = (keys,)
        rows.append(_metric_row(group, dict(zip(group_columns, keys, strict=True))))
    return pd.DataFrame(rows)


def _build_audit(connection: duckdb.DuckDBPyConnection, events: pd.DataFrame, paths: dict[str, list[Path]]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for domain in ("market_daily_raw", "security_status", "universe_snapshot", "industry_concept"):
        source = _scan([str(p) for p in paths[domain]])
        counts = connection.execute(
            f"""
            SELECT COUNT(*) AS n_rows,
                   COUNT(DISTINCT CAST(symbol AS VARCHAR)||'|'||CAST(TRY_CAST(trade_date AS DATE) AS VARCHAR)) AS n_keys,
                   COUNT(DISTINCT CAST(symbol AS VARCHAR)) AS n_symbols,
                   COUNT(DISTINCT TRY_CAST(trade_date AS DATE)) AS n_dates
            FROM {source}
            WHERE TRY_CAST(trade_date AS DATE) BETWEEN CAST(? AS DATE) AND CAST(? AS DATE)
            """,
            [[str(p) for p in paths[domain]], START_DATE, END_DATE],
        ).fetchone()
        rows.append(
            {
                "scope": "source_domain",
                "domain": domain,
                "rows": int(counts[0]),
                "distinct_keys": int(counts[1]),
                "symbols": int(counts[2]),
                "dates": int(counts[3]),
            }
        )
    rows.extend(
        [
            {"scope": "event", "domain": "A", "rows": int((events.pattern == "A").sum()), "distinct_keys": int(events.loc[events.pattern == "A", ["symbol", "event_date"]].drop_duplicates().shape[0]), "symbols": int(events.loc[events.pattern == "A", "symbol"].nunique()), "dates": int(events.loc[events.pattern == "A", "event_date"].nunique())},
            {"scope": "event", "domain": "B", "rows": int((events.pattern == "B").sum()), "distinct_keys": int(events.loc[events.pattern == "B", ["symbol", "event_date"]].drop_duplicates().shape[0]), "symbols": int(events.loc[events.pattern == "B", "symbol"].nunique()), "dates": int(events.loc[events.pattern == "B", "event_date"].nunique())},
        ]
    )
    return pd.DataFrame(rows)


def _build_industry_fill_audit(connection: duckdb.DuckDBPyConnection) -> pd.DataFrame:
    return connection.execute(
        """
        SELECT industry_join_method AS fill_method,
               COUNT(*) AS row_count,
               COUNT(DISTINCT symbol) AS symbol_count,
               COUNT(DISTINCT trade_date) AS date_count
        FROM industry
        GROUP BY industry_join_method
        ORDER BY industry_join_method
        """
    ).fetchdf()


def _fmt_pct(value: Any) -> str:
    if value is None or pd.isna(value):
        return "n/a"
    return f"{float(value) * 100:.2f}%"


def _fmt_return(value: Any) -> str:
    if value is None or pd.isna(value):
        return "n/a"
    return f"{float(value) * 100:+.2f}%"


def _headline_row(group: pd.DataFrame) -> dict[str, Any]:
    return _metric_row(group, {})


def _build_headline_summary(events: pd.DataFrame) -> pd.DataFrame:
    specs = [
        ("A", "all", events.pattern.eq("A")),
        ("A", "ordinary", events.pattern.eq("A") & ~events.is_st.astype(bool)),
        ("B", "all", events.pattern.eq("B")),
        ("B", "ordinary", events.pattern.eq("B") & ~events.is_st.astype(bool)),
        (
            "B",
            "immediate",
            events.pattern.eq("B") & events.reexpansion_gap.eq(1),
        ),
        (
            "B",
            "immediate_ordinary",
            events.pattern.eq("B")
            & events.reexpansion_gap.eq(1)
            & ~events.is_st.astype(bool),
        ),
    ]
    rows: list[dict[str, Any]] = []
    for pattern, scope, mask in specs:
        row = _headline_row(events[mask])
        row["pattern"] = pattern
        row["scope"] = scope
        rows.append(row)
    return pd.DataFrame(rows)


def _write_summary(
    path: Path,
    events: pd.DataFrame,
    audit: pd.DataFrame,
    industry_fill_audit: pd.DataFrame,
    args: argparse.Namespace,
) -> None:
    all_a = _headline_row(events[events.pattern == "A"])
    all_b = _headline_row(events[events.pattern == "B"])
    ordinary_a = _headline_row(events[(events.pattern == "A") & (~events.is_st.astype(bool))])
    ordinary_b = _headline_row(events[(events.pattern == "B") & (~events.is_st.astype(bool))])
    immediate_b = _headline_row(
        events[(events.pattern == "B") & (events.reexpansion_gap == 1)]
    )
    fill_lookup = {
        str(row.fill_method): int(row.row_count)
        for row in industry_fill_audit.itertuples(index=False)
    }
    event_missing_industry = int(events.industry_member_count.isna().sum())
    industry_a = events[(events.pattern == "A") & (~events.is_st.astype(bool))]
    industry_b = events[
        (events.pattern == "B")
        & (events.reexpansion_gap == 1)
        & (~events.is_st.astype(bool))
    ]

    def resonance_rows(frame: pd.DataFrame) -> list[str]:
        rows: list[str] = []
        for bin_name in ("0", "1", "2", "3+"):
            group = frame[frame.industry_resonance_bin == bin_name]
            if group.empty:
                continue
            success = group.success_to_3board.dropna()
            rows.append(
                f"| {bin_name} | {len(group):,} | "
                f"{_fmt_pct(success.mean()) if len(success) else 'n/a'} | "
                f"{_fmt_return(group.d1_close_return.mean())} | "
                f"{_fmt_pct((group.d1_close_return > 0).mean())} | "
                f"{_fmt_return(group.d5_close_return.mean())} | "
                f"{_fmt_pct((group.d5_close_return > 0).mean())} |"
            )
        return rows

    lines = [
        "# Limit-up volume pattern study (2020-2025)",
        "",
        "This is a descriptive event study over active point-in-time QDP main-board daily data. It is not a portfolio backtest.",
        "",
        "## Scope and definitions",
        "",
        f"- Signal window: {START_DATE} through {END_DATE}; ordinary main-board and ST rows are reported separately.",
        f"- Upper limit: 10% for ordinary rows and 5% for ST rows, rounded to two decimals with a half-tick tolerance of {PRICE_TOLERANCE}.",
        f"- The first {IPO_NO_LIMIT_OBSERVATIONS} observed trading rows of each symbol are excluded from limit-up classification to avoid IPO no-limit windows.",
        "- Pattern A: first board volume is greater than the previous trading day and greater than 1.5 times its preceding ten-observation average; second board volume is greater than first board volume; the two boards must be consecutive market days.",
        "- Pattern B: first board meets the same 1.5-times ten-observation volume threshold, followed immediately by at least one consecutive one-word limit-up day whose volume is below both the first-board volume and that day's preceding ten-observation average. The shrink days do not need to be monotonically decreasing.",
        "- Pattern B re-expansion: the first later observed day within five market days whose volume exceeds 1.5 times its preceding ten-observation average; `immediate` means the first market day directly after the shrink run.",
        "- Premium is measured from the event-day close. Open, close, and high returns are reported for the next 1, 2, 3, and 5 market days. Missing outcomes remain missing.",
        "- Resonance is same-day same-industry breadth on the event date, excluding the event stock. Concept-level resonance is not claimed.",
        "- Industry codes are used when present; a unique same-day code is used for uncoded historical labels, otherwise a `NAME:` fallback key is retained. The fill method is reported separately.",
        "",
        "## Event counts",
        "",
        f"- Pattern A events: {int((events.pattern == 'A').sum()):,}; Pattern B re-expansion events: {int((events.pattern == 'B').sum()):,}.",
        f"- Pattern B immediate re-expansion events: {int(((events.pattern == 'B') & (events.reexpansion_gap == 1)).sum()):,}.",
        "",
        "## Headline outcomes",
        "",
        "| Group | Events | D+1 close mean | D+1 close positive | D+5 close mean | D+5 close positive | A: 3rd-board rate |",
        "|---|---:|---:|---:|---:|---:|---:|",
        f"| A all | {all_a['event_count']:,} | {_fmt_return(all_a['d1_close_mean'])} | {_fmt_pct(all_a['d1_close_positive_probability'])} | {_fmt_return(all_a['d5_close_mean'])} | {_fmt_pct(all_a['d5_close_positive_probability'])} | {_fmt_pct(all_a['success_to_3board_rate'])} |",
        f"| A ordinary | {ordinary_a['event_count']:,} | {_fmt_return(ordinary_a['d1_close_mean'])} | {_fmt_pct(ordinary_a['d1_close_positive_probability'])} | {_fmt_return(ordinary_a['d5_close_mean'])} | {_fmt_pct(ordinary_a['d5_close_positive_probability'])} | {_fmt_pct(ordinary_a['success_to_3board_rate'])} |",
        f"| B all | {all_b['event_count']:,} | {_fmt_return(all_b['d1_close_mean'])} | {_fmt_pct(all_b['d1_close_positive_probability'])} | {_fmt_return(all_b['d5_close_mean'])} | {_fmt_pct(all_b['d5_close_positive_probability'])} | n/a |",
        f"| B ordinary | {ordinary_b['event_count']:,} | {_fmt_return(ordinary_b['d1_close_mean'])} | {_fmt_pct(ordinary_b['d1_close_positive_probability'])} | {_fmt_return(ordinary_b['d5_close_mean'])} | {_fmt_pct(ordinary_b['d5_close_positive_probability'])} | n/a |",
        f"| B immediate | {immediate_b['event_count']:,} | {_fmt_return(immediate_b['d1_close_mean'])} | {_fmt_pct(immediate_b['d1_close_positive_probability'])} | {_fmt_return(immediate_b['d5_close_mean'])} | {_fmt_pct(immediate_b['d5_close_positive_probability'])} | n/a |",
        "",
        "Pattern A's success rate is the proportion whose next market day is still a valid third board. Pattern B rows start on the first qualifying re-expansion day, not on the shrink-board day.",
        "",
        "## Industry resonance (ordinary main board)",
        "",
        "The bin is the count of other same-industry stocks that closed limit-up on the event date.",
        "",
        "### Pattern A",
        "",
        "| Other same-industry limit-ups | Events | 3rd-board rate | D+1 close mean | D+1 close positive | D+5 close mean | D+5 close positive |",
        "|---|---:|---:|---:|---:|---:|---:|",
        *resonance_rows(industry_a),
        "",
        "### Pattern B immediate re-expansion",
        "",
        "| Other same-industry limit-ups | Events | 3rd-board rate | D+1 close mean | D+1 close positive | D+5 close mean | D+5 close positive |",
        "|---|---:|---:|---:|---:|---:|---:|",
        *resonance_rows(industry_b),
        "",
        "## Industry coverage audit",
        "",
        f"- Event rows without a same-day industry aggregate after the fallback logic: {event_missing_industry}.",
        f"- Source industry-code rows: {fill_lookup.get('source_code', 0):,}; same-day name-to-code fills: {fill_lookup.get('same_day_name_code', 0):,}; name fallback rows: {fill_lookup.get('name_fallback', 0):,}; unresolved rows: {fill_lookup.get('missing', 0):,}.",
        "- The `name_fallback` key is retained for auditability and should not be interpreted as a verified standard industry code.",
        "",
        "## Interpretation boundaries",
        "",
        "- These are event-level descriptive statistics, not a portfolio backtest. Same-day events can be correlated, and a close-to-close premium is not automatically executable.",
        "- Pattern A's aggregate positive mean can be dominated by the minority that reaches a third board; inspect medians, probabilities, annual splits, and costs before treating it as an edge.",
        "- Pattern B's re-expansion day is often an opening/disagreement day; a negative average is evidence against treating the event as an unconditional buy signal, not proof that every instance should be sold.",
        "- Industry breadth is a descriptive filter. QDP does not provide a reliable historical concept/主题 taxonomy for this run.",
        "",
        "The CSV files contain the complete grouped statistics; `event_records.parquet` contains the underlying event rows and all forward outcomes. A positive conditional mean is not treated as evidence of a tradable edge without checking event count, date-level consistency, missing outcomes, costs, and account constraints.",
        "",
        "## Output files",
        "",
        "- `pattern_summary.csv`: overall and ordinary/ST summaries.",
        "- `annual_summary.csv`: year-by-year summaries.",
        "- `resonance_summary.csv`: industry and market resonance bins.",
        "- `industry_resonance_summary.csv`: industry resonance bins without a market-bin cross join.",
        "- `market_resonance_summary.csv`: market resonance bins without an industry-bin cross join.",
        "- `run_length_summary.csv`: Pattern B by shrink one-word run length and re-expansion gap.",
        "- `outcome_by_success.csv`: Pattern A outcomes split by third-board success/failure.",
        "- `headline_summary.csv`: compact overall, ordinary, and immediate-re-expansion rows.",
        "- `data_audit.csv`: QDP coverage and event audit.",
        "- `industry_fill_audit.csv`: source-code and fallback coverage for the industry join.",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run(args: argparse.Namespace) -> dict[str, Any]:
    workspace = Path(args.workspace_root).resolve()
    output = (workspace / args.output_root).resolve()
    output.mkdir(parents=True, exist_ok=True)
    start = time.perf_counter()
    initial_available = _check_memory(args.memory_floor_gib)
    domains = ("market_daily_raw", "security_status", "universe_snapshot", "industry_concept", "trading_calendar")
    paths = {domain: _active_paths(workspace, domain) for domain in domains}
    connection = _configure_connection(workspace, args.memory_limit, args.threads)
    try:
        _build_base(connection, paths)
        _check_memory(args.memory_floor_gib)
        _build_events(connection)
        _attach_outcomes(connection)
        _attach_resonance(connection)
        events = _read_events(connection)
        if events.empty:
            raise RuntimeError("no_pattern_events")
        events["is_st_label"] = np.where(events["is_st"].astype(bool), "ST", "ordinary")
        event_path = output / "event_records.parquet"
        events.to_parquet(event_path, index=False)
        pattern = _summarize(events, ["pattern", "is_st_label"])
        all_rows = _summarize(events.assign(is_st_label="all"), ["pattern", "is_st_label"])
        pattern = pd.concat([pattern, all_rows], ignore_index=True)
        annual = _summarize(events, ["pattern", "is_st_label", "event_year"])
        resonance = _summarize(events, ["pattern", "is_st_label", "industry_resonance_bin", "market_resonance_bin"])
        industry_resonance = _summarize(events, ["pattern", "is_st_label", "industry_resonance_bin"])
        market_resonance = _summarize(events, ["pattern", "is_st_label", "market_resonance_bin"])
        run_length = _summarize(events[events.pattern == "B"], ["is_st_label", "shrink_run_bucket", "reexpansion_scope"])
        success_events = events[events.pattern == "A"].copy()
        success_events["success_group"] = np.select(
            [
                success_events["success_to_3board"].eq(True).fillna(False),
                success_events["success_to_3board"].eq(False).fillna(False),
            ],
            ["success", "failure"],
            default="unobserved",
        )
        outcome_by_success = _summarize(success_events, ["success_group", "is_st_label"])
        headline = _build_headline_summary(events)
        audit = _build_audit(connection, events, paths)
        industry_fill_audit = _build_industry_fill_audit(connection)
        _csv(output / "pattern_summary.csv", pattern)
        _csv(output / "annual_summary.csv", annual)
        _csv(output / "resonance_summary.csv", resonance)
        _csv(output / "industry_resonance_summary.csv", industry_resonance)
        _csv(output / "market_resonance_summary.csv", market_resonance)
        _csv(output / "run_length_summary.csv", run_length)
        _csv(output / "outcome_by_success.csv", outcome_by_success)
        _csv(output / "headline_summary.csv", headline)
        _csv(output / "data_audit.csv", audit)
        _csv(output / "industry_fill_audit.csv", industry_fill_audit)
        _write_summary(output / "research_summary.md", events, audit, industry_fill_audit, args)
        available = _check_memory(args.memory_floor_gib)
        result = {
            "schema": "quantlab.limit_up_volume_patterns/2",
            "status": "ok",
            "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            "elapsed_seconds": time.perf_counter() - start,
            "workspace": str(workspace),
            "output_root": str(output),
            "date_start": START_DATE,
            "date_end": END_DATE,
            "load_start_date": LOAD_START_DATE,
            "load_end_date": LOAD_END_DATE,
            "initial_available_gib": initial_available,
            "final_available_gib": available,
            "event_count": int(len(events)),
            "pattern_a_count": int((events.pattern == "A").sum()),
            "pattern_b_count": int((events.pattern == "B").sum()),
            "pattern_b_immediate_count": int(((events.pattern == "B") & (events.reexpansion_gap == 1)).sum()),
            "qdp_domains": domains,
            "files": {
                "event_records": "event_records.parquet",
                "pattern_summary": "pattern_summary.csv",
                "annual_summary": "annual_summary.csv",
                "resonance_summary": "resonance_summary.csv",
                "industry_resonance_summary": "industry_resonance_summary.csv",
                "market_resonance_summary": "market_resonance_summary.csv",
                "run_length_summary": "run_length_summary.csv",
                "outcome_by_success": "outcome_by_success.csv",
                "headline_summary": "headline_summary.csv",
                "data_audit": "data_audit.csv",
                "industry_fill_audit": "industry_fill_audit.csv",
                "research_summary": "research_summary.md",
            },
        }
        _dump(output / "result.json", result)
        return result
    finally:
        connection.close()
        gc.collect()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace-root", default=".")
    parser.add_argument("--output-root", default="research/records/limit_up_volume_patterns_2020_2025_v2")
    parser.add_argument("--memory-limit", default="3GB")
    parser.add_argument("--memory-floor-gib", type=float, default=0.5)
    parser.add_argument("--threads", type=int, default=4)
    return parser


if __name__ == "__main__":
    namespace = build_parser().parse_args()
    result = run(namespace)
    print(json.dumps(result, ensure_ascii=False, indent=2, default=_json_default))
