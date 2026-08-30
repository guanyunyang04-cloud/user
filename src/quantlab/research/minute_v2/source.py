"""Point-in-time QDP source resolution for minute-v2."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path
from typing import Any

import pandas as pd

from quantlab.data.qdp_v2.active import resolve_active_domain

from .contracts import DAILY_WINDOWS, MinuteV2Config, MinuteV2Error

DOMAIN_NAMES = (
    "market_intraday_1m",
    "market_opening_auction",
    "market_daily_raw",
    "universe_snapshot",
    "security_status",
    "adjust_factor",
    "industry_concept",
    "trading_calendar",
    "share_capital",
    "valuation",
    "corporate_actions",
)


@dataclass(frozen=True)
class SourceSnapshot:
    workspace_root: Path
    dataset_ids: dict[str, str]
    manifests: dict[str, str]
    shard_paths: dict[str, tuple[Path, ...]]
    minute_quality_root: Path

    def as_dict(self) -> dict[str, Any]:
        return {
            "workspace_root": str(self.workspace_root),
            "dataset_ids": dict(self.dataset_ids),
            "manifests": dict(self.manifests),
            "shard_counts": {name: len(paths) for name, paths in self.shard_paths.items()},
            "minute_quality_root": str(self.minute_quality_root),
        }


def resolve_source_snapshot(workspace_root: str | Path) -> SourceSnapshot:
    root = Path(workspace_root).resolve()
    contexts = {name: resolve_active_domain(name, workspace_root=root) for name in DOMAIN_NAMES}
    return SourceSnapshot(
        workspace_root=root,
        dataset_ids={name: context.dataset_id for name, context in contexts.items()},
        manifests={name: str(context.manifest_path) for name, context in contexts.items()},
        shard_paths={name: tuple(context.shard_paths) for name, context in contexts.items()},
        minute_quality_root=(root / "data" / "qdp" / "source_archives" / "minute" / "quality").resolve(),
    )


def month_bounds(year: int, month: int) -> tuple[str, str]:
    start = date(int(year), int(month), 1)
    if int(month) == 12:
        following = date(int(year) + 1, 1, 1)
    else:
        following = date(int(year), int(month) + 1, 1)
    return start.isoformat(), (following - timedelta(days=1)).isoformat()


def prior_open_date(
    snapshot: SourceSnapshot,
    *,
    start_date: str,
    open_days: int,
) -> str:
    """Return a bounded market-calendar warm-up date without assuming weekdays are open."""

    frames = [
        pd.read_parquet(path, columns=["trade_date", "is_open"])
        for path in snapshot.shard_paths["trading_calendar"]
    ]
    calendar = pd.concat(frames, ignore_index=True)
    values = sorted(
        calendar.loc[
            calendar["is_open"].fillna(False)
            & calendar["trade_date"].astype(str).lt(str(start_date)),
            "trade_date",
        ]
        .astype(str)
        .unique()
        .tolist()
    )
    required = int(open_days)
    if required <= 0:
        return str(start_date)
    if len(values) < required:
        raise MinuteV2Error(
            f"minute_v2_history_calendar_incomplete:{start_date}:{len(values)}:{required}"
        )
    return str(values[-required])


def overlapping_minute_paths(
    snapshot: SourceSnapshot,
    *,
    start_date: str,
    end_date: str,
) -> list[Path]:
    context = resolve_active_domain("market_intraday_1m", workspace_root=snapshot.workspace_root)
    selected = [
        path
        for shard, path in zip(context.manifest.shards, context.shard_paths, strict=True)
        if str(shard.end_date) >= str(start_date) and str(shard.start_date) <= str(end_date)
    ]
    if not selected:
        raise MinuteV2Error(f"minute_v2_minute_shards_missing:{start_date}:{end_date}")
    return selected


def _sql_literal(value: str | Path) -> str:
    return "'" + str(value).replace("'", "''") + "'"


def _finite_double_sql(expression: str) -> str:
    value = f"TRY_CAST({expression} AS DOUBLE)"
    return f"CASE WHEN isfinite({value}) THEN {value} END"


def _finite_positive_double_sql(expression: str) -> str:
    value = f"TRY_CAST({expression} AS DOUBLE)"
    return f"CASE WHEN isfinite({value}) AND {value} > 0 THEN {value} END"


def _safe_ln_sql(expression: str) -> str:
    return f"CASE WHEN isfinite({expression}) AND {expression} > 0 THEN LN({expression}) END"


def _safe_ratio_sql(numerator: str, denominator: str) -> str:
    ratio = f"({numerator}) / ({denominator})"
    return (
        f"CASE WHEN isfinite({numerator}) AND isfinite({denominator}) "
        f"AND {denominator} <> 0 AND isfinite({ratio}) THEN {ratio} END"
    )


def _scan(paths: tuple[Path, ...] | list[Path]) -> str:
    if not paths:
        raise MinuteV2Error("minute_v2_parquet_scan_empty")
    values = ",".join(_sql_literal(path) for path in paths)
    return f"read_parquet([{values}], union_by_name=true)"


def _date_literal(value: str) -> str:
    parsed = date.fromisoformat(str(value))
    return _sql_literal(parsed.isoformat())


def _quality_frames(
    snapshot: SourceSnapshot,
    *,
    start_date: str,
    end_date: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    years = range(int(str(start_date)[:4]), int(str(end_date)[:4]) + 1)
    minute_parts: list[pd.DataFrame] = []
    session_parts: list[pd.DataFrame] = []
    for year in years:
        directory = snapshot.minute_quality_root / f"year={year}"
        minute_path = directory / "minute_feature_exclusions.parquet"
        session_path = directory / "session_feature_exclusions.parquet"
        if minute_path.is_file():
            current = pd.read_parquet(
                minute_path,
                columns=[
                    "symbol",
                    "trade_date",
                    "exclude_open",
                    "exclude_high",
                    "exclude_low",
                    "exclude_close",
                ],
            )
            minute_parts.append(current)
        if session_path.is_file():
            current = pd.read_parquet(session_path)
            if not current.empty:
                session_parts.append(current.loc[:, ["symbol", "trade_date", "exclusion_reason"]])
    minute = (
        pd.concat(minute_parts, ignore_index=True)
        if minute_parts
        else pd.DataFrame(
            {
                "symbol": pd.Series(dtype="string"),
                "trade_date": pd.Series(dtype="string"),
                "exclude_open": pd.Series(dtype=bool),
                "exclude_high": pd.Series(dtype=bool),
                "exclude_low": pd.Series(dtype=bool),
                "exclude_close": pd.Series(dtype=bool),
            }
        )
    )
    session = (
        pd.concat(session_parts, ignore_index=True)
        if session_parts
        else pd.DataFrame(
            {
                "symbol": pd.Series(dtype="string"),
                "trade_date": pd.Series(dtype="string"),
                "exclusion_reason": pd.Series(dtype="string"),
            }
        )
    )
    for frame in (minute, session):
        if not frame.empty:
            frame["symbol"] = frame["symbol"].astype(str)
            frame["trade_date"] = frame["trade_date"].astype(str)
            frame.drop_duplicates(["symbol", "trade_date"], inplace=True)
    return minute, session


def register_source_views(
    connection: Any,
    snapshot: SourceSnapshot,
    *,
    start_date: str,
    end_date: str,
    minute_history_start_date: str | None = None,
    minute_history_end_date: str | None = None,
    minute_end_date: str | None = None,
    minute_extended_start_date: str | None = None,
) -> None:
    """Register bounded minute views and lazy auxiliary-domain views."""

    start = _date_literal(start_date)
    end = _date_literal(end_date)
    history_start_value = str(minute_history_start_date or start_date)
    history_start = _date_literal(history_start_value)
    history_end_value = str(minute_history_end_date or end_date)
    history_end = _date_literal(history_end_value)
    extended_start_value = str(minute_extended_start_date or start_date)
    extended_start = _date_literal(extended_start_value)
    extended_end_value = str(minute_end_date or end_date)
    extended_end = _date_literal(extended_end_value)
    target_paths = overlapping_minute_paths(snapshot, start_date=start_date, end_date=end_date)
    history_paths = overlapping_minute_paths(
        snapshot,
        start_date=history_start_value,
        end_date=history_end_value,
    )
    extended_paths = overlapping_minute_paths(
        snapshot,
        start_date=extended_start_value,
        end_date=extended_end_value,
    )
    connection.execute(
        f"CREATE OR REPLACE TEMP VIEW minute_bars AS SELECT * FROM {_scan(target_paths)} "
        f"WHERE trade_date BETWEEN {start} AND {end}"
    )
    connection.execute(
        f"CREATE OR REPLACE TEMP VIEW minute_bars_history AS "
        f"SELECT * FROM {_scan(history_paths)} "
        f"WHERE trade_date BETWEEN {history_start} AND {history_end}"
    )
    connection.execute(
        f"CREATE OR REPLACE TEMP VIEW minute_bars_extended AS SELECT * FROM {_scan(extended_paths)} "
        f"WHERE trade_date BETWEEN {extended_start} AND {extended_end}"
    )
    for name, view in (
        ("market_opening_auction", "opening_auction"),
        ("market_daily_raw", "daily_raw"),
        ("universe_snapshot", "universe_snapshot"),
        ("security_status", "security_status"),
        ("adjust_factor", "adjust_factor"),
        ("industry_concept", "industry_concept"),
        ("trading_calendar", "trading_calendar"),
        ("share_capital", "share_capital"),
        ("valuation", "valuation"),
        ("corporate_actions", "corporate_actions"),
    ):
        connection.execute(
            f"CREATE OR REPLACE TEMP VIEW {view} AS SELECT * FROM {_scan(snapshot.shard_paths[name])}"
        )
    minute_quality, session_quality = _quality_frames(
        snapshot,
        start_date=history_start_value,
        end_date=extended_end_value,
    )
    connection.register("minute_feature_exclusions_frame", minute_quality)
    connection.register("session_feature_exclusions_frame", session_quality)
    connection.execute(
        "CREATE OR REPLACE TEMP VIEW minute_feature_exclusions AS "
        "SELECT * FROM minute_feature_exclusions_frame"
    )
    connection.execute(
        "CREATE OR REPLACE TEMP VIEW session_feature_exclusions AS "
        "SELECT * FROM session_feature_exclusions_frame"
    )


def stock_day_query(
    *,
    start_date: str,
    end_date: str,
    config: MinuteV2Config,
) -> str:
    config.validate()
    start_sql = _date_literal(start_date)
    end_sql = _date_literal(end_date)
    st_clause = "AND st.is_st = FALSE" if config.exclude_st else ""
    lookback = int(config.daily_history_lookback_open_days)
    minimum_history = int(config.minimum_daily_history)
    daily_state_expressions: list[str] = []
    for window in DAILY_WINDOWS:
        daily_state_expressions.extend(
            [
                f"""CASE WHEN COUNT(*) OVER (PARTITION BY symbol ORDER BY trade_date
                         ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING) >= {window + 1}
                     AND LAG(adjusted_close, {window + 1}) OVER w > 0
                     THEN {_safe_ratio_sql(
                          "LAG(adjusted_close, 1) OVER w",
                         f"LAG(adjusted_close, {window + 1}) OVER w",
                     )} - 1.0
                END AS previous_return_{window}d""",
                f"""CASE WHEN COUNT(*) OVER (PARTITION BY symbol ORDER BY trade_date
                         ROWS BETWEEN {window} PRECEDING AND 1 PRECEDING) = {window}
                     AND AVG(adjusted_close) OVER (PARTITION BY symbol ORDER BY trade_date
                         ROWS BETWEEN {window} PRECEDING AND 1 PRECEDING) > 0
                     THEN {_safe_ratio_sql(
                          "LAG(adjusted_close, 1) OVER w",
                         f"AVG(adjusted_close) OVER (PARTITION BY symbol ORDER BY trade_date ROWS BETWEEN {window} PRECEDING AND 1 PRECEDING)",
                     )} - 1.0
                END AS previous_close_to_sma_{window}d""",
                f"""CASE WHEN COUNT(adjusted_log_return) OVER (
                         PARTITION BY symbol ORDER BY trade_date
                         ROWS BETWEEN {window} PRECEDING AND 1 PRECEDING) = {window}
                     THEN STDDEV_SAMP(adjusted_log_return) OVER (
                         PARTITION BY symbol ORDER BY trade_date
                         ROWS BETWEEN {window} PRECEDING AND 1 PRECEDING)
                END AS previous_volatility_{window}d""",
                f"""CASE WHEN COUNT(amount) OVER (PARTITION BY symbol ORDER BY trade_date
                         ROWS BETWEEN {window} PRECEDING AND 1 PRECEDING) = {window}
                     AND AVG(amount) OVER (PARTITION BY symbol ORDER BY trade_date
                         ROWS BETWEEN {window} PRECEDING AND 1 PRECEDING) > 0
                     THEN {_safe_ratio_sql(
                          "LAG(amount, 1) OVER w",
                         f"AVG(amount) OVER (PARTITION BY symbol ORDER BY trade_date ROWS BETWEEN {window} PRECEDING AND 1 PRECEDING)",
                     )} - 1.0
                END AS previous_amount_ratio_{window}d""",
            ]
        )
    daily_state_sql = ",\n                ".join(daily_state_expressions)
    daily_projection = ",\n                ".join(
        name
        for window in DAILY_WINDOWS
        for name in (
            f"d.previous_return_{window}d",
            f"d.previous_close_to_sma_{window}d",
            f"d.previous_volatility_{window}d",
            f"d.previous_amount_ratio_{window}d",
        )
    )
    return f"""
        WITH prior_open_dates AS (
            SELECT trade_date
            FROM (
                SELECT DISTINCT trade_date
                FROM trading_calendar
                WHERE is_open AND trade_date < {start_sql}
                ORDER BY trade_date DESC
                LIMIT {lookback}
            )
        ),
        history_bounds AS (
            SELECT MIN(trade_date) AS warmup_date FROM prior_open_dates
        ),
        factors AS (
            SELECT
                symbol,
                trade_date,
                ANY_VALUE({_finite_positive_double_sql("adjust_factor")}) AS adjust_factor
            FROM adjust_factor
            WHERE trade_date BETWEEN (SELECT warmup_date FROM history_bounds) AND {end_sql}
            GROUP BY symbol, trade_date
        ),
        daily_joined AS (
            SELECT
                d.symbol,
                d.trade_date,
                {_finite_positive_double_sql("d.close")} AS close,
                {_finite_double_sql("d.amount")} AS raw_amount,
                CASE WHEN isfinite({_finite_positive_double_sql("d.close")} * f.adjust_factor)
                     THEN {_finite_positive_double_sql("d.close")} * f.adjust_factor END AS adjusted_close,
                f.adjust_factor
            FROM daily_raw d
            JOIN factors f USING(symbol, trade_date)
            WHERE d.trade_date BETWEEN (SELECT warmup_date FROM history_bounds) AND {end_sql}
              AND {_finite_positive_double_sql("d.close")} IS NOT NULL
              AND f.adjust_factor IS NOT NULL
              AND isfinite({_finite_positive_double_sql("d.close")} * f.adjust_factor)
        ),
        daily_returns AS (
            SELECT
                *,
                CASE WHEN isfinite(adjusted_close)
                          AND adjusted_close > 0
                          AND isfinite(LAG(adjusted_close) OVER w)
                          AND LAG(adjusted_close) OVER w > 0
                          AND isfinite(adjusted_close / LAG(adjusted_close) OVER w)
                          AND adjusted_close / LAG(adjusted_close) OVER w > 0
                     THEN LN(adjusted_close / LAG(adjusted_close) OVER w) END AS adjusted_log_return,
                CASE WHEN isfinite(raw_amount) THEN GREATEST(raw_amount, 0.0) END AS amount
            FROM daily_joined
            WINDOW w AS (PARTITION BY symbol ORDER BY trade_date)
        ),
        daily_state AS (
            SELECT
                symbol,
                trade_date,
                adjust_factor,
                LAG(close, 1) OVER w AS previous_close,
                LAG(adjust_factor, 1) OVER w AS previous_adjust_factor,
                 CASE WHEN LAG(adjusted_close, 2) OVER w > 0
                      THEN {_safe_ratio_sql(
                          "LAG(adjusted_close, 1) OVER w",
                          "LAG(adjusted_close, 2) OVER w",
                      )} - 1.0
                 END AS previous_return_1d,
                CASE WHEN COUNT(amount) OVER (
                    PARTITION BY symbol ORDER BY trade_date ROWS BETWEEN 20 PRECEDING AND 1 PRECEDING
                ) = 20 THEN AVG(amount) OVER (
                    PARTITION BY symbol ORDER BY trade_date ROWS BETWEEN 20 PRECEDING AND 1 PRECEDING
                ) END AS previous_amount_20d,
                COUNT(*) OVER (
                    PARTITION BY symbol ORDER BY trade_date
                    ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING
                ) AS prior_history_count,
                {daily_state_sql}
            FROM daily_returns
            WINDOW w AS (PARTITION BY symbol ORDER BY trade_date)
        ),
        capital_state AS (
            SELECT
                symbol,
                trade_date,
                LAG({_finite_double_sql("total_share")}, 1) OVER w AS previous_total_share,
                LAG({_finite_double_sql("float_share")}, 1) OVER w AS previous_float_share
            FROM share_capital
            WHERE trade_date BETWEEN (SELECT warmup_date FROM history_bounds) AND {end_sql}
            WINDOW w AS (PARTITION BY symbol ORDER BY trade_date)
        ),
        valuation_state AS (
            SELECT
                symbol,
                trade_date,
                LAG({_finite_double_sql("total_mv")}, 1) OVER w AS previous_total_mv,
                LAG({_finite_double_sql("circ_mv")}, 1) OVER w AS previous_circ_mv,
                LAG({_finite_double_sql("turnover_rate")}, 1) OVER w AS previous_turnover_rate,
                LAG({_finite_double_sql("pe")}, 1) OVER w AS previous_pe,
                LAG({_finite_double_sql("pb")}, 1) OVER w AS previous_pb
            FROM valuation
            WHERE trade_date BETWEEN (SELECT warmup_date FROM history_bounds) AND {end_sql}
            WINDOW w AS (PARTITION BY symbol ORDER BY trade_date)
        ),
        actions AS (
            SELECT
                symbol,
                COALESCE(NULLIF(ex_date, ''), trade_date) AS action_date,
                TRUE AS corporate_action_today,
                SUM(COALESCE(CAST(cash_dividend_per_10 AS DOUBLE), 0.0)) AS cash_dividend_per_10,
                SUM(COALESCE(CAST(bonus_share_per_10 AS DOUBLE), 0.0)) AS bonus_share_per_10,
                SUM(COALESCE(CAST(transfer_share_per_10 AS DOUBLE), 0.0)) AS transfer_share_per_10
            FROM corporate_actions
            WHERE COALESCE(NULLIF(ex_date, ''), trade_date) BETWEEN {start_sql} AND {end_sql}
              AND (
                  announcement_date IS NULL
                  OR announcement_date = ''
                  OR announcement_date < COALESCE(NULLIF(ex_date, ''), trade_date)
              )
            GROUP BY symbol, action_date
        ),
        auctions AS (
            SELECT
                symbol,
                trade_date,
                CASE WHEN SUM(GREATEST({_finite_double_sql("volume")}, 0.0)) > 0
                          AND isfinite(
                              SUM(GREATEST({_finite_double_sql("amount")}, 0.0))
                              / SUM(GREATEST({_finite_double_sql("volume")}, 0.0))
                          )
                     THEN SUM(GREATEST({_finite_double_sql("amount")}, 0.0))
                          / SUM(GREATEST({_finite_double_sql("volume")}, 0.0))
                END AS auction_price,
                SUM(GREATEST({_finite_double_sql("amount")}, 0.0)) AS auction_amount
            FROM opening_auction
            WHERE trade_date BETWEEN {start_sql} AND {end_sql}
            GROUP BY symbol, trade_date
        ),
        eligible AS (
            SELECT
                u.symbol,
                u.trade_date,
                COALESCE(NULLIF(i.industry_name, ''), NULLIF(i.industry, ''), 'UNKNOWN') AS industry_name,
                d.adjust_factor,
                d.previous_close,
                d.previous_adjust_factor,
                a.auction_price,
                a.auction_amount,
                d.previous_return_1d,
                {daily_projection},
                d.previous_amount_20d,
                d.prior_history_count >= 120 AS history_120d_available,
                d.prior_history_count >= 240 AS history_240d_available,
                c.previous_total_share,
                c.previous_float_share,
                v.previous_total_mv,
                v.previous_circ_mv,
                v.previous_turnover_rate,
                v.previous_pe,
                v.previous_pb,
                COALESCE(ca.corporate_action_today, FALSE) AS corporate_action_today,
                COALESCE(ca.cash_dividend_per_10, 0.0) AS cash_dividend_per_10,
                COALESCE(ca.bonus_share_per_10, 0.0) AS bonus_share_per_10,
                COALESCE(ca.transfer_share_per_10, 0.0) AS transfer_share_per_10,
                COALESCE(q.exclude_open, FALSE) AS exclude_open,
                COALESCE(q.exclude_high, FALSE) AS exclude_high,
                COALESCE(q.exclude_low, FALSE) AS exclude_low,
                COALESCE(q.exclude_close, FALSE) AS exclude_close
            FROM universe_snapshot u
            JOIN security_status st USING(symbol, trade_date)
            JOIN daily_state d USING(symbol, trade_date)
            LEFT JOIN capital_state c USING(symbol, trade_date)
            LEFT JOIN valuation_state v USING(symbol, trade_date)
            LEFT JOIN actions ca ON ca.symbol = u.symbol AND ca.action_date = u.trade_date
            LEFT JOIN industry_concept i USING(symbol, trade_date)
            LEFT JOIN auctions a USING(symbol, trade_date)
            LEFT JOIN minute_feature_exclusions q USING(symbol, trade_date)
            LEFT JOIN session_feature_exclusions sx USING(symbol, trade_date)
            WHERE u.trade_date BETWEEN {start_sql} AND {end_sql}
              AND u.board = {_sql_literal(config.board)}
              {st_clause}
              AND st.is_suspended = FALSE
              AND st.is_delisted = FALSE
              AND sx.symbol IS NULL
              AND d.prior_history_count >= {minimum_history}
              AND d.adjust_factor > 0
              AND d.previous_adjust_factor > 0
              AND d.previous_close > 0
              AND d.previous_amount_20d > 0
        )
        SELECT
            *,
            PERCENT_RANK() OVER (
                PARTITION BY trade_date ORDER BY previous_amount_20d NULLS FIRST
            ) AS daily_liquidity_rank
        FROM eligible
        ORDER BY symbol, trade_date
    """


def materialize_stock_days(
    connection: Any,
    *,
    start_date: str,
    end_date: str,
    config: MinuteV2Config,
) -> dict[str, int]:
    connection.execute(
        "CREATE OR REPLACE TEMP TABLE stock_days AS "
        + stock_day_query(start_date=start_date, end_date=end_date, config=config)
    )
    row = connection.execute(
        "SELECT count(*),count(DISTINCT symbol),count(DISTINCT trade_date) FROM stock_days"
    ).fetchone()
    if not row or int(row[0]) == 0:
        raise MinuteV2Error(f"minute_v2_stock_days_empty:{start_date}:{end_date}")
    return {"stock_days": int(row[0]), "symbols": int(row[1]), "dates": int(row[2])}


__all__ = [
    "DOMAIN_NAMES",
    "SourceSnapshot",
    "materialize_stock_days",
    "month_bounds",
    "overlapping_minute_paths",
    "prior_open_date",
    "register_source_views",
    "resolve_source_snapshot",
    "stock_day_query",
]
