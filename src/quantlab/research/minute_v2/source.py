"""Point-in-time QDP source resolution for minute-v2."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path
from typing import Any

import pandas as pd

from quantlab.data.qdp_v2 import resolve_active_domain

from .contracts import MinuteV2Config, MinuteV2Error

DOMAIN_NAMES = (
    "market_intraday_1m",
    "market_opening_auction",
    "market_daily_raw",
    "universe_snapshot",
    "security_status",
    "adjust_factor",
    "industry_concept",
    "trading_calendar",
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
    minute_end_date: str | None = None,
) -> None:
    """Register bounded minute views and lazy auxiliary-domain views."""

    start = _date_literal(start_date)
    end = _date_literal(end_date)
    extended_end_value = str(minute_end_date or end_date)
    extended_end = _date_literal(extended_end_value)
    target_paths = overlapping_minute_paths(snapshot, start_date=start_date, end_date=end_date)
    extended_paths = overlapping_minute_paths(
        snapshot,
        start_date=start_date,
        end_date=extended_end_value,
    )
    connection.execute(
        f"CREATE OR REPLACE TEMP VIEW minute_bars AS SELECT * FROM {_scan(target_paths)} "
        f"WHERE trade_date BETWEEN {start} AND {end}"
    )
    connection.execute(
        f"CREATE OR REPLACE TEMP VIEW minute_bars_extended AS SELECT * FROM {_scan(extended_paths)} "
        f"WHERE trade_date BETWEEN {start} AND {extended_end}"
    )
    for name, view in (
        ("market_opening_auction", "opening_auction"),
        ("market_daily_raw", "daily_raw"),
        ("universe_snapshot", "universe_snapshot"),
        ("security_status", "security_status"),
        ("adjust_factor", "adjust_factor"),
        ("industry_concept", "industry_concept"),
        ("trading_calendar", "trading_calendar"),
    ):
        connection.execute(
            f"CREATE OR REPLACE TEMP VIEW {view} AS SELECT * FROM {_scan(snapshot.shard_paths[name])}"
        )
    minute_quality, session_quality = _quality_frames(
        snapshot,
        start_date=start_date,
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
    start = date.fromisoformat(str(start_date))
    warmup = (start - timedelta(days=140)).isoformat()
    start_sql = _date_literal(start_date)
    end_sql = _date_literal(end_date)
    warmup_sql = _date_literal(warmup)
    st_clause = "AND st.is_st = FALSE" if config.exclude_st else ""
    return f"""
        WITH factors AS (
            SELECT symbol, trade_date, ANY_VALUE(adjust_factor) AS adjust_factor
            FROM adjust_factor
            WHERE trade_date BETWEEN {warmup_sql} AND {end_sql}
            GROUP BY symbol, trade_date
        ),
        daily_joined AS (
            SELECT
                d.symbol,
                d.trade_date,
                CAST(d.close AS DOUBLE) AS close,
                CAST(d.amount AS DOUBLE) AS amount,
                CAST(d.close AS DOUBLE) * CAST(f.adjust_factor AS DOUBLE) AS adjusted_close,
                CAST(f.adjust_factor AS DOUBLE) AS adjust_factor
            FROM daily_raw d
            JOIN factors f USING(symbol, trade_date)
            WHERE d.trade_date BETWEEN {warmup_sql} AND {end_sql}
              AND d.close > 0
        ),
        daily_returns AS (
            SELECT
                *,
                CASE WHEN adjusted_close > 0 AND LAG(adjusted_close) OVER w > 0
                     THEN LN(adjusted_close / LAG(adjusted_close) OVER w) END AS adjusted_log_return
            FROM daily_joined
            WINDOW w AS (PARTITION BY symbol ORDER BY trade_date)
        ),
        daily_state AS (
            SELECT
                symbol,
                trade_date,
                adjust_factor,
                LAG(close, 1) OVER w AS previous_close,
                CASE WHEN LAG(adjusted_close, 2) OVER w > 0
                     THEN LAG(adjusted_close, 1) OVER w / LAG(adjusted_close, 2) OVER w - 1.0
                END AS previous_return_1d,
                CASE WHEN LAG(adjusted_close, 6) OVER w > 0
                     THEN LAG(adjusted_close, 1) OVER w / LAG(adjusted_close, 6) OVER w - 1.0
                END AS previous_return_5d,
                CASE WHEN LAG(adjusted_close, 21) OVER w > 0
                     THEN LAG(adjusted_close, 1) OVER w / LAG(adjusted_close, 21) OVER w - 1.0
                END AS previous_return_20d,
                STDDEV_SAMP(adjusted_log_return) OVER (
                    PARTITION BY symbol ORDER BY trade_date ROWS BETWEEN 20 PRECEDING AND 1 PRECEDING
                ) AS previous_volatility_20d,
                AVG(amount) OVER (
                    PARTITION BY symbol ORDER BY trade_date ROWS BETWEEN 20 PRECEDING AND 1 PRECEDING
                ) AS previous_amount_20d,
                COUNT(*) OVER (
                    PARTITION BY symbol ORDER BY trade_date ROWS BETWEEN 20 PRECEDING AND 1 PRECEDING
                ) AS prior_history_count
            FROM daily_returns
            WINDOW w AS (PARTITION BY symbol ORDER BY trade_date)
        ),
        auctions AS (
            SELECT
                symbol,
                trade_date,
                CASE WHEN SUM(GREATEST(CAST(volume AS DOUBLE), 0.0)) > 0
                     THEN SUM(GREATEST(CAST(amount AS DOUBLE), 0.0))
                          / SUM(GREATEST(CAST(volume AS DOUBLE), 0.0))
                END AS auction_price,
                SUM(GREATEST(CAST(amount AS DOUBLE), 0.0)) AS auction_amount
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
                a.auction_price,
                a.auction_amount,
                d.previous_return_1d,
                d.previous_return_5d,
                d.previous_return_20d,
                d.previous_volatility_20d,
                d.previous_amount_20d,
                COALESCE(q.exclude_open, FALSE) AS exclude_open,
                COALESCE(q.exclude_high, FALSE) AS exclude_high,
                COALESCE(q.exclude_low, FALSE) AS exclude_low,
                COALESCE(q.exclude_close, FALSE) AS exclude_close
            FROM universe_snapshot u
            JOIN security_status st USING(symbol, trade_date)
            JOIN daily_state d USING(symbol, trade_date)
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
              AND d.prior_history_count >= 20
              AND d.adjust_factor > 0
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
    "register_source_views",
    "resolve_source_snapshot",
    "stock_day_query",
]
