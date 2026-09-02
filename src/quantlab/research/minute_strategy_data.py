"""Bounded point-in-time data loading for the hand-designed minute study.

The existing ``minute_ma`` helpers are intentionally convenient for small
fixtures.  This module keeps the same field definitions while reducing the
historical input to complete hourly bars before pandas sees it.  Target-day
minutes remain in memory so causal event rules and market/sector context can
be evaluated without materialising the multi-year minute store.
"""

from __future__ import annotations

import gc
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from quantlab.data.qdp_v2.active import resolve_active_domain
from quantlab.data.qdp_v2.duckdb_resources import (
    DEFAULT_MEMORY_FLOOR_BYTES,
    open_guarded_duckdb,
)
from quantlab.research.minute_ma import (
    EVENT_KEY_COLUMNS,
    MinuteMAConfig,
    build_minute_ma_states_from_history,
    normalize_bar_time,
    prepare_minute_bars,
    session_minute_ordinal,
)


class MinuteStrategyDataError(ValueError):
    """Raised when a point-in-time strategy input cannot be constructed."""


@dataclass(frozen=True)
class StrategyDataConfig:
    """Data-side assumptions shared by the full development runner."""

    board: str = "main"
    exclude_st: bool = True
    history_open_days: int = 66
    outcome_open_days: int = 5
    liquidity_lookback_days: int = 20
    liquidity_match_band: float = 2.0
    duckdb_threads: int | str = 2
    temp_directory: str | None = None
    memory_floor_gib: float = 0.5
    duckdb_memory_floor_gib: float = 2.0

    def validate(self) -> None:
        if self.board != "main":
            raise MinuteStrategyDataError("strategy_data_board_unsupported")
        for name, value, minimum in (
            ("history_open_days", self.history_open_days, 1),
            ("outcome_open_days", self.outcome_open_days, 1),
            ("liquidity_lookback_days", self.liquidity_lookback_days, 1),
        ):
            if isinstance(value, bool) or not isinstance(value, (int, np.integer)) or int(value) < minimum:
                raise MinuteStrategyDataError(f"strategy_data_{name}_invalid")
        if not np.isfinite(float(self.liquidity_match_band)) or float(self.liquidity_match_band) < 1.0:
            raise MinuteStrategyDataError("strategy_data_liquidity_match_band_invalid")
        if isinstance(self.duckdb_threads, str):
            if self.duckdb_threads.strip().lower() != "auto":
                raise MinuteStrategyDataError("strategy_data_duckdb_threads_invalid")
        elif isinstance(self.duckdb_threads, bool) or not isinstance(self.duckdb_threads, (int, np.integer)) or int(self.duckdb_threads) < 1:
            raise MinuteStrategyDataError("strategy_data_duckdb_threads_invalid")
        if not np.isfinite(float(self.memory_floor_gib)) or float(self.memory_floor_gib) < 0.5:
            raise MinuteStrategyDataError("strategy_data_memory_floor_invalid")
        if (
            not np.isfinite(float(self.duckdb_memory_floor_gib))
            or float(self.duckdb_memory_floor_gib) < float(self.memory_floor_gib)
        ):
            raise MinuteStrategyDataError("strategy_data_duckdb_memory_floor_invalid")

    def as_dict(self) -> dict[str, Any]:
        self.validate()
        return {
            "board": self.board,
            "exclude_st": bool(self.exclude_st),
            "history_open_days": int(self.history_open_days),
            "outcome_open_days": int(self.outcome_open_days),
            "liquidity_lookback_days": int(self.liquidity_lookback_days),
            "liquidity_match_band": float(self.liquidity_match_band),
            "duckdb_threads": self.duckdb_threads,
            "temp_directory": self.temp_directory,
            "memory_floor_gib": float(self.memory_floor_gib),
            "duckdb_memory_floor_gib": float(self.duckdb_memory_floor_gib),
        }


def _open_strategy_duckdb(
    workspace_root: str | Path,
    config: StrategyDataConfig | None = None,
):
    """Open a guarded connection while reserving the configured RAM floor."""

    selected = config or StrategyDataConfig()
    selected.validate()
    root = Path(workspace_root).resolve()
    spill = (
        Path(selected.temp_directory).resolve()
        if selected.temp_directory
        else root / ".tmp" / "minute_strategy_duckdb"
    )
    return open_guarded_duckdb(
        ":memory:",
        temp_directory=spill,
        threads=selected.duckdb_threads,
        floor_bytes=max(
            DEFAULT_MEMORY_FLOOR_BYTES,
            int(float(selected.duckdb_memory_floor_gib) * (1024**3)),
        ),
        minimum_limit_bytes=128 * 1024**2,
    )


def _scan(paths: Sequence[Path]) -> str:
    if not paths:
        raise MinuteStrategyDataError("strategy_data_source_paths_empty")
    values = ",".join("'" + str(path).replace("'", "''") + "'" for path in paths)
    return f"read_parquet([{values}], union_by_name=true)"


def _context(workspace_root: str | Path, domain: str):
    return resolve_active_domain(domain, workspace_root=Path(workspace_root).resolve())


def _normalise_dates(values: Iterable[Any]) -> list[str]:
    raw = pd.Series(list(values), dtype="object").astype("string").str.strip()
    compact = raw.str.fullmatch(r"\d{8}", na=False)
    parsed = pd.Series(pd.NaT, index=raw.index, dtype="datetime64[ns]")
    if compact.any():
        parsed.loc[compact] = pd.to_datetime(raw.loc[compact], format="%Y%m%d", errors="coerce")
    if (~compact).any():
        parsed.loc[~compact] = pd.to_datetime(raw.loc[~compact], errors="coerce")
    if parsed.isna().any():
        raise MinuteStrategyDataError("strategy_data_date_invalid")
    return parsed.dt.strftime("%Y-%m-%d").tolist()


def _normalise_date_series(values: pd.Series) -> pd.Series:
    return pd.Series(_normalise_dates(values.tolist()), index=values.index, dtype="string")


def _symbols_frame(symbols: Sequence[str]) -> pd.DataFrame:
    values = sorted({str(value).strip() for value in symbols if value is not None and str(value).strip()})
    if not values:
        raise MinuteStrategyDataError("strategy_data_symbols_empty")
    return pd.DataFrame({"symbol": values})


def load_trading_calendar(
    workspace_root: str | Path,
    *,
    start_date: str,
    end_date: str,
    config: StrategyDataConfig | None = None,
) -> pd.DataFrame:
    """Load the explicit exchange calendar, including only open dates."""

    start = date.fromisoformat(str(start_date)).isoformat()
    end = date.fromisoformat(str(end_date)).isoformat()
    context = _context(workspace_root, "trading_calendar")
    con = _open_strategy_duckdb(workspace_root, config)
    try:
        frame = con.execute(
            f"""
            SELECT CAST(trade_date AS VARCHAR) AS trade_date,
                   BOOL_OR(COALESCE(is_open, FALSE)) AS is_open
            FROM {_scan(context.shard_paths)}
            WHERE trade_date BETWEEN ? AND ?
            GROUP BY trade_date
            ORDER BY trade_date
            """,
            [start, end],
        ).df()
    finally:
        con.close()
    if frame.empty:
        raise MinuteStrategyDataError(f"strategy_data_calendar_empty:{start}:{end}")
    frame["trade_date"] = _normalise_date_series(frame["trade_date"])
    frame["is_open"] = frame["is_open"].fillna(False).astype(bool)
    return frame.loc[frame["is_open"], ["trade_date"]].reset_index(drop=True)


def open_dates_with_history(
    workspace_root: str | Path,
    *,
    start_date: str,
    end_date: str,
    history_open_days: int,
    outcome_open_days: int = 0,
) -> tuple[list[str], list[str]]:
    """Return support dates and target/future open dates from the exchange calendar."""

    calendar = load_trading_calendar(
        workspace_root,
        start_date="2010-01-01",
        end_date=end_date,
    )["trade_date"].astype(str).tolist()
    target = date.fromisoformat(str(start_date)).isoformat()
    if target not in calendar:
        eligible = [value for value in calendar if value >= target]
        if not eligible:
            raise MinuteStrategyDataError("strategy_data_target_calendar_empty")
        target = eligible[0]
    target_index = calendar.index(target)
    end = date.fromisoformat(str(end_date)).isoformat()
    end_index = max(index for index, value in enumerate(calendar) if value <= end)
    support_start_index = max(0, target_index - int(history_open_days))
    future_end_index = min(len(calendar) - 1, end_index + int(outcome_open_days))
    return calendar[support_start_index:target_index], calendar[target_index : future_end_index + 1]


def load_point_in_time_universe(
    workspace_root: str | Path,
    *,
    trade_dates: Sequence[str],
    config: StrategyDataConfig | None = None,
) -> pd.DataFrame:
    """Return the declared in-market stock pool at each decision date."""

    selected = config or StrategyDataConfig()
    selected.validate()
    dates = _normalise_dates(trade_dates)
    if not dates:
        raise MinuteStrategyDataError("strategy_data_universe_dates_empty")
    root = Path(workspace_root).resolve()
    universe = _context(root, "universe_snapshot")
    status = _context(root, "security_status")
    industry = _context(root, "industry_concept")
    date_frame = pd.DataFrame({"trade_date": dates})
    con = _open_strategy_duckdb(root, selected)
    con.register("wanted_strategy_dates", date_frame)
    try:
        frame = con.execute(
            f"""
            WITH u AS (
                SELECT CAST(symbol AS VARCHAR) AS symbol,
                       CAST(trade_date AS VARCHAR) AS trade_date,
                       ANY_VALUE(board) AS board,
                       ANY_VALUE(list_status) AS list_status,
                       ANY_VALUE(exchange) AS exchange,
                       ANY_VALUE(name) AS name,
                       ANY_VALUE(list_date) AS list_date,
                       ANY_VALUE(delist_date) AS delist_date
                FROM {_scan(universe.shard_paths)}
                JOIN wanted_strategy_dates USING(trade_date)
                GROUP BY symbol, trade_date
            ),
            s AS (
                SELECT CAST(symbol AS VARCHAR) AS symbol,
                       CAST(trade_date AS VARCHAR) AS trade_date,
                       BOOL_OR(COALESCE(is_st, FALSE)) AS is_st,
                       BOOL_OR(COALESCE(is_suspended, FALSE)) AS is_suspended,
                       BOOL_OR(COALESCE(is_delisted, FALSE)) AS is_delisted
                FROM {_scan(status.shard_paths)}
                JOIN wanted_strategy_dates USING(trade_date)
                GROUP BY symbol, trade_date
            ),
            i AS (
                SELECT CAST(symbol AS VARCHAR) AS symbol,
                       CAST(trade_date AS VARCHAR) AS trade_date,
                       COALESCE(
                           MAX(NULLIF(industry_name, '')),
                           MAX(NULLIF(industry, '')),
                           'UNKNOWN'
                       ) AS industry_name
                FROM {_scan(industry.shard_paths)}
                JOIN wanted_strategy_dates USING(trade_date)
                GROUP BY symbol, trade_date
            )
            SELECT u.symbol, u.trade_date, u.board, u.list_status, u.exchange, u.name,
                   u.list_date, u.delist_date,
                   COALESCE(s.is_st, FALSE) AS is_st,
                   COALESCE(s.is_suspended, FALSE) AS is_suspended,
                   COALESCE(s.is_delisted, FALSE) AS is_delisted,
                   COALESCE(i.industry_name, 'UNKNOWN') AS industry_name
            FROM u
            JOIN s USING(symbol, trade_date)
            LEFT JOIN i USING(symbol, trade_date)
            WHERE u.board = ?
              AND COALESCE(u.list_status, 'L') = 'L'
              AND NOT s.is_suspended
              AND NOT s.is_delisted
              {"AND NOT s.is_st" if selected.exclude_st else ""}
            ORDER BY u.trade_date, u.symbol
            """,
            [selected.board],
        ).df()
    finally:
        con.close()
    if frame.empty:
        raise MinuteStrategyDataError("strategy_data_universe_empty")
    frame["trade_date"] = _normalise_date_series(frame["trade_date"])
    frame["symbol"] = frame["symbol"].astype(str)
    if frame.duplicated(["symbol", "trade_date"]).any():
        raise MinuteStrategyDataError("strategy_data_universe_duplicate_key")
    return frame


def _quality_exclusions(workspace_root: str | Path, start_date: str, end_date: str) -> set[tuple[str, str]]:
    root = Path(workspace_root).resolve() / "data" / "qdp" / "source_archives" / "minute" / "quality"
    excluded: set[tuple[str, str]] = set()
    for year in range(int(str(start_date)[:4]), int(str(end_date)[:4]) + 1):
        directory = root / f"year={year}"
        minute_path = directory / "minute_feature_exclusions.parquet"
        if minute_path.is_file():
            frame = pd.read_parquet(minute_path)
            flags = [name for name in ("exclude_open", "exclude_high", "exclude_low", "exclude_close") if name in frame]
            if flags:
                bad = frame.loc[frame[flags].fillna(False).any(axis=1), ["symbol", "trade_date"]]
                excluded.update(zip(bad["symbol"].astype(str), _normalise_date_series(bad["trade_date"]), strict=False))
        session_path = directory / "session_feature_exclusions.parquet"
        if session_path.is_file():
            frame = pd.read_parquet(session_path, columns=["symbol", "trade_date"])
            excluded.update(zip(frame["symbol"].astype(str), _normalise_date_series(frame["trade_date"]), strict=False))
    return {key for key in excluded if str(start_date) <= key[1] <= str(end_date)}


def load_daily_context(
    workspace_root: str | Path,
    *,
    symbols: Sequence[str],
    start_date: str,
    end_date: str,
    config: StrategyDataConfig | None = None,
) -> pd.DataFrame:
    """Load daily history and derive only prior-day causal context fields."""

    selected = config or StrategyDataConfig()
    selected.validate()
    root = Path(workspace_root).resolve()
    symbol_frame = _symbols_frame(symbols)
    daily = _context(root, "market_daily_raw")
    factors = _context(root, "adjust_factor")
    status = _context(root, "security_status")
    industry = _context(root, "industry_concept")
    auction = _context(root, "market_opening_auction")
    con = _open_strategy_duckdb(root, selected)
    con.register("wanted_strategy_symbols", symbol_frame)
    try:
        frame = con.execute(
            f"""
            WITH f AS (
                SELECT symbol, trade_date, ANY_VALUE(adjust_factor) AS adjust_factor
                FROM {_scan(factors.shard_paths)}
                WHERE trade_date BETWEEN ? AND ?
                GROUP BY symbol, trade_date
            ),
            s AS (
                SELECT symbol, trade_date,
                       BOOL_OR(COALESCE(is_st, FALSE)) AS is_st
                FROM {_scan(status.shard_paths)}
                WHERE trade_date BETWEEN ? AND ?
                GROUP BY symbol, trade_date
            ),
            i AS (
                SELECT symbol, trade_date,
                       COALESCE(MAX(NULLIF(industry_name, '')), MAX(NULLIF(industry, '')), 'UNKNOWN') AS industry_name
                FROM {_scan(industry.shard_paths)}
                WHERE trade_date BETWEEN ? AND ?
                GROUP BY symbol, trade_date
            ),
            a AS (
                SELECT symbol, trade_date,
                       CASE WHEN SUM(GREATEST(TRY_CAST(volume AS DOUBLE), 0.0)) > 0
                            THEN SUM(GREATEST(TRY_CAST(amount AS DOUBLE), 0.0))
                                 / SUM(GREATEST(TRY_CAST(volume AS DOUBLE), 0.0)) END AS auction_price,
                       SUM(GREATEST(TRY_CAST(amount AS DOUBLE), 0.0)) AS auction_amount
                FROM {_scan(auction.shard_paths)}
                WHERE trade_date BETWEEN ? AND ?
                GROUP BY symbol, trade_date
            )
            SELECT CAST(d.symbol AS VARCHAR) AS symbol,
                   CAST(d.trade_date AS VARCHAR) AS trade_date,
                   TRY_CAST(d.open AS DOUBLE) AS open,
                   TRY_CAST(d.high AS DOUBLE) AS high,
                   TRY_CAST(d.low AS DOUBLE) AS low,
                   TRY_CAST(d.close AS DOUBLE) AS close,
                   TRY_CAST(d.volume AS DOUBLE) AS volume,
                   TRY_CAST(d.amount AS DOUBLE) AS amount,
                   TRY_CAST(f.adjust_factor AS DOUBLE) AS adjust_factor,
                   COALESCE(s.is_st, FALSE) AS is_st,
                   COALESCE(i.industry_name, 'UNKNOWN') AS industry_name,
                   TRY_CAST(a.auction_price AS DOUBLE) AS auction_price,
                   TRY_CAST(a.auction_amount AS DOUBLE) AS auction_amount
            FROM {_scan(daily.shard_paths)} d
            JOIN wanted_strategy_symbols w ON w.symbol = d.symbol
            LEFT JOIN f ON f.symbol = d.symbol AND f.trade_date = d.trade_date
            LEFT JOIN s ON s.symbol = d.symbol AND s.trade_date = d.trade_date
            LEFT JOIN i ON i.symbol = d.symbol AND i.trade_date = d.trade_date
            LEFT JOIN a ON a.symbol = d.symbol AND a.trade_date = d.trade_date
            WHERE d.trade_date BETWEEN ? AND ?
            ORDER BY d.symbol, d.trade_date
            """,
            [start_date, end_date, start_date, end_date, start_date, end_date, start_date, end_date, start_date, end_date],
        ).df()
    finally:
        con.close()
    if frame.empty:
        raise MinuteStrategyDataError("strategy_data_daily_context_empty")
    frame["symbol"] = frame["symbol"].astype(str)
    frame["trade_date"] = _normalise_date_series(frame["trade_date"])
    numeric = ["open", "high", "low", "close", "volume", "amount", "adjust_factor", "auction_price", "auction_amount"]
    frame[numeric] = frame[numeric].apply(pd.to_numeric, errors="coerce")
    valid = (
        np.isfinite(frame[["open", "high", "low", "close", "volume", "amount", "adjust_factor"]]).all(axis=1)
        & frame[["open", "high", "low", "close", "adjust_factor"]].gt(0).all(axis=1)
        & frame[["volume", "amount"]].ge(0).all(axis=1)
        & frame["high"].ge(frame["low"])
    )
    frame = frame.loc[valid].copy()
    if frame.empty:
        raise MinuteStrategyDataError("strategy_data_daily_context_no_valid_rows")
    if frame.duplicated(["symbol", "trade_date"]).any():
        raise MinuteStrategyDataError("strategy_data_daily_context_duplicate_key")
    frame["adjusted_open"] = frame["open"] * frame["adjust_factor"]
    frame["adjusted_high"] = frame["high"] * frame["adjust_factor"]
    frame["adjusted_low"] = frame["low"] * frame["adjust_factor"]
    frame["adjusted_close"] = frame["close"] * frame["adjust_factor"]
    frame.sort_values(["symbol", "trade_date"], inplace=True, kind="stable", ignore_index=True)
    parts: list[pd.DataFrame] = []
    for _, group in frame.groupby("symbol", sort=False):
        current = group.copy()
        close = current["adjusted_close"]
        previous = close.shift(1)
        current["previous_close_adjusted"] = previous
        current["previous_return_1d"] = close.shift(1) / close.shift(2) - 1.0
        current["previous_return_5d"] = close.shift(1) / close.shift(6) - 1.0
        current["previous_return_20d"] = close.shift(1) / close.shift(21) - 1.0
        current["previous_return_60d"] = close.shift(1) / close.shift(61) - 1.0
        current["previous_close_to_sma_20d"] = close.shift(1) / close.shift(1).rolling(20, min_periods=20).mean() - 1.0
        current["prior_20d_median_amount"] = current["amount"].shift(1).rolling(
            int(selected.liquidity_lookback_days), min_periods=int(selected.liquidity_lookback_days)
        ).median()
        current["previous_daily_high_20"] = current["adjusted_high"].shift(1).rolling(20, min_periods=20).max()
        breakout_reference = current["adjusted_high"].shift(2).rolling(20, min_periods=20).max()
        current["daily_breakout_prior"] = current["adjusted_close"].shift(1) > breakout_reference
        current["breakout_recent"] = current["daily_breakout_prior"].rolling(5, min_periods=1).max().fillna(0).astype(bool)
        current["prior_acceleration"] = current["previous_return_5d"] - current["previous_return_20d"] > 0.0
        current["daily_trend_positive"] = (
            (current["previous_return_20d"] > 0.0) & (current["previous_close_to_sma_20d"] > 0.0)
        )
        current["limit_pct"] = np.where(current["is_st"], 0.05, 0.10)
        current["up_limit"] = previous * (1.0 + current["limit_pct"])
        current["down_limit"] = previous * (1.0 - current["limit_pct"])
        current["previous_limit_up"] = current["adjusted_high"].shift(1) >= close.shift(2) * (1.0 + current["limit_pct"].shift(1)) * (1.0 - 0.001)
        current["previous_limit_down"] = current["adjusted_low"].shift(1) <= close.shift(2) * (1.0 - current["limit_pct"].shift(1)) * (1.0 + 0.001)
        current["auction_gap"] = current["auction_price"] * current["adjust_factor"] / previous - 1.0
        current["auction_gap"] = current["auction_gap"].where(previous.gt(0))
        current["daily_trend_position"] = current["previous_close_to_sma_20d"]
        parts.append(current)
    return pd.concat(parts, ignore_index=True)


def _minute_paths(workspace_root: str | Path, start_date: str, end_date: str) -> list[Path]:
    context = _context(workspace_root, "market_intraday_1m")
    paths = [
        path
        for shard, path in zip(context.manifest.shards, context.shard_paths, strict=True)
        if str(shard.end_date) >= str(start_date) and str(shard.start_date) <= str(end_date)
    ]
    if not paths:
        raise MinuteStrategyDataError("strategy_data_minute_shards_empty")
    return paths


def load_hourly_history(
    workspace_root: str | Path,
    *,
    symbols: Sequence[str],
    start_date: str,
    end_date: str,
    config: StrategyDataConfig | None = None,
    symbol_chunk_size: int = 0,
) -> pd.DataFrame:
    """Aggregate historical minutes to complete 60-minute bars in DuckDB.

    A full-universe history query can require several gigabytes of transient
    hash/group-by state before it returns the relatively small hourly result.
    When ``symbol_chunk_size`` is positive, execute independent symbol chunks
    and release each guarded DuckDB connection before starting the next one.
    The final concatenation and sort are identical to the unchunked path, but
    the peak is bounded by one chunk rather than the whole universe.
    """

    root = Path(workspace_root).resolve()
    symbol_values = _symbols_frame(symbols)["symbol"].astype(str).tolist()
    try:
        requested_chunk = int(symbol_chunk_size)
    except (TypeError, ValueError) as exc:
        raise MinuteStrategyDataError("strategy_data_symbol_chunk_size_invalid") from exc
    if requested_chunk < 0:
        raise MinuteStrategyDataError("strategy_data_symbol_chunk_size_invalid")
    factors = _context(root, "adjust_factor")
    paths = _minute_paths(root, start_date, end_date)
    chunk_size = requested_chunk or len(symbol_values)
    frames: list[pd.DataFrame] = []
    for offset in range(0, len(symbol_values), chunk_size):
        symbol_frame = _symbols_frame(symbol_values[offset : offset + chunk_size])
        con = _open_strategy_duckdb(root, config)
        con.register("wanted_strategy_symbols", symbol_frame)
        try:
            chunk = con.execute(
                f"""
                WITH raw AS (
                    SELECT CAST(m.symbol AS VARCHAR) AS symbol,
                           CAST(m.trade_date AS VARCHAR) AS trade_date,
                           CAST(m.bar_time AS VARCHAR) AS bar_time,
                           TRY_CAST(m.open AS DOUBLE) AS open,
                           TRY_CAST(m.high AS DOUBLE) AS high,
                           TRY_CAST(m.low AS DOUBLE) AS low,
                           TRY_CAST(m.close AS DOUBLE) AS close,
                           TRY_CAST(m.volume AS DOUBLE) AS volume,
                           TRY_CAST(m.amount AS DOUBLE) AS amount,
                           TRY_CAST(f.adjust_factor AS DOUBLE) AS adjust_factor,
                           CASE
                             WHEN m.bar_time BETWEEN '093100000' AND '103000000' THEN 1
                             WHEN m.bar_time BETWEEN '103100000' AND '113000000' THEN 2
                             WHEN m.bar_time BETWEEN '130100000' AND '140000000' THEN 3
                             WHEN m.bar_time BETWEEN '140100000' AND '150000000' THEN 4
                           END AS sixty_minute_bucket
                    FROM {_scan(paths)} m
                    JOIN wanted_strategy_symbols w ON w.symbol = m.symbol
                    LEFT JOIN {_scan(factors.shard_paths)} f
                      ON f.symbol = m.symbol AND f.trade_date = m.trade_date
                    WHERE m.trade_date BETWEEN ? AND ?
                      AND (
                        m.bar_time BETWEEN '093100000' AND '113000000'
                        OR m.bar_time BETWEEN '130100000' AND '150000000'
                      )
                )
                SELECT symbol, trade_date, sixty_minute_bucket,
                       ARG_MIN(open, bar_time) AS open,
                       MAX(high) AS high,
                       MIN(low) AS low,
                       ARG_MAX(close, bar_time) AS close,
                       SUM(volume) AS volume,
                       SUM(amount) AS amount,
                       ANY_VALUE(adjust_factor) AS adjust_factor,
                       COUNT(*) AS bar_count,
                       COUNT(DISTINCT bar_time) AS distinct_bar_count
                FROM raw
                WHERE sixty_minute_bucket IS NOT NULL
                  AND adjust_factor > 0
                  AND open > 0 AND high > 0 AND low > 0 AND close > 0
                GROUP BY symbol, trade_date, sixty_minute_bucket
                HAVING bar_count = 60 AND distinct_bar_count = 60
                """,
                [start_date, end_date],
            ).df()
        finally:
            con.close()
        if not chunk.empty:
            frames.append(chunk)
        del chunk, symbol_frame, con
        gc.collect()
    frame = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    if frame.empty:
        raise MinuteStrategyDataError("strategy_data_hourly_history_empty")
    frame["trade_date"] = _normalise_date_series(frame["trade_date"])
    frame["symbol"] = frame["symbol"].astype(str)
    for column in ("open", "high", "low", "close", "volume", "amount", "adjust_factor"):
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    excluded = _quality_exclusions(root, start_date, end_date)
    if excluded:
        keys = pd.MultiIndex.from_frame(frame[["symbol", "trade_date"]])
        frame = frame.loc[~keys.isin(pd.MultiIndex.from_tuples(sorted(excluded)))].copy()
    frame["adjusted_open"] = frame["open"] * frame["adjust_factor"]
    frame["adjusted_high"] = frame["high"] * frame["adjust_factor"]
    frame["adjusted_low"] = frame["low"] * frame["adjust_factor"]
    frame["adjusted_close"] = frame["close"] * frame["adjust_factor"]
    frame.sort_values(["symbol", "trade_date", "sixty_minute_bucket"], inplace=True, kind="stable", ignore_index=True)
    frame["hour_sequence"] = frame.groupby("symbol", sort=False).cumcount().astype("int32")
    return frame


def load_target_bars(
    workspace_root: str | Path,
    *,
    symbols: Sequence[str],
    trade_dates: Sequence[str],
    daily_context: pd.DataFrame | None = None,
    require_complete_session: bool = False,
    config: StrategyDataConfig | None = None,
) -> pd.DataFrame:
    """Load target or forward bars with optional execution-state fields."""

    dates = _normalise_dates(trade_dates)
    if not dates:
        raise MinuteStrategyDataError("strategy_data_target_dates_empty")
    root = Path(workspace_root).resolve()
    symbol_frame = _symbols_frame(symbols)
    date_frame = pd.DataFrame({"trade_date": dates})
    factors = _context(root, "adjust_factor")
    status = _context(root, "security_status")
    paths = _minute_paths(root, min(dates), max(dates))
    con = _open_strategy_duckdb(root, config)
    con.register("wanted_strategy_symbols", symbol_frame)
    con.register("wanted_strategy_dates", date_frame)
    try:
        frame = con.execute(
            f"""
            WITH f AS (
                SELECT symbol, trade_date, ANY_VALUE(adjust_factor) AS adjust_factor
                FROM {_scan(factors.shard_paths)}
                JOIN wanted_strategy_dates USING(trade_date)
                GROUP BY symbol, trade_date
            ),
            s AS (
                SELECT symbol, trade_date,
                       BOOL_OR(is_st) AS is_st,
                       BOOL_OR(is_suspended) AS is_suspended,
                       BOOL_OR(is_delisted) AS is_delisted
                FROM {_scan(status.shard_paths)}
                JOIN wanted_strategy_dates USING(trade_date)
                GROUP BY symbol, trade_date
            )
            SELECT CAST(m.symbol AS VARCHAR) AS symbol,
                   CAST(m.trade_date AS VARCHAR) AS trade_date,
                   CAST(m.bar_time AS VARCHAR) AS bar_time,
                   TRY_CAST(m.open AS DOUBLE) AS open,
                   TRY_CAST(m.high AS DOUBLE) AS high,
                   TRY_CAST(m.low AS DOUBLE) AS low,
                   TRY_CAST(m.close AS DOUBLE) AS close,
                   TRY_CAST(m.volume AS DOUBLE) AS volume,
                   TRY_CAST(m.amount AS DOUBLE) AS amount,
                   TRY_CAST(f.adjust_factor AS DOUBLE) AS adjust_factor,
                   s.is_st AS is_st,
                   s.is_suspended AS is_suspended,
                   s.is_delisted AS is_delisted
            FROM {_scan(paths)} m
            JOIN wanted_strategy_symbols w ON w.symbol = m.symbol
            JOIN wanted_strategy_dates d ON d.trade_date = m.trade_date
            LEFT JOIN f ON f.symbol = m.symbol AND f.trade_date = m.trade_date
            LEFT JOIN s ON s.symbol = m.symbol AND s.trade_date = m.trade_date
            WHERE (
                m.bar_time BETWEEN '093100000' AND '113000000'
                OR m.bar_time BETWEEN '130100000' AND '150000000'
            )
            ORDER BY symbol, trade_date, bar_time
            """
        ).df()
    finally:
        con.close()
    if frame.empty:
        raise MinuteStrategyDataError("strategy_data_target_bars_empty")
    frame["symbol"] = frame["symbol"].astype(str)
    frame["trade_date"] = _normalise_date_series(frame["trade_date"])
    frame["status_known"] = frame[["is_st", "is_suspended", "is_delisted"]].notna().all(axis=1)
    frame["bar_time"] = frame["bar_time"].map(normalize_bar_time).astype("string")
    for column in ("open", "high", "low", "close", "volume", "amount", "adjust_factor"):
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    finite = np.isfinite(frame[["open", "high", "low", "close", "volume", "amount", "adjust_factor"]]).all(axis=1)
    valid = finite & frame[["open", "high", "low", "close", "adjust_factor"]].gt(0).all(axis=1)
    frame = frame.loc[valid].copy()
    frame["session_minute_ordinal"] = frame["bar_time"].map(session_minute_ordinal)
    frame = frame.loc[frame["session_minute_ordinal"].notna()].copy()
    frame["session_minute_ordinal"] = frame["session_minute_ordinal"].astype("int16")
    frame["sixty_minute_bucket"] = frame["session_minute_ordinal"].floordiv(60).add(1).astype("int8")
    if frame.duplicated(["symbol", "trade_date", "bar_time"]).any():
        raise MinuteStrategyDataError("strategy_data_target_duplicate_minute_key")
    excluded = _quality_exclusions(root, min(dates), max(dates))
    if excluded:
        keys = pd.MultiIndex.from_frame(frame[["symbol", "trade_date"]])
        frame = frame.loc[~keys.isin(pd.MultiIndex.from_tuples(sorted(excluded)))].copy()
    if require_complete_session:
        # A target-day signal must have an explicit point-in-time status.  For
        # forward outcome bars we retain unknown statuses so the event study
        # can report ``status_unknown`` instead of silently dropping them.
        frame = frame.loc[frame["status_known"]].copy()
        counts = frame.groupby(["symbol", "trade_date"], sort=False)["session_minute_ordinal"].agg(
            count="size", distinct="nunique", minimum="min", maximum="max"
        )
        complete = counts.loc[
            counts["count"].eq(240)
            & counts["distinct"].eq(240)
            & counts["minimum"].eq(0)
            & counts["maximum"].eq(239)
        ].index
        keys = pd.MultiIndex.from_frame(frame[["symbol", "trade_date"]])
        frame = frame.loc[keys.isin(complete)].copy()
    for column in ("open", "high", "low", "close"):
        frame[f"adjusted_{column}"] = frame[column] * frame["adjust_factor"]
    if daily_context is not None and not daily_context.empty:
        context_columns = [
            "symbol", "trade_date", "industry_name", "previous_close_adjusted", "prior_20d_median_amount",
            "previous_return_1d", "previous_return_5d", "previous_return_20d", "previous_return_60d",
            "previous_close_to_sma_20d", "previous_daily_high_20", "breakout_recent", "prior_acceleration",
            "daily_trend_positive", "up_limit", "down_limit", "auction_gap", "auction_amount",
            "daily_trend_position",
        ]
        available = [name for name in context_columns if name in daily_context.columns]
        context_frame = daily_context.loc[:, available].drop_duplicates(["symbol", "trade_date"])
        collisions = set(frame.columns).intersection(context_frame.columns).difference({"symbol", "trade_date"})
        if collisions:
            raise MinuteStrategyDataError(f"strategy_data_target_context_collision:{','.join(sorted(collisions))}")
        frame = frame.merge(context_frame, on=["symbol", "trade_date"], how="left", validate="many_to_one")
    return frame.sort_values(["symbol", "trade_date", "session_minute_ordinal"], kind="stable").reset_index(drop=True)


def build_minute_market_context(
    bars: pd.DataFrame,
    *,
    daily_context: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Derive causal market, industry, auction and volume proxy fields."""

    if bars.empty:
        return pd.DataFrame()
    frame = bars.copy()
    if "session_minute_ordinal" not in frame.columns or "adjusted_close" not in frame.columns:
        frame = prepare_minute_bars(frame)
    required = {"symbol", "trade_date", "bar_time", "session_minute_ordinal", "adjusted_high", "adjusted_low", "adjusted_close", "volume", "amount"}
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise MinuteStrategyDataError(f"strategy_data_context_columns_missing:{','.join(missing)}")
    if daily_context is not None and not daily_context.empty:
        context_columns = [
            "symbol", "trade_date", "industry_name", "previous_close_adjusted", "prior_20d_median_amount",
            "breakout_recent", "prior_acceleration", "daily_trend_positive", "up_limit", "down_limit",
            "auction_gap", "auction_amount", "daily_trend_position",
        ]
        available = [name for name in context_columns if name in daily_context.columns]
        context = daily_context.loc[:, available].drop_duplicates(["symbol", "trade_date"])
        collisions = set(frame.columns).intersection(context.columns).difference({"symbol", "trade_date"})
        for name in sorted(collisions):
            left = frame[["symbol", "trade_date", name]].drop_duplicates(["symbol", "trade_date"])
            right = context[["symbol", "trade_date", name]]
            check = left.merge(right, on=["symbol", "trade_date"], how="inner", suffixes=("_left", "_right"))
            if not check.empty:
                left_values = check[f"{name}_left"]
                right_values = check[f"{name}_right"]
                comparable = left_values.notna() & right_values.notna()
                if comparable.any():
                    if pd.api.types.is_numeric_dtype(left_values) or pd.api.types.is_numeric_dtype(right_values):
                        left_numeric = pd.to_numeric(left_values.loc[comparable], errors="coerce")
                        right_numeric = pd.to_numeric(right_values.loc[comparable], errors="coerce")
                        same = np.isclose(left_numeric, right_numeric, equal_nan=True)
                    else:
                        same = left_values.loc[comparable].astype(str).eq(right_values.loc[comparable].astype(str)).to_numpy()
                    if not bool(np.asarray(same).all()):
                        raise MinuteStrategyDataError(f"strategy_data_context_collision:{name}")
            context = context.drop(columns=name)
        frame = frame.merge(context, on=["symbol", "trade_date"], how="left", validate="many_to_one")
    if "industry_name" not in frame:
        frame["industry_name"] = "UNKNOWN"
    frame["industry_name"] = frame["industry_name"].fillna("UNKNOWN").astype(str)
    frame.sort_values(["symbol", "trade_date", "session_minute_ordinal"], inplace=True, kind="stable")
    frame.reset_index(drop=True, inplace=True)
    grouped = frame.groupby(["symbol", "trade_date"], sort=False, observed=True)
    previous_close = frame.get("previous_close_adjusted", pd.Series(np.nan, index=frame.index))
    frame["return_from_previous_close"] = np.where(
        pd.to_numeric(previous_close, errors="coerce").gt(0),
        frame["adjusted_close"] / pd.to_numeric(previous_close, errors="coerce") - 1.0,
        np.nan,
    )
    frame["return_1m"] = grouped["adjusted_close"].pct_change()
    frame["return_5m"] = grouped["adjusted_close"].pct_change(5)
    frame["return_20m"] = grouped["adjusted_close"].pct_change(20)
    frame["cumulative_amount"] = grouped["amount"].cumsum()
    frame["cumulative_volume"] = grouped["volume"].cumsum()
    frame["cumulative_vwap"] = frame["cumulative_amount"] / frame["cumulative_volume"].where(frame["cumulative_volume"].gt(0))
    frame["vwap_deviation"] = frame["close"] / frame["cumulative_vwap"] - 1.0
    frame["recent_volume_5"] = grouped["volume"].transform(
        lambda values: values.rolling(5, min_periods=5).mean()
    )
    frame["previous_volume_20"] = grouped["volume"].transform(
        lambda values: values.shift(5).rolling(20, min_periods=20).mean()
    )
    frame["volume_acceleration_5_20"] = frame["recent_volume_5"] / frame["previous_volume_20"] - 1.0
    prior_median_amount = pd.to_numeric(
        frame.get("prior_20d_median_amount", pd.Series(np.nan, index=frame.index)),
        errors="coerce",
    )
    frame["amount_curve_surprise"] = np.where(
        prior_median_amount.gt(0),
        frame["cumulative_amount"] / (prior_median_amount * (frame["session_minute_ordinal"] + 1.0) / 240.0) - 1.0,
        np.nan,
    )
    frame["price_impact_1m"] = np.abs(frame["return_1m"]) / (frame["amount"] / 1_000_000.0 + 1.0)
    frame["previous_high_20m"] = grouped["adjusted_high"].transform(
        lambda values: values.shift(1).rolling(20, min_periods=20).max()
    )
    frame["recent_high_breakout"] = frame["adjusted_close"] > frame["previous_high_20m"]
    frame["prior_acceleration"] = (
        frame["return_5m"].fillna(0.0) - frame["return_20m"].fillna(0.0) > 0.0
    )
    frame["vwap_supportive"] = frame["vwap_deviation"].ge(0.0).fillna(False)
    frame["amount_acceleration_positive"] = (
        frame["amount_curve_surprise"].gt(0.0) | frame["volume_acceleration_5_20"].gt(0.0)
    ).fillna(False)
    frame["volume_normal"] = frame["volume_acceleration_5_20"].between(-0.5, 1.5, inclusive="both").fillna(False)
    if "up_limit" in frame:
        frame["touched_limit_up"] = frame["adjusted_high"] >= pd.to_numeric(frame["up_limit"], errors="coerce") * (1.0 - 0.001)
        frame["touched_limit_down"] = frame["adjusted_low"] <= pd.to_numeric(frame["down_limit"], errors="coerce") * (1.0 + 0.001)
    else:
        frame["touched_limit_up"] = False
        frame["touched_limit_down"] = False
    frame["_return_positive"] = frame["return_from_previous_close"].gt(0.0)
    market = (
        frame.groupby(["trade_date", "bar_time"], sort=False, observed=True)
        .agg(
            market_return_mean=("return_from_previous_close", "mean"),
            market_breadth_positive=("_return_positive", "mean"),
            market_return_dispersion=("return_from_previous_close", "std"),
            market_total_amount=("amount", "sum"),
            market_member_count=("symbol", "nunique"),
            limit_up_count=("touched_limit_up", "sum"),
            limit_down_count=("touched_limit_down", "sum"),
        )
        .reset_index()
    )
    market["market_regime"] = np.select(
        [
            market["market_breadth_positive"].lt(0.35)
            | market["market_return_mean"].lt(-0.003)
            | market["limit_down_count"].gt(np.maximum(10.0, market["market_member_count"] * 0.05)),
            market["market_breadth_positive"].ge(0.55) & market["market_return_mean"].ge(-0.001),
        ],
        ["defensive", "supportive"],
        default="neutral",
    )
    market["market_supportive"] = market["market_regime"].eq("supportive")
    frame = frame.merge(market, on=["trade_date", "bar_time"], how="left", validate="many_to_one")
    industry = (
        frame.groupby(["trade_date", "bar_time", "industry_name"], sort=False, observed=True)
        .agg(
            sector_return_mean=("return_from_previous_close", "mean"),
            sector_breadth=("_return_positive", "mean"),
            sector_amount=("amount", "sum"),
            sector_member_count=("symbol", "nunique"),
        )
        .reset_index()
    )
    industry["sector_strength_rank"] = industry.groupby(["trade_date", "bar_time"], observed=True)["sector_return_mean"].rank(pct=True, method="average")
    industry["sector_strong"] = industry["sector_strength_rank"].ge(0.70) & industry["sector_breadth"].ge(0.50)
    frame = frame.merge(industry, on=["trade_date", "bar_time", "industry_name"], how="left", validate="many_to_one")
    frame["stock_return_rank"] = frame.groupby(["trade_date", "bar_time"], observed=True)["return_from_previous_close"].rank(pct=True, method="average")
    frame["industry_stock_return_rank"] = frame.groupby(["trade_date", "bar_time", "industry_name"], observed=True)["return_from_previous_close"].rank(pct=True, method="average")
    frame["leader_relative_return"] = frame["return_from_previous_close"] - frame["sector_return_mean"]
    frame["leader_sync"] = frame["industry_stock_return_rank"].ge(0.70) & frame["sector_breadth"].ge(0.50)
    frame["sector_amount_share"] = frame["sector_amount"] / frame["market_total_amount"].where(frame["market_total_amount"].gt(0))
    first_five = (
        frame.groupby(["symbol", "trade_date"], sort=False, observed=True)["adjusted_close"]
        .agg(
            first_5m_open="first",
            first_5m_close=lambda values: values.iloc[4] if len(values) >= 5 else np.nan,
        )
        .reset_index()
    )
    first_five["first_5m_return"] = np.where(
        first_five["first_5m_open"].gt(0),
        first_five["first_5m_close"] / first_five["first_5m_open"] - 1.0,
        np.nan,
    )
    frame = frame.merge(
        first_five.drop(columns="first_5m_open"),
        on=["symbol", "trade_date"],
        how="left",
        validate="many_to_one",
    )
    frame["auction_confirmed"] = (
        frame["session_minute_ordinal"].ge(4)
        & (
            (frame.get("auction_gap", pd.Series(0.0, index=frame.index)).ge(-0.002) & frame["first_5m_return"].ge(-0.002))
            | (frame.get("auction_gap", pd.Series(0.0, index=frame.index)).lt(-0.002) & frame["return_from_previous_close"].gt(0.0))
        )
    ).fillna(False)
    frame["breakout_recent"] = frame.get("breakout_recent", pd.Series(False, index=frame.index)).fillna(False).astype(bool)
    frame["daily_trend_positive"] = frame.get("daily_trend_positive", pd.Series(False, index=frame.index)).fillna(False).astype(bool)
    frame["not_repeated_cross"] = True
    frame["market_supportive"] = frame["market_supportive"].fillna(False).astype(bool)
    frame["sector_strong"] = frame["sector_strong"].fillna(False).astype(bool)
    frame["leader_sync"] = frame["leader_sync"].fillna(False).astype(bool)
    frame.drop(columns="_return_positive", inplace=True)
    return frame


def build_enriched_minute_ma_states(
    target_bars: pd.DataFrame,
    hourly_history: pd.DataFrame | None,
    *,
    daily_context: pd.DataFrame | None = None,
    context: pd.DataFrame | None = None,
    config: MinuteMAConfig | None = None,
    hourly_ma_history: pd.DataFrame | None = None,
    prior_touch_counts: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Build MA states and attach the rule-level causal flags."""

    context = (
        build_minute_market_context(target_bars, daily_context=daily_context)
        if context is None
        else context.copy()
    )
    context_keys = ["symbol", "trade_date", "bar_time"]
    state_columns = set(target_bars.columns) | {
        "session_minute_ordinal",
        "sixty_minute_bucket",
        "adjusted_open",
        "adjusted_high",
        "adjusted_low",
        "adjusted_close",
    }
    context = context.loc[:, [
        name for name in context.columns
        if name in context_keys or name not in state_columns
    ]].copy()
    states = build_minute_ma_states_from_history(
        target_bars,
        hourly_history,
        config=config,
        context=context,
        hourly_ma_history=hourly_ma_history,
        prior_touch_counts=prior_touch_counts,
    )
    if states.empty:
        return states
    group_key = list(EVENT_KEY_COLUMNS)
    side = states["close_below_intersection"].fillna(False).astype(int)
    previous_side = (
        states.groupby(group_key, sort=False, observed=True)["close_below_intersection"]
        .shift(1)
        .astype("boolean")
        .fillna(False)
        .astype(int)
    )
    cross_count = (side.ne(previous_side)).groupby([states[key] for key in group_key], sort=False).cumsum()
    states["not_repeated_cross"] = cross_count.lt(3)
    for name, default in (
        ("recent_high_breakout", False),
        ("prior_acceleration", False),
        ("breakout_recent", False),
        ("daily_trend_positive", False),
        ("market_supportive", False),
        ("sector_strong", False),
        ("leader_sync", False),
        ("auction_confirmed", False),
        ("volume_normal", False),
        ("vwap_supportive", False),
        ("amount_acceleration_positive", False),
    ):
        if name not in states:
            states[name] = default
        else:
            states[name] = states[name].fillna(default).astype(bool)
    return states


def build_live_ma_alignment(
    target_bars: pd.DataFrame,
    hourly_history: pd.DataFrame | None,
    *,
    config: MinuteMAConfig | None = None,
    hourly_ma_history: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Return compact all-period live-MA alignment flags for target minutes."""

    selected = config or MinuteMAConfig()
    selected.validate()
    if target_bars.empty or (hourly_history is None and hourly_ma_history is None):
        return pd.DataFrame(columns=["symbol", "trade_date", "bar_time", "ma_alignment_score", "bullish_ma_stack", "bearish_ma_stack"])
    target_source = target_bars.copy()
    if "adjusted_close" not in target_source.columns:
        target_source = prepare_minute_bars(target_source)
    target = target_source.loc[:, ["symbol", "trade_date", "bar_time", "adjusted_close"]].copy()
    target["symbol"] = target["symbol"].astype(str)
    target["trade_date"] = _normalise_date_series(target["trade_date"])
    target["bar_time"] = target["bar_time"].map(normalize_bar_time)
    keys = ["symbol", "trade_date", "sixty_minute_bucket"]
    target["session_minute_ordinal"] = target["bar_time"].map(session_minute_ordinal)
    target["sixty_minute_bucket"] = target["session_minute_ordinal"].floordiv(60).add(1).astype("int16")
    live_columns: list[str] = []
    target_dates = set(target["trade_date"].astype(str).unique())
    if hourly_ma_history is not None:
        history = hourly_ma_history.loc[
            :, ["symbol", "trade_date", "sixty_minute_bucket", "ma_period", "prior_close_sum_adjusted"]
        ].copy()
        history["symbol"] = history["symbol"].astype(str)
        history["trade_date"] = _normalise_date_series(history["trade_date"])
        history["sixty_minute_bucket"] = pd.to_numeric(
            history["sixty_minute_bucket"], errors="coerce"
        ).astype("int16")
        history["ma_period"] = pd.to_numeric(history["ma_period"], errors="coerce").astype("int16")
        history = history.loc[history["trade_date"].isin(target_dates)].copy()
        for period in selected.periods:
            current = history.loc[
                history["ma_period"].eq(int(period)),
                [*keys, "prior_close_sum_adjusted"],
            ].rename(columns={"prior_close_sum_adjusted": f"prior_sum_{int(period)}"})
            target = target.merge(current, on=keys, how="left", validate="many_to_one")
            name = f"live_ma_{int(period)}"
            target[name] = (target[f"prior_sum_{int(period)}"] + target["adjusted_close"]) / float(period)
            live_columns.append(name)
            target.drop(columns=f"prior_sum_{int(period)}", inplace=True)
    else:
        hourly = hourly_history.loc[
            :, ["symbol", "trade_date", "sixty_minute_bucket", "adjusted_close"]
        ].copy()
        hourly["symbol"] = hourly["symbol"].astype(str)
        hourly["trade_date"] = _normalise_date_series(hourly["trade_date"])
        hourly["sixty_minute_bucket"] = pd.to_numeric(hourly["sixty_minute_bucket"], errors="coerce")
        hourly = hourly.loc[hourly["sixty_minute_bucket"].notna()].copy()
        hourly["sixty_minute_bucket"] = hourly["sixty_minute_bucket"].astype("int16")
        hourly["adjusted_close"] = pd.to_numeric(hourly["adjusted_close"], errors="coerce")
        hourly = hourly.loc[np.isfinite(hourly["adjusted_close"])].copy()
        hourly.sort_values(["symbol", "trade_date", "sixty_minute_bucket"], inplace=True, kind="stable")
        hourly.reset_index(drop=True, inplace=True)
        grouped_close = hourly.groupby("symbol", sort=False, observed=True)["adjusted_close"]
        for period in selected.periods:
            window = int(period) - 1
            prior_sum = grouped_close.transform(
                lambda values, size=window: values.rolling(size, min_periods=size).sum().shift(1)
            )
            available = prior_sum.notna() & hourly["trade_date"].astype(str).isin(target_dates)
            current = hourly.loc[available, keys].copy()
            current[f"prior_sum_{int(period)}"] = prior_sum.loc[available].to_numpy()
            target = target.merge(current, on=keys, how="left", validate="many_to_one")
            name = f"live_ma_{int(period)}"
            target[name] = (target[f"prior_sum_{int(period)}"] + target["adjusted_close"]) / float(period)
            live_columns.append(name)
            target.drop(columns=f"prior_sum_{int(period)}", inplace=True)
    pivot = target.set_index(["symbol", "trade_date", "bar_time"])[live_columns]
    score = pd.Series(0.0, index=pivot.index)
    comparisons = pd.Series(0, index=pivot.index, dtype="int16")
    for left, right in zip(live_columns[:-1], live_columns[1:], strict=True):
        available = pivot[left].notna() & pivot[right].notna()
        score.loc[available] += np.sign(pivot.loc[available, left] - pivot.loc[available, right])
        comparisons.loc[available] += 1
    result = score.div(comparisons.replace(0, np.nan)).rename("ma_alignment_score").reset_index()
    result["bullish_ma_stack"] = result["ma_alignment_score"].eq(1.0)
    result["bearish_ma_stack"] = result["ma_alignment_score"].eq(-1.0)
    return result


__all__ = [
    "MinuteStrategyDataError",
    "StrategyDataConfig",
    "build_enriched_minute_ma_states",
    "build_minute_market_context",
    "build_live_ma_alignment",
    "load_daily_context",
    "load_hourly_history",
    "load_point_in_time_universe",
    "load_target_bars",
    "load_trading_calendar",
    "open_dates_with_history",
]
