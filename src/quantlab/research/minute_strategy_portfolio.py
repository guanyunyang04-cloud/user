"""Minute-level portfolio replay for the hand-designed strategy events.

The event-study layer deliberately stops at independent signal outcomes.  This
module is the first account-level layer: it consumes causal signal rows, fills
at the declared next-minute entry, enforces A-share T+1, limits the account to
five open symbols, and evaluates a small, explicit family of exit policies.
It is intentionally conservative about intrabar information: a stop or target
observed in a minute bar is filled at the next available minute open.
"""

from __future__ import annotations

import argparse
import gc
import json
import math
import sys
import time
import zlib
from collections import Counter, defaultdict
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import psutil

from quantlab.research.minute_account import default_minute_costs
from quantlab.research.minute_ma import normalize_bar_time, session_minute_ordinal
from quantlab.research.minute_strategy_data import (
    MinuteStrategyDataError,
    StrategyDataConfig,
    load_target_bars,
    load_trading_calendar,
)
from quantlab.research.portfolio import Position, buy_position, position_value, sell_position

GIB = 1024**3
CONTROL_STRATEGIES = frozenset({"s0_random_matched", "s0_liquidity_matched"})
SELECTION_RANKERS = (
    "random_hash",
    "sector_leader",
    "trend_structure",
    "flow_quality",
)
SIGNAL_COLUMNS = (
    "signal_id",
    "strategy_id",
    "strategy_family",
    "symbol",
    "signal_date",
    "signal_time",
    "entry_date",
    "entry_time",
    "entry_price",
    "entry_adjusted_price",
    "entry_observed",
    "entry_executable",
    "signal_executable",
    "diagnostic_only",
    "causal_only",
    "sixty_minute_bucket",
    "ma_period",
    "event_trigger",
    "signal_adjusted_close",
    "market_regime",
    "sector_strength_rank",
    "sector_breadth",
    "leader_relative_return",
    "recent_high_breakout",
    "prior_acceleration",
    "daily_trend_positive",
    "volume_acceleration_5_20",
    "amount_curve_surprise",
    "auction_confirmed",
    "volume_normal",
    "not_repeated_cross",
)

_RANKER_REQUIRED_COLUMNS = {
    "random_hash": frozenset(),
    "sector_leader": frozenset(
        {"sector_strength_rank", "sector_breadth", "leader_relative_return"}
    ),
    "trend_structure": frozenset(
        {"recent_high_breakout", "prior_acceleration", "daily_trend_positive"}
    ),
    "flow_quality": frozenset(
        {
            "volume_acceleration_5_20",
            "amount_curve_surprise",
            "auction_confirmed",
            "volume_normal",
            "not_repeated_cross",
        }
    ),
}


class MinutePortfolioError(RuntimeError):
    """Raised when the account or execution contract cannot be satisfied."""


@dataclass(frozen=True)
class ExitPolicy:
    """One finite, causal exit rule used by the development comparison."""

    policy_id: str
    max_hold_days: int
    stop_loss_bps: float | None = None
    take_profit_bps: float | None = None
    trail_activation_bps: float | None = None
    trail_drawdown_bps: float | None = None
    description: str = ""

    def validate(self) -> None:
        if not self.policy_id or self.policy_id.strip() != self.policy_id:
            raise MinutePortfolioError("minute_portfolio_policy_id_invalid")
        if isinstance(self.max_hold_days, bool) or int(self.max_hold_days) < 1:
            raise MinutePortfolioError("minute_portfolio_max_hold_days_invalid")
        for name in (
            "stop_loss_bps",
            "take_profit_bps",
            "trail_activation_bps",
            "trail_drawdown_bps",
        ):
            value = getattr(self, name)
            if value is not None and (
                not np.isfinite(float(value)) or float(value) <= 0.0
            ):
                raise MinutePortfolioError(f"minute_portfolio_{name}_invalid")
        if (self.trail_activation_bps is None) != (self.trail_drawdown_bps is None):
            raise MinutePortfolioError("minute_portfolio_trail_pair_invalid")
        if (
            self.trail_activation_bps is not None
            and float(self.trail_drawdown_bps) >= float(self.trail_activation_bps)
        ):
            raise MinutePortfolioError("minute_portfolio_trail_order_invalid")

    def as_dict(self) -> dict[str, Any]:
        self.validate()
        return {
            "policy_id": self.policy_id,
            "max_hold_days": int(self.max_hold_days),
            "stop_loss_bps": self.stop_loss_bps,
            "take_profit_bps": self.take_profit_bps,
            "trail_activation_bps": self.trail_activation_bps,
            "trail_drawdown_bps": self.trail_drawdown_bps,
            "description": self.description,
        }


def exit_policy_catalog() -> tuple[ExitPolicy, ...]:
    policies = (
        ExitPolicy(
            "next_open_1d",
            1,
            description="Exit at the first executable minute of the next trading day.",
        ),
        ExitPolicy(
            "next_open_2d",
            2,
            description="Two-session time stop at the first executable minute.",
        ),
        ExitPolicy(
            "next_open_3d",
            3,
            description="Three-session time stop at the first executable minute.",
        ),
        ExitPolicy(
            "next_open_5d",
            5,
            description="Five-session time stop at the first executable minute.",
        ),
        ExitPolicy(
            "protect_3pct_target6pct_3d",
            3,
            stop_loss_bps=300.0,
            take_profit_bps=600.0,
            trail_activation_bps=300.0,
            trail_drawdown_bps=200.0,
            description="Three-percent protection, six-percent target, then a two-percent trail.",
        ),
        ExitPolicy(
            "momentum_trail_4pct_5d",
            5,
            stop_loss_bps=400.0,
            trail_activation_bps=400.0,
            trail_drawdown_bps=250.0,
            description="Wider stop and trailing exit for breakout/momentum entries.",
        ),
        ExitPolicy(
            "fast_failure_2pct_target4pct_2d",
            2,
            stop_loss_bps=200.0,
            take_profit_bps=400.0,
            trail_activation_bps=200.0,
            trail_drawdown_bps=150.0,
            description="Fast failure control for weak reclaim attempts.",
        ),
    )
    for policy in policies:
        policy.validate()
    return policies


@dataclass(frozen=True)
class PortfolioConfig:
    starting_cash: float = 100_000.0
    max_positions: int = 5
    slippage_multiplier: float = 1.0
    signal_deduplication: str = "first_per_symbol_hour"
    selection_ranker: str = "random_hash"
    same_minute_tiebreak: str = "deterministic_hash"
    selection_seed: int = 7
    memory_floor_gib: float = 0.5
    soft_memory_floor_gib: float = 1.0

    def validate(self) -> None:
        if not np.isfinite(float(self.starting_cash)) or float(self.starting_cash) <= 0:
            raise MinutePortfolioError("minute_portfolio_starting_cash_invalid")
        if isinstance(self.max_positions, bool) or int(self.max_positions) < 1:
            raise MinutePortfolioError("minute_portfolio_max_positions_invalid")
        if not np.isfinite(float(self.slippage_multiplier)) or float(self.slippage_multiplier) <= 0:
            raise MinutePortfolioError("minute_portfolio_slippage_multiplier_invalid")
        if self.signal_deduplication != "first_per_symbol_hour":
            raise MinutePortfolioError("minute_portfolio_signal_deduplication_invalid")
        if self.selection_ranker not in SELECTION_RANKERS:
            raise MinutePortfolioError("minute_portfolio_selection_ranker_invalid")
        if self.same_minute_tiebreak != "deterministic_hash":
            raise MinutePortfolioError("minute_portfolio_same_minute_tiebreak_invalid")
        if isinstance(self.selection_seed, bool) or not isinstance(
            self.selection_seed, (int, np.integer)
        ):
            raise MinutePortfolioError("minute_portfolio_selection_seed_invalid")
        if not np.isfinite(float(self.memory_floor_gib)) or float(self.memory_floor_gib) < 0.5:
            raise MinutePortfolioError("minute_portfolio_memory_floor_invalid")
        if (
            not np.isfinite(float(self.soft_memory_floor_gib))
            or float(self.soft_memory_floor_gib) < float(self.memory_floor_gib)
        ):
            raise MinutePortfolioError("minute_portfolio_soft_memory_floor_invalid")

    def as_dict(self) -> dict[str, Any]:
        self.validate()
        return {
            "starting_cash": float(self.starting_cash),
            "max_positions": int(self.max_positions),
            "slippage_multiplier": float(self.slippage_multiplier),
            "signal_deduplication": self.signal_deduplication,
            "selection_ranker": self.selection_ranker,
            "same_minute_tiebreak": self.same_minute_tiebreak,
            "selection_seed": int(self.selection_seed),
            "memory_floor_gib": float(self.memory_floor_gib),
            "soft_memory_floor_gib": float(self.soft_memory_floor_gib),
        }


@dataclass
class _PendingEntry:
    record: dict[str, Any]
    signal_date_idx: int
    selection_score: float


@dataclass
class _Holding:
    trade_id: int
    symbol: str
    strategy_id: str
    signal_id: str
    signal_date: str
    signal_time: str
    entry_date_idx: int
    entry_date: str
    entry_time: str
    entry_ordinal: int
    selection_score: float
    position: Position
    entry_adjusted_price: float
    peak_adjusted_price: float
    peak_time: str
    trailing_active: bool = False
    pending_exit: bool = False
    pending_exit_reason: str = ""
    last_adjusted_close: float = math.nan
    exit_attempt_count: int = 0


@dataclass
class _Account:
    selection_ranker: str
    selection_seed: int
    strategy_id: str
    policy: ExitPolicy
    starting_cash: float
    max_positions: int
    cash: float
    previous_equity: float
    holdings: dict[int, _Holding] = field(default_factory=dict)
    holdings_by_symbol: dict[str, int] = field(default_factory=dict)
    pending_entries: dict[tuple[int, int], list[_PendingEntry]] = field(
        default_factory=lambda: defaultdict(list)
    )
    pending_symbols: set[str] = field(default_factory=set)
    pending_exits: dict[tuple[int, int], list[tuple[int, str]]] = field(
        default_factory=lambda: defaultdict(list)
    )
    equity_rows: list[dict[str, Any]] = field(default_factory=list)
    trade_rows: list[dict[str, Any]] = field(default_factory=list)
    exit_reason_counts: Counter[str] = field(default_factory=Counter)
    trade_id: int = 0
    selection_count: int = 0
    accepted_signal_count: int = 0
    filled_entry_count: int = 0
    skipped_slot_count: int = 0
    skipped_cash_count: int = 0
    skipped_duplicate_count: int = 0
    delayed_exit_count: int = 0
    unresolved_count: int = 0


def _memory_check(config: PortfolioConfig, stage: str, *, soft: bool = False) -> None:
    available = int(psutil.virtual_memory().available)
    floor = float(config.soft_memory_floor_gib if soft else config.memory_floor_gib) * GIB
    if available < int(floor):
        raise MinutePortfolioError(
            f"minute_portfolio_memory_floor_breached:{stage}:available={available}:floor={int(floor)}"
        )


def _normalise_signal_frame(frame: pd.DataFrame, strategy_id: str) -> dict[int, list[dict[str, Any]]]:
    """Filter and compact one date's signal rows for one strategy."""

    if frame.empty:
        return {}
    required = {
        "signal_id",
        "strategy_id",
        "symbol",
        "signal_date",
        "signal_time",
        "entry_date",
        "entry_time",
        "entry_price",
        "entry_adjusted_price",
        "entry_observed",
        "entry_executable",
        "signal_executable",
        "diagnostic_only",
        "sixty_minute_bucket",
        "ma_period",
    }
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise MinutePortfolioError(f"minute_portfolio_signal_columns_missing:{','.join(missing)}")
    working = frame.loc[frame["strategy_id"].astype(str).eq(str(strategy_id))].copy()
    if working.empty:
        return {}
    bool_columns = ("entry_observed", "entry_executable", "signal_executable", "diagnostic_only")
    for name in bool_columns:
        working[name] = working[name].fillna(False).astype(bool)
    if "causal_only" in working.columns:
        working["causal_only"] = working["causal_only"].fillna(False).astype(bool)
        causal_mask = working["causal_only"]
    else:
        # Pilot outputs predating the explicit field are retained as causal
        # compatibility inputs; new outputs always carry this column.
        causal_mask = True
    working = working.loc[
        ~working["diagnostic_only"]
        & causal_mask
        & working["signal_executable"]
        & working["entry_observed"]
        & working["entry_executable"]
    ].copy()
    if working.empty:
        return {}
    working["symbol"] = working["symbol"].astype(str).str.strip()
    working["signal_date"] = working["signal_date"].astype(str).str.strip()
    working["entry_date"] = working["entry_date"].astype(str).str.strip()
    try:
        signal_dates = pd.to_datetime(working["signal_date"], format="%Y-%m-%d", errors="raise")
        entry_dates = pd.to_datetime(working["entry_date"], format="%Y-%m-%d", errors="raise")
    except (TypeError, ValueError) as exc:
        raise MinutePortfolioError("minute_portfolio_signal_date_invalid") from exc
    if (signal_dates.isna() | entry_dates.isna()).any():
        raise MinutePortfolioError("minute_portfolio_signal_date_invalid")
    if (entry_dates < signal_dates).any():
        raise MinutePortfolioError("minute_portfolio_entry_before_signal")
    working["signal_time"] = working["signal_time"].map(normalize_bar_time)
    working["entry_time"] = working["entry_time"].map(normalize_bar_time)
    working["signal_ordinal"] = working["signal_time"].map(session_minute_ordinal)
    working["entry_ordinal"] = working["entry_time"].map(session_minute_ordinal)
    working["sixty_minute_bucket"] = pd.to_numeric(
        working["sixty_minute_bucket"], errors="coerce"
    )
    working["ma_period"] = pd.to_numeric(working["ma_period"], errors="coerce")
    working["entry_price"] = pd.to_numeric(working["entry_price"], errors="coerce")
    working["entry_adjusted_price"] = pd.to_numeric(
        working["entry_adjusted_price"], errors="coerce"
    )
    valid = (
        working["signal_ordinal"].notna()
        & working["entry_ordinal"].notna()
        & np.isfinite(working["sixty_minute_bucket"])
        & np.isfinite(working["ma_period"])
        & working["sixty_minute_bucket"].between(1, 4)
        & working["ma_period"].gt(1)
        & np.isfinite(working["entry_price"])
        & np.isfinite(working["entry_adjusted_price"])
        & working["entry_price"].gt(0)
        & working["entry_adjusted_price"].gt(0)
    )
    working = working.loc[valid].copy()
    if working.empty:
        return {}
    # Multiple MA periods can produce the same symbol/hour signal.  Keep the
    # earliest causal event for that symbol/hour; later hours remain eligible,
    # so a full-day signal stream is not converted into an artificial daily cap.
    working.sort_values(
        ["signal_ordinal", "sixty_minute_bucket", "ma_period", "symbol", "signal_id"],
        kind="stable",
        inplace=True,
    )
    working = working.drop_duplicates(
        ["symbol", "signal_date", "sixty_minute_bucket"], keep="first"
    )
    keep = [
        name
        for name in SIGNAL_COLUMNS
        if name in working.columns
    ] + ["signal_ordinal", "entry_ordinal"]
    working = working.loc[:, list(dict.fromkeys(keep))]
    grouped: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in working.to_dict("records"):
        ordinal = int(row.pop("signal_ordinal"))
        row["entry_ordinal"] = int(row["entry_ordinal"])
        grouped[ordinal].append(row)
    return dict(grouped)


def _selection_tiebreak(row: Mapping[str, Any], seed: int) -> int:
    payload = "|".join(
        (
            str(int(seed)),
            str(row.get("strategy_id", "")),
            str(row.get("signal_date", "")),
            normalize_bar_time(row.get("signal_time", "")),
            str(row.get("symbol", "")),
        )
    )
    return int(zlib.crc32(payload.encode("utf-8")) & 0xFFFFFFFF)


def _finite_number(row: Mapping[str, Any], name: str, *, default: float = 0.0) -> float:
    try:
        value = float(row.get(name, math.nan))
    except (TypeError, ValueError):
        return float(default)
    return value if np.isfinite(value) else float(default)


def _causal_flag(row: Mapping[str, Any], name: str) -> float:
    value = row.get(name)
    return float(pd.notna(value) and bool(value))


def _finite_percentiles(records: Sequence[Mapping[str, Any]], name: str) -> np.ndarray:
    values = np.asarray(
        [_finite_number(row, name, default=math.nan) for row in records],
        dtype=np.float64,
    )
    result = np.zeros(len(values), dtype=np.float64)
    finite = np.isfinite(values)
    if finite.any():
        result[finite] = (
            pd.Series(values[finite]).rank(method="average", pct=True).to_numpy(dtype=float)
        )
    return result


def _rank_same_minute_records(
    records: Sequence[dict[str, Any]],
    *,
    ranker: str,
    seed: int,
) -> list[tuple[dict[str, Any], float]]:
    if ranker not in SELECTION_RANKERS:
        raise MinutePortfolioError(f"minute_portfolio_unknown_selection_ranker:{ranker}")
    if not records:
        return []
    missing = sorted(
        name
        for name in _RANKER_REQUIRED_COLUMNS[ranker]
        if any(name not in row for row in records)
    )
    if missing:
        raise MinutePortfolioError(
            f"minute_portfolio_ranker_columns_missing:{ranker}:{','.join(missing)}"
        )

    scores = np.zeros(len(records), dtype=np.float64)
    if ranker == "sector_leader":
        sector = np.clip(
            np.asarray(
                [_finite_number(row, "sector_strength_rank") for row in records],
                dtype=np.float64,
            ),
            0.0,
            1.0,
        )
        breadth = np.clip(
            np.asarray(
                [_finite_number(row, "sector_breadth") for row in records],
                dtype=np.float64,
            ),
            0.0,
            1.0,
        )
        leader = _finite_percentiles(records, "leader_relative_return")
        scores = (sector + breadth + leader) / 3.0
    elif ranker == "trend_structure":
        scores = np.asarray(
            [
                (
                    _causal_flag(row, "recent_high_breakout")
                    + _causal_flag(row, "prior_acceleration")
                    + _causal_flag(row, "daily_trend_positive")
                )
                / 3.0
                for row in records
            ],
            dtype=np.float64,
        )
    elif ranker == "flow_quality":
        scores = np.asarray(
            [
                (
                    _causal_flag(row, "volume_normal")
                    + _causal_flag(row, "not_repeated_cross")
                    + _causal_flag(row, "auction_confirmed")
                    + float(
                        _finite_number(
                            row, "amount_curve_surprise", default=-math.inf
                        )
                        > 0.0
                        or _finite_number(
                            row, "volume_acceleration_5_20", default=-math.inf
                        )
                        > 0.0
                    )
                )
                / 4.0
                for row in records
            ],
            dtype=np.float64,
        )

    order = sorted(
        range(len(records)),
        key=lambda index: (
            -float(scores[index]),
            _selection_tiebreak(records[index], seed),
            str(records[index].get("signal_id")),
        ),
    )
    return [(records[index], float(scores[index])) for index in order]


def _bar_maps(bars: pd.DataFrame) -> dict[str, dict[int, dict[str, Any]]]:
    if bars.empty:
        return {}
    required = {"symbol", "bar_time", "open", "high", "low", "close", "adjusted_open", "adjusted_high", "adjusted_low", "adjusted_close", "session_minute_ordinal"}
    missing = sorted(required.difference(bars.columns))
    if missing:
        raise MinutePortfolioError(f"minute_portfolio_bar_columns_missing:{','.join(missing)}")
    result: dict[str, dict[int, dict[str, Any]]] = defaultdict(dict)
    columns = [
        "symbol",
        "bar_time",
        "open",
        "high",
        "low",
        "close",
        "adjusted_open",
        "adjusted_high",
        "adjusted_low",
        "adjusted_close",
        "session_minute_ordinal",
        "status_known",
        "is_suspended",
        "is_delisted",
    ]
    available = [name for name in columns if name in bars.columns]
    for row in bars.loc[:, available].to_dict("records"):
        ordinal = row.get("session_minute_ordinal")
        if ordinal is None or pd.isna(ordinal):
            continue
        symbol = str(row["symbol"])
        result[symbol][int(ordinal)] = row
    return dict(result)


def _load_portfolio_bars(
    bars_loader: Callable[[Sequence[str], str], pd.DataFrame],
    symbols: Sequence[str],
    date_text: str,
) -> pd.DataFrame:
    """Load a day while treating an explicitly empty symbol batch as no bars."""

    try:
        return bars_loader(symbols, date_text)
    except MinuteStrategyDataError as exc:
        if str(exc) == "strategy_data_target_bars_empty":
            return pd.DataFrame()
        raise


def _next_slot(
    calendar_index: int,
    ordinal: int,
    symbol: str,
    day_maps: Mapping[str, Mapping[int, Mapping[str, Any]]],
    calendar_length: int,
) -> tuple[int, int] | None:
    available = day_maps.get(symbol, {})
    later = sorted(value for value in available if int(value) > int(ordinal))
    if later:
        return calendar_index, int(later[0])
    if calendar_index + 1 < calendar_length:
        return calendar_index + 1, 0
    return None


def _next_exit_slot(
    holding: _Holding,
    *,
    date_idx: int,
    ordinal: int,
    day_maps: Mapping[str, Mapping[int, Mapping[str, Any]]],
    calendar_length: int,
) -> tuple[int, int] | None:
    """Return the earliest legal sell slot after an observed trigger.

    A position bought during a session is not sellable until the next trading
    day.  Once that day has started, a later minute in the same session is a
    legal retry slot (useful when the opening bar is suspended or locked at a
    limit).  The distinction is kept here instead of being implicit in each
    exit rule.
    """

    if int(date_idx) <= int(holding.entry_date_idx):
        next_date = int(holding.entry_date_idx) + 1
        if next_date >= int(calendar_length):
            return None
        return next_date, 0
    return _next_slot(
        int(date_idx),
        int(ordinal),
        holding.symbol,
        day_maps,
        int(calendar_length),
    )


def _schedule_exit(account: _Account, holding: _Holding, date_idx: int, ordinal: int, reason: str) -> None:
    if holding.pending_exit:
        return
    holding.pending_exit = True
    holding.pending_exit_reason = str(reason)
    account.pending_exits[(int(date_idx), int(ordinal))].append((holding.trade_id, str(reason)))


def _can_sell(row: Mapping[str, Any], holding: _Holding) -> tuple[bool, str]:
    if "status_known" in row and pd.notna(row["status_known"]) and not bool(row["status_known"]):
        return False, "status_unknown"
    if "is_suspended" in row and (pd.isna(row["is_suspended"]) or bool(row["is_suspended"])):
        return False, "suspended"
    if "is_delisted" in row and (pd.isna(row["is_delisted"]) or bool(row["is_delisted"])):
        return False, "delisted"
    adjusted_open = float(row.get("adjusted_open", math.nan))
    if not np.isfinite(adjusted_open) or adjusted_open <= 0.0:
        return False, "price_missing"
    high = float(row.get("adjusted_high", math.nan))
    low = float(row.get("adjusted_low", math.nan))
    close = float(row.get("adjusted_close", math.nan))
    if (
        np.isfinite(high)
        and np.isfinite(low)
        and np.isfinite(close)
        and np.isfinite(holding.last_adjusted_close)
        and math.isclose(high, low, rel_tol=0.0, abs_tol=1.0e-12)
        and math.isclose(low, close, rel_tol=0.0, abs_tol=1.0e-12)
        and close < holding.last_adjusted_close * (1.0 - 1.0e-6)
    ):
        return False, "one_price_down"
    return True, "ok"


def _close_holding(
    account: _Account,
    holding: _Holding,
    row: Mapping[str, Any],
    *,
    date_idx: int,
    date_text: str,
    ordinal: int,
    reason: str,
    costs: Any,
    slippage_multiplier: float,
) -> bool:
    sellable, sell_reason = _can_sell(row, holding)
    holding.exit_attempt_count += 1
    if not sellable:
        account.delayed_exit_count += 1
        holding.pending_exit = False
        holding.pending_exit_reason = ""
        account.exit_reason_counts[f"delayed_{sell_reason}"] += 1
        return False
    adjusted_open = float(row["adjusted_open"])
    proceeds, sale = sell_position(
        position=holding.position,
        adjusted_open=adjusted_open,
        trade_date=date_text,
        costs=costs,
        slippage_multiplier=float(slippage_multiplier),
    )
    account.cash += proceeds
    pnl = float(proceeds - holding.position.net_cash_outflow)
    account.trade_rows.append(
        {
            "trade_id": holding.trade_id,
            "selection_ranker": account.selection_ranker,
            "selection_seed": int(account.selection_seed),
            "strategy_id": account.strategy_id,
            "policy_id": account.policy.policy_id,
            "signal_id": holding.signal_id,
            "symbol": holding.symbol,
            "signal_date": holding.signal_date,
            "signal_time": holding.signal_time,
            "entry_date": holding.entry_date,
            "entry_time": holding.entry_time,
            "selection_score": float(holding.selection_score),
            "exit_date": date_text,
            "exit_time": str(row.get("bar_time", "")),
            "exit_reason": reason,
            "holding_trading_days": int(date_idx - holding.entry_date_idx),
            "shares": int(holding.position.shares),
            "gross_entry_notional": float(holding.position.gross_entry_notional),
            "net_cash_outflow": float(holding.position.net_cash_outflow),
            "proceeds": float(proceeds),
            "pnl": pnl,
            "trade_net_return": pnl / float(holding.position.net_cash_outflow),
            "buy_cost": float(holding.position.net_cash_outflow - holding.position.gross_cash_outflow),
            "sell_cost": float(sale.get("total_cost", 0.0)),
            "exit_attempt_count": int(holding.exit_attempt_count),
        }
    )
    account.exit_reason_counts[str(reason)] += 1
    account.holdings.pop(holding.trade_id, None)
    account.holdings_by_symbol.pop(holding.symbol, None)
    return True


def _execute_scheduled_exits(
    account: _Account,
    *,
    date_idx: int,
    date_text: str,
    ordinal: int,
    day_maps: Mapping[str, Mapping[int, Mapping[str, Any]]],
    calendar_length: int,
    costs: Any,
    slippage_multiplier: float,
) -> None:
    pending = account.pending_exits.pop((int(date_idx), int(ordinal)), [])
    for trade_id, reason in pending:
        holding = account.holdings.get(int(trade_id))
        if holding is None:
            continue
        row = day_maps.get(holding.symbol, {}).get(int(ordinal))
        if row is not None and _close_holding(
            account,
            holding,
            row,
            date_idx=date_idx,
            date_text=date_text,
            ordinal=ordinal,
            reason=reason,
            costs=costs,
            slippage_multiplier=slippage_multiplier,
        ):
            continue
        next_slot = _next_slot(
            date_idx,
            ordinal,
            holding.symbol,
            day_maps,
            calendar_length,
        )
        if next_slot is None:
            holding.pending_exit = False
            holding.pending_exit_reason = ""
            account.unresolved_count += 1
            continue
        _schedule_exit(account, holding, next_slot[0], next_slot[1], reason)


def _execute_pending_entries(
    account: _Account,
    *,
    date_idx: int,
    ordinal: int,
    date_text: str,
    costs: Any,
    slippage_multiplier: float,
) -> None:
    pending = account.pending_entries.pop((int(date_idx), int(ordinal)), [])
    if not pending:
        return
    for item in pending:
        account.pending_symbols.discard(str(item.record["symbol"]))
    # All orders at one executable timestamp compete on equal information.
    # Allocate the currently available cash across the remaining slots before
    # filling, so iteration order cannot silently give the first symbol a
    # larger stake than later symbols.
    available_slots = max(0, account.max_positions - len(account.holdings))
    if available_slots <= 0:
        account.skipped_slot_count += len(pending)
        return
    eligible: list[_PendingEntry] = []
    seen_symbols: set[str] = set()
    for item in pending:
        symbol = str(item.record["symbol"])
        if symbol in account.holdings_by_symbol or symbol in seen_symbols:
            account.skipped_duplicate_count += 1
            continue
        if len(eligible) >= available_slots:
            account.skipped_slot_count += 1
            continue
        seen_symbols.add(symbol)
        eligible.append(item)
    if not eligible:
        return
    remaining_slots = available_slots
    slot_budget = account.cash / float(available_slots)
    for item in eligible:
        row = item.record
        symbol = str(row["symbol"])
        allocated_cash = min(
            slot_budget,
            account.cash / float(max(1, remaining_slots)),
        )
        raw_open = float(row["entry_price"])
        adjusted_open = float(row["entry_adjusted_price"])
        position, buy = buy_position(
            available_cash=account.cash,
            allocated_cash=allocated_cash,
            symbol_idx=int(zlib.crc32(symbol.encode("utf-8")) & 0x7FFFFFFF),
            signal_date_idx=int(item.signal_date_idx),
            execution_date_idx=int(date_idx),
            raw_open=raw_open,
            adjusted_open=adjusted_open,
            costs=costs,
            slippage_multiplier=float(slippage_multiplier),
        )
        if position is None:
            account.skipped_cash_count += 1
            continue
        account.cash -= float(position.net_cash_outflow)
        remaining_slots -= 1
        account.trade_id += 1
        entry_time = normalize_bar_time(row["entry_time"])
        holding = _Holding(
            trade_id=account.trade_id,
            symbol=symbol,
            strategy_id=account.strategy_id,
            signal_id=str(row["signal_id"]),
            signal_date=str(row["signal_date"]),
            signal_time=normalize_bar_time(row["signal_time"]),
            entry_date_idx=int(date_idx),
            entry_date=date_text,
            entry_time=entry_time,
            entry_ordinal=int(row["entry_ordinal"]),
            selection_score=float(item.selection_score),
            position=position,
            entry_adjusted_price=adjusted_open,
            peak_adjusted_price=adjusted_open,
            peak_time=entry_time,
        )
        account.holdings[holding.trade_id] = holding
        account.holdings_by_symbol[symbol] = holding.trade_id
        account.filled_entry_count += 1


def _observe_holdings(
    account: _Account,
    *,
    date_idx: int,
    ordinal: int,
    day_maps: Mapping[str, Mapping[int, Mapping[str, Any]]],
    calendar_length: int,
) -> None:
    for holding in list(account.holdings.values()):
        if holding.pending_exit:
            continue
        if int(date_idx) < holding.entry_date_idx:
            continue
        if int(date_idx) == holding.entry_date_idx and int(ordinal) < holding.entry_ordinal:
            continue
        row = day_maps.get(holding.symbol, {}).get(int(ordinal))
        if row is None:
            continue
        adjusted_close = float(row.get("adjusted_close", math.nan))
        adjusted_high = float(row.get("adjusted_high", math.nan))
        adjusted_low = float(row.get("adjusted_low", math.nan))
        if np.isfinite(adjusted_close):
            holding.last_adjusted_close = adjusted_close
        if np.isfinite(adjusted_high) and adjusted_high > holding.peak_adjusted_price:
            holding.peak_adjusted_price = adjusted_high
            holding.peak_time = str(row.get("bar_time", ""))
        policy = account.policy
        reason: str | None = None
        if policy.stop_loss_bps is not None and np.isfinite(adjusted_low):
            stop = holding.entry_adjusted_price * (1.0 - float(policy.stop_loss_bps) / 10_000.0)
            if adjusted_low <= stop:
                reason = "stop_loss"
        if (
            reason is None
            and policy.trail_activation_bps is not None
            and holding.peak_adjusted_price
            >= holding.entry_adjusted_price * (1.0 + float(policy.trail_activation_bps) / 10_000.0)
        ):
            holding.trailing_active = True
        if (
            reason is None
            and holding.trailing_active
            and np.isfinite(adjusted_low)
            and adjusted_low
            <= holding.peak_adjusted_price * (1.0 - float(policy.trail_drawdown_bps) / 10_000.0)
        ):
            reason = "trailing_stop"
        if (
            reason is None
            and policy.take_profit_bps is not None
            and np.isfinite(adjusted_high)
            and adjusted_high
            >= holding.entry_adjusted_price * (1.0 + float(policy.take_profit_bps) / 10_000.0)
        ):
            reason = "take_profit"
        if reason is not None:
            next_slot = _next_exit_slot(
                holding,
                date_idx=date_idx,
                ordinal=ordinal,
                day_maps=day_maps,
                calendar_length=calendar_length,
            )
            if next_slot is not None:
                _schedule_exit(account, holding, next_slot[0], next_slot[1], reason)


def _schedule_due_exits(account: _Account, *, date_idx: int, ordinal: int = 0) -> None:
    for holding in list(account.holdings.values()):
        if holding.pending_exit:
            continue
        if int(date_idx) >= holding.entry_date_idx + int(account.policy.max_hold_days):
            _schedule_exit(account, holding, date_idx, ordinal, "time_stop")


def _mark_account(
    account: _Account,
    *,
    date_idx: int,
    date_text: str,
    day_maps: Mapping[str, Mapping[int, Mapping[str, Any]]],
) -> None:
    equity = float(account.cash)
    for holding in account.holdings.values():
        rows = day_maps.get(holding.symbol, {})
        if rows:
            last_row = rows[max(rows)]
            price = float(last_row.get("adjusted_close", math.nan))
            if np.isfinite(price) and price > 0:
                holding.last_adjusted_close = price
        mark = holding.last_adjusted_close
        if not np.isfinite(mark) or mark <= 0:
            mark = holding.position.last_adjusted_price
        equity += position_value(holding.position, float(mark))
    account.equity_rows.append(
        {
            "selection_ranker": account.selection_ranker,
            "selection_seed": int(account.selection_seed),
            "strategy_id": account.strategy_id,
            "policy_id": account.policy.policy_id,
            "trade_date": date_text,
            "date_idx": int(date_idx),
            "cash": float(account.cash),
            "position_count": int(len(account.holdings)),
            "equity": float(equity),
            "daily_return": (
                float(equity / account.previous_equity - 1.0)
                if account.previous_equity > 0.0
                else 0.0
            ),
        }
    )
    account.previous_equity = float(equity)


def _account_result(account: _Account) -> dict[str, Any]:
    equity = pd.DataFrame(account.equity_rows)
    trades = pd.DataFrame(account.trade_rows)
    if equity.empty:
        ending = float(account.cash)
        drawdown = 0.0
    else:
        equity["equity_peak"] = equity["equity"].cummax()
        equity["drawdown"] = equity["equity"] / equity["equity_peak"] - 1.0
        ending = float(equity["equity"].iloc[-1])
        drawdown = float(equity["drawdown"].min())
    return {
        "selection_ranker": account.selection_ranker,
        "selection_seed": int(account.selection_seed),
        "strategy_id": account.strategy_id,
        "policy_id": account.policy.policy_id,
        "starting_cash": float(account.starting_cash),
        "ending_cash": float(account.cash),
        "ending_equity": ending,
        "net_return": ending / float(account.starting_cash) - 1.0,
        "maximum_drawdown": drawdown,
        "selection_count": int(account.selection_count),
        "accepted_signal_count": int(account.accepted_signal_count),
        "filled_entry_count": int(account.filled_entry_count),
        "skipped_slot_count": int(account.skipped_slot_count),
        "skipped_cash_count": int(account.skipped_cash_count),
        "skipped_duplicate_count": int(account.skipped_duplicate_count),
        "closed_trade_count": int(len(trades)),
        "positive_trade_fraction": (
            float(trades["pnl"].gt(0).mean()) if not trades.empty else None
        ),
        "mean_trade_net_return": (
            float(trades["trade_net_return"].mean()) if not trades.empty else None
        ),
        "delayed_exit_count": int(account.delayed_exit_count),
        "unresolved_position_count": int(len(account.holdings)),
        "unresolved_exit_count": int(account.unresolved_count),
        "unresolved_position_value": float(
            sum(
                position_value(holding.position, holding.last_adjusted_close)
                for holding in account.holdings.values()
                if np.isfinite(holding.last_adjusted_close) and holding.last_adjusted_close > 0
            )
        ),
        "exit_reason_counts": dict(account.exit_reason_counts),
        "annual": (
            [
                {
                    "year": int(year),
                    "net_return": float((1.0 + group["daily_return"]).prod() - 1.0),
                    "maximum_drawdown": float(group["drawdown"].min()),
                }
                for year, group in equity.assign(
                    year=equity["trade_date"].astype(str).str[:4].astype(int)
                ).groupby("year", sort=True)
            ]
            if not equity.empty
            else []
        ),
    }


def simulate_portfolio_accounts(
    signal_frames_by_date: Mapping[str, pd.DataFrame] | Callable[[str], pd.DataFrame | None],
    *,
    bars_loader: Callable[[Sequence[str], str], pd.DataFrame],
    calendar: Sequence[str],
    strategy_ids: Sequence[str],
    policies: Sequence[ExitPolicy],
    config: PortfolioConfig | None = None,
    selection_seeds: Sequence[int] | None = None,
    selection_rankers: Sequence[str] | None = None,
) -> tuple[list[dict[str, Any]], pd.DataFrame, pd.DataFrame]:
    """Replay all requested strategy/policy accounts over a shared bar stream."""

    selected = config or PortfolioConfig()
    selected.validate()
    for policy in policies:
        policy.validate()
    dates = [str(value) for value in calendar]
    if not dates or len(set(dates)) != len(dates) or dates != sorted(dates):
        raise MinutePortfolioError("minute_portfolio_calendar_invalid")
    strategy_ids = tuple(dict.fromkeys(str(value) for value in strategy_ids))
    if not strategy_ids or any(not value.strip() for value in strategy_ids):
        raise MinutePortfolioError("minute_portfolio_strategy_ids_empty")
    policies = tuple(policies)
    if not policies or len({policy.policy_id for policy in policies}) != len(policies):
        raise MinutePortfolioError("minute_portfolio_policies_invalid")
    raw_seeds = tuple(
        selection_seeds if selection_seeds is not None else (selected.selection_seed,)
    )
    if not raw_seeds or any(
        isinstance(value, (bool, np.bool_))
        or not isinstance(value, (int, np.integer))
        for value in raw_seeds
    ):
        raise MinutePortfolioError("minute_portfolio_selection_seeds_invalid")
    seeds = tuple(dict.fromkeys(int(value) for value in raw_seeds))
    rankers = tuple(
        dict.fromkeys(
            str(value)
            for value in (
                selection_rankers
                if selection_rankers is not None
                else (selected.selection_ranker,)
            )
        )
    )
    if not rankers or any(value not in SELECTION_RANKERS for value in rankers):
        raise MinutePortfolioError("minute_portfolio_selection_rankers_invalid")
    date_index = {value: index for index, value in enumerate(dates)}
    accounts = {
        (ranker, seed, str(strategy_id), policy.policy_id): _Account(
            selection_ranker=ranker,
            selection_seed=int(seed),
            strategy_id=str(strategy_id),
            policy=policy,
            starting_cash=float(selected.starting_cash),
            max_positions=int(selected.max_positions),
            cash=float(selected.starting_cash),
            previous_equity=float(selected.starting_cash),
        )
        for ranker in rankers
        for seed in seeds
        for strategy_id in strategy_ids
        for policy in policies
    }
    costs = default_minute_costs()
    all_equity: list[pd.DataFrame] = []
    all_trades: list[pd.DataFrame] = []
    for date_idx, date_text in enumerate(dates):
        _memory_check(selected, f"date_start:{date_text}")
        raw_signals = (
            signal_frames_by_date(date_text)
            if callable(signal_frames_by_date)
            else signal_frames_by_date.get(date_text)
        )
        grouped_signals: dict[str, dict[int, list[dict[str, Any]]]] = {}
        if raw_signals is not None and not raw_signals.empty:
            for strategy_id in strategy_ids:
                normalised = _normalise_signal_frame(raw_signals, str(strategy_id))
                grouped_signals[str(strategy_id)] = normalised
        active_symbols = {
            holding.symbol
            for account in accounts.values()
            for holding in account.holdings.values()
        }
        day_maps = _bar_maps(
            _load_portfolio_bars(bars_loader, sorted(active_symbols), date_text)
            if active_symbols
            else pd.DataFrame()
        )
        loaded_symbols = set(day_maps)
        for account in accounts.values():
            _schedule_due_exits(account, date_idx=date_idx, ordinal=0)
        for ordinal in range(240):
            for account in accounts.values():
                _execute_scheduled_exits(
                    account,
                    date_idx=date_idx,
                    date_text=date_text,
                    ordinal=ordinal,
                    day_maps=day_maps,
                    calendar_length=len(dates),
                    costs=costs,
                    slippage_multiplier=selected.slippage_multiplier,
                )
                _execute_pending_entries(
                    account,
                    date_idx=date_idx,
                    ordinal=ordinal,
                    date_text=date_text,
                    costs=costs,
                    slippage_multiplier=selected.slippage_multiplier,
                )
            # Entries are filled from the signal's declared next-open price,
            # but their same-session high/low/close must still be observed for
            # peak tracking and end-of-day marking.  Load newly opened symbols
            # once, immediately after their fill and before observation.
            newly_opened = sorted(
                {
                    holding.symbol
                    for account in accounts.values()
                    for holding in account.holdings.values()
                    if holding.symbol not in loaded_symbols
                }
            )
            if newly_opened:
                _memory_check(selected, f"same_day_observation_load:{date_text}", soft=True)
                new_maps = _bar_maps(
                    _load_portfolio_bars(bars_loader, newly_opened, date_text)
                )
                day_maps.update(new_maps)
                loaded_symbols.update(newly_opened)
                del new_maps
            for account in accounts.values():
                _observe_holdings(
                    account,
                    date_idx=date_idx,
                    ordinal=ordinal,
                    day_maps=day_maps,
                    calendar_length=len(dates),
                )
            for strategy_id in strategy_ids:
                records = grouped_signals.get(str(strategy_id), {}).get(ordinal, [])
                if not records:
                    continue
                for ranker in rankers:
                    for seed in seeds:
                        ranked = _rank_same_minute_records(
                            records,
                            ranker=ranker,
                            seed=seed,
                        )
                        for policy in policies:
                            account = accounts[
                                (ranker, seed, str(strategy_id), policy.policy_id)
                            ]
                            for row, selection_score in ranked:
                                account.selection_count += 1
                                symbol = str(row["symbol"])
                                if symbol in account.holdings_by_symbol:
                                    account.skipped_duplicate_count += 1
                                    continue
                                if symbol in account.pending_symbols:
                                    account.skipped_duplicate_count += 1
                                    continue
                                entry_date = str(row.get("entry_date") or date_text)
                                entry_idx = date_index.get(entry_date)
                                entry_ordinal = row.get("entry_ordinal")
                                if entry_idx is None or entry_ordinal is None:
                                    continue
                                if int(entry_idx) < date_idx or (
                                    int(entry_idx) == date_idx
                                    and int(entry_ordinal) <= ordinal
                                ):
                                    continue
                                if (
                                    len(account.holdings)
                                    + len(account.pending_symbols)
                                    >= account.max_positions
                                ):
                                    account.skipped_slot_count += 1
                                    continue
                                account.accepted_signal_count += 1
                                account.pending_entries[
                                    (int(entry_idx), int(entry_ordinal))
                                ].append(
                                    _PendingEntry(
                                        record=row,
                                        signal_date_idx=date_idx,
                                        selection_score=selection_score,
                                    )
                                )
                                account.pending_symbols.add(symbol)
            # The next bar's open is the earliest causal fill/exit point; no
            # same-day exit is allowed for a position entered today.
        # A symbol bought during this session was not part of ``active_symbols``
        # when the day was loaded.  Fetch only those newly opened symbols for
        # the end-of-day mark so the equity path reflects the observed close
        # without loading every candidate signal's minute history.
        marked_symbols = set(day_maps)
        new_holding_symbols = sorted(
            {
                holding.symbol
                for account in accounts.values()
                for holding in account.holdings.values()
                if holding.symbol not in marked_symbols
            }
        )
        if new_holding_symbols:
            _memory_check(selected, f"end_mark_load_start:{date_text}", soft=True)
            extra_maps = _bar_maps(
                _load_portfolio_bars(bars_loader, new_holding_symbols, date_text)
            )
            for symbol, rows in extra_maps.items():
                day_maps[symbol] = rows
            del extra_maps
        for account in accounts.values():
            _mark_account(
                account,
                date_idx=date_idx,
                date_text=date_text,
                day_maps=day_maps,
            )
        del raw_signals, day_maps, grouped_signals
        gc.collect()
        _memory_check(selected, f"date_complete:{date_text}")
    for account in accounts.values():
        if account.equity_rows:
            all_equity.append(pd.DataFrame(account.equity_rows))
        if account.trade_rows:
            all_trades.append(pd.DataFrame(account.trade_rows))
    results = [_account_result(account) for account in accounts.values()]
    equity = pd.concat(all_equity, ignore_index=True) if all_equity else pd.DataFrame()
    trades = pd.concat(all_trades, ignore_index=True) if all_trades else pd.DataFrame()
    return results, equity, trades


def summarize_selection_seed_results(results: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Summarize account sensitivity without selecting a winning threshold."""

    if not results:
        raise MinutePortfolioError("minute_portfolio_seed_results_empty")
    required = {
        "selection_ranker",
        "selection_seed",
        "strategy_id",
        "policy_id",
        "net_return",
        "maximum_drawdown",
        "unresolved_position_count",
        "annual",
    }
    grouped: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    seen: set[tuple[str, int, str, str]] = set()
    all_rankers: set[str] = set()
    all_seeds: set[int] = set()
    all_years: set[int] = set()
    for raw in results:
        missing = sorted(required.difference(raw))
        if missing:
            raise MinutePortfolioError(
                f"minute_portfolio_seed_result_columns_missing:{','.join(missing)}"
            )
        seed = int(raw["selection_seed"])
        ranker = str(raw["selection_ranker"])
        strategy_id = str(raw["strategy_id"])
        policy_id = str(raw["policy_id"])
        key = (ranker, seed, strategy_id, policy_id)
        if key in seen:
            raise MinutePortfolioError("minute_portfolio_seed_result_duplicate")
        seen.add(key)
        annual = {
            int(item["year"]): float(item["net_return"])
            for item in raw.get("annual", [])
        }
        all_rankers.add(ranker)
        all_seeds.add(seed)
        all_years.update(annual)
        grouped[(ranker, strategy_id, policy_id)].append(
            {
                "selection_seed": seed,
                "net_return": float(raw["net_return"]),
                "maximum_drawdown": float(raw["maximum_drawdown"]),
                "resolved": int(raw["unresolved_position_count"]) == 0,
                "annual": annual,
            }
        )

    seeds = sorted(all_seeds)
    years = sorted(all_years)

    def distribution(values: Sequence[float]) -> dict[str, float]:
        array = np.asarray(values, dtype=np.float64)
        return {
            "mean": float(array.mean()),
            "median": float(np.median(array)),
            "standard_deviation": float(array.std(ddof=0)),
            "minimum": float(array.min()),
            "q25": float(np.quantile(array, 0.25)),
            "q75": float(np.quantile(array, 0.75)),
            "maximum": float(array.max()),
        }

    summaries: list[dict[str, Any]] = []
    for (ranker, strategy_id, policy_id), rows in grouped.items():
        rows.sort(key=lambda item: int(item["selection_seed"]))
        if [int(item["selection_seed"]) for item in rows] != seeds:
            raise MinutePortfolioError(
                f"minute_portfolio_seed_coverage_incomplete:{ranker}:{strategy_id}:{policy_id}"
            )
        resolved = [item for item in rows if bool(item["resolved"])]
        positive = [item for item in rows if float(item["net_return"]) > 0.0]
        resolved_positive = [
            item for item in resolved if float(item["net_return"]) > 0.0
        ]
        all_periods_positive = [
            item
            for item in rows
            if years
            and all(
                year in item["annual"] and float(item["annual"][year]) > 0.0
                for year in years
            )
        ]
        resolved_all_periods_positive = [
            item for item in all_periods_positive if bool(item["resolved"])
        ]
        annual_summaries = []
        for year in years:
            year_values = [
                float(item["annual"][year])
                for item in rows
                if year in item["annual"]
            ]
            annual_summaries.append(
                {
                    "year": int(year),
                    "observed_seed_count": len(year_values),
                    "positive_seed_count": sum(value > 0.0 for value in year_values),
                    "positive_seed_fraction": (
                        sum(value > 0.0 for value in year_values) / len(year_values)
                        if year_values
                        else None
                    ),
                    "net_return": distribution(year_values) if year_values else None,
                }
            )
        summaries.append(
            {
                "selection_ranker": ranker,
                "strategy_id": strategy_id,
                "policy_id": policy_id,
                "seed_count": len(rows),
                "resolved_seed_count": len(resolved),
                "positive_seed_count": len(positive),
                "positive_seed_fraction": len(positive) / len(rows),
                "resolved_positive_seed_count": len(resolved_positive),
                "resolved_positive_seed_fraction": (
                    len(resolved_positive) / len(resolved) if resolved else None
                ),
                "all_periods_positive_seed_count": len(all_periods_positive),
                "resolved_all_periods_positive_seed_count": len(
                    resolved_all_periods_positive
                ),
                "combined_net_return": distribution(
                    [float(item["net_return"]) for item in rows]
                ),
                "maximum_drawdown": distribution(
                    [float(item["maximum_drawdown"]) for item in rows]
                ),
                "annual": annual_summaries,
            }
        )
    summaries.sort(
        key=lambda item: (
            -float(item["combined_net_return"]["median"]),
            str(item["selection_ranker"]),
            str(item["strategy_id"]),
            str(item["policy_id"]),
        )
    )
    return {
        "schema": "quantlab.minute_strategy_selection_seed_summary/1",
        "selection_rankers": sorted(all_rankers),
        "selection_seeds": seeds,
        "seed_count": len(seeds),
        "years": years,
        "account_variant_count": len(summaries),
        "account_result_count": len(results),
        "accounts": summaries,
    }


def _signal_paths(output_roots: Sequence[str | Path]) -> dict[str, Path]:
    paths: dict[str, Path] = {}
    for root in output_roots:
        for path in sorted((Path(root).resolve() / "outcomes").glob("date=*/outcomes.parquet")):
            date_text = path.parent.name.split("=", 1)[-1]
            if date_text in paths:
                raise MinutePortfolioError(f"minute_portfolio_duplicate_signal_date:{date_text}")
            paths[date_text] = path
    if not paths:
        raise MinutePortfolioError("minute_portfolio_signal_paths_empty")
    return paths


def _read_signal_file(path: Path, strategy_ids: Sequence[str]) -> pd.DataFrame:
    try:
        import pyarrow.parquet as pq

        available_columns = set(pq.ParquetFile(path).schema.names)
    except (ImportError, OSError, ValueError) as exc:
        raise MinutePortfolioError(f"minute_portfolio_signal_schema_unreadable:{path}") from exc
    columns = [name for name in SIGNAL_COLUMNS if name in available_columns]
    # The schema probe lets older pilot outputs remain readable when optional
    # fields differ, without loading the complete file a second time.
    frame = pd.read_parquet(path, columns=columns)
    frame = frame.loc[frame["strategy_id"].astype(str).isin([str(x) for x in strategy_ids])].copy()
    return frame


def _infer_strategy_ids(paths: Mapping[str, Path]) -> tuple[str, ...]:
    """Union executable strategy ids across every supplied signal partition."""

    found: set[str] = set()
    for path in paths.values():
        try:
            import pyarrow.parquet as pq

            available = set(pq.ParquetFile(path).schema.names)
        except (ImportError, OSError, ValueError) as exc:
            raise MinutePortfolioError(f"minute_portfolio_signal_schema_unreadable:{path}") from exc
        if "strategy_id" not in available:
            raise MinutePortfolioError(f"minute_portfolio_signal_columns_missing:strategy_id:{path}")
        columns = ["strategy_id"]
        if "diagnostic_only" in available:
            columns.append("diagnostic_only")
        frame = pd.read_parquet(path, columns=columns)
        diagnostic = (
            frame["diagnostic_only"].fillna(False).astype(bool)
            if "diagnostic_only" in frame
            else pd.Series(False, index=frame.index)
        )
        found.update(
            str(value)
            for value in frame.loc[~diagnostic, "strategy_id"].dropna().unique()
            if str(value) not in CONTROL_STRATEGIES
        )
    return tuple(sorted(found))


def run_portfolio_exit_study(
    workspace_root: str | Path,
    output_roots: Sequence[str | Path],
    *,
    output_root: str | Path,
    strategy_ids: Sequence[str] | None = None,
    policy_ids: Sequence[str] | None = None,
    config: PortfolioConfig | None = None,
    data_config: StrategyDataConfig | None = None,
    selection_seeds: Sequence[int] | None = None,
    selection_rankers: Sequence[str] | None = None,
) -> dict[str, Any]:
    """Run the finite-cash exit comparison over existing event outputs."""

    selected = config or PortfolioConfig()
    selected.validate()
    roots = [Path(value).resolve() for value in output_roots]
    paths = _signal_paths(roots)
    if strategy_ids is None:
        strategy_ids = _infer_strategy_ids(paths)
    strategy_ids = tuple(str(value) for value in strategy_ids if str(value) not in CONTROL_STRATEGIES)
    if not strategy_ids:
        raise MinutePortfolioError("minute_portfolio_strategy_ids_empty")
    catalog = {policy.policy_id: policy for policy in exit_policy_catalog()}
    requested_policy_ids = tuple(str(value) for value in (policy_ids or tuple(catalog)))
    unknown_policies = sorted(set(requested_policy_ids).difference(catalog))
    if unknown_policies:
        raise MinutePortfolioError(f"minute_portfolio_unknown_policy:{','.join(unknown_policies)}")
    policies = tuple(catalog[value] for value in requested_policy_ids)
    if not policies:
        raise MinutePortfolioError("minute_portfolio_policy_ids_empty")
    min_date = min(paths)
    max_date = max(paths)
    max_hold = max(policy.max_hold_days for policy in policies)
    calendar_end = (pd.Timestamp(max_date) + pd.Timedelta(days=max(14, max_hold * 3))).strftime("%Y-%m-%d")
    selected_data = data_config or StrategyDataConfig(
        duckdb_threads=2,
        memory_floor_gib=selected.memory_floor_gib,
        duckdb_memory_floor_gib=2.0,
    )
    calendar = load_trading_calendar(
        workspace_root,
        start_date=min_date,
        end_date=calendar_end,
        config=selected_data,
    )["trade_date"].astype(str).tolist()
    signal_dates = set(paths)
    calendar = [value for value in calendar if value >= min_date]
    bars_config = selected_data

    def signal_loader(date_text: str) -> pd.DataFrame | None:
        path = paths.get(str(date_text))
        if path is None:
            return None
        _memory_check(selected, f"signal_load_start:{date_text}", soft=True)
        return _read_signal_file(path, strategy_ids)

    def bars_loader(symbols: Sequence[str], date_text: str) -> pd.DataFrame:
        if not symbols:
            return pd.DataFrame()
        return load_target_bars(
            workspace_root,
            symbols=symbols,
            trade_dates=[date_text],
            daily_context=None,
            require_complete_session=False,
            config=bars_config,
        )

    started = time.perf_counter()
    results, equity, trades = simulate_portfolio_accounts(
        signal_loader,
        bars_loader=bars_loader,
        calendar=calendar,
        strategy_ids=strategy_ids,
        policies=policies,
        config=selected,
        selection_seeds=selection_seeds,
        selection_rankers=selection_rankers,
    )
    out = Path(output_root).resolve()
    out.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(results).to_json(out / "portfolio_summary.json", orient="records", force_ascii=False, indent=2)
    if not equity.empty:
        equity.to_parquet(out / "equity.parquet", index=False)
    if not trades.empty:
        trades.to_parquet(out / "trades.parquet", index=False)
    seed_summary = summarize_selection_seed_results(results)
    seed_summary_path = out / "selection_seed_summary.json"
    seed_summary_path.write_text(
        json.dumps(seed_summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    payload = {
        "schema": "quantlab.minute_strategy_portfolio_exit_study/1",
        "status": "ok",
        "signal_output_roots": [str(value) for value in roots],
        "signal_date_count": len(signal_dates),
        "calendar_start": calendar[0] if calendar else None,
        "calendar_end": calendar[-1] if calendar else None,
        "strategy_ids": list(strategy_ids),
        "policies": [policy.as_dict() for policy in policies],
        "portfolio_config": selected.as_dict(),
        "selection_rankers": sorted(
            {str(result["selection_ranker"]) for result in results}
        ),
        "selection_seeds": sorted(
            {int(result["selection_seed"]) for result in results}
        ),
        "selection_seed_summary_path": str(seed_summary_path),
        "data_config": selected_data.as_dict(),
        "elapsed_seconds": round(time.perf_counter() - started, 3),
        "results": results,
        "output_root": str(out),
    }
    (out / "run_manifest.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return payload


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="quantlab research minute-strategy-portfolio")
    parser.add_argument("--workspace-root", default=".")
    parser.add_argument(
        "--signal-root",
        action="append",
        dest="signal_roots",
        required=True,
        help="A minute-strategy event output root; repeat for additional months.",
    )
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--strategy-id", action="append", dest="strategy_ids")
    parser.add_argument("--policy-id", action="append", dest="policy_ids")
    parser.add_argument("--starting-cash", type=float, default=100_000.0)
    parser.add_argument("--max-positions", type=int, default=5)
    parser.add_argument("--slippage-multiplier", type=float, default=1.0)
    parser.add_argument(
        "--selection-ranker",
        action="append",
        dest="selection_rankers",
        choices=SELECTION_RANKERS,
        help="Causal same-minute ranking rule; repeat to compare rankers.",
    )
    parser.add_argument(
        "--selection-seed",
        type=int,
        action="append",
        dest="selection_seeds",
        help="Deterministic same-minute tie-break seed; repeat to study stability.",
    )
    parser.add_argument("--memory-floor-gib", type=float, default=0.5)
    parser.add_argument("--soft-memory-floor-gib", type=float, default=1.0)
    parser.add_argument("--duckdb-threads", default="2")
    parser.add_argument("--duckdb-memory-floor-gib", type=float, default=2.0)
    parser.add_argument("--temp-directory", default="")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(list(sys.argv[1:] if argv is None else argv))
    duckdb_threads: int | str = (
        args.duckdb_threads
        if str(args.duckdb_threads).strip().lower() == "auto"
        else int(args.duckdb_threads)
    )
    portfolio_config = PortfolioConfig(
        starting_cash=args.starting_cash,
        max_positions=args.max_positions,
        slippage_multiplier=args.slippage_multiplier,
        selection_ranker=(args.selection_rankers or ["random_hash"])[0],
        selection_seed=(args.selection_seeds or [7])[0],
        memory_floor_gib=args.memory_floor_gib,
        soft_memory_floor_gib=args.soft_memory_floor_gib,
    )
    data_config = StrategyDataConfig(
        duckdb_threads=duckdb_threads,
        temp_directory=args.temp_directory or None,
        memory_floor_gib=args.memory_floor_gib,
        duckdb_memory_floor_gib=args.duckdb_memory_floor_gib,
    )
    result = run_portfolio_exit_study(
        args.workspace_root,
        args.signal_roots,
        output_root=args.output_root,
        strategy_ids=args.strategy_ids,
        policy_ids=args.policy_ids,
        config=portfolio_config,
        data_config=data_config,
        selection_seeds=args.selection_seeds,
        selection_rankers=args.selection_rankers,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


__all__ = [
    "CONTROL_STRATEGIES",
    "ExitPolicy",
    "MinutePortfolioError",
    "PortfolioConfig",
    "exit_policy_catalog",
    "main",
    "run_portfolio_exit_study",
    "simulate_portfolio_accounts",
    "summarize_selection_seed_results",
]
