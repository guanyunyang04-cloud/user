"""Causal event replay with finite cash and ordinary-share T+1."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from .contracts import MinuteV2Config, MinuteV2Error


@dataclass(frozen=True)
class EventReplayConfig:
    starting_cash: float = 1_000_000.0
    top_k_per_minute: int = 3
    maximum_open_positions: int = 5
    maximum_new_positions_per_day: int = 3
    score_threshold: float = 0.0
    lot_size: int = 100

    def validate(self) -> None:
        if self.starting_cash <= 0:
            raise MinuteV2Error("minute_v2_replay_starting_cash_invalid")
        if min(self.top_k_per_minute, self.maximum_open_positions, self.maximum_new_positions_per_day) <= 0:
            raise MinuteV2Error("minute_v2_replay_position_limit_invalid")
        if self.lot_size <= 0:
            raise MinuteV2Error("minute_v2_replay_lot_size_invalid")


@dataclass
class _Holding:
    trade_id: int
    symbol: str
    signal_date: str
    signal_time: str
    entry_date: str
    planned_exit_date: str
    actual_exit_date: str
    shares: int
    cash_outflow: float
    gross_notional: float
    score: float
    label_net_return: float


def _required_columns() -> set[str]:
    return {
        "symbol",
        "trade_date",
        "bar_time",
        "score",
        "planned_exit_date",
        "entry_bar_time",
        "entry_price",
        "entry_amount",
        "entry_executable",
        "actual_exit_date",
        "exit_amount",
        "label_net_return",
        "label_observed",
    }


def replay_events(
    scored: pd.DataFrame,
    *,
    research_config: MinuteV2Config | None = None,
    replay_config: EventReplayConfig | None = None,
) -> tuple[dict[str, Any], pd.DataFrame, pd.DataFrame]:
    """Replay the first qualifying signals without outcome-aware replacement.

    The current ledger marks open positions at cost between entry and the fixed
    legal exit window.  It is intended for model comparison; a future broker
    adapter must add independent daily marks and order-state reconciliation.
    """

    research = research_config or MinuteV2Config()
    replay = replay_config or EventReplayConfig()
    research.validate()
    replay.validate()
    missing = sorted(_required_columns().difference(scored.columns))
    if missing:
        raise MinuteV2Error(f"minute_v2_replay_columns_missing:{','.join(missing)}")
    frame = scored.copy()
    frame["symbol"] = frame["symbol"].astype(str)
    frame["trade_date"] = pd.to_datetime(frame["trade_date"], errors="raise").dt.strftime("%Y-%m-%d")
    frame["bar_time"] = frame["bar_time"].astype(str)
    for name in ("planned_exit_date", "actual_exit_date"):
        valid = frame[name].notna()
        frame.loc[valid, name] = pd.to_datetime(
            frame.loc[valid, name], errors="raise"
        ).dt.strftime("%Y-%m-%d")
    frame["score"] = pd.to_numeric(frame["score"], errors="coerce")
    frame = frame.sort_values(
        ["trade_date", "bar_time", "score", "symbol"],
        ascending=[True, True, False, True],
        kind="stable",
    )
    calendar = sorted(
        set(frame["trade_date"].dropna().astype(str))
        | set(frame["actual_exit_date"].dropna().astype(str))
    )
    cash = float(replay.starting_cash)
    holdings: dict[int, _Holding] = {}
    open_symbols: set[str] = set()
    attempted: set[tuple[str, str]] = set()
    trade_rows: list[dict[str, Any]] = []
    daily_rows: list[dict[str, Any]] = []
    unfilled = 0
    overlap = 0
    selected = 0
    next_trade_id = 1
    buy_cost_rate = (
        research.commission_bps + research.transfer_fee_bps + research.slippage_bps
    ) / 10_000.0
    schedule = {
        str(date_value): group
        for date_value, group in frame.groupby("trade_date", sort=True)
    }
    for current_date in calendar:
        due = [holding for holding in holdings.values() if holding.actual_exit_date == current_date]
        for holding in due:
            proceeds = holding.cash_outflow * (1.0 + holding.label_net_return)
            cash += proceeds
            trade_rows.append(
                {
                    "trade_id": holding.trade_id,
                    "symbol": holding.symbol,
                    "signal_date": holding.signal_date,
                    "signal_time": holding.signal_time,
                    "entry_date": holding.entry_date,
                    "planned_exit_date": holding.planned_exit_date,
                    "exit_date": current_date,
                    "shares": holding.shares,
                    "gross_entry_notional": holding.gross_notional,
                    "net_cash_outflow": holding.cash_outflow,
                    "net_exit_proceeds": proceeds,
                    "net_pnl": proceeds - holding.cash_outflow,
                    "trade_net_return": holding.label_net_return,
                    "score": holding.score,
                    "t1_legal": current_date > holding.entry_date,
                }
            )
            open_symbols.discard(holding.symbol)
            holdings.pop(holding.trade_id, None)
        new_positions = 0
        current = schedule.get(current_date)
        if current is not None:
            for _, minute_group in current.groupby("bar_time", sort=True):
                ranked = minute_group.loc[
                    minute_group["score"].notna()
                    & minute_group["score"].ge(float(replay.score_threshold))
                ].head(int(replay.top_k_per_minute))
                for row in ranked.itertuples(index=False):
                    if len(holdings) >= replay.maximum_open_positions:
                        break
                    if new_positions >= replay.maximum_new_positions_per_day:
                        break
                    selected += 1
                    symbol = str(row.symbol)
                    attempt_key = (current_date, symbol)
                    if attempt_key in attempted or symbol in open_symbols:
                        overlap += 1
                        continue
                    attempted.add(attempt_key)
                    if not bool(row.entry_executable):
                        unfilled += 1
                        continue
                    actual_exit = row.actual_exit_date
                    if (
                        pd.isna(row.label_observed)
                        or not bool(row.label_observed)
                        or pd.isna(actual_exit)
                        or not str(actual_exit)
                    ):
                        unfilled += 1
                        continue
                    entry_price = float(row.entry_price)
                    if not math.isfinite(entry_price) or entry_price <= 0:
                        unfilled += 1
                        continue
                    entry_capacity = float(row.entry_amount) * research.maximum_participation_rate
                    exit_capacity = float(row.exit_amount) * research.maximum_participation_rate
                    position_slots = max(1, replay.maximum_open_positions - len(holdings))
                    cash_budget = cash / position_slots
                    gross_budget = min(entry_capacity, exit_capacity, cash_budget / (1.0 + buy_cost_rate))
                    shares = math.floor(gross_budget / entry_price / replay.lot_size) * replay.lot_size
                    if shares <= 0:
                        unfilled += 1
                        continue
                    gross_notional = shares * entry_price
                    cash_outflow = gross_notional * (1.0 + buy_cost_rate)
                    if cash_outflow > cash * (1.0 + 1.0e-12):
                        raise MinuteV2Error("minute_v2_replay_negative_cash_prevented")
                    cash -= cash_outflow
                    label_return = float(row.label_net_return)
                    holding = _Holding(
                        trade_id=next_trade_id,
                        symbol=symbol,
                        signal_date=current_date,
                        signal_time=str(row.bar_time),
                        entry_date=current_date,
                        planned_exit_date=str(row.planned_exit_date),
                        actual_exit_date=str(row.actual_exit_date),
                        shares=shares,
                        cash_outflow=cash_outflow,
                        gross_notional=gross_notional,
                        score=float(row.score),
                        label_net_return=label_return,
                    )
                    holdings[next_trade_id] = holding
                    open_symbols.add(symbol)
                    next_trade_id += 1
                    new_positions += 1
        marked_cost = sum(holding.cash_outflow for holding in holdings.values())
        daily_rows.append(
            {
                "trade_date": current_date,
                "cash": cash,
                "open_position_count": len(holdings),
                "realized_cost_mark_equity": cash + marked_cost,
            }
        )
    trades = pd.DataFrame(trade_rows)
    daily = pd.DataFrame(daily_rows)
    realized_pnl = float(trades["net_pnl"].sum()) if not trades.empty else 0.0
    closed_outflow = float(trades["net_cash_outflow"].sum()) if not trades.empty else 0.0
    result = {
        "starting_cash": float(replay.starting_cash),
        "ending_cash": cash,
        "closed_trade_count": int(len(trades)),
        "open_position_count": int(len(holdings)),
        "selected_event_count": int(selected),
        "unfilled_selection_count": int(unfilled),
        "overlap_skip_count": int(overlap),
        "realized_pnl": realized_pnl,
        "realized_return_on_starting_cash": realized_pnl / replay.starting_cash,
        "closed_capital_weighted_return": realized_pnl / closed_outflow if closed_outflow > 0 else None,
        "positive_trade_fraction": float(trades["net_pnl"].gt(0).mean()) if len(trades) else None,
        "t1_violation_count": int((~trades["t1_legal"]).sum()) if len(trades) else 0,
        "marking_policy": "open positions remain at net entry cost until the fixed legal exit window",
    }
    if result["t1_violation_count"]:
        raise MinuteV2Error("minute_v2_replay_t1_violation")
    if cash < -1.0e-6 or not np.isfinite(cash):
        raise MinuteV2Error("minute_v2_replay_cash_invalid")
    return result, daily, trades


__all__ = ["EventReplayConfig", "replay_events"]
