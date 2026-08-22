"""Finite-cash minute-window replay with T+1 and volume capacity."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from quantlab.research.minute_panel import MinuteResearchError
from quantlab.research.portfolio import (
    ExecutionCosts,
    Position,
    buy_position,
    position_value,
    sell_position,
)


@dataclass
class MinuteHolding:
    trade_id: int
    symbol: str
    entry_date: str
    planned_exit_date: str
    position: Position
    buy: dict[str, float]
    entry_window_amount: float
    entry_window_volume: float
    remaining_fraction: float = 1.0
    realized_proceeds: float = 0.0
    exit_fill_count: int = 0
    first_exit_date: str = ""
    last_exit_date: str = ""
    sell_cost: float = 0.0


@dataclass
class MinuteAccountState:
    cash: float
    previous_equity: float
    holdings: dict[int, MinuteHolding] = field(default_factory=dict)
    open_symbols: set[str] = field(default_factory=set)
    equity_rows: list[dict[str, Any]] = field(default_factory=list)
    trade_rows: list[dict[str, Any]] = field(default_factory=list)
    fill_rows: list[dict[str, Any]] = field(default_factory=list)
    entry_count: int = 0
    selection_count: int = 0
    unfilled_selection_count: int = 0
    overlap_skip_count: int = 0


def default_minute_costs() -> ExecutionCosts:
    return ExecutionCosts(
        lot_size=100,
        commission_bps=3.0,
        minimum_commission_cny=5.0,
        transfer_fee_bps=0.1,
        slippage_bps=7.0,
        stress_slippage_multiplier=2.0,
        stamp_tax_schedule=(("1900-01-01", 10.0), ("2023-08-28", 5.0)),
    )


def _window_lookup(windows: pd.DataFrame) -> dict[tuple[str, str], dict[str, Any]]:
    required = {
        "symbol",
        "trade_date",
        "window_price",
        "adjusted_price",
        "window_volume",
        "window_amount",
        "exit_executable",
    }
    missing = sorted(required.difference(windows.columns))
    if missing:
        raise MinuteResearchError(f"minute_window_columns_missing:{','.join(missing)}")
    if windows.duplicated(["symbol", "trade_date"]).any():
        raise MinuteResearchError("minute_windows_duplicate_keys")
    return {
        (str(row.trade_date), str(row.symbol)): row._asdict()
        for row in windows.itertuples(index=False)
    }


def _partial_position(holding: MinuteHolding, fraction: float) -> Position:
    original = holding.position
    scale = holding.remaining_fraction * float(fraction)
    return Position(
        symbol_idx=original.symbol_idx,
        shares=max(1, int(round(original.shares * scale))),
        entry_signal_date_idx=original.entry_signal_date_idx,
        entry_date_idx=original.entry_date_idx,
        entry_raw_open=original.entry_raw_open,
        entry_adjusted_open=original.entry_adjusted_open,
        gross_entry_notional=original.gross_entry_notional * scale,
        net_cash_outflow=original.net_cash_outflow * scale,
        gross_cash_outflow=original.gross_cash_outflow * scale,
        last_adjusted_price=original.last_adjusted_price,
    )


def _credited_adjusted_price(
    holding: MinuteHolding,
    actual_price: float,
    maximum_credited_gross_return: float | None,
) -> float:
    if maximum_credited_gross_return is None:
        return float(actual_price)
    cap = holding.position.entry_adjusted_open * (1.0 + float(maximum_credited_gross_return))
    return float(min(actual_price, cap))


def _sell_capacity_fraction(
    holding: MinuteHolding,
    window: dict[str, Any],
    *,
    maximum_participation_rate: float,
    costs: ExecutionCosts,
) -> tuple[float, float]:
    adjusted_price = float(window["adjusted_price"])
    raw_price = float(window["window_price"])
    full_value = position_value(holding.position, adjusted_price) * holding.remaining_fraction
    capacity_lots = math.floor(
        float(window["window_volume"]) * float(maximum_participation_rate) / costs.lot_size
    )
    capacity_value = float(capacity_lots * costs.lot_size) * raw_price
    if full_value <= 0.0 or capacity_value <= 0.0:
        return 0.0, full_value
    if full_value <= capacity_value * (1.0 + 1.0e-12):
        return 1.0, full_value
    return float(capacity_value / full_value), full_value


def _close_trade(state: MinuteAccountState, holding: MinuteHolding) -> None:
    cash_outflow = float(holding.position.net_cash_outflow)
    proceeds = float(holding.realized_proceeds)
    state.trade_rows.append(
        {
            "trade_id": holding.trade_id,
            "symbol": holding.symbol,
            "entry_date": holding.entry_date,
            "planned_exit_date": holding.planned_exit_date,
            "first_exit_date": holding.first_exit_date,
            "exit_date": holding.last_exit_date,
            "gross_entry_notional": float(holding.position.gross_entry_notional),
            "net_cash_outflow": cash_outflow,
            "exit_proceeds": proceeds,
            "pnl": proceeds - cash_outflow,
            "trade_net_return": proceeds / cash_outflow - 1.0,
            "entry_window_amount": holding.entry_window_amount,
            "entry_window_volume": holding.entry_window_volume,
            "entry_participation_rate": (
                holding.position.gross_entry_notional / holding.entry_window_amount
            ),
            "exit_fill_count": holding.exit_fill_count,
            "partial_exit": holding.exit_fill_count > 1,
            "delayed_exit": holding.last_exit_date > holding.planned_exit_date,
            "buy_cost": float(holding.buy.get("total_cost", 0.0)),
            "sell_cost": holding.sell_cost,
        }
    )
    state.open_symbols.discard(holding.symbol)
    state.holdings.pop(holding.trade_id, None)


def _attempt_exit(
    state: MinuteAccountState,
    holding: MinuteHolding,
    *,
    date: str,
    window: dict[str, Any] | None,
    maximum_participation_rate: float,
    maximum_credited_gross_return: float | None,
    costs: ExecutionCosts,
    slippage_multiplier: float,
) -> None:
    if window is None or not bool(window["exit_executable"]):
        return
    fraction, full_value = _sell_capacity_fraction(
        holding,
        window,
        maximum_participation_rate=maximum_participation_rate,
        costs=costs,
    )
    if fraction <= 0.0:
        return
    partial = _partial_position(holding, fraction)
    credited_price = _credited_adjusted_price(
        holding,
        float(window["adjusted_price"]),
        maximum_credited_gross_return,
    )
    proceeds, sell = sell_position(
        position=partial,
        adjusted_open=credited_price,
        trade_date=date,
        costs=costs,
        slippage_multiplier=float(slippage_multiplier),
    )
    sold_remaining_fraction = holding.remaining_fraction * fraction
    holding.remaining_fraction = max(0.0, holding.remaining_fraction - sold_remaining_fraction)
    if holding.remaining_fraction < 1.0e-10:
        holding.remaining_fraction = 0.0
    holding.realized_proceeds += proceeds
    holding.sell_cost += float(sell["total_cost"])
    holding.exit_fill_count += 1
    holding.first_exit_date = holding.first_exit_date or date
    holding.last_exit_date = date
    state.cash += proceeds
    participation = float(sell["gross_notional"] / float(window["window_amount"]))
    state.fill_rows.append(
        {
            "trade_id": holding.trade_id,
            "symbol": holding.symbol,
            "fill_date": date,
            "planned_exit_date": holding.planned_exit_date,
            "gross_position_value_before": full_value,
            "gross_sold_value": float(sell["gross_notional"]),
            "window_amount": float(window["window_amount"]),
            "window_volume": float(window["window_volume"]),
            "participation_rate": participation,
            "remaining_fraction": holding.remaining_fraction,
            **{f"sell_{key}": float(value) for key, value in sell.items()},
        }
    )
    if participation > float(maximum_participation_rate) + 1.0e-9:
        raise MinuteResearchError("minute_exit_capacity_contract_breached")
    if holding.remaining_fraction == 0.0:
        _close_trade(state, holding)


def _process_exits(
    state: MinuteAccountState,
    *,
    date: str,
    windows: dict[tuple[str, str], dict[str, Any]],
    maximum_participation_rate: float,
    maximum_credited_gross_return: float | None,
    costs: ExecutionCosts,
    slippage_multiplier: float,
) -> None:
    due = [holding for holding in state.holdings.values() if holding.planned_exit_date <= date]
    for holding in due:
        _attempt_exit(
            state,
            holding,
            date=date,
            window=windows.get((date, holding.symbol)),
            maximum_participation_rate=maximum_participation_rate,
            maximum_credited_gross_return=maximum_credited_gross_return,
            costs=costs,
            slippage_multiplier=slippage_multiplier,
        )


def _open_orders(
    state: MinuteAccountState,
    current: pd.DataFrame,
    *,
    date: str,
    date_index: int,
    top_k: int,
    maximum_participation_rate: float,
    costs: ExecutionCosts,
    slippage_multiplier: float,
) -> None:
    ranked = current.sort_values(["prediction", "symbol"], ascending=[False, True], kind="stable").head(int(top_k))
    state.selection_count += len(ranked)
    per_order_budget = state.cash / max(int(top_k), 1)
    for row in ranked.itertuples(index=False):
        symbol = str(row.symbol)
        if not bool(row.entry_filled):
            state.unfilled_selection_count += 1
            continue
        if symbol in state.open_symbols:
            state.overlap_skip_count += 1
            continue
        capacity_lots = math.floor(
            float(row.entry_window_volume) * float(maximum_participation_rate) / costs.lot_size
        )
        capacity_shares = capacity_lots * costs.lot_size
        fill_price = float(row.entry_price) * (
            1.0 + costs.slippage_bps * float(slippage_multiplier) / 10_000.0
        )
        capacity_fill_notional = capacity_shares * fill_price
        capacity_cash = (
            capacity_fill_notional
            + max(
                costs.minimum_commission_cny,
                capacity_fill_notional * costs.commission_bps / 10_000.0,
            )
            + capacity_fill_notional * costs.transfer_fee_bps / 10_000.0
        )
        position, buy = buy_position(
            available_cash=state.cash,
            allocated_cash=min(per_order_budget, capacity_cash),
            symbol_idx=0,
            signal_date_idx=date_index,
            execution_date_idx=date_index,
            raw_open=float(row.entry_price),
            adjusted_open=float(row.entry_adjusted_price),
            costs=costs,
            slippage_multiplier=float(slippage_multiplier),
        )
        if position is None:
            state.unfilled_selection_count += 1
            continue
        state.cash -= position.net_cash_outflow
        if state.cash < -1.0e-6:
            raise MinuteResearchError("minute_account_cash_became_negative")
        state.cash = max(state.cash, 0.0)
        state.entry_count += 1
        holding = MinuteHolding(
            trade_id=state.entry_count,
            symbol=symbol,
            entry_date=date,
            planned_exit_date=str(row.planned_exit_date),
            position=position,
            buy={str(key): float(value) for key, value in buy.items()},
            entry_window_amount=float(row.entry_window_amount),
            entry_window_volume=float(row.entry_window_volume),
        )
        state.holdings[holding.trade_id] = holding
        state.open_symbols.add(symbol)


def _holding_value(
    holding: MinuteHolding,
    *,
    date: str,
    windows: dict[tuple[str, str], dict[str, Any]],
) -> float:
    current = windows.get((date, holding.symbol))
    adjusted_price = math.nan if current is None else float(current.get("adjusted_price", math.nan))
    return position_value(holding.position, adjusted_price) * holding.remaining_fraction


def _account_result(
    state: MinuteAccountState,
    equity: pd.DataFrame,
    trades: pd.DataFrame,
    fills: pd.DataFrame,
    *,
    starting_cash: float,
    slippage_multiplier: float,
    maximum_credited_gross_return: float | None,
    maximum_participation_rate: float,
    unresolved_value: float,
    evaluation_years: set[int],
) -> dict[str, Any]:
    ending_equity = float(equity["equity"].iloc[-1])
    delayed = int(trades["delayed_exit"].sum()) if not trades.empty else 0
    partial = int(trades["partial_exit"].sum()) if not trades.empty else 0
    annual: list[dict[str, Any]] = []
    for year, group in equity.groupby(equity["trade_date"].astype(str).str[:4].astype(int), sort=True):
        wealth = (1.0 + group["daily_return"]).cumprod()
        drawdown = wealth / wealth.cummax() - 1.0
        year_trades = trades.loc[trades["exit_date"].astype(str).str.startswith(f"{year}-")] if not trades.empty else trades
        annual.append(
            {
                "year": int(year),
                "date_count": int(len(group)),
                "net_return": float(wealth.iloc[-1] - 1.0),
                "maximum_drawdown": float(drawdown.min()),
                "closed_trade_count": int(len(year_trades)),
                "positive_trade_fraction": (
                    float(year_trades["pnl"].gt(0.0).mean()) if len(year_trades) else None
                ),
                "evaluation_year": int(year) in evaluation_years,
            }
        )
    positive = (
        np.sort(trades.loc[trades["pnl"] > 0.0, "pnl"].to_numpy(dtype=np.float64))[::-1]
        if not trades.empty
        else np.asarray([], dtype=np.float64)
    )
    positive_total = float(positive.sum())

    def winner_share(fraction: float) -> float | None:
        if positive_total <= 0.0:
            return None
        count = max(1, math.ceil(len(positive) * fraction))
        return float(positive[:count].sum() / positive_total)

    return {
        "starting_cash": float(starting_cash),
        "ending_cash": float(state.cash),
        "ending_equity": ending_equity,
        "net_return": ending_equity / float(starting_cash) - 1.0,
        "maximum_drawdown": float(equity["drawdown"].min()),
        "account_date_count": int(len(equity)),
        "selection_count": state.selection_count,
        "filled_entry_count": state.entry_count,
        "unfilled_selection_count": state.unfilled_selection_count,
        "overlap_skip_count": state.overlap_skip_count,
        "closed_trade_count": int(len(trades)),
        "partial_exit_episode_count": partial,
        "delayed_exit_episode_count": delayed,
        "exit_fill_count": int(len(fills)),
        "unresolved_position_count": int(len(state.holdings)),
        "unresolved_position_value": unresolved_value,
        "slippage_multiplier": float(slippage_multiplier),
        "maximum_credited_gross_return": maximum_credited_gross_return,
        "maximum_participation_rate": float(maximum_participation_rate),
        "mean_cash_fraction": float((equity["cash"] / equity["equity"]).mean()),
        "exit_participation_p95": (
            float(fills["participation_rate"].quantile(0.95)) if not fills.empty else None
        ),
        "exit_participation_max": (
            float(fills["participation_rate"].max()) if not fills.empty else None
        ),
        "exit_capacity_breach_count": 0,
        "positive_year_count": int(
            sum(bool(row["evaluation_year"]) and float(row["net_return"]) > 0.0 for row in annual)
        ),
        "annual": annual,
        "winner_concentration": {
            "top_1pct_positive_pnl_share": winner_share(0.01),
            "top_5pct_positive_pnl_share": winner_share(0.05),
            "top_10pct_positive_pnl_share": winner_share(0.10),
        },
        "t1_contract": "10:00 decision; 10:01-10:10 VWAP entry; exits start in the same window next market day",
        "capacity_contract": "each entry and partial exit is capped by a fixed share of its own 10-minute window",
    }


def simulate_minute_account(
    panel: pd.DataFrame,
    windows: pd.DataFrame,
    *,
    evaluation_start_date: str,
    evaluation_end_date: str = "",
    top_k: int = 5,
    starting_cash: float = 1_000_000.0,
    slippage_multiplier: float = 2.0,
    maximum_credited_gross_return: float | None = None,
    maximum_participation_rate: float = 0.01,
) -> tuple[dict[str, Any], pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Replay candidate-first selections and capacity-limited partial exits."""

    if not 0.0 < float(maximum_participation_rate) <= 1.0:
        raise MinuteResearchError("minute_participation_rate_invalid")
    mask = (panel["signal_date"] >= str(evaluation_start_date)) & panel["prediction"].notna()
    if evaluation_end_date:
        mask &= panel["signal_date"] <= str(evaluation_end_date)
    evaluated = panel.loc[mask].copy()
    if evaluated.empty:
        raise MinuteResearchError("minute_evaluation_is_empty")
    schedule = {str(date): group for date, group in evaluated.groupby("signal_date", sort=True)}
    signal_dates = sorted(schedule)
    first_signal, last_signal = signal_dates[0], signal_dates[-1]
    calendar = sorted(
        date for date in windows["trade_date"].astype(str).unique().tolist() if date >= first_signal
    )
    if not calendar or last_signal not in calendar:
        raise MinuteResearchError("minute_execution_calendar_incomplete")
    date_positions = {date: index for index, date in enumerate(calendar)}
    lookup = _window_lookup(windows)
    costs = default_minute_costs()
    state = MinuteAccountState(cash=float(starting_cash), previous_equity=float(starting_cash))
    for date in calendar:
        _process_exits(
            state,
            date=date,
            windows=lookup,
            maximum_participation_rate=maximum_participation_rate,
            maximum_credited_gross_return=maximum_credited_gross_return,
            costs=costs,
            slippage_multiplier=slippage_multiplier,
        )
        current = schedule.get(date)
        if current is not None:
            _open_orders(
                state,
                current,
                date=date,
                date_index=date_positions[date],
                top_k=top_k,
                maximum_participation_rate=maximum_participation_rate,
                costs=costs,
                slippage_multiplier=slippage_multiplier,
            )
        holding_value = sum(_holding_value(holding, date=date, windows=lookup) for holding in state.holdings.values())
        equity = state.cash + holding_value
        state.equity_rows.append(
            {
                "trade_date": date,
                "cash": state.cash,
                "position_count": len(state.holdings),
                "equity": equity,
                "daily_return": equity / state.previous_equity - 1.0,
            }
        )
        state.previous_equity = equity
        if date >= last_signal and not state.holdings:
            break
    equity = pd.DataFrame(state.equity_rows)
    equity["equity_peak"] = equity["equity"].cummax()
    equity["drawdown"] = equity["equity"] / equity["equity_peak"] - 1.0
    trades = pd.DataFrame(state.trade_rows)
    fills = pd.DataFrame(state.fill_rows)
    unresolved_value = float(equity["equity"].iloc[-1] - state.cash)
    result = _account_result(
        state,
        equity,
        trades,
        fills,
        starting_cash=starting_cash,
        slippage_multiplier=slippage_multiplier,
        maximum_credited_gross_return=maximum_credited_gross_return,
        maximum_participation_rate=maximum_participation_rate,
        unresolved_value=unresolved_value,
        evaluation_years={int(date[:4]) for date in signal_dates},
    )
    return result, equity, trades, fills


__all__ = ["default_minute_costs", "simulate_minute_account"]
