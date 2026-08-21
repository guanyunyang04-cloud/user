from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class ExecutionCosts:
    lot_size: int
    commission_bps: float
    minimum_commission_cny: float
    transfer_fee_bps: float
    slippage_bps: float
    stress_slippage_multiplier: float
    stamp_tax_schedule: tuple[tuple[str, float], ...]


@dataclass
class Position:
    symbol_idx: int
    shares: int
    entry_signal_date_idx: int
    entry_date_idx: int
    entry_raw_open: float
    entry_adjusted_open: float
    gross_entry_notional: float
    net_cash_outflow: float
    gross_cash_outflow: float
    last_adjusted_price: float


@dataclass
class Holding:
    trade_id: int
    position: Position
    signal_date_idx: int
    signal_date: str
    symbol: str
    selection_rank: int
    legal_gross_return: float
    exit_date_idx: int
    exit_phase: str
    buy_costs: dict[str, float]


class PortfolioError(RuntimeError):
    pass


def parse_execution_costs(config: Mapping[str, Any]) -> ExecutionCosts:
    raw = config.get("execution_costs", config)
    if not isinstance(raw, Mapping):
        raise PortfolioError("execution costs are missing")

    def non_negative(name: str, default: float | None = None) -> float:
        value = raw.get(name, default)
        if value is None:
            raise PortfolioError(f"execution cost is missing: {name}")
        number = float(value)
        if not math.isfinite(number) or number < 0.0:
            raise PortfolioError(f"invalid execution cost: {name}")
        return number

    lot_size = int(raw.get("lot_size", 0))
    if lot_size != 100:
        raise PortfolioError("A-share account requires 100-share lots")
    schedule_raw = raw.get("stamp_tax_schedule")
    if not isinstance(schedule_raw, Sequence) or isinstance(schedule_raw, (str, bytes)):
        raise PortfolioError("stamp-tax schedule is missing")
    schedule: list[tuple[str, float]] = []
    previous = ""
    for item in schedule_raw:
        if not isinstance(item, Mapping):
            raise PortfolioError("invalid stamp-tax schedule entry")
        date_text = pd.Timestamp(str(item["effective_date"])).strftime("%Y-%m-%d")
        rate = float(item["stamp_tax_bps"])
        if date_text <= previous or not math.isfinite(rate) or rate < 0.0:
            raise PortfolioError("stamp-tax schedule is not strictly valid")
        schedule.append((date_text, rate))
        previous = date_text
    costs = ExecutionCosts(
        lot_size=lot_size,
        commission_bps=non_negative("commission_bps"),
        minimum_commission_cny=non_negative("minimum_commission_cny"),
        transfer_fee_bps=non_negative("transfer_fee_bps"),
        slippage_bps=non_negative("slippage_bps", 7.0),
        stress_slippage_multiplier=non_negative("stress_slippage_multiplier", 2.0),
        stamp_tax_schedule=tuple(schedule),
    )
    if not math.isclose(costs.slippage_bps, 7.0, abs_tol=1.0e-12):
        raise PortfolioError("slippage contract changed")
    if not math.isclose(costs.stress_slippage_multiplier, 2.0, abs_tol=1.0e-12):
        raise PortfolioError("stress slippage contract changed")
    return costs


def _stamp_tax_bps(costs: ExecutionCosts, trade_date: str) -> float:
    matches = [
        float(rate)
        for effective_date, rate in costs.stamp_tax_schedule
        if str(effective_date) <= str(trade_date)
    ]
    return float(matches[-1]) if matches else 0.0


def position_value(position: Position, adjusted_price: float) -> float:
    if (
        not math.isfinite(adjusted_price)
        or adjusted_price <= 0.0
        or not math.isfinite(position.entry_adjusted_open)
        or position.entry_adjusted_open <= 0.0
    ):
        return float(
            position.gross_entry_notional
            * position.last_adjusted_price
            / position.entry_adjusted_open
        )
    position.last_adjusted_price = float(adjusted_price)
    return float(
        position.gross_entry_notional
        * float(adjusted_price)
        / position.entry_adjusted_open
    )


def buy_position(
    *,
    available_cash: float,
    allocated_cash: float,
    symbol_idx: int,
    signal_date_idx: int,
    execution_date_idx: int,
    raw_open: float,
    adjusted_open: float,
    costs: ExecutionCosts,
    slippage_multiplier: float,
) -> tuple[Position | None, dict[str, float]]:
    if (
        available_cash <= 0.0
        or allocated_cash <= 0.0
        or not math.isfinite(raw_open)
        or raw_open <= 0.0
        or not math.isfinite(adjusted_open)
        or adjusted_open <= 0.0
    ):
        return None, {}
    allocation = min(float(available_cash), float(allocated_cash))
    slippage_rate = costs.slippage_bps * float(slippage_multiplier) / 10_000.0
    fill_price = float(raw_open) * (1.0 + slippage_rate)
    maximum_lots = math.floor(allocation / (fill_price * costs.lot_size))

    def cash_outflow_for_lots(lots: int) -> float:
        fill_notional = int(lots) * costs.lot_size * fill_price
        commission = max(
            costs.minimum_commission_cny,
            fill_notional * costs.commission_bps / 10_000.0,
        )
        transfer = fill_notional * costs.transfer_fee_bps / 10_000.0
        return float(fill_notional + commission + transfer)

    cash_limit = min(float(available_cash), allocation)
    if cash_outflow_for_lots(maximum_lots) <= cash_limit + 1.0e-9:
        affordable_lots = maximum_lots
    else:
        lower, upper = 0, maximum_lots
        while lower < upper:
            middle = (lower + upper + 1) // 2
            if cash_outflow_for_lots(middle) <= cash_limit + 1.0e-9:
                lower = middle
            else:
                upper = middle - 1
        affordable_lots = lower
    shares = int(affordable_lots) * costs.lot_size
    if shares <= 0:
        return None, {}
    gross_notional = float(shares) * float(raw_open)
    fill_notional = float(shares) * fill_price
    commission = max(
        costs.minimum_commission_cny,
        fill_notional * costs.commission_bps / 10_000.0,
    )
    transfer = fill_notional * costs.transfer_fee_bps / 10_000.0
    slippage = fill_notional - gross_notional
    cash_outflow = fill_notional + commission + transfer
    position = Position(
        symbol_idx=int(symbol_idx),
        shares=int(shares),
        entry_signal_date_idx=int(signal_date_idx),
        entry_date_idx=int(execution_date_idx),
        entry_raw_open=float(raw_open),
        entry_adjusted_open=float(adjusted_open),
        gross_entry_notional=float(gross_notional),
        net_cash_outflow=float(cash_outflow),
        gross_cash_outflow=float(gross_notional),
        last_adjusted_price=float(adjusted_open),
    )
    details = {
        "gross_notional": gross_notional,
        "fill_notional": fill_notional,
        "commission": float(commission),
        "transfer_fee": float(transfer),
        "stamp_tax": 0.0,
        "slippage": float(slippage),
        "cash_flow": float(-cash_outflow),
        "total_cost": float(commission + transfer + slippage),
    }
    return position, details


def sell_position(
    *,
    position: Position,
    adjusted_open: float,
    trade_date: str,
    costs: ExecutionCosts,
    slippage_multiplier: float,
) -> tuple[float, dict[str, float]]:
    gross_value = position_value(position, float(adjusted_open))
    slippage_rate = costs.slippage_bps * float(slippage_multiplier) / 10_000.0
    fill_notional = gross_value * max(0.0, 1.0 - slippage_rate)
    commission = max(
        costs.minimum_commission_cny,
        fill_notional * costs.commission_bps / 10_000.0,
    )
    transfer = fill_notional * costs.transfer_fee_bps / 10_000.0
    stamp = fill_notional * _stamp_tax_bps(costs, trade_date) / 10_000.0
    proceeds = fill_notional - commission - transfer - stamp
    slippage = gross_value - fill_notional
    details = {
        "gross_notional": float(gross_value),
        "fill_notional": float(fill_notional),
        "commission": float(commission),
        "transfer_fee": float(transfer),
        "stamp_tax": float(stamp),
        "slippage": float(slippage),
        "cash_flow": float(proceeds),
        "total_cost": float(commission + transfer + stamp + slippage),
    }
    return float(proceeds), details


def account_task_id(spec: Mapping[str, Any]) -> str:
    task_id = f"top{int(spec['top_k'])}__{spec['cost_scenario']}"
    cap = spec.get("maximum_credited_gross_return")
    if cap is not None:
        task_id += f"__cap{round(float(cap) * 100)}pct"
    if not bool(spec.get("allow_overlapping_same_symbol", True)):
        task_id += "__no_overlap"
    return task_id


def account_specs(horizon: int = 10) -> tuple[dict[str, Any], ...]:
    specs: list[dict[str, Any]] = []
    for top_k in (1, 3, 5, 10):
        for scenario, multiplier in (("base", 1.0), ("stress", 2.0)):
            specs.append(
                {
                    "top_k": top_k,
                    "cost_scenario": scenario,
                    "slippage_multiplier": multiplier,
                    "cohort_equity_fraction": 1.0 / float(horizon),
                    "planned_fill_day": horizon,
                }
            )
    for cap in (0.05, 0.10):
        for scenario, multiplier in (("base", 1.0), ("stress", 2.0)):
            specs.append(
                {
                    "top_k": 10,
                    "cost_scenario": scenario,
                    "slippage_multiplier": multiplier,
                    "cohort_equity_fraction": 1.0 / float(horizon),
                    "planned_fill_day": horizon,
                    "maximum_credited_gross_return": cap,
                }
            )
    for cap in (None, 0.05, 0.10):
        spec: dict[str, Any] = {
            "top_k": 10,
            "cost_scenario": "stress",
            "slippage_multiplier": 2.0,
            "cohort_equity_fraction": 1.0 / float(horizon),
            "planned_fill_day": horizon,
            "allow_overlapping_same_symbol": False,
        }
        if cap is not None:
            spec["maximum_credited_gross_return"] = cap
        specs.append(spec)
    return tuple(specs)


def simulate_account(
    *,
    spec: Mapping[str, Any],
    selections: pd.DataFrame,
    date_values: np.ndarray,
    symbol_values: Sequence[str],
    cutoff_idx: int,
    daily_raw: np.ndarray,
    raw_open: np.ndarray,
    entry_filled: np.ndarray,
    costs: ExecutionCosts,
    starting_cash: float = 1_000_000.0,
) -> tuple[dict[str, Any], pd.DataFrame, pd.DataFrame]:
    selected = selections.loc[selections["selection_rank"] <= int(spec["top_k"])].copy()
    selected = selected.drop_duplicates(
        ["date_idx", "candidate_id", "selection_rank"], keep="first"
    ).sort_values(["date_idx", "selection_rank"])
    if selected.empty:
        raise PortfolioError(f"selection schedule is empty: {spec}")
    if "symbol_idx" not in selected:
        mapping = {str(symbol): index for index, symbol in enumerate(symbol_values)}
        selected["symbol_idx"] = selected["symbol"].map(mapping)
    if selected["symbol_idx"].isna().any():
        raise PortfolioError("selection symbol is absent from pack")

    schedule = {
        int(date_idx): group.copy()
        for date_idx, group in selected.groupby("date_idx", sort=True)
    }
    first_signal_idx = int(selected["date_idx"].min())
    last_signal_idx = int(selected["date_idx"].max())
    holdings: dict[int, Holding] = {}
    open_exits: dict[int, list[int]] = {}
    close_exits: dict[int, list[int]] = {}
    cash = float(starting_cash)
    previous_equity = float(starting_cash)
    trade_id = 0
    equity_rows: list[dict[str, Any]] = []
    trade_rows: list[dict[str, Any]] = []
    symbol_open_counts: dict[int, int] = {}
    overlap_orders = 0
    overlap_filled = 0
    overlap_skipped = 0
    maximum_symbol_cohorts = 0
    total_costs = {
        "commission": 0.0,
        "transfer_fee": 0.0,
        "stamp_tax": 0.0,
        "slippage": 0.0,
    }

    def close_trade(identifier: int, current_idx: int) -> None:
        nonlocal cash
        holding = holdings.pop(identifier)
        symbol_idx = holding.position.symbol_idx
        remaining = symbol_open_counts.get(symbol_idx, 0) - 1
        if remaining < 0:
            raise PortfolioError("negative symbol cohort count")
        if remaining:
            symbol_open_counts[symbol_idx] = remaining
        else:
            symbol_open_counts.pop(symbol_idx, None)
        if holding.legal_gross_return <= -1.0:
            proceeds = 0.0
            sale = {name: 0.0 for name in (*total_costs, "total_cost")}
        else:
            adjusted_exit = holding.position.entry_adjusted_open * (
                1.0 + holding.legal_gross_return
            )
            proceeds, sale = sell_position(
                position=holding.position,
                adjusted_open=adjusted_exit,
                trade_date=str(date_values[current_idx]),
                costs=costs,
                slippage_multiplier=float(spec["slippage_multiplier"]),
            )
        cash += proceeds
        for name in total_costs:
            total_costs[name] += float(sale.get(name, 0.0))
        pnl = proceeds - holding.position.net_cash_outflow
        trade_rows.append(
            {
                "trade_id": identifier,
                "signal_date_idx": holding.signal_date_idx,
                "signal_date": holding.signal_date,
                "entry_date_idx": holding.position.entry_date_idx,
                "entry_date": str(date_values[holding.position.entry_date_idx]),
                "exit_date_idx": current_idx,
                "exit_date": str(date_values[current_idx]),
                "exit_phase": holding.exit_phase,
                "symbol": holding.symbol,
                "symbol_idx": symbol_idx,
                "selection_rank": holding.selection_rank,
                "shares": holding.position.shares,
                "gross_entry_notional": holding.position.gross_entry_notional,
                "net_cash_outflow": holding.position.net_cash_outflow,
                "proceeds": proceeds,
                "pnl": pnl,
                "trade_net_return": pnl / holding.position.net_cash_outflow,
                "legal_gross_return": holding.legal_gross_return,
                "buy_cost": float(holding.buy_costs.get("total_cost", 0.0)),
                "sell_cost": float(sale.get("total_cost", 0.0)),
            }
        )

    for current_idx in range(first_signal_idx, int(cutoff_idx) + 1):
        for identifier in open_exits.pop(current_idx, []):
            if identifier in holdings:
                close_trade(identifier, current_idx)

        signal_idx = current_idx - 1
        orders = schedule.get(signal_idx)
        requested = 0 if orders is None else len(orders)
        filled = 0
        cohort_budget = min(
            cash, previous_equity * float(spec["cohort_equity_fraction"])
        )
        per_order_budget = (
            cohort_budget / int(spec["top_k"])
            if requested and int(spec["top_k"]) > 0
            else 0.0
        )
        if orders is not None:
            for row in orders.itertuples(index=False):
                symbol_idx = int(row.symbol_idx)
                if not bool(entry_filled[signal_idx, symbol_idx]):
                    continue
                existing = symbol_open_counts.get(symbol_idx, 0)
                if existing:
                    overlap_orders += 1
                    if not bool(spec.get("allow_overlapping_same_symbol", True)):
                        overlap_skipped += 1
                        continue
                adjusted_entry = float(daily_raw[current_idx, symbol_idx, 0])
                entry_raw = float(raw_open[current_idx, symbol_idx])
                position, buy = buy_position(
                    available_cash=cash,
                    allocated_cash=per_order_budget,
                    symbol_idx=symbol_idx,
                    signal_date_idx=signal_idx,
                    execution_date_idx=current_idx,
                    raw_open=entry_raw,
                    adjusted_open=adjusted_entry,
                    costs=costs,
                    slippage_multiplier=float(spec["slippage_multiplier"]),
                )
                if position is None:
                    continue
                cash -= position.net_cash_outflow
                if cash < -1.0e-6:
                    raise PortfolioError("account cash became negative")
                cash = max(cash, 0.0)
                filled += 1
                if existing:
                    overlap_filled += 1
                trade_id += 1
                fill_day = int(row.fill_day)
                exit_idx = signal_idx + fill_day
                uncapped = float(row.legal_gross_return)
                cap = spec.get("maximum_credited_gross_return")
                legal_return = (
                    min(uncapped, float(cap)) if cap is not None else uncapped
                )
                if legal_return <= -1.0:
                    phase = "terminal_recovery"
                    close_exits.setdefault(exit_idx, []).append(trade_id)
                elif fill_day == int(spec.get("planned_fill_day", 2)):
                    phase = "planned_close"
                    close_exits.setdefault(exit_idx, []).append(trade_id)
                else:
                    phase = "delayed_open"
                    open_exits.setdefault(exit_idx, []).append(trade_id)
                holdings[trade_id] = Holding(
                    trade_id=trade_id,
                    position=position,
                    signal_date_idx=signal_idx,
                    signal_date=str(date_values[signal_idx]),
                    symbol=str(row.symbol),
                    selection_rank=int(row.selection_rank),
                    legal_gross_return=legal_return,
                    exit_date_idx=exit_idx,
                    exit_phase=phase,
                    buy_costs={str(key): float(value) for key, value in buy.items()},
                )
                symbol_open_counts[symbol_idx] = existing + 1
                maximum_symbol_cohorts = max(maximum_symbol_cohorts, existing + 1)
                for name in total_costs:
                    total_costs[name] += float(buy.get(name, 0.0))

        for identifier in close_exits.pop(current_idx, []):
            if identifier in holdings:
                close_trade(identifier, current_idx)

        adjusted_close = np.asarray(daily_raw[current_idx, :, 3], dtype=np.float64)
        equity = cash
        for holding in holdings.values():
            equity += position_value(
                holding.position, adjusted_close[holding.position.symbol_idx]
            )
        daily_return = equity / previous_equity - 1.0 if previous_equity > 0.0 else 0.0
        equity_rows.append(
            {
                "date_idx": current_idx,
                "trade_date": str(date_values[current_idx]),
                "year": int(str(date_values[current_idx])[:4]),
                "cash": cash,
                "position_count": len(holdings),
                "unique_symbol_count": len(symbol_open_counts),
                "maximum_same_symbol_open_cohorts": max(
                    symbol_open_counts.values(), default=0
                ),
                "requested_entry_count": requested,
                "filled_entry_count": filled,
                "equity": equity,
                "daily_net_return": daily_return,
            }
        )
        previous_equity = equity

    if holdings or open_exits or close_exits:
        raise PortfolioError(
            f"positions remain unresolved at {date_values[cutoff_idx]} cutoff"
        )
    equity = pd.DataFrame(equity_rows)
    trades = pd.DataFrame(trade_rows)
    equity["equity_peak"] = equity["equity"].cummax()
    equity["drawdown"] = equity["equity"] / equity["equity_peak"] - 1.0
    annual: list[dict[str, Any]] = []
    for year, group in equity.groupby("year", sort=True):
        trade_count = 0
        if not trades.empty:
            trade_count = int(
                trades["exit_date"].astype(str).str.startswith(f"{year}-").sum()
            )
        annual.append(
            {
                "year": int(year),
                "net_return": float(np.prod(1.0 + group["daily_net_return"]) - 1.0),
                "maximum_drawdown": float(group["drawdown"].min()),
                "trade_count": trade_count,
            }
        )
    annual_frame = pd.DataFrame(annual)
    daily_returns = equity["daily_net_return"].to_numpy(dtype=np.float64)
    result = {
        "task_id": account_task_id(spec),
        "spec": dict(spec),
        "first_signal_date": str(date_values[first_signal_idx]),
        "last_signal_date": str(date_values[last_signal_idx]),
        "last_account_date": str(date_values[cutoff_idx]),
        "starting_cash": float(starting_cash),
        "ending_equity": float(equity["equity"].iloc[-1]),
        "total_net_return": float(equity["equity"].iloc[-1] / starting_cash - 1.0),
        "maximum_drawdown": float(equity["drawdown"].min()),
        "annualized_daily_sharpe": (
            float(np.sqrt(242.0) * daily_returns.mean() / daily_returns.std(ddof=1))
            if daily_returns.std(ddof=1) > 0.0
            else math.nan
        ),
        "trade_count": len(trades),
        "positive_trade_fraction": (
            float(trades["pnl"].gt(0.0).mean()) if len(trades) else math.nan
        ),
        "mean_trade_net_return": (
            float(trades["trade_net_return"].mean()) if len(trades) else math.nan
        ),
        "positive_year_count": int(annual_frame["net_return"].gt(0.0).sum()),
        "negative_year_count": int(annual_frame["net_return"].lt(0.0).sum()),
        "worst_year_return": float(annual_frame["net_return"].min()),
        "total_costs": total_costs,
        "minimum_cash": float(equity["cash"].min()),
        "maximum_position_count": int(equity["position_count"].max()),
        "maximum_unique_symbol_count": int(equity["unique_symbol_count"].max()),
        "maximum_same_symbol_open_cohorts": maximum_symbol_cohorts,
        "allow_overlapping_same_symbol": bool(
            spec.get("allow_overlapping_same_symbol", True)
        ),
        "same_symbol_overlap_order_count": overlap_orders,
        "same_symbol_overlap_filled_count": overlap_filled,
        "same_symbol_overlap_skipped_count": overlap_skipped,
        "unresolved_position_count": 0,
        "cutoff_violation_count": 0,
        "annual": annual,
    }
    return result, equity, trades


def newey_west_interval(
    values: np.ndarray, *, lag: int = 20, z_value: float = 1.96
) -> dict[str, float | int | None]:
    current = np.asarray(values, dtype=np.float64)
    current = current[np.isfinite(current)]
    count = len(current)
    if count < 2:
        return {
            "count": count,
            "mean": float(current.mean()) if count else None,
            "standard_error": None,
            "lower": None,
            "upper": None,
        }
    demeaned = current - current.mean()
    maximum_lag = min(int(lag), count - 1)
    long_run = float(np.dot(demeaned, demeaned) / count)
    for offset in range(1, maximum_lag + 1):
        covariance = float(np.dot(demeaned[offset:], demeaned[:-offset]) / count)
        long_run += 2.0 * (1.0 - offset / (maximum_lag + 1.0)) * covariance
    standard_error = math.sqrt(max(long_run, 0.0) / count)
    mean = float(current.mean())
    return {
        "count": count,
        "mean": mean,
        "standard_error": standard_error,
        "lower": mean - z_value * standard_error,
        "upper": mean + z_value * standard_error,
    }


def add_account_diagnostics(
    *,
    result: dict[str, Any],
    equity: pd.DataFrame,
    trades: pd.DataFrame,
    selections: pd.DataFrame,
) -> dict[str, Any]:
    daily_returns = equity["daily_net_return"].to_numpy(dtype=np.float64)
    result["daily_hac20_net_return"] = newey_west_interval(daily_returns, lag=20)
    result["daily_positive_fraction"] = float(np.mean(daily_returns > 0.0))
    result["daily_return_p01"] = float(np.quantile(daily_returns, 0.01))
    result["daily_return_p05"] = float(np.quantile(daily_returns, 0.05))
    date_folds = (
        selections.groupby("date_idx", sort=False)["fold"].first().astype("Int64")
    )
    with_fold = equity.copy()
    with_fold["fold"] = with_fold["date_idx"].map(date_folds)
    result["fold_account_metrics"] = [
        {
            "fold": int(fold),
            "date_count": len(group),
            "mean_daily_net_return": float(group["daily_net_return"].mean()),
            "compound_net_return": float(
                np.prod(1.0 + group["daily_net_return"]) - 1.0
            ),
            "hac20_net_return": newey_west_interval(
                group["daily_net_return"].to_numpy(dtype=np.float64), lag=20
            ),
        }
        for fold, group in with_fold.dropna(subset=["fold"]).groupby("fold", sort=True)
    ]
    if trades.empty:
        result["winner_concentration"] = {"trade_count": 0}
        return result
    positive = np.sort(
        trades.loc[trades["pnl"] > 0.0, "pnl"].to_numpy(dtype=np.float64)
    )[::-1]
    positive_total = float(positive.sum())

    def share(fraction: float) -> float | None:
        if positive_total <= 0.0 or not len(positive):
            return None
        count = max(1, math.ceil(len(positive) * fraction))
        return float(positive[:count].sum() / positive_total)

    result["winner_concentration"] = {
        "trade_count": len(trades),
        "positive_pnl_total": positive_total,
        "top_1pct_positive_pnl_share": share(0.01),
        "top_5pct_positive_pnl_share": share(0.05),
        "largest_winner_trade_net_return": float(trades["trade_net_return"].max()),
        "largest_loser_trade_net_return": float(trades["trade_net_return"].min()),
    }
    return result
