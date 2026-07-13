from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Mapping, Protocol, Sequence

import numpy as np
import pandas as pd

from daily_research.path_policy.seq100_qcurve import (
    QCURVE_ENTRY_HORIZONS,
    QCURVE_HOLD_HORIZONS,
    QCurveCostContract,
    allocate_dynamic_qcurve_portfolio,
)
from daily_research.path_policy.seq100_qcurve_data import QCurvePack


class QCurvePredictionProvider(Protocol):
    def predict_symbols(self, date_idx: int, symbols: np.ndarray) -> Mapping[str, np.ndarray]: ...


@dataclass
class Position:
    shares: int
    basis_cash: float
    entry_date_idx: int
    last_price: float


@dataclass
class PortfolioState:
    cash: float = 1_000_000.0
    positions: dict[int, Position] = field(default_factory=dict)
    fees_paid: float = 0.0
    traded_notional: float = 0.0
    realized_pnls: list[float] = field(default_factory=list)
    holding_days: list[int] = field(default_factory=list)


@dataclass
class ReplayAccumulator:
    equities: list[float] = field(default_factory=list)
    daily_log_returns: list[float] = field(default_factory=list)
    benchmark_log_returns: list[float] = field(default_factory=list)
    cash_weights: list[float] = field(default_factory=list)
    position_counts: list[int] = field(default_factory=list)
    turnover_values: list[float] = field(default_factory=list)
    decision_rows: list[dict[str, Any]] = field(default_factory=list)
    trade_rows: list[dict[str, Any]] = field(default_factory=list)


def _stamp_tax_bps(contract: QCurveCostContract, date: np.datetime64) -> float:
    selected: float | None = None
    for effective, rate in sorted(contract.stamp_tax_schedule):
        if date >= np.datetime64(effective):
            selected = float(rate)
    if selected is None:
        raise ValueError("trade date predates stamp tax schedule")
    return selected


def _mark_price(pack: QCurvePack, date_idx: int, symbol_idx: int, position: Position, *, at_open: bool) -> float:
    panel = pack.execution["open_raw"] if at_open else pack.execution["signal_close_raw"]
    value = float(panel[int(date_idx), int(symbol_idx)])
    if math.isfinite(value) and value > 0.0:
        position.last_price = value
        return value
    return float(position.last_price)


def _equity(pack: QCurvePack, state: PortfolioState, date_idx: int, *, at_open: bool) -> float:
    return float(state.cash) + sum(
        int(position.shares) * _mark_price(pack, date_idx, symbol, position, at_open=at_open)
        for symbol, position in state.positions.items()
    )


def _current_weights(pack: QCurvePack, state: PortfolioState, date_idx: int) -> tuple[float, dict[int, float]]:
    equity = _equity(pack, state, date_idx, at_open=False)
    weights = {
        symbol: int(position.shares)
        * _mark_price(pack, date_idx, symbol, position, at_open=False)
        / max(equity, 1.0e-12)
        for symbol, position in state.positions.items()
    }
    return equity, weights


def _trade_cost_fraction(contract: QCurveCostContract, *, sell: bool, slippage_multiplier: float) -> float:
    result = (contract.commission_bps + contract.transfer_fee_bps) / 10_000.0
    result += contract.slippage_bps * float(slippage_multiplier) / 10_000.0
    if sell:
        result += max(rate for _, rate in contract.stamp_tax_schedule) / 10_000.0
    return result


def _apply_turnover_gate(
    allocation: pd.DataFrame,
    *,
    equity: float,
    contract: QCurveCostContract,
    slippage_multiplier: float,
) -> tuple[pd.DataFrame, dict[str, float | None]]:
    working = allocation.copy()
    current = working.set_index("symbol")["current_weight"].astype(float).clip(lower=0.0)
    proposed = working.set_index("symbol")["target_weight"].astype(float).clip(lower=0.0)
    mandatory_exit = working[
        working["is_held"].astype(bool)
        & working["sellable_next_open"].astype(bool)
        & (~np.isfinite(working["path_value"]) | working["path_value"].le(0.0))
    ]
    if not mandatory_exit.empty:
        return working, {"turnover_gate_applied": 0.0, "expected_gain": None, "estimated_cost": 0.0}
    values = working.set_index("symbol")["path_value"].replace([np.inf, -np.inf], np.nan).fillna(0.0)
    old_utility = float((current * values).sum())
    new_utility = float((proposed * values).sum())
    sells = float((current - proposed).clip(lower=0.0).sum())
    buys = float((proposed - current).clip(lower=0.0).sum())
    changed_orders = int(np.count_nonzero(np.abs(proposed - current) > 1.0e-6))
    estimated_cost = (
        sells * _trade_cost_fraction(contract, sell=True, slippage_multiplier=slippage_multiplier)
        + buys * _trade_cost_fraction(contract, sell=False, slippage_multiplier=slippage_multiplier)
        + changed_orders * contract.minimum_commission_cny / max(float(equity), 1.0)
    )
    expected_gain = new_utility - old_utility
    if expected_gain <= estimated_cost:
        target = current.to_dict()
        working["target_weight"] = working["symbol"].map(target).fillna(0.0).astype(float)
        working["selected"] = working["target_weight"].gt(1.0e-12)
        applied = 1.0
    else:
        applied = 0.0
    return working, {
        "turnover_gate_applied": applied,
        "expected_gain": float(expected_gain),
        "estimated_cost": float(estimated_cost),
    }


def _sell_order(
    state: PortfolioState,
    *,
    symbol: int,
    shares: int,
    raw_open: float,
    trade_date: np.datetime64,
    date_idx: int,
    contract: QCurveCostContract,
    slippage_multiplier: float,
) -> dict[str, Any]:
    position = state.positions[symbol]
    quantity = min(int(shares), int(position.shares))
    if quantity <= 0:
        return {}
    execution_price = float(raw_open) * (
        1.0 - contract.slippage_bps * float(slippage_multiplier) / 10_000.0
    )
    notional = quantity * execution_price
    commission = max(contract.minimum_commission_cny, notional * contract.commission_bps / 10_000.0)
    transfer = notional * contract.transfer_fee_bps / 10_000.0
    stamp = notional * _stamp_tax_bps(contract, trade_date) / 10_000.0
    fees = commission + transfer + stamp
    basis = position.basis_cash * quantity / int(position.shares)
    net_proceeds = notional - fees
    state.cash += net_proceeds
    state.fees_paid += fees
    state.traded_notional += notional
    position.shares -= quantity
    position.basis_cash -= basis
    state.realized_pnls.append(float(net_proceeds - basis))
    state.holding_days.append(int(date_idx) - int(position.entry_date_idx))
    if position.shares == 0:
        del state.positions[symbol]
    return {
        "side": "sell",
        "symbol_idx": int(symbol),
        "shares": quantity,
        "raw_open": float(raw_open),
        "execution_price": execution_price,
        "notional": notional,
        "fees": fees,
    }


def _buy_order(
    state: PortfolioState,
    *,
    symbol: int,
    desired_shares: int,
    raw_open: float,
    date_idx: int,
    contract: QCurveCostContract,
    slippage_multiplier: float,
) -> dict[str, Any]:
    current = int(state.positions[symbol].shares) if symbol in state.positions else 0
    quantity = max(int(desired_shares) - current, 0)
    quantity = quantity // int(contract.lot_size) * int(contract.lot_size)
    execution_price = float(raw_open) * (
        1.0 + contract.slippage_bps * float(slippage_multiplier) / 10_000.0
    )
    while quantity > 0:
        notional = quantity * execution_price
        commission = max(contract.minimum_commission_cny, notional * contract.commission_bps / 10_000.0)
        transfer = notional * contract.transfer_fee_bps / 10_000.0
        required = notional + commission + transfer
        if required <= state.cash + 1.0e-9:
            break
        quantity -= int(contract.lot_size)
    if quantity <= 0:
        return {}
    notional = quantity * execution_price
    commission = max(contract.minimum_commission_cny, notional * contract.commission_bps / 10_000.0)
    transfer = notional * contract.transfer_fee_bps / 10_000.0
    fees = commission + transfer
    required = notional + fees
    state.cash -= required
    state.fees_paid += fees
    state.traded_notional += notional
    if symbol in state.positions:
        position = state.positions[symbol]
        position.shares += quantity
        position.basis_cash += required
        position.last_price = float(raw_open)
    else:
        state.positions[symbol] = Position(
            shares=quantity,
            basis_cash=required,
            entry_date_idx=int(date_idx),
            last_price=float(raw_open),
        )
    return {
        "side": "buy",
        "symbol_idx": int(symbol),
        "shares": quantity,
        "raw_open": float(raw_open),
        "execution_price": execution_price,
        "notional": notional,
        "fees": fees,
    }


def _execute_targets(
    pack: QCurvePack,
    state: PortfolioState,
    *,
    execution_date_idx: int,
    targets: Mapping[int, float],
    contract: QCurveCostContract,
    slippage_multiplier: float,
) -> tuple[float, list[dict[str, Any]], float]:
    equity_before = _equity(pack, state, execution_date_idx, at_open=True)
    trades: list[dict[str, Any]] = []
    lot = int(contract.lot_size)
    desired: dict[int, int] = {}
    for symbol, weight in targets.items():
        if symbol in state.positions and not bool(pack.masks["open_sellable"][execution_date_idx, symbol]):
            desired[symbol] = int(state.positions[symbol].shares)
            continue
        raw_open = float(pack.execution["open_raw"][execution_date_idx, symbol])
        if not (math.isfinite(raw_open) and raw_open > 0.0 and float(weight) > 0.0):
            desired[symbol] = 0
            continue
        desired[symbol] = int(math.floor(float(weight) * equity_before / raw_open / lot) * lot)

    for symbol in list(state.positions):
        current = int(state.positions[symbol].shares)
        wanted = int(desired.get(symbol, 0))
        if wanted >= current or not bool(pack.masks["open_sellable"][execution_date_idx, symbol]):
            continue
        raw_open = float(pack.execution["open_raw"][execution_date_idx, symbol])
        trade = _sell_order(
            state,
            symbol=symbol,
            shares=current - wanted,
            raw_open=raw_open,
            trade_date=pack.date_values[execution_date_idx],
            date_idx=execution_date_idx,
            contract=contract,
            slippage_multiplier=slippage_multiplier,
        )
        if trade:
            trades.append(trade)

    for symbol, wanted in sorted(desired.items(), key=lambda item: (-float(targets[item[0]]), item[0])):
        current = int(state.positions[symbol].shares) if symbol in state.positions else 0
        if wanted <= current or not bool(pack.masks["open_buyable"][execution_date_idx, symbol]):
            continue
        if symbol not in state.positions and len(state.positions) >= 3:
            continue
        raw_open = float(pack.execution["open_raw"][execution_date_idx, symbol])
        trade = _buy_order(
            state,
            symbol=symbol,
            desired_shares=wanted,
            raw_open=raw_open,
            date_idx=execution_date_idx,
            contract=contract,
            slippage_multiplier=slippage_multiplier,
        )
        if trade:
            trades.append(trade)
    equity_after = _equity(pack, state, execution_date_idx, at_open=True)
    turnover = sum(float(item["notional"]) for item in trades) / max(equity_before, 1.0e-12)
    if len(state.positions) > 3:
        raise RuntimeError("portfolio exceeded the three-position hard cap")
    return equity_after, trades, float(turnover)


def _benchmark_log_return(pack: QCurvePack, previous_open_idx: int, current_open_idx: int) -> float:
    eligible = np.asarray(pack.masks["signal_eligible"][previous_open_idx - 1], dtype=bool)
    previous = np.asarray(pack.execution["open_raw"][previous_open_idx], dtype=np.float64)
    current = np.asarray(pack.execution["open_raw"][current_open_idx], dtype=np.float64)
    valid = eligible & np.isfinite(previous) & np.isfinite(current) & (previous > 0.0) & (current > 0.0)
    if not bool(valid.any()):
        return 0.0
    return float(np.mean(np.log(current[valid] / previous[valid])))


def _enter_action_values(mean: np.ndarray, q20: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
    horizons = np.asarray(QCURVE_ENTRY_HORIZONS, dtype=np.int16)
    holding_days = horizons.astype(np.float64) - 1.0
    eligible = np.isfinite(mean) & np.isfinite(q20) & (q20 > 0.0)
    rates = np.where(eligible, mean / holding_days[None, :], -np.inf)
    best_rates = np.max(rates, axis=1)
    positive = np.sort(best_rates[np.isfinite(best_rates) & (best_rates > 0.0)])[::-1]
    rho = max(0.0, float(positive[3])) if positive.size >= 4 else 0.0
    values = np.where(eligible, mean - rho * holding_days[None, :], -np.inf)
    best_idx = np.argmax(values, axis=1)
    best_value = values[np.arange(values.shape[0]), best_idx]
    return best_value, best_idx, rho


def _allocation_frame(
    *,
    pack: QCurvePack,
    date_idx: int,
    symbols: np.ndarray,
    predictions: Mapping[str, np.ndarray],
    state: PortfolioState,
) -> tuple[pd.DataFrame, float]:
    equity, weights = _current_weights(pack, state, date_idx)
    symbol_to_row = {int(symbol): idx for idx, symbol in enumerate(symbols)}
    rows: list[dict[str, Any]] = []
    for symbol in symbols:
        idx = symbol_to_row[int(symbol)]
        rows.append(
            {
                "symbol": str(int(symbol)),
                "symbol_idx": int(symbol),
                "is_held": int(symbol) in state.positions,
                "current_weight": float(weights.get(int(symbol), 0.0)),
                "sellable_next_open": bool(pack.masks["open_sellable"][date_idx + 1, int(symbol)]),
                "enter_mean_curve": predictions["enter_mean"][idx],
                "enter_q20_curve": predictions["enter_q20"][idx],
                "hold_mean_curve": predictions["hold_mean"][idx],
                "hold_q20_curve": predictions["hold_q20"][idx],
            }
        )
    return pd.DataFrame(rows), equity


def _replay_summary(state: PortfolioState, accumulator: ReplayAccumulator) -> dict[str, Any]:
    logs = np.asarray(accumulator.daily_log_returns, dtype=np.float64)
    benchmark = np.asarray(accumulator.benchmark_log_returns, dtype=np.float64)
    equities = np.asarray(accumulator.equities, dtype=np.float64)
    equity_path = np.r_[1_000_000.0, equities]
    running_max = np.maximum.accumulate(equity_path) if equity_path.size else np.asarray([], dtype=np.float64)
    drawdown = equity_path / np.maximum(running_max, 1.0e-12) - 1.0 if equity_path.size else np.asarray([])
    pnls = np.asarray(state.realized_pnls, dtype=np.float64)
    wins = pnls[pnls > 0.0]
    losses = pnls[pnls < 0.0]
    return {
        "ending_equity": float(equities[-1]) if equities.size else float(state.cash),
        "net_return": float(math.exp(float(logs.sum())) - 1.0) if logs.size else 0.0,
        "total_log_return": float(logs.sum()),
        "average_daily_log_return": float(logs.mean()) if logs.size else 0.0,
        "benchmark_total_log_return": float(benchmark.sum()),
        "alpha_total_log_return": float(logs.sum() - benchmark.sum()),
        "alpha_average_daily_log_return": float((logs - benchmark).mean()) if logs.size else 0.0,
        "maximum_drawdown": float(drawdown.min()) if drawdown.size else 0.0,
        "fees_paid": float(state.fees_paid),
        "turnover": float(np.sum(accumulator.turnover_values)),
        "average_cash_weight": float(np.mean(accumulator.cash_weights)) if accumulator.cash_weights else 1.0,
        "capital_utilization": float(1.0 - np.mean(accumulator.cash_weights)) if accumulator.cash_weights else 0.0,
        "average_position_count": float(np.mean(accumulator.position_counts)) if accumulator.position_counts else 0.0,
        "maximum_position_count": int(max(accumulator.position_counts, default=0)),
        "closed_trade_count": int(pnls.size),
        "win_rate": float(np.mean(pnls > 0.0)) if pnls.size else 0.0,
        "profit_loss_ratio": float(wins.mean() / abs(losses.mean())) if wins.size and losses.size else None,
        "average_holding_days": float(np.mean(state.holding_days)) if state.holding_days else 0.0,
    }


def evaluate_qcurve_fold(
    *,
    pack: QCurvePack,
    fold: Mapping[str, Any],
    provider: QCurvePredictionProvider,
) -> dict[str, Any]:
    contract = QCurveCostContract.from_manifest(pack.manifest)
    states = {"base": PortfolioState(), "stress": PortfolioState()}
    accumulators = {"base": ReplayAccumulator(), "stress": ReplayAccumulator()}
    topk_rows = {value: [] for value in (1, 3, 5, 10)}
    calibration = {
        action: {name: [0, 0] for name in ("q20", "q50", "q80")}
        for action in ("enter", "hold")
    }
    probability_square_error = 0.0
    probability_count = 0
    prediction_finite = 0
    prediction_total = 0
    oracle_regrets: list[float] = []
    capture_values: list[float] = []
    horizon_counts = np.zeros(len(QCURVE_ENTRY_HORIZONS), dtype=np.int64)
    previous_execution_idx: int | None = None

    for span in list(fold["development_spans"]):
        date_idx = int(span["date_idx"])
        start = int(span["candidate_start"])
        stop = int(span["candidate_stop"])
        symbols = pack.candidate_symbols(start, stop)
        union_symbols = np.asarray(sorted(set(symbols.tolist()).union(*[set(state.positions) for state in states.values()])), dtype=np.int32)
        predictions = {name: np.asarray(value) for name, value in provider.predict_symbols(date_idx, union_symbols).items()}
        candidate_lookup = np.searchsorted(union_symbols, symbols)
        candidate_predictions = {name: value[candidate_lookup] for name, value in predictions.items()}
        actual_enter, actual_hold = pack.q_targets(start, stop, cost="base")

        for action, actual in (("enter", actual_enter), ("hold", actual_hold)):
            for quantile in ("q20", "q50", "q80"):
                pred = candidate_predictions[f"{action}_{quantile}"]
                finite = np.isfinite(pred) & np.isfinite(actual)
                calibration[action][quantile][0] += int(np.count_nonzero((actual < pred) & finite))
                calibration[action][quantile][1] += int(np.count_nonzero(finite))
            probability = candidate_predictions[f"{action}_p_positive"]
            finite = np.isfinite(probability) & np.isfinite(actual)
            probability_square_error += float(np.square(probability[finite] - (actual[finite] > 0.0)).sum())
            probability_count += int(np.count_nonzero(finite))
        required_prediction_names = (
            "enter_mean",
            "enter_q20",
            "enter_q50",
            "enter_q80",
            "enter_p_positive",
            "hold_mean",
            "hold_q20",
            "hold_q50",
            "hold_q80",
            "hold_p_positive",
        )
        prediction_finite += sum(int(np.count_nonzero(np.isfinite(candidate_predictions[name]))) for name in required_prediction_names)
        prediction_total += sum(int(candidate_predictions[name].size) for name in required_prediction_names)

        values, best_idx, _ = _enter_action_values(
            candidate_predictions["enter_mean"],
            candidate_predictions["enter_q20"],
        )
        order = np.lexsort((symbols, -np.where(np.isfinite(values), values, -np.inf)))
        eligible_order = order[np.isfinite(values[order]) & (values[order] > 0.0)]
        selected_for_diagnostics = eligible_order[:3]
        if selected_for_diagnostics.size:
            selected_horizons = best_idx[selected_for_diagnostics]
            horizon_counts += np.bincount(selected_horizons, minlength=horizon_counts.size)
            realized = actual_enter[selected_for_diagnostics, selected_horizons]
            oracle = np.max(actual_enter[selected_for_diagnostics], axis=1)
            oracle_regrets.extend((oracle - realized).astype(float).tolist())
            all_oracle = np.maximum(np.max(actual_enter, axis=1), 0.0)
            oracle_top = np.sort(all_oracle)[-3:]
            denominator = float(oracle_top.sum())
            if denominator > 0.0:
                capture_values.append(float(np.maximum(oracle, 0.0).sum() / denominator))
        for topk in topk_rows:
            selected = eligible_order[:topk]
            if selected.size == 0:
                topk_rows[topk].append((0.0, 0.0))
                continue
            indices = best_idx[selected]
            realized = actual_enter[selected, indices]
            benchmark_values = np.asarray(
                [np.mean(actual_enter[:, horizon]) for horizon in indices],
                dtype=np.float64,
            )
            topk_rows[topk].append((float(np.mean(realized)), float(np.mean(realized - benchmark_values))))

        for cost_name, state in states.items():
            frame, equity_at_close = _allocation_frame(
                pack=pack,
                date_idx=date_idx,
                symbols=union_symbols,
                predictions=predictions,
                state=state,
            )
            allocated, diagnostics = allocate_dynamic_qcurve_portfolio(frame)
            multiplier = 1.0 if cost_name == "base" else float(contract.stress_slippage_multiplier)
            gated, gate = _apply_turnover_gate(
                allocated,
                equity=equity_at_close,
                contract=contract,
                slippage_multiplier=multiplier,
            )
            targets = {
                int(row.symbol_idx): float(row.target_weight)
                for row in gated.itertuples(index=False)
                if float(row.target_weight) > 1.0e-12
            }
            execution_idx = date_idx + 1
            equity_after, trades, turnover = _execute_targets(
                pack,
                state,
                execution_date_idx=execution_idx,
                targets=targets,
                contract=contract,
                slippage_multiplier=multiplier,
            )
            accumulator = accumulators[cost_name]
            previous_equity = accumulator.equities[-1] if accumulator.equities else 1_000_000.0
            accumulator.daily_log_returns.append(float(math.log(equity_after / previous_equity)))
            if accumulator.equities:
                assert previous_execution_idx is not None
                accumulator.benchmark_log_returns.append(
                    _benchmark_log_return(pack, previous_execution_idx, execution_idx)
                )
            else:
                accumulator.benchmark_log_returns.append(0.0)
            accumulator.equities.append(equity_after)
            accumulator.turnover_values.append(turnover)
            accumulator.position_counts.append(len(state.positions))
            accumulator.cash_weights.append(float(state.cash / max(equity_after, 1.0e-12)))
            accumulator.decision_rows.append(
                {
                    "signal_date_idx": date_idx,
                    "execution_date_idx": execution_idx,
                    "rho": float(diagnostics["rho"]),
                    "target_count": int(diagnostics["target_count"]),
                    "target_gross_exposure": float(gated["target_weight"].sum()),
                    **gate,
                }
            )
            for trade in trades:
                accumulator.trade_rows.append({"execution_date_idx": execution_idx, **trade})
        previous_execution_idx = date_idx + 1

    topk = {}
    for value, rows in topk_rows.items():
        array = np.asarray(rows, dtype=np.float64)
        topk[f"top{value}"] = {
            "cost_adjusted_absolute_mean_log_return": float(array[:, 0].mean()),
            "cost_adjusted_alpha_mean_log_return": float(array[:, 1].mean()),
            "positive_date_rate": float(np.mean(array[:, 0] > 0.0)),
            "date_count": int(array.shape[0]),
        }
    coverage = {
        action: {
            name: {
                "actual_breach_rate": numerator / max(denominator, 1),
                "count": denominator,
            }
            for name, (numerator, denominator) in values.items()
        }
        for action, values in calibration.items()
    }
    return {
        "schema_version": 1,
        "development_year": int(fold["development_year"]),
        "portfolio": {name: _replay_summary(states[name], accumulators[name]) for name in states},
        "topk": topk,
        "calibration": {
            "quantile_coverage": coverage,
            "positive_probability_brier": probability_square_error / max(probability_count, 1),
            "positive_probability_count": probability_count,
        },
        "diagnostics": {
            "prediction_coverage": prediction_finite / max(prediction_total, 1),
            "candidate_coverage": 1.0,
            "execution_mask_coverage": 1.0,
            "q_label_coverage": 1.0,
            "oracle_executable_regret_mean": float(np.mean(oracle_regrets)) if oracle_regrets else None,
            "opportunity_capture_rate_mean": float(np.mean(capture_values)) if capture_values else None,
            "recommended_exit_horizon_counts": {
                str(horizon): int(horizon_counts[idx])
                for idx, horizon in enumerate(QCURVE_ENTRY_HORIZONS)
                if horizon_counts[idx] > 0
            },
        },
        "daily": {name: accumulators[name].decision_rows for name in accumulators},
        "trades": {name: accumulators[name].trade_rows for name in accumulators},
    }
