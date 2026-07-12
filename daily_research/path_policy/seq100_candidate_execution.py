from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd


DEFAULT_DAILY_COHORT_CASH_CNY = 1_000_000.0
DEFAULT_TOP_K_VALUES = (1, 3, 5, 10)
REQUIRED_LOT_SIZE = 100
REQUIRED_BASE_SLIPPAGE_BPS = 7.0
REQUIRED_STRESS_SLIPPAGE_MULTIPLIER = 2.0


@dataclass(frozen=True)
class ExecutionCostContract:
    lot_size: int
    commission_bps: float
    minimum_commission_cny: float
    transfer_fee_bps: float
    slippage_bps: float
    stress_slippage_multiplier: float
    stamp_tax_schedule: tuple[tuple[str, float], ...]
    semantic_sha256: str


@dataclass(frozen=True)
class CandidateExecutionEvaluation:
    candidates: pd.DataFrame
    daily_topk: pd.DataFrame
    execution_cost_contract_sha256: str


@dataclass(frozen=True)
class _ResolvedPlan:
    planned_day: int | None
    exit_day: int | None
    exit_date: str | None
    exit_price: float | None
    status: str
    covered: bool
    terminal_recovery: bool


@dataclass(frozen=True)
class _Cashflow:
    order_filled: bool
    shares: int
    ending_cash: float
    net_return: float
    total_cost: float
    cash_utilization: float


def _canonical_json(payload: Any) -> str:
    return json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def execution_cost_contract_sha256(manifest_or_contract: Mapping[str, Any]) -> str:
    """Hash the manifest-bound cost semantics independently of JSON key order."""

    payload = dict(manifest_or_contract)
    contract = payload.get("execution_cost_contract", payload)
    if not isinstance(contract, Mapping) or not contract:
        raise ValueError("manifest is missing execution_cost_contract")
    return hashlib.sha256(_canonical_json(dict(contract)).encode("utf-8")).hexdigest()


def parse_execution_cost_contract(manifest: Mapping[str, Any]) -> ExecutionCostContract:
    raw = manifest.get("execution_cost_contract")
    if not isinstance(raw, Mapping) or not raw:
        raise ValueError("manifest is missing execution_cost_contract")
    contract = dict(raw)

    def finite_non_negative(name: str, default: float | None = None) -> float:
        value = contract.get(name, default)
        if value is None:
            raise ValueError(f"execution_cost_contract is missing {name}")
        number = float(value)
        if not math.isfinite(number) or number < 0.0:
            raise ValueError(f"execution_cost_contract.{name} must be finite and non-negative")
        return number

    lot_size = int(contract.get("lot_size", 0))
    if lot_size != REQUIRED_LOT_SIZE:
        raise ValueError(f"seq100 execution requires lot_size={REQUIRED_LOT_SIZE}")
    schedule_raw = contract.get("stamp_tax_schedule")
    if not schedule_raw:
        schedule_raw = [
            {
                "effective_date": "1900-01-01",
                "stamp_tax_bps": finite_non_negative("stamp_tax_bps", 0.0),
            }
        ]
    if not isinstance(schedule_raw, Sequence) or isinstance(schedule_raw, (str, bytes)):
        raise ValueError("execution_cost_contract.stamp_tax_schedule must be a sequence")
    schedule: list[tuple[str, float]] = []
    previous = ""
    for item in schedule_raw:
        if isinstance(item, Mapping):
            effective_date = item.get("effective_date")
            rate_value = item.get("stamp_tax_bps")
        elif isinstance(item, Sequence) and not isinstance(item, (str, bytes)) and len(item) == 2:
            effective_date, rate_value = item
        else:
            raise ValueError("invalid stamp-tax schedule entry")
        try:
            date_text = pd.Timestamp(str(effective_date)).strftime("%Y-%m-%d")
        except (TypeError, ValueError) as exc:
            raise ValueError(f"invalid stamp-tax effective date: {effective_date}") from exc
        rate = float(rate_value)
        if not math.isfinite(rate) or rate < 0.0:
            raise ValueError("stamp-tax rates must be finite and non-negative")
        if previous and date_text <= previous:
            raise ValueError("stamp-tax schedule must be strictly increasing")
        schedule.append((date_text, rate))
        previous = date_text

    stress_multiplier = finite_non_negative("stress_slippage_multiplier", 2.0)
    if not math.isclose(
        stress_multiplier,
        REQUIRED_STRESS_SLIPPAGE_MULTIPLIER,
        rel_tol=0.0,
        abs_tol=1.0e-12,
    ):
        raise ValueError(
            "seq100 execution requires stress_slippage_multiplier="
            f"{REQUIRED_STRESS_SLIPPAGE_MULTIPLIER}"
        )
    slippage_bps = finite_non_negative("slippage_bps", REQUIRED_BASE_SLIPPAGE_BPS)
    if not math.isclose(
        slippage_bps,
        REQUIRED_BASE_SLIPPAGE_BPS,
        rel_tol=0.0,
        abs_tol=1.0e-12,
    ):
        raise ValueError(f"seq100 execution requires slippage_bps={REQUIRED_BASE_SLIPPAGE_BPS}")
    return ExecutionCostContract(
        lot_size=lot_size,
        commission_bps=finite_non_negative("commission_bps"),
        minimum_commission_cny=finite_non_negative("minimum_commission_cny"),
        transfer_fee_bps=finite_non_negative("transfer_fee_bps"),
        slippage_bps=slippage_bps,
        stress_slippage_multiplier=stress_multiplier,
        stamp_tax_schedule=tuple(schedule),
        semantic_sha256=execution_cost_contract_sha256(manifest),
    )


def _terminal_recovery_fraction(manifest: Mapping[str, Any]) -> float:
    terminal = manifest.get("terminal_execution_contract", {})
    if terminal is None:
        terminal = {}
    if not isinstance(terminal, Mapping):
        raise ValueError("terminal_execution_contract must be an object")
    value = float(terminal.get("recovery_fraction_of_entry_notional", 0.0))
    if not math.isfinite(value) or not 0.0 <= value <= 1.0:
        raise ValueError("terminal recovery fraction must be within [0, 1]")
    return value


def _normalize_date(value: Any, *, name: str) -> str:
    try:
        timestamp = pd.Timestamp(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"invalid {name}: {value}") from exc
    if pd.isna(timestamp):
        raise ValueError(f"invalid {name}: {value}")
    return timestamp.strftime("%Y-%m-%d")


def _coerce_required_bool(series: pd.Series, *, name: str) -> np.ndarray:
    values: list[bool] = []
    for value in series.tolist():
        if isinstance(value, (bool, np.bool_)):
            values.append(bool(value))
            continue
        if isinstance(value, (int, float, np.integer, np.floating)) and math.isfinite(float(value)):
            if float(value) in (0.0, 1.0):
                values.append(bool(int(value)))
                continue
        text = str(value).strip().lower()
        if text in {"true", "t", "yes", "y", "1"}:
            values.append(True)
            continue
        if text in {"false", "f", "no", "n", "0"}:
            values.append(False)
            continue
        raise ValueError(f"{name} contains a non-boolean value: {value}")
    return np.asarray(values, dtype=bool)


def _exit_date_lookup(
    *,
    candidate_count: int,
    horizon: int,
    candidate_frame: pd.DataFrame,
    exit_trade_date_path: np.ndarray | Sequence[Sequence[Any]] | None,
    signal_date_idx: Sequence[int] | np.ndarray | None,
    date_values: Sequence[Any] | np.ndarray | None,
):
    if exit_trade_date_path is not None:
        date_path = np.asarray(exit_trade_date_path)
        if date_path.ndim != 2 or date_path.shape[0] != candidate_count or date_path.shape[1] < horizon:
            raise ValueError("exit_trade_date_path must have shape [candidate, forward+tail]")

        def from_path(row: int, day: int) -> str:
            return _normalize_date(date_path[row, day - 1], name="exit trade date")

        return from_path

    indices_source: Sequence[int] | np.ndarray | pd.Series | None = signal_date_idx
    if indices_source is None and "signal_date_idx" in candidate_frame.columns:
        indices_source = candidate_frame["signal_date_idx"]
    if indices_source is None or date_values is None:
        raise ValueError(
            "provide exit_trade_date_path or both signal_date_idx and global date_values"
        )
    indices = pd.to_numeric(pd.Series(indices_source), errors="coerce").to_numpy(dtype=np.float64)
    if indices.shape != (candidate_count,) or not bool(np.isfinite(indices).all()):
        raise ValueError("signal_date_idx must contain one finite integer per candidate")
    rounded = np.rint(indices).astype(np.int64)
    if not bool(np.equal(indices, rounded).all()):
        raise ValueError("signal_date_idx values must be integers")
    normalized_dates = tuple(_normalize_date(value, name="global date") for value in date_values)
    if any(int(index) < 0 or int(index) + horizon >= len(normalized_dates) for index in rounded):
        raise ValueError("global date_values does not cover the complete exit window")

    def from_global(row: int, day: int) -> str:
        return normalized_dates[int(rounded[row]) + int(day)]

    return from_global


def _stamp_tax_bps(contract: ExecutionCostContract, exit_date: str) -> float:
    matches = [rate for effective_date, rate in contract.stamp_tax_schedule if effective_date <= exit_date]
    if not matches:
        raise ValueError("exit trade date predates the stamp-tax schedule")
    return float(matches[-1])


def _resolve_plan(
    *,
    predicted_exit_day: float,
    entry_filled: bool,
    entry_price: float,
    exit_prices: np.ndarray,
    exit_sellable: np.ndarray,
    forward_days: int,
    horizon: int,
    terminal_recovery_fraction: float,
    date_for_day,
    row: int,
) -> _ResolvedPlan:
    if not entry_filled:
        return _ResolvedPlan(None, None, None, None, "entry_unfilled_cash", True, False)
    if not math.isfinite(entry_price) or entry_price <= 0.0:
        return _ResolvedPlan(None, None, None, None, "invalid_entry_cash", True, False)
    if math.isfinite(predicted_exit_day):
        planned_day = max(2, min(int(round(predicted_exit_day)), int(forward_days)))
        candidate_days = range(planned_day, horizon + 1)
        for day in candidate_days:
            exit_price = float(exit_prices[day - 1])
            if bool(exit_sellable[day - 1]) and math.isfinite(exit_price) and exit_price >= 0.0:
                status = "filled_planned_exit" if day == planned_day else "filled_deferred_exit"
                return _ResolvedPlan(
                    planned_day,
                    day,
                    date_for_day(row, day),
                    exit_price,
                    status,
                    True,
                    False,
                )
    else:
        planned_day = None
    terminal_date = date_for_day(row, horizon)
    return _ResolvedPlan(
        planned_day,
        horizon,
        terminal_date,
        entry_price * terminal_recovery_fraction,
        "terminal_recovery",
        True,
        True,
    )


def _cashflow(
    *,
    allocated_cash: float,
    entry_filled: bool,
    entry_price: float,
    plan: _ResolvedPlan,
    contract: ExecutionCostContract,
    slippage_multiplier: float,
) -> _Cashflow:
    cash = float(allocated_cash)
    if not math.isfinite(cash) or cash <= 0.0:
        raise ValueError("allocated cash must be finite and positive")
    if (
        not entry_filled
        or not math.isfinite(entry_price)
        or entry_price <= 0.0
        or plan.exit_price is None
        or plan.exit_date is None
    ):
        return _Cashflow(False, 0, cash, 0.0, 0.0, 0.0)
    multiplier = float(slippage_multiplier)
    if not math.isfinite(multiplier) or multiplier < 0.0:
        raise ValueError("slippage multiplier must be finite and non-negative")
    slippage_rate = contract.slippage_bps * multiplier / 10_000.0
    buy_price = entry_price * (1.0 + slippage_rate)
    sell_price = float(plan.exit_price) * max(0.0, 1.0 - slippage_rate)
    commission_rate = contract.commission_bps / 10_000.0
    transfer_rate = contract.transfer_fee_bps / 10_000.0
    shares = int(cash // (buy_price * contract.lot_size)) * contract.lot_size
    while shares > 0:
        buy_notional = shares * buy_price
        buy_commission = max(contract.minimum_commission_cny, buy_notional * commission_rate)
        buy_transfer = buy_notional * transfer_rate
        buy_cash = buy_notional + buy_commission + buy_transfer
        if buy_cash <= cash + 1.0e-9:
            break
        shares -= contract.lot_size
    if shares <= 0:
        return _Cashflow(False, 0, cash, 0.0, 0.0, 0.0)

    buy_notional = shares * buy_price
    buy_commission = max(contract.minimum_commission_cny, buy_notional * commission_rate)
    buy_transfer = buy_notional * transfer_rate
    buy_cash = buy_notional + buy_commission + buy_transfer
    sell_notional = shares * sell_price
    sell_commission = max(contract.minimum_commission_cny, sell_notional * commission_rate)
    sell_transfer = sell_notional * transfer_rate
    stamp_tax = sell_notional * _stamp_tax_bps(contract, plan.exit_date) / 10_000.0
    ending_cash = cash - buy_cash + sell_notional - sell_commission - sell_transfer - stamp_tax
    explicit_cost = buy_commission + buy_transfer + sell_commission + sell_transfer + stamp_tax
    slippage_cost = shares * ((buy_price - entry_price) + (float(plan.exit_price) - sell_price))
    return _Cashflow(
        True,
        int(shares),
        float(ending_cash),
        float(ending_cash / cash - 1.0),
        float(explicit_cost + slippage_cost),
        float(buy_cash / cash),
    )


def _gross_return(*, entry_filled: bool, entry_price: float, plan: _ResolvedPlan, order_filled: bool) -> float:
    if not entry_filled or not order_filled or plan.exit_price is None:
        return 0.0
    return float(plan.exit_price / entry_price - 1.0)


def _net_value(gross_value: float, gross_return: float, cashflow: _Cashflow) -> float:
    if not math.isfinite(gross_value):
        return math.nan
    if not cashflow.order_filled:
        return 0.0
    return float(gross_value + cashflow.net_return - gross_return)


def _evaluate_rows_for_allocation(
    *,
    frame: pd.DataFrame,
    plans: Sequence[_ResolvedPlan],
    entry_filled: np.ndarray,
    entry_prices: np.ndarray,
    gross_values: np.ndarray | None,
    allocated_cash: float,
    contract: ExecutionCostContract,
) -> dict[str, np.ndarray]:
    count = len(frame)
    output: dict[str, np.ndarray] = {
        "gross_return": np.empty(count, dtype=np.float64),
        "net_return_base": np.empty(count, dtype=np.float64),
        "net_return_stress": np.empty(count, dtype=np.float64),
        "net_value_base": np.full(count, np.nan, dtype=np.float64),
        "net_value_stress": np.full(count, np.nan, dtype=np.float64),
        "filled_base": np.empty(count, dtype=bool),
        "filled_stress": np.empty(count, dtype=bool),
        "shares_base": np.empty(count, dtype=np.int64),
        "shares_stress": np.empty(count, dtype=np.int64),
        "cost_base": np.empty(count, dtype=np.float64),
        "cost_stress": np.empty(count, dtype=np.float64),
        "cash_utilization_base": np.empty(count, dtype=np.float64),
        "cash_utilization_stress": np.empty(count, dtype=np.float64),
        "ending_cash_base": np.empty(count, dtype=np.float64),
        "ending_cash_stress": np.empty(count, dtype=np.float64),
    }
    for row, plan in enumerate(plans):
        base = _cashflow(
            allocated_cash=allocated_cash,
            entry_filled=bool(entry_filled[row]),
            entry_price=float(entry_prices[row]),
            plan=plan,
            contract=contract,
            slippage_multiplier=1.0,
        )
        stress = _cashflow(
            allocated_cash=allocated_cash,
            entry_filled=bool(entry_filled[row]),
            entry_price=float(entry_prices[row]),
            plan=plan,
            contract=contract,
            slippage_multiplier=contract.stress_slippage_multiplier,
        )
        gross = _gross_return(
            entry_filled=bool(entry_filled[row]),
            entry_price=float(entry_prices[row]),
            plan=plan,
            order_filled=base.order_filled,
        )
        output["gross_return"][row] = gross
        output["net_return_base"][row] = base.net_return
        output["net_return_stress"][row] = stress.net_return
        output["filled_base"][row] = base.order_filled
        output["filled_stress"][row] = stress.order_filled
        output["shares_base"][row] = base.shares
        output["shares_stress"][row] = stress.shares
        output["cost_base"][row] = base.total_cost
        output["cost_stress"][row] = stress.total_cost
        output["cash_utilization_base"][row] = base.cash_utilization
        output["cash_utilization_stress"][row] = stress.cash_utilization
        output["ending_cash_base"][row] = base.ending_cash
        output["ending_cash_stress"][row] = stress.ending_cash
        if gross_values is not None:
            output["net_value_base"][row] = _net_value(float(gross_values[row]), gross, base)
            output["net_value_stress"][row] = _net_value(float(gross_values[row]), gross, stress)
    return output


def _mean_complete(values: np.ndarray) -> float:
    numeric = np.asarray(values, dtype=np.float64)
    return float(numeric.mean()) if numeric.size and bool(np.isfinite(numeric).all()) else math.nan


def _mean_finite(values: np.ndarray) -> float:
    numeric = np.asarray(values, dtype=np.float64)
    finite = numeric[np.isfinite(numeric)]
    return float(finite.mean()) if finite.size else math.nan


def _daily_topk_rows(
    *,
    frame: pd.DataFrame,
    plans: Sequence[_ResolvedPlan],
    entry_filled: np.ndarray,
    entry_prices: np.ndarray,
    gross_values: np.ndarray | None,
    contract: ExecutionCostContract,
    top_k_values: tuple[int, ...],
    daily_cohort_cash_cny: float,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    date_text = frame["trade_date"].map(lambda value: _normalize_date(value, name="trade_date"))
    for trade_date in sorted(date_text.unique()):
        positions = np.flatnonzero(date_text.to_numpy(dtype=object) == trade_date)
        date_frame = frame.iloc[positions].reset_index(drop=True).copy()
        date_frame["_position"] = np.arange(len(date_frame), dtype=np.int64)
        date_frame["_score"] = pd.to_numeric(date_frame["score"], errors="coerce")
        date_frame["_symbol"] = date_frame["symbol"].astype(str)
        date_plans = tuple(plans[position] for position in positions)
        date_entry_filled = entry_filled[positions]
        date_entry_prices = entry_prices[positions]
        date_gross_values = gross_values[positions] if gross_values is not None else None
        ranked = date_frame.sort_values(
            ["_score", "_symbol"],
            ascending=[False, True],
            kind="mergesort",
            na_position="last",
        )
        ranked_positions = ranked["_position"].to_numpy(dtype=np.int64)
        universe_symbols = ranked["_symbol"].tolist()
        universe_hash = hashlib.sha256("\n".join(sorted(universe_symbols)).encode("utf-8")).hexdigest()
        for top_k in top_k_values:
            allocation = float(daily_cohort_cash_cny) / int(top_k)
            scenario = _evaluate_rows_for_allocation(
                frame=date_frame,
                plans=date_plans,
                entry_filled=date_entry_filled,
                entry_prices=date_entry_prices,
                gross_values=date_gross_values,
                allocated_cash=allocation,
                contract=contract,
            )
            selected_positions = ranked_positions[: min(int(top_k), len(ranked_positions))]
            selected_symbols = ranked.loc[ranked["_position"].isin(selected_positions), "_symbol"].tolist()
            selected_count = int(len(selected_positions))
            unused_slots = int(top_k) - selected_count

            def selected_portfolio_return(name: str) -> float:
                ending = float(scenario[name][selected_positions].sum()) + unused_slots * allocation
                return float(ending / daily_cohort_cash_cny - 1.0)

            selected_base = selected_portfolio_return("ending_cash_base")
            selected_stress = selected_portfolio_return("ending_cash_stress")
            universe_base = _mean_complete(scenario["net_return_base"][ranked_positions])
            universe_stress = _mean_complete(scenario["net_return_stress"][ranked_positions])
            # Value is a label-derived diagnostic, not a candidate-complete cash outcome.
            # Aggregate only finite labels and expose coverage explicitly; missing labels
            # must never become zero-valued paths.
            selected_value_base = _mean_finite(scenario["net_value_base"][selected_positions])
            selected_value_stress = _mean_finite(scenario["net_value_stress"][selected_positions])
            universe_value_base = _mean_finite(scenario["net_value_base"][ranked_positions])
            universe_value_stress = _mean_finite(scenario["net_value_stress"][ranked_positions])
            selected_value_coverage = (
                float(np.isfinite(scenario["net_value_base"][selected_positions]).mean())
                if selected_count
                else 1.0
            )
            universe_value_coverage = float(
                np.isfinite(scenario["net_value_base"][ranked_positions]).mean()
            )
            selected_coverage = float(np.mean([date_plans[pos].covered for pos in selected_positions]))
            universe_coverage = float(np.mean([date_plans[pos].covered for pos in ranked_positions]))
            selected_fill_count_base = int(scenario["filled_base"][selected_positions].sum())
            selected_fill_count_stress = int(scenario["filled_stress"][selected_positions].sum())
            universe_fill_count_base = int(scenario["filled_base"][ranked_positions].sum())
            universe_fill_count_stress = int(scenario["filled_stress"][ranked_positions].sum())
            selected_fill_base = float(selected_fill_count_base / int(top_k))
            selected_fill_stress = float(selected_fill_count_stress / int(top_k))
            universe_fill_base = float(scenario["filled_base"][ranked_positions].mean())
            universe_fill_stress = float(scenario["filled_stress"][ranked_positions].mean())
            selected_terminal_base = int(
                sum(date_plans[pos].terminal_recovery and bool(scenario["filled_base"][pos]) for pos in selected_positions)
            )
            selected_terminal_stress = int(
                sum(date_plans[pos].terminal_recovery and bool(scenario["filled_stress"][pos]) for pos in selected_positions)
            )
            universe_terminal_base = int(
                sum(date_plans[pos].terminal_recovery and bool(scenario["filled_base"][pos]) for pos in ranked_positions)
            )
            universe_terminal_stress = int(
                sum(date_plans[pos].terminal_recovery and bool(scenario["filled_stress"][pos]) for pos in ranked_positions)
            )
            selected_cash_base = float(scenario["cash_utilization_base"][selected_positions].sum() / int(top_k))
            selected_cash_stress = float(scenario["cash_utilization_stress"][selected_positions].sum() / int(top_k))
            universe_cash_base = float(scenario["cash_utilization_base"][ranked_positions].mean())
            universe_cash_stress = float(scenario["cash_utilization_stress"][ranked_positions].mean())
            selected_cost_total_base = float(scenario["cost_base"][selected_positions].sum())
            selected_cost_total_stress = float(scenario["cost_stress"][selected_positions].sum())
            selected_cost_base = selected_cost_total_base / int(top_k)
            selected_cost_stress = selected_cost_total_stress / int(top_k)
            universe_cost_base = float(scenario["cost_base"][ranked_positions].mean())
            universe_cost_stress = float(scenario["cost_stress"][ranked_positions].mean())
            selected_cash_retained_base = int(top_k) - int(scenario["filled_base"][selected_positions].sum())
            selected_cash_retained_stress = int(top_k) - int(scenario["filled_stress"][selected_positions].sum())
            universe_cash_retained_base = int((~scenario["filled_base"][ranked_positions]).sum())
            universe_cash_retained_stress = int((~scenario["filled_stress"][ranked_positions]).sum())
            selected_status_counts = {
                status: int(sum(date_plans[pos].status == status for pos in selected_positions))
                for status in sorted({date_plans[pos].status for pos in selected_positions})
            }
            universe_status_counts = {
                status: int(sum(date_plans[pos].status == status for pos in ranked_positions))
                for status in sorted({date_plans[pos].status for pos in ranked_positions})
            }
            row: dict[str, Any] = {
                "trade_date": trade_date,
                "top_k": int(top_k),
                "selected_count": selected_count,
                "universe_count": int(len(ranked_positions)),
                "unused_cash_slot_count": unused_slots,
                "selected_symbols": selected_symbols,
                "selected_symbols_json": json.dumps(selected_symbols, ensure_ascii=False, separators=(",", ":")),
                "universe_hash": universe_hash,
                "daily_cohort_cash_cny": float(daily_cohort_cash_cny),
                "per_name_allocation_cash_cny": allocation,
                "execution_cost_contract_sha256": contract.semantic_sha256,
                "selected_net_realized_plan_return_base": selected_base,
                "universe_net_realized_plan_return_base": universe_base,
                "alpha_net_realized_plan_return_base": selected_base - universe_base,
                "selected_net_realized_plan_return_stress": selected_stress,
                "universe_net_realized_plan_return_stress": universe_stress,
                "alpha_net_realized_plan_return_stress": selected_stress - universe_stress,
                "selected_net_realized_plan_value_base": selected_value_base,
                "universe_net_realized_plan_value_base": universe_value_base,
                "alpha_net_realized_plan_value_base": selected_value_base - universe_value_base,
                "selected_net_realized_plan_value_stress": selected_value_stress,
                "universe_net_realized_plan_value_stress": universe_value_stress,
                "alpha_net_realized_plan_value_stress": selected_value_stress - universe_value_stress,
                "selected_realized_plan_coverage": selected_coverage,
                "universe_realized_plan_coverage": universe_coverage,
                "alpha_realized_plan_coverage": selected_coverage - universe_coverage,
                "selected_realized_plan_covered_count": int(
                    sum(date_plans[pos].covered for pos in selected_positions)
                ),
                "universe_realized_plan_covered_count": int(
                    sum(date_plans[pos].covered for pos in ranked_positions)
                ),
                "alpha_realized_plan_covered_count": int(
                    sum(date_plans[pos].covered for pos in selected_positions)
                    - sum(date_plans[pos].covered for pos in ranked_positions)
                ),
                "selected_realized_plan_return_coverage": selected_coverage,
                "universe_realized_plan_return_coverage": universe_coverage,
                "selected_realized_plan_value_coverage": selected_value_coverage,
                "universe_realized_plan_value_coverage": universe_value_coverage,
                "selected_entry_fill_rate": selected_fill_base,
                "universe_entry_fill_rate": universe_fill_base,
                "alpha_entry_fill_rate": selected_fill_base - universe_fill_base,
                "selected_entry_fill_count_base": selected_fill_count_base,
                "universe_entry_fill_count_base": universe_fill_count_base,
                "alpha_entry_fill_count_base": selected_fill_count_base - universe_fill_count_base,
                "selected_entry_fill_rate_stress": selected_fill_stress,
                "universe_entry_fill_rate_stress": universe_fill_stress,
                "alpha_entry_fill_rate_stress": selected_fill_stress - universe_fill_stress,
                "selected_entry_fill_count_stress": selected_fill_count_stress,
                "universe_entry_fill_count_stress": universe_fill_count_stress,
                "alpha_entry_fill_count_stress": selected_fill_count_stress - universe_fill_count_stress,
                "selected_cash_utilization_base": selected_cash_base,
                "universe_cash_utilization_base": universe_cash_base,
                "alpha_cash_utilization_base": selected_cash_base - universe_cash_base,
                "selected_cash_utilization_stress": selected_cash_stress,
                "universe_cash_utilization_stress": universe_cash_stress,
                "alpha_cash_utilization_stress": selected_cash_stress - universe_cash_stress,
                "selected_execution_cost_base_cny": selected_cost_base,
                "universe_execution_cost_base_cny": universe_cost_base,
                "alpha_execution_cost_base_cny": selected_cost_base - universe_cost_base,
                "selected_execution_cost_stress_cny": selected_cost_stress,
                "universe_execution_cost_stress_cny": universe_cost_stress,
                "alpha_execution_cost_stress_cny": selected_cost_stress - universe_cost_stress,
                "selected_execution_cost_total_base_cny": selected_cost_total_base,
                "selected_execution_cost_total_stress_cny": selected_cost_total_stress,
                "selected_cash_retained_count_base": selected_cash_retained_base,
                "universe_cash_retained_count_base": universe_cash_retained_base,
                "alpha_cash_retained_count_base": selected_cash_retained_base - universe_cash_retained_base,
                "selected_cash_retained_count_stress": selected_cash_retained_stress,
                "universe_cash_retained_count_stress": universe_cash_retained_stress,
                "alpha_cash_retained_count_stress": selected_cash_retained_stress - universe_cash_retained_stress,
                "selected_terminal_recovery_count_base": selected_terminal_base,
                "universe_terminal_recovery_count_base": universe_terminal_base,
                "alpha_terminal_recovery_count_base": selected_terminal_base - universe_terminal_base,
                "selected_terminal_recovery_count_stress": selected_terminal_stress,
                "universe_terminal_recovery_count_stress": universe_terminal_stress,
                "alpha_terminal_recovery_count_stress": selected_terminal_stress - universe_terminal_stress,
                "selected_exit_status_counts_json": json.dumps(
                    selected_status_counts, ensure_ascii=False, sort_keys=True, separators=(",", ":")
                ),
                "universe_exit_status_counts_json": json.dumps(
                    universe_status_counts, ensure_ascii=False, sort_keys=True, separators=(",", ":")
                ),
            }
            rows.append(row)
    return rows


def evaluate_candidate_execution(
    candidate_frame: pd.DataFrame,
    exit_close_raw_path: np.ndarray | Sequence[Sequence[float]],
    exit_sellable_path: np.ndarray | Sequence[Sequence[bool]],
    *,
    manifest: Mapping[str, Any],
    exit_trade_date_path: np.ndarray | Sequence[Sequence[Any]] | None = None,
    signal_date_idx: Sequence[int] | np.ndarray | None = None,
    date_values: Sequence[Any] | np.ndarray | None = None,
    top_k_values: Sequence[int] = DEFAULT_TOP_K_VALUES,
    daily_cohort_cash_cny: float = DEFAULT_DAILY_COHORT_CASH_CNY,
    legacy_gross_value_column: str | None = None,
) -> CandidateExecutionEvaluation:
    """Evaluate a candidate-complete A-share execution cohort without dropping rows.

    The returned candidate columns use ``daily_cohort_cash_cny / max(top_k_values)`` as
    their canonical per-name allocation.  ``daily_topk`` recomputes cashflows for every K
    using exactly ``daily_cohort_cash_cny / K``; this is necessary because board lots and
    minimum commissions make net returns allocation-dependent.
    """

    required = {
        "trade_date",
        "symbol",
        "score",
        "predicted_exit_day",
        "entry_filled",
        "entry_open_raw",
    }
    missing = sorted(required.difference(candidate_frame.columns))
    if missing:
        raise ValueError(f"candidate frame is missing required columns: {missing}")
    count = int(len(candidate_frame))
    if count <= 0:
        raise ValueError("candidate frame must not be empty")
    forward_days = int(manifest.get("forward_days", 0))
    execution_tail_days = int(manifest.get("execution_tail_days", 0))
    if forward_days <= 0 or execution_tail_days < 0:
        raise ValueError("manifest forward_days/execution_tail_days are invalid")
    horizon = forward_days + execution_tail_days
    if horizon < 2:
        raise ValueError("execution window must include path day 2 for T+1")
    close_path = np.asarray(exit_close_raw_path)
    sellable_path = np.asarray(exit_sellable_path)
    expected_shape = (count, horizon)
    if close_path.ndim != 2 or close_path.shape[0] != count or close_path.shape[1] < horizon:
        raise ValueError("exit_close_raw_path must have shape [candidate, forward+tail]")
    if sellable_path.ndim != 2 or sellable_path.shape[0] != count or sellable_path.shape[1] < horizon:
        raise ValueError("exit_sellable_path must have shape [candidate, forward+tail]")
    close_path = close_path[:, :horizon]
    sellable_path = sellable_path[:, :horizon].astype(bool, copy=False)
    if close_path.shape != expected_shape or sellable_path.shape != expected_shape:
        raise AssertionError("execution paths were not normalized to the manifest horizon")
    normalized_top_k = tuple(sorted({int(value) for value in top_k_values}))
    if not normalized_top_k or any(value <= 0 for value in normalized_top_k):
        raise ValueError("top_k_values must contain positive integers")
    cohort_cash = float(daily_cohort_cash_cny)
    if not math.isfinite(cohort_cash) or cohort_cash <= 0.0:
        raise ValueError("daily_cohort_cash_cny must be finite and positive")

    frame = candidate_frame.reset_index(drop=True).copy()
    frame["trade_date"] = frame["trade_date"].map(lambda value: _normalize_date(value, name="trade_date"))
    frame["symbol"] = frame["symbol"].astype(str)
    if bool(frame.duplicated(["trade_date", "symbol"]).any()):
        raise ValueError("candidate frame must contain at most one row per trade_date/symbol")
    entry_filled = _coerce_required_bool(frame["entry_filled"], name="entry_filled")
    entry_prices = pd.to_numeric(frame["entry_open_raw"], errors="coerce").to_numpy(dtype=np.float64)
    predicted_days = pd.to_numeric(frame["predicted_exit_day"], errors="coerce").to_numpy(dtype=np.float64)
    date_for_day = _exit_date_lookup(
        candidate_count=count,
        horizon=horizon,
        candidate_frame=frame,
        exit_trade_date_path=exit_trade_date_path,
        signal_date_idx=signal_date_idx,
        date_values=date_values,
    )
    terminal_recovery = _terminal_recovery_fraction(manifest)
    contract = parse_execution_cost_contract(manifest)
    plans = tuple(
        _resolve_plan(
            predicted_exit_day=float(predicted_days[row]),
            entry_filled=bool(entry_filled[row]),
            entry_price=float(entry_prices[row]),
            exit_prices=close_path[row],
            exit_sellable=sellable_path[row],
            forward_days=forward_days,
            horizon=horizon,
            terminal_recovery_fraction=terminal_recovery,
            date_for_day=date_for_day,
            row=row,
        )
        for row in range(count)
    )
    if legacy_gross_value_column is None:
        legacy_gross_value_column = next(
            (name for name in ("gross_realized_plan_value", "realized_plan_value") if name in frame.columns),
            None,
        )
    if legacy_gross_value_column is not None and legacy_gross_value_column not in frame.columns:
        raise ValueError(f"legacy gross value column does not exist: {legacy_gross_value_column}")
    gross_values = (
        pd.to_numeric(frame[legacy_gross_value_column], errors="coerce").to_numpy(dtype=np.float64)
        if legacy_gross_value_column is not None
        else None
    )
    canonical_allocation = cohort_cash / max(normalized_top_k)
    canonical = _evaluate_rows_for_allocation(
        frame=frame,
        plans=plans,
        entry_filled=entry_filled,
        entry_prices=entry_prices,
        gross_values=gross_values,
        allocated_cash=canonical_allocation,
        contract=contract,
    )
    frame["gross_realized_plan_return"] = canonical["gross_return"]
    frame["realized_plan_gross_return"] = canonical["gross_return"]
    if "realized_plan_return" not in frame.columns:
        frame["realized_plan_return"] = canonical["gross_return"]
    frame["net_realized_plan_return_base"] = canonical["net_return_base"]
    frame["net_realized_plan_return_stress"] = canonical["net_return_stress"]
    frame["net_realized_plan_value_base"] = canonical["net_value_base"]
    frame["net_realized_plan_value_stress"] = canonical["net_value_stress"]
    frame["realized_plan_covered"] = np.asarray([plan.covered for plan in plans], dtype=bool)
    frame["realized_plan_entry_filled"] = entry_filled
    frame["realized_plan_planned_exit_day"] = pd.array(
        [plan.planned_day for plan in plans], dtype="Int64"
    )
    frame["realized_plan_resolved_exit_day"] = pd.array(
        [plan.exit_day for plan in plans], dtype="Int64"
    )
    frame["realized_plan_exit_day"] = frame["realized_plan_resolved_exit_day"]
    frame["realized_plan_resolved_exit_date"] = pd.array(
        [plan.exit_date for plan in plans], dtype="string"
    )
    frame["realized_plan_exit_date"] = frame["realized_plan_resolved_exit_date"]
    frame["realized_plan_exit_status"] = [plan.status for plan in plans]
    frame["realized_plan_terminal_recovery"] = np.asarray(
        [plan.terminal_recovery for plan in plans], dtype=bool
    )
    frame["entry_order_filled_base"] = canonical["filled_base"]
    frame["entry_order_filled_stress"] = canonical["filled_stress"]
    frame["shares"] = canonical["shares_base"]
    frame["shares_base"] = canonical["shares_base"]
    frame["shares_stress"] = canonical["shares_stress"]
    frame["execution_cost_base_cny"] = canonical["cost_base"]
    frame["execution_cost_stress_cny"] = canonical["cost_stress"]
    frame["cash_utilization"] = canonical["cash_utilization_base"]
    frame["cash_utilization_base"] = canonical["cash_utilization_base"]
    frame["cash_utilization_stress"] = canonical["cash_utilization_stress"]
    frame["ending_cash_base_cny"] = canonical["ending_cash_base"]
    frame["ending_cash_stress_cny"] = canonical["ending_cash_stress"]
    frame["candidate_allocation_cash_cny"] = canonical_allocation
    frame["execution_cost_contract_sha256"] = contract.semantic_sha256
    del canonical

    daily_topk = pd.DataFrame(
        _daily_topk_rows(
            frame=frame,
            plans=plans,
            entry_filled=entry_filled,
            entry_prices=entry_prices,
            gross_values=gross_values,
            contract=contract,
            top_k_values=normalized_top_k,
            daily_cohort_cash_cny=cohort_cash,
        )
    )
    if len(frame) != count:
        raise AssertionError("candidate-complete execution evaluation dropped rows")
    return CandidateExecutionEvaluation(
        candidates=frame,
        daily_topk=daily_topk,
        execution_cost_contract_sha256=contract.semantic_sha256,
    )


evaluate_candidate_complete_execution = evaluate_candidate_execution


__all__ = [
    "CandidateExecutionEvaluation",
    "DEFAULT_DAILY_COHORT_CASH_CNY",
    "DEFAULT_TOP_K_VALUES",
    "ExecutionCostContract",
    "evaluate_candidate_complete_execution",
    "evaluate_candidate_execution",
    "execution_cost_contract_sha256",
    "parse_execution_cost_contract",
]
