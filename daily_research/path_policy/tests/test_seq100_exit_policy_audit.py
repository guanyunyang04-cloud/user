from __future__ import annotations

import math

import numpy as np
import pytest

from daily_research.path_policy.seq100_candidate_execution import (
    _cashflow,
    _resolve_plan,
    parse_execution_cost_contract,
)
from daily_research.path_policy.seq100_exit_policy_audit import (
    _next_valid_exit_indices,
    cashflow_batch,
    oracle_executable_outcome_batch,
    resolve_planned_exit_batch,
)


def _manifest() -> dict:
    return {
        "execution_cost_contract": {
            "contract": "a_share_round_trip_cashflow_v1",
            "lot_size": 100,
            "commission_bps": 3.0,
            "minimum_commission_cny": 5.0,
            "transfer_fee_bps": 0.1,
            "slippage_bps": 7.0,
            "stress_slippage_multiplier": 2.0,
            "stamp_tax_schedule": [
                {"effective_date": "1900-01-01", "stamp_tax_bps": 10.0},
                {"effective_date": "2023-08-28", "stamp_tax_bps": 5.0},
            ],
        }
    }


def _fixture() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    entry_filled = np.asarray([True, True, True, False])
    entry_prices = np.asarray([10.0, 12.0, 8.0, 9.0])
    exit_prices = np.asarray(
        [
            [9.0, 11.0, 12.0, 8.0, 7.0, 13.0, 14.0, 15.0],
            [11.0, 13.0, 14.0, 15.0, 16.0, 17.0, 18.0, 19.0],
            [7.0, 7.5, 7.0, 6.5, 6.0, 5.5, 5.0, 4.5],
            [9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0],
        ],
        dtype=np.float64,
    )
    exit_sellable = np.asarray(
        [
            [True, True, True, True, True, True, True, True],
            [False, False, False, True, True, True, True, True],
            [False, False, False, False, False, False, False, False],
            [True, True, True, True, True, True, True, True],
        ]
    )
    return entry_filled, entry_prices, exit_prices, exit_sellable


def _date_values() -> np.ndarray:
    return np.asarray(
        [
            "2023-08-23",
            "2023-08-24",
            "2023-08-25",
            "2023-08-28",
            "2023-08-29",
            "2023-08-30",
            "2023-08-31",
            "2023-09-01",
            "2023-09-04",
        ],
        dtype=object,
    )


@pytest.mark.parametrize("slippage_multiplier", [1.0, 2.0])
def test_vector_fixed_policy_matches_authoritative_scalar_cashflow(
    slippage_multiplier: float,
) -> None:
    filled, entries, prices, sellable = _fixture()
    planned_days = np.asarray([2, 2, 6, 3])
    next_valid = _next_valid_exit_indices(prices, sellable)
    contract = parse_execution_cost_contract(_manifest())
    dates = _date_values()
    plan = resolve_planned_exit_batch(
        signal_date_idx=0,
        entry_filled=filled,
        entry_prices=entries,
        exit_prices=prices,
        next_valid_exit_idx=next_valid,
        planned_days=planned_days,
        forward_days=6,
        execution_days=8,
        terminal_recovery_fraction=0.25,
    )
    batch = cashflow_batch(
        allocated_cash=250_000.0,
        entry_filled=filled,
        entry_prices=entries,
        plan=plan,
        date_values=dates,
        contract=contract,
        slippage_multiplier=slippage_multiplier,
    )

    assert plan.exit_day.tolist() == [2, 4, 8, -1]
    assert plan.terminal_recovery.tolist() == [False, False, True, False]
    for row in range(len(entries)):
        scalar_plan = _resolve_plan(
            predicted_exit_day=float(planned_days[row]),
            entry_filled=bool(filled[row]),
            entry_price=float(entries[row]),
            exit_prices=prices[row],
            exit_sellable=sellable[row],
            forward_days=6,
            horizon=8,
            terminal_recovery_fraction=0.25,
            date_for_day=lambda _row, day: str(dates[day]),
            row=row,
        )
        scalar = _cashflow(
            allocated_cash=250_000.0,
            entry_filled=bool(filled[row]),
            entry_price=float(entries[row]),
            plan=scalar_plan,
            contract=contract,
            slippage_multiplier=slippage_multiplier,
        )
        expected_exit_day = -1 if scalar_plan.exit_day is None else scalar_plan.exit_day
        assert int(plan.exit_day[row]) == expected_exit_day
        if scalar_plan.exit_price is None:
            assert math.isnan(float(plan.exit_price[row]))
        else:
            assert float(plan.exit_price[row]) == pytest.approx(scalar_plan.exit_price)
        assert bool(batch.order_filled[row]) is scalar.order_filled
        assert int(batch.shares[row]) == scalar.shares
        assert float(batch.ending_cash[row]) == pytest.approx(scalar.ending_cash, abs=1.0e-8)
        assert float(batch.net_return[row]) == pytest.approx(scalar.net_return, abs=1.0e-12)
        assert float(batch.total_cost[row]) == pytest.approx(scalar.total_cost, abs=1.0e-8)
        assert float(batch.cash_utilization[row]) == pytest.approx(
            scalar.cash_utilization, abs=1.0e-12
        )


@pytest.mark.parametrize("slippage_multiplier", [1.0, 2.0])
def test_vector_oracle_matches_scalar_search_and_dominates_every_fixed_day(
    slippage_multiplier: float,
) -> None:
    filled, entries, prices, sellable = _fixture()
    next_valid = _next_valid_exit_indices(prices, sellable)
    contract = parse_execution_cost_contract(_manifest())
    dates = _date_values()
    oracle_plan, oracle_cash = oracle_executable_outcome_batch(
        signal_date_idx=0,
        allocated_cash=250_000.0,
        entry_filled=filled,
        entry_prices=entries,
        exit_prices=prices,
        next_valid_exit_idx=next_valid,
        date_values=dates,
        contract=contract,
        slippage_multiplier=slippage_multiplier,
        forward_days=6,
        execution_days=8,
        terminal_recovery_fraction=0.25,
    )

    scalar_best_day: list[int] = []
    scalar_best_return: list[float] = []
    for row in range(len(entries)):
        outcomes: list[tuple[int, float]] = []
        for day in range(2, 7):
            scalar_plan = _resolve_plan(
                predicted_exit_day=float(day),
                entry_filled=bool(filled[row]),
                entry_price=float(entries[row]),
                exit_prices=prices[row],
                exit_sellable=sellable[row],
                forward_days=6,
                horizon=8,
                terminal_recovery_fraction=0.25,
                date_for_day=lambda _row, exit_day: str(dates[exit_day]),
                row=row,
            )
            scalar_cash = _cashflow(
                allocated_cash=250_000.0,
                entry_filled=bool(filled[row]),
                entry_price=float(entries[row]),
                plan=scalar_plan,
                contract=contract,
                slippage_multiplier=slippage_multiplier,
            )
            outcomes.append((day, scalar_cash.net_return))
        best = max(outcomes, key=lambda value: value[1])
        scalar_best_day.append(best[0])
        scalar_best_return.append(best[1])

    assert oracle_plan.planned_day.tolist() == scalar_best_day
    np.testing.assert_allclose(oracle_cash.net_return, scalar_best_return, atol=1.0e-12, rtol=0.0)
    for fixed_day in range(2, 7):
        plan = resolve_planned_exit_batch(
            signal_date_idx=0,
            entry_filled=filled,
            entry_prices=entries,
            exit_prices=prices,
            next_valid_exit_idx=next_valid,
            planned_days=fixed_day,
            forward_days=6,
            execution_days=8,
            terminal_recovery_fraction=0.25,
        )
        fixed = cashflow_batch(
            allocated_cash=250_000.0,
            entry_filled=filled,
            entry_prices=entries,
            plan=plan,
            date_values=dates,
            contract=contract,
            slippage_multiplier=slippage_multiplier,
        )
        assert bool(np.all(oracle_cash.net_return >= fixed.net_return - 1.0e-12))
