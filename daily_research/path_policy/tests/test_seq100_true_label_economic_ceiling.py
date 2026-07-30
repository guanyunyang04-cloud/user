from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import dataclass

import numpy as np
import pandas as pd

from daily_research.path_policy import (
    seq100_true_label_economic_ceiling as oracle,
)
from daily_research.path_policy import (
    seq100_v4_economic_realizability as economic,
)
from daily_research.path_policy.seq100_candidate_execution import ExecutionCosts


def _costs() -> ExecutionCosts:
    return ExecutionCosts(
        lot_size=100,
        commission_bps=3.0,
        minimum_commission_cny=5.0,
        transfer_fee_bps=0.1,
        slippage_bps=7.0,
        stress_slippage_multiplier=2.0,
        stamp_tax_schedule=(
            ("1900-01-01", 10.0),
            ("2023-08-28", 5.0),
        ),
    )


def test_oracle_inventory_and_forbidden_year_are_frozen() -> None:
    study = oracle.load_study()
    specs = oracle.task_specs()

    assert len(specs) == 864
    assert len({spec.task_id for spec in specs}) == 864
    assert study["period"]["signal_years"] == [2023, 2024, 2025]
    assert study["period"]["forbidden_year"] == 2026
    assert study["period"]["maximum_consumed_outcome_date"] == "2025-12-31"
    assert study["oracle"]["daily_rerank_expected_tasks"] == 576
    assert study["oracle"]["scheduled_hold_expected_tasks"] == 288


def test_peak_derivation_excludes_d1_and_uses_first_legal_maximum() -> None:
    sellable = np.ones((20, 2), dtype=bool)
    sellable[12, 1] = False
    peaks, derived = oracle.derive_peak_close_dates(
        date_idx=10,
        symbols=np.asarray([0, 1], dtype=np.int32),
        horizon=4,
        label_mfe=np.asarray([0.2, 0.1], dtype=np.float64),
        future_close=np.asarray(
            [
                [0.5, 0.2, 0.2, 0.1],
                [0.4, 0.3, 0.1, 0.05],
            ],
            dtype=np.float64,
        ),
        d1_open=np.asarray([0.0, 0.0], dtype=np.float64),
        exit_sellable=sellable,
        cutoff_idx=19,
    )

    assert peaks.tolist() == [12, 13]
    assert np.allclose(derived, np.asarray([0.2, 0.1]))


def test_true_rank_panel_has_noncompensating_dual_and_state_semantics() -> None:
    panel, valid10, valid20, dual_valid = oracle._day_true_ranks(
        mfe10=np.asarray([0.1, 0.4, 0.3]),
        mfe20=np.asarray([0.3, 0.2, 0.5]),
        risk10=np.asarray([-0.2, -0.1, 0.0]),
        risk20=np.asarray([-0.3, -0.2, -0.1]),
        state10=np.asarray([0, 1, 2], dtype=np.int8),
    )

    assert valid10.all() and valid20.all() and dual_valid.all()
    assert int(np.argmax(panel[:, 2])) == 0
    assert int(np.argmax(panel[:, 4])) == 2
    expected = economic._rank01(np.minimum(panel[:, 0], panel[:, 1]))
    assert np.allclose(panel[:, 7], expected)
    assert panel[2, 5] > panel[0, 5]
    assert panel[2, 6] > panel[0, 6]


@dataclass
class _EmptyPlanningBook:
    study: dict

    def selector_column(self, family: str) -> int:
        return economic.SignalBook.selector_column(self, family)

    def symbols_for_day(self, family: str, day: int) -> np.ndarray:
        del family, day
        return np.empty(0, dtype=np.int32)


def test_incomplete_label_tail_does_not_force_oracle_liquidation() -> None:
    book = _EmptyPlanningBook(study=economic.load_study())
    spec = economic.TaskSpec(
        family="mfe10_primary",
        exposure_mode="strong_candidate_cash",
        slot_count=1,
        buffer_multiplier=0.0,
        cost_scenario="base",
    )
    position = economic.Position(
        symbol_idx=1,
        shares=100,
        entry_signal_date_idx=0,
        entry_date_idx=1,
        entry_raw_open=10.0,
        entry_adjusted_open=10.0,
        gross_entry_notional=1_000.0,
        net_cash_outflow=1_010.0,
        gross_cash_outflow=1_000.0,
        last_adjusted_price=10.0,
    )

    orders = economic.plan_orders(
        book=book,
        spec=spec,
        day=0,
        positions={1: position},
    )

    assert orders.sells == ()
    assert orders.unpaired_buys == ()


def test_scheduled_sale_distinguishes_peak_close_from_next_open() -> None:
    class Book:
        costs = _costs()
        symbol_values = np.asarray(["000001.SZ"])

        @staticmethod
        def trailing_amount(*, signal_date_idx: int, symbol_idx: int) -> float:
            del signal_date_idx, symbol_idx
            return 1.0e9

    def sell_at(adjusted_price: float) -> float:
        position = economic.Position(
            symbol_idx=0,
            shares=100,
            entry_signal_date_idx=0,
            entry_date_idx=1,
            entry_raw_open=10.0,
            entry_adjusted_open=10.0,
            gross_entry_notional=1_000.0,
            net_cash_outflow=1_010.0,
            gross_cash_outflow=1_000.0,
            last_adjusted_price=10.0,
        )
        positions = {0: position}
        sold, cash, *_ = oracle._execute_scheduled_sale(
            book=Book(),
            spec=oracle.OracleTaskSpec(
                mode="true_label_peak_close_hold",
                family="mfe10_primary",
                exposure_mode="target_full",
                slot_count=1,
                buffer_multiplier=None,
                cost_scenario="base",
            ),
            symbol_idx=0,
            signal_date="2023-01-05",
            execution_date="2023-01-05",
            execution_date_idx=2,
            adjusted_price=adjusted_price,
            sellable=True,
            reason="test",
            cash=0.0,
            gross_cash=0.0,
            positions=positions,
            scheduled_peak={0: 2},
            attempts=[],
            counters=defaultdict(int),
            consecutive_blocked_sells={},
            holding_days=[],
            sell_delay_days=[],
            realized_contribution=defaultdict(float),
            multiplier=1.0,
        )
        assert sold
        return cash

    peak_close_cash = sell_at(12.0)
    next_open_cash = sell_at(10.5)
    assert peak_close_cash > next_open_cash


def test_bounded_lot_solver_matches_legacy_and_handles_large_cash() -> None:
    costs = _costs()

    def legacy_shares(allocation: float, raw_open: float) -> int:
        fill_price = raw_open * (1.0 + costs.slippage_bps / 10_000.0)
        shares = math.floor(allocation / (fill_price * costs.lot_size)) * costs.lot_size
        while shares > 0:
            fill_notional = shares * fill_price
            commission = max(
                costs.minimum_commission_cny,
                fill_notional * costs.commission_bps / 10_000.0,
            )
            transfer = fill_notional * costs.transfer_fee_bps / 10_000.0
            if fill_notional + commission + transfer <= allocation + 1.0e-9:
                break
            shares -= costs.lot_size
        return int(shares)

    for allocation in (1_000.0, 10_000.0, 1_000_000.0):
        for raw_open in (2.31, 10.07, 103.29):
            position, _ = economic._buy_position(
                available_cash=allocation,
                allocated_cash=allocation,
                symbol_idx=0,
                signal_date_idx=0,
                execution_date_idx=1,
                raw_open=raw_open,
                adjusted_open=raw_open,
                costs=costs,
                slippage_multiplier=1.0,
            )
            observed = 0 if position is None else position.shares
            assert observed == legacy_shares(allocation, raw_open)

    huge, _ = economic._buy_position(
        available_cash=1.0e100,
        allocated_cash=1.0e100,
        symbol_idx=0,
        signal_date_idx=0,
        execution_date_idx=1,
        raw_open=10.0,
        adjusted_open=10.0,
        costs=costs,
        slippage_multiplier=1.0,
    )
    assert huge is not None
    assert huge.shares > 10**90


def test_slot_bands_and_weighted_interval_ceiling() -> None:
    surface = pd.DataFrame(
        {
            "slot_count": list(oracle.SLOT_COUNTS),
            "economic_pass": [value in (3, 6, 12) for value in oracle.SLOT_COUNTS],
        }
    )
    bands = oracle.stable_slot_bands(
        surface,
        flag="economic_pass",
        width=3,
    )
    assert bands == [{"slot_counts": [3, 6, 12], "cell_count": 3}]

    trades = pd.DataFrame(
        [
            {
                "signal_date_idx": 0,
                "exit_date_idx": 2,
                "log_wealth_increment": math.log(1.2),
            },
            {
                "signal_date_idx": 2,
                "exit_date_idx": 4,
                "log_wealth_increment": math.log(1.3),
            },
            {
                "signal_date_idx": 0,
                "exit_date_idx": 4,
                "log_wealth_increment": math.log(1.5),
            },
        ]
    )
    log_wealth, selected = oracle.weighted_interval_ceiling(
        trades,
        first_date_idx=0,
        last_date_idx=4,
    )

    assert selected.tolist() == [0, 1]
    assert math.isclose(math.exp(log_wealth), 1.56, abs_tol=1.0e-12)
