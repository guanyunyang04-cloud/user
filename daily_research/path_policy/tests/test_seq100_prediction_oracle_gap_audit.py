from __future__ import annotations

import math
from dataclasses import replace

import numpy as np
import pandas as pd

from daily_research.path_policy import (
    seq100_prediction_oracle_gap_audit as gap,
)
from daily_research.path_policy import (
    seq100_v4_economic_realizability as economic,
)


def test_gap_inventory_and_forbidden_year_are_frozen() -> None:
    study = gap.load_study()
    tasks = gap.task_specs(study)

    assert len(tasks) == 1200
    assert len({task.task_id for task in tasks}) == 1200
    assert study["period"]["signal_years"] == [2023, 2024, 2025]
    assert study["period"]["forbidden_year"] == 2026
    assert study["account"]["profits_reinvested"] is False
    assert study["account"]["allocation_mode"] == ("fixed_initial_notional_per_slot")


def test_common_support_reranking_and_top_capture() -> None:
    ranks, symbols = gap._rank_on_support(
        np.asarray([0.4, 0.1, 0.3, 0.2]),
        np.asarray([True, False, True, True]),
    )
    assert symbols.tolist() == [0, 2, 3]
    assert np.allclose(ranks[symbols], np.asarray([1.0, 0.5, 0.0]))
    assert np.isnan(ranks[1])

    top = gap._top_metrics(
        predicted_rank=np.asarray([1.0, 0.5, 0.0, np.nan]),
        true_rank=np.asarray([0.5, 1.0, 0.0, np.nan]),
        true_mfe=np.asarray([0.2, 0.4, 0.1, np.nan]),
        peak_day=np.asarray([3.0, 4.0, 2.0, np.nan]),
        fraction=0.5,
    )
    assert top["intersection_count"] == 2
    assert top["overlap_over_smaller"] == 1.0
    assert top["predicted_top_true_mfe"] == top["oracle_top_true_mfe"]


def test_custom_fixed_notional_allocation_does_not_reinvest_profit() -> None:
    class Book:
        @staticmethod
        def position_allocation(
            *,
            spec: economic.TaskSpec,
            cash: float,
            equity_open: float,
        ) -> float:
            del cash, equity_open
            return gap.STARTING_CASH_CNY / spec.slot_count

    spec = economic.TaskSpec(
        family="mfe10_primary",
        exposure_mode="target_full",
        slot_count=6,
        buffer_multiplier=0.0,
        cost_scenario="base",
    )
    first = economic._position_allocation(
        book=Book(),
        spec=spec,
        cash=2_000_000.0,
        equity_open=2_000_000.0,
    )
    second = economic._position_allocation(
        book=Book(),
        spec=spec,
        cash=20_000_000.0,
        equity_open=20_000_000.0,
    )
    assert first == second == gap.STARTING_CASH_CNY / 6


def test_mixed_book_routes_each_coordinate_to_its_declared_source() -> None:
    book = gap.MixedSignalBook.__new__(gap.MixedSignalBook)
    book.variant = gap.VariantSpec(
        variant_id="synthetic",
        family="dual_mfe_state_risk_veto",
        mfe10_source="true",
        mfe20_source="predicted",
        state_source="true",
        risk_source="predicted",
        role="test",
    )
    book.component_index = {
        "mfe10_true_d20": 0,
        "mfe20_predicted_d20": 1,
    }
    book.dual_index = {"dual_true_predicted": 0}
    book.component_ranks = np.asarray(
        [
            [[0.1, 0.9]],
            [[0.8, 0.2]],
        ],
        dtype=np.float32,
    )
    book.dual_ranks = np.asarray([[[0.4, 0.6]]], dtype=np.float32)
    book.predicted_rank_panel = np.zeros(
        (1, 2, len(economic.RANK_COLUMNS)),
        dtype=np.float32,
    )
    book.true_rank_panel = np.ones(
        (1, 2, len(economic.RANK_COLUMNS)),
        dtype=np.float32,
    )
    symbols = np.asarray([0, 1], dtype=np.int32)

    assert np.allclose(
        book.rank_values(day=0, symbols=symbols, column=0),
        [0.1, 0.9],
    )
    assert np.allclose(
        book.rank_values(day=0, symbols=symbols, column=1),
        [0.8, 0.2],
    )
    assert np.allclose(
        book.rank_values(day=0, symbols=symbols, column=2),
        [1.0, 1.0],
    )
    assert np.allclose(
        book.rank_values(day=0, symbols=symbols, column=5),
        [0.0, 0.0],
    )
    assert np.allclose(
        book.rank_values(day=0, symbols=symbols, column=7),
        [0.4, 0.6],
    )


def test_gate_uses_rank_vector_interface_without_materialized_panel() -> None:
    class Book:
        def __init__(self) -> None:
            self.study = economic.load_study()
            self.values = np.zeros(
                (100, len(economic.RANK_COLUMNS)),
                dtype=np.float64,
            )
            base = np.linspace(0.0, 1.0, 100)
            self.values[:, :] = base[:, None]

        def selector_column(self, family: str) -> int:
            return economic.SignalBook.selector_column(self, family)

        def symbols_for_day(self, family: str, day: int) -> np.ndarray:
            del family, day
            return np.arange(99, -1, -1, dtype=np.int32)

        def rank_values(
            self,
            *,
            day: int,
            symbols: np.ndarray,
            column: int,
        ) -> np.ndarray:
            del day
            return self.values[symbols, column]

        def rank(self, day: int, symbol_idx: int, column: int) -> float:
            del day
            return float(self.values[symbol_idx, column])

    spec = economic.TaskSpec(
        family="dual_mfe_state_risk_veto",
        exposure_mode="target_full",
        slot_count=3,
        buffer_multiplier=2.0,
        cost_scenario="base",
    )
    orders = economic.plan_orders(
        book=Book(),
        spec=spec,
        day=0,
        positions={},
    )
    assert len(orders.unpaired_buys) > 0


def test_effect_row_requires_matched_account_dimensions() -> None:
    base = {
        "task_id": "left",
        "mode": "daily_rerank",
        "variant_id": "left",
        "exposure_mode": "target_full",
        "slot_count": 3,
        "buffer_multiplier": 2.0,
        "cost_scenario": "base",
        "terminal_cost_accrued_return": 0.3,
        "net_cagr": 0.1,
        "maximum_drawdown": -0.2,
        "turnover_to_starting_cash": 3.0,
        "total_cost": 100.0,
        "net_return_2023": 0.1,
        "net_return_2024": 0.1,
        "net_return_2025": 0.1,
    }
    right = {
        **base,
        "task_id": "right",
        "variant_id": "right",
        "terminal_cost_accrued_return": 0.1,
        "net_cagr": 0.03,
    }
    rows: list[dict] = []
    gap._append_effect(
        rows,
        effect="test",
        layer="test",
        left=base,
        right=right,
    )
    assert math.isclose(rows[0]["terminal_return_delta"], 0.2)
    assert math.isclose(rows[0]["terminal_pnl_delta_cny"], 200_000.0)

    changed = replace(
        gap.GapTaskSpec(
            mode="daily_rerank",
            variant_id="left",
            exposure_mode="target_full",
            slot_count=3,
            buffer_multiplier=2.0,
            cost_scenario="base",
        ),
        slot_count=6,
    )
    assert changed.slot_count == 6


def test_period_cash_metrics_separate_pnl_from_growing_equity_return() -> None:
    frame = pd.DataFrame(
        {
            "task_id": ["a", "a", "a"],
            "year": [2023, 2024, 2025],
            "ending_equity": [2_000_000.0, 3_000_000.0, 4_000_000.0],
        }
    )
    enriched = gap._add_period_cash_metrics(frame, period_column="year")

    assert enriched["starting_equity"].tolist() == [
        1_000_000.0,
        2_000_000.0,
        3_000_000.0,
    ]
    assert enriched["period_pnl_cny"].tolist() == [
        1_000_000.0,
        1_000_000.0,
        1_000_000.0,
    ]
    assert enriched["period_pnl_on_initial_cash"].tolist() == [1.0, 1.0, 1.0]
    assert enriched["cumulative_pnl_cny"].tolist() == [
        1_000_000.0,
        2_000_000.0,
        3_000_000.0,
    ]
