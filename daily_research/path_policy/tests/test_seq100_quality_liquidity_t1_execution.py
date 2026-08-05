from __future__ import annotations

import math

import numpy as np

from daily_research.path_policy import (
    seq100_quality_liquidity_t1_execution as execution,
)
from daily_research.path_policy import (
    seq100_v4_economic_realizability as economic,
)


def test_task_surface_is_bounded_to_one_through_three_positions() -> None:
    study = execution.load_study()
    specs = execution.task_specs()

    assert len(specs) == 3456
    assert len({spec.task_id for spec in specs}) == 3456
    assert {spec.slot_count for spec in specs} == {1, 2, 3}
    assert study["period"]["last_entry_signal_date"] == "2025-12-29"
    assert study["period"]["last_mark_date"] == "2025-12-31"
    assert study["period"]["forbidden_year"] == 2026


def test_rank_blends_and_agreement_are_cross_sectional() -> None:
    ranks, confidence = execution._family_rank_and_confidence(
        np.asarray([0.0, 1.0, 2.0]),
        np.asarray([2.0, 1.0, 0.0]),
    )

    open_idx = execution.SCORE_FAMILIES.index("open_to_open")
    close_idx = execution.SCORE_FAMILIES.index("open_to_close_2")
    blend_idx = execution.SCORE_FAMILIES.index("blend_open_weight_50")
    agreement_idx = execution.SCORE_FAMILIES.index("agreement_min")
    np.testing.assert_allclose(ranks[open_idx], [0.0, 0.5, 1.0])
    np.testing.assert_allclose(ranks[close_idx], [1.0, 0.5, 0.0])
    np.testing.assert_allclose(ranks[blend_idx], [0.5, 0.5, 0.5])
    np.testing.assert_allclose(ranks[agreement_idx], [0.25, 1.0, 0.25])
    np.testing.assert_allclose(confidence[blend_idx], [1.0, 1.0, 1.0])
    np.testing.assert_allclose(confidence[agreement_idx], [0.0, 1.0, 0.0])


def test_confidence_percentile_only_uses_prior_values() -> None:
    values = np.asarray([1.0, 2.0, 3.0, 0.0], dtype=np.float64)

    observed = execution._causal_percentile(
        values,
        lookback=3,
        minimum_history=2,
    )

    np.testing.assert_allclose(observed, [1.0, 1.0, 1.0, 0.0])
    changed_future = values.copy()
    changed_future[-1] = 100.0
    changed = execution._causal_percentile(
        changed_future,
        lookback=3,
        minimum_history=2,
    )
    np.testing.assert_allclose(changed[:-1], observed[:-1])


class _FixedPlannerBook:
    signal_date_idx = np.asarray([10, 11, 12], dtype=np.int32)
    date_values = np.asarray([str(index) for index in range(20)])

    @staticmethod
    def date_text(day: int) -> str:
        return "2025-01-02"

    @staticmethod
    def confidence_pass(*, family: str, day: int, mode: str) -> bool:
        return True

    @staticmethod
    def symbols_for_day(family: str, day: int) -> np.ndarray:
        return np.asarray([1, 2, 3, 4], dtype=np.int32)


def _fixed_spec(
    *,
    exit_mode: str,
    cadence: str,
    slots: int,
) -> execution.ExecutionSpec:
    return execution.ExecutionSpec(
        score_family="open_to_close_2",
        entry_mode="open",
        exit_mode=exit_mode,
        cadence=cadence,
        confidence_mode="always",
        slot_count=slots,
        buffer_multiplier=0.0,
        cost_scenario="base",
    )


def test_fixed_close_staggering_caps_each_daily_cohort() -> None:
    book = _FixedPlannerBook()
    positions = {4: object()}

    staggered = execution._plan_fixed_buys(
        book=book,
        spec=_fixed_spec(exit_mode="fixed_close_2", cadence="staggered", slots=3),
        day=0,
        positions=positions,
        exit_due={4: 12},
    )
    full = execution._plan_fixed_buys(
        book=book,
        spec=_fixed_spec(exit_mode="fixed_close_2", cadence="full_cohort", slots=3),
        day=0,
        positions=positions,
        exit_due={4: 12},
    )

    assert [item.symbol_idx for item in staggered] == [1, 2]
    assert full == ()


def test_fixed_open_can_sell_then_rebuy_a_due_symbol() -> None:
    book = _FixedPlannerBook()
    positions = {1: object(), 2: object()}

    planned = execution._plan_fixed_buys(
        book=book,
        spec=_fixed_spec(exit_mode="fixed_open_2", cadence="full_cohort", slots=2),
        day=0,
        positions=positions,
        exit_due={1: 11, 2: 11},
    )

    assert [item.symbol_idx for item in planned] == [1, 2]


def test_close_renewal_uses_the_previous_signal_and_rank_band() -> None:
    class Book(_FixedPlannerBook):
        @staticmethod
        def candidate_count(day: int) -> int:
            return 4

        @staticmethod
        def symbols_for_day(family: str, day: int) -> np.ndarray:
            return np.asarray([1, 2, 3, 4], dtype=np.int32)

    spec = execution.ExecutionSpec(
        score_family="open_to_close_2",
        entry_mode="open",
        exit_mode="renewed_close_2",
        cadence="staggered",
        confidence_mode="always",
        slot_count=1,
        buffer_multiplier=1.0,
        cost_scenario="base",
    )

    assert execution._renew_close_position(book=Book(), spec=spec, day=1, symbol_idx=2)
    assert not execution._renew_close_position(
        book=Book(), spec=spec, day=1, symbol_idx=3
    )


def test_first_5m_vwap_uses_the_pack_adjustment_ratio() -> None:
    book = object.__new__(execution.SignalBook)
    book.raw_open = np.asarray([[10.0]], dtype=np.float32)
    book.adjusted_open = np.asarray([[20.0]], dtype=np.float32)
    book.first_5m_vwap = np.asarray([[11.0]], dtype=np.float32)

    raw, adjusted = book.execution_entry_prices(
        entry_mode="first_5m_vwap",
        signal_day=0,
        execution_date_idx=0,
        symbol_idx=0,
    )

    assert math.isclose(raw, 11.0)
    assert math.isclose(adjusted, 22.0)


def test_rolling_rank_buffer_keeps_an_incumbent_inside_the_band() -> None:
    class Book:
        @staticmethod
        def confidence_pass(*, family: str, day: int, mode: str) -> bool:
            return True

        @staticmethod
        def symbols_for_day(family: str, day: int) -> np.ndarray:
            return np.asarray([2, 1, 3], dtype=np.int32)

        @staticmethod
        def rank_value(*, family: str, day: int, symbol_idx: int) -> float:
            return {1: 0.50, 2: 0.75, 3: 0.0}[symbol_idx]

    spec = economic.TaskSpec(
        family="unused",
        exposure_mode="target_full",
        slot_count=1,
        buffer_multiplier=1.0,
        cost_scenario="base",
    )

    orders = execution._plan_rolling_orders(
        book=Book(),
        spec=spec,
        score_family="open_to_open",
        confidence_mode="always",
        day=0,
        positions={1: object()},
    )

    assert orders.sells == ()
    assert orders.unpaired_buys == ()
