from __future__ import annotations

import math

import numpy as np

from daily_research.path_policy import (
    seq100_quality_liquidity_d1_execution as d1,
)
from daily_research.path_policy import (
    seq100_v4_economic_realizability as economic,
)


def test_d1_task_surface_and_t1_boundary_are_explicit() -> None:
    study = d1.load_study()
    specs = d1.task_specs()

    assert len(specs) == 140
    assert len({spec.task_id for spec in specs}) == 140
    assert study["model_contract"]["target"] == "return_1"
    assert study["model_contract"]["source_label"] == "g_1"
    assert study["period"]["last_entry_signal_date"] == "2025-12-23"
    assert study["period"]["last_mark_date"] == "2025-12-31"
    assert study["period"]["forbidden_year"] == 2026


def test_first_5m_vwap_is_converted_to_the_pack_adjustment() -> None:
    book = object.__new__(d1.SignalBook)
    book.raw_open = np.asarray([[10.0]], dtype=np.float32)
    book.adjusted_open = np.asarray([[20.0]], dtype=np.float32)
    book.first_5m_vwap = np.asarray([[11.0]], dtype=np.float32)
    spec = economic.TaskSpec(
        family="d1_first_5m_vwap",
        exposure_mode="target_full",
        slot_count=1,
        buffer_multiplier=0.0,
        cost_scenario="base",
    )

    raw, adjusted = book.entry_prices(
        spec=spec,
        signal_day=0,
        execution_date_idx=0,
        symbol_idx=0,
    )

    assert math.isclose(raw, 11.0)
    assert math.isclose(adjusted, 22.0)


def test_end_of_sample_planner_only_sells_and_never_adds_positions() -> None:
    book = object.__new__(d1.SignalBook)
    book.date_text = lambda day: "2025-12-24"
    spec = economic.TaskSpec(
        family="d1_open",
        exposure_mode="target_full",
        slot_count=2,
        buffer_multiplier=0.0,
        cost_scenario="base",
    )
    positions = {
        8: object(),
        3: object(),
    }

    orders = book.plan_orders(spec=spec, day=1, positions=positions)

    assert [item.symbol_idx for item in orders.sells] == [3, 8]
    assert all(item.reason == "end_of_sample_liquidation" for item in orders.sells)
    assert orders.unpaired_buys == ()
