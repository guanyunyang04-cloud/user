from __future__ import annotations

import math

import numpy as np

from daily_research.path_policy import (
    seq100_quality_liquidity_profit_timeout as profit_timeout,
)
from daily_research.path_policy import seq100_v4_economic_realizability as economic


def _position() -> economic.Position:
    return economic.Position(
        symbol_idx=1,
        shares=100,
        entry_signal_date_idx=100,
        entry_date_idx=101,
        entry_raw_open=10.0,
        entry_adjusted_open=10.0,
        gross_entry_notional=1000.0,
        net_cash_outflow=1002.0,
        gross_cash_outflow=1000.0,
        last_adjusted_price=10.0,
    )


def _spec(*, timeout: int = 10, take_profit: float | None = 0.08):
    return profit_timeout.ProfitTimeoutSpec(
        slot_count=6,
        timeout_day=timeout,
        take_profit=take_profit,
        cost_scenario="base",
    )


def test_study_and_task_inventory_are_compact_and_frozen() -> None:
    study = profit_timeout.load_study()
    specs = profit_timeout.task_specs()

    assert len(specs) == 32
    assert len({spec.task_id for spec in specs}) == 32
    assert study["signal"]["feature_count"] == 557
    assert study["signal"]["future_buyability_used_for_selection"] is False
    assert study["policies"]["stop_loss"] is None
    assert study["period"]["forbidden_year"] == 2026


def test_optional_summary_metric_preserves_null_as_nan() -> None:
    assert math.isnan(profit_timeout._optional_float(None))
    assert profit_timeout._optional_float(0.25) == 0.25


def test_take_profit_is_inactive_on_entry_day_and_active_from_d2() -> None:
    position = _position()

    entry_day = profit_timeout.resolve_exit_decision(
        position=position,
        spec=_spec(),
        date_idx=101,
        adjusted_ohlc=np.asarray([10.0, 11.0, 9.8, 10.9]),
        sellable=True,
    )
    d2 = profit_timeout.resolve_exit_decision(
        position=position,
        spec=_spec(),
        date_idx=102,
        adjusted_ohlc=np.asarray([10.1, 10.9, 10.0, 10.7]),
        sellable=True,
    )

    assert entry_day is None
    assert d2 is not None
    assert d2.reason == "take_profit"
    assert d2.fill_phase == "standing_limit"
    assert d2.fill_price == 10.8


def test_take_profit_gap_and_blocked_semantics() -> None:
    gap = profit_timeout.resolve_exit_decision(
        position=_position(),
        spec=_spec(),
        date_idx=102,
        adjusted_ohlc=np.asarray([11.0, 11.2, 10.7, 10.9]),
        sellable=True,
    )
    blocked = profit_timeout.resolve_exit_decision(
        position=_position(),
        spec=_spec(),
        date_idx=102,
        adjusted_ohlc=np.asarray([10.2, 10.9, 9.7, 9.8]),
        sellable=False,
    )

    assert gap is not None
    assert gap.fill_phase == "gap_open"
    assert gap.fill_price == 11.0
    assert blocked is not None
    assert blocked.reason == "take_profit"
    assert blocked.fill_phase == "blocked"
    assert blocked.fill_price is None


def test_timeout_only_exits_at_precommitted_close() -> None:
    before = profit_timeout.resolve_exit_decision(
        position=_position(),
        spec=_spec(timeout=10, take_profit=None),
        date_idx=109,
        adjusted_ohlc=np.asarray([10.1, 12.0, 9.5, 11.5]),
        sellable=True,
    )
    timeout = profit_timeout.resolve_exit_decision(
        position=_position(),
        spec=_spec(timeout=10, take_profit=None),
        date_idx=110,
        adjusted_ohlc=np.asarray([10.1, 12.0, 9.5, 9.7]),
        sellable=True,
    )

    assert before is None
    assert timeout is not None
    assert timeout.reason == "timeout"
    assert timeout.fill_phase == "precommitted_close"
    assert timeout.fill_price == 9.7


class _CandidateBook:
    def __init__(self) -> None:
        self.study = {
            "policies": {
                "auxiliary_top_fraction": 1.0,
                "auxiliary_retention_fraction": 0.5,
            }
        }
        self.rank_panel = np.zeros((1, 4, 7), dtype=np.float64)
        self.rank_panel[0, :, 5] = [0.9, 0.1, 0.8, 0.7]
        self.rank_panel[0, :, 6] = [0.8, 0.2, 0.7, 0.6]

    def symbols_for_day(self, family: str, day: int) -> np.ndarray:
        assert family == profit_timeout.FAMILY
        assert day == 0
        return np.asarray([0, 1, 2, 3], dtype=np.int32)


def test_entry_selection_uses_risk_gate_without_future_buyability() -> None:
    selected = profit_timeout.entry_candidates(
        book=_CandidateBook(),  # type: ignore[arg-type]
        day=0,
        held_symbols={0},
        limit=2,
    )

    assert selected == [2]
