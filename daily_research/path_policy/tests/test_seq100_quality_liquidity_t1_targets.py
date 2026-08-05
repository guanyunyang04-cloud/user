from __future__ import annotations

import numpy as np
import pytest

from daily_research.path_policy import (
    seq100_quality_liquidity_t1_targets as targets,
)


def _panels(*, days: int = 6, symbols: int = 2) -> dict[str, np.ndarray]:
    daily = np.full((days, symbols, 4), np.nan, dtype=np.float32)
    for day in range(days):
        for symbol in range(symbols):
            value = 10.0 + day + symbol
            daily[day, symbol, 0] = value
            daily[day, symbol, 1] = value + 1.0
            daily[day, symbol, 2] = value - 1.0
            daily[day, symbol, 3] = value + 0.5
    shape = (days, symbols)
    return {
        "daily_raw": daily,
        "price_observed": np.ones(shape, dtype=bool),
        "entry_filled": np.ones(shape, dtype=bool),
        "status_valid": np.ones(shape, dtype=bool),
        "suspended": np.zeros(shape, dtype=bool),
        "delisted": np.zeros(shape, dtype=bool),
        "raw_open": daily[:, :, 0].copy(),
        "raw_down_limit": np.full(shape, 1.0, dtype=np.float32),
    }


def _derive(
    panels: dict[str, np.ndarray],
    *,
    signal_date_idx: np.ndarray,
    symbol_idx: np.ndarray,
    cutoff_idx: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    return targets._derive_batch(
        signal_date_idx=signal_date_idx,
        symbol_idx=symbol_idx,
        cutoff_idx=cutoff_idx,
        **panels,
    )


def test_study_freezes_t1_target_without_training() -> None:
    study = targets.load_study()

    assert study["target"]["primary"] == targets.PRIMARY_LABEL
    assert study["target"]["entry"] == "next_trading_day_open"
    assert study["target"]["exit"] == "second_next_trading_day_open"
    assert study["source"]["forbidden_year"] == 2026
    assert not study["source"]["candidate_selection_uses_future_fill"]
    assert not study["execution"]["training_performed"]


def test_open_to_open_label_and_components_reconcile() -> None:
    panels = _panels()
    panels["daily_raw"][1, 0, 0] = 10.0
    panels["daily_raw"][1, 0, 3] = 11.0
    panels["daily_raw"][2, 0, 0] = 12.0

    returns, valid, flags = _derive(
        panels,
        signal_date_idx=np.asarray([0], dtype=np.int32),
        symbol_idx=np.asarray([0], dtype=np.int32),
        cutoff_idx=4,
    )

    assert valid.tolist() == [1]
    assert returns[0, 0] == pytest.approx(0.20)
    assert returns[0, 1] == pytest.approx(0.10)
    assert returns[0, 2] == pytest.approx(12.0 / 11.0 - 1.0)
    assert (1.0 + returns[0, 1]) * (1.0 + returns[0, 2]) - 1.0 == pytest.approx(
        returns[0, 0]
    )
    assert flags[0] & targets.FLAG_TARGET_VALID
    assert flags[0] & targets.FLAG_DECOMPOSITION_VALID


def test_future_fill_state_does_not_gate_price_label() -> None:
    panels = _panels()
    panels["entry_filled"][0, 0] = False
    panels["raw_down_limit"][2, 0] = panels["raw_open"][2, 0]

    returns, valid, flags = _derive(
        panels,
        signal_date_idx=np.asarray([0], dtype=np.int32),
        symbol_idx=np.asarray([0], dtype=np.int32),
        cutoff_idx=4,
    )

    assert valid.tolist() == [1]
    assert np.isfinite(returns[0, 0])
    assert flags[0] & targets.FLAG_ENTRY_EXECUTION_STATE_KNOWN
    assert flags[0] & targets.FLAG_EXIT_EXECUTION_STATE_KNOWN
    assert not flags[0] & targets.FLAG_ENTRY_BUYABLE
    assert not flags[0] & targets.FLAG_EXIT_OPEN_SELLABLE


def test_cutoff_prevents_second_next_open_from_becoming_a_label() -> None:
    panels = _panels()
    panels["daily_raw"][4:, :, :] = 999.0

    returns, valid, flags = _derive(
        panels,
        signal_date_idx=np.asarray([2, 3], dtype=np.int32),
        symbol_idx=np.asarray([0, 0], dtype=np.int32),
        cutoff_idx=3,
    )

    assert valid.tolist() == [0, 0]
    assert np.isnan(returns).all()
    assert not (flags & targets.FLAG_OUTCOME_WITHIN_CUTOFF).any()


def test_known_suspension_is_separate_from_missing_price() -> None:
    panels = _panels()
    panels["price_observed"][1, 0] = False
    panels["daily_raw"][1, 0, 0] = 10.0
    panels["suspended"][1, 0] = True
    panels["entry_filled"][0, 0] = False

    returns, valid, flags = _derive(
        panels,
        signal_date_idx=np.asarray([0], dtype=np.int32),
        symbol_idx=np.asarray([0], dtype=np.int32),
        cutoff_idx=4,
    )

    assert valid.tolist() == [0]
    assert np.isnan(returns).all()
    assert flags[0] & targets.FLAG_ENTRY_EXECUTION_STATE_KNOWN
    assert not flags[0] & targets.FLAG_ENTRY_PRICE_OBSERVED
    assert not flags[0] & targets.FLAG_ENTRY_BUYABLE


def test_frozen_baseline_metrics_reward_correct_ordering() -> None:
    dates = np.repeat(np.asarray([1, 2], dtype=np.int32), 100)
    actual = np.tile(np.linspace(-0.10, 0.10, 100), 2)

    metrics = targets._daily_return_metrics(
        date_idx=dates,
        actual=actual,
        prediction=actual,
    )

    assert len(metrics) == 2
    assert metrics["rank_ic"].tolist() == pytest.approx([1.0, 1.0])
    assert metrics["top1_capture"].tolist() == pytest.approx([1.0, 1.0])
    assert metrics["top5_capture"].tolist() == pytest.approx([1.0, 1.0])
    assert metrics["top5_excess_mean"].gt(0.0).all()
