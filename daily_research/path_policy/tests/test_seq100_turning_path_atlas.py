from __future__ import annotations

import numpy as np
import pytest

from daily_research.path_policy import seq100_turning_path_atlas as turning


def _arrays(values: list[float]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    prices = np.asarray(values, dtype=float)
    dates = np.asarray([f"2020-01-{index + 1:02d}" for index in range(len(values))])
    indices = np.arange(len(values), dtype=int)
    return prices, dates, indices


def test_contract_uses_cost_scaled_turns_without_fixed_horizon() -> None:
    study = turning.load_study()
    definition = study["turning_definition"]
    assert definition["threshold_multipliers"] == [1, 2, 3, 4, 6, 8, 12, 16, 24]
    assert "prominence" in definition["continuous_significance"]
    assert study["episode_definition"]["no_fixed_resolution_horizon"] is True
    assert study["analysis"]["profit_claim_allowed"] is False


def test_directional_change_extrema_alternate_and_confirm_later() -> None:
    prices, dates, indices = _arrays([0.0, -0.02, -0.05, -0.01, 0.04, 0.01, -0.02])
    events, _ = turning.detect_directional_changes(
        prices, dates, indices, threshold=0.04
    )
    assert [event["event_type"] for event in events] == [
        "peak",
        "trough",
        "peak",
    ]
    assert events[1]["extreme_date_idx"] == 2
    assert events[1]["confirmation_date_idx"] == 3
    assert all(
        event["confirmation_date_idx"] >= event["extreme_date_idx"] for event in events
    )


def test_probe_distinguishes_recovered_pullback_from_terminal_top() -> None:
    prices, dates, indices = _arrays([0.0, -0.05, 0.0, 0.06, 0.04, 0.07, 0.05, 0.00])
    _, episodes = turning.detect_directional_changes(
        prices, dates, indices, threshold=0.04, probe=0.015
    )
    up = [episode for episode in episodes if episode["episode_type"] == "up_pullback"]
    assert [episode["outcome"] for episode in up] == [
        "pullback_recovered",
        "terminal_top",
    ]
    assert all(
        episode["resolution_date_idx"] >= episode["onset_date_idx"] for episode in up
    )


def test_probe_distinguishes_bottom_confirmation_from_continuation() -> None:
    prices, dates, indices = _arrays([0.0, 0.05, 0.0, -0.05, -0.03, -0.06, -0.04, 0.0])
    _, episodes = turning.detect_directional_changes(
        prices, dates, indices, threshold=0.04, probe=0.015
    )
    down = [
        episode for episode in episodes if episode["episode_type"] == "down_rebound"
    ]
    assert [episode["outcome"] for episode in down] == [
        "downtrend_continued",
        "major_bottom_confirmed",
    ]


def test_prominence_is_continuous_and_symmetric_for_peak_and_trough() -> None:
    prices, dates, indices = _arrays([0.0, 0.03, 0.0, -0.04, 0.0])
    records = turning.detect_path_prominence(
        prices, dates, indices, cost_log_return=0.006
    )
    by_type = {record["extremum_type"]: record for record in records}
    assert by_type["peak"]["prominence_log_return"] == pytest.approx(0.03)
    assert by_type["trough"]["prominence_log_return"] == pytest.approx(0.04)
    assert by_type["peak"]["prominence_cost_multiple"] == pytest.approx(5.0)
