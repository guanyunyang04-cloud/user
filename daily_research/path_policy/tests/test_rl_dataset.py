from __future__ import annotations

import pytest

from daily_research.path_policy.rl_dataset import (
    build_path20_sequence_trajectory_dataset,
    build_sequence_tensors,
    full_year_window,
    validate_no_oracle_or_future_inputs,
)
from daily_research.path_policy.tests.fixtures import make_prepared_policy_inputs


def test_rl_trajectory_uses_current_features_and_blocks_future_columns() -> None:
    prepared = make_prepared_policy_inputs(days=45, stocks=("AAA", "BBB", "CCC"))
    trajectory = build_path20_sequence_trajectory_dataset(
        prepared,
        start_date="20240102",
        end_date="20240220",
        lake_dataset_id="policy_input_bundle__fixture",
        sequence_length=5,
        min_trading_days=2,
    )

    assert trajectory.manifest["status"] == "completed"
    assert trajectory.manifest["policy_version"] == "alpha_path20_sequence_policy_v1"
    assert trajectory.manifest["leakage_guard"]["status"] == "passed"
    assert trajectory.feature_columns
    assert not any(column.startswith("future_") for column in trajectory.feature_columns)
    assert not any(column.startswith("oracle_path20_") for column in trajectory.feature_columns)
    assert "next_open_return" in trajectory.daily_frames[0].columns
    assert "next_open_excess_return" in trajectory.daily_frames[0].columns


def test_rl_sequence_tensor_shapes_preserve_order() -> None:
    prepared = make_prepared_policy_inputs(days=35, stocks=("AAA", "BBB"))
    trajectory = build_path20_sequence_trajectory_dataset(
        prepared,
        start_date="20240102",
        end_date="20240209",
        lake_dataset_id="policy_input_bundle__fixture",
        sequence_length=4,
        min_trading_days=2,
    )
    tensors = build_sequence_tensors(trajectory, sequence_length=4)

    assert tensors["state"].ndim == 4
    assert tensors["state"].shape[1] == 4
    assert tensors["state"].shape[2] == 2
    assert tensors["mask"].shape == tensors["current_weight"].shape
    assert str(tensors["dates"][0]) >= trajectory.dates[3].strftime("%Y-%m-%d")


def test_rl_leakage_guard_rejects_oracle_and_future_columns() -> None:
    with pytest.raises(ValueError):
        validate_no_oracle_or_future_inputs(["score_blend", "future_return_1d"])
    with pytest.raises(ValueError):
        validate_no_oracle_or_future_inputs(["oracle_path20_future_cum_excess_return_20d"])
    with pytest.raises(ValueError):
        validate_no_oracle_or_future_inputs(["path_q10_1d"])


def test_full_year_windows_are_calendar_year_requests() -> None:
    assert full_year_window(2019) == ("20190101", "20191231")
    assert full_year_window("2024") == ("20240101", "20241231")


def test_annual_manifest_marks_low_trading_day_window_incomplete() -> None:
    prepared = make_prepared_policy_inputs(days=30, stocks=("AAA", "BBB"))
    trajectory = build_path20_sequence_trajectory_dataset(
        prepared,
        start_date="20240102",
        end_date="20240215",
        lake_dataset_id="policy_input_bundle__fixture",
        year=2024,
        sequence_length=4,
        min_trading_days=180,
    )

    assert trajectory.manifest["status"] == "incomplete"
    assert trajectory.manifest["incomplete_reason"] == "trading_day_count_below_180"
    assert trajectory.manifest["trading_day_count"] < 180
