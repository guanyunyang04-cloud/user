from __future__ import annotations

import math

import numpy as np
import pandas as pd

from daily_research.path_policy import seq100_structured_180x35_2026 as study
from daily_research.path_policy import seq100_structured_180x35_2026_vintages as vintages


def test_vintage_scope_adds_only_the_two_missing_inference_tasks() -> None:
    assert vintages.VINTAGES == (2023, 2024, 2025, 2026)
    assert vintages.NEW_INFERENCE_VINTAGES == (2023, 2024)
    assert vintages.MODEL_IDS[2026] == study.MODEL_2026_180
    assert vintages.HISTORICAL_YEARS == (2023, 2024, 2025)
    assert vintages.EXPECTED_HISTORICAL_POLICY_ALPHA_DAILY_ROWS == 177_388


def test_historical_checkpoint_discovery_is_unique_and_complete() -> None:
    for year in (2023, 2024):
        run_dir = vintages._run_dir(year)
        assert f"lookback180_turnover_{year}_seed7" in run_dir.name
        assert (run_dir / "best_model.pt").is_file()
        assert (run_dir / "sequence_path_training_summary.json").is_file()


def test_vintage_log_growth_uses_the_observed_account_sessions() -> None:
    actual = vintages._annualized_log_growth(0.12, 48)
    expected = math.log1p(0.12) * 252.0 / 48.0
    assert actual == expected


def test_vintage_forecast_book_uses_each_models_own_exit_plan() -> None:
    frame = pd.DataFrame(
        {
            "date_idx": [10, 10, 10],
            "symbol_idx": [2, 1, 3],
            "score": [0.5, 0.5, 0.4],
            "predicted_exit_day": [7, 12, 4],
        }
    )
    book = vintages._forecast_book(frame, vintage=2023, top_k=1)
    day = book.days[10]
    assert day.top3_symbol_idx == (1,)
    np.testing.assert_array_equal(day.planned_day, np.asarray([12, 7, 4]))


def test_2026_checkpoint_health_reconciles_native_training_rows() -> None:
    health = vintages._checkpoint_health(2026)
    assert health["completed_epochs"] == 1
    assert health["best_epoch"] == 1
    assert health["train_row_count"] == 7_396_133
    assert health["development_candidate_count"] == 143_300
    assert health["normalization_cutoff"] == "2026-01-05"
    assert health["model_state_finite"] is True
    assert health["scaler_state_finite"] is True
    assert health["losses_finite"] is True


def test_rolling_plan_only_accelerates_and_uses_next_close_for_nonpositive_score() -> None:
    lookup = {
        11: (
            np.asarray([1, 2], dtype=np.int64),
            np.asarray([-0.1, 0.5], dtype=np.float64),
            np.asarray([50, 2], dtype=np.int16),
        ),
        12: (
            np.asarray([1, 2], dtype=np.int64),
            np.asarray([0.5, -0.1], dtype=np.float64),
            np.asarray([60, 60], dtype=np.int16),
        ),
    }
    actual = vintages._rolling_planned_days(
        signal_date_idx=10,
        symbol_idx=np.asarray([1, 2], dtype=np.int64),
        initial_planned_day=np.asarray([10, 10], dtype=np.int16),
        forecast_lookup=lookup,
    )
    np.testing.assert_array_equal(actual, np.asarray([2, 3], dtype=np.int16))


def test_policy_alpha_aggregation_keeps_mean_and_robust_diagnostics_separate() -> None:
    values = [0.01] * 47 + [1.0]
    frame = pd.DataFrame(
        {
            "checkpoint_vintage": [2026] * 48,
            "model_id": ["model"] * 48,
            "policy_name": ["fixed_d7"] * 48,
            "policy_kind": ["fixed"] * 48,
            "fixed_day": [7] * 48,
            "top_k": [1] * 48,
            "cost_scenario": ["double_slippage"] * 48,
            "trade_date": [f"date_{idx}" for idx in range(48)],
            "selected_mean_net_return": values,
            "universe_mean_net_return": [0.0] * 48,
            "alpha_mean_net_return": values,
            "selected_entry_fill_rate": [1.0] * 48,
            "selected_terminal_recovery_rate": [0.0] * 48,
            "selected_deferred_exit_rate": [0.0] * 48,
            "selected_mean_planned_exit_day": [7.0] * 48,
            "selected_mean_resolved_exit_day": [7.0] * 48,
            "universe_mean_resolved_exit_day": [7.0] * 48,
        }
    )
    row = vintages._aggregate_policy_alpha(frame).iloc[0]
    assert row["alpha_mean_net_return"] > 0.02
    assert row["alpha_median"] == 0.01
    assert row["alpha_trimmed_mean_10pct"] == 0.01
    assert row["alpha_log_speed_median"] < row["annualized_alpha_log_speed"]
    assert row["positive_alpha_log_speed_day"] == 1.0


def test_policy_alpha_aggregation_preserves_explicit_evaluation_year() -> None:
    frame = pd.DataFrame(
        {
            "evaluation_year": [2023, 2024],
            "checkpoint_vintage": [2023, 2024],
            "model_id": ["model_2023", "model_2024"],
            "policy_name": ["fixed_d7", "fixed_d7"],
            "policy_kind": ["fixed", "fixed"],
            "fixed_day": [7, 7],
            "top_k": [1, 1],
            "cost_scenario": ["double_slippage", "double_slippage"],
            "trade_date": ["2023-01-03", "2024-01-02"],
            "selected_mean_net_return": [0.02, 0.03],
            "universe_mean_net_return": [0.01, 0.01],
            "alpha_mean_net_return": [0.01, 0.02],
            "selected_entry_fill_rate": [1.0, 1.0],
            "selected_terminal_recovery_rate": [0.0, 0.0],
            "selected_deferred_exit_rate": [0.0, 0.0],
            "selected_mean_planned_exit_day": [7.0, 7.0],
            "selected_mean_resolved_exit_day": [7.0, 7.0],
            "universe_mean_resolved_exit_day": [7.0, 7.0],
        }
    )
    result = vintages._aggregate_policy_alpha(frame)
    assert result["evaluation_year"].tolist() == [2023, 2024]


def test_historical_policy_contract_covers_all_registered_exits() -> None:
    contract = vintages._historical_policy_alpha_contract()["contract"]
    assert contract["evaluation_years"] == [2023, 2024, 2025]
    assert contract["daily_aggregation"] == "equal_weight_by_signal_date"
    assert contract["year_aggregation"] == "equal_weight_by_development_year"
    assert contract["policies"][0] == "fixed_d2"
    assert contract["policies"][-2:] == ["model_plan", "rolling_path"]
    assert "alpha_log_growth_per_resolved_session" in contract["selection_objective"]


def test_best_policy_rows_keeps_one_complete_dynamic_policy_row() -> None:
    frame = pd.DataFrame(
        {
            "top_k": [1, 1],
            "policy_name": ["rolling_path", "fixed_d2"],
            "fixed_day": [np.nan, 2.0],
            "annualized_alpha_log_speed": [0.9, 0.8],
            "marker": ["rolling", "fixed"],
        }
    )
    row = vintages._best_policy_rows(
        frame,
        group_columns=["top_k"],
        objective="annualized_alpha_log_speed",
    ).iloc[0]
    assert row["policy_name"] == "rolling_path"
    assert math.isnan(float(row["fixed_day"]))
    assert row["marker"] == "rolling"
