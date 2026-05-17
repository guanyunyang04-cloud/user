from __future__ import annotations

import pandas as pd
import torch

from daily_research.path_policy.forecast_dataset import build_forecast_sequence_dataset
from daily_research.path_policy.forecast_training import (
    forecast_evidence_verdict,
    make_forecast_model,
    train_forecast_models,
)
from daily_research.path_policy.tests.fixtures import make_prepared_policy_inputs


def test_forecast_model_families_emit_path20_sequence_contract() -> None:
    x = torch.randn(4, 6, 5)
    for family in ("linear_last_day", "gru_sequence", "patch_transformer"):
        model = make_forecast_model(family, input_dim=5, hidden_dim=16, horizon=20)
        pred = model(x)
        assert set(pred) == {"mu", "q10", "q50", "q90", "aux"}
        assert pred["mu"].shape == (4, 20)
        assert pred["q10"].shape == (4, 20)
        assert pred["aux"].shape == (4, 6)
        assert torch.all(pred["q10"] <= pred["q50"])
        assert torch.all(pred["q50"] <= pred["q90"])


def test_train_forecast_models_writes_summary_predictions_and_artifacts(tmp_path) -> None:
    prepared = make_prepared_policy_inputs(days=820, stocks=("AAA", "BBB", "CCC", "DDD"), start_date="2018-01-02")
    dataset = build_forecast_sequence_dataset(
        prepared,
        train_start_year=2018,
        train_end_year=2019,
        validation_year=2020,
        test_year=2021,
        lookback_days=5,
        horizon=20,
        max_samples_per_role=8,
    )

    summary = train_forecast_models(
        dataset,
        study_root=tmp_path,
        model_families=("linear_last_day", "gru_sequence", "patch_transformer"),
        epochs=1,
        batch_size=4,
        lr=1.0e-3,
        hidden_dim=16,
    )

    assert summary["status"] == "completed"
    assert summary["shadow_only"] is True
    assert summary["promotion_allowed"] is False
    assert summary["active_execution_strategy_expected_diff"] == "none"
    assert set(summary["models"]) == {"linear_last_day", "gru_sequence", "patch_transformer"}
    assert (tmp_path / "forecast_training_summary.json").exists()
    assert (tmp_path / "forecast_predictions_validation.csv").exists()
    assert (tmp_path / "forecast_predictions_test.csv").exists()
    assert (tmp_path / "forecast_model_linear_last_day.pt").exists()

    validation_predictions = pd.read_csv(tmp_path / "forecast_predictions_validation.csv")
    assert {"pred_cum_mu_20d", "future_cum_excess_return_20d", "pred_q10_1d", "pred_q90_20d"}.issubset(
        validation_predictions.columns
    )


def test_forecast_evidence_verdict_requires_positive_validation_and_calibration() -> None:
    failed = forecast_evidence_verdict(
        validation_metrics={"status": "completed", "rank_ic_20d": 0.01, "top_bottom_spread_20d": -0.01, "q10_coverage_mean": 0.8, "q90_coverage_mean": 0.8},
        test_metrics={"status": "completed", "rank_ic_20d": 0.1, "top_bottom_spread_20d": 0.1},
    )
    assert failed == "forecast_failed"

    promising = forecast_evidence_verdict(
        validation_metrics={"status": "completed", "rank_ic_20d": 0.01, "top_bottom_spread_20d": 0.01, "q10_coverage_mean": 0.8, "q90_coverage_mean": 0.8},
        test_metrics={"status": "completed", "rank_ic_20d": -0.1, "top_bottom_spread_20d": -0.1},
    )
    assert promising == "forecast_promising"

    confirmed = forecast_evidence_verdict(
        validation_metrics={"status": "completed", "rank_ic_20d": 0.01, "top_bottom_spread_20d": 0.01, "q10_coverage_mean": 0.8, "q90_coverage_mean": 0.8},
        test_metrics={"status": "completed", "rank_ic_20d": 0.1, "top_bottom_spread_20d": 0.1},
    )
    assert confirmed == "forecast_test_confirmed"
