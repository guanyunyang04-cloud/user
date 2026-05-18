from __future__ import annotations

import pandas as pd
import torch

from daily_research.path_policy.forecast_dataset import build_forecast_sequence_dataset
from daily_research.path_policy.forecast_training import (
    forecast_evidence_verdict,
    forecast_prediction_metrics,
    make_forecast_model,
    train_forecast_models,
)
from daily_research.path_policy.tests.fixtures import make_prepared_policy_inputs


def test_forecast_model_families_emit_path20_sequence_contract() -> None:
    x = torch.randn(4, 6, 5)
    for family in ("linear_last_day", "mlp_last_day", "gru_sequence", "patch_transformer"):
        model = make_forecast_model(
            family,
            input_dim=5,
            hidden_dim=24,
            horizon=20,
            gru_layers=2,
            transformer_layers=2,
            transformer_heads=3,
            patch_sizes=(2, 3),
        )
        pred = model(x)
        assert set(pred) == {"mu", "q10", "q50", "q90", "aux"}
        assert pred["mu"].shape == (4, 20)
        assert pred["q10"].shape == (4, 20)
        assert pred["aux"].shape == (4, 8)
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
        model_families=("linear_last_day", "mlp_last_day", "gru_sequence", "patch_transformer"),
        epochs=5,
        min_epochs=1,
        early_stop_patience=1,
        batch_size=4,
        lr=1.0e-3,
        hidden_dim=24,
        dropout=0.0,
        seeds=(7, 11),
        device="cpu",
        amp=False,
        gru_layers=2,
        transformer_layers=1,
        transformer_heads=3,
        patch_sizes=(2, 3),
    )

    assert summary["status"] == "completed"
    assert summary["shadow_only"] is True
    assert summary["promotion_allowed"] is False
    assert summary["active_execution_strategy_expected_diff"] == "none"
    assert set(summary["models"]) == {"linear_last_day", "mlp_last_day", "gru_sequence", "patch_transformer"}
    assert summary["selected_seed"] in {7, 11}
    assert summary["selected_signal_profile"] in {"trend_20d", "short_burst", "multiscale", "failed"}
    assert "validation_multiscale_score" in summary
    assert summary["feature_profile"] == dataset.manifest["feature_profile"]
    assert summary["feature_manifest"]["feature_count_after_cap"] == dataset.x.shape[2]
    assert "family_summary" in summary
    assert "seed_summaries" in summary["models"]["patch_transformer"]
    assert (tmp_path / "forecast_training_summary.json").exists()
    assert (tmp_path / "forecast_learning_curve.csv").exists()
    assert (tmp_path / "forecast_predictions_validation.csv").exists()
    assert (tmp_path / "forecast_predictions_test.csv").exists()
    assert (tmp_path / "forecast_model_linear_last_day_seed7_best.pt").exists()

    validation_predictions = pd.read_csv(tmp_path / "forecast_predictions_validation.csv")
    assert {
        "pred_aux_cum_1d",
        "pred_aux_cum_3d",
        "pred_cum_mu_20d",
        "future_cum_excess_return_1d",
        "future_cum_excess_return_3d",
        "future_cum_excess_return_20d",
        "pred_q10_1d",
        "pred_q90_20d",
    }.issubset(validation_predictions.columns)
    validation_metrics = summary["validation_metrics"]
    for horizon in (1, 3, 5, 10, 20):
        assert f"rank_ic_{horizon}d" in validation_metrics
        assert f"top_bottom_spread_{horizon}d" in validation_metrics
        assert f"direction_accuracy_{horizon}d" in validation_metrics
    assert "rank_ic_upside_20d" in validation_metrics
    assert "top_bottom_spread_upside_20d" in validation_metrics


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


def test_forecast_prediction_metrics_accepts_amp_float16_predictions() -> None:
    frame = pd.DataFrame(
        {
            "date": ["2023-01-03"] * 5,
            "stock": ["AAA", "BBB", "CCC", "DDD", "EEE"],
            "future_path_upside_capture_20d": pd.Series([0.03, 0.01, -0.02, 0.04, 0.0], dtype="float16"),
            "pred_aux_upside_20d": pd.Series([0.02, 0.01, -0.03, 0.05, 0.0], dtype="float16"),
        }
    )
    for horizon in (1, 3, 5, 10, 20):
        frame[f"future_cum_excess_return_{horizon}d"] = pd.Series(
            [0.01, -0.01, 0.02, 0.03, -0.02],
            dtype="float16",
        )
        frame[f"pred_cum_mu_{horizon}d"] = pd.Series(
            [0.02, -0.02, 0.01, 0.04, -0.01],
            dtype="float16",
        )
    for step in range(1, 21):
        frame[f"target_excess_{step}d"] = pd.Series([0.001, -0.001, 0.002, 0.003, -0.002], dtype="float16")
        frame[f"pred_q10_{step}d"] = pd.Series([-0.01, -0.01, -0.01, -0.01, -0.01], dtype="float16")
        frame[f"pred_q90_{step}d"] = pd.Series([0.01, 0.01, 0.01, 0.01, 0.01], dtype="float16")

    metrics = forecast_prediction_metrics(frame)

    assert metrics["status"] == "completed"
    assert "top_bottom_spread_20d" in metrics
    assert "top_bottom_spread_upside_20d" in metrics
