from __future__ import annotations

import json

import pandas as pd
import pytest
import torch

from daily_research.path_policy.forecast_dataset import build_forecast_memmap_dataset, build_forecast_sequence_dataset
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
    prepared = make_prepared_policy_inputs(days=420, stocks=("AAA", "BBB", "CCC", "DDD"), start_date="2019-07-01")
    dataset = build_forecast_sequence_dataset(
        prepared,
        train_start_year=2019,
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


def test_train_forecast_models_accepts_memmap_dataset_view(tmp_path) -> None:
    prepared = make_prepared_policy_inputs(days=820, stocks=("AAA", "BBB", "CCC", "DDD"), start_date="2018-01-02")
    dataset = build_forecast_memmap_dataset(
        prepared,
        root=tmp_path / "dataset",
        train_start_year=2018,
        train_end_year=2019,
        validation_year=2020,
        test_year=2021,
        lookback_days=5,
        horizon=20,
        max_samples_per_role=8,
        min_lookback_valid_ratio=0.80,
    )

    summary = train_forecast_models(
        dataset,
        study_root=tmp_path / "study",
        model_families=("linear_last_day", "mlp_last_day"),
        epochs=2,
        min_epochs=1,
        early_stop_patience=1,
        batch_size=4,
        lr=1.0e-3,
        hidden_dim=24,
        dropout=0.0,
        seeds=(7,),
        device="cpu",
        amp=False,
        dataloader_num_workers=0,
    )

    assert summary["status"] == "completed"
    assert summary["dataset_mode"] == "memmap"
    assert summary["feature_manifest"]["feature_count_after_cap"] == dataset.input_dim
    assert (tmp_path / "study" / "forecast_predictions_validation.csv").exists()
    assert "validation_stratified_metrics" in summary


def test_train_forecast_models_writes_last_checkpoint_progress_and_incremental_curve(tmp_path) -> None:
    prepared = make_prepared_policy_inputs(days=420, stocks=("AAA", "BBB", "CCC", "DDD"), start_date="2019-07-01")
    dataset = build_forecast_sequence_dataset(
        prepared,
        train_start_year=2019,
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
        model_families=("linear_last_day",),
        epochs=2,
        min_epochs=2,
        early_stop_patience=5,
        batch_size=4,
        lr=1.0e-3,
        hidden_dim=24,
        dropout=0.0,
        seeds=(7,),
        device="cpu",
        amp=False,
    )

    best_path = tmp_path / "forecast_model_linear_last_day_seed7_best.pt"
    last_path = tmp_path / "forecast_model_linear_last_day_seed7_last.pt"
    progress_path = tmp_path / "forecast_progress.json"
    learning_curve_path = tmp_path / "forecast_learning_curve.csv"
    assert best_path.exists()
    assert last_path.exists()
    assert progress_path.exists()
    assert learning_curve_path.exists()

    checkpoint = torch.load(last_path, map_location="cpu", weights_only=False)
    assert checkpoint["checkpoint_kind"] == "last"
    assert checkpoint["model_family"] == "linear_last_day"
    assert checkpoint["seed"] == 7
    assert checkpoint["epoch"] == 2
    assert checkpoint["best_epoch"] >= 1
    assert "optimizer_state_dict" in checkpoint
    assert "scaler_state_dict" in checkpoint
    assert checkpoint["patience_used"] >= 0
    assert checkpoint["training_config"]["epochs"] == 2
    assert checkpoint["feature_columns"] == list(dataset.feature_columns)

    learning_curve = pd.read_csv(learning_curve_path)
    assert list(learning_curve["epoch"]) == [1, 2]
    assert "epoch_seconds" in learning_curve.columns

    progress = json.loads(progress_path.read_text(encoding="utf-8"))
    assert progress["status"] == "completed"
    assert progress["current_epoch"] == 2
    assert progress["max_epochs"] == 2
    assert progress["estimated_remaining_seconds"] == 0.0
    assert progress["best_epoch"] >= 1
    assert progress["last_checkpoint_pt"] == str(last_path.resolve())
    assert progress["best_checkpoint_pt"] == str(best_path.resolve())
    assert "estimated_remaining_seconds" in progress
    assert "eta_at" in progress

    seed_summary = summary["models"]["linear_last_day"]["seed_summaries"]["7"]
    assert seed_summary["epochs_ran"] == 2
    assert seed_summary["last_checkpoint_pt"] == str(last_path.resolve())
    assert summary["progress_json"] == str(progress_path.resolve())


def test_train_forecast_models_resumes_from_strict_last_checkpoint(tmp_path) -> None:
    prepared = make_prepared_policy_inputs(days=420, stocks=("AAA", "BBB", "CCC", "DDD"), start_date="2019-07-01")
    dataset = build_forecast_sequence_dataset(
        prepared,
        train_start_year=2019,
        train_end_year=2019,
        validation_year=2020,
        test_year=2021,
        lookback_days=5,
        horizon=20,
        max_samples_per_role=8,
    )

    first_summary = train_forecast_models(
        dataset,
        study_root=tmp_path / "first",
        model_families=("linear_last_day",),
        epochs=1,
        min_epochs=1,
        early_stop_patience=5,
        batch_size=4,
        lr=1.0e-3,
        hidden_dim=24,
        dropout=0.0,
        seeds=(7,),
        device="cpu",
        amp=False,
    )
    resume_path = tmp_path / "first" / "forecast_model_linear_last_day_seed7_last.pt"
    assert first_summary["models"]["linear_last_day"]["seed_summaries"]["7"]["epochs_ran"] == 1

    resumed_summary = train_forecast_models(
        dataset,
        study_root=tmp_path / "resumed",
        model_families=("linear_last_day",),
        epochs=2,
        min_epochs=2,
        early_stop_patience=5,
        batch_size=4,
        lr=1.0e-3,
        hidden_dim=24,
        dropout=0.0,
        seeds=(7,),
        device="cpu",
        amp=False,
        resume_from=resume_path,
    )

    seed_summary = resumed_summary["models"]["linear_last_day"]["seed_summaries"]["7"]
    assert resumed_summary["resume_from_checkpoint_pt"] == str(resume_path.resolve())
    assert seed_summary["resume_from_checkpoint_pt"] == str(resume_path.resolve())
    assert seed_summary["resume_start_epoch"] == 2
    assert seed_summary["epochs_ran"] == 2
    resumed_last = torch.load(
        tmp_path / "resumed" / "forecast_model_linear_last_day_seed7_last.pt",
        map_location="cpu",
        weights_only=False,
    )
    assert resumed_last["epoch"] == 2


def test_train_forecast_models_rejects_resume_checkpoint_contract_mismatch(tmp_path) -> None:
    prepared = make_prepared_policy_inputs(days=420, stocks=("AAA", "BBB", "CCC", "DDD"), start_date="2019-07-01")
    dataset = build_forecast_sequence_dataset(
        prepared,
        train_start_year=2019,
        train_end_year=2019,
        validation_year=2020,
        test_year=2021,
        lookback_days=5,
        horizon=20,
        max_samples_per_role=8,
    )
    train_forecast_models(
        dataset,
        study_root=tmp_path / "first",
        model_families=("linear_last_day",),
        epochs=1,
        min_epochs=1,
        early_stop_patience=5,
        batch_size=4,
        lr=1.0e-3,
        hidden_dim=24,
        dropout=0.0,
        seeds=(7,),
        device="cpu",
        amp=False,
    )

    with pytest.raises(ValueError, match="model_family"):
        train_forecast_models(
            dataset,
            study_root=tmp_path / "mismatch",
            model_families=("mlp_last_day",),
            epochs=2,
            min_epochs=2,
            early_stop_patience=5,
            batch_size=4,
            lr=1.0e-3,
            hidden_dim=24,
            dropout=0.0,
            seeds=(7,),
            device="cpu",
            amp=False,
            resume_from=tmp_path / "first" / "forecast_model_linear_last_day_seed7_last.pt",
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
