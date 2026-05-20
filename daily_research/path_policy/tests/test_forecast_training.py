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
    run_forecast_ranking_baseline,
    train_forecast_models,
)
from daily_research.path_policy.tests.fixtures import make_prepared_policy_inputs


def test_forecast_model_families_emit_path20_sequence_contract() -> None:
    x = torch.randn(4, 6, 5)
    static_ids = torch.tensor(
        [
            [1, 1, 1, 1, 1, 1],
            [2, 2, 2, 2, 2, 2],
            [3, 1, 0, 0, 3, 2],
            [4, 2, 1, 3, 4, 3],
        ],
        dtype=torch.long,
    )
    for family in (
        "linear_last_day",
        "mlp_last_day",
        "gru_sequence",
        "patch_transformer",
        "gru_sequence_static_context",
        "patch_transformer_static_context",
        "stock_mixer_sequence",
        "sector_slot_mixer_sequence",
    ):
        model = make_forecast_model(
            family,
            input_dim=5,
            hidden_dim=24,
            horizon=20,
            gru_layers=2,
            transformer_layers=2,
            transformer_heads=3,
            patch_sizes=(2, 3),
            static_context_vocab_sizes={
                "symbol": 8,
                "exchange": 4,
                "industry": 4,
                "board": 5,
                "liquidity_bucket": 6,
                "price_bucket": 6,
            },
            static_context_embedding_dims={
                "symbol": 4,
                "exchange": 2,
                "industry": 3,
                "board": 3,
                "liquidity_bucket": 2,
                "price_bucket": 2,
            },
        )
        pred = model(x, static_context_ids=static_ids) if "static_context" in family else model(x)
        assert set(pred) == {"mu", "q10", "q50", "q90", "aux"}
        assert pred["mu"].shape == (4, 20)
        assert pred["q10"].shape == (4, 20)
        assert pred["aux"].shape == (4, 8)
        assert torch.all(pred["q10"] <= pred["q50"])
        assert torch.all(pred["q50"] <= pred["q90"])


def test_train_forecast_models_static_context_checkpoint_contract(tmp_path) -> None:
    prepared = make_prepared_policy_inputs(days=420, stocks=("AAA.SZ", "BBB.SH", "CCC.SZ", "DDD.SH"), start_date="2019-07-01")
    prepared.metadata_frames["industry_map"] = pd.DataFrame(
        {
            "symbol": ["AAA.SZ", "BBB.SH", "CCC.SZ"],
            "industry": ["bank", "electronics", "healthcare"],
        }
    )
    dataset = build_forecast_memmap_dataset(
        prepared,
        root=tmp_path / "dataset",
        train_start_year=2019,
        train_end_year=2019,
        validation_year=2020,
        test_year=2021,
        lookback_days=5,
        horizon=20,
        max_samples_per_role=8,
        min_lookback_valid_ratio=0.80,
        include_static_context=True,
    )

    summary = train_forecast_models(
        dataset,
        study_root=tmp_path / "study",
        model_families=("gru_sequence_static_context",),
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
        dataloader_num_workers=0,
    )

    assert summary["status"] == "completed"
    assert summary["models"]["gru_sequence_static_context"]["status"] == "completed"
    assert summary["training_config"]["static_context_schema"]["enabled"] is True
    assert "seen_in_train=true" in summary["validation_stratified_metrics"]

    last_path = tmp_path / "study" / "forecast_model_gru_sequence_static_context_seed7_last.pt"
    checkpoint = torch.load(last_path, map_location="cpu", weights_only=False)
    assert checkpoint["resume_contract"]["static_context_schema"]["enabled"] is True
    assert checkpoint["resume_contract"]["symbol_vocab_fingerprint"] == dataset.manifest["symbol_vocab_fingerprint"]
    assert checkpoint["resume_contract"]["loss_profile"] == "default"

    bad_manifest = {**dataset.manifest, "symbol_vocab_fingerprint": "changed"}
    dataset.manifest = bad_manifest
    with pytest.raises(ValueError, match="symbol_vocab_fingerprint"):
        train_forecast_models(
            dataset,
            study_root=tmp_path / "resume_bad",
            model_families=("gru_sequence_static_context",),
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
            resume_from=last_path,
        )


def test_train_forecast_models_records_loss_profile_in_summary_and_resume_contract(tmp_path) -> None:
    prepared = make_prepared_policy_inputs(days=420, stocks=("AAA.SZ", "BBB.SH", "CCC.SZ", "DDD.SH"), start_date="2019-07-01")
    dataset = build_forecast_memmap_dataset(
        prepared,
        root=tmp_path / "dataset",
        train_start_year=2019,
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
        model_families=("gru_sequence",),
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
        dataloader_num_workers=0,
        loss_profile="rank_aux",
    )

    assert summary["training_config"]["loss_profile"] == "rank_aux"
    seed_summary = summary["models"]["gru_sequence"]["seed_summaries"]["7"]
    checkpoint = torch.load(seed_summary["last_checkpoint_pt"], map_location="cpu", weights_only=False)
    assert checkpoint["resume_contract"]["loss_profile"] == "rank_aux"
    assert checkpoint["training_config"]["loss_profile"] == "rank_aux"


def test_ranking_baseline_dependency_missing_reports_status(tmp_path) -> None:
    prepared = make_prepared_policy_inputs(days=420, stocks=("AAA.SZ", "BBB.SH", "CCC.SZ", "DDD.SH"), start_date="2019-07-01")
    dataset = build_forecast_memmap_dataset(
        prepared,
        root=tmp_path / "dataset",
        train_start_year=2019,
        train_end_year=2019,
        validation_year=2020,
        test_year=2021,
        lookback_days=5,
        horizon=20,
        max_samples_per_role=8,
        min_lookback_valid_ratio=0.80,
    )

    summary = run_forecast_ranking_baseline(dataset, study_root=tmp_path / "study", baseline="missing_ranker")

    assert summary["status"] == "dependency_missing"
    assert summary["baseline"] == "missing_ranker"
    assert summary["shadow_only"] is True
    assert summary["active_execution_strategy_expected_diff"] == "none"


def test_sector_slot_diagnostics_handles_missing_metadata(tmp_path) -> None:
    prepared = make_prepared_policy_inputs(days=420, stocks=("AAA.SZ", "BBB.SH", "CCC.SZ", "DDD.SH"), start_date="2019-07-01")
    dataset = build_forecast_memmap_dataset(
        prepared,
        root=tmp_path / "dataset",
        train_start_year=2019,
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
        model_families=("sector_slot_mixer_sequence",),
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
        dataloader_num_workers=0,
        slot_diagnostics=True,
    )

    slot_path = summary["models"]["sector_slot_mixer_sequence"]["seed_summaries"]["7"]["slot_diagnostics_path"]
    diagnostics = json.loads(__import__("pathlib").Path(slot_path).read_text(encoding="utf-8"))
    assert diagnostics["status"] == "completed"
    assert diagnostics["slot_semantics"] == "dynamic_theme_factor_not_static_board"
    assert diagnostics["slot_count"] == 8
    assert diagnostics["industry_available"] is False


def test_stock_mixer_uses_memmap_date_level_batching(tmp_path) -> None:
    prepared = make_prepared_policy_inputs(days=420, stocks=("AAA.SZ", "BBB.SH", "CCC.SZ", "DDD.SH"), start_date="2019-07-01")
    dataset = build_forecast_memmap_dataset(
        prepared,
        root=tmp_path / "dataset",
        train_start_year=2019,
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
        model_families=("stock_mixer_sequence",),
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
        dataloader_num_workers=0,
    )

    assert summary["status"] == "completed"
    assert summary["models"]["stock_mixer_sequence"]["cross_section_batching_enabled"] is True
    assert summary["models"]["stock_mixer_sequence"]["seed_summaries"]["7"]["status"] == "completed"
    assert summary["training_config"]["cross_section_batching_enabled"] is True
    assert summary["training_config"]["effective_batch_size"] == 1


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
