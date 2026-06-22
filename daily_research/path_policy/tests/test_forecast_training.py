from __future__ import annotations

import json
from pathlib import Path
from dataclasses import replace

import pandas as pd
import pytest
import torch

from daily_research.path_policy.forecast_training import (
    FORECAST_MODEL_FAMILIES,
    _date_grouped_forecast_loss,
    _forecast_finite_guard,
    forecast_evidence_verdict,
    forecast_loss_profile_contract,
    forecast_prediction_metrics,
    make_forecast_model,
    ranking_relevance_labels,
    run_forecast_ranking_baseline,
    train_forecast_models,
)
from daily_research.path_policy.forecast_dataset import (
    build_qdp_date_slate_training_pack,
    build_qdp_training_pack,
    load_forecast_memmap_dataset,
)
from daily_research.path_policy.tests.fixtures import make_tiny_forecast_memmap_dataset, make_tiny_forecast_sequence_dataset
from daily_research.path_policy.tests.test_forecast_memmap_dataset import _write_qdp_fixture_shard


pytestmark = [pytest.mark.research, pytest.mark.slow]


def test_forecast_model_families_emit_path20_sequence_contract() -> None:
    from daily_research.path_policy.models import PATH20_DEFAULT_CUMULATIVE_HORIZONS, path20_forecast_aux_dim

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
        "dlinear_sequence",
        "mlp_last_day",
        "gru_sequence",
        "patch_transformer",
        "gru_sequence_static_context",
        "patch_transformer_static_context",
        "stock_mixer_sequence",
        "sector_slot_mixer_sequence",
        "hybrid_expert_fusion_static_context",
        "hybrid_multiscale_recency_aware_v1",
        "hybrid_structured_alpha_v2",
        "regime_routed_multi_expert_horizon_v1",
    ):
        static_fields = ("exchange", "industry") if family == "hybrid_structured_alpha_v2" else None
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
            static_context_fields=static_fields,
        )
        pred = (
            model(x, static_context_ids=static_ids)
            if "static_context" in family
            or family
            in {
                "hybrid_multiscale_recency_aware_v1",
                "hybrid_structured_alpha_v2",
                "regime_routed_multi_expert_horizon_v1",
            }
            else model(x)
        )
        expected = {"mu", "q10", "q50", "q90", "aux"}
        if family in {"hybrid_multiscale_recency_aware_v1", "hybrid_structured_alpha_v2", "regime_routed_multi_expert_horizon_v1"}:
            assert expected.issubset(pred)
            expected_router_width = 5 if family == "regime_routed_multi_expert_horizon_v1" else 4
            assert pred["router_weights"].shape == (4, expected_router_width)
            assert torch.isfinite(pred["router_entropy"]).all()
        else:
            assert set(pred) == expected
        assert pred["mu"].shape == (4, 20)
        assert pred["q10"].shape == (4, 20)
        assert pred["aux"].shape == (4, path20_forecast_aux_dim(PATH20_DEFAULT_CUMULATIVE_HORIZONS))
        assert torch.all(pred["q10"] <= pred["q50"])
        assert torch.all(pred["q50"] <= pred["q90"])


def test_make_forecast_model_registers_hybrid_expert_fusion_static_context() -> None:
    assert "hybrid_expert_fusion_static_context" in FORECAST_MODEL_FAMILIES
    model = make_forecast_model(
        "hybrid_expert_fusion_static_context",
        input_dim=5,
        hidden_dim=12,
        horizon=20,
        gru_layers=1,
        transformer_layers=1,
        transformer_heads=3,
        patch_sizes=(2,),
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
    pred = model(torch.randn(2, 6, 5), static_context_ids=torch.ones(2, 6, dtype=torch.long))
    assert set(pred) == {"mu", "q10", "q50", "q90", "aux"}


def test_make_forecast_model_registers_regime_routed_multi_expert_horizon_v1() -> None:
    from daily_research.path_policy.models import path20_decision_aux_dim, path20_forecast_aux_dim

    horizons = (1, 2, 3, 5, 8, 10, 15, 20, 30)
    assert "regime_routed_multi_expert_horizon_v1" in FORECAST_MODEL_FAMILIES
    model = make_forecast_model(
        "regime_routed_multi_expert_horizon_v1",
        input_dim=5,
        hidden_dim=16,
        horizon=30,
        gru_layers=1,
        transformer_layers=1,
        transformer_heads=4,
        patch_sizes=(2,),
        output_profile="decision_utility_v1",
        cumulative_horizons=horizons,
        static_context_vocab_sizes={
            "symbol": 8,
            "exchange": 4,
            "industry": 4,
            "liquidity_bucket": 6,
            "price_bucket": 6,
        },
        static_context_embedding_dims={
            "symbol": 4,
            "exchange": 2,
            "industry": 3,
            "liquidity_bucket": 2,
            "price_bucket": 2,
        },
    )

    pred = model(torch.randn(2, 6, 5), static_context_ids=torch.ones(2, 5, dtype=torch.long))

    assert {"mu", "q10", "q50", "q90", "aux", "decision_aux", "router_weights", "router_entropy"}.issubset(pred)
    assert pred["mu"].shape == (2, 30)
    assert pred["aux"].shape == (2, path20_forecast_aux_dim(horizons, horizon=30))
    assert pred["decision_aux"].shape == (2, path20_decision_aux_dim(horizons, horizon=30))
    assert pred["router_weights"].shape == (2, 5)
    assert torch.allclose(pred["router_weights"].sum(dim=-1), torch.ones(2), atol=1.0e-6)


def test_make_forecast_model_registers_date_slate_alpha_fusion_v1() -> None:
    assert "date_slate_alpha_fusion_v1" in FORECAST_MODEL_FAMILIES
    model = make_forecast_model(
        "date_slate_alpha_fusion_v1",
        input_dim=9,
        hidden_dim=12,
        horizon=20,
        dropout=0.0,
        transformer_layers=1,
        transformer_heads=3,
        patch_sizes=(2,),
        output_profile="forecast_incremental_path_v2",
        static_context_vocab_sizes={"exchange": 4, "industry": 6},
        static_context_embedding_dims={"exchange": 2, "industry": 3},
        static_context_fields=("exchange", "industry"),
        feature_group_indices={
            "daily_price_volume": (0, 1),
            "cross_section": (2,),
            "market_regime": (3,),
            "industry_peer": (4,),
            "valuation_liquidity": (5,),
            "event_quality": (6,),
            "intraday": (7, 8),
        },
    )
    pred = model(torch.randn(2, 6, 9), static_context_ids=torch.ones(2, 2, dtype=torch.long))

    assert {"mu", "q10", "q50", "q90", "aux", "derived_cum_mu", "router_weights"}.issubset(pred)
    assert pred["mu"].shape == (2, 20)
    assert pred["router_weights"].shape == (2, 3)

    assert "date_slate_cross_stock_alpha_fusion_v1" in FORECAST_MODEL_FAMILIES
    cross_model = make_forecast_model(
        "date_slate_cross_stock_alpha_fusion_v1",
        input_dim=9,
        hidden_dim=12,
        horizon=20,
        dropout=0.0,
        transformer_layers=1,
        transformer_heads=3,
        patch_sizes=(2,),
        output_profile="forecast_incremental_path_v2",
        static_context_vocab_sizes={"exchange": 4, "industry": 6},
        static_context_embedding_dims={"exchange": 2, "industry": 3},
        static_context_fields=("exchange", "industry"),
        feature_group_indices={
            "daily_price_volume": (0, 1),
            "cross_section": (2,),
            "market_regime": (3,),
            "industry_peer": (4,),
            "valuation_liquidity": (5,),
            "event_quality": (6,),
            "intraday": (7, 8),
        },
    )
    cross_pred = cross_model(
        torch.randn(4, 6, 9),
        static_context_ids=torch.ones(4, 2, dtype=torch.long),
        date_group_ids=torch.tensor([0, 0, 1, 1], dtype=torch.long),
    )
    assert {"mu", "q10", "q50", "q90", "aux", "derived_cum_mu", "base_mu", "cross_gate_mean"}.issubset(cross_pred)
    assert cross_pred["mu"].shape == (4, 20)


def test_date_grouped_alpha_score_contract_and_loss_respects_date_groups() -> None:
    contract = forecast_loss_profile_contract("date_grouped_alpha_score_v1", cumulative_horizons=(1, 3), forecast_horizon=5)
    assert contract["required_output_profile"] == "forecast_incremental_path_v2"
    assert contract["alpha_score_objective"]["rank_loss_scope"] == "within_same_prediction_date_only"
    prediction = {
        "mu": torch.tensor(
            [
                [0.0, 0.0, 0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0, 0.0, 0.0],
            ],
            dtype=torch.float32,
        ),
        "q10": torch.zeros(4, 5),
        "q50": torch.zeros(4, 5),
        "q90": torch.zeros(4, 5),
        "aux": torch.zeros(4, 8),
    }
    y_daily = torch.zeros(4, 5)
    y_cum = torch.tensor([[0.0, 0.0], [1.0, 1.0], [10.0, 10.0], [11.0, 11.0]], dtype=torch.float32)
    y_risk = torch.zeros(4, 2, 3)
    same_date_loss = _date_grouped_forecast_loss(
        prediction,
        y_daily,
        y_cum,
        y_risk,
        torch.tensor([0, 0, 1, 1], dtype=torch.long),
        loss_profile="date_grouped_alpha_score_v1",
        cumulative_horizons=(1, 3),
        rank_min_group_size=2,
        rank_max_pairs_per_date=16,
    )
    cross_mixed_loss = _date_grouped_forecast_loss(
        prediction,
        y_daily,
        y_cum,
        y_risk,
        torch.tensor([0, 1, 0, 1], dtype=torch.long),
        loss_profile="date_grouped_alpha_score_v1",
        cumulative_horizons=(1, 3),
        rank_min_group_size=2,
        rank_max_pairs_per_date=16,
    )
    assert same_date_loss < cross_mixed_loss


def test_date_listwise_alpha_score_contract_and_loss_uses_cross_stock_outputs() -> None:
    contract = forecast_loss_profile_contract("date_listwise_alpha_score_v2", cumulative_horizons=(1, 3), forecast_horizon=5)
    assert contract["required_output_profile"] == "forecast_incremental_path_v2"
    assert contract["alpha_score_objective"]["requires_model_level_cross_stock_context"] is True
    prediction = {
        "mu": torch.tensor(
            [
                [0.0, 0.0, 0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0, 0.0, 0.0],
            ],
            dtype=torch.float32,
        ),
        "q10": torch.zeros(4, 5),
        "q50": torch.zeros(4, 5),
        "q90": torch.zeros(4, 5),
        "aux": torch.zeros(4, 8),
        "base_mu": torch.zeros(4, 5),
        "cross_gate_mean": torch.full((4,), 0.10),
        "cross_residual_norm": torch.full((4,), 0.20),
    }
    y_daily = torch.zeros(4, 5)
    y_cum = torch.tensor([[0.0, 0.0], [1.0, 1.0], [10.0, 10.0], [11.0, 11.0]], dtype=torch.float32)
    y_risk = torch.zeros(4, 2, 3)
    loss = _date_grouped_forecast_loss(
        prediction,
        y_daily,
        y_cum,
        y_risk,
        torch.tensor([0, 0, 1, 1], dtype=torch.long),
        loss_profile="date_listwise_alpha_score_v2",
        cumulative_horizons=(1, 3),
        rank_min_group_size=2,
        rank_max_pairs_per_date=16,
    )
    assert torch.isfinite(loss)
    assert loss.item() > 0.0


def test_forecast_finite_guard_writes_bad_batch_dump(tmp_path) -> None:
    with pytest.raises(FloatingPointError, match="forecast_nonfinite_detected"):
        _forecast_finite_guard(
            enabled=True,
            dump_dir=tmp_path,
            reason_prefix="unit",
            model_family="linear_last_day",
            seed=7,
            epoch=1,
            step=2,
            role="train",
            row_ids=torch.tensor([10, 11]),
            date_group_ids=torch.tensor([0, 0]),
            tensors={"x": torch.tensor([[1.0, float("nan")]])},
        )
    dumps = sorted(tmp_path.glob("bad_batch_*.json"))
    assert len(dumps) == 1
    payload = json.loads(dumps[0].read_text(encoding="utf-8"))
    assert payload["status"] == "nonfinite_detected"
    assert payload["tensor_summaries"]["x"]["nonfinite_count"] == 1


def test_train_forecast_models_date_slate_alpha_fusion_v1_end_to_end(tmp_path) -> None:
    stocks = ("AAA.SZ", "BBB.SH")
    feature_columns = ["daily_price_return_1d", "cs_rank_price_return_1d", "intraday_close_last"]
    shards = [
        _write_qdp_fixture_shard(tmp_path / "source", year=year, stocks=stocks, feature_columns=feature_columns)
        for year in (2019, 2020, 2021)
    ]
    source_manifest = {
        "artifact_type": "qdp_sharded_memmap",
        "profile": "style_structural_alpha_v2",
        "feature_profile": "style_structural_alpha_v2",
        "canonical_dataset_id": "policy_input_bundle__unit",
        "source_pool_view_id": "policy_pool_view__unit",
        "source_pool_view_kind": "tradeable_mainboard",
        "lookback_days": 3,
        "horizon": 1,
        "forecast_horizon": 1,
        "execution_mode": "next_open",
        "cumulative_horizons": [1],
        "feature_columns": feature_columns,
        "feature_count": len(feature_columns),
        "label_schema_name": "path20_basic_v2",
        "label_schema_version": 2,
        "static_context_schema": {
            "enabled": True,
            "fields": ["symbol", "exchange", "industry"],
            "id_columns": ["symbol_id", "exchange_id", "industry_id"],
            "vocab_sizes": {"symbol": 3, "exchange": 3, "industry": 12},
            "embedding_defaults": {"symbol": 4, "exchange": 2, "industry": 3, "dropout": 0.0},
        },
        "shards": shards,
    }
    source_manifest_path = tmp_path / "qdp_sharded_manifest.json"
    source_manifest_path.write_text(json.dumps(source_manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    training_pack = build_qdp_training_pack(
        source_manifest_path,
        output_root=tmp_path / "training_pack",
        train_start_year=2019,
        train_end_year=2019,
        validation_year=2020,
        test_year=2021,
        feature_dtype="float32",
        stock_chunk_size=1,
        resume=False,
    )
    date_slate_pack = build_qdp_date_slate_training_pack(
        training_pack["manifest_json"],
        output_root=tmp_path / "date_slate_pack",
        feature_dtype="float32",
        stock_chunk_size=1,
        resume=False,
    )
    dataset = load_forecast_memmap_dataset(date_slate_pack["manifest_json"])

    summary = train_forecast_models(
        dataset,
        study_root=tmp_path / "study",
        model_families=("date_slate_alpha_fusion_v1",),
        epochs=1,
        min_epochs=1,
        early_stop_patience=2,
        batch_size=2,
        lr=1.0e-3,
        hidden_dim=12,
        dropout=0.0,
        transformer_layers=1,
        transformer_heads=3,
        patch_sizes=(2,),
        seeds=(7,),
        device="cpu",
        amp=False,
        dataloader_num_workers=0,
        selection_profile="validation_loss",
        output_profile="forecast_incremental_path_v2",
        loss_profile="date_grouped_alpha_score_v1",
        static_context_fields_override=("exchange", "industry"),
        date_slate_dates_per_batch=1,
        date_slate_stocks_per_date=2,
        rank_min_group_size=2,
        rank_max_pairs_per_date=16,
        finite_guard=True,
        bad_batch_dump_dir=tmp_path / "bad_batches",
        per_epoch_prediction_metrics=False,
    )

    assert summary["status"] == "completed"
    family_summary = summary["models"]["date_slate_alpha_fusion_v1"]
    assert family_summary["date_slate_batching_enabled"] is True
    assert family_summary["effective_batch_size"] == 2
    assert family_summary["output_profile"] == "forecast_incremental_path_v2"
    assert family_summary["loss_profile"] == "date_grouped_alpha_score_v1"
    assert family_summary["static_context_schema"]["fields"] == ["exchange", "industry"]
    seed_summary = family_summary["seed_summaries"]["7"]
    assert seed_summary["best_epoch"] == 1
    checkpoint = torch.load(seed_summary["last_checkpoint_pt"], map_location="cpu", weights_only=False)
    assert checkpoint["resume_contract"]["output_profile"] == "forecast_incremental_path_v2"
    assert checkpoint["resume_contract"]["loss_profile"] == "date_grouped_alpha_score_v1"
    assert checkpoint["training_config"]["date_slate_batching_enabled"] is True
    assert checkpoint["training_config"]["rank_min_group_size"] == 2
    assert not list((tmp_path / "bad_batches").glob("bad_batch_*.json"))

    cross_summary = train_forecast_models(
        dataset,
        study_root=tmp_path / "cross_study",
        model_families=("date_slate_cross_stock_alpha_fusion_v1",),
        epochs=1,
        min_epochs=1,
        early_stop_patience=2,
        batch_size=2,
        lr=1.0e-3,
        hidden_dim=12,
        dropout=0.0,
        transformer_layers=1,
        transformer_heads=3,
        patch_sizes=(2,),
        seeds=(7,),
        device="cpu",
        amp=False,
        dataloader_num_workers=0,
        selection_profile="validation_loss",
        output_profile="forecast_incremental_path_v2",
        loss_profile="date_listwise_alpha_score_v2",
        static_context_fields_override=("exchange", "industry"),
        date_slate_dates_per_batch=1,
        date_slate_stocks_per_date=2,
        rank_min_group_size=2,
        rank_max_pairs_per_date=16,
        finite_guard=True,
        bad_batch_dump_dir=tmp_path / "cross_bad_batches",
        per_epoch_prediction_metrics=False,
    )

    assert cross_summary["status"] == "completed"
    cross_family_summary = cross_summary["models"]["date_slate_cross_stock_alpha_fusion_v1"]
    assert cross_family_summary["date_slate_batching_enabled"] is True
    assert cross_family_summary["loss_profile"] == "date_listwise_alpha_score_v2"
    assert cross_family_summary["loss_profile_contract"]["alpha_score_objective"]["requires_model_level_cross_stock_context"] is True
    cross_seed_summary = cross_family_summary["seed_summaries"]["7"]
    cross_checkpoint = torch.load(cross_seed_summary["last_checkpoint_pt"], map_location="cpu", weights_only=False)
    assert cross_checkpoint["resume_contract"]["output_profile"] == "forecast_incremental_path_v2"
    assert cross_checkpoint["resume_contract"]["loss_profile"] == "date_listwise_alpha_score_v2"
    assert cross_checkpoint["training_config"]["date_slate_batching_enabled"] is True
    assert cross_checkpoint["training_config"]["date_slate_semantics"]["full_train_date_slate"] is True
    assert not list((tmp_path / "cross_bad_batches").glob("bad_batch_*.json"))


def test_train_forecast_models_accepts_dlinear_sequence_checkpoint_and_predictions(tmp_path) -> None:
    dataset = make_tiny_forecast_sequence_dataset(lookback_days=5, horizon=20)

    summary = train_forecast_models(
        dataset,
        study_root=tmp_path / "study",
        model_families=("dlinear_sequence",),
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

    assert summary["status"] == "completed"
    assert summary["models"]["dlinear_sequence"]["status"] == "completed"
    assert (tmp_path / "study" / "forecast_predictions_validation.csv").exists()
    assert (tmp_path / "study" / "forecast_predictions_test.csv").exists()
    checkpoint = torch.load(
        tmp_path / "study" / "forecast_model_dlinear_sequence_seed7_last.pt",
        map_location="cpu",
        weights_only=False,
    )
    assert checkpoint["model_family"] == "dlinear_sequence"
    assert checkpoint["resume_contract"]["model_family"] == "dlinear_sequence"


def test_train_forecast_models_static_context_checkpoint_contract(tmp_path) -> None:
    dataset = make_tiny_forecast_memmap_dataset(tmp_path / "dataset", lookback_days=5, horizon=20, include_static_context=True)

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
    dataset = make_tiny_forecast_memmap_dataset(tmp_path / "dataset", lookback_days=5, horizon=20)

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


def test_auxiliary_decision_loss_profiles_record_weight_contract_and_finite_loss() -> None:
    from daily_research.path_policy.forecast_training import _forecast_loss
    from daily_research.path_policy.models import path20_decision_aux_dim, path20_forecast_aux_dim

    horizons = (1, 2, 3, 5, 8, 10, 15, 20, 30)
    for profile in (
        "forecast_path_v1_baseline",
        "decision_utility_v1_baseline",
        "decision_utility_path_aux_v1",
        "decision_utility_hit_risk_aux_v1",
        "decision_utility_rank_aux_v1",
        "score_monthly_robust_v1",
        "horizon_entropy_regularized_v1",
        "risk_drawdown_reweighted_v1",
        "horizon_target_normalized_v1",
        "horizon_head_soft_constraint_v1",
        "target_norm_head_constraint_v1",
        "horizon_30d_soft_penalty_v1",
        "topn_excess_rank_v1",
        "score_to_weight_proxy_v1",
        "bad_month_aware_v1",
        "personal_alpha_scorer_hybrid_v1",
        "personal_time_efficient_topk_v1",
    ):
        contract = forecast_loss_profile_contract(profile, cumulative_horizons=horizons, forecast_horizon=30)
        assert contract["loss_profile"] == profile
        assert contract["status"] == "active"
        assert contract["cumulative_horizons"] == list(horizons)
        assert "loss_component_weights" in contract
        if contract["required_output_profile"] == "decision_utility_v1":
            weights = contract["loss_component_weights"]
            assert contract["required_output_profile"] == "decision_utility_v1"
            assert weights["decision_utility"] > weights["path_daily"]
            assert weights["decision_utility"] > weights["risk_aux"]
            if profile not in {
                "score_monthly_robust_v1",
                "horizon_head_soft_constraint_v1",
                "topn_excess_rank_v1",
                "score_to_weight_proxy_v1",
                "bad_month_aware_v1",
                "personal_alpha_scorer_hybrid_v1",
            }:
                assert weights["decision_utility"] > weights["rank_aux"]
            if profile == "score_monthly_robust_v1":
                assert weights["decision_rank_aux"] > 0.45
                assert weights["rank_aux"] >= 1.0
            if profile == "horizon_entropy_regularized_v1":
                assert weights["horizon_entropy"] > 0.0
                assert weights["horizon_classification"] < 0.20
            if profile == "risk_drawdown_reweighted_v1":
                assert weights["risk_aux"] >= 0.20
                assert weights["downside_rank_aux"] >= 0.02
            if profile == "horizon_target_normalized_v1":
                assert contract["target_normalization"] == "per_horizon_utility_zscore"
                assert weights["horizon_target_normalization"] > 0.0
                assert weights["horizon_head_soft_constraint"] == 0.0
            if profile == "horizon_head_soft_constraint_v1":
                assert contract["target_normalization"] == "none"
                assert weights["horizon_head_soft_constraint"] > 0.0
                assert weights["horizon_head_soft_constraint"] < weights["decision_utility"]
                assert contract["horizon_head_constraint"]["max_30d_probability"] == 0.75
            if profile == "target_norm_head_constraint_v1":
                assert contract["target_normalization"] == "per_horizon_utility_zscore"
                assert weights["horizon_target_normalization"] > 0.0
                assert weights["horizon_head_soft_constraint"] > 0.0
            if profile == "horizon_30d_soft_penalty_v1":
                calibration = contract["decision_score_calibration"]
                assert calibration["enabled"] is True
                assert calibration["method"] == "max_horizon_utility_soft_penalty"
                assert calibration["penalized_horizon"] == 30
                assert calibration["utility_penalty"] == pytest.approx(0.005)
                assert weights["calibrated_decision_rank_aux"] > 0.0
            proxy = contract["high_return_proxy_objective"]
            if profile == "topn_excess_rank_v1":
                assert proxy["enabled"] is True
                assert proxy["method"] == "batch_top_quintile_excess_rank_surrogate"
                assert proxy["uses_active_execution_artifact"] is False
                assert proxy["not_a_backtest"] is True
                assert weights["topn_excess_rank"] > 0.0
            elif profile == "score_to_weight_proxy_v1":
                assert proxy["enabled"] is True
                assert proxy["method"] == "batch_soft_topn_score_to_weight_surrogate"
                assert proxy["uses_active_execution_artifact"] is False
                assert proxy["not_a_backtest"] is True
                assert weights["score_to_weight_proxy"] > 0.0
            elif profile == "bad_month_aware_v1":
                assert proxy["enabled"] is True
                assert proxy["method"] == "batch_downside_tail_reweighted_score_surrogate"
                assert proxy["uses_active_execution_artifact"] is False
                assert proxy["not_a_backtest"] is True
                assert weights["bad_month_aware"] > 0.0
            elif profile == "personal_alpha_scorer_hybrid_v1":
                assert proxy["enabled"] is True
                assert proxy["method"] == "hybrid_batch_top_tail_soft_weight_downside_surrogate"
                assert "not date-cross-sectional topK" in proxy["description"]
                assert proxy["uses_active_execution_artifact"] is False
                assert proxy["not_a_backtest"] is True
                assert weights["topn_excess_rank"] > 0.0
                assert weights["score_to_weight_proxy"] > 0.0
                assert weights["bad_month_aware"] > 0.0
                assert weights["downside_rank_aux"] > 0.0
            elif profile == "personal_time_efficient_topk_v1":
                time_eff = contract["time_efficient_topk_objective"]
                assert contract["primary_objective"] == "personal_time_efficient_topk"
                assert contract["target_normalization"] == "per_horizon_utility_divided_by_day"
                assert time_eff["enabled"] is True
                assert time_eff["method"] == "max_net_utility_per_horizon_day"
                assert weights["decision_utility"] > weights["time_eff_topk_alignment"]
                assert weights["decision_rank_aux"] > weights["rank_aux"]
            else:
                assert proxy["enabled"] is False

            prediction = {
                "mu": torch.randn(6, 30) * 0.01,
                "q10": torch.randn(6, 30) * 0.01 - 0.01,
                "q50": torch.randn(6, 30) * 0.01,
                "q90": torch.randn(6, 30) * 0.01 + 0.01,
                "aux": torch.randn(6, path20_forecast_aux_dim(horizons, horizon=30)) * 0.01,
                "decision_aux": torch.randn(6, path20_decision_aux_dim(horizons, horizon=30)) * 0.01,
            }
            loss = _forecast_loss(
                prediction,
                torch.randn(6, 30) * 0.01,
                torch.randn(6, len(horizons)) * 0.01,
                torch.randn(6, len(horizons), 3) * 0.01,
                loss_profile=profile,
                target_scale=1.0,
                decision_cost_bps=20.0,
                decision_hit_threshold_bps=10.0,
                decision_drawdown_penalty=0.10,
                cumulative_horizons=horizons,
            )
            assert torch.isfinite(loss)


def test_horizon_30d_soft_penalty_profile_calibrates_prediction_score_and_best_horizon() -> None:
    from daily_research.path_policy.forecast_training import _add_decision_utility_columns
    from daily_research.path_policy.models import path20_decision_aux_dim

    horizons = (1, 2, 3, 5, 8, 10, 15, 20, 30)
    utility = torch.zeros(2, len(horizons), dtype=torch.float32)
    utility[:, horizons.index(20)] = 0.100
    utility[:, horizons.index(30)] = 0.103
    hit_logits = torch.zeros_like(utility)
    horizon_logits = torch.zeros_like(utility)
    horizon_logits[:, horizons.index(30)] = 4.0
    decision_aux = torch.cat([utility, hit_logits, horizon_logits], dim=1).numpy()
    assert decision_aux.shape[1] == path20_decision_aux_dim(horizons, horizon=30)

    columns: dict[str, object] = {}
    _add_decision_utility_columns(
        columns,
        predictions={"decision_aux": decision_aux},
        y_cum=torch.zeros(2, len(horizons)).numpy(),
        drawdown_by_horizon=torch.zeros(2, len(horizons)).numpy(),
        target_scale=1.0,
        decision_cost_bps=20.0,
        decision_hit_threshold_bps=10.0,
        decision_drawdown_penalty=0.10,
        cumulative_horizons=horizons,
        max_horizon=30,
        loss_profile="horizon_30d_soft_penalty_v1",
    )

    assert list(columns["pred_best_horizon"]) == [20, 20]
    assert columns["pred_best_horizon_source"] == "calibrated_utility_argmax"
    assert columns["decision_score_calibration_method"] == "max_horizon_utility_soft_penalty"
    assert columns["decision_score_calibration_penalty"] == pytest.approx(0.005)
    assert list(columns["pred_decision_score"]) == pytest.approx([0.100, 0.100])
    assert list(columns["pred_decision_utility_30d"]) == pytest.approx([0.103, 0.103])
    assert list(columns["calibrated_pred_decision_utility_30d"]) == pytest.approx([0.098, 0.098])


def test_time_efficient_topk_profile_writes_prediction_columns_and_scores_metrics() -> None:
    from daily_research.path_policy.forecast_training import _add_decision_utility_columns

    horizons = (1, 3, 5, 10, 20)
    utility = torch.tensor(
        [
            [0.010, 0.006, 0.005, 0.004, 0.003],
            [0.001, 0.002, 0.003, 0.004, 0.005],
            [-0.003, -0.002, -0.001, 0.000, 0.001],
            [0.004, 0.003, 0.002, 0.001, 0.000],
            [-0.004, -0.003, -0.002, -0.001, 0.000],
        ],
        dtype=torch.float32,
    )
    decision_aux = torch.cat([utility, torch.zeros_like(utility), torch.zeros_like(utility)], dim=1).numpy()
    y_cum = torch.tensor(
        [
            [0.010, 0.018, 0.025, 0.040, 0.060],
            [0.001, 0.006, 0.015, 0.040, 0.100],
            [-0.003, -0.006, -0.005, 0.000, 0.020],
            [0.004, 0.009, 0.010, 0.010, 0.000],
            [-0.004, -0.009, -0.010, -0.010, 0.000],
        ],
        dtype=torch.float32,
    ).numpy()
    risk = torch.zeros((5, len(horizons)), dtype=torch.float32).numpy()
    columns: dict[str, object] = {}
    _add_decision_utility_columns(
        columns,
        predictions={"decision_aux": decision_aux},
        y_cum=y_cum,
        drawdown_by_horizon=risk,
        worst_by_horizon=risk,
        target_scale=1.0,
        decision_cost_bps=0.0,
        decision_hit_threshold_bps=0.0,
        decision_drawdown_penalty=0.0,
        cumulative_horizons=horizons,
        max_horizon=20,
        loss_profile="personal_time_efficient_topk_v1",
    )

    assert {
        "pred_time_eff_utility_1d",
        "future_time_eff_utility_20d",
        "pred_time_eff_score",
        "future_time_eff_score",
        "pred_time_eff_best_horizon",
        "future_time_eff_best_horizon",
        "pred_decision_score",
        "future_decision_score",
    }.issubset(columns)
    assert columns["decision_score_calibration_method"] == "time_efficient_per_day"
    assert list(columns["pred_time_eff_best_horizon"]) == [1, 20, 20, 1, 20]
    assert list(columns["future_time_eff_best_horizon"]) == [1, 20, 20, 1, 20]

    frame = pd.DataFrame(
        {
            "date": ["2023-01-03"] * 5,
            "stock": ["AAA", "BBB", "CCC", "DDD", "EEE"],
            **columns,
        }
    )
    for horizon in horizons:
        frame[f"future_cum_excess_return_{horizon}d"] = y_cum[:, horizons.index(horizon)]
        frame[f"pred_cum_mu_{horizon}d"] = utility[:, horizons.index(horizon)].numpy()
    frame["future_path_upside_capture_20d"] = [0.02, 0.01, 0.00, -0.01, -0.02]
    frame["pred_aux_upside_20d"] = [0.02, 0.01, 0.00, -0.01, -0.02]
    for step in range(1, 21):
        frame[f"target_excess_{step}d"] = [0.001, 0.001, 0.000, -0.001, -0.001]
        frame[f"pred_q10_{step}d"] = [-0.01] * 5
        frame[f"pred_q90_{step}d"] = [0.01] * 5

    metrics = forecast_prediction_metrics(frame)

    assert metrics["time_eff_score_rank_ic"] > 0.0
    assert metrics["time_eff_score_top_bottom_spread"] > 0.0
    assert metrics["time_eff_best_horizon_accuracy"] == pytest.approx(1.0)
    assert metrics["time_eff_profile_status"] == "completed"


def test_train_forecast_models_records_auxiliary_loss_contract_in_summary_and_checkpoint(tmp_path) -> None:
    dataset = make_tiny_forecast_sequence_dataset(lookback_days=5, horizon=20)

    summary = train_forecast_models(
        dataset,
        study_root=tmp_path / "study",
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
        output_profile="decision_utility_v1",
        loss_profile="decision_utility_path_aux_v1",
        selection_profile="decision_utility",
        decision_cost_bps=20.0,
        decision_hit_threshold_bps=10.0,
        decision_drawdown_penalty=0.10,
    )

    contract = summary["training_config"]["loss_profile_contract"]
    assert contract["loss_profile"] == "decision_utility_path_aux_v1"
    assert contract["required_output_profile"] == "decision_utility_v1"
    assert contract["loss_component_weights"]["decision_utility"] > contract["loss_component_weights"]["path_aux"]
    assert summary["loss_component_weights"] == contract["loss_component_weights"]

    seed_summary = summary["models"]["linear_last_day"]["seed_summaries"]["7"]
    checkpoint = torch.load(seed_summary["last_checkpoint_pt"], map_location="cpu", weights_only=False)
    assert checkpoint["resume_contract"]["loss_profile_contract"] == contract
    assert checkpoint["training_config"]["loss_profile_contract"] == contract


def test_train_forecast_models_selects_time_efficient_topk_by_validation_loss(tmp_path) -> None:
    dataset = make_tiny_forecast_sequence_dataset(lookback_days=5, horizon=20)

    summary = train_forecast_models(
        dataset,
        study_root=tmp_path / "study",
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
        output_profile="decision_utility_v1",
        loss_profile="personal_time_efficient_topk_v1",
        selection_profile="validation_loss",
        decision_cost_bps=20.0,
        decision_hit_threshold_bps=10.0,
        decision_drawdown_penalty=0.10,
    )

    assert summary["status"] == "completed"
    assert summary["training_config"]["loss_profile"] == "personal_time_efficient_topk_v1"
    assert summary["training_config"]["loss_profile_contract"]["primary_objective"] == "personal_time_efficient_topk"
    assert summary["selection_rule"] == "lowest_validation_loss_for_training_loss_profile_then_seed_score"
    assert summary["validation_selection_score"] == pytest.approx(-summary["models"]["linear_last_day"]["seed_summaries"]["7"]["best_validation_loss"])
    assert summary["test_interpretable"] is True
    assert "time_eff_score_rank_ic" in summary["validation_metrics"]

    validation_predictions = pd.read_csv(tmp_path / "study" / "forecast_predictions_validation.csv")
    assert {
        "pred_time_eff_utility_1d",
        "future_time_eff_utility_20d",
        "pred_time_eff_score",
        "future_time_eff_score",
        "pred_time_eff_best_horizon",
        "future_time_eff_best_horizon",
    }.issubset(validation_predictions.columns)

    checkpoint = torch.load(
        tmp_path / "study" / "forecast_model_linear_last_day_seed7_best.pt",
        map_location="cpu",
        weights_only=False,
    )
    assert checkpoint["resume_contract"]["selection_profile"] == "validation_loss"
    assert checkpoint["resume_contract"]["loss_profile_contract"]["primary_objective"] == "personal_time_efficient_topk"


def test_train_forecast_models_can_override_static_context_fields_without_rebuilding_pack(tmp_path) -> None:
    dataset = make_tiny_forecast_sequence_dataset(lookback_days=5, horizon=20, include_static_context=True)
    assert dataset.manifest["static_context_schema"]["fields"][:3] == ["symbol", "exchange", "industry"]

    summary = train_forecast_models(
        dataset,
        study_root=tmp_path / "study",
        model_families=("hybrid_expert_fusion_static_context",),
        epochs=1,
        min_epochs=1,
        early_stop_patience=5,
        batch_size=4,
        lr=1.0e-3,
        hidden_dim=12,
        dropout=0.0,
        gru_layers=1,
        transformer_layers=1,
        transformer_heads=3,
        patch_sizes=(2,),
        seeds=(7,),
        device="cpu",
        amp=False,
        output_profile="decision_utility_v1",
        loss_profile="personal_time_efficient_topk_v1",
        selection_profile="validation_loss",
        static_context_fields_override=("exchange", "industry"),
        per_epoch_prediction_metrics=False,
    )

    assert summary["status"] == "completed"
    assert summary["training_config"]["static_context_source_fields"] == [
        "symbol",
        "exchange",
        "industry",
        "liquidity_bucket",
        "price_bucket",
    ]
    assert summary["training_config"]["static_context_training_fields"] == ["exchange", "industry"]
    assert summary["training_config"]["static_context_training_field_indices"] == [1, 2]
    assert summary["training_config"]["static_context_schema"]["fields"] == ["exchange", "industry"]
    assert summary["training_config"]["static_context_schema"]["source_fields"][0] == "symbol"

    seed_summary = summary["models"]["hybrid_expert_fusion_static_context"]["seed_summaries"]["7"]
    checkpoint = torch.load(seed_summary["last_checkpoint_pt"], map_location="cpu", weights_only=False)
    assert checkpoint["model_config"]["static_context_fields"] == ["exchange", "industry"]
    assert "symbol" not in checkpoint["model_config"]["static_context_vocab_sizes"]
    assert checkpoint["resume_contract"]["static_context_schema"]["fields"] == ["exchange", "industry"]


def test_train_forecast_models_supports_no_symbol_hybrid_alpha_score_loss(tmp_path) -> None:
    dataset = make_tiny_forecast_sequence_dataset(lookback_days=5, horizon=20, include_static_context=True)

    summary = train_forecast_models(
        dataset,
        study_root=tmp_path / "study",
        model_families=("hybrid_expert_fusion_static_context",),
        epochs=1,
        min_epochs=1,
        early_stop_patience=5,
        batch_size=4,
        lr=1.0e-3,
        hidden_dim=12,
        dropout=0.0,
        gru_layers=1,
        transformer_layers=1,
        transformer_heads=3,
        patch_sizes=(2,),
        seeds=(7,),
        device="cpu",
        amp=False,
        output_profile="forecast_path_v1",
        loss_profile="hybrid_alpha_score_v1",
        selection_profile="validation_loss",
        static_context_fields_override=("exchange", "industry"),
        per_epoch_prediction_metrics=False,
    )

    assert summary["status"] == "completed"
    contract = summary["training_config"]["loss_profile_contract"]
    assert contract["loss_profile"] == "hybrid_alpha_score_v1"
    assert contract["required_output_profile"] == "forecast_path_v1"
    assert contract["primary_objective"] == "hybrid_alpha_score"
    assert contract["alpha_score_objective"]["execution_cost_in_loss"] is False
    assert contract["alpha_score_objective"]["batch_topk_alignment_in_loss"] is False
    assert contract["loss_component_weights"]["alpha_efficiency_rank_aux"] > 0.0
    assert contract["loss_component_weights"]["decision_utility"] == 0.0
    assert summary["training_config"]["static_context_training_fields"] == ["exchange", "industry"]

    validation_predictions = pd.read_csv(tmp_path / "study" / "forecast_predictions_validation.csv")
    assert "pred_cum_mu_20d" in validation_predictions.columns
    assert "pred_decision_score" not in validation_predictions.columns

    seed_summary = summary["models"]["hybrid_expert_fusion_static_context"]["seed_summaries"]["7"]
    checkpoint = torch.load(seed_summary["last_checkpoint_pt"], map_location="cpu", weights_only=False)
    assert checkpoint["model_config"]["output_profile"] == "forecast_path_v1"
    assert checkpoint["resume_contract"]["loss_profile_contract"]["primary_objective"] == "hybrid_alpha_score"
    assert checkpoint["model_config"]["static_context_fields"] == ["exchange", "industry"]


def test_hybrid_alpha_score_v2_contract_is_prediction_first_and_unit_time_aware() -> None:
    contract = forecast_loss_profile_contract("hybrid_alpha_score_v2", cumulative_horizons=(1, 3, 5, 10, 20))

    assert contract["loss_profile"] == "hybrid_alpha_score_v2"
    assert contract["required_output_profile"] == "forecast_path_v1"
    assert contract["primary_objective"] == "hybrid_alpha_score"
    assert contract["alpha_score_objective"]["direct_train_target"] == "market_fact_alpha_score_not_execution_policy"
    assert contract["alpha_score_objective"]["execution_cost_in_loss"] is False
    assert contract["alpha_score_objective"]["execution_risk_penalty_in_loss"] is False
    assert contract["alpha_score_objective"]["batch_topk_alignment_in_loss"] is False
    assert contract["alpha_score_objective"]["unit_time_efficiency_in_loss"] is True
    assert contract["alpha_score_objective"]["rank_horizon_bias"] == "short_mid_horizon_unit_time_tilt"
    assert contract["loss_component_weights"]["decision_utility"] == 0.0
    assert contract["loss_component_weights"]["time_eff_topk_alignment"] == 0.0
    rank_weights = contract["alpha_score_objective"]["rank_loss_horizon_weights"]
    assert rank_weights["1"] == pytest.approx(rank_weights["5"])
    assert rank_weights["20"] < rank_weights["1"]


def test_train_forecast_models_supports_structured_alpha_v2_with_hybrid_alpha_score_v2(tmp_path) -> None:
    dataset = make_tiny_forecast_sequence_dataset(
        lookback_days=6,
        horizon=20,
        feature_count=10,
        include_static_context=True,
    )
    feature_columns = [
        "raw_close_from_prev_close_1d",
        "cs_rank_ret_20d",
        "market_breadth_20",
        "industry_momentum_20d",
        "valuation_peTTM_cs_z",
        "history_valid_ratio_252d",
        "intraday_first_5m_ret",
        "cs_z_intraday_close_to_vwap",
        "turn",
        "amount",
    ]
    manifest = dict(dataset.manifest)
    manifest["feature_columns"] = list(feature_columns)
    manifest["normalization"] = dict(manifest["normalization"])
    manifest["normalization"]["feature_columns"] = list(feature_columns)
    dataset = replace(dataset, feature_columns=feature_columns, manifest=manifest, normalization_manifest=dict(manifest["normalization"]))

    summary = train_forecast_models(
        dataset,
        study_root=tmp_path / "study",
        model_families=("hybrid_structured_alpha_v2",),
        epochs=1,
        min_epochs=1,
        early_stop_patience=5,
        batch_size=4,
        lr=1.0e-3,
        hidden_dim=12,
        dropout=0.0,
        gru_layers=1,
        transformer_layers=1,
        transformer_heads=3,
        patch_sizes=(2,),
        seeds=(7,),
        device="cpu",
        amp=False,
        output_profile="forecast_path_v1",
        loss_profile="hybrid_alpha_score_v2",
        selection_profile="validation_loss",
        static_context_fields_override=("exchange", "industry"),
        per_epoch_prediction_metrics=False,
    )

    assert summary["status"] == "completed"
    contract = summary["training_config"]["loss_profile_contract"]
    assert contract["loss_profile"] == "hybrid_alpha_score_v2"
    assert contract["alpha_score_objective"]["rank_horizon_bias"] == "short_mid_horizon_unit_time_tilt"
    assert contract["loss_component_weights"]["decision_utility"] == 0.0
    assert summary["training_config"]["static_context_training_fields"] == ["exchange", "industry"]

    validation_predictions = pd.read_csv(tmp_path / "study" / "forecast_predictions_validation.csv")
    assert "pred_cum_mu_5d" in validation_predictions.columns
    assert "pred_decision_score" not in validation_predictions.columns

    seed_summary = summary["models"]["hybrid_structured_alpha_v2"]["seed_summaries"]["7"]
    checkpoint = torch.load(seed_summary["last_checkpoint_pt"], map_location="cpu", weights_only=False)
    assert checkpoint["model_config"]["fusion_version"] == "hybrid_structured_alpha_v2"
    assert checkpoint["model_config"]["output_profile"] == "forecast_path_v1"
    assert checkpoint["resume_contract"]["loss_profile_contract"]["loss_profile"] == "hybrid_alpha_score_v2"
    assert checkpoint["resume_contract"]["loss_profile_contract"]["alpha_score_objective"]["execution_cost_in_loss"] is False


def test_train_forecast_models_records_multiscale_recency_hybrid_contract(tmp_path) -> None:
    dataset = make_tiny_forecast_sequence_dataset(
        lookback_days=6,
        horizon=20,
        feature_count=8,
        include_static_context=True,
    )
    feature_columns = [
        "raw_close_from_prev_close_1d",
        "intraday_first_5m_ret",
        "cs_rank_amount_20d",
        "intraday_last_30m_ret",
        "market_breadth_20",
        "valuation_peTTM_cs_z",
        "cs_rank_intraday_close_to_vwap",
        "industry_momentum_20d",
    ]
    manifest = dict(dataset.manifest)
    manifest["feature_columns"] = list(feature_columns)
    manifest["normalization"] = dict(manifest["normalization"])
    manifest["normalization"]["feature_columns"] = list(feature_columns)
    dataset = replace(dataset, feature_columns=feature_columns, manifest=manifest, normalization_manifest=dict(manifest["normalization"]))

    summary = train_forecast_models(
        dataset,
        study_root=tmp_path / "study",
        model_families=("hybrid_multiscale_recency_aware_v1",),
        epochs=1,
        min_epochs=1,
        early_stop_patience=5,
        batch_size=4,
        lr=1.0e-3,
        hidden_dim=12,
        dropout=0.0,
        gru_layers=1,
        transformer_layers=1,
        transformer_heads=3,
        patch_sizes=(2,),
        seeds=(7,),
        device="cpu",
        amp=False,
        output_profile="forecast_path_v1",
        loss_profile="hybrid_alpha_score_v1",
        selection_profile="validation_loss",
        static_context_fields_override=("exchange", "industry"),
        per_epoch_prediction_metrics=False,
    )

    assert summary["status"] == "completed"
    model_summary = summary["models"]["hybrid_multiscale_recency_aware_v1"]
    seed_summary = model_summary["seed_summaries"]["7"]
    checkpoint = torch.load(seed_summary["last_checkpoint_pt"], map_location="cpu", weights_only=False)
    model_config = checkpoint["model_config"]
    assert model_config["fusion_version"] == "hybrid_multiscale_recency_aware_v1"
    assert model_config["intraday_feature_count"] == 3
    assert model_config["intraday_feature_indices"] == [1, 3, 6]
    assert model_config["main_feature_count"] == 5
    assert model_config["static_context_fields"] == ["exchange", "industry"]
    assert checkpoint["resume_contract"]["model_config"]["fusion_version"] == "hybrid_multiscale_recency_aware_v1"


def test_train_forecast_models_records_structured_alpha_v2_contract(tmp_path) -> None:
    dataset = make_tiny_forecast_sequence_dataset(
        lookback_days=6,
        horizon=20,
        feature_count=10,
        include_static_context=True,
    )
    feature_columns = [
        "raw_close_from_prev_close_1d",
        "cs_rank_ret_20d",
        "market_breadth_20",
        "industry_momentum_20d",
        "valuation_peTTM_cs_z",
        "history_valid_ratio_252d",
        "intraday_first_5m_ret",
        "cs_z_intraday_close_to_vwap",
        "turn",
        "amount",
    ]
    manifest = dict(dataset.manifest)
    manifest["feature_columns"] = list(feature_columns)
    manifest["normalization"] = dict(manifest["normalization"])
    manifest["normalization"]["feature_columns"] = list(feature_columns)
    dataset = replace(dataset, feature_columns=feature_columns, manifest=manifest, normalization_manifest=dict(manifest["normalization"]))

    summary = train_forecast_models(
        dataset,
        study_root=tmp_path / "study",
        model_families=("hybrid_structured_alpha_v2",),
        epochs=1,
        min_epochs=1,
        early_stop_patience=5,
        batch_size=4,
        lr=1.0e-3,
        hidden_dim=12,
        dropout=0.0,
        gru_layers=1,
        transformer_layers=1,
        transformer_heads=3,
        patch_sizes=(2,),
        seeds=(7,),
        device="cpu",
        amp=False,
        output_profile="forecast_path_v1",
        loss_profile="hybrid_alpha_score_v1",
        selection_profile="validation_loss",
        static_context_fields_override=("exchange", "industry"),
        per_epoch_prediction_metrics=False,
    )

    assert summary["status"] == "completed"
    model_summary = summary["models"]["hybrid_structured_alpha_v2"]
    seed_summary = model_summary["seed_summaries"]["7"]
    checkpoint = torch.load(seed_summary["last_checkpoint_pt"], map_location="cpu", weights_only=False)
    model_config = checkpoint["model_config"]
    assert model_config["fusion_version"] == "hybrid_structured_alpha_v2"
    assert model_config["architecture_contract"]["prediction_first"] is True
    assert model_config["architecture_contract"]["execution_utility_in_model"] is False
    assert model_config["architecture_contract"]["intraday_role"] == "residual_short_horizon_correction"
    assert model_config["static_context_fields"] == ["exchange", "industry"]
    assert "symbol" not in model_config["static_context_fields"]
    assert model_config["feature_group_counts"]["intraday"] == 2
    assert model_config["feature_group_counts"]["valuation_liquidity"] == 3
    assert model_config["feature_group_counts"]["cross_section"] == 1
    assert model_config["feature_group_indices"]["intraday"] == [6, 7]
    assert checkpoint["resume_contract"]["model_config"]["fusion_version"] == "hybrid_structured_alpha_v2"

    from daily_research.path_policy.forecast_checkpoint_topk_reselection import _model_from_checkpoint
    from daily_research.path_policy.forecast_training import _ForecastDatasetView

    reloaded = _model_from_checkpoint(
        checkpoint,
        _ForecastDatasetView(dataset, static_context_fields_override=("exchange", "industry")),
    )
    assert getattr(reloaded, "feature_group_counts")["intraday"] == 2
    assert getattr(reloaded, "static_context_fields") == ("exchange", "industry")


def test_train_forecast_models_uses_structured_alpha_v2_pack_feature_groups(tmp_path) -> None:
    dataset = make_tiny_forecast_sequence_dataset(
        lookback_days=6,
        horizon=20,
        feature_count=7,
        include_static_context=True,
    )
    feature_columns = [
        "raw_close",
        "cs_rank_ret_20d",
        "market_breadth_20",
        "industry_momentum_20d",
        "turn_20d",
        "event_quality_score",
        "intraday_first_5m_ret",
    ]
    manifest = dict(dataset.manifest)
    manifest["feature_columns"] = list(feature_columns)
    manifest["structured_alpha_v2_pack"] = {
        "enabled": True,
        "feature_group_indices": {
            "daily_price_volume": [0],
            "cross_section": [1],
            "market_regime": [2],
            "industry_peer": [3],
            "valuation_liquidity": [4],
            "event_quality": [5],
            "intraday": [6],
        },
    }
    manifest["normalization"] = dict(manifest["normalization"])
    manifest["normalization"]["feature_columns"] = list(feature_columns)
    dataset = replace(dataset, feature_columns=feature_columns, manifest=manifest, normalization_manifest=dict(manifest["normalization"]))

    summary = train_forecast_models(
        dataset,
        study_root=tmp_path / "study",
        model_families=("hybrid_structured_alpha_v2",),
        epochs=1,
        min_epochs=1,
        early_stop_patience=5,
        batch_size=4,
        lr=1.0e-3,
        hidden_dim=12,
        dropout=0.0,
        gru_layers=1,
        transformer_layers=1,
        transformer_heads=3,
        patch_sizes=(2,),
        seeds=(7,),
        device="cpu",
        amp=False,
        output_profile="forecast_path_v1",
        loss_profile="hybrid_alpha_score_v2",
        selection_profile="validation_loss",
        static_context_fields_override=("exchange", "industry"),
        per_epoch_prediction_metrics=False,
    )

    seed_summary = summary["models"]["hybrid_structured_alpha_v2"]["seed_summaries"]["7"]
    checkpoint = torch.load(seed_summary["last_checkpoint_pt"], map_location="cpu", weights_only=False)
    model_config = checkpoint["model_config"]

    assert summary["status"] == "completed"
    assert summary["training_config"]["structured_alpha_v2_feature_group_source"] == "manifest_structured_alpha_v2_pack"
    assert model_config["feature_group_source"] == "manifest_structured_alpha_v2_pack"
    assert model_config["feature_group_indices"]["daily_price_volume"] == [0]
    assert model_config["feature_group_indices"]["cross_section"] == [1]
    assert model_config["feature_group_indices"]["market_regime"] == [2]
    assert model_config["feature_group_indices"]["industry_peer"] == [3]
    assert model_config["feature_group_indices"]["valuation_liquidity"] == [4]
    assert model_config["feature_group_indices"]["event_quality"] == [5]
    assert model_config["feature_group_indices"]["intraday"] == [6]


def test_train_forecast_models_applies_train_date_stride_without_thinning_validation_or_test(tmp_path) -> None:
    dataset = make_tiny_forecast_sequence_dataset(
        lookback_days=5,
        horizon=20,
        train_days=6,
        validation_days=2,
        test_days=2,
        include_static_context=True,
    )

    summary = train_forecast_models(
        dataset,
        study_root=tmp_path / "study",
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
        output_profile="forecast_path_v1",
        loss_profile="hybrid_alpha_score_v1",
        selection_profile="validation_loss",
        train_date_stride=3,
        per_epoch_prediction_metrics=False,
    )

    assert summary["status"] == "completed"
    assert summary["models"]["linear_last_day"]["train_rows"] == 8
    assert summary["models"]["linear_last_day"]["source_train_rows"] == 24
    assert summary["models"]["linear_last_day"]["validation_rows"] == 8
    assert summary["models"]["linear_last_day"]["test_rows"] == 8

    sampling = summary["training_config"]["sampling_config"]
    assert sampling["enabled"] is True
    assert sampling["mode"] == "train_date_stride"
    assert sampling["train_date_stride"] == 3
    assert sampling["source_train_dates"] == 6
    assert sampling["kept_train_dates"] == 2
    assert sampling["source_train_rows"] == 24
    assert sampling["kept_train_rows"] == 8
    assert sampling["validation_test_full"] is True

    learning_curve = pd.read_csv(tmp_path / "study" / "forecast_learning_curve.csv")
    assert learning_curve.loc[0, "train_sample_count"] == 8

    seed_summary = summary["models"]["linear_last_day"]["seed_summaries"]["7"]
    checkpoint = torch.load(seed_summary["last_checkpoint_pt"], map_location="cpu", weights_only=False)
    assert checkpoint["training_config"]["sampling_config"] == sampling
    assert checkpoint["resume_contract"]["sampling_config"] == sampling


def test_train_forecast_models_can_defer_per_epoch_prediction_metrics(tmp_path) -> None:
    dataset = make_tiny_forecast_sequence_dataset(lookback_days=5, horizon=20)

    summary = train_forecast_models(
        dataset,
        study_root=tmp_path / "study",
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
        output_profile="decision_utility_v1",
        loss_profile="personal_time_efficient_topk_v1",
        selection_profile="validation_loss",
        per_epoch_prediction_metrics=False,
    )

    assert summary["status"] == "completed"
    assert summary["training_config"]["per_epoch_prediction_metrics"] is False
    assert summary["selection_rule"] == "lowest_validation_loss_for_training_loss_profile_then_seed_score"
    assert summary["models"]["linear_last_day"]["per_epoch_prediction_metrics"] is False

    learning_curve = pd.read_csv(tmp_path / "study" / "forecast_learning_curve.csv")
    assert learning_curve.loc[0, "per_epoch_prediction_metrics"] == "validation_loss_only"
    assert learning_curve.loc[0, "validation_selection_score"] == pytest.approx(-learning_curve.loc[0, "validation_loss"])

    assert summary["validation_metrics"]["status"] == "completed"
    assert "time_eff_score_rank_ic" in summary["validation_metrics"]
    validation_predictions = pd.read_csv(tmp_path / "study" / "forecast_predictions_validation.csv")
    assert "pred_time_eff_score" in validation_predictions.columns


def test_decision_utility_targets_and_loss_are_finite() -> None:
    from daily_research.path_policy.forecast_training import _decision_utility_targets, _forecast_loss

    y_cum = torch.tensor([[0.010, 0.020, 0.015, 0.030, 0.025]], dtype=torch.float32)
    y_risk = torch.tensor(
        [[[-0.040, -0.020, 0.060]] * 5],
        dtype=torch.float32,
    )

    targets = _decision_utility_targets(
        y_cum,
        y_risk,
        target_scale=1.0,
        cost_bps=20.0,
        hit_threshold_bps=20.0,
        drawdown_penalty=0.25,
    )

    expected = y_cum - 0.002 - 0.25 * 0.04 * torch.sqrt(torch.tensor([[1, 3, 5, 10, 20]], dtype=torch.float32) / 20.0)
    assert torch.allclose(targets["utility"], expected, atol=1.0e-7)
    assert torch.equal(targets["hit_label"], expected > 0.002)
    assert targets["best_horizon_index"].item() == int(torch.argmax(expected, dim=1).item())

    prediction = {
        "mu": torch.zeros(4, 20),
        "q10": torch.full((4, 20), -0.01),
        "q50": torch.zeros(4, 20),
        "q90": torch.full((4, 20), 0.01),
        "aux": torch.zeros(4, 20),
        "decision_aux": torch.randn(4, 15) * 0.01,
    }
    loss = _forecast_loss(
        prediction,
        torch.randn(4, 20) * 0.01,
        torch.randn(4, 5) * 0.01,
        torch.randn(4, 5, 3) * 0.01,
        loss_profile="decision_utility_v1",
        target_scale=1.0,
        decision_cost_bps=20.0,
        decision_hit_threshold_bps=20.0,
        decision_drawdown_penalty=0.25,
    )
    assert torch.isfinite(loss)


def test_time_efficient_utility_targets_normalize_by_capital_time_and_cost() -> None:
    from daily_research.path_policy.forecast_training import _time_efficient_utility_targets

    horizons = (1, 3, 5, 10, 20)
    y_cum = torch.tensor([[0.010, 0.030, 0.050, 0.100, 0.200]], dtype=torch.float32)
    y_risk = torch.zeros((1, len(horizons), 3), dtype=torch.float32)

    no_cost = _time_efficient_utility_targets(
        y_cum,
        y_risk,
        target_scale=1.0,
        cost_bps=0.0,
        hit_threshold_bps=0.0,
        drawdown_penalty=0.0,
        worst_day_penalty=0.0,
        cumulative_horizons=horizons,
        max_horizon=20,
    )

    assert torch.allclose(no_cost["efficient_utility"], torch.full((1, len(horizons)), 0.010), atol=1.0e-7)
    assert no_cost["best_horizon_index"].item() == 0

    with_cost = _time_efficient_utility_targets(
        y_cum,
        y_risk,
        target_scale=1.0,
        cost_bps=20.0,
        hit_threshold_bps=0.0,
        drawdown_penalty=0.0,
        worst_day_penalty=0.0,
        cumulative_horizons=horizons,
        max_horizon=20,
    )

    assert with_cost["efficient_utility"][0, 0] == pytest.approx(0.008)
    assert with_cost["efficient_utility"][0, -1] == pytest.approx(0.0099)
    assert with_cost["best_horizon_index"].item() == len(horizons) - 1


def test_train_forecast_models_records_decision_output_contract_predictions_and_resume_mismatch(tmp_path) -> None:
    dataset = make_tiny_forecast_sequence_dataset(lookback_days=5, horizon=20)

    summary = train_forecast_models(
        dataset,
        study_root=tmp_path / "study",
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
        output_profile="decision_utility_v1",
        loss_profile="decision_utility_v1",
        selection_profile="decision_utility",
        decision_cost_bps=20.0,
        decision_hit_threshold_bps=20.0,
        decision_drawdown_penalty=0.25,
    )

    assert summary["status"] == "completed"
    assert summary["training_config"]["output_profile"] == "decision_utility_v1"
    assert summary["training_config"]["decision_utility"]["cost_bps"] == pytest.approx(20.0)
    assert "decision_score_rank_ic" in summary["validation_metrics"]
    assert "decision_utility_profile_status" in summary["validation_metrics"]

    validation_predictions = pd.read_csv(tmp_path / "study" / "forecast_predictions_validation.csv")
    assert {
        "pred_decision_utility_1d",
        "future_decision_utility_20d",
        "pred_hit_prob_5d",
        "future_hit_label_10d",
        "pred_best_horizon",
        "future_best_horizon",
        "pred_decision_score",
        "future_decision_score",
    }.issubset(validation_predictions.columns)

    seed_summary = summary["models"]["linear_last_day"]["seed_summaries"]["7"]
    checkpoint = torch.load(seed_summary["last_checkpoint_pt"], map_location="cpu", weights_only=False)
    assert checkpoint["resume_contract"]["output_profile"] == "decision_utility_v1"
    assert checkpoint["resume_contract"]["decision_cost_bps"] == pytest.approx(20.0)
    assert checkpoint["training_config"]["decision_utility"]["drawdown_penalty"] == pytest.approx(0.25)

    with pytest.raises(ValueError, match="decision_cost_bps"):
        train_forecast_models(
            dataset,
            study_root=tmp_path / "resume_mismatch",
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
            output_profile="decision_utility_v1",
            loss_profile="decision_utility_v1",
            selection_profile="decision_utility",
            decision_cost_bps=25.0,
            decision_hit_threshold_bps=20.0,
            decision_drawdown_penalty=0.25,
            resume_from=seed_summary["last_checkpoint_pt"],
        )


def test_train_forecast_models_accepts_custom_horizon_decision_utility_contract(tmp_path) -> None:
    horizons = (1, 2, 3, 5, 8, 10, 15, 20, 30)
    dataset = make_tiny_forecast_memmap_dataset(
        tmp_path / "dataset",
        lookback_days=5,
        horizon=30,
        cumulative_horizons=horizons,
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
        output_profile="decision_utility_v1",
        loss_profile="decision_utility_v1",
        selection_profile="decision_utility",
        decision_cost_bps=20.0,
        decision_hit_threshold_bps=10.0,
        decision_drawdown_penalty=0.10,
        dataloader_num_workers=0,
    )

    assert summary["status"] == "completed"
    assert summary["training_config"]["forecast_horizon"] == 30
    assert summary["training_config"]["cumulative_horizons"] == list(horizons)
    assert summary["training_config"]["decision_utility"]["horizons"] == list(horizons)

    validation_predictions = pd.read_csv(tmp_path / "study" / "forecast_predictions_validation.csv")
    assert {
        "pred_decision_utility_30d",
        "future_decision_utility_30d",
        "pred_hit_prob_8d",
        "future_hit_label_8d",
        "pred_best_horizon",
        "future_best_horizon",
        "trade_utility_score",
    }.issubset(validation_predictions.columns)

    seed_summary = summary["models"]["gru_sequence_static_context"]["seed_summaries"]["7"]
    checkpoint = torch.load(seed_summary["last_checkpoint_pt"], map_location="cpu", weights_only=False)
    assert checkpoint["resume_contract"]["horizon"] == 30
    assert checkpoint["resume_contract"]["forecast_horizon"] == 30
    assert checkpoint["resume_contract"]["cumulative_horizons"] == list(horizons)

    bad_dataset = dataset
    bad_dataset.manifest = {**dataset.manifest, "cumulative_horizons": [1, 3, 5, 10, 20, 30]}
    with pytest.raises(ValueError, match="cumulative_horizons"):
        train_forecast_models(
            bad_dataset,
            study_root=tmp_path / "resume_bad_horizons",
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
            output_profile="decision_utility_v1",
            loss_profile="decision_utility_v1",
            selection_profile="decision_utility",
            decision_cost_bps=20.0,
            decision_hit_threshold_bps=10.0,
            decision_drawdown_penalty=0.10,
            resume_from=seed_summary["last_checkpoint_pt"],
        )


def test_train_forecast_models_accepts_daily_1_to_45_grid_feasibility_contract(tmp_path) -> None:
    horizons = tuple(range(1, 46))
    dataset = make_tiny_forecast_sequence_dataset(lookback_days=5, horizon=45, cumulative_horizons=horizons)

    summary = train_forecast_models(
        dataset,
        study_root=tmp_path / "study",
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
        output_profile="decision_utility_v1",
        loss_profile="decision_utility_path_aux_v1",
        selection_profile="decision_utility",
        decision_cost_bps=20.0,
        decision_hit_threshold_bps=10.0,
        decision_drawdown_penalty=0.10,
    )

    assert summary["status"] == "completed"
    assert summary["training_config"]["forecast_horizon"] == 45
    assert summary["training_config"]["cumulative_horizons"] == list(horizons)
    assert summary["training_config"]["loss_profile_contract"]["cumulative_horizons"] == list(horizons)
    validation_predictions = pd.read_csv(tmp_path / "study" / "forecast_predictions_validation.csv")
    assert {
        "pred_cum_mu_45d",
        "future_cum_excess_return_45d",
        "pred_decision_utility_45d",
        "future_decision_utility_45d",
        "pred_hit_prob_45d",
        "future_hit_label_45d",
    }.issubset(validation_predictions.columns)


def test_ranking_baseline_dependency_missing_reports_status(tmp_path) -> None:
    dataset = make_tiny_forecast_memmap_dataset(tmp_path / "dataset", lookback_days=5, horizon=20)

    summary = run_forecast_ranking_baseline(dataset, study_root=tmp_path / "study", baseline="missing_ranker")

    assert summary["status"] == "dependency_missing"
    assert summary["baseline"] == "missing_ranker"
    assert summary["shadow_only"] is True
    assert summary["active_execution_strategy_expected_diff"] == "none"


def test_ranking_relevance_labels_are_integer_deciles_by_date() -> None:
    frame = pd.DataFrame(
        {
            "date": ["2024-01-02"] * 5 + ["2024-01-03"] * 3,
            "ranker_feature_0": range(8),
        }
    )
    y = pd.Series([0.10, 0.30, 0.20, 0.50, 0.40, -0.20, 0.00, 0.20], dtype=float)

    labels = ranking_relevance_labels(frame, y, relevance_levels=5)

    assert labels.dtype == "int32"
    assert labels.tolist() == [0, 2, 1, 4, 3, 0, 2, 4]
    assert labels.min() >= 0
    assert labels.max() <= 4


def test_sector_slot_diagnostics_handles_missing_metadata(tmp_path) -> None:
    dataset = make_tiny_forecast_memmap_dataset(tmp_path / "dataset", lookback_days=5, horizon=20)

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
    dataset = make_tiny_forecast_memmap_dataset(tmp_path / "dataset", lookback_days=5, horizon=20)

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
    dataset = make_tiny_forecast_sequence_dataset(lookback_days=5, horizon=20)

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
    dataset = make_tiny_forecast_memmap_dataset(tmp_path / "dataset", lookback_days=5, horizon=20)

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
    dataset = make_tiny_forecast_sequence_dataset(lookback_days=5, horizon=20)

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
    dataset = make_tiny_forecast_sequence_dataset(lookback_days=5, horizon=20)

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


def test_train_forecast_models_can_resume_for_evaluation_only_at_checkpoint_epoch(tmp_path) -> None:
    dataset = make_tiny_forecast_sequence_dataset(lookback_days=5, horizon=20)
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
    resume_path = first_summary["models"]["linear_last_day"]["seed_summaries"]["7"]["last_checkpoint_pt"]

    eval_summary = train_forecast_models(
        dataset,
        study_root=tmp_path / "eval_only",
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
        resume_from=resume_path,
    )

    seed_summary = eval_summary["models"]["linear_last_day"]["seed_summaries"]["7"]
    assert eval_summary["status"] == "completed"
    assert eval_summary["resume_from_checkpoint_pt"] == str(Path(resume_path).resolve())
    assert seed_summary["resume_start_epoch"] == 2
    assert seed_summary["epochs_ran"] == 1
    assert seed_summary["stopped_reason"] == "resume_evaluation_only"
    assert (tmp_path / "eval_only" / "forecast_predictions_test.csv").exists()


def test_train_forecast_models_rejects_resume_checkpoint_contract_mismatch(tmp_path) -> None:
    dataset = make_tiny_forecast_sequence_dataset(lookback_days=5, horizon=20)
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


def test_forecast_prediction_metrics_scores_decision_utility_columns() -> None:
    frame = pd.DataFrame(
        {
            "date": ["2023-01-03"] * 5,
            "stock": ["AAA", "BBB", "CCC", "DDD", "EEE"],
            "pred_decision_score": [0.05, 0.04, 0.01, -0.01, -0.02],
            "future_decision_score": [0.06, 0.03, 0.00, -0.01, -0.03],
            "pred_best_horizon": [5, 5, 3, 1, 1],
            "future_best_horizon": [5, 3, 3, 1, 10],
        }
    )
    for horizon in (1, 3, 5, 10, 20):
        frame[f"future_cum_excess_return_{horizon}d"] = [0.01, 0.00, -0.01, 0.02, -0.02]
        frame[f"pred_cum_mu_{horizon}d"] = [0.02, 0.01, -0.02, 0.01, -0.03]
        frame[f"future_hit_label_{horizon}d"] = [1, 1, 0, 0, 0]
    frame["future_path_upside_capture_20d"] = [0.02, 0.01, 0.00, -0.01, -0.02]
    frame["pred_aux_upside_20d"] = [0.02, 0.01, 0.00, -0.01, -0.02]
    for step in range(1, 21):
        frame[f"target_excess_{step}d"] = [0.001, 0.001, 0.000, -0.001, -0.001]
        frame[f"pred_q10_{step}d"] = [-0.01] * 5
        frame[f"pred_q90_{step}d"] = [0.01] * 5

    metrics = forecast_prediction_metrics(frame)

    assert metrics["decision_score_rank_ic"] > 0.0
    assert metrics["decision_score_top_bottom_spread"] > 0.0
    assert metrics["decision_hit_lift_top20_mean"] > 0.0
    assert metrics["decision_best_horizon_accuracy"] == pytest.approx(0.6)
    assert metrics["decision_utility_profile_status"] == "passed"
