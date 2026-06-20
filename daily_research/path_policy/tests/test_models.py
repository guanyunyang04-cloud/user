from __future__ import annotations

import torch

from daily_research.path_policy.models import (
    DLinearPath20Forecaster,
    ExpertFusionPath20Forecaster,
    GRUPath20Forecaster,
    HybridStructuredAlphaV2Forecaster,
    HybridMultiScaleRecencyAwarePath20Forecaster,
    LinearPath20Forecaster,
    NeuralTargetWeightPolicy,
    PATH20_DECISION_AUX_DIM,
    PATH20_DEFAULT_CUMULATIVE_HORIZONS,
    path20_decision_aux_dim,
    path20_forecast_aux_dim,
    PatchTransformerPath20Forecaster,
    Path20ForecasterMLP,
    PathPolicyModelConfig,
    pairwise_rank_loss,
    pinball_loss,
    portfolio_utility_loss,
    RegimeRoutedMultiExpertHorizonForecaster,
)


def test_forecaster_variants_emit_path20_contract() -> None:
    x2 = torch.randn(6, 5)
    x3 = torch.randn(6, 4, 5)
    models = [
        LinearPath20Forecaster(input_dim=5),
        Path20ForecasterMLP(input_dim=5, hidden_dim=16),
        DLinearPath20Forecaster(input_dim=5, hidden_dim=8),
        GRUPath20Forecaster(input_dim=5, hidden_dim=8),
        PatchTransformerPath20Forecaster(input_dim=5, hidden_dim=8, num_heads=2, num_layers=1),
    ]
    for model in models:
        prediction = model(x3 if model.__class__.__name__ != "Path20ForecasterMLP" and model.__class__.__name__ != "LinearPath20Forecaster" else x2)
        assert set(prediction) == {"mu", "q10", "q50", "q90", "aux"}
        assert prediction["mu"].shape == (6, 20)
        assert prediction["q10"].shape == (6, 20)
        assert prediction["aux"].shape == (6, path20_forecast_aux_dim(PATH20_DEFAULT_CUMULATIVE_HORIZONS))
        assert torch.all(prediction["q10"] <= prediction["q50"])
        assert torch.all(prediction["q50"] <= prediction["q90"])


def test_forecaster_variants_emit_optional_decision_utility_head() -> None:
    x2 = torch.randn(6, 5)
    x3 = torch.randn(6, 4, 5)
    models = [
        LinearPath20Forecaster(input_dim=5, output_profile="decision_utility_v1"),
        Path20ForecasterMLP(input_dim=5, hidden_dim=16, output_profile="decision_utility_v1"),
        DLinearPath20Forecaster(input_dim=5, hidden_dim=8, output_profile="decision_utility_v1"),
        GRUPath20Forecaster(input_dim=5, hidden_dim=8, output_profile="decision_utility_v1"),
        PatchTransformerPath20Forecaster(
            input_dim=5,
            hidden_dim=8,
            num_heads=2,
            num_layers=1,
            output_profile="decision_utility_v1",
        ),
    ]
    for model in models:
        prediction = model(x3 if model.__class__.__name__ not in {"Path20ForecasterMLP", "LinearPath20Forecaster"} else x2)
        assert set(prediction) == {"mu", "q10", "q50", "q90", "aux", "decision_aux"}
        assert prediction["mu"].shape == (6, 20)
        assert prediction["aux"].shape == (6, path20_forecast_aux_dim(PATH20_DEFAULT_CUMULATIVE_HORIZONS))
        assert prediction["decision_aux"].shape == (6, PATH20_DECISION_AUX_DIM)


def test_forecaster_variants_emit_dynamic_horizon_contract() -> None:
    horizons = (1, 2, 3, 5, 8, 10, 15, 20, 30)
    x = torch.randn(6, 4, 5)
    model = GRUPath20Forecaster(
        input_dim=5,
        hidden_dim=8,
        horizon=30,
        output_profile="decision_utility_v1",
        cumulative_horizons=horizons,
    )

    prediction = model(x)

    assert prediction["mu"].shape == (6, 30)
    assert prediction["q10"].shape == (6, 30)
    assert prediction["aux"].shape == (6, path20_forecast_aux_dim(horizons, horizon=30))
    assert prediction["decision_aux"].shape == (6, path20_decision_aux_dim(horizons, horizon=30))
    assert prediction["decision_aux"].shape[1] == 27


def test_decision_forecaster_supports_daily_1_to_45_feasibility_grid() -> None:
    horizons = tuple(range(1, 46))
    model = GRUPath20Forecaster(
        input_dim=5,
        hidden_dim=8,
        horizon=45,
        output_profile="decision_utility_v1",
        cumulative_horizons=horizons,
    )

    prediction = model(torch.randn(4, 6, 5))

    assert prediction["mu"].shape == (4, 45)
    assert prediction["aux"].shape == (4, path20_forecast_aux_dim(horizons, horizon=45))
    assert prediction["decision_aux"].shape == (4, path20_decision_aux_dim(horizons, horizon=45))
    assert prediction["decision_aux"].shape[1] == 135


def test_dlinear_dynamic_horizon_contract() -> None:
    horizons = (1, 2, 3, 5, 8, 10, 15, 20, 30)
    model = DLinearPath20Forecaster(
        input_dim=5,
        hidden_dim=8,
        horizon=30,
        output_profile="decision_utility_v1",
        cumulative_horizons=horizons,
    )

    prediction = model(torch.randn(6, 4, 5))

    assert prediction["mu"].shape == (6, 30)
    assert prediction["aux"].shape == (6, 36)
    assert prediction["decision_aux"].shape == (6, 27)


def test_sequence_forecasters_use_temporal_pooling_and_position_information() -> None:
    x = torch.randn(3, 8, 5)
    gru = GRUPath20Forecaster(input_dim=5, hidden_dim=8, num_layers=2, dropout=0.1)
    gru_prediction = gru(x)
    assert hasattr(gru, "attention_pool")
    assert torch.isfinite(gru_prediction["mu"]).all()

    transformer = PatchTransformerPath20Forecaster(
        input_dim=5,
        hidden_dim=8,
        patch_sizes=(2,),
        num_heads=2,
        num_layers=1,
        dropout=0.0,
    )
    transformer.eval()
    ordered = torch.zeros(1, 4, 5)
    ordered[:, :2, :] = 1.0
    ordered[:, 2:, :] = -1.0
    swapped = ordered[:, [2, 3, 0, 1], :]
    with torch.no_grad():
        ordered_prediction = transformer(ordered)["mu"]
        swapped_prediction = transformer(swapped)["mu"]
    assert not torch.allclose(ordered_prediction, swapped_prediction)


def test_expert_fusion_forecaster_emits_path20_contract_and_router_weights() -> None:
    model = ExpertFusionPath20Forecaster(
        input_dim=5,
        hidden_dim=12,
        horizon=20,
        dropout=0.0,
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

    prediction = model(x, static_context_ids=static_ids)
    weights = model.expert_weights(x, static_context_ids=static_ids)

    assert set(prediction) == {"mu", "q10", "q50", "q90", "aux"}
    assert prediction["mu"].shape == (4, 20)
    assert weights.shape == (4, 3)
    assert torch.isfinite(weights).all()
    assert torch.allclose(weights.sum(dim=-1), torch.ones(4), atol=1.0e-6)


def test_multiscale_recency_aware_hybrid_uses_intraday_bottleneck_and_router_weights() -> None:
    model = HybridMultiScaleRecencyAwarePath20Forecaster(
        input_dim=8,
        hidden_dim=12,
        horizon=20,
        dropout=0.0,
        gru_layers=1,
        transformer_layers=1,
        transformer_heads=3,
        patch_sizes=(2,),
        intraday_feature_indices=(1, 3, 7),
        static_context_vocab_sizes={"exchange": 4, "industry": 6},
        static_context_embedding_dims={"exchange": 2, "industry": 3},
        static_context_fields=("exchange", "industry"),
    )
    x = torch.randn(4, 6, 8)
    static_ids = torch.tensor([[1, 1], [2, 2], [1, 3], [0, 4]], dtype=torch.long)

    prediction = model(x, static_context_ids=static_ids)
    weights = model.expert_weights(x, static_context_ids=static_ids)

    assert set(prediction) == {"mu", "q10", "q50", "q90", "aux", "router_weights", "router_entropy"}
    assert prediction["mu"].shape == (4, 20)
    assert prediction["aux"].shape == (4, path20_forecast_aux_dim(PATH20_DEFAULT_CUMULATIVE_HORIZONS))
    assert weights.shape == (4, 4)
    assert model.intraday_input_dim == 3
    assert model.main_input_dim == 5
    assert torch.isfinite(prediction["router_weights"]).all()
    assert torch.allclose(weights.sum(dim=-1), torch.ones(4), atol=1.0e-6)


def test_structured_alpha_v2_uses_group_tokens_context_conditioning_and_intraday_residual() -> None:
    feature_groups = {
        "daily_price_volume": (0, 1),
        "cross_section": (2,),
        "market_regime": (3,),
        "industry_peer": (4,),
        "valuation_liquidity": (5,),
        "event_quality": (6,),
        "intraday": (7, 8),
    }
    model = HybridStructuredAlphaV2Forecaster(
        input_dim=9,
        hidden_dim=12,
        horizon=20,
        dropout=0.0,
        gru_layers=1,
        transformer_layers=1,
        transformer_heads=3,
        patch_sizes=(2,),
        feature_group_indices=feature_groups,
        static_context_vocab_sizes={"exchange": 4, "industry": 6},
        static_context_embedding_dims={"exchange": 2, "industry": 3},
        static_context_fields=("exchange", "industry"),
    )
    x = torch.randn(4, 6, 9)
    static_ids = torch.tensor([[1, 1], [2, 2], [1, 3], [0, 4]], dtype=torch.long)

    prediction = model(x, static_context_ids=static_ids)
    weights = model.expert_weights(x, static_context_ids=static_ids)

    assert {"mu", "q10", "q50", "q90", "aux", "router_weights", "router_entropy"}.issubset(prediction)
    assert prediction["mu"].shape == (4, 20)
    assert prediction["aux"].shape == (4, path20_forecast_aux_dim(PATH20_DEFAULT_CUMULATIVE_HORIZONS))
    assert weights.shape == (4, 4)
    assert prediction["feature_group_weights"].shape == (4, 6)
    assert model.feature_group_counts["intraday"] == 2
    assert model.static_context_fields == ("exchange", "industry")
    assert torch.isfinite(prediction["intraday_residual_norm"]).all()
    assert torch.isfinite(prediction["context_gate_abs_mean"]).all()
    assert torch.allclose(weights.sum(dim=-1), torch.ones(4), atol=1.0e-6)
    assert torch.allclose(prediction["feature_group_weights"].sum(dim=-1), torch.ones(4), atol=1.0e-6)


def test_structured_alpha_v2_rejects_symbol_static_context() -> None:
    try:
        HybridStructuredAlphaV2Forecaster(
            input_dim=5,
            hidden_dim=12,
            horizon=20,
            dropout=0.0,
            gru_layers=1,
            transformer_layers=1,
            transformer_heads=3,
            patch_sizes=(2,),
            static_context_vocab_sizes={"symbol": 8, "exchange": 4, "industry": 6},
            static_context_embedding_dims={"symbol": 4, "exchange": 2, "industry": 3},
            static_context_fields=("symbol", "exchange", "industry"),
        )
    except ValueError as exc:
        assert "symbol" in str(exc)
    else:
        raise AssertionError("hybrid_structured_alpha_v2 must reject symbol static context")


def test_regime_routed_multi_expert_forecaster_emits_decision_contract_and_router_diagnostics() -> None:
    horizons = (1, 2, 3, 5, 8, 10, 15, 20, 30)
    model = RegimeRoutedMultiExpertHorizonForecaster(
        input_dim=5,
        hidden_dim=16,
        horizon=30,
        dropout=0.0,
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
    x = torch.randn(4, 6, 5)
    static_ids = torch.ones(4, 5, dtype=torch.long)

    prediction = model(x, static_context_ids=static_ids)
    weights = model.expert_weights(x, static_context_ids=static_ids)

    assert {"mu", "q10", "q50", "q90", "aux", "decision_aux"}.issubset(prediction)
    assert prediction["mu"].shape == (4, 30)
    assert prediction["aux"].shape == (4, 36)
    assert prediction["decision_aux"].shape == (4, 27)
    assert prediction["router_weights"].shape == (4, 5)
    assert weights.shape == (4, 5)
    assert torch.isfinite(prediction["router_weights"]).all()
    assert torch.isfinite(prediction["router_entropy"]).all()
    assert torch.isfinite(prediction["expert_token_diversity"]).all()
    assert torch.allclose(prediction["router_weights"].sum(dim=-1), torch.ones(4), atol=1.0e-6)
    assert torch.allclose(weights.sum(dim=-1), torch.ones(4), atol=1.0e-6)


def test_regime_routed_multi_expert_forecaster_supports_2d_static_fallback_and_date_batches() -> None:
    model = RegimeRoutedMultiExpertHorizonForecaster(
        input_dim=5,
        hidden_dim=16,
        horizon=20,
        dropout=0.0,
        gru_layers=1,
        transformer_layers=1,
        transformer_heads=4,
        patch_sizes=(2,),
    )

    two_dimensional = model(torch.randn(3, 5))
    assert two_dimensional["mu"].shape == (3, 20)
    assert two_dimensional["router_weights"].shape == (3, 5)

    date_batch = torch.randn(2, 4, 6, 5)
    static_ids = torch.ones(2, 4, 5, dtype=torch.long)
    stock_mask = torch.tensor([[True, True, False, True], [True, False, False, True]])
    prediction = model(date_batch, static_context_ids=static_ids, stock_mask=stock_mask)

    assert prediction["mu"].shape == (8, 20)
    assert prediction["router_weights"].shape == (8, 5)
    assert torch.isfinite(prediction["bad_state_intensity"]).all()


def test_losses_are_finite_and_allocator_respects_weight_constraints() -> None:
    pred = torch.tensor([0.0, 0.1, -0.1])
    target = torch.tensor([0.05, -0.02, 0.2])
    assert torch.isfinite(pinball_loss(pred, target, 0.5))
    assert torch.isfinite(pairwise_rank_loss(pred, target))

    policy = NeuralTargetWeightPolicy(
        PathPolicyModelConfig(
            stock_feature_dim=4,
            path_feature_dim=88,
            portfolio_feature_dim=8,
            hidden_dim=16,
            dropout=0.0,
            max_position_weight=0.15,
        )
    )
    out = policy(
        torch.randn(10, 4),
        torch.randn(10, 88),
        torch.zeros(8),
        tradable_mask=torch.ones(10, dtype=torch.bool),
    )
    assert float(out["target_weight"].min()) >= 0.0
    assert float(out["target_weight"].max()) <= 0.15 + 1.0e-6
    assert float(out["target_weight"].sum()) <= 1.0 + 1.0e-6
    loss = portfolio_utility_loss(out["target_weight"], torch.randn(10) * 0.01)
    assert torch.isfinite(loss)
