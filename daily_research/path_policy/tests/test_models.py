from __future__ import annotations

import torch

from daily_research.path_policy.models import (
    DLinearPath20Forecaster,
    ExpertFusionPath20Forecaster,
    GRUPath20Forecaster,
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
