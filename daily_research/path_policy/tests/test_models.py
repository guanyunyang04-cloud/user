from __future__ import annotations

import torch

from daily_research.path_policy.models import (
    DLinearPath20Forecaster,
    GRUPath20Forecaster,
    LinearPath20Forecaster,
    NeuralTargetWeightPolicy,
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
        assert prediction["aux"].shape == (6, 8)
        assert torch.all(prediction["q10"] <= prediction["q50"])
        assert torch.all(prediction["q50"] <= prediction["q90"])


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
