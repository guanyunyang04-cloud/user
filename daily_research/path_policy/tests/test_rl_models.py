from __future__ import annotations

import torch

from daily_research.path_policy.rl_models import (
    DecisionTransformerTargetWeightPolicy,
    PortfolioDecisionTransformerPolicy,
    SequencePolicyConfig,
    SequenceTargetWeightPolicy,
    sequence_policy_utility_loss,
)


def test_sequence_policy_variants_emit_raw_target_contract() -> None:
    config = SequencePolicyConfig(
        stock_feature_dim=7,
        portfolio_feature_dim=8,
        hidden_dim=16,
        dropout=0.0,
        max_position_weight=0.12,
    )
    state = torch.randn(3, 5, 4, 7)
    portfolio = torch.randn(3, 5, 8)
    mask = torch.ones(3, 4, dtype=torch.bool)
    models = [
        SequenceTargetWeightPolicy(config),
        DecisionTransformerTargetWeightPolicy(config, num_layers=1, num_heads=2),
        PortfolioDecisionTransformerPolicy(config, temporal_layers=1, cross_layers=1, num_heads=2),
    ]

    for model in models:
        out = model(state, portfolio, tradable_mask=mask)
        assert set(out) == {"raw_target_weight", "cash_logit", "score_logits", "policy_aux"}
        assert out["raw_target_weight"].shape == (3, 4)
        assert out["score_logits"].shape == (3, 4)
        assert float(out["raw_target_weight"].min()) >= 0.0
        assert float(out["raw_target_weight"].max()) <= 0.12 + 1.0e-6
        assert torch.all(out["raw_target_weight"].sum(dim=1) <= 1.0 + 1.0e-6)


def test_sequence_policy_utility_loss_is_finite() -> None:
    target_weight = torch.tensor([[0.10, 0.20, 0.0], [0.0, 0.05, 0.15]])
    future_return = torch.tensor([[0.01, -0.02, 0.03], [0.02, 0.01, -0.01]])
    current_weight = torch.zeros_like(target_weight)

    loss = sequence_policy_utility_loss(target_weight, future_return, current_weight)

    assert torch.isfinite(loss)


def test_portfolio_decision_transformer_time_order_changes_output() -> None:
    torch.manual_seed(7)
    config = SequencePolicyConfig(
        stock_feature_dim=5,
        portfolio_feature_dim=8,
        hidden_dim=16,
        dropout=0.0,
        max_position_weight=0.15,
        max_sequence_length=12,
        max_stock_slots=8,
    )
    model = PortfolioDecisionTransformerPolicy(config, temporal_layers=1, cross_layers=1, num_heads=2)
    model.eval()
    state = torch.randn(2, 6, 3, 5)
    portfolio = torch.randn(2, 6, 8)
    previous_weight = torch.rand(2, 6, 3) * 0.05
    previous_reward = torch.randn(2, 6) * 0.001
    mask = torch.ones(2, 3, dtype=torch.bool)

    out_forward = model(
        state,
        portfolio,
        previous_weight_sequence=previous_weight,
        previous_reward_sequence=previous_reward,
        tradable_mask=mask,
    )["raw_target_weight"]
    out_reversed = model(
        torch.flip(state, dims=[1]),
        torch.flip(portfolio, dims=[1]),
        previous_weight_sequence=torch.flip(previous_weight, dims=[1]),
        previous_reward_sequence=torch.flip(previous_reward, dims=[1]),
        tradable_mask=mask,
    )["raw_target_weight"]

    assert not torch.allclose(out_forward, out_reversed)


def test_portfolio_decision_transformer_temporal_causal_mask() -> None:
    torch.manual_seed(11)
    config = SequencePolicyConfig(
        stock_feature_dim=4,
        portfolio_feature_dim=8,
        hidden_dim=16,
        dropout=0.0,
        max_position_weight=0.15,
        max_sequence_length=10,
        max_stock_slots=8,
    )
    model = PortfolioDecisionTransformerPolicy(config, temporal_layers=1, cross_layers=1, num_heads=2)
    model.eval()
    state = torch.randn(1, 5, 3, 4)
    portfolio = torch.randn(1, 5, 8)
    encoded = model.temporal_encode(state, portfolio)
    perturbed = state.clone()
    perturbed[:, 4, :, :] = perturbed[:, 4, :, :] + 100.0
    encoded_perturbed = model.temporal_encode(perturbed, portfolio)

    assert torch.allclose(encoded[:, 0, :, :], encoded_perturbed[:, 0, :, :], atol=1.0e-5)
