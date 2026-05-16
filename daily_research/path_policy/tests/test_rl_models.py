from __future__ import annotations

import torch

from daily_research.path_policy.rl_models import (
    DecisionTransformerTargetWeightPolicy,
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
