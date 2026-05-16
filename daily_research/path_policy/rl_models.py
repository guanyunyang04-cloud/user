from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn
import torch.nn.functional as F


@dataclass(frozen=True)
class SequencePolicyConfig:
    stock_feature_dim: int
    portfolio_feature_dim: int = 8
    hidden_dim: int = 96
    dropout: float = 0.10
    max_position_weight: float = 0.10


class SequenceTargetWeightPolicy(nn.Module):
    def __init__(self, config: SequencePolicyConfig) -> None:
        super().__init__()
        self.config = config
        input_dim = int(config.stock_feature_dim) + int(config.portfolio_feature_dim)
        self.encoder = nn.GRU(
            input_size=input_dim,
            hidden_size=int(config.hidden_dim),
            batch_first=True,
        )
        self.score_head = nn.Sequential(
            nn.LayerNorm(int(config.hidden_dim)),
            nn.Dropout(float(config.dropout)),
            nn.Linear(int(config.hidden_dim), 1),
        )
        self.cash_head = nn.Sequential(
            nn.LayerNorm(int(config.portfolio_feature_dim)),
            nn.Linear(int(config.portfolio_feature_dim), max(int(config.hidden_dim // 2), 8)),
            nn.GELU(),
            nn.Linear(max(int(config.hidden_dim // 2), 8), 1),
        )

    def forward(
        self,
        state_sequence: torch.Tensor,
        portfolio_sequence: torch.Tensor,
        tradable_mask: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        if state_sequence.ndim != 4:
            raise ValueError("state_sequence must have shape [batch, steps, stocks, features].")
        batch, steps, stocks, _ = state_sequence.shape
        if portfolio_sequence.ndim == 2:
            portfolio_sequence = portfolio_sequence.unsqueeze(1).expand(batch, steps, -1)
        portfolio_expanded = portfolio_sequence.unsqueeze(2).expand(batch, steps, stocks, portfolio_sequence.shape[-1])
        encoded_input = torch.cat([state_sequence, portfolio_expanded], dim=-1).permute(0, 2, 1, 3)
        encoded_input = encoded_input.reshape(batch * stocks, steps, encoded_input.shape[-1])
        _, hidden = self.encoder(encoded_input)
        latent = hidden[-1].reshape(batch, stocks, -1)
        logits = self.score_head(latent).squeeze(-1)
        if tradable_mask is not None:
            logits = logits.masked_fill(~tradable_mask.bool(), -1.0e9)
        stock_weight = torch.softmax(logits, dim=-1)
        latest_portfolio = portfolio_sequence[:, -1, :]
        cash_weight = torch.sigmoid(self.cash_head(latest_portfolio).squeeze(-1)).clamp(0.0, 0.95)
        gross = (1.0 - cash_weight).clamp(0.0, 1.0)
        raw_target_weight = (stock_weight * gross.unsqueeze(-1)).clamp(min=0.0, max=float(self.config.max_position_weight))
        total = raw_target_weight.sum(dim=-1, keepdim=True).clamp_min(1.0e-8)
        raw_target_weight = torch.where(total > gross.unsqueeze(-1), raw_target_weight / total * gross.unsqueeze(-1), raw_target_weight)
        return {
            "raw_target_weight": raw_target_weight,
            "cash_logit": torch.logit(cash_weight.clamp(1.0e-6, 1.0 - 1.0e-6)),
            "score_logits": logits,
            "policy_aux": latent.mean(dim=1),
        }


class DecisionTransformerTargetWeightPolicy(nn.Module):
    def __init__(
        self,
        config: SequencePolicyConfig,
        *,
        num_layers: int = 2,
        num_heads: int = 4,
    ) -> None:
        super().__init__()
        self.config = config
        input_dim = int(config.stock_feature_dim) + int(config.portfolio_feature_dim) + 2
        self.input_proj = nn.Linear(input_dim, int(config.hidden_dim))
        layer = nn.TransformerEncoderLayer(
            d_model=int(config.hidden_dim),
            nhead=max(int(num_heads), 1),
            dim_feedforward=int(config.hidden_dim) * 4,
            dropout=float(config.dropout),
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=max(int(num_layers), 1))
        self.score_head = nn.Linear(int(config.hidden_dim), 1)
        self.cash_head = nn.Sequential(
            nn.LayerNorm(int(config.hidden_dim)),
            nn.Linear(int(config.hidden_dim), 1),
        )

    def forward(
        self,
        state_sequence: torch.Tensor,
        portfolio_sequence: torch.Tensor,
        previous_weight_sequence: torch.Tensor | None = None,
        previous_reward_sequence: torch.Tensor | None = None,
        tradable_mask: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        if state_sequence.ndim != 4:
            raise ValueError("state_sequence must have shape [batch, steps, stocks, features].")
        batch, steps, stocks, _ = state_sequence.shape
        if portfolio_sequence.ndim == 2:
            portfolio_sequence = portfolio_sequence.unsqueeze(1).expand(batch, steps, -1)
        if previous_weight_sequence is None:
            previous_weight_sequence = torch.zeros(batch, steps, stocks, device=state_sequence.device, dtype=state_sequence.dtype)
        if previous_reward_sequence is None:
            previous_reward_sequence = torch.zeros(batch, steps, device=state_sequence.device, dtype=state_sequence.dtype)
        portfolio_expanded = portfolio_sequence.unsqueeze(2).expand(batch, steps, stocks, portfolio_sequence.shape[-1])
        reward_expanded = previous_reward_sequence.unsqueeze(-1).unsqueeze(-1).expand(batch, steps, stocks, 1)
        weight_expanded = previous_weight_sequence.unsqueeze(-1)
        tokens = torch.cat([state_sequence, portfolio_expanded, weight_expanded, reward_expanded], dim=-1)
        tokens = self.input_proj(tokens.reshape(batch, steps * stocks, tokens.shape[-1]))
        encoded = self.encoder(tokens).reshape(batch, steps, stocks, -1)
        latest = encoded[:, -1, :, :]
        logits = self.score_head(latest).squeeze(-1)
        if tradable_mask is not None:
            logits = logits.masked_fill(~tradable_mask.bool(), -1.0e9)
        stock_weight = torch.softmax(logits, dim=-1)
        pooled = latest.mean(dim=1)
        cash_weight = torch.sigmoid(self.cash_head(pooled).squeeze(-1)).clamp(0.0, 0.95)
        gross = (1.0 - cash_weight).clamp(0.0, 1.0)
        raw_target_weight = (stock_weight * gross.unsqueeze(-1)).clamp(min=0.0, max=float(self.config.max_position_weight))
        total = raw_target_weight.sum(dim=-1, keepdim=True).clamp_min(1.0e-8)
        raw_target_weight = torch.where(total > gross.unsqueeze(-1), raw_target_weight / total * gross.unsqueeze(-1), raw_target_weight)
        return {
            "raw_target_weight": raw_target_weight,
            "cash_logit": torch.logit(cash_weight.clamp(1.0e-6, 1.0 - 1.0e-6)),
            "score_logits": logits,
            "policy_aux": pooled,
        }


def sequence_policy_utility_loss(
    raw_target_weight: torch.Tensor,
    future_excess_return: torch.Tensor,
    current_weight: torch.Tensor,
    *,
    transaction_cost_rate: float = 0.001,
    turnover_penalty: float = 0.20,
    concentration_penalty: float = 0.02,
    entropy_bonus: float = 0.001,
) -> torch.Tensor:
    turnover = (raw_target_weight - current_weight).abs().sum(dim=-1)
    gross_return = (raw_target_weight * future_excess_return.nan_to_num(0.0)).sum(dim=-1)
    cost = turnover * float(transaction_cost_rate)
    concentration = (raw_target_weight**2).sum(dim=-1)
    weights = raw_target_weight.clamp_min(1.0e-12)
    entropy = -(weights * weights.log()).sum(dim=-1)
    utility = torch.log1p((gross_return - cost).clamp(min=-0.95))
    loss = -utility
    loss = loss + float(turnover_penalty) * turnover
    loss = loss + float(concentration_penalty) * concentration
    loss = loss - float(entropy_bonus) * entropy
    return loss.mean()
