from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn
import torch.nn.functional as F


@dataclass(frozen=True)
class PathPolicyModelConfig:
    stock_feature_dim: int
    path_feature_dim: int
    portfolio_feature_dim: int = 8
    hidden_dim: int = 128
    dropout: float = 0.10
    max_position_weight: float = 0.20


class Path20ForecasterMLP(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 128, dropout: float = 0.10, horizon: int = 20) -> None:
        super().__init__()
        self.horizon = int(horizon)
        output_dim = self.horizon * 4 + 6
        self.net = nn.Sequential(
            nn.LayerNorm(int(input_dim)),
            nn.Linear(int(input_dim), int(hidden_dim)),
            nn.GELU(),
            nn.Dropout(float(dropout)),
            nn.Linear(int(hidden_dim), int(hidden_dim)),
            nn.GELU(),
            nn.Dropout(float(dropout)),
            nn.Linear(int(hidden_dim), output_dim),
        )

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        raw = self.net(x)
        horizon = self.horizon
        mu = raw[:, :horizon]
        q_raw = raw[:, horizon : horizon * 4].reshape(raw.shape[0], horizon, 3)
        q_sorted = torch.sort(q_raw, dim=-1).values
        aux = raw[:, horizon * 4 :]
        return {
            "mu": mu,
            "q10": q_sorted[:, :, 0],
            "q50": q_sorted[:, :, 1],
            "q90": q_sorted[:, :, 2],
            "aux": aux,
        }


def _split_path20_outputs(raw: torch.Tensor, horizon: int) -> dict[str, torch.Tensor]:
    horizon = int(horizon)
    mu = raw[:, :horizon]
    q_raw = raw[:, horizon : horizon * 4].reshape(raw.shape[0], horizon, 3)
    q_sorted = torch.sort(q_raw, dim=-1).values
    aux = raw[:, horizon * 4 :]
    return {
        "mu": mu,
        "q10": q_sorted[:, :, 0],
        "q50": q_sorted[:, :, 1],
        "q90": q_sorted[:, :, 2],
        "aux": aux,
    }


class LinearPath20Forecaster(nn.Module):
    def __init__(self, input_dim: int, horizon: int = 20) -> None:
        super().__init__()
        self.horizon = int(horizon)
        self.head = nn.Linear(int(input_dim), self.horizon * 4 + 6)

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        return _split_path20_outputs(self.head(x), self.horizon)


class DLinearPath20Forecaster(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 64, horizon: int = 20) -> None:
        super().__init__()
        self.horizon = int(horizon)
        self.seasonal = nn.Linear(int(input_dim), int(hidden_dim))
        self.trend = nn.Linear(int(input_dim), int(hidden_dim))
        self.head = nn.Linear(int(hidden_dim) * 2, self.horizon * 4 + 6)

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        if x.ndim == 2:
            x = x.unsqueeze(1)
        trend_input = x.mean(dim=1)
        seasonal_input = x[:, -1, :] - trend_input
        encoded = torch.cat([F.gelu(self.trend(trend_input)), F.gelu(self.seasonal(seasonal_input))], dim=-1)
        return _split_path20_outputs(self.head(encoded), self.horizon)


class GRUPath20Forecaster(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 96, dropout: float = 0.10, horizon: int = 20) -> None:
        super().__init__()
        self.horizon = int(horizon)
        self.gru = nn.GRU(
            input_size=int(input_dim),
            hidden_size=int(hidden_dim),
            batch_first=True,
            dropout=0.0,
        )
        self.head = nn.Sequential(
            nn.LayerNorm(int(hidden_dim)),
            nn.Dropout(float(dropout)),
            nn.Linear(int(hidden_dim), self.horizon * 4 + 6),
        )

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        if x.ndim == 2:
            x = x.unsqueeze(1)
        _, hidden = self.gru(x)
        return _split_path20_outputs(self.head(hidden[-1]), self.horizon)


class PatchTransformerPath20Forecaster(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 96,
        horizon: int = 20,
        patch_size: int = 4,
        num_layers: int = 2,
        num_heads: int = 4,
        dropout: float = 0.10,
    ) -> None:
        super().__init__()
        self.horizon = int(horizon)
        self.patch_size = max(int(patch_size), 1)
        self.input_dim = int(input_dim)
        self.patch_proj = nn.Linear(self.input_dim * self.patch_size, int(hidden_dim))
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=int(hidden_dim),
            nhead=max(int(num_heads), 1),
            dim_feedforward=int(hidden_dim) * 4,
            dropout=float(dropout),
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=max(int(num_layers), 1))
        self.head = nn.Sequential(
            nn.LayerNorm(int(hidden_dim)),
            nn.Linear(int(hidden_dim), self.horizon * 4 + 6),
        )

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        if x.ndim == 2:
            x = x.unsqueeze(1)
        batch, steps, features = x.shape
        if int(features) != self.input_dim:
            raise ValueError(f"Expected input_dim={self.input_dim}, got {features}.")
        pad = (-steps) % self.patch_size
        if pad:
            x = F.pad(x, (0, 0, 0, pad))
        patches = x.reshape(batch, -1, self.patch_size * self.input_dim)
        encoded = self.encoder(self.patch_proj(patches))
        pooled = encoded.mean(dim=1)
        return _split_path20_outputs(self.head(pooled), self.horizon)


class NeuralTargetWeightPolicy(nn.Module):
    def __init__(self, config: PathPolicyModelConfig) -> None:
        super().__init__()
        self.config = config
        input_dim = int(config.stock_feature_dim) + int(config.path_feature_dim) + int(config.portfolio_feature_dim)
        self.score_net = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, int(config.hidden_dim)),
            nn.GELU(),
            nn.Dropout(float(config.dropout)),
            nn.Linear(int(config.hidden_dim), int(config.hidden_dim)),
            nn.GELU(),
            nn.Dropout(float(config.dropout)),
            nn.Linear(int(config.hidden_dim), 1),
        )
        self.cash_head = nn.Sequential(
            nn.LayerNorm(max(int(config.portfolio_feature_dim), 1)),
            nn.Linear(max(int(config.portfolio_feature_dim), 1), max(int(config.hidden_dim // 2), 8)),
            nn.GELU(),
            nn.Linear(max(int(config.hidden_dim // 2), 8), 1),
        )

    def forward(
        self,
        stock_features: torch.Tensor,
        path_features: torch.Tensor,
        portfolio_features: torch.Tensor,
        tradable_mask: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        if portfolio_features.ndim == 1:
            portfolio_features = portfolio_features.unsqueeze(0).expand(stock_features.shape[0], -1)
        if portfolio_features.shape[0] == 1 and stock_features.shape[0] > 1:
            portfolio_features = portfolio_features.expand(stock_features.shape[0], -1)
        x = torch.cat([stock_features, path_features, portfolio_features], dim=-1)
        logits = self.score_net(x).squeeze(-1)
        if tradable_mask is not None:
            logits = logits.masked_fill(~tradable_mask.bool(), -1.0e9)
        weights = torch.softmax(logits, dim=0)
        cash_logit = self.cash_head(portfolio_features[:1]).squeeze()
        cash_weight = torch.sigmoid(cash_logit).clamp(0.0, 0.95)
        gross = (1.0 - cash_weight).clamp(0.0, 1.0)
        target_weight = weights * gross
        target_weight = torch.clamp(target_weight, min=0.0, max=float(self.config.max_position_weight))
        total = target_weight.sum().clamp_min(1.0e-8)
        target_weight = torch.where(total > gross, target_weight / total * gross, target_weight)
        return {
            "target_weight": target_weight,
            "cash_weight": (1.0 - target_weight.sum()).clamp(0.0, 1.0),
            "logits": logits,
        }


def path_feature_tensor_from_prediction(prediction: dict[str, torch.Tensor]) -> torch.Tensor:
    return torch.cat(
        [
            prediction["mu"],
            prediction["q10"],
            prediction["q50"],
            prediction["q90"],
            prediction["aux"],
        ],
        dim=-1,
    )


def pinball_loss(pred: torch.Tensor, target: torch.Tensor, quantile: float) -> torch.Tensor:
    diff = target - pred
    q = float(quantile)
    return torch.maximum(q * diff, (q - 1.0) * diff).mean()


def pairwise_rank_loss(score: torch.Tensor, target: torch.Tensor, max_pairs: int = 4096) -> torch.Tensor:
    valid = torch.isfinite(score) & torch.isfinite(target)
    score = score[valid]
    target = target[valid]
    if score.numel() < 2:
        return score.new_tensor(0.0)
    diff_target = target.unsqueeze(0) - target.unsqueeze(1)
    pair_mask = diff_target.abs() > 1.0e-8
    if not bool(pair_mask.any().item()):
        return score.new_tensor(0.0)
    diff_score = score.unsqueeze(0) - score.unsqueeze(1)
    sign = torch.sign(diff_target)
    losses = F.softplus(-sign[pair_mask] * diff_score[pair_mask])
    if losses.numel() > int(max_pairs):
        idx = torch.randperm(losses.numel(), device=losses.device)[: int(max_pairs)]
        losses = losses[idx]
    return losses.mean()


def portfolio_utility_loss(
    target_weight: torch.Tensor,
    future_return: torch.Tensor,
    current_weight: torch.Tensor | None = None,
    *,
    transaction_cost_rate: float = 0.0013,
    concentration_penalty: float = 0.02,
    turnover_penalty: float = 0.20,
) -> torch.Tensor:
    current = torch.zeros_like(target_weight) if current_weight is None else current_weight.to(target_weight)
    turnover = (target_weight - current).abs().sum()
    gross_return = (target_weight * future_return.nan_to_num(0.0)).sum()
    cost = turnover * float(transaction_cost_rate)
    concentration = (target_weight**2).sum()
    utility = torch.log1p((gross_return - cost).clamp(min=-0.95))
    return -utility + float(turnover_penalty) * cost + float(concentration_penalty) * concentration
