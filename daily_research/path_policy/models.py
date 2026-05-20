from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn
import torch.nn.functional as F


PATH20_FORECAST_AUX_DIM = 8


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
        output_dim = self.horizon * 4 + PATH20_FORECAST_AUX_DIM
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
        self.head = nn.Linear(int(input_dim), self.horizon * 4 + PATH20_FORECAST_AUX_DIM)

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        return _split_path20_outputs(self.head(x), self.horizon)


class DLinearPath20Forecaster(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 64, horizon: int = 20) -> None:
        super().__init__()
        self.horizon = int(horizon)
        self.seasonal = nn.Linear(int(input_dim), int(hidden_dim))
        self.trend = nn.Linear(int(input_dim), int(hidden_dim))
        self.head = nn.Linear(int(hidden_dim) * 2, self.horizon * 4 + PATH20_FORECAST_AUX_DIM)

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        if x.ndim == 2:
            x = x.unsqueeze(1)
        trend_input = x.mean(dim=1)
        seasonal_input = x[:, -1, :] - trend_input
        encoded = torch.cat([F.gelu(self.trend(trend_input)), F.gelu(self.seasonal(seasonal_input))], dim=-1)
        return _split_path20_outputs(self.head(encoded), self.horizon)


class GRUPath20Forecaster(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 96,
        dropout: float = 0.10,
        horizon: int = 20,
        num_layers: int = 1,
    ) -> None:
        super().__init__()
        self.horizon = int(horizon)
        self.hidden_dim = int(hidden_dim)
        self.num_layers = max(int(num_layers), 1)
        self.gru = nn.GRU(
            input_size=int(input_dim),
            hidden_size=self.hidden_dim,
            num_layers=self.num_layers,
            batch_first=True,
            dropout=float(dropout) if self.num_layers > 1 else 0.0,
        )
        self.attention_pool = nn.Sequential(
            nn.LayerNorm(self.hidden_dim),
            nn.Linear(self.hidden_dim, 1),
        )
        self.head = nn.Sequential(
            nn.LayerNorm(self.hidden_dim * 2),
            nn.Dropout(float(dropout)),
            nn.Linear(self.hidden_dim * 2, self.hidden_dim),
            nn.GELU(),
            nn.Dropout(float(dropout)),
            nn.Linear(self.hidden_dim, self.horizon * 4 + PATH20_FORECAST_AUX_DIM),
        )

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        if x.ndim == 2:
            x = x.unsqueeze(1)
        outputs, hidden = self.gru(x)
        attention_logits = self.attention_pool(outputs).squeeze(-1)
        attention_weight = torch.softmax(attention_logits, dim=1).unsqueeze(-1)
        pooled = torch.sum(outputs * attention_weight, dim=1)
        encoded = torch.cat([pooled, hidden[-1]], dim=-1)
        return _split_path20_outputs(self.head(encoded), self.horizon)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim == 2:
            x = x.unsqueeze(1)
        outputs, hidden = self.gru(x)
        attention_logits = self.attention_pool(outputs).squeeze(-1)
        attention_weight = torch.softmax(attention_logits, dim=1).unsqueeze(-1)
        pooled = torch.sum(outputs * attention_weight, dim=1)
        return torch.cat([pooled, hidden[-1]], dim=-1)


class PatchTransformerPath20Forecaster(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 96,
        horizon: int = 20,
        patch_size: int = 4,
        patch_sizes: tuple[int, ...] | list[int] | None = None,
        num_layers: int = 2,
        num_heads: int = 4,
        dropout: float = 0.10,
        max_patches: int = 512,
    ) -> None:
        super().__init__()
        self.horizon = int(horizon)
        resolved_patch_sizes = tuple(int(item) for item in (patch_sizes or (patch_size,)) if int(item) > 0)
        self.patch_sizes = tuple(dict.fromkeys(resolved_patch_sizes or (max(int(patch_size), 1),)))
        self.patch_size = self.patch_sizes[0]
        self.input_dim = int(input_dim)
        self.hidden_dim = int(hidden_dim)
        self.max_patches = max(int(max_patches), 1)
        self.patch_projs = nn.ModuleList(
            [nn.Linear(self.input_dim * int(size), self.hidden_dim) for size in self.patch_sizes]
        )
        self.position_embeddings = nn.ParameterList(
            [nn.Parameter(torch.zeros(1, self.max_patches, self.hidden_dim)) for _ in self.patch_sizes]
        )
        self.scale_embeddings = nn.Parameter(torch.zeros(1, len(self.patch_sizes), self.hidden_dim))
        self.cls_token = nn.Parameter(torch.zeros(1, 1, self.hidden_dim))
        self.input_dropout = nn.Dropout(float(dropout))
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.hidden_dim,
            nhead=max(int(num_heads), 1),
            dim_feedforward=self.hidden_dim * 4,
            dropout=float(dropout),
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=max(int(num_layers), 1),
            enable_nested_tensor=False,
        )
        self.head = nn.Sequential(
            nn.LayerNorm(self.hidden_dim),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.GELU(),
            nn.Dropout(float(dropout)),
            nn.Linear(self.hidden_dim, self.horizon * 4 + PATH20_FORECAST_AUX_DIM),
        )
        nn.init.normal_(self.cls_token, std=0.02)
        nn.init.normal_(self.scale_embeddings, std=0.02)
        for embedding in self.position_embeddings:
            nn.init.normal_(embedding, std=0.02)

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        if x.ndim == 2:
            x = x.unsqueeze(1)
        batch, steps, features = x.shape
        if int(features) != self.input_dim:
            raise ValueError(f"Expected input_dim={self.input_dim}, got {features}.")
        tokens: list[torch.Tensor] = []
        for scale_idx, patch_size in enumerate(self.patch_sizes):
            work = x
            pad = (-steps) % int(patch_size)
            if pad:
                work = F.pad(work, (0, 0, 0, pad))
            patches = work.reshape(batch, -1, int(patch_size) * self.input_dim)
            if patches.shape[1] > self.max_patches:
                raise ValueError(
                    f"Patch count {patches.shape[1]} exceeds max_patches={self.max_patches} for patch_size={patch_size}."
                )
            projected = self.patch_projs[scale_idx](patches)
            projected = projected + self.position_embeddings[scale_idx][:, : projected.shape[1], :]
            projected = projected + self.scale_embeddings[:, scale_idx : scale_idx + 1, :]
            tokens.append(projected)
        token_sequence = torch.cat(tokens, dim=1)
        cls = self.cls_token.expand(batch, -1, -1)
        encoded = self.encoder(self.input_dropout(torch.cat([cls, token_sequence], dim=1)))
        pooled = encoded[:, 0, :]
        return _split_path20_outputs(self.head(pooled), self.horizon)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim == 2:
            x = x.unsqueeze(1)
        batch, steps, features = x.shape
        if int(features) != self.input_dim:
            raise ValueError(f"Expected input_dim={self.input_dim}, got {features}.")
        tokens: list[torch.Tensor] = []
        for scale_idx, patch_size in enumerate(self.patch_sizes):
            work = x
            pad = (-steps) % int(patch_size)
            if pad:
                work = F.pad(work, (0, 0, 0, pad))
            patches = work.reshape(batch, -1, int(patch_size) * self.input_dim)
            if patches.shape[1] > self.max_patches:
                raise ValueError(
                    f"Patch count {patches.shape[1]} exceeds max_patches={self.max_patches} for patch_size={patch_size}."
                )
            projected = self.patch_projs[scale_idx](patches)
            projected = projected + self.position_embeddings[scale_idx][:, : projected.shape[1], :]
            projected = projected + self.scale_embeddings[:, scale_idx : scale_idx + 1, :]
            tokens.append(projected)
        token_sequence = torch.cat(tokens, dim=1)
        cls = self.cls_token.expand(batch, -1, -1)
        encoded = self.encoder(self.input_dropout(torch.cat([cls, token_sequence], dim=1)))
        return encoded[:, 0, :]


class StaticContextEncoder(nn.Module):
    field_order = ("symbol", "exchange", "industry", "liquidity_bucket", "price_bucket")

    def __init__(
        self,
        *,
        vocab_sizes: dict[str, int] | None = None,
        embedding_dims: dict[str, int] | None = None,
        dropout: float = 0.20,
    ) -> None:
        super().__init__()
        resolved_vocab = dict(vocab_sizes or {})
        resolved_dims = {
            "symbol": 16,
            "exchange": 4,
            "industry": 8,
            "liquidity_bucket": 4,
            "price_bucket": 4,
            **dict(embedding_dims or {}),
        }
        self.embeddings = nn.ModuleDict()
        total_dim = 0
        for field in self.field_order:
            vocab_size = max(int(resolved_vocab.get(field, 1)), 1)
            dim = max(int(resolved_dims.get(field, 1)), 1)
            self.embeddings[field] = nn.Embedding(vocab_size, dim)
            total_dim += dim
        self.output_dim = int(total_dim)
        self.dropout = nn.Dropout(float(dropout))

    def forward(self, static_context_ids: torch.Tensor | None, *, batch_size: int, device: torch.device) -> torch.Tensor:
        if static_context_ids is None:
            static_context_ids = torch.zeros((batch_size, len(self.field_order)), dtype=torch.long, device=device)
        static_context_ids = static_context_ids.to(device=device, dtype=torch.long)
        if static_context_ids.ndim != 2 or static_context_ids.shape[1] < len(self.field_order):
            raise ValueError("static_context_ids must have shape [batch, 5].")
        vectors = []
        for pos, field in enumerate(self.field_order):
            embedding = self.embeddings[field]
            ids = static_context_ids[:, pos].clamp(min=0, max=embedding.num_embeddings - 1)
            vectors.append(embedding(ids))
        return self.dropout(torch.cat(vectors, dim=-1))


class StaticContextPath20Forecaster(nn.Module):
    def __init__(
        self,
        *,
        temporal_encoder: nn.Module,
        temporal_dim: int,
        hidden_dim: int,
        horizon: int,
        vocab_sizes: dict[str, int] | None = None,
        embedding_dims: dict[str, int] | None = None,
        static_dropout: float = 0.20,
        dropout: float = 0.10,
    ) -> None:
        super().__init__()
        self.horizon = int(horizon)
        self.temporal_encoder = temporal_encoder
        self.static_encoder = StaticContextEncoder(
            vocab_sizes=vocab_sizes,
            embedding_dims=embedding_dims,
            dropout=static_dropout,
        )
        self.head = nn.Sequential(
            nn.LayerNorm(int(temporal_dim) + self.static_encoder.output_dim),
            nn.Dropout(float(dropout)),
            nn.Linear(int(temporal_dim) + self.static_encoder.output_dim, int(hidden_dim)),
            nn.GELU(),
            nn.Dropout(float(dropout)),
            nn.Linear(int(hidden_dim), self.horizon * 4 + PATH20_FORECAST_AUX_DIM),
        )

    def forward(self, x: torch.Tensor, static_context_ids: torch.Tensor | None = None) -> dict[str, torch.Tensor]:
        encoded = self.temporal_encoder.encode(x)
        static = self.static_encoder(static_context_ids, batch_size=int(encoded.shape[0]), device=encoded.device)
        return _split_path20_outputs(self.head(torch.cat([encoded, static], dim=-1)), self.horizon)


class StockMixerPath20Forecaster(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 96, horizon: int = 20, dropout: float = 0.10) -> None:
        super().__init__()
        self.horizon = int(horizon)
        self.encoder = GRUPath20Forecaster(input_dim=input_dim, hidden_dim=hidden_dim, dropout=dropout, horizon=horizon, num_layers=1)
        encoded_dim = int(hidden_dim) * 2
        self.stock_norm = nn.LayerNorm(encoded_dim)
        self.stock_attention = nn.MultiheadAttention(
            embed_dim=encoded_dim,
            num_heads=4 if encoded_dim % 4 == 0 else 1,
            dropout=float(dropout),
            batch_first=True,
        )
        self.mixer = nn.Sequential(
            nn.LayerNorm(encoded_dim),
            nn.Linear(encoded_dim, encoded_dim),
            nn.GELU(),
            nn.Dropout(float(dropout)),
        )
        self.head = nn.Linear(encoded_dim, self.horizon * 4 + PATH20_FORECAST_AUX_DIM)

    def forward(self, x: torch.Tensor, stock_mask: torch.Tensor | None = None) -> dict[str, torch.Tensor]:
        if x.ndim == 4:
            dates, stocks, steps, features = x.shape
            encoded = self.encoder.encode(x.reshape(dates * stocks, steps, features)).reshape(dates, stocks, -1)
            key_padding_mask = None
            if stock_mask is not None:
                key_padding_mask = ~stock_mask.to(device=encoded.device, dtype=torch.bool)
            tokens = self.stock_norm(encoded)
            attended, _ = self.stock_attention(
                tokens,
                tokens,
                tokens,
                key_padding_mask=key_padding_mask,
                need_weights=False,
            )
            mixed = self.mixer(encoded + attended).reshape(dates * stocks, -1)
            return _split_path20_outputs(self.head(mixed), self.horizon)
        encoded = self.encoder.encode(x)
        tokens = self.stock_norm(encoded).unsqueeze(0)
        attended, _ = self.stock_attention(tokens, tokens, tokens, need_weights=False)
        mixed = self.mixer(encoded + attended.squeeze(0))
        return _split_path20_outputs(self.head(mixed), self.horizon)


class SectorSlotMixerPath20Forecaster(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 96,
        horizon: int = 20,
        dropout: float = 0.10,
        slot_count: int = 8,
    ) -> None:
        super().__init__()
        self.horizon = int(horizon)
        self.encoder = GRUPath20Forecaster(input_dim=input_dim, hidden_dim=hidden_dim, dropout=dropout, horizon=horizon, num_layers=1)
        self.slots = nn.Parameter(torch.zeros(max(int(slot_count), 1), int(hidden_dim) * 2))
        self.slot_proj = nn.Linear(int(hidden_dim) * 4, int(hidden_dim) * 2)
        self.head = nn.Sequential(
            nn.LayerNorm(int(hidden_dim) * 2),
            nn.Dropout(float(dropout)),
            nn.Linear(int(hidden_dim) * 2, self.horizon * 4 + PATH20_FORECAST_AUX_DIM),
        )
        nn.init.normal_(self.slots, std=0.02)

    def _slot_context(self, encoded: torch.Tensor, stock_mask: torch.Tensor | None = None) -> torch.Tensor:
        if encoded.ndim == 2:
            encoded = encoded.unsqueeze(0)
            squeeze = True
        else:
            squeeze = False
        slots = self.slots.unsqueeze(0).expand(encoded.shape[0], -1, -1)
        slot_scores = torch.einsum("bkd,bnd->bkn", F.normalize(slots, dim=-1), F.normalize(encoded, dim=-1))
        if stock_mask is not None:
            mask = stock_mask.to(device=encoded.device, dtype=torch.bool)
            slot_scores = slot_scores.masked_fill(~mask.unsqueeze(1), torch.finfo(slot_scores.dtype).min)
        slot_weights = torch.softmax(slot_scores, dim=-1)
        slot_context = slot_weights @ encoded
        token_scores = torch.einsum("bnd,bkd->bnk", F.normalize(encoded, dim=-1), F.normalize(slot_context, dim=-1))
        token_weights = torch.softmax(token_scores, dim=-1)
        context = token_weights @ slot_context
        return context.squeeze(0) if squeeze else context

    def forward(self, x: torch.Tensor, stock_mask: torch.Tensor | None = None) -> dict[str, torch.Tensor]:
        if x.ndim == 4:
            dates, stocks, steps, features = x.shape
            encoded = self.encoder.encode(x.reshape(dates * stocks, steps, features)).reshape(dates, stocks, -1)
            context = self._slot_context(encoded, stock_mask=stock_mask)
            fused = F.gelu(self.slot_proj(torch.cat([encoded, context], dim=-1))).reshape(dates * stocks, -1)
            return _split_path20_outputs(self.head(fused), self.horizon)
        encoded = self.encoder.encode(x)
        context = self._slot_context(encoded, stock_mask=stock_mask)
        fused = F.gelu(self.slot_proj(torch.cat([encoded, context], dim=-1)))
        return _split_path20_outputs(self.head(fused), self.horizon)


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
