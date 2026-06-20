from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn
import torch.nn.functional as F


PATH20_FORECAST_AUX_DIM = 20
PATH20_DECISION_AUX_DIM = 15
PATH20_DEFAULT_CUMULATIVE_HORIZONS: tuple[int, ...] = (1, 3, 5, 10, 20)
PATH20_FORECAST_OUTPUT_PROFILES = ("forecast_path_v1", "decision_utility_v1")


def normalize_path20_output_profile(output_profile: str | None) -> str:
    profile = str(output_profile or "forecast_path_v1").strip().lower()
    if profile not in PATH20_FORECAST_OUTPUT_PROFILES:
        raise ValueError(f"Unsupported path20 forecast output profile: {profile}")
    return profile


def normalize_path20_cumulative_horizons(
    cumulative_horizons: tuple[int, ...] | list[int] | str | None = None,
    *,
    horizon: int = 20,
) -> tuple[int, ...]:
    max_horizon = int(horizon)
    if max_horizon <= 0:
        raise ValueError("horizon must be positive.")
    if cumulative_horizons is None:
        values = list(PATH20_DEFAULT_CUMULATIVE_HORIZONS)
    elif isinstance(cumulative_horizons, str):
        values = [int(item.strip()) for item in cumulative_horizons.split(",") if item.strip()]
    else:
        values = [int(item) for item in cumulative_horizons]
    if not values:
        raise ValueError("cumulative_horizons must contain at least one horizon.")
    resolved = tuple(sorted(dict.fromkeys(values)))
    invalid = [item for item in resolved if int(item) <= 0 or int(item) > max_horizon]
    if invalid:
        raise ValueError(f"cumulative_horizons must be in [1, {max_horizon}], got {invalid}.")
    return resolved


def path20_forecast_aux_dim(cumulative_horizons: tuple[int, ...] | list[int] | str | None = None, *, horizon: int = 20) -> int:
    return len(normalize_path20_cumulative_horizons(cumulative_horizons, horizon=horizon)) * 4


def path20_decision_aux_dim(cumulative_horizons: tuple[int, ...] | list[int] | str | None = None, *, horizon: int = 20) -> int:
    return len(normalize_path20_cumulative_horizons(cumulative_horizons, horizon=horizon)) * 3


def path20_forecast_output_dim(
    horizon: int,
    output_profile: str | None = "forecast_path_v1",
    cumulative_horizons: tuple[int, ...] | list[int] | str | None = None,
) -> int:
    profile = normalize_path20_output_profile(output_profile)
    output_dim = int(horizon) * 4 + path20_forecast_aux_dim(cumulative_horizons, horizon=int(horizon))
    if profile == "decision_utility_v1":
        output_dim += path20_decision_aux_dim(cumulative_horizons, horizon=int(horizon))
    return output_dim


@dataclass(frozen=True)
class PathPolicyModelConfig:
    stock_feature_dim: int
    path_feature_dim: int
    portfolio_feature_dim: int = 8
    hidden_dim: int = 128
    dropout: float = 0.10
    max_position_weight: float = 0.20


class Path20ForecasterMLP(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 128,
        dropout: float = 0.10,
        horizon: int = 20,
        output_profile: str = "forecast_path_v1",
        cumulative_horizons: tuple[int, ...] | list[int] | str | None = None,
    ) -> None:
        super().__init__()
        self.horizon = int(horizon)
        self.cumulative_horizons = normalize_path20_cumulative_horizons(cumulative_horizons, horizon=self.horizon)
        self.output_profile = normalize_path20_output_profile(output_profile)
        output_dim = path20_forecast_output_dim(self.horizon, self.output_profile, self.cumulative_horizons)
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
        return _split_path20_outputs(raw, self.horizon, self.output_profile, self.cumulative_horizons)


def _split_path20_outputs(
    raw: torch.Tensor,
    horizon: int,
    output_profile: str | None = "forecast_path_v1",
    cumulative_horizons: tuple[int, ...] | list[int] | str | None = None,
) -> dict[str, torch.Tensor]:
    horizon = int(horizon)
    resolved_horizons = normalize_path20_cumulative_horizons(cumulative_horizons, horizon=horizon)
    profile = normalize_path20_output_profile(output_profile)
    mu = raw[:, :horizon]
    q_raw = raw[:, horizon : horizon * 4].reshape(raw.shape[0], horizon, 3)
    q_sorted = torch.sort(q_raw, dim=-1).values
    aux_start = horizon * 4
    aux_end = aux_start + path20_forecast_aux_dim(resolved_horizons, horizon=horizon)
    aux = raw[:, aux_start:aux_end]
    output = {
        "mu": mu,
        "q10": q_sorted[:, :, 0],
        "q50": q_sorted[:, :, 1],
        "q90": q_sorted[:, :, 2],
        "aux": aux,
    }
    if profile == "decision_utility_v1":
        output["decision_aux"] = raw[:, aux_end : aux_end + path20_decision_aux_dim(resolved_horizons, horizon=horizon)]
    return output


class LinearPath20Forecaster(nn.Module):
    def __init__(
        self,
        input_dim: int,
        horizon: int = 20,
        output_profile: str = "forecast_path_v1",
        cumulative_horizons: tuple[int, ...] | list[int] | str | None = None,
    ) -> None:
        super().__init__()
        self.horizon = int(horizon)
        self.cumulative_horizons = normalize_path20_cumulative_horizons(cumulative_horizons, horizon=self.horizon)
        self.output_profile = normalize_path20_output_profile(output_profile)
        self.head = nn.Linear(int(input_dim), path20_forecast_output_dim(self.horizon, self.output_profile, self.cumulative_horizons))

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        return _split_path20_outputs(self.head(x), self.horizon, self.output_profile, self.cumulative_horizons)


class DLinearPath20Forecaster(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 64,
        horizon: int = 20,
        output_profile: str = "forecast_path_v1",
        cumulative_horizons: tuple[int, ...] | list[int] | str | None = None,
    ) -> None:
        super().__init__()
        self.horizon = int(horizon)
        self.cumulative_horizons = normalize_path20_cumulative_horizons(cumulative_horizons, horizon=self.horizon)
        self.output_profile = normalize_path20_output_profile(output_profile)
        self.seasonal = nn.Linear(int(input_dim), int(hidden_dim))
        self.trend = nn.Linear(int(input_dim), int(hidden_dim))
        self.head = nn.Linear(int(hidden_dim) * 2, path20_forecast_output_dim(self.horizon, self.output_profile, self.cumulative_horizons))

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        if x.ndim == 2:
            x = x.unsqueeze(1)
        trend_input = x.mean(dim=1)
        seasonal_input = x[:, -1, :] - trend_input
        encoded = torch.cat([F.gelu(self.trend(trend_input)), F.gelu(self.seasonal(seasonal_input))], dim=-1)
        return _split_path20_outputs(self.head(encoded), self.horizon, self.output_profile, self.cumulative_horizons)


class GRUPath20Forecaster(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 96,
        dropout: float = 0.10,
        horizon: int = 20,
        num_layers: int = 1,
        output_profile: str = "forecast_path_v1",
        cumulative_horizons: tuple[int, ...] | list[int] | str | None = None,
    ) -> None:
        super().__init__()
        self.horizon = int(horizon)
        self.cumulative_horizons = normalize_path20_cumulative_horizons(cumulative_horizons, horizon=self.horizon)
        self.output_profile = normalize_path20_output_profile(output_profile)
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
            nn.Linear(self.hidden_dim, path20_forecast_output_dim(self.horizon, self.output_profile, self.cumulative_horizons)),
        )

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        if x.ndim == 2:
            x = x.unsqueeze(1)
        outputs, hidden = self.gru(x)
        attention_logits = self.attention_pool(outputs).squeeze(-1)
        attention_weight = torch.softmax(attention_logits, dim=1).unsqueeze(-1)
        pooled = torch.sum(outputs * attention_weight, dim=1)
        encoded = torch.cat([pooled, hidden[-1]], dim=-1)
        return _split_path20_outputs(self.head(encoded), self.horizon, self.output_profile, self.cumulative_horizons)

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
        output_profile: str = "forecast_path_v1",
        cumulative_horizons: tuple[int, ...] | list[int] | str | None = None,
    ) -> None:
        super().__init__()
        self.horizon = int(horizon)
        self.cumulative_horizons = normalize_path20_cumulative_horizons(cumulative_horizons, horizon=self.horizon)
        self.output_profile = normalize_path20_output_profile(output_profile)
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
            nn.Linear(self.hidden_dim, path20_forecast_output_dim(self.horizon, self.output_profile, self.cumulative_horizons)),
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
        return _split_path20_outputs(self.head(pooled), self.horizon, self.output_profile, self.cumulative_horizons)

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
    default_field_order = ("symbol", "exchange", "industry", "liquidity_bucket", "price_bucket")

    def __init__(
        self,
        *,
        vocab_sizes: dict[str, int] | None = None,
        embedding_dims: dict[str, int] | None = None,
        fields: tuple[str, ...] | list[str] | None = None,
        dropout: float = 0.20,
    ) -> None:
        super().__init__()
        self.field_order = tuple(str(item).strip() for item in (fields or self.default_field_order) if str(item).strip())
        resolved_vocab = dict(vocab_sizes or {})
        resolved_dims = {
            "symbol": 16,
            "exchange": 4,
            "industry": 8,
            "board": 4,
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
            raise ValueError(f"static_context_ids must have shape [batch, {len(self.field_order)}].")
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
        static_fields: tuple[str, ...] | list[str] | None = None,
        static_dropout: float = 0.20,
        dropout: float = 0.10,
        output_profile: str = "forecast_path_v1",
        cumulative_horizons: tuple[int, ...] | list[int] | str | None = None,
    ) -> None:
        super().__init__()
        self.horizon = int(horizon)
        self.cumulative_horizons = normalize_path20_cumulative_horizons(cumulative_horizons, horizon=self.horizon)
        self.output_profile = normalize_path20_output_profile(output_profile)
        self.temporal_encoder = temporal_encoder
        self.static_encoder = StaticContextEncoder(
            vocab_sizes=vocab_sizes,
            embedding_dims=embedding_dims,
            fields=static_fields,
            dropout=static_dropout,
        )
        self.head = nn.Sequential(
            nn.LayerNorm(int(temporal_dim) + self.static_encoder.output_dim),
            nn.Dropout(float(dropout)),
            nn.Linear(int(temporal_dim) + self.static_encoder.output_dim, int(hidden_dim)),
            nn.GELU(),
            nn.Dropout(float(dropout)),
            nn.Linear(int(hidden_dim), path20_forecast_output_dim(self.horizon, self.output_profile, self.cumulative_horizons)),
        )

    def forward(self, x: torch.Tensor, static_context_ids: torch.Tensor | None = None) -> dict[str, torch.Tensor]:
        encoded = self.temporal_encoder.encode(x)
        static = self.static_encoder(static_context_ids, batch_size=int(encoded.shape[0]), device=encoded.device)
        return _split_path20_outputs(self.head(torch.cat([encoded, static], dim=-1)), self.horizon, self.output_profile, self.cumulative_horizons)


class StockMixerPath20Forecaster(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 96,
        horizon: int = 20,
        dropout: float = 0.10,
        output_profile: str = "forecast_path_v1",
        cumulative_horizons: tuple[int, ...] | list[int] | str | None = None,
    ) -> None:
        super().__init__()
        self.horizon = int(horizon)
        self.cumulative_horizons = normalize_path20_cumulative_horizons(cumulative_horizons, horizon=self.horizon)
        self.output_profile = normalize_path20_output_profile(output_profile)
        self.encoder = GRUPath20Forecaster(input_dim=input_dim, hidden_dim=hidden_dim, dropout=dropout, horizon=horizon, num_layers=1, cumulative_horizons=self.cumulative_horizons)
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
        self.head = nn.Linear(encoded_dim, path20_forecast_output_dim(self.horizon, self.output_profile, self.cumulative_horizons))

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
            return _split_path20_outputs(self.head(mixed), self.horizon, self.output_profile, self.cumulative_horizons)
        encoded = self.encoder.encode(x)
        tokens = self.stock_norm(encoded).unsqueeze(0)
        attended, _ = self.stock_attention(tokens, tokens, tokens, need_weights=False)
        mixed = self.mixer(encoded + attended.squeeze(0))
        return _split_path20_outputs(self.head(mixed), self.horizon, self.output_profile, self.cumulative_horizons)


class SectorSlotMixerPath20Forecaster(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 96,
        horizon: int = 20,
        dropout: float = 0.10,
        slot_count: int = 8,
        output_profile: str = "forecast_path_v1",
        cumulative_horizons: tuple[int, ...] | list[int] | str | None = None,
    ) -> None:
        super().__init__()
        self.horizon = int(horizon)
        self.cumulative_horizons = normalize_path20_cumulative_horizons(cumulative_horizons, horizon=self.horizon)
        self.output_profile = normalize_path20_output_profile(output_profile)
        self.encoder = GRUPath20Forecaster(input_dim=input_dim, hidden_dim=hidden_dim, dropout=dropout, horizon=horizon, num_layers=1, cumulative_horizons=self.cumulative_horizons)
        self.slots = nn.Parameter(torch.zeros(max(int(slot_count), 1), int(hidden_dim) * 2))
        self.slot_proj = nn.Linear(int(hidden_dim) * 4, int(hidden_dim) * 2)
        self.head = nn.Sequential(
            nn.LayerNorm(int(hidden_dim) * 2),
            nn.Dropout(float(dropout)),
            nn.Linear(int(hidden_dim) * 2, path20_forecast_output_dim(self.horizon, self.output_profile, self.cumulative_horizons)),
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
            return _split_path20_outputs(self.head(fused), self.horizon, self.output_profile, self.cumulative_horizons)
        encoded = self.encoder.encode(x)
        context = self._slot_context(encoded, stock_mask=stock_mask)
        fused = F.gelu(self.slot_proj(torch.cat([encoded, context], dim=-1)))
        return _split_path20_outputs(self.head(fused), self.horizon, self.output_profile, self.cumulative_horizons)


class ExpertFusionPath20Forecaster(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 192,
        horizon: int = 20,
        dropout: float = 0.15,
        gru_layers: int = 2,
        transformer_layers: int = 4,
        transformer_heads: int = 4,
        patch_sizes: tuple[int, ...] | list[int] | None = None,
        router_temperature: float = 1.0,
        static_context_vocab_sizes: dict[str, int] | None = None,
        static_context_embedding_dims: dict[str, int] | None = None,
        static_context_fields: tuple[str, ...] | list[str] | None = None,
        static_context_dropout: float = 0.20,
        output_profile: str = "forecast_path_v1",
        cumulative_horizons: tuple[int, ...] | list[int] | str | None = None,
    ) -> None:
        super().__init__()
        self.horizon = int(horizon)
        self.cumulative_horizons = normalize_path20_cumulative_horizons(cumulative_horizons, horizon=self.horizon)
        self.output_profile = normalize_path20_output_profile(output_profile)
        self.hidden_dim = int(hidden_dim)
        self.router_temperature = max(float(router_temperature), 1.0e-4)
        self.expert_names = ("gru", "patch_transformer", "dlinear")
        self.gru_encoder = GRUPath20Forecaster(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            dropout=dropout,
            horizon=horizon,
            num_layers=gru_layers,
            output_profile=output_profile,
            cumulative_horizons=self.cumulative_horizons,
        )
        requested_heads = max(int(transformer_heads), 1)
        heads = requested_heads if int(hidden_dim) % requested_heads == 0 else 1
        self.patch_encoder = PatchTransformerPath20Forecaster(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            horizon=horizon,
            patch_sizes=tuple(int(item) for item in (patch_sizes or (4, 20)) if int(item) > 0),
            num_layers=transformer_layers,
            num_heads=heads,
            dropout=dropout,
            output_profile=output_profile,
            cumulative_horizons=self.cumulative_horizons,
        )
        self.dlinear = DLinearPath20Forecaster(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            horizon=horizon,
            output_profile=output_profile,
            cumulative_horizons=self.cumulative_horizons,
        )
        self.gru_proj = nn.Linear(int(hidden_dim) * 2, int(hidden_dim))
        self.dlinear_proj = nn.Linear(int(hidden_dim) * 2, int(hidden_dim))
        self.static_encoder = StaticContextEncoder(
            vocab_sizes=static_context_vocab_sizes,
            embedding_dims=static_context_embedding_dims,
            fields=static_context_fields,
            dropout=static_context_dropout,
        )
        self.static_proj = nn.Sequential(
            nn.LayerNorm(self.static_encoder.output_dim),
            nn.Linear(self.static_encoder.output_dim, int(hidden_dim)),
            nn.GELU(),
            nn.Dropout(float(dropout)),
        )
        self.router = nn.Sequential(
            nn.LayerNorm(int(hidden_dim) * 4),
            nn.Linear(int(hidden_dim) * 4, int(hidden_dim)),
            nn.GELU(),
            nn.Dropout(float(dropout)),
            nn.Linear(int(hidden_dim), len(self.expert_names)),
        )
        fusion_layer = nn.TransformerEncoderLayer(
            d_model=int(hidden_dim),
            nhead=heads,
            dim_feedforward=int(hidden_dim) * 4,
            dropout=float(dropout),
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.fusion_encoder = nn.TransformerEncoder(fusion_layer, num_layers=1, enable_nested_tensor=False)
        self.head = nn.Sequential(
            nn.LayerNorm(int(hidden_dim) * 3),
            nn.Dropout(float(dropout)),
            nn.Linear(int(hidden_dim) * 3, int(hidden_dim)),
            nn.GELU(),
            nn.Dropout(float(dropout)),
            nn.Linear(int(hidden_dim), path20_forecast_output_dim(self.horizon, self.output_profile, self.cumulative_horizons)),
        )

    def _expert_tokens(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim == 2:
            x = x.unsqueeze(1)
        gru_token = self.gru_proj(self.gru_encoder.encode(x))
        patch_token = self.patch_encoder.encode(x)
        trend_input = x.mean(dim=1)
        seasonal_input = x[:, -1, :] - trend_input
        dlinear_token = self.dlinear_proj(torch.cat([F.gelu(self.dlinear.trend(trend_input)), F.gelu(self.dlinear.seasonal(seasonal_input))], dim=-1))
        return torch.stack([gru_token, patch_token, dlinear_token], dim=1)

    def _static_hidden(self, static_context_ids: torch.Tensor | None, *, batch_size: int, device: torch.device) -> torch.Tensor:
        static = self.static_encoder(static_context_ids, batch_size=int(batch_size), device=device)
        return self.static_proj(static)

    def expert_weights(self, x: torch.Tensor, static_context_ids: torch.Tensor | None = None) -> torch.Tensor:
        tokens = self._expert_tokens(x)
        static = self._static_hidden(static_context_ids, batch_size=int(tokens.shape[0]), device=tokens.device)
        pooled = torch.cat([tokens.mean(dim=1), tokens.max(dim=1).values, tokens[:, 0, :], static], dim=-1)
        return torch.softmax(self.router(pooled) / self.router_temperature, dim=-1)

    def forward(self, x: torch.Tensor, static_context_ids: torch.Tensor | None = None) -> dict[str, torch.Tensor]:
        tokens = self._expert_tokens(x)
        static = self._static_hidden(static_context_ids, batch_size=int(tokens.shape[0]), device=tokens.device)
        pooled = torch.cat([tokens.mean(dim=1), tokens.max(dim=1).values, tokens[:, 0, :], static], dim=-1)
        weights = torch.softmax(self.router(pooled) / self.router_temperature, dim=-1)
        weighted = tokens * weights.unsqueeze(-1)
        fused_tokens = self.fusion_encoder(weighted)
        fused = torch.cat([weighted.sum(dim=1), fused_tokens.mean(dim=1), static], dim=-1)
        return _split_path20_outputs(self.head(fused), self.horizon, self.output_profile, self.cumulative_horizons)


def _as_long_index_tensor(values: tuple[int, ...] | list[int] | torch.Tensor | None) -> torch.Tensor:
    if values is None:
        return torch.empty((0,), dtype=torch.long)
    if isinstance(values, torch.Tensor):
        return values.detach().to(dtype=torch.long).reshape(-1).cpu()
    return torch.as_tensor([int(item) for item in values], dtype=torch.long).reshape(-1)


class MultiScaleEWMATrendEncoder(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        *,
        half_lives: tuple[float, ...] | list[float] = (5.0, 10.0, 20.0, 60.0, 120.0),
        dropout: float = 0.10,
    ) -> None:
        super().__init__()
        self.input_dim = int(input_dim)
        self.hidden_dim = int(hidden_dim)
        self.half_lives = tuple(float(item) for item in half_lives if float(item) > 0.0)
        if not self.half_lives:
            raise ValueError("half_lives must contain at least one positive value.")
        self.proj = nn.Sequential(
            nn.LayerNorm(self.input_dim * (len(self.half_lives) + 3)),
            nn.Linear(self.input_dim * (len(self.half_lives) + 3), self.hidden_dim),
            nn.GELU(),
            nn.Dropout(float(dropout)),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.GELU(),
        )

    @staticmethod
    def _ewma_weight_matrix(
        steps: int,
        half_lives: tuple[float, ...],
        *,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        age = torch.arange(int(steps) - 1, -1, -1, device=device, dtype=dtype).reshape(1, -1)
        half_life = torch.as_tensor(tuple(float(item) for item in half_lives), device=device, dtype=dtype).reshape(-1, 1)
        weights = torch.pow(torch.as_tensor(0.5, device=device, dtype=dtype), age / half_life)
        return weights / weights.sum(dim=1, keepdim=True).clamp_min(torch.finfo(dtype).eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim == 2:
            x = x.unsqueeze(1)
        if int(x.shape[-1]) != self.input_dim:
            raise ValueError(f"Expected input_dim={self.input_dim}, got {x.shape[-1]}.")
        steps = int(x.shape[1])
        weights = self._ewma_weight_matrix(steps, self.half_lives, device=x.device, dtype=x.dtype)
        summaries = torch.matmul(x.transpose(1, 2), weights.t()).transpose(1, 2).reshape(x.shape[0], -1)
        last = x[:, -1, :]
        long_mean = x.mean(dim=1)
        recent = x[:, -min(5, steps) :, :].mean(dim=1)
        features = torch.cat([summaries, last - long_mean, recent - long_mean, last], dim=-1)
        return self.proj(features)


class RecencyAwarePatchTransformerPath20Encoder(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        *,
        patch_size: int = 4,
        patch_sizes: tuple[int, ...] | list[int] | None = None,
        num_layers: int = 2,
        num_heads: int = 4,
        dropout: float = 0.10,
        max_patches: int = 512,
        recency_halflife: float = 8.0,
    ) -> None:
        super().__init__()
        self.input_dim = int(input_dim)
        self.hidden_dim = int(hidden_dim)
        self.recency_halflife = max(float(recency_halflife), 1.0e-6)
        resolved_patch_sizes = tuple(int(item) for item in (patch_sizes or (patch_size,)) if int(item) > 0)
        self.patch_sizes = tuple(dict.fromkeys(resolved_patch_sizes or (max(int(patch_size), 1),)))
        self.max_patches = max(int(max_patches), 1)
        self.patch_projs = nn.ModuleList([nn.Linear(self.input_dim * int(size), self.hidden_dim) for size in self.patch_sizes])
        self.position_embeddings = nn.ParameterList(
            [nn.Parameter(torch.zeros(1, self.max_patches, self.hidden_dim)) for _ in self.patch_sizes]
        )
        self.scale_embeddings = nn.Parameter(torch.zeros(1, len(self.patch_sizes), self.hidden_dim))
        self.cls_token = nn.Parameter(torch.zeros(1, 1, self.hidden_dim))
        self.input_dropout = nn.Dropout(float(dropout))
        requested_heads = max(int(num_heads), 1)
        heads = requested_heads if self.hidden_dim % requested_heads == 0 else 1
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.hidden_dim,
            nhead=heads,
            dim_feedforward=self.hidden_dim * 4,
            dropout=float(dropout),
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=max(int(num_layers), 1), enable_nested_tensor=False)
        nn.init.normal_(self.cls_token, std=0.02)
        nn.init.normal_(self.scale_embeddings, std=0.02)
        for embedding in self.position_embeddings:
            nn.init.normal_(embedding, std=0.02)

    def _patch_recency_scale(
        self,
        *,
        patch_count: int,
        patch_size: int,
        total_steps: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        patch_end = torch.arange(1, int(patch_count) + 1, device=device, dtype=dtype) * float(patch_size)
        age = torch.clamp(float(total_steps) - patch_end, min=0.0)
        weights = torch.pow(torch.as_tensor(0.5, device=device, dtype=dtype), age / self.recency_halflife)
        # Keep old patches alive as weak context instead of deleting long history.
        return (0.25 + 0.75 * weights).reshape(1, int(patch_count), 1)

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
            scale = self._patch_recency_scale(
                patch_count=int(projected.shape[1]),
                patch_size=int(patch_size),
                total_steps=int(steps),
                device=x.device,
                dtype=x.dtype,
            )
            projected = projected * scale
            projected = projected + self.position_embeddings[scale_idx][:, : projected.shape[1], :]
            projected = projected + self.scale_embeddings[:, scale_idx : scale_idx + 1, :]
            tokens.append(projected)
        token_sequence = torch.cat(tokens, dim=1)
        cls = self.cls_token.expand(batch, -1, -1)
        encoded = self.encoder(self.input_dropout(torch.cat([cls, token_sequence], dim=1)))
        return encoded[:, 0, :]


STRUCTURED_ALPHA_V2_FEATURE_GROUPS: tuple[str, ...] = (
    "daily_price_volume",
    "cross_section",
    "market_regime",
    "industry_peer",
    "valuation_liquidity",
    "event_quality",
    "intraday",
)


class TemporalConvAlphaEncoder(nn.Module):
    def __init__(
        self,
        hidden_dim: int,
        *,
        dilations: tuple[int, ...] | list[int] = (1, 3, 5),
        dropout: float = 0.10,
    ) -> None:
        super().__init__()
        self.hidden_dim = int(hidden_dim)
        self.dilations = tuple(int(item) for item in dilations if int(item) > 0) or (1,)
        self.blocks = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv1d(self.hidden_dim, self.hidden_dim, kernel_size=3, padding=int(dilation), dilation=int(dilation), groups=1),
                    nn.GELU(),
                    nn.Dropout(float(dropout)),
                    nn.Conv1d(self.hidden_dim, self.hidden_dim, kernel_size=1),
                )
                for dilation in self.dilations
            ]
        )
        self.norm = nn.LayerNorm(self.hidden_dim)
        self.pool = nn.Sequential(
            nn.LayerNorm(self.hidden_dim * 3),
            nn.Linear(self.hidden_dim * 3, self.hidden_dim),
            nn.GELU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim == 2:
            x = x.unsqueeze(1)
        work = x.transpose(1, 2)
        for block in self.blocks:
            work = work + block(work)
        encoded = self.norm(work.transpose(1, 2))
        last = encoded[:, -1, :]
        recent = encoded[:, -min(10, int(encoded.shape[1])) :, :].mean(dim=1)
        long_mean = encoded.mean(dim=1)
        return self.pool(torch.cat([last, recent - long_mean, long_mean], dim=-1))


class StructuredPath20PredictionHead(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        *,
        horizon: int,
        cumulative_horizons: tuple[int, ...],
        output_profile: str,
        dropout: float,
    ) -> None:
        super().__init__()
        self.horizon = int(horizon)
        self.cumulative_horizons = tuple(int(item) for item in cumulative_horizons)
        self.output_profile = normalize_path20_output_profile(output_profile)
        self.hidden_dim = int(hidden_dim)
        self.trunk = nn.Sequential(
            nn.LayerNorm(int(input_dim)),
            nn.Linear(int(input_dim), self.hidden_dim),
            nn.GELU(),
            nn.Dropout(float(dropout)),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.GELU(),
        )
        self.daily_head = nn.Linear(self.hidden_dim, self.horizon)
        self.quantile_head = nn.Linear(self.hidden_dim, self.horizon * 3)
        self.horizon_embeddings = nn.Parameter(torch.zeros(1, len(self.cumulative_horizons), self.hidden_dim))
        self.horizon_context = nn.Sequential(
            nn.LayerNorm(self.hidden_dim),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.GELU(),
            nn.Dropout(float(dropout)),
        )
        self.cum_head = nn.Linear(self.hidden_dim, 1)
        self.risk_head = nn.Linear(self.hidden_dim, 3)
        self.decision_head = (
            nn.Linear(self.hidden_dim, 3)
            if self.output_profile == "decision_utility_v1"
            else None
        )
        nn.init.normal_(self.horizon_embeddings, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        hidden = self.trunk(x)
        daily = self.daily_head(hidden)
        quantile = self.quantile_head(hidden)
        horizon_hidden = self.horizon_context(hidden.unsqueeze(1) + self.horizon_embeddings)
        cum = self.cum_head(horizon_hidden).squeeze(-1)
        risk = self.risk_head(horizon_hidden).reshape(x.shape[0], -1)
        parts = [daily, quantile, torch.cat([cum, risk], dim=-1)]
        if self.decision_head is not None:
            decision = self.decision_head(horizon_hidden)
            parts.append(
                torch.cat(
                    [
                        decision[:, :, 0],
                        decision[:, :, 1],
                        decision[:, :, 2],
                    ],
                    dim=-1,
                )
            )
        return torch.cat(parts, dim=-1)


class HybridStructuredAlphaV2Forecaster(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 192,
        horizon: int = 20,
        dropout: float = 0.15,
        gru_layers: int = 2,
        transformer_layers: int = 4,
        transformer_heads: int = 6,
        patch_sizes: tuple[int, ...] | list[int] | None = None,
        router_temperature: float = 1.0,
        static_context_vocab_sizes: dict[str, int] | None = None,
        static_context_embedding_dims: dict[str, int] | None = None,
        static_context_fields: tuple[str, ...] | list[str] | None = None,
        static_context_dropout: float = 0.20,
        feature_group_indices: dict[str, tuple[int, ...] | list[int] | torch.Tensor] | None = None,
        recency_half_lives: tuple[float, ...] | list[float] = (3.0, 5.0, 10.0, 20.0, 60.0, 120.0),
        patch_recency_halflife: float = 6.0,
        intraday_recency_halflife: float = 3.0,
        output_profile: str = "forecast_path_v1",
        cumulative_horizons: tuple[int, ...] | list[int] | str | None = None,
    ) -> None:
        super().__init__()
        self.horizon = int(horizon)
        self.cumulative_horizons = normalize_path20_cumulative_horizons(cumulative_horizons, horizon=self.horizon)
        self.output_profile = normalize_path20_output_profile(output_profile)
        if self.output_profile != "forecast_path_v1":
            raise ValueError("hybrid_structured_alpha_v2 is prediction-first and only supports forecast_path_v1 outputs.")
        self.input_dim = int(input_dim)
        self.hidden_dim = int(hidden_dim)
        self.router_temperature = max(float(router_temperature), 1.0e-4)
        self.feature_group_names = STRUCTURED_ALPHA_V2_FEATURE_GROUPS
        self.main_feature_group_names = tuple(name for name in self.feature_group_names if name != "intraday")
        self.expert_names = ("gru_continuity", "patch_recency", "multi_ewma", "local_tcn")
        self.recency_half_lives = tuple(float(item) for item in recency_half_lives if float(item) > 0.0)
        self.patch_recency_halflife = float(patch_recency_halflife)
        self.intraday_recency_halflife = float(intraday_recency_halflife)

        static_fields = tuple(str(item).strip() for item in (static_context_fields or ("exchange", "industry")) if str(item).strip())
        if any(field == "symbol" for field in static_fields):
            raise ValueError("hybrid_structured_alpha_v2 does not accept symbol static context; use exchange,industry.")
        self.static_context_fields = static_fields

        provided_groups = dict(feature_group_indices or {})
        used = torch.zeros((self.input_dim,), dtype=torch.bool)
        group_index_tensors: list[torch.Tensor] = []
        for group_name in self.feature_group_names:
            idx = _as_long_index_tensor(provided_groups.get(group_name, ()))
            idx = idx[(idx >= 0) & (idx < self.input_dim)].unique(sorted=True)
            if idx.numel():
                used[idx] = True
            group_index_tensors.append(idx)
        if not any(idx.numel() for idx in group_index_tensors):
            group_index_tensors[0] = torch.arange(self.input_dim, dtype=torch.long)
        elif not bool(used.all()):
            remainder = torch.arange(self.input_dim, dtype=torch.long)[~used]
            group_index_tensors[0] = torch.cat([group_index_tensors[0], remainder]).unique(sorted=True)
        for group_pos, idx in enumerate(group_index_tensors):
            self.register_buffer(f"_feature_group_indices_{group_pos}", idx, persistent=False)

        self.group_encoders = nn.ModuleDict()
        self.feature_group_counts: dict[str, int] = {}
        for group_pos, group_name in enumerate(self.feature_group_names):
            count = int(getattr(self, f"_feature_group_indices_{group_pos}").numel())
            self.feature_group_counts[group_name] = count
            if group_name == "intraday" or count <= 0:
                continue
            self.group_encoders[group_name] = nn.Sequential(
                nn.LayerNorm(count),
                nn.Linear(count, self.hidden_dim),
                nn.GELU(),
                nn.Dropout(float(dropout)),
                nn.Linear(self.hidden_dim, self.hidden_dim),
                nn.GELU(),
            )
        self.group_embeddings = nn.Parameter(torch.zeros(1, 1, len(self.main_feature_group_names), self.hidden_dim))
        requested_heads = max(int(transformer_heads), 1)
        heads = requested_heads if self.hidden_dim % requested_heads == 0 else 1
        self.group_mixer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=self.hidden_dim,
                nhead=heads,
                dim_feedforward=self.hidden_dim * 2,
                dropout=float(dropout),
                activation="gelu",
                batch_first=True,
                norm_first=True,
            ),
            num_layers=1,
            enable_nested_tensor=False,
        )
        self.group_weight_head = nn.Sequential(
            nn.LayerNorm(self.hidden_dim),
            nn.Linear(self.hidden_dim, 1),
        )

        self.static_encoder = StaticContextEncoder(
            vocab_sizes=static_context_vocab_sizes,
            embedding_dims=static_context_embedding_dims,
            fields=self.static_context_fields,
            dropout=static_context_dropout,
        )
        self.static_proj = nn.Sequential(
            nn.LayerNorm(self.static_encoder.output_dim),
            nn.Linear(self.static_encoder.output_dim, self.hidden_dim),
            nn.GELU(),
            nn.Dropout(float(dropout)),
        )
        self.context_builder = nn.Sequential(
            nn.LayerNorm(self.hidden_dim * 3),
            nn.Linear(self.hidden_dim * 3, self.hidden_dim),
            nn.GELU(),
            nn.Dropout(float(dropout)),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.GELU(),
        )
        self.sequence_film = nn.Linear(self.hidden_dim, self.hidden_dim * 2)
        self.expert_film = nn.Linear(self.hidden_dim, len(self.expert_names) * self.hidden_dim * 2)

        self.gru = nn.GRU(
            input_size=self.hidden_dim,
            hidden_size=self.hidden_dim,
            num_layers=max(int(gru_layers), 1),
            batch_first=True,
            dropout=float(dropout) if max(int(gru_layers), 1) > 1 else 0.0,
        )
        self.gru_attention = nn.Sequential(
            nn.LayerNorm(self.hidden_dim),
            nn.Linear(self.hidden_dim, 1),
        )
        self.gru_proj = nn.Sequential(
            nn.LayerNorm(self.hidden_dim * 2),
            nn.Linear(self.hidden_dim * 2, self.hidden_dim),
            nn.GELU(),
        )
        self.patch_encoder = RecencyAwarePatchTransformerPath20Encoder(
            input_dim=self.hidden_dim,
            hidden_dim=self.hidden_dim,
            patch_sizes=tuple(int(item) for item in (patch_sizes or (3, 5, 20)) if int(item) > 0),
            num_layers=max(int(transformer_layers), 1),
            num_heads=heads,
            dropout=dropout,
            recency_halflife=float(patch_recency_halflife),
        )
        self.ewma_encoder = MultiScaleEWMATrendEncoder(
            self.hidden_dim,
            self.hidden_dim,
            half_lives=self.recency_half_lives or (3.0, 5.0, 10.0, 20.0, 60.0, 120.0),
            dropout=dropout,
        )
        self.tcn_encoder = TemporalConvAlphaEncoder(self.hidden_dim, dropout=dropout)
        self.router = nn.Sequential(
            nn.LayerNorm(self.hidden_dim * 7),
            nn.Linear(self.hidden_dim * 7, self.hidden_dim),
            nn.GELU(),
            nn.Dropout(float(dropout)),
            nn.Linear(self.hidden_dim, len(self.expert_names)),
        )
        self.fusion_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=self.hidden_dim,
                nhead=heads,
                dim_feedforward=self.hidden_dim * 4,
                dropout=float(dropout),
                activation="gelu",
                batch_first=True,
                norm_first=True,
            ),
            num_layers=2,
            enable_nested_tensor=False,
        )
        self.base_head = StructuredPath20PredictionHead(
            self.hidden_dim * 4,
            self.hidden_dim,
            horizon=self.horizon,
            cumulative_horizons=self.cumulative_horizons,
            output_profile=self.output_profile,
            dropout=dropout,
        )
        intraday_summary_dim = max(int(self.feature_group_counts.get("intraday", 0)) * 4, 1)
        self.register_buffer(
            "intraday_residual_enabled",
            torch.as_tensor(1.0 if int(self.feature_group_counts.get("intraday", 0)) > 0 else 0.0, dtype=torch.float32),
            persistent=False,
        )
        self.intraday_encoder = nn.Sequential(
            nn.LayerNorm(intraday_summary_dim),
            nn.Linear(intraday_summary_dim, max(self.hidden_dim // 2, 16)),
            nn.GELU(),
            nn.Dropout(float(dropout) + 0.05),
            nn.Linear(max(self.hidden_dim // 2, 16), self.hidden_dim),
            nn.GELU(),
        )
        self.intraday_residual_head = nn.Sequential(
            nn.LayerNorm(self.hidden_dim * 2),
            nn.Linear(self.hidden_dim * 2, self.hidden_dim),
            nn.GELU(),
            nn.Dropout(float(dropout)),
            nn.Linear(self.hidden_dim, path20_forecast_output_dim(self.horizon, self.output_profile, self.cumulative_horizons)),
        )
        self.register_buffer(
            "intraday_residual_mask",
            self._build_intraday_residual_mask(
                self.horizon,
                self.cumulative_horizons,
                self.output_profile,
            ),
            persistent=False,
        )
        nn.init.normal_(self.group_embeddings, std=0.02)

    @staticmethod
    def _build_intraday_residual_mask(
        horizon: int,
        cumulative_horizons: tuple[int, ...],
        output_profile: str,
    ) -> torch.Tensor:
        horizon = int(horizon)
        days = torch.arange(1, horizon + 1, dtype=torch.float32)
        daily = torch.pow(torch.as_tensor(0.5, dtype=torch.float32), (days - 1.0) / 5.0)
        horizon_values = torch.as_tensor(tuple(int(item) for item in cumulative_horizons), dtype=torch.float32)
        aux_decay = torch.pow(torch.as_tensor(0.5, dtype=torch.float32), torch.clamp(horizon_values - 1.0, min=0.0) / 8.0)
        parts = [daily, daily.repeat(3), aux_decay, aux_decay.repeat_interleave(3)]
        if normalize_path20_output_profile(output_profile) == "decision_utility_v1":
            parts.append(aux_decay.repeat(3))
        return torch.cat(parts, dim=0).reshape(1, -1)

    def _indices_for_group(self, group_name: str) -> torch.Tensor:
        group_pos = self.feature_group_names.index(group_name)
        return getattr(self, f"_feature_group_indices_{group_pos}")

    def _select_group(self, x: torch.Tensor, group_name: str) -> torch.Tensor | None:
        idx = self._indices_for_group(group_name)
        if idx.numel() == 0:
            return None
        return torch.index_select(x, dim=-1, index=idx.to(device=x.device))

    def _group_sequence(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if x.ndim == 2:
            x = x.unsqueeze(1)
        batch, steps, _ = x.shape
        tokens: list[torch.Tensor] = []
        for group_name in self.main_feature_group_names:
            values = self._select_group(x, group_name)
            if values is None or group_name not in self.group_encoders:
                token = torch.zeros((batch, steps, self.hidden_dim), device=x.device, dtype=x.dtype)
            else:
                token = self.group_encoders[group_name](values)
            tokens.append(token)
        group_tokens = torch.stack(tokens, dim=2) + self.group_embeddings.to(device=x.device, dtype=x.dtype)
        mixed = self.group_mixer(group_tokens.reshape(batch * steps, len(self.main_feature_group_names), self.hidden_dim))
        mixed = mixed.reshape(batch, steps, len(self.main_feature_group_names), self.hidden_dim)
        logits = self.group_weight_head(mixed).squeeze(-1)
        weights = torch.softmax(logits, dim=-1)
        sequence = torch.sum(mixed * weights.unsqueeze(-1), dim=2)
        return sequence, mixed.mean(dim=1), weights.mean(dim=1)

    def _static_hidden(self, static_context_ids: torch.Tensor | None, *, batch_size: int, device: torch.device) -> torch.Tensor:
        static = self.static_encoder(static_context_ids, batch_size=int(batch_size), device=device)
        return self.static_proj(static)

    def _context_token(self, sequence: torch.Tensor, group_summary: torch.Tensor, static: torch.Tensor) -> torch.Tensor:
        context_names = ("market_regime", "industry_peer", "event_quality")
        context_positions = [self.main_feature_group_names.index(name) for name in context_names if name in self.main_feature_group_names]
        if context_positions:
            context_group = group_summary[:, context_positions, :].mean(dim=1)
        else:
            context_group = group_summary.mean(dim=1)
        recent = sequence[:, -min(10, int(sequence.shape[1])) :, :].mean(dim=1)
        return self.context_builder(torch.cat([static, context_group, recent], dim=-1))

    def _condition_sequence(self, sequence: torch.Tensor, context: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        gamma, beta = self.sequence_film(context).chunk(2, dim=-1)
        gamma = 0.25 * torch.tanh(gamma)
        beta = 0.25 * torch.tanh(beta)
        conditioned = sequence * (1.0 + gamma.unsqueeze(1)) + beta.unsqueeze(1)
        return conditioned, gamma, beta

    def _gru_token(self, sequence: torch.Tensor) -> torch.Tensor:
        outputs, hidden = self.gru(sequence)
        attention_logits = self.gru_attention(outputs).squeeze(-1)
        attention = torch.softmax(attention_logits, dim=1).unsqueeze(-1)
        pooled = torch.sum(outputs * attention, dim=1)
        return self.gru_proj(torch.cat([pooled, hidden[-1]], dim=-1))

    def _intraday_token(self, x: torch.Tensor, *, batch_size: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        intraday = self._select_group(x, "intraday")
        if intraday is None:
            return torch.zeros((batch_size, self.hidden_dim), device=device, dtype=dtype)
        steps = int(intraday.shape[1])
        age = torch.arange(steps - 1, -1, -1, device=device, dtype=dtype)
        weights = torch.pow(torch.as_tensor(0.5, device=device, dtype=dtype), age / max(self.intraday_recency_halflife, 1.0e-6))
        weights = weights / weights.sum().clamp_min(torch.finfo(dtype).eps)
        recent = torch.sum(intraday * weights.reshape(1, -1, 1), dim=1)
        last = intraday[:, -1, :]
        mean = intraday.mean(dim=1)
        volatility = intraday.std(dim=1, unbiased=False)
        summary = torch.cat([recent, last - mean, volatility, last], dim=-1)
        return self.intraday_encoder(summary)

    def _expert_tokens(
        self,
        sequence: torch.Tensor,
        context: torch.Tensor,
    ) -> torch.Tensor:
        tokens = torch.stack(
            [
                self._gru_token(sequence),
                self.patch_encoder.encode(sequence),
                self.ewma_encoder(sequence),
                self.tcn_encoder(sequence),
            ],
            dim=1,
        )
        film = self.expert_film(context).reshape(context.shape[0], len(self.expert_names), 2, self.hidden_dim)
        gamma = 0.25 * torch.tanh(film[:, :, 0, :])
        beta = 0.25 * torch.tanh(film[:, :, 1, :])
        return tokens * (1.0 + gamma) + beta

    def _router_input(self, tokens: torch.Tensor, context: torch.Tensor, group_summary: torch.Tensor) -> torch.Tensor:
        return torch.cat(
            [
                tokens.mean(dim=1),
                tokens.max(dim=1).values,
                tokens.std(dim=1, unbiased=False),
                tokens[:, 0, :],
                tokens[:, 1, :],
                context,
                group_summary.mean(dim=1),
            ],
            dim=-1,
        )

    @staticmethod
    def _expert_diversity(tokens: torch.Tensor) -> torch.Tensor:
        if int(tokens.shape[1]) < 2:
            return tokens.new_zeros((tokens.shape[0],))
        normed = F.normalize(tokens, dim=-1)
        similarity = torch.matmul(normed, normed.transpose(1, 2))
        mask = ~torch.eye(int(tokens.shape[1]), dtype=torch.bool, device=tokens.device).unsqueeze(0)
        distance = 1.0 - similarity.masked_select(mask).reshape(tokens.shape[0], -1)
        return distance.mean(dim=1)

    def expert_weights(self, x: torch.Tensor, static_context_ids: torch.Tensor | None = None) -> torch.Tensor:
        if x.ndim == 2:
            x = x.unsqueeze(1)
        sequence, group_summary, _ = self._group_sequence(x)
        static = self._static_hidden(static_context_ids, batch_size=int(sequence.shape[0]), device=sequence.device)
        context = self._context_token(sequence, group_summary, static)
        sequence, _, _ = self._condition_sequence(sequence, context)
        tokens = self._expert_tokens(sequence, context)
        return torch.softmax(self.router(self._router_input(tokens, context, group_summary)) / self.router_temperature, dim=-1)

    def forward(self, x: torch.Tensor, static_context_ids: torch.Tensor | None = None) -> dict[str, torch.Tensor]:
        if x.ndim == 2:
            x = x.unsqueeze(1)
        sequence, group_summary, group_weights = self._group_sequence(x)
        static = self._static_hidden(static_context_ids, batch_size=int(sequence.shape[0]), device=sequence.device)
        context = self._context_token(sequence, group_summary, static)
        conditioned_sequence, context_gamma, _ = self._condition_sequence(sequence, context)
        tokens = self._expert_tokens(conditioned_sequence, context)
        weights = torch.softmax(self.router(self._router_input(tokens, context, group_summary)) / self.router_temperature, dim=-1)
        weighted = tokens * weights.unsqueeze(-1)
        fused_tokens = self.fusion_encoder(weighted)
        fused = torch.cat([weighted.sum(dim=1), fused_tokens.mean(dim=1), context, group_summary.mean(dim=1)], dim=-1)
        raw = self.base_head(fused)
        intraday_token = self._intraday_token(
            x,
            batch_size=int(x.shape[0]),
            device=x.device,
            dtype=x.dtype,
        )
        residual_raw = self.intraday_residual_head(torch.cat([intraday_token, context], dim=-1))
        effective_residual = (
            residual_raw
            * self.intraday_residual_mask.to(device=x.device, dtype=x.dtype)
            * self.intraday_residual_enabled.to(device=x.device, dtype=x.dtype)
        )
        raw = raw + effective_residual
        output = _split_path20_outputs(raw, self.horizon, self.output_profile, self.cumulative_horizons)
        entropy = -(weights * torch.log(torch.clamp(weights, min=1.0e-8))).sum(dim=-1)
        output.update(
            {
                "router_weights": weights,
                "router_entropy": entropy,
                "expert_token_diversity": self._expert_diversity(tokens),
                "feature_group_weights": group_weights,
                "context_gate_abs_mean": context_gamma.abs().mean(dim=-1),
                "intraday_residual_norm": effective_residual.norm(dim=-1) / effective_residual.shape[-1] ** 0.5,
            }
        )
        return output


class HybridMultiScaleRecencyAwarePath20Forecaster(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 256,
        horizon: int = 20,
        dropout: float = 0.15,
        gru_layers: int = 2,
        transformer_layers: int = 4,
        transformer_heads: int = 8,
        patch_sizes: tuple[int, ...] | list[int] | None = None,
        router_temperature: float = 1.0,
        static_context_vocab_sizes: dict[str, int] | None = None,
        static_context_embedding_dims: dict[str, int] | None = None,
        static_context_fields: tuple[str, ...] | list[str] | None = None,
        static_context_dropout: float = 0.20,
        intraday_feature_indices: tuple[int, ...] | list[int] | torch.Tensor | None = None,
        recency_half_lives: tuple[float, ...] | list[float] = (5.0, 10.0, 20.0, 60.0, 120.0),
        patch_recency_halflife: float = 8.0,
        intraday_recency_halflife: float = 5.0,
        intraday_bottleneck_dim: int = 32,
        output_profile: str = "forecast_path_v1",
        cumulative_horizons: tuple[int, ...] | list[int] | str | None = None,
    ) -> None:
        super().__init__()
        self.horizon = int(horizon)
        self.cumulative_horizons = normalize_path20_cumulative_horizons(cumulative_horizons, horizon=self.horizon)
        self.output_profile = normalize_path20_output_profile(output_profile)
        self.input_dim = int(input_dim)
        self.hidden_dim = int(hidden_dim)
        self.router_temperature = max(float(router_temperature), 1.0e-4)
        intraday_idx = _as_long_index_tensor(intraday_feature_indices)
        intraday_idx = intraday_idx[(intraday_idx >= 0) & (intraday_idx < self.input_dim)].unique(sorted=True)
        all_idx = torch.arange(self.input_dim, dtype=torch.long)
        if intraday_idx.numel():
            main_mask = torch.ones((self.input_dim,), dtype=torch.bool)
            main_mask[intraday_idx] = False
            main_idx = all_idx[main_mask]
        else:
            main_idx = all_idx
        if main_idx.numel() == 0:
            main_idx = all_idx
            intraday_idx = torch.empty((0,), dtype=torch.long)
        self.register_buffer("main_feature_indices", main_idx, persistent=False)
        self.register_buffer("intraday_feature_indices", intraday_idx, persistent=False)
        self.main_input_dim = int(main_idx.numel())
        self.intraday_input_dim = int(intraday_idx.numel())
        self.expert_names = ("gru_main", "patch_recency", "multi_ewma", "intraday_bottleneck")
        self.recency_half_lives = tuple(float(item) for item in recency_half_lives)
        self.patch_recency_halflife = float(patch_recency_halflife)
        self.intraday_recency_halflife = float(intraday_recency_halflife)

        self.main_norm = nn.LayerNorm(self.main_input_dim)
        self.gru_encoder = GRUPath20Forecaster(
            input_dim=self.main_input_dim,
            hidden_dim=hidden_dim,
            dropout=dropout,
            horizon=horizon,
            num_layers=gru_layers,
            output_profile=output_profile,
            cumulative_horizons=self.cumulative_horizons,
        )
        requested_heads = max(int(transformer_heads), 1)
        heads = requested_heads if int(hidden_dim) % requested_heads == 0 else 1
        self.patch_encoder = RecencyAwarePatchTransformerPath20Encoder(
            input_dim=self.main_input_dim,
            hidden_dim=hidden_dim,
            patch_sizes=tuple(int(item) for item in (patch_sizes or (4, 20)) if int(item) > 0),
            num_layers=transformer_layers,
            num_heads=heads,
            dropout=dropout,
            recency_halflife=float(patch_recency_halflife),
        )
        self.ewma_encoder = MultiScaleEWMATrendEncoder(
            self.main_input_dim,
            hidden_dim,
            half_lives=tuple(float(item) for item in recency_half_lives),
            dropout=dropout,
        )
        bottleneck_dim = max(int(intraday_bottleneck_dim), 1)
        intraday_summary_dim = max(self.intraday_input_dim * 4, 1)
        self.intraday_bottleneck = nn.Sequential(
            nn.LayerNorm(intraday_summary_dim),
            nn.Linear(intraday_summary_dim, bottleneck_dim),
            nn.GELU(),
            nn.Dropout(float(dropout) + 0.05),
            nn.Linear(bottleneck_dim, hidden_dim),
            nn.GELU(),
        )
        self.gru_proj = nn.Linear(int(hidden_dim) * 2, int(hidden_dim))
        self.static_encoder = StaticContextEncoder(
            vocab_sizes=static_context_vocab_sizes,
            embedding_dims=static_context_embedding_dims,
            fields=static_context_fields,
            dropout=static_context_dropout,
        )
        self.static_proj = nn.Sequential(
            nn.LayerNorm(self.static_encoder.output_dim),
            nn.Linear(self.static_encoder.output_dim, int(hidden_dim)),
            nn.GELU(),
            nn.Dropout(float(dropout)),
        )
        router_input_dim = int(hidden_dim) * 6
        self.router = nn.Sequential(
            nn.LayerNorm(router_input_dim),
            nn.Linear(router_input_dim, int(hidden_dim)),
            nn.GELU(),
            nn.Dropout(float(dropout)),
            nn.Linear(int(hidden_dim), len(self.expert_names)),
        )
        fusion_layer = nn.TransformerEncoderLayer(
            d_model=int(hidden_dim),
            nhead=heads,
            dim_feedforward=int(hidden_dim) * 4,
            dropout=float(dropout),
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.fusion_encoder = nn.TransformerEncoder(fusion_layer, num_layers=1, enable_nested_tensor=False)
        self.head = nn.Sequential(
            nn.LayerNorm(int(hidden_dim) * 3),
            nn.Dropout(float(dropout)),
            nn.Linear(int(hidden_dim) * 3, int(hidden_dim)),
            nn.GELU(),
            nn.Dropout(float(dropout)),
            nn.Linear(int(hidden_dim), path20_forecast_output_dim(self.horizon, self.output_profile, self.cumulative_horizons)),
        )

    def _split_inputs(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor | None]:
        if x.ndim == 2:
            x = x.unsqueeze(1)
        main = torch.index_select(x, dim=-1, index=self.main_feature_indices.to(device=x.device))
        main = self.main_norm(main)
        if self.intraday_feature_indices.numel() == 0:
            return main, None
        intraday = torch.index_select(x, dim=-1, index=self.intraday_feature_indices.to(device=x.device))
        return main, intraday

    def _intraday_token(self, intraday: torch.Tensor | None, *, batch_size: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        if intraday is None:
            return torch.zeros((batch_size, self.hidden_dim), device=device, dtype=dtype)
        steps = int(intraday.shape[1])
        age = torch.arange(steps - 1, -1, -1, device=device, dtype=dtype)
        weights = torch.pow(torch.as_tensor(0.5, device=device, dtype=dtype), age / max(self.intraday_recency_halflife, 1.0e-6))
        weights = weights / weights.sum().clamp_min(torch.finfo(dtype).eps)
        recent = torch.sum(intraday * weights.reshape(1, -1, 1), dim=1)
        last = intraday[:, -1, :]
        mean = intraday.mean(dim=1)
        volatility = intraday.std(dim=1, unbiased=False)
        summary = torch.cat([recent, last - mean, volatility, last], dim=-1)
        return self.intraday_bottleneck(summary)

    def _static_hidden(self, static_context_ids: torch.Tensor | None, *, batch_size: int, device: torch.device) -> torch.Tensor:
        static = self.static_encoder(static_context_ids, batch_size=int(batch_size), device=device)
        return self.static_proj(static)

    def _expert_tokens(self, x: torch.Tensor) -> torch.Tensor:
        main, intraday = self._split_inputs(x)
        gru_token = self.gru_proj(self.gru_encoder.encode(main))
        patch_token = self.patch_encoder.encode(main)
        ewma_token = self.ewma_encoder(main)
        intraday_token = self._intraday_token(
            intraday,
            batch_size=int(main.shape[0]),
            device=main.device,
            dtype=main.dtype,
        )
        return torch.stack([gru_token, patch_token, ewma_token, intraday_token], dim=1)

    def _router_input(self, tokens: torch.Tensor, static: torch.Tensor) -> torch.Tensor:
        return torch.cat(
            [
                tokens.mean(dim=1),
                tokens.max(dim=1).values,
                tokens.std(dim=1, unbiased=False),
                tokens[:, 0, :],
                tokens[:, 1, :],
                static,
            ],
            dim=-1,
        )

    def expert_weights(self, x: torch.Tensor, static_context_ids: torch.Tensor | None = None) -> torch.Tensor:
        tokens = self._expert_tokens(x)
        static = self._static_hidden(static_context_ids, batch_size=int(tokens.shape[0]), device=tokens.device)
        return torch.softmax(self.router(self._router_input(tokens, static)) / self.router_temperature, dim=-1)

    def forward(self, x: torch.Tensor, static_context_ids: torch.Tensor | None = None) -> dict[str, torch.Tensor]:
        tokens = self._expert_tokens(x)
        static = self._static_hidden(static_context_ids, batch_size=int(tokens.shape[0]), device=tokens.device)
        weights = torch.softmax(self.router(self._router_input(tokens, static)) / self.router_temperature, dim=-1)
        weighted = tokens * weights.unsqueeze(-1)
        fused_tokens = self.fusion_encoder(weighted)
        fused = torch.cat([weighted.sum(dim=1), fused_tokens.mean(dim=1), static], dim=-1)
        output = _split_path20_outputs(self.head(fused), self.horizon, self.output_profile, self.cumulative_horizons)
        entropy = -(weights * torch.log(torch.clamp(weights, min=1.0e-8))).sum(dim=-1)
        output.update(
            {
                "router_weights": weights,
                "router_entropy": entropy,
            }
        )
        return output


class RegimeRoutedMultiExpertHorizonForecaster(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 384,
        horizon: int = 20,
        dropout: float = 0.15,
        gru_layers: int = 2,
        transformer_layers: int = 4,
        transformer_heads: int = 8,
        patch_sizes: tuple[int, ...] | list[int] | None = None,
        fusion_layers: int = 2,
        router_temperature: float = 1.0,
        static_context_vocab_sizes: dict[str, int] | None = None,
        static_context_embedding_dims: dict[str, int] | None = None,
        static_context_fields: tuple[str, ...] | list[str] | None = None,
        static_context_dropout: float = 0.20,
        output_profile: str = "forecast_path_v1",
        cumulative_horizons: tuple[int, ...] | list[int] | str | None = None,
    ) -> None:
        super().__init__()
        self.horizon = int(horizon)
        self.cumulative_horizons = normalize_path20_cumulative_horizons(cumulative_horizons, horizon=self.horizon)
        self.output_profile = normalize_path20_output_profile(output_profile)
        self.input_dim = int(input_dim)
        self.hidden_dim = int(hidden_dim)
        self.router_temperature = max(float(router_temperature), 1.0e-4)
        self.expert_names = ("gru", "patch_transformer", "dlinear", "stock_mixer", "local_state")
        self.input_projection = nn.Sequential(
            nn.LayerNorm(self.input_dim),
            nn.Linear(self.input_dim, self.hidden_dim),
            nn.GELU(),
            nn.Dropout(float(dropout)),
        )
        self.gru_encoder = GRUPath20Forecaster(
            input_dim=self.hidden_dim,
            hidden_dim=self.hidden_dim,
            dropout=dropout,
            horizon=horizon,
            num_layers=gru_layers,
            output_profile=output_profile,
            cumulative_horizons=self.cumulative_horizons,
        )
        requested_heads = max(int(transformer_heads), 1)
        heads = requested_heads if self.hidden_dim % requested_heads == 0 else 1
        self.patch_encoder = PatchTransformerPath20Forecaster(
            input_dim=self.hidden_dim,
            hidden_dim=self.hidden_dim,
            horizon=horizon,
            patch_sizes=tuple(int(item) for item in (patch_sizes or (4, 10, 20)) if int(item) > 0),
            num_layers=transformer_layers,
            num_heads=heads,
            dropout=dropout,
            output_profile=output_profile,
            cumulative_horizons=self.cumulative_horizons,
        )
        self.dlinear_trend = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.dlinear_seasonal = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.gru_proj = nn.Linear(self.hidden_dim * 2, self.hidden_dim)
        self.dlinear_proj = nn.Linear(self.hidden_dim * 2, self.hidden_dim)
        self.stock_attention = nn.MultiheadAttention(
            embed_dim=self.hidden_dim,
            num_heads=heads,
            dropout=float(dropout),
            batch_first=True,
        )
        self.stock_proj = nn.Sequential(
            nn.LayerNorm(self.hidden_dim * 2),
            nn.Linear(self.hidden_dim * 2, self.hidden_dim),
            nn.GELU(),
            nn.Dropout(float(dropout)),
        )
        self.local_state_proj = nn.Sequential(
            nn.LayerNorm(self.input_dim * 6),
            nn.Linear(self.input_dim * 6, self.hidden_dim),
            nn.GELU(),
            nn.Dropout(float(dropout)),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.GELU(),
        )
        self.static_encoder = StaticContextEncoder(
            vocab_sizes=static_context_vocab_sizes,
            embedding_dims=static_context_embedding_dims,
            fields=static_context_fields,
            dropout=static_context_dropout,
        )
        self.static_proj = nn.Sequential(
            nn.LayerNorm(self.static_encoder.output_dim),
            nn.Linear(self.static_encoder.output_dim, self.hidden_dim),
            nn.GELU(),
            nn.Dropout(float(dropout)),
        )
        router_input_dim = self.hidden_dim * 8
        self.router = nn.Sequential(
            nn.LayerNorm(router_input_dim),
            nn.Linear(router_input_dim, self.hidden_dim),
            nn.GELU(),
            nn.Dropout(float(dropout)),
            nn.Linear(self.hidden_dim, max(self.hidden_dim // 2, len(self.expert_names))),
            nn.GELU(),
            nn.Linear(max(self.hidden_dim // 2, len(self.expert_names)), len(self.expert_names)),
        )
        fusion_layer = nn.TransformerEncoderLayer(
            d_model=self.hidden_dim,
            nhead=heads,
            dim_feedforward=self.hidden_dim * 4,
            dropout=float(dropout),
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.fusion_encoder = nn.TransformerEncoder(
            fusion_layer,
            num_layers=max(int(fusion_layers), 1),
            enable_nested_tensor=False,
        )
        self.head = nn.Sequential(
            nn.LayerNorm(self.hidden_dim * 4),
            nn.Dropout(float(dropout)),
            nn.Linear(self.hidden_dim * 4, self.hidden_dim * 2),
            nn.GELU(),
            nn.Dropout(float(dropout)),
            nn.Linear(self.hidden_dim * 2, self.hidden_dim),
            nn.GELU(),
            nn.Dropout(float(dropout)),
            nn.Linear(self.hidden_dim, path20_forecast_output_dim(self.horizon, self.output_profile, self.cumulative_horizons)),
        )

    def _flatten_inputs(
        self,
        x: torch.Tensor,
        static_context_ids: torch.Tensor | None,
        stock_mask: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor | None, tuple[int, ...] | None]:
        if x.ndim == 2:
            x = x.unsqueeze(1)
        if x.ndim == 3:
            return x, static_context_ids, stock_mask, None
        if x.ndim != 4:
            raise ValueError("x must have shape [batch, features], [batch, steps, features], or [dates, stocks, steps, features].")
        dates, stocks, steps, features = x.shape
        flat_static = static_context_ids
        if static_context_ids is not None:
            if static_context_ids.ndim != 3:
                raise ValueError("static_context_ids must have shape [dates, stocks, fields] for date-level batches.")
            flat_static = static_context_ids.reshape(dates * stocks, static_context_ids.shape[-1])
        return x.reshape(dates * stocks, steps, features), flat_static, stock_mask, (dates, stocks)

    def _local_state_features(self, raw_x: torch.Tensor) -> torch.Tensor:
        if raw_x.ndim == 2:
            raw_x = raw_x.unsqueeze(1)
        last = raw_x[:, -1, :]
        mean = raw_x.mean(dim=1)
        delta = last - raw_x[:, 0, :]
        recent_window = raw_x[:, -min(int(raw_x.shape[1]), 5) :, :]
        recent = recent_window.mean(dim=1) - mean
        volatility = raw_x.std(dim=1, unbiased=False)
        drawdown = last - raw_x.max(dim=1).values
        return torch.cat([last, mean, delta, recent, volatility, drawdown], dim=-1)

    def _static_hidden(self, static_context_ids: torch.Tensor | None, *, batch_size: int, device: torch.device) -> torch.Tensor:
        static = self.static_encoder(static_context_ids, batch_size=int(batch_size), device=device)
        return self.static_proj(static)

    def _stock_token(self, flat_temporal: torch.Tensor, stock_mask: torch.Tensor | None, batch_shape: tuple[int, ...] | None) -> torch.Tensor:
        if batch_shape is None:
            tokens = flat_temporal.unsqueeze(0)
            attended, _ = self.stock_attention(tokens, tokens, tokens, need_weights=False)
            return self.stock_proj(torch.cat([flat_temporal, attended.squeeze(0)], dim=-1))
        dates, stocks = int(batch_shape[0]), int(batch_shape[1])
        tokens = flat_temporal.reshape(dates, stocks, self.hidden_dim)
        key_padding_mask = None
        if stock_mask is not None:
            key_padding_mask = ~stock_mask.reshape(dates, stocks).to(device=flat_temporal.device, dtype=torch.bool)
        attended, _ = self.stock_attention(tokens, tokens, tokens, key_padding_mask=key_padding_mask, need_weights=False)
        return self.stock_proj(torch.cat([tokens, attended], dim=-1)).reshape(dates * stocks, self.hidden_dim)

    def _expert_tokens(
        self,
        x: torch.Tensor,
        static_context_ids: torch.Tensor | None = None,
        stock_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        flat_x, flat_static_ids, flat_stock_mask, batch_shape = self._flatten_inputs(x, static_context_ids, stock_mask)
        projected = self.input_projection(flat_x)
        gru_token = self.gru_proj(self.gru_encoder.encode(projected))
        patch_token = self.patch_encoder.encode(projected)
        trend_input = projected.mean(dim=1)
        seasonal_input = projected[:, -1, :] - trend_input
        dlinear_token = self.dlinear_proj(
            torch.cat([F.gelu(self.dlinear_trend(trend_input)), F.gelu(self.dlinear_seasonal(seasonal_input))], dim=-1)
        )
        stock_token = self._stock_token(projected[:, -1, :], flat_stock_mask, batch_shape)
        local_token = self.local_state_proj(self._local_state_features(flat_x))
        static = self._static_hidden(flat_static_ids, batch_size=int(flat_x.shape[0]), device=flat_x.device)
        tokens = torch.stack([gru_token, patch_token, dlinear_token, stock_token, local_token], dim=1)
        return tokens, static, self._local_state_features(flat_x)

    def _router_input(self, tokens: torch.Tensor, static: torch.Tensor, local_state: torch.Tensor) -> torch.Tensor:
        local_summary = local_state.reshape(local_state.shape[0], 6, self.input_dim).mean(dim=-1)
        local_summary = F.pad(local_summary, (0, max(self.hidden_dim - local_summary.shape[1], 0)))[:, : self.hidden_dim]
        return torch.cat(
            [
                tokens.mean(dim=1),
                tokens.max(dim=1).values,
                tokens.std(dim=1, unbiased=False),
                static,
                tokens[:, 0, :],
                tokens[:, 1, :],
                tokens[:, 4, :],
                local_summary,
            ],
            dim=-1,
        )

    def expert_weights(
        self,
        x: torch.Tensor,
        static_context_ids: torch.Tensor | None = None,
        stock_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        tokens, static, local_state = self._expert_tokens(x, static_context_ids=static_context_ids, stock_mask=stock_mask)
        return torch.softmax(self.router(self._router_input(tokens, static, local_state)) / self.router_temperature, dim=-1)

    def forward(
        self,
        x: torch.Tensor,
        static_context_ids: torch.Tensor | None = None,
        stock_mask: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        tokens, static, local_state = self._expert_tokens(x, static_context_ids=static_context_ids, stock_mask=stock_mask)
        router_logits = self.router(self._router_input(tokens, static, local_state)) / self.router_temperature
        weights = torch.softmax(router_logits, dim=-1)
        weighted = tokens * weights.unsqueeze(-1)
        fused_tokens = self.fusion_encoder(torch.cat([weighted, static.unsqueeze(1)], dim=1))
        fused = torch.cat([weighted.sum(dim=1), fused_tokens.mean(dim=1), static, tokens[:, 4, :]], dim=-1)
        output = _split_path20_outputs(self.head(fused), self.horizon, self.output_profile, self.cumulative_horizons)
        entropy = -(weights * torch.log(torch.clamp(weights, min=1.0e-8))).sum(dim=-1)
        centered = tokens - tokens.mean(dim=1, keepdim=True)
        diversity = centered.square().mean(dim=(1, 2))
        local_view = local_state.reshape(local_state.shape[0], 6, self.input_dim)
        bad_state_intensity = torch.sigmoid(local_view[:, 4, :].mean(dim=-1) - local_view[:, 5, :].mean(dim=-1))
        output.update(
            {
                "router_weights": weights,
                "router_entropy": entropy,
                "expert_token_diversity": diversity,
                "bad_state_intensity": bad_state_intensity,
            }
        )
        return output


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
