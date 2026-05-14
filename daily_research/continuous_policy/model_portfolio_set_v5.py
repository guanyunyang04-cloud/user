from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from daily_research.continuous_policy.core_v4_release_targets import build_core_v4_release_first_targets
from daily_research.continuous_policy.model_seq_v3 import SEQUENCE_STEP_ORDER, resolve_sequence_columns
from daily_research.continuous_policy.model_v2 import _apply_matrix, _prepare_matrix, _split_indices
from daily_research.continuous_policy.semantic_budget_intent import derive_release_first_intent
from daily_research.continuous_policy.training_contracts import TRAINER_BACKEND_FORMAL_PORTFOLIO_SET_V5
from daily_research.continuous_policy.training_runtime_acceleration import (
    autocast_context,
    configure_torch_training_acceleration,
    move_to_device,
)


PORTFOLIO_SET_V5_ARTIFACT_TYPE = "continuous_policy_torch_portfolio_set_v5"
PORTFOLIO_SET_V5_ARTIFACT_FILENAME = "continuous_policy_portfolio_set_v5_artifact.pt"
PORTFOLIO_SET_V5_DEFAULT_STRICT_GOLD_DATASET_ID = "continuous_policy_training_matrices__strict_train__36c234208d5f375ea1cccfc1"

PORTFOLIO_SET_V5_OUTPUT_NAMES: tuple[str, ...] = (
    "target_weight",
    "target_delta",
    "source_supply_score",
    "receiver_demand_score",
    "cash_buffer_score",
    "release_intent",
    "reduce_quality",
    "exit_hazard",
)
PORTFOLIO_SET_V5_LOSS_ALIASES: tuple[str, ...] = (
    "alpha_result_value_budget_split_v48",
    "portfolio_set_release_first_decision_v1",
)
PORTFOLIO_SET_V5_LOSS_PROFILE_NAMES: tuple[str, ...] = PORTFOLIO_SET_V5_LOSS_ALIASES
PORTFOLIO_SET_V5_MAX_TRAIN_DAYS = 256
PORTFOLIO_SET_V5_MAX_STOCKS_PER_DAY = 3070


def resolve_portfolio_set_v5_loss_profile(profile_name: str | None) -> tuple[str, dict[str, dict[str, float]]]:
    name = str(profile_name or "alpha_result_value_budget_split_v48").strip() or "alpha_result_value_budget_split_v48"
    if name not in PORTFOLIO_SET_V5_LOSS_ALIASES:
        raise ValueError(
            f"Unsupported portfolio-set v5 loss profile: {profile_name!r}. "
            "Only alpha_result_value_budget_split_v48 / portfolio_set_release_first_decision_v1 are supported."
        )
    return "alpha_result_value_budget_split_v48", {
        "multi_objective_loss_weights": {
            "action_total": 0.0,
            "duration_total": 0.0,
            "target_weight_closure_total": 1.0,
            "source_supply_total": 0.52,
            "receiver_demand_total": 0.52,
            "cash_buffer_total": 0.20,
            "target_delta_weight_coherence_total": 0.34,
            "release_flow_balance_total": 0.48,
            "underdeployment_high_cash_total": 0.36,
            "intent_translation_conflict_total": 0.34,
        }
    }


def _numeric_series(frame: pd.DataFrame, name: str, default: float = 0.0) -> pd.Series:
    if name not in frame.columns:
        return pd.Series(float(default), index=frame.index, dtype=float)
    return pd.to_numeric(frame[name], errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(float(default))


def _current_weight(frame: pd.DataFrame) -> pd.Series:
    for name in ("current_weight", "position_weight", "portfolio_weight", "weight"):
        if name in frame.columns:
            return _numeric_series(frame, name, 0.0).clip(0.0, 1.0)
    return pd.Series(0.0, index=frame.index, dtype=float)


def _ensure_features(frame: pd.DataFrame, feature_names: list[str]) -> pd.DataFrame:
    result = frame.copy()
    for name in feature_names:
        if name not in result.columns:
            result[name] = np.nan
    return result


def _date_column(frame: pd.DataFrame) -> str:
    if "date" in frame.columns:
        return "date"
    if "trade_date" in frame.columns:
        return "trade_date"
    raise ValueError("portfolio-set v5 requires a date or trade_date column.")


def build_portfolio_set_v5_targets(sample_frame: pd.DataFrame, *, deadband: float = 0.003) -> pd.DataFrame:
    targets = build_core_v4_release_first_targets(sample_frame, deadband=deadband)
    current = _current_weight(sample_frame)
    target_delta = pd.to_numeric(targets["target_delta"], errors="coerce").fillna(0.0)
    receiver_support = pd.to_numeric(targets["receiver_support"], errors="coerce").fillna(0.0).clip(0.0, 1.0)
    receiver_score = pd.to_numeric(targets["receiver_score"], errors="coerce").fillna(0.0).clip(0.0, 1.0)
    release_intent = pd.to_numeric(targets["release_intent"], errors="coerce").fillna(0.0).clip(0.0, 1.0)
    source_score = pd.to_numeric(targets["source_score"], errors="coerce").fillna(0.0).clip(0.0, 1.0)
    held = current > float(deadband)
    receiver = (target_delta > float(deadband)) & (receiver_support > 0.0)
    source = (target_delta < -float(deadband)) & held & (release_intent > 0.0)
    enriched = pd.DataFrame(index=sample_frame.index)
    enriched["target_weight"] = pd.to_numeric(targets["target_weight"], errors="coerce").fillna(current).clip(0.0, 0.24)
    enriched["target_delta"] = (enriched["target_weight"] - current).where(lambda s: s.abs() >= float(deadband), 0.0)
    enriched["source_supply_score"] = (release_intent * source_score).where(source, 0.0).clip(0.0, 1.0)
    enriched["receiver_demand_score"] = (receiver_support * receiver_score).where(receiver, 0.0).clip(0.0, 1.0)
    day_col = _date_column(sample_frame)
    day_receiver = enriched["receiver_demand_score"].groupby(sample_frame[day_col].astype(str)).transform("sum")
    day_source = enriched["source_supply_score"].groupby(sample_frame[day_col].astype(str)).transform("sum")
    defensive = _numeric_series(sample_frame, "market_downside_pressure", 0.0).clip(0.0, 1.0)
    enriched["cash_buffer_score"] = ((day_source - day_receiver).clip(lower=0.0) + defensive * 0.30).clip(0.0, 1.0)
    enriched["release_intent"] = release_intent.where(source, 0.0).clip(0.0, 1.0)
    enriched["reduce_quality"] = pd.to_numeric(targets["reduce_quality"], errors="coerce").fillna(0.0).clip(0.0, 1.0)
    enriched["exit_hazard"] = pd.to_numeric(targets["exit_hazard"], errors="coerce").fillna(0.0).clip(0.0, 1.0)
    enriched["held_mask"] = held.astype(float)
    enriched["source_mask"] = source.astype(float)
    enriched["receiver_mask"] = receiver.astype(float)
    enriched["target_delta_weight_conflict"] = (
        ((enriched["target_weight"] - current) * enriched["target_delta"] < -(float(deadband) ** 2))
    ).astype(float)
    return enriched.astype(float)


def _resolve_static_and_sequence_columns(feature_names: list[str]) -> tuple[list[str], list[str], list[str]]:
    try:
        sequence_bases, sequence_columns = resolve_sequence_columns(feature_names)
    except ValueError:
        return list(feature_names), [], []
    static_columns = [name for name in feature_names if name not in set(sequence_columns)]
    if not static_columns:
        static_columns = list(feature_names)
        sequence_bases = []
        sequence_columns = []
    return static_columns, sequence_bases, sequence_columns


def _sequence_column_name(base_name: str, step: int) -> str:
    return str(base_name) if int(step) == 0 else f"{base_name}_lag{int(step)}"


def _build_sequence_array(frame: pd.DataFrame, sequence_bases: list[str], *, fill: np.ndarray, means: np.ndarray, stds: np.ndarray) -> np.ndarray:
    if not sequence_bases:
        return np.zeros((len(frame), 1, 1), dtype=np.float32)
    steps = list(SEQUENCE_STEP_ORDER)
    matrices: list[np.ndarray] = []
    offset = 0
    for step in steps:
        cols = [_sequence_column_name(base, step) for base in sequence_bases]
        raw = frame.reindex(columns=cols).replace([np.inf, -np.inf], np.nan).to_numpy(dtype=np.float32)
        step_fill = fill[offset : offset + len(cols)]
        step_means = means[offset : offset + len(cols)]
        step_stds = stds[offset : offset + len(cols)]
        raw = np.where(np.isfinite(raw), raw, step_fill)
        raw = (raw - step_means) / step_stds
        matrices.append(raw.astype(np.float32))
        offset += len(cols)
    return np.stack(matrices, axis=1).astype(np.float32)


def _daily_lookup(daily_frame: pd.DataFrame) -> dict[str, pd.Series]:
    day_col = _date_column(daily_frame)
    return {str(row[day_col]): row for _, row in daily_frame.iterrows()}


class PortfolioSetDayDataset(Dataset[dict[str, np.ndarray]]):
    def __init__(
        self,
        *,
        sample_frame: pd.DataFrame,
        daily_frame: pd.DataFrame,
        static_feature_names: list[str],
        sequence_bases: list[str],
        daily_feature_names: list[str],
        targets: pd.DataFrame,
        static_fill: np.ndarray,
        static_means: np.ndarray,
        static_stds: np.ndarray,
        sequence_fill: np.ndarray,
        sequence_means: np.ndarray,
        sequence_stds: np.ndarray,
        daily_fill: np.ndarray,
        daily_means: np.ndarray,
        daily_stds: np.ndarray,
        max_stocks_per_day: int = PORTFOLIO_SET_V5_MAX_STOCKS_PER_DAY,
    ) -> None:
        self.sample_frame = sample_frame.copy()
        self.targets = targets.copy()
        self.daily_by_date = _daily_lookup(daily_frame)
        self.static_feature_names = list(static_feature_names)
        self.sequence_bases = list(sequence_bases)
        self.daily_feature_names = list(daily_feature_names)
        self.static_fill = np.asarray(static_fill, dtype=np.float32)
        self.static_means = np.asarray(static_means, dtype=np.float32)
        self.static_stds = np.asarray(static_stds, dtype=np.float32)
        self.sequence_fill = np.asarray(sequence_fill, dtype=np.float32)
        self.sequence_means = np.asarray(sequence_means, dtype=np.float32)
        self.sequence_stds = np.asarray(sequence_stds, dtype=np.float32)
        self.daily_fill = np.asarray(daily_fill, dtype=np.float32)
        self.daily_means = np.asarray(daily_means, dtype=np.float32)
        self.daily_stds = np.asarray(daily_stds, dtype=np.float32)
        self.max_stocks_per_day = max(1, int(max_stocks_per_day or 1))
        date_col = _date_column(self.sample_frame)
        self.date_col = date_col
        self.dates = sorted(str(item) for item in self.sample_frame[date_col].astype(str).unique())

    def __len__(self) -> int:
        return len(self.dates)

    def __getitem__(self, index: int) -> dict[str, np.ndarray]:
        date_text = self.dates[int(index)]
        day = self.sample_frame[self.sample_frame[self.date_col].astype(str) == date_text].copy()
        if len(day) > self.max_stocks_per_day:
            day = day.head(self.max_stocks_per_day).copy()
        target = self.targets.loc[day.index]
        static = day.reindex(columns=self.static_feature_names).replace([np.inf, -np.inf], np.nan).to_numpy(dtype=np.float32)
        static = np.where(np.isfinite(static), static, self.static_fill)
        static = ((static - self.static_means) / self.static_stds).astype(np.float32)
        sequence = _build_sequence_array(
            day,
            self.sequence_bases,
            fill=self.sequence_fill,
            means=self.sequence_means,
            stds=self.sequence_stds,
        )
        daily_row = self.daily_by_date.get(date_text)
        if daily_row is None:
            daily_raw = np.full(len(self.daily_feature_names), np.nan, dtype=np.float32)
        else:
            daily_raw = pd.to_numeric(daily_row.reindex(self.daily_feature_names), errors="coerce").to_numpy(dtype=np.float32)
        daily_raw = np.where(np.isfinite(daily_raw), daily_raw, self.daily_fill)
        daily = ((daily_raw - self.daily_means) / self.daily_stds).astype(np.float32)
        return {
            "static_x": static,
            "sequence_x": sequence,
            "daily_x": daily,
            "current_weight": _current_weight(day).to_numpy(dtype=np.float32),
            "target_y": target.reindex(columns=PORTFOLIO_SET_V5_OUTPUT_NAMES).to_numpy(dtype=np.float32),
            "held_mask": target["held_mask"].to_numpy(dtype=np.float32),
            "source_mask": target["source_mask"].to_numpy(dtype=np.float32),
            "receiver_mask": target["receiver_mask"].to_numpy(dtype=np.float32),
        }


def collate_portfolio_set_days(batch: list[dict[str, np.ndarray]]) -> dict[str, torch.Tensor]:
    max_items = max(int(item["static_x"].shape[0]) for item in batch)
    batch_size = len(batch)
    static_dim = int(batch[0]["static_x"].shape[1])
    sequence_steps = int(batch[0]["sequence_x"].shape[1])
    sequence_dim = int(batch[0]["sequence_x"].shape[2])
    output_dim = len(PORTFOLIO_SET_V5_OUTPUT_NAMES)
    static_x = np.zeros((batch_size, max_items, static_dim), dtype=np.float32)
    sequence_x = np.zeros((batch_size, max_items, sequence_steps, sequence_dim), dtype=np.float32)
    target_y = np.zeros((batch_size, max_items, output_dim), dtype=np.float32)
    current_weight = np.zeros((batch_size, max_items), dtype=np.float32)
    held_mask = np.zeros((batch_size, max_items), dtype=np.float32)
    source_mask = np.zeros((batch_size, max_items), dtype=np.float32)
    receiver_mask = np.zeros((batch_size, max_items), dtype=np.float32)
    sample_mask = np.zeros((batch_size, max_items), dtype=bool)
    daily_x = np.stack([item["daily_x"] for item in batch]).astype(np.float32)
    for batch_idx, item in enumerate(batch):
        size = int(item["static_x"].shape[0])
        static_x[batch_idx, :size] = item["static_x"]
        sequence_x[batch_idx, :size] = item["sequence_x"]
        target_y[batch_idx, :size] = item["target_y"]
        current_weight[batch_idx, :size] = item["current_weight"]
        held_mask[batch_idx, :size] = item["held_mask"]
        source_mask[batch_idx, :size] = item["source_mask"]
        receiver_mask[batch_idx, :size] = item["receiver_mask"]
        sample_mask[batch_idx, :size] = True
    return {
        "static_x": torch.as_tensor(static_x, dtype=torch.float32),
        "sequence_x": torch.as_tensor(sequence_x, dtype=torch.float32),
        "daily_x": torch.as_tensor(daily_x, dtype=torch.float32),
        "target_y": torch.as_tensor(target_y, dtype=torch.float32),
        "current_weight": torch.as_tensor(current_weight, dtype=torch.float32),
        "held_mask": torch.as_tensor(held_mask, dtype=torch.float32),
        "source_mask": torch.as_tensor(source_mask, dtype=torch.float32),
        "receiver_mask": torch.as_tensor(receiver_mask, dtype=torch.float32),
        "sample_mask": torch.as_tensor(sample_mask, dtype=torch.bool),
    }


class TemporalEncoder(nn.Module):
    def __init__(self, *, input_dim: int, model_dim: int, layers: int, dropout: float) -> None:
        super().__init__()
        self.input_dim = int(input_dim)
        self.model_dim = int(model_dim)
        self.input_proj = nn.Linear(int(input_dim), int(model_dim))
        self.gru = nn.GRU(
            input_size=int(model_dim),
            hidden_size=int(model_dim),
            num_layers=max(1, int(layers)),
            batch_first=True,
            dropout=float(dropout) if int(layers) > 1 else 0.0,
        )
        self.norm = nn.LayerNorm(int(model_dim))

    def forward(self, sequence_x: torch.Tensor) -> torch.Tensor:
        bsz, count, steps, dim = sequence_x.shape
        encoded = self.input_proj(sequence_x.reshape(bsz * count, steps, dim))
        _, hidden = self.gru(encoded)
        return self.norm(hidden[-1].reshape(bsz, count, self.model_dim))


class LatentSetEncoder(nn.Module):
    def __init__(self, *, model_dim: int, latent_count: int, heads: int, layers: int, dropout: float) -> None:
        super().__init__()
        self.latents = nn.Parameter(torch.randn(1, max(1, int(latent_count)), int(model_dim)) * 0.02)
        self.cross_attn = nn.MultiheadAttention(int(model_dim), max(1, int(heads)), batch_first=True, dropout=float(dropout))
        self.self_layers = nn.ModuleList(
            [
                nn.TransformerEncoderLayer(
                    d_model=int(model_dim),
                    nhead=max(1, int(heads)),
                    dim_feedforward=int(model_dim) * 4,
                    dropout=float(dropout),
                    batch_first=True,
                    activation="gelu",
                    norm_first=True,
                )
                for _ in range(max(1, int(layers)))
            ]
        )
        self.norm = nn.LayerNorm(int(model_dim))

    def forward(self, tokens: torch.Tensor, sample_mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        batch_size = int(tokens.shape[0])
        latents = self.latents.expand(batch_size, -1, -1)
        key_padding_mask = ~sample_mask.bool()
        latents, _ = self.cross_attn(latents, tokens, tokens, key_padding_mask=key_padding_mask)
        for layer in self.self_layers:
            latents = layer(latents)
        context = self.norm(latents.mean(dim=1))
        return latents, context


class PortfolioSetPolicyNetV5(nn.Module):
    def __init__(
        self,
        *,
        static_input_dim: int,
        sequence_input_dim: int,
        daily_input_dim: int,
        model_dim: int = 192,
        temporal_layers: int = 2,
        cross_layers: int = 2,
        latent_count: int = 64,
        dropout: float = 0.18,
    ) -> None:
        super().__init__()
        self.model_dim = int(model_dim)
        self.latent_count = int(latent_count)
        self.static_encoder = nn.Sequential(
            nn.Linear(int(static_input_dim), int(model_dim)),
            nn.GELU(),
            nn.Dropout(float(dropout)),
            nn.Linear(int(model_dim), int(model_dim)),
            nn.GELU(),
        )
        self.temporal_encoder = TemporalEncoder(
            input_dim=int(sequence_input_dim),
            model_dim=int(model_dim),
            layers=int(temporal_layers),
            dropout=float(dropout),
        )
        self.daily_encoder = nn.Sequential(
            nn.Linear(int(daily_input_dim), int(model_dim)),
            nn.GELU(),
            nn.Dropout(float(dropout)),
            nn.Linear(int(model_dim), int(model_dim)),
            nn.GELU(),
        )
        self.stock_fusion = nn.Sequential(
            nn.Linear(int(model_dim) * 2, int(model_dim)),
            nn.GELU(),
            nn.Dropout(float(dropout)),
        )
        self.set_encoder = LatentSetEncoder(
            model_dim=int(model_dim),
            latent_count=int(latent_count),
            heads=4,
            layers=int(cross_layers),
            dropout=float(dropout),
        )
        self.stock_context = nn.Sequential(
            nn.Linear(int(model_dim) * 3, int(model_dim)),
            nn.GELU(),
            nn.Dropout(float(dropout)),
            nn.Linear(int(model_dim), int(model_dim)),
            nn.GELU(),
        )
        self.head = nn.Linear(int(model_dim), len(PORTFOLIO_SET_V5_OUTPUT_NAMES))

    def forward(
        self,
        static_x: torch.Tensor,
        sequence_x: torch.Tensor,
        daily_x: torch.Tensor,
        sample_mask: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        static_hidden = self.static_encoder(static_x)
        temporal_hidden = self.temporal_encoder(sequence_x)
        stock_hidden = self.stock_fusion(torch.cat([static_hidden, temporal_hidden], dim=-1))
        _, set_context = self.set_encoder(stock_hidden, sample_mask)
        daily_context = self.daily_encoder(daily_x)
        repeated_set = set_context.unsqueeze(1).expand(-1, stock_hidden.shape[1], -1)
        repeated_daily = daily_context.unsqueeze(1).expand(-1, stock_hidden.shape[1], -1)
        stock_context = self.stock_context(torch.cat([stock_hidden, repeated_set, repeated_daily], dim=-1))
        return {"raw": self.head(stock_context)}


@dataclass
class TorchPortfolioSetV5Artifact:
    feature_names: list[str]
    daily_feature_names: list[str]
    static_feature_names: list[str]
    sequence_bases: list[str]
    feature_fill_values: np.ndarray
    feature_means: np.ndarray
    feature_stds: np.ndarray
    sequence_fill_values: np.ndarray
    sequence_means: np.ndarray
    sequence_stds: np.ndarray
    daily_fill_values: np.ndarray
    daily_means: np.ndarray
    daily_stds: np.ndarray
    train_summary: dict[str, Any]
    training_diagnostics: dict[str, Any]
    training_contract: dict[str, Any]
    trained_at: str
    model_config: dict[str, Any] = field(default_factory=dict)
    model_state_dict: dict[str, Any] | None = None
    global_target_defaults: dict[str, float] = field(default_factory=dict)

    def save(self, path: Path) -> Path:
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "artifact_type": PORTFOLIO_SET_V5_ARTIFACT_TYPE,
            "feature_names": list(self.feature_names),
            "daily_feature_names": list(self.daily_feature_names),
            "static_feature_names": list(self.static_feature_names),
            "sequence_bases": list(self.sequence_bases),
            "feature_fill_values": self.feature_fill_values.astype(np.float32).tolist(),
            "feature_means": self.feature_means.astype(np.float32).tolist(),
            "feature_stds": self.feature_stds.astype(np.float32).tolist(),
            "sequence_fill_values": self.sequence_fill_values.astype(np.float32).tolist(),
            "sequence_means": self.sequence_means.astype(np.float32).tolist(),
            "sequence_stds": self.sequence_stds.astype(np.float32).tolist(),
            "daily_fill_values": self.daily_fill_values.astype(np.float32).tolist(),
            "daily_means": self.daily_means.astype(np.float32).tolist(),
            "daily_stds": self.daily_stds.astype(np.float32).tolist(),
            "train_summary": self.train_summary,
            "training_diagnostics": self.training_diagnostics,
            "training_contract": self.training_contract,
            "trained_at": self.trained_at,
            "model_config": self.model_config,
            "model_state_dict": self.model_state_dict,
            "global_target_defaults": self.global_target_defaults,
        }
        torch.save(payload, path)
        return path


def load_torch_portfolio_set_v5_artifact(path: str | Path) -> TorchPortfolioSetV5Artifact:
    payload = torch.load(Path(path), map_location="cpu", weights_only=False)
    if str(payload.get("artifact_type", "") or "") != PORTFOLIO_SET_V5_ARTIFACT_TYPE:
        raise TypeError(f"Unsupported portfolio-set v5 artifact type: {payload.get('artifact_type')!r}")
    return TorchPortfolioSetV5Artifact(
        feature_names=list(payload.get("feature_names", []) or []),
        daily_feature_names=list(payload.get("daily_feature_names", []) or []),
        static_feature_names=list(payload.get("static_feature_names", []) or []),
        sequence_bases=list(payload.get("sequence_bases", []) or []),
        feature_fill_values=np.asarray(payload.get("feature_fill_values", []), dtype=np.float32),
        feature_means=np.asarray(payload.get("feature_means", []), dtype=np.float32),
        feature_stds=np.asarray(payload.get("feature_stds", []), dtype=np.float32),
        sequence_fill_values=np.asarray(payload.get("sequence_fill_values", []), dtype=np.float32),
        sequence_means=np.asarray(payload.get("sequence_means", []), dtype=np.float32),
        sequence_stds=np.asarray(payload.get("sequence_stds", []), dtype=np.float32),
        daily_fill_values=np.asarray(payload.get("daily_fill_values", []), dtype=np.float32),
        daily_means=np.asarray(payload.get("daily_means", []), dtype=np.float32),
        daily_stds=np.asarray(payload.get("daily_stds", []), dtype=np.float32),
        train_summary=dict(payload.get("train_summary", {}) or {}),
        training_diagnostics=dict(payload.get("training_diagnostics", {}) or {}),
        training_contract=dict(payload.get("training_contract", {}) or {}),
        trained_at=str(payload.get("trained_at", "") or ""),
        model_config=dict(payload.get("model_config", {}) or {}),
        model_state_dict=payload.get("model_state_dict"),
        global_target_defaults=dict(payload.get("global_target_defaults", {}) or {}),
    )


def _decode_raw(raw: torch.Tensor) -> dict[str, torch.Tensor]:
    return {
        "target_weight": torch.sigmoid(raw[..., 0]) * 0.24,
        "target_delta": torch.tanh(raw[..., 1]) * 0.18,
        "source_supply_score": torch.sigmoid(raw[..., 2]),
        "receiver_demand_score": torch.sigmoid(raw[..., 3]),
        "cash_buffer_score": torch.sigmoid(raw[..., 4]),
        "release_intent": torch.sigmoid(raw[..., 5]),
        "reduce_quality": torch.sigmoid(raw[..., 6]),
        "exit_hazard": torch.sigmoid(raw[..., 7]),
    }


def _masked_mean(value: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    weight = mask.to(dtype=value.dtype)
    while weight.ndim < value.ndim:
        weight = weight.unsqueeze(-1)
    return (value * weight).sum() / weight.sum().clamp_min(1.0)


def _portfolio_set_loss(raw: torch.Tensor, batch: dict[str, torch.Tensor], weights: dict[str, float]) -> torch.Tensor:
    pred = _decode_raw(raw)
    target = batch["target_y"]
    mask = batch["sample_mask"].float()
    current = batch["current_weight"].clamp(0.0, 1.0)
    target_weight = target[..., 0].clamp(0.0, 0.24)
    target_delta = target[..., 1].clamp(-0.18, 0.18)
    source_target = target[..., 2].clamp(0.0, 1.0)
    receiver_target = target[..., 3].clamp(0.0, 1.0)
    cash_target = target[..., 4].clamp(0.0, 1.0)
    release_target = target[..., 5].clamp(0.0, 1.0)
    reduce_target = target[..., 6].clamp(0.0, 1.0)
    exit_target = target[..., 7].clamp(0.0, 1.0)
    source_supply = pred["source_supply_score"] * pred["release_intent"] * (current > 0.003).float()
    receiver_demand = pred["receiver_demand_score"] * (current < 0.24 - 0.003).float()
    day_source = (source_supply * mask).sum(dim=1)
    day_receiver = (receiver_demand * mask).sum(dim=1)
    target_day_source = (source_target * release_target * mask).sum(dim=1)
    target_day_receiver = (receiver_target * mask).sum(dim=1)
    pred_delta_from_weight = pred["target_weight"] - current
    conflict = torch.relu(-(pred_delta_from_weight * pred["target_delta"]) - (0.003 ** 2))
    loss = (
        float(weights["target_weight_closure_total"]) * _masked_mean(nn.functional.smooth_l1_loss(pred["target_weight"], target_weight, reduction="none"), mask)
        + float(weights["source_supply_total"]) * _masked_mean(nn.functional.binary_cross_entropy_with_logits(raw[..., 2], source_target, reduction="none"), mask)
        + float(weights["receiver_demand_total"]) * _masked_mean(nn.functional.binary_cross_entropy_with_logits(raw[..., 3], receiver_target, reduction="none"), mask)
        + float(weights["cash_buffer_total"]) * _masked_mean(nn.functional.smooth_l1_loss(pred["cash_buffer_score"], cash_target, reduction="none"), mask)
        + float(weights["target_delta_weight_coherence_total"]) * _masked_mean(nn.functional.smooth_l1_loss(pred["target_delta"], target_delta, reduction="none") + conflict, mask)
        + float(weights["release_flow_balance_total"])
        * (
            nn.functional.smooth_l1_loss(day_source, target_day_source)
            + nn.functional.smooth_l1_loss(day_receiver, target_day_receiver)
            + torch.relu(day_source - day_receiver - pred["cash_buffer_score"].mean(dim=1) - 0.10).mean()
        )
        + float(weights["underdeployment_high_cash_total"]) * torch.relu(target_weight.sum(dim=1) - pred["target_weight"].sum(dim=1)).mean()
        + float(weights["intent_translation_conflict_total"]) * _masked_mean(conflict, mask)
        + 0.10 * _masked_mean(nn.functional.binary_cross_entropy_with_logits(raw[..., 5], release_target, reduction="none"), mask)
        + 0.08 * _masked_mean(nn.functional.binary_cross_entropy_with_logits(raw[..., 6], reduce_target, reduction="none"), mask)
        + 0.08 * _masked_mean(nn.functional.binary_cross_entropy_with_logits(raw[..., 7], exit_target, reduction="none"), mask)
    )
    return loss


def _global_defaults(daily_frame: pd.DataFrame) -> dict[str, float]:
    defaults: dict[str, float] = {
        "release_first_allocation_v3_mode": 1.0,
        "allocation_intent_v2_mode": 1.0,
        "gross_exposure_target": 0.72,
        "candidate_budget": 8.0,
        "turnover_budget": 0.24,
        "max_position_weight_target": 0.24,
    }
    for key in ("gross_exposure_target", "candidate_budget", "turnover_budget", "max_position_weight_target"):
        if key in daily_frame.columns and len(daily_frame):
            defaults[key] = float(pd.to_numeric(daily_frame[key], errors="coerce").replace([np.inf, -np.inf], np.nan).dropna().median())
    return defaults


def _select_train_days(sample_frame: pd.DataFrame, random_seed: int, max_days: int = PORTFOLIO_SET_V5_MAX_TRAIN_DAYS) -> tuple[pd.DataFrame, dict[str, int]]:
    date_col = _date_column(sample_frame)
    dates = sorted(str(item) for item in sample_frame[date_col].astype(str).unique())
    raw_day_count = len(dates)
    if raw_day_count <= int(max_days):
        return sample_frame.copy(), {"portfolio_set_v5_raw_train_day_count": raw_day_count, "portfolio_set_v5_train_day_count": raw_day_count}
    rng = np.random.default_rng(int(random_seed))
    selected = sorted(rng.choice(np.asarray(dates, dtype=object), size=int(max_days), replace=False).tolist())
    return sample_frame[sample_frame[date_col].astype(str).isin(selected)].copy(), {
        "portfolio_set_v5_raw_train_day_count": raw_day_count,
        "portfolio_set_v5_train_day_count": len(selected),
        "portfolio_set_v5_train_day_cap": int(max_days),
    }


def _make_model(artifact: TorchPortfolioSetV5Artifact) -> PortfolioSetPolicyNetV5:
    cfg = dict(artifact.model_config or {})
    model = PortfolioSetPolicyNetV5(
        static_input_dim=len(artifact.static_feature_names),
        sequence_input_dim=max(len(artifact.sequence_bases), 1),
        daily_input_dim=max(len(artifact.daily_feature_names), 1),
        model_dim=int(cfg.get("model_dim", 192)),
        temporal_layers=int(cfg.get("temporal_layers", 2)),
        cross_layers=int(cfg.get("cross_layers", 2)),
        latent_count=int(cfg.get("latent_count", 64)),
        dropout=float(cfg.get("dropout", 0.18)),
    )
    if artifact.model_state_dict:
        model.load_state_dict(artifact.model_state_dict, strict=True)
    return model


def _predict_outputs(artifact: TorchPortfolioSetV5Artifact, state_frame: pd.DataFrame, daily_features: dict[str, float]) -> pd.DataFrame:
    frame = _ensure_features(state_frame.copy(), artifact.feature_names)
    static_frame = _ensure_features(frame, artifact.static_feature_names)
    static_x = _apply_matrix(static_frame, artifact.static_feature_names, artifact.feature_fill_values, artifact.feature_means, artifact.feature_stds)
    sequence_x = _build_sequence_array(
        frame,
        artifact.sequence_bases,
        fill=artifact.sequence_fill_values,
        means=artifact.sequence_means,
        stds=artifact.sequence_stds,
    )
    daily_raw = np.asarray([float(daily_features.get(name, 0.0) or 0.0) for name in artifact.daily_feature_names], dtype=np.float32)
    if len(daily_raw) == 0:
        daily_raw = np.zeros(1, dtype=np.float32)
    daily_x = np.where(np.isfinite(daily_raw), daily_raw, artifact.daily_fill_values[: len(daily_raw)])
    daily_x = ((daily_x - artifact.daily_means[: len(daily_x)]) / artifact.daily_stds[: len(daily_x)]).astype(np.float32)
    model = _make_model(artifact).eval()
    with torch.no_grad():
        raw = model(
            torch.as_tensor(static_x[None, :, :], dtype=torch.float32),
            torch.as_tensor(sequence_x[None, :, :, :], dtype=torch.float32),
            torch.as_tensor(daily_x[None, :], dtype=torch.float32),
            torch.ones((1, len(frame)), dtype=torch.bool),
        )["raw"][0]
        decoded = _decode_raw(raw)
    return pd.DataFrame({name: tensor.detach().cpu().numpy().astype(float) for name, tensor in decoded.items()}, index=state_frame.index)


def predict_policy_portfolio_set_v5(
    artifact: TorchPortfolioSetV5Artifact,
    *,
    state_frame: pd.DataFrame,
    daily_features: dict[str, float],
) -> tuple[pd.DataFrame, dict[str, float]]:
    if state_frame.empty:
        raise ValueError("state_frame is empty.")
    policy = state_frame.copy()
    outputs = _predict_outputs(artifact, state_frame, daily_features)
    current = _current_weight(policy)
    source_score = outputs["source_supply_score"].clip(0.0, 1.0)
    receiver_score = outputs["receiver_demand_score"].clip(0.0, 1.0)
    raw_target_weight = outputs["target_weight"].clip(0.0, 0.24)
    raw_delta = outputs["target_delta"].clip(-0.18, 0.18)
    release_candidate = (current > 0.003) & ((raw_delta < -0.003) | (source_score >= 0.30))
    receiver_candidate = (current < 0.24 - 0.003) & ((raw_delta > 0.003) | (receiver_score >= 0.30))
    target_weight = raw_target_weight.copy()
    target_weight = target_weight.where(~release_candidate, np.minimum(target_weight, (current - raw_delta.abs().clip(lower=0.01)).clip(lower=0.0)))
    target_weight = target_weight.where(~receiver_candidate, np.maximum(target_weight, (current + raw_delta.clip(lower=0.01)).clip(upper=0.24)))
    target_delta = (target_weight - current).clip(-0.18, 0.18)
    target_delta = target_delta.where(target_delta.abs() >= 0.003, 0.0)
    target_weight = (current + target_delta).clip(0.0, 0.24)
    intent_frame = policy.copy()
    intent_frame["portfolio_daily_target_delta_intent"] = target_delta.astype(float)
    intent_frame["portfolio_daily_source_score"] = source_score.astype(float)
    intent_frame["portfolio_daily_source_release_quality"] = outputs["release_intent"].clip(0.0, 1.0).astype(float)
    intent_frame["portfolio_daily_source_economic_block_risk"] = 0.0
    release_first = derive_release_first_intent(intent_frame)
    release_intent = pd.to_numeric(release_first["release_first_intent_score"], errors="coerce").fillna(0.0).clip(0.0, 1.0)
    policy["portfolio_daily_target_weight_intent"] = target_weight.astype(float)
    policy["portfolio_daily_target_delta_intent"] = target_delta.astype(float)
    policy["portfolio_daily_release_first_intent"] = release_intent.astype(float)
    policy["release_first_action_hint"] = release_first["release_first_action_hint"].astype(str)
    policy["release_first_block_reason"] = release_first["release_first_block_reason"].astype(str)
    policy["portfolio_daily_source_score"] = source_score.astype(float)
    policy["portfolio_daily_unified_source_score"] = source_score.astype(float)
    policy["portfolio_daily_source_executable_candidate"] = ((current > 0.003) & (release_intent >= 0.30)).astype(float)
    policy["portfolio_daily_receiver_score"] = receiver_score.astype(float)
    policy["portfolio_daily_unified_receiver_score"] = receiver_score.astype(float)
    policy["portfolio_daily_receiver_executable_candidate"] = ((current < 0.24 - 0.003) & (receiver_score >= 0.30) & (target_delta > 0.003)).astype(float)
    policy["portfolio_set_v5_source_supply_score"] = source_score.astype(float)
    policy["portfolio_set_v5_receiver_demand_score"] = receiver_score.astype(float)
    policy["portfolio_set_v5_cash_buffer_score"] = outputs["cash_buffer_score"].clip(0.0, 1.0).astype(float)
    policy["portfolio_set_v5_target_delta_weight_conflict_count"] = int((((target_weight - current) * target_delta) < -(0.003 ** 2)).sum())
    add_mask = (current > 0.0) & (target_delta > 0.003)
    open_mask = (current <= 0.0) & (target_delta > 0.003)
    reduce_mask = (current > 0.0) & (target_delta < -0.003) & (target_weight > 0.003)
    exit_mask = (current > 0.0) & (target_delta < -0.003) & (target_weight <= 0.003)
    policy["action_label"] = "hold"
    policy.loc[open_mask, "action_label"] = "open"
    policy.loc[add_mask, "action_label"] = "add"
    policy.loc[reduce_mask, "action_label"] = "reduce"
    policy.loc[exit_mask, "action_label"] = "exit"
    global_targets = dict(artifact.global_target_defaults or {})
    global_targets.update(
        {
            "release_first_allocation_v3_mode": 1.0,
            "allocation_intent_v2_mode": 1.0,
            "target_weight_intent_mode": 1.0,
        }
    )
    return policy, {key: float(value) for key, value in global_targets.items()}


def fit_policy_models_portfolio_set_v5(
    *,
    sample_frame: pd.DataFrame,
    daily_frame: pd.DataFrame,
    feature_names: list[str],
    daily_feature_names: list[str],
    run_root: Path,
    random_seed: int = 7,
    train_summary: dict[str, Any] | None = None,
    trained_at: str = "",
    training_contract: dict[str, Any] | None = None,
    epochs: int = 12,
    min_epochs: int = 8,
    batch_size: int = 1,
    learning_rate: float = 5.0e-5,
    model_dim: int = 192,
    temporal_layers: int = 2,
    cross_layers: int = 2,
    latent_count: int = 64,
    dropout: float = 0.18,
    early_stop_patience: int = 10,
    resume_mode: str = "strict",
    loss_profile: str = "alpha_result_value_budget_split_v48",
    progress_sink: Any | None = None,
) -> TorchPortfolioSetV5Artifact:
    if sample_frame.empty or daily_frame.empty:
        raise ValueError("formal_torch_portfolio_set_v5 received empty training data.")
    contract = dict(training_contract or {})
    if str(contract.get("trainer_backend", "") or "") != TRAINER_BACKEND_FORMAL_PORTFOLIO_SET_V5:
        raise ValueError("fit_policy_models_portfolio_set_v5 requires the formal_torch_portfolio_set_v5 training contract.")
    resolved_loss_profile, loss_config = resolve_portfolio_set_v5_loss_profile(loss_profile)
    if not torch.cuda.is_available() and bool(contract.get("gpu_required", False)):
        raise RuntimeError("continuous_policy formal_torch_portfolio_set_v5 requires CUDA, but torch.cuda.is_available() is False.")
    device = torch.device("cuda" if bool(contract.get("gpu_required", False)) else "cpu")
    runtime = configure_torch_training_acceleration(device, cvxpy_layers_enabled=False)
    torch.manual_seed(int(random_seed))
    np.random.seed(int(random_seed))
    run_root.mkdir(parents=True, exist_ok=True)
    train_frame, day_diagnostics = _select_train_days(sample_frame, int(random_seed))
    targets = build_portfolio_set_v5_targets(train_frame)
    static_feature_names, sequence_bases, sequence_columns = _resolve_static_and_sequence_columns(feature_names)
    static_matrix, static_fill, static_means, static_stds = _prepare_matrix(_ensure_features(train_frame, static_feature_names), static_feature_names)
    if sequence_columns:
        _, sequence_fill, sequence_means, sequence_stds = _prepare_matrix(_ensure_features(train_frame, sequence_columns), sequence_columns)
    else:
        sequence_fill = np.zeros(1, dtype=np.float32)
        sequence_means = np.zeros(1, dtype=np.float32)
        sequence_stds = np.ones(1, dtype=np.float32)
    _, daily_fill, daily_means, daily_stds = _prepare_matrix(_ensure_features(daily_frame, daily_feature_names), daily_feature_names)
    dataset = PortfolioSetDayDataset(
        sample_frame=train_frame,
        daily_frame=daily_frame,
        static_feature_names=static_feature_names,
        sequence_bases=sequence_bases,
        daily_feature_names=daily_feature_names,
        targets=targets,
        static_fill=static_fill,
        static_means=static_means,
        static_stds=static_stds,
        sequence_fill=sequence_fill,
        sequence_means=sequence_means,
        sequence_stds=sequence_stds,
        daily_fill=daily_fill,
        daily_means=daily_means,
        daily_stds=daily_stds,
    )
    train_idx, val_idx = _split_indices(len(dataset), int(random_seed))
    train_subset = torch.utils.data.Subset(dataset, train_idx.tolist())
    val_subset = torch.utils.data.Subset(dataset, val_idx.tolist())
    loader = DataLoader(
        train_subset,
        batch_size=max(1, min(int(batch_size or 1), 2)),
        shuffle=True,
        collate_fn=collate_portfolio_set_days,
        pin_memory=runtime.pin_memory,
    )
    val_loader = DataLoader(
        val_subset,
        batch_size=1,
        shuffle=False,
        collate_fn=collate_portfolio_set_days,
        pin_memory=runtime.pin_memory,
    )
    if progress_sink is not None:
        progress_sink.emit("train_dataframe_ready", train_sample_rows=int(len(train_frame)), portfolio_set_v5=True)
        progress_sink.emit("train_dataloader_ready", batch_count=int(len(loader)), data_loader_pin_memory=runtime.pin_memory)
    model = PortfolioSetPolicyNetV5(
        static_input_dim=len(static_feature_names),
        sequence_input_dim=max(len(sequence_bases), 1),
        daily_input_dim=max(len(daily_feature_names), 1),
        model_dim=int(model_dim),
        temporal_layers=int(temporal_layers),
        cross_layers=int(cross_layers),
        latent_count=int(latent_count),
        dropout=float(dropout),
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(learning_rate), weight_decay=1.0e-4)
    scaler = torch.amp.GradScaler("cuda", enabled=runtime.amp_enabled)
    weights = loss_config["multi_objective_loss_weights"]
    best_loss = float("inf")
    best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
    best_epoch = 0
    completed_epochs = 0
    progress_event_count = 0
    started = time.monotonic()
    for epoch in range(1, max(int(epochs or 0), 1) + 1):
        epoch_started = time.monotonic()
        model.train()
        train_loss_sum = 0.0
        train_count = 0
        for batch in loader:
            batch = move_to_device(batch, device, non_blocking=runtime.non_blocking_transfer)
            optimizer.zero_grad(set_to_none=True)
            with autocast_context(runtime):
                raw = model(batch["static_x"], batch["sequence_x"], batch["daily_x"], batch["sample_mask"])["raw"]
                loss = _portfolio_set_loss(raw, batch, weights)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            train_loss_sum += float(loss.detach().cpu())
            train_count += 1
        model.eval()
        val_losses: list[float] = []
        with torch.no_grad():
            for batch in val_loader:
                batch = move_to_device(batch, device, non_blocking=runtime.non_blocking_transfer)
                raw = model(batch["static_x"], batch["sequence_x"], batch["daily_x"], batch["sample_mask"])["raw"]
                val_losses.append(float(_portfolio_set_loss(raw, batch, weights).detach().cpu()))
        val_loss = float(np.mean(val_losses)) if val_losses else float(train_loss_sum / max(train_count, 1))
        completed_epochs = epoch
        if val_loss < best_loss:
            best_loss = val_loss
            best_epoch = epoch
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
        if progress_sink is not None:
            progress_sink.emit(
                "train_epoch_complete",
                epoch=epoch,
                completed_epochs=completed_epochs,
                train_loss=train_loss_sum / max(train_count, 1),
                validation_loss=val_loss,
                epoch_seconds=round(time.monotonic() - epoch_started, 3),
                amp_enabled=runtime.amp_enabled,
                data_loader_pin_memory=runtime.pin_memory,
                non_blocking_transfer=runtime.non_blocking_transfer,
                portfolio_set_v5=True,
            )
            progress_event_count += 1
        if epoch >= int(min_epochs or 0) and epoch - best_epoch >= int(early_stop_patience or 0):
            break
    model.load_state_dict(best_state, strict=True)
    diagnostics = {
        "trainer_backend": TRAINER_BACKEND_FORMAL_PORTFOLIO_SET_V5,
        "loss_profile": resolved_loss_profile,
        "status": "portfolio_set_v5_complete",
        "device": str(device),
        "gpu_acceleration": runtime.to_diagnostics(),
        "amp_enabled": runtime.amp_enabled,
        "data_loader_pin_memory": runtime.pin_memory,
        "non_blocking_transfer": runtime.non_blocking_transfer,
        "completed_epochs": int(completed_epochs),
        "best_epoch": int(best_epoch),
        "best_validation_loss": float(best_loss),
        "train_sample_rows": int(len(train_frame)),
        "raw_train_sample_rows": int(len(sample_frame)),
        "train_day_count": int(len(dataset)),
        "portfolio_set_v5_latent_count": int(latent_count),
        "portfolio_set_v5_uses_latent_attention": True,
        "portfolio_set_v5_full_self_attention": False,
        "portfolio_set_v5_shadow_only": True,
        "supports_release_first_allocation_v3_mode": True,
        "progress_event_count": int(progress_event_count),
        "train_seconds": round(time.monotonic() - started, 3),
        **day_diagnostics,
    }
    if progress_sink is not None:
        progress_sink.emit("training_complete", completed_epochs=completed_epochs, best_epoch=best_epoch, portfolio_set_v5=True)
    artifact = TorchPortfolioSetV5Artifact(
        feature_names=list(feature_names),
        daily_feature_names=list(daily_feature_names),
        static_feature_names=list(static_feature_names),
        sequence_bases=list(sequence_bases),
        feature_fill_values=static_fill,
        feature_means=static_means,
        feature_stds=static_stds,
        sequence_fill_values=sequence_fill,
        sequence_means=sequence_means,
        sequence_stds=sequence_stds,
        daily_fill_values=daily_fill if len(daily_fill) else np.zeros(1, dtype=np.float32),
        daily_means=daily_means if len(daily_means) else np.zeros(1, dtype=np.float32),
        daily_stds=daily_stds if len(daily_stds) else np.ones(1, dtype=np.float32),
        train_summary=dict(train_summary or {}),
        training_diagnostics=diagnostics,
        training_contract=contract,
        trained_at=str(trained_at or ""),
        model_config={
            "model_dim": int(model_dim),
            "temporal_layers": int(temporal_layers),
            "cross_layers": int(cross_layers),
            "latent_count": int(latent_count),
            "dropout": float(dropout),
        },
        model_state_dict={key: value.detach().cpu() for key, value in model.state_dict().items()},
        global_target_defaults=_global_defaults(daily_frame),
    )
    artifact.save(run_root / PORTFOLIO_SET_V5_ARTIFACT_FILENAME)
    return artifact
