from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from daily_research.continuous_policy.model_v2 import (
    ACTION_CLASSES,
    DURATION_CLASSES,
    HOLDING_DAYS_BY_BUCKET,
    _action_weights,
    _finite_scalar,
    _prepare_matrix,
    _resolve_device,
    _save_checkpoint,
    _signature_hash,
    _signature_payload,
    _split_indices,
    resolve_decoder_profile,
)
from daily_research.continuous_policy.model_seq_v3 import SEQUENCE_STEP_ORDER, resolve_sequence_columns
from daily_research.continuous_policy.training_contracts import TRAINER_BACKEND_FORMAL_HIER_V4


GLOBAL_TARGET_NAMES_V4: tuple[str, ...] = (
    "gross_exposure_target",
    "candidate_budget",
    "turnover_budget",
    "max_position_weight_target",
    "hold_bias_target",
    "reduce_bias_target",
    "exit_patience_target",
    "reentry_guard_target",
)


class LearnedAttentionPool(nn.Module):
    def __init__(self, dim: int, heads: int = 4) -> None:
        super().__init__()
        self.query = nn.Parameter(torch.randn(1, 1, dim) * 0.02)
        self.attn = nn.MultiheadAttention(dim, heads, batch_first=True)

    def forward(self, tokens: torch.Tensor, valid_mask: torch.Tensor) -> torch.Tensor:
        batch_size = int(tokens.shape[0])
        query = self.query.expand(batch_size, -1, -1)
        key_padding_mask = ~valid_mask.bool()
        pooled, _ = self.attn(query, tokens, tokens, key_padding_mask=key_padding_mask)
        return pooled.squeeze(1)


class TemporalPathEncoder(nn.Module):
    def __init__(self, *, input_dim: int, model_dim: int, layers: int = 2, heads: int = 4, dropout: float = 0.10, steps: int = 5) -> None:
        super().__init__()
        self.input_proj = nn.Linear(int(input_dim), int(model_dim))
        self.positional = nn.Parameter(torch.randn(int(steps), int(model_dim)) * 0.02)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=int(model_dim),
            nhead=int(heads),
            dim_feedforward=int(model_dim) * 4,
            dropout=float(dropout),
            batch_first=True,
            activation="gelu",
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=int(layers))
        self.output_norm = nn.LayerNorm(int(model_dim))

    def forward(self, sequence_x: torch.Tensor) -> torch.Tensor:
        encoded = self.input_proj(sequence_x) + self.positional.unsqueeze(0)
        encoded = self.encoder(encoded)
        return self.output_norm(encoded.mean(dim=1))


class HierarchicalContinuousPolicyNet(nn.Module):
    def __init__(
        self,
        *,
        static_input_dim: int,
        sequence_feature_dim: int,
        sequence_steps: int,
        daily_input_dim: int,
        model_dim: int = 256,
        temporal_layers: int = 2,
        temporal_heads: int = 4,
        cross_layers: int = 2,
        cross_heads: int = 4,
        dropout: float = 0.10,
    ) -> None:
        super().__init__()
        self.model_dim = int(model_dim)
        self.sequence_steps = int(sequence_steps)
        self.temporal_layers = int(temporal_layers)
        self.temporal_heads = int(temporal_heads)
        self.cross_layers = int(cross_layers)
        self.cross_heads = int(cross_heads)
        self.static_encoder = nn.Sequential(
            nn.Linear(int(static_input_dim), int(model_dim)),
            nn.GELU(),
            nn.Dropout(float(dropout)),
            nn.Linear(int(model_dim), int(model_dim)),
            nn.GELU(),
            nn.Dropout(float(dropout)),
        )
        self.temporal_encoder = TemporalPathEncoder(
            input_dim=int(sequence_feature_dim),
            model_dim=int(model_dim),
            layers=int(temporal_layers),
            heads=int(temporal_heads),
            dropout=float(dropout),
            steps=int(sequence_steps),
        )
        self.stock_fusion = nn.Sequential(
            nn.Linear(int(model_dim) * 2, int(model_dim)),
            nn.GELU(),
            nn.Dropout(float(dropout)),
            nn.Linear(int(model_dim), int(model_dim)),
            nn.GELU(),
            nn.Dropout(float(dropout)),
        )
        self.market_encoder = nn.Sequential(
            nn.Linear(int(daily_input_dim), int(model_dim)),
            nn.GELU(),
            nn.Dropout(float(dropout) * 0.5),
            nn.Linear(int(model_dim), int(model_dim)),
            nn.GELU(),
        )
        self.universe_pool = LearnedAttentionPool(int(model_dim), heads=max(1, int(cross_heads)))
        self.portfolio_pool = LearnedAttentionPool(int(model_dim), heads=max(1, int(cross_heads)))
        cross_layer = nn.TransformerEncoderLayer(
            d_model=int(model_dim),
            nhead=int(cross_heads),
            dim_feedforward=int(model_dim) * 4,
            dropout=float(dropout),
            batch_first=True,
            activation="gelu",
            norm_first=True,
        )
        self.cross_encoder = nn.TransformerEncoder(cross_layer, num_layers=int(cross_layers))
        self.global_backbone = nn.Sequential(
            nn.Linear(int(model_dim) * 3, int(model_dim)),
            nn.GELU(),
            nn.Dropout(float(dropout)),
            nn.Linear(int(model_dim), int(model_dim)),
            nn.GELU(),
        )
        self.stock_context = nn.Sequential(
            nn.Linear(int(model_dim) * 4, int(model_dim)),
            nn.GELU(),
            nn.Dropout(float(dropout)),
            nn.Linear(int(model_dim), int(model_dim)),
            nn.GELU(),
            nn.Dropout(float(dropout)),
        )
        self.global_head = nn.Linear(int(model_dim), len(GLOBAL_TARGET_NAMES_V4))
        self.action_head = nn.Linear(int(model_dim), len(ACTION_CLASSES))
        self.duration_head = nn.Linear(int(model_dim), len(DURATION_CLASSES))
        self.delta_head = nn.Linear(int(model_dim), 1)
        self.entry_head = nn.Linear(int(model_dim), 1)
        self.hold_head = nn.Linear(int(model_dim), 1)
        self.add_head = nn.Linear(int(model_dim), 1)
        self.reduce_head = nn.Linear(int(model_dim), 1)
        self.exit_head = nn.Linear(int(model_dim), 1)
        self.reentry_head = nn.Linear(int(model_dim), 1)

    def _decode_global_targets(self, raw_targets: torch.Tensor) -> dict[str, torch.Tensor]:
        sigmoid = torch.sigmoid(raw_targets)
        return {
            "gross_exposure_target": 0.15 + sigmoid[:, 0] * 0.83,
            "candidate_budget": 2.0 + sigmoid[:, 1] * 10.0,
            "turnover_budget": 0.08 + sigmoid[:, 2] * 0.92,
            "max_position_weight_target": 0.08 + sigmoid[:, 3] * 0.20,
            "hold_bias_target": 0.05 + sigmoid[:, 4] * 0.90,
            "reduce_bias_target": sigmoid[:, 5] * 0.65,
            "exit_patience_target": 0.05 + sigmoid[:, 6] * 0.90,
            "reentry_guard_target": sigmoid[:, 7] * 0.45,
        }

    def forward(
        self,
        static_x: torch.Tensor,
        sequence_x: torch.Tensor,
        daily_x: torch.Tensor,
        sample_mask: torch.Tensor,
        holding_weight: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        batch_size, stock_count, _, _ = sequence_x.shape
        static_hidden = self.static_encoder(static_x)
        sequence_hidden = self.temporal_encoder(sequence_x.reshape(batch_size * stock_count, self.sequence_steps, sequence_x.shape[-1]))
        sequence_hidden = sequence_hidden.reshape(batch_size, stock_count, self.model_dim)
        stock_hidden = self.stock_fusion(torch.cat([static_hidden, sequence_hidden], dim=-1))

        valid_mask = sample_mask.bool()
        holding_mask = valid_mask & (holding_weight > 1e-8)
        holding_fallback = torch.where(holding_mask.any(dim=1, keepdim=True), holding_mask, valid_mask)

        market_token = self.market_encoder(daily_x).unsqueeze(1)
        universe_token = self.universe_pool(stock_hidden, valid_mask).unsqueeze(1)
        portfolio_token = self.portfolio_pool(stock_hidden, holding_fallback).unsqueeze(1)
        tokens = torch.cat([market_token, portfolio_token, universe_token, stock_hidden], dim=1)
        token_padding_mask = torch.cat([torch.zeros((batch_size, 3), dtype=torch.bool, device=tokens.device), ~valid_mask], dim=1)
        encoded = self.cross_encoder(tokens, src_key_padding_mask=token_padding_mask)
        market_context = encoded[:, 0]
        portfolio_context = encoded[:, 1]
        universe_context = encoded[:, 2]
        stock_context = encoded[:, 3:]
        global_context = self.global_backbone(torch.cat([market_context, portfolio_context, universe_context], dim=-1))
        global_targets = self._decode_global_targets(self.global_head(global_context))
        repeated_global = global_context.unsqueeze(1).expand(-1, stock_count, -1)
        stock_context = self.stock_context(torch.cat([stock_hidden, static_hidden, sequence_hidden, repeated_global], dim=-1))
        return {
            "action_logits": self.action_head(stock_context),
            "duration_logits": self.duration_head(stock_context),
            "target_delta_hint": self.delta_head(stock_context).squeeze(-1),
            "entry_quality": self.entry_head(stock_context).squeeze(-1),
            "hold_quality": self.hold_head(stock_context).squeeze(-1),
            "add_quality": self.add_head(stock_context).squeeze(-1),
            "reduce_quality": self.reduce_head(stock_context).squeeze(-1),
            "exit_urgency": self.exit_head(stock_context).squeeze(-1),
            "reentry_readiness": self.reentry_head(stock_context).squeeze(-1),
            **global_targets,
        }


@dataclass
class TorchContinuousPolicyHierV4Artifact:
    model: HierarchicalContinuousPolicyNet
    static_feature_names: list[str]
    sequence_base_names: list[str]
    sequence_steps: list[int]
    sequence_columns: list[str]
    daily_feature_names: list[str]
    static_fill_values: np.ndarray
    static_means: np.ndarray
    static_stds: np.ndarray
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

    def save(self, path: Path) -> Path:
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "artifact_type": "continuous_policy_torch_hier_v4",
            "static_feature_names": self.static_feature_names,
            "sequence_base_names": self.sequence_base_names,
            "sequence_steps": self.sequence_steps,
            "sequence_columns": self.sequence_columns,
            "daily_feature_names": self.daily_feature_names,
            "static_fill_values": self.static_fill_values.tolist(),
            "static_means": self.static_means.tolist(),
            "static_stds": self.static_stds.tolist(),
            "sequence_fill_values": self.sequence_fill_values.tolist(),
            "sequence_means": self.sequence_means.tolist(),
            "sequence_stds": self.sequence_stds.tolist(),
            "daily_fill_values": self.daily_fill_values.tolist(),
            "daily_means": self.daily_means.tolist(),
            "daily_stds": self.daily_stds.tolist(),
            "model_state_dict": self.model.state_dict(),
            "model_config": {
                "static_input_dim": len(self.static_feature_names),
                "sequence_feature_dim": len(self.sequence_base_names),
                "sequence_steps": len(self.sequence_steps),
                "daily_input_dim": len(self.daily_feature_names),
                "model_dim": int(self.model.model_dim),
                "temporal_layers": int(self.model.temporal_layers),
                "temporal_heads": int(self.model.temporal_heads),
                "cross_layers": int(self.model.cross_layers),
                "cross_heads": int(self.model.cross_heads),
            },
            "train_summary": self.train_summary,
            "training_diagnostics": self.training_diagnostics,
            "training_contract": self.training_contract,
            "trained_at": self.trained_at,
        }
        torch.save(payload, path)
        return path


class DailyGroupedDataset(Dataset[dict[str, np.ndarray]]):
    def __init__(self, groups: list[dict[str, np.ndarray]]) -> None:
        self.groups = list(groups)

    def __len__(self) -> int:
        return len(self.groups)

    def __getitem__(self, idx: int) -> dict[str, np.ndarray]:
        return self.groups[idx]


def _masked_mean_squared_error(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    mask_f = mask.to(dtype=pred.dtype)
    diff = (pred - target) ** 2 * mask_f
    denom = torch.clamp(mask_f.sum(), min=1.0)
    return diff.sum() / denom


def _masked_cross_entropy(logits: torch.Tensor, target: torch.Tensor, mask: torch.Tensor, *, weight: torch.Tensor | None = None) -> torch.Tensor:
    flat_mask = mask.reshape(-1).bool()
    flat_logits = logits.reshape(-1, logits.shape[-1])
    flat_target = target.reshape(-1)
    if not bool(flat_mask.any()):
        return flat_logits.sum() * 0.0
    return nn.functional.cross_entropy(flat_logits[flat_mask], flat_target[flat_mask], weight=weight)


def _masked_scalar_loss(outputs: dict[str, torch.Tensor], targets: dict[str, torch.Tensor], mask: torch.Tensor) -> torch.Tensor:
    pieces = []
    for name, target in targets.items():
        pieces.append(_masked_mean_squared_error(outputs[name], target, mask))
    if not pieces:
        return mask.sum() * 0.0
    return torch.stack(pieces).mean()


def _derive_extra_daily_targets(daily_frame: pd.DataFrame) -> pd.DataFrame:
    working = daily_frame.copy()
    market_downside = working.get("market_downside_pressure", pd.Series(0.0, index=working.index)).astype(float).clip(0.0, 1.0)
    cash_pressure = working.get("portfolio_cash_pressure", pd.Series(0.0, index=working.index)).astype(float).clip(0.0, 1.0)
    reversal_rate = working.get("recent_reversal_rate_20d", pd.Series(0.0, index=working.index)).astype(float).clip(0.0, 1.0)
    hold_bias = working["hold_bias_target"].astype(float).clip(0.0, 1.0)
    if "reduce_bias_target" not in working.columns:
        working["reduce_bias_target"] = (
            0.06
            + market_downside * 0.22
            + cash_pressure * 0.22
            + reversal_rate * 0.20
            + (1.0 - hold_bias) * 0.12
        ).clip(0.0, 0.65)
    if "exit_patience_target" not in working.columns:
        working["exit_patience_target"] = (
            0.12
            + hold_bias * 0.46
            + (1.0 - market_downside) * 0.12
            - cash_pressure * 0.08
            - reversal_rate * 0.06
        ).clip(0.05, 0.95)
    if "reentry_guard_target" not in working.columns:
        working["reentry_guard_target"] = (
            reversal_rate * 0.26
            + market_downside * 0.12
            + cash_pressure * 0.14
        ).clip(0.0, 0.45)
    return working


def _build_grouped_day_batches(
    *,
    sample_frame: pd.DataFrame,
    daily_frame: pd.DataFrame,
    static_feature_names: list[str],
    sequence_columns: list[str],
    sequence_base_names: list[str],
    daily_feature_names: list[str],
) -> tuple[list[dict[str, np.ndarray]], np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    sample_working = sample_frame.copy().sort_values(["date", "stock"], ascending=[True, True]).reset_index(drop=True)
    daily_working = _derive_extra_daily_targets(daily_frame.copy()).sort_values("date").reset_index(drop=True)
    X_static, static_fill, static_means, static_stds = _prepare_matrix(sample_working, static_feature_names)
    X_sequence_flat, sequence_fill, sequence_means, sequence_stds = _prepare_matrix(sample_working, sequence_columns)
    X_daily, daily_fill, daily_means, daily_stds = _prepare_matrix(daily_working, daily_feature_names)
    X_sequence = X_sequence_flat.reshape(len(sample_working), len(SEQUENCE_STEP_ORDER), len(sequence_base_names)).astype(np.float32)

    action_lookup = {name: idx for idx, name in enumerate(ACTION_CLASSES)}
    duration_lookup = {name: idx for idx, name in enumerate(DURATION_CLASSES)}
    action_codes = np.asarray([action_lookup.get(str(value), 0) for value in sample_working["action_label"].astype(str)], dtype=np.int64)
    duration_codes = np.asarray(
        [
            duration_lookup.get(str(value), 0)
            for value in sample_working["planned_holding_bucket"].astype(str).where(
                sample_working["planned_holding_bucket"].astype(str).isin(DURATION_CLASSES),
                "avoid",
            )
        ],
        dtype=np.int64,
    )
    scalar_targets = {
        "target_delta_hint": sample_working["target_delta_hint"].astype(float).to_numpy(dtype=np.float32),
        "entry_quality": sample_working["entry_quality"].astype(float).to_numpy(dtype=np.float32),
        "hold_quality": sample_working["hold_quality"].astype(float).to_numpy(dtype=np.float32),
        "add_quality": sample_working["add_quality"].astype(float).to_numpy(dtype=np.float32),
        "reduce_quality": sample_working["reduce_quality"].astype(float).to_numpy(dtype=np.float32),
        "exit_urgency": sample_working["exit_urgency"].astype(float).to_numpy(dtype=np.float32),
        "reentry_readiness": sample_working["reentry_readiness"].astype(float).to_numpy(dtype=np.float32),
    }
    current_weight = sample_working["current_weight"].astype(float).fillna(0.0).to_numpy(dtype=np.float32)

    sample_positions_by_date: dict[str, np.ndarray] = {
        str(date_value): positions.to_numpy(dtype=np.int64)
        for date_value, positions in sample_working.groupby("date").groups.items()
    }
    daily_position_by_date = {str(date_value): idx for idx, date_value in enumerate(daily_working["date"].astype(str))}
    groups: list[dict[str, np.ndarray]] = []
    for date_value in daily_working["date"].astype(str):
        positions = sample_positions_by_date.get(str(date_value))
        if positions is None or len(positions) == 0:
            continue
        daily_idx = int(daily_position_by_date[str(date_value)])
        groups.append(
            {
                "date": np.asarray([str(date_value)]),
                "static_x": X_static[positions],
                "sequence_x": X_sequence[positions],
                "current_weight": current_weight[positions],
                "action_target": action_codes[positions],
                "duration_target": duration_codes[positions],
                "target_delta_hint": scalar_targets["target_delta_hint"][positions],
                "entry_quality": scalar_targets["entry_quality"][positions],
                "hold_quality": scalar_targets["hold_quality"][positions],
                "add_quality": scalar_targets["add_quality"][positions],
                "reduce_quality": scalar_targets["reduce_quality"][positions],
                "exit_urgency": scalar_targets["exit_urgency"][positions],
                "reentry_readiness": scalar_targets["reentry_readiness"][positions],
                "daily_x": X_daily[daily_idx],
                **{
                    name: np.float32(daily_working.at[daily_idx, name] if name in daily_working.columns else 0.0)
                    for name in GLOBAL_TARGET_NAMES_V4
                },
            }
        )
    return groups, static_fill, static_means, static_stds, sequence_fill, sequence_means, sequence_stds, daily_fill, daily_means, daily_stds


def _collate_day_groups(batch: list[dict[str, np.ndarray]]) -> dict[str, torch.Tensor]:
    max_items = max(int(item["static_x"].shape[0]) for item in batch)
    static_dim = int(batch[0]["static_x"].shape[1])
    sequence_steps = int(batch[0]["sequence_x"].shape[1])
    sequence_dim = int(batch[0]["sequence_x"].shape[2])
    batch_size = len(batch)

    static_x = np.zeros((batch_size, max_items, static_dim), dtype=np.float32)
    sequence_x = np.zeros((batch_size, max_items, sequence_steps, sequence_dim), dtype=np.float32)
    mask = np.zeros((batch_size, max_items), dtype=bool)
    current_weight = np.zeros((batch_size, max_items), dtype=np.float32)
    action_target = np.zeros((batch_size, max_items), dtype=np.int64)
    duration_target = np.zeros((batch_size, max_items), dtype=np.int64)
    scalar_targets = {name: np.zeros((batch_size, max_items), dtype=np.float32) for name in (
        "target_delta_hint",
        "entry_quality",
        "hold_quality",
        "add_quality",
        "reduce_quality",
        "exit_urgency",
        "reentry_readiness",
    )}
    daily_x = np.stack([item["daily_x"] for item in batch]).astype(np.float32)
    global_targets = {
        name: np.asarray([float(item[name]) for item in batch], dtype=np.float32)
        for name in GLOBAL_TARGET_NAMES_V4
    }

    for batch_idx, item in enumerate(batch):
        size = int(item["static_x"].shape[0])
        static_x[batch_idx, :size] = item["static_x"]
        sequence_x[batch_idx, :size] = item["sequence_x"]
        current_weight[batch_idx, :size] = item["current_weight"]
        action_target[batch_idx, :size] = item["action_target"]
        duration_target[batch_idx, :size] = item["duration_target"]
        mask[batch_idx, :size] = True
        for name in scalar_targets:
            scalar_targets[name][batch_idx, :size] = item[name]

    payload: dict[str, torch.Tensor] = {
        "static_x": torch.as_tensor(static_x, dtype=torch.float32),
        "sequence_x": torch.as_tensor(sequence_x, dtype=torch.float32),
        "sample_mask": torch.as_tensor(mask, dtype=torch.bool),
        "current_weight": torch.as_tensor(current_weight, dtype=torch.float32),
        "action_target": torch.as_tensor(action_target, dtype=torch.long),
        "duration_target": torch.as_tensor(duration_target, dtype=torch.long),
        "daily_x": torch.as_tensor(daily_x, dtype=torch.float32),
    }
    for name, values in scalar_targets.items():
        payload[name] = torch.as_tensor(values, dtype=torch.float32)
    for name, values in global_targets.items():
        payload[name] = torch.as_tensor(values, dtype=torch.float32)
    return payload


def load_torch_hier_v4_artifact(path: str | Path) -> TorchContinuousPolicyHierV4Artifact:
    payload = torch.load(Path(path), map_location="cpu", weights_only=False)
    if str(payload.get("artifact_type", "") or "") != "continuous_policy_torch_hier_v4":
        raise TypeError(f"Unsupported hierarchical artifact type: {payload.get('artifact_type')!r}")
    model_cfg = dict(payload.get("model_config", {}) or {})
    model = HierarchicalContinuousPolicyNet(**model_cfg)
    model.load_state_dict(payload["model_state_dict"])
    model.eval()
    return TorchContinuousPolicyHierV4Artifact(
        model=model,
        static_feature_names=list(payload.get("static_feature_names", []) or []),
        sequence_base_names=list(payload.get("sequence_base_names", []) or []),
        sequence_steps=[int(item) for item in payload.get("sequence_steps", []) or []],
        sequence_columns=list(payload.get("sequence_columns", []) or []),
        daily_feature_names=list(payload.get("daily_feature_names", []) or []),
        static_fill_values=np.asarray(payload.get("static_fill_values", []), dtype=np.float32),
        static_means=np.asarray(payload.get("static_means", []), dtype=np.float32),
        static_stds=np.asarray(payload.get("static_stds", []), dtype=np.float32),
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
    )


def fit_policy_models_v4(
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
    epochs: int = 32,
    min_epochs: int = 32,
    batch_size: int = 8,
    learning_rate: float = 9e-4,
    model_dim: int = 256,
    temporal_layers: int = 2,
    cross_layers: int = 2,
    dropout: float = 0.10,
    early_stop_patience: int = 10,
    resume_mode: str = "strict",
) -> TorchContinuousPolicyHierV4Artifact:
    if sample_frame.empty or daily_frame.empty:
        raise ValueError("continuous_policy formal_torch_hier_v4 received empty training data.")
    contract = dict(training_contract or {})
    if str(contract.get("trainer_backend", "") or "") != TRAINER_BACKEND_FORMAL_HIER_V4:
        raise ValueError("fit_policy_models_v4 requires the formal_torch_hier_v4 training contract.")
    device = _resolve_device(contract)
    torch.manual_seed(int(random_seed))
    np.random.seed(int(random_seed))
    run_root.mkdir(parents=True, exist_ok=True)

    sequence_base_names, sequence_columns = resolve_sequence_columns(feature_names)
    static_feature_names = [name for name in feature_names if name not in set(sequence_columns)]
    if not static_feature_names:
        raise ValueError("formal_torch_hier_v4 requires at least one static feature.")

    groups, static_fill, static_means, static_stds, sequence_fill, sequence_means, sequence_stds, daily_fill, daily_means, daily_stds = _build_grouped_day_batches(
        sample_frame=sample_frame,
        daily_frame=daily_frame,
        static_feature_names=static_feature_names,
        sequence_columns=sequence_columns,
        sequence_base_names=sequence_base_names,
        daily_feature_names=daily_feature_names,
    )
    if len(groups) < 24:
        raise ValueError("continuous_policy formal_torch_hier_v4 requires at least 24 grouped daily batches.")

    group_dates = [str(item["date"][0]) for item in groups]
    train_idx, val_idx = _split_indices(len(groups), random_seed + 31)
    train_groups = [groups[int(idx)] for idx in train_idx]
    val_groups = [groups[int(idx)] for idx in val_idx]

    model = HierarchicalContinuousPolicyNet(
        static_input_dim=len(static_feature_names),
        sequence_feature_dim=len(sequence_base_names),
        sequence_steps=len(SEQUENCE_STEP_ORDER),
        daily_input_dim=len(daily_feature_names),
        model_dim=max(192, int(model_dim)),
        temporal_layers=max(2, int(temporal_layers)),
        temporal_heads=4,
        cross_layers=max(2, int(cross_layers)),
        cross_heads=4,
        dropout=max(0.05, float(dropout)),
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(learning_rate), weight_decay=1e-4)
    action_weight_tensor = _action_weights(sample_frame).to(device)

    signature_payload = _signature_payload(
        feature_names=feature_names,
        daily_feature_names=daily_feature_names,
        train_summary=dict(train_summary or {}),
        training_contract=contract,
    )
    signature_hash = _signature_hash(signature_payload)
    checkpoint_last = run_root / "checkpoint_last.pt"
    checkpoint_best = run_root / "checkpoint_best.pt"
    artifact_path = run_root / "continuous_policy_hier_v4_artifact.pt"
    diagnostics_path = run_root / "training_diagnostics.json"

    start_epoch = 0
    best_epoch = 0
    best_val_loss = math.inf
    resumed_from = ""
    history: list[dict[str, float | int]] = []
    if checkpoint_last.exists() and resume_mode == "strict":
        state = torch.load(checkpoint_last, map_location="cpu", weights_only=False)
        previous_hash = str(state.get("signature_hash", "") or "")
        if previous_hash and previous_hash != signature_hash:
            raise RuntimeError("continuous_policy hier v4 strict resume rejected because checkpoint lineage does not match the current run signature.")
        model.load_state_dict(state["model_state_dict"])
        optimizer.load_state_dict(state["optimizer_state_dict"])
        start_epoch = int(state.get("epoch", 0) or 0)
        best_epoch = int(state.get("best_epoch", 0) or 0)
        best_val_loss = float(state.get("best_val_loss", math.inf) or math.inf)
        history = list(state.get("history", []) or [])
        resumed_from = str(checkpoint_last.resolve())
    if resume_mode == "strict" and checkpoint_last.exists() and int(epochs) <= int(start_epoch):
        raise RuntimeError(f"continuous_policy hier v4 strict resume requires epochs > completed epochs ({start_epoch}), got {epochs}.")

    train_loader = DataLoader(
        DailyGroupedDataset(train_groups),
        batch_size=max(1, int(batch_size)),
        shuffle=True,
        drop_last=False,
        collate_fn=_collate_day_groups,
    )
    val_loader = DataLoader(
        DailyGroupedDataset(val_groups),
        batch_size=max(1, int(batch_size)),
        shuffle=False,
        drop_last=False,
        collate_fn=_collate_day_groups,
    )

    scalar_target_names = (
        "target_delta_hint",
        "entry_quality",
        "hold_quality",
        "add_quality",
        "reduce_quality",
        "exit_urgency",
        "reentry_readiness",
    )

    def _run_epoch(loader: DataLoader[Any], *, train_mode: bool) -> float:
        if train_mode:
            model.train()
        else:
            model.eval()
        epoch_losses: list[float] = []
        for batch in loader:
            batch_tensors = {key: value.to(device) for key, value in batch.items()}
            with torch.set_grad_enabled(train_mode):
                outputs = model(
                    batch_tensors["static_x"],
                    batch_tensors["sequence_x"],
                    batch_tensors["daily_x"],
                    batch_tensors["sample_mask"],
                    batch_tensors["current_weight"],
                )
                action_loss = _masked_cross_entropy(
                    outputs["action_logits"],
                    batch_tensors["action_target"],
                    batch_tensors["sample_mask"],
                    weight=action_weight_tensor,
                )
                duration_loss = _masked_cross_entropy(
                    outputs["duration_logits"],
                    batch_tensors["duration_target"],
                    batch_tensors["sample_mask"],
                )
                scalar_loss = _masked_scalar_loss(
                    outputs,
                    {name: batch_tensors[name] for name in scalar_target_names},
                    batch_tensors["sample_mask"],
                )
                global_loss = torch.stack(
                    [
                        nn.functional.mse_loss(outputs[name], batch_tensors[name])
                        for name in GLOBAL_TARGET_NAMES_V4
                    ]
                ).mean()
                loss = action_loss + 0.55 * duration_loss + 0.42 * scalar_loss + 0.35 * global_loss
                if train_mode:
                    optimizer.zero_grad(set_to_none=True)
                    loss.backward()
                    nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)
                    optimizer.step()
            epoch_losses.append(float(loss.detach().cpu()))
        return float(np.mean(epoch_losses)) if epoch_losses else 0.0

    patience_used = 0
    for epoch in range(start_epoch + 1, int(epochs) + 1):
        train_loss = _run_epoch(train_loader, train_mode=True)
        val_loss = _run_epoch(val_loader, train_mode=False)
        history.append({"epoch": int(epoch), "train_loss": train_loss, "validation_loss": val_loss})
        checkpoint_payload = {
            "epoch": int(epoch),
            "best_epoch": int(best_epoch),
            "best_val_loss": float(best_val_loss),
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "history": history[-200:],
            "signature_hash": signature_hash,
        }
        _save_checkpoint(checkpoint_last, checkpoint_payload)
        if val_loss < best_val_loss - 1e-6:
            best_val_loss = val_loss
            best_epoch = int(epoch)
            patience_used = 0
            checkpoint_payload["best_epoch"] = int(best_epoch)
            checkpoint_payload["best_val_loss"] = float(best_val_loss)
            _save_checkpoint(checkpoint_best, checkpoint_payload)
        else:
            patience_used += 1
        if int(epoch) >= int(min_epochs) and int(patience_used) >= int(early_stop_patience):
            break

    best_state = torch.load(checkpoint_best if checkpoint_best.exists() else checkpoint_last, map_location="cpu", weights_only=False)
    model.load_state_dict(best_state["model_state_dict"])
    model.eval()

    diagnostics = {
        "trainer_backend": TRAINER_BACKEND_FORMAL_HIER_V4,
        "device": str(device),
        "cuda_available": bool(torch.cuda.is_available()),
        "epochs_requested": int(epochs),
        "min_epochs": int(min_epochs),
        "completed_epochs": int(history[-1]["epoch"]) if history else 0,
        "best_epoch": int(best_epoch),
        "best_validation_loss": float(best_val_loss),
        "resume_mode": str(resume_mode or ""),
        "resumed_from_checkpoint": resumed_from,
        "checkpoint_last": str(checkpoint_last.resolve()),
        "checkpoint_best": str(checkpoint_best.resolve()) if checkpoint_best.exists() else "",
        "training_diagnostics_json": str(diagnostics_path.resolve()),
        "signature_hash": signature_hash,
        "sequence_base_count": len(sequence_base_names),
        "sequence_step_count": len(SEQUENCE_STEP_ORDER),
        "grouped_day_count": len(groups),
        "train_day_count": len(train_groups),
        "validation_day_count": len(val_groups),
        "temporal_layers": max(2, int(temporal_layers)),
        "cross_layers": max(2, int(cross_layers)),
        "global_target_count": len(GLOBAL_TARGET_NAMES_V4),
        "history_tail": history[-8:],
        "train_dates_tail": group_dates[-8:],
    }
    artifact = TorchContinuousPolicyHierV4Artifact(
        model=model.cpu(),
        static_feature_names=list(static_feature_names),
        sequence_base_names=list(sequence_base_names),
        sequence_steps=[int(item) for item in SEQUENCE_STEP_ORDER],
        sequence_columns=list(sequence_columns),
        daily_feature_names=list(daily_feature_names),
        static_fill_values=static_fill,
        static_means=static_means,
        static_stds=static_stds,
        sequence_fill_values=sequence_fill,
        sequence_means=sequence_means,
        sequence_stds=sequence_stds,
        daily_fill_values=daily_fill,
        daily_means=daily_means,
        daily_stds=daily_stds,
        train_summary=dict(train_summary or {}),
        training_diagnostics=diagnostics,
        training_contract=contract,
        trained_at=str(trained_at or ""),
    )
    artifact.save(artifact_path)
    diagnostics_path.write_text(json.dumps(diagnostics, ensure_ascii=False, indent=2), encoding="utf-8")
    return artifact


def _apply_normalization(values: np.ndarray, *, fill: np.ndarray, means: np.ndarray, stds: np.ndarray) -> np.ndarray:
    safe = np.where(np.isfinite(values), values, fill[None, ...]).astype(np.float32)
    return ((safe - means[None, ...]) / stds[None, ...]).astype(np.float32)


def predict_policy_v4(
    artifact: TorchContinuousPolicyHierV4Artifact,
    *,
    state_frame: pd.DataFrame,
    daily_features: dict[str, float],
) -> tuple[pd.DataFrame, dict[str, float]]:
    if state_frame.empty:
        raise ValueError("state_frame is empty.")

    working = state_frame.copy()
    static_raw = working.loc[:, artifact.static_feature_names].replace([np.inf, -np.inf], np.nan).to_numpy(dtype=np.float32, copy=True)
    sequence_raw = working.loc[:, artifact.sequence_columns].replace([np.inf, -np.inf], np.nan).to_numpy(dtype=np.float32, copy=True)
    static_x = _apply_normalization(static_raw, fill=artifact.static_fill_values, means=artifact.static_means, stds=artifact.static_stds)
    sequence_x = _apply_normalization(sequence_raw, fill=artifact.sequence_fill_values, means=artifact.sequence_means, stds=artifact.sequence_stds)
    sequence_x = sequence_x.reshape(len(working), len(artifact.sequence_steps), len(artifact.sequence_base_names)).astype(np.float32)
    daily_row = np.asarray([[float(daily_features.get(name, 0.0) or 0.0) for name in artifact.daily_feature_names]], dtype=np.float32)
    daily_x = _apply_normalization(daily_row, fill=artifact.daily_fill_values, means=artifact.daily_means, stds=artifact.daily_stds)
    current_weight = working["current_weight"].astype(float).fillna(0.0).to_numpy(dtype=np.float32)
    decoder_profile_name, decoder_profile = resolve_decoder_profile(artifact.train_summary.get("decoder_profile"))

    model = artifact.model
    model.eval()
    with torch.no_grad():
        outputs = model(
            torch.as_tensor(static_x[None, ...], dtype=torch.float32),
            torch.as_tensor(sequence_x[None, ...], dtype=torch.float32),
            torch.as_tensor(daily_x, dtype=torch.float32),
            torch.ones((1, len(working)), dtype=torch.bool),
            torch.as_tensor(current_weight[None, ...], dtype=torch.float32),
        )

    action_prob = torch.softmax(outputs["action_logits"], dim=-1).cpu().numpy()[0]
    duration_prob = torch.softmax(outputs["duration_logits"], dim=-1).cpu().numpy()[0]
    target_delta_hint = outputs["target_delta_hint"].cpu().numpy()[0]
    entry_quality = outputs["entry_quality"].cpu().numpy()[0]
    hold_quality = outputs["hold_quality"].cpu().numpy()[0]
    add_quality = outputs["add_quality"].cpu().numpy()[0]
    reduce_quality = outputs["reduce_quality"].cpu().numpy()[0]
    exit_urgency = np.clip(outputs["exit_urgency"].cpu().numpy()[0], 0.0, None)
    reentry_readiness = np.clip(outputs["reentry_readiness"].cpu().numpy()[0], 0.0, None)
    predicted_duration_codes = duration_prob.argmax(axis=1)
    predicted_duration_labels = np.asarray([DURATION_CLASSES[int(code)] for code in predicted_duration_codes], dtype=object)
    planned_holding_days = np.asarray([HOLDING_DAYS_BY_BUCKET.get(str(label), 0.0) for label in predicted_duration_labels], dtype=float)

    global_targets = {name: float(np.asarray(outputs[name].cpu().numpy()).reshape(-1)[0]) for name in GLOBAL_TARGET_NAMES_V4}
    global_targets = {
        "gross_exposure_target": float(np.clip(_finite_scalar(global_targets["gross_exposure_target"], default=0.35), 0.15, 0.98)),
        "candidate_budget": float(np.clip(_finite_scalar(global_targets["candidate_budget"], default=4.0), 2.0, 12.0)),
        "turnover_budget": float(np.clip(_finite_scalar(global_targets["turnover_budget"], default=0.18), 0.08, 1.00)),
        "max_position_weight_target": float(np.clip(_finite_scalar(global_targets["max_position_weight_target"], default=0.12), 0.08, 0.28)),
        "hold_bias_target": float(np.clip(_finite_scalar(global_targets["hold_bias_target"], default=0.24), 0.05, 0.95)),
        "reduce_bias_target": float(np.clip(_finite_scalar(global_targets["reduce_bias_target"], default=0.10), 0.0, 0.65)),
        "exit_patience_target": float(np.clip(_finite_scalar(global_targets["exit_patience_target"], default=0.20), 0.05, 0.95)),
        "reentry_guard_target": float(np.clip(_finite_scalar(global_targets["reentry_guard_target"], default=0.0), 0.0, 0.45)),
    }

    market_downside_pressure = working.get("market_downside_pressure", pd.Series(0.0, index=working.index)).astype(float).to_numpy(dtype=float)
    portfolio_cash_pressure = working.get("portfolio_cash_pressure", pd.Series(0.0, index=working.index)).astype(float).to_numpy(dtype=float)
    reentry_cooldown = working.get("reentry_cooldown", pd.Series(0.0, index=working.index)).astype(float).to_numpy(dtype=float)
    signal_decay_speed = working.get("signal_decay_speed", pd.Series(0.0, index=working.index)).astype(float).to_numpy(dtype=float)
    drawdown_from_peak = working.get("drawdown_from_peak", pd.Series(0.0, index=working.index)).astype(float).to_numpy(dtype=float)
    holding_flag = working.get("holding_flag", pd.Series(0.0, index=working.index)).astype(float).to_numpy(dtype=float)
    reversal_rate = working.get("recent_reversal_rate_20d", pd.Series(0.0, index=working.index)).astype(float).to_numpy(dtype=float)
    reduce_reversal_pressure = working.get("reduce_reversal_pressure", pd.Series(0.0, index=working.index)).astype(float).to_numpy(dtype=float)
    exit_reentry_pressure = working.get("exit_reentry_pressure", pd.Series(0.0, index=working.index)).astype(float).to_numpy(dtype=float)
    cash_regime_pressure = working.get("cash_regime_pressure", pd.Series(0.0, index=working.index)).astype(float).to_numpy(dtype=float)
    hold_continuity_pressure = working.get("hold_continuity_pressure", pd.Series(0.0, index=working.index)).astype(float).to_numpy(dtype=float)
    hold_days = working.get("hold_days", pd.Series(0.0, index=working.index)).astype(float).to_numpy(dtype=float)

    action_lookup = {name: action_prob[:, idx] for idx, name in enumerate(ACTION_CLASSES)}
    duration_bonus = np.clip(planned_holding_days / 15.0, 0.0, 1.0)
    market_risk_score = np.clip(
        0.55 * market_downside_pressure + 0.25 * portfolio_cash_pressure + 0.20 * cash_regime_pressure,
        0.0,
        1.0,
    )
    reversal_pressure = np.clip(
        0.45 * reversal_rate + 0.35 * reduce_reversal_pressure + 0.20 * exit_reentry_pressure,
        0.0,
        1.0,
    )
    risk_off_score = np.clip(market_risk_score + reversal_pressure * 0.10, 0.0, 1.0)
    market_risk_scalar = float(np.mean(market_risk_score)) if market_risk_score.size else 0.0
    reversal_pressure_scalar = float(np.mean(reversal_pressure)) if reversal_pressure.size else 0.0
    hold_continuity_scalar = float(np.mean(hold_continuity_pressure)) if hold_continuity_pressure.size else 0.0
    global_targets["gross_exposure_target"] = float(
        np.clip(
            global_targets["gross_exposure_target"] - market_risk_scalar * decoder_profile["defensive_cash_scale"],
            0.12,
            0.96,
        )
    )
    global_targets["candidate_budget"] = float(
        np.clip(
            global_targets["candidate_budget"] - market_risk_scalar * decoder_profile["candidate_defensive_penalty"],
            2.0,
            12.0,
        )
    )
    global_targets["turnover_budget"] = float(
        np.clip(
            global_targets["turnover_budget"] * (1.0 - market_risk_scalar * decoder_profile["turnover_defensive_penalty"] - reversal_pressure_scalar * 0.22),
            0.08,
            1.0,
        )
    )
    global_targets["max_position_weight_target"] = float(
        np.clip(
            global_targets["max_position_weight_target"] - market_risk_scalar * decoder_profile["position_cap_defensive_penalty"],
            0.08,
            0.28,
        )
    )
    global_targets["hold_bias_target"] = float(
        np.clip(
            global_targets["hold_bias_target"] + decoder_profile["hold_bias_bonus"] * 0.18 + hold_continuity_scalar * 0.08,
            0.05,
            0.95,
        )
    )
    global_targets["reduce_bias_target"] = float(
        np.clip(
            global_targets["reduce_bias_target"] + reversal_pressure_scalar * 0.04 + decoder_profile["reduce_bias_bonus"] * 0.40,
            0.0,
            0.65,
        )
    )
    global_targets["exit_patience_target"] = float(
        np.clip(
            global_targets["exit_patience_target"] + decoder_profile["exit_patience_bonus"] * 0.50 + hold_continuity_scalar * 0.05 - market_risk_scalar * 0.06,
            0.05,
            0.95,
        )
    )
    global_targets["reentry_guard_target"] = float(
        np.clip(
            global_targets["reentry_guard_target"] + decoder_profile["reversal_cooldown_bonus"] * 0.60 + reversal_pressure_scalar * 0.12,
            0.0,
            0.45,
        )
    )
    open_utility = (
        action_lookup["open"] * 0.45
        + entry_quality * 0.35
        + reentry_readiness * 0.15
        + duration_bonus * 0.10
        - (reentry_cooldown + exit_reentry_pressure) * (0.20 + global_targets["reentry_guard_target"])
        - market_risk_score * 0.20
    )
    hold_utility = (
        action_lookup["hold"] * 0.40
        + hold_quality * 0.28
        + duration_bonus * 0.10
        + global_targets["hold_bias_target"] * 0.12
        + hold_continuity_pressure * 0.10
        - signal_decay_speed * 0.12
        - market_risk_score * 0.08
    )
    add_utility = (
        action_lookup["add"] * 0.36
        + add_quality * 0.28
        + hold_quality * 0.12
        + duration_bonus * 0.08
        - market_risk_score * 0.10
    )
    reduce_utility = (
        action_lookup["reduce"] * 0.38
        + reduce_quality * 0.34
        + global_targets["reduce_bias_target"] * 0.18
        + market_risk_score * 0.10
        + signal_decay_speed * 0.08
        - reduce_reversal_pressure * 0.12
    )
    exit_utility = (
        action_lookup["exit"] * 0.42
        + exit_urgency * 0.38
        + np.clip(-drawdown_from_peak, 0.0, None) * 0.20
        + market_risk_score * 0.10
        - global_targets["exit_patience_target"] * 0.08
    )
    open_candidate_score = (
        open_utility
        + entry_quality * 0.20
        + duration_bonus * 0.08
        - reentry_cooldown * 0.08
    )

    active_preference = np.where(holding_flag > 0.5, np.maximum(hold_utility, add_utility * 0.95), open_utility)
    active_preference = np.where(active_preference > 0.0, active_preference, 0.0)
    candidate_budget_target = max(1, int(round(global_targets["candidate_budget"])))
    candidate_scores = pd.Series(active_preference, index=working.index.astype(str), dtype=float)
    keep_index = list(candidate_scores.nlargest(min(candidate_budget_target, len(candidate_scores))).index)
    active_mask = working.index.astype(str).isin(keep_index)
    if not bool(np.any(active_mask)):
        fallback_order = np.argsort(open_candidate_score)[::-1]
        active_mask = np.zeros(len(working), dtype=bool)
        promoted = 0
        for candidate_idx in fallback_order:
            if promoted >= candidate_budget_target:
                break
            if holding_flag[candidate_idx] <= 0.5 and str(predicted_duration_labels[candidate_idx]) == "avoid":
                continue
            active_mask[candidate_idx] = True
            promoted += 1

    utility = np.where(
        holding_flag > 0.5,
        np.maximum(hold_utility + add_utility * 0.35 - reduce_utility * 0.15 - exit_utility * 0.20, 0.0),
        np.maximum(open_utility, 0.0),
    )
    utility = np.where(active_mask, utility, 0.0)
    if float(np.sum(utility)) <= 1e-8:
        fallback_basis = np.where(
            holding_flag > 0.5,
            np.maximum(current_weight, 0.0) + np.maximum(hold_utility + add_utility * 0.20, 0.0),
            np.maximum(open_candidate_score, 0.0),
        )
        fallback_basis = np.where(active_mask, fallback_basis, 0.0)
        if float(np.sum(fallback_basis)) <= 1e-8 and np.any(active_mask):
            active_indices = np.flatnonzero(active_mask)
            shifted = open_candidate_score[active_indices] - float(np.min(open_candidate_score[active_indices]))
            fallback_basis[active_indices] = shifted + 1.0e-4
        utility = fallback_basis
    score_series = pd.Series(utility, index=working.index.astype(str), dtype=float)
    if float(score_series.sum()) > 1e-12:
        raw_target = score_series / float(score_series.sum()) * float(global_targets["gross_exposure_target"])
    else:
        raw_target = pd.Series(0.0, index=score_series.index, dtype=float)
    raw_target = raw_target.clip(upper=float(global_targets["max_position_weight_target"]))
    current_series = pd.Series(current_weight, index=working.index.astype(str), dtype=float)
    hold_floor_ratio = np.clip(
        0.74
        + global_targets["hold_bias_target"] * 0.12
        + global_targets["exit_patience_target"] * 0.08
        + hold_continuity_pressure * 0.08
        - market_risk_score * 0.10,
        0.60,
        0.96,
    )
    protected_floor = np.where(
        holding_flag > 0.5,
        current_series.to_numpy(dtype=float) * hold_floor_ratio,
        0.0,
    )
    raw_target = pd.Series(
        np.where(
            (holding_flag > 0.5) & (hold_utility >= reduce_utility - 0.03) & (exit_utility < hold_utility + 0.04),
            np.maximum(raw_target.to_numpy(dtype=float), protected_floor),
            raw_target.to_numpy(dtype=float),
        ),
        index=raw_target.index,
        dtype=float,
    )
    if float(raw_target.sum()) > float(global_targets["gross_exposure_target"]) and float(raw_target.sum()) > 0:
        raw_target = raw_target / float(raw_target.sum()) * float(global_targets["gross_exposure_target"])
    delta = raw_target - current_series
    raw_turnover = float(delta.abs().sum())
    if raw_turnover > float(global_targets["turnover_budget"]) > 0:
        delta = delta * (float(global_targets["turnover_budget"]) / raw_turnover)
    target_weight = (current_series + delta).clip(lower=0.0)
    if float(target_weight.sum()) > 0.999:
        target_weight = target_weight / float(target_weight.sum())
    delta = target_weight - current_series

    policy_rows: list[dict[str, Any]] = []
    for idx, stock in enumerate(working.index.astype(str)):
        current = float(current_series.iloc[idx])
        target = float(target_weight.iloc[idx])
        diff = float(delta.iloc[idx])
        if current > 1e-8 and target <= 1e-5:
            action_label = "exit" if exit_utility[idx] >= reduce_utility[idx] + 0.04 else "reduce"
        elif current <= 1e-8 and target > 1e-5:
            action_label = "open"
        elif diff > 0.006:
            action_label = "add" if current > 1e-8 else "open"
        elif diff < -0.006:
            action_label = "reduce"
        elif current > 1e-8:
            action_label = "hold"
        else:
            action_label = "skip"
        if current > 1e-8 and hold_quality[idx] > reduce_quality[idx] - 0.03 and exit_urgency[idx] < 0.22 and hold_days[idx] <= max(planned_holding_days[idx], 3.0):
            action_label = "hold"
        if current > 1e-8 and action_label == "reduce" and reduce_reversal_pressure[idx] > 0.22 and drawdown_from_peak[idx] > -0.08:
            action_label = "hold"
        hold_boost = max(hold_utility[idx] - reduce_utility[idx] - exit_utility[idx] * 0.5, 0.0)
        policy_rows.append(
            {
                "stock": stock,
                "action_label": action_label,
                "action_strength": float(max(abs(diff), 0.0)),
                "target_delta_hint": float(np.clip(abs(diff), 0.0, 0.25)),
                "hold_boost": float(np.clip(hold_boost, 0.0, 1.0)),
                "entry_quality": float(entry_quality[idx]),
                "hold_quality": float(hold_quality[idx]),
                "add_quality": float(add_quality[idx]),
                "reduce_quality": float(reduce_quality[idx]),
                "exit_urgency": float(exit_urgency[idx]),
                "reentry_readiness": float(reentry_readiness[idx]),
                "planned_holding_bucket": str(predicted_duration_labels[idx]),
                "planned_holding_days": float(planned_holding_days[idx]),
                "target_weight": float(target),
                "current_weight": float(current),
                "open_utility": float(open_utility[idx]),
                "hold_utility": float(hold_utility[idx]),
                "add_utility": float(add_utility[idx]),
                "reduce_utility": float(reduce_utility[idx]),
                "exit_utility": float(exit_utility[idx]),
                "risk_off_score": float(risk_off_score[idx]),
                "decoder_profile": decoder_profile_name,
            }
        )
    policy_frame = pd.DataFrame(policy_rows).set_index("stock")
    return policy_frame, global_targets
