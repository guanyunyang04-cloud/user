from __future__ import annotations

import hashlib
import json
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from daily_research.continuous_policy.training_contracts import TRAINER_BACKEND_FORMAL_V2


ACTION_CLASSES = ("skip", "open", "hold", "add", "reduce", "exit")
DURATION_CLASSES = ("avoid", "short", "swing", "extended")
ACTION_WEIGHT_BOOST = {"skip": 0.80, "open": 1.10, "hold": 1.80, "add": 1.55, "reduce": 1.45, "exit": 1.45}
HOLDING_DAYS_BY_BUCKET = {"avoid": 0.0, "short": 3.0, "swing": 8.0, "extended": 15.0}
DECODER_PROFILES: dict[str, dict[str, float]] = {
    "default_v2": {
        "defensive_cash_scale": 0.0,
        "candidate_defensive_penalty": 0.0,
        "turnover_defensive_penalty": 0.0,
        "position_cap_defensive_penalty": 0.0,
        "hold_bias_bonus": 0.0,
        "hold_override_margin": 0.0,
        "reduce_gate_bonus": 0.0,
        "open_gate_bonus": 0.0,
        "reentry_penalty": 0.0,
        "hold_delta_bonus": 0.0,
        "reduce_delta_softener": 0.0,
        "reduce_bias_bonus": 0.0,
        "exit_patience_bonus": 0.0,
        "reversal_cooldown_bonus": 0.0,
        "risk_off_open_penalty": 0.0,
    },
    "budget_v3": {
        "defensive_cash_scale": 0.18,
        "candidate_defensive_penalty": 2.0,
        "turnover_defensive_penalty": 0.20,
        "position_cap_defensive_penalty": 0.03,
        "hold_bias_bonus": 0.10,
        "hold_override_margin": 0.02,
        "reduce_gate_bonus": 0.03,
        "open_gate_bonus": 0.015,
        "reentry_penalty": 0.06,
        "hold_delta_bonus": 0.010,
        "reduce_delta_softener": 0.08,
        "reduce_bias_bonus": 0.04,
        "exit_patience_bonus": 0.02,
        "reversal_cooldown_bonus": 0.04,
        "risk_off_open_penalty": 0.01,
    },
    "holdcash_v3": {
        "defensive_cash_scale": 0.28,
        "candidate_defensive_penalty": 3.5,
        "turnover_defensive_penalty": 0.28,
        "position_cap_defensive_penalty": 0.05,
        "hold_bias_bonus": 0.18,
        "hold_override_margin": 0.05,
        "reduce_gate_bonus": 0.06,
        "open_gate_bonus": 0.025,
        "reentry_penalty": 0.10,
        "hold_delta_bonus": 0.020,
        "reduce_delta_softener": 0.14,
        "reduce_bias_bonus": 0.08,
        "exit_patience_bonus": 0.06,
        "reversal_cooldown_bonus": 0.08,
        "risk_off_open_penalty": 0.02,
    },
    "holdcash_v5": {
        "defensive_cash_scale": 0.26,
        "candidate_defensive_penalty": 2.8,
        "turnover_defensive_penalty": 0.34,
        "position_cap_defensive_penalty": 0.05,
        "hold_bias_bonus": 0.24,
        "hold_override_margin": 0.08,
        "reduce_gate_bonus": 0.10,
        "open_gate_bonus": 0.020,
        "reentry_penalty": 0.16,
        "hold_delta_bonus": 0.024,
        "reduce_delta_softener": 0.22,
        "reduce_bias_bonus": 0.04,
        "exit_patience_bonus": 0.12,
        "reversal_cooldown_bonus": 0.20,
        "risk_off_open_penalty": 0.04,
    },
    "reduceexit_v4": {
        "defensive_cash_scale": 0.20,
        "candidate_defensive_penalty": 2.5,
        "turnover_defensive_penalty": 0.18,
        "position_cap_defensive_penalty": 0.04,
        "hold_bias_bonus": 0.12,
        "hold_override_margin": 0.04,
        "reduce_gate_bonus": 0.08,
        "open_gate_bonus": 0.020,
        "reentry_penalty": 0.14,
        "hold_delta_bonus": 0.018,
        "reduce_delta_softener": 0.05,
        "reduce_bias_bonus": 0.16,
        "exit_patience_bonus": 0.12,
        "reversal_cooldown_bonus": 0.12,
        "risk_off_open_penalty": 0.03,
    },
    "cash_v4": {
        "defensive_cash_scale": 0.40,
        "candidate_defensive_penalty": 4.2,
        "turnover_defensive_penalty": 0.36,
        "position_cap_defensive_penalty": 0.06,
        "hold_bias_bonus": 0.14,
        "hold_override_margin": 0.05,
        "reduce_gate_bonus": 0.05,
        "open_gate_bonus": 0.030,
        "reentry_penalty": 0.10,
        "hold_delta_bonus": 0.015,
        "reduce_delta_softener": 0.10,
        "reduce_bias_bonus": 0.08,
        "exit_patience_bonus": 0.08,
        "reversal_cooldown_bonus": 0.10,
        "risk_off_open_penalty": 0.05,
    },
    "reduceexit_cash_v4": {
        "defensive_cash_scale": 0.42,
        "candidate_defensive_penalty": 4.5,
        "turnover_defensive_penalty": 0.34,
        "position_cap_defensive_penalty": 0.07,
        "hold_bias_bonus": 0.16,
        "hold_override_margin": 0.06,
        "reduce_gate_bonus": 0.09,
        "open_gate_bonus": 0.035,
        "reentry_penalty": 0.18,
        "hold_delta_bonus": 0.020,
        "reduce_delta_softener": 0.04,
        "reduce_bias_bonus": 0.18,
        "exit_patience_bonus": 0.15,
        "reversal_cooldown_bonus": 0.18,
        "risk_off_open_penalty": 0.06,
    },
}
DECODER_PROFILE_NAMES = tuple(sorted(DECODER_PROFILES))


def resolve_decoder_profile(value: str | None) -> tuple[str, dict[str, float]]:
    key = str(value or "default_v2").strip().lower() or "default_v2"
    resolved = DECODER_PROFILES.get(key)
    if resolved is None:
        available = ", ".join(sorted(DECODER_PROFILES))
        raise KeyError(f"Unknown continuous_policy decoder profile: {value}. Available profiles: {available}")
    return key, dict(resolved)


class SamplePolicyNet(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 192, dropout: float = 0.10) -> None:
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.action_head = nn.Linear(hidden_dim, len(ACTION_CLASSES))
        self.duration_head = nn.Linear(hidden_dim, len(DURATION_CLASSES))
        self.delta_head = nn.Linear(hidden_dim, 1)
        self.entry_head = nn.Linear(hidden_dim, 1)
        self.hold_head = nn.Linear(hidden_dim, 1)
        self.add_head = nn.Linear(hidden_dim, 1)
        self.reduce_head = nn.Linear(hidden_dim, 1)
        self.exit_head = nn.Linear(hidden_dim, 1)
        self.reentry_head = nn.Linear(hidden_dim, 1)

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        hidden = self.backbone(x)
        return {
            "action_logits": self.action_head(hidden),
            "duration_logits": self.duration_head(hidden),
            "target_delta_hint": self.delta_head(hidden).squeeze(-1),
            "entry_quality": self.entry_head(hidden).squeeze(-1),
            "hold_quality": self.hold_head(hidden).squeeze(-1),
            "add_quality": self.add_head(hidden).squeeze(-1),
            "reduce_quality": self.reduce_head(hidden).squeeze(-1),
            "exit_urgency": self.exit_head(hidden).squeeze(-1),
            "reentry_readiness": self.reentry_head(hidden).squeeze(-1),
        }


class DailyControllerNet(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 96, dropout: float = 0.05) -> None:
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.output = nn.Linear(hidden_dim, 5)

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        raw = self.output(self.backbone(x))
        return {
            "gross_exposure_target": raw[:, 0],
            "candidate_budget": raw[:, 1],
            "turnover_budget": raw[:, 2],
            "max_position_weight_target": raw[:, 3],
            "hold_bias_target": raw[:, 4],
        }


@dataclass
class TorchContinuousPolicyArtifact:
    sample_model: SamplePolicyNet
    daily_model: DailyControllerNet
    feature_names: list[str]
    daily_feature_names: list[str]
    sample_fill_values: np.ndarray
    sample_means: np.ndarray
    sample_stds: np.ndarray
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
            "artifact_type": "continuous_policy_torch_v2",
            "feature_names": self.feature_names,
            "daily_feature_names": self.daily_feature_names,
            "sample_fill_values": self.sample_fill_values.tolist(),
            "sample_means": self.sample_means.tolist(),
            "sample_stds": self.sample_stds.tolist(),
            "daily_fill_values": self.daily_fill_values.tolist(),
            "daily_means": self.daily_means.tolist(),
            "daily_stds": self.daily_stds.tolist(),
            "sample_model_state_dict": self.sample_model.state_dict(),
            "daily_model_state_dict": self.daily_model.state_dict(),
            "sample_model_config": {"input_dim": len(self.feature_names), "hidden_dim": int(self.sample_model.backbone[0].out_features), "dropout": float(self.sample_model.backbone[2].p)},
            "daily_model_config": {"input_dim": len(self.daily_feature_names), "hidden_dim": int(self.daily_model.backbone[0].out_features), "dropout": float(self.daily_model.backbone[2].p)},
            "train_summary": self.train_summary,
            "training_diagnostics": self.training_diagnostics,
            "training_contract": self.training_contract,
            "trained_at": self.trained_at,
        }
        torch.save(payload, path)
        return path


def _safe_std(values: np.ndarray) -> np.ndarray:
    std = np.nanstd(values, axis=0)
    std = np.where(np.isfinite(std) & (std > 1e-6), std, 1.0)
    return std.astype(np.float32)


def _prepare_matrix(frame: pd.DataFrame, feature_names: list[str]) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    raw = frame.loc[:, feature_names].replace([np.inf, -np.inf], np.nan).to_numpy(dtype=np.float32, copy=True)
    fill = np.nanmedian(raw, axis=0)
    fill = np.where(np.isfinite(fill), fill, 0.0).astype(np.float32)
    filled = np.where(np.isfinite(raw), raw, fill[None, :]).astype(np.float32)
    means = np.nanmean(filled, axis=0)
    means = np.where(np.isfinite(means), means, 0.0).astype(np.float32)
    stds = _safe_std(filled)
    normalized = ((filled - means[None, :]) / stds[None, :]).astype(np.float32)
    return normalized, fill, means, stds


def _apply_matrix(values: pd.DataFrame, feature_names: list[str], fill: np.ndarray, means: np.ndarray, stds: np.ndarray) -> np.ndarray:
    raw = values.loc[:, feature_names].replace([np.inf, -np.inf], np.nan).to_numpy(dtype=np.float32, copy=True)
    filled = np.where(np.isfinite(raw), raw, fill[None, :]).astype(np.float32)
    return ((filled - means[None, :]) / stds[None, :]).astype(np.float32)


def _finite_scalar(value: float | np.ndarray | torch.Tensor, *, default: float) -> float:
    if isinstance(value, torch.Tensor):
        arr = value.detach().cpu().numpy()
        scalar = float(arr.reshape(-1)[0]) if arr.size else float(default)
    elif isinstance(value, np.ndarray):
        scalar = float(value.reshape(-1)[0]) if value.size else float(default)
    else:
        scalar = float(value)
    return scalar if np.isfinite(scalar) else float(default)


def _daily_feature_scalar(daily_features: dict[str, float], key: str, *, default: float = 0.0) -> float:
    raw = daily_features.get(key, default)
    try:
        value = float(raw)
    except Exception:
        value = float(default)
    return value if np.isfinite(value) else float(default)


def _signature_payload(*, feature_names: list[str], daily_feature_names: list[str], train_summary: dict[str, Any], training_contract: dict[str, Any]) -> dict[str, Any]:
    contract = dict(training_contract or {})
    summary = dict(train_summary or {})
    return {
        "trainer_backend": str(contract.get("trainer_backend", "") or ""),
        "contract_class": str(contract.get("contract_class", "") or ""),
        "pool_name": str(summary.get("pool_name", "") or ""),
        "benchmark": str(summary.get("benchmark", "") or ""),
        "start_date": str(summary.get("start_date", "") or ""),
        "end_date": str(summary.get("end_date", "") or ""),
        "label_preset": str(summary.get("label_preset", "") or ""),
        "feature_names": list(feature_names),
        "daily_feature_names": list(daily_feature_names),
    }


def _signature_hash(payload: dict[str, Any]) -> str:
    text = json.dumps(payload, ensure_ascii=False, sort_keys=True)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _resolve_device(training_contract: dict[str, Any]) -> torch.device:
    wants_gpu = bool(training_contract.get("gpu_required", False))
    if wants_gpu and not torch.cuda.is_available():
        raise RuntimeError("continuous_policy formal_torch_v2 requires CUDA, but torch.cuda.is_available() is False.")
    if wants_gpu:
        return torch.device("cuda")
    return torch.device("cpu")


def _action_weights(sample_frame: pd.DataFrame) -> torch.Tensor:
    counts = sample_frame["action_label"].astype(str).value_counts()
    weights = []
    for label in ACTION_CLASSES:
        count = max(float(counts.get(label, 1.0)), 1.0)
        weights.append(float(ACTION_WEIGHT_BOOST.get(label, 1.0)) / count)
    arr = np.asarray(weights, dtype=np.float32)
    arr = arr / max(float(arr.mean()), 1e-6)
    return torch.as_tensor(arr, dtype=torch.float32)


def _split_indices(size: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    if size <= 8:
        idx = np.arange(size, dtype=int)
        return idx, idx
    rng = np.random.default_rng(int(seed))
    shuffled = np.arange(size, dtype=int)
    rng.shuffle(shuffled)
    split = max(1, int(round(size * 0.12)))
    val_idx = np.sort(shuffled[:split])
    train_idx = np.sort(shuffled[split:])
    if len(train_idx) == 0:
        train_idx = val_idx
    return train_idx, val_idx


def _scalar_heads_loss(outputs: dict[str, torch.Tensor], targets: dict[str, torch.Tensor]) -> torch.Tensor:
    loss = torch.tensor(0.0, device=next(iter(outputs.values())).device)
    for name, target in targets.items():
        loss = loss + nn.functional.smooth_l1_loss(outputs[name], target)
    return loss


def _save_checkpoint(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, path)


def load_torch_artifact(path: str | Path) -> TorchContinuousPolicyArtifact:
    payload = torch.load(Path(path), map_location="cpu", weights_only=False)
    if str(payload.get("artifact_type", "") or "") != "continuous_policy_torch_v2":
        raise TypeError(f"Unsupported torch artifact type: {payload.get('artifact_type')!r}")
    sample_cfg = dict(payload.get("sample_model_config", {}) or {})
    daily_cfg = dict(payload.get("daily_model_config", {}) or {})
    sample_model = SamplePolicyNet(**sample_cfg)
    daily_model = DailyControllerNet(**daily_cfg)
    sample_model.load_state_dict(payload["sample_model_state_dict"])
    daily_model.load_state_dict(payload["daily_model_state_dict"])
    sample_model.eval()
    daily_model.eval()
    return TorchContinuousPolicyArtifact(
        sample_model=sample_model,
        daily_model=daily_model,
        feature_names=list(payload.get("feature_names", []) or []),
        daily_feature_names=list(payload.get("daily_feature_names", []) or []),
        sample_fill_values=np.asarray(payload.get("sample_fill_values", []), dtype=np.float32),
        sample_means=np.asarray(payload.get("sample_means", []), dtype=np.float32),
        sample_stds=np.asarray(payload.get("sample_stds", []), dtype=np.float32),
        daily_fill_values=np.asarray(payload.get("daily_fill_values", []), dtype=np.float32),
        daily_means=np.asarray(payload.get("daily_means", []), dtype=np.float32),
        daily_stds=np.asarray(payload.get("daily_stds", []), dtype=np.float32),
        train_summary=dict(payload.get("train_summary", {}) or {}),
        training_diagnostics=dict(payload.get("training_diagnostics", {}) or {}),
        training_contract=dict(payload.get("training_contract", {}) or {}),
        trained_at=str(payload.get("trained_at", "") or ""),
    )


def fit_policy_models_v2(
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
    batch_size: int = 512,
    learning_rate: float = 1.5e-3,
    hidden_dim: int = 192,
    daily_hidden_dim: int = 96,
    dropout: float = 0.10,
    daily_dropout: float = 0.05,
    early_stop_patience: int = 10,
    resume_mode: str = "strict",
) -> TorchContinuousPolicyArtifact:
    if sample_frame.empty or daily_frame.empty:
        raise ValueError("continuous_policy formal_torch_v2 received empty training data.")
    contract = dict(training_contract or {})
    if str(contract.get("trainer_backend", "") or "") != TRAINER_BACKEND_FORMAL_V2:
        raise ValueError("fit_policy_models_v2 requires the formal_torch_v2 training contract.")
    device = _resolve_device(contract)
    torch.manual_seed(int(random_seed))
    np.random.seed(int(random_seed))
    run_root.mkdir(parents=True, exist_ok=True)

    X_sample, sample_fill, sample_means, sample_stds = _prepare_matrix(sample_frame, feature_names)
    X_daily, daily_fill, daily_means, daily_stds = _prepare_matrix(daily_frame, daily_feature_names)

    action_lookup = {name: idx for idx, name in enumerate(ACTION_CLASSES)}
    duration_lookup = {name: idx for idx, name in enumerate(DURATION_CLASSES)}
    y_action = np.asarray([action_lookup.get(str(value), 0) for value in sample_frame["action_label"].astype(str)], dtype=np.int64)
    y_duration = np.asarray([
        duration_lookup.get(str(value), 0)
        for value in sample_frame["planned_holding_bucket"].astype(str).where(sample_frame["planned_holding_bucket"].astype(str).isin(DURATION_CLASSES), "avoid")
    ], dtype=np.int64)

    sample_targets = {
        "target_delta_hint": sample_frame["target_delta_hint"].astype(float).to_numpy(dtype=np.float32),
        "entry_quality": sample_frame["entry_quality"].astype(float).to_numpy(dtype=np.float32),
        "hold_quality": sample_frame["hold_quality"].astype(float).to_numpy(dtype=np.float32),
        "add_quality": sample_frame["add_quality"].astype(float).to_numpy(dtype=np.float32),
        "reduce_quality": sample_frame["reduce_quality"].astype(float).to_numpy(dtype=np.float32),
        "exit_urgency": sample_frame["exit_urgency"].astype(float).to_numpy(dtype=np.float32),
        "reentry_readiness": sample_frame["reentry_readiness"].astype(float).to_numpy(dtype=np.float32),
    }
    daily_targets = {
        "gross_exposure_target": daily_frame["gross_exposure_target"].astype(float).to_numpy(dtype=np.float32),
        "candidate_budget": daily_frame["candidate_budget"].astype(float).to_numpy(dtype=np.float32),
        "turnover_budget": daily_frame["turnover_budget"].astype(float).to_numpy(dtype=np.float32),
        "max_position_weight_target": daily_frame["max_position_weight_target"].astype(float).to_numpy(dtype=np.float32),
        "hold_bias_target": daily_frame["hold_bias_target"].astype(float).to_numpy(dtype=np.float32),
    }

    train_idx, val_idx = _split_indices(len(sample_frame), random_seed)
    daily_train_idx, daily_val_idx = _split_indices(len(daily_frame), random_seed + 17)

    sample_model = SamplePolicyNet(len(feature_names), hidden_dim=hidden_dim, dropout=dropout).to(device)
    daily_model = DailyControllerNet(len(daily_feature_names), hidden_dim=daily_hidden_dim, dropout=daily_dropout).to(device)
    sample_optimizer = torch.optim.AdamW(sample_model.parameters(), lr=float(learning_rate), weight_decay=1e-4)
    daily_optimizer = torch.optim.AdamW(daily_model.parameters(), lr=float(learning_rate), weight_decay=1e-4)
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
    artifact_path = run_root / "continuous_policy_v2_artifact.pt"
    diagnostics_path = run_root / "training_diagnostics.json"

    start_epoch = 0
    best_epoch = 0
    best_val_loss = math.inf
    resumed_from = ""
    history: list[dict[str, float | int]] = []
    if checkpoint_last.exists() and resume_mode == "strict":
        state = torch.load(checkpoint_last, map_location="cpu", weights_only=False)
        previous_hash = str(state.get("signature_hash", "") or "")
        if resume_mode == "strict" and previous_hash and previous_hash != signature_hash:
            raise RuntimeError("continuous_policy v2 strict resume rejected because checkpoint lineage does not match the current run signature.")
        sample_model.load_state_dict(state["sample_model_state_dict"])
        daily_model.load_state_dict(state["daily_model_state_dict"])
        sample_optimizer.load_state_dict(state["sample_optimizer_state_dict"])
        daily_optimizer.load_state_dict(state["daily_optimizer_state_dict"])
        start_epoch = int(state.get("epoch", 0) or 0)
        best_epoch = int(state.get("best_epoch", 0) or 0)
        best_val_loss = float(state.get("best_val_loss", math.inf) or math.inf)
        history = list(state.get("history", []) or [])
        resumed_from = str(checkpoint_last.resolve())
    if resume_mode == "strict" and checkpoint_last.exists() and int(epochs) <= int(start_epoch):
        raise RuntimeError(f"continuous_policy v2 strict resume requires epochs > completed epochs ({start_epoch}), got {epochs}.")

    dataset = TensorDataset(
        torch.as_tensor(X_sample[train_idx], dtype=torch.float32),
        torch.as_tensor(y_action[train_idx], dtype=torch.long),
        torch.as_tensor(y_duration[train_idx], dtype=torch.long),
        *[torch.as_tensor(sample_targets[name][train_idx], dtype=torch.float32) for name in sample_targets],
    )
    loader = DataLoader(dataset, batch_size=max(32, int(batch_size)), shuffle=True, drop_last=False)
    X_val = torch.as_tensor(X_sample[val_idx], dtype=torch.float32, device=device)
    y_action_val = torch.as_tensor(y_action[val_idx], dtype=torch.long, device=device)
    y_duration_val = torch.as_tensor(y_duration[val_idx], dtype=torch.long, device=device)
    val_targets = {name: torch.as_tensor(values[val_idx], dtype=torch.float32, device=device) for name, values in sample_targets.items()}
    X_daily_train = torch.as_tensor(X_daily[daily_train_idx], dtype=torch.float32, device=device)
    X_daily_val = torch.as_tensor(X_daily[daily_val_idx], dtype=torch.float32, device=device)
    daily_targets_train = {name: torch.as_tensor(values[daily_train_idx], dtype=torch.float32, device=device) for name, values in daily_targets.items()}
    daily_targets_val = {name: torch.as_tensor(values[daily_val_idx], dtype=torch.float32, device=device) for name, values in daily_targets.items()}

    patience_used = 0
    for epoch in range(start_epoch + 1, int(epochs) + 1):
        sample_model.train()
        daily_model.train()
        epoch_sample_loss = 0.0
        batch_count = 0
        for batch in loader:
            batch = [item.to(device) for item in batch]
            x_batch, action_batch, duration_batch, delta_batch, entry_batch, hold_batch, add_batch, reduce_batch, exit_batch, reentry_batch = batch
            outputs = sample_model(x_batch)
            action_loss = nn.functional.cross_entropy(outputs["action_logits"], action_batch, weight=action_weight_tensor)
            duration_loss = nn.functional.cross_entropy(outputs["duration_logits"], duration_batch)
            scalar_loss = _scalar_heads_loss(
                outputs,
                {
                    "target_delta_hint": delta_batch,
                    "entry_quality": entry_batch,
                    "hold_quality": hold_batch,
                    "add_quality": add_batch,
                    "reduce_quality": reduce_batch,
                    "exit_urgency": exit_batch,
                    "reentry_readiness": reentry_batch,
                },
            )
            loss = action_loss + 0.55 * duration_loss + 0.40 * scalar_loss
            sample_optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(sample_model.parameters(), max_norm=2.0)
            sample_optimizer.step()
            epoch_sample_loss += float(loss.detach().cpu())
            batch_count += 1

        daily_outputs = daily_model(X_daily_train)
        daily_loss = _scalar_heads_loss(daily_outputs, daily_targets_train)
        daily_optimizer.zero_grad(set_to_none=True)
        daily_loss.backward()
        nn.utils.clip_grad_norm_(daily_model.parameters(), max_norm=2.0)
        daily_optimizer.step()

        sample_model.eval()
        daily_model.eval()
        with torch.no_grad():
            val_outputs = sample_model(X_val)
            val_action_loss = nn.functional.cross_entropy(val_outputs["action_logits"], y_action_val, weight=action_weight_tensor)
            val_duration_loss = nn.functional.cross_entropy(val_outputs["duration_logits"], y_duration_val)
            val_scalar_loss = _scalar_heads_loss(val_outputs, val_targets)
            val_daily_outputs = daily_model(X_daily_val)
            val_daily_loss = _scalar_heads_loss(val_daily_outputs, daily_targets_val)
            val_loss = float((val_action_loss + 0.55 * val_duration_loss + 0.40 * val_scalar_loss + 0.30 * val_daily_loss).detach().cpu())

        train_loss = float(epoch_sample_loss / max(batch_count, 1) + 0.30 * float(daily_loss.detach().cpu()))
        history.append({"epoch": epoch, "train_loss": train_loss, "validation_loss": val_loss})
        checkpoint_payload = {
            "signature_hash": signature_hash,
            "signature_payload": signature_payload,
            "epoch": epoch,
            "best_epoch": best_epoch,
            "best_val_loss": best_val_loss,
            "history": history,
            "sample_model_state_dict": sample_model.state_dict(),
            "daily_model_state_dict": daily_model.state_dict(),
            "sample_optimizer_state_dict": sample_optimizer.state_dict(),
            "daily_optimizer_state_dict": daily_optimizer.state_dict(),
        }
        _save_checkpoint(checkpoint_last, checkpoint_payload)
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch
            patience_used = 0
            checkpoint_payload["best_epoch"] = best_epoch
            checkpoint_payload["best_val_loss"] = best_val_loss
            _save_checkpoint(checkpoint_best, checkpoint_payload)
        else:
            patience_used += 1
        if epoch >= max(int(min_epochs), 1) and patience_used >= max(int(early_stop_patience), 1):
            break

    best_state = torch.load(checkpoint_best if checkpoint_best.exists() else checkpoint_last, map_location="cpu", weights_only=False)
    sample_model.load_state_dict(best_state["sample_model_state_dict"])
    daily_model.load_state_dict(best_state["daily_model_state_dict"])
    sample_model.eval()
    daily_model.eval()

    diagnostics = {
        "trainer_backend": TRAINER_BACKEND_FORMAL_V2,
        "device": str(device),
        "cuda_available": bool(torch.cuda.is_available()),
        "epochs_requested": int(epochs),
        "min_epochs": int(min_epochs),
        "completed_epochs": int(history[-1]["epoch"]) if history else 0,
        "best_epoch": int(best_state.get("best_epoch", best_epoch) or 0),
        "best_validation_loss": float(best_state.get("best_val_loss", best_val_loss) or 0.0),
        "resume_mode": str(resume_mode or ""),
        "resumed_from_checkpoint": resumed_from,
        "checkpoint_last": str(checkpoint_last.resolve()),
        "checkpoint_best": str(checkpoint_best.resolve()) if checkpoint_best.exists() else str(checkpoint_last.resolve()),
        "training_diagnostics_json": str(diagnostics_path.resolve()),
        "signature_hash": signature_hash,
        "history_tail": history[-8:],
    }
    artifact = TorchContinuousPolicyArtifact(
        sample_model=sample_model.cpu(),
        daily_model=daily_model.cpu(),
        feature_names=list(feature_names),
        daily_feature_names=list(daily_feature_names),
        sample_fill_values=sample_fill,
        sample_means=sample_means,
        sample_stds=sample_stds,
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


def predict_policy_v2(
    artifact: TorchContinuousPolicyArtifact,
    *,
    state_frame: pd.DataFrame,
    daily_features: dict[str, float],
) -> tuple[pd.DataFrame, dict[str, float]]:
    if state_frame.empty:
        raise ValueError("state_frame is empty.")
    with torch.no_grad():
        sample_x = torch.as_tensor(
            _apply_matrix(state_frame, artifact.feature_names, artifact.sample_fill_values, artifact.sample_means, artifact.sample_stds),
            dtype=torch.float32,
        )
        outputs = artifact.sample_model(sample_x)
        action_prob = torch.softmax(outputs["action_logits"], dim=-1).cpu().numpy()
        duration_prob = torch.softmax(outputs["duration_logits"], dim=-1).cpu().numpy()
        predicted_labels = np.asarray(ACTION_CLASSES, dtype=object)[action_prob.argmax(axis=1)]
        predicted_duration_labels = np.asarray(DURATION_CLASSES, dtype=object)[duration_prob.argmax(axis=1)]
        delta_hint = outputs["target_delta_hint"].cpu().numpy()
        entry_quality = outputs["entry_quality"].cpu().numpy()
        hold_quality = outputs["hold_quality"].cpu().numpy()
        add_quality = outputs["add_quality"].cpu().numpy()
        reduce_quality = outputs["reduce_quality"].cpu().numpy()
        exit_urgency = np.clip(outputs["exit_urgency"].cpu().numpy(), 0.0, None)
        reentry_readiness = np.clip(outputs["reentry_readiness"].cpu().numpy(), 0.0, None)

        daily_row = pd.DataFrame([{name: float(daily_features.get(name, 0.0) or 0.0) for name in artifact.daily_feature_names}])
        daily_x = torch.as_tensor(
            _apply_matrix(daily_row, artifact.daily_feature_names, artifact.daily_fill_values, artifact.daily_means, artifact.daily_stds),
            dtype=torch.float32,
        )
        daily_out = artifact.daily_model(daily_x)
        global_targets = {
            "gross_exposure_target": float(np.clip(_finite_scalar(daily_out["gross_exposure_target"], default=0.35), 0.15, 0.98)),
            "candidate_budget": float(np.clip(_finite_scalar(daily_out["candidate_budget"], default=4.0), 2.0, 12.0)),
            "turnover_budget": float(np.clip(_finite_scalar(daily_out["turnover_budget"], default=0.18), 0.08, 1.00)),
            "max_position_weight_target": float(np.clip(_finite_scalar(daily_out["max_position_weight_target"], default=0.12), 0.08, 0.28)),
            "hold_bias_target": float(np.clip(_finite_scalar(daily_out["hold_bias_target"], default=0.24), 0.10, 0.95)),
        }

    decoder_profile_name, decoder_profile = resolve_decoder_profile(artifact.train_summary.get("decoder_profile"))
    defensive_score = (
        0.40 * max(-_daily_feature_scalar(daily_features, "benchmark_trend_gap"), 0.0)
        + 0.15 * max(_daily_feature_scalar(daily_features, "benchmark_vol_ratio"), 0.0)
        + 0.15 * max(-_daily_feature_scalar(daily_features, "portfolio_drawdown_20d") - 0.02, 0.0)
        + 0.10 * max(_daily_feature_scalar(daily_features, "portfolio_turnover_pressure") - 0.80, 0.0)
    )
    risk_off_score = (
        defensive_score
        + 0.25 * max(_daily_feature_scalar(daily_features, "market_downside_pressure"), 0.0)
        + 0.20 * max(_daily_feature_scalar(daily_features, "portfolio_cash_pressure"), 0.0)
        + 0.18 * max(_daily_feature_scalar(daily_features, "recent_reversal_rate_20d") - 0.18, 0.0)
    )
    global_targets["gross_exposure_target"] = float(
        np.clip(
            global_targets["gross_exposure_target"] - risk_off_score * decoder_profile["defensive_cash_scale"],
            0.12,
            0.98,
        )
    )
    global_targets["candidate_budget"] = float(
        np.clip(
            global_targets["candidate_budget"] - risk_off_score * decoder_profile["candidate_defensive_penalty"],
            2.0,
            12.0,
        )
    )
    global_targets["turnover_budget"] = float(
        np.clip(
            global_targets["turnover_budget"] * (1.0 - risk_off_score * decoder_profile["turnover_defensive_penalty"]),
            0.08,
            1.00,
        )
    )
    global_targets["max_position_weight_target"] = float(
        np.clip(
            global_targets["max_position_weight_target"] - risk_off_score * decoder_profile["position_cap_defensive_penalty"],
            0.08,
            0.28,
        )
    )
    global_targets["hold_bias_target"] = float(
        np.clip(
            global_targets["hold_bias_target"] + risk_off_score * decoder_profile["hold_bias_bonus"] - _daily_feature_scalar(daily_features, "recent_reversal_rate_20d") * 0.04,
            0.10,
            0.95,
        )
    )
    global_targets["reduce_bias_target"] = float(
        np.clip(
            0.10
            + risk_off_score * 0.28
            + _daily_feature_scalar(daily_features, "recent_reversal_rate_20d") * 0.12
            + decoder_profile["reduce_bias_bonus"],
            0.0,
            0.65,
        )
    )
    global_targets["exit_patience_target"] = float(
        np.clip(
            global_targets["hold_bias_target"] + decoder_profile["exit_patience_bonus"] - risk_off_score * 0.20,
            0.05,
            0.95,
        )
    )
    global_targets["reentry_guard_target"] = float(
        np.clip(
            decoder_profile["reversal_cooldown_bonus"]
            + _daily_feature_scalar(daily_features, "recent_reversal_rate_20d") * 0.20
            + risk_off_score * 0.18,
            0.0,
            0.45,
        )
    )
    global_targets = {
        "gross_exposure_target": float(np.clip(_finite_scalar(global_targets["gross_exposure_target"], default=0.35), 0.15, 0.98)),
        "candidate_budget": float(np.clip(_finite_scalar(global_targets["candidate_budget"], default=4.0), 2.0, 12.0)),
        "turnover_budget": float(np.clip(_finite_scalar(global_targets["turnover_budget"], default=0.18), 0.08, 1.00)),
        "max_position_weight_target": float(np.clip(_finite_scalar(global_targets["max_position_weight_target"], default=0.12), 0.08, 0.28)),
        "hold_bias_target": float(np.clip(_finite_scalar(global_targets["hold_bias_target"], default=0.24), 0.10, 0.95)),
        "reduce_bias_target": float(np.clip(_finite_scalar(global_targets["reduce_bias_target"], default=0.10), 0.0, 0.65)),
        "exit_patience_target": float(np.clip(_finite_scalar(global_targets["exit_patience_target"], default=0.20), 0.05, 0.95)),
        "reentry_guard_target": float(np.clip(_finite_scalar(global_targets["reentry_guard_target"], default=0.0), 0.0, 0.45)),
    }

    probability_map = {label: action_prob[:, idx] for idx, label in enumerate(ACTION_CLASSES)}
    current_weight = state_frame["current_weight"].astype(float).to_numpy(dtype=float) if "current_weight" in state_frame.columns else np.zeros(len(state_frame), dtype=float)
    hold_days = state_frame["hold_days"].astype(float).to_numpy(dtype=float) if "hold_days" in state_frame.columns else np.zeros(len(state_frame), dtype=float)
    days_since_last_buy = state_frame["days_since_last_buy"].astype(float).to_numpy(dtype=float) if "days_since_last_buy" in state_frame.columns else np.full(len(state_frame), 99.0, dtype=float)
    days_since_last_reduce = state_frame["days_since_last_reduce"].astype(float).to_numpy(dtype=float) if "days_since_last_reduce" in state_frame.columns else np.full(len(state_frame), 99.0, dtype=float)
    days_since_last_exit = state_frame["days_since_last_exit"].astype(float).to_numpy(dtype=float) if "days_since_last_exit" in state_frame.columns else np.full(len(state_frame), 99.0, dtype=float)
    reentry_cooldown = state_frame["reentry_cooldown"].astype(float).to_numpy(dtype=float) if "reentry_cooldown" in state_frame.columns else np.zeros(len(state_frame), dtype=float)
    signal_decay_speed = state_frame["signal_decay_speed"].astype(float).to_numpy(dtype=float) if "signal_decay_speed" in state_frame.columns else np.zeros(len(state_frame), dtype=float)
    market_downside_pressure = state_frame["market_downside_pressure"].astype(float).to_numpy(dtype=float) if "market_downside_pressure" in state_frame.columns else np.zeros(len(state_frame), dtype=float)
    portfolio_cash_pressure = state_frame["portfolio_cash_pressure"].astype(float).to_numpy(dtype=float) if "portfolio_cash_pressure" in state_frame.columns else np.zeros(len(state_frame), dtype=float)
    recent_reversal_rate = state_frame["recent_reversal_rate_20d"].astype(float).to_numpy(dtype=float) if "recent_reversal_rate_20d" in state_frame.columns else np.zeros(len(state_frame), dtype=float)
    drawdown_from_peak = state_frame["drawdown_from_peak"].astype(float).to_numpy(dtype=float) if "drawdown_from_peak" in state_frame.columns else np.zeros(len(state_frame), dtype=float)
    duration_days = np.asarray([HOLDING_DAYS_BY_BUCKET.get(str(label), 0.0) for label in predicted_duration_labels], dtype=float)
    adjusted_labels = predicted_labels.astype(object).copy()
    reduce_bias_target = float(global_targets["reduce_bias_target"])
    exit_patience_target = float(global_targets["exit_patience_target"])
    reentry_guard_target = float(global_targets["reentry_guard_target"])
    for idx in range(len(adjusted_labels)):
        label = str(adjusted_labels[idx])
        held = float(current_weight[idx]) > 1e-8
        duration_name = str(predicted_duration_labels[idx])
        if held:
            if label in {"reduce", "exit"} and market_downside_pressure[idx] < 0.12 and signal_decay_speed[idx] < 0.05 and drawdown_from_peak[idx] > -0.05 and hold_quality[idx] > reduce_quality[idx] - decoder_profile["hold_override_margin"]:
                label = "hold"
            if label in {"reduce", "exit"} and hold_days[idx] < 2 and exit_urgency[idx] < 0.24 and hold_quality[idx] > -0.02:
                label = "hold"
            if label in {"reduce", "exit"} and days_since_last_buy[idx] <= max(3.0, hold_days[idx]) and exit_urgency[idx] < 0.26 and hold_quality[idx] > -0.04:
                label = "hold"
            if label == "reduce" and days_since_last_reduce[idx] <= 2.0 and hold_quality[idx] > reduce_quality[idx] - decoder_profile["hold_override_margin"]:
                label = "hold"
            if label == "exit" and exit_urgency[idx] < 0.18 + exit_patience_target * 0.06 and hold_quality[idx] > reduce_quality[idx] - decoder_profile["reduce_gate_bonus"]:
                label = "reduce" if reduce_quality[idx] > 0.08 else "hold"
            if label == "reduce" and reduce_quality[idx] < 0.09 + decoder_profile["reduce_gate_bonus"] and hold_quality[idx] > 0.03 - decoder_profile["hold_override_margin"]:
                label = "hold"
            if label in {"reduce", "exit"} and duration_name in {"swing", "extended"} and hold_quality[idx] >= reduce_quality[idx] - decoder_profile["hold_override_margin"]:
                label = "hold"
            if label in {"hold", "skip"} and (market_downside_pressure[idx] > 0.18 or portfolio_cash_pressure[idx] > 0.18 or signal_decay_speed[idx] > 0.10) and hold_days[idx] >= 3.0 and current_weight[idx] > 0.02:
                label = "reduce"
            if label in {"hold", "skip"} and add_quality[idx] > 0.14 and duration_name in {"swing", "extended"} and current_weight[idx] < 0.12:
                label = "add"
        else:
            open_gate = (
                0.08
                + risk_off_score * decoder_profile["open_gate_bonus"]
                + reentry_cooldown[idx] * decoder_profile["reentry_penalty"]
                + reentry_guard_target
                + market_downside_pressure[idx] * decoder_profile["risk_off_open_penalty"]
            )
            if label == "open" and (entry_quality[idx] < open_gate or duration_name == "avoid"):
                label = "skip"
            if label in {"skip", "hold"} and duration_name != "avoid" and (entry_quality[idx] > (0.075 + defensive_score * 0.01) or (entry_quality[idx] > 0.055 and probability_map["open"][idx] > 0.035)):
                if probability_map["open"][idx] > 0.035 or duration_name in {"swing", "extended"}:
                    if (reentry_cooldown[idx] <= 0.25 and days_since_last_exit[idx] > 3.0) or entry_quality[idx] > open_gate + 0.035:
                        label = "open"
        adjusted_labels[idx] = label

    candidate_budget_target = max(1, int(round(global_targets["candidate_budget"])))
    open_candidate_scores = entry_quality + probability_map["open"] * 0.45 + np.clip(duration_days - 3.0, 0.0, None) / 30.0
    open_candidate_scores = np.where(current_weight > 1e-8, -1e9, open_candidate_scores)
    open_candidate_scores = np.where(np.asarray(predicted_duration_labels) == "avoid", -1e9, open_candidate_scores)
    active_candidates = sum(1 for idx, label in enumerate(adjusted_labels) if str(label) in {"open", "add", "hold"} or current_weight[idx] > 1e-8)
    missing_candidates = max(0, min(candidate_budget_target, len(adjusted_labels)) - active_candidates)
    if missing_candidates > 0 and np.isfinite(open_candidate_scores).any():
        promoted = 0
        for idx in np.argsort(open_candidate_scores)[::-1]:
            if promoted >= missing_candidates or float(open_candidate_scores[idx]) < 0.06 or current_weight[idx] > 1e-8:
                continue
            adjusted_labels[idx] = "open"
            promoted += 1

    blended_delta = np.asarray(delta_hint, dtype=float).copy()
    action_strength = np.zeros(len(state_frame), dtype=float)
    hold_boost = np.zeros(len(state_frame), dtype=float)
    for idx, label in enumerate(adjusted_labels):
        duration_bonus = max(duration_days[idx] - 3.0, 0.0) / 20.0
        if label == "open":
            blended_delta[idx] = np.clip(max(blended_delta[idx], 0.02 + entry_quality[idx] * 0.55 + duration_bonus * 0.05), 0.0, 0.22)
            action_strength[idx] = np.clip(entry_quality[idx] + probability_map["open"][idx] * 0.55 + duration_bonus * 0.40, 0.0, None)
            hold_boost[idx] = np.clip(reentry_readiness[idx] * 0.25 + duration_bonus * 0.20, 0.0, None)
        elif label == "add":
            blended_delta[idx] = np.clip(max(blended_delta[idx], 0.01 + add_quality[idx] * 0.35 + duration_bonus * 0.03), 0.0, 0.18)
            action_strength[idx] = np.clip(add_quality[idx] + probability_map["add"][idx] * 0.45 + duration_bonus * 0.30, 0.0, None)
            hold_boost[idx] = np.clip(hold_quality[idx] + duration_bonus * 0.25, 0.0, None)
        elif label == "hold":
            blended_delta[idx] = np.clip(max(blended_delta[idx] * 0.30, 0.0) + hold_quality[idx] * 0.10 + decoder_profile["hold_delta_bonus"], 0.0, 0.08 + decoder_profile["hold_delta_bonus"])
            action_strength[idx] = np.clip(hold_quality[idx] + probability_map["hold"][idx] * 0.35 + duration_bonus * 0.25, 0.0, None)
            hold_boost[idx] = np.clip(hold_quality[idx] + duration_bonus * 0.30 + decoder_profile["hold_bias_bonus"] * (0.20 + exit_patience_target), 0.0, None)
        elif label == "reduce":
            blended_delta[idx] = -np.clip(
                max(
                    -blended_delta[idx],
                    0.08
                    + reduce_quality[idx] * (0.28 + reduce_bias_target - decoder_profile["reduce_delta_softener"])
                    + market_downside_pressure[idx] * 0.16
                    + signal_decay_speed[idx] * 0.18
                    - hold_quality[idx] * 0.08,
                ),
                0.0,
                0.75,
            )
            action_strength[idx] = np.clip(reduce_quality[idx] + probability_map["reduce"][idx] * 0.35 + reduce_bias_target * 0.25, 0.0, None)
        elif label == "exit":
            blended_delta[idx] = -1.0
            action_strength[idx] = np.clip(
                exit_urgency[idx]
                + probability_map["exit"][idx] * 0.45
                + market_downside_pressure[idx] * 0.12
                + signal_decay_speed[idx] * 0.15
                - exit_patience_target * 0.10,
                0.0,
                None,
            )
        else:
            blended_delta[idx] = 0.0
            action_strength[idx] = probability_map["skip"][idx] * 0.20

    policy = pd.DataFrame(
        {
            "stock": state_frame["stock"].astype(str).to_numpy(),
            "action_label": adjusted_labels,
            "action_strength": action_strength,
            "target_delta_hint": blended_delta,
            "hold_boost": hold_boost,
            "exit_urgency": exit_urgency + probability_map["exit"] * 0.55 + probability_map["reduce"] * 0.35 + np.clip(-blended_delta, 0.0, None),
            "entry_quality": entry_quality,
            "hold_quality": hold_quality,
            "add_quality": add_quality,
            "reduce_quality": reduce_quality,
            "reentry_readiness": reentry_readiness,
            "planned_holding_bucket": predicted_duration_labels,
            "planned_holding_days": duration_days,
            "decoder_profile": decoder_profile_name,
        }
    ).set_index("stock")
    for label in ACTION_CLASSES:
        policy[f"prob_{label}"] = probability_map[label]
    return policy, global_targets
