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
from torch.utils.data import DataLoader, TensorDataset

from daily_research.continuous_policy.model_v2 import (
    ACTION_CLASSES,
    DURATION_CLASSES,
    HOLDING_DAYS_BY_BUCKET,
    _action_weights,
    _apply_matrix,
    _daily_feature_scalar,
    _finite_scalar,
    _prepare_matrix,
    _resolve_device,
    _save_checkpoint,
    _scalar_heads_loss,
    _signature_hash,
    _signature_payload,
    _split_indices,
    resolve_decoder_profile,
)
from daily_research.continuous_policy.state_builder import STATE_SEQUENCE_BASES, STATE_SEQUENCE_LAGS
from daily_research.continuous_policy.training_contracts import TRAINER_BACKEND_FORMAL_SEQ_V3


SEQUENCE_STEP_ORDER: tuple[int, ...] = tuple(sorted(STATE_SEQUENCE_LAGS, reverse=True)) + (0,)


def _sequence_column_name(base_name: str, step: int) -> str:
    return str(base_name) if int(step) == 0 else f"{base_name}_lag{int(step)}"


def resolve_sequence_columns(feature_names: list[str]) -> tuple[list[str], list[str]]:
    available = set(feature_names)
    sequence_columns: list[str] = []
    sequence_bases: list[str] = []
    for base_name in STATE_SEQUENCE_BASES:
        required = [_sequence_column_name(base_name, step) for step in SEQUENCE_STEP_ORDER]
        if all(column in available for column in required):
            sequence_bases.append(str(base_name))
            sequence_columns.extend(required)
    if not sequence_bases:
        raise ValueError("formal_torch_seq_v3 requires lagged sequence features, but none were found in the training matrix.")
    return sequence_bases, sequence_columns


class TemporalSamplePolicyNet(nn.Module):
    def __init__(
        self,
        *,
        static_input_dim: int,
        sequence_feature_dim: int,
        sequence_steps: int,
        hidden_dim: int = 224,
        sequence_hidden_dim: int = 128,
        sequence_layers: int = 1,
        dropout: float = 0.10,
    ) -> None:
        super().__init__()
        self.sequence_steps = int(sequence_steps)
        self.sequence_feature_dim = int(sequence_feature_dim)
        self.sequence_layers = max(1, int(sequence_layers))
        self.sequence_encoder = nn.GRU(
            input_size=self.sequence_feature_dim,
            hidden_size=int(sequence_hidden_dim),
            num_layers=self.sequence_layers,
            batch_first=True,
        )
        self.static_backbone = nn.Sequential(
            nn.Linear(int(static_input_dim), int(hidden_dim)),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(int(hidden_dim), int(hidden_dim)),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        merged_dim = int(hidden_dim) + int(sequence_hidden_dim)
        self.fusion = nn.Sequential(
            nn.Linear(merged_dim, int(hidden_dim)),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.action_head = nn.Linear(int(hidden_dim), len(ACTION_CLASSES))
        self.duration_head = nn.Linear(int(hidden_dim), len(DURATION_CLASSES))
        self.delta_head = nn.Linear(int(hidden_dim), 1)
        self.entry_head = nn.Linear(int(hidden_dim), 1)
        self.hold_head = nn.Linear(int(hidden_dim), 1)
        self.add_head = nn.Linear(int(hidden_dim), 1)
        self.reduce_head = nn.Linear(int(hidden_dim), 1)
        self.exit_head = nn.Linear(int(hidden_dim), 1)
        self.reentry_head = nn.Linear(int(hidden_dim), 1)

    def forward(self, static_x: torch.Tensor, sequence_x: torch.Tensor) -> dict[str, torch.Tensor]:
        _, hidden = self.sequence_encoder(sequence_x)
        seq_hidden = hidden[-1]
        static_hidden = self.static_backbone(static_x)
        fused = self.fusion(torch.cat([static_hidden, seq_hidden], dim=-1))
        return {
            "action_logits": self.action_head(fused),
            "duration_logits": self.duration_head(fused),
            "target_delta_hint": self.delta_head(fused).squeeze(-1),
            "entry_quality": self.entry_head(fused).squeeze(-1),
            "hold_quality": self.hold_head(fused).squeeze(-1),
            "add_quality": self.add_head(fused).squeeze(-1),
            "reduce_quality": self.reduce_head(fused).squeeze(-1),
            "exit_urgency": self.exit_head(fused).squeeze(-1),
            "reentry_readiness": self.reentry_head(fused).squeeze(-1),
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
class TorchContinuousPolicySeqArtifact:
    sample_model: TemporalSamplePolicyNet
    daily_model: DailyControllerNet
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
            "artifact_type": "continuous_policy_torch_seq_v3",
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
            "sample_model_state_dict": self.sample_model.state_dict(),
            "daily_model_state_dict": self.daily_model.state_dict(),
            "sample_model_config": {
                "static_input_dim": len(self.static_feature_names),
                "sequence_feature_dim": len(self.sequence_base_names),
                "sequence_steps": len(self.sequence_steps),
                "hidden_dim": int(self.sample_model.static_backbone[0].out_features),
                "sequence_hidden_dim": int(self.sample_model.sequence_encoder.hidden_size),
                "sequence_layers": int(getattr(self.sample_model, "sequence_layers", 1)),
                "dropout": float(self.sample_model.static_backbone[2].p),
            },
            "daily_model_config": {
                "input_dim": len(self.daily_feature_names),
                "hidden_dim": int(self.daily_model.backbone[0].out_features),
                "dropout": float(self.daily_model.backbone[2].p),
            },
            "train_summary": self.train_summary,
            "training_diagnostics": self.training_diagnostics,
            "training_contract": self.training_contract,
            "trained_at": self.trained_at,
        }
        torch.save(payload, path)
        return path


def _reshape_sequence_matrix(values: np.ndarray, *, steps: int, feature_dim: int) -> np.ndarray:
    sample_count = int(values.shape[0])
    return values.reshape(sample_count, steps, feature_dim).astype(np.float32)


def load_torch_seq_artifact(path: str | Path) -> TorchContinuousPolicySeqArtifact:
    payload = torch.load(Path(path), map_location="cpu", weights_only=False)
    if str(payload.get("artifact_type", "") or "") != "continuous_policy_torch_seq_v3":
        raise TypeError(f"Unsupported seq artifact type: {payload.get('artifact_type')!r}")
    sample_cfg = dict(payload.get("sample_model_config", {}) or {})
    daily_cfg = dict(payload.get("daily_model_config", {}) or {})
    sample_model = TemporalSamplePolicyNet(**sample_cfg)
    daily_model = DailyControllerNet(**daily_cfg)
    sample_model.load_state_dict(payload["sample_model_state_dict"])
    daily_model.load_state_dict(payload["daily_model_state_dict"])
    sample_model.eval()
    daily_model.eval()
    return TorchContinuousPolicySeqArtifact(
        sample_model=sample_model,
        daily_model=daily_model,
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


def fit_policy_models_v3(
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
    learning_rate: float = 1.2e-3,
    hidden_dim: int = 224,
    sequence_hidden_dim: int = 128,
    sequence_layers: int = 1,
    daily_hidden_dim: int = 96,
    dropout: float = 0.12,
    daily_dropout: float = 0.05,
    early_stop_patience: int = 10,
    resume_mode: str = "strict",
) -> TorchContinuousPolicySeqArtifact:
    if sample_frame.empty or daily_frame.empty:
        raise ValueError("continuous_policy formal_torch_seq_v3 received empty training data.")
    contract = dict(training_contract or {})
    if str(contract.get("trainer_backend", "") or "") != TRAINER_BACKEND_FORMAL_SEQ_V3:
        raise ValueError("fit_policy_models_v3 requires the formal_torch_seq_v3 training contract.")
    device = _resolve_device(contract)
    torch.manual_seed(int(random_seed))
    np.random.seed(int(random_seed))
    run_root.mkdir(parents=True, exist_ok=True)

    sequence_base_names, sequence_columns = resolve_sequence_columns(feature_names)
    static_feature_names = [name for name in feature_names if name not in set(sequence_columns)]
    if not static_feature_names:
        raise ValueError("formal_torch_seq_v3 requires at least one static feature.")

    X_static, static_fill, static_means, static_stds = _prepare_matrix(sample_frame, static_feature_names)
    X_sequence_flat, sequence_fill, sequence_means, sequence_stds = _prepare_matrix(sample_frame, sequence_columns)
    X_sequence = _reshape_sequence_matrix(
        X_sequence_flat,
        steps=len(SEQUENCE_STEP_ORDER),
        feature_dim=len(sequence_base_names),
    )
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

    sample_model = TemporalSamplePolicyNet(
        static_input_dim=len(static_feature_names),
        sequence_feature_dim=len(sequence_base_names),
        sequence_steps=len(SEQUENCE_STEP_ORDER),
        hidden_dim=hidden_dim,
        sequence_hidden_dim=sequence_hidden_dim,
        sequence_layers=sequence_layers,
        dropout=dropout,
    ).to(device)
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
    artifact_path = run_root / "continuous_policy_v3_seq_artifact.pt"
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
            raise RuntimeError("continuous_policy v3 strict resume rejected because checkpoint lineage does not match the current run signature.")
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
        raise RuntimeError(f"continuous_policy v3 strict resume requires epochs > completed epochs ({start_epoch}), got {epochs}.")

    dataset = TensorDataset(
        torch.as_tensor(X_static[train_idx], dtype=torch.float32),
        torch.as_tensor(X_sequence[train_idx], dtype=torch.float32),
        torch.as_tensor(y_action[train_idx], dtype=torch.long),
        torch.as_tensor(y_duration[train_idx], dtype=torch.long),
        *[torch.as_tensor(sample_targets[name][train_idx], dtype=torch.float32) for name in sample_targets],
    )
    loader = DataLoader(dataset, batch_size=max(32, int(batch_size)), shuffle=True, drop_last=False)
    X_static_val = torch.as_tensor(X_static[val_idx], dtype=torch.float32, device=device)
    X_sequence_val = torch.as_tensor(X_sequence[val_idx], dtype=torch.float32, device=device)
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
            static_batch, sequence_batch, action_batch, duration_batch, delta_batch, entry_batch, hold_batch, add_batch, reduce_batch, exit_batch, reentry_batch = batch
            outputs = sample_model(static_batch, sequence_batch)
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
            val_outputs = sample_model(X_static_val, X_sequence_val)
            val_action_loss = nn.functional.cross_entropy(val_outputs["action_logits"], y_action_val, weight=action_weight_tensor)
            val_duration_loss = nn.functional.cross_entropy(val_outputs["duration_logits"], y_duration_val)
            val_scalar_loss = _scalar_heads_loss(val_outputs, val_targets)
            val_daily_outputs = daily_model(X_daily_val)
            val_daily_loss = _scalar_heads_loss(val_daily_outputs, daily_targets_val)
            val_loss = float((val_action_loss + 0.55 * val_duration_loss + 0.40 * val_scalar_loss + 0.30 * val_daily_loss).detach().cpu())

        train_loss = float(epoch_sample_loss / max(batch_count, 1))
        history.append({"epoch": int(epoch), "train_loss": train_loss, "validation_loss": val_loss})
        checkpoint_payload = {
            "epoch": int(epoch),
            "best_epoch": int(best_epoch),
            "best_val_loss": float(best_val_loss),
            "sample_model_state_dict": sample_model.state_dict(),
            "daily_model_state_dict": daily_model.state_dict(),
            "sample_optimizer_state_dict": sample_optimizer.state_dict(),
            "daily_optimizer_state_dict": daily_optimizer.state_dict(),
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
    sample_model.load_state_dict(best_state["sample_model_state_dict"])
    daily_model.load_state_dict(best_state["daily_model_state_dict"])
    sample_model.eval()
    daily_model.eval()

    diagnostics = {
        "trainer_backend": TRAINER_BACKEND_FORMAL_SEQ_V3,
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
        "grouped_day_count": int(len(daily_frame)),
        "train_day_count": int(len(daily_train_idx)),
        "validation_day_count": int(len(daily_val_idx)),
        "train_sample_rows": int(len(train_idx)),
        "validation_sample_rows": int(len(val_idx)),
        "history_tail": history[-8:],
    }
    artifact = TorchContinuousPolicySeqArtifact(
        sample_model=sample_model.cpu(),
        daily_model=daily_model.cpu(),
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


def predict_policy_v3(
    artifact: TorchContinuousPolicySeqArtifact,
    *,
    state_frame: pd.DataFrame,
    daily_features: dict[str, float],
) -> tuple[pd.DataFrame, dict[str, float]]:
    if state_frame.empty:
        raise ValueError("state_frame is empty.")
    with torch.no_grad():
        static_x = torch.as_tensor(
            _apply_matrix(
                state_frame,
                artifact.static_feature_names,
                artifact.static_fill_values,
                artifact.static_means,
                artifact.static_stds,
            ),
            dtype=torch.float32,
        )
        sequence_x = torch.as_tensor(
            _reshape_sequence_matrix(
                _apply_matrix(
                    state_frame,
                    artifact.sequence_columns,
                    artifact.sequence_fill_values,
                    artifact.sequence_means,
                    artifact.sequence_stds,
                ),
                steps=len(artifact.sequence_steps),
                feature_dim=len(artifact.sequence_base_names),
            ),
            dtype=torch.float32,
        )
        outputs = artifact.sample_model(static_x, sequence_x)
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
            _apply_matrix(
                daily_row,
                artifact.daily_feature_names,
                artifact.daily_fill_values,
                artifact.daily_means,
                artifact.daily_stds,
            ),
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
    reversal_pressure = (
        0.45 * max(_daily_feature_scalar(daily_features, "recent_reversal_rate_20d") - 0.12, 0.0)
        + 0.30 * max(_daily_feature_scalar(daily_features, "reduce_reversal_pressure") - 0.14, 0.0)
        + 0.25 * max(_daily_feature_scalar(daily_features, "exit_reentry_pressure") - 0.14, 0.0)
    )
    risk_off_score = (
        defensive_score
        + 0.32 * max(_daily_feature_scalar(daily_features, "market_downside_pressure"), 0.0)
        + 0.20 * max(_daily_feature_scalar(daily_features, "portfolio_cash_pressure"), 0.0)
        + 0.18 * max(_daily_feature_scalar(daily_features, "cash_regime_pressure") - 0.12, 0.0)
    )
    is_holdcash_v3_decoder = decoder_profile_name == "holdcash_v3"
    min_gross_exposure_target = 0.18 if is_holdcash_v3_decoder else 0.12
    min_candidate_budget = 4.0 if is_holdcash_v3_decoder else 2.0
    min_position_cap_target = 0.10 if is_holdcash_v3_decoder else 0.08

    global_targets["gross_exposure_target"] = float(
        np.clip(
            global_targets["gross_exposure_target"] - risk_off_score * decoder_profile["defensive_cash_scale"],
            min_gross_exposure_target,
            0.98,
        )
    )
    global_targets["candidate_budget"] = float(
        np.clip(
            global_targets["candidate_budget"] - risk_off_score * decoder_profile["candidate_defensive_penalty"],
            min_candidate_budget,
            12.0,
        )
    )
    global_targets["turnover_budget"] = float(
        np.clip(
            global_targets["turnover_budget"] * (1.0 - risk_off_score * decoder_profile["turnover_defensive_penalty"] - reversal_pressure * 0.20),
            0.08,
            1.00,
        )
    )
    global_targets["max_position_weight_target"] = float(
        np.clip(
            global_targets["max_position_weight_target"] - risk_off_score * decoder_profile["position_cap_defensive_penalty"],
            min_position_cap_target,
            0.28,
        )
    )
    global_targets["hold_bias_target"] = float(
        np.clip(
            global_targets["hold_bias_target"] + risk_off_score * decoder_profile["hold_bias_bonus"] + reversal_pressure * (0.02 if is_holdcash_v3_decoder else 0.04),
            0.10,
            0.95,
        )
    )
    global_targets["reduce_bias_target"] = float(
        np.clip(
            (0.08 if is_holdcash_v3_decoder else 0.10)
            + risk_off_score * (0.12 if is_holdcash_v3_decoder else 0.22)
            + reversal_pressure * (0.05 if is_holdcash_v3_decoder else 0.06)
            + decoder_profile["reduce_bias_bonus"],
            0.0,
            0.65,
        )
    )
    global_targets["exit_patience_target"] = float(
        np.clip(
            (
                0.18
                + global_targets["hold_bias_target"] * 0.16
                + decoder_profile["exit_patience_bonus"]
                + reversal_pressure * 0.08
                - risk_off_score * (0.08 if is_holdcash_v3_decoder else 0.20)
            )
            if is_holdcash_v3_decoder
            else (
                global_targets["hold_bias_target"] + decoder_profile["exit_patience_bonus"] - risk_off_score * 0.20
            ),
            0.10 if is_holdcash_v3_decoder else 0.05,
            0.95,
        )
    )
    global_targets["reentry_guard_target"] = float(
        np.clip(
            decoder_profile["reversal_cooldown_bonus"]
            + reversal_pressure * (0.22 if is_holdcash_v3_decoder else 0.28)
            + risk_off_score * (0.04 if is_holdcash_v3_decoder else 0.08),
            0.0,
            0.35 if is_holdcash_v3_decoder else 0.45,
        )
    )
    global_targets = {
        "gross_exposure_target": float(np.clip(_finite_scalar(global_targets["gross_exposure_target"], default=0.35), min_gross_exposure_target, 0.98)),
        "candidate_budget": float(np.clip(_finite_scalar(global_targets["candidate_budget"], default=4.0), min_candidate_budget, 12.0)),
        "turnover_budget": float(np.clip(_finite_scalar(global_targets["turnover_budget"], default=0.18), 0.08, 1.00)),
        "max_position_weight_target": float(np.clip(_finite_scalar(global_targets["max_position_weight_target"], default=0.12), min_position_cap_target, 0.28)),
        "hold_bias_target": float(np.clip(_finite_scalar(global_targets["hold_bias_target"], default=0.24), 0.10, 0.95)),
        "reduce_bias_target": float(np.clip(_finite_scalar(global_targets["reduce_bias_target"], default=0.10), 0.0, 0.65)),
        "exit_patience_target": float(np.clip(_finite_scalar(global_targets["exit_patience_target"], default=0.20), 0.10 if is_holdcash_v3_decoder else 0.05, 0.95)),
        "reentry_guard_target": float(np.clip(_finite_scalar(global_targets["reentry_guard_target"], default=0.0), 0.0, 0.35 if is_holdcash_v3_decoder else 0.45)),
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
    reduce_reversal_pressure = state_frame["reduce_reversal_pressure"].astype(float).to_numpy(dtype=float) if "reduce_reversal_pressure" in state_frame.columns else np.zeros(len(state_frame), dtype=float)
    exit_reentry_pressure = state_frame["exit_reentry_pressure"].astype(float).to_numpy(dtype=float) if "exit_reentry_pressure" in state_frame.columns else np.zeros(len(state_frame), dtype=float)
    cash_regime_pressure = state_frame["cash_regime_pressure"].astype(float).to_numpy(dtype=float) if "cash_regime_pressure" in state_frame.columns else np.zeros(len(state_frame), dtype=float)
    hold_continuity_pressure = state_frame["hold_continuity_pressure"].astype(float).to_numpy(dtype=float) if "hold_continuity_pressure" in state_frame.columns else np.zeros(len(state_frame), dtype=float)
    drawdown_from_peak = state_frame["drawdown_from_peak"].astype(float).to_numpy(dtype=float) if "drawdown_from_peak" in state_frame.columns else np.zeros(len(state_frame), dtype=float)
    current_gross_exposure = float(np.clip(np.nansum(current_weight), 0.0, 1.0))
    deployment_gap = float(
        np.clip(
            global_targets["gross_exposure_target"] - current_gross_exposure,
            0.0,
            1.0,
        )
    )
    cash_pressure_scalar = float(np.clip(np.nanmean(portfolio_cash_pressure), 0.0, 1.0)) if len(portfolio_cash_pressure) else 0.0
    turnover_ramp_bonus = float(
        np.clip(
            max(deployment_gap - 0.06, 0.0) * (0.42 if is_holdcash_v3_decoder else 0.34)
            + max(cash_pressure_scalar - 0.10, 0.0) * 0.18
            - reversal_pressure * 0.04,
            0.0,
            0.18 if is_holdcash_v3_decoder else 0.14,
        )
    )
    if deployment_gap > 0.12 or cash_pressure_scalar > 0.18:
        turnover_floor = float(
            np.clip(
                0.14
                + max(deployment_gap - 0.10, 0.0) * 0.40
                + max(cash_pressure_scalar - 0.14, 0.0) * 0.16,
                0.14,
                0.34 if is_holdcash_v3_decoder else 0.30,
            )
        )
        global_targets["turnover_budget"] = float(
            np.clip(
                max(
                    global_targets["turnover_budget"] + turnover_ramp_bonus,
                    turnover_floor,
                ),
                0.08,
                1.00,
            )
        )
    duration_days = np.asarray([HOLDING_DAYS_BY_BUCKET.get(str(label), 0.0) for label in predicted_duration_labels], dtype=float)
    adjusted_labels = predicted_labels.astype(object).copy()
    reduce_bias_target = float(global_targets["reduce_bias_target"])
    exit_patience_target = float(global_targets["exit_patience_target"])
    reentry_guard_target = float(global_targets["reentry_guard_target"])
    for idx in range(len(adjusted_labels)):
        label = str(adjusted_labels[idx])
        held = float(current_weight[idx]) > 1e-8
        duration_name = str(predicted_duration_labels[idx])
        add_prob = float(probability_map["add"][idx])
        reduce_prob = float(probability_map["reduce"][idx])
        exit_prob = float(probability_map["exit"][idx])
        if held:
            if label in {"reduce", "exit"} and market_downside_pressure[idx] < 0.12 and signal_decay_speed[idx] < 0.05 and drawdown_from_peak[idx] > -0.05 and hold_quality[idx] > reduce_quality[idx] - decoder_profile["hold_override_margin"]:
                label = "hold"
            if label in {"reduce", "exit"} and hold_days[idx] < 2 and exit_urgency[idx] < 0.24 and hold_quality[idx] > -0.02:
                label = "hold"
            if label in {"reduce", "exit"} and days_since_last_buy[idx] <= max(3.0, hold_days[idx]) and exit_urgency[idx] < 0.26 and hold_quality[idx] > -0.04:
                label = "hold"
            if label in {"reduce", "exit"} and hold_continuity_pressure[idx] > 0.24 and hold_quality[idx] > reduce_quality[idx] - 0.05 and market_downside_pressure[idx] < 0.20:
                label = "hold"
            if label == "reduce" and days_since_last_reduce[idx] <= 2.0 and hold_quality[idx] > reduce_quality[idx] - decoder_profile["hold_override_margin"]:
                label = "hold"
            if label == "reduce" and reduce_reversal_pressure[idx] > 0.22 and hold_quality[idx] > reduce_quality[idx] - decoder_profile["reduce_gate_bonus"] and drawdown_from_peak[idx] > -0.08:
                label = "hold"
            if label == "exit" and exit_urgency[idx] < 0.18 + exit_patience_target * 0.06 and hold_quality[idx] > reduce_quality[idx] - decoder_profile["reduce_gate_bonus"]:
                label = "reduce" if reduce_quality[idx] > 0.08 else "hold"
            if label == "reduce" and reduce_quality[idx] < 0.09 + decoder_profile["reduce_gate_bonus"] and hold_quality[idx] > 0.03 - decoder_profile["hold_override_margin"]:
                label = "hold"
            if label in {"reduce", "exit"} and duration_name in {"swing", "extended"} and hold_quality[idx] >= reduce_quality[idx] - decoder_profile["hold_override_margin"]:
                label = "hold"
            exit_rescue = (
                exit_prob > 0.55
                and hold_days[idx] >= 8.0
                and (
                    exit_urgency[idx] > 0.34
                    or drawdown_from_peak[idx] < -0.12
                    or (market_downside_pressure[idx] > 0.16 and signal_decay_speed[idx] > 0.04)
                )
            )
            if exit_rescue and not (
                hold_quality[idx] > add_quality[idx] + 0.14
                and drawdown_from_peak[idx] > -0.05
                and market_downside_pressure[idx] < 0.12
            ):
                label = "exit"
            if label in {"hold", "skip"} and (market_downside_pressure[idx] > 0.18 or portfolio_cash_pressure[idx] > 0.18 or signal_decay_speed[idx] > 0.10) and hold_days[idx] >= 3.0 and current_weight[idx] > 0.02 and drawdown_from_peak[idx] < -0.03:
                label = "reduce"
            if (
                label in {"hold", "skip"}
                and add_quality[idx] > 0.14
                and duration_name in {"swing", "extended"}
                and current_weight[idx] < 0.12
            ):
                label = "add"
        else:
            open_gate = (
                0.08
                + risk_off_score * decoder_profile["open_gate_bonus"]
                + (reentry_cooldown[idx] + exit_reentry_pressure[idx]) * decoder_profile["reentry_penalty"]
                + reentry_guard_target
                + (market_downside_pressure[idx] + cash_regime_pressure[idx] * 0.5) * decoder_profile["risk_off_open_penalty"]
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
            action_strength[idx] = np.clip(entry_quality[idx] + probability_map["open"][idx] * 0.55 + duration_bonus * 0.40 - exit_reentry_pressure[idx] * 0.12, 0.0, None)
            hold_boost[idx] = np.clip(reentry_readiness[idx] * 0.25 + duration_bonus * 0.20, 0.0, None)
        elif label == "add":
            blended_delta[idx] = np.clip(max(blended_delta[idx], 0.01 + add_quality[idx] * 0.35 + duration_bonus * 0.03), 0.0, 0.18)
            action_strength[idx] = np.clip(add_quality[idx] + probability_map["add"][idx] * 0.45 + duration_bonus * 0.30, 0.0, None)
            hold_boost[idx] = np.clip(hold_quality[idx] + duration_bonus * 0.25, 0.0, None)
        elif label == "hold":
            blended_delta[idx] = np.clip(max(blended_delta[idx] * 0.30, 0.0) + hold_quality[idx] * 0.10 + decoder_profile["hold_delta_bonus"], 0.0, 0.08 + decoder_profile["hold_delta_bonus"])
            action_strength[idx] = np.clip(hold_quality[idx] + probability_map["hold"][idx] * 0.35 + duration_bonus * 0.25, 0.0, None)
            hold_boost[idx] = np.clip(
                hold_quality[idx]
                + duration_bonus * 0.30
                + decoder_profile["hold_bias_bonus"] * (0.20 + exit_patience_target)
                + hold_continuity_pressure[idx] * 0.08,
                0.0,
                None,
            )
        elif label == "reduce":
            blended_delta[idx] = -np.clip(
                max(
                    -blended_delta[idx],
                    0.08
                    + reduce_quality[idx] * (0.28 + reduce_bias_target - decoder_profile["reduce_delta_softener"])
                    + market_downside_pressure[idx] * 0.16
                    + signal_decay_speed[idx] * 0.18
                    + cash_regime_pressure[idx] * 0.10
                    - hold_quality[idx] * 0.08,
                ),
                0.0,
                0.75,
            )
            action_strength[idx] = np.clip(
                reduce_quality[idx]
                + probability_map["reduce"][idx] * 0.35
                + reduce_bias_target * 0.25
                - reduce_reversal_pressure[idx] * 0.12,
                0.0,
                None,
            )
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
