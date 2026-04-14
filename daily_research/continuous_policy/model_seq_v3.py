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
        dropout: float = 0.10,
    ) -> None:
        super().__init__()
        self.sequence_steps = int(sequence_steps)
        self.sequence_feature_dim = int(sequence_feature_dim)
        self.sequence_encoder = nn.GRU(
            input_size=self.sequence_feature_dim,
            hidden_size=int(sequence_hidden_dim),
            num_layers=1,
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
        "history_tail": history[-8:],
    }
    artifact = TorchContinuousPolicySeqArtifact(
        sample_model=sample_model,
        daily_model=daily_model,
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
