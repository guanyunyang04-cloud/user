from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from daily_research.continuous_policy.model_v2 import _apply_matrix, _prepare_matrix, _split_indices
from daily_research.continuous_policy.core_v4_release_targets import build_core_v4_release_first_targets
from daily_research.continuous_policy.semantic_budget_intent import derive_release_first_intent
from daily_research.continuous_policy.training_contracts import TRAINER_BACKEND_FORMAL_CORE_V4
from daily_research.continuous_policy.training_runtime_acceleration import (
    autocast_context,
    configure_torch_training_acceleration,
    move_to_device,
)


CORE_V4_RELEASE_FIRST_LOSS_ALIASES: tuple[str, ...] = (
    "alpha_result_value_budget_split_v46",
    "core_v4_release_first_v1",
    "alpha_result_value_budget_split_v47",
    "core_v4_release_first_decision_v2",
)
CORE_V4_LOSS_PROFILE_NAMES: tuple[str, ...] = CORE_V4_RELEASE_FIRST_LOSS_ALIASES
CORE_V4_OUTPUT_NAMES: tuple[str, ...] = (
    "target_weight",
    "target_delta",
    "receiver_score",
    "receiver_support",
    "release_intent",
    "source_score",
    "source_release_quality",
    "source_economic_block_risk",
    "reduce_quality",
    "exit_hazard",
)
CORE_V4_MAX_TRAIN_ROWS = 4096


def resolve_core_v4_loss_profile(profile_name: str | None) -> tuple[str, dict[str, dict[str, float]]]:
    name = str(profile_name or "alpha_result_value_budget_split_v46").strip() or "alpha_result_value_budget_split_v46"
    if name not in CORE_V4_RELEASE_FIRST_LOSS_ALIASES:
        raise ValueError(f"Unsupported core-v4 loss profile: {profile_name!r}. Only r56 release-first v46 is supported.")
    decision_focused = name in {"alpha_result_value_budget_split_v47", "core_v4_release_first_decision_v2"}
    return ("alpha_result_value_budget_split_v47" if decision_focused else "alpha_result_value_budget_split_v46"), {
        "multi_objective_loss_weights": {
            "action_total": 0.0,
            "duration_total": 0.0,
            "target_weight_closure_total": 1.0,
            "cash_timing_directional_total": 0.32,
            "source_release_intent_total": 0.42,
            "reduce_exit_intent_total": 0.28,
            "release_first_allocation_total": 0.66,
            "receiver_support_total": 0.34 if decision_focused else 0.0,
            "target_delta_weight_coherence_total": 0.28 if decision_focused else 0.0,
            "release_receiver_flow_surrogate_total": 0.44 if decision_focused else 0.0,
        }
    }


class CoreV4PolicyNet(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 192, dropout: float = 0.30) -> None:
        super().__init__()
        self.input_dim = int(input_dim)
        self.hidden_dim = int(hidden_dim)
        self.dropout = float(dropout)
        self.net = nn.Sequential(
            nn.Linear(int(input_dim), int(hidden_dim)),
            nn.GELU(),
            nn.Dropout(float(dropout)),
            nn.Linear(int(hidden_dim), int(hidden_dim)),
            nn.GELU(),
            nn.Dropout(float(dropout)),
            nn.Linear(int(hidden_dim), len(CORE_V4_OUTPUT_NAMES)),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


@dataclass
class TorchContinuousPolicyCoreV4Artifact:
    feature_names: list[str]
    daily_feature_names: list[str]
    feature_fill_values: np.ndarray
    feature_means: np.ndarray
    feature_stds: np.ndarray
    daily_fill_values: np.ndarray
    daily_means: np.ndarray
    daily_stds: np.ndarray
    linear_weights: dict[str, dict[str, float]]
    linear_biases: dict[str, float]
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
            "artifact_type": "continuous_policy_torch_core_v4",
            "feature_names": list(self.feature_names),
            "daily_feature_names": list(self.daily_feature_names),
            "feature_fill_values": self.feature_fill_values.astype(np.float32).tolist(),
            "feature_means": self.feature_means.astype(np.float32).tolist(),
            "feature_stds": self.feature_stds.astype(np.float32).tolist(),
            "daily_fill_values": self.daily_fill_values.astype(np.float32).tolist(),
            "daily_means": self.daily_means.astype(np.float32).tolist(),
            "daily_stds": self.daily_stds.astype(np.float32).tolist(),
            "linear_weights": self.linear_weights,
            "linear_biases": self.linear_biases,
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


def load_torch_core_v4_artifact(path: str | Path) -> TorchContinuousPolicyCoreV4Artifact:
    payload = torch.load(Path(path), map_location="cpu", weights_only=False)
    if str(payload.get("artifact_type", "") or "") != "continuous_policy_torch_core_v4":
        raise TypeError(f"Unsupported core-v4 artifact type: {payload.get('artifact_type')!r}")
    return TorchContinuousPolicyCoreV4Artifact(
        feature_names=list(payload.get("feature_names", []) or []),
        daily_feature_names=list(payload.get("daily_feature_names", []) or []),
        feature_fill_values=np.asarray(payload.get("feature_fill_values", []), dtype=np.float32),
        feature_means=np.asarray(payload.get("feature_means", []), dtype=np.float32),
        feature_stds=np.asarray(payload.get("feature_stds", []), dtype=np.float32),
        daily_fill_values=np.asarray(payload.get("daily_fill_values", []), dtype=np.float32),
        daily_means=np.asarray(payload.get("daily_means", []), dtype=np.float32),
        daily_stds=np.asarray(payload.get("daily_stds", []), dtype=np.float32),
        linear_weights=dict(payload.get("linear_weights", {}) or {}),
        linear_biases=dict(payload.get("linear_biases", {}) or {}),
        train_summary=dict(payload.get("train_summary", {}) or {}),
        training_diagnostics=dict(payload.get("training_diagnostics", {}) or {}),
        training_contract=dict(payload.get("training_contract", {}) or {}),
        trained_at=str(payload.get("trained_at", "") or ""),
        model_config=dict(payload.get("model_config", {}) or {}),
        model_state_dict=payload.get("model_state_dict"),
        global_target_defaults=dict(payload.get("global_target_defaults", {}) or {}),
    )


def _numeric_series(frame: pd.DataFrame, name: str, default: float = 0.0) -> pd.Series:
    if name not in frame.columns:
        return pd.Series(float(default), index=frame.index, dtype=float)
    return pd.to_numeric(frame[name], errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(float(default))


def _current_weight(frame: pd.DataFrame) -> pd.Series:
    for name in ("current_weight", "position_weight", "portfolio_weight", "weight"):
        if name in frame.columns:
            return _numeric_series(frame, name, 0.0).clip(lower=0.0, upper=1.0)
    return pd.Series(0.0, index=frame.index, dtype=float)


def _ensure_features(frame: pd.DataFrame, feature_names: list[str]) -> pd.DataFrame:
    result = frame.copy()
    for name in feature_names:
        if name not in result.columns:
            result[name] = np.nan
    return result


def _linear_score(frame: pd.DataFrame, weights: dict[str, float], bias: float) -> pd.Series:
    score = pd.Series(float(bias), index=frame.index, dtype=float)
    for name, weight in dict(weights or {}).items():
        score = score + _numeric_series(frame, name, 0.0) * float(weight)
    return score.replace([np.inf, -np.inf], np.nan).fillna(float(bias))


def _decode_raw_outputs(raw: np.ndarray, current_weight: pd.Series) -> dict[str, pd.Series]:
    frame_index = current_weight.index
    sigmoid = lambda values: 1.0 / (1.0 + np.exp(-np.clip(values, -40.0, 40.0)))
    target_weight = pd.Series(sigmoid(raw[:, 0]) * 0.18, index=frame_index).clip(lower=0.0, upper=0.24)
    target_delta = pd.Series(np.tanh(raw[:, 1]) * 0.10, index=frame_index).clip(lower=-0.18, upper=0.18)
    return {
        "target_weight": target_weight,
        "target_delta": target_delta,
        "receiver_score": pd.Series(sigmoid(raw[:, 2]), index=frame_index).clip(0.0, 1.0),
        "receiver_support": pd.Series(sigmoid(raw[:, 3]), index=frame_index).clip(0.0, 1.0),
        "release_intent": pd.Series(sigmoid(raw[:, 4]), index=frame_index).clip(0.0, 1.0),
        "source_score": pd.Series(sigmoid(raw[:, 5]), index=frame_index).clip(0.0, 1.0),
        "source_release_quality": pd.Series(sigmoid(raw[:, 6]), index=frame_index).clip(0.0, 1.0),
        "source_economic_block_risk": pd.Series(sigmoid(raw[:, 7]), index=frame_index).clip(0.0, 1.0),
        "reduce_quality": pd.Series(sigmoid(raw[:, 8]), index=frame_index).clip(0.0, 1.0),
        "exit_hazard": pd.Series(sigmoid(raw[:, 9]), index=frame_index).clip(0.0, 1.0),
    }


def _heuristic_outputs(artifact: TorchContinuousPolicyCoreV4Artifact, state_frame: pd.DataFrame) -> dict[str, pd.Series]:
    current = _current_weight(state_frame)
    target_weight = _linear_score(
        state_frame,
        artifact.linear_weights.get("target_weight", {}),
        float(artifact.linear_biases.get("target_weight", 0.0)),
    ).clip(lower=0.0, upper=0.24)
    target_delta = _linear_score(
        state_frame,
        artifact.linear_weights.get("target_delta", {}),
        float(artifact.linear_biases.get("target_delta", 0.0)),
    ).clip(lower=-0.18, upper=0.18)
    if "portfolio_daily_target_delta_intent" in state_frame.columns:
        target_delta = (0.70 * _numeric_series(state_frame, "portfolio_daily_target_delta_intent", 0.0) + 0.30 * target_delta).clip(-0.18, 0.18)
    if "portfolio_daily_target_weight_intent" in state_frame.columns:
        target_weight = _numeric_series(state_frame, "portfolio_daily_target_weight_intent", 0.0).clip(0.0, 0.24)
    release_raw = _linear_score(
        state_frame,
        artifact.linear_weights.get("release_intent", {}),
        float(artifact.linear_biases.get("release_intent", 0.0)),
    )
    release_intent = (1.0 / (1.0 + np.exp(-np.clip(release_raw, -40.0, 40.0)))).where(current > 0.0, 0.0)
    release_intent = release_intent.clip(0.0, 1.0)
    receiver_raw = _linear_score(
        state_frame,
        artifact.linear_weights.get("receiver_score", {}),
        float(artifact.linear_biases.get("receiver_score", 0.0)),
    )
    receiver_score = (1.0 / (1.0 + np.exp(-np.clip(receiver_raw, -40.0, 40.0)))).where(current < 0.24, 0.0)
    receiver_support = receiver_score.where(current < 0.24, 0.0).clip(0.0, 1.0)
    return {
        "target_weight": target_weight,
        "target_delta": target_delta,
        "receiver_score": receiver_score.clip(0.0, 1.0),
        "receiver_support": receiver_support,
        "release_intent": release_intent,
        "source_score": (release_intent * 0.72 + (target_delta < -0.003).astype(float) * 0.18).clip(0.0, 1.0),
        "source_release_quality": (release_intent * 0.80 + current.clip(0.0, 0.20) * 0.50).clip(0.0, 1.0),
        "source_economic_block_risk": (0.10 - release_intent * 0.10).clip(0.0, 1.0),
        "reduce_quality": release_intent.clip(0.0, 1.0),
        "exit_hazard": (release_intent * (target_delta.abs() >= (current * 0.60).clip(lower=0.02)).astype(float)).clip(0.0, 1.0),
    }


def _model_outputs(artifact: TorchContinuousPolicyCoreV4Artifact, state_frame: pd.DataFrame) -> dict[str, pd.Series]:
    if not artifact.model_state_dict:
        return _heuristic_outputs(artifact, state_frame)
    feature_frame = _ensure_features(state_frame, artifact.feature_names)
    matrix = _apply_matrix(
        feature_frame,
        artifact.feature_names,
        artifact.feature_fill_values,
        artifact.feature_means,
        artifact.feature_stds,
    )
    config = dict(artifact.model_config or {})
    model = CoreV4PolicyNet(
        input_dim=int(config.get("input_dim", len(artifact.feature_names))),
        hidden_dim=int(config.get("hidden_dim", 192)),
        dropout=float(config.get("dropout", 0.30)),
    )
    model.load_state_dict(artifact.model_state_dict, strict=True)
    model.eval()
    with torch.no_grad():
        raw = model(torch.as_tensor(matrix, dtype=torch.float32)).detach().cpu().numpy()
    outputs = _decode_raw_outputs(raw, _current_weight(state_frame))
    if "portfolio_daily_target_delta_intent" in state_frame.columns:
        outputs["target_delta"] = (
            0.60 * _numeric_series(state_frame, "portfolio_daily_target_delta_intent", 0.0)
            + 0.40 * outputs["target_delta"]
        ).clip(-0.18, 0.18)
    if "portfolio_daily_target_weight_intent" in state_frame.columns:
        outputs["target_weight"] = (
            0.60 * _numeric_series(state_frame, "portfolio_daily_target_weight_intent", 0.0)
            + 0.40 * outputs["target_weight"]
        ).clip(0.0, 0.24)
    if "portfolio_daily_receiver_score" in state_frame.columns:
        outputs["receiver_score"] = pd.concat(
            [
                outputs["receiver_score"].rename("model_receiver"),
                _numeric_series(state_frame, "portfolio_daily_receiver_score", 0.0).rename("policy_receiver"),
                _numeric_series(state_frame, "portfolio_daily_unified_receiver_score", 0.0).rename("unified_receiver"),
            ],
            axis=1,
        ).max(axis=1).clip(0.0, 1.0)
        outputs["receiver_support"] = outputs["receiver_score"].clip(0.0, 1.0)
    return outputs


def predict_policy_core_v4(
    artifact: TorchContinuousPolicyCoreV4Artifact,
    *,
    state_frame: pd.DataFrame,
    daily_features: dict[str, float],
) -> tuple[pd.DataFrame, dict[str, float]]:
    if state_frame.empty:
        raise ValueError("state_frame is empty.")
    policy = state_frame.copy()
    current = _current_weight(policy)
    outputs = _model_outputs(artifact, policy)
    raw_target_weight = outputs["target_weight"].clip(0.0, 0.24)
    raw_target_delta = outputs["target_delta"].clip(-0.18, 0.18)
    receiver_score = outputs.get("receiver_score", pd.Series(0.0, index=policy.index, dtype=float)).clip(0.0, 1.0)
    receiver_support = outputs.get("receiver_support", receiver_score).clip(0.0, 1.0)
    headroom = (0.24 - current).clip(lower=0.0)
    release_candidate = (current > 0.003) & (raw_target_delta < -0.003)
    receiver_candidate = (headroom > 0.003) & (raw_target_delta > 0.003) & (receiver_support >= 0.30)
    coherent_target_weight = raw_target_weight.copy()
    coherent_target_weight = coherent_target_weight.where(
        ~release_candidate,
        pd.concat(
            [
                coherent_target_weight.rename("target_weight"),
                (current + raw_target_delta).clip(lower=0.0).rename("delta_weight"),
            ],
            axis=1,
        ).min(axis=1),
    )
    coherent_target_weight = coherent_target_weight.where(
        ~receiver_candidate,
        pd.concat(
            [
                coherent_target_weight.rename("target_weight"),
                (current + raw_target_delta.clip(lower=0.0)).rename("delta_weight"),
            ],
            axis=1,
        ).max(axis=1),
    )
    target_delta_weight_conflict = (
        ((raw_target_weight - current) * raw_target_delta < -(0.003 ** 2))
        | ((raw_target_delta < -0.003) & (raw_target_weight > current + 0.003))
        | ((raw_target_delta > 0.003) & (raw_target_weight < current - 0.003))
    )
    target_weight = coherent_target_weight.clip(0.0, 0.24)
    target_delta = (target_weight - current).clip(-0.18, 0.18)
    target_delta = target_delta.where(target_delta.abs() >= 0.003, 0.0)
    target_weight = (current + target_delta).clip(0.0, 0.24)
    source_score = outputs["source_score"].where(current > 0.0, 0.0).clip(0.0, 1.0)
    source_release_quality = outputs["source_release_quality"].where(current > 0.0, 0.0).clip(0.0, 1.0)
    source_economic_block_risk = outputs["source_economic_block_risk"].clip(0.0, 1.0)
    reduce_quality = outputs["reduce_quality"].clip(0.0, 1.0)
    exit_hazard = outputs["exit_hazard"].clip(0.0, 1.0)
    intent_frame = policy.copy()
    intent_frame["current_weight"] = current.astype(float)
    intent_frame["portfolio_daily_target_delta_intent"] = target_delta.astype(float)
    intent_frame["portfolio_daily_source_score"] = source_score.astype(float)
    intent_frame["portfolio_daily_source_release_quality"] = source_release_quality.astype(float)
    intent_frame["portfolio_daily_source_economic_block_risk"] = source_economic_block_risk.astype(float)
    intent_frame["reduce_quality"] = reduce_quality.astype(float)
    intent_frame["exit_hazard"] = exit_hazard.astype(float)
    release_first = derive_release_first_intent(intent_frame, deadband=0.003, min_intent=0.35)
    release_intent = pd.to_numeric(release_first["release_first_intent_score"], errors="coerce").fillna(0.0).clip(0.0, 1.0)
    release_action_hint = release_first["release_first_action_hint"].astype(str)
    add_mask = (current > 0.0) & (target_delta > 0.003)
    open_mask = (current <= 0.0) & (target_delta > 0.003)
    actions = pd.Series("hold", index=policy.index, dtype=object)
    actions.loc[current <= 0.0] = "skip"
    actions.loc[add_mask] = "add"
    actions.loc[open_mask] = "open"
    release_mask = release_action_hint.isin(["reduce", "exit"])
    actions.loc[release_mask] = release_action_hint.loc[release_mask]

    policy["action_label"] = actions.astype(str)
    policy["planned_holding_bucket"] = np.where(actions.isin(["open", "add", "hold"]), "swing", "short")
    policy["target_delta_hint"] = target_delta.astype(float)
    policy["portfolio_daily_target_weight_intent"] = target_weight.astype(float)
    policy["portfolio_daily_target_delta_intent"] = target_delta.astype(float)
    policy["portfolio_daily_release_first_intent"] = release_intent.astype(float)
    policy["release_first_intent_delta"] = release_first["release_first_intent_delta"].astype(float)
    policy["release_first_action_hint"] = release_action_hint
    policy["release_first_block_reason"] = release_first["release_first_block_reason"].astype(str)
    policy["portfolio_daily_source_score"] = source_score
    policy["portfolio_daily_unified_source_score"] = policy["portfolio_daily_source_score"]
    policy["portfolio_daily_source_executable_candidate"] = ((current > 0.003) & (release_intent >= 0.30)).astype(float)
    policy["portfolio_daily_receiver_score"] = receiver_score.astype(float)
    policy["portfolio_daily_unified_receiver_score"] = receiver_score.astype(float)
    policy["portfolio_daily_receiver_executable_candidate"] = (
        (current < 0.24 - 0.003) & (receiver_score >= 0.30) & (target_delta > 0.003)
    ).astype(float)
    policy["core_v4_target_delta_weight_conflict_count"] = int(target_delta_weight_conflict.sum())
    policy["portfolio_daily_source_release_quality"] = source_release_quality
    policy["portfolio_daily_source_economic_block_risk"] = source_economic_block_risk
    policy["reduce_quality"] = reduce_quality
    policy["exit_hazard"] = exit_hazard
    policy["exit_urgency"] = policy["exit_hazard"]
    policy["entry_quality"] = np.where(open_mask, 0.64, 0.36)
    policy["hold_quality"] = np.where(current > 0.0, 0.58, 0.34)
    policy["add_quality"] = np.where(add_mask | open_mask, 0.62, 0.34)
    policy["reentry_readiness"] = np.where(open_mask, 0.58, 0.30)

    risk = float(daily_features.get("market_risk", daily_features.get("downside_risk", 0.0)) or 0.0)
    defaults = {
        "gross_exposure_target": float(np.clip(0.82 - 0.18 * risk, 0.55, 0.88)),
        "candidate_budget": 0.12,
        "turnover_budget": 0.18,
        "max_position_weight_target": 0.08,
        "hold_bias_target": 0.55,
        "reduce_bias_target": 0.48,
        "exit_patience_target": 0.42,
        "reentry_guard_target": 0.42,
        **{key: float(value) for key, value in dict(artifact.global_target_defaults or {}).items()},
    }
    defaults["release_first_allocation_v3_mode"] = 1.0
    defaults["allocation_intent_v2_mode"] = 1.0
    defaults["core_v4_shadow_only"] = 1.0
    return policy, defaults


def _training_targets(sample_frame: pd.DataFrame) -> np.ndarray:
    targets = build_core_v4_release_first_targets(sample_frame, deadband=0.003, position_cap=0.24)
    return np.column_stack(
        [
            targets["target_weight"].clip(0.0, 0.24) / 0.18,
            np.clip(targets["target_delta"] / 0.10, -1.0, 1.0),
            targets["receiver_score"].clip(0.0, 1.0),
            targets["receiver_support"].clip(0.0, 1.0),
            targets["release_intent"].clip(0.0, 1.0),
            targets["source_score"].clip(0.0, 1.0),
            targets["source_release_quality"].clip(0.0, 1.0),
            targets["source_economic_block_risk"].clip(0.0, 1.0),
            targets["reduce_quality"].clip(0.0, 1.0),
            targets["exit_hazard"].clip(0.0, 1.0),
        ]
    ).astype(np.float32)


def _select_core_v4_train_frame(sample_frame: pd.DataFrame, random_seed: int) -> tuple[pd.DataFrame, dict[str, int]]:
    def _held_mask(frame: pd.DataFrame) -> pd.Series:
        holding = _numeric_series(frame, "holding_flag", 0.0) > 0.5
        return holding | (_current_weight(frame) > 0.003)

    if len(sample_frame) <= CORE_V4_MAX_TRAIN_ROWS:
        targets = build_core_v4_release_first_targets(sample_frame, deadband=0.003, position_cap=0.24)
        held = _held_mask(sample_frame)
        return sample_frame, {
            "core_v4_train_release_rows": int(((targets["release_intent"] > 0.0) & (targets["target_delta"] < -0.003)).sum()),
            "core_v4_train_receiver_rows": int((targets["receiver_support"] > 0.0).sum()),
            "core_v4_train_held_rows": int(held.sum()),
        }

    targets = build_core_v4_release_first_targets(sample_frame, deadband=0.003, position_cap=0.24)
    held = _held_mask(sample_frame)
    action = sample_frame.get("action_label", pd.Series("hold", index=sample_frame.index)).fillna("hold").astype(str).str.lower()
    masks = (
        (targets["release_intent"] > 0.0) & (targets["target_delta"] < -0.003),
        held & (targets["release_intent"] <= 0.0),
        (targets["receiver_support"] > 0.0) | action.isin({"open", "add"}),
        (~held) & action.isin({"skip", "hold"}),
    )
    quota = max(1, CORE_V4_MAX_TRAIN_ROWS // (len(masks) + 1))
    selected: list[pd.Index] = []
    used: set[Any] = set()
    for mask in masks:
        candidates = sample_frame.index[mask.fillna(False)]
        if len(candidates) == 0:
            continue
        take = min(quota, len(candidates))
        sampled = sample_frame.loc[candidates].sample(n=take, random_state=int(random_seed)).index
        selected.append(sampled)
        used.update(sampled.tolist())
    remaining = max(0, CORE_V4_MAX_TRAIN_ROWS - len(used))
    if remaining > 0:
        rest = sample_frame.loc[~sample_frame.index.isin(list(used))]
        if len(rest) > 0:
            selected.append(rest.sample(n=min(remaining, len(rest)), random_state=int(random_seed)).index)
    selected_values: list[Any] = []
    for index in selected:
        selected_values.extend(pd.Index(index).tolist())
    selected_index = pd.Index(pd.unique(selected_values))
    train_frame = sample_frame.loc[selected_index].sort_index()
    selected_targets = targets.loc[train_frame.index]
    selected_held = held.loc[train_frame.index]
    return train_frame, {
        "core_v4_train_release_rows": int(((selected_targets["release_intent"] > 0.0) & (selected_targets["target_delta"] < -0.003)).sum()),
        "core_v4_train_receiver_rows": int((selected_targets["receiver_support"] > 0.0).sum()),
        "core_v4_train_held_rows": int(selected_held.sum()),
    }


def _global_defaults(daily_frame: pd.DataFrame) -> dict[str, float]:
    defaults = {
        "gross_exposure_target": 0.80,
        "candidate_budget": 0.12,
        "turnover_budget": 0.18,
        "max_position_weight_target": 0.08,
        "hold_bias_target": 0.55,
        "reduce_bias_target": 0.48,
        "exit_patience_target": 0.42,
        "reentry_guard_target": 0.42,
    }
    for name in list(defaults):
        if name in daily_frame.columns:
            value = pd.to_numeric(daily_frame[name], errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
            if not value.empty:
                defaults[name] = float(np.clip(value.mean(), 0.0, 1.0))
    return defaults


def fit_policy_models_core_v4(
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
    batch_size: int = 2,
    learning_rate: float = 7.0e-5,
    hidden_dim: int = 192,
    sequence_layers: int = 2,
    daily_hidden_dim: int = 128,
    dropout: float = 0.30,
    daily_dropout: float = 0.16,
    early_stop_patience: int = 10,
    resume_mode: str = "strict",
    loss_profile: str = "alpha_result_value_budget_split_v46",
    progress_sink: Any | None = None,
) -> TorchContinuousPolicyCoreV4Artifact:
    if sample_frame.empty or daily_frame.empty:
        raise ValueError("formal_torch_core_v4 received empty training data.")
    contract = dict(training_contract or {})
    if str(contract.get("trainer_backend", "") or "") != TRAINER_BACKEND_FORMAL_CORE_V4:
        raise ValueError("fit_policy_models_core_v4 requires the formal_torch_core_v4 training contract.")
    resolved_loss_profile, loss_config = resolve_core_v4_loss_profile(loss_profile)
    if not torch.cuda.is_available() and bool(contract.get("gpu_required", False)):
        raise RuntimeError("continuous_policy formal_torch_core_v4 requires CUDA, but torch.cuda.is_available() is False.")
    device = torch.device("cuda" if bool(contract.get("gpu_required", False)) else "cpu")
    runtime = configure_torch_training_acceleration(device, cvxpy_layers_enabled=False)
    torch.manual_seed(int(random_seed))
    np.random.seed(int(random_seed))
    run_root.mkdir(parents=True, exist_ok=True)

    train_frame, stratified_diagnostics = _select_core_v4_train_frame(sample_frame, int(random_seed))
    feature_frame = _ensure_features(train_frame, feature_names)
    X, fill, means, stds = _prepare_matrix(feature_frame, feature_names)
    daily_feature_frame = _ensure_features(daily_frame, daily_feature_names)
    _, daily_fill, daily_means, daily_stds = _prepare_matrix(daily_feature_frame, daily_feature_names)
    y = _training_targets(train_frame)
    current_y = _current_weight(train_frame).to_numpy(dtype=np.float32).reshape(-1, 1)
    train_idx, val_idx = _split_indices(len(X), int(random_seed))
    train_dataset = TensorDataset(
        torch.as_tensor(X[train_idx], dtype=torch.float32),
        torch.as_tensor(y[train_idx], dtype=torch.float32),
        torch.as_tensor(current_y[train_idx], dtype=torch.float32),
    )
    val_x = torch.as_tensor(X[val_idx], dtype=torch.float32)
    val_y = torch.as_tensor(y[val_idx], dtype=torch.float32)
    val_current = torch.as_tensor(current_y[val_idx], dtype=torch.float32)
    loader = DataLoader(
        train_dataset,
        batch_size=max(1, int(batch_size or 1)),
        shuffle=True,
        pin_memory=runtime.pin_memory,
    )
    if progress_sink is not None:
        progress_sink.emit("train_dataframe_ready", train_sample_rows=int(len(train_frame)), core_v4=True)
        progress_sink.emit("train_dataloader_ready", batch_count=int(len(loader)), data_loader_pin_memory=runtime.pin_memory)

    model = CoreV4PolicyNet(input_dim=len(feature_names), hidden_dim=int(hidden_dim), dropout=float(dropout)).to(device)
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
        for batch_x, batch_y, batch_current in loader:
            batch_x, batch_y, batch_current = move_to_device(
                (batch_x, batch_y, batch_current),
                device,
                non_blocking=runtime.non_blocking_transfer,
            )
            optimizer.zero_grad(set_to_none=True)
            with autocast_context(runtime):
                raw = model(batch_x)
                pred_weight = torch.sigmoid(raw[:, 0])
                pred_delta = torch.tanh(raw[:, 1])
                target_weight = torch.clamp(batch_y[:, 0], 0.0, 1.333)
                target_delta = torch.clamp(batch_y[:, 1], -1.0, 1.0)
                target_receiver_score = torch.clamp(batch_y[:, 2], 0.0, 1.0)
                target_receiver_support = torch.clamp(batch_y[:, 3], 0.0, 1.0)
                target_release = torch.clamp(batch_y[:, 4], 0.0, 1.0)
                target_source = torch.clamp(batch_y[:, 5], 0.0, 1.0)
                target_reduce = torch.clamp(batch_y[:, 8], 0.0, 1.0)
                target_exit = torch.clamp(batch_y[:, 9], 0.0, 1.0)
                pred_receiver_score = torch.sigmoid(raw[:, 2])
                pred_receiver_support = torch.sigmoid(raw[:, 3])
                pred_release = torch.sigmoid(raw[:, 4])
                pred_source = torch.sigmoid(raw[:, 5])
                pred_reduce = torch.sigmoid(raw[:, 8])
                pred_exit = torch.sigmoid(raw[:, 9])
                current_actual = torch.clamp(batch_current[:, 0], 0.0, 1.0)
                pred_weight_actual = pred_weight * 0.18
                pred_delta_actual = pred_delta * 0.10
                coherence_target = torch.clamp((pred_weight_actual - current_actual) / 0.10, -1.0, 1.0)
                release_supply = pred_release * pred_source * (current_actual > 0.003).float()
                receiver_demand = pred_receiver_support * pred_receiver_score
                release_target_pressure = target_release * target_source
                receiver_target_pressure = target_receiver_support * target_receiver_score
                flow_surrogate = (
                    torch.relu(release_target_pressure.mean() - release_supply.mean())
                    + torch.relu(receiver_target_pressure.mean() - receiver_demand.mean())
                    + torch.relu(release_supply.mean() - receiver_demand.mean() - 0.15)
                    + (pred_release * (current_actual <= 0.003).float()).mean()
                    + (pred_release * torch.clamp(batch_y[:, 7], 0.0, 1.0)).mean()
                    + torch.relu(target_weight.mean() - pred_weight.mean())
                )
                loss = (
                    weights["target_weight_closure_total"] * nn.functional.smooth_l1_loss(pred_weight, target_weight)
                    + weights["cash_timing_directional_total"] * nn.functional.smooth_l1_loss(pred_delta, target_delta)
                    + weights["source_release_intent_total"]
                    * nn.functional.binary_cross_entropy_with_logits(raw[:, 4], target_release)
                    + weights["reduce_exit_intent_total"]
                    * (
                        nn.functional.binary_cross_entropy_with_logits(raw[:, 8], target_reduce)
                        + nn.functional.binary_cross_entropy_with_logits(raw[:, 9], target_exit)
                    )
                    + weights["release_first_allocation_total"]
                    * nn.functional.smooth_l1_loss(pred_source, target_source)
                    + weights.get("receiver_support_total", 0.0)
                    * (
                        nn.functional.smooth_l1_loss(pred_receiver_score, target_receiver_score)
                        + nn.functional.binary_cross_entropy_with_logits(raw[:, 3], target_receiver_support)
                    )
                    + weights.get("target_delta_weight_coherence_total", 0.0)
                    * nn.functional.smooth_l1_loss(pred_delta, coherence_target)
                    + weights.get("release_receiver_flow_surrogate_total", 0.0) * flow_surrogate
                )
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            train_loss_sum += float(loss.detach().cpu()) * int(batch_x.shape[0])
            train_count += int(batch_x.shape[0])
        model.eval()
        with torch.no_grad():
            val_x_device, val_y_device, _ = move_to_device(
                (val_x, val_y, val_current),
                device,
                non_blocking=runtime.non_blocking_transfer,
            )
            raw = model(val_x_device)
            val_pred = torch.cat([torch.sigmoid(raw[:, :1]), torch.tanh(raw[:, 1:2]), torch.sigmoid(raw[:, 2:])], dim=1)
            val_loss = float(nn.functional.smooth_l1_loss(val_pred, val_y_device).detach().cpu())
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
            )
            progress_event_count += 1
        if epoch >= int(min_epochs or 0) and epoch - best_epoch >= int(early_stop_patience or 0):
            break

    model.load_state_dict(best_state, strict=True)
    diagnostics = {
        "trainer_backend": TRAINER_BACKEND_FORMAL_CORE_V4,
        "loss_profile": resolved_loss_profile,
        "status": "core_v4_complete",
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
        "train_sample_cap": int(CORE_V4_MAX_TRAIN_ROWS),
        **stratified_diagnostics,
        "progress_event_count": int(progress_event_count),
        "supports_release_first_allocation_v3_mode": True,
        "core_v4_shadow_only": True,
        "train_seconds": round(time.monotonic() - started, 3),
    }
    if progress_sink is not None:
        progress_sink.emit("training_complete", completed_epochs=completed_epochs, best_epoch=best_epoch, core_v4=True)
    artifact = TorchContinuousPolicyCoreV4Artifact(
        feature_names=list(feature_names),
        daily_feature_names=list(daily_feature_names),
        feature_fill_values=fill,
        feature_means=means,
        feature_stds=stds,
        daily_fill_values=daily_fill,
        daily_means=daily_means,
        daily_stds=daily_stds,
        linear_weights={
            "target_weight": {"current_weight": 0.65, "alpha_score": 0.03},
            "target_delta": {"portfolio_daily_target_delta_intent": 1.0, "alpha_score": 0.01},
            "release_intent": {"current_weight": 1.0, "portfolio_daily_target_delta_intent": -4.0},
            "receiver_score": {"alpha_score": 1.0, "current_weight": -1.0, "portfolio_daily_target_delta_intent": 2.0},
            "receiver_support": {"alpha_score": 1.0, "current_weight": -1.0, "portfolio_daily_target_delta_intent": 2.0},
        },
        linear_biases={"target_weight": 0.02, "target_delta": 0.0, "release_intent": 0.0, "receiver_score": 0.0, "receiver_support": 0.0},
        train_summary=dict(train_summary or {}),
        training_diagnostics=diagnostics,
        training_contract=contract,
        trained_at=str(trained_at or ""),
        model_config={"input_dim": len(feature_names), "hidden_dim": int(hidden_dim), "dropout": float(dropout)},
        model_state_dict={key: value.detach().cpu() for key, value in model.state_dict().items()},
        global_target_defaults=_global_defaults(daily_frame),
    )
    artifact.save(run_root / "continuous_policy_core_v4_artifact.pt")
    return artifact
