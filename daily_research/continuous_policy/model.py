from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder


ACTION_CLASSES = ("skip", "open", "hold", "add", "reduce", "exit")
DURATION_CLASSES = ("avoid", "short", "swing", "extended")
ACTION_WEIGHT_BOOST = {
    "skip": 0.80,
    "open": 1.10,
    "hold": 1.80,
    "add": 1.55,
    "reduce": 1.45,
    "exit": 1.45,
}
HOLDING_DAYS_BY_BUCKET = {
    "avoid": 0.0,
    "short": 3.0,
    "swing": 8.0,
    "extended": 15.0,
}


@dataclass
class ContinuousPolicyArtifact:
    action_model: Pipeline
    delta_model: Pipeline
    duration_model: Pipeline
    entry_quality_model: Pipeline
    hold_quality_model: Pipeline
    add_quality_model: Pipeline
    reduce_quality_model: Pipeline
    exit_urgency_model: Pipeline
    reentry_readiness_model: Pipeline
    gross_exposure_model: Pipeline
    candidate_budget_model: Pipeline
    turnover_budget_model: Pipeline
    max_position_weight_model: Pipeline
    hold_bias_model: Pipeline
    feature_names: list[str]
    daily_feature_names: list[str]
    label_encoder: LabelEncoder
    duration_encoder: LabelEncoder
    train_summary: dict[str, Any]
    trained_at: str

    def save(self, path: Path) -> Path:
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, path)
        return path


def load_artifact(path: str | Path) -> Any:
    resolved = Path(path)
    if resolved.suffix.lower() == ".pt":
        import torch

        payload = torch.load(resolved, map_location="cpu", weights_only=False)
        artifact_type = str(payload.get("artifact_type", "") or "")
        if artifact_type == "continuous_policy_torch_core_v4":
            from daily_research.continuous_policy.model_core_v4 import load_torch_core_v4_artifact

            return load_torch_core_v4_artifact(resolved)
        if artifact_type == "continuous_policy_torch_portfolio_set_v5":
            from daily_research.continuous_policy.model_portfolio_set_v5 import load_torch_portfolio_set_v5_artifact

            return load_torch_portfolio_set_v5_artifact(resolved)
        if artifact_type == "continuous_policy_decision_core_v6_artifact":
            from daily_research.continuous_policy.decision_core_v6 import load_decision_core_v6_artifact

            return load_decision_core_v6_artifact(resolved)
        if artifact_type == "continuous_policy_torch_hier_v4":
            from daily_research.continuous_policy.model_hier_v4 import load_torch_hier_v4_artifact

            return load_torch_hier_v4_artifact(resolved)
        if artifact_type == "continuous_policy_torch_seq_v3":
            from daily_research.continuous_policy.model_seq_v3 import load_torch_seq_artifact

            return load_torch_seq_artifact(resolved)
        from daily_research.continuous_policy.model_v2 import load_torch_artifact

        return load_torch_artifact(resolved)
    artifact = joblib.load(resolved)
    if not isinstance(artifact, ContinuousPolicyArtifact):
        raise TypeError(f"Unsupported continuous-policy artifact type: {type(artifact)!r}")
    return artifact


def _classifier_pipeline(random_seed: int) -> Pipeline:
    return Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            (
                "model",
                HistGradientBoostingClassifier(
                    max_depth=6,
                    learning_rate=0.08,
                    max_iter=260,
                    random_state=int(random_seed),
                ),
            ),
        ]
    )


def _regressor_pipeline(random_seed: int) -> Pipeline:
    return Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            (
                "model",
                HistGradientBoostingRegressor(
                    max_depth=6,
                    learning_rate=0.06,
                    max_iter=260,
                    random_state=int(random_seed),
                ),
            ),
        ]
    )


def fit_policy_models(
    *,
    sample_frame: pd.DataFrame,
    daily_frame: pd.DataFrame,
    feature_names: list[str],
    daily_feature_names: list[str],
    random_seed: int = 7,
    train_summary: dict[str, Any] | None = None,
    trained_at: str = "",
) -> ContinuousPolicyArtifact:
    if sample_frame.empty:
        raise ValueError("Continuous-policy sample frame is empty.")
    if daily_frame.empty:
        raise ValueError("Continuous-policy daily frame is empty.")

    clean_samples = sample_frame.copy()
    clean_daily = daily_frame.copy()
    clean_samples = clean_samples.replace([np.inf, -np.inf], np.nan)
    clean_daily = clean_daily.replace([np.inf, -np.inf], np.nan)

    label_encoder = LabelEncoder()
    label_encoder.fit(list(ACTION_CLASSES))
    y_action = label_encoder.transform(clean_samples["action_label"].astype(str))
    action_model = _classifier_pipeline(random_seed=random_seed)

    class_counts = clean_samples["action_label"].astype(str).value_counts()
    sample_weights = clean_samples["action_label"].astype(str).map(
        lambda value: ACTION_WEIGHT_BOOST.get(str(value), 1.0) / max(float(class_counts.get(value, 1.0)), 1.0)
    )
    sample_weights = sample_weights / float(sample_weights.mean())

    action_model.fit(clean_samples[feature_names], y_action, model__sample_weight=sample_weights.to_numpy(dtype=float))

    delta_model = _regressor_pipeline(random_seed=random_seed)
    delta_model.fit(clean_samples[feature_names], clean_samples["target_delta_hint"].astype(float))

    duration_encoder = LabelEncoder()
    duration_encoder.fit(list(DURATION_CLASSES))
    y_duration = duration_encoder.transform(clean_samples["planned_holding_bucket"].astype(str).where(
        clean_samples["planned_holding_bucket"].astype(str).isin(DURATION_CLASSES),
        "avoid",
    ))
    duration_model = _classifier_pipeline(random_seed=random_seed + 1)
    duration_model.fit(clean_samples[feature_names], y_duration, model__sample_weight=sample_weights.to_numpy(dtype=float))

    entry_quality_model = _regressor_pipeline(random_seed=random_seed + 2)
    entry_quality_model.fit(clean_samples[feature_names], clean_samples["entry_quality"].astype(float))

    hold_quality_model = _regressor_pipeline(random_seed=random_seed + 3)
    hold_quality_model.fit(clean_samples[feature_names], clean_samples["hold_quality"].astype(float))

    add_quality_model = _regressor_pipeline(random_seed=random_seed + 4)
    add_quality_model.fit(clean_samples[feature_names], clean_samples["add_quality"].astype(float))

    reduce_quality_model = _regressor_pipeline(random_seed=random_seed + 5)
    reduce_quality_model.fit(clean_samples[feature_names], clean_samples["reduce_quality"].astype(float))

    exit_urgency_model = _regressor_pipeline(random_seed=random_seed + 6)
    exit_urgency_model.fit(clean_samples[feature_names], clean_samples["exit_urgency"].astype(float))

    reentry_readiness_model = _regressor_pipeline(random_seed=random_seed + 7)
    reentry_readiness_model.fit(clean_samples[feature_names], clean_samples["reentry_readiness"].astype(float))

    gross_model = _regressor_pipeline(random_seed=random_seed)
    gross_model.fit(clean_daily[daily_feature_names], clean_daily["gross_exposure_target"].astype(float))

    candidate_model = _regressor_pipeline(random_seed=random_seed)
    candidate_model.fit(clean_daily[daily_feature_names], clean_daily["candidate_budget"].astype(float))

    turnover_model = _regressor_pipeline(random_seed=random_seed)
    turnover_model.fit(clean_daily[daily_feature_names], clean_daily["turnover_budget"].astype(float))

    max_position_weight_model = _regressor_pipeline(random_seed=random_seed + 8)
    max_position_weight_model.fit(clean_daily[daily_feature_names], clean_daily["max_position_weight_target"].astype(float))

    hold_bias_model = _regressor_pipeline(random_seed=random_seed + 9)
    hold_bias_model.fit(clean_daily[daily_feature_names], clean_daily["hold_bias_target"].astype(float))

    return ContinuousPolicyArtifact(
        action_model=action_model,
        delta_model=delta_model,
        duration_model=duration_model,
        entry_quality_model=entry_quality_model,
        hold_quality_model=hold_quality_model,
        add_quality_model=add_quality_model,
        reduce_quality_model=reduce_quality_model,
        exit_urgency_model=exit_urgency_model,
        reentry_readiness_model=reentry_readiness_model,
        gross_exposure_model=gross_model,
        candidate_budget_model=candidate_model,
        turnover_budget_model=turnover_model,
        max_position_weight_model=max_position_weight_model,
        hold_bias_model=hold_bias_model,
        feature_names=list(feature_names),
        daily_feature_names=list(daily_feature_names),
        label_encoder=label_encoder,
        duration_encoder=duration_encoder,
        train_summary=dict(train_summary or {}),
        trained_at=str(trained_at or ""),
    )


def _probability_lookup(
    *,
    probabilities: np.ndarray,
    encoded_classes: np.ndarray,
    label_encoder: LabelEncoder,
) -> dict[str, np.ndarray]:
    label_names = label_encoder.inverse_transform(encoded_classes.astype(int))
    mapping = {name: probabilities[:, idx] for idx, name in enumerate(label_names)}
    return {label: mapping.get(label, np.zeros(probabilities.shape[0], dtype=float)) for label in ACTION_CLASSES}


def _finite_value(value: float | np.ndarray, *, default: float) -> float:
    scalar = float(np.asarray(value).reshape(-1)[0]) if np.asarray(value).size else float(default)
    return scalar if np.isfinite(scalar) else float(default)


def predict_policy(
    artifact: Any,
    *,
    state_frame: pd.DataFrame,
    daily_features: dict[str, float],
) -> tuple[pd.DataFrame, dict[str, float]]:
    if not isinstance(artifact, ContinuousPolicyArtifact):
        from daily_research.continuous_policy.model_core_v4 import TorchContinuousPolicyCoreV4Artifact, predict_policy_core_v4
        from daily_research.continuous_policy.decision_core_v6 import TorchDecisionCoreV6Artifact, predict_policy_decision_core_v6
        from daily_research.continuous_policy.model_hier_v4 import TorchContinuousPolicyHierV4Artifact, predict_policy_v4
        from daily_research.continuous_policy.model_portfolio_set_v5 import TorchPortfolioSetV5Artifact, predict_policy_portfolio_set_v5
        from daily_research.continuous_policy.model_seq_v3 import TorchContinuousPolicySeqArtifact, predict_policy_v3
        from daily_research.continuous_policy.model_v2 import TorchContinuousPolicyArtifact, predict_policy_v2

        if isinstance(artifact, TorchContinuousPolicyCoreV4Artifact):
            return predict_policy_core_v4(artifact, state_frame=state_frame, daily_features=daily_features)
        if isinstance(artifact, TorchDecisionCoreV6Artifact):
            return predict_policy_decision_core_v6(artifact, state_frame=state_frame, daily_features=daily_features)
        if isinstance(artifact, TorchPortfolioSetV5Artifact):
            return predict_policy_portfolio_set_v5(artifact, state_frame=state_frame, daily_features=daily_features)
        if isinstance(artifact, TorchContinuousPolicyArtifact):
            return predict_policy_v2(artifact, state_frame=state_frame, daily_features=daily_features)
        if isinstance(artifact, TorchContinuousPolicySeqArtifact):
            return predict_policy_v3(artifact, state_frame=state_frame, daily_features=daily_features)
        if isinstance(artifact, TorchContinuousPolicyHierV4Artifact):
            return predict_policy_v4(artifact, state_frame=state_frame, daily_features=daily_features)
        raise TypeError(f"Unsupported continuous-policy artifact type: {type(artifact)!r}")
    if state_frame.empty:
        raise ValueError("state_frame is empty.")
    X = state_frame[artifact.feature_names].replace([np.inf, -np.inf], np.nan)
    probabilities = artifact.action_model.predict_proba(X)
    encoded_classes = getattr(artifact.action_model.named_steps["model"], "classes_", np.arange(probabilities.shape[1]))
    probability_map = _probability_lookup(
        probabilities=probabilities,
        encoded_classes=np.asarray(encoded_classes),
        label_encoder=artifact.label_encoder,
    )
    predicted_codes = artifact.action_model.predict(X)
    predicted_labels = artifact.label_encoder.inverse_transform(np.asarray(predicted_codes, dtype=int))
    delta_hint = artifact.delta_model.predict(X)
    predicted_duration_codes = artifact.duration_model.predict(X)
    predicted_duration_labels = artifact.duration_encoder.inverse_transform(np.asarray(predicted_duration_codes, dtype=int))
    entry_quality = artifact.entry_quality_model.predict(X)
    hold_quality = artifact.hold_quality_model.predict(X)
    add_quality = artifact.add_quality_model.predict(X)
    reduce_quality = artifact.reduce_quality_model.predict(X)
    exit_urgency = np.clip(artifact.exit_urgency_model.predict(X), 0.0, None)
    reentry_readiness = np.clip(artifact.reentry_readiness_model.predict(X), 0.0, None)

    daily_row = pd.DataFrame([{name: float(daily_features.get(name, 0.0) or 0.0) for name in artifact.daily_feature_names}])
    global_targets = {
        "gross_exposure_target": float(np.clip(_finite_value(artifact.gross_exposure_model.predict(daily_row)[0], default=0.35), 0.15, 0.98)),
        "candidate_budget": float(np.clip(_finite_value(artifact.candidate_budget_model.predict(daily_row)[0], default=4.0), 2.0, 12.0)),
        "turnover_budget": float(np.clip(_finite_value(artifact.turnover_budget_model.predict(daily_row)[0], default=0.18), 0.08, 1.00)),
        "max_position_weight_target": float(np.clip(_finite_value(artifact.max_position_weight_model.predict(daily_row)[0], default=0.12), 0.08, 0.28)),
        "hold_bias_target": float(np.clip(_finite_value(artifact.hold_bias_model.predict(daily_row)[0], default=0.24), 0.10, 0.95)),
    }
    candidate_budget_target = max(1, int(round(global_targets["candidate_budget"])))

    current_weight = state_frame["current_weight"].astype(float).to_numpy(dtype=float) if "current_weight" in state_frame.columns else np.zeros(len(state_frame), dtype=float)
    hold_days = state_frame["hold_days"].astype(float).to_numpy(dtype=float) if "hold_days" in state_frame.columns else np.zeros(len(state_frame), dtype=float)
    duration_days = np.asarray([HOLDING_DAYS_BY_BUCKET.get(str(label), 0.0) for label in predicted_duration_labels], dtype=float)
    adjusted_labels = predicted_labels.astype(object).copy()
    for idx in range(len(adjusted_labels)):
        label = str(adjusted_labels[idx])
        held = float(current_weight[idx]) > 1e-8
        duration_name = str(predicted_duration_labels[idx])
        if held:
            if label in {"reduce", "exit"} and hold_days[idx] < 2 and exit_urgency[idx] < 0.24 and hold_quality[idx] > -0.02:
                label = "hold"
            if label == "exit" and exit_urgency[idx] < 0.18 and hold_quality[idx] > reduce_quality[idx]:
                label = "reduce" if reduce_quality[idx] > 0.08 else "hold"
            if label == "reduce" and reduce_quality[idx] < 0.09 and hold_quality[idx] > 0.03:
                label = "hold"
            if label in {"hold", "skip"} and add_quality[idx] > 0.14 and duration_name in {"swing", "extended"} and current_weight[idx] < 0.12:
                label = "add"
        else:
            if label == "open" and (entry_quality[idx] < 0.08 or duration_name == "avoid"):
                label = "skip"
            if (
                label in {"skip", "hold"}
                and duration_name != "avoid"
                and (
                    entry_quality[idx] > 0.075
                    or (entry_quality[idx] > 0.055 and probability_map["open"][idx] > 0.035)
                )
                and (
                    probability_map["open"][idx] > 0.035
                    or duration_name in {"swing", "extended"}
                )
            ):
                label = "open"
        adjusted_labels[idx] = label

    open_candidate_scores = entry_quality + probability_map["open"] * 0.45 + np.clip(duration_days - 3.0, 0.0, None) / 30.0
    open_candidate_scores = np.where(current_weight > 1e-8, -1e9, open_candidate_scores)
    open_candidate_scores = np.where(np.asarray(predicted_duration_labels) == "avoid", -1e9, open_candidate_scores)
    active_candidates = sum(1 for idx, label in enumerate(adjusted_labels) if str(label) in {"open", "add", "hold"} or current_weight[idx] > 1e-8)
    missing_candidates = max(0, min(candidate_budget_target, len(adjusted_labels)) - active_candidates)
    if missing_candidates > 0 and np.isfinite(open_candidate_scores).any():
        ranked_indices = np.argsort(open_candidate_scores)[::-1]
        promoted = 0
        for idx in ranked_indices:
            if promoted >= missing_candidates:
                break
            if float(open_candidate_scores[idx]) < 0.06:
                break
            if current_weight[idx] > 1e-8:
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
            blended_delta[idx] = np.clip(max(blended_delta[idx] * 0.30, 0.0) + hold_quality[idx] * 0.10, 0.0, 0.08)
            action_strength[idx] = np.clip(hold_quality[idx] + probability_map["hold"][idx] * 0.35 + duration_bonus * 0.25, 0.0, None)
            hold_boost[idx] = np.clip(hold_quality[idx] + duration_bonus * 0.30, 0.0, None)
        elif label == "reduce":
            blended_delta[idx] = -np.clip(max(-blended_delta[idx], 0.10 + reduce_quality[idx] * 0.35), 0.0, 0.70)
            action_strength[idx] = np.clip(reduce_quality[idx] + probability_map["reduce"][idx] * 0.35, 0.0, None)
            hold_boost[idx] = 0.0
        elif label == "exit":
            blended_delta[idx] = -1.0
            action_strength[idx] = np.clip(exit_urgency[idx] + probability_map["exit"][idx] * 0.45, 0.0, None)
            hold_boost[idx] = 0.0
        else:
            blended_delta[idx] = 0.0
            action_strength[idx] = probability_map["skip"][idx] * 0.20
            hold_boost[idx] = 0.0

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
        }
    ).set_index("stock")
    for label in ACTION_CLASSES:
        policy[f"prob_{label}"] = probability_map[label]
    return policy, global_targets
