from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.linear_model import Ridge

try:
    from lightgbm import LGBMRegressor
except Exception:  # pragma: no cover
    LGBMRegressor = None


RETURN_TARGET_PREFIX = "fwd_excess_"
RISK_TARGET_NAME = "risk_downside_20"


@dataclass
class ScoreHeadArtifact:
    method: str
    feature_columns: List[str]
    task_weights: Dict[str, float]
    model: object | None


def _cross_sectional_standardize(series: pd.Series) -> pd.Series:
    grouped = series.groupby(level=0)
    mean = grouped.transform("mean")
    std = grouped.transform("std").replace(0, np.nan)
    return ((series - mean) / std).fillna(0.0)


def _cross_sectional_rank(series: pd.Series) -> pd.Series:
    return series.groupby(level=0).rank(pct=True).fillna(0.5)


def _build_feature_frame(pred_df: pd.DataFrame, target_names: List[str]) -> pd.DataFrame:
    indexed = pred_df.set_index(["date", "stock"]).sort_index()
    feature_dict: Dict[str, pd.Series] = {}
    for target_name in target_names:
        pred_col = f"pred_{target_name}"
        if pred_col not in indexed.columns:
            continue
        pred_series = indexed[pred_col].astype(float)
        feature_dict[f"{pred_col}_z"] = _cross_sectional_standardize(pred_series)
        feature_dict[f"{pred_col}_rank"] = _cross_sectional_rank(pred_series)
    return pd.DataFrame(feature_dict).reset_index()


def compute_task_rankic_summary(pred_df: pd.DataFrame, target_names: List[str]) -> pd.DataFrame:
    rows = []
    for target_name in target_names:
        pred_col = f"pred_{target_name}"
        true_col = f"true_{target_name}"
        for dt, g in pred_df.groupby("date"):
            if len(g) < 5:
                continue
            corr = spearmanr(g[pred_col], g[true_col], nan_policy="omit").correlation
            rows.append({"date": dt, "target": target_name, "rankic": float(corr) if corr is not None and np.isfinite(corr) else np.nan})
    out = pd.DataFrame(rows)
    if out.empty:
        return pd.DataFrame(columns=["target", "rankic_mean", "rankic_std", "rankic_ir"])
    summary = out.groupby("target")["rankic"].agg(["mean", "std"]).reset_index()
    summary["rankic_ir"] = summary["mean"] / summary["std"].replace(0, np.nan)
    return summary.rename(columns={"mean": "rankic_mean", "std": "rankic_std"})


def _slice_recent_window(pred_df: pd.DataFrame, window_days: int | None) -> pd.DataFrame:
    if not window_days or window_days <= 0:
        return pred_df
    dates = sorted(pd.to_datetime(pred_df["date"]).unique())
    if not dates:
        return pred_df
    selected = set(dates[-min(window_days, len(dates)):])
    out = pred_df[pd.to_datetime(pred_df["date"]).isin(selected)].copy()
    return out if not out.empty else pred_df


def derive_adaptive_task_weights(
    pred_df: pd.DataFrame,
    target_names: List[str],
    recent_window_days: int | None = None,
) -> Dict[str, float]:
    summary = compute_task_rankic_summary(_slice_recent_window(pred_df, recent_window_days), target_names)
    weights: Dict[str, float] = {}
    for target_name in target_names:
        row = summary.loc[summary["target"] == target_name]
        rankic = float(row["rankic_mean"].iloc[0]) if not row.empty else 0.0
        if target_name.startswith(RETURN_TARGET_PREFIX):
            weights[target_name] = max(rankic, 0.01)
        elif target_name == RISK_TARGET_NAME:
            weights[target_name] = max(rankic, 0.05)
        else:
            weights[target_name] = max(abs(rankic), 0.01)
    total = sum(weights.values()) or 1.0
    return {k: v / total for k, v in weights.items()}


def _build_training_target(pred_df: pd.DataFrame, task_weights: Dict[str, float]) -> pd.Series:
    indexed = pred_df.set_index(["date", "stock"]).sort_index()
    utility = pd.Series(0.0, index=indexed.index, dtype=float)
    for target_name, weight in task_weights.items():
        true_col = f"true_{target_name}"
        if true_col not in indexed.columns:
            continue
        utility = utility.add(indexed[true_col].astype(float) * float(weight), fill_value=0.0)
    target = _cross_sectional_rank(utility)
    target.name = "score_target"
    return target


def fit_score_head(
    train_pred_df: pd.DataFrame,
    target_names: List[str],
    method: str = "ridge",
    adaptive_task_weights: bool = True,
    adaptive_window_days: int | None = None,
) -> ScoreHeadArtifact:
    if method == "manual":
        task_weights = derive_adaptive_task_weights(train_pred_df, target_names, recent_window_days=adaptive_window_days) if adaptive_task_weights else {}
        return ScoreHeadArtifact(method="manual", feature_columns=[], task_weights=task_weights, model=None)

    features = _build_feature_frame(train_pred_df, target_names)
    indexed_features = features.set_index(["date", "stock"]).sort_index()
    task_weights = derive_adaptive_task_weights(train_pred_df, target_names, recent_window_days=adaptive_window_days) if adaptive_task_weights else {
        name: 1.0 / len(target_names) for name in target_names
    }
    target = _build_training_target(train_pred_df, task_weights)
    aligned = indexed_features.join(target, how="inner").dropna()
    if aligned.empty:
        raise RuntimeError("Score head training data is empty.")
    X = aligned.drop(columns=["score_target"])
    y = aligned["score_target"].values
    if method == "ridge":
        model = Ridge(alpha=1.0, random_state=7)
    elif method == "lgbm":
        if LGBMRegressor is None:
            raise RuntimeError("LightGBM is not available in current environment.")
        model = LGBMRegressor(
            n_estimators=120,
            learning_rate=0.05,
            num_leaves=31,
            subsample=0.9,
            colsample_bytree=0.9,
            random_state=7,
        )
    else:
        raise ValueError(f"Unsupported score head method: {method}")
    model.fit(X, y)
    return ScoreHeadArtifact(
        method=method,
        feature_columns=list(aligned.drop(columns=["score_target"]).columns),
        task_weights=task_weights,
        model=model,
    )


def apply_score_head(
    artifact: ScoreHeadArtifact,
    pred_df: pd.DataFrame,
    target_names: List[str],
) -> pd.DataFrame:
    if artifact.method == "manual" or artifact.model is None:
        raise ValueError("apply_score_head requires a trained non-manual score head.")
    features = _build_feature_frame(pred_df, target_names)
    indexed = features.set_index(["date", "stock"]).sort_index()
    aligned = indexed.reindex(columns=artifact.feature_columns).fillna(0.0)
    score = artifact.model.predict(aligned)
    out = aligned.reset_index()[["date", "stock"]].copy()
    out["learned_score"] = score
    return out
