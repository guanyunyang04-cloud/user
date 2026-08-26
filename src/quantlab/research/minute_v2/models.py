"""Transparent, linear, and tree baselines for minute-v2 events."""

from __future__ import annotations

import json
import math
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .contracts import KEY_COLUMNS, MODEL_FEATURE_COLUMNS, MinuteV2Error


def rule_score(frame: pd.DataFrame) -> pd.Series:
    """One deliberately simple causal benchmark, not an expert vote."""

    required = {
        "stock_return_5m_rank",
        "stock_volume_acceleration_rank",
        "industry_strength_rank",
        "industry_stock_return_rank",
        "market_breadth_positive",
        "vwap_deviation",
        "drawdown_from_day_high",
    }
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise MinuteV2Error(f"minute_v2_rule_features_missing:{','.join(missing)}")
    values = (
        0.25 * frame["stock_return_5m_rank"].fillna(0.5)
        + 0.20 * frame["stock_volume_acceleration_rank"].fillna(0.5)
        + 0.20 * frame["industry_strength_rank"].fillna(0.5)
        + 0.15 * frame["industry_stock_return_rank"].fillna(0.5)
        + 0.10 * frame["market_breadth_positive"].fillna(0.5)
        + 0.05 * frame["vwap_deviation"].clip(-0.03, 0.03).fillna(0.0) / 0.03
        + 0.05 * frame["drawdown_from_day_high"].clip(-0.05, 0.0).fillna(-0.05) / 0.05
    )
    return values.astype(float).rename("score")


@dataclass(frozen=True)
class RidgeMinuteModel:
    feature_names: tuple[str, ...]
    medians: tuple[float, ...]
    means: tuple[float, ...]
    scales: tuple[float, ...]
    coefficients: tuple[float, ...]
    intercept: float
    alpha: float

    def predict(self, frame: pd.DataFrame) -> np.ndarray:
        matrix = _numeric_matrix(frame, self.feature_names)
        medians = np.asarray(self.medians, dtype=np.float64)
        matrix = np.where(np.isfinite(matrix), matrix, medians)
        normalized = (matrix - np.asarray(self.means)) / np.asarray(self.scales)
        return normalized @ np.asarray(self.coefficients) + float(self.intercept)

    def as_dict(self) -> dict[str, Any]:
        return {"schema": "quantlab.minute_v2_ridge/1", **asdict(self)}

    def save(self, path: str | Path) -> None:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(self.as_dict(), ensure_ascii=False, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> RidgeMinuteModel:
        value = json.loads(Path(path).read_text(encoding="utf-8"))
        value.pop("schema", None)
        for name in ("feature_names", "medians", "means", "scales", "coefficients"):
            value[name] = tuple(value[name])
        return cls(**value)


def _numeric_matrix(frame: pd.DataFrame, features: Sequence[str]) -> np.ndarray:
    missing = sorted(set(features).difference(frame.columns))
    if missing:
        raise MinuteV2Error(f"minute_v2_model_features_missing:{','.join(missing)}")
    return frame.loc[:, list(features)].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=np.float64)


def fit_ridge(
    frame: pd.DataFrame,
    *,
    target: str = "label_net_return",
    feature_names: Sequence[str] = MODEL_FEATURE_COLUMNS,
    alpha: float = 10.0,
) -> RidgeMinuteModel:
    if target not in frame:
        raise MinuteV2Error(f"minute_v2_ridge_target_missing:{target}")
    matrix = _numeric_matrix(frame, feature_names)
    labels = pd.to_numeric(frame[target], errors="coerce").to_numpy(dtype=np.float64)
    valid_label = np.isfinite(labels)
    matrix = matrix[valid_label]
    labels = labels[valid_label]
    if len(labels) < max(100, len(feature_names) * 3):
        raise MinuteV2Error(f"minute_v2_ridge_support_too_small:{len(labels)}")
    medians = np.nanmedian(np.where(np.isfinite(matrix), matrix, np.nan), axis=0)
    medians = np.where(np.isfinite(medians), medians, 0.0)
    matrix = np.where(np.isfinite(matrix), matrix, medians)
    means = matrix.mean(axis=0)
    scales = matrix.std(axis=0)
    scales = np.where(scales > 1.0e-12, scales, 1.0)
    normalized = (matrix - means) / scales
    label_mean = float(labels.mean())
    centered = labels - label_mean
    gram = normalized.T @ normalized
    regularized = gram + float(alpha) * np.eye(gram.shape[0], dtype=np.float64)
    coefficients = np.linalg.solve(regularized, normalized.T @ centered)
    return RidgeMinuteModel(
        feature_names=tuple(str(name) for name in feature_names),
        medians=tuple(float(value) for value in medians),
        means=tuple(float(value) for value in means),
        scales=tuple(float(value) for value in scales),
        coefficients=tuple(float(value) for value in coefficients),
        intercept=label_mean,
        alpha=float(alpha),
    )


def _ranker_frame(
    frame: pd.DataFrame,
    *,
    target: str,
    features: Sequence[str],
) -> tuple[pd.DataFrame, np.ndarray, list[int]]:
    required = {*KEY_COLUMNS, target, *features}
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise MinuteV2Error(f"minute_v2_ranker_columns_missing:{','.join(missing)}")
    working = frame.loc[frame[target].notna(), list(required)].copy()
    working = working.sort_values(["trade_date", "bar_time", "symbol"], kind="stable")
    group_size = working.groupby(["trade_date", "bar_time"], sort=False)["symbol"].transform("size")
    working = working.loc[group_size >= 5].copy()
    if len(working) < 500:
        raise MinuteV2Error(f"minute_v2_ranker_support_too_small:{len(working)}")
    percentile = working.groupby(["trade_date", "bar_time"], sort=False)[target].rank(
        method="first", pct=True
    )
    relevance = np.minimum(4, np.floor(percentile.to_numpy(dtype=float) * 5.0)).astype(np.int32)
    groups = (
        working.groupby(["trade_date", "bar_time"], sort=False)
        .size()
        .astype(int)
        .tolist()
    )
    matrix = working.loc[:, list(features)].apply(pd.to_numeric, errors="coerce").astype("float32")
    return matrix, relevance, groups


def fit_lightgbm_ranker(
    train: pd.DataFrame,
    *,
    validation: pd.DataFrame | None = None,
    target: str = "label_net_return",
    feature_names: Sequence[str] = MODEL_FEATURE_COLUMNS,
    random_state: int = 17,
) -> tuple[Any, dict[str, Any]]:
    try:
        from lightgbm import LGBMRanker, early_stopping
    except ImportError as exc:  # pragma: no cover - required project dependency
        raise MinuteV2Error("lightgbm_required_for_minute_v2") from exc
    train_x, train_y, train_groups = _ranker_frame(
        train,
        target=target,
        features=feature_names,
    )
    model = LGBMRanker(
        objective="lambdarank",
        metric="ndcg",
        n_estimators=400,
        learning_rate=0.03,
        num_leaves=31,
        min_child_samples=200,
        subsample=0.85,
        colsample_bytree=0.80,
        reg_alpha=1.0,
        reg_lambda=5.0,
        lambdarank_truncation_level=8,
        random_state=int(random_state),
        n_jobs=4,
        verbosity=-1,
    )
    fit_kwargs: dict[str, Any] = {"group": train_groups, "eval_at": [1, 3, 5]}
    validation_rows = 0
    if validation is not None and not validation.empty:
        valid_x, valid_y, valid_groups = _ranker_frame(
            validation,
            target=target,
            features=feature_names,
        )
        validation_rows = len(valid_x)
        fit_kwargs.update(
            {
                "eval_set": [(valid_x, valid_y)],
                "eval_group": [valid_groups],
                "callbacks": [early_stopping(40, verbose=False)],
            }
        )
    model.fit(train_x, train_y, **fit_kwargs)
    importance = {
        str(name): float(value)
        for name, value in zip(feature_names, model.feature_importances_, strict=True)
    }
    metadata = {
        "train_rows": int(len(train_x)),
        "train_groups": int(len(train_groups)),
        "validation_rows": int(validation_rows),
        "feature_names": [str(name) for name in feature_names],
        "feature_importance": importance,
        "best_iteration": int(getattr(model, "best_iteration_", 0) or 0),
        "random_state": int(random_state),
    }
    return model, metadata


def grouped_rank_ic(
    frame: pd.DataFrame,
    *,
    score: str = "score",
    target: str = "label_net_return",
) -> pd.Series:
    values: dict[str, float] = {}
    for (trade_date, bar_time), group in frame.groupby(["trade_date", "bar_time"], sort=True):
        current = group[[score, target]].dropna()
        if len(current) < 5:
            continue
        correlation = current[score].rank().corr(current[target].rank())
        if math.isfinite(float(correlation)):
            values[f"{trade_date}T{bar_time}"] = float(correlation)
    return pd.Series(values, dtype=float)


def evaluate_scores(
    frame: pd.DataFrame,
    *,
    score: str = "score",
    target: str = "label_net_return",
    top_k: int = 3,
) -> dict[str, Any]:
    current = frame.loc[frame[score].notna() & frame[target].notna()].copy()
    if current.empty:
        raise MinuteV2Error("minute_v2_score_evaluation_empty")
    rank_ic = grouped_rank_ic(current, score=score, target=target)
    top = (
        current.sort_values(
            ["trade_date", "bar_time", score, "symbol"],
            ascending=[True, True, False, True],
            kind="stable",
        )
        .groupby(["trade_date", "bar_time"], sort=False)
        .head(int(top_k))
    )
    group_means = current.groupby(["trade_date", "bar_time"], sort=False)[target].mean()
    top = top.join(group_means.rename("group_mean_net_return"), on=["trade_date", "bar_time"])
    top["excess_over_group_mean"] = top[target] - top["group_mean_net_return"]
    by_day = top.groupby("trade_date", sort=True)[target].mean()
    excess_by_group = top.groupby(["trade_date", "bar_time"], sort=True)[
        "excess_over_group_mean"
    ].mean()
    return {
        "row_count": int(len(current)),
        "group_count": int(current.groupby(["trade_date", "bar_time"]).ngroups),
        "rank_ic_mean": float(rank_ic.mean()) if len(rank_ic) else None,
        "rank_ic_positive_fraction": float(rank_ic.gt(0).mean()) if len(rank_ic) else None,
        "universe_row_mean_net_return": float(current[target].mean()),
        "universe_group_mean_net_return": float(group_means.mean()),
        "top_k": int(top_k),
        "top_k_mean_net_return": float(top[target].mean()),
        "top_k_mean_excess_over_group_mean": float(top["excess_over_group_mean"].mean()),
        "top_k_positive_excess_group_fraction": float(excess_by_group.gt(0).mean()),
        "daily_top_k_mean_net_return": float(by_day.mean()),
        "positive_day_fraction": float(by_day.gt(0).mean()),
    }


__all__ = [
    "RidgeMinuteModel",
    "evaluate_scores",
    "fit_lightgbm_ranker",
    "fit_ridge",
    "grouped_rank_ic",
    "rule_score",
]
