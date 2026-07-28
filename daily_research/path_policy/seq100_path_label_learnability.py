from __future__ import annotations

import argparse
import gc
import json
import math
import os
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
from scipy import stats

from daily_research.path_policy import seq100_signal_quality as signal_quality


WORKSPACE_ROOT = Path(__file__).resolve().parents[2]
STUDY_ID = "seq100_path_label_learnability_v1"
DEFAULT_STUDY_PATH = (
    WORKSPACE_ROOT
    / "daily_research/research_records/seq100/seq100_path_label_learnability_v1/config.json"
)
DEFAULT_OUTPUT_ROOT = (
    WORKSPACE_ROOT
    / "daily_research/output/path_policy/studies/seq100_path_label_learnability_v1"
)
HORIZONS = (5, 10, 20, 40, 60)
FOLD_YEARS = (2023, 2024, 2025)
REGRESSION_TARGETS = ("g", "mfe", "pre_peak_mae")
PATH_TARGETS = (*REGRESSION_TARGETS, "state")
SEED = 7
PROGRESS_SCHEMA = "seq100_path_label_learnability_progress/v1"
RESULT_SCHEMA = "seq100_path_label_learnability_task/v1"


def _now() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def _atomic_write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, default=_json_default) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _append_event(path: Path, event: str, **payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    record = {"event": event, "at": _now(), **payload}
    with path.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(
            json.dumps(record, ensure_ascii=False, default=_json_default) + "\n"
        )


def _json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise TypeError(f"unsupported JSON value: {type(value).__name__}")


def _resolve_path(value: str | Path) -> Path:
    path = Path(value)
    if not path.is_absolute():
        path = WORKSPACE_ROOT / path
    return path.resolve()


def load_study(path: Path = DEFAULT_STUDY_PATH) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if str(payload.get("study_id", "")) != STUDY_ID:
        raise ValueError(f"study_id must be {STUDY_ID}")
    protocol = dict(payload.get("folds", payload.get("protocol", {})) or {})
    if tuple(int(value) for value in protocol.get("fold_years", [])) != FOLD_YEARS:
        raise ValueError("study fold years changed")
    if tuple(int(value) for value in protocol.get("horizons", [])) != HORIZONS:
        raise ValueError("study horizons changed")
    return payload


def _verify_file_record(
    record: Mapping[str, Any],
) -> Path:
    path = _resolve_path(str(record["path"]))
    if not path.is_file():
        raise FileNotFoundError(path)
    declared_size = int(record.get("size", path.stat().st_size))
    if path.stat().st_size != declared_size:
        raise ValueError(f"input file size changed: {path}")
    return path


@dataclass(frozen=True)
class FoldRows:
    year: int
    dependency_days: int
    oos_start_date_idx: int
    oos_end_date_idx: int
    maximum_train_signal_date_idx: int
    train_rows: np.ndarray
    evaluation_rows: np.ndarray


class LearnabilityInputs:
    def __init__(
        self,
        study: Mapping[str, Any],
    ) -> None:
        bindings = dict(study.get("data", study.get("inputs", {})) or {})
        self.label_manifest_path = _resolve_path(bindings["label_manifest"]["path"])
        self.feature_manifest_path = _resolve_path(
            bindings["base_feature_manifest"]["path"]
        )
        for name, path in (
            ("label_manifest", self.label_manifest_path),
            ("base_feature_manifest", self.feature_manifest_path),
        ):
            if not path.is_file():
                raise FileNotFoundError(path)
        self.label_manifest = json.loads(
            self.label_manifest_path.read_text(encoding="utf-8")
        )
        self.feature_manifest = json.loads(
            self.feature_manifest_path.read_text(encoding="utf-8")
        )
        self.maximum_outcome_date = str(
            dict(study.get("folds", study.get("protocol", {})) or {}).get(
                "maximum_outcome_date", "2025-12-31"
            )
        )
        if (
            self.label_manifest.get("maximum_outcome_date_read")
            != self.maximum_outcome_date
        ):
            raise ValueError("label source exceeds the configured outcome boundary")
        if bool(self.label_manifest.get("training_performed")):
            raise ValueError("input preparation unexpectedly trained a model")
        self.candidate_count = int(self.label_manifest["candidate_count"])
        if int(self.feature_manifest["candidate_alignment"]["candidate_count"]) != self.candidate_count:
            raise ValueError("feature and label candidate counts differ")

        feature_files = dict(self.feature_manifest["files"])
        label_files = dict(self.label_manifest["files"])
        continuous_path = _verify_file_record(
            feature_files["continuous"]
        )
        categorical_path = _verify_file_record(
            feature_files["categorical"]
        )
        date_path = _verify_file_record(
            feature_files["candidate_date_idx"]
        )
        symbol_path = _verify_file_record(
            feature_files["candidate_symbol_idx"]
        )
        labels_path = _verify_file_record(
            label_files["candidate_labels"]
        )
        states_path = _verify_file_record(
            label_files["state_labels"]
        )
        entry_path = _verify_file_record(
            label_files["entry_fill"]
        )

        self.continuous_shape = tuple(
            int(value) for value in self.feature_manifest["continuous_shape"]
        )
        self.categorical_shape = tuple(
            int(value) for value in self.feature_manifest["categorical_shape"]
        )
        self.labels_shape = tuple(
            int(value) for value in label_files["candidate_labels"]["shape"]
        )
        self.states_shape = tuple(
            int(value) for value in label_files["state_labels"]["shape"]
        )
        if self.continuous_shape[0] != self.candidate_count:
            raise ValueError("continuous row count changed")
        if self.categorical_shape[0] != self.candidate_count:
            raise ValueError("categorical row count changed")
        self.continuous = np.memmap(
            continuous_path,
            mode="r",
            dtype=np.float32,
            shape=self.continuous_shape,
        )
        self.categorical = np.memmap(
            categorical_path,
            mode="r",
            dtype=np.int64,
            shape=self.categorical_shape,
        )
        self.candidate_date_idx = np.memmap(
            date_path,
            mode="r",
            dtype=np.int32,
            shape=(self.candidate_count,),
        )
        self.candidate_symbol_idx = np.memmap(
            symbol_path,
            mode="r",
            dtype=np.int32,
            shape=(self.candidate_count,),
        )
        self.labels = np.memmap(
            labels_path,
            mode="r",
            dtype=np.float32,
            shape=self.labels_shape,
        )
        self.states = np.memmap(
            states_path,
            mode="r",
            dtype=np.int8,
            shape=self.states_shape,
        )
        self.entry_fill = np.memmap(
            entry_path,
            mode="r",
            dtype=np.int8,
            shape=(self.candidate_count,),
        )
        self.continuous_catalog = [
            dict(item) for item in self.feature_manifest["continuous_catalog"]
        ]
        self.categorical_catalog = [
            dict(item) for item in self.feature_manifest["categorical_catalog"]
        ]
        self.continuous_columns = np.asarray(
            [int(item["column_index"]) for item in self.continuous_catalog],
            dtype=np.int32,
        )
        self.categorical_columns = np.asarray(
            [int(item["column_index"]) for item in self.categorical_catalog],
            dtype=np.int32,
        )
        self.feature_names = [
            str(item["name"])
            for item in (*self.continuous_catalog, *self.categorical_catalog)
        ]
        self.label_columns = tuple(str(value) for value in self.label_manifest["label_columns"])
        self.state_columns = tuple(str(value) for value in self.label_manifest["state_columns"])
        expected_labels = tuple(
            f"{target}_{horizon}"
            for horizon in HORIZONS
            for target in REGRESSION_TARGETS
        )
        expected_states = tuple(f"state_{horizon}" for horizon in HORIZONS)
        if self.label_columns != expected_labels or self.state_columns != expected_states:
            raise ValueError("candidate label column order changed")
        pack_path = _resolve_path(self.label_manifest["source"]["pack_manifest"])
        if not pack_path.is_file():
            raise FileNotFoundError(pack_path)
        pack = json.loads(pack_path.read_text(encoding="utf-8"))
        self.date_values = np.asarray(pack["date_values"], dtype=str)
        cutoff = np.flatnonzero(self.date_values == self.maximum_outcome_date)
        if cutoff.size != 1:
            raise ValueError("configured outcome cutoff is absent or duplicated")
        self.cutoff_date_idx = int(cutoff[0])
        self._year_bounds = {
            year: self._resolve_year_bounds(year) for year in FOLD_YEARS
        }
        validate_outcome_boundary(
            candidate_date_idx=self.candidate_date_idx,
            date_values=self.date_values,
            labels=self.labels,
            states=self.states,
            entry_fill=self.entry_fill,
            maximum_outcome_date=self.maximum_outcome_date,
        )

    def _resolve_year_bounds(self, year: int) -> tuple[int, int]:
        matches = np.flatnonzero(np.char.startswith(self.date_values, f"{int(year)}-"))
        if matches.size == 0:
            raise ValueError(f"calendar lacks fold year {year}")
        return int(matches[0]), int(matches[-1])

    def label_values(self, target: str, horizon: int) -> np.ndarray:
        if target == "state":
            return self.states[:, HORIZONS.index(int(horizon))]
        column = self.label_columns.index(f"{target}_{int(horizon)}")
        return self.labels[:, column]

    def common_path_rows(self, year: int, horizon: int) -> FoldRows:
        values = self.label_values("g", horizon)
        return build_fold_rows(
            candidate_date_idx=self.candidate_date_idx,
            date_values=self.date_values,
            values=values,
            year=year,
            dependency_days=horizon,
            valid=lambda item: np.isfinite(item),
        )

    def entry_rows(self, year: int) -> FoldRows:
        return build_fold_rows(
            candidate_date_idx=self.candidate_date_idx,
            date_values=self.date_values,
            values=self.entry_fill,
            year=year,
            dependency_days=1,
            valid=lambda item: item >= 0,
        )


def validate_outcome_boundary(
    *,
    candidate_date_idx: np.ndarray,
    date_values: np.ndarray,
    labels: np.ndarray,
    states: np.ndarray,
    entry_fill: np.ndarray,
    maximum_outcome_date: str,
) -> dict[str, int]:
    dates = np.asarray(candidate_date_idx, dtype=np.int32)
    calendar = np.asarray(date_values, dtype=str)
    if dates.ndim != 1 or bool(np.any(dates[1:] < dates[:-1])):
        raise ValueError("candidate dates must be one-dimensional and ordered")
    post_dates = np.flatnonzero(calendar > str(maximum_outcome_date))
    post_start = (
        int(np.searchsorted(dates, int(post_dates[0]), side="left"))
        if post_dates.size
        else len(dates)
    )
    count = int(len(dates) - post_start)
    if count:
        if not bool(np.isnan(np.asarray(labels[post_start:])).all()):
            raise ValueError("post-boundary candidate has a numeric path label")
        if not bool((np.asarray(states[post_start:]) == -1).all()):
            raise ValueError("post-boundary candidate has a path state")
        if not bool((np.asarray(entry_fill[post_start:]) == -1).all()):
            raise ValueError("post-boundary candidate has an entry outcome")
    return {"post_boundary_candidate_count": count}


def build_fold_rows(
    *,
    candidate_date_idx: np.ndarray,
    date_values: np.ndarray,
    values: np.ndarray,
    year: int,
    dependency_days: int,
    valid: Any,
) -> FoldRows:
    dates = np.asarray(candidate_date_idx, dtype=np.int32)
    calendar = np.asarray(date_values, dtype=str)
    year_dates = np.flatnonzero(np.char.startswith(calendar, f"{int(year)}-"))
    if year_dates.size == 0:
        raise ValueError(f"calendar lacks fold year {year}")
    oos_start = int(year_dates[0])
    oos_end = int(year_dates[-1])
    maximum_train_signal = oos_start - int(dependency_days) - 1
    if maximum_train_signal < int(dates[0]):
        raise ValueError("fold purge leaves no training history")
    train_stop = int(np.searchsorted(dates, maximum_train_signal, side="right"))
    evaluation_start = int(np.searchsorted(dates, oos_start, side="left"))
    evaluation_stop = int(np.searchsorted(dates, oos_end, side="right"))
    raw_values = np.asarray(values)
    train_mask = np.asarray(valid(raw_values[:train_stop]), dtype=bool)
    evaluation_mask = np.asarray(
        valid(raw_values[evaluation_start:evaluation_stop]), dtype=bool
    )
    train_rows = np.flatnonzero(train_mask).astype(np.int64, copy=False)
    evaluation_rows = (
        np.flatnonzero(evaluation_mask).astype(np.int64, copy=False)
        + evaluation_start
    )
    if not train_rows.size or not evaluation_rows.size:
        raise ValueError("fold has an empty training or evaluation split")
    if int(dates[train_rows[-1]]) + int(dependency_days) >= oos_start:
        raise AssertionError("training label dependency overlaps the evaluation year")
    if str(calendar[int(dates[evaluation_rows[0]])])[:4] != str(year):
        raise AssertionError("evaluation rows start outside their declared year")
    if str(calendar[int(dates[evaluation_rows[-1]])])[:4] != str(year):
        raise AssertionError("evaluation rows end outside their declared year")
    return FoldRows(
        year=int(year),
        dependency_days=int(dependency_days),
        oos_start_date_idx=oos_start,
        oos_end_date_idx=oos_end,
        maximum_train_signal_date_idx=maximum_train_signal,
        train_rows=train_rows,
        evaluation_rows=evaluation_rows,
    )


def date_equal_weights(date_idx: np.ndarray) -> np.ndarray:
    dates = np.asarray(date_idx, dtype=np.int32)
    if dates.ndim != 1 or not dates.size:
        raise ValueError("date weights require a non-empty one-dimensional array")
    if bool(np.any(dates[1:] < dates[:-1])):
        raise ValueError("date weights require ordered rows")
    _, inverse, counts = np.unique(dates, return_inverse=True, return_counts=True)
    weights = 1.0 / counts[inverse].astype(np.float64)
    weights *= float(len(weights)) / float(len(counts))
    return weights.astype(np.float32)


def aligned_target_weights(
    *,
    values: np.ndarray,
    date_idx: np.ndarray,
    valid: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    target = np.asarray(values).copy()
    mask = np.asarray(valid, dtype=bool)
    if target.ndim != 1 or target.shape != mask.shape:
        raise ValueError("target validity arrays must be aligned and one-dimensional")
    weights = np.zeros(len(target), dtype=np.float32)
    if not bool(mask.any()):
        raise ValueError("target has no valid rows")
    weights[mask] = date_equal_weights(np.asarray(date_idx, dtype=np.int32)[mask])
    if np.issubdtype(target.dtype, np.floating):
        target[~mask] = 0.0
    else:
        target[~mask] = 0
    return target, weights


def _weighted_mean(values: np.ndarray, weights: np.ndarray) -> float:
    value = np.asarray(values, dtype=np.float64)
    weight = np.asarray(weights, dtype=np.float64)
    valid = np.isfinite(value) & np.isfinite(weight) & (weight > 0.0)
    if not bool(valid.any()):
        return math.nan
    return float(np.average(value[valid], weights=weight[valid]))


def _hac_mean_test(values: np.ndarray, maximum_lag: int) -> dict[str, Any]:
    series = np.asarray(values, dtype=np.float64)
    series = series[np.isfinite(series)]
    count = int(series.size)
    if count < 2:
        return {
            "count": count,
            "maximum_lag": 0,
            "mean": float(series.mean()) if count else math.nan,
            "standard_error": math.nan,
            "t_statistic": math.nan,
            "p_value_two_sided": math.nan,
        }
    lag_count = min(max(int(maximum_lag), 0), count - 1)
    mean = float(series.mean())
    centered = series - mean
    long_run_variance = float(np.dot(centered, centered) / count)
    for lag in range(1, lag_count + 1):
        covariance = float(np.dot(centered[lag:], centered[:-lag]) / count)
        bartlett = 1.0 - lag / float(lag_count + 1)
        long_run_variance += 2.0 * bartlett * covariance
    long_run_variance = max(long_run_variance, 0.0)
    standard_error = math.sqrt(long_run_variance / count)
    t_statistic = mean / standard_error if standard_error > 0.0 else math.inf
    p_value = (
        float(2.0 * stats.norm.sf(abs(t_statistic)))
        if math.isfinite(t_statistic)
        else 0.0
    )
    return {
        "count": count,
        "maximum_lag": lag_count,
        "mean": mean,
        "standard_error": float(standard_error),
        "t_statistic": float(t_statistic),
        "p_value_two_sided": p_value,
    }


def _fixed_bin_calibration(
    probability: np.ndarray,
    outcome: np.ndarray,
    weights: np.ndarray,
    *,
    bin_count: int = 10,
) -> dict[str, Any]:
    predicted = np.clip(np.asarray(probability, dtype=np.float64), 0.0, 1.0)
    actual = np.asarray(outcome, dtype=np.float64)
    weight = np.asarray(weights, dtype=np.float64)
    valid = (
        np.isfinite(predicted)
        & np.isfinite(actual)
        & np.isfinite(weight)
        & (weight > 0.0)
    )
    predicted = predicted[valid]
    actual = actual[valid]
    weight = weight[valid]
    if not predicted.size:
        raise ValueError("calibration requires finite weighted rows")
    bin_id = np.minimum((predicted * int(bin_count)).astype(np.int32), bin_count - 1)
    total_weight = float(weight.sum())
    expected_calibration_error = 0.0
    maximum_calibration_error = 0.0
    rows: list[dict[str, Any]] = []
    for current in range(bin_count):
        mask = bin_id == current
        if not bool(mask.any()):
            rows.append(
                {
                    "bin": current,
                    "lower": current / bin_count,
                    "upper": (current + 1) / bin_count,
                    "row_count": 0,
                    "weight": 0.0,
                    "mean_probability": None,
                    "observed_rate": None,
                }
            )
            continue
        current_weight = float(weight[mask].sum())
        mean_probability = float(np.average(predicted[mask], weights=weight[mask]))
        observed_rate = float(np.average(actual[mask], weights=weight[mask]))
        gap = abs(mean_probability - observed_rate)
        expected_calibration_error += current_weight / total_weight * gap
        maximum_calibration_error = max(maximum_calibration_error, gap)
        rows.append(
            {
                "bin": current,
                "lower": current / bin_count,
                "upper": (current + 1) / bin_count,
                "row_count": int(mask.sum()),
                "weight": current_weight,
                "mean_probability": mean_probability,
                "observed_rate": observed_rate,
            }
        )
    return {
        "expected_calibration_error": float(expected_calibration_error),
        "maximum_calibration_error": float(maximum_calibration_error),
        "bins": rows,
    }


def daily_score_metrics(
    *,
    date_idx: np.ndarray,
    actual: np.ndarray,
    score: np.ndarray,
    date_values: np.ndarray | None = None,
    horizon: int = 1,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    dates = np.asarray(date_idx, dtype=np.int32)
    truth = np.asarray(actual, dtype=np.float64)
    predicted = np.asarray(score, dtype=np.float64)
    if dates.shape != truth.shape or dates.shape != predicted.shape:
        raise ValueError("daily score arrays must be aligned")
    if bool(np.any(dates[1:] < dates[:-1])):
        raise ValueError("daily score rows must be ordered by date")
    rows: list[dict[str, Any]] = []
    decile_daily: list[list[float]] = []
    boundaries = np.flatnonzero(np.r_[True, dates[1:] != dates[:-1], True])
    for start, stop in zip(boundaries[:-1], boundaries[1:]):
        valid = np.isfinite(truth[start:stop]) & np.isfinite(predicted[start:stop])
        if int(valid.sum()) < 10:
            continue
        current_actual = truth[start:stop][valid]
        current_score = predicted[start:stop][valid]
        count = int(current_actual.size)
        order = np.argsort(current_score, kind="mergesort")
        decile_parts = np.array_split(order, 10)
        decile_means = [float(current_actual[part].mean()) for part in decile_parts]
        decile_daily.append(decile_means)
        top1_count = max(1, int(math.ceil(0.01 * count)))
        top5_count = max(1, int(math.ceil(0.05 * count)))
        top1 = order[-top1_count:]
        top5 = order[-top5_count:]
        baseline_mean = float(current_actual.mean())
        positive = current_actual > 0.0
        rank_ic = stats.spearmanr(current_actual, current_score).statistic
        current_date_idx = int(dates[start])
        trade_date = (
            str(np.asarray(date_values, dtype=str)[current_date_idx])
            if date_values is not None
            else str(current_date_idx)
        )
        rows.append(
            {
                "date_idx": current_date_idx,
                "trade_date": trade_date,
                "candidate_count": count,
                "rank_ic": (
                    float(rank_ic) if math.isfinite(float(rank_ic)) else math.nan
                ),
                "baseline_mean": baseline_mean,
                "top_1pct_mean": float(current_actual[top1].mean()),
                "top_5pct_mean": float(current_actual[top5].mean()),
                "top_1pct_lift": float(current_actual[top1].mean() - baseline_mean),
                "top_5pct_lift": float(current_actual[top5].mean() - baseline_mean),
                "baseline_positive_rate": float(positive.mean()),
                "top_1pct_positive_rate": float(positive[top1].mean()),
                "top_5pct_positive_rate": float(positive[top5].mean()),
                "top_1pct_positive_rate_lift": float(
                    positive[top1].mean() - positive.mean()
                ),
                "top_5pct_positive_rate_lift": float(
                    positive[top5].mean() - positive.mean()
                ),
            }
        )
    frame = pd.DataFrame(rows)
    if frame.empty:
        raise ValueError("daily score metrics have no valid dates")
    deciles = np.asarray(decile_daily, dtype=np.float64)
    decile_means = np.nanmean(deciles, axis=0)
    decile_spearman = stats.spearmanr(
        np.arange(10, dtype=np.float64), decile_means
    ).statistic
    monthly = frame.assign(month=frame["trade_date"].str.slice(0, 7)).groupby(
        "month", sort=True
    )[["rank_ic", "top_1pct_lift", "top_5pct_lift"]].mean()
    summary = {
        "date_count": int(len(frame)),
        "rank_ic_mean": float(frame["rank_ic"].mean()),
        "rank_ic_median": float(frame["rank_ic"].median()),
        "rank_ic_positive_date_fraction": float((frame["rank_ic"] > 0.0).mean()),
        "top_1pct_mean": float(frame["top_1pct_mean"].mean()),
        "top_5pct_mean": float(frame["top_5pct_mean"].mean()),
        "top_1pct_lift": float(frame["top_1pct_lift"].mean()),
        "top_5pct_lift": float(frame["top_5pct_lift"].mean()),
        "top_1pct_positive_rate_lift": float(
            frame["top_1pct_positive_rate_lift"].mean()
        ),
        "top_5pct_positive_rate_lift": float(
            frame["top_5pct_positive_rate_lift"].mean()
        ),
        "decile_means": decile_means.tolist(),
        "decile_spearman": (
            float(decile_spearman)
            if math.isfinite(float(decile_spearman))
            else math.nan
        ),
        "adjacent_decile_increase_fraction": float(
            np.mean(np.diff(decile_means) > 0.0)
        ),
        "rank_ic_hac": _hac_mean_test(
            frame["rank_ic"].to_numpy(), max(int(horizon) - 1, 0)
        ),
        "top_5pct_lift_hac": _hac_mean_test(
            frame["top_5pct_lift"].to_numpy(), max(int(horizon) - 1, 0)
        ),
        "monthly": [
            {
                "month": str(index),
                **{key: float(value) for key, value in row.items()},
            }
            for index, row in monthly.iterrows()
        ],
    }
    return frame, summary


def regression_metrics(
    *,
    date_idx: np.ndarray,
    actual: np.ndarray,
    prediction: np.ndarray,
    date_values: np.ndarray,
    horizon: int,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    truth = np.asarray(actual, dtype=np.float64)
    predicted = np.asarray(prediction, dtype=np.float64)
    valid = np.isfinite(truth) & np.isfinite(predicted)
    dates = np.asarray(date_idx, dtype=np.int32)[valid]
    truth = truth[valid]
    predicted = predicted[valid]
    weights = date_equal_weights(dates)
    daily, ranking = daily_score_metrics(
        date_idx=dates,
        actual=truth,
        score=predicted,
        date_values=date_values,
        horizon=horizon,
    )
    error = predicted - truth
    return daily, {
        "row_count": int(len(truth)),
        "date_equal_mae": _weighted_mean(np.abs(error), weights),
        "date_equal_rmse": math.sqrt(_weighted_mean(np.square(error), weights)),
        "date_equal_target_mean": _weighted_mean(truth, weights),
        "date_equal_prediction_mean": _weighted_mean(predicted, weights),
        "ranking": ranking,
    }


def _binary_probability_metrics(
    *,
    date_idx: np.ndarray,
    outcome: np.ndarray,
    probability: np.ndarray,
    date_values: np.ndarray,
    horizon: int,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    from sklearn.metrics import average_precision_score, roc_auc_score

    actual = np.asarray(outcome, dtype=np.int8)
    predicted = np.clip(np.asarray(probability, dtype=np.float64), 1.0e-7, 1.0 - 1.0e-7)
    dates = np.asarray(date_idx, dtype=np.int32)
    valid = np.isfinite(predicted) & np.isin(actual, [0, 1])
    actual = actual[valid]
    predicted = predicted[valid]
    dates = dates[valid]
    weights = date_equal_weights(dates)
    prevalence = _weighted_mean(actual, weights)
    brier = _weighted_mean(np.square(predicted - actual), weights)
    baseline_brier = prevalence * (1.0 - prevalence)
    logloss = -_weighted_mean(
        actual * np.log(predicted) + (1 - actual) * np.log(1.0 - predicted),
        weights,
    )
    baseline_logloss = -(
        prevalence * math.log(max(prevalence, 1.0e-12))
        + (1.0 - prevalence) * math.log(max(1.0 - prevalence, 1.0e-12))
    )
    daily, ranking = daily_score_metrics(
        date_idx=dates,
        actual=actual,
        score=predicted,
        date_values=date_values,
        horizon=horizon,
    )
    return daily, {
        "row_count": int(len(actual)),
        "event_count": int(actual.sum()),
        "date_equal_prevalence": prevalence,
        "date_equal_pr_auc": float(
            average_precision_score(actual, predicted, sample_weight=weights)
        ),
        "date_equal_roc_auc": float(
            roc_auc_score(actual, predicted, sample_weight=weights)
        ),
        "date_equal_brier": brier,
        "baseline_brier": baseline_brier,
        "brier_skill": 1.0 - brier / baseline_brier if baseline_brier > 0 else math.nan,
        "date_equal_logloss": logloss,
        "baseline_logloss": baseline_logloss,
        "logloss_skill": (
            1.0 - logloss / baseline_logloss if baseline_logloss > 0 else math.nan
        ),
        "calibration": _fixed_bin_calibration(predicted, actual, weights),
        "ranking": ranking,
    }


def state_metrics(
    *,
    date_idx: np.ndarray,
    actual: np.ndarray,
    probability: np.ndarray,
    date_values: np.ndarray,
    horizon: int,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    truth = np.asarray(actual, dtype=np.int8)
    predicted = np.asarray(probability, dtype=np.float64)
    dates = np.asarray(date_idx, dtype=np.int32)
    if predicted.ndim != 2 or predicted.shape[1] != 3 or predicted.shape[0] != len(truth):
        raise ValueError("state probability must have shape [rows, 3]")
    valid = np.isin(truth, [0, 1, 2]) & np.isfinite(predicted).all(axis=1)
    truth = truth[valid]
    predicted = np.clip(predicted[valid], 1.0e-7, 1.0 - 1.0e-7)
    predicted /= predicted.sum(axis=1, keepdims=True)
    dates = dates[valid]
    weights = date_equal_weights(dates)
    expected_state = predicted[:, 1] + 2.0 * predicted[:, 2]
    ordinal_daily, ordinal_ranking = daily_score_metrics(
        date_idx=dates,
        actual=truth,
        score=expected_state,
        date_values=date_values,
        horizon=horizon,
    )
    high_actual = (truth == 2).astype(np.int8)
    high_daily, high_probability = _binary_probability_metrics(
        date_idx=dates,
        outcome=high_actual,
        probability=predicted[:, 2],
        date_values=date_values,
        horizon=horizon,
    )
    one_hot = np.eye(3, dtype=np.float64)[truth]
    class_frequency = np.asarray(
        [_weighted_mean((truth == item).astype(np.float64), weights) for item in range(3)]
    )
    baseline = np.broadcast_to(class_frequency, predicted.shape)
    brier = _weighted_mean(np.sum(np.square(predicted - one_hot), axis=1), weights)
    baseline_brier = _weighted_mean(
        np.sum(np.square(baseline - one_hot), axis=1), weights
    )
    logloss = -_weighted_mean(np.log(predicted[np.arange(len(truth)), truth]), weights)
    baseline_logloss = -_weighted_mean(
        np.log(np.clip(class_frequency[truth], 1.0e-12, 1.0)), weights
    )
    merged_daily = ordinal_daily.merge(
        high_daily[["date_idx", "top_1pct_lift", "top_5pct_lift"]].rename(
            columns={
                "top_1pct_lift": "high_state_top_1pct_lift",
                "top_5pct_lift": "high_state_top_5pct_lift",
            }
        ),
        on="date_idx",
        how="left",
    )
    return merged_daily, {
        "row_count": int(len(truth)),
        "class_frequency": class_frequency.tolist(),
        "ordinal_ranking": ordinal_ranking,
        "high_state_probability": high_probability,
        "multiclass_brier": brier,
        "baseline_multiclass_brier": baseline_brier,
        "multiclass_brier_skill": (
            1.0 - brier / baseline_brier if baseline_brier > 0 else math.nan
        ),
        "multiclass_logloss": logloss,
        "baseline_multiclass_logloss": baseline_logloss,
        "multiclass_logloss_skill": (
            1.0 - logloss / baseline_logloss if baseline_logloss > 0 else math.nan
        ),
    }


@dataclass
class LightGBMDatasets:
    train_rows: np.ndarray
    evaluation_rows: np.ndarray
    train_sequence: Any
    evaluation_sequence: Any
    train_set: Any
    evaluation_set: Any
    category_vocabularies: list[np.ndarray]


def _memory_trimmed_sequence(sequence: Any, study: Mapping[str, Any]) -> Any:
    import lightgbm as lgb

    trim_config = dict(study["model"]["working_set_trim"])
    if not bool(trim_config["enabled"]):
        return sequence
    interval = int(trim_config["check_interval_batches"])
    trigger_gb = float(trim_config["trigger_available_gb"])

    class MemoryTrimmedSequence(lgb.Sequence):
        def __init__(self) -> None:
            self.batch_size = int(sequence.batch_size)
            self.batch_count = 0
            self.trim_events: list[dict[str, Any]] = []

        def __len__(self) -> int:
            return int(len(sequence))

        def __getitem__(self, index: Any) -> np.ndarray:
            is_batch = isinstance(index, slice) or (
                isinstance(index, (list, tuple, np.ndarray))
                and len(index) >= self.batch_size
            )
            if is_batch:
                self.batch_count += 1
                available_before = (
                    signal_quality.sequence_training._available_physical_memory_gb()
                )
                should_check = self.batch_count == 1 or self.batch_count % interval == 0
                should_trim = bool(
                    should_check
                    and (
                        self.batch_count == 1
                        or available_before is None
                        or available_before < trigger_gb
                    )
                )
                if should_trim:
                    process_before = (
                        signal_quality.sequence_training._current_process_memory_gb()
                    )
                    trim_succeeded = bool(
                        signal_quality.sequence_training._trim_working_set()
                    )
                    available_after = (
                        signal_quality.sequence_training._available_physical_memory_gb()
                    )
                    process_after = (
                        signal_quality.sequence_training._current_process_memory_gb()
                    )
                    self.trim_events.append(
                        {
                            "batch": self.batch_count,
                            "trim_succeeded": trim_succeeded,
                            "available_before_gb": available_before,
                            "available_after_gb": available_after,
                            "working_set_before_gb": process_before["working_set_gb"],
                            "working_set_after_gb": process_after["working_set_gb"],
                            "private_before_gb": process_before["private_gb"],
                            "private_after_gb": process_after["private_gb"],
                        }
                    )
            return sequence[index]

    return MemoryTrimmedSequence()


def _model_parameters(
    study: Mapping[str, Any],
    *,
    target: str,
) -> tuple[dict[str, Any], int, int]:
    config = dict(study["model"])
    parameters: dict[str, Any] = {
        "boosting_type": "gbdt",
        "device_type": "cpu",
        "learning_rate": float(config["learning_rate"]),
        "num_leaves": int(config["num_leaves"]),
        "max_depth": int(config["max_depth"]),
        "min_data_in_leaf": int(config["min_data_in_leaf"]),
        "feature_fraction": float(config["feature_fraction"]),
        "bagging_fraction": float(config["bagging_fraction"]),
        "bagging_freq": int(config["bagging_freq"]),
        "lambda_l1": float(config["lambda_l1"]),
        "lambda_l2": float(config["lambda_l2"]),
        "max_bin": int(config["max_bin"]),
        "deterministic": bool(config["deterministic"]),
        "force_col_wise": bool(config["force_col_wise"]),
        "num_threads": int(config["num_threads"]),
        "histogram_pool_size": int(config["histogram_pool_size_mb"]),
        "seed": SEED,
        "feature_fraction_seed": SEED,
        "bagging_seed": SEED,
        "data_random_seed": SEED,
        "drop_seed": SEED,
        "extra_seed": SEED,
        "verbosity": -1,
        "feature_pre_filter": False,
    }
    if target in REGRESSION_TARGETS:
        parameters.update(
            {
                "objective": "huber",
                "metric": "huber",
                "alpha": float(config["huber_alpha"]),
            }
        )
    elif target == "state":
        parameters.update(
            {
                "objective": "multiclass",
                "metric": "multi_logloss",
                "num_class": 3,
            }
        )
    elif target == "entry_unfilled":
        parameters.update({"objective": "binary", "metric": "binary_logloss"})
    else:
        raise ValueError(f"unsupported model target: {target}")
    return (
        parameters,
        int(config["num_boost_round"]),
        int(config["early_stopping_rounds"]),
    )


def _memory_snapshot() -> dict[str, Any]:
    try:
        import psutil

        process = psutil.Process(os.getpid())
        virtual = psutil.virtual_memory()
        return {
            "process_rss_bytes": int(process.memory_info().rss),
            "available_physical_bytes": int(virtual.available),
            "system_memory_percent": float(virtual.percent),
        }
    except Exception:
        return {}


def _require_available_memory(minimum_gb: float = 1.5) -> None:
    snapshot = _memory_snapshot()
    available = snapshot.get("available_physical_bytes")
    if available is not None and int(available) < float(minimum_gb) * 1024**3:
        raise MemoryError(
            f"learnability training requires at least {minimum_gb:.1f} GiB available memory"
        )


def _save_npy(path: Path, values: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as handle:
        np.save(handle, np.asarray(values), allow_pickle=False)
    os.replace(temporary, path)


def _dataset_feature_names(inputs: LearnabilityInputs) -> tuple[list[str], int]:
    return list(inputs.feature_names), len(inputs.categorical_columns)


def build_lgb_datasets(
    *,
    inputs: LearnabilityInputs,
    train_rows: np.ndarray,
    evaluation_rows: np.ndarray,
    initial_train_label: np.ndarray,
    initial_evaluation_label: np.ndarray,
    initial_train_weight: np.ndarray,
    initial_evaluation_weight: np.ndarray,
    study: Mapping[str, Any],
) -> LightGBMDatasets:
    import lightgbm as lgb

    _require_available_memory()
    category_vocabularies = signal_quality._fit_category_vocabularies(
        inputs.categorical,
        np.asarray(train_rows, dtype=np.int64),
        inputs.categorical_columns,
    )
    batch_size = int(study["model"]["sequence_batch_size"])
    train_sequence = signal_quality._make_lgb_sequence(
        continuous=inputs.continuous,
        categorical=inputs.categorical,
        row_ids=np.asarray(train_rows, dtype=np.int64),
        continuous_columns=inputs.continuous_columns,
        categorical_columns=inputs.categorical_columns,
        category_vocabularies=category_vocabularies,
        batch_size=batch_size,
    )
    train_sequence = _memory_trimmed_sequence(train_sequence, study)
    evaluation_sequence = signal_quality._make_lgb_sequence(
        continuous=inputs.continuous,
        categorical=inputs.categorical,
        row_ids=np.asarray(evaluation_rows, dtype=np.int64),
        continuous_columns=inputs.continuous_columns,
        categorical_columns=inputs.categorical_columns,
        category_vocabularies=category_vocabularies,
        batch_size=batch_size,
    )
    evaluation_sequence = _memory_trimmed_sequence(evaluation_sequence, study)
    feature_names, categorical_count = _dataset_feature_names(inputs)
    continuous_count = len(feature_names) - categorical_count
    categorical_positions = list(range(continuous_count, len(feature_names)))
    construction = {
        "max_bin": int(study["model"]["max_bin"]),
        "data_random_seed": SEED,
        "feature_pre_filter": False,
        "verbosity": -1,
    }
    train_set = lgb.Dataset(
        train_sequence,
        label=np.asarray(initial_train_label),
        weight=np.asarray(initial_train_weight, dtype=np.float32),
        feature_name=feature_names,
        categorical_feature=categorical_positions,
        free_raw_data=True,
        params=construction,
    )
    evaluation_set = lgb.Dataset(
        evaluation_sequence,
        label=np.asarray(initial_evaluation_label),
        weight=np.asarray(initial_evaluation_weight, dtype=np.float32),
        feature_name=feature_names,
        categorical_feature=categorical_positions,
        reference=train_set,
        free_raw_data=True,
        params=construction,
    )
    train_set.construct()
    signal_quality.sequence_training._trim_working_set()
    evaluation_set.construct()
    signal_quality.sequence_training._trim_working_set()
    return LightGBMDatasets(
        train_rows=np.asarray(train_rows, dtype=np.int64),
        evaluation_rows=np.asarray(evaluation_rows, dtype=np.int64),
        train_sequence=train_sequence,
        evaluation_sequence=evaluation_sequence,
        train_set=train_set,
        evaluation_set=evaluation_set,
        category_vocabularies=category_vocabularies,
    )


def _predict_model(model: Any, sequence: Any, *, chunk_size: int = 250_000) -> np.ndarray:
    first = np.asarray(
        model.predict(
            sequence[0 : min(len(sequence), 1)],
            num_iteration=model.best_iteration,
        )
    )
    shape = (len(sequence),) if first.ndim == 1 else (len(sequence), first.shape[1])
    output = np.empty(shape, dtype=np.float32)
    for start in range(0, len(sequence), int(chunk_size)):
        stop = min(start + int(chunk_size), len(sequence))
        output[start:stop] = np.asarray(
            model.predict(
                sequence[start:stop], num_iteration=model.best_iteration
            ),
            dtype=np.float32,
        )
    return output


def _training_callback(events_path: Path, task_id: str) -> Any:
    def callback(environment: Any) -> None:
        iteration = int(environment.iteration) + 1
        if iteration == 1 or iteration % 10 == 0:
            available = signal_quality.sequence_training._available_physical_memory_gb()
            if available is None or available < 2.0:
                signal_quality.sequence_training._trim_working_set()
        if iteration == 1 or iteration % 50 == 0:
            metrics = {
                f"{dataset}.{metric}": float(value)
                for dataset, metric, value, _higher_is_better in environment.evaluation_result_list
            }
            _append_event(
                events_path,
                "boosting_progress",
                task_id=task_id,
                iteration=iteration,
                metrics=metrics,
                **_memory_snapshot(),
            )

    callback.order = 20
    callback.before_iteration = False
    return callback


def _top_feature_importance(model: Any, feature_names: Sequence[str]) -> list[dict[str, Any]]:
    gain = np.asarray(model.feature_importance(importance_type="gain"), dtype=np.float64)
    split = np.asarray(model.feature_importance(importance_type="split"), dtype=np.int64)
    order = np.argsort(-gain, kind="mergesort")[:30]
    return [
        {
            "feature": str(feature_names[int(index)]),
            "gain": float(gain[int(index)]),
            "split": int(split[int(index)]),
        }
        for index in order
        if gain[int(index)] > 0.0 or split[int(index)] > 0
    ]


def train_one_model(
    *,
    study: Mapping[str, Any],
    inputs: LearnabilityInputs,
    datasets: LightGBMDatasets,
    target: str,
    horizon: int,
    year: int,
    train_label: np.ndarray,
    evaluation_label: np.ndarray,
    metric_evaluation_label: np.ndarray,
    train_weight: np.ndarray,
    evaluation_weight: np.ndarray,
    output_dir: Path,
    events_path: Path,
    study_id: str = STUDY_ID,
    result_schema: str = RESULT_SCHEMA,
) -> dict[str, Any]:
    import lightgbm as lgb

    output_dir.mkdir(parents=True, exist_ok=True)
    task_id = f"fold_{int(year)}_h{int(horizon):02d}_{target}"
    parameters, maximum_rounds, patience = _model_parameters(study, target=target)
    datasets.train_set.set_label(np.asarray(train_label))
    datasets.train_set.set_weight(np.asarray(train_weight, dtype=np.float32))
    datasets.evaluation_set.set_label(np.asarray(evaluation_label))
    datasets.evaluation_set.set_weight(
        np.asarray(evaluation_weight, dtype=np.float32)
    )
    _append_event(
        events_path,
        "model_started",
        task_id=task_id,
        train_row_count=int(len(train_label)),
        evaluation_row_count=int(len(evaluation_label)),
        **_memory_snapshot(),
    )
    print(json.dumps({"event": "model_started", "task_id": task_id}), flush=True)
    started = time.perf_counter()
    model = lgb.train(
        parameters,
        datasets.train_set,
        num_boost_round=maximum_rounds,
        valid_sets=[datasets.evaluation_set],
        valid_names=["evaluation"],
        callbacks=[
            _training_callback(events_path, task_id),
            lgb.early_stopping(
                stopping_rounds=patience,
                first_metric_only=True,
                verbose=False,
            ),
        ],
    )
    training_seconds = float(time.perf_counter() - started)
    model_path = output_dir / "model.txt"
    model.save_model(str(model_path), num_iteration=model.best_iteration)
    prediction = _predict_model(model, datasets.evaluation_sequence)
    prediction_path = output_dir / "prediction.npy"
    _save_npy(prediction_path, prediction)
    if target in REGRESSION_TARGETS:
        daily, metrics = regression_metrics(
            date_idx=inputs.candidate_date_idx[datasets.evaluation_rows],
            actual=np.asarray(metric_evaluation_label),
            prediction=prediction,
            date_values=inputs.date_values,
            horizon=horizon,
        )
    elif target == "state":
        daily, metrics = state_metrics(
            date_idx=inputs.candidate_date_idx[datasets.evaluation_rows],
            actual=np.asarray(metric_evaluation_label),
            probability=prediction,
            date_values=inputs.date_values,
            horizon=horizon,
        )
    else:
        daily, metrics = _binary_probability_metrics(
            date_idx=inputs.candidate_date_idx[datasets.evaluation_rows],
            outcome=np.asarray(metric_evaluation_label),
            probability=prediction,
            date_values=inputs.date_values,
            horizon=horizon,
        )
    daily_path = output_dir / "daily_metrics.parquet"
    daily.to_parquet(daily_path, index=False, compression="zstd")
    result = {
        "schema": str(result_schema),
        "status": "completed",
        "completed_at": _now(),
        "study_id": str(study_id),
        "task_id": task_id,
        "fold_year": int(year),
        "target": target,
        "horizon": int(horizon),
        "train_row_count": int(len(train_label)),
        "train_positive_weight_count": int(np.count_nonzero(train_weight > 0.0)),
        "evaluation_row_count": int(len(evaluation_label)),
        "evaluation_positive_weight_count": int(
            np.count_nonzero(evaluation_weight > 0.0)
        ),
        "best_iteration": int(model.best_iteration),
        "training_seconds": training_seconds,
        "parameters": parameters,
        "best_score": {
            dataset: {metric: float(value) for metric, value in scores.items()}
            for dataset, scores in model.best_score.items()
        },
        "metrics": metrics,
        "top_feature_importance": _top_feature_importance(
            model, inputs.feature_names
        ),
        "files": {
            "model": {
                "path": str(model_path.resolve()),
                "size": int(model_path.stat().st_size),
            },
            "prediction": {
                "path": str(prediction_path.resolve()),
                "size": int(prediction_path.stat().st_size),
                "shape": list(prediction.shape),
                "dtype": str(prediction.dtype),
            },
            "daily_metrics": {
                "path": str(daily_path.resolve()),
                "size": int(daily_path.stat().st_size),
            },
        },
        "working_set_trim": {
            "train_sequence": list(
                getattr(datasets.train_sequence, "trim_events", [])
            ),
            "evaluation_sequence": list(
                getattr(datasets.evaluation_sequence, "trim_events", [])
            ),
        },
    }
    _atomic_write_json(output_dir / "task_result.json", result)
    _append_event(
        events_path,
        "model_completed",
        task_id=task_id,
        best_iteration=int(model.best_iteration),
        training_seconds=training_seconds,
        **_memory_snapshot(),
    )
    print(
        json.dumps(
            {
                "event": "model_completed",
                "task_id": task_id,
                "best_iteration": int(model.best_iteration),
                "training_seconds": training_seconds,
            }
        ),
        flush=True,
    )
    del model, prediction, daily
    gc.collect()
    return result


def _task_result_complete(
    path: Path,
    *,
    expected: Mapping[str, Any] | None = None,
    result_schema: str = RESULT_SCHEMA,
) -> bool:
    if not path.is_file():
        return False
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload.get("schema") != str(result_schema) or payload.get("status") != "completed":
            return False
        if expected is not None and any(
            payload.get(key) != value for key, value in expected.items()
        ):
            return False
        files = dict(payload["files"])
        model_path = _resolve_path(str(files["model"]["path"]))
        prediction_path = _resolve_path(str(files["prediction"]["path"]))
        daily_path = _resolve_path(str(files["daily_metrics"]["path"]))
        if not all(item.is_file() for item in (model_path, prediction_path, daily_path)):
            return False
        import lightgbm as lgb

        lgb.Booster(model_file=str(model_path))
        prediction = np.load(prediction_path, mmap_mode="r", allow_pickle=False)
        declared_shape = tuple(int(value) for value in files["prediction"].get("shape", []))
        if declared_shape and tuple(prediction.shape) != declared_shape:
            return False
        daily = pd.read_parquet(daily_path)
        if daily.empty:
            return False
        for record in files.values():
            artifact = _resolve_path(str(record["path"]))
            if int(record.get("size", artifact.stat().st_size)) != artifact.stat().st_size:
                return False
        return True
    except (KeyError, OSError, TypeError, ValueError, json.JSONDecodeError):
        return False


def _save_npz(path: Path, arrays: Sequence[np.ndarray]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as handle:
        np.savez_compressed(
            handle,
            **{f"category_{index}": value for index, value in enumerate(arrays)},
        )
    os.replace(temporary, path)


def _write_group_material(
    *,
    output_dir: Path,
    study: Mapping[str, Any],
    inputs: LearnabilityInputs,
    fold: FoldRows,
    datasets: LightGBMDatasets,
    study_id: str = STUDY_ID,
    group_schema: str = "seq100_path_label_learnability_group/v1",
) -> dict[str, Any]:
    evaluation_rows_path = output_dir / "evaluation_rows.npy"
    vocabulary_path = output_dir / "category_vocabularies.npz"
    _save_npy(evaluation_rows_path, datasets.evaluation_rows)
    _save_npz(vocabulary_path, datasets.category_vocabularies)
    payload = {
        "schema": str(group_schema),
        "created_at": _now(),
        "study_id": str(study_id),
        "fold_year": int(fold.year),
        "dependency_days": int(fold.dependency_days),
        "oos_start_date_idx": int(fold.oos_start_date_idx),
        "oos_start_trade_date": str(inputs.date_values[fold.oos_start_date_idx]),
        "oos_end_date_idx": int(fold.oos_end_date_idx),
        "oos_end_trade_date": str(inputs.date_values[fold.oos_end_date_idx]),
        "maximum_train_signal_date_idx": int(fold.maximum_train_signal_date_idx),
        "maximum_train_signal_trade_date": str(
            inputs.date_values[fold.maximum_train_signal_date_idx]
        ),
        "maximum_train_outcome_date_idx": int(
            fold.maximum_train_signal_date_idx + fold.dependency_days
        ),
        "maximum_train_outcome_trade_date": str(
            inputs.date_values[
                fold.maximum_train_signal_date_idx + fold.dependency_days
            ]
        ),
        "purge_rule": "signal_date_idx + dependency_days < oos_start_date_idx",
        "train_row_count": int(len(datasets.train_rows)),
        "evaluation_row_count": int(len(datasets.evaluation_rows)),
        "feature_count": len(inputs.feature_names),
        "categorical_count": len(inputs.categorical_columns),
        "files": {
            "evaluation_rows": {
                "path": str(evaluation_rows_path.resolve()),
                "size": int(evaluation_rows_path.stat().st_size),
            },
            "category_vocabularies": {
                "path": str(vocabulary_path.resolve()),
                "size": int(vocabulary_path.stat().st_size),
            },
        },
    }
    _atomic_write_json(output_dir / "group_manifest.json", payload)
    return payload


def _release_training_memory(datasets: LightGBMDatasets) -> None:
    datasets.train_set = None
    datasets.evaluation_set = None
    datasets.train_sequence = None
    datasets.evaluation_sequence = None
    datasets.category_vocabularies.clear()
    gc.collect()
    try:
        signal_quality.sequence_training._trim_working_set()
    except Exception:
        pass


def _path_task_dir(output_root: Path, year: int, horizon: int, target: str) -> Path:
    return output_root / "folds" / f"fold_{year}" / f"h{horizon:02d}" / target


def _entry_task_dir(output_root: Path, year: int) -> Path:
    return output_root / "folds" / f"fold_{year}" / "entry_unfilled"


def _task_semantics(
    study: Mapping[str, Any],
    *,
    year: int,
    horizon: int,
    target: str,
) -> dict[str, Any]:
    return {
        "study_id": STUDY_ID,
        "task_id": f"fold_{int(year)}_h{int(horizon):02d}_{target}",
        "fold_year": int(year),
        "target": str(target),
        "horizon": int(horizon),
        "parameters": _model_parameters(study, target=target)[0],
    }


def _completed_task_count(output_root: Path, study: Mapping[str, Any]) -> int:
    count = 0
    for path in output_root.glob("folds/fold_*/**/task_result.json"):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            expected = _task_semantics(
                study,
                year=int(payload["fold_year"]),
                horizon=int(payload["horizon"]),
                target=str(payload["target"]),
            )
        except (KeyError, TypeError, ValueError, json.JSONDecodeError):
            continue
        if _task_result_complete(path, expected=expected):
            count += 1
    return count


def _progress_payload(
    *,
    study: Mapping[str, Any],
    status: str,
    phase: str,
    output_root: Path,
    started_at: str,
    current_task: str | None = None,
) -> dict[str, Any]:
    completed = _completed_task_count(output_root, study)
    return {
        "schema": PROGRESS_SCHEMA,
        "study_id": STUDY_ID,
        "status": status,
        "phase": phase,
        "started_at": started_at,
        "updated_at": _now(),
        "task_count": len(FOLD_YEARS) * (1 + len(HORIZONS) * len(PATH_TARGETS)),
        "completed_task_count": completed,
        "current_task": current_task,
        **_memory_snapshot(),
    }


def _run_entry_fold(
    *,
    study: Mapping[str, Any],
    inputs: LearnabilityInputs,
    output_root: Path,
    events_path: Path,
    year: int,
) -> bool:
    task_dir = _entry_task_dir(output_root, year)
    result_path = task_dir / "task_result.json"
    expected = _task_semantics(
        study, year=year, horizon=1, target="entry_unfilled"
    )
    if _task_result_complete(result_path, expected=expected):
        return False
    fold = inputs.entry_rows(year)
    train_raw = (np.asarray(inputs.entry_fill[fold.train_rows]) == 0).astype(np.int8)
    evaluation_raw = (
        np.asarray(inputs.entry_fill[fold.evaluation_rows]) == 0
    ).astype(np.int8)
    train_weight = date_equal_weights(inputs.candidate_date_idx[fold.train_rows])
    evaluation_weight = date_equal_weights(
        inputs.candidate_date_idx[fold.evaluation_rows]
    )
    datasets = build_lgb_datasets(
        inputs=inputs,
        train_rows=fold.train_rows,
        evaluation_rows=fold.evaluation_rows,
        initial_train_label=train_raw,
        initial_evaluation_label=evaluation_raw,
        initial_train_weight=train_weight,
        initial_evaluation_weight=evaluation_weight,
        study=study,
    )
    _write_group_material(
        output_dir=task_dir,
        study=study,
        inputs=inputs,
        fold=fold,
        datasets=datasets,
    )
    train_one_model(
        study=study,
        inputs=inputs,
        datasets=datasets,
        target="entry_unfilled",
        horizon=1,
        year=year,
        train_label=train_raw,
        evaluation_label=evaluation_raw,
        metric_evaluation_label=evaluation_raw,
        train_weight=train_weight,
        evaluation_weight=evaluation_weight,
        output_dir=task_dir,
        events_path=events_path,
    )
    _release_training_memory(datasets)
    return True


def _valid_target(values: np.ndarray, target: str) -> np.ndarray:
    if target == "state":
        return np.isin(np.asarray(values), [0, 1, 2])
    return np.isfinite(np.asarray(values, dtype=np.float64))


def _run_path_group(
    *,
    study: Mapping[str, Any],
    inputs: LearnabilityInputs,
    output_root: Path,
    events_path: Path,
    year: int,
    horizon: int,
) -> int:
    pending = [
        target
        for target in PATH_TARGETS
        if not _task_result_complete(
            _path_task_dir(output_root, year, horizon, target)
            / "task_result.json",
            expected=_task_semantics(
                study,
                year=year,
                horizon=horizon,
                target=target,
            ),
        )
    ]
    if not pending:
        return 0
    fold = inputs.common_path_rows(year, horizon)
    initial_train = np.asarray(inputs.label_values("g", horizon)[fold.train_rows])
    initial_evaluation = np.asarray(
        inputs.label_values("g", horizon)[fold.evaluation_rows]
    )
    initial_train_weight = date_equal_weights(
        inputs.candidate_date_idx[fold.train_rows]
    )
    initial_evaluation_weight = date_equal_weights(
        inputs.candidate_date_idx[fold.evaluation_rows]
    )
    datasets = build_lgb_datasets(
        inputs=inputs,
        train_rows=fold.train_rows,
        evaluation_rows=fold.evaluation_rows,
        initial_train_label=initial_train,
        initial_evaluation_label=initial_evaluation,
        initial_train_weight=initial_train_weight,
        initial_evaluation_weight=initial_evaluation_weight,
        study=study,
    )
    group_dir = output_root / "folds" / f"fold_{year}" / f"h{horizon:02d}"
    _write_group_material(
        output_dir=group_dir,
        study=study,
        inputs=inputs,
        fold=fold,
        datasets=datasets,
    )
    completed = 0
    for target in pending:
        values = inputs.label_values(target, horizon)
        train_raw = np.asarray(values[fold.train_rows])
        evaluation_raw = np.asarray(values[fold.evaluation_rows])
        train_valid = _valid_target(train_raw, target)
        evaluation_valid = _valid_target(evaluation_raw, target)
        train_label, train_weight = aligned_target_weights(
            values=train_raw,
            date_idx=inputs.candidate_date_idx[fold.train_rows],
            valid=train_valid,
        )
        evaluation_label, evaluation_weight = aligned_target_weights(
            values=evaluation_raw,
            date_idx=inputs.candidate_date_idx[fold.evaluation_rows],
            valid=evaluation_valid,
        )
        train_one_model(
            study=study,
            inputs=inputs,
            datasets=datasets,
            target=target,
            horizon=horizon,
            year=year,
            train_label=train_label,
            evaluation_label=evaluation_label,
            metric_evaluation_label=evaluation_raw,
            train_weight=train_weight,
            evaluation_weight=evaluation_weight,
            output_dir=_path_task_dir(output_root, year, horizon, target),
            events_path=events_path,
        )
        completed += 1
    _release_training_memory(datasets)
    return completed


def _result_ranking_fields(result: Mapping[str, Any]) -> dict[str, Any]:
    target = str(result["target"])
    metrics = dict(result["metrics"])
    if target in REGRESSION_TARGETS:
        ranking = dict(metrics["ranking"])
        return {
            "rank_ic": float(ranking["rank_ic_mean"]),
            "decile_spearman": float(ranking["decile_spearman"]),
            "top_1pct_lift": float(ranking["top_1pct_lift"]),
            "top_5pct_lift": float(ranking["top_5pct_lift"]),
            "top_5pct_positive_rate_lift": float(
                ranking["top_5pct_positive_rate_lift"]
            ),
            "rank_ic_hac_p_value": float(
                ranking["rank_ic_hac"]["p_value_two_sided"]
            ),
            "probability_skill": None,
        }
    if target == "state":
        ordinal = dict(metrics["ordinal_ranking"])
        high = dict(metrics["high_state_probability"])
        high_ranking = dict(high["ranking"])
        return {
            "rank_ic": float(ordinal["rank_ic_mean"]),
            "decile_spearman": float(ordinal["decile_spearman"]),
            "top_1pct_lift": float(high_ranking["top_1pct_lift"]),
            "top_5pct_lift": float(high_ranking["top_5pct_lift"]),
            "top_5pct_positive_rate_lift": float(
                high_ranking["top_5pct_lift"]
            ),
            "rank_ic_hac_p_value": float(
                ordinal["rank_ic_hac"]["p_value_two_sided"]
            ),
            "probability_skill": float(high["brier_skill"]),
            "multiclass_brier_skill": float(metrics["multiclass_brier_skill"]),
        }
    raise ValueError(f"target is not a path decision candidate: {target}")


def decide_from_results(
    results: Sequence[Mapping[str, Any]],
    decision_config: Mapping[str, Any],
) -> dict[str, Any]:
    thresholds = dict(decision_config["qualification_thresholds"])
    candidates: list[dict[str, Any]] = []
    for target in PATH_TARGETS:
        for horizon in HORIZONS:
            current = sorted(
                [
                    item
                    for item in results
                    if str(item.get("target")) == target
                    and int(item.get("horizon", -1)) == horizon
                ],
                key=lambda item: int(item["fold_year"]),
            )
            if len(current) != len(FOLD_YEARS):
                raise ValueError(f"incomplete decision evidence for {target} D{horizon}")
            annual = [
                {
                    "year": int(item["fold_year"]),
                    **_result_ranking_fields(item),
                }
                for item in current
            ]
            all_positive_rank_ic = all(item["rank_ic"] > 0.0 for item in annual)
            all_positive_top5 = all(item["top_5pct_lift"] > 0.0 for item in annual)
            top1_positive_years = sum(
                item["top_1pct_lift"] > 0.0 for item in annual
            )
            minimum_monotonicity = float(
                thresholds["minimum_each_year_decile_spearman"]
            )
            strong_monotonicity = float(
                thresholds["strong_decile_spearman"]
            )
            all_monotone = all(
                item["decile_spearman"] >= minimum_monotonicity
                for item in annual
            )
            strong_monotone_years = sum(
                item["decile_spearman"] >= strong_monotonicity
                for item in annual
            )
            hac_supported_years = sum(
                item["rank_ic"] > 0.0
                and item["rank_ic_hac_p_value"]
                < float(thresholds["maximum_hac_p_value"])
                for item in annual
            )
            positive_rate_gate = True
            if target in {"g", "mfe"}:
                positive_rate_gate = all(
                    item["top_5pct_positive_rate_lift"] > 0.0
                    for item in annual
                )
            probability_gate = True
            if target == "state":
                probability_gate = all(
                    float(item["probability_skill"]) > 0.0
                    and float(item["multiclass_brier_skill"]) > 0.0
                    for item in annual
                )
            worst_rank_ic = min(item["rank_ic"] for item in annual)
            directional_stable = bool(
                all_positive_rank_ic
                and all_positive_top5
                and all_monotone
                and positive_rate_gate
                and probability_gate
            )
            qualified = bool(
                directional_stable
                and worst_rank_ic
                >= float(thresholds["minimum_worst_year_rank_ic"])
                and top1_positive_years
                >= int(thresholds["minimum_top1_positive_years"])
                and strong_monotone_years
                >= int(thresholds["minimum_strong_monotone_years"])
                and hac_supported_years
                >= int(thresholds["minimum_hac_supported_years"])
            )
            candidates.append(
                {
                    "target": target,
                    "horizon": horizon,
                    "annual": annual,
                    "worst_year_rank_ic": float(worst_rank_ic),
                    "median_annual_rank_ic": float(
                        np.median([item["rank_ic"] for item in annual])
                    ),
                    "top1_positive_years": int(top1_positive_years),
                    "strong_monotone_years": int(strong_monotone_years),
                    "hac_supported_years": int(hac_supported_years),
                    "directionally_stable": directional_stable,
                    "qualified": qualified,
                    "gates": {
                        "all_positive_rank_ic": bool(all_positive_rank_ic),
                        "all_positive_top5_lift": bool(all_positive_top5),
                        "all_decile_monotone": bool(all_monotone),
                        "positive_rate_enrichment": bool(positive_rate_gate),
                        "probability_skill": bool(probability_gate),
                    },
                }
            )

    family_rows: list[dict[str, Any]] = []
    for target in ("g", "mfe", "state"):
        qualified = [
            item for item in candidates if item["target"] == target and item["qualified"]
        ]
        directional = [
            item
            for item in candidates
            if item["target"] == target and item["directionally_stable"]
        ]
        family_rows.append(
            {
                "target": target,
                "qualified_horizons": [int(item["horizon"]) for item in qualified],
                "directionally_stable_horizons": [
                    int(item["horizon"]) for item in directional
                ],
                "qualified_horizon_count": len(qualified),
                "median_qualified_worst_year_rank_ic": (
                    float(np.median([item["worst_year_rank_ic"] for item in qualified]))
                    if qualified
                    else None
                ),
            }
        )
    priority = {"g": 0, "mfe": 1, "state": 2}
    ordered_families = sorted(
        family_rows,
        key=lambda item: (
            -int(item["qualified_horizon_count"]),
            -float(item["median_qualified_worst_year_rank_ic"] or -math.inf),
            priority[str(item["target"])],
        ),
    )
    winner = ordered_families[0]
    if int(winner["qualified_horizon_count"]) == 0:
        primary_target = None
        verdict = "current_inputs_not_sufficient_for_successor_target"
        output_structure = None
        retained_horizons: list[int] = []
    else:
        primary_target = str(winner["target"])
        retained_horizons = list(winner["qualified_horizons"])
        if primary_target == "g":
            verdict = "learn_scalar_endpoint_return"
            output_structure = "one independent scalar g_H score per retained horizon"
        elif primary_target == "mfe":
            verdict = "learn_upside_opportunity_then_manage_realized_path"
            output_structure = "one independent scalar mfe_H opportunity score per retained horizon"
        else:
            verdict = "learn_low_mid_high_path_state_probabilities"
            output_structure = "three calibrated state_H probabilities per retained horizon"
    mae_horizons = [
        int(item["horizon"])
        for item in candidates
        if item["target"] == "pre_peak_mae" and item["qualified"]
    ]
    return {
        "schema": "seq100_path_label_learnability_decision/v1",
        "decision_rule": "qualified horizon count, then median worst-year Rank IC, then simplicity tie-break g > mfe > state",
        "verdict": verdict,
        "primary_target": primary_target,
        "output_structure": output_structure,
        "retained_horizons": retained_horizons,
        "pre_peak_mae_auxiliary_horizons": mae_horizons,
        "family_summary": family_rows,
        "candidate_summary": candidates,
        "thresholds": thresholds,
        "does_not_select": [
            "exit_rule",
            "holding_period",
            "slot_count",
            "leverage",
            "stop_loss",
            "additional_future_confirmation",
        ],
    }


def _load_complete_results(
    output_root: Path,
    study: Mapping[str, Any],
) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    expected: list[tuple[Path, int, int, str]] = []
    for year in FOLD_YEARS:
        expected.append(
            (
                _entry_task_dir(output_root, year) / "task_result.json",
                year,
                1,
                "entry_unfilled",
            )
        )
        for horizon in HORIZONS:
            for target in PATH_TARGETS:
                expected.append(
                    (
                        _path_task_dir(output_root, year, horizon, target)
                        / "task_result.json",
                        year,
                        horizon,
                        target,
                    )
                )
    for path, year, horizon, target in expected:
        expected_semantics = _task_semantics(
            study,
            year=year,
            horizon=horizon,
            target=target,
        )
        if not _task_result_complete(path, expected=expected_semantics):
            raise RuntimeError(f"missing or changed learnability task result: {path}")
        results.append(json.loads(path.read_text(encoding="utf-8")))
    return results


def aggregate_results(
    *,
    study_path: Path = DEFAULT_STUDY_PATH,
    output_root: Path = DEFAULT_OUTPUT_ROOT,
) -> dict[str, Any]:
    study = load_study(study_path)
    results = _load_complete_results(output_root, study)
    path_results = [item for item in results if item["target"] in PATH_TARGETS]
    decision = decide_from_results(path_results, dict(study["decision"]))
    entry = [
        {
            "fold_year": int(item["fold_year"]),
            "metrics": item["metrics"],
            "best_iteration": int(item["best_iteration"]),
        }
        for item in results
        if item["target"] == "entry_unfilled"
    ]
    summary = {
        "schema": "seq100_path_label_learnability_summary/v1",
        "status": "completed",
        "completed_at": _now(),
        "study_id": STUDY_ID,
        "fold_years": list(FOLD_YEARS),
        "horizons": list(HORIZONS),
        "task_count": len(results),
        "path_task_count": len(path_results),
        "entry_unfilled_audit": entry,
        "decision": decision,
        "disclosure": {
            "years_2023_2025": "burned discovery/evaluation folds, not pristine holdouts",
            "early_stopping": "each fold year participates only in early stopping and evaluation for that fold under the common rule",
            "outcome_boundary": str(
                dict(study.get("folds", {})).get("maximum_outcome_date", "")
            ),
            "economic_use": "statistical learnability does not establish account-level economic usability",
        },
    }
    _atomic_write_json(output_root / "decision.json", decision)
    _atomic_write_json(output_root / "summary.json", summary)
    return summary


def run_study(
    *,
    study_path: Path = DEFAULT_STUDY_PATH,
    output_root: Path = DEFAULT_OUTPUT_ROOT,
) -> dict[str, Any]:
    study = load_study(study_path)
    inputs = LearnabilityInputs(study)
    output_root.mkdir(parents=True, exist_ok=True)
    events_path = output_root / "events.jsonl"
    progress_path = output_root / "progress.json"
    started_at = _now()
    _atomic_write_json(
        progress_path,
        _progress_payload(
            study=study,
            status="running",
            phase="training",
            output_root=output_root,
            started_at=started_at,
        ),
    )
    try:
        for year in FOLD_YEARS:
            current = f"fold_{year}_entry_unfilled"
            _atomic_write_json(
                progress_path,
                _progress_payload(
                    study=study,
                    status="running",
                    phase="training",
                    output_root=output_root,
                    started_at=started_at,
                    current_task=current,
                ),
            )
            _run_entry_fold(
                study=study,
                inputs=inputs,
                output_root=output_root,
                events_path=events_path,
                year=year,
            )
            for horizon in HORIZONS:
                current = f"fold_{year}_h{horizon:02d}"
                _atomic_write_json(
                    progress_path,
                    _progress_payload(
                        study=study,
                        status="running",
                        phase="training",
                        output_root=output_root,
                        started_at=started_at,
                        current_task=current,
                    ),
                )
                _run_path_group(
                    study=study,
                    inputs=inputs,
                    output_root=output_root,
                    events_path=events_path,
                    year=year,
                    horizon=horizon,
                )
        summary = aggregate_results(study_path=study_path, output_root=output_root)
        progress = _progress_payload(
            study=study,
            status="completed",
            phase="completed",
            output_root=output_root,
            started_at=started_at,
        )
        progress["summary"] = str((output_root / "summary.json").resolve())
        progress["verdict"] = summary["decision"]["verdict"]
        _atomic_write_json(progress_path, progress)
        _append_event(
            events_path,
            "study_completed",
            verdict=summary["decision"]["verdict"],
            **_memory_snapshot(),
        )
        return summary
    except BaseException as exc:
        progress = _progress_payload(
            study=study,
            status="failed",
            phase="failed",
            output_root=output_root,
            started_at=started_at,
        )
        progress["error_type"] = type(exc).__name__
        progress["error"] = str(exc)
        _atomic_write_json(progress_path, progress)
        _append_event(
            events_path,
            "study_failed",
            error_type=type(exc).__name__,
            error=str(exc),
            **_memory_snapshot(),
        )
        raise


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train or summarize the Seq100 future-path learnability study."
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_STUDY_PATH)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("run")
    subparsers.add_parser("decide")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "run":
        result = run_study(
            study_path=args.config,
            output_root=args.output_root,
        )
    else:
        result = aggregate_results(
            study_path=args.config,
            output_root=args.output_root,
        )
    print(json.dumps(result, ensure_ascii=False, default=_json_default), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
