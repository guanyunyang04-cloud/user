from __future__ import annotations

import argparse
import json
import math
import os
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime
from itertools import combinations, pairwise
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats

from daily_research.path_policy import seq100_mfe_feature_family_audit as source
from daily_research.path_policy import seq100_mfe_objective_alignment as objective
from daily_research.path_policy import seq100_path_label_learnability as base

WORKSPACE_ROOT = source.WORKSPACE_ROOT
STUDY_ID = "seq100_entry_role_synthesis_v1"
DEFAULT_STUDY_PATH = (
    WORKSPACE_ROOT / "daily_research/studies/seq100_entry_role_synthesis_v1.json"
)
DEFAULT_OUTPUT_ROOT = (
    WORKSPACE_ROOT
    / "daily_research/output/path_policy/studies/seq100_entry_role_synthesis_v1"
)
FOLD_YEARS = (2023, 2024, 2025)
TARGET_HORIZONS = (10, 20)
EXPECTED_VARIANTS = {
    10: ("baseline", "turnover_cost_proxy"),
    20: ("baseline", "breakout_retest_levels", "traditional_indicators"),
}
SUMMARY_SCHEMA = "seq100_entry_role_synthesis_summary/v1"

PATH_METRICS = (
    "mfe_mean",
    "mfe_tail_rate",
    "early_opportunity_share",
    "peak_day_mean",
    "pre_peak_mae_mean",
    "deep_adverse_rate",
    "endpoint_return_mean",
    "low_state_rate",
    "mid_state_rate",
    "high_state_rate",
    "state_expected_mean",
)


def _now() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def _json_default(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"cannot serialize {type(value)!r}")


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, default=_json_default) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _resolve(value: str | Path) -> Path:
    path = Path(value)
    return (path if path.is_absolute() else WORKSPACE_ROOT / path).resolve()


def _file_record(path: Path) -> dict[str, Any]:
    resolved = path.resolve()
    try:
        display = resolved.relative_to(WORKSPACE_ROOT).as_posix()
    except ValueError:
        display = str(resolved)
    return {"path": display, "size": int(resolved.stat().st_size)}


def _write_parquet(path: Path, frame: pd.DataFrame) -> dict[str, Any]:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(path, index=False, compression="zstd")
    return _file_record(path)


def load_study(path: Path = DEFAULT_STUDY_PATH) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("study_id") != STUDY_ID:
        raise ValueError(f"study_id must be {STUDY_ID}")
    folds = dict(payload["folds"])
    if tuple(int(value) for value in folds["fold_years"]) != FOLD_YEARS:
        raise ValueError("entry-role fold years changed")
    if folds["maximum_outcome_date"] != "2025-12-31":
        raise ValueError("entry-role audit may not consume outcomes after 2025")
    if bool(payload["models"].get("train_new_boosters", True)):
        raise ValueError("entry-role audit may not train a booster")
    variants = {
        int(horizon): tuple(values)
        for horizon, values in payload["models"]["mfe_variants"].items()
    }
    if variants != EXPECTED_VARIANTS:
        raise ValueError("entry-role MFE variants changed")
    evaluation = dict(payload["evaluation"])
    top_fractions = tuple(float(value) for value in evaluation["mfe_top_fractions"])
    retention = tuple(float(value) for value in evaluation["retention_fractions"])
    if top_fractions != (0.01, 0.05):
        raise ValueError("entry-role high-MFE fractions changed")
    if sorted(set(retention)) != list(retention) or retention[-1] != 1.0:
        raise ValueError(
            "retention fractions must be unique, ascending, and end at one"
        )
    if float(evaluation["reference_retention_fraction"]) not in retention:
        raise ValueError("reference retention must be present in the curve")
    if float(evaluation["reference_top_fraction"]) not in top_fractions:
        raise ValueError("reference MFE fraction must be present in the curve")
    if not bool(payload["decision"].get("no_fusion")):
        raise ValueError("entry-role audit must not fit or select a fused score")
    return payload


@dataclass(frozen=True)
class ScoreBundle:
    year: int
    horizon: int
    name: str
    rows: np.ndarray
    prediction: np.ndarray
    result: dict[str, Any]


@dataclass(frozen=True)
class PathActual:
    rows: np.ndarray
    date_idx: np.ndarray
    target_mfe: np.ndarray
    early_mfe: np.ndarray
    peak_day: np.ndarray
    pre_peak_mae: np.ndarray
    endpoint_return: np.ndarray
    state: np.ndarray
    mfe_tail_event: np.ndarray
    deep_adverse_event: np.ndarray


def _load_mfe_bundle(
    *,
    feature_study: Mapping[str, Any],
    output_root: Path,
    year: int,
    horizon: int,
    variant: str,
) -> ScoreBundle:
    result, prediction, rows = source._load_outer_result(
        output_root,
        year=year,
        horizon=horizon,
        variant=variant,
        study=feature_study,
    )
    return ScoreBundle(
        year=int(year),
        horizon=int(horizon),
        name=str(variant),
        rows=np.asarray(rows, dtype=np.int64),
        prediction=np.asarray(prediction, dtype=np.float32),
        result=result,
    )


def _load_path_bundle(
    *,
    path_study: Mapping[str, Any],
    output_root: Path,
    year: int,
    horizon: int,
    target: str,
) -> ScoreBundle:
    result_path = (
        output_root
        / "folds"
        / f"fold_{int(year)}"
        / f"h{int(horizon):02d}"
        / str(target)
        / "task_result.json"
    )
    expected = base._task_semantics(
        path_study,
        year=year,
        horizon=horizon,
        target=target,
    )
    if not base._task_result_complete(result_path, expected=expected):
        raise ValueError(f"path prediction is incomplete: {result_path}")
    result = json.loads(result_path.read_text(encoding="utf-8"))
    group_path = result_path.parent.parent / "group_manifest.json"
    group = json.loads(group_path.read_text(encoding="utf-8"))
    rows_path = base._verify_file_record(group["files"]["evaluation_rows"])
    prediction_path = base._verify_file_record(result["files"]["prediction"])
    rows = np.load(rows_path, allow_pickle=False)
    prediction = np.load(prediction_path, allow_pickle=False)
    if len(prediction) != len(rows):
        raise ValueError(
            f"path prediction row count differs: {target} D{horizon} {year}"
        )
    return ScoreBundle(
        year=int(year),
        horizon=int(horizon),
        name=str(target),
        rows=np.asarray(rows, dtype=np.int64),
        prediction=np.asarray(prediction, dtype=np.float32),
        result=result,
    )


def _within_date_event(
    *, date_idx: np.ndarray, values: np.ndarray, fraction: float, upper: bool
) -> np.ndarray:
    dates = np.asarray(date_idx, dtype=np.int32)
    current_values = np.asarray(values, dtype=np.float64)
    result = np.zeros(len(dates), dtype=bool)
    for start, stop in pairwise(objective._date_boundaries(dates)):
        valid = np.isfinite(current_values[start:stop])
        local = np.flatnonzero(valid)
        if not local.size:
            continue
        order = np.argsort(current_values[start:stop][valid], kind="mergesort")
        count = max(1, math.ceil(float(fraction) * len(order)))
        selected = order[-count:] if upper else order[:count]
        result[start + local[selected]] = True
    return result


def _path_actual(
    *,
    inputs: base.LearnabilityInputs,
    reader: objective.FuturePathReader,
    rows: np.ndarray,
    horizon: int,
    early_horizon: int,
) -> PathActual:
    candidate_rows = np.asarray(rows, dtype=np.int64)
    dates = np.asarray(inputs.candidate_date_idx[candidate_rows], dtype=np.int32)
    target = np.asarray(
        inputs.label_values("mfe", horizon)[candidate_rows], dtype=np.float32
    )
    early = np.asarray(
        inputs.label_values("mfe", early_horizon)[candidate_rows], dtype=np.float32
    )
    adverse = np.asarray(
        inputs.label_values("pre_peak_mae", horizon)[candidate_rows], dtype=np.float32
    )
    endpoint = np.asarray(
        inputs.label_values("g", horizon)[candidate_rows], dtype=np.float32
    )
    state = np.asarray(
        inputs.label_values("state", horizon)[candidate_rows], dtype=np.int8
    )
    peak_day = reader.peak_day(
        inputs=inputs,
        rows=candidate_rows,
        horizon=horizon,
        expected_mfe=target,
    )
    valid_target = np.isfinite(target)
    invalid_companion = valid_target & ~(
        np.isfinite(peak_day)
        & np.isfinite(adverse)
        & np.isfinite(endpoint)
        & np.isin(state, (0, 1, 2))
    )
    if bool(invalid_companion.any()):
        raise ValueError(f"finite D{horizon} MFE lacks a required path companion")
    return PathActual(
        rows=candidate_rows,
        date_idx=dates,
        target_mfe=target,
        early_mfe=early,
        peak_day=peak_day,
        pre_peak_mae=adverse,
        endpoint_return=endpoint,
        state=state,
        mfe_tail_event=_within_date_event(
            date_idx=dates, values=target, fraction=0.20, upper=True
        ),
        deep_adverse_event=_within_date_event(
            date_idx=dates, values=adverse, fraction=0.20, upper=False
        ),
    )


def _top_indices(values: np.ndarray, fraction: float) -> np.ndarray:
    scores = np.asarray(values, dtype=np.float64)
    count = max(1, math.ceil(float(fraction) * len(scores)))
    return np.argsort(-scores, kind="mergesort")[:count]


def _finite_mean(values: np.ndarray) -> float:
    current = np.asarray(values, dtype=np.float64)
    current = current[np.isfinite(current)]
    return float(current.mean()) if current.size else math.nan


def _positive_share(early: np.ndarray, target: np.ndarray) -> float:
    early_values = np.asarray(early, dtype=np.float64)
    target_values = np.asarray(target, dtype=np.float64)
    numerator = float(
        np.where(np.isfinite(early_values), np.maximum(early_values, 0.0), 0.0).sum()
    )
    denominator = float(
        np.where(np.isfinite(target_values), np.maximum(target_values, 0.0), 0.0).sum()
    )
    return numerator / denominator if denominator > 1.0e-12 else math.nan


def _selection_metrics(
    indices: np.ndarray,
    *,
    target_mfe: np.ndarray,
    early_mfe: np.ndarray,
    peak_day: np.ndarray,
    pre_peak_mae: np.ndarray,
    endpoint_return: np.ndarray,
    state: np.ndarray,
    mfe_tail_event: np.ndarray,
    deep_adverse_event: np.ndarray,
) -> dict[str, Any]:
    selected = np.asarray(indices, dtype=np.int64)
    if not selected.size:
        return {"selected_count": 0, **{metric: math.nan for metric in PATH_METRICS}}
    selected_state = np.asarray(state[selected], dtype=np.int8)
    return {
        "selected_count": len(selected),
        "mfe_mean": _finite_mean(target_mfe[selected]),
        "mfe_tail_rate": _finite_mean(mfe_tail_event[selected]),
        "early_opportunity_share": _positive_share(
            early_mfe[selected], target_mfe[selected]
        ),
        "peak_day_mean": _finite_mean(peak_day[selected]),
        "pre_peak_mae_mean": _finite_mean(pre_peak_mae[selected]),
        "deep_adverse_rate": _finite_mean(deep_adverse_event[selected]),
        "endpoint_return_mean": _finite_mean(endpoint_return[selected]),
        "low_state_rate": _finite_mean(selected_state == 0),
        "mid_state_rate": _finite_mean(selected_state == 1),
        "high_state_rate": _finite_mean(selected_state == 2),
        "state_expected_mean": _finite_mean(selected_state),
    }


def _metric_delta(
    upper: Mapping[str, Any], lower: Mapping[str, Any]
) -> dict[str, float]:
    return {
        metric: float(upper[metric]) - float(lower[metric]) for metric in PATH_METRICS
    }


def _safe_correlation(left: np.ndarray, right: np.ndarray, *, method: str) -> float:
    x = np.asarray(left, dtype=np.float64)
    y = np.asarray(right, dtype=np.float64)
    valid = np.isfinite(x) & np.isfinite(y)
    if (
        int(valid.sum()) < 3
        or np.std(x[valid]) <= 1.0e-12
        or np.std(y[valid]) <= 1.0e-12
    ):
        return math.nan
    if method == "spearman":
        return float(stats.spearmanr(x[valid], y[valid]).statistic)
    if method == "pearson":
        return float(stats.pearsonr(x[valid], y[valid]).statistic)
    raise ValueError(f"unknown correlation method: {method}")


def _daily_model_relationship(
    *,
    actual: PathActual,
    left_name: str,
    left_score: np.ndarray,
    right_name: str,
    right_score: np.ndarray,
) -> pd.DataFrame:
    left_values = np.asarray(left_score, dtype=np.float64)
    right_values = np.asarray(right_score, dtype=np.float64)
    if (
        left_values.shape != actual.date_idx.shape
        or right_values.shape != actual.date_idx.shape
    ):
        raise ValueError("model relationship arrays are not aligned")
    records: list[dict[str, Any]] = []
    for start, stop in pairwise(objective._date_boundaries(actual.date_idx)):
        valid = (
            np.isfinite(left_values[start:stop])
            & np.isfinite(right_values[start:stop])
            & np.isfinite(actual.target_mfe[start:stop])
            & np.isfinite(actual.peak_day[start:stop])
            & np.isfinite(actual.pre_peak_mae[start:stop])
            & np.isfinite(actual.endpoint_return[start:stop])
            & np.isin(actual.state[start:stop], (0, 1, 2))
        )
        local = np.flatnonzero(valid)
        if len(local) < 20:
            continue
        left = left_values[start:stop][valid]
        right = right_values[start:stop][valid]
        arrays = {
            "target_mfe": actual.target_mfe[start:stop][valid],
            "early_mfe": actual.early_mfe[start:stop][valid],
            "peak_day": actual.peak_day[start:stop][valid],
            "pre_peak_mae": actual.pre_peak_mae[start:stop][valid],
            "endpoint_return": actual.endpoint_return[start:stop][valid],
            "state": actual.state[start:stop][valid],
            "mfe_tail_event": actual.mfe_tail_event[start:stop][valid],
            "deep_adverse_event": actual.deep_adverse_event[start:stop][valid],
        }
        row: dict[str, Any] = {
            "date_idx": int(actual.date_idx[start]),
            "candidate_count": len(local),
            "prediction_spearman": _safe_correlation(left, right, method="spearman"),
            "prediction_pearson": _safe_correlation(left, right, method="pearson"),
        }
        for label, fraction in (("top1", 0.01), ("top5", 0.05)):
            left_top = _top_indices(left, fraction)
            right_top = _top_indices(right, fraction)
            left_mask = np.zeros(len(left), dtype=bool)
            right_mask = np.zeros(len(right), dtype=bool)
            left_mask[left_top] = True
            right_mask[right_top] = True
            intersection = int(np.sum(left_mask & right_mask))
            union = int(np.sum(left_mask | right_mask))
            row[f"{label}_overlap_rate"] = intersection / float(len(left_top))
            row[f"{label}_jaccard"] = intersection / float(union)
            left_only = np.flatnonzero(left_mask & ~right_mask)
            right_only = np.flatnonzero(right_mask & ~left_mask)
            left_metrics = _selection_metrics(left_only, **arrays)
            right_metrics = _selection_metrics(right_only, **arrays)
            row[f"{label}_left_only_count"] = len(left_only)
            row[f"{label}_right_only_count"] = len(right_only)
            for metric, value in _metric_delta(right_metrics, left_metrics).items():
                row[f"{label}_right_only_minus_left_only_{metric}"] = value

        for host_name, host, guest_name, guest in (
            (left_name, left, right_name, right),
            (right_name, right, left_name, left),
        ):
            host_top = _top_indices(host, 0.05)
            guest_inside = guest[host_top]
            target_inside = arrays["target_mfe"][host_top]
            prefix = f"{guest_name}_within_{host_name}_top5"
            row[f"{prefix}_rank_ic"] = _safe_correlation(
                guest_inside, target_inside, method="spearman"
            )
            order = np.argsort(guest_inside, kind="mergesort")
            half = max(1, len(order) // 2)
            low = host_top[order[:half]]
            high = host_top[order[-half:]]
            low_metrics = _selection_metrics(low, **arrays)
            high_metrics = _selection_metrics(high, **arrays)
            for metric, value in _metric_delta(high_metrics, low_metrics).items():
                row[f"{prefix}_high_minus_low_{metric}"] = value
        records.append(row)
    frame = pd.DataFrame(records)
    if frame.empty:
        raise ValueError("model relationship has no valid dates")
    return frame


def _summarize_daily_frame(
    frame: pd.DataFrame, *, horizon: int, excluded: Sequence[str] = ("date_idx",)
) -> dict[str, Any]:
    metrics: dict[str, Any] = {}
    for column in frame.columns:
        if column in excluded:
            continue
        values = frame[column].to_numpy(dtype=np.float64)
        metrics[column] = {
            "mean": _finite_mean(values),
            "hac": base._hac_mean_test(values, max(int(horizon) - 1, 0)),
        }
    return {"date_count": int(frame["date_idx"].nunique()), "metrics": metrics}


def _compact_role_metrics(metrics: Mapping[str, Any]) -> dict[str, Any]:
    broad = dict(metrics["broad_ranking"])
    tail = dict(metrics["upper_tail"])
    path = dict(metrics["path_quality"])
    return {
        "rank_ic": float(broad["rank_ic_mean"]),
        "top1_mfe_mean": float(broad["top_1pct_mean"]),
        "top1_mfe_lift": float(broad["top_1pct_lift"]),
        "top5_mfe_mean": float(broad["top_5pct_mean"]),
        "top5_mfe_lift": float(broad["top_5pct_lift"]),
        "top1_mfe_tail_rate": float(path["top1_daily_tail_rate"]),
        "top1_mfe_tail_lift": float(tail["daily_tail_top1_lift"]),
        "top5_mfe_tail_rate": float(path["top5_daily_tail_rate"]),
        "top5_mfe_tail_lift": float(tail["daily_tail_top5_lift"]),
        **{key: value for key, value in path.items() if not str(key).endswith("_hac")},
    }


def _conditional_daily_frames(
    *,
    actual: PathActual,
    host_score: np.ndarray,
    auxiliary_score: np.ndarray,
    top_fractions: Sequence[float],
    retention_fractions: Sequence[float],
    quantile_count: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    host_values = np.asarray(host_score, dtype=np.float64)
    auxiliary_values = np.asarray(auxiliary_score, dtype=np.float64)
    if (
        host_values.shape != actual.date_idx.shape
        or auxiliary_values.shape != actual.date_idx.shape
    ):
        raise ValueError("conditional score arrays are not aligned")
    curve_rows: list[dict[str, Any]] = []
    grid_rows: list[dict[str, Any]] = []
    for start, stop in pairwise(objective._date_boundaries(actual.date_idx)):
        valid = (
            np.isfinite(host_values[start:stop])
            & np.isfinite(auxiliary_values[start:stop])
            & np.isfinite(actual.target_mfe[start:stop])
            & np.isfinite(actual.peak_day[start:stop])
            & np.isfinite(actual.pre_peak_mae[start:stop])
            & np.isfinite(actual.endpoint_return[start:stop])
            & np.isin(actual.state[start:stop], (0, 1, 2))
        )
        local = np.flatnonzero(valid)
        if len(local) < 20:
            continue
        host = host_values[start:stop][valid]
        auxiliary = auxiliary_values[start:stop][valid]
        arrays = {
            "target_mfe": actual.target_mfe[start:stop][valid],
            "early_mfe": actual.early_mfe[start:stop][valid],
            "peak_day": actual.peak_day[start:stop][valid],
            "pre_peak_mae": actual.pre_peak_mae[start:stop][valid],
            "endpoint_return": actual.endpoint_return[start:stop][valid],
            "state": actual.state[start:stop][valid],
            "mfe_tail_event": actual.mfe_tail_event[start:stop][valid],
            "deep_adverse_event": actual.deep_adverse_event[start:stop][valid],
        }
        for top_fraction in top_fractions:
            host_top = _top_indices(host, float(top_fraction))
            order_descending = host_top[
                np.argsort(-auxiliary[host_top], kind="mergesort")
            ]
            for retention in retention_fractions:
                count = max(1, math.ceil(float(retention) * len(host_top)))
                selected = order_descending[:count]
                curve_rows.append(
                    {
                        "date_idx": int(actual.date_idx[start]),
                        "candidate_count": len(local),
                        "mfe_top_fraction": float(top_fraction),
                        "retention_fraction": float(retention),
                        **_selection_metrics(selected, **arrays),
                    }
                )
            order_ascending = host_top[
                np.argsort(auxiliary[host_top], kind="mergesort")
            ]
            for quantile, selected in enumerate(
                np.array_split(order_ascending, int(quantile_count)), start=1
            ):
                grid_rows.append(
                    {
                        "date_idx": int(actual.date_idx[start]),
                        "candidate_count": len(local),
                        "mfe_top_fraction": float(top_fraction),
                        "auxiliary_quantile": int(quantile),
                        **_selection_metrics(selected, **arrays),
                    }
                )
    curve = pd.DataFrame(curve_rows)
    grid = pd.DataFrame(grid_rows)
    if curve.empty or grid.empty:
        raise ValueError("conditional evaluation has no valid dates")
    return curve, grid


def _aggregate_conditional_curve(
    frame: pd.DataFrame, *, horizon: int
) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for top_fraction in sorted(frame["mfe_top_fraction"].unique()):
        current_top = frame[frame["mfe_top_fraction"] == top_fraction]
        unfiltered = current_top[current_top["retention_fraction"] == 1.0][
            ["date_idx", *PATH_METRICS]
        ].rename(columns={metric: f"{metric}_unfiltered" for metric in PATH_METRICS})
        for retention in sorted(current_top["retention_fraction"].unique()):
            selected = current_top[current_top["retention_fraction"] == retention]
            paired = selected.merge(unfiltered, on="date_idx", validate="one_to_one")
            metric_summary: dict[str, Any] = {}
            for metric in PATH_METRICS:
                values = paired[metric].to_numpy(dtype=np.float64)
                baseline_values = paired[f"{metric}_unfiltered"].to_numpy(
                    dtype=np.float64
                )
                delta = values - baseline_values
                metric_summary[metric] = {
                    "mean": _finite_mean(values),
                    "unfiltered_mean": _finite_mean(baseline_values),
                    "mean_delta": _finite_mean(delta),
                    "delta_hac": base._hac_mean_test(delta, max(int(horizon) - 1, 0)),
                }
            denominator = float(metric_summary["mfe_mean"]["unfiltered_mean"])
            ratio = (
                float(metric_summary["mfe_mean"]["mean"]) / denominator
                if math.isfinite(denominator) and abs(denominator) > 1.0e-12
                else math.nan
            )
            output.append(
                {
                    "mfe_top_fraction": float(top_fraction),
                    "retention_fraction": float(retention),
                    "date_count": len(paired),
                    "selected_count_mean": float(selected["selected_count"].mean()),
                    "mfe_mean_retention_ratio": ratio,
                    "metrics": metric_summary,
                }
            )
    return output


def _aggregate_conditional_grid(
    frame: pd.DataFrame, *, horizon: int, quantile_count: int
) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for top_fraction in sorted(frame["mfe_top_fraction"].unique()):
        current = frame[frame["mfe_top_fraction"] == top_fraction]
        bins: list[dict[str, Any]] = []
        for quantile in range(1, int(quantile_count) + 1):
            subset = current[current["auxiliary_quantile"] == quantile]
            bins.append(
                {
                    "auxiliary_quantile": int(quantile),
                    "date_count": int(subset["date_idx"].nunique()),
                    "selected_count_mean": float(subset["selected_count"].mean()),
                    "metrics": {
                        metric: _finite_mean(subset[metric].to_numpy())
                        for metric in PATH_METRICS
                    },
                }
            )
        low = current[current["auxiliary_quantile"] == 1][["date_idx", *PATH_METRICS]]
        high = current[current["auxiliary_quantile"] == int(quantile_count)][
            ["date_idx", *PATH_METRICS]
        ]
        paired = low.merge(
            high, on="date_idx", suffixes=("_low", "_high"), validate="one_to_one"
        )
        high_minus_low: dict[str, Any] = {}
        monotonicity: dict[str, float] = {}
        for metric in PATH_METRICS:
            delta = (paired[f"{metric}_high"] - paired[f"{metric}_low"]).to_numpy(
                dtype=np.float64
            )
            high_minus_low[metric] = {
                "mean_delta": _finite_mean(delta),
                "delta_hac": base._hac_mean_test(delta, max(int(horizon) - 1, 0)),
            }
            bin_values = np.asarray(
                [record["metrics"][metric] for record in bins], dtype=np.float64
            )
            monotonicity[metric] = _safe_correlation(
                np.arange(1, int(quantile_count) + 1),
                bin_values,
                method="spearman",
            )
        output.append(
            {
                "mfe_top_fraction": float(top_fraction),
                "bins_low_to_high_auxiliary": bins,
                "highest_minus_lowest": high_minus_low,
                "bin_spearman": monotonicity,
            }
        )
    return output


def _quality_record(
    *,
    inputs: base.LearnabilityInputs,
    actual: PathActual,
    mfe_bundles: Mapping[str, ScoreBundle],
    auxiliary_bundles: Mapping[str, ScoreBundle],
) -> dict[str, Any]:
    rows = actual.rows
    if not bool(np.all(rows[1:] > rows[:-1])):
        raise AssertionError("evaluation rows are not strictly increasing")
    if any(not np.array_equal(bundle.rows, rows) for bundle in mfe_bundles.values()):
        raise AssertionError("MFE variants do not share exact evaluation rows")
    if any(
        not np.array_equal(bundle.rows, rows) for bundle in auxiliary_bundles.values()
    ):
        raise AssertionError("auxiliary predictions do not share exact evaluation rows")
    symbol = np.asarray(inputs.candidate_symbol_idx[rows], dtype=np.int64)
    key = actual.date_idx.astype(np.int64) * (int(symbol.max()) + 2) + symbol
    trade_dates = np.asarray(inputs.date_values[actual.date_idx], dtype=str)
    boundaries = objective._date_boundaries(actual.date_idx)
    date_counts = np.diff(boundaries)
    score_profile = {
        name: {
            "shape": list(bundle.prediction.shape),
            "finite_fraction": float(np.isfinite(bundle.prediction).mean()),
            "minimum": float(np.min(bundle.prediction)),
            "maximum": float(np.max(bundle.prediction)),
        }
        for name, bundle in {**mfe_bundles, **auxiliary_bundles}.items()
    }
    state_probability = auxiliary_bundles.get("state_high_probability_10")
    state_checks: dict[str, Any] | None = None
    if state_probability is not None:
        probability = np.asarray(state_probability.prediction, dtype=np.float64)
        if probability.ndim != 2 or probability.shape[1] != 3:
            raise ValueError("state prediction must have three probability columns")
        state_checks = {
            "minimum_probability": float(probability.min()),
            "maximum_probability": float(probability.max()),
            "maximum_row_sum_error": float(
                np.max(np.abs(probability.sum(axis=1) - 1.0))
            ),
        }
        if (
            state_checks["minimum_probability"] < -1.0e-6
            or state_checks["maximum_probability"] > 1.0 + 1.0e-6
            or state_checks["maximum_row_sum_error"] > 1.0e-5
        ):
            raise ValueError("state probabilities fail probability integrity checks")
    return {
        "year": int(str(trade_dates[0])[:4]),
        "horizon": int(mfe_bundles["baseline"].horizon),
        "evaluation_row_count": len(rows),
        "date_count": int(len(boundaries) - 1),
        "first_trade_date": str(inputs.date_values[int(actual.date_idx[0])]),
        "last_trade_date": str(inputs.date_values[int(actual.date_idx[-1])]),
        "forbidden_2026_row_count": int(
            np.sum(np.char.startswith(trade_dates, "2026-"))
        ),
        "duplicate_candidate_key_count": int(np.sum(key[1:] == key[:-1])),
        "candidate_count_per_date": {
            "minimum": int(date_counts.min()),
            "median": float(np.median(date_counts)),
            "maximum": int(date_counts.max()),
        },
        "required_label_missing_count": {
            "mfe": int(np.sum(~np.isfinite(actual.target_mfe))),
            "peak_day": int(np.sum(~np.isfinite(actual.peak_day))),
            "pre_peak_mae": int(np.sum(~np.isfinite(actual.pre_peak_mae))),
            "endpoint_return": int(np.sum(~np.isfinite(actual.endpoint_return))),
            "state": int(np.sum(~np.isin(actual.state, (0, 1, 2)))),
        },
        "prediction_profile": score_profile,
        "state_probability_integrity": state_checks,
        "exact_row_alignment": True,
    }


def _compile_feature_roles(
    *,
    study: Mapping[str, Any],
    paired: Sequence[Mapping[str, Any]],
    relationships: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    strong_rules = dict(study["decision"]["strong_candidate_role"])
    broad_rules = dict(study["decision"]["broad_ranking_role"])
    replacement_rules = dict(study["decision"]["replacement_evidence"])
    roles: dict[str, Any] = {}
    for horizon, variants in EXPECTED_VARIANTS.items():
        for challenger in variants[1:]:
            annual = sorted(
                (
                    row
                    for row in paired
                    if int(row["horizon"]) == horizon
                    and row["challenger"] == challenger
                ),
                key=lambda row: int(row["year"]),
            )
            relation = sorted(
                (
                    row
                    for row in relationships
                    if int(row["horizon"]) == horizon
                    and row["left"] == "baseline"
                    and row["right"] == challenger
                ),
                key=lambda row: int(row["year"]),
            )
            if len(annual) != len(FOLD_YEARS) or len(relation) != len(FOLD_YEARS):
                raise AssertionError(
                    f"feature role evidence is incomplete: {challenger}"
                )
            rank_ic = [float(row["metrics"]["rank_ic"]["mean_delta"]) for row in annual]
            top5 = [
                float(row["metrics"]["top_5pct_lift"]["mean_delta"]) for row in annual
            ]
            tail = [
                float(row["metrics"]["daily_tail_top5_lift"]["mean_delta"])
                for row in annual
            ]
            exclusive = [
                float(
                    row["summary"]["metrics"][
                        "top5_right_only_minus_left_only_mfe_mean"
                    ]["mean"]
                )
                for row in relation
            ]
            strong_candidate = bool(
                sum(value > 0.0 for value in top5)
                >= int(strong_rules["minimum_positive_top5_mfe_years"])
                and sum(value > 0.0 for value in tail)
                >= int(strong_rules["minimum_positive_tail_hit_years"])
                and min(rank_ic)
                >= float(strong_rules["minimum_worst_year_rank_ic_delta"])
            )
            broad_ranking = bool(
                sum(value > 0.0 for value in rank_ic)
                >= int(broad_rules["minimum_positive_rank_ic_years"])
            )
            exclusive_support = sum(value > 0.0 for value in exclusive)
            replace_baseline = bool(
                strong_candidate
                and exclusive_support
                >= int(
                    replacement_rules["minimum_positive_challenger_exclusive_mfe_years"]
                )
            )
            if strong_candidate and broad_ranking:
                role = "strong_candidate_and_broad_ranking"
            elif strong_candidate:
                role = "strong_candidate_selection"
            elif broad_ranking:
                role = "broad_ranking_only"
            else:
                role = "diagnostic_only"
            roles[challenger] = {
                "horizon": int(horizon),
                "role": role,
                "replace_baseline": replace_baseline,
                "rank_ic_delta_2023_2024_2025": rank_ic,
                "top5_mfe_delta_2023_2024_2025": top5,
                "top5_tail_hit_delta_2023_2024_2025": tail,
                "challenger_only_minus_baseline_only_top5_mfe_2023_2024_2025": exclusive,
                "gate_checks": {
                    "strong_candidate": strong_candidate,
                    "broad_ranking": broad_ranking,
                    "positive_challenger_exclusive_mfe_years": exclusive_support,
                },
            }
    return roles


def _find_curve_point(
    record: Mapping[str, Any], *, top_fraction: float, retention: float
) -> Mapping[str, Any]:
    matches = [
        row
        for row in record["curve"]
        if math.isclose(float(row["mfe_top_fraction"]), float(top_fraction))
        and math.isclose(float(row["retention_fraction"]), float(retention))
    ]
    if len(matches) != 1:
        raise AssertionError(
            "conditional reference curve point is absent or duplicated"
        )
    return matches[0]


def _find_grid_point(
    record: Mapping[str, Any], *, top_fraction: float
) -> Mapping[str, Any]:
    matches = [
        row
        for row in record["grid"]
        if math.isclose(float(row["mfe_top_fraction"]), float(top_fraction))
    ]
    if len(matches) != 1:
        raise AssertionError("conditional reference grid is absent or duplicated")
    return matches[0]


def _classify_conditional_coordinate(
    *,
    study: Mapping[str, Any],
    records: Sequence[Mapping[str, Any]],
    auxiliary: str,
    horizon: int,
    host_variant: str,
) -> dict[str, Any]:
    evaluation = dict(study["evaluation"])
    rules = dict(study["decision"]["conditional_coordinate"])
    top_fraction = float(evaluation["reference_top_fraction"])
    retention = float(evaluation["reference_retention_fraction"])
    annual = sorted(
        (
            row
            for row in records
            if row["auxiliary"] == auxiliary
            and int(row["horizon"]) == int(horizon)
            and row["host_variant"] == host_variant
        ),
        key=lambda row: int(row["year"]),
    )
    if len(annual) != len(FOLD_YEARS):
        raise AssertionError(f"conditional evidence is incomplete: {auxiliary}")
    points = [
        _find_curve_point(row, top_fraction=top_fraction, retention=retention)
        for row in annual
    ]
    grids = [_find_grid_point(row, top_fraction=top_fraction) for row in annual]
    mfe_retention = [float(row["mfe_mean_retention_ratio"]) for row in points]
    tail_delta = [
        float(row["metrics"]["mfe_tail_rate"]["mean_delta"]) for row in points
    ]
    if auxiliary == "state_high_probability_10":
        intended_curve = {
            "endpoint_return_delta": [
                float(row["metrics"]["endpoint_return_mean"]["mean_delta"])
                for row in points
            ],
            "high_state_rate_delta": [
                float(row["metrics"]["high_state_rate"]["mean_delta"]) for row in points
            ],
        }
        intended_grid = {
            "endpoint_return_highest_minus_lowest": [
                float(row["highest_minus_lowest"]["endpoint_return_mean"]["mean_delta"])
                for row in grids
            ],
            "high_state_rate_highest_minus_lowest": [
                float(row["highest_minus_lowest"]["high_state_rate"]["mean_delta"])
                for row in grids
            ],
        }
        curve_pass = all(
            value > 0.0 for values in intended_curve.values() for value in values
        )
        grid_pass = all(
            value > 0.0 for values in intended_grid.values() for value in values
        )
    elif auxiliary == "state_expected_value_10":
        intended_curve = {
            "endpoint_return_delta": [
                float(row["metrics"]["endpoint_return_mean"]["mean_delta"])
                for row in points
            ],
            "state_expected_delta": [
                float(row["metrics"]["state_expected_mean"]["mean_delta"])
                for row in points
            ],
        }
        intended_grid = {
            "endpoint_return_highest_minus_lowest": [
                float(row["highest_minus_lowest"]["endpoint_return_mean"]["mean_delta"])
                for row in grids
            ],
            "state_expected_highest_minus_lowest": [
                float(row["highest_minus_lowest"]["state_expected_mean"]["mean_delta"])
                for row in grids
            ],
        }
        curve_pass = all(
            value > 0.0 for values in intended_curve.values() for value in values
        )
        grid_pass = all(
            value > 0.0 for values in intended_grid.values() for value in values
        )
    elif auxiliary == "state_low_avoidance_10":
        intended_curve = {
            "endpoint_return_delta": [
                float(row["metrics"]["endpoint_return_mean"]["mean_delta"])
                for row in points
            ],
            "low_state_rate_delta": [
                float(row["metrics"]["low_state_rate"]["mean_delta"]) for row in points
            ],
        }
        intended_grid = {
            "endpoint_return_highest_minus_lowest": [
                float(row["highest_minus_lowest"]["endpoint_return_mean"]["mean_delta"])
                for row in grids
            ],
            "low_state_rate_highest_minus_lowest": [
                float(row["highest_minus_lowest"]["low_state_rate"]["mean_delta"])
                for row in grids
            ],
        }
        curve_pass = bool(
            all(value > 0.0 for value in intended_curve["endpoint_return_delta"])
            and all(value < 0.0 for value in intended_curve["low_state_rate_delta"])
        )
        grid_pass = bool(
            all(
                value > 0.0
                for value in intended_grid["endpoint_return_highest_minus_lowest"]
            )
            and all(
                value < 0.0
                for value in intended_grid["low_state_rate_highest_minus_lowest"]
            )
        )
    else:
        intended_curve = {
            "pre_peak_mae_delta": [
                float(row["metrics"]["pre_peak_mae_mean"]["mean_delta"])
                for row in points
            ],
            "deep_adverse_rate_delta": [
                float(row["metrics"]["deep_adverse_rate"]["mean_delta"])
                for row in points
            ],
        }
        intended_grid = {
            "pre_peak_mae_highest_minus_lowest": [
                float(row["highest_minus_lowest"]["pre_peak_mae_mean"]["mean_delta"])
                for row in grids
            ],
            "deep_adverse_rate_highest_minus_lowest": [
                float(row["highest_minus_lowest"]["deep_adverse_rate"]["mean_delta"])
                for row in grids
            ],
        }
        curve_pass = bool(
            all(value > 0.0 for value in intended_curve["pre_peak_mae_delta"])
            and all(value < 0.0 for value in intended_curve["deep_adverse_rate_delta"])
        )
        grid_pass = bool(
            all(
                value > 0.0
                for value in intended_grid["pre_peak_mae_highest_minus_lowest"]
            )
            and all(
                value < 0.0
                for value in intended_grid["deep_adverse_rate_highest_minus_lowest"]
            )
        )
    opportunity_pass = bool(
        min(mfe_retention) >= float(rules["minimum_mfe_mean_retention_ratio"])
        and min(tail_delta) >= -float(rules["maximum_tail_hit_rate_loss"])
    )
    if curve_pass and grid_pass and opportunity_pass:
        status = "conditional_filter_supported"
    elif curve_pass and grid_pass:
        status = "informative_but_opportunity_tradeoff"
    else:
        status = "not_supported_as_pre_entry_filter"
    return {
        "auxiliary": auxiliary,
        "horizon": int(horizon),
        "host_variant": host_variant,
        "status": status,
        "reference_mfe_top_fraction": top_fraction,
        "reference_retention_fraction": retention,
        "mfe_mean_retention_ratio_2023_2024_2025": mfe_retention,
        "mfe_tail_hit_delta_2023_2024_2025": tail_delta,
        "intended_curve_evidence": intended_curve,
        "intended_grid_evidence": intended_grid,
        "gate_checks": {
            "intended_curve_improves_each_year": curve_pass,
            "highest_vs_lowest_quintile_improves_each_year": grid_pass,
            "opportunity_preservation": opportunity_pass,
        },
    }


def _build_decision(
    *,
    study: Mapping[str, Any],
    feature_roles: Mapping[str, Any],
    conditional: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    d10_variant = (
        "turnover_cost_proxy"
        if feature_roles["turnover_cost_proxy"]["replace_baseline"]
        else "baseline"
    )
    d20_variant = (
        "breakout_retest_levels"
        if feature_roles["breakout_retest_levels"]["replace_baseline"]
        else "baseline"
    )
    state_projection_decisions = {
        "probability_of_high_state": _classify_conditional_coordinate(
            study=study,
            records=conditional,
            auxiliary="state_high_probability_10",
            horizon=10,
            host_variant=d10_variant,
        ),
        "expected_state": _classify_conditional_coordinate(
            study=study,
            records=conditional,
            auxiliary="state_expected_value_10",
            horizon=10,
            host_variant=d10_variant,
        ),
        "low_state_avoidance": _classify_conditional_coordinate(
            study=study,
            records=conditional,
            auxiliary="state_low_avoidance_10",
            horizon=10,
            host_variant=d10_variant,
        ),
    }
    status_order = {
        "not_supported_as_pre_entry_filter": 0,
        "informative_but_opportunity_tradeoff": 1,
        "conditional_filter_supported": 2,
    }
    preferred_state_projection, preferred_state_evidence = max(
        state_projection_decisions.items(),
        key=lambda item: status_order[item[1]["status"]],
    )
    coordinate_decisions = {
        "state_10": {
            "status": preferred_state_evidence["status"],
            "preferred_conditional_projection": preferred_state_projection,
            "projection_evidence": state_projection_decisions,
        },
        "pre_peak_mae_10": _classify_conditional_coordinate(
            study=study,
            records=conditional,
            auxiliary="pre_peak_mae_10",
            horizon=10,
            host_variant=d10_variant,
        ),
        "pre_peak_mae_20": _classify_conditional_coordinate(
            study=study,
            records=conditional,
            auxiliary="pre_peak_mae_20",
            horizon=20,
            host_variant=d20_variant,
        ),
    }
    output_coordinates: list[dict[str, Any]] = [
        {"name": "mfe_10", "model_variant": d10_variant, "type": "scalar"},
        {"name": "mfe_20", "model_variant": d20_variant, "type": "scalar"},
    ]
    if coordinate_decisions["state_10"]["status"] != (
        "not_supported_as_pre_entry_filter"
    ):
        output_coordinates.append(
            {
                "name": "state_10",
                "model_variant": "base_features",
                "type": "three_probability_vector",
                "conditional_audit_score": preferred_state_projection,
            }
        )
    for name in ("pre_peak_mae_10", "pre_peak_mae_20"):
        if coordinate_decisions[name]["status"] != "not_supported_as_pre_entry_filter":
            output_coordinates.append(
                {"name": name, "model_variant": "base_features", "type": "scalar"}
            )
    return {
        "status": "completed_without_retraining_or_fusion",
        "primary_mfe_heads": {
            "10": {"variant": d10_variant},
            "20": {"variant": d20_variant},
        },
        "feature_variant_roles": dict(feature_roles),
        "conditional_coordinate_decisions": coordinate_decisions,
        "state_20": study["decision"]["state_20_role"],
        "entry_output_coordinates": output_coordinates,
        "entry_output_coordinate_count": len(output_coordinates),
        "next_step": "design_post_entry_state_update_targets_from_entry_information_plus_realized_D1_D3_D5_path_only_after_freezing_this_entry_output_contract",
        "does_not_select": list(study["non_selections"]),
    }


def run_audit(
    *,
    study_path: Path = DEFAULT_STUDY_PATH,
    output_root: Path = DEFAULT_OUTPUT_ROOT,
) -> dict[str, Any]:
    started = datetime.now().astimezone()
    study = load_study(study_path)
    feature_study = source.load_study(
        _resolve(study["sources"]["feature_study_config"])
    )
    path_study = base.load_study(_resolve(study["sources"]["path_study_config"]))
    feature_output_root = _resolve(study["sources"]["feature_output_root"])
    path_output_root = _resolve(study["sources"]["path_output_root"])
    inputs, pack = source._load_validated_inputs(feature_study)
    if inputs.maximum_outcome_date != study["folds"]["maximum_outcome_date"]:
        raise ValueError("label boundary differs from entry-role boundary")
    reader = objective.FuturePathReader(pack)
    evaluation = dict(study["evaluation"])
    top_fractions = tuple(float(value) for value in evaluation["mfe_top_fractions"])
    retention = tuple(float(value) for value in evaluation["retention_fractions"])
    quantile_count = int(evaluation["auxiliary_quantile_count"])

    annual: list[dict[str, Any]] = []
    paired: list[dict[str, Any]] = []
    relationships: list[dict[str, Any]] = []
    conditional: list[dict[str, Any]] = []
    quality: list[dict[str, Any]] = []
    consumed_files: set[str] = {
        str(study_path.resolve()),
        str(_resolve(study["sources"]["feature_study_config"])),
        str(_resolve(study["sources"]["path_study_config"])),
        str(inputs.label_manifest_path),
        str(inputs.feature_manifest_path),
    }

    for horizon in TARGET_HORIZONS:
        early_horizon = int(evaluation["early_horizon_by_target"][str(horizon)])
        for year in FOLD_YEARS:
            mfe_bundles = {
                variant: _load_mfe_bundle(
                    feature_study=feature_study,
                    output_root=feature_output_root,
                    year=year,
                    horizon=horizon,
                    variant=variant,
                )
                for variant in EXPECTED_VARIANTS[horizon]
            }
            baseline_rows = mfe_bundles["baseline"].rows
            actual = _path_actual(
                inputs=inputs,
                reader=reader,
                rows=baseline_rows,
                horizon=horizon,
                early_horizon=early_horizon,
            )
            auxiliary_bundles: dict[str, ScoreBundle] = {}
            risk_name = f"pre_peak_mae_{horizon}"
            risk_bundle = _load_path_bundle(
                path_study=path_study,
                output_root=path_output_root,
                year=year,
                horizon=horizon,
                target="pre_peak_mae",
            )
            auxiliary_bundles[risk_name] = risk_bundle
            if horizon == 10:
                state_bundle = _load_path_bundle(
                    path_study=path_study,
                    output_root=path_output_root,
                    year=year,
                    horizon=horizon,
                    target="state",
                )
                auxiliary_bundles["state_high_probability_10"] = state_bundle

            quality.append(
                _quality_record(
                    inputs=inputs,
                    actual=actual,
                    mfe_bundles=mfe_bundles,
                    auxiliary_bundles=auxiliary_bundles,
                )
            )
            daily_by_variant: dict[str, pd.DataFrame] = {}
            for variant, bundle in mfe_bundles.items():
                objective_bundle = objective.PredictionBundle(
                    variant=variant,
                    year=year,
                    horizon=horizon,
                    rows=bundle.rows,
                    prediction=bundle.prediction,
                    result=bundle.result,
                )
                daily, metrics = objective._daily_evaluation(
                    inputs=inputs,
                    bundle=objective_bundle,
                    peak_day=actual.peak_day,
                    early_horizon=early_horizon,
                )
                daily_path = (
                    output_root
                    / "role_comparison"
                    / f"h{horizon:02d}"
                    / f"fold_{year}"
                    / f"{variant}_daily.parquet"
                )
                daily_by_variant[variant] = daily
                annual.append(
                    {
                        "year": int(year),
                        "horizon": int(horizon),
                        "early_horizon": int(early_horizon),
                        "variant": variant,
                        "metrics": _compact_role_metrics(metrics),
                        "daily_metrics": _write_parquet(daily_path, daily),
                    }
                )
                files = dict(bundle.result["files"])
                for key in ("prediction", "evaluation_rows", "daily_metrics", "model"):
                    if key in files:
                        consumed_files.add(str(_resolve(files[key]["path"])))
            baseline_daily = daily_by_variant["baseline"]
            for challenger in EXPECTED_VARIANTS[horizon][1:]:
                paired.append(
                    {
                        "year": int(year),
                        "horizon": int(horizon),
                        "baseline": "baseline",
                        "challenger": challenger,
                        "metrics": objective._paired_delta(
                            challenger=daily_by_variant[challenger],
                            baseline=baseline_daily,
                            horizon=horizon,
                        ),
                    }
                )
            for left, right in combinations(EXPECTED_VARIANTS[horizon], 2):
                relationship_daily = _daily_model_relationship(
                    actual=actual,
                    left_name=left,
                    left_score=mfe_bundles[left].prediction,
                    right_name=right,
                    right_score=mfe_bundles[right].prediction,
                )
                relationship_path = (
                    output_root
                    / "model_relationship"
                    / f"h{horizon:02d}"
                    / f"fold_{year}_{left}_vs_{right}.parquet"
                )
                relationships.append(
                    {
                        "year": int(year),
                        "horizon": int(horizon),
                        "left": left,
                        "right": right,
                        "summary": _summarize_daily_frame(
                            relationship_daily, horizon=horizon
                        ),
                        "daily_metrics": _write_parquet(
                            relationship_path, relationship_daily
                        ),
                    }
                )

            auxiliary_scores = {
                risk_name: np.asarray(risk_bundle.prediction, dtype=np.float32)
            }
            if horizon == 10:
                state_probability = np.asarray(
                    auxiliary_bundles["state_high_probability_10"].prediction,
                    dtype=np.float32,
                )
                auxiliary_scores["state_high_probability_10"] = state_probability[:, 2]
                auxiliary_scores["state_expected_value_10"] = (
                    state_probability[:, 1] + 2.0 * state_probability[:, 2]
                ).astype(np.float32, copy=False)
                auxiliary_scores["state_low_avoidance_10"] = (
                    1.0 - state_probability[:, 0]
                ).astype(np.float32, copy=False)
            for host_variant, host_bundle in mfe_bundles.items():
                for auxiliary_name, auxiliary_score in auxiliary_scores.items():
                    curve_daily, grid_daily = _conditional_daily_frames(
                        actual=actual,
                        host_score=host_bundle.prediction,
                        auxiliary_score=auxiliary_score,
                        top_fractions=top_fractions,
                        retention_fractions=retention,
                        quantile_count=quantile_count,
                    )
                    artifact_root = (
                        output_root
                        / "conditional"
                        / auxiliary_name
                        / f"h{horizon:02d}"
                        / host_variant
                    )
                    conditional.append(
                        {
                            "year": int(year),
                            "horizon": int(horizon),
                            "host_variant": host_variant,
                            "auxiliary": auxiliary_name,
                            "curve": _aggregate_conditional_curve(
                                curve_daily, horizon=horizon
                            ),
                            "grid": _aggregate_conditional_grid(
                                grid_daily,
                                horizon=horizon,
                                quantile_count=quantile_count,
                            ),
                            "curve_daily": _write_parquet(
                                artifact_root / f"fold_{year}_retention_curve.parquet",
                                curve_daily,
                            ),
                            "grid_daily": _write_parquet(
                                artifact_root / f"fold_{year}_conditional_grid.parquet",
                                grid_daily,
                            ),
                        }
                    )
            for bundle in auxiliary_bundles.values():
                files = dict(bundle.result["files"])
                for key in ("prediction", "daily_metrics", "model"):
                    if key in files:
                        consumed_files.add(str(_resolve(files[key]["path"])))

    feature_roles = _compile_feature_roles(
        study=study, paired=paired, relationships=relationships
    )
    decision = _build_decision(
        study=study, feature_roles=feature_roles, conditional=conditional
    )
    if any(row["forbidden_2026_row_count"] for row in quality):
        raise AssertionError("entry-role audit consumed a forbidden 2026 row")
    if any(row["duplicate_candidate_key_count"] for row in quality):
        raise AssertionError("entry-role evaluation grain is not date-symbol unique")
    summary = {
        "schema": SUMMARY_SCHEMA,
        "status": "completed",
        "completed_at": _now(),
        "study_id": STUDY_ID,
        "scope": {
            "candidate_count": int(inputs.candidate_count),
            "fold_years": list(FOLD_YEARS),
            "horizons": list(TARGET_HORIZONS),
            "fold_role": study["folds"]["fold_role"],
            "maximum_consumed_outcome_date": inputs.maximum_outcome_date,
            "new_booster_count": 0,
            "retained_meta_model_count": 0,
            "reused_mfe_prediction_count": sum(
                len(variants) for variants in EXPECTED_VARIANTS.values()
            )
            * len(FOLD_YEARS),
            "reused_auxiliary_prediction_count": 3 * len(FOLD_YEARS),
            "consumed_evidence_file_count": len(consumed_files),
        },
        "data_quality": quality,
        "role_comparison": {
            "annual": annual,
            "paired_against_baseline": paired,
            "model_relationships": relationships,
            "feature_roles": feature_roles,
        },
        "conditional_path_audit": conditional,
        "decision": decision,
        "runtime": {
            "elapsed_seconds": (datetime.now().astimezone() - started).total_seconds()
        },
        "disclosure": {
            "2023_2025_reuse": "These owner-authorized folds have been reused for research decisions; the result is comparative recent-market evidence, not a pristine-holdout claim.",
            "2026": "No 2026 row, outcome, prediction, metric, or decision input was read.",
            "selection_semantics": "Top fractions and retention fractions diagnose score roles and conditional path tradeoffs; they are not slots, portfolio rules, or account policies.",
            "causality": "Every consumed prediction is OOS and every feature is bounded by signal-date close; future paths are used only as evaluation outcomes.",
            "no_fusion": "No blended score, meta model, successor model, holding rule, or exit rule was fit or selected.",
        },
    }
    _write_json(output_root / "summary.json", summary)
    _write_json(output_root / "decision.json", decision)
    return summary


def self_test() -> dict[str, Any]:
    dates = np.repeat(np.arange(3, dtype=np.int32), 100)
    within = np.tile(np.arange(100, dtype=np.float64), 3)
    rows = np.arange(len(dates), dtype=np.int64)
    target = within / 100.0
    adverse = -0.20 + within / 1000.0
    state = np.tile(np.repeat(np.arange(3, dtype=np.int8), [34, 33, 33]), 3)
    actual = PathActual(
        rows=rows,
        date_idx=dates,
        target_mfe=target.astype(np.float32),
        early_mfe=(0.5 * target).astype(np.float32),
        peak_day=np.full(len(rows), 5.0, dtype=np.float32),
        pre_peak_mae=adverse.astype(np.float32),
        endpoint_return=(0.5 * target).astype(np.float32),
        state=state,
        mfe_tail_event=_within_date_event(
            date_idx=dates, values=target, fraction=0.20, upper=True
        ),
        deep_adverse_event=_within_date_event(
            date_idx=dates, values=adverse, fraction=0.20, upper=False
        ),
    )
    relationship = _daily_model_relationship(
        actual=actual,
        left_name="left",
        left_score=within,
        right_name="right",
        right_score=within + np.sin(within),
    )
    if float(relationship["prediction_spearman"].mean()) < 0.95:
        raise AssertionError("relationship correlation drifted")
    curve, grid = _conditional_daily_frames(
        actual=actual,
        host_score=within,
        auxiliary_score=within,
        top_fractions=(0.05,),
        retention_fractions=(0.5, 1.0),
        quantile_count=5,
    )
    curve_summary = _aggregate_conditional_curve(curve, horizon=10)
    selected = next(row for row in curve_summary if row["retention_fraction"] == 0.5)
    if selected["metrics"]["mfe_mean"]["mean_delta"] <= 0.0:
        raise AssertionError("conditional retention direction drifted")
    grid_summary = _aggregate_conditional_grid(grid, horizon=10, quantile_count=5)
    if grid_summary[0]["highest_minus_lowest"]["mfe_mean"]["mean_delta"] <= 0.0:
        raise AssertionError("conditional grid direction drifted")
    return {"status": "passed", "test_count": 3}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Reassess Seq100 entry score roles and conditional path coordinates."
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_STUDY_PATH)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("command", choices=("self-test", "audit"))
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "self-test":
        result = self_test()
    else:
        summary = run_audit(
            study_path=args.config.resolve(), output_root=args.output_root.resolve()
        )
        result = {
            "status": summary["status"],
            "study_id": summary["study_id"],
            "decision": summary["decision"],
            "runtime": summary["runtime"],
        }
    print(json.dumps(result, ensure_ascii=False, default=_json_default), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
