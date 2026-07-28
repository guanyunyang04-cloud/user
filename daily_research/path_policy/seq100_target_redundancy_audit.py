from __future__ import annotations

import argparse
import json
import math
import os
from dataclasses import dataclass
from datetime import datetime
from itertools import combinations
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
from scipy import stats

from daily_research.path_policy import seq100_path_label_learnability as base
from daily_research.path_policy import seq100_short_horizon_target_reaudit as source


WORKSPACE_ROOT = source.WORKSPACE_ROOT
STUDY_ID = "seq100_target_redundancy_audit_v1"
DEFAULT_STUDY_PATH = (
    WORKSPACE_ROOT
    / "daily_research/research_records/seq100/seq100_target_redundancy_audit_v1/config.json"
)
DEFAULT_OUTPUT_ROOT = (
    WORKSPACE_ROOT
    / "daily_research/output/path_policy/studies/seq100_target_redundancy_audit_v1"
)
MFE_HORIZONS = (5, 10, 20, 40)
STATE_HORIZONS = (3, 5, 10, 20)
FOLD_YEARS = (2023, 2024, 2025)
STATE_TO_NEAREST_MFE = {3: 5, 5: 5, 10: 10, 20: 20}
SUMMARY_SCHEMA = "seq100_target_redundancy_audit_summary/v1"


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


def load_study(path: Path = DEFAULT_STUDY_PATH) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("study_id") != STUDY_ID:
        raise ValueError(f"study_id must be {STUDY_ID}")
    folds = dict(payload.get("folds", {}))
    if tuple(folds["fold_years"]) != FOLD_YEARS:
        raise ValueError("target redundancy fold years changed")
    if tuple(folds["mfe_horizons"]) != MFE_HORIZONS:
        raise ValueError("target redundancy MFE horizons changed")
    if tuple(folds["state_horizons"]) != STATE_HORIZONS:
        raise ValueError("target redundancy state horizons changed")
    if bool(payload.get("model", {}).get("train_new_boosters", True)):
        raise ValueError("target redundancy audit may not train a booster")
    return payload


def _resolve(path: str) -> Path:
    candidate = Path(path)
    return candidate if candidate.is_absolute() else WORKSPACE_ROOT / candidate


def _task_specs() -> list[tuple[int, str, int]]:
    return [
        (year, target, horizon)
        for year in FOLD_YEARS
        for target, horizon in (
            *(("mfe", horizon) for horizon in MFE_HORIZONS),
            ("state", 3),
            *(("state", horizon) for horizon in (5, 10, 20)),
        )
    ]


def _task_root(target: str, horizon: int) -> Path:
    del target
    return source.DEFAULT_OUTPUT_ROOT if horizon in source.SHORT_TARGETS else source.OLD_OUTPUT_ROOT


def _load_all_bundles(
    inputs: source.ShortHorizonInputs,
) -> dict[tuple[int, str, int], source.PredictionBundle]:
    return {
        (year, target, horizon): source._prediction_bundle(
            inputs=inputs,
            output_root=source.DEFAULT_OUTPUT_ROOT,
            year=year,
            horizon=horizon,
            target=target,
        )
        for year, target, horizon in _task_specs()
    }


@dataclass(frozen=True)
class OrthogonalizedScore:
    common_rows: np.ndarray
    date_idx: np.ndarray
    actual: np.ndarray
    residual_score: np.ndarray
    target_score: np.ndarray
    score_explained_r2: float
    active_predictor_horizons: list[int]
    standardized_coefficients: list[float]
    predictor_scales: list[float]


def orthogonalize_score(
    *,
    target_bundle: source.PredictionBundle,
    predictor_bundles: Sequence[source.PredictionBundle],
    ridge_penalty: float,
) -> OrthogonalizedScore:
    bundles = (*predictor_bundles, target_bundle)
    common, indices = source._common_rows(bundles)
    dates = target_bundle.date_idx[indices[-1]]
    predictors = np.column_stack(
        [
            source._date_demean(dates, source._score_from_bundle(bundle)[index])
            for bundle, index in zip(predictor_bundles, indices[:-1])
        ]
    )
    target_score = source._date_demean(
        dates, source._score_from_bundle(target_bundle)[indices[-1]]
    )
    actual = np.asarray(target_bundle.actual[indices[-1]], dtype=np.float64)
    valid = (
        np.isfinite(predictors).all(axis=1)
        & np.isfinite(target_score)
        & np.isfinite(actual)
    )
    common = common[valid]
    dates = dates[valid]
    predictors = predictors[valid]
    target_score = target_score[valid]
    actual = actual[valid]
    weights = base.date_equal_weights(dates).astype(np.float64)
    predictor_scale = np.sqrt(
        np.average(np.square(predictors), axis=0, weights=weights)
    )
    active = predictor_scale > 1.0e-10
    standardized = predictors[:, active] / predictor_scale[active]
    square_root = np.sqrt(weights)
    if standardized.shape[1]:
        design = standardized * square_root[:, None]
        response = target_score * square_root
        coefficients = np.linalg.solve(
            design.T @ design
            + float(ridge_penalty) * np.eye(design.shape[1], dtype=np.float64),
            design.T @ response,
        )
        fitted = standardized @ coefficients
    else:
        coefficients = np.empty(0, dtype=np.float64)
        fitted = np.zeros_like(target_score)
    residual = target_score - fitted
    weighted_mean = float(np.average(target_score, weights=weights))
    total = float(np.sum(weights * np.square(target_score - weighted_mean)))
    unexplained = float(np.sum(weights * np.square(residual)))
    return OrthogonalizedScore(
        common_rows=np.asarray(common, dtype=np.int64),
        date_idx=np.asarray(dates, dtype=np.int32),
        actual=actual,
        residual_score=residual,
        target_score=target_score,
        score_explained_r2=float(1.0 - unexplained / max(total, 1.0e-18)),
        active_predictor_horizons=[
            bundle.horizon
            for bundle, is_active in zip(predictor_bundles, active)
            if bool(is_active)
        ],
        standardized_coefficients=coefficients.tolist(),
        predictor_scales=predictor_scale.tolist(),
    )


def _score_metrics(
    *,
    date_idx: np.ndarray,
    actual: np.ndarray,
    score: np.ndarray,
    date_values: np.ndarray,
    horizon: int,
) -> dict[str, Any]:
    dates = np.asarray(date_idx, dtype=np.int32)
    predicted = np.asarray(score, dtype=np.float64)
    boundaries = np.flatnonzero(np.r_[True, dates[1:] != dates[:-1], True])
    variable_dates = sum(
        float(np.nanstd(predicted[start:stop])) > 1.0e-12
        for start, stop in zip(boundaries[:-1], boundaries[1:])
    )
    if variable_dates == 0:
        return {
            "rank_ic": 0.0,
            "top_5pct_mean_lift": 0.0,
            "rank_ic_hac_p_value": 1.0,
            "daily_tail_top5_lift": 0.0,
            "daily_tail_top5_lift_hac_p_value": 1.0,
        }
    _daily, ranking = base.daily_score_metrics(
        date_idx=date_idx,
        actual=actual,
        score=score,
        date_values=date_values,
        horizon=horizon,
    )
    _tail_daily, tail = source._daily_event_enrichment(
        date_idx=date_idx,
        actual=actual,
        score=score,
        absolute_threshold=math.inf,
        horizon=horizon,
    )
    return {
        "rank_ic": float(ranking["rank_ic_mean"]),
        "top_5pct_mean_lift": float(ranking["top_5pct_lift"]),
        "rank_ic_hac_p_value": float(ranking["rank_ic_hac"]["p_value_two_sided"]),
        "daily_tail_top5_lift": float(tail["daily_tail_top5_lift"]),
        "daily_tail_top5_lift_hac_p_value": float(
            tail["daily_tail_top5_lift_hac"]["p_value_two_sided"]
        ),
    }


def _residual_record(
    *,
    year: int,
    target_horizon: int,
    predictor_horizons: Sequence[int],
    orthogonalized: OrthogonalizedScore,
    date_values: np.ndarray,
    actual_override: np.ndarray | None = None,
) -> dict[str, Any]:
    actual = orthogonalized.actual if actual_override is None else np.asarray(actual_override)
    metrics = _score_metrics(
        date_idx=orthogonalized.date_idx,
        actual=actual,
        score=orthogonalized.residual_score,
        date_values=date_values,
        horizon=target_horizon,
    )
    return {
        "year": year,
        "target_horizon": target_horizon,
        "predictor_horizons": list(predictor_horizons),
        "common_row_count": int(len(orthogonalized.common_rows)),
        "score_explained_r2": orthogonalized.score_explained_r2,
        "active_predictor_horizons": orthogonalized.active_predictor_horizons,
        "standardized_coefficients": orthogonalized.standardized_coefficients,
        "predictor_scales": orthogonalized.predictor_scales,
        **metrics,
    }


def _classify_residual(
    rows: Sequence[Mapping[str, Any]],
    thresholds: Mapping[str, Any],
) -> dict[str, Any]:
    rank_ic = [float(row["rank_ic"]) for row in rows]
    tail = [float(row["daily_tail_top5_lift"]) for row in rows]
    strong_cutoff = float(thresholds["minimum_strong_residual_rank_ic"])
    negative_tolerance = float(thresholds["secondary_worst_rank_ic"])
    core = bool(
        all(value > 0.0 for value in rank_ic)
        and all(value > 0.0 for value in tail)
        and sum(value >= strong_cutoff for value in rank_ic)
        >= int(thresholds["core_minimum_strong_years"])
    )
    secondary = bool(
        sum(value > 0.0 for value in rank_ic)
        >= int(thresholds["secondary_minimum_positive_years"])
        and sum(value > 0.0 for value in tail)
        >= int(thresholds["secondary_minimum_positive_tail_years"])
        and min(rank_ic) >= negative_tolerance
    )
    status = "core" if core else "secondary" if secondary else "omit"
    return {
        "status": status,
        "rank_ic_2023_2024_2025": rank_ic,
        "daily_tail_top5_lift_2023_2024_2025": tail,
        "strong_year_count": sum(value >= strong_cutoff for value in rank_ic),
        "positive_rank_year_count": sum(value > 0.0 for value in rank_ic),
        "positive_tail_year_count": sum(value > 0.0 for value in tail),
        "worst_rank_ic": min(rank_ic),
    }


def _stronger_status(left: str, right: str) -> str:
    order = {"omit": 0, "secondary": 1, "core": 2}
    return left if order[left] >= order[right] else right


def _mfe_redundancy_audit(
    *,
    bundles: Mapping[tuple[int, str, int], source.PredictionBundle],
    inputs: source.ShortHorizonInputs,
    config: Mapping[str, Any],
) -> dict[str, Any]:
    ridge = float(config["evaluation"]["ridge_penalty"])
    pair_rows: list[dict[str, Any]] = []
    leave_one_out: list[dict[str, Any]] = []
    forward: list[dict[str, Any]] = []
    incremental: list[dict[str, Any]] = []
    for year in FOLD_YEARS:
        current = {h: bundles[(year, "mfe", h)] for h in MFE_HORIZONS}
        for left, right in combinations(MFE_HORIZONS, 2):
            pair_rows.append(source._pair_prediction_redundancy(current[left], current[right]))
        for horizon in MFE_HORIZONS:
            predictors = tuple(current[item] for item in MFE_HORIZONS if item != horizon)
            orthogonalized = orthogonalize_score(
                target_bundle=current[horizon],
                predictor_bundles=predictors,
                ridge_penalty=ridge,
            )
            leave_one_out.append(
                _residual_record(
                    year=year,
                    target_horizon=horizon,
                    predictor_horizons=[item.horizon for item in predictors],
                    orthogonalized=orthogonalized,
                    date_values=inputs.date_values,
                )
            )
        for position, horizon in enumerate(MFE_HORIZONS[1:], start=1):
            predecessor_horizons = MFE_HORIZONS[:position]
            predictors = tuple(current[item] for item in predecessor_horizons)
            orthogonalized = orthogonalize_score(
                target_bundle=current[horizon],
                predictor_bundles=predictors,
                ridge_penalty=ridge,
            )
            forward.append(
                _residual_record(
                    year=year,
                    target_horizon=horizon,
                    predictor_horizons=predecessor_horizons,
                    orthogonalized=orthogonalized,
                    date_values=inputs.date_values,
                )
            )
            immediate = current[predecessor_horizons[-1]]
            immediate_index = np.searchsorted(immediate.rows, orthogonalized.common_rows)
            if not np.array_equal(
                immediate.rows[immediate_index], orthogonalized.common_rows
            ):
                raise AssertionError("incremental MFE row alignment drifted")
            actual_increment = (
                orthogonalized.actual
                - np.asarray(immediate.actual[immediate_index], dtype=np.float64)
            )
            record = _residual_record(
                year=year,
                target_horizon=horizon,
                predictor_horizons=predecessor_horizons,
                orthogonalized=orthogonalized,
                date_values=inputs.date_values,
                actual_override=actual_increment,
            )
            record["increment_from_horizon"] = predecessor_horizons[-1]
            record["actual_increment_negative_fraction"] = float(
                np.mean(actual_increment < -1.0e-7)
            )
            incremental.append(record)

    thresholds = dict(config["decision"]["residual_head_thresholds"])
    heads: list[dict[str, Any]] = []
    for horizon in MFE_HORIZONS:
        loo_rows = [row for row in leave_one_out if row["target_horizon"] == horizon]
        loo_gate = _classify_residual(loo_rows, thresholds)
        if horizon == MFE_HORIZONS[0]:
            increment_gate = {"status": "not_applicable"}
            final_status = loo_gate["status"]
        else:
            increment_rows = [
                row for row in incremental if row["target_horizon"] == horizon
            ]
            increment_gate = _classify_residual(increment_rows, thresholds)
            final_status = _stronger_status(
                loo_gate["status"], increment_gate["status"]
            )
        heads.append(
            {
                "horizon": horizon,
                "leave_one_out": loo_gate,
                "incremental_window": increment_gate,
                "final_status": final_status,
            }
        )
    return {
        "prediction_pair_metrics": pair_rows,
        "leave_one_out_residual": leave_one_out,
        "forward_residual": forward,
        "incremental_window_opportunity": incremental,
        "head_decisions": heads,
    }


def _daily_percentile_rank(date_idx: np.ndarray, values: np.ndarray) -> np.ndarray:
    dates = np.asarray(date_idx, dtype=np.int32)
    scores = np.asarray(values, dtype=np.float64)
    result = np.full(len(scores), np.nan, dtype=np.float64)
    boundaries = np.flatnonzero(np.r_[True, dates[1:] != dates[:-1], True])
    for start, stop in zip(boundaries[:-1], boundaries[1:]):
        current = scores[start:stop]
        valid = np.isfinite(current)
        if np.any(valid):
            ranks = stats.rankdata(current[valid], method="average")
            result[start:stop][valid] = (ranks - 0.5) / len(ranks)
    return result


@dataclass(frozen=True)
class StateMetaData:
    year: int
    date_idx: np.ndarray
    actual: np.ndarray
    mfe_features: np.ndarray
    state_probability: np.ndarray


def _state_meta_data(
    *,
    year: int,
    state_bundle: source.PredictionBundle,
    mfe_bundles: Sequence[source.PredictionBundle],
) -> StateMetaData:
    common, indices = source._common_rows((*mfe_bundles, state_bundle))
    del common
    dates = state_bundle.date_idx[indices[-1]]
    actual = np.asarray(state_bundle.actual[indices[-1]], dtype=np.int8)
    probability = np.asarray(state_bundle.prediction[indices[-1]], dtype=np.float64)
    mfe_features = np.column_stack(
        [
            _daily_percentile_rank(dates, source._score_from_bundle(bundle)[index])
            for bundle, index in zip(mfe_bundles, indices[:-1])
        ]
    )
    valid = (
        np.isin(actual, [0, 1, 2])
        & np.isfinite(mfe_features).all(axis=1)
        & np.isfinite(probability).all(axis=1)
    )
    return StateMetaData(
        year=year,
        date_idx=np.asarray(dates[valid], dtype=np.int32),
        actual=actual[valid],
        mfe_features=mfe_features[valid],
        state_probability=probability[valid],
    )


def _multiclass_metrics(
    *, actual: np.ndarray, probability: np.ndarray, weight: np.ndarray
) -> dict[str, float]:
    truth = np.asarray(actual, dtype=np.int64)
    predicted = np.clip(np.asarray(probability, dtype=np.float64), 1.0e-12, 1.0)
    predicted /= predicted.sum(axis=1, keepdims=True)
    normalized_weight = np.asarray(weight, dtype=np.float64)
    normalized_weight /= normalized_weight.sum()
    one_hot = np.eye(3, dtype=np.float64)[truth]
    return {
        "multiclass_brier": float(
            np.sum(normalized_weight * np.sum(np.square(predicted - one_hot), axis=1))
        ),
        "multiclass_logloss": float(
            -np.sum(normalized_weight * np.log(predicted[np.arange(len(truth)), truth]))
        ),
    }


def _fit_meta_classifier(
    *, features: np.ndarray, actual: np.ndarray, weights: np.ndarray, c_value: float
) -> Any:
    from sklearn.linear_model import LogisticRegression

    adjusted_weights = np.asarray(weights, dtype=np.float64)
    adjusted_weights *= len(adjusted_weights) / adjusted_weights.sum()
    model = LogisticRegression(
        C=float(c_value),
        solver="lbfgs",
        max_iter=200,
        tol=1.0e-6,
        random_state=7,
    )
    model.fit(features, actual, sample_weight=adjusted_weights)
    if not np.array_equal(model.classes_, np.asarray([0, 1, 2])):
        raise ValueError("state meta classifier did not observe all three classes")
    return model


def _sequential_state_probability_increment(
    *,
    horizon: int,
    yearly: Sequence[StateMetaData],
    config: Mapping[str, Any],
) -> list[dict[str, Any]]:
    c_value = float(config["evaluation"]["diagnostic_logistic_c"])
    rows: list[dict[str, Any]] = []
    history: list[StateMetaData] = []
    for current in yearly:
        if history:
            history_mfe = np.concatenate([item.mfe_features for item in history])
            history_probability = np.concatenate(
                [item.state_probability for item in history]
            )
            history_actual = np.concatenate([item.actual for item in history])
            history_dates = np.concatenate([item.date_idx for item in history])
            history_weights = base.date_equal_weights(history_dates)
            baseline = _fit_meta_classifier(
                features=history_mfe,
                actual=history_actual,
                weights=history_weights,
                c_value=c_value,
            )
            augmented = _fit_meta_classifier(
                features=np.column_stack((history_mfe, history_probability)),
                actual=history_actual,
                weights=history_weights,
                c_value=c_value,
            )
            evaluation_weights = base.date_equal_weights(current.date_idx)
            baseline_metrics = _multiclass_metrics(
                actual=current.actual,
                probability=baseline.predict_proba(current.mfe_features),
                weight=evaluation_weights,
            )
            augmented_metrics = _multiclass_metrics(
                actual=current.actual,
                probability=augmented.predict_proba(
                    np.column_stack((current.mfe_features, current.state_probability))
                ),
                weight=evaluation_weights,
            )
            rows.append(
                {
                    "state_horizon": horizon,
                    "evaluation_year": current.year,
                    "calibration_source_years": [item.year for item in history],
                    "evaluation_row_count": int(len(current.actual)),
                    "baseline_mfe_only": baseline_metrics,
                    "augmented_with_state_probability": augmented_metrics,
                    "relative_brier_improvement": float(
                        (baseline_metrics["multiclass_brier"] - augmented_metrics["multiclass_brier"])
                        / baseline_metrics["multiclass_brier"]
                    ),
                    "relative_logloss_improvement": float(
                        (baseline_metrics["multiclass_logloss"] - augmented_metrics["multiclass_logloss"])
                        / baseline_metrics["multiclass_logloss"]
                    ),
                }
            )
        history.append(current)
    return rows


def _classify_probability_increment(
    rows: Sequence[Mapping[str, Any]], thresholds: Mapping[str, Any]
) -> dict[str, Any]:
    brier = [float(row["relative_brier_improvement"]) for row in rows]
    logloss = [float(row["relative_logloss_improvement"]) for row in rows]
    joint = [left > 0.0 and right > 0.0 for left, right in zip(brier, logloss)]
    tolerance = float(thresholds["secondary_maximum_relative_deterioration"])
    if rows and all(joint):
        status = "supported"
    elif (
        any(joint)
        and min(brier) >= -tolerance
        and min(logloss) >= -tolerance
    ):
        status = "conditional"
    else:
        status = "unsupported"
    return {
        "status": status,
        "relative_brier_improvement_2024_2025": brier,
        "relative_logloss_improvement_2024_2025": logloss,
        "joint_improvement_year_count": sum(joint),
    }


def _state_vs_mfe_audit(
    *,
    bundles: Mapping[tuple[int, str, int], source.PredictionBundle],
    inputs: source.ShortHorizonInputs,
    config: Mapping[str, Any],
) -> dict[str, Any]:
    ridge = float(config["evaluation"]["ridge_penalty"])
    pair_rows: list[dict[str, Any]] = []
    residual_rows: list[dict[str, Any]] = []
    label_relationship: list[dict[str, Any]] = []
    probability_rows: list[dict[str, Any]] = []
    yearly_meta: dict[int, list[StateMetaData]] = {horizon: [] for horizon in STATE_HORIZONS}
    for year in FOLD_YEARS:
        mfe = {h: bundles[(year, "mfe", h)] for h in MFE_HORIZONS}
        for state_horizon in STATE_HORIZONS:
            state_bundle = bundles[(year, "state", state_horizon)]
            for mfe_horizon in MFE_HORIZONS:
                pair = source._pair_prediction_redundancy(
                    state_bundle, mfe[mfe_horizon]
                )
                pair["left_target"] = "state"
                pair["right_target"] = "mfe"
                pair_rows.append(pair)
            orthogonalized = orthogonalize_score(
                target_bundle=state_bundle,
                predictor_bundles=tuple(mfe[horizon] for horizon in MFE_HORIZONS),
                ridge_penalty=ridge,
            )
            ordinal = _score_metrics(
                date_idx=orthogonalized.date_idx,
                actual=orthogonalized.actual,
                score=orthogonalized.residual_score,
                date_values=inputs.date_values,
                horizon=state_horizon,
            )
            high = _score_metrics(
                date_idx=orthogonalized.date_idx,
                actual=(orthogonalized.actual == 2).astype(np.float64),
                score=orthogonalized.residual_score,
                date_values=inputs.date_values,
                horizon=state_horizon,
            )
            residual_rows.append(
                {
                    "year": year,
                    "state_horizon": state_horizon,
                    "mfe_predictor_horizons": list(MFE_HORIZONS),
                    "common_row_count": int(len(orthogonalized.common_rows)),
                    "state_score_explained_r2": orthogonalized.score_explained_r2,
                    "ordinal_rank_ic": ordinal["rank_ic"],
                    "ordinal_top_5pct_lift": ordinal["top_5pct_mean_lift"],
                    "ordinal_rank_ic_hac_p_value": ordinal["rank_ic_hac_p_value"],
                    "high_state_top_5pct_lift": high["top_5pct_mean_lift"],
                    "high_state_rank_ic": high["rank_ic"],
                    "high_state_rank_ic_hac_p_value": high["rank_ic_hac_p_value"],
                    "daily_state_tail_top5_lift": ordinal["daily_tail_top5_lift"],
                    "standardized_coefficients": orthogonalized.standardized_coefficients,
                }
            )
            nearest = mfe[STATE_TO_NEAREST_MFE[state_horizon]]
            common, indices = source._common_rows((state_bundle, nearest))
            dates = state_bundle.date_idx[indices[0]]
            state_actual = np.asarray(state_bundle.actual[indices[0]], dtype=np.int8)
            mfe_actual = np.asarray(nearest.actual[indices[1]], dtype=np.float64)
            valid = np.isin(state_actual, [0, 1, 2]) & np.isfinite(mfe_actual)
            mfe_tertile = source._daily_tertiles(
                date_idx=dates,
                values=mfe_actual,
                valid=valid,
            )
            relation = source._contingency_metrics(state_actual, mfe_tertile)
            label_relationship.append(
                {
                    "year": year,
                    "state_horizon": state_horizon,
                    "nearest_mfe_horizon": STATE_TO_NEAREST_MFE[state_horizon],
                    **relation,
                }
            )
            yearly_meta[state_horizon].append(
                _state_meta_data(
                    year=year,
                    state_bundle=state_bundle,
                    mfe_bundles=tuple(mfe[horizon] for horizon in MFE_HORIZONS),
                )
            )
    for horizon in STATE_HORIZONS:
        probability_rows.extend(
            _sequential_state_probability_increment(
                horizon=horizon,
                yearly=yearly_meta[horizon],
                config=config,
            )
        )

    residual_thresholds = dict(config["decision"]["residual_head_thresholds"])
    probability_thresholds = dict(
        config["decision"]["probability_increment_thresholds"]
    )
    head_decisions: list[dict[str, Any]] = []
    for horizon in STATE_HORIZONS:
        current = [row for row in residual_rows if row["state_horizon"] == horizon]
        normalized = [
            {
                "rank_ic": row["ordinal_rank_ic"],
                "daily_tail_top5_lift": row["high_state_top_5pct_lift"],
            }
            for row in current
        ]
        ordinal_gate = _classify_residual(normalized, residual_thresholds)
        probability_gate = _classify_probability_increment(
            [row for row in probability_rows if row["state_horizon"] == horizon],
            probability_thresholds,
        )
        head_decisions.append(
            {
                "horizon": horizon,
                "ordinal_residual_status": ordinal_gate,
                "literal_probability_status": probability_gate,
                "feature_audit_status": ordinal_gate["status"],
            }
        )
    return {
        "prediction_pair_metrics": pair_rows,
        "residual_after_all_mfe_scores": residual_rows,
        "actual_state_vs_nearest_mfe_tertile": label_relationship,
        "sequential_probability_increment": probability_rows,
        "head_decisions": head_decisions,
    }


def run_audit(
    *,
    study_path: Path = DEFAULT_STUDY_PATH,
    output_root: Path = DEFAULT_OUTPUT_ROOT,
) -> dict[str, Any]:
    started = datetime.now().astimezone()
    study = load_study(study_path)
    source_config = source.load_study(
        _resolve(str(study["data"]["source_config"]["path"]))
    )
    inputs = source.ShortHorizonInputs(source_config)
    bundles = _load_all_bundles(inputs)
    mfe = _mfe_redundancy_audit(
        bundles=bundles, inputs=inputs, config=study
    )
    state = _state_vs_mfe_audit(
        bundles=bundles, inputs=inputs, config=study
    )
    mfe_heads = {
        str(item["horizon"]): item["final_status"] for item in mfe["head_decisions"]
    }
    state_heads = {
        str(item["horizon"]): item["feature_audit_status"]
        for item in state["head_decisions"]
    }
    literal_probabilities = {
        str(item["horizon"]): item["literal_probability_status"]["status"]
        for item in state["head_decisions"]
    }
    decision = {
        "mfe_heads": mfe_heads,
        "mfe_heads_for_feature_audit": [
            f"mfe_{horizon}"
            for horizon in MFE_HORIZONS
            if mfe_heads[str(horizon)] != "omit"
        ],
        "state_heads": state_heads,
        "state_heads_for_feature_audit": [
            f"state_{horizon}"
            for horizon in STATE_HORIZONS
            if state_heads[str(horizon)] != "omit"
        ],
        "state_literal_probability_status": literal_probabilities,
        "risk_heads_unchanged": [
            "pre_peak_mae_3",
            "pre_peak_mae_5",
            "pre_peak_mae_10",
            "pre_peak_mae_20",
            "pre_peak_mae_40",
        ],
        "next_step": "run_feature_family_increment_audit_using_user_authorized_2023_2025_folds",
        "does_not_select": [
            "feature_family_winner",
            "joint_loss",
            "score_fusion",
            "exit_rule",
            "holding_period",
            "slot_count",
            "leverage",
            "stop_loss",
            "successor_architecture",
        ],
    }
    summary = {
        "schema": SUMMARY_SCHEMA,
        "status": "completed",
        "completed_at": _now(),
        "study_id": STUDY_ID,
        "scope": {
            "fold_years": list(FOLD_YEARS),
            "fold_role": "user-approved reused recent confirmation/decision folds; not a pristine holdout",
            "prediction_task_count": len(bundles),
            "maximum_consumed_outcome_date": str(study["folds"]["maximum_outcome_date"]),
            "new_booster_count": 0,
            "diagnostic_meta_models_retained": 0,
        },
        "mfe_cross_horizon": mfe,
        "state_vs_mfe": state,
        "decision": decision,
        "runtime": {
            "elapsed_seconds": (datetime.now().astimezone() - started).total_seconds()
        },
        "disclosure": {
            "2023_2025_reuse": "The owner explicitly chose the recent 2023-2025 folds despite prior reuse. Results are comparative recent-market evidence, not a pristine-holdout claim.",
            "2026": "No 2026 row, outcome, score, calibration fit, metric, or decision input was read.",
            "meta_models": "Small sequential logistic regressions are diagnostic only, use preceding OOS years, are not retained, and are not successor models.",
            "economics": "Target redundancy does not establish tradable return, an exit rule, or an account policy.",
        },
    }
    _write_json(output_root / "mfe_cross_horizon.json", mfe)
    _write_json(output_root / "state_vs_mfe.json", state)
    _write_json(output_root / "decision.json", decision)
    _write_json(output_root / "summary.json", summary)
    return summary


def self_test() -> dict[str, Any]:
    dates = np.repeat(np.arange(4, dtype=np.int32), 50)
    predictor = np.tile(np.linspace(-1.0, 1.0, 50), 4)
    independent = np.tile(np.cos(np.linspace(-math.pi, math.pi, 50)), 4)
    target_score = 2.0 * predictor + 0.5 * independent
    actual = independent + 0.01 * predictor
    rows = np.arange(len(dates), dtype=np.int64)
    predictor_bundle = source.PredictionBundle(
        year=2023,
        horizon=5,
        target="mfe",
        rows=rows,
        date_idx=dates,
        actual=actual,
        prediction=predictor,
        result={},
    )
    target_bundle = source.PredictionBundle(
        year=2023,
        horizon=10,
        target="mfe",
        rows=rows,
        date_idx=dates,
        actual=actual,
        prediction=target_score,
        result={},
    )
    orthogonalized = orthogonalize_score(
        target_bundle=target_bundle,
        predictor_bundles=(predictor_bundle,),
        ridge_penalty=1.0e-3,
    )
    correlation = stats.spearmanr(orthogonalized.residual_score, actual).statistic
    if float(correlation) < 0.95:
        raise AssertionError("orthogonalization lost the independent score component")
    core = _classify_residual(
        [
            {"rank_ic": value, "daily_tail_top5_lift": 0.02}
            for value in (0.02, 0.03, 0.04)
        ],
        {
            "minimum_strong_residual_rank_ic": 0.01,
            "core_minimum_strong_years": 2,
            "secondary_minimum_positive_years": 2,
            "secondary_minimum_positive_tail_years": 2,
            "secondary_worst_rank_ic": -0.01,
        },
    )
    if core["status"] != "core":
        raise AssertionError("residual core gate drifted")
    probability = _classify_probability_increment(
        [
            {"relative_brier_improvement": 0.01, "relative_logloss_improvement": 0.02},
            {"relative_brier_improvement": 0.02, "relative_logloss_improvement": 0.01},
        ],
        {"secondary_maximum_relative_deterioration": 0.005},
    )
    if probability["status"] != "supported":
        raise AssertionError("probability increment gate drifted")
    return {"status": "passed", "test_count": 3}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Audit Seq100 MFE horizon redundancy and state information beyond MFE."
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
        result = run_audit(
            study_path=args.config.resolve(),
            output_root=args.output_root.resolve(),
        )
    print(json.dumps(result, ensure_ascii=False, default=_json_default), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
