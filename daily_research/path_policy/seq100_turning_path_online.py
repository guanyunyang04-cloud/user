"""Chronological probability tests for quality-pool turning-path episodes.

The target is a causal stopping-time outcome, not a hand-written "good stock"
label.  At each probe onset the study estimates whether a pullback/rebound will
resolve as a major turn before the previous local extreme is recovered.  Every
fit is scale specific and may use only episodes resolved before the evaluated
year's first decision date.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import duckdb
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import sklearn
from scipy import stats
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

from daily_research.path_policy import seq100_hot_path_atlas as atlas

plt.switch_backend("Agg")


WORKSPACE_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_STUDY_PATH = (
    WORKSPACE_ROOT / "daily_research/studies/seq100_turning_path_online_v1.json"
)
DEFAULT_OUTPUT_ROOT = (
    WORKSPACE_ROOT
    / "daily_research/output/path_policy/studies/seq100_turning_path_online_v1"
)
STUDY_ID = "seq100_turning_path_online_v1"
MANIFEST_SCHEMA = "seq100_turning_path_online_analysis/1"
BUILDER_VERSION = 1
EPISODE_TYPES = ("down_rebound", "up_pullback")
FUTURE_RESOLUTION_FILTER = "NOT right_censored AND resolution_date_idx > onset_date_idx"
GEOMETRY_FEATURES = (
    "log_probe_overshoot",
    "log_major_leg_magnitude",
    "log_anchor_age_years",
    "log_days_since_major_confirmation_years",
)
RANK_FEATURES = (
    "turnover_proxy_rank",
    "attention_rank",
    "attention_mean5_rank",
    "amount_shock_rank",
    "volume_shock_rank",
    "range_shock_rank",
    "amount_cross_section_rank",
    "ret1_rank",
    "ret5_rank",
    "position20_rank",
    "ret20_rank",
    "trend_efficiency20_rank",
    "volatility20_rank",
    "pullback5_rank",
    "distance_ma20_rank",
    "close_location_rank",
    "bar_open_close_rank",
    "bar_range_rank",
    "bar_mfe_rank",
    "bar_mae_rank",
    "bar_vwap_close_rank",
)
CONTROL_FEATURES = ("size_decile_centered", "legacy_rank_state_missing")
MODEL_NAMES = ("scale_prior", "geometry", "activity_price")
COMPARISONS = {
    "geometry_vs_prior": ("scale_prior", "geometry"),
    "activity_price_vs_prior": ("scale_prior", "activity_price"),
    "activity_price_vs_geometry": ("geometry", "activity_price"),
}


def load_study(path: str | Path = DEFAULT_STUDY_PATH) -> dict[str, Any]:
    study_path = atlas._resolve_path(path)
    study = atlas._read_json(study_path)
    if study.get("study_id") != STUDY_ID:
        raise ValueError("turning_online_study_id_mismatch")
    source = dict(study.get("source", {}) or {})
    if str(source.get("quality_pool_name")) != "quality_liquidity_pit":
        raise ValueError("turning_online_quality_pool_mismatch")
    if str(source.get("maximum_outcome_date")) != "2025-12-31":
        raise ValueError("turning_online_outcome_cutoff_mismatch")
    if int(source.get("forbidden_year", -1)) != 2026:
        raise ValueError("turning_online_forbidden_year_mismatch")
    period = dict(study.get("period", {}) or {})
    if tuple(period.get("regularization_selection_years", ())) != (2016, 2017, 2018):
        raise ValueError("turning_online_selection_years_mismatch")
    if tuple(period.get("rolling_evaluation_years", ())) != tuple(range(2019, 2026)):
        raise ValueError("turning_online_evaluation_years_mismatch")
    boundaries = dict(study.get("boundaries", {}) or {})
    if bool(boundaries.get("profit_claim_allowed", True)):
        raise ValueError("turning_online_profit_claim_forbidden")
    configured_geometry = tuple(
        dict(study.get("feature_sets", {}) or {}).get("geometry", ())
    )
    configured_ranks = tuple(
        dict(study.get("feature_sets", {}) or {}).get("activity_price_rank_inputs", ())
    )
    if configured_geometry != GEOMETRY_FEATURES or configured_ranks != RANK_FEATURES:
        raise ValueError("turning_online_feature_contract_mismatch")
    if _scale_pairs(study) != (
        (1, 4),
        (1, 8),
        (1, 16),
        (2, 4),
        (2, 8),
        (2, 16),
        (4, 8),
        (4, 16),
    ):
        raise ValueError("turning_online_scale_pairs_mismatch")
    return study


def _scale_pairs(study: Mapping[str, Any]) -> tuple[tuple[int, int], ...]:
    return tuple(
        (int(item["probe_multiplier"]), int(item["major_multiplier"]))
        for item in study.get("scale_pairs", ())
    )


def _source_contract(
    study_path: Path, study: Mapping[str, Any]
) -> tuple[dict[str, Any], Path, str]:
    source = dict(study["source"])
    manifest_path = atlas._resolve_path(str(source["quality_pool_analysis_manifest"]))
    manifest = atlas._read_json(manifest_path)
    expected = str(source["expected_quality_pool_fingerprint"])
    if manifest.get("status") != "completed":
        raise ValueError("turning_online_quality_source_incomplete")
    if str(manifest.get("experiment_fingerprint")) != expected:
        raise ValueError("turning_online_quality_fingerprint_mismatch")
    audit = dict(manifest.get("audit", {}) or {})
    if int(audit.get("forbidden_rows", -1)) != 0:
        raise ValueError("turning_online_quality_source_contains_2026")
    if int(audit.get("duplicate_keys", -1)) != 0:
        raise ValueError("turning_online_quality_source_has_duplicates")
    panel_path = Path(str(dict(manifest["outputs"])["quality_episode_features"]))
    if not panel_path.is_file():
        raise FileNotFoundError(f"turning_online_panel_missing:{panel_path}")
    payload = {
        "builder_version": BUILDER_VERSION,
        "study_sha256": atlas._sha256_file(study_path),
        "quality_analysis_sha256": atlas._sha256_file(manifest_path),
        "quality_experiment_fingerprint": expected,
        "sklearn_version": sklearn.__version__,
    }
    fingerprint = hashlib.sha256(
        json.dumps(payload, ensure_ascii=True, sort_keys=True).encode("utf-8")
    ).hexdigest()
    contract = {
        "quality_analysis_manifest": atlas._file_record(manifest_path),
        "quality_experiment_fingerprint": expected,
        "quality_episode_features": atlas._file_record(panel_path, include_hash=False),
        "quality_source_audit": audit,
        "fingerprint_payload": payload,
    }
    return contract, panel_path, fingerprint


def _target_expression(episode_type: str) -> str:
    if episode_type == "up_pullback":
        return "(outcome = 'terminal_top')::INTEGER"
    if episode_type == "down_rebound":
        return "(outcome = 'major_bottom_confirmed')::INTEGER"
    raise ValueError(f"turning_online_unknown_episode_type:{episode_type}")


def _episode_query(
    panel_path: Path,
    *,
    episode_type: str,
    probe_multiplier: int,
    major_multiplier: int,
    maximum_signal_year: int | None = None,
    maximum_resolution_date_idx_exclusive: int | None = None,
) -> str:
    predicates = [
        f"episode_type = '{episode_type}'",
        FUTURE_RESOLUTION_FILTER,
        f"probe_multiplier = {int(probe_multiplier)}",
        f"major_threshold_multiplier = {int(major_multiplier)}",
        "signal_year BETWEEN 2012 AND 2025",
    ]
    if maximum_signal_year is not None:
        predicates.append(f"signal_year <= {int(maximum_signal_year)}")
    if maximum_resolution_date_idx_exclusive is not None:
        predicates.append(
            f"resolution_date_idx < {int(maximum_resolution_date_idx_exclusive)}"
        )
    where = "\n          AND ".join(predicates)
    rank_columns = ",\n        ".join(RANK_FEATURES)
    return f"""
    SELECT
        symbol,
        episode_order,
        onset_date,
        onset_date_idx,
        resolution_date_idx,
        signal_year,
        {_target_expression(episode_type)} AS target,
        onset_reversal_log_return,
        probe_log_return,
        signed_leg_move_to_anchor,
        major_threshold_log_return,
        anchor_age_market_days,
        days_since_major_confirmation,
        all_rank_features_complete,
        size_bin_10,
        {rank_columns}
    FROM read_parquet({atlas._sql_quote(panel_path)})
    WHERE {where}
    ORDER BY onset_date_idx, symbol, episode_order
    """


def _load_episode_frame(
    connection: duckdb.DuckDBPyConnection,
    panel_path: Path,
    *,
    episode_type: str,
    probe_multiplier: int,
    major_multiplier: int,
    maximum_signal_year: int | None = None,
    maximum_resolution_date_idx_exclusive: int | None = None,
) -> pd.DataFrame:
    frame = connection.execute(
        _episode_query(
            panel_path,
            episode_type=episode_type,
            probe_multiplier=probe_multiplier,
            major_multiplier=major_multiplier,
            maximum_signal_year=maximum_signal_year,
            maximum_resolution_date_idx_exclusive=(
                maximum_resolution_date_idx_exclusive
            ),
        )
    ).fetchdf()
    if frame.empty:
        raise ValueError(
            "turning_online_empty_episode_frame:"
            f"{episode_type}:{probe_multiplier}:{major_multiplier}"
        )
    if frame.duplicated(["symbol", "episode_order"]).any():
        raise ValueError("turning_online_duplicate_episode_rows")
    target_values = set(frame["target"].astype(int).unique().tolist())
    if target_values != {0, 1}:
        raise ValueError(f"turning_online_degenerate_target:{target_values}")
    return frame


def _geometry_matrix(frame: pd.DataFrame) -> np.ndarray:
    reversal = frame["onset_reversal_log_return"].to_numpy(np.float64)
    probe = frame["probe_log_return"].to_numpy(np.float64)
    leg = frame["signed_leg_move_to_anchor"].to_numpy(np.float64)
    major = frame["major_threshold_log_return"].to_numpy(np.float64)
    anchor_age = frame["anchor_age_market_days"].to_numpy(np.float64)
    confirmation_age = frame["days_since_major_confirmation"].to_numpy(np.float64)
    if (
        np.any(probe <= 0)
        or np.any(major <= 0)
        or np.any(anchor_age < 0)
        or np.any(confirmation_age < 0)
    ):
        raise ValueError("turning_online_invalid_geometry_domain")
    one_year = math.log(253.0)
    matrix = np.column_stack(
        [
            np.log1p(np.maximum(np.abs(reversal) / probe - 1.0, 0.0)),
            np.log1p(np.abs(leg) / major),
            np.log1p(anchor_age) / one_year,
            np.log1p(confirmation_age) / one_year,
        ]
    )
    if not np.isfinite(matrix).all():
        raise ValueError("turning_online_nonfinite_geometry")
    return matrix


def _model_matrix(
    frame: pd.DataFrame, feature_set: str
) -> tuple[np.ndarray, tuple[str, ...]]:
    geometry = _geometry_matrix(frame)
    if feature_set == "geometry":
        return geometry, GEOMETRY_FEATURES
    if feature_set != "activity_price":
        raise ValueError(f"turning_online_unknown_feature_set:{feature_set}")
    ranks = frame.loc[:, RANK_FEATURES].to_numpy(np.float64)
    rank_missing = ~np.isfinite(ranks)
    ranks = np.where(rank_missing, 0.5, ranks)
    ranks = 2.0 * (ranks - 0.5)
    size = frame["size_bin_10"].to_numpy(np.float64)
    size_missing = ~np.isfinite(size)
    size = np.where(size_missing, 5.5, size)
    size = (size - 5.5) / 4.5
    declared_complete = frame["all_rank_features_complete"].fillna(False).to_numpy(bool)
    missing_state = (
        rank_missing.any(axis=1) | size_missing | ~declared_complete
    ).astype(np.float64)
    matrix = np.column_stack([geometry, ranks, size, missing_state])
    names = (*GEOMETRY_FEATURES, *RANK_FEATURES, *CONTROL_FEATURES)
    if not np.isfinite(matrix).all():
        raise ValueError("turning_online_nonfinite_model_matrix")
    return matrix, names


def _fold_masks(
    frame: pd.DataFrame, evaluation_year: int
) -> tuple[np.ndarray, np.ndarray, int]:
    test = frame["signal_year"].to_numpy(np.int64) == int(evaluation_year)
    if not np.any(test):
        raise ValueError(f"turning_online_missing_evaluation_year:{evaluation_year}")
    cutoff = int(frame.loc[test, "onset_date_idx"].min())
    train = frame["resolution_date_idx"].to_numpy(np.int64) < cutoff
    if np.any(train & test):
        raise AssertionError("turning_online_test_label_entered_training")
    if not np.any(train):
        raise ValueError("turning_online_empty_training_fold")
    return train, test, cutoff


def _fit_predict(
    matrix: np.ndarray,
    target: np.ndarray,
    train: np.ndarray,
    test: np.ndarray,
    *,
    feature_names: Sequence[str],
    regularization_c: float,
    maximum_iterations: int,
    tolerance: float,
) -> tuple[np.ndarray, dict[str, Any], list[dict[str, Any]]]:
    x_train = np.asarray(matrix[train], dtype=np.float64)
    x_test = np.asarray(matrix[test], dtype=np.float64)
    y_train = np.asarray(target[train], dtype=np.int8)
    mean = x_train.mean(axis=0)
    scale = x_train.std(axis=0)
    scale = np.where(scale > 1.0e-10, scale, 1.0)
    x_train = (x_train - mean) / scale
    x_test = (x_test - mean) / scale
    model = LogisticRegression(
        C=float(regularization_c),
        solver="newton-cholesky",
        max_iter=int(maximum_iterations),
        tol=float(tolerance),
        fit_intercept=True,
    )
    model.fit(x_train, y_train)
    probability = np.clip(model.predict_proba(x_test)[:, 1], 1.0e-6, 1 - 1.0e-6)
    fit = {
        "training_rows": int(train.sum()),
        "training_positives": int(y_train.sum()),
        "training_positive_rate": float(y_train.mean()),
        "regularization_c": float(regularization_c),
        "iterations": int(model.n_iter_[0]),
        "intercept": float(model.intercept_[0]),
    }
    coefficients = [
        {
            "feature": str(name),
            "coefficient": float(value),
            "training_mean": float(center),
            "training_scale": float(spread),
        }
        for name, value, center, spread in zip(
            feature_names, model.coef_[0], mean, scale
        )
    ]
    return probability, fit, coefficients


def _jeffreys_prior(
    target: np.ndarray, train: np.ndarray, test_rows: int
) -> np.ndarray:
    successes = float(np.asarray(target[train], dtype=np.float64).sum())
    total = int(train.sum())
    probability = (successes + 0.5) / (total + 1.0)
    return np.full(int(test_rows), probability, dtype=np.float64)


def _log_loss_values(target: np.ndarray, probability: np.ndarray) -> np.ndarray:
    y = np.asarray(target, dtype=np.float64)
    p = np.clip(np.asarray(probability, dtype=np.float64), 1.0e-12, 1 - 1.0e-12)
    return -(y * np.log(p) + (1.0 - y) * np.log1p(-p))


def _safe_auc(target: np.ndarray, probability: np.ndarray) -> float:
    y = np.asarray(target, dtype=np.int8)
    if len(np.unique(y)) < 2 or float(np.ptp(probability)) <= 1.0e-15:
        return 0.5
    return float(roc_auc_score(y, probability))


def _mean_daily_auc(
    dates: np.ndarray, target: np.ndarray, probability: np.ndarray
) -> tuple[float, int]:
    frame = pd.DataFrame(
        {"date_idx": dates, "target": target, "probability": probability}
    )
    values: list[float] = []
    for _, group in frame.groupby("date_idx", sort=False):
        if group["target"].nunique() < 2:
            continue
        values.append(
            _safe_auc(
                group["target"].to_numpy(np.int8),
                group["probability"].to_numpy(np.float64),
            )
        )
    return (float(np.mean(values)) if values else 0.5, len(values))


def _model_metrics(
    *,
    target: np.ndarray,
    dates: np.ndarray,
    probability: np.ndarray,
) -> dict[str, Any]:
    loss = _log_loss_values(target, probability)
    brier = np.square(np.asarray(probability) - np.asarray(target))
    daily = (
        pd.DataFrame({"date_idx": dates, "log_loss": loss, "brier": brier})
        .groupby("date_idx", sort=False)
        .mean()
    )
    daily_auc, daily_auc_dates = _mean_daily_auc(dates, target, probability)
    return {
        "rows": len(target),
        "dates": len(daily),
        "positive_rate": float(np.mean(target)),
        "mean_probability": float(np.mean(probability)),
        "log_loss": float(np.mean(loss)),
        "date_equal_log_loss": float(daily["log_loss"].mean()),
        "brier": float(np.mean(brier)),
        "date_equal_brier": float(daily["brier"].mean()),
        "event_auc": _safe_auc(target, probability),
        "mean_daily_auc": daily_auc,
        "daily_auc_dates": daily_auc_dates,
    }


def _calibration_rows(
    *,
    target: np.ndarray,
    probability: np.ndarray,
    bins: int,
) -> list[dict[str, Any]]:
    p = np.clip(np.asarray(probability, dtype=np.float64), 0.0, 1.0)
    y = np.asarray(target, dtype=np.float64)
    index = np.minimum((p * int(bins)).astype(np.int64), int(bins) - 1)
    rows: list[dict[str, Any]] = []
    for value in range(int(bins)):
        keep = index == value
        if not np.any(keep):
            continue
        rows.append(
            {
                "calibration_bin": int(value),
                "rows": int(keep.sum()),
                "probability_sum": float(p[keep].sum()),
                "target_sum": float(y[keep].sum()),
                "mean_probability": float(p[keep].mean()),
                "observed_rate": float(y[keep].mean()),
            }
        )
    return rows


def _daily_loss_frame(
    *,
    frame: pd.DataFrame,
    test: np.ndarray,
    target: np.ndarray,
    probabilities: Mapping[str, np.ndarray],
) -> pd.DataFrame:
    tested = frame.loc[test, ["onset_date", "onset_date_idx"]].reset_index(drop=True)
    values = tested.copy()
    for name, probability in probabilities.items():
        values[f"{name}_log_loss"] = _log_loss_values(target, probability)
        values[f"{name}_brier"] = np.square(probability - target)
    return values.groupby(["onset_date", "onset_date_idx"], as_index=False).mean()


def _selection_diagnostics(
    daily_rows: pd.DataFrame, study: Mapping[str, Any]
) -> tuple[pd.DataFrame, dict[str, float]]:
    lag = int(dict(study["evaluation"])["hac_lag"])
    aggregated = (
        daily_rows.groupby(
            ["feature_set", "regularization_c", "evaluation_year", "onset_date_idx"],
            as_index=False,
        )["daily_log_loss"]
        .mean()
        .sort_values(["feature_set", "regularization_c", "onset_date_idx"])
    )
    rows: list[dict[str, Any]] = []
    selected: dict[str, float] = {}
    for feature_set, family in aggregated.groupby("feature_set", sort=True):
        family_rows: list[dict[str, Any]] = []
        for value, group in family.groupby("regularization_c", sort=True):
            estimate = atlas._hac_mean(group["daily_log_loss"], lag=lag)
            family_rows.append(
                {
                    "feature_set": str(feature_set),
                    "regularization_c": float(value),
                    "discovery_dates": int(group["onset_date_idx"].nunique()),
                    "date_equal_log_loss": float(estimate["mean"]),
                    "hac_se": float(estimate["se"]),
                    "hac_lcb_95": float(estimate["lcb_95"]),
                    "hac_ucb_95": float(estimate["ucb_95"]),
                }
            )
        minimum = min(family_rows, key=lambda item: item["date_equal_log_loss"])
        threshold = float(minimum["date_equal_log_loss"] + minimum["hac_se"])
        eligible = [
            item
            for item in family_rows
            if float(item["date_equal_log_loss"]) <= threshold
        ]
        choice = min(eligible, key=lambda item: item["regularization_c"])
        selected[str(feature_set)] = float(choice["regularization_c"])
        for item in family_rows:
            rows.append(
                {
                    **item,
                    "one_se_threshold": threshold,
                    "minimum_loss_c": float(minimum["regularization_c"]),
                    "selected": bool(
                        float(item["regularization_c"])
                        == float(choice["regularization_c"])
                    ),
                }
            )
    return pd.DataFrame(rows), selected


def _discover_regularization(
    connection: duckdb.DuckDBPyConnection,
    panel_path: Path,
    study: Mapping[str, Any],
    *,
    first_oos_date_idx: int,
) -> tuple[pd.DataFrame, dict[str, float]]:
    model = dict(study["models"])
    years = tuple(
        int(value) for value in dict(study["period"])["regularization_selection_years"]
    )
    candidates = tuple(float(value) for value in model["regularization_c_candidates"])
    rows: list[pd.DataFrame] = []
    for episode_type in EPISODE_TYPES:
        for probe, major in _scale_pairs(study):
            frame = _load_episode_frame(
                connection,
                panel_path,
                episode_type=episode_type,
                probe_multiplier=probe,
                major_multiplier=major,
                maximum_signal_year=max(years),
                maximum_resolution_date_idx_exclusive=first_oos_date_idx,
            )
            target_all = frame["target"].to_numpy(np.int8)
            for feature_set in ("geometry", "activity_price"):
                matrix, names = _model_matrix(frame, feature_set)
                for year in years:
                    train, test, _ = _fold_masks(frame, year)
                    target = target_all[test]
                    dates = frame.loc[test, "onset_date_idx"].to_numpy(np.int32)
                    for regularization_c in candidates:
                        probability, _, _ = _fit_predict(
                            matrix,
                            target_all,
                            train,
                            test,
                            feature_names=names,
                            regularization_c=regularization_c,
                            maximum_iterations=int(model["maximum_iterations"]),
                            tolerance=float(model["tolerance"]),
                        )
                        daily = (
                            pd.DataFrame(
                                {
                                    "onset_date_idx": dates,
                                    "daily_log_loss": _log_loss_values(
                                        target, probability
                                    ),
                                }
                            )
                            .groupby("onset_date_idx", as_index=False)
                            .mean()
                        )
                        daily.insert(0, "evaluation_year", int(year))
                        daily.insert(0, "major_multiplier", int(major))
                        daily.insert(0, "probe_multiplier", int(probe))
                        daily.insert(0, "episode_type", episode_type)
                        daily.insert(0, "regularization_c", regularization_c)
                        daily.insert(0, "feature_set", feature_set)
                        rows.append(daily)
    return _selection_diagnostics(pd.concat(rows, ignore_index=True), study)


def _prediction_chunk(
    frame: pd.DataFrame,
    test: np.ndarray,
    *,
    episode_type: str,
    probe: int,
    major: int,
    target: np.ndarray,
    probabilities: Mapping[str, np.ndarray],
) -> pd.DataFrame:
    output = frame.loc[
        test,
        [
            "symbol",
            "episode_order",
            "onset_date",
            "onset_date_idx",
            "resolution_date_idx",
            "signal_year",
        ],
    ].reset_index(drop=True)
    output.insert(2, "episode_type", episode_type)
    output.insert(3, "probe_multiplier", np.int8(probe))
    output.insert(4, "major_threshold_multiplier", np.int8(major))
    output["target"] = np.asarray(target, dtype=np.int8)
    for name, probability in probabilities.items():
        output[f"{name}_probability"] = np.asarray(probability, dtype=np.float32)
    return output


def _evaluate_oos(
    connection: duckdb.DuckDBPyConnection,
    panel_path: Path,
    study: Mapping[str, Any],
    selected_c: Mapping[str, float],
    predictions_path: Path,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    model = dict(study["models"])
    evaluation = dict(study["evaluation"])
    years = tuple(
        int(value) for value in dict(study["period"])["rolling_evaluation_years"]
    )
    metric_rows: list[dict[str, Any]] = []
    daily_rows: list[pd.DataFrame] = []
    calibration_rows: list[dict[str, Any]] = []
    coefficient_rows: list[dict[str, Any]] = []
    temporary = predictions_path.with_suffix(".parquet.tmp")
    temporary.parent.mkdir(parents=True, exist_ok=True)
    temporary.unlink(missing_ok=True)
    writer: pq.ParquetWriter | None = None
    try:
        for episode_type in EPISODE_TYPES:
            for probe, major in _scale_pairs(study):
                frame = _load_episode_frame(
                    connection,
                    panel_path,
                    episode_type=episode_type,
                    probe_multiplier=probe,
                    major_multiplier=major,
                )
                target_all = frame["target"].to_numpy(np.int8)
                matrices = {
                    feature_set: _model_matrix(frame, feature_set)
                    for feature_set in ("geometry", "activity_price")
                }
                for year in years:
                    train, test, cutoff = _fold_masks(frame, year)
                    target = target_all[test]
                    dates = frame.loc[test, "onset_date_idx"].to_numpy(np.int32)
                    probabilities: dict[str, np.ndarray] = {
                        "scale_prior": _jeffreys_prior(
                            target_all, train, int(test.sum())
                        )
                    }
                    fits: dict[str, dict[str, Any]] = {}
                    for feature_set in ("geometry", "activity_price"):
                        matrix, names = matrices[feature_set]
                        probability, fit, coefficients = _fit_predict(
                            matrix,
                            target_all,
                            train,
                            test,
                            feature_names=names,
                            regularization_c=float(selected_c[feature_set]),
                            maximum_iterations=int(model["maximum_iterations"]),
                            tolerance=float(model["tolerance"]),
                        )
                        probabilities[feature_set] = probability
                        fits[feature_set] = fit
                        coefficient_rows.extend(
                            {
                                "evaluation_year": year,
                                "episode_type": episode_type,
                                "probe_multiplier": probe,
                                "major_threshold_multiplier": major,
                                "model": feature_set,
                                "training_cutoff_date_idx_exclusive": cutoff,
                                **record,
                            }
                            for record in coefficients
                        )
                    for name, probability in probabilities.items():
                        metrics = _model_metrics(
                            target=target,
                            dates=dates,
                            probability=probability,
                        )
                        fit = fits.get(name, {})
                        metric_rows.append(
                            {
                                "evaluation_year": year,
                                "maturity_status": (
                                    "provisional_right_truncated"
                                    if year
                                    in dict(study["period"])[
                                        "provisional_right_truncated_years"
                                    ]
                                    else "complete"
                                ),
                                "episode_type": episode_type,
                                "probe_multiplier": probe,
                                "major_threshold_multiplier": major,
                                "model": name,
                                "training_cutoff_date_idx_exclusive": cutoff,
                                "training_rows": int(train.sum()),
                                "training_positives": int(target_all[train].sum()),
                                "training_positive_rate": float(
                                    target_all[train].mean()
                                ),
                                "regularization_c": fit.get("regularization_c", np.nan),
                                "iterations": fit.get("iterations", 0),
                                **metrics,
                            }
                        )
                        calibration_rows.extend(
                            {
                                "evaluation_year": year,
                                "episode_type": episode_type,
                                "probe_multiplier": probe,
                                "major_threshold_multiplier": major,
                                "model": name,
                                **record,
                            }
                            for record in _calibration_rows(
                                target=target,
                                probability=probability,
                                bins=int(evaluation["calibration_bins"]),
                            )
                        )
                    daily = _daily_loss_frame(
                        frame=frame,
                        test=test,
                        target=target,
                        probabilities=probabilities,
                    )
                    daily.insert(0, "evaluation_year", year)
                    daily.insert(0, "major_threshold_multiplier", major)
                    daily.insert(0, "probe_multiplier", probe)
                    daily.insert(0, "episode_type", episode_type)
                    daily_rows.append(daily)
                    chunk = _prediction_chunk(
                        frame,
                        test,
                        episode_type=episode_type,
                        probe=probe,
                        major=major,
                        target=target,
                        probabilities=probabilities,
                    )
                    table = pa.Table.from_pandas(chunk, preserve_index=False)
                    if writer is None:
                        writer = pq.ParquetWriter(
                            temporary, table.schema, compression="zstd"
                        )
                    writer.write_table(table)
        if writer is None:
            raise ValueError("turning_online_no_predictions_written")
    finally:
        if writer is not None:
            writer.close()
    os.replace(temporary, predictions_path)
    return (
        pd.DataFrame(metric_rows),
        pd.concat(daily_rows, ignore_index=True),
        pd.DataFrame(calibration_rows),
        pd.DataFrame(coefficient_rows),
    )


def _bh_q_values(p_values: Sequence[float]) -> np.ndarray:
    values = np.asarray(p_values, dtype=np.float64)
    count = len(values)
    if count == 0:
        return values
    order = np.argsort(values, kind="mergesort")
    ranked = values[order]
    adjusted = ranked * count / np.arange(1, count + 1)
    adjusted = np.minimum.accumulate(adjusted[::-1])[::-1]
    output = np.empty(count, dtype=np.float64)
    output[order] = np.clip(adjusted, 0.0, 1.0)
    return output


def _comparison_daily(daily: pd.DataFrame) -> pd.DataFrame:
    rows: list[pd.DataFrame] = []
    for name, (baseline, candidate) in COMPARISONS.items():
        work = daily[
            [
                "evaluation_year",
                "episode_type",
                "probe_multiplier",
                "major_threshold_multiplier",
                "onset_date",
                "onset_date_idx",
                f"{baseline}_log_loss",
                f"{candidate}_log_loss",
                f"{baseline}_brier",
                f"{candidate}_brier",
            ]
        ].copy()
        work["comparison"] = name
        work["log_loss_gain_nats"] = work.pop(f"{baseline}_log_loss") - work.pop(
            f"{candidate}_log_loss"
        )
        work["brier_gain"] = work.pop(f"{baseline}_brier") - work.pop(
            f"{candidate}_brier"
        )
        rows.append(work)
    return pd.concat(rows, ignore_index=True)


def _hac_information_row(
    group: pd.DataFrame, *, hac_lag: int, cell_dates: int | None = None
) -> dict[str, Any]:
    estimate = atlas._hac_mean(group["log_loss_gain_nats"], lag=hac_lag)
    brier = atlas._hac_mean(group["brier_gain"], lag=hac_lag)
    se = float(estimate["se"])
    z_value = float(estimate["mean"] / se) if se > 0 else np.nan
    p_value = (
        float(2.0 * stats.norm.sf(abs(z_value))) if np.isfinite(z_value) else np.nan
    )
    inverse_log_two = 1.0 / math.log(2.0)
    return {
        "dates": int(group["onset_date_idx"].nunique()),
        "cell_dates": len(group) if cell_dates is None else int(cell_dates),
        "information_gain_nats": float(estimate["mean"]),
        "information_gain_bits": float(estimate["mean"] * inverse_log_two),
        "information_hac_se_nats": se,
        "information_lcb_95_bits": float(estimate["lcb_95"] * inverse_log_two),
        "information_ucb_95_bits": float(estimate["ucb_95"] * inverse_log_two),
        "information_p_value_two_sided": p_value,
        "brier_gain": float(brier["mean"]),
        "brier_gain_hac_se": float(brier["se"]),
    }


def _aggregate_cells_by_date(group: pd.DataFrame) -> pd.DataFrame:
    return (
        group.groupby(
            ["evaluation_year", "onset_date", "onset_date_idx"], as_index=False
        )[["log_loss_gain_nats", "brier_gain"]]
        .mean()
        .sort_values("onset_date_idx")
    )


def _information_summaries(
    daily: pd.DataFrame, study: Mapping[str, Any]
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    comparisons = _comparison_daily(daily)
    period = dict(study["period"])
    lag = int(dict(study["evaluation"])["hac_lag"])
    annual_rows: list[dict[str, Any]] = []
    for (year, comparison), group in comparisons.groupby(
        ["evaluation_year", "comparison"], sort=True
    ):
        collapsed = _aggregate_cells_by_date(group)
        annual_rows.append(
            {
                "evaluation_year": int(year),
                "episode_type": "all",
                "comparison": str(comparison),
                **_hac_information_row(collapsed, hac_lag=lag, cell_dates=len(group)),
            }
        )
        for episode_type, typed in group.groupby("episode_type", sort=True):
            typed_daily = _aggregate_cells_by_date(typed)
            annual_rows.append(
                {
                    "evaluation_year": int(year),
                    "episode_type": str(episode_type),
                    "comparison": str(comparison),
                    **_hac_information_row(
                        typed_daily, hac_lag=lag, cell_dates=len(typed)
                    ),
                }
            )
    scopes = {
        "complete_2019_2024": {
            int(value) for value in period["complete_evaluation_years"]
        },
        "all_2019_2025_provisional": {
            int(value) for value in period["rolling_evaluation_years"]
        },
    }
    headline_rows: list[dict[str, Any]] = []
    for scope, years in scopes.items():
        scoped = comparisons[comparisons["evaluation_year"].isin(years)]
        for comparison, group in scoped.groupby("comparison", sort=True):
            collapsed = _aggregate_cells_by_date(group)
            headline_rows.append(
                {
                    "scope": scope,
                    "episode_type": "all",
                    "comparison": str(comparison),
                    **_hac_information_row(
                        collapsed, hac_lag=lag, cell_dates=len(group)
                    ),
                }
            )
            for episode_type, typed in group.groupby("episode_type", sort=True):
                typed_daily = _aggregate_cells_by_date(typed)
                headline_rows.append(
                    {
                        "scope": scope,
                        "episode_type": str(episode_type),
                        "comparison": str(comparison),
                        **_hac_information_row(
                            typed_daily, hac_lag=lag, cell_dates=len(typed)
                        ),
                    }
                )
    complete = comparisons[
        comparisons["evaluation_year"].isin(scopes["complete_2019_2024"])
    ]
    scale_rows: list[dict[str, Any]] = []
    keys = [
        "comparison",
        "episode_type",
        "probe_multiplier",
        "major_threshold_multiplier",
    ]
    for values, group in complete.groupby(keys, sort=True):
        comparison, episode_type, probe, major = values
        collapsed = _aggregate_cells_by_date(group)
        annual_means = collapsed.groupby("evaluation_year")["log_loss_gain_nats"].mean()
        scale_rows.append(
            {
                "comparison": comparison,
                "episode_type": episode_type,
                "probe_multiplier": int(probe),
                "major_threshold_multiplier": int(major),
                "positive_years": int((annual_means > 0).sum()),
                "years": len(annual_means),
                **_hac_information_row(collapsed, hac_lag=lag, cell_dates=len(group)),
            }
        )
    scale = pd.DataFrame(scale_rows)
    scale["information_bh_q_two_sided"] = np.nan
    for comparison, index in scale.groupby("comparison").groups.items():
        del comparison
        positions = np.asarray(list(index), dtype=np.int64)
        scale.loc[positions, "information_bh_q_two_sided"] = _bh_q_values(
            scale.loc[positions, "information_p_value_two_sided"].to_numpy(float)
        )
    return (
        comparisons,
        pd.DataFrame(annual_rows),
        pd.DataFrame(headline_rows),
        scale,
    )


def _coefficient_stability(coefficients: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for keys, group in coefficients.groupby(
        ["model", "episode_type", "feature"], sort=True
    ):
        model, episode_type, feature = keys
        values = group["coefficient"].to_numpy(np.float64)
        median = float(np.median(values))
        direction = np.sign(median)
        rows.append(
            {
                "model": model,
                "episode_type": episode_type,
                "feature": feature,
                "fits": len(values),
                "median_standardized_coefficient": median,
                "mean_standardized_coefficient": float(values.mean()),
                "median_absolute_coefficient": float(np.median(np.abs(values))),
                "same_median_sign_share": float(
                    np.mean(np.sign(values) == direction)
                    if direction != 0
                    else np.mean(values == 0)
                ),
                "minimum_coefficient": float(values.min()),
                "maximum_coefficient": float(values.max()),
            }
        )
    return pd.DataFrame(rows)


def _aggregate_calibration(
    calibration: pd.DataFrame, study: Mapping[str, Any]
) -> pd.DataFrame:
    complete = {
        int(value) for value in dict(study["period"])["complete_evaluation_years"]
    }
    frame = calibration[calibration["evaluation_year"].isin(complete)].copy()
    grouped = (
        frame.groupby(["episode_type", "model", "calibration_bin"], as_index=False)[
            ["rows", "probability_sum", "target_sum"]
        ]
        .sum()
        .sort_values(["episode_type", "model", "calibration_bin"])
    )
    grouped["mean_probability"] = grouped["probability_sum"] / grouped["rows"]
    grouped["observed_rate"] = grouped["target_sum"] / grouped["rows"]
    return grouped


def _plot_information(annual: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    work = annual[
        annual["episode_type"].eq("all")
        & annual["comparison"].isin(["geometry_vs_prior", "activity_price_vs_geometry"])
    ].copy()
    labels = {
        "geometry_vs_prior": "Path geometry vs scale prior",
        "activity_price_vs_geometry": "Activity + price vs geometry",
    }
    colors = {
        "geometry_vs_prior": "#0f766e",
        "activity_price_vs_geometry": "#c2410c",
    }
    fig, axis = plt.subplots(figsize=(10.5, 5.4))
    for comparison, group in work.groupby("comparison", sort=True):
        group = group.sort_values("evaluation_year")
        axis.errorbar(
            group["evaluation_year"],
            group["information_gain_bits"],
            yerr=[
                group["information_gain_bits"] - group["information_lcb_95_bits"],
                group["information_ucb_95_bits"] - group["information_gain_bits"],
            ],
            marker="o",
            linewidth=2,
            capsize=3,
            label=labels[str(comparison)],
            color=colors[str(comparison)],
        )
    axis.axhline(0.0, color="#111827", linewidth=1)
    axis.axvspan(2024.7, 2025.3, color="#d1d5db", alpha=0.35)
    axis.text(
        2025,
        axis.get_ylim()[0],
        "right-truncated",
        ha="center",
        va="bottom",
        fontsize=8,
    )
    axis.set_title(
        "Chronological information gain across all turning scales", loc="left"
    )
    axis.set_xlabel("Probe-onset year")
    axis.set_ylabel("Out-of-sample log-loss reduction (bits / episode)")
    axis.legend(frameon=False)
    axis.grid(axis="y", alpha=0.2)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _plot_calibration(calibration: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(11.2, 5.2), sharex=True, sharey=True)
    titles = {
        "down_rebound": "Rebound becomes a major bottom",
        "up_pullback": "Pullback becomes a terminal top",
    }
    colors = {"geometry": "#0f766e", "activity_price": "#c2410c"}
    for axis, episode_type in zip(axes, EPISODE_TYPES):
        axis.plot([0, 1], [0, 1], color="#9ca3af", linestyle="--", linewidth=1)
        for model in ("geometry", "activity_price"):
            group = calibration[
                calibration["episode_type"].eq(episode_type)
                & calibration["model"].eq(model)
            ]
            axis.plot(
                group["mean_probability"],
                group["observed_rate"],
                marker="o",
                linewidth=2,
                color=colors[model],
                label=model.replace("_", " + ").title(),
            )
        axis.set_title(titles[episode_type], loc="left", fontsize=11)
        axis.set_xlabel("Predicted probability")
        axis.grid(alpha=0.2)
    axes[0].set_ylabel("Observed frequency")
    axes[1].legend(frameon=False)
    fig.suptitle(
        "Calibration on complete 2019-2024 outcomes", x=0.07, ha="left", fontsize=13
    )
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _plot_scale_metrics(
    metrics: pd.DataFrame, path: Path, study: Mapping[str, Any]
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    complete = {
        int(value) for value in dict(study["period"])["complete_evaluation_years"]
    }
    work = metrics[metrics["evaluation_year"].isin(complete)].copy()
    weighted_rows: list[dict[str, Any]] = []
    for keys, group in work.groupby(
        ["episode_type", "probe_multiplier", "major_threshold_multiplier", "model"],
        sort=True,
    ):
        episode_type, probe, major, model = keys
        weights = group["rows"].to_numpy(float)
        weighted_rows.append(
            {
                "episode_type": episode_type,
                "probe_multiplier": int(probe),
                "major_threshold_multiplier": int(major),
                "model": model,
                "actual": float(np.average(group["positive_rate"], weights=weights)),
                "predicted": float(
                    np.average(group["mean_probability"], weights=weights)
                ),
            }
        )
    values = pd.DataFrame(weighted_rows)
    pairs = _scale_pairs(study)
    labels = [f"{probe} -> {major}x" for probe, major in pairs]
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.3), sharey=True)
    titles = {
        "down_rebound": "Major-bottom probability",
        "up_pullback": "Terminal-top probability",
    }
    for axis, episode_type in zip(axes, EPISODE_TYPES):
        group = values[
            values["episode_type"].eq(episode_type)
            & values["model"].eq("activity_price")
        ].set_index(["probe_multiplier", "major_threshold_multiplier"])
        actual = [group.loc[pair, "actual"] for pair in pairs]
        predicted = [group.loc[pair, "predicted"] for pair in pairs]
        positions = np.arange(len(pairs))
        axis.plot(positions, actual, marker="o", color="#111827", label="Observed")
        axis.plot(positions, predicted, marker="s", color="#c2410c", label="Predicted")
        axis.set_xticks(positions, labels, rotation=35, ha="right")
        axis.set_title(titles[episode_type], loc="left")
        axis.set_xlabel("Probe -> major cost multiple")
        axis.grid(axis="y", alpha=0.2)
    axes[0].set_ylabel("Probability")
    axes[1].legend(frameon=False)
    fig.suptitle(
        "Multi-scale probability levels, complete 2019-2024",
        x=0.06,
        ha="left",
        fontsize=13,
    )
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _runtime_record(
    connection: duckdb.DuckDBPyConnection, study: Mapping[str, Any]
) -> dict[str, Any]:
    settings = connection.execute(
        "SELECT current_setting('memory_limit'), current_setting('threads')"
    ).fetchone()
    return {
        **atlas._duckdb_runtime_resources(study),
        "active_memory_limit": str(settings[0]),
        "active_threads": int(settings[1]),
    }


def run_study(
    *,
    study_path: str | Path = DEFAULT_STUDY_PATH,
    output_root: str | Path = DEFAULT_OUTPUT_ROOT,
) -> dict[str, Any]:
    study_path = atlas._resolve_path(study_path)
    output_root = atlas._resolve_path(output_root)
    study = load_study(study_path)
    contract, panel_path, fingerprint = _source_contract(study_path, study)
    output_root.mkdir(parents=True, exist_ok=True)
    progress_path = output_root / "progress.json"
    atlas._write_json(
        progress_path,
        {"status": "selecting_regularization", "fingerprint": fingerprint},
    )
    connection = atlas._connect(output_root, study)
    try:
        runtime = _runtime_record(connection, study)
        audit_frame = (
            connection.execute(
                f"""
            SELECT
                count(*) FILTER (WHERE {FUTURE_RESOLUTION_FILTER})
                    AS future_resolved_rows,
                count(*) FILTER (
                    WHERE NOT right_censored
                      AND resolution_date_idx = onset_date_idx
                ) AS same_day_resolved_rows,
                count(*) FILTER (WHERE right_censored) AS right_censored_rows,
                count(*) FILTER (
                    WHERE right_censored AND signal_year = 2025
                ) AS right_censored_2025_rows,
                count(*) FILTER (WHERE signal_year = 2025) AS all_2025_rows,
                count(*) FILTER (WHERE signal_year = 2026) AS forbidden_rows,
                min(onset_date_idx) FILTER (WHERE signal_year = 2019)
                    AS first_oos_date_idx
            FROM read_parquet({atlas._sql_quote(panel_path)})
            """
            )
            .fetchdf()
            .iloc[0]
        )
        audit = {key: int(value) for key, value in audit_frame.items()}
        if audit["forbidden_rows"] != 0:
            raise ValueError("turning_online_forbidden_rows")
        selection, selected_c = _discover_regularization(
            connection,
            panel_path,
            study,
            first_oos_date_idx=int(audit["first_oos_date_idx"]),
        )
        atlas._write_json(
            progress_path,
            {
                "status": "running_rolling_oos",
                "fingerprint": fingerprint,
                "selected_regularization_c": selected_c,
            },
        )
        predictions_path = output_root / "oos_predictions.parquet"
        metrics, daily, calibration, coefficients = _evaluate_oos(
            connection,
            panel_path,
            study,
            selected_c,
            predictions_path,
        )
        prediction_audit = (
            connection.execute(
                f"""
                SELECT
                    count(*) AS prediction_rows,
                    count(*) - count(DISTINCT
                        symbol || '|' || episode_type || '|'
                        || probe_multiplier::VARCHAR || '|'
                        || major_threshold_multiplier::VARCHAR || '|'
                        || episode_order::VARCHAR
                    ) AS prediction_duplicate_keys,
                    count(*) FILTER (WHERE signal_year = 2026)
                        AS prediction_forbidden_rows,
                    count(*) FILTER (
                        WHERE NOT isfinite(scale_prior_probability)
                           OR NOT isfinite(geometry_probability)
                           OR NOT isfinite(activity_price_probability)
                    ) AS nonfinite_probability_rows,
                    least(
                        min(scale_prior_probability),
                        min(geometry_probability),
                        min(activity_price_probability)
                    ) AS minimum_probability,
                    greatest(
                        max(scale_prior_probability),
                        max(geometry_probability),
                        max(activity_price_probability)
                    ) AS maximum_probability
                FROM read_parquet({atlas._sql_quote(predictions_path)})
                """
            )
            .fetchdf()
            .iloc[0]
            .to_dict()
        )
        audit.update(
            {
                key: (
                    float(value)
                    if key in {"minimum_probability", "maximum_probability"}
                    else int(value)
                )
                for key, value in prediction_audit.items()
            }
        )
        audit["selection_maximum_signal_year"] = 2018
        audit["selection_resolution_date_idx_exclusive"] = int(
            audit["first_oos_date_idx"]
        )
        if (
            int(audit["prediction_duplicate_keys"]) != 0
            or int(audit["prediction_forbidden_rows"]) != 0
            or int(audit["nonfinite_probability_rows"]) != 0
            or not 0.0 < float(audit["minimum_probability"])
            or not float(audit["maximum_probability"]) < 1.0
        ):
            raise ValueError("turning_online_prediction_audit_failed")
    finally:
        connection.close()

    comparisons, annual, headline, scale = _information_summaries(daily, study)
    coefficient_stability = _coefficient_stability(coefficients)
    calibration_aggregate = _aggregate_calibration(calibration, study)
    audit["oos_prediction_rows"] = int(
        metrics[metrics["model"].eq("scale_prior")]["rows"].sum()
    )
    if int(audit["prediction_rows"]) != int(audit["oos_prediction_rows"]):
        raise ValueError("turning_online_prediction_row_count_mismatch")
    audit["right_censored_2025_share"] = audit["right_censored_2025_rows"] / max(
        audit["all_2025_rows"], 1
    )

    outputs = {
        "regularization_selection": output_root / "regularization_selection.csv",
        "fold_metrics": output_root / "fold_metrics.csv",
        "daily_losses": output_root / "daily_losses.csv",
        "daily_information": output_root / "daily_information.csv",
        "annual_information": output_root / "annual_information.csv",
        "headline_information": output_root / "headline_information.csv",
        "scale_information": output_root / "scale_information.csv",
        "calibration": output_root / "calibration.csv",
        "calibration_complete": output_root / "calibration_complete_2019_2024.csv",
        "coefficients": output_root / "coefficients.csv",
        "coefficient_stability": output_root / "coefficient_stability.csv",
        "predictions": output_root / "oos_predictions.parquet",
        "information_figure": output_root / "figures/information_by_year.png",
        "calibration_figure": output_root / "figures/calibration.png",
        "scale_figure": output_root / "figures/multi_scale_probabilities.png",
    }
    selection.to_csv(outputs["regularization_selection"], index=False)
    metrics.to_csv(outputs["fold_metrics"], index=False)
    daily.to_csv(outputs["daily_losses"], index=False)
    comparisons.to_csv(outputs["daily_information"], index=False)
    annual.to_csv(outputs["annual_information"], index=False)
    headline.to_csv(outputs["headline_information"], index=False)
    scale.to_csv(outputs["scale_information"], index=False)
    calibration.to_csv(outputs["calibration"], index=False)
    calibration_aggregate.to_csv(outputs["calibration_complete"], index=False)
    coefficients.to_csv(outputs["coefficients"], index=False)
    coefficient_stability.to_csv(outputs["coefficient_stability"], index=False)
    _plot_information(annual, outputs["information_figure"])
    _plot_calibration(calibration_aggregate, outputs["calibration_figure"])
    _plot_scale_metrics(metrics, outputs["scale_figure"], study)
    manifest = {
        "schema": MANIFEST_SCHEMA,
        "status": "completed",
        "study_id": STUDY_ID,
        "builder_version": BUILDER_VERSION,
        "experiment_fingerprint": fingerprint,
        "study": atlas._file_record(study_path),
        "source_contract": contract,
        "runtime": runtime,
        "selected_regularization_c": selected_c,
        "audit": audit,
        "outputs": {key: str(path.resolve()) for key, path in outputs.items()},
        "training_performed": True,
        "portfolio_selection_performed": False,
        "execution_backtest_performed": False,
        "profit_claim_allowed": False,
        "causal_effect_claim_allowed": False,
    }
    manifest_path = output_root / "analysis_manifest.json"
    atlas._write_json(manifest_path, manifest)
    atlas._write_json(
        progress_path,
        {"status": "completed", "analysis_manifest": str(manifest_path.resolve())},
    )
    return manifest


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run chronological quality-pool turning probability tests."
    )
    parser.add_argument("--study", default=str(DEFAULT_STUDY_PATH))
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT))
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    result = run_study(study_path=args.study, output_root=args.output_root)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
