"""Conditional-information audit for fixed hot-path archetypes.

This module is deliberately downstream of ``seq100_hot_path_atlas``.  The
future path clusters are fixed before any conditional model is fitted.  The
models here are information probes: they estimate whether signal-close daily
and five-minute state variables improve the conditional path law out of time.
They are not portfolio policies or profit claims.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, log_loss, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from daily_research.path_policy import seq100_hot_path_atlas as atlas
from daily_research.path_policy.seq100_hot_money_event_study import _minute_query


WORKSPACE_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_STUDY_PATH = (
    WORKSPACE_ROOT / "daily_research/studies/seq100_hot_path_state_v1.json"
)
DEFAULT_OUTPUT_ROOT = (
    WORKSPACE_ROOT
    / "daily_research/output/path_policy/studies/seq100_hot_path_state_v1"
)
STUDY_ID = "seq100_hot_path_state_v1"
MANIFEST_SCHEMA = "seq100_hot_path_state_manifest/2"
ANALYSIS_SCHEMA = "seq100_hot_path_state_analysis/2"
BUILDER_VERSION = 2
YEARS = tuple(range(2012, 2026))
CLASS_LABELS = (0, 1, 2)


def load_study(path: str | Path = DEFAULT_STUDY_PATH) -> dict[str, Any]:
    study_path = atlas._resolve_path(path)
    study = atlas._read_json(study_path)
    if study.get("study_id") != STUDY_ID:
        raise ValueError("hot_path_state_study_id_mismatch")
    source = dict(study.get("source", {}) or {})
    if int(source.get("forbidden_year", -1)) != 2026:
        raise ValueError("hot_path_state_forbidden_year_contract_missing")
    if str(source.get("maximum_outcome_date")) != "2025-12-31":
        raise ValueError("hot_path_state_outcome_cutoff_mismatch")
    periods = dict(study.get("period", {}) or {})
    if list(periods.get("model_fit_years", [])) != list(range(2012, 2019)):
        raise ValueError("hot_path_state_fit_years_mismatch")
    if list(periods.get("fixed_evaluation_years", [])) != list(range(2019, 2026)):
        raise ValueError("hot_path_state_evaluation_years_mismatch")
    return study


def _source_contract(
    study_path: Path, study: Mapping[str, Any]
) -> tuple[dict[str, Any], list[Path], list[Path], str]:
    source = dict(study.get("source", {}) or {})
    atlas_manifest_path = atlas._resolve_path(str(source["atlas_manifest"]))
    atlas_analysis_path = atlas._resolve_path(str(source["atlas_analysis_manifest"]))
    assignments_path = atlas._resolve_path(str(source["path_cluster_assignments"]))
    active_path = atlas._resolve_path(str(source["qdp_active_manifest"]))
    for path in (
        atlas_manifest_path,
        atlas_analysis_path,
        assignments_path,
        active_path,
    ):
        if not path.is_file():
            raise FileNotFoundError(f"hot_path_state_source_missing:{path}")
    atlas_manifest = atlas._read_json(atlas_manifest_path)
    atlas_analysis = atlas._read_json(atlas_analysis_path)
    expected = str(source.get("expected_atlas_fingerprint", ""))
    if str(atlas_manifest.get("experiment_fingerprint")) != expected:
        raise ValueError("hot_path_state_atlas_fingerprint_mismatch")
    if str(atlas_analysis.get("experiment_fingerprint")) != expected:
        raise ValueError("hot_path_state_analysis_fingerprint_mismatch")
    panel_paths = atlas._panel_paths(atlas_manifest)
    active = atlas._read_json(active_path)
    qdp_root = active_path.parents[1]
    minute_domain = str(source.get("minute_domain", "market_intraday_5m"))
    minute_paths, minute_manifest_path, minute_manifest = (
        atlas._dataset_paths_and_manifest(qdp_root, active, minute_domain)
    )
    payload = {
        "builder_version": BUILDER_VERSION,
        "study_sha256": atlas._sha256_file(study_path),
        "atlas_manifest_sha256": atlas._sha256_file(atlas_manifest_path),
        "atlas_analysis_sha256": atlas._sha256_file(atlas_analysis_path),
        "assignments_sha256": atlas._sha256_file(assignments_path),
        "active_sha256": atlas._sha256_file(active_path),
        "minute_manifest_sha256": atlas._sha256_file(minute_manifest_path),
    }
    fingerprint = hashlib.sha256(
        json.dumps(payload, sort_keys=True, ensure_ascii=True).encode("utf-8")
    ).hexdigest()
    contract = {
        "atlas_manifest": atlas._file_record(atlas_manifest_path),
        "atlas_analysis_manifest": atlas._file_record(atlas_analysis_path),
        "path_cluster_assignments": atlas._file_record(assignments_path),
        "atlas_panels": [
            atlas._file_record(path, include_hash=False) for path in panel_paths
        ],
        "active_manifest": atlas._file_record(active_path),
        "minute_dataset": {
            "dataset_id": str(dict(active.get("datasets", {}) or {})[minute_domain]),
            "manifest": atlas._file_record(minute_manifest_path),
            "row_count": int(minute_manifest.get("row_count", 0) or 0),
            "shard_count": int(len(minute_paths)),
        },
        "fingerprint_payload": payload,
    }
    return contract, panel_paths, minute_paths, fingerprint


def _manifest_valid(manifest: Mapping[str, Any], fingerprint: str) -> bool:
    if (
        manifest.get("schema") != MANIFEST_SCHEMA
        or manifest.get("status") != "prepared"
        or str(manifest.get("experiment_fingerprint")) != str(fingerprint)
    ):
        return False
    records = list(manifest.get("minute_partitions", []) or [])
    daily_bar = dict(manifest.get("daily_bar_state", {}) or {})
    return (
        len(records) == len(YEARS)
        and Path(str(daily_bar.get("path", ""))).is_file()
        and all(Path(str(record.get("path", ""))).is_file() for record in records)
    )


def _daily_bar_query(panel_paths: Sequence[Path], assignments_path: Path) -> str:
    scan = atlas._parquet_scan(panel_paths)
    assignments = atlas._sql_quote(assignments_path)
    return f"""
    WITH keys AS (
        SELECT DISTINCT symbol, trade_date
        FROM read_parquet({assignments})
    )
    SELECT
        p.symbol,
        p.trade_date,
        p.adj_close / NULLIF(p.adj_open, 0) - 1.0 AS bar_open_close_return,
        p.range_1d AS bar_range,
        p.close_location_1d AS bar_close_location,
        p.body_location_1d AS bar_body_location,
        p.upper_shadow_1d AS bar_upper_shadow,
        p.lower_shadow_1d AS bar_lower_shadow,
        p.adj_high / NULLIF(p.adj_open, 0) - 1.0 AS bar_mfe_from_open,
        p.adj_low / NULLIF(p.adj_open, 0) - 1.0 AS bar_mae_from_open,
        p.raw_close / NULLIF(p.amount / NULLIF(p.volume, 0), 0) - 1.0
            AS bar_vwap_close_deviation
    FROM {scan} p
    INNER JOIN keys k USING (symbol, trade_date)
    """


def prepare_minute_features(
    *,
    study_path: str | Path = DEFAULT_STUDY_PATH,
    output_root: str | Path = DEFAULT_OUTPUT_ROOT,
    force: bool = False,
) -> dict[str, Any]:
    study_path = atlas._resolve_path(study_path)
    output_root = atlas._resolve_path(output_root)
    study = load_study(study_path)
    contract, panel_paths, minute_paths, fingerprint = _source_contract(
        study_path, study
    )
    manifest_path = output_root / "manifest.json"
    if manifest_path.is_file() and not force:
        current = atlas._read_json(manifest_path)
        if _manifest_valid(current, fingerprint):
            return current
        if str(current.get("experiment_fingerprint", "")) not in {"", fingerprint}:
            raise ValueError("hot_path_state_existing_fingerprint_mismatch")
    output_root.mkdir(parents=True, exist_ok=True)
    progress_path = output_root / "progress.json"
    assignments_path = Path(
        str(contract["path_cluster_assignments"]["path"])
    ).resolve()
    minute_root = output_root / "minute_features"
    minute_root.mkdir(parents=True, exist_ok=True)
    connection = atlas._connect(output_root, study)
    records: list[dict[str, Any]] = []
    try:
        daily_bar_path = output_root / "daily_bar_state.parquet"
        if force or not daily_bar_path.is_file():
            atlas._write_json(
                progress_path,
                {
                    "status": "building_daily_bar_state",
                    "experiment_fingerprint": fingerprint,
                },
            )
            atlas._copy_query(
                connection,
                _daily_bar_query(panel_paths, assignments_path),
                daily_bar_path,
            )
        daily_bar_audit = connection.execute(
            f"""
            WITH expected AS (
                SELECT count(*) AS rows
                FROM read_parquet({atlas._sql_quote(assignments_path)})
            ), actual AS (
                SELECT
                    count(*) AS rows,
                    count(*) - count(DISTINCT symbol || '|' || trade_date)
                        AS duplicate_keys
                FROM read_parquet({atlas._sql_quote(daily_bar_path)})
            )
            SELECT expected.rows AS expected_rows, actual.*
            FROM expected CROSS JOIN actual
            """
        ).fetchdf().iloc[0].to_dict()
        if int(daily_bar_audit["duplicate_keys"]) != 0:
            raise ValueError("hot_path_state_daily_bar_duplicate_keys")
        if int(daily_bar_audit["rows"]) != int(daily_bar_audit["expected_rows"]):
            raise ValueError("hot_path_state_daily_bar_coverage_mismatch")
        for year in YEARS:
            output_path = minute_root / f"minute_state_{year}.parquet"
            if force or not output_path.is_file():
                atlas._write_json(
                    progress_path,
                    {
                        "status": "building_minute_partition",
                        "year": int(year),
                        "completed_years": [int(item["year"]) for item in records],
                        "experiment_fingerprint": fingerprint,
                    },
                )
                query = _minute_query(list(minute_paths), assignments_path, int(year))
                atlas._copy_query(connection, query, output_path)
            audit = connection.execute(
                f"""
                WITH expected AS (
                    SELECT count(*) AS rows
                    FROM read_parquet({atlas._sql_quote(assignments_path)})
                    WHERE left(trade_date, 4) = '{year}'
                ), actual AS (
                    SELECT
                        count(*) AS rows,
                        count(*) FILTER (WHERE minute_bar_count >= 30) AS covered_rows,
                        count(*) - count(DISTINCT symbol || '|' || trade_date) AS duplicate_keys
                    FROM read_parquet({atlas._sql_quote(output_path)})
                )
                SELECT expected.rows AS expected_rows, actual.*
                FROM expected CROSS JOIN actual
                """
            ).fetchdf().iloc[0].to_dict()
            if int(audit["duplicate_keys"]) != 0:
                raise ValueError(f"hot_path_state_minute_duplicate_keys:{year}")
            if int(audit["rows"]) > int(audit["expected_rows"]):
                raise ValueError(f"hot_path_state_minute_exceeds_sample:{year}")
            records.append(
                {
                    "year": int(year),
                    **atlas._file_record(output_path, include_hash=False),
                    "expected_rows": int(audit["expected_rows"]),
                    "rows": int(audit["rows"]),
                    "covered_rows": int(audit["covered_rows"]),
                }
            )
    finally:
        connection.close()
    manifest = {
        "schema": MANIFEST_SCHEMA,
        "status": "prepared",
        "study_id": STUDY_ID,
        "builder_version": BUILDER_VERSION,
        "experiment_fingerprint": fingerprint,
        "study": atlas._file_record(study_path),
        "source_contract": contract,
        "daily_bar_state": {
            **atlas._file_record(daily_bar_path, include_hash=False),
            "rows": int(daily_bar_audit["rows"]),
            "duplicate_keys": int(daily_bar_audit["duplicate_keys"]),
        },
        "minute_partitions": records,
        "expected_sample_rows": int(sum(item["expected_rows"] for item in records)),
        "minute_rows": int(sum(item["rows"] for item in records)),
        "minute_covered_rows": int(sum(item["covered_rows"] for item in records)),
        "training_performed": False,
        "portfolio_selection_performed": False,
    }
    atlas._write_json(manifest_path, manifest)
    atlas._write_json(
        progress_path,
        {
            "status": "prepared",
            "manifest": str(manifest_path.resolve()),
            "minute_rows": manifest["minute_rows"],
        },
    )
    return manifest


def _build_model(
    model_name: str,
    feature_columns: Sequence[str],
    study: Mapping[str, Any],
) -> Pipeline:
    laws = dict(study.get("conditional_laws", {}) or {})
    if model_name == "additive_logit":
        config = dict(laws["additive_logit"])
        estimator = LogisticRegression(
            C=float(config.get("C", 0.1)),
            max_iter=int(config.get("maximum_iterations", 500)),
            solver="lbfgs",
        )
        transformer = Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                ("scale", StandardScaler()),
            ]
        )
    elif model_name == "bounded_histogram_boosting":
        config = dict(laws["bounded_histogram_boosting"])
        estimator = HistGradientBoostingClassifier(
            max_iter=int(config.get("maximum_iterations", 160)),
            learning_rate=float(config.get("learning_rate", 0.05)),
            max_leaf_nodes=int(config.get("maximum_leaf_nodes", 15)),
            min_samples_leaf=int(config.get("minimum_samples_leaf", 300)),
            l2_regularization=float(config.get("l2_regularization", 10.0)),
            random_state=int(config.get("random_seed", 17)),
        )
        transformer = Pipeline([("imputer", SimpleImputer(strategy="median"))])
    else:
        raise ValueError(f"unknown_hot_path_state_model:{model_name}")
    preprocess = ColumnTransformer(
        [("features", transformer, list(feature_columns))], remainder="drop"
    )
    return Pipeline([("preprocess", preprocess), ("model", estimator)])


def _multiclass_brier(y: np.ndarray, probability: np.ndarray) -> float:
    target = np.eye(probability.shape[1], dtype=np.float64)[y.astype(int)]
    return float(np.mean(np.sum(np.square(probability - target), axis=1)))


def _row_log_loss(y: np.ndarray, probability: np.ndarray) -> np.ndarray:
    selected = probability[np.arange(len(y)), y.astype(int)]
    return -np.log(np.clip(selected, 1e-12, 1.0))


def _safe_binary_auc(target: np.ndarray, score: np.ndarray) -> float:
    binary = np.asarray(target, dtype=bool)
    if np.unique(binary).size < 2:
        return float("nan")
    return float(roc_auc_score(binary, score))


def _causal_rolling_prior(
    frame: pd.DataFrame,
    target: np.ndarray,
    evaluation_frame: pd.DataFrame,
    fallback_prior: np.ndarray,
    *,
    lookback_days: int,
    label_delay_days: int,
    smoothing: float,
) -> np.ndarray:
    date_index = frame["date_idx"].to_numpy(dtype=np.int64)
    minimum = int(date_index.min())
    maximum = int(date_index.max())
    counts = np.zeros((maximum - minimum + 1, len(CLASS_LABELS)), dtype=np.float64)
    for label in CLASS_LABELS:
        selected = target == int(label)
        positions = date_index[selected] - minimum
        np.add.at(counts[:, int(label)], positions, 1.0)
    cumulative = np.cumsum(counts, axis=0)
    result = np.empty((len(evaluation_frame), len(CLASS_LABELS)), dtype=np.float64)
    for position, current in enumerate(
        evaluation_frame["date_idx"].to_numpy(dtype=np.int64)
    ):
        high = int(current) - int(label_delay_days)
        low = high - int(lookback_days) + 1
        high_position = min(high, maximum) - minimum
        low_position = max(low, minimum) - minimum
        if high_position < 0 or low_position > high_position:
            result[position] = fallback_prior
            continue
        observed = cumulative[high_position].copy()
        if low_position > 0:
            observed -= cumulative[low_position - 1]
        denominator = float(observed.sum() + smoothing * len(CLASS_LABELS))
        if denominator <= 0:
            result[position] = fallback_prior
        else:
            result[position] = (observed + smoothing) / denominator
    return result


def _date_equal_auc(
    frame: pd.DataFrame,
    target: np.ndarray,
    score: np.ndarray,
) -> dict[str, float]:
    work = pd.DataFrame(
        {
            "trade_date": frame["trade_date"].astype(str).to_numpy(),
            "target": np.asarray(target, dtype=bool),
            "score": np.asarray(score, dtype=np.float64),
        }
    )
    values: list[float] = []
    for _, group in work.groupby("trade_date", sort=True):
        value = _safe_binary_auc(
            group["target"].to_numpy(dtype=bool),
            group["score"].to_numpy(dtype=np.float64),
        )
        if np.isfinite(value):
            values.append(float(value))
    estimate = atlas._hac_mean(np.asarray(values, dtype=np.float64), lag=20)
    return {
        "dates": int(len(values)),
        "mean": float(estimate["mean"]),
        "hac_se": float(estimate["se"]),
        "hac_lcb_95": float(estimate["lcb_95"]),
        "hac_ucb_95": float(estimate["ucb_95"]),
    }


def _evaluate_predictions(
    frame: pd.DataFrame,
    y: np.ndarray,
    probability: np.ndarray,
    prior: np.ndarray,
    causal_prior_probability: np.ndarray,
    *,
    model_name: str,
    feature_set: str,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    prior_probability = np.tile(prior, (len(y), 1))
    model_loss = _row_log_loss(y, probability)
    prior_loss = _row_log_loss(y, prior_probability)
    gain = prior_loss - model_loss
    causal_prior_loss = _row_log_loss(y, causal_prior_probability)
    causal_gain = causal_prior_loss - model_loss
    rows: list[dict[str, Any]] = []
    for year in sorted(frame["signal_year"].unique()):
        selected = frame["signal_year"].to_numpy(dtype=int) == int(year)
        yy = y[selected]
        pp = probability[selected]
        rising_date_auc = _date_equal_auc(
            frame.loc[selected], yy == 2, pp[:, 2]
        )
        bad_date_auc = _date_equal_auc(frame.loc[selected], yy == 0, pp[:, 0])
        rows.append(
            {
                "model": model_name,
                "feature_set": feature_set,
                "evaluation_year": int(year),
                "rows": int(selected.sum()),
                "log_loss": float(log_loss(yy, pp, labels=CLASS_LABELS)),
                "prior_log_loss": float(
                    log_loss(
                        yy,
                        np.tile(prior, (len(yy), 1)),
                        labels=CLASS_LABELS,
                    )
                ),
                "log_loss_gain": float(np.mean(gain[selected])),
                "causal_prior_log_loss": float(
                    np.mean(causal_prior_loss[selected])
                ),
                "log_loss_gain_over_causal_prior": float(
                    np.mean(causal_gain[selected])
                ),
                "multiclass_brier": _multiclass_brier(yy, pp),
                "rising_path_auc": _safe_binary_auc(yy == 2, pp[:, 2]),
                "bad_path_auc": _safe_binary_auc(yy == 0, pp[:, 0]),
                "date_equal_rising_path_auc": rising_date_auc["mean"],
                "date_equal_bad_path_auc": bad_date_auc["mean"],
                "accuracy": float(accuracy_score(yy, np.argmax(pp, axis=1))),
            }
        )
    date_gain = pd.DataFrame(
        {
            "trade_date": frame["trade_date"].astype(str).to_numpy(),
            "signal_year": frame["signal_year"].to_numpy(dtype=int),
            "gain": gain,
        }
    ).groupby(["trade_date", "signal_year"], as_index=False)["gain"].mean()
    estimate = atlas._hac_mean(date_gain["gain"].to_numpy(dtype=float), lag=20)
    causal_date_gain = pd.DataFrame(
        {
            "trade_date": frame["trade_date"].astype(str).to_numpy(),
            "gain": causal_gain,
        }
    ).groupby("trade_date", as_index=False)["gain"].mean()
    causal_estimate = atlas._hac_mean(
        causal_date_gain["gain"].to_numpy(dtype=float), lag=20
    )
    rising_date_auc = _date_equal_auc(frame, y == 2, probability[:, 2])
    bad_date_auc = _date_equal_auc(frame, y == 0, probability[:, 0])
    pooled = pd.DataFrame(
        [
            {
                "model": model_name,
                "feature_set": feature_set,
                "dates": int(date_gain["trade_date"].nunique()),
                "rows": int(len(frame)),
                "log_loss_gain_date_mean": estimate["mean"],
                "log_loss_gain_hac_se": estimate["se"],
                "log_loss_gain_hac_lcb_95": estimate["lcb_95"],
                "log_loss_gain_hac_ucb_95": estimate["ucb_95"],
                "causal_prior_log_loss_gain_date_mean": causal_estimate["mean"],
                "causal_prior_log_loss_gain_hac_se": causal_estimate["se"],
                "causal_prior_log_loss_gain_hac_lcb_95": causal_estimate[
                    "lcb_95"
                ],
                "causal_prior_log_loss_gain_hac_ucb_95": causal_estimate[
                    "ucb_95"
                ],
                "rising_path_auc": _safe_binary_auc(y == 2, probability[:, 2]),
                "bad_path_auc": _safe_binary_auc(y == 0, probability[:, 0]),
                "date_equal_rising_path_auc": rising_date_auc["mean"],
                "date_equal_rising_path_auc_hac_lcb_95": rising_date_auc[
                    "hac_lcb_95"
                ],
                "date_equal_bad_path_auc": bad_date_auc["mean"],
                "date_equal_bad_path_auc_hac_lcb_95": bad_date_auc[
                    "hac_lcb_95"
                ],
                "multiclass_brier": _multiclass_brier(y, probability),
            }
        ]
    )
    prediction_frame = frame[
        [
            "symbol",
            "trade_date",
            "date_idx",
            "signal_year",
            "path_cluster",
            "terminal_log_return_5",
            "terminal_log_return_20",
        ]
    ].copy()
    prediction_frame["model"] = model_name
    prediction_frame["feature_set"] = feature_set
    for label in CLASS_LABELS:
        prediction_frame[f"probability_cluster_{label}"] = probability[:, label]
    prediction_frame["row_log_loss_gain"] = gain
    prediction_frame["row_log_loss_gain_over_causal_prior"] = causal_gain
    prediction_frame["row_model_log_loss"] = model_loss
    return pd.DataFrame(rows), pooled, prediction_frame


def _top_decile_diagnostics(predictions: pd.DataFrame) -> pd.DataFrame:
    work = predictions.copy()
    work["rising_probability_rank"] = work.groupby(
        ["model", "feature_set", "trade_date"]
    )["probability_cluster_2"].rank(method="average", pct=True)
    rows: list[dict[str, Any]] = []
    for keys, group in work.groupby(
        ["model", "feature_set", "signal_year"], sort=True
    ):
        model, feature_set, year = keys
        top = group[group["rising_probability_rank"] > 0.90]
        bottom = group[group["rising_probability_rank"] <= 0.10]
        rows.append(
            {
                "model": str(model),
                "feature_set": str(feature_set),
                "evaluation_year": int(year),
                "top_rows": int(len(top)),
                "top_rising_path_rate": float((top["path_cluster"] == 2).mean()),
                "bottom_rising_path_rate": float(
                    (bottom["path_cluster"] == 2).mean()
                ),
                "top_net_return_5": float(
                    np.expm1(top["terminal_log_return_5"]).mean() - 0.006
                ),
                "top_net_return_20": float(
                    np.expm1(top["terminal_log_return_20"]).mean() - 0.006
                ),
                "bottom_net_return_20": float(
                    np.expm1(bottom["terminal_log_return_20"]).mean() - 0.006
                ),
            }
        )
    return pd.DataFrame(rows)


def _permutation_importance(
    model: Pipeline,
    frame: pd.DataFrame,
    y: np.ndarray,
    features: Sequence[str],
    *,
    seed: int = 17,
    maximum_rows: int = 20_000,
) -> pd.DataFrame:
    rng = np.random.default_rng(int(seed))
    positions = np.arange(len(frame), dtype=np.int64)
    if len(positions) > int(maximum_rows):
        positions = np.sort(rng.choice(positions, int(maximum_rows), replace=False))
    sample = frame.iloc[positions].copy()
    target = y[positions]
    base_probability = model.predict_proba(sample)
    base_loss = float(log_loss(target, base_probability, labels=CLASS_LABELS))
    rows: list[dict[str, Any]] = []
    for feature in features:
        permuted = sample.copy()
        permuted[feature] = rng.permutation(permuted[feature].to_numpy())
        probability = model.predict_proba(permuted)
        rows.append(
            {
                "feature": str(feature),
                "permutation_scope": "global_time_and_cross_section",
                "permutation_log_loss_increase": float(
                    log_loss(target, probability, labels=CLASS_LABELS) - base_loss
                ),
                "evaluation_rows": int(len(sample)),
            }
        )
    return pd.DataFrame(rows).sort_values(
        "permutation_log_loss_increase", ascending=False, kind="mergesort"
    )


def _information_increments(
    predictions: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    comparisons = (
        ("stock_state_over_market", "market_only_all", "daily_all"),
        ("bar_geometry_on_all", "daily_all", "daily_plus_bar_all"),
        ("bar_geometry_on_covered", "daily_covered", "daily_plus_bar_covered"),
        (
            "intraday_path_over_complete_daily",
            "daily_plus_bar_covered",
            "daily_plus_intraday",
        ),
        ("all_minute_source_over_daily", "daily_covered", "daily_plus_intraday"),
    )
    pooled_rows: list[dict[str, Any]] = []
    annual_rows: list[dict[str, Any]] = []
    for model, group in predictions.groupby("model", sort=True):
        for comparison, base_name, enhanced_name in comparisons:
            base = group[group["feature_set"] == base_name][
                ["symbol", "trade_date", "signal_year", "row_model_log_loss"]
            ].rename(columns={"row_model_log_loss": "base_loss"})
            enhanced = group[group["feature_set"] == enhanced_name][
                ["symbol", "trade_date", "row_model_log_loss"]
            ].rename(columns={"row_model_log_loss": "enhanced_loss"})
            joined = base.merge(
                enhanced,
                on=["symbol", "trade_date"],
                how="inner",
                validate="one_to_one",
            )
            joined["increment"] = joined["base_loss"] - joined["enhanced_loss"]
            date_mean = (
                joined.groupby(["trade_date", "signal_year"], sort=True)["increment"]
                .mean()
                .reset_index()
            )
            estimate = atlas._hac_mean(
                date_mean["increment"].to_numpy(dtype=float), lag=20
            )
            pooled_rows.append(
                {
                    "model": str(model),
                    "comparison": comparison,
                    "base_feature_set": base_name,
                    "enhanced_feature_set": enhanced_name,
                    "rows": int(len(joined)),
                    "dates": int(len(date_mean)),
                    "log_loss_increment": estimate["mean"],
                    "increment_hac_se": estimate["se"],
                    "increment_hac_lcb_95": estimate["lcb_95"],
                    "increment_hac_ucb_95": estimate["ucb_95"],
                }
            )
            for year, year_group in date_mean.groupby("signal_year", sort=True):
                annual_rows.append(
                    {
                        "model": str(model),
                        "comparison": comparison,
                        "evaluation_year": int(year),
                        "dates": int(len(year_group)),
                        "log_loss_increment": float(year_group["increment"].mean()),
                    }
                )
    return pd.DataFrame(pooled_rows), pd.DataFrame(annual_rows)


def _geometry_reconciliation(frame: pd.DataFrame) -> pd.DataFrame:
    pairs = (
        ("open_close_return", "bar_open_close_return", "minute_open_close_return"),
        ("range", "bar_range", "minute_range"),
        ("close_location", "bar_close_location", "minute_close_location"),
        ("mfe_from_open", "bar_mfe_from_open", "minute_mfe_from_open"),
        ("mae_from_open", "bar_mae_from_open", "minute_mae_from_open"),
        (
            "vwap_close_deviation",
            "bar_vwap_close_deviation",
            "minute_vwap_close_deviation",
        ),
    )
    rows: list[dict[str, Any]] = []
    for name, daily_column, minute_column in pairs:
        daily = pd.to_numeric(frame[daily_column], errors="coerce").to_numpy(float)
        minute = pd.to_numeric(frame[minute_column], errors="coerce").to_numpy(float)
        valid = np.isfinite(daily) & np.isfinite(minute)
        difference = np.abs(daily[valid] - minute[valid])
        rows.append(
            {
                "coordinate": name,
                "comparable_rows": int(valid.sum()),
                "mean_absolute_difference": float(np.mean(difference)),
                "p99_absolute_difference": float(np.quantile(difference, 0.99)),
                "maximum_absolute_difference": float(np.max(difference)),
                "share_within_1e_6": float(np.mean(difference <= 1e-6)),
            }
        )
    return pd.DataFrame(rows)


def _linear_coefficients(
    model: Pipeline, feature_columns: Sequence[str]
) -> pd.DataFrame:
    estimator = model.named_steps["model"]
    if not isinstance(estimator, LogisticRegression):
        return pd.DataFrame()
    rows: list[dict[str, Any]] = []
    for class_position, label in enumerate(estimator.classes_):
        for feature, coefficient in zip(
            feature_columns, estimator.coef_[class_position], strict=True
        ):
            rows.append(
                {
                    "path_cluster": int(label),
                    "feature": str(feature),
                    "standardized_coefficient": float(coefficient),
                }
            )
    return pd.DataFrame(rows)


def analyze_conditional_information(
    *,
    study_path: str | Path = DEFAULT_STUDY_PATH,
    output_root: str | Path = DEFAULT_OUTPUT_ROOT,
    force: bool = False,
) -> dict[str, Any]:
    study_path = atlas._resolve_path(study_path)
    output_root = atlas._resolve_path(output_root)
    study = load_study(study_path)
    manifest_path = output_root / "manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError("hot_path_state_manifest_missing")
    manifest = atlas._read_json(manifest_path)
    if manifest.get("schema") != MANIFEST_SCHEMA:
        raise ValueError("hot_path_state_manifest_schema_mismatch")
    analysis_root = output_root / "analysis"
    analysis_manifest_path = analysis_root / "manifest.json"
    if analysis_manifest_path.is_file() and not force:
        current = atlas._read_json(analysis_manifest_path)
        if (
            current.get("schema") == ANALYSIS_SCHEMA
            and current.get("status") == "completed"
            and current.get("experiment_fingerprint")
            == manifest.get("experiment_fingerprint")
        ):
            return current
    analysis_root.mkdir(parents=True, exist_ok=True)
    assignments_path = Path(
        str(manifest["source_contract"]["path_cluster_assignments"]["path"])
    )
    assignments = pd.read_parquet(assignments_path)
    daily_bar = pd.read_parquet(Path(str(manifest["daily_bar_state"]["path"])))
    if bool(daily_bar.duplicated(["symbol", "trade_date"]).any()):
        raise ValueError("hot_path_state_combined_daily_bar_duplicate_keys")
    minute = pd.concat(
        [pd.read_parquet(Path(str(record["path"]))) for record in manifest["minute_partitions"]],
        ignore_index=True,
    )
    if bool(minute.duplicated(["symbol", "trade_date"]).any()):
        raise ValueError("hot_path_state_combined_minute_duplicate_keys")
    frame = assignments.merge(
        daily_bar, on=["symbol", "trade_date"], how="inner", validate="one_to_one"
    )
    if len(frame) != len(assignments):
        raise ValueError("hot_path_state_daily_bar_join_coverage_mismatch")
    frame = frame.merge(
        minute, on=["symbol", "trade_date"], how="left", validate="one_to_one"
    )
    frame["minute_covered"] = frame["minute_bar_count"].fillna(0) >= 30
    periods = dict(study.get("period", {}) or {})
    fit_years = set(int(value) for value in periods["model_fit_years"])
    evaluation_years = set(int(value) for value in periods["fixed_evaluation_years"])
    features = dict(study.get("feature_sets", {}) or {})
    market_features = [str(value) for value in features["market_state"]]
    daily_features = [str(value) for value in features["daily_state"]]
    daily_bar_features = [str(value) for value in features["daily_bar_state"]]
    intraday_features = [str(value) for value in features["intraday_state"]]
    missing_market = [column for column in market_features if column not in frame]
    missing_daily = [column for column in daily_features if column not in frame]
    missing_bar = [column for column in daily_bar_features if column not in frame]
    missing_intraday = [column for column in intraday_features if column not in frame]
    if missing_market or missing_daily or missing_bar or missing_intraday:
        raise ValueError(
            "hot_path_state_feature_columns_missing:"
            f"market={missing_market}:daily={missing_daily}:"
            f"bar={missing_bar}:intraday={missing_intraday}"
        )

    all_rows = np.ones(len(frame), dtype=bool)
    covered_rows = frame["minute_covered"].to_numpy(dtype=bool)
    feature_specs = (
        ("market_only_all", market_features, all_rows),
        ("daily_all", daily_features, all_rows),
        ("daily_covered", daily_features, covered_rows),
        ("daily_plus_bar_all", daily_features + daily_bar_features, all_rows),
        (
            "daily_plus_bar_covered",
            daily_features + daily_bar_features,
            covered_rows,
        ),
        (
            "daily_plus_intraday",
            daily_features + daily_bar_features + intraday_features,
            covered_rows,
        ),
    )
    model_specs = [
        (model_name, feature_set, columns, mask)
        for model_name in ("additive_logit", "bounded_histogram_boosting")
        for feature_set, columns, mask in feature_specs
    ]
    causal_prior_config = dict(
        dict(study.get("evaluation", {}) or {}).get("causal_prior", {}) or {}
    )
    annual_frames: list[pd.DataFrame] = []
    pooled_frames: list[pd.DataFrame] = []
    prediction_frames: list[pd.DataFrame] = []
    importance_frames: list[pd.DataFrame] = []
    coefficient_frames: list[pd.DataFrame] = []
    for model_name, feature_set, feature_columns, coverage_mask in model_specs:
        selected = frame.loc[np.asarray(coverage_mask, dtype=bool)].copy()
        fit_mask = selected["signal_year"].isin(fit_years).to_numpy()
        evaluation_mask = selected["signal_year"].isin(evaluation_years).to_numpy()
        if not fit_mask.any() or not evaluation_mask.any():
            raise ValueError(f"hot_path_state_empty_period:{model_name}:{feature_set}")
        target = selected["path_cluster"].to_numpy(dtype=int)
        prior = np.bincount(target[fit_mask], minlength=len(CLASS_LABELS)).astype(float)
        prior /= prior.sum()
        model = _build_model(model_name, feature_columns, study)
        model.fit(selected.loc[fit_mask], target[fit_mask])
        evaluation_frame = selected.loc[evaluation_mask].reset_index(drop=True)
        evaluation_target = target[evaluation_mask]
        probability = model.predict_proba(evaluation_frame)
        causal_prior_probability = _causal_rolling_prior(
            selected,
            target,
            evaluation_frame,
            prior,
            lookback_days=int(causal_prior_config.get("lookback_days", 252)),
            label_delay_days=int(
                causal_prior_config.get("label_delay_days", 20)
            ),
            smoothing=float(causal_prior_config.get("smoothing", 0.5)),
        )
        annual, pooled, prediction = _evaluate_predictions(
            evaluation_frame,
            evaluation_target,
            probability,
            prior,
            causal_prior_probability,
            model_name=model_name,
            feature_set=feature_set,
        )
        annual_frames.append(annual)
        pooled_frames.append(pooled)
        prediction_frames.append(prediction)
        importance = _permutation_importance(
            model,
            evaluation_frame,
            evaluation_target,
            feature_columns,
            seed=17,
        )
        importance.insert(0, "feature_set", feature_set)
        importance.insert(0, "model", model_name)
        importance_frames.append(importance)
        coefficients = _linear_coefficients(model, feature_columns)
        if not coefficients.empty:
            coefficients.insert(0, "feature_set", feature_set)
            coefficients.insert(0, "model", model_name)
            coefficient_frames.append(coefficients)

    annual_metrics = pd.concat(annual_frames, ignore_index=True)
    pooled_metrics = pd.concat(pooled_frames, ignore_index=True)
    predictions = pd.concat(prediction_frames, ignore_index=True)
    feature_importance = pd.concat(importance_frames, ignore_index=True)
    linear_coefficients = pd.concat(coefficient_frames, ignore_index=True)
    top_decile = _top_decile_diagnostics(predictions)
    information_increments, annual_information_increments = _information_increments(
        predictions
    )
    geometry_reconciliation = _geometry_reconciliation(frame)
    predictions.to_parquet(
        analysis_root / "fixed_period_predictions.parquet",
        index=False,
        compression="zstd",
    )
    atlas._write_csv(analysis_root / "annual_metrics.csv", annual_metrics)
    atlas._write_csv(analysis_root / "pooled_metrics.csv", pooled_metrics)
    atlas._write_csv(analysis_root / "top_decile_diagnostics.csv", top_decile)
    atlas._write_csv(
        analysis_root / "information_increments.csv", information_increments
    )
    atlas._write_csv(
        analysis_root / "annual_information_increments.csv",
        annual_information_increments,
    )
    atlas._write_csv(
        analysis_root / "geometry_reconciliation.csv", geometry_reconciliation
    )
    atlas._write_csv(analysis_root / "permutation_importance.csv", feature_importance)
    atlas._write_csv(analysis_root / "linear_coefficients.csv", linear_coefficients)
    coverage_by_year = (
        frame.groupby("signal_year", sort=True)
        .agg(
            sample_rows=("symbol", "size"),
            minute_rows=("minute_bar_count", "count"),
            minute_covered_rows=("minute_covered", "sum"),
        )
        .reset_index()
    )
    atlas._write_csv(analysis_root / "minute_coverage_by_year.csv", coverage_by_year)
    coverage_by_year_cluster = (
        frame.groupby(["signal_year", "path_cluster"], sort=True)
        .agg(
            sample_rows=("symbol", "size"),
            minute_covered_rows=("minute_covered", "sum"),
        )
        .reset_index()
    )
    coverage_by_year_cluster["minute_coverage_rate"] = (
        coverage_by_year_cluster["minute_covered_rows"]
        / coverage_by_year_cluster["sample_rows"]
    )
    atlas._write_csv(
        analysis_root / "minute_coverage_by_year_cluster.csv",
        coverage_by_year_cluster,
    )
    report_lines = [
        "# Hot Path State Conditional-Information Audit",
        "",
        "The three future path archetypes were frozen upstream. Models here only test whether signal-close state changes their conditional probabilities out of time.",
        "",
        "## Pooled fixed-period metrics",
        "",
        pooled_metrics.to_markdown(index=False, floatfmt=".5f"),
        "",
        "## Information-source increments",
        "",
        information_increments.to_markdown(index=False, floatfmt=".5f"),
        "",
        "## Daily/minute geometry reconciliation",
        "",
        geometry_reconciliation.to_markdown(index=False, floatfmt=".8f"),
        "",
        "## Year-by-year metrics",
        "",
        annual_metrics.to_markdown(index=False, floatfmt=".5f"),
        "",
        "## Boundary",
        "",
        "The causal prior uses only path labels whose 20-day outcome window has already elapsed. Daily OHLC/amount-volume geometry is credited to the daily baseline; the intraday increment therefore contains only order-, concentration- and realized-path-dependent five-minute coordinates. A positive log-loss gain establishes conditional information, not a profitable account. Date-equal AUC measures stock selection within a day, while pooled AUC can also contain market-regime information. Top-decile returns are sampled-path diagnostics and cannot select a production rule.",
    ]
    report_path = analysis_root / "research_record.md"
    report_path.write_text("\n".join(report_lines) + "\n", encoding="utf-8")
    result = {
        "schema": ANALYSIS_SCHEMA,
        "status": "completed",
        "study_id": STUDY_ID,
        "experiment_fingerprint": manifest.get("experiment_fingerprint"),
        "sample_rows": int(len(frame)),
        "minute_covered_rows": int(frame["minute_covered"].sum()),
        "outputs": {
            "pooled_metrics": str((analysis_root / "pooled_metrics.csv").resolve()),
            "annual_metrics": str((analysis_root / "annual_metrics.csv").resolve()),
            "information_increments": str(
                (analysis_root / "information_increments.csv").resolve()
            ),
            "geometry_reconciliation": str(
                (analysis_root / "geometry_reconciliation.csv").resolve()
            ),
            "top_decile_diagnostics": str(
                (analysis_root / "top_decile_diagnostics.csv").resolve()
            ),
            "research_record": str(report_path.resolve()),
        },
        "training_role": "conditional_information_probe_only",
        "portfolio_selection_performed": False,
        "profit_claim_allowed": False,
    }
    atlas._write_json(analysis_manifest_path, result)
    return result


def run_study(
    *,
    study_path: str | Path = DEFAULT_STUDY_PATH,
    output_root: str | Path = DEFAULT_OUTPUT_ROOT,
    force_prepare: bool = False,
    force_analysis: bool = False,
) -> dict[str, Any]:
    prepared = prepare_minute_features(
        study_path=study_path, output_root=output_root, force=force_prepare
    )
    analysis = analyze_conditional_information(
        study_path=study_path, output_root=output_root, force=force_analysis
    )
    return {"prepared": prepared, "analysis": analysis}


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Audit daily and five-minute information about fixed path states."
    )
    parser.add_argument(
        "command", choices=("prepare", "analyze", "run"), nargs="?", default="run"
    )
    parser.add_argument("--study", default=str(DEFAULT_STUDY_PATH))
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT))
    parser.add_argument("--force-prepare", action="store_true")
    parser.add_argument("--force-analysis", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> dict[str, Any]:
    args = build_arg_parser().parse_args(argv)
    if args.command == "prepare":
        result = prepare_minute_features(
            study_path=args.study,
            output_root=args.output_root,
            force=bool(args.force_prepare),
        )
    elif args.command == "analyze":
        result = analyze_conditional_information(
            study_path=args.study,
            output_root=args.output_root,
            force=bool(args.force_analysis),
        )
    else:
        result = run_study(
            study_path=args.study,
            output_root=args.output_root,
            force_prepare=bool(args.force_prepare),
            force_analysis=bool(args.force_analysis),
        )
    print(json.dumps(result, ensure_ascii=False, indent=2, default=atlas._json_default))
    return result


if __name__ == "__main__":
    main()
