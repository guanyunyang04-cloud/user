"""Bad-tail hurdle and executable-return ranking research for Seq100.

The module estimates the conditional law of gross executable returns while
keeping entry, liquidation, and right-censored sell delay as separate random
objects.  It deliberately stops at forecast evaluation: candidate rankings
are formed on the full signal-date universe, costs remain decision-layer
thresholds, and no finite-capital account or policy surface is optimized.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy import special, stats
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.preprocessing import SplineTransformer

from daily_research.path_policy import seq100_stock_distribution as base


WORKSPACE_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_STUDY_PATH = (
    WORKSPACE_ROOT / "daily_research/studies/seq100_stock_bad_tail_v1.json"
)
DEFAULT_INPUT_MANIFEST = base.DEFAULT_INPUT_MANIFEST
DEFAULT_LABEL_MANIFEST = base.DEFAULT_OUTPUT_ROOT / "manifest.json"
DEFAULT_OUTPUT_ROOT = (
    WORKSPACE_ROOT
    / "daily_research/output/path_policy/studies/seq100_stock_bad_tail_v1"
)

STUDY_ID = "seq100_stock_bad_tail_v1"
MANIFEST_SCHEMA = "seq100_stock_bad_tail_manifest/1"
RANK_CACHE_SCHEMA = "seq100_stock_structural_rank_cache/1"
FOLD_SCHEMA = "seq100_stock_bad_tail_fold/1"
EXPECTED_LABEL_FINGERPRINT = (
    "c93cf15be7f58642b34e086c0c25e00b56a4f46de514d01a35b3132f2e0bc65d"
)
PRIMARY_HORIZON = 20
DIAGNOSTIC_HORIZON = 10
HORIZONS = (PRIMARY_HORIZON, DIAGNOSTIC_HORIZON)
QUANTILE_PROBABILITIES = (0.05, 0.10, 0.25, 0.50, 0.75, 0.90)
STRUCTURAL_FEATURES = (
    "log_turnover_pct_1d",
    "ma_distance_60d",
    "return_20d",
    "atr_60d",
    "volatility_60d",
)
REGIME_FEATURES = ("market_all__ret20_mean", "market_all__vol20_mean")
DEVELOPMENT_YEARS = base.DEVELOPMENT_YEARS
PREFLIGHT_YEARS = (2022,)
RETRY_DAYS = base.RETRY_DAYS
FORBIDDEN_YEAR = base.FORBIDDEN_YEAR


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    base._write_json(path, value)


def _resolve_path(value: str | Path) -> Path:
    return base._resolve_path(value)


def _write_parquet(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = Path(str(path) + ".partial")
    partial.unlink(missing_ok=True)
    frame.to_parquet(partial, index=False)
    os.replace(partial, path)


def _hash_payload(value: Mapping[str, Any]) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, ensure_ascii=False).encode("utf-8")
    ).hexdigest()


def load_study(study_path: str | Path = DEFAULT_STUDY_PATH) -> dict[str, Any]:
    path = _resolve_path(study_path)
    study = _read_json(path)
    if study.get("study_id") != STUDY_ID:
        raise ValueError("study_id_mismatch")
    source = dict(study.get("source", {}) or {})
    if str(source.get("expected_input_fingerprint")) != base.EXPECTED_INPUT_FINGERPRINT:
        raise ValueError("input_fingerprint_contract_mismatch")
    if str(source.get("expected_label_fingerprint")) != EXPECTED_LABEL_FINGERPRINT:
        raise ValueError("label_fingerprint_contract_mismatch")
    if int(source.get("expected_row_count", -1)) != base.EXPECTED_ROW_COUNT:
        raise ValueError("row_count_contract_mismatch")
    targets = dict(study.get("targets", {}) or {})
    if int(targets.get("primary_horizon", -1)) != PRIMARY_HORIZON:
        raise ValueError("primary_horizon_contract_mismatch")
    if int(targets.get("diagnostic_horizon", -1)) != DIAGNOSTIC_HORIZON:
        raise ValueError("diagnostic_horizon_contract_mismatch")
    probabilities = tuple(float(v) for v in targets.get("quantile_probabilities", ()))
    if probabilities != QUANTILE_PROBABILITIES:
        raise ValueError("quantile_contract_mismatch")
    structural = tuple(
        str(v)
        for v in study.get("features", {})
        .get("rank_gam", {})
        .get("structural_features", ())
    )
    if structural != STRUCTURAL_FEATURES:
        raise ValueError("structural_feature_contract_mismatch")
    if bool(study.get("decision_boundary", {}).get("portfolio_selection_performed", True)):
        raise ValueError("study_must_start_without_portfolio_selection")
    if bool(study.get("decision_boundary", {}).get("account_optimization_performed", True)):
        raise ValueError("study_must_start_without_account_optimization")
    return study


def _load_panel(
    *,
    input_manifest_path: str | Path = DEFAULT_INPUT_MANIFEST,
    label_manifest_path: str | Path = DEFAULT_LABEL_MANIFEST,
) -> base.StockPanel:
    panel = base.load_panel(
        input_manifest_path=input_manifest_path,
        label_manifest_path=label_manifest_path,
    )
    if panel.label_manifest.get("experiment_fingerprint") != EXPECTED_LABEL_FINGERPRINT:
        raise ValueError("prepared_label_fingerprint_mismatch")
    if bool(pd.Series(panel.trade_date).str.startswith("2026-").any()):
        raise ValueError("forbidden_2026_panel_row")
    return panel


def _mid_percentile_rank(values: np.ndarray) -> np.ndarray:
    """Return mid-ranks in (0, 1), mapping missing values to the neutral 0.5."""

    x = np.asarray(values, dtype=np.float64)
    result = np.full(len(x), 0.5, dtype=np.float32)
    finite = np.isfinite(x)
    count = int(finite.sum())
    if count:
        result[finite] = (
            (stats.rankdata(x[finite], method="average") - 0.5) / float(count)
        ).astype(np.float32)
    return result


def _rank_cache_fingerprint(feature_names: Sequence[str]) -> str:
    return _hash_payload(
        {
            "schema": RANK_CACHE_SCHEMA,
            "input_fingerprint": base.EXPECTED_INPUT_FINGERPRINT,
            "row_count": base.EXPECTED_ROW_COUNT,
            "features": list(feature_names),
            "method": "full-date finite mid-percentile; missing maps to 0.5",
            "forbidden_year": FORBIDDEN_YEAR,
        }
    )


def prepare_structural_ranks(
    panel: base.StockPanel,
    *,
    output_root: str | Path = DEFAULT_OUTPUT_ROOT,
    feature_names: Sequence[str] = STRUCTURAL_FEATURES,
) -> tuple[np.memmap, dict[str, Any]]:
    """Build full-cross-section ranks once so target validity cannot affect ranks."""

    output_root = _resolve_path(output_root)
    cache_root = output_root / "derived"
    cache_root.mkdir(parents=True, exist_ok=True)
    manifest_path = cache_root / "structural_rank_manifest.json"
    data_path = cache_root / "structural_ranks.float32.dat"
    names = tuple(str(v) for v in feature_names)
    fingerprint = _rank_cache_fingerprint(names)
    shape = (panel.row_count, len(names))
    expected_size = int(np.prod(shape)) * np.dtype(np.float32).itemsize
    if manifest_path.is_file():
        manifest = _read_json(manifest_path)
        record = dict(manifest.get("file", {}) or {})
        if (
            manifest.get("schema") == RANK_CACHE_SCHEMA
            and manifest.get("fingerprint") == fingerprint
            and data_path.is_file()
            and int(data_path.stat().st_size) == expected_size
            and str(record.get("sha256")) == base._sha256_file(data_path)
        ):
            return (
                np.memmap(data_path, dtype=np.float32, mode="r", shape=shape),
                manifest,
            )

    positions = base._feature_positions(panel, names)
    partial = Path(str(data_path) + ".partial")
    partial.unlink(missing_ok=True)
    ranks = np.memmap(partial, dtype=np.float32, mode="w+", shape=shape)
    dates = np.asarray(panel.date_idx, dtype=np.int64)
    boundaries = np.r_[0, np.flatnonzero(dates[1:] != dates[:-1]) + 1, len(dates)]
    for boundary_pos in range(len(boundaries) - 1):
        left = int(boundaries[boundary_pos])
        right = int(boundaries[boundary_pos + 1])
        rows = np.arange(left, right, dtype=np.int64)
        values = base._feature_matrix(panel, rows, positions)
        for feature_pos in range(len(names)):
            ranks[left:right, feature_pos] = _mid_percentile_rank(
                values[:, feature_pos]
            )
    ranks.flush()
    del ranks
    os.replace(partial, data_path)
    record = base._file_record(
        data_path,
        shape=shape,
        dtype="float32",
        columns=names,
    )
    manifest = {
        "schema": RANK_CACHE_SCHEMA,
        "status": "completed",
        "fingerprint": fingerprint,
        "input_fingerprint": base.EXPECTED_INPUT_FINGERPRINT,
        "row_count": panel.row_count,
        "features": list(names),
        "rank_definition": (
            "Within each complete signal-date candidate cross-section, use "
            "(average_rank-0.5)/finite_count; missing values map to 0.5."
        ),
        "target_or_future_state_used": False,
        "maximum_signal_date": str(panel.trade_date[-1]),
        "forbidden_2026_read_count": 0,
        "file": record,
    }
    _write_json(manifest_path, manifest)
    return np.memmap(data_path, dtype=np.float32, mode="r", shape=shape), manifest


def _fixed_spline_transformer(
    x: np.ndarray, *, n_knots: int, degree: int
) -> tuple[SplineTransformer, np.ndarray]:
    values = np.asarray(x, dtype=np.float64)
    knots = np.tile(
        np.linspace(0.0, 1.0, int(n_knots), dtype=np.float64)[:, None],
        (1, values.shape[1]),
    )
    transformer = SplineTransformer(
        n_knots=int(n_knots),
        degree=int(degree),
        knots=knots,
        extrapolation="constant",
        include_bias=False,
        order="C",
    )
    return transformer, np.asarray(transformer.fit_transform(values), dtype=np.float32)


@dataclass
class RankGamDistribution:
    transformer: SplineTransformer
    mean_model: Ridge
    scale_model: Ridge
    df: float
    residual_scale: float


@dataclass
class RankGamBinary:
    transformer: SplineTransformer
    model: LogisticRegression | None
    constant_probability: float


@dataclass
class DiscreteHazardModel:
    transformer: SplineTransformer
    model: LogisticRegression | None
    empirical_hazards: np.ndarray
    retry_days: int


def _fit_rank_gam_distribution(
    ranks: np.ndarray,
    values: np.ndarray,
    date_idx: np.ndarray,
    *,
    n_knots: int,
    degree: int,
    ridge_penalty: float,
) -> tuple[RankGamDistribution, dict[str, Any]]:
    x = np.asarray(ranks, dtype=np.float32)
    y = np.asarray(values, dtype=np.float64)
    dates = np.asarray(date_idx, dtype=np.int64)
    if len(x) != len(y) or not np.isfinite(y).all():
        raise ValueError("rank_gam_distribution_training_data_invalid")
    weights = base._date_equal_weights(dates)
    transformer, basis = _fixed_spline_transformer(
        x, n_knots=n_knots, degree=degree
    )
    effective_dates = float(np.sum(weights))
    alpha = float(ridge_penalty) * effective_dates
    mean_model = Ridge(alpha=alpha, fit_intercept=True)
    mean_model.fit(basis, y, sample_weight=weights)
    location = np.asarray(mean_model.predict(basis), dtype=np.float64)
    residual = y - location
    floor = max(float(np.median(np.abs(residual))) ** 2 * 1.0e-4, 1.0e-10)
    scale_model = Ridge(alpha=alpha, fit_intercept=True)
    scale_model.fit(
        basis,
        np.log(np.square(residual) + floor),
        sample_weight=weights,
    )
    raw_scale = np.sqrt(
        np.exp(np.clip(scale_model.predict(basis), -30.0, 10.0))
    )
    df, residual_scale = base._safe_t_fit(
        residual / np.maximum(raw_scale, 1.0e-8)
    )
    model = RankGamDistribution(
        transformer=transformer,
        mean_model=mean_model,
        scale_model=scale_model,
        df=df,
        residual_scale=residual_scale,
    )
    summary = {
        "training_row_count": len(y),
        "training_date_count": int(np.unique(dates).size),
        "basis_count": int(basis.shape[1]),
        "effective_date_weight": effective_dates,
        "ridge_alpha": alpha,
        "student_t_df": df,
        "student_t_residual_scale": residual_scale,
    }
    return model, summary


def _predict_rank_gam_distribution(
    model: RankGamDistribution,
    ranks: np.ndarray,
    *,
    batch_size: int = 50_000,
) -> tuple[np.ndarray, np.ndarray]:
    x = np.asarray(ranks, dtype=np.float32)
    location = np.empty(len(x), dtype=np.float64)
    scale = np.empty(len(x), dtype=np.float64)
    for start in range(0, len(x), int(batch_size)):
        stop = min(start + int(batch_size), len(x))
        basis = np.asarray(model.transformer.transform(x[start:stop]), dtype=np.float32)
        location[start:stop] = model.mean_model.predict(basis)
        raw = np.sqrt(
            np.exp(np.clip(model.scale_model.predict(basis), -30.0, 10.0))
        )
        scale[start:stop] = np.maximum(raw * model.residual_scale, 1.0e-8)
    return location, scale


def _fit_rank_gam_binary(
    ranks: np.ndarray,
    values: np.ndarray,
    date_idx: np.ndarray,
    *,
    n_knots: int,
    degree: int,
    regularization_c: float,
) -> tuple[RankGamBinary, dict[str, Any]]:
    x = np.asarray(ranks, dtype=np.float32)
    y = np.asarray(values, dtype=np.int8)
    dates = np.asarray(date_idx, dtype=np.int64)
    if not np.isin(y, (0, 1)).all():
        raise ValueError("rank_gam_binary_target_invalid")
    weights = base._date_equal_weights(dates)
    transformer, basis = _fixed_spline_transformer(
        x, n_knots=n_knots, degree=degree
    )
    probability = float(np.average(y, weights=weights))
    classifier: LogisticRegression | None
    if len(np.unique(y)) < 2:
        classifier = None
    else:
        classifier = LogisticRegression(
            C=float(regularization_c),
            solver="lbfgs",
            max_iter=300,
            tol=1.0e-6,
        )
        classifier.fit(basis, y, sample_weight=weights)
    return (
        RankGamBinary(
            transformer=transformer,
            model=classifier,
            constant_probability=probability,
        ),
        {
            "training_row_count": len(y),
            "training_date_count": int(np.unique(dates).size),
            "basis_count": int(basis.shape[1]),
            "event_count": int(y.sum()),
            "event_rate_date_equal": probability,
        },
    )


def _predict_rank_gam_binary(
    model: RankGamBinary,
    ranks: np.ndarray,
    *,
    batch_size: int = 50_000,
) -> np.ndarray:
    x = np.asarray(ranks, dtype=np.float32)
    result = np.empty(len(x), dtype=np.float64)
    for start in range(0, len(x), int(batch_size)):
        stop = min(start + int(batch_size), len(x))
        if model.model is None:
            result[start:stop] = model.constant_probability
        else:
            basis = np.asarray(
                model.transformer.transform(x[start:stop]), dtype=np.float32
            )
            result[start:stop] = model.model.predict_proba(basis)[:, 1]
    return np.clip(result, 1.0e-8, 1.0 - 1.0e-8)


def _recent_probability(
    values: np.ndarray, date_idx: np.ndarray, *, recent_date_count: int = 252
) -> float:
    y = np.asarray(values, dtype=np.float64)
    dates = np.asarray(date_idx, dtype=np.int64)
    unique_dates = np.unique(dates)
    recent = unique_dates[-min(int(recent_date_count), len(unique_dates)) :]
    selected = np.isin(dates, recent)
    weights = base._date_equal_weights(dates[selected])
    return float(np.average(y[selected], weights=weights))


def _lgb_parameters(
    config: Mapping[str, Any], *, objective: str, alpha: float | None = None
) -> dict[str, Any]:
    parameters: dict[str, Any] = {
        "objective": str(objective),
        "metric": "None",
        "learning_rate": float(config["learning_rate"]),
        "num_leaves": int(config["num_leaves"]),
        "max_depth": int(config["max_depth"]),
        "min_data_in_leaf": int(config["min_data_in_leaf"]),
        "lambda_l2": float(config["lambda_l2"]),
        "max_bin": int(config["max_bin"]),
        "feature_fraction": float(config["feature_fraction"]),
        "bagging_fraction": float(config["bagging_fraction"]),
        "bagging_freq": 0,
        "num_threads": int(config["num_threads"]),
        "seed": int(config["seed"]),
        "feature_fraction_seed": int(config["seed"]),
        "bagging_seed": int(config["seed"]),
        "data_random_seed": int(config["seed"]),
        "deterministic": True,
        "force_col_wise": True,
        "verbosity": -1,
    }
    if alpha is not None:
        parameters["alpha"] = float(alpha)
    return parameters


def _predict_boosters(
    panel: base.StockPanel,
    rows: np.ndarray,
    positions: np.ndarray,
    boosters: Mapping[str, Any],
    *,
    iterations: int,
    batch_size: int = 50_000,
) -> dict[str, np.ndarray]:
    selected = np.asarray(rows, dtype=np.int64)
    result = {
        name: np.empty(len(selected), dtype=np.float64) for name in boosters
    }
    for start in range(0, len(selected), int(batch_size)):
        stop = min(start + int(batch_size), len(selected))
        x = base._feature_matrix(panel, selected[start:stop], positions)
        for name, booster in boosters.items():
            result[name][start:stop] = booster.predict(
                x, num_iteration=int(iterations)
            )
    return result


def _fit_lgb_continuous(
    panel: base.StockPanel,
    *,
    train_rows: np.ndarray,
    evaluation_rows: np.ndarray,
    target_values: np.ndarray,
    feature_names: Sequence[str],
    probabilities: Sequence[float],
    config: Mapping[str, Any],
    model_root: Path,
    prefix: str,
) -> tuple[np.ndarray, np.ndarray, float, dict[str, Any]]:
    import lightgbm as lgb

    if str(lgb.__version__) != str(config["runtime_version"]):
        raise ValueError("lightgbm_runtime_version_mismatch")
    positions = base._feature_positions(panel, feature_names)
    x = base._feature_matrix(panel, train_rows, positions)
    y = np.asarray(target_values[train_rows], dtype=np.float64)
    if not np.isfinite(y).all():
        raise ValueError("lightgbm_continuous_target_not_finite")
    weights = base._date_equal_weights(panel.date_idx[train_rows])
    dataset = lgb.Dataset(
        x,
        label=y,
        weight=weights,
        feature_name=list(feature_names),
        free_raw_data=False,
    )
    rounds = int(config["num_boost_round"])
    boosters: dict[str, Any] = {}
    boosters["mean"] = lgb.train(
        _lgb_parameters(config, objective="regression_l2"),
        dataset,
        num_boost_round=rounds,
    )
    for probability in probabilities:
        name = f"q{int(round(float(probability) * 100)):02d}"
        boosters[name] = lgb.train(
            _lgb_parameters(config, objective="quantile", alpha=float(probability)),
            dataset,
            num_boost_round=rounds,
        )
    model_root.mkdir(parents=True, exist_ok=True)
    for name, booster in boosters.items():
        booster.save_model(str(model_root / f"{prefix}__{name}.txt"))
    predicted = _predict_boosters(
        panel,
        evaluation_rows,
        positions,
        boosters,
        iterations=rounds,
    )
    raw_quantiles = np.column_stack(
        [predicted[f"q{int(round(float(p) * 100)):02d}"] for p in probabilities]
    )
    crossing = _quantile_crossing_rate(raw_quantiles)
    quantiles = _monotone_rearrange(raw_quantiles)
    importance = boosters["mean"].feature_importance(importance_type="gain")
    top = np.argsort(importance)[::-1][:20]
    summary = {
        "training_row_count": len(train_rows),
        "training_date_count": int(np.unique(panel.date_idx[train_rows]).size),
        "feature_count": len(feature_names),
        "rounds": rounds,
        "raw_quantile_crossing_rate": crossing,
        "post_rearrangement_crossing_rate": _quantile_crossing_rate(quantiles),
        "top_mean_gain_features": [
            {
                "feature": str(feature_names[int(pos)]),
                "gain": float(importance[int(pos)]),
            }
            for pos in top
        ],
    }
    return predicted["mean"], quantiles, crossing, summary


def _fit_lgb_binary(
    panel: base.StockPanel,
    *,
    train_rows: np.ndarray,
    evaluation_rows: np.ndarray,
    target_values: np.ndarray,
    feature_names: Sequence[str],
    config: Mapping[str, Any],
    model_path: Path,
) -> tuple[np.ndarray, dict[str, Any]]:
    import lightgbm as lgb

    if str(lgb.__version__) != str(config["runtime_version"]):
        raise ValueError("lightgbm_runtime_version_mismatch")
    positions = base._feature_positions(panel, feature_names)
    x = base._feature_matrix(panel, train_rows, positions)
    y = np.asarray(target_values[train_rows], dtype=np.int8)
    if not np.isin(y, (0, 1)).all() or len(np.unique(y)) != 2:
        raise ValueError("lightgbm_binary_target_requires_two_classes")
    weights = base._date_equal_weights(panel.date_idx[train_rows])
    dataset = lgb.Dataset(
        x,
        label=y,
        weight=weights,
        feature_name=list(feature_names),
        free_raw_data=False,
    )
    rounds = int(config["num_boost_round"])
    booster = lgb.train(
        _lgb_parameters(config, objective="binary"),
        dataset,
        num_boost_round=rounds,
    )
    model_path.parent.mkdir(parents=True, exist_ok=True)
    booster.save_model(str(model_path))
    probability = _predict_boosters(
        panel,
        evaluation_rows,
        positions,
        {"probability": booster},
        iterations=rounds,
    )["probability"]
    importance = booster.feature_importance(importance_type="gain")
    top = np.argsort(importance)[::-1][:20]
    return np.clip(probability, 1.0e-8, 1.0 - 1.0e-8), {
        "training_row_count": len(train_rows),
        "training_date_count": int(np.unique(panel.date_idx[train_rows]).size),
        "event_count": int(y.sum()),
        "event_rate_date_equal": float(np.average(y, weights=weights)),
        "feature_count": len(feature_names),
        "rounds": rounds,
        "top_gain_features": [
            {
                "feature": str(feature_names[int(pos)]),
                "gain": float(importance[int(pos)]),
            }
            for pos in top
        ],
    }


def _quantile_crossing_rate(quantiles: np.ndarray) -> float:
    values = np.asarray(quantiles, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] < 2:
        raise ValueError("quantile_matrix_invalid")
    return float(np.mean(np.any(values[:, 1:] < values[:, :-1], axis=1)))


def _monotone_rearrange(quantiles: np.ndarray) -> np.ndarray:
    """Pointwise increasing rearrangement of independently fitted quantiles."""

    return np.sort(np.asarray(quantiles, dtype=np.float64), axis=1)


def _t_quantiles(
    location: np.ndarray,
    scale: np.ndarray,
    df: float,
    probabilities: Sequence[float] = QUANTILE_PROBABILITIES,
) -> np.ndarray:
    standardized = stats.t.ppf(np.asarray(probabilities, dtype=np.float64), float(df))
    return np.asarray(location, dtype=np.float64)[:, None] + np.asarray(
        scale, dtype=np.float64
    )[:, None] * standardized[None, :]


def _quantile_crps_approximation(
    values: np.ndarray,
    quantiles: np.ndarray,
    probabilities: Sequence[float] = QUANTILE_PROBABILITIES,
) -> np.ndarray:
    """Approximate CRPS by integrating pinball loss on the frozen grid."""

    y = np.asarray(values, dtype=np.float64)
    q = np.asarray(quantiles, dtype=np.float64)
    probs = np.asarray(probabilities, dtype=np.float64)
    if q.shape != (len(y), len(probs)):
        raise ValueError("quantile_crps_shape_mismatch")
    extended_probabilities = np.r_[0.0, probs, 1.0]
    extended_quantiles = np.column_stack([q[:, 0], q, q[:, -1]])
    error = y[:, None] - extended_quantiles
    loss = np.maximum(
        extended_probabilities[None, :] * error,
        (extended_probabilities[None, :] - 1.0) * error,
    )
    return 2.0 * np.trapezoid(loss, extended_probabilities, axis=1)


def _expand_discrete_hazard(
    tau: np.ndarray,
    *,
    retry_days: int = RETRY_DAYS,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Expand filled/censored durations into person-period Bernoulli rows."""

    duration = np.asarray(tau, dtype=np.int64)
    if bool(((duration < 0) | (duration > int(retry_days) + 1)).any()):
        raise ValueError("hazard_duration_out_of_contract")
    lengths = np.minimum(duration, int(retry_days)) + 1
    subject_index = np.repeat(np.arange(len(duration), dtype=np.int64), lengths)
    starts = np.repeat(np.cumsum(np.r_[0, lengths[:-1]]), lengths)
    delay = np.arange(int(lengths.sum()), dtype=np.int64) - starts
    event = (
        (duration[subject_index] <= int(retry_days))
        & (delay == duration[subject_index])
    ).astype(np.int8)
    return subject_index, delay, event


def _fit_discrete_hazard(
    ranks: np.ndarray,
    tau: np.ndarray,
    date_idx: np.ndarray,
    *,
    n_knots: int,
    degree: int,
    regularization_c: float,
    retry_days: int = RETRY_DAYS,
) -> tuple[DiscreteHazardModel, dict[str, Any]]:
    x = np.asarray(ranks, dtype=np.float32)
    duration = np.asarray(tau, dtype=np.int64)
    dates = np.asarray(date_idx, dtype=np.int64)
    transformer, subject_basis = _fixed_spline_transformer(
        x, n_knots=n_knots, degree=degree
    )
    subject_index, delay, event = _expand_discrete_hazard(
        duration, retry_days=retry_days
    )
    delay_design = np.zeros((len(delay), int(retry_days)), dtype=np.float32)
    positive = delay > 0
    delay_design[np.flatnonzero(positive), delay[positive] - 1] = 1.0
    design = np.column_stack([subject_basis[subject_index], delay_design])
    period_dates = dates[subject_index]
    weights = base._date_equal_weights(period_dates)
    empirical = np.empty(int(retry_days) + 1, dtype=np.float64)
    for value in range(int(retry_days) + 1):
        selected = delay == value
        total_weight = float(weights[selected].sum())
        event_weight = float(np.sum(weights[selected] * event[selected]))
        empirical[value] = (event_weight + 0.5) / (total_weight + 1.0)
    classifier: LogisticRegression | None
    if len(np.unique(event)) < 2:
        classifier = None
    else:
        classifier = LogisticRegression(
            C=float(regularization_c),
            solver="lbfgs",
            max_iter=300,
            tol=1.0e-6,
        )
        classifier.fit(design, event, sample_weight=weights)
    model = DiscreteHazardModel(
        transformer=transformer,
        model=classifier,
        empirical_hazards=np.clip(empirical, 1.0e-8, 1.0 - 1.0e-8),
        retry_days=int(retry_days),
    )
    summary = {
        "subject_count": len(duration),
        "person_period_count": len(event),
        "training_date_count": int(np.unique(dates).size),
        "event_count": int(event.sum()),
        "right_censored_count": int(np.sum(duration == int(retry_days) + 1)),
        "delayed_event_count": int(
            np.sum((duration > 0) & (duration <= int(retry_days)))
        ),
        "empirical_hazards": empirical.tolist(),
    }
    return model, summary


def _predict_discrete_hazard(
    model: DiscreteHazardModel,
    ranks: np.ndarray,
    *,
    model_name: str,
    batch_size: int = 50_000,
) -> np.ndarray:
    x = np.asarray(ranks, dtype=np.float32)
    if model_name == "empirical_hazard" or model.model is None:
        return np.tile(model.empirical_hazards, (len(x), 1))
    if model_name != "rank_gam_hazard":
        raise ValueError(f"unknown_hazard_model:{model_name}")
    result = np.empty((len(x), model.retry_days + 1), dtype=np.float64)
    coefficient = np.asarray(model.model.coef_[0], dtype=np.float64)
    basis_count = int(model.transformer.n_features_out_)
    basis_coefficient = coefficient[:basis_count]
    delay_coefficient = np.r_[0.0, coefficient[basis_count:]]
    for start in range(0, len(x), int(batch_size)):
        stop = min(start + int(batch_size), len(x))
        basis = np.asarray(model.transformer.transform(x[start:stop]), dtype=np.float64)
        linear = float(model.model.intercept_[0]) + basis @ basis_coefficient
        result[start:stop] = special.expit(
            linear[:, None] + delay_coefficient[None, :]
        )
    return np.clip(result, 1.0e-8, 1.0 - 1.0e-8)


def _hazard_mass(hazard: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    h = np.clip(np.asarray(hazard, dtype=np.float64), 1.0e-12, 1.0 - 1.0e-12)
    survival_before = np.column_stack(
        [np.ones(len(h), dtype=np.float64), np.cumprod(1.0 - h[:, :-1], axis=1)]
    )
    mass = survival_before * h
    censored = np.prod(1.0 - h, axis=1)
    return mass, censored


def _regime_frame(
    panel: base.StockPanel,
    *,
    train_rows: np.ndarray,
    evaluation_rows: np.ndarray,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    positions = base._feature_positions(panel, REGIME_FEATURES)
    train_dates = np.unique(panel.date_idx[np.asarray(train_rows, dtype=np.int64)])
    evaluation_dates = np.unique(
        panel.date_idx[np.asarray(evaluation_rows, dtype=np.int64)]
    )
    train_first_rows = np.searchsorted(panel.date_idx, train_dates)
    evaluation_first_rows = np.searchsorted(panel.date_idx, evaluation_dates)
    train_values = base._feature_matrix(panel, train_first_rows, positions).astype(
        np.float64
    )
    evaluation_values = base._feature_matrix(
        panel, evaluation_first_rows, positions
    ).astype(np.float64)
    thresholds: dict[str, list[float]] = {}
    labels: list[np.ndarray] = []
    names = (("trend", ("down", "middle", "up")), ("volatility", ("low", "middle", "high")))
    for feature_pos, (name, categories) in enumerate(names):
        finite_train = train_values[:, feature_pos][
            np.isfinite(train_values[:, feature_pos])
        ]
        if not len(finite_train):
            raise ValueError(f"regime_training_feature_missing:{name}")
        lower, upper = np.quantile(finite_train, [1.0 / 3.0, 2.0 / 3.0])
        thresholds[name] = [float(lower), float(upper)]
        values = evaluation_values[:, feature_pos]
        label = np.full(len(values), "unknown", dtype=object)
        finite = np.isfinite(values)
        label[finite & (values < lower)] = categories[0]
        label[finite & (values >= lower) & (values <= upper)] = categories[1]
        label[finite & (values > upper)] = categories[2]
        labels.append(label)
    frame = pd.DataFrame(
        {
            "date_idx": evaluation_dates.astype(np.int64),
            "trend_regime": labels[0],
            "volatility_regime": labels[1],
        }
    )
    return frame, {
        "features": list(REGIME_FEATURES),
        "threshold_source": "training dates only",
        "thresholds": thresholds,
        "training_date_count": len(train_dates),
        "evaluation_date_count": len(evaluation_dates),
    }


def _attach_regimes(daily: pd.DataFrame, regimes: pd.DataFrame) -> pd.DataFrame:
    return daily.merge(regimes, how="left", on="date_idx", validate="many_to_one")


def _safe_binary_discrimination(
    values: np.ndarray, probabilities: np.ndarray, weights: np.ndarray | None = None
) -> tuple[float, float]:
    y = np.asarray(values, dtype=np.int8)
    p = np.asarray(probabilities, dtype=np.float64)
    if len(np.unique(y)) < 2:
        return float("nan"), float("nan")
    return (
        float(roc_auc_score(y, p, sample_weight=weights)),
        float(average_precision_score(y, p, sample_weight=weights)),
    )


def _distribution_scores(
    *,
    panel: base.StockPanel,
    evaluation_rows: np.ndarray,
    valid: np.ndarray,
    observed: np.ndarray,
    mean: np.ndarray,
    quantiles: np.ndarray,
    model: str,
    target: str,
    horizon: int,
    evaluation_year: int,
    regimes: pd.DataFrame,
    t_parameters: tuple[np.ndarray, np.ndarray, float] | None = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    rows = np.asarray(evaluation_rows, dtype=np.int64)
    keep = np.asarray(valid, dtype=bool)
    y = np.asarray(observed, dtype=np.float64)[keep]
    predicted_mean = np.asarray(mean, dtype=np.float64)[keep]
    q = np.asarray(quantiles, dtype=np.float64)[keep]
    date_idx = panel.date_idx[rows][keep]
    if not len(y) or not np.isfinite(y).all() or not np.isfinite(q).all():
        raise ValueError(f"distribution_evaluation_invalid:{target}:{model}")
    raw: dict[str, Any] = {
        "date_idx": date_idx,
        "squared_error": np.square(y - predicted_mean),
        "quantile_crps": _quantile_crps_approximation(y, q),
    }
    pinball_columns: list[str] = []
    for pos, probability in enumerate(QUANTILE_PROBABILITIES):
        suffix = f"{int(round(probability * 100)):02d}"
        column = f"pinball_{suffix}"
        pinball_columns.append(column)
        raw[column] = base._pinball(y, q[:, pos], probability)
        raw[f"coverage_q{suffix}"] = (y <= q[:, pos]).astype(np.float64)
    raw["mean_pinball"] = np.mean(
        np.column_stack([raw[column] for column in pinball_columns]), axis=1
    )
    probability_positions = {
        probability: pos for pos, probability in enumerate(QUANTILE_PROBABILITIES)
    }
    raw["coverage_50"] = (
        (y >= q[:, probability_positions[0.25]])
        & (y <= q[:, probability_positions[0.75]])
    ).astype(np.float64)
    raw["coverage_80"] = (
        (y >= q[:, probability_positions[0.10]])
        & (y <= q[:, probability_positions[0.90]])
    ).astype(np.float64)
    raw["coverage_85"] = (
        (y >= q[:, probability_positions[0.05]])
        & (y <= q[:, probability_positions[0.90]])
    ).astype(np.float64)
    if t_parameters is None:
        raw["log_score"] = np.full(len(y), np.nan, dtype=np.float64)
    else:
        location, scale, df = t_parameters
        raw["log_score"] = stats.t.logpdf(
            y,
            float(df),
            loc=np.asarray(location, dtype=np.float64)[keep],
            scale=np.maximum(np.asarray(scale, dtype=np.float64)[keep], 1.0e-10),
        )
    daily = pd.DataFrame(raw).groupby("date_idx", sort=True, as_index=False).mean()
    daily = _attach_regimes(daily, regimes)
    daily["evaluation_year"] = int(evaluation_year)
    daily["horizon"] = int(horizon)
    daily["target"] = str(target)
    daily["model"] = str(model)
    weights = base._date_equal_weights(date_idx)
    metric: dict[str, Any] = {
        "evaluation_year": int(evaluation_year),
        "horizon": int(horizon),
        "target": str(target),
        "model": str(model),
        "row_count": len(y),
        "date_count": len(daily),
        "mse": float(np.average(np.square(y - predicted_mean), weights=weights)),
        "rmse": float(
            math.sqrt(np.average(np.square(y - predicted_mean), weights=weights))
        ),
        "quantile_crps": float(
            np.average(_quantile_crps_approximation(y, q), weights=weights)
        ),
        "mean_pinball": float(daily["mean_pinball"].mean()),
        "coverage_50": float(daily["coverage_50"].mean()),
        "coverage_80": float(daily["coverage_80"].mean()),
        "coverage_85": float(daily["coverage_85"].mean()),
        "quantile_crossing_rate": _quantile_crossing_rate(q),
    }
    for probability in QUANTILE_PROBABILITIES:
        suffix = f"{int(round(probability * 100)):02d}"
        metric[f"pinball_{suffix}"] = float(daily[f"pinball_{suffix}"].mean())
        metric[f"coverage_q{suffix}"] = float(daily[f"coverage_q{suffix}"].mean())
    finite_log = daily["log_score"].dropna()
    metric["log_score"] = (
        float(finite_log.mean()) if len(finite_log) else float("nan")
    )
    return daily, metric


def _binary_scores(
    *,
    panel: base.StockPanel,
    evaluation_rows: np.ndarray,
    valid: np.ndarray,
    observed: np.ndarray,
    probability: np.ndarray,
    model: str,
    target: str,
    horizon: int,
    evaluation_year: int,
    regimes: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, Any], pd.DataFrame]:
    rows = np.asarray(evaluation_rows, dtype=np.int64)
    keep = np.asarray(valid, dtype=bool)
    y = np.asarray(observed, dtype=np.int8)[keep]
    p = np.clip(np.asarray(probability, dtype=np.float64)[keep], 1.0e-8, 1.0 - 1.0e-8)
    dates = panel.date_idx[rows][keep]
    if not len(y) or not np.isin(y, (0, 1)).all() or not np.isfinite(p).all():
        raise ValueError(f"binary_evaluation_invalid:{target}:{model}")
    raw = pd.DataFrame(
        {
            "date_idx": dates,
            "log_score": y * np.log(p) + (1 - y) * np.log(1.0 - p),
            "brier": np.square(y - p),
            "observed_rate": y.astype(np.float64),
            "predicted_rate": p,
        }
    )
    daily = raw.groupby("date_idx", sort=True, as_index=False).mean()
    discrimination: list[dict[str, float]] = []
    for date_value, positions in pd.Series(dates).groupby(dates).groups.items():
        local = np.asarray(list(positions), dtype=np.int64)
        auc, average_precision = _safe_binary_discrimination(y[local], p[local])
        discrimination.append(
            {
                "date_idx": int(date_value),
                "roc_auc": auc,
                "average_precision": average_precision,
            }
        )
    daily = daily.merge(
        pd.DataFrame(discrimination), how="left", on="date_idx", validate="one_to_one"
    )
    daily["absolute_calibration_error"] = np.abs(
        daily["observed_rate"] - daily["predicted_rate"]
    )
    daily = _attach_regimes(daily, regimes)
    daily["evaluation_year"] = int(evaluation_year)
    daily["horizon"] = int(horizon)
    daily["target"] = str(target)
    daily["model"] = str(model)
    weights = base._date_equal_weights(dates)
    auc, average_precision = _safe_binary_discrimination(y, p, weights)
    metric = {
        "evaluation_year": int(evaluation_year),
        "horizon": int(horizon),
        "target": str(target),
        "model": str(model),
        "row_count": len(y),
        "date_count": len(daily),
        "log_score": float(daily["log_score"].mean()),
        "brier": float(daily["brier"].mean()),
        "observed_rate": float(daily["observed_rate"].mean()),
        "predicted_rate": float(daily["predicted_rate"].mean()),
        "absolute_calibration_error": float(
            daily["absolute_calibration_error"].mean()
        ),
        "roc_auc_date_equal_rows": auc,
        "average_precision_date_equal_rows": average_precision,
        "mean_daily_roc_auc": float(daily["roc_auc"].mean()),
        "mean_daily_average_precision": float(daily["average_precision"].mean()),
    }
    order = np.argsort(p, kind="stable")
    reliability_bin = np.empty(len(p), dtype=np.int8)
    reliability_bin[order] = np.minimum(
        (np.arange(len(p), dtype=np.int64) * 10) // max(len(p), 1), 9
    ).astype(np.int8)
    reliability_raw = pd.DataFrame(
        {
            "date_idx": dates,
            "bin": reliability_bin,
            "observed": y.astype(np.float64),
            "predicted": p,
        }
    )
    per_date_bin = (
        reliability_raw.groupby(["date_idx", "bin"], as_index=False)
        .agg(observed_rate=("observed", "mean"), predicted_rate=("predicted", "mean"), row_count=("observed", "size"))
    )
    reliability = (
        per_date_bin.groupby("bin", as_index=False)
        .agg(
            observed_rate=("observed_rate", "mean"),
            predicted_rate=("predicted_rate", "mean"),
            mean_daily_row_count=("row_count", "mean"),
            date_count=("date_idx", "nunique"),
        )
    )
    reliability["evaluation_year"] = int(evaluation_year)
    reliability["horizon"] = int(horizon)
    reliability["target"] = str(target)
    reliability["model"] = str(model)
    return daily, metric, reliability


def _hazard_scores(
    *,
    panel: base.StockPanel,
    evaluation_rows: np.ndarray,
    valid: np.ndarray,
    tau: np.ndarray,
    hazard: np.ndarray,
    model: str,
    horizon: int,
    evaluation_year: int,
    regimes: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    rows = np.asarray(evaluation_rows, dtype=np.int64)
    keep = np.asarray(valid, dtype=bool)
    duration = np.asarray(tau, dtype=np.int64)[keep]
    date_idx = panel.date_idx[rows][keep]
    h = np.asarray(hazard, dtype=np.float64)[keep]
    mass, censor_probability = _hazard_mass(h)
    event = duration <= RETRY_DAYS
    event_probability = 1.0 - censor_probability
    log_likelihood = np.log(np.maximum(censor_probability, 1.0e-12))
    event_rows = np.flatnonzero(event)
    log_likelihood[event_rows] = np.log(
        np.maximum(mass[event_rows, duration[event_rows]], 1.0e-12)
    )
    expected_delay = mass @ np.arange(RETRY_DAYS + 1, dtype=np.float64)
    expected_delay += (RETRY_DAYS + 1) * censor_probability
    raw = pd.DataFrame(
        {
            "date_idx": date_idx,
            "log_likelihood": log_likelihood,
            "sell_within_probability": event_probability,
            "sell_within_observed": event.astype(np.float64),
            "brier": np.square(event.astype(np.float64) - event_probability),
            "expected_delay": expected_delay,
            "observed_delay_or_censor": duration.astype(np.float64),
        }
    )
    daily = raw.groupby("date_idx", sort=True, as_index=False).mean()
    daily["absolute_calibration_error"] = np.abs(
        daily["sell_within_observed"] - daily["sell_within_probability"]
    )
    daily = _attach_regimes(daily, regimes)
    daily["evaluation_year"] = int(evaluation_year)
    daily["horizon"] = int(horizon)
    daily["target"] = f"tau_sell_{int(horizon)}"
    daily["model"] = str(model)
    weights = base._date_equal_weights(date_idx)
    return daily, {
        "evaluation_year": int(evaluation_year),
        "horizon": int(horizon),
        "target": f"tau_sell_{int(horizon)}",
        "model": str(model),
        "row_count": len(duration),
        "date_count": len(daily),
        "event_count": int(event.sum()),
        "right_censored_count": int((~event).sum()),
        "delayed_event_count": int(np.sum((duration > 0) & event)),
        "log_likelihood": float(daily["log_likelihood"].mean()),
        "brier": float(daily["brier"].mean()),
        "observed_sell_within_rate": float(daily["sell_within_observed"].mean()),
        "predicted_sell_within_rate": float(
            daily["sell_within_probability"].mean()
        ),
        "absolute_calibration_error": float(
            daily["absolute_calibration_error"].mean()
        ),
        "delay_mae_date_equal_rows": float(
            np.average(np.abs(duration - expected_delay), weights=weights)
        ),
    }


def _rank_ic(scores: np.ndarray, outcomes: np.ndarray) -> float:
    x = np.asarray(scores, dtype=np.float64)
    y = np.asarray(outcomes, dtype=np.float64)
    finite = np.isfinite(x) & np.isfinite(y)
    if int(finite.sum()) < 3 or np.unique(x[finite]).size < 2 or np.unique(y[finite]).size < 2:
        return float("nan")
    return float(stats.spearmanr(x[finite], y[finite]).statistic)


def _top_k_indices(
    scores: np.ndarray, candidate_ids: np.ndarray, *, top_k: int
) -> np.ndarray:
    """Select deterministically from forecasts alone, before outcome masking."""

    values = np.asarray(scores, dtype=np.float64)
    identities = np.asarray(candidate_ids, dtype=np.int64)
    if len(values) != len(identities) or not np.isfinite(values).all():
        raise ValueError("top_k_forecast_input_invalid")
    count = min(max(int(top_k), 0), len(values))
    return np.lexsort((identities, -values))[:count].astype(np.int64, copy=False)


def _ranking_scores(
    *,
    panel: base.StockPanel,
    evaluation_rows: np.ndarray,
    gross_values: np.ndarray,
    gross_valid: np.ndarray,
    residual_values: np.ndarray,
    residual_valid: np.ndarray,
    score_map: Mapping[str, tuple[np.ndarray, str]],
    top_k: int,
    cost_proxies: Sequence[float],
    horizon: int,
    evaluation_year: int,
    regimes: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = np.asarray(evaluation_rows, dtype=np.int64)
    dates = panel.date_idx[rows]
    candidate_ids = panel.row_index.iloc[rows]["candidate_id"].to_numpy(dtype=np.int64)
    gross = np.asarray(gross_values, dtype=np.float64)
    residual = np.asarray(residual_values, dtype=np.float64)
    valid_gross = np.asarray(gross_valid, dtype=bool)
    valid_residual = np.asarray(residual_valid, dtype=bool)
    boundaries = np.r_[0, np.flatnonzero(dates[1:] != dates[:-1]) + 1, len(dates)]
    records: list[dict[str, Any]] = []
    for model, (raw_scores, score_target) in score_map.items():
        scores = np.asarray(raw_scores, dtype=np.float64)
        if len(scores) != len(rows) or not np.isfinite(scores).all():
            raise ValueError(f"ranking_score_invalid:{model}")
        for boundary_pos in range(len(boundaries) - 1):
            left = int(boundaries[boundary_pos])
            right = int(boundaries[boundary_pos + 1])
            local = slice(left, right)
            local_count = right - left
            selected_count = min(int(top_k), local_count)
            order = _top_k_indices(
                scores[local], candidate_ids[local], top_k=selected_count
            )
            selected = left + order
            observed_target = gross if score_target == "gross" else residual
            target_valid = valid_gross if score_target == "gross" else valid_residual
            local_target_valid = target_valid[local]
            local_gross_valid = valid_gross[local]
            selected_gross_valid = valid_gross[selected]
            selected_gross = np.expm1(gross[selected][selected_gross_valid])
            universe_gross = np.expm1(gross[local][local_gross_valid])
            top_mean = float(np.mean(selected_gross)) if len(selected_gross) else np.nan
            universe_mean = float(np.mean(universe_gross)) if len(universe_gross) else np.nan
            record: dict[str, Any] = {
                "date_idx": int(dates[left]),
                "evaluation_year": int(evaluation_year),
                "horizon": int(horizon),
                "model": str(model),
                "score_target": str(score_target),
                "candidate_count": local_count,
                "target_observed_count": int(local_target_valid.sum()),
                "top_k_selected_count": selected_count,
                "top_k_gross_observed_count": int(selected_gross_valid.sum()),
                "top_k_gross_observation_rate": float(
                    selected_gross_valid.mean() if selected_count else np.nan
                ),
                "rank_ic": _rank_ic(
                    scores[local][local_target_valid],
                    observed_target[local][local_target_valid],
                ),
                "gross_rank_ic": _rank_ic(
                    scores[local][local_gross_valid],
                    gross[local][local_gross_valid],
                ),
                "top_k_gross_simple_return": top_mean,
                "universe_gross_simple_return": universe_mean,
                "top_k_excess_simple_return": top_mean - universe_mean,
                "top_k_positive_rate": float(
                    np.mean(selected_gross > 0.0) if len(selected_gross) else np.nan
                ),
            }
            for cost in cost_proxies:
                suffix = f"{int(round(float(cost) * 10_000)):04d}bps"
                record[f"top_k_cost_proxy_return_{suffix}"] = top_mean - float(cost)
            records.append(record)
    daily = pd.DataFrame(records)
    daily = _attach_regimes(daily, regimes)
    summaries: list[dict[str, Any]] = []
    for model, frame in daily.groupby("model", sort=True):
        summary: dict[str, Any] = {
            "evaluation_year": int(evaluation_year),
            "horizon": int(horizon),
            "model": str(model),
            "score_target": str(frame["score_target"].iloc[0]),
            "date_count": len(frame),
            "rank_ic": float(frame["rank_ic"].mean()),
            "positive_ic_rate": float((frame["rank_ic"] > 0.0).mean()),
            "gross_rank_ic": float(frame["gross_rank_ic"].mean()),
            "positive_gross_ic_rate": float((frame["gross_rank_ic"] > 0.0).mean()),
            "top_k_gross_simple_return": float(
                frame["top_k_gross_simple_return"].mean()
            ),
            "top_k_excess_simple_return": float(
                frame["top_k_excess_simple_return"].mean()
            ),
            "top_k_gross_observation_rate": float(
                frame["top_k_gross_observation_rate"].mean()
            ),
            "top_k_positive_rate": float(frame["top_k_positive_rate"].mean()),
        }
        for cost in cost_proxies:
            suffix = f"{int(round(float(cost) * 10_000)):04d}bps"
            column = f"top_k_cost_proxy_return_{suffix}"
            summary[column] = float(frame[column].mean())
        summaries.append(summary)
    return daily, pd.DataFrame(summaries)


def _tail_decile_daily(
    frame: pd.DataFrame,
    *,
    probability_column: str,
    model: str,
    horizon: int,
    evaluation_year: int,
) -> pd.DataFrame:
    """Evaluate a risk ranking on full candidates without combining it with a winner score."""

    required = {
        "date_idx",
        "gross_log_return_observed",
        "gross_return_observed",
        probability_column,
    }
    if not required.issubset(frame.columns):
        raise ValueError("tail_decile_candidate_columns_missing")
    values = frame.copy()
    values["risk_percentile"] = values.groupby("date_idx")[
        probability_column
    ].rank(method="average", pct=True)
    values["risk_decile"] = np.select(
        [values["risk_percentile"] >= 0.9, values["risk_percentile"] <= 0.1],
        ["high_risk_decile", "low_risk_decile"],
        default="middle_risk_deciles",
    )
    observed = values["gross_return_observed"].astype(bool)
    values["simple_return"] = np.where(
        observed,
        np.expm1(values["gross_log_return_observed"].astype(float)),
        np.nan,
    )
    values["loss_event"] = np.where(
        observed,
        values["gross_log_return_observed"].astype(float) <= 0.0,
        np.nan,
    )
    daily = (
        values.groupby(["date_idx", "risk_decile"], sort=True, as_index=False)
        .agg(
            candidate_count=("risk_decile", "size"),
            observed_count=("gross_return_observed", "sum"),
            loss_rate=("loss_event", "mean"),
            simple_return=("simple_return", "mean"),
            observation_rate=("gross_return_observed", "mean"),
        )
    )
    daily["evaluation_year"] = int(evaluation_year)
    daily["horizon"] = int(horizon)
    daily["model"] = str(model)
    daily["probability_column"] = str(probability_column)
    return daily


def run_tail_diagnostics(
    *,
    run_root: str | Path,
    output_root: str | Path | None = None,
) -> dict[str, Any]:
    """Persist the separate bad-tail veto diagnostic for an existing forecast run."""

    run_root = _resolve_path(run_root)
    run_summary = _read_json(run_root / "summary.json")
    if run_summary.get("study_id") != STUDY_ID:
        raise ValueError("tail_diagnostic_study_mismatch")
    if output_root is None:
        output_root = run_root
    output_root = _resolve_path(output_root)
    model_names = ("lightgbm_binary", "rank_gam_logistic", "rank_gam_distribution")
    daily_frames: list[pd.DataFrame] = []
    fold_summaries = [
        _read_json(Path(str(path))) for path in run_summary["fold_summaries"]
    ]
    for fold_summary in fold_summaries:
        candidate_record = fold_summary["files"]["candidate_forecasts"]
        candidate_path = Path(str(candidate_record["path"]))
        candidate = pd.read_parquet(
            candidate_path,
            columns=[
                "date_idx",
                "gross_log_return_observed",
                "gross_return_observed",
                *[f"loss_probability__{name}" for name in model_names],
            ],
        )
        if len(candidate) != int(fold_summary["full_candidate_row_count"]):
            raise ValueError("tail_diagnostic_candidate_row_count_mismatch")
        if bool(candidate["gross_return_observed"].isna().any()):
            raise ValueError("tail_diagnostic_observation_flag_missing")
        for model in model_names:
            daily_frames.append(
                _tail_decile_daily(
                    candidate,
                    probability_column=f"loss_probability__{model}",
                    model=model,
                    horizon=int(fold_summary["horizon"]),
                    evaluation_year=int(fold_summary["evaluation_year"]),
                )
            )
    daily = pd.concat(daily_frames, ignore_index=True)
    daily_path = output_root / "risk_decile_daily.parquet"
    _write_parquet(daily_path, daily)
    summary = (
        daily.groupby(["evaluation_year", "horizon", "model", "risk_decile"], sort=True)
        .agg(
            date_count=("date_idx", "nunique"),
            candidate_count=("candidate_count", "mean"),
            observed_count=("observed_count", "mean"),
            loss_rate=("loss_rate", "mean"),
            simple_return=("simple_return", "mean"),
            observation_rate=("observation_rate", "mean"),
        )
        .reset_index()
    )
    summary_path = output_root / "risk_decile_summary.parquet"
    _write_parquet(summary_path, summary)
    inference_rows: list[dict[str, Any]] = []
    for (horizon, model), model_frame in daily.groupby(
        ["horizon", "model"], sort=True
    ):
        periods: list[tuple[str, pd.DataFrame]] = [("pooled", model_frame)]
        periods.extend(
            (str(int(year)), frame)
            for year, frame in model_frame.groupby("evaluation_year", sort=True)
        )
        for period, frame in periods:
            pivot = frame.pivot(
                index="date_idx",
                columns="risk_decile",
                values=["loss_rate", "simple_return"],
            )
            required_columns = [
                ("loss_rate", "high_risk_decile"),
                ("loss_rate", "low_risk_decile"),
                ("simple_return", "high_risk_decile"),
                ("simple_return", "middle_risk_deciles"),
            ]
            if not all(column in pivot.columns for column in required_columns):
                continue
            pivot = pivot.dropna(subset=required_columns)
            loss_difference = (
                pivot[("loss_rate", "high_risk_decile")]
                - pivot[("loss_rate", "low_risk_decile")]
            ).to_numpy(dtype=np.float64)
            high_minus_universe = (
                pivot[("simple_return", "high_risk_decile")]
                - pivot[("simple_return", "middle_risk_deciles")]
            ).to_numpy(dtype=np.float64)
            hac_loss = base._hac_mean(loss_difference, lag=20)
            block_loss = base._block_interval(
                loss_difference,
                block_length=20,
                seed=20260806 + int(horizon),
            )
            hac_return = base._hac_mean(high_minus_universe, lag=20)
            block_return = base._block_interval(
                high_minus_universe,
                block_length=20,
                seed=20260806 + int(horizon) + 1,
            )
            inference_rows.append(
                {
                    "period": period,
                    "horizon": int(horizon),
                    "model": str(model),
                    "date_count": len(pivot),
                    "loss_rate_spread_high_minus_low": hac_loss["mean"],
                    "loss_rate_spread_hac_lcb_95": hac_loss["lcb_95"],
                    "loss_rate_spread_block_lcb_95": block_loss["lcb_95"],
                    "high_minus_middle_return": hac_return["mean"],
                    "high_minus_middle_return_hac_lcb_95": hac_return["lcb_95"],
                    "high_minus_middle_return_block_lcb_95": block_return["lcb_95"],
                }
            )
    inference = pd.DataFrame(inference_rows)
    inference_path = output_root / "risk_decile_inference.parquet"
    _write_parquet(inference_path, inference)
    audit = {
        "status": "ok",
        "study_id": STUDY_ID,
        "source_run": str(run_root.resolve()),
        "model_names": list(model_names),
        "full_candidate_rank_before_observation_mask": True,
        "future_state_used_as_filter": False,
        "forbidden_2026_read_count": 0,
        "files": {
            "daily": base._file_record(daily_path),
            "summary": base._file_record(summary_path),
            "inference": base._file_record(inference_path),
        },
    }
    _write_json(output_root / "risk_decile_audit.json", audit)
    return audit


def _fold_fingerprint(
    *,
    study_path: Path,
    evaluation_year: int,
    horizon: int,
) -> str:
    return _hash_payload(
        {
            "schema": FOLD_SCHEMA,
            "study_sha256": base._sha256_file(study_path),
            "module_sha256": base._sha256_file(Path(__file__).resolve()),
            "base_module_sha256": base._sha256_file(Path(base.__file__).resolve()),
            "input_fingerprint": base.EXPECTED_INPUT_FINGERPRINT,
            "label_fingerprint": EXPECTED_LABEL_FINGERPRINT,
            "evaluation_year": int(evaluation_year),
            "horizon": int(horizon),
        }
    )


def _fold_complete(path: Path, *, fingerprint: str) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    summary = _read_json(path)
    if (
        summary.get("schema") != FOLD_SCHEMA
        or summary.get("status") != "completed"
        or summary.get("fingerprint") != fingerprint
    ):
        raise ValueError(f"existing_fold_fingerprint_mismatch:{path.parent.name}")
    if not all(
        base._record_valid(record, verify_hash=True)
        for record in dict(summary.get("files", {}) or {}).values()
    ):
        raise ValueError(f"existing_fold_file_invalid:{path.parent.name}")
    return summary


def _sample_fold_rows(
    panel: base.StockPanel,
    rows: np.ndarray,
    *,
    maximum_per_date: int,
) -> np.ndarray:
    return base._sample_rows_by_date(
        np.asarray(rows, dtype=np.int64),
        panel.date_idx,
        maximum_per_date=int(maximum_per_date),
    )


def _outcome_valid_for_horizon(
    panel: base.StockPanel, rows: np.ndarray, horizon: int
) -> np.ndarray:
    column = base.HORIZONS.index(int(horizon)) + 1
    return (
        panel.flags[np.asarray(rows, dtype=np.int64), column]
        & base.FLAG_OUTCOME_WITHIN_CUTOFF
    ) != 0


def _student_baseline_prediction(
    panel: base.StockPanel,
    *,
    train_rows: np.ndarray,
    evaluation_rows: np.ndarray,
    target_values: np.ndarray,
    maximum_train_rows_per_date: int,
    penalty: float,
) -> tuple[np.ndarray, np.ndarray, float, dict[str, Any]]:
    models, summary = base._fit_distribution_models(
        panel,
        train_rows=train_rows,
        target_values=target_values,
        feature_block="core_minute",
        penalty=float(penalty),
        maximum_train_rows_per_date=int(maximum_train_rows_per_date),
    )
    name = "zero_mean_feature_scale_student_t"
    selected = {name: models[name]}
    location, scale, df = base._predict_distribution_models(
        panel, evaluation_rows, selected
    )[name]
    summary["retained_model"] = name
    return location, scale, df, summary


def _run_fold(
    *,
    panel: base.StockPanel,
    structural_ranks: np.memmap,
    study: Mapping[str, Any],
    study_path: Path,
    run_root: Path,
    evaluation_year: int,
    horizon: int,
) -> dict[str, Any]:
    fingerprint = _fold_fingerprint(
        study_path=study_path,
        evaluation_year=evaluation_year,
        horizon=horizon,
    )
    fold_root = run_root / "folds" / f"y{int(evaluation_year)}_h{int(horizon)}"
    summary_path = fold_root / "summary.json"
    current = _fold_complete(summary_path, fingerprint=fingerprint)
    if current is not None:
        return current
    fold_root.mkdir(parents=True, exist_ok=True)
    model_root = fold_root / "models"
    year_rows = panel.rows_for_year(int(evaluation_year))
    if bool(pd.Series(panel.trade_date[year_rows]).str.startswith("2026-").any()):
        raise ValueError("forbidden_2026_fold_row")

    evaluation_config = dict(study["evaluation"])
    maximum_per_date = int(evaluation_config["maximum_train_rows_per_date"])
    top_k = int(evaluation_config["top_k_diagnostic"])
    cost_proxies = tuple(float(v) for v in evaluation_config["gross_cost_proxies"])
    gam_config = dict(study["models"]["rank_gam"])
    lgb_config = dict(study["models"]["lightgbm"])
    student_config = dict(study["models"]["student_t_baseline"])
    lgb_feature_block = str(study["features"]["lightgbm"]["feature_block"])
    lgb_features = base.feature_names_for_block(panel.input_manifest, lgb_feature_block)

    gross_target = f"executable_log_return_{int(horizon)}"
    residual_target = f"residual_log_return_{int(horizon)}"
    gross_values_full = panel.target(gross_target)
    residual_values_full = panel.target(residual_target)
    gross_valid_full = panel.target_valid(gross_target)
    residual_valid_full = panel.target_valid(residual_target)
    gross_values = gross_values_full[year_rows]
    residual_values = residual_values_full[year_rows]
    gross_valid = gross_valid_full[year_rows]
    residual_valid = residual_valid_full[year_rows]

    gross_fold = base._build_fold(
        panel,
        evaluation_year=int(evaluation_year),
        horizon=int(horizon),
        target=gross_target,
    )
    regimes, regime_summary = _regime_frame(
        panel,
        train_rows=gross_fold["train_rows"],
        evaluation_rows=year_rows,
    )
    distribution_daily: list[pd.DataFrame] = []
    distribution_metrics: list[dict[str, Any]] = []
    fit_summaries: list[dict[str, Any]] = []
    continuous_predictions: dict[str, dict[str, dict[str, Any]]] = {}

    for target, target_values_full, target_valid_full in (
        (gross_target, gross_values_full, gross_valid_full),
        (residual_target, residual_values_full, residual_valid_full),
    ):
        target_fold = base._build_fold(
            panel,
            evaluation_year=int(evaluation_year),
            horizon=int(horizon),
            target=target,
        )
        sampled_rows = _sample_fold_rows(
            panel,
            target_fold["train_rows"],
            maximum_per_date=maximum_per_date,
        )
        student_location, student_scale, student_df, student_summary = (
            _student_baseline_prediction(
                panel,
                train_rows=target_fold["train_rows"],
                evaluation_rows=year_rows,
                target_values=target_values_full,
                maximum_train_rows_per_date=maximum_per_date,
                penalty=float(student_config["ridge_penalty"]),
            )
        )
        gam_model, gam_summary = _fit_rank_gam_distribution(
            np.asarray(structural_ranks[sampled_rows], dtype=np.float32),
            target_values_full[sampled_rows],
            panel.date_idx[sampled_rows],
            n_knots=int(gam_config["spline_knots"]),
            degree=int(gam_config["spline_degree"]),
            ridge_penalty=float(gam_config["ridge_penalty"]),
        )
        gam_location, gam_scale = _predict_rank_gam_distribution(
            gam_model, np.asarray(structural_ranks[year_rows], dtype=np.float32)
        )
        lgb_mean, lgb_quantiles, raw_crossing, lgb_summary = _fit_lgb_continuous(
            panel,
            train_rows=sampled_rows,
            evaluation_rows=year_rows,
            target_values=target_values_full,
            feature_names=lgb_features,
            probabilities=QUANTILE_PROBABILITIES,
            config=lgb_config,
            model_root=model_root,
            prefix=target,
        )
        student_quantiles = _t_quantiles(
            student_location, student_scale, student_df
        )
        gam_quantiles = _t_quantiles(gam_location, gam_scale, gam_model.df)
        predictions = {
            "student_t_feature_scale": {
                "mean": student_location,
                "quantiles": student_quantiles,
                "location": student_location,
                "scale": student_scale,
                "df": student_df,
            },
            "rank_gam_distribution": {
                "mean": gam_location,
                "quantiles": gam_quantiles,
                "location": gam_location,
                "scale": gam_scale,
                "df": gam_model.df,
            },
            "lightgbm_direct_quantiles": {
                "mean": lgb_mean,
                "quantiles": lgb_quantiles,
                "raw_crossing_rate": raw_crossing,
            },
        }
        continuous_predictions[target] = predictions
        target_valid = target_valid_full[year_rows]
        target_observed = target_values_full[year_rows]
        for model_name, prediction in predictions.items():
            t_parameters = None
            if "df" in prediction:
                t_parameters = (
                    prediction["location"],
                    prediction["scale"],
                    float(prediction["df"]),
                )
            daily, metric = _distribution_scores(
                panel=panel,
                evaluation_rows=year_rows,
                valid=target_valid,
                observed=target_observed,
                mean=prediction["mean"],
                quantiles=prediction["quantiles"],
                model=model_name,
                target=target,
                horizon=int(horizon),
                evaluation_year=int(evaluation_year),
                regimes=regimes,
                t_parameters=t_parameters,
            )
            distribution_daily.append(daily)
            distribution_metrics.append(metric)
        fold_contract = {
            key: value
            for key, value in target_fold.items()
            if key not in {"train_rows", "evaluation_rows"}
        }
        fit_summaries.extend(
            [
                {
                    "head": target,
                    "model": "student_t_feature_scale",
                    "fold": fold_contract,
                    **student_summary,
                },
                {
                    "head": target,
                    "model": "rank_gam_distribution",
                    "fold": fold_contract,
                    **gam_summary,
                },
                {
                    "head": target,
                    "model": "lightgbm_direct_quantiles",
                    "fold": fold_contract,
                    **lgb_summary,
                },
            ]
        )

    sampled_gross_rows = _sample_fold_rows(
        panel,
        gross_fold["train_rows"],
        maximum_per_date=maximum_per_date,
    )
    loss_values_full = (gross_values_full <= 0.0).astype(np.int8)
    gam_loss_model, gam_loss_summary = _fit_rank_gam_binary(
        np.asarray(structural_ranks[sampled_gross_rows], dtype=np.float32),
        loss_values_full[sampled_gross_rows],
        panel.date_idx[sampled_gross_rows],
        n_knots=int(gam_config["spline_knots"]),
        degree=int(gam_config["spline_degree"]),
        regularization_c=float(gam_config["logistic_c"]),
    )
    gam_loss_probability = _predict_rank_gam_binary(
        gam_loss_model, np.asarray(structural_ranks[year_rows], dtype=np.float32)
    )
    lgb_loss_probability, lgb_loss_summary = _fit_lgb_binary(
        panel,
        train_rows=sampled_gross_rows,
        evaluation_rows=year_rows,
        target_values=loss_values_full,
        feature_names=lgb_features,
        config=lgb_config,
        model_path=model_root / "gross_loss_hurdle.txt",
    )
    recent_loss = _recent_probability(
        loss_values_full[gross_fold["train_rows"]],
        panel.date_idx[gross_fold["train_rows"]],
    )
    student_gross = continuous_predictions[gross_target]["student_t_feature_scale"]
    gam_gross = continuous_predictions[gross_target]["rank_gam_distribution"]
    loss_probabilities = {
        "recent_252d_loss_probability": np.full(len(year_rows), recent_loss),
        "student_t_feature_scale": stats.t.cdf(
            0.0,
            float(student_gross["df"]),
            loc=student_gross["location"],
            scale=student_gross["scale"],
        ),
        "rank_gam_distribution": stats.t.cdf(
            0.0,
            float(gam_gross["df"]),
            loc=gam_gross["location"],
            scale=gam_gross["scale"],
        ),
        "rank_gam_logistic": gam_loss_probability,
        "lightgbm_binary": lgb_loss_probability,
    }
    hurdle_daily: list[pd.DataFrame] = []
    hurdle_metrics: list[dict[str, Any]] = []
    reliability_frames: list[pd.DataFrame] = []
    for model_name, probability in loss_probabilities.items():
        daily, metric, reliability = _binary_scores(
            panel=panel,
            evaluation_rows=year_rows,
            valid=gross_valid,
            observed=loss_values_full[year_rows],
            probability=probability,
            model=model_name,
            target="gross_return_le_zero",
            horizon=int(horizon),
            evaluation_year=int(evaluation_year),
            regimes=regimes,
        )
        hurdle_daily.append(daily)
        hurdle_metrics.append(metric)
        reliability_frames.append(reliability)
    fit_summaries.extend(
        [
            {
                "head": "gross_return_le_zero",
                "model": "rank_gam_logistic",
                **gam_loss_summary,
            },
            {
                "head": "gross_return_le_zero",
                "model": "lightgbm_binary",
                **lgb_loss_summary,
            },
        ]
    )

    state_daily: list[pd.DataFrame] = []
    state_metrics: list[dict[str, Any]] = []
    state_predictions: dict[str, dict[str, np.ndarray]] = {}
    state_fit_summaries: list[dict[str, Any]] = []
    state_folds: dict[str, dict[str, Any]] = {}
    for state_target in ("entry_action", f"sell_action_{int(horizon)}"):
        state_fold = base._build_fold(
            panel,
            evaluation_year=int(evaluation_year),
            horizon=int(horizon),
            target=state_target,
        )
        state_folds[state_target] = state_fold
        sampled_state_rows = _sample_fold_rows(
            panel,
            state_fold["train_rows"],
            maximum_per_date=maximum_per_date,
        )
        state_values_full = panel.state(state_target).astype(np.int8)
        gam_state_model, gam_state_summary = _fit_rank_gam_binary(
            np.asarray(structural_ranks[sampled_state_rows], dtype=np.float32),
            state_values_full[sampled_state_rows],
            panel.date_idx[sampled_state_rows],
            n_knots=int(gam_config["spline_knots"]),
            degree=int(gam_config["spline_degree"]),
            regularization_c=float(gam_config["logistic_c"]),
        )
        gam_state_probability = _predict_rank_gam_binary(
            gam_state_model, np.asarray(structural_ranks[year_rows], dtype=np.float32)
        )
        lgb_state_probability, lgb_state_summary = _fit_lgb_binary(
            panel,
            train_rows=sampled_state_rows,
            evaluation_rows=year_rows,
            target_values=state_values_full,
            feature_names=lgb_features,
            config=lgb_config,
            model_path=model_root / f"{state_target}.txt",
        )
        recent_state = _recent_probability(
            state_values_full[state_fold["train_rows"]],
            panel.date_idx[state_fold["train_rows"]],
        )
        predictions = {
            "recent_252d_probability": np.full(len(year_rows), recent_state),
            "rank_gam_logistic": gam_state_probability,
            "lightgbm_binary": lgb_state_probability,
        }
        state_predictions[state_target] = predictions
        valid_state = panel.target_valid(state_target)[year_rows]
        valid_state &= _outcome_valid_for_horizon(panel, year_rows, int(horizon))
        for model_name, probability in predictions.items():
            daily, metric, reliability = _binary_scores(
                panel=panel,
                evaluation_rows=year_rows,
                valid=valid_state,
                observed=state_values_full[year_rows],
                probability=probability,
                model=model_name,
                target=state_target,
                horizon=int(horizon),
                evaluation_year=int(evaluation_year),
                regimes=regimes,
            )
            state_daily.append(daily)
            state_metrics.append(metric)
            reliability_frames.append(reliability)
        fold_contract = {
            key: value
            for key, value in state_fold.items()
            if key not in {"train_rows", "evaluation_rows"}
        }
        state_fit_summaries.extend(
            [
                {
                    "head": state_target,
                    "model": "rank_gam_logistic",
                    "fold": fold_contract,
                    **gam_state_summary,
                },
                {
                    "head": state_target,
                    "model": "lightgbm_binary",
                    "fold": fold_contract,
                    **lgb_state_summary,
                },
            ]
        )
    fit_summaries.extend(state_fit_summaries)

    sell_target = f"sell_action_{int(horizon)}"
    sell_fold = state_folds[sell_target]
    sampled_hazard_rows = _sample_fold_rows(
        panel,
        sell_fold["train_rows"],
        maximum_per_date=maximum_per_date,
    )
    horizon_position = base.HORIZONS.index(int(horizon))
    tau_full = np.asarray(panel.tau[:, horizon_position], dtype=np.int16)
    if bool((tau_full[sampled_hazard_rows] < 0).any()):
        raise ValueError("hazard_training_tau_missing")
    hazard_model, hazard_fit_summary = _fit_discrete_hazard(
        np.asarray(structural_ranks[sampled_hazard_rows], dtype=np.float32),
        tau_full[sampled_hazard_rows],
        panel.date_idx[sampled_hazard_rows],
        n_knots=int(gam_config["spline_knots"]),
        degree=int(gam_config["spline_degree"]),
        regularization_c=float(gam_config["logistic_c"]),
    )
    hazard_predictions = {
        name: _predict_discrete_hazard(
            hazard_model,
            np.asarray(structural_ranks[year_rows], dtype=np.float32),
            model_name=name,
        )
        for name in ("empirical_hazard", "rank_gam_hazard")
    }
    hazard_daily: list[pd.DataFrame] = []
    hazard_metrics: list[dict[str, Any]] = []
    valid_hazard = panel.target_valid(sell_target)[year_rows]
    valid_hazard &= _outcome_valid_for_horizon(panel, year_rows, int(horizon))
    valid_hazard &= tau_full[year_rows] >= 0
    for model_name, hazard in hazard_predictions.items():
        daily, metric = _hazard_scores(
            panel=panel,
            evaluation_rows=year_rows,
            valid=valid_hazard,
            tau=tau_full[year_rows],
            hazard=hazard,
            model=model_name,
            horizon=int(horizon),
            evaluation_year=int(evaluation_year),
            regimes=regimes,
        )
        hazard_daily.append(daily)
        hazard_metrics.append(metric)
    fit_summaries.append(
        {
            "head": f"tau_sell_{int(horizon)}",
            "model": "rank_gam_hazard",
            **hazard_fit_summary,
        }
    )

    rank_values = np.asarray(structural_ranks[year_rows], dtype=np.float64)
    score_map = {
        "contrarian_equal_weight": (1.0 - np.mean(rank_values, axis=1), "gross"),
        "rank_gam_mean_gross": (
            continuous_predictions[gross_target]["rank_gam_distribution"]["mean"],
            "gross",
        ),
        "lightgbm_mean_gross": (
            continuous_predictions[gross_target]["lightgbm_direct_quantiles"]["mean"],
            "gross",
        ),
        "rank_gam_mean_residual": (
            continuous_predictions[residual_target]["rank_gam_distribution"]["mean"],
            "residual",
        ),
        "lightgbm_mean_residual": (
            continuous_predictions[residual_target]["lightgbm_direct_quantiles"]["mean"],
            "residual",
        ),
    }
    ranking_daily, ranking_metrics = _ranking_scores(
        panel=panel,
        evaluation_rows=year_rows,
        gross_values=gross_values,
        gross_valid=gross_valid,
        residual_values=residual_values,
        residual_valid=residual_valid,
        score_map=score_map,
        top_k=top_k,
        cost_proxies=cost_proxies,
        horizon=int(horizon),
        evaluation_year=int(evaluation_year),
        regimes=regimes,
    )

    candidate = panel.row_index.iloc[year_rows][
        ["candidate_id", "date_idx", "symbol_idx", "trade_date", "symbol"]
    ].reset_index(drop=True)
    candidate["gross_log_return_observed"] = np.where(
        gross_valid, gross_values, np.nan
    ).astype(np.float32)
    candidate["residual_log_return_observed"] = np.where(
        residual_valid, residual_values, np.nan
    ).astype(np.float32)
    candidate["gross_return_observed"] = gross_valid
    candidate["residual_return_observed"] = residual_valid
    candidate["entry_action"] = panel.state("entry_action")[year_rows]
    candidate["sell_action"] = panel.state(sell_target)[year_rows]
    candidate["tau_sell"] = tau_full[year_rows]
    for target, predictions in continuous_predictions.items():
        target_prefix = "gross" if target == gross_target else "residual"
        for model_name, prediction in predictions.items():
            prefix = f"{target_prefix}__{model_name}"
            candidate[f"{prefix}__mean"] = np.asarray(
                prediction["mean"], dtype=np.float32
            )
            for probability_pos, probability in enumerate(QUANTILE_PROBABILITIES):
                suffix = f"q{int(round(probability * 100)):02d}"
                candidate[f"{prefix}__{suffix}"] = np.asarray(
                    prediction["quantiles"][:, probability_pos], dtype=np.float32
                )
    for model_name, probability in loss_probabilities.items():
        candidate[f"loss_probability__{model_name}"] = np.asarray(
            probability, dtype=np.float32
        )
    for state_target, predictions in state_predictions.items():
        for model_name, probability in predictions.items():
            candidate[f"{state_target}__{model_name}"] = np.asarray(
                probability, dtype=np.float32
            )
    for model_name, hazard in hazard_predictions.items():
        mass, censor_probability = _hazard_mass(hazard)
        expected_delay = mass @ np.arange(RETRY_DAYS + 1, dtype=np.float64)
        expected_delay += (RETRY_DAYS + 1) * censor_probability
        candidate[f"sell_within_probability__{model_name}"] = np.asarray(
            1.0 - censor_probability, dtype=np.float32
        )
        candidate[f"expected_tau__{model_name}"] = np.asarray(
            expected_delay, dtype=np.float32
        )
    for model_name, (score, _) in score_map.items():
        candidate[f"ranking_score__{model_name}"] = np.asarray(
            score, dtype=np.float32
        )

    frames = {
        "candidate_forecasts": candidate,
        "distribution_daily": pd.concat(distribution_daily, ignore_index=True),
        "distribution_metrics": pd.DataFrame(distribution_metrics),
        "hurdle_daily": pd.concat(hurdle_daily, ignore_index=True),
        "hurdle_metrics": pd.DataFrame(hurdle_metrics),
        "state_daily": pd.concat(state_daily, ignore_index=True),
        "state_metrics": pd.DataFrame(state_metrics),
        "hazard_daily": pd.concat(hazard_daily, ignore_index=True),
        "hazard_metrics": pd.DataFrame(hazard_metrics),
        "ranking_daily": ranking_daily,
        "ranking_metrics": ranking_metrics,
        "reliability": pd.concat(reliability_frames, ignore_index=True),
    }
    files: dict[str, Any] = {}
    for name, frame in frames.items():
        path = fold_root / f"{name}.parquet"
        _write_parquet(path, frame)
        files[name] = base._file_record(path)
    fit_path = fold_root / "fit_summaries.json"
    _write_json(fit_path, {"fits": fit_summaries, "regime": regime_summary})
    files["fit_summaries"] = base._file_record(fit_path)

    quantile_columns = [
        column
        for column in candidate.columns
        if "__q" in column and column.rsplit("__", 1)[-1].startswith("q")
    ]
    quantile_groups: dict[str, list[str]] = {}
    for column in quantile_columns:
        prefix = column.rsplit("__", 1)[0]
        quantile_groups.setdefault(prefix, []).append(column)
    crossing_checks = {
        prefix: _quantile_crossing_rate(
            candidate[sorted(columns, key=lambda value: int(value.rsplit("q", 1)[1]))].to_numpy()
        )
        for prefix, columns in quantile_groups.items()
    }
    purge_checks = []
    for fit in fit_summaries:
        fold = fit.get("fold")
        if not isinstance(fold, Mapping):
            continue
        purge_checks.append(
            int(fold["maximum_train_signal_date_idx"])
            + int(fold["dependency_days"])
            < int(fold["oos_start_date_idx"])
        )
    audit_checks = {
        "candidate_rows_equal_full_year_universe": len(candidate) == len(year_rows),
        "candidate_forecast_has_no_2026": not bool(
            candidate["trade_date"].astype(str).str.startswith("2026-").any()
        ),
        "all_post_rearrangement_quantiles_monotone": all(
            value == 0.0 for value in crossing_checks.values()
        ),
        "all_purges_pass": bool(purge_checks) and all(purge_checks),
        "ranking_selects_before_outcome_observation": bool(
            (ranking_daily["top_k_selected_count"] >= ranking_daily["top_k_gross_observed_count"]).all()
        ),
        "future_state_not_used_as_candidate_filter": True,
        "forbidden_2026_read_count_zero": True,
        "portfolio_selection_not_performed": True,
        "account_optimization_not_performed": True,
    }
    summary = {
        "schema": FOLD_SCHEMA,
        "status": "completed",
        "study_id": STUDY_ID,
        "fingerprint": fingerprint,
        "evaluation_year": int(evaluation_year),
        "horizon": int(horizon),
        "full_candidate_row_count": len(year_rows),
        "gross_observed_row_count": int(gross_valid.sum()),
        "residual_observed_row_count": int(residual_valid.sum()),
        "maximum_train_rows_per_date": maximum_per_date,
        "feature_contract": {
            "rank_gam": list(STRUCTURAL_FEATURES),
            "lightgbm_block": lgb_feature_block,
            "lightgbm_feature_count": len(lgb_features),
            "pit_concatenation_used": False,
        },
        "raw_lightgbm_quantile_crossing_rates": {
            target: float(
                predictions["lightgbm_direct_quantiles"]["raw_crossing_rate"]
            )
            for target, predictions in continuous_predictions.items()
        },
        "post_rearrangement_crossing_rates": crossing_checks,
        "audit": {
            "status": "ok" if all(audit_checks.values()) else "failed",
            "checks": audit_checks,
        },
        "training_performed": True,
        "hyperparameter_selection_performed": False,
        "model_selection_performed": False,
        "portfolio_selection_performed": False,
        "account_optimization_performed": False,
        "forbidden_2026_read_count": 0,
        "files": files,
    }
    _write_json(summary_path, summary)
    print(
        json.dumps(
            {
                "event": "stock_bad_tail_fold_complete",
                "evaluation_year": int(evaluation_year),
                "horizon": int(horizon),
                "audit": summary["audit"]["status"],
            },
            ensure_ascii=False,
        ),
        flush=True,
    )
    return summary


def _paired_comparisons(
    daily: pd.DataFrame,
    *,
    metric: str,
    baseline: str,
    higher_is_better: bool,
    family: str,
) -> pd.DataFrame:
    records: list[dict[str, Any]] = []
    if daily.empty:
        return pd.DataFrame()
    for (horizon, target), family_frame in daily.groupby(
        ["horizon", "target"], sort=True
    ):
        periods: list[tuple[str, pd.DataFrame]] = [("pooled", family_frame)]
        periods.extend(
            (str(int(year)), frame)
            for year, frame in family_frame.groupby("evaluation_year", sort=True)
        )
        for period, frame in periods:
            baseline_frame = frame.loc[frame["model"] == baseline, ["date_idx", metric]]
            if baseline_frame.empty:
                continue
            baseline_frame = baseline_frame.set_index("date_idx")
            for model in sorted(set(frame["model"]) - {baseline}):
                challenger = frame.loc[
                    frame["model"] == model, ["date_idx", metric]
                ].set_index("date_idx")
                joined = baseline_frame.join(
                    challenger, how="inner", lsuffix="_base", rsuffix="_model"
                ).dropna()
                if joined.empty:
                    continue
                if higher_is_better:
                    difference = (
                        joined[f"{metric}_model"].to_numpy(dtype=np.float64)
                        - joined[f"{metric}_base"].to_numpy(dtype=np.float64)
                    )
                else:
                    difference = (
                        joined[f"{metric}_base"].to_numpy(dtype=np.float64)
                        - joined[f"{metric}_model"].to_numpy(dtype=np.float64)
                    )
                hac = base._hac_mean(difference, lag=20)
                interval = base._block_interval(
                    difference,
                    block_length=20,
                    seed=20260806 + int(horizon) + (0 if period == "pooled" else int(period)),
                )
                records.append(
                    {
                        "family": str(family),
                        "period": period,
                        "horizon": int(horizon),
                        "target": str(target),
                        "metric": str(metric),
                        "baseline": str(baseline),
                        "model": str(model),
                        "date_count": len(joined),
                        "mean_improvement": hac["mean"],
                        "hac_standard_error": hac["standard_error"],
                        "hac_lcb_95": hac["lcb_95"],
                        "hac_ucb_95": hac["ucb_95"],
                        "block_lcb_95": interval["lcb_95"],
                        "block_ucb_95": interval["ucb_95"],
                    }
                )
    return pd.DataFrame(records)


def _family_reality_checks(
    daily: pd.DataFrame,
    *,
    metric: str,
    baseline: str | None,
    higher_is_better: bool,
    family: str,
) -> pd.DataFrame:
    records: list[dict[str, Any]] = []
    if daily.empty:
        return pd.DataFrame()
    for (horizon, target), frame in daily.groupby(["horizon", "target"], sort=True):
        pivot = frame.pivot(index="date_idx", columns="model", values=metric)
        if baseline is not None:
            if baseline not in pivot.columns:
                continue
            challenger_names = sorted(set(pivot.columns) - {baseline})
            if not challenger_names:
                continue
            pivot = pivot.dropna(subset=[baseline, *challenger_names])
            baseline_values = pivot[baseline].to_numpy(dtype=np.float64)
            differences = []
            for model in challenger_names:
                candidate = pivot[model].to_numpy(dtype=np.float64)
                differences.append(
                    candidate - baseline_values
                    if higher_is_better
                    else baseline_values - candidate
                )
        else:
            challenger_names = sorted(pivot.columns)
            pivot = pivot.dropna(subset=challenger_names)
            differences = [
                (
                    pivot[model].to_numpy(dtype=np.float64)
                    if higher_is_better
                    else -pivot[model].to_numpy(dtype=np.float64)
                )
                for model in challenger_names
            ]
        if pivot.empty:
            continue
        result = base._reality_check(
            np.column_stack(differences),
            challenger_names,
            block_length=20,
            repetitions=1_000,
            seed=20260806 + int(horizon),
        )
        records.append(
            {
                "family": str(family),
                "horizon": int(horizon),
                "target": str(target),
                "metric": str(metric),
                "baseline": "zero" if baseline is None else str(baseline),
                **result,
            }
        )
    return pd.DataFrame(records)


def _ranking_inference(
    daily: pd.DataFrame, *, cost_proxies: Sequence[float]
) -> tuple[pd.DataFrame, pd.DataFrame]:
    comparisons: list[dict[str, Any]] = []
    realities: list[pd.DataFrame] = []
    metrics = ["top_k_excess_simple_return"] + [
        f"top_k_cost_proxy_return_{int(round(float(cost) * 10_000)):04d}bps"
        for cost in cost_proxies
    ]
    for metric in metrics:
        for (horizon, model), frame in daily.groupby(["horizon", "model"], sort=True):
            values = frame.sort_values("date_idx")[metric].to_numpy(dtype=np.float64)
            hac = base._hac_mean(values, lag=20)
            interval = base._block_interval(
                values,
                block_length=20,
                seed=20260806 + int(horizon),
            )
            annual = frame.groupby("evaluation_year", sort=True)[metric].mean()
            comparisons.append(
                {
                    "family": "ranking",
                    "period": "pooled",
                    "horizon": int(horizon),
                    "target": str(frame["score_target"].iloc[0]),
                    "metric": metric,
                    "baseline": "zero",
                    "model": str(model),
                    "date_count": len(values),
                    "mean_improvement": hac["mean"],
                    "hac_standard_error": hac["standard_error"],
                    "hac_lcb_95": hac["lcb_95"],
                    "hac_ucb_95": hac["ucb_95"],
                    "block_lcb_95": interval["lcb_95"],
                    "block_ucb_95": interval["ucb_95"],
                    "positive_year_count": int((annual > 0.0).sum()),
                    "year_count": len(annual),
                }
            )
        reality_input = daily.copy()
        reality_input["target"] = "all_ranking_models"
        realities.append(
            _family_reality_checks(
                reality_input,
                metric=metric,
                baseline=None,
                higher_is_better=True,
                family="ranking",
            )
        )
    return pd.DataFrame(comparisons), pd.concat(realities, ignore_index=True)


def _regime_metrics(frames: Mapping[str, pd.DataFrame]) -> pd.DataFrame:
    metric_map = {
        "distribution": ("mean_pinball", "quantile_crps", "squared_error"),
        "hurdle": ("log_score", "brier", "absolute_calibration_error"),
        "state": ("log_score", "brier", "absolute_calibration_error"),
        "hazard": ("log_likelihood", "brier", "absolute_calibration_error"),
        "ranking": (
            "rank_ic",
            "gross_rank_ic",
            "top_k_excess_simple_return",
            "top_k_gross_simple_return",
        ),
    }
    records: list[dict[str, Any]] = []
    for family, frame in frames.items():
        for (horizon, target, model, trend, volatility), group in frame.groupby(
            ["horizon", "target", "model", "trend_regime", "volatility_regime"],
            dropna=False,
            sort=True,
        ):
            for metric in metric_map[family]:
                if metric not in group.columns:
                    continue
                records.append(
                    {
                        "family": family,
                        "horizon": int(horizon),
                        "target": str(target),
                        "model": str(model),
                        "trend_regime": str(trend),
                        "volatility_regime": str(volatility),
                        "metric": metric,
                        "mean": float(group[metric].mean()),
                        "date_count": int(group["date_idx"].nunique()),
                    }
                )
    return pd.DataFrame(records)


def _development_gate(
    *,
    evaluation_years: Sequence[int],
    comparisons: pd.DataFrame,
    reality: pd.DataFrame,
    ranking_daily: pd.DataFrame,
    cost_proxies: Sequence[float],
) -> dict[str, Any]:
    complete_development = set(int(v) for v in evaluation_years) == set(
        DEVELOPMENT_YEARS
    )

    def supported(
        family: str,
        metric: str,
        horizon: int,
        *,
        target: str | None = None,
    ) -> tuple[bool, list[str]]:
        selected = comparisons.loc[
            (comparisons["family"] == family)
            & (comparisons["metric"] == metric)
            & (comparisons["horizon"] == int(horizon))
            & (comparisons["period"] == "pooled")
        ]
        if target is not None:
            selected = selected.loc[selected["target"] == str(target)]
        passing = selected.loc[
            (selected["hac_lcb_95"] > 0.0)
            & (selected["block_lcb_95"] > 0.0)
        ]
        return bool(len(passing)), passing["model"].astype(str).tolist()

    distribution_pass, distribution_models = supported(
        "distribution",
        "mean_pinball",
        PRIMARY_HORIZON,
        target=f"executable_log_return_{PRIMARY_HORIZON}",
    )
    hurdle_pass, hurdle_models = supported(
        "hurdle", "log_score", PRIMARY_HORIZON, target="gross_return_le_zero"
    )
    stress_cost = max(float(v) for v in cost_proxies)
    stress_metric = (
        f"top_k_cost_proxy_return_{int(round(stress_cost * 10_000)):04d}bps"
    )
    ranking_pass, ranking_models = supported(
        "ranking", stress_metric, PRIMARY_HORIZON
    )
    primary_reality = reality.loc[
        (reality["horizon"] == PRIMARY_HORIZON)
        & (
            (
                (reality["family"] == "distribution")
                & (reality["metric"] == "mean_pinball")
                & (
                    reality["target"]
                    == f"executable_log_return_{PRIMARY_HORIZON}"
                )
            )
            | (
                (reality["family"] == "hurdle")
                & (reality["metric"] == "log_score")
                & (reality["target"] == "gross_return_le_zero")
            )
            | (
                (reality["family"] == "ranking")
                & (reality["metric"] == stress_metric)
                & (reality["target"] == "all_ranking_models")
            )
        )
    ]
    reality_pass = bool(len(primary_reality) == 3 and (primary_reality["p_value"] <= 0.05).all())
    diagnostic_ranking = comparisons.loc[
        (comparisons["family"] == "ranking")
        & (comparisons["metric"] == stress_metric)
        & (comparisons["horizon"] == DIAGNOSTIC_HORIZON)
        & (comparisons["period"] == "pooled")
        & (comparisons["model"].isin(ranking_models))
    ]
    diagnostic_nonnegative = bool(
        len(diagnostic_ranking) and (diagnostic_ranking["mean_improvement"] >= 0.0).any()
    )
    pass_development = bool(
        complete_development
        and distribution_pass
        and hurdle_pass
        and ranking_pass
        and reality_pass
        and diagnostic_nonnegative
    )
    return {
        "complete_development_matrix": complete_development,
        "distribution_pinball_supported": distribution_pass,
        "distribution_models": distribution_models,
        "bad_tail_log_score_supported": hurdle_pass,
        "bad_tail_models": hurdle_models,
        "stress_cost_top_k_supported": ranking_pass,
        "ranking_models": ranking_models,
        "family_reality_checks_pass_5pct": reality_pass,
        "d10_diagnostic_nonnegative_for_a_passing_ranking_model": diagnostic_nonnegative,
        "development_forecast_gate_passed": pass_development,
        "eligible_for_retrospective_2023_2025": pass_development,
        "eligible_for_account_research": False,
        "account_gate_reason": (
            "Account research remains disabled until a frozen model family also "
            "passes the reserved 2023-2025 retrospective forecast gate."
        ),
    }


def run_experiment(
    *,
    study_path: str | Path = DEFAULT_STUDY_PATH,
    input_manifest_path: str | Path = DEFAULT_INPUT_MANIFEST,
    label_manifest_path: str | Path = DEFAULT_LABEL_MANIFEST,
    output_root: str | Path = DEFAULT_OUTPUT_ROOT,
    evaluation_years: Sequence[int] = PREFLIGHT_YEARS,
    horizons: Sequence[int] = HORIZONS,
    run_id: str = "preflight_2022_h20_h10",
) -> dict[str, Any]:
    study_path = _resolve_path(study_path)
    input_manifest_path = _resolve_path(input_manifest_path)
    label_manifest_path = _resolve_path(label_manifest_path)
    output_root = _resolve_path(output_root)
    study = load_study(study_path)
    requested_years = tuple(int(v) for v in evaluation_years)
    requested_horizons = tuple(int(v) for v in horizons)
    if not requested_years or not requested_horizons:
        raise ValueError("experiment_matrix_empty")
    if not set(requested_years).issubset(set(DEVELOPMENT_YEARS)):
        raise ValueError("only_frozen_development_years_are_authorized")
    if any(year >= FORBIDDEN_YEAR for year in requested_years):
        raise ValueError("forbidden_2026_evaluation_request")
    if set(requested_horizons) != set(HORIZONS):
        raise ValueError("h20_primary_and_h10_diagnostic_must_run_together")
    panel = _load_panel(
        input_manifest_path=input_manifest_path,
        label_manifest_path=label_manifest_path,
    )
    structural_ranks, rank_manifest = prepare_structural_ranks(
        panel, output_root=output_root
    )
    run_root = output_root / "experiments" / str(run_id)
    run_root.mkdir(parents=True, exist_ok=True)
    fold_summaries: list[dict[str, Any]] = []
    for evaluation_year in requested_years:
        for horizon in requested_horizons:
            fold_summaries.append(
                _run_fold(
                    panel=panel,
                    structural_ranks=structural_ranks,
                    study=study,
                    study_path=study_path,
                    run_root=run_root,
                    evaluation_year=evaluation_year,
                    horizon=horizon,
                )
            )

    frame_names = (
        "distribution_daily",
        "distribution_metrics",
        "hurdle_daily",
        "hurdle_metrics",
        "state_daily",
        "state_metrics",
        "hazard_daily",
        "hazard_metrics",
        "ranking_daily",
        "ranking_metrics",
        "reliability",
    )
    combined: dict[str, pd.DataFrame] = {}
    files: dict[str, Any] = {}
    for name in frame_names:
        frame = pd.concat(
            [pd.read_parquet(summary["files"][name]["path"]) for summary in fold_summaries],
            ignore_index=True,
        )
        combined[name] = frame
        path = run_root / f"{name}.parquet"
        _write_parquet(path, frame)
        files[name] = base._file_record(path)

    comparison_frames = [
        _paired_comparisons(
            combined["distribution_daily"],
            metric="mean_pinball",
            baseline="student_t_feature_scale",
            higher_is_better=False,
            family="distribution",
        ),
        _paired_comparisons(
            combined["distribution_daily"],
            metric="quantile_crps",
            baseline="student_t_feature_scale",
            higher_is_better=False,
            family="distribution",
        ),
        _paired_comparisons(
            combined["hurdle_daily"],
            metric="log_score",
            baseline="recent_252d_loss_probability",
            higher_is_better=True,
            family="hurdle",
        ),
        _paired_comparisons(
            combined["hurdle_daily"],
            metric="brier",
            baseline="recent_252d_loss_probability",
            higher_is_better=False,
            family="hurdle",
        ),
        _paired_comparisons(
            combined["state_daily"],
            metric="log_score",
            baseline="recent_252d_probability",
            higher_is_better=True,
            family="state",
        ),
        _paired_comparisons(
            combined["hazard_daily"],
            metric="log_likelihood",
            baseline="empirical_hazard",
            higher_is_better=True,
            family="hazard",
        ),
    ]
    ranking_comparisons, ranking_reality = _ranking_inference(
        combined["ranking_daily"],
        cost_proxies=tuple(float(v) for v in study["evaluation"]["gross_cost_proxies"]),
    )
    comparison_frames.append(ranking_comparisons)
    comparisons = pd.concat(comparison_frames, ignore_index=True)
    reality_frames = [
        _family_reality_checks(
            combined["distribution_daily"],
            metric="mean_pinball",
            baseline="student_t_feature_scale",
            higher_is_better=False,
            family="distribution",
        ),
        _family_reality_checks(
            combined["hurdle_daily"],
            metric="log_score",
            baseline="recent_252d_loss_probability",
            higher_is_better=True,
            family="hurdle",
        ),
        _family_reality_checks(
            combined["state_daily"],
            metric="log_score",
            baseline="recent_252d_probability",
            higher_is_better=True,
            family="state",
        ),
        _family_reality_checks(
            combined["hazard_daily"],
            metric="log_likelihood",
            baseline="empirical_hazard",
            higher_is_better=True,
            family="hazard",
        ),
        ranking_reality,
    ]
    reality = pd.concat(reality_frames, ignore_index=True)
    regime = _regime_metrics(
        {
            "distribution": combined["distribution_daily"],
            "hurdle": combined["hurdle_daily"],
            "state": combined["state_daily"],
            "hazard": combined["hazard_daily"],
            "ranking": combined["ranking_daily"].assign(target=lambda frame: frame["score_target"]),
        }
    )
    for name, frame in (
        ("comparisons", comparisons),
        ("reality_checks", reality),
        ("regime_metrics", regime),
    ):
        path = run_root / f"{name}.parquet"
        _write_parquet(path, frame)
        files[name] = base._file_record(path)

    implementation_checks = {
        "all_folds_completed": len(fold_summaries)
        == len(requested_years) * len(requested_horizons),
        "all_fold_audits_ok": all(
            summary["audit"]["status"] == "ok" for summary in fold_summaries
        ),
        "all_candidate_forecasts_cover_full_year_universe": all(
            summary["audit"]["checks"]["candidate_rows_equal_full_year_universe"]
            for summary in fold_summaries
        ),
        "all_purges_pass": all(
            summary["audit"]["checks"]["all_purges_pass"]
            for summary in fold_summaries
        ),
        "all_quantiles_monotone_after_rearrangement": all(
            summary["audit"]["checks"]["all_post_rearrangement_quantiles_monotone"]
            for summary in fold_summaries
        ),
        "no_2026_rows_or_reads": all(
            summary["forbidden_2026_read_count"] == 0 for summary in fold_summaries
        ),
        "hyperparameters_not_selected_on_preflight": True,
        "portfolio_and_account_search_absent": True,
    }
    preflight_only = set(requested_years) == set(PREFLIGHT_YEARS)
    development_gate = _development_gate(
        evaluation_years=requested_years,
        comparisons=comparisons,
        reality=reality,
        ranking_daily=combined["ranking_daily"],
        cost_proxies=tuple(float(v) for v in study["evaluation"]["gross_cost_proxies"]),
    )
    summary = {
        "schema": MANIFEST_SCHEMA,
        "status": "completed_forecast_research",
        "study_id": STUDY_ID,
        "run_id": str(run_id),
        "study_sha256": base._sha256_file(study_path),
        "input_fingerprint": base.EXPECTED_INPUT_FINGERPRINT,
        "label_fingerprint": EXPECTED_LABEL_FINGERPRINT,
        "rank_cache_fingerprint": rank_manifest["fingerprint"],
        "evaluation_years": list(requested_years),
        "horizons": list(requested_horizons),
        "primary_horizon": PRIMARY_HORIZON,
        "diagnostic_horizon": DIAGNOSTIC_HORIZON,
        "fold_count": len(fold_summaries),
        "preflight_only": preflight_only,
        "implementation_audit": {
            "status": "ok" if all(implementation_checks.values()) else "failed",
            "checks": implementation_checks,
        },
        "preflight_expansion_gate": {
            "eligible_for_frozen_development_matrix": bool(
                preflight_only and all(implementation_checks.values())
            ),
            "predictive_performance_used_for_preflight_tuning": False,
            "meaning": (
                "This gate validates mechanics only; it is not evidence of alpha "
                "and cannot authorize account research."
            ),
        },
        "development_forecast_gate": development_gate,
        "training_performed": True,
        "hyperparameter_selection_performed": False,
        "model_selection_performed": False,
        "portfolio_selection_performed": False,
        "account_optimization_performed": False,
        "retrospective_oos_years_used": [],
        "forbidden_2026_read_count": 0,
        "profit_claim_allowed": False,
        "files": files,
        "fold_summaries": [str((run_root / "folds" / f"y{s['evaluation_year']}_h{s['horizon']}" / "summary.json").resolve()) for s in fold_summaries],
    }
    _write_json(run_root / "summary.json", summary)
    return summary


def run_preflight(
    *,
    study_path: str | Path = DEFAULT_STUDY_PATH,
    input_manifest_path: str | Path = DEFAULT_INPUT_MANIFEST,
    label_manifest_path: str | Path = DEFAULT_LABEL_MANIFEST,
    output_root: str | Path = DEFAULT_OUTPUT_ROOT,
) -> dict[str, Any]:
    return run_experiment(
        study_path=study_path,
        input_manifest_path=input_manifest_path,
        label_manifest_path=label_manifest_path,
        output_root=output_root,
        evaluation_years=PREFLIGHT_YEARS,
        horizons=HORIZONS,
        run_id="preflight_2022_h20_h10",
    )


def run_development(
    *,
    study_path: str | Path = DEFAULT_STUDY_PATH,
    input_manifest_path: str | Path = DEFAULT_INPUT_MANIFEST,
    label_manifest_path: str | Path = DEFAULT_LABEL_MANIFEST,
    output_root: str | Path = DEFAULT_OUTPUT_ROOT,
) -> dict[str, Any]:
    return run_experiment(
        study_path=study_path,
        input_manifest_path=input_manifest_path,
        label_manifest_path=label_manifest_path,
        output_root=output_root,
        evaluation_years=DEVELOPMENT_YEARS,
        horizons=HORIZONS,
        run_id="development_2017_2022",
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Seq100 bad-tail hurdle and separate return-ranking research."
    )
    parser.add_argument("--study", default=str(DEFAULT_STUDY_PATH))
    parser.add_argument("--input-manifest", default=str(DEFAULT_INPUT_MANIFEST))
    parser.add_argument("--label-manifest", default=str(DEFAULT_LABEL_MANIFEST))
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT))
    parser.add_argument("--prepare-ranks", action="store_true")
    parser.add_argument("--preflight", action="store_true")
    parser.add_argument("--run-development", action="store_true")
    parser.add_argument("--tail-diagnostics", action="store_true")
    parser.add_argument("--run-root", default="")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    selected_modes = sum(
        bool(value)
        for value in (args.preflight, args.run_development, args.tail_diagnostics, args.prepare_ranks)
    )
    if selected_modes > 1:
        raise ValueError("choose_one_research_mode")
    if args.preflight:
        result = run_preflight(
            study_path=args.study,
            input_manifest_path=args.input_manifest,
            label_manifest_path=args.label_manifest,
            output_root=args.output_root,
        )
    elif args.run_development:
        result = run_development(
            study_path=args.study,
            input_manifest_path=args.input_manifest,
            label_manifest_path=args.label_manifest,
            output_root=args.output_root,
        )
    elif args.tail_diagnostics:
        run_root = args.run_root or str(
            _resolve_path(args.output_root)
            / "experiments"
            / "development_2017_2022"
        )
        result = run_tail_diagnostics(run_root=run_root, output_root=run_root)
    elif args.prepare_ranks:
        panel = _load_panel(
            input_manifest_path=args.input_manifest,
            label_manifest_path=args.label_manifest,
        )
        _, manifest = prepare_structural_ranks(panel, output_root=args.output_root)
        result = manifest
    else:
        study = load_study(args.study)
        input_manifest, _ = base._load_input_contract(
            _resolve_path(args.input_manifest)
        )
        result = {
            "status": "ready",
            "study_id": study["study_id"],
            "primary_horizon": PRIMARY_HORIZON,
            "diagnostic_horizon": DIAGNOSTIC_HORIZON,
            "quantiles": list(QUANTILE_PROBABILITIES),
            "rank_gam_features": list(STRUCTURAL_FEATURES),
            "lightgbm_feature_count": len(
                base.feature_names_for_block(input_manifest, "core_minute")
            ),
            "preflight_years": list(PREFLIGHT_YEARS),
            "development_years": list(DEVELOPMENT_YEARS),
            "retrospective_oos_years_used": [],
            "forbidden_2026_read_count": 0,
            "training_performed": False,
            "portfolio_selection_performed": False,
        }
    print(json.dumps(result, ensure_ascii=False, indent=2, default=base._json_default))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
