"""Candidate-neutral market distribution research.

This module is deliberately a market-layer experiment, not a trading policy.
It compares direct multi-horizon forecasts, a coherent iterated state kernel,
joint path forecasts, and an augmented long-memory volatility state on the
corrected point-in-time input. Every model is scored on dates rather than
candidate rows because the market observation is shared by the cross-section.
Development folds are reported separately from the repeatedly inspected
2023-2025 retrospective OOS folds; no model is selected by this module.
"""

from __future__ import annotations

import argparse
import json
import math
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
from scipy import optimize, stats
from sklearn.covariance import LedoitWolf
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error

try:  # ``arch`` is used only for a validated volatility baseline.
    from arch import arch_model
except Exception:  # pragma: no cover - optional import for lightweight tests
    arch_model = None


WORKSPACE_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INPUT_MANIFEST = (
    WORKSPACE_ROOT
    / "daily_research/output/path_policy/studies/"
    / "seq100_quality_liquidity_training_ready/model_inputs/manifest.json"
)
DEFAULT_OUTPUT_ROOT = (
    WORKSPACE_ROOT
    / "daily_research/output/path_policy/studies/"
    / "seq100_mathematical_market_model"
)
FORMAL_START_YEAR = 2012
FORBIDDEN_YEAR = 2026
OOS_YEARS = (2023, 2024, 2025)
DEVELOPMENT_YEARS = (2017, 2018, 2019, 2020, 2021, 2022)
HORIZONS = (1, 2, 5, 10, 20)
MARKET_TARGET = "market_all__ret1_mean"
EXPECTED_INPUT_FINGERPRINT = (
    "a2bea9ed174b0a04ea2e97af475e9b4f1da79d739288a0cd92114998e31784ae"
)
EXPECTED_CANDIDATE_ROWS = 4_191_476
EXPECTED_MARKET_DATES = 3_400
MINUTE_STATE_FEATURES = (
    "minute_realized_volatility",
    "minute_downside_semivolatility",
    "minute_upside_semivolatility",
    "minute_amount_entropy",
    "minute_open_close_return",
)
LONG_MEMORY_FEATURES = (
    "log_mean_realized_volatility_d1",
    "log_mean_realized_volatility_w1",
    "log_mean_realized_volatility_m1",
    "log_mean_realized_volatility_q1",
    "realized_semivariance_balance",
    "mean_minute_open_close_return",
    "mean_minute_amount_entropy",
)
# sklearn minimizes sum of squared errors plus alpha * ||beta||^2. Scaling
# alpha by n makes these penalties comparable across expanding folds.
RIDGE_PENALTIES = (1e-4, 1e-3, 1e-2)
EWMA_HALF_LIVES = (5, 20, 60)
HAC_LAGS = (20, 60)
BOOTSTRAP_BLOCK_LENGTHS = (20, 60)
BOOTSTRAP_REPETITIONS = 2_000
RNG_SEED = 20260806


@dataclass(frozen=True)
class MarketPanel:
    dates: pd.DatetimeIndex
    state: np.ndarray
    feature_names: tuple[str, ...]
    log_return: np.ndarray
    minute_state: np.ndarray | None = None
    minute_feature_names: tuple[str, ...] = ()
    long_memory_state: np.ndarray | None = None
    long_memory_feature_names: tuple[str, ...] = ()


@dataclass(frozen=True)
class TParams:
    df: float
    loc: float
    scale: float


@dataclass(frozen=True)
class MarginalForecast:
    loc: np.ndarray
    scale: np.ndarray
    df: float


@dataclass(frozen=True)
class JointForecast:
    loc: np.ndarray
    covariance: np.ndarray
    df: float | None = None


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _json_default(value: Any) -> Any:
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, (pd.Timestamp, np.datetime64)):
        return str(value)
    raise TypeError(type(value).__name__)


def _robust_location_scale(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    location = np.nanmedian(x, axis=0)
    mad = np.nanmedian(np.abs(x - location), axis=0)
    scale = 1.4826 * mad
    fallback = np.nanstd(x, axis=0, ddof=1)
    scale = np.where(np.isfinite(scale) & (scale > 1e-10), scale, fallback)
    scale = np.where(np.isfinite(scale) & (scale > 1e-10), scale, 1.0)
    return location.astype(np.float64), scale.astype(np.float64)


def _fit_standardizer(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    location, scale = _robust_location_scale(x)
    return location, scale


def _apply_standardizer(
    x: np.ndarray, location: np.ndarray, scale: np.ndarray
) -> np.ndarray:
    z = (np.asarray(x, dtype=np.float64) - location) / scale
    z[~np.isfinite(z)] = 0.0
    return np.clip(z, -12.0, 12.0)


def _trailing_mean(values: np.ndarray, window: int) -> np.ndarray:
    """Inclusive trailing mean that never consumes observations after t."""

    current = np.asarray(values, dtype=np.float64)
    if int(window) <= 0:
        raise ValueError("trailing_window_must_be_positive")
    result = np.full(current.shape, np.nan, dtype=np.float64)
    if len(current) < int(window):
        return result
    prefix = np.r_[0.0, np.cumsum(current)]
    result[int(window) - 1 :] = (
        prefix[int(window) :] - prefix[: -int(window)]
    ) / float(window)
    return result


def build_long_memory_state(minute_state: np.ndarray) -> np.ndarray:
    """Construct causal HAR-style volatility state from date-level minute data.

    The realized-volatility input is a cross-sectional mean of individual
    quadratic-variation proxies. It is a market state proxy, not the realized
    variance of the equal-weight portfolio, because cross-stock covariances are
    not observed in that scalar.
    """

    minute = np.asarray(minute_state, dtype=np.float64)
    if minute.ndim != 2 or minute.shape[1] != len(MINUTE_STATE_FEATURES):
        raise ValueError("minute_state_shape_mismatch")
    realized = np.maximum(minute[:, 0], 1e-10)
    log_realized = np.log(realized)
    downside_variance = np.square(np.maximum(minute[:, 1], 0.0))
    upside_variance = np.square(np.maximum(minute[:, 2], 0.0))
    semivariance_balance = (downside_variance - upside_variance) / np.maximum(
        downside_variance + upside_variance, 1e-12
    )
    return np.column_stack(
        [
            log_realized,
            _trailing_mean(log_realized, 5),
            _trailing_mean(log_realized, 22),
            _trailing_mean(log_realized, 66),
            semivariance_balance,
            minute[:, 4],
            minute[:, 3],
        ]
    )


def load_market_panel(
    input_manifest: str | Path = DEFAULT_INPUT_MANIFEST,
    *,
    include_minute_state: bool = False,
    verify_all_market_rows: bool = False,
) -> MarketPanel:
    """Load one date-level market observation from the certified row spine.

    The compact matrix repeats market-wide fields for every candidate on a
    date.  We verify that invariant explicitly instead of silently selecting a
    row.  The target is the log of the equal-weight current-day market return;
    the forecast at date ``t`` uses only rows and fields at ``t``.
    """

    manifest_path = Path(input_manifest).resolve()
    manifest = _read_json(manifest_path)
    if manifest.get("status") != "completed":
        raise ValueError("model_input_manifest_not_completed")
    if bool(manifest.get("training_performed")):
        raise ValueError("input_manifest_training_flag_must_be_false")
    if manifest.get("input_fingerprint") != EXPECTED_INPUT_FINGERPRINT:
        raise ValueError("input_fingerprint_mismatch")
    if int(manifest.get("row_count", -1)) != EXPECTED_CANDIDATE_ROWS:
        raise ValueError("input_row_count_mismatch")
    source = dict(manifest.get("source", {}))
    if int(source.get("forbidden_year", -1)) != FORBIDDEN_YEAR:
        raise ValueError("forbidden_year_contract_mismatch")

    feature_names = list(manifest["feature_groups"]["compact_core"])
    market_indices = [
        i for i, name in enumerate(feature_names) if name.startswith("market_")
    ]
    market_names = tuple(feature_names[i] for i in market_indices)
    if MARKET_TARGET not in market_names:
        raise ValueError("market_target_missing")

    row_index_path = Path(manifest["row_index"]["path"])
    row_index = pd.read_parquet(row_index_path, columns=["date_idx", "trade_date"])
    date_idx = row_index["date_idx"].to_numpy(dtype=np.int64)
    if np.any(date_idx[1:] < date_idx[:-1]):
        raise ValueError("row_index_not_date_sorted")
    starts = np.r_[0, np.flatnonzero(date_idx[1:] != date_idx[:-1]) + 1]
    ends = np.r_[starts[1:] - 1, len(row_index) - 1]
    dates = pd.DatetimeIndex(row_index["trade_date"].iloc[starts])

    storage = dict(manifest["storage"]["compact"])
    shape = tuple(int(v) for v in storage["shape"])
    if shape[0] != EXPECTED_CANDIDATE_ROWS or shape[1] != len(feature_names):
        raise ValueError("compact_shape_contract_mismatch")
    compact = np.memmap(storage["path"], dtype=np.float32, mode="r", shape=shape)
    first = np.asarray(compact[starts][:, market_indices], dtype=np.float64)
    last = np.asarray(compact[ends][:, market_indices], dtype=np.float64)
    difference = np.abs(first - last)
    if np.any(np.nan_to_num(difference, nan=0.0) > 1e-6):
        raise ValueError("market_fields_are_not_date_level_invariant")
    if verify_all_market_rows:
        # Inspect every cross-section while keeping only one date-sized slice
        # in memory. The formal experiment enables this expensive audit; quick
        # status and focused tests retain the endpoint contract above.
        for start, end in zip(starts, ends):
            group = np.asarray(
                compact[int(start) : int(end) + 1, market_indices], dtype=np.float64
            )
            for column in range(group.shape[1]):
                finite_values = group[:, column][np.isfinite(group[:, column])]
                if (
                    len(finite_values) > 1
                    and float(np.max(finite_values) - np.min(finite_values)) > 1e-6
                ):
                    raise ValueError("market_fields_are_not_date_level_invariant")
    state = first
    target_index = market_names.index(MARKET_TARGET)
    arithmetic_return = state[:, target_index]
    if np.any(~np.isfinite(arithmetic_return)) or np.any(arithmetic_return <= -1.0):
        raise ValueError("market_return_not_finite_or_positive_wealth")
    log_return = np.log1p(arithmetic_return)
    minute_state: np.ndarray | None = None
    long_memory_state: np.ndarray | None = None
    if include_minute_state:
        minute_indices = [feature_names.index(name) for name in MINUTE_STATE_FEATURES]
        minute_values = np.asarray(compact[:, minute_indices], dtype=np.float64)
        aggregated: list[np.ndarray] = []
        for column_position in range(len(minute_indices)):
            values = minute_values[:, column_position]
            observed = np.isfinite(values)
            totals = np.add.reduceat(np.where(observed, values, 0.0), starts)
            counts = np.add.reduceat(observed.astype(np.int64), starts)
            if np.any(counts == 0):
                raise ValueError("minute_market_state_missing_date")
            aggregated.append(totals / counts)
        minute_state = np.column_stack(aggregated)
        long_memory_state = build_long_memory_state(minute_state)
    formal = dates.year >= FORMAL_START_YEAR
    if not np.all(formal):
        dates = dates[formal]
        state = state[formal]
        log_return = log_return[formal]
        if minute_state is not None:
            minute_state = minute_state[formal]
        if long_memory_state is not None:
            long_memory_state = long_memory_state[formal]
    if dates.year.max() >= FORBIDDEN_YEAR:
        raise ValueError("forbidden_year_present_in_market_panel")
    if len(dates) != EXPECTED_MARKET_DATES:
        raise ValueError("market_date_count_mismatch")
    return MarketPanel(
        dates=dates,
        state=state,
        feature_names=market_names,
        log_return=log_return,
        minute_state=minute_state,
        minute_feature_names=MINUTE_STATE_FEATURES if include_minute_state else (),
        long_memory_state=long_memory_state,
        long_memory_feature_names=LONG_MEMORY_FEATURES if include_minute_state else (),
    )


def build_horizon_targets(log_return: np.ndarray, horizons: Sequence[int]) -> dict[int, np.ndarray]:
    """Return cumulative future log returns for each signal date.

    ``target[h][t]`` is ``sum(log_return[t+1:t+h+1])`` and is NaN whenever the
    complete future window is unavailable.  The current observation is never
    included in its own target.
    """

    values = np.asarray(log_return, dtype=np.float64)
    prefix = np.r_[0.0, np.cumsum(values)]
    result: dict[int, np.ndarray] = {}
    for horizon in horizons:
        h = int(horizon)
        target = np.full(values.shape, np.nan, dtype=np.float64)
        if h > 0 and len(values) > h:
            target[:-h] = prefix[1 + h :] - prefix[1 : -h]
        result[h] = target
    return result


def _fit_t(values: np.ndarray) -> TParams:
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    if len(values) < 10:
        raise ValueError("too_few_values_for_t_fit")
    try:
        df, loc, scale = stats.t.fit(values)
    except Exception:
        df, loc, scale = 30.0, float(np.mean(values)), float(np.std(values, ddof=1))
    return TParams(
        df=float(np.clip(df, 2.05, 200.0)),
        loc=float(loc),
        scale=float(max(scale, 1e-8)),
    )


def _fit_normal(values: np.ndarray) -> TParams:
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    return TParams(
        df=math.inf,
        loc=float(np.mean(values)),
        scale=float(max(np.std(values, ddof=1), 1e-8)),
    )


def _logpdf(values: np.ndarray, params: TParams) -> np.ndarray:
    if math.isinf(params.df):
        return stats.norm.logpdf(values, loc=params.loc, scale=params.scale)
    return stats.t.logpdf(values, params.df, loc=params.loc, scale=params.scale)


def _cdf(values: np.ndarray, params: TParams) -> np.ndarray:
    if math.isinf(params.df):
        return stats.norm.cdf(values, loc=params.loc, scale=params.scale)
    return stats.t.cdf(values, params.df, loc=params.loc, scale=params.scale)


def _student_t_scale_from_standard_deviation(
    standard_deviation: np.ndarray | float, df: float
) -> np.ndarray:
    """Convert a target standard deviation to scipy's Student-t scale."""

    standard_deviation = np.asarray(standard_deviation, dtype=np.float64)
    if math.isinf(df):
        return standard_deviation
    return standard_deviation * math.sqrt((float(df) - 2.0) / float(df))


def crps_from_quantiles(
    values: np.ndarray, loc: np.ndarray, scale: np.ndarray, df: float, *, n: int = 512
) -> np.ndarray:
    """Deterministic CRPS approximation from equally weighted predictive draws."""

    values = np.asarray(values, dtype=np.float64)
    loc = np.asarray(loc, dtype=np.float64)
    scale = np.maximum(np.asarray(scale, dtype=np.float64), 1e-10)
    probs = (np.arange(n, dtype=np.float64) + 0.5) / n
    if math.isinf(df):
        draws = stats.norm.ppf(probs)[None, :]
    else:
        draws = stats.t.ppf(probs, df)[None, :]
    draws = loc[:, None] + scale[:, None] * draws
    first = np.mean(np.abs(draws - values[:, None]), axis=1)
    # For sorted draws, the pairwise absolute-distance sum is exact in O(n).
    sorted_draws = np.sort(draws, axis=1)
    weights = 2.0 * np.arange(n, dtype=np.float64) - n + 1.0
    pairwise = 2.0 * np.sum(sorted_draws * weights[None, :], axis=1) / (n * n)
    return first - 0.5 * pairwise


def _fit_direct_t(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_test: np.ndarray,
    *,
    penalty: float = 1e-3,
    heteroskedastic: bool = False,
) -> MarginalForecast:
    location, scale = _fit_standardizer(x_train)
    train_z = _apply_standardizer(x_train, location, scale)
    test_z = _apply_standardizer(x_test, location, scale)
    alpha = float(len(y_train)) * float(penalty)
    mean_model = Ridge(alpha=alpha, fit_intercept=True)
    mean_model.fit(train_z, y_train)
    train_mean = mean_model.predict(train_z)
    residual = y_train - train_mean
    params = _fit_t(residual)
    if not heteroskedastic:
        predicted_scale = np.full(len(x_test), params.scale, dtype=np.float64)
    else:
        log_residual_sq = np.log(np.square(residual) + 1e-8)
        scale_model = Ridge(alpha=alpha, fit_intercept=True)
        scale_model.fit(train_z, log_residual_sq)
        predicted_scale = np.sqrt(
            np.maximum(np.exp(scale_model.predict(test_z)), 1e-8)
        )
        # Preserve the fitted Student-t scale calibration on average.
        calibration = params.scale / max(float(np.median(np.sqrt(np.exp(scale_model.predict(train_z))))), 1e-8)
        predicted_scale *= calibration
    return MarginalForecast(
        loc=mean_model.predict(test_z),
        scale=np.maximum(predicted_scale, 1e-8),
        df=params.df,
    )


def _fit_ewma_parameters(values: np.ndarray, decay: float) -> TParams:
    values = np.asarray(values, dtype=np.float64)
    mu = float(np.mean(values))
    long_var = float(max(np.var(values, ddof=1), 1e-10))
    variance = long_var
    scales: list[float] = []
    for value in values:
        scales.append(math.sqrt(max(variance, 1e-10)))
        variance = decay * variance + (1.0 - decay) * max((value - mu) ** 2, 1e-12)
    residual = (values - mu) / np.asarray(scales)
    params = _fit_t(residual)
    return TParams(df=params.df, loc=mu, scale=math.sqrt(long_var))


def _ewma_forecast(
    values: np.ndarray,
    train_end: int,
    test_indices: np.ndarray,
    horizons: Sequence[int],
    decay: float,
) -> dict[int, MarginalForecast]:
    train = values[:train_end]
    params = _fit_ewma_parameters(train, decay=decay)
    variance = params.scale * params.scale
    # Update through the end of the training sample using only observed data.
    for value in train:
        variance = decay * variance + (1.0 - decay) * max((value - params.loc) ** 2, 1e-12)
    one_step_variance: dict[int, float] = {}
    result: dict[int, MarginalForecast] = {}
    for horizon in horizons:
        loc = np.full(len(test_indices), horizon * params.loc, dtype=np.float64)
        scale = np.empty(len(test_indices), dtype=np.float64)
        for j, index in enumerate(test_indices):
            var = variance
            # Walk from the train boundary to the signal date with observed
            # returns; the forecast itself never consumes future observations.
            for pos in range(train_end, int(index) + 1):
                var = decay * var + (1.0 - decay) * max((values[pos] - params.loc) ** 2, 1e-12)
            future_var = 0.0
            current = var
            for _ in range(horizon):
                current = decay * current + (1.0 - decay) * params.scale**2
                future_var += current
            scale[j] = float(
                _student_t_scale_from_standard_deviation(
                    math.sqrt(max(future_var, 1e-10)), params.df
                )
            )
        result[horizon] = MarginalForecast(loc=loc, scale=scale, df=params.df)
    return result


def _fit_garch_forecast(
    values: np.ndarray, train_end: int, test_indices: np.ndarray, horizons: Sequence[int], distribution: str
) -> dict[int, MarginalForecast]:
    if arch_model is None:
        raise RuntimeError("arch_dependency_missing")
    train = np.asarray(values[:train_end], dtype=np.float64) * 100.0
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = arch_model(
            train,
            mean="Constant",
            vol="GARCH",
            p=1,
            q=1,
            dist="StudentsT" if distribution == "t" else "normal",
            rescale=False,
        )
        fitted = model.fit(disp="off", show_warning=False)
    params = fitted.params
    mu = float(params.get("mu", 0.0)) / 100.0
    omega = float(params.get("omega", 1e-6)) / 10000.0
    alpha = float(params.get("alpha[1]", 0.05))
    beta = float(params.get("beta[1]", 0.9))
    nu = float(params.get("nu", math.inf)) if distribution == "t" else math.inf
    residuals = train / 100.0 - mu
    variance = float(np.var(residuals, ddof=1))
    for residual in residuals:
        variance = omega + alpha * residual * residual + beta * variance
    result: dict[int, MarginalForecast] = {}
    for horizon in horizons:
        locs: list[float] = []
        scales: list[float] = []
        current_variance = variance
        cursor = train_end
        for index in test_indices:
            for pos in range(cursor, int(index) + 1):
                residual = values[pos] - mu
                current_variance = omega + alpha * residual * residual + beta * current_variance
            cursor = int(index) + 1
            future_var = 0.0
            one = current_variance
            for _ in range(horizon):
                one = omega + (alpha + beta) * one
                future_var += one
            locs.append(horizon * mu)
            scales.append(
                float(
                    _student_t_scale_from_standard_deviation(
                        math.sqrt(max(future_var, 1e-10)), nu
                    )
                )
            )
        result[horizon] = MarginalForecast(
            loc=np.asarray(locs), scale=np.asarray(scales), df=float(np.clip(nu, 2.05, 200.0))
        )
    return result


def _fit_joint_gaussian(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_test: np.ndarray,
    *,
    penalty: float = 1e-3,
) -> JointForecast:
    location, scale = _fit_standardizer(x_train)
    train_z = _apply_standardizer(x_train, location, scale)
    test_z = _apply_standardizer(x_test, location, scale)
    model = Ridge(alpha=float(len(y_train)) * float(penalty), fit_intercept=True)
    model.fit(train_z, y_train)
    residual = y_train - model.predict(train_z)
    covariance = LedoitWolf().fit(residual).covariance_
    covariance = covariance + np.eye(covariance.shape[0]) * 1e-8
    return JointForecast(loc=model.predict(test_z), covariance=covariance)


def _fit_joint_static_gaussian(y_train: np.ndarray, n_test: int) -> JointForecast:
    """Unconditional joint path baseline with shrinkage covariance."""

    loc = np.mean(y_train, axis=0)
    covariance = LedoitWolf().fit(y_train - loc).covariance_
    covariance = covariance + np.eye(covariance.shape[0]) * 1e-8
    return JointForecast(loc=np.repeat(loc[None, :], n_test, axis=0), covariance=covariance)


def _estimate_multivariate_t_df(
    residual: np.ndarray, covariance: np.ndarray
) -> float:
    """Estimate one elliptical tail parameter with covariance held fixed.

    ``scipy.stats.multivariate_t`` takes a shape matrix, whereas Ledoit-Wolf
    estimates the covariance. For df > 2, shape = covariance * (df-2)/df.
    Holding the shrinkage covariance fixed keeps this one-dimensional fit
    identifiable and prevents tail estimation from destabilizing dependence.
    """

    residual = np.asarray(residual, dtype=np.float64)
    covariance = np.asarray(covariance, dtype=np.float64)

    def objective(log_df_minus_two: float) -> float:
        df = 2.0 + math.exp(float(log_df_minus_two))
        shape = covariance * ((df - 2.0) / df)
        values = stats.multivariate_t.logpdf(
            residual, loc=np.zeros(residual.shape[1]), shape=shape, df=df
        )
        return -float(np.sum(values))

    lower = math.log(0.05)
    upper = math.log(198.0)
    fitted = optimize.minimize_scalar(objective, bounds=(lower, upper), method="bounded")
    if not fitted.success or not math.isfinite(float(fitted.fun)):
        return 30.0
    return float(np.clip(2.0 + math.exp(float(fitted.x)), 2.05, 200.0))


def _fit_joint_student_t(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_test: np.ndarray,
    *,
    penalty: float = 1e-3,
) -> JointForecast:
    location, scale = _fit_standardizer(x_train)
    train_z = _apply_standardizer(x_train, location, scale)
    test_z = _apply_standardizer(x_test, location, scale)
    model = Ridge(alpha=float(len(y_train)) * float(penalty), fit_intercept=True)
    model.fit(train_z, y_train)
    residual = y_train - model.predict(train_z)
    covariance = LedoitWolf().fit(residual).covariance_
    covariance = covariance + np.eye(covariance.shape[0]) * 1e-8
    df = _estimate_multivariate_t_df(residual, covariance)
    return JointForecast(
        loc=model.predict(test_z), covariance=covariance, df=df
    )


def _fit_joint_static_student_t(y_train: np.ndarray, n_test: int) -> JointForecast:
    loc = np.mean(y_train, axis=0)
    residual = y_train - loc
    covariance = LedoitWolf().fit(residual).covariance_
    covariance = covariance + np.eye(covariance.shape[0]) * 1e-8
    df = _estimate_multivariate_t_df(residual, covariance)
    return JointForecast(
        loc=np.repeat(loc[None, :], n_test, axis=0),
        covariance=covariance,
        df=df,
    )


def _fit_iterated_var_models(
    x_train: np.ndarray,
    g_train: np.ndarray,
    x_test: np.ndarray,
    g_test: np.ndarray,
    *,
    horizons: Sequence[int],
    components: int,
    alpha: float = 1.0,
) -> tuple[dict[int, MarginalForecast], JointForecast]:
    x_location, x_scale = _fit_standardizer(x_train)
    train_z = _apply_standardizer(x_train, x_location, x_scale)
    test_z = _apply_standardizer(x_test, x_location, x_scale)
    pca = PCA(n_components=components, svd_solver="full")
    train_pc = pca.fit_transform(train_z)
    test_pc = pca.transform(test_z)
    g_location = float(np.mean(g_train))
    g_scale = float(max(np.std(g_train, ddof=1), 1e-8))
    train_g_z = (np.asarray(g_train, dtype=np.float64) - g_location) / g_scale
    test_g_z = (np.asarray(g_test, dtype=np.float64) - g_location) / g_scale
    train_state = np.column_stack([train_pc, train_g_z])
    design = train_state[:-1]
    target = train_state[1:]
    transition = Ridge(alpha=alpha, fit_intercept=True)
    transition.fit(design, target)
    residual = target - transition.predict(design)
    covariance = LedoitWolf().fit(residual).covariance_ + np.eye(components + 1) * 1e-8
    # The first forecast starts from the observed state at each signal date.
    result: dict[int, MarginalForecast] = {}
    dimension = components + 1
    e = np.zeros(dimension)
    e[-1] = 1.0
    a = np.asarray(transition.coef_, dtype=np.float64)
    b = np.asarray(transition.intercept_, dtype=np.float64)
    # Augment the transition with cumulative future log return. This preserves
    # all cross-horizon covariance terms instead of summing marginal variances.
    augmented_transition = np.zeros((dimension + 1, dimension + 1), dtype=np.float64)
    augmented_transition[:dimension, :dimension] = a
    augmented_transition[-1, :dimension] = g_scale * (e @ a)
    augmented_transition[-1, -1] = 1.0
    augmented_intercept = np.r_[b, g_location + g_scale * float(e @ b)]
    augmented_covariance = np.zeros((dimension + 1, dimension + 1), dtype=np.float64)
    augmented_covariance[:dimension, :dimension] = covariance
    cross = g_scale * (covariance @ e)
    augmented_covariance[:dimension, -1] = cross
    augmented_covariance[-1, :dimension] = cross
    augmented_covariance[-1, -1] = g_scale * g_scale * float(e @ covariance @ e)
    ordered_horizons = tuple(int(h) for h in horizons)
    horizon_set = set(ordered_horizons)
    joint_loc = np.empty((len(test_z), len(ordered_horizons)), dtype=np.float64)
    marginal_variances: dict[int, float] = {}
    for row in range(len(test_z)):
        state = np.r_[test_pc[row], float(test_g_z[row])]
        # The current return is in the information set at the signal close.
        mean = np.r_[state, 0.0]
        cov = np.zeros_like(augmented_covariance)
        horizon_position = 0
        for step in range(1, max(ordered_horizons) + 1):
            mean = augmented_intercept + augmented_transition @ mean
            cov = (
                augmented_transition @ cov @ augmented_transition.T
                + augmented_covariance
            )
            if step in horizon_set:
                while ordered_horizons[horizon_position] != step:
                    horizon_position += 1
                joint_loc[row, horizon_position] = float(mean[-1])
                marginal_variances[step] = float(max(cov[-1, -1], 1e-10))
                horizon_position += 1

    # Cov(C_h, C_l) follows from the linear innovation representation. It does
    # not depend on the observed starting state, so one covariance matrix is
    # shared by every forecast date in this homoskedastic VAR challenger.
    selector = np.zeros(dimension + 1, dtype=np.float64)
    selector[-1] = 1.0
    powers = [np.eye(dimension + 1, dtype=np.float64)]
    for _ in range(max(ordered_horizons)):
        powers.append(augmented_transition @ powers[-1])
    joint_covariance = np.empty(
        (len(ordered_horizons), len(ordered_horizons)), dtype=np.float64
    )
    for left_position, left_horizon in enumerate(ordered_horizons):
        for right_position, right_horizon in enumerate(ordered_horizons):
            covariance_value = 0.0
            for innovation_step in range(1, min(left_horizon, right_horizon) + 1):
                left_loading = selector @ powers[left_horizon - innovation_step]
                right_loading = selector @ powers[right_horizon - innovation_step]
                covariance_value += float(
                    left_loading @ augmented_covariance @ right_loading.T
                )
            joint_covariance[left_position, right_position] = covariance_value
    joint_covariance = 0.5 * (joint_covariance + joint_covariance.T)
    minimum_eigenvalue = float(np.min(np.linalg.eigvalsh(joint_covariance)))
    if minimum_eigenvalue <= 1e-10:
        joint_covariance += np.eye(len(ordered_horizons)) * (1e-10 - minimum_eigenvalue)

    for position, horizon in enumerate(ordered_horizons):
        result[horizon] = MarginalForecast(
            loc=joint_loc[:, position],
            scale=np.full(
                len(test_z), math.sqrt(max(joint_covariance[position, position], 1e-10))
            ),
            df=math.inf,
        )
    return result, JointForecast(loc=joint_loc, covariance=joint_covariance)


def _fit_iterated_var(
    x_train: np.ndarray,
    g_train: np.ndarray,
    x_test: np.ndarray,
    g_test: np.ndarray,
    *,
    horizons: Sequence[int],
    components: int,
    alpha: float = 1.0,
) -> dict[int, MarginalForecast]:
    """Backward-compatible marginal wrapper used by focused tests."""

    marginal, _ = _fit_iterated_var_models(
        x_train,
        g_train,
        x_test,
        g_test,
        horizons=horizons,
        components=components,
        alpha=alpha,
    )
    return marginal


def _energy_score(samples: np.ndarray, observation: np.ndarray) -> float:
    samples = np.asarray(samples, dtype=np.float64)
    observation = np.asarray(observation, dtype=np.float64)
    first = np.mean(np.linalg.norm(samples - observation[None, :], axis=1))
    # Independent adjacent pairs give an unbiased O(n) estimate of E|X-X'|.
    pair_count = len(samples) // 2
    pairwise = np.linalg.norm(
        samples[: 2 * pair_count : 2] - samples[1 : 2 * pair_count : 2], axis=1
    )
    return float(first - 0.5 * np.mean(pairwise))


def _variogram_score(samples: np.ndarray, observation: np.ndarray, power: float = 0.5) -> float:
    samples = np.asarray(samples, dtype=np.float64)
    observation = np.asarray(observation, dtype=np.float64)
    n_dim = observation.shape[0]
    score = 0.0
    for i in range(n_dim):
        for j in range(i + 1, n_dim):
            observed = abs(observation[i] - observation[j]) ** power
            predicted = np.mean(np.abs(samples[:, i] - samples[:, j]) ** power)
            score += (observed - predicted) ** 2
    return float(score)


def _joint_shape(forecast: JointForecast) -> np.ndarray:
    if forecast.df is None or math.isinf(float(forecast.df)):
        return forecast.covariance
    return forecast.covariance * ((float(forecast.df) - 2.0) / float(forecast.df))


def _joint_samples(
    forecast: JointForecast, n: int, rng: np.random.Generator
) -> np.ndarray:
    if forecast.df is None or math.isinf(float(forecast.df)):
        return rng.multivariate_normal(forecast.loc, forecast.covariance, size=n)
    return stats.multivariate_t.rvs(
        loc=forecast.loc,
        shape=_joint_shape(forecast),
        df=float(forecast.df),
        size=n,
        random_state=rng,
    )


def _joint_logpdf(observation: np.ndarray, forecast: JointForecast) -> float:
    if forecast.df is None or math.isinf(float(forecast.df)):
        return float(
            stats.multivariate_normal.logpdf(
                observation, mean=forecast.loc, cov=forecast.covariance
            )
        )
    return float(
        stats.multivariate_t.logpdf(
            observation,
            loc=forecast.loc,
            shape=_joint_shape(forecast),
            df=float(forecast.df),
        )
    )


def _joint_marginals(forecast: JointForecast) -> dict[int, MarginalForecast]:
    standard_deviation = np.sqrt(np.maximum(np.diag(forecast.covariance), 1e-10))
    df = math.inf if forecast.df is None else float(forecast.df)
    scale = _student_t_scale_from_standard_deviation(standard_deviation, df)
    return {
        horizon: MarginalForecast(
            loc=forecast.loc[:, position],
            scale=np.full(len(forecast.loc), float(scale[position])),
            df=df,
        )
        for position, horizon in enumerate(HORIZONS)
    }


def _marginal_score_arrays(
    values: np.ndarray, forecast: MarginalForecast
) -> dict[str, np.ndarray]:
    log_scores = _logpdf(
        values,
        TParams(df=forecast.df, loc=forecast.loc, scale=forecast.scale),
    )
    cdf_values = _cdf(
        values,
        TParams(df=forecast.df, loc=forecast.loc, scale=forecast.scale),
    )
    crps = crps_from_quantiles(values, forecast.loc, forecast.scale, forecast.df)
    return {
        "log_score": np.asarray(log_scores, dtype=np.float64),
        "crps": np.asarray(crps, dtype=np.float64),
        "pit": np.asarray(cdf_values, dtype=np.float64),
    }


def _metric_row(
    *,
    year: int,
    model: str,
    horizon: int,
    values: np.ndarray,
    forecast: MarginalForecast,
) -> dict[str, Any]:
    scores = _marginal_score_arrays(values, forecast)
    log_scores = scores["log_score"]
    cdf_values = scores["pit"]
    crps = scores["crps"]
    z = stats.norm.ppf(np.clip(cdf_values, 1e-8, 1.0 - 1e-8))
    return {
        "year": int(year),
        "model": model,
        "horizon": int(horizon),
        "n": int(len(values)),
        "log_score": float(np.mean(log_scores)),
        "crps": float(np.mean(crps)),
        "rmse": float(math.sqrt(mean_squared_error(values, forecast.loc))),
        "coverage_50": float(np.mean(np.abs(z) <= stats.norm.ppf(0.75))),
        "coverage_90": float(np.mean(np.abs(z) <= stats.norm.ppf(0.95))),
        "pit_mean": float(np.mean(cdf_values)),
        "pit_variance": float(np.var(cdf_values)),
    }


def _fit_fold_models(
    panel: MarketPanel,
    train_end: int,
    test_indices: np.ndarray,
    targets: Mapping[int, np.ndarray],
) -> tuple[dict[str, dict[int, MarginalForecast]], dict[str, JointForecast]]:
    x = panel.state
    g = panel.log_return
    forecasts: dict[str, dict[int, MarginalForecast]] = {}
    joints: dict[str, JointForecast] = {}
    # Unconditional distributions are calculated separately at each horizon.
    for kind in ("gaussian", "student_t"):
        model_name = f"static_{kind}"
        forecasts[model_name] = {}
        for horizon in HORIZONS:
            # A horizon-h label at signal index t consumes observations through
            # t+h.  Remove labels whose outcome window would cross the fold
            # boundary, even for an unconditional baseline.
            values = targets[horizon][: max(0, train_end - horizon)]
            values = values[np.isfinite(values)]
            p = _fit_normal(values) if kind == "gaussian" else _fit_t(values)
            forecasts[model_name][horizon] = MarginalForecast(
                loc=np.full(len(test_indices), p.loc),
                scale=np.full(len(test_indices), p.scale),
                df=p.df,
            )
    # Treat regularization as a declared sensitivity family. With standardized
    # inputs, alpha=n*penalty corresponds to mean squared error plus
    # penalty*||beta||^2 and therefore has the same semantics across folds.
    direct_variants: dict[str, tuple[np.ndarray, bool]] = {
        "direct_market_student_t": (x, False),
        "direct_market_student_t_hetero": (x, True),
    }
    if panel.minute_state is not None and panel.long_memory_state is not None:
        direct_variants.update(
            {
                "direct_minute_student_t_hetero": (panel.minute_state, True),
                "direct_market_plus_minute_student_t_hetero": (
                    np.column_stack([panel.state, panel.minute_state]),
                    True,
                ),
                "direct_har_student_t_hetero": (panel.long_memory_state, True),
            }
        )
    for base_name, (variant, heteroskedastic) in direct_variants.items():
        for penalty in RIDGE_PENALTIES:
            name = f"{base_name}_ridge_{penalty:.0e}"
            forecasts[name] = {}
            for horizon in HORIZONS:
                h_train_end = max(0, train_end - horizon)
                y = targets[horizon][:h_train_end]
                valid = np.isfinite(y)
                forecasts[name][horizon] = _fit_direct_t(
                    variant[:h_train_end][valid],
                    y[valid],
                    variant[test_indices],
                    penalty=penalty,
                    heteroskedastic=heteroskedastic,
                )
    for half_life in EWMA_HALF_LIVES:
        decay = math.exp(math.log(0.5) / float(half_life))
        forecasts[f"ewma_student_t_hl{half_life}"] = _ewma_forecast(
            g, train_end, test_indices, HORIZONS, decay
        )
    for distribution in ("normal", "t"):
        try:
            forecasts[f"garch_{distribution}"] = _fit_garch_forecast(
                g, train_end, test_indices, HORIZONS, distribution
            )
        except Exception as exc:  # pragma: no cover - environment/data dependent
            forecasts[f"garch_{distribution}"] = {}
            warnings.warn(f"garch_{distribution}_skipped:{type(exc).__name__}:{exc}")
    # A joint direct path model uses exactly the same state input and a single
    # residual covariance, so horizon dependence is learned rather than
    # imposed by multiplying independent marginal forecasts.
    joint_train_end = train_end - max(HORIZONS)
    joint_y = np.column_stack([targets[h][:joint_train_end] for h in HORIZONS])
    valid = np.isfinite(joint_y).all(axis=1)
    joints["joint_static_gaussian"] = _fit_joint_static_gaussian(
        joint_y[valid], len(test_indices)
    )
    joints["joint_static_student_t"] = _fit_joint_static_student_t(
        joint_y[valid], len(test_indices)
    )
    joint_inputs = {"market": x}
    if panel.long_memory_state is not None:
        joint_inputs["har"] = panel.long_memory_state
    for input_name, joint_x in joint_inputs.items():
        for penalty in RIDGE_PENALTIES:
            gaussian_name = f"joint_direct_{input_name}_gaussian_ridge_{penalty:.0e}"
            student_name = f"joint_direct_{input_name}_student_t_ridge_{penalty:.0e}"
            joints[gaussian_name] = _fit_joint_gaussian(
                joint_x[:joint_train_end][valid],
                joint_y[valid],
                joint_x[test_indices],
                penalty=penalty,
            )
            joints[student_name] = _fit_joint_student_t(
                joint_x[:joint_train_end][valid],
                joint_y[valid],
                joint_x[test_indices],
                penalty=penalty,
            )
    # The iterated kernel is fitted on the same pre-fold state history. Its
    # state sufficiency and time-homogeneity are hypotheses under test.
    target_position = panel.feature_names.index(MARKET_TARGET)
    kernel_x = np.delete(x, target_position, axis=1)
    for components in (1, 3, 5):
        name = f"iterated_var_gaussian_k{components}"
        marginal, joint = _fit_iterated_var_models(
            kernel_x[:train_end],
            g[:train_end],
            kernel_x[test_indices],
            g[test_indices],
            horizons=HORIZONS,
            components=components,
        )
        forecasts[name] = marginal
        joints[f"joint_{name}"] = joint
    if panel.long_memory_state is not None:
        augmented_x = np.column_stack([kernel_x, panel.long_memory_state])
        name = "iterated_var_augmented_gaussian_k3"
        marginal, joint = _fit_iterated_var_models(
            augmented_x[:train_end],
            g[:train_end],
            augmented_x[test_indices],
            g[test_indices],
            horizons=HORIZONS,
            components=3,
        )
        forecasts[name] = marginal
        joints[f"joint_{name}"] = joint
    # Joint models must also earn acceptable marginal scores; otherwise a good
    # dependence score could conceal poorly calibrated one-dimensional laws.
    for joint_name, joint_forecast in joints.items():
        if joint_name.startswith("joint_iterated_var"):
            continue
        forecasts[f"{joint_name}_marginal"] = _joint_marginals(joint_forecast)
    return forecasts, joints


def _acf(values: np.ndarray, lag: int) -> float:
    current = np.asarray(values, dtype=np.float64)
    current = current[np.isfinite(current)]
    if lag <= 0:
        return 1.0
    if len(current) <= lag:
        return math.nan
    left = current[:-lag] - float(np.mean(current[:-lag]))
    right = current[lag:] - float(np.mean(current[lag:]))
    denominator = math.sqrt(float(np.dot(left, left) * np.dot(right, right)))
    return float(np.dot(left, right) / denominator) if denominator > 0.0 else math.nan


def _gph_estimate(values: np.ndarray) -> dict[str, float]:
    """Geweke-Porter-Hudak low-frequency memory diagnostic.

    This is a diagnostic for fractional behavior, not a proof of fractional
    integration. The estimate is intentionally reported with its bandwidth so
    that changing the low-frequency window is visible.
    """

    current = np.asarray(values, dtype=np.float64)
    current = current[np.isfinite(current)]
    n = len(current)
    if n < 32:
        return {"n": float(n), "bandwidth": math.nan, "d": math.nan, "standard_error": math.nan}
    bandwidth = max(10, min(int(n**0.5), n // 2))
    frequencies = 2.0 * math.pi * np.arange(1, bandwidth + 1) / float(n)
    periodogram = np.abs(np.fft.fft(current - np.mean(current)))[1 : bandwidth + 1] ** 2
    periodogram = np.maximum(periodogram, 1e-14)
    design = -2.0 * np.log(2.0 * np.sin(frequencies / 2.0))
    response = np.log(periodogram)
    matrix = np.column_stack([np.ones(bandwidth), design])
    coefficients, _, _, _ = np.linalg.lstsq(matrix, response, rcond=None)
    residual = response - matrix @ coefficients
    covariance = np.linalg.pinv(matrix.T @ matrix) * float(np.dot(residual, residual)) / max(
        bandwidth - 2, 1
    )
    return {
        "n": float(n),
        "bandwidth": float(bandwidth),
        "d": float(coefficients[1]),
        "standard_error": float(math.sqrt(max(covariance[1, 1], 0.0))),
    }


def _ljung_box(values: np.ndarray, lag: int) -> dict[str, float]:
    current = np.asarray(values, dtype=np.float64)
    current = current[np.isfinite(current)]
    n = len(current)
    if n <= lag + 1:
        return {"n": float(n), "lag": float(lag), "q": math.nan, "p_value": math.nan}
    q = 0.0
    for offset in range(1, lag + 1):
        autocorrelation = _acf(current, offset)
        if math.isfinite(autocorrelation):
            q += autocorrelation * autocorrelation / float(n - offset)
    q *= float(n) * (n + 2.0)
    return {
        "n": float(n),
        "lag": float(lag),
        "q": float(q),
        "p_value": float(stats.chi2.sf(q, lag)),
    }


def _state_diagnostics(panel: MarketPanel) -> dict[str, Any]:
    robust_location, robust_scale = _fit_standardizer(panel.state)
    robust_z = _apply_standardizer(panel.state, robust_location, robust_scale)
    ordinary_scale = np.nanstd(panel.state, axis=0, ddof=1)
    ordinary_scale = np.where(ordinary_scale > 1e-10, ordinary_scale, 1.0)
    ordinary_z = (panel.state - np.nanmean(panel.state, axis=0)) / ordinary_scale
    ordinary_z[~np.isfinite(ordinary_z)] = 0.0
    robust_pca = PCA(svd_solver="full").fit(robust_z)
    ordinary_pca = PCA(svd_solver="full").fit(ordinary_z)
    return_fields = [
        index
        for index, name in enumerate(panel.feature_names)
        if name.split("__", 1)[-1].startswith(
            ("ret1_mean", "ret5_mean", "ret20_mean")
        )
    ]
    return_pca = PCA(svd_solver="full").fit(robust_z[:, return_fields])
    rv = (
        panel.long_memory_state[:, 0]
        if panel.long_memory_state is not None
        else np.full(len(panel.dates), np.nan)
    )
    return {
        "market_feature_count": int(panel.state.shape[1]),
        "market_feature_names": list(panel.feature_names),
        "pca_explained_variance_ratio_robust_first10": robust_pca.explained_variance_ratio_[
            :10
        ].tolist(),
        "pca_explained_variance_ratio_ordinary_first10": ordinary_pca.explained_variance_ratio_[
            :10
        ].tolist(),
        "return_feature_count": int(len(return_fields)),
        "return_feature_names": [panel.feature_names[index] for index in return_fields],
        "pca_explained_variance_ratio_return_first10": return_pca.explained_variance_ratio_[
            :10
        ].tolist(),
        "log_return_acf": {str(lag): _acf(panel.log_return, lag) for lag in (1, 5, 20)},
        "absolute_log_return_acf": {
            str(lag): _acf(np.abs(panel.log_return), lag) for lag in (1, 5, 20, 60)
        },
        "log_realized_volatility_acf": {
            str(lag): _acf(rv, lag) for lag in (1, 5, 20, 60, 120)
        },
        "gph_log_realized_volatility": _gph_estimate(rv),
        "gph_absolute_log_return": _gph_estimate(np.log(np.maximum(np.abs(panel.log_return), 1e-8))),
    }


def _volatility_model_diagnostics(panel: MarketPanel) -> pd.DataFrame:
    """Evaluate constant, AR(1), and HAR forecasts of log realized volatility."""

    if panel.long_memory_state is None:
        return pd.DataFrame()
    state = panel.long_memory_state
    target = state[1:, 0]
    signal_state = state[:-1]
    rows: list[dict[str, Any]] = []
    for year in (*DEVELOPMENT_YEARS, *OOS_YEARS):
        test = np.flatnonzero(panel.dates.year[:-1] == year)
        if len(test) == 0:
            continue
        train_end = int(test[0])
        train_y = target[: max(0, train_end - 1)]
        valid_train = np.isfinite(train_y)
        train_y = train_y[valid_train]
        if len(train_y) < 100:
            continue
        variants = {
            "constant": None,
            "ar1": signal_state[:, [0]],
            "har": signal_state[:, :],
        }
        for name, features in variants.items():
            if features is None:
                loc = float(np.mean(train_y))
                scale = float(max(np.std(train_y, ddof=1), 1e-8))
                predictions = np.full(len(test), loc)
                residual = train_y - loc
            else:
                train_features = features[: max(0, train_end - 1)]
                train_valid = np.isfinite(train_features).all(axis=1) & np.isfinite(
                    target[: max(0, train_end - 1)]
                )
                train_features = train_features[train_valid]
                fit_y = target[: max(0, train_end - 1)][train_valid]
                if len(fit_y) < 100:
                    continue
                location, scale_features = _fit_standardizer(train_features)
                fit_z = _apply_standardizer(train_features, location, scale_features)
                model = Ridge(alpha=float(len(fit_y)) * 1e-3, fit_intercept=True)
                model.fit(fit_z, fit_y)
                residual = fit_y - model.predict(fit_z)
                scale = float(max(np.std(residual, ddof=1), 1e-8))
                test_features = _apply_standardizer(
                    features[test], location, scale_features
                )
                predictions = model.predict(test_features)
            observations = target[test]
            valid = np.isfinite(observations) & np.isfinite(predictions)
            if not valid.any():
                continue
            test_residual = observations[valid] - predictions[valid]
            test_ljung_box = _ljung_box(test_residual, 20)
            rows.append(
                {
                    "year": int(year),
                    "model": name,
                    "n": int(valid.sum()),
                    "rmse": float(
                        math.sqrt(mean_squared_error(observations[valid], predictions[valid]))
                    ),
                    "log_score": float(
                        np.mean(
                            stats.norm.logpdf(
                                observations[valid],
                                loc=predictions[valid],
                                scale=scale,
                            )
                        )
                    ),
                    "test_residual_ljung_box_q20": float(test_ljung_box["q"]),
                    "test_residual_ljung_box_p20": float(test_ljung_box["p_value"]),
                }
            )
    return pd.DataFrame(rows)


def _hac_mean(values: np.ndarray, lag: int) -> dict[str, float]:
    current = np.asarray(values, dtype=np.float64)
    current = current[np.isfinite(current)]
    n = len(current)
    if n < 3:
        return {"n": float(n), "mean": math.nan, "standard_error": math.nan, "t": math.nan, "p": math.nan}
    mean = float(np.mean(current))
    centered = current - mean
    lag = min(max(int(lag), 0), n - 1)
    long_run = float(np.dot(centered, centered) / n)
    for offset in range(1, lag + 1):
        covariance = float(np.dot(centered[offset:], centered[:-offset]) / n)
        long_run += 2.0 * (1.0 - offset / float(lag + 1)) * covariance
    standard_error = math.sqrt(max(long_run, 0.0) / n)
    t_stat = (
        mean / standard_error
        if standard_error > 0.0
        else math.inf
        if mean > 0.0
        else -math.inf
        if mean < 0.0
        else 0.0
    )
    return {
        "n": float(n),
        "mean": mean,
        "standard_error": float(standard_error),
        "t": float(t_stat),
        "p": float(stats.norm.sf(t_stat)) if math.isfinite(t_stat) else (0.0 if t_stat > 0 else 1.0),
    }


def _moving_block_bootstrap(
    values: np.ndarray,
    *,
    block_length: int,
    repetitions: int = BOOTSTRAP_REPETITIONS,
    seed: int = RNG_SEED,
) -> dict[str, float]:
    current = np.asarray(values, dtype=np.float64)
    current = current[np.isfinite(current)]
    n = len(current)
    if n < max(3, int(block_length)):
        return {"lower": math.nan, "upper": math.nan, "positive_probability": math.nan}
    rng = np.random.default_rng(int(seed))
    blocks = int(math.ceil(n / float(block_length)))
    offsets = np.arange(int(block_length), dtype=np.int64)
    draws = np.empty(int(repetitions), dtype=np.float64)
    for repetition in range(int(repetitions)):
        starts = rng.integers(0, n, size=blocks)
        indices = (starts[:, None] + offsets[None, :]).ravel()[:n] % n
        draws[repetition] = float(np.mean(current[indices]))
    return {
        "lower": float(np.quantile(draws, 0.025)),
        "upper": float(np.quantile(draws, 0.975)),
        "positive_probability": float(np.mean(draws > 0.0)),
    }


def _reality_check(
    differences: np.ndarray,
    model_names: Sequence[str],
    *,
    block_length: int,
    repetitions: int = BOOTSTRAP_REPETITIONS,
    seed: int = RNG_SEED,
) -> dict[str, Any]:
    """White's max-statistic bootstrap for a predeclared model family."""

    matrix = np.asarray(differences, dtype=np.float64)
    if matrix.ndim != 2 or matrix.shape[1] != len(model_names):
        raise ValueError("reality_check_matrix_shape_mismatch")
    valid = np.isfinite(matrix).all(axis=1)
    matrix = matrix[valid]
    if len(matrix) < max(3, int(block_length)) or matrix.shape[1] == 0:
        return {"n": int(len(matrix)), "p_value": math.nan, "observed_max": math.nan, "winner": None}
    observed_means = np.mean(matrix, axis=0)
    observed_max = max(0.0, float(np.max(observed_means)))
    centered = matrix - observed_means[None, :]
    rng = np.random.default_rng(int(seed))
    blocks = int(math.ceil(len(matrix) / float(block_length)))
    offsets = np.arange(int(block_length), dtype=np.int64)
    boot_max = np.empty(int(repetitions), dtype=np.float64)
    for repetition in range(int(repetitions)):
        starts = rng.integers(0, len(matrix), size=blocks)
        indices = (starts[:, None] + offsets[None, :]).ravel()[: len(matrix)] % len(matrix)
        boot_max[repetition] = max(0.0, float(np.max(np.mean(centered[indices], axis=0))))
    p_value = float((1.0 + np.sum(boot_max >= observed_max)) / (len(boot_max) + 1.0))
    winner_position = int(np.argmax(observed_means))
    return {
        "n": int(len(matrix)),
        "p_value": p_value,
        "observed_max": float(observed_max),
        "winner": str(model_names[winner_position]),
        "winner_mean": float(observed_means[winner_position]),
    }


def _marginal_inference(
    forecast_frame: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    outer = forecast_frame[forecast_frame["split_role"] == "retrospective_oos"]
    comparison_rows: list[dict[str, Any]] = []
    reality_rows: list[dict[str, Any]] = []
    baseline = "static_student_t"
    for horizon in HORIZONS:
        subset = outer[outer["horizon"] == horizon]
        if subset.empty:
            continue
        for metric, higher_is_better in (("log_score", True), ("crps", False)):
            table = subset.pivot(index="date", columns="model", values=metric)
            if baseline not in table:
                continue
            models = [name for name in table.columns if name != baseline]
            differences: list[np.ndarray] = []
            valid_models: list[str] = []
            for model in models:
                if higher_is_better:
                    values = table[model] - table[baseline]
                else:
                    values = table[baseline] - table[model]
                values = values.dropna().to_numpy(dtype=np.float64)
                if len(values) < 3:
                    continue
                valid_models.append(model)
                differences.append(values)
                row: dict[str, Any] = {
                    "horizon": int(horizon),
                    "metric": metric,
                    "model": model,
                    "baseline": baseline,
                    "n": int(len(values)),
                    "improvement": float(np.mean(values)),
                }
                for lag in HAC_LAGS:
                    hac = _hac_mean(values, lag)
                    row[f"hac{lag}_se"] = hac["standard_error"]
                    row[f"hac{lag}_t"] = hac["t"]
                    row[f"hac{lag}_p_one_sided"] = hac["p"]
                for block in BOOTSTRAP_BLOCK_LENGTHS:
                    boot = _moving_block_bootstrap(
                        values,
                        block_length=block,
                        seed=RNG_SEED + horizon + block + len(comparison_rows),
                    )
                    row[f"mbb{block}_lower"] = boot["lower"]
                    row[f"mbb{block}_upper"] = boot["upper"]
                    row[f"mbb{block}_positive_probability"] = boot["positive_probability"]
                comparison_rows.append(row)
            if differences:
                common = table[[baseline, *valid_models]].dropna()
                matrix = np.column_stack(
                    [
                        (common[model] - common[baseline])
                        if higher_is_better
                        else (common[baseline] - common[model])
                        for model in valid_models
                    ]
                )
                for block in BOOTSTRAP_BLOCK_LENGTHS:
                    result = _reality_check(
                        matrix,
                        valid_models,
                        block_length=block,
                        seed=RNG_SEED + horizon * 100 + block + (0 if higher_is_better else 50_000),
                    )
                    reality_rows.append(
                        {
                            "horizon": int(horizon),
                            "metric": metric,
                            "baseline": baseline,
                            "block_length": int(block),
                            **result,
                        }
                    )
    return pd.DataFrame(comparison_rows), pd.DataFrame(reality_rows)


def _joint_inference(joint_frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    outer = joint_frame[joint_frame["split_role"] == "retrospective_oos"]
    baseline = "joint_static_student_t"
    comparison_rows: list[dict[str, Any]] = []
    reality_rows: list[dict[str, Any]] = []
    for metric, higher_is_better in (
        ("log_score", True),
        ("energy_score", False),
        ("variogram_score", False),
    ):
        table = outer.pivot(index="date", columns="model", values=metric)
        if baseline not in table:
            continue
        models = [name for name in table.columns if name != baseline]
        differences: list[np.ndarray] = []
        valid_models: list[str] = []
        for model in models:
            values = (
                table[model] - table[baseline]
                if higher_is_better
                else table[baseline] - table[model]
            ).dropna().to_numpy(dtype=np.float64)
            if len(values) < 3:
                continue
            valid_models.append(model)
            differences.append(values)
            row: dict[str, Any] = {
                "metric": metric,
                "model": model,
                "baseline": baseline,
                "n": int(len(values)),
                "improvement": float(np.mean(values)),
            }
            for lag in HAC_LAGS:
                hac = _hac_mean(values, lag)
                row[f"hac{lag}_se"] = hac["standard_error"]
                row[f"hac{lag}_t"] = hac["t"]
                row[f"hac{lag}_p_one_sided"] = hac["p"]
            for block in BOOTSTRAP_BLOCK_LENGTHS:
                boot = _moving_block_bootstrap(
                    values,
                    block_length=block,
                    seed=RNG_SEED + block + len(comparison_rows),
                )
                row[f"mbb{block}_lower"] = boot["lower"]
                row[f"mbb{block}_upper"] = boot["upper"]
                row[f"mbb{block}_positive_probability"] = boot["positive_probability"]
            comparison_rows.append(row)
        if differences:
            common = table[[baseline, *valid_models]].dropna()
            matrix = np.column_stack(
                [
                    (common[model] - common[baseline])
                    if higher_is_better
                    else (common[baseline] - common[model])
                    for model in valid_models
                ]
            )
            for block in BOOTSTRAP_BLOCK_LENGTHS:
                result = _reality_check(
                    matrix,
                    valid_models,
                    block_length=block,
                    seed=RNG_SEED + block + 100_000,
                )
                reality_rows.append(
                    {
                        "metric": metric,
                        "baseline": baseline,
                        "block_length": int(block),
                        **result,
                    }
                )
    return pd.DataFrame(comparison_rows), pd.DataFrame(reality_rows)


def run_experiment(
    *, input_manifest: str | Path = DEFAULT_INPUT_MANIFEST, output_root: str | Path = DEFAULT_OUTPUT_ROOT
) -> dict[str, Any]:
    panel = load_market_panel(
        input_manifest,
        include_minute_state=True,
        verify_all_market_rows=True,
    )
    targets = build_horizon_targets(panel.log_return, HORIZONS)
    rows: list[dict[str, Any]] = []
    joint_rows: list[dict[str, Any]] = []
    forecast_records: list[dict[str, Any]] = []
    for split_role, years in (
        ("development", DEVELOPMENT_YEARS),
        ("retrospective_oos", OOS_YEARS),
    ):
        for year in years:
            test_indices = np.flatnonzero(panel.dates.year == year)
            if len(test_indices) == 0:
                raise ValueError(f"missing_evaluation_year:{year}")
            # The common support deliberately excludes the final 20 dates of
            # the entire panel, where a complete H=20 target is unavailable.
            test_indices = test_indices[test_indices + max(HORIZONS) < len(panel.dates)]
            if len(test_indices) == 0:
                continue
            train_end = int(test_indices[0])
            forecasts, joints = _fit_fold_models(panel, train_end, test_indices, targets)
            for model_name, by_horizon in forecasts.items():
                for horizon, forecast in by_horizon.items():
                    values = targets[horizon][test_indices]
                    valid = np.isfinite(values)
                    if not valid.any():
                        continue
                    valid_forecast = MarginalForecast(
                        loc=forecast.loc[valid], scale=forecast.scale[valid], df=forecast.df
                    )
                    row = _metric_row(
                        year=year,
                        model=model_name,
                        horizon=horizon,
                        values=values[valid],
                        forecast=valid_forecast,
                    )
                    row["split_role"] = split_role
                    rows.append(row)
                    scores = _marginal_score_arrays(values[valid], valid_forecast)
                    for position, (index, value, loc, scale, log_score, crps, pit) in enumerate(
                        zip(
                            test_indices[valid],
                            values[valid],
                            forecast.loc[valid],
                            forecast.scale[valid],
                            scores["log_score"],
                            scores["crps"],
                            scores["pit"],
                        )
                    ):
                        forecast_records.append(
                            {
                                "date": str(panel.dates[index].date()),
                                "year": int(year),
                                "split_role": split_role,
                                "model": model_name,
                                "horizon": int(horizon),
                                "observation": float(value),
                                "loc": float(loc),
                                "scale": float(scale),
                                "df": float(forecast.df),
                                "log_score": float(log_score),
                                "crps": float(crps),
                                "pit": float(pit),
                            }
                        )
            observations = np.column_stack([targets[h][test_indices] for h in HORIZONS])
            valid = np.isfinite(observations).all(axis=1)
            for model_position, (model_name, forecast) in enumerate(joints.items()):
                rng_local = np.random.default_rng(
                    RNG_SEED + int(year) * 1000 + model_position
                )
                for index, observation, mean in zip(
                    test_indices[valid], observations[valid], forecast.loc[valid]
                ):
                    per_date_forecast = JointForecast(
                        loc=np.asarray(mean, dtype=np.float64),
                        covariance=forecast.covariance,
                        df=forecast.df,
                    )
                    samples = _joint_samples(per_date_forecast, 512, rng_local)
                    joint_rows.append(
                        {
                            "date": str(panel.dates[index].date()),
                            "year": int(year),
                            "split_role": split_role,
                            "model": model_name,
                            "log_score": _joint_logpdf(observation, per_date_forecast),
                            "energy_score": _energy_score(samples, observation),
                            "variogram_score": _variogram_score(samples, observation),
                        }
                    )
    output = Path(output_root).resolve()
    output.mkdir(parents=True, exist_ok=True)
    metrics = pd.DataFrame(rows)
    metrics.to_csv(output / "marginal_metrics.csv", index=False)
    forecast_frame = pd.DataFrame(forecast_records)
    forecast_frame.to_parquet(output / "marginal_forecasts.parquet", index=False)
    joint_frame = pd.DataFrame(joint_rows)
    if not joint_frame.empty:
        joint_summary = (
            joint_frame.groupby(["split_role", "year", "model"], as_index=False)
            .agg(
                n=("date", "count"),
                log_score=("log_score", "mean"),
                energy_score=("energy_score", "mean"),
                variogram_score=("variogram_score", "mean"),
            )
        )
    else:
        joint_summary = pd.DataFrame()
    joint_summary.to_csv(output / "joint_metrics.csv", index=False)
    marginal_summary = (
        metrics.groupby(["split_role", "year", "model", "horizon"], as_index=False)
        .agg(
            n=("n", "first"),
            log_score=("log_score", "first"),
            crps=("crps", "first"),
            rmse=("rmse", "first"),
            coverage_50=("coverage_50", "first"),
            coverage_90=("coverage_90", "first"),
            pit_mean=("pit_mean", "first"),
            pit_variance=("pit_variance", "first"),
        )
    )
    marginal_summary.to_csv(output / "marginal_summary.csv", index=False)
    marginal_comparison, marginal_reality = _marginal_inference(forecast_frame)
    marginal_comparison.to_csv(output / "marginal_comparisons.csv", index=False)
    marginal_reality.to_csv(output / "marginal_reality_check.csv", index=False)
    joint_comparison, joint_reality = _joint_inference(joint_frame)
    joint_comparison.to_csv(output / "joint_comparisons.csv", index=False)
    joint_reality.to_csv(output / "joint_reality_check.csv", index=False)
    diagnostics = _state_diagnostics(panel)
    (output / "state_diagnostics.json").write_text(
        json.dumps(diagnostics, ensure_ascii=False, indent=2, default=_json_default) + "\n",
        encoding="utf-8",
    )
    volatility_metrics = _volatility_model_diagnostics(panel)
    volatility_metrics.to_csv(output / "volatility_metrics.csv", index=False)
    panel_summary = {
        "date_count": int(len(panel.dates)),
        "start_date": str(panel.dates.min().date()),
        "end_date": str(panel.dates.max().date()),
        "feature_count": int(panel.state.shape[1]),
        "feature_names": list(panel.feature_names),
        "minute_feature_count": int(
            0 if panel.minute_state is None else panel.minute_state.shape[1]
        ),
        "minute_feature_names": list(panel.minute_feature_names),
        "long_memory_feature_count": int(
            0 if panel.long_memory_state is None else panel.long_memory_state.shape[1]
        ),
        "long_memory_feature_names": list(panel.long_memory_feature_names),
        "target": MARKET_TARGET,
        "target_mean_log_return": float(np.mean(panel.log_return)),
        "target_std_log_return": float(np.std(panel.log_return, ddof=1)),
        "target_skew": float(stats.skew(panel.log_return, bias=False)),
        "target_excess_kurtosis": float(stats.kurtosis(panel.log_return, bias=False)),
        "target_abs_return_acf1": float(
            np.corrcoef(np.abs(panel.log_return[:-1]), np.abs(panel.log_return[1:]))[0, 1]
        ),
        "state_diagnostics_file": str(output / "state_diagnostics.json"),
        "volatility_metrics_file": str(output / "volatility_metrics.csv"),
    }
    (output / "panel_summary.json").write_text(
        json.dumps(panel_summary, ensure_ascii=False, indent=2, default=_json_default) + "\n",
        encoding="utf-8",
    )
    summary = {
        "status": "completed",
        "study_id": "seq100_mathematical_market_model_v1",
        "input_manifest": str(Path(input_manifest).resolve()),
        "input_fingerprint": _read_json(Path(input_manifest)).get("input_fingerprint"),
        "formal_start_year": FORMAL_START_YEAR,
        "forbidden_year": FORBIDDEN_YEAR,
        "oos_years": list(OOS_YEARS),
        "development_years": list(DEVELOPMENT_YEARS),
        "horizons": list(HORIZONS),
        "models": sorted(metrics["model"].unique().tolist()) if not metrics.empty else [],
        "joint_models": sorted(joint_frame["model"].unique().tolist()) if not joint_frame.empty else [],
        "model_selection_performed": False,
        "portfolio_selection_performed": False,
        "profit_claim_allowed": False,
        "all_market_rows_invariance_verified": True,
        "output_files": {
            "marginal_metrics": str(output / "marginal_metrics.csv"),
            "marginal_summary": str(output / "marginal_summary.csv"),
            "joint_metrics": str(output / "joint_metrics.csv"),
            "marginal_forecasts": str(output / "marginal_forecasts.parquet"),
            "marginal_comparisons": str(output / "marginal_comparisons.csv"),
            "marginal_reality_check": str(output / "marginal_reality_check.csv"),
            "joint_comparisons": str(output / "joint_comparisons.csv"),
            "joint_reality_check": str(output / "joint_reality_check.csv"),
            "panel_summary": str(output / "panel_summary.json"),
            "state_diagnostics": str(output / "state_diagnostics.json"),
            "volatility_metrics": str(output / "volatility_metrics.csv"),
        },
    }
    (output / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, default=_json_default) + "\n",
        encoding="utf-8",
    )
    return summary


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Candidate-neutral market distribution experiment.")
    parser.add_argument("--input-manifest", default=str(DEFAULT_INPUT_MANIFEST))
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT))
    parser.add_argument("--run", action="store_true", help="run the bounded OOS experiment")
    parser.add_argument("--status", action="store_true", help="show the input contract")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.run:
        result = run_experiment(input_manifest=args.input_manifest, output_root=args.output_root)
    else:
        manifest = _read_json(Path(args.input_manifest))
        result = {
            "status": "ready",
            "input_manifest": str(Path(args.input_manifest).resolve()),
            "input_status": manifest.get("status"),
            "row_count": manifest.get("row_count"),
            "input_fingerprint": manifest.get("input_fingerprint"),
            "training_performed": manifest.get("training_performed"),
            "feature_set_selected": manifest.get("feature_set_selected"),
            "horizons": list(HORIZONS),
            "development_years": list(DEVELOPMENT_YEARS),
            "oos_years": list(OOS_YEARS),
            "model_selection_performed": False,
        }
    print(json.dumps(result, ensure_ascii=False, indent=2, default=_json_default))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
