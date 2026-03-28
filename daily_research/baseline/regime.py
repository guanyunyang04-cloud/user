from __future__ import annotations

from typing import Tuple

import numpy as np
import pandas as pd

from daily_research.baseline.config import ResearchConfig


REGIME_ENGINE_VERSION = 2
REGIME_SELECTOR_COLUMNS: tuple[str, ...] = (
    "quadrant",
    "market_state",
    "trend_bucket",
    "vol_bucket",
)
LEGACY_QUADRANT_LABELS: tuple[str, ...] = (
    "trend_up_low_vol",
    "trend_up_high_vol",
    "trend_down_low_vol",
    "trend_down_high_vol",
)
TREND_BUCKET_LABELS: tuple[str, ...] = (
    "trend_down",
    "trend_flat",
    "trend_up",
)
VOL_BUCKET_LABELS: tuple[str, ...] = (
    "vol_low",
    "vol_mid",
    "vol_high",
)
MARKET_STATE_LABELS: tuple[str, ...] = tuple(
    f"{trend}_{vol}"
    for trend in TREND_BUCKET_LABELS
    for vol in VOL_BUCKET_LABELS
)


def _classify_trend_bucket(
    trend_gap: pd.Series,
    flat_band: float,
) -> pd.Series:
    bucket = pd.Series("trend_flat", index=trend_gap.index, dtype="object")
    band = max(float(flat_band), 0.0)
    bucket.loc[trend_gap > band] = "trend_up"
    bucket.loc[trend_gap < -band] = "trend_down"
    bucket.loc[trend_gap.isna()] = np.nan
    return bucket


def _classify_vol_bucket(
    annual_vol: pd.Series,
    vol_threshold: float,
    transition_band: float,
) -> pd.Series:
    threshold = float(vol_threshold)
    band = max(float(transition_band), 0.0)
    low_cutoff = threshold * max(0.0, 1.0 - band)
    high_cutoff = threshold * (1.0 + band)

    bucket = pd.Series("vol_mid", index=annual_vol.index, dtype="object")
    bucket.loc[annual_vol <= low_cutoff] = "vol_low"
    bucket.loc[annual_vol > high_cutoff] = "vol_high"
    bucket.loc[annual_vol.isna()] = np.nan
    return bucket


def _compose_market_state(
    trend_bucket: pd.Series,
    vol_bucket: pd.Series,
) -> pd.Series:
    market_state = trend_bucket.astype("object").str.cat(vol_bucket.astype("object"), sep="_")
    missing_mask = trend_bucket.isna() | vol_bucket.isna()
    market_state.loc[missing_mask] = np.nan
    return market_state


def _normalize_allowed_regime_labels(config: ResearchConfig) -> set[str]:
    labels = {
        str(name).strip().lower()
        for name in getattr(config, "regime_allowed_quadrants", [])
        if str(name).strip()
    }
    if not labels:
        labels = {"trend_up_low_vol"}
    return labels


def normalize_regime_state_selector(selector: str | None) -> str:
    normalized = str(selector or "quadrant").strip().lower()
    if normalized not in REGIME_SELECTOR_COLUMNS:
        raise ValueError(
            "regime_state_selector must be one of: quadrant, market_state, trend_bucket, vol_bucket."
        )
    return normalized


def resolve_regime_label_series(
    regime_state: pd.DataFrame,
    selector: str | None = "quadrant",
) -> pd.Series:
    normalized = normalize_regime_state_selector(selector)
    if normalized not in regime_state.columns:
        raise KeyError(f"Regime state does not contain selector column: {normalized}")
    return regime_state[normalized].copy()


def _resolve_regime_on_mask(
    *,
    quadrant: pd.Series,
    market_state: pd.Series,
    trend_bucket: pd.Series,
    vol_bucket: pd.Series,
    allowed_labels: set[str],
) -> pd.Series:
    selector_frame = pd.DataFrame(
        {
            "quadrant": quadrant.astype("string").str.lower(),
            "market_state": market_state.astype("string").str.lower(),
            "trend_bucket": trend_bucket.astype("string").str.lower(),
            "vol_bucket": vol_bucket.astype("string").str.lower(),
        }
    )
    selector_mask = pd.Series(False, index=selector_frame.index, dtype=bool)
    for column_name in selector_frame.columns:
        selector_mask = selector_mask | selector_frame[column_name].isin(allowed_labels).fillna(False)
    return selector_mask


def compute_market_regime_state(
    benchmark_close: pd.Series,
    config: ResearchConfig,
) -> pd.DataFrame:
    benchmark_close = benchmark_close.astype(float).sort_index().dropna()
    ma = benchmark_close.rolling(config.regime_ma_window).mean()
    returns = benchmark_close.pct_change(fill_method=None)
    annual_vol = returns.rolling(config.regime_vol_window).std() * np.sqrt(252)
    regime_ready = ma.notna() & annual_vol.notna()

    benchmark_trend_gap = benchmark_close.div(ma).sub(1.0)
    benchmark_vol_gap = annual_vol.sub(float(config.regime_max_annual_vol))
    benchmark_vol_ratio = annual_vol.div(float(config.regime_max_annual_vol)).replace([np.inf, -np.inf], np.nan)

    trend_pass = (benchmark_close > ma).astype("boolean")
    vol_pass = (annual_vol <= float(config.regime_max_annual_vol)).astype("boolean")
    trend_up = trend_pass.fillna(False).astype(bool)
    low_vol = vol_pass.fillna(False).astype(bool)

    quadrant = pd.Series(index=benchmark_close.index, dtype="object")
    quadrant.loc[trend_up & low_vol] = "trend_up_low_vol"
    quadrant.loc[trend_up & ~low_vol] = "trend_up_high_vol"
    quadrant.loc[~trend_up & low_vol] = "trend_down_low_vol"
    quadrant.loc[~trend_up & ~low_vol] = "trend_down_high_vol"
    quadrant.loc[~regime_ready] = np.nan

    trend_bucket = _classify_trend_bucket(
        benchmark_trend_gap,
        flat_band=float(config.regime_trend_flat_band),
    )
    vol_bucket = _classify_vol_bucket(
        annual_vol,
        vol_threshold=float(config.regime_max_annual_vol),
        transition_band=float(config.regime_vol_transition_band),
    )
    market_state = _compose_market_state(trend_bucket, vol_bucket)

    allowed_labels = _normalize_allowed_regime_labels(config)
    regime_on = _resolve_regime_on_mask(
        quadrant=quadrant,
        market_state=market_state,
        trend_bucket=trend_bucket,
        vol_bucket=vol_bucket,
        allowed_labels=allowed_labels,
    ).fillna(False) & regime_ready.fillna(False)

    return pd.DataFrame(
        {
            "benchmark_close": benchmark_close,
            "benchmark_ma": ma,
            "benchmark_trend_gap": benchmark_trend_gap,
            "benchmark_annual_vol": annual_vol,
            "benchmark_vol_gap": benchmark_vol_gap,
            "benchmark_vol_ratio": benchmark_vol_ratio,
            "regime_ready": regime_ready.astype(bool),
            "trend_pass": trend_pass,
            "vol_pass": vol_pass,
            "trend_bucket": trend_bucket,
            "vol_bucket": vol_bucket,
            "market_state": market_state,
            "quadrant": quadrant,
            "regime_on": regime_on,
        }
    )


def apply_market_regime_filter(
    target_weights: pd.DataFrame,
    target_scores: pd.DataFrame,
    regime_state: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    regime_on = regime_state["regime_on"].reindex(target_weights.index)
    regime_on = regime_on.astype("boolean").fillna(False).astype(bool)
    filtered_weights = target_weights.mul(regime_on.astype(float), axis=0)
    filtered_scores = target_scores.where(regime_on, 0.0)
    return filtered_weights, filtered_scores
