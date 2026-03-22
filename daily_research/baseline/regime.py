from typing import Tuple

import numpy as np
import pandas as pd

from daily_research.baseline.config import ResearchConfig


def compute_market_regime_state(
    benchmark_close: pd.Series,
    config: ResearchConfig,
) -> pd.DataFrame:
    benchmark_close = benchmark_close.astype(float).sort_index().dropna()
    ma = benchmark_close.rolling(config.regime_ma_window).mean()
    returns = benchmark_close.pct_change(fill_method=None)
    annual_vol = returns.rolling(config.regime_vol_window).std() * np.sqrt(252)

    trend_pass = benchmark_close > ma
    vol_pass = annual_vol <= float(config.regime_max_annual_vol)
    trend_up = trend_pass.fillna(False)
    low_vol = vol_pass.fillna(False)

    quadrant = pd.Series(index=benchmark_close.index, dtype="object")
    quadrant.loc[trend_up & low_vol] = "trend_up_low_vol"
    quadrant.loc[trend_up & ~low_vol] = "trend_up_high_vol"
    quadrant.loc[~trend_up & low_vol] = "trend_down_low_vol"
    quadrant.loc[~trend_up & ~low_vol] = "trend_down_high_vol"

    allowed_quadrants = {
        str(name).strip().lower()
        for name in getattr(config, "regime_allowed_quadrants", [])
        if str(name).strip()
    }
    if not allowed_quadrants:
        allowed_quadrants = {"trend_up_low_vol"}

    regime_on = quadrant.isin(allowed_quadrants).fillna(False)

    return pd.DataFrame(
        {
            "benchmark_close": benchmark_close,
            "benchmark_ma": ma,
            "benchmark_annual_vol": annual_vol,
            "trend_pass": trend_up,
            "vol_pass": low_vol,
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
