from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.mixture import GaussianMixture


@dataclass
class MarketStateArtifact:
    features: pd.DataFrame
    state_series: pd.Series
    state_names: dict[int, str]
    model: GaussianMixture


def build_market_state_features(benchmark_close: pd.Series) -> pd.DataFrame:
    close = pd.Series(benchmark_close).dropna().astype(float)
    ret_5 = close.pct_change(5)
    ret_20 = close.pct_change(20)
    ma_20 = close.rolling(20).mean()
    ma_60 = close.rolling(60).mean()
    trend_gap_20 = close.div(ma_20).sub(1.0)
    trend_gap_60 = close.div(ma_60).sub(1.0)
    vol_20 = close.pct_change().rolling(20).std() * np.sqrt(252.0)
    vol_60 = close.pct_change().rolling(60).std() * np.sqrt(252.0)
    drawdown_60 = close.div(close.rolling(60).max()).sub(1.0)
    df = pd.DataFrame(
        {
            "ret_5": ret_5,
            "ret_20": ret_20,
            "trend_gap_20": trend_gap_20,
            "trend_gap_60": trend_gap_60,
            "vol_20": vol_20,
            "vol_60": vol_60,
            "drawdown_60": drawdown_60,
        }
    ).dropna()
    return df


def _name_states(features: pd.DataFrame, states: pd.Series) -> dict[int, str]:
    summary = features.assign(state=states).groupby("state").agg(
        trend_gap_60=("trend_gap_60", "mean"),
        vol_20=("vol_20", "mean"),
    )
    trend_med = float(summary["trend_gap_60"].median())
    vol_med = float(summary["vol_20"].median())
    names: dict[int, str] = {}
    for state, row in summary.iterrows():
        trend_name = "trend_up" if float(row["trend_gap_60"]) >= trend_med else "trend_down"
        vol_name = "high_vol" if float(row["vol_20"]) >= vol_med else "low_vol"
        names[int(state)] = f"{trend_name}_{vol_name}"
    return names


def fit_market_state_model(
    benchmark_close: pd.Series,
    n_states: int = 4,
    random_seed: int = 7,
) -> MarketStateArtifact:
    features = build_market_state_features(benchmark_close)
    model = GaussianMixture(
        n_components=n_states,
        covariance_type="full",
        random_state=random_seed,
        reg_covar=1e-5,
    )
    state_values = model.fit_predict(features.values)
    state_series = pd.Series(state_values, index=features.index, name="state_id")
    state_names = _name_states(features, state_series)
    return MarketStateArtifact(
        features=features,
        state_series=state_series,
        state_names=state_names,
        model=model,
    )


def build_state_frame(artifact: MarketStateArtifact) -> pd.DataFrame:
    out = artifact.features.copy()
    out["state_id"] = artifact.state_series.astype(int)
    out["state_name"] = out["state_id"].map(artifact.state_names)
    return out

