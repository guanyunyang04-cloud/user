from __future__ import annotations

import numpy as np
import pandas as pd

from daily_research.baseline.advanced_ml_runtime import HistoryWindow
from daily_research.continuous_policy.state_builder import (
    ALPHA_PRIOR_FRAME_NAMES,
    STATE_SEQUENCE_BASES,
    STATE_SEQUENCE_LAGS,
    PreparedPolicyInputs,
)


def make_prepared_policy_inputs(
    *,
    days: int = 34,
    stocks: tuple[str, ...] = ("AAA", "BBB", "CCC", "DDD"),
    start_date: str = "2024-01-02",
) -> PreparedPolicyInputs:
    dates = pd.bdate_range(str(start_date), periods=int(days))
    columns = list(stocks)
    base = np.arange(len(dates), dtype=float).reshape(-1, 1)
    multipliers = np.linspace(1.0, 1.6, len(columns)).reshape(1, -1)
    open_values = 10.0 + base * multipliers + np.arange(len(columns), dtype=float).reshape(1, -1)
    close_values = open_values * (1.0 + 0.001 * multipliers)
    open_ = pd.DataFrame(open_values, index=dates, columns=columns)
    close = pd.DataFrame(close_values, index=dates, columns=columns)
    high = close * 1.01
    low = open_ * 0.99
    volume_values = 100000.0 + base * 100.0 + np.arange(len(columns), dtype=float).reshape(1, -1) * 1000.0
    volume = pd.DataFrame(volume_values, index=dates, columns=columns)
    amount = volume * close
    benchmark_open = pd.Series(100.0 + np.arange(len(dates), dtype=float) * 0.75, index=dates)
    benchmark_close = benchmark_open * 1.001
    membership = pd.DataFrame(True, index=dates, columns=columns)
    score_none = pd.DataFrame(0.0, index=dates, columns=columns)
    score_v2 = pd.DataFrame(np.tile(np.linspace(0.1, 0.4, len(columns)), (len(dates), 1)), index=dates, columns=columns)
    score_blend = score_v2.copy()
    feature_frames = {
        name: pd.DataFrame(0.0, index=dates, columns=columns)
        for name in ("z_score_none", "z_score_v2", "adv20_rank", "price_rank", "ma20_gap", "ma60_gap", "volume_rank")
    }
    derived_names = [
        "ret_1d",
        "ret_3d",
        "ret_5d",
        "ret_10d",
        "ret_20d",
        "vol_5d",
        "vol_20d",
        "score_delta_1d",
        "score_delta_5d",
        "score_delta_accel",
        "ret_accel_5_20",
        "volume_ratio_5_20",
        "distance_to_20d_high",
        "distance_to_60d_high",
        "distance_to_20d_low",
        "volatility_expansion",
        "adv_ratio_5_20",
        *ALPHA_PRIOR_FRAME_NAMES,
        *[f"{base_name}_lag{lag}" for base_name in STATE_SEQUENCE_BASES for lag in STATE_SEQUENCE_LAGS],
    ]
    derived_frames = {
        name: pd.DataFrame(0.0, index=dates, columns=columns)
        for name in derived_names
    }
    for horizon in (1, 3, 5, 10, 20):
        derived_frames[f"ret_{horizon}d"] = close.pct_change(horizon).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    derived_frames["vol_5d"] = close.pct_change().rolling(5).std().fillna(0.0)
    derived_frames["vol_20d"] = close.pct_change().rolling(20).std().fillna(0.0)
    return PreparedPolicyInputs(
        universe=tuple(columns),
        pool_name="fixture",
        benchmark="BENCH",
        data_source="fixture",
        csv_folder="",
        start_date=dates.min().strftime("%Y%m%d"),
        end_date=dates.max().strftime("%Y%m%d"),
        requested_start_date=dates.min().strftime("%Y%m%d"),
        history_window=HistoryWindow(
            mode="fixture",
            requested_start_date=dates.min().strftime("%Y%m%d"),
            effective_start_date=dates.min().strftime("%Y%m%d"),
            end_date=dates.max().strftime("%Y%m%d"),
            required_trading_days=0,
        ),
        raw_cache_meta={},
        prepared_cache_meta={},
        close=close,
        open_=open_,
        high=high,
        low=low,
        volume=volume,
        amount=amount,
        benchmark_close=benchmark_close,
        benchmark_open=benchmark_open,
        score_none=score_none,
        score_v2=score_v2,
        score_blend=score_blend,
        feature_frames=feature_frames,
        market_features={},
        membership_frame=membership,
        rolling_pool_summary={},
        alpha_prior_summary={},
        derived_frames=derived_frames,
    )
