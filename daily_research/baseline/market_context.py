from __future__ import annotations

from typing import Dict

import numpy as np
import pandas as pd


def _align_mask(frame: pd.DataFrame, membership_mask: pd.DataFrame | None) -> pd.DataFrame:
    if membership_mask is None:
        return pd.DataFrame(True, index=frame.index, columns=frame.columns, dtype=bool)
    return membership_mask.reindex(index=frame.index, columns=frame.columns).fillna(False).astype(bool)


def _masked_share(condition: pd.DataFrame, valid_mask: pd.DataFrame) -> pd.Series:
    numer = condition.where(valid_mask).astype(float).sum(axis=1, min_count=1)
    denom = valid_mask.sum(axis=1).replace(0, np.nan)
    return numer.div(denom)


def _rolling_percentile(series: pd.Series, window: int, min_periods: int | None = None) -> pd.Series:
    series = series.astype(float)
    lookback = max(int(window), 5)
    min_obs = int(min_periods) if min_periods is not None else max(40, lookback // 3)

    def _last_rank_pct(values: np.ndarray) -> float:
        arr = np.asarray(values, dtype=float)
        if arr.size == 0 or not np.isfinite(arr[-1]):
            return float("nan")
        valid = arr[np.isfinite(arr)]
        if valid.size < min_obs:
            return float("nan")
        last = valid[-1]
        less = float((valid < last).sum())
        equal = float((valid == last).sum())
        return (less + 0.5 * equal) / float(valid.size)

    return series.rolling(lookback, min_periods=min_obs).apply(_last_rank_pct, raw=True)


def compute_continuous_market_context(
    df_dict: Dict[str, pd.DataFrame],
    regime_state: pd.DataFrame,
    membership_mask: pd.DataFrame | None = None,
) -> pd.DataFrame:
    close = df_dict["Close"].astype(float).copy()
    amount = df_dict["Amount"].astype(float).reindex_like(close)

    close.index = pd.to_datetime(close.index)
    amount.index = pd.to_datetime(amount.index)
    close = close.sort_index()
    amount = amount.sort_index()

    mask = _align_mask(close, membership_mask)
    ema20 = close.ewm(span=20, adjust=False).mean()
    ema60 = close.ewm(span=60, adjust=False).mean()
    returns_1d = close.pct_change(fill_method=None)
    returns_20d = close.div(close.shift(20)).sub(1.0)

    valid_price = mask & close.notna()
    valid_ma20 = valid_price & ema20.notna()
    valid_ma60 = valid_price & ema60.notna()
    valid_ret_1d = valid_price & returns_1d.notna()
    valid_ret_20d = valid_price & returns_20d.notna()

    breadth_ma20 = _masked_share(close.gt(ema20), valid_ma20)
    breadth_ma60 = _masked_share(close.gt(ema60), valid_ma60)
    adv_ratio = _masked_share(returns_1d.gt(0.0), valid_ret_1d)
    dec_ratio = _masked_share(returns_1d.lt(0.0), valid_ret_1d)
    advance_decline_balance = adv_ratio - dec_ratio
    down_gt3_ratio = _masked_share(returns_1d.le(-0.03), valid_ret_1d)
    dispersion_1d = returns_1d.where(valid_ret_1d).std(axis=1)
    dispersion_20d = returns_20d.where(valid_ret_20d).std(axis=1)

    pool_amount = amount.where(mask).sum(axis=1, min_count=1)
    liquidity_ratio_5_20 = pool_amount.rolling(5).mean().div(pool_amount.rolling(20).mean()).sub(1.0)

    benchmark_trend_gap = regime_state["benchmark_close"].div(regime_state["benchmark_ma"]).sub(1.0)
    benchmark_annual_vol = regime_state["benchmark_annual_vol"].astype(float)
    benchmark_vol_percentile = _rolling_percentile(benchmark_annual_vol, 126)

    score_trend = _rolling_percentile(benchmark_trend_gap, 252)
    score_vol = 1.0 - benchmark_vol_percentile
    score_breadth20 = _rolling_percentile(breadth_ma20, 252)
    score_breadth60 = _rolling_percentile(breadth_ma60, 252)
    score_advance_decline = _rolling_percentile(advance_decline_balance, 252)
    score_dispersion = 1.0 - _rolling_percentile(dispersion_20d, 252)
    score_liquidity = _rolling_percentile(liquidity_ratio_5_20, 252)
    score_stress = 1.0 - _rolling_percentile(down_gt3_ratio, 252)

    component_scores = pd.concat(
        [
            score_trend.rename("score_trend"),
            score_vol.rename("score_vol"),
            score_breadth20.rename("score_breadth20"),
            score_breadth60.rename("score_breadth60"),
            score_advance_decline.rename("score_advance_decline"),
            score_dispersion.rename("score_dispersion"),
            score_liquidity.rename("score_liquidity"),
            score_stress.rename("score_stress"),
        ],
        axis=1,
    )
    context_score = component_scores.mean(axis=1, skipna=True)

    out = pd.DataFrame(
        {
            "benchmark_trend_gap": benchmark_trend_gap,
            "benchmark_annual_vol": benchmark_annual_vol,
            "benchmark_vol_percentile_126": benchmark_vol_percentile,
            "breadth_ma20": breadth_ma20,
            "breadth_ma60": breadth_ma60,
            "advance_decline_balance": advance_decline_balance,
            "down_gt3_ratio": down_gt3_ratio,
            "dispersion_1d": dispersion_1d,
            "dispersion_20d": dispersion_20d,
            "liquidity_ratio_5_20": liquidity_ratio_5_20,
            "score_trend": score_trend,
            "score_vol": score_vol,
            "score_breadth20": score_breadth20,
            "score_breadth60": score_breadth60,
            "score_advance_decline": score_advance_decline,
            "score_dispersion": score_dispersion,
            "score_liquidity": score_liquidity,
            "score_stress": score_stress,
            "context_score": context_score,
            "pool_member_count": mask.sum(axis=1),
        }
    )
    return out.reindex(regime_state.index)
