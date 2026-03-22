from typing import Dict

import numpy as np
import pandas as pd


def _zscore_cs(df: pd.DataFrame) -> pd.DataFrame:
    mean = df.mean(axis=1)
    std = df.std(axis=1).replace(0, np.nan)
    return df.sub(mean, axis=0).div(std, axis=0)


def _safe_div(a: pd.DataFrame, b: pd.DataFrame) -> pd.DataFrame:
    return a.div(b.replace(0, np.nan))


def _rolling_sum_bool(condition: pd.DataFrame, window: int) -> pd.DataFrame:
    return condition.astype(float).rolling(window).sum()


def _cross_over(a: pd.DataFrame, b: pd.DataFrame) -> pd.DataFrame:
    return (a > b) & (a.shift(1) <= b.shift(1))


def _adaptive_dma(price_df: pd.DataFrame, alpha_df: pd.DataFrame) -> pd.DataFrame:
    price = price_df.to_numpy(dtype=float)
    alpha = alpha_df.to_numpy(dtype=float)
    alpha = np.clip(np.nan_to_num(alpha, nan=0.0, posinf=1.0, neginf=0.0), 0.0, 1.0)

    if price.size == 0:
        return pd.DataFrame(index=price_df.index, columns=price_df.columns, dtype=float)

    out = np.full_like(price, np.nan, dtype=float)
    out[0] = price[0]
    for t in range(1, price.shape[0]):
        prev = out[t - 1]
        curr = price[t]
        a_t = alpha[t]
        res = prev.copy()

        curr_valid = np.isfinite(curr)
        prev_valid = np.isfinite(prev)

        use_mask = curr_valid & prev_valid
        res[use_mask] = a_t[use_mask] * curr[use_mask] + (1.0 - a_t[use_mask]) * prev[use_mask]

        init_mask = curr_valid & ~prev_valid
        res[init_mask] = curr[init_mask]

        out[t] = res

    return pd.DataFrame(out, index=price_df.index, columns=price_df.columns)


def compute_factors(df_dict: Dict[str, pd.DataFrame]) -> Dict[str, Dict[str, pd.DataFrame]]:
    open_df = df_dict["Open"].copy()
    high_df = df_dict["High"].copy()
    low_df = df_dict["Low"].copy()
    close_df = df_dict["Close"].copy()
    volume_df = df_dict["Volume"].copy()
    amount_df = df_dict["Amount"].copy()

    returns = close_df.pct_change(fill_method=None)
    ema10 = close_df.ewm(span=10, adjust=False).mean()
    ema20 = close_df.ewm(span=20, adjust=False).mean()
    ema60 = close_df.ewm(span=60, adjust=False).mean()
    ema12 = close_df.ewm(span=12, adjust=False).mean()
    ema26 = close_df.ewm(span=26, adjust=False).mean()
    dea = (ema12 - ema26).ewm(span=9, adjust=False).mean()

    rolling_high_20 = close_df.rolling(20).max()
    rolling_low_20 = close_df.rolling(20).min()

    tr = pd.concat(
        [
            (high_df - low_df).stack(future_stack=True),
            (high_df - close_df.shift(1)).abs().stack(future_stack=True),
            (low_df - close_df.shift(1)).abs().stack(future_stack=True),
        ],
        axis=1,
    ).max(axis=1)
    tr = tr.unstack()
    atr14 = tr.rolling(14).mean()
    mid20 = close_df.ewm(span=20, adjust=False).mean()
    upp = mid20 + 2.0 * atr14
    lowr = mid20 - 2.0 * atr14

    volatility_20_raw = returns.rolling(20).std()
    volatility_contraction_raw = _safe_div(
        volatility_20_raw.rolling(5).mean(),
        volatility_20_raw.rolling(20).mean(),
    ) - 1.0

    d1 = (close_df - close_df.shift(10)).abs()
    v1 = (close_df - close_df.shift(1)).abs().rolling(10).sum()
    er1 = _safe_div(d1, v1).fillna(0.0)
    cs1 = er1 * (2.0 / 3.0 - 2.0 / 31.0) + 2.0 / 31.0
    cq1 = cs1 * cs1
    kbas = _adaptive_dma(close_df, cq1)
    kama = kbas.ewm(span=2, adjust=False).mean()
    bull = ema60
    bup = bull >= bull.shift(1)

    volume_ma5 = volume_df.rolling(5).mean()
    volume_ma10 = volume_df.rolling(10).mean()
    volume_ma20 = volume_df.rolling(20).mean()
    vrat = _safe_div(volume_df, volume_ma5.shift(1)).replace([np.inf, -np.inf], np.nan).fillna(1.0)
    vup = (volume_df > volume_ma5) & (volume_df > volume_ma10) & (vrat > 1.5)

    cmid_proxy = _safe_div(amount_df.rolling(60).sum(), volume_df.rolling(60).sum())
    dif = ema12 - ema26
    mbar = (dif - dea) * 2.0
    maca = (mbar > mbar.shift(1)) & (dif > dif.shift(1))
    bias = (close_df - kama) / kama.replace(0, np.nan) * 100.0

    long_regime = (close_df > kama) & (kama >= kama.shift(1)) & bup
    yang = close_df > open_df
    yin = close_df <= open_df
    prev_yin = (close_df.shift(1) <= open_df.shift(1)) & close_df.shift(1).notna() & open_df.shift(1).notna()

    cross_close_kama = _cross_over(close_df, kama)
    cross_close_cmid = _cross_over(close_df, cmid_proxy)
    msup = cross_close_kama & bup & vup & (close_df > cmid_proxy)
    mrun = cross_close_kama & maca & (bias < 5.0) & vup & (close_df > cmid_proxy) & bup
    mbuy = msup | mrun

    rvol_proxy = (volume_df > volume_ma20 * 1.5) & (vrat > 1.5)
    zjtp = cross_close_cmid & rvol_proxy & yang & long_regime
    hcw = long_regime & (bias > 0.0) & (bias < 3.0) & yang & prev_yin & (close_df > cmid_proxy)

    raw_factors: Dict[str, pd.DataFrame] = {}
    raw_factors["mom_5"] = close_df / close_df.shift(5) - 1.0
    raw_factors["mom_20"] = close_df / close_df.shift(20) - 1.0
    raw_factors["mom_60"] = close_df / close_df.shift(60) - 1.0
    raw_factors["ma_gap_10"] = close_df / ema10 - 1.0
    raw_factors["ma_gap_20_60"] = ema20 / ema60 - 1.0
    raw_factors["trend_slope_20"] = ema20 / ema20.shift(5) - 1.0
    raw_factors["breakout_20"] = close_df / rolling_high_20 - 1.0

    raw_factors["vol_ratio_5_20"] = _safe_div(volume_df.rolling(5).mean(), volume_df.rolling(20).mean()) - 1.0
    raw_factors["breakout_volume"] = (raw_factors["breakout_20"].clip(lower=0.0) + 1.0) * (
        raw_factors["vol_ratio_5_20"].clip(lower=-1.0) + 1.0
    ) - 1.0
    raw_factors["volume_contraction"] = -raw_factors["vol_ratio_5_20"]
    raw_factors["price_volume_divergence"] = raw_factors["mom_20"] - raw_factors["vol_ratio_5_20"]

    raw_factors["atr_14_pct"] = -(atr14 / close_df.replace(0, np.nan))
    raw_factors["volatility_20"] = -volatility_20_raw
    raw_factors["volatility_contraction"] = -volatility_contraction_raw

    raw_factors["range_position_20"] = (close_df - rolling_low_20) / (rolling_high_20 - rolling_low_20).replace(0, np.nan)
    raw_factors["drawdown_20"] = close_df / rolling_high_20 - 1.0
    raw_factors["close_strength"] = (close_df - low_df) / (high_df - low_df).replace(0, np.nan)
    raw_factors["body_strength"] = (close_df - open_df) / close_df.replace(0, np.nan)
    raw_factors["up_day_ratio_10"] = (returns > 0).astype(float).rolling(10).mean()
    raw_factors["trend_streak"] = _rolling_sum_bool(returns > 0, 10) - _rolling_sum_bool(returns < 0, 10)
    raw_factors["kama_gap"] = close_df / kama.replace(0, np.nan) - 1.0
    raw_factors["kama_slope"] = kama / kama.shift(5) - 1.0
    raw_factors["long_regime_flag"] = long_regime.astype(float)
    raw_factors["mbuy_flag"] = mbuy.astype(float)
    raw_factors["zjtp_flag"] = zjtp.astype(float)
    raw_factors["hcw_flag"] = hcw.astype(float)

    zscore_factors = {name: _zscore_cs(value) for name, value in raw_factors.items()}

    return {
        "raw_factors": raw_factors,
        "zscore_factors": zscore_factors,
        "raw_inputs": {
            "Open": open_df,
            "High": high_df,
            "Low": low_df,
            "Close": close_df,
            "Volume": volume_df,
            "Amount": amount_df,
            "KAMA": kama,
            "CMID_PROXY": cmid_proxy,
            "LOWR": lowr,
            "UPP": upp,
        },
    }
