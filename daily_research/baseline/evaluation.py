from __future__ import annotations

from typing import Dict, Iterable, Tuple

import numpy as np
import pandas as pd


FORWARD_HORIZONS = (1, 5, 10, 20)


def compute_forward_returns(close: pd.DataFrame, horizons: Iterable[int] = FORWARD_HORIZONS) -> Dict[str, pd.DataFrame]:
    forward_returns: Dict[str, pd.DataFrame] = {}
    for horizon in horizons:
        label = f"fwd_{horizon}d"
        forward_returns[label] = close.shift(-horizon).div(close).sub(1.0)
    return forward_returns


def _rank_corr(x: pd.Series, y: pd.Series) -> float:
    pair = pd.concat([x, y], axis=1).dropna()
    if len(pair) < 2:
        return np.nan
    if pair.iloc[:, 0].nunique() <= 1 or pair.iloc[:, 1].nunique() <= 1:
        return np.nan
    xr = pair.iloc[:, 0].rank(method="average")
    yr = pair.iloc[:, 1].rank(method="average")
    return float(xr.corr(yr))


def _pearson_corr(x: pd.Series, y: pd.Series) -> float:
    pair = pd.concat([x, y], axis=1).dropna()
    if len(pair) < 2:
        return np.nan
    if pair.iloc[:, 0].nunique() <= 1 or pair.iloc[:, 1].nunique() <= 1:
        return np.nan
    return float(pair.iloc[:, 0].corr(pair.iloc[:, 1]))


def _cross_section_quantiles(row: pd.Series, quantiles: int) -> pd.Series:
    valid = row.dropna()
    if len(valid) < quantiles:
        return pd.Series(np.nan, index=row.index)
    try:
        labels = list(range(1, quantiles + 1))
        binned = pd.qcut(valid.rank(method="first"), quantiles, labels=labels)
        out = pd.Series(np.nan, index=row.index)
        out.loc[valid.index] = binned.astype(float)
        return out
    except ValueError:
        return pd.Series(np.nan, index=row.index)


def summarize_factor_ic(
    factor_frames: Dict[str, pd.DataFrame],
    forward_returns: Dict[str, pd.DataFrame],
) -> pd.DataFrame:
    rows = []
    for factor_name, factor_df in factor_frames.items():
        for horizon_name, ret_df in forward_returns.items():
            aligned_factor, aligned_ret = factor_df.align(ret_df, join="inner", axis=0)
            ic_series = aligned_factor.apply(
                lambda row: _pearson_corr(row, aligned_ret.loc[row.name]), axis=1
            )
            rank_ic_series = aligned_factor.apply(
                lambda row: _rank_corr(row, aligned_ret.loc[row.name]), axis=1
            )
            ic_mean = float(ic_series.mean()) if not ic_series.dropna().empty else np.nan
            rank_ic_mean = float(rank_ic_series.mean()) if not rank_ic_series.dropna().empty else np.nan
            ic_std = float(ic_series.std()) if len(ic_series.dropna()) > 1 else np.nan
            rank_ic_std = float(rank_ic_series.std()) if len(rank_ic_series.dropna()) > 1 else np.nan
            rows.append(
                {
                    "factor": factor_name,
                    "horizon": horizon_name,
                    "ic": ic_mean,
                    "rank_ic": rank_ic_mean,
                    "icir": float(ic_mean / ic_std * np.sqrt(252)) if ic_std and not np.isnan(ic_std) and ic_std > 0 else np.nan,
                    "rank_icir": float(rank_ic_mean / rank_ic_std * np.sqrt(252))
                    if rank_ic_std and not np.isnan(rank_ic_std) and rank_ic_std > 0
                    else np.nan,
                    "sample_days": int(ic_series.notna().sum()),
                }
            )
    return pd.DataFrame(rows)


def summarize_quantile_returns(
    factor_frames: Dict[str, pd.DataFrame],
    forward_returns: Dict[str, pd.DataFrame],
    quantiles: int = 5,
) -> pd.DataFrame:
    rows = []
    for factor_name, factor_df in factor_frames.items():
        quantile_map = factor_df.apply(lambda row: _cross_section_quantiles(row, quantiles), axis=1)
        for horizon_name, ret_df in forward_returns.items():
            aligned_quantile, aligned_ret = quantile_map.align(ret_df, join="inner", axis=0)
            per_quantile_returns = {q: [] for q in range(1, quantiles + 1)}
            long_short_returns = []
            for dt in aligned_quantile.index:
                q_row = aligned_quantile.loc[dt]
                r_row = aligned_ret.loc[dt]
                q_ret = {}
                for quantile in range(1, quantiles + 1):
                    mask = q_row == float(quantile)
                    ret = r_row[mask].dropna()
                    if ret.empty:
                        q_ret[quantile] = np.nan
                    else:
                        mean_ret = float(ret.mean())
                        q_ret[quantile] = mean_ret
                        per_quantile_returns[quantile].append(mean_ret)
                if not np.isnan(q_ret.get(quantiles, np.nan)) and not np.isnan(q_ret.get(1, np.nan)):
                    long_short_returns.append(q_ret[quantiles] - q_ret[1])
            for quantile in range(1, quantiles + 1):
                values = per_quantile_returns[quantile]
                rows.append(
                    {
                        "factor": factor_name,
                        "horizon": horizon_name,
                        "bucket": f"Q{quantile}",
                        "mean_return": float(np.mean(values)) if values else np.nan,
                        "sample_days": int(len(values)),
                    }
                )
            rows.append(
                {
                    "factor": factor_name,
                    "horizon": horizon_name,
                    "bucket": f"Q{quantiles}-Q1",
                    "mean_return": float(np.mean(long_short_returns)) if long_short_returns else np.nan,
                    "sample_days": int(len(long_short_returns)),
                }
            )
    return pd.DataFrame(rows)


def evaluate_factor_bundle(
    raw_factors: Dict[str, pd.DataFrame],
    score: pd.DataFrame,
    close: pd.DataFrame,
    quantiles: int = 5,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    factor_frames = dict(raw_factors)
    factor_frames["composite_score"] = score
    forward_returns = compute_forward_returns(close)
    factor_ic_summary = summarize_factor_ic(factor_frames, forward_returns)
    factor_quantile_returns = summarize_quantile_returns(
        factor_frames,
        forward_returns,
        quantiles=quantiles,
    )
    return factor_ic_summary, factor_quantile_returns
