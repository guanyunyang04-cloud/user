from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

import numpy as np
import pandas as pd
import torch
from torch.utils.data import BatchSampler, Dataset


STRUCTURE_LABELS = [
    "trend_breakout",
    "pullback_rebound",
    "high_vol_expansion",
    "low_vol_trend",
    "weak_structure",
    "neutral_mixed",
]
STRUCTURE_LABEL_TO_ID = {name: idx for idx, name in enumerate(STRUCTURE_LABELS)}
STRUCTURE_ID_TO_LABEL = {idx: name for name, idx in STRUCTURE_LABEL_TO_ID.items()}


def _zscore_2d(arr: np.ndarray) -> np.ndarray:
    mean = np.nanmean(arr, axis=0, keepdims=True)
    std = np.nanstd(arr, axis=0, keepdims=True)
    std[std == 0] = 1.0
    out = (arr - mean) / std
    out = np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)
    return out


def _cross_sectional_rank_frame(frame: pd.DataFrame) -> pd.DataFrame:
    rank = frame.rank(axis=1, pct=True)
    centered = (rank - 0.5) * 2.0
    return centered.fillna(0.0)


def _cross_sectional_zscore_frame(frame: pd.DataFrame) -> pd.DataFrame:
    mean = frame.mean(axis=1)
    std = frame.std(axis=1).replace(0, np.nan)
    z = frame.sub(mean, axis=0).div(std, axis=0)
    return z.fillna(0.0)


def build_liquidity_bucket_frame(
    amount_frame: pd.DataFrame,
    window: int = 20,
    n_buckets: int = 5,
) -> pd.DataFrame:
    adv = amount_frame.astype(float).rolling(window).mean()
    rank = adv.rank(axis=1, pct=True)
    rank_arr = rank.to_numpy(dtype=float)
    bucket_arr = np.floor(np.clip(rank_arr, 0.0, 0.999999) * int(n_buckets))
    bucket_arr[~np.isfinite(rank_arr)] = np.nan
    return pd.DataFrame(bucket_arr, index=adv.index, columns=adv.columns)


def build_structure_label_frame(
    close: pd.DataFrame,
    open_df: pd.DataFrame,
    high: pd.DataFrame,
    low: pd.DataFrame,
    amount: pd.DataFrame,
    benchmark_close: pd.Series,
) -> pd.DataFrame:
    ret_20 = close.pct_change(20, fill_method=None)
    rel_ret_5 = close.pct_change(5, fill_method=None).sub(benchmark_close.pct_change(5, fill_method=None), axis=0)
    ma_20_gap = close.div(close.rolling(20).mean()).sub(1.0)
    ma_60_gap = close.div(close.rolling(60).mean()).sub(1.0)
    amount_ratio_5_20 = amount.rolling(5).mean().div(amount.rolling(20).mean().replace(0, np.nan))
    prev_close = close.shift(1)
    range_pct = high.sub(low).div(prev_close.replace(0, np.nan))
    intraday_body = close.div(open_df.replace(0, np.nan)).sub(1.0)

    high_vol_cut = range_pct.quantile(0.8, axis=1)
    low_vol_cut = range_pct.quantile(0.5, axis=1)
    structure = pd.DataFrame(index=close.index, columns=close.columns, dtype="object")

    trend_breakout = (ret_20 > 0.05) & (ma_20_gap > 0.03) & (amount_ratio_5_20 > 1.0)
    pullback_rebound = (ma_60_gap > 0.0) & (rel_ret_5 < -0.01) & (intraday_body > 0.0)
    high_vol_expansion = range_pct.ge(high_vol_cut, axis=0) & (amount_ratio_5_20 > 1.0)
    low_vol_trend = (ret_20 > 0.03) & (ma_20_gap > 0.0) & range_pct.le(low_vol_cut, axis=0)
    weak_structure = (ma_20_gap < 0.0) | (ret_20 < 0.0)

    structure = structure.mask(trend_breakout, "trend_breakout")
    structure = structure.mask(structure.isna() & pullback_rebound, "pullback_rebound")
    structure = structure.mask(structure.isna() & high_vol_expansion, "high_vol_expansion")
    structure = structure.mask(structure.isna() & low_vol_trend, "low_vol_trend")
    structure = structure.mask(structure.isna() & weak_structure, "weak_structure")
    structure = structure.fillna("neutral_mixed")
    return structure


def transform_return_target_frames(
    target_frames: Dict[str, pd.DataFrame],
    mode: str,
) -> Dict[str, pd.DataFrame]:
    if mode == "raw":
        return {name: frame.copy() for name, frame in target_frames.items()}

    transformed: Dict[str, pd.DataFrame] = {}
    for name, frame in target_frames.items():
        if not name.startswith("fwd_excess_"):
            transformed[name] = frame.copy()
            continue
        if mode == "cs_rank":
            transformed[name] = _cross_sectional_rank_frame(frame)
        elif mode == "cs_zscore":
            transformed[name] = _cross_sectional_zscore_frame(frame)
        else:
            raise ValueError(f"Unsupported return_target_transform: {mode}")
    return transformed


def _group_mean_feature(value_df: pd.DataFrame, group_map: pd.Series) -> pd.DataFrame:
    out = pd.DataFrame(index=value_df.index, columns=value_df.columns, dtype=float)
    valid_map = group_map.dropna()
    for group_name in valid_map.unique():
        cols = [stock for stock in valid_map.index[valid_map == group_name] if stock in value_df.columns]
        if not cols:
            continue
        group_mean = value_df[cols].mean(axis=1)
        out.loc[:, cols] = np.repeat(group_mean.values[:, None], len(cols), axis=1)
    return out


def _group_rank_feature(value_df: pd.DataFrame, group_map: pd.Series) -> pd.DataFrame:
    out = pd.DataFrame(index=value_df.index, columns=value_df.columns, dtype=float)
    valid_map = group_map.dropna()
    for group_name in valid_map.unique():
        cols = [stock for stock in valid_map.index[valid_map == group_name] if stock in value_df.columns]
        if len(cols) < 2:
            continue
        out.loc[:, cols] = value_df[cols].rank(axis=1, pct=True)
    return out.fillna(0.5)


def _dynamic_group_mean_feature(
    value_df: pd.DataFrame,
    group_df: pd.DataFrame,
    groups: List[int],
) -> pd.DataFrame:
    value_arr = value_df.to_numpy(dtype=float)
    group_arr = group_df.to_numpy(dtype=float)
    out_arr = np.full_like(value_arr, np.nan, dtype=float)
    for group in groups:
        mask = np.isfinite(group_arr) & (group_arr == float(group))
        counts = mask.sum(axis=1)
        sums = np.where(mask, value_arr, 0.0).sum(axis=1)
        means = np.divide(
            sums,
            counts,
            out=np.full(len(counts), np.nan, dtype=float),
            where=counts > 0,
        )
        out_arr = np.where(mask, means[:, None], out_arr)
    return pd.DataFrame(out_arr, index=value_df.index, columns=value_df.columns)


def _build_static_graph_prior(
    columns: pd.Index,
    industry_map: pd.Series | None,
    style_map: pd.DataFrame | None,
    industry_boost: float,
    style_boost: float,
) -> np.ndarray:
    n_stocks = len(columns)
    prior = np.zeros((n_stocks, n_stocks), dtype=np.float32)
    if industry_map is not None and not industry_map.empty and industry_boost != 0.0:
        industry = industry_map.reindex(columns)
        industry_arr = industry.astype("object").to_numpy()
        same_industry = (
            pd.notna(industry_arr)[:, None]
            & pd.notna(industry_arr)[None, :]
            & (industry_arr[:, None] == industry_arr[None, :])
        )
        prior += same_industry.astype(np.float32) * float(industry_boost)
    if style_map is not None and not style_map.empty and style_boost != 0.0:
        style_bool = style_map.reindex(columns).fillna(False).astype(bool).to_numpy(dtype=np.float32, copy=False)
        if style_bool.size > 0:
            overlap = style_bool @ style_bool.T
            style_counts = style_bool.sum(axis=1, keepdims=True)
            denom = np.maximum(np.minimum(style_counts, style_counts.T), 1.0)
            prior += (overlap / denom).astype(np.float32) * float(style_boost)
    np.fill_diagonal(prior, 0.0)
    return prior


def _aggregate_dynamic_graph_feature(
    base_features: np.ndarray,
    source_arr: np.ndarray,
    valid_mask: np.ndarray,
    static_prior: np.ndarray,
    top_k: int,
    temperature: float,
) -> tuple[np.ndarray, np.ndarray]:
    out = np.full(source_arr.shape, np.nan, dtype=np.float32)
    similarity_out = np.full(source_arr.shape, np.nan, dtype=np.float32)
    if base_features.ndim != 3:
        raise ValueError(f"Expected base_features to have 3 dims, got {base_features.shape}")
    n_dates = base_features.shape[0]
    safe_temp = max(float(temperature), 1e-3)
    for row_idx in range(n_dates):
        active_idx = np.flatnonzero(valid_mask[row_idx])
        if active_idx.size <= 1:
            continue
        k = min(int(top_k), int(active_idx.size) - 1)
        if k <= 0:
            continue
        x = np.nan_to_num(base_features[row_idx, active_idx, :], nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32, copy=False)
        norms = np.linalg.norm(x, axis=1, keepdims=True)
        norms[norms == 0.0] = 1.0
        x = x / norms
        sim = (x @ x.T).astype(np.float32, copy=False)
        sim += static_prior[np.ix_(active_idx, active_idx)]
        np.fill_diagonal(sim, -np.inf)
        top_pos = np.argpartition(sim, kth=sim.shape[1] - k, axis=1)[:, -k:]
        top_sim = np.take_along_axis(sim, top_pos, axis=1)
        logits = top_sim / safe_temp
        finite_logits = np.where(np.isfinite(logits), logits, -np.inf)
        row_max = np.max(finite_logits, axis=1, keepdims=True)
        row_max[~np.isfinite(row_max)] = 0.0
        weights = np.exp(finite_logits - row_max)
        weights[~np.isfinite(top_sim)] = 0.0
        weight_denom = weights.sum(axis=1, keepdims=True)
        weight_denom[weight_denom == 0.0] = 1.0
        weights = weights / weight_denom

        source_vals = source_arr[row_idx, active_idx]
        neighbor_vals = source_vals[top_pos]
        finite_neighbor = np.isfinite(neighbor_vals)
        neighbor_weights = weights * finite_neighbor.astype(np.float32)
        neighbor_denom = neighbor_weights.sum(axis=1)
        weighted_sum = (neighbor_weights * np.nan_to_num(neighbor_vals, nan=0.0, posinf=0.0, neginf=0.0)).sum(axis=1)
        neighbor_mean = np.divide(
            weighted_sum,
            neighbor_denom,
            out=np.full(active_idx.size, np.nan, dtype=np.float32),
            where=neighbor_denom > 0.0,
        )
        sim_weight_sum = (weights * np.where(np.isfinite(top_sim), top_sim, 0.0)).sum(axis=1)

        out[row_idx, active_idx] = neighbor_mean
        similarity_out[row_idx, active_idx] = sim_weight_sum.astype(np.float32, copy=False)
    return out, similarity_out


def build_dynamic_graph_feature_frames(
    close: pd.DataFrame,
    ret_5: pd.DataFrame,
    ret_20: pd.DataFrame,
    ma_20_gap: pd.DataFrame,
    amount_ratio_5_20: pd.DataFrame,
    liquidity_rank_20: pd.DataFrame,
    industry_map: pd.Series | None = None,
    style_map: pd.DataFrame | None = None,
    top_k: int = 8,
    temperature: float = 0.35,
    industry_boost: float = 0.15,
    style_boost: float = 0.05,
) -> Dict[str, pd.DataFrame]:
    columns = close.columns
    valid_mask = close.notna().to_numpy(dtype=bool, copy=False)
    static_prior = _build_static_graph_prior(
        columns=columns,
        industry_map=industry_map,
        style_map=style_map,
        industry_boost=industry_boost,
        style_boost=style_boost,
    )
    signal_frames = {
        "ret_5": ret_5,
        "ret_20": ret_20,
        "ma_20_gap": ma_20_gap,
        "amount_ratio_5_20": amount_ratio_5_20.sub(1.0),
        "liquidity_rank_20": (liquidity_rank_20 - 0.5) * 2.0,
    }
    base_features = np.stack(
        [
            _cross_sectional_zscore_frame(frame).to_numpy(dtype=np.float32, copy=False)
            for frame in signal_frames.values()
        ],
        axis=-1,
    )
    peer_outputs: Dict[str, pd.DataFrame] = {}
    similarity_arr = None
    for name, frame in signal_frames.items():
        peer_arr, sim_arr = _aggregate_dynamic_graph_feature(
            base_features=base_features,
            source_arr=frame.to_numpy(dtype=np.float32, copy=False),
            valid_mask=valid_mask,
            static_prior=static_prior,
            top_k=top_k,
            temperature=temperature,
        )
        peer_frame = pd.DataFrame(peer_arr, index=frame.index, columns=frame.columns)
        peer_outputs[f"graph_peer_{name}"] = peer_frame.fillna(0.0)
        peer_outputs[f"graph_rel_{name}"] = frame.sub(peer_frame, axis=0).fillna(0.0)
        if similarity_arr is None:
            similarity_arr = sim_arr
    if similarity_arr is not None:
        peer_outputs["graph_peer_similarity"] = pd.DataFrame(
            similarity_arr,
            index=close.index,
            columns=close.columns,
        ).fillna(0.0)
    return peer_outputs


def build_sequence_features(
    df_dict: Dict[str, pd.DataFrame],
    benchmark_close: pd.Series,
    state_frame: pd.DataFrame,
    industry_map: pd.Series | None = None,
    style_map: pd.DataFrame | None = None,
    liquidity_layer: bool = False,
    liquidity_bucket_count: int = 5,
    liquidity_bucket_frame: pd.DataFrame | None = None,
    dynamic_graph_layer: bool = False,
    dynamic_graph_top_k: int = 8,
    dynamic_graph_temperature: float = 0.35,
    dynamic_graph_industry_boost: float = 0.15,
    dynamic_graph_style_boost: float = 0.05,
    short_alpha_features: bool = False,
) -> Dict[str, pd.DataFrame]:
    close = df_dict["Close"].astype(float)
    open_df = df_dict["Open"].astype(float)
    high = df_dict["High"].astype(float)
    low = df_dict["Low"].astype(float)
    volume = df_dict["Volume"].astype(float)
    amount = df_dict["Amount"].astype(float)
    benchmark_close = benchmark_close.astype(float).reindex(close.index)
    benchmark_ret_1 = benchmark_close.pct_change().fillna(0.0)
    benchmark_ret_5 = benchmark_close.pct_change(5).fillna(0.0)
    benchmark_ret_20 = benchmark_close.pct_change(20).fillna(0.0)

    ret_1 = close.pct_change().fillna(0.0)
    ret_5 = close.pct_change(5).fillna(0.0)
    ret_20 = close.pct_change(20).fillna(0.0)
    rel_ret_1 = ret_1.sub(benchmark_ret_1, axis=0)
    rel_ret_5 = ret_5.sub(benchmark_ret_5, axis=0)
    intraday_body = close.div(open_df.replace(0, np.nan)).sub(1.0).fillna(0.0)
    range_pct = high.sub(low).div(close.shift(1).replace(0, np.nan)).fillna(0.0)
    high_close_gap = high.div(close.replace(0, np.nan)).sub(1.0).fillna(0.0)
    low_close_gap = close.div(low.replace(0, np.nan)).sub(1.0).fillna(0.0)
    vol_ratio_5_20 = volume.rolling(5).mean().div(volume.rolling(20).mean().replace(0, np.nan)).fillna(1.0)
    amount_ratio_5_20 = amount.rolling(5).mean().div(amount.rolling(20).mean().replace(0, np.nan)).fillna(1.0)
    adv20 = amount.rolling(20).mean()
    adv60 = amount.rolling(60).mean()
    ma_20_gap = close.div(close.rolling(20).mean().replace(0, np.nan)).sub(1.0).fillna(0.0)
    ma_60_gap = close.div(close.rolling(60).mean().replace(0, np.nan)).sub(1.0).fillna(0.0)
    bench_gap_20 = benchmark_close.div(benchmark_close.rolling(20).mean().replace(0, np.nan)).sub(1.0).fillna(0.0)

    state_id = state_frame["state_id"].reindex(close.index).ffill().fillna(0).astype(int)
    state_one_hot = {
        f"state_{idx}": pd.DataFrame(
            np.repeat((state_id == idx).astype(float).values[:, None], close.shape[1], axis=1),
            index=close.index,
            columns=close.columns,
        )
        for idx in sorted(state_frame["state_id"].dropna().astype(int).unique())
    }
    benchmark_feature_frames = {
        "benchmark_ret_1": pd.DataFrame(np.repeat(benchmark_ret_1.fillna(0.0).values[:, None], close.shape[1], axis=1), index=close.index, columns=close.columns),
        "benchmark_ret_5": pd.DataFrame(np.repeat(benchmark_ret_5.fillna(0.0).values[:, None], close.shape[1], axis=1), index=close.index, columns=close.columns),
        "benchmark_gap_20": pd.DataFrame(np.repeat(bench_gap_20.fillna(0.0).values[:, None], close.shape[1], axis=1), index=close.index, columns=close.columns),
    }

    features = {
        "ret_1": ret_1,
        "ret_5": ret_5,
        "ret_20": ret_20,
        "rel_ret_1": rel_ret_1,
        "rel_ret_5": rel_ret_5,
        "intraday_body": intraday_body,
        "range_pct": range_pct,
        "high_close_gap": high_close_gap,
        "low_close_gap": low_close_gap,
        "vol_ratio_5_20": vol_ratio_5_20,
        "amount_ratio_5_20": amount_ratio_5_20,
        "ma_20_gap": ma_20_gap,
        "ma_60_gap": ma_60_gap,
    }
    features.update(state_one_hot)
    features.update(benchmark_feature_frames)

    if industry_map is not None and not industry_map.empty:
        industry_map = industry_map.reindex(close.columns)
        industry_ret_20 = _group_mean_feature(ret_20, industry_map)
        industry_ma_20_gap = _group_mean_feature(ma_20_gap, industry_map)
        industry_rank_ret_20 = _group_rank_feature(ret_20, industry_map)
        industry_rank_ma_20_gap = _group_rank_feature(ma_20_gap, industry_map)
        benchmark_ret_20_df = pd.DataFrame(
            np.repeat(benchmark_ret_20.fillna(0.0).values[:, None], close.shape[1], axis=1),
            index=close.index,
            columns=close.columns,
        )
        features["industry_ret_20"] = industry_ret_20.sub(benchmark_ret_20_df, axis=0).fillna(0.0)
        features["industry_ma_20_gap"] = industry_ma_20_gap.fillna(0.0)
        features["industry_rank_ret_20"] = industry_rank_ret_20.fillna(0.5)
        features["industry_rank_ma_20_gap"] = industry_rank_ma_20_gap.fillna(0.5)

    if style_map is not None and not style_map.empty:
        style_map = style_map.reindex(close.columns).fillna(False).astype(bool)
        for style_name in style_map.columns:
            members = [stock for stock in close.columns if bool(style_map.loc[stock, style_name])]
            if not members:
                continue
            style_ret_20 = ret_20[members].mean(axis=1)
            style_strength = style_ret_20.sub(benchmark_ret_20.reindex(ret_20.index), fill_value=0.0)
            style_strength_df = pd.DataFrame(
                np.repeat(style_strength.fillna(0.0).values[:, None], close.shape[1], axis=1),
                index=close.index,
                columns=close.columns,
            )
            style_member_df = pd.DataFrame(
                np.repeat(style_map[style_name].astype(float).values[None, :], len(close.index), axis=0),
                index=close.index,
                columns=close.columns,
            )
            features[f"style_{style_name}_strength_20"] = style_strength_df
            features[f"style_{style_name}_member"] = style_member_df

    liquidity_rank_20 = adv20.rank(axis=1, pct=True).fillna(0.5)
    if liquidity_layer:
        if liquidity_bucket_frame is None:
            liquidity_bucket_frame = build_liquidity_bucket_frame(amount, window=20, n_buckets=liquidity_bucket_count)
        liquidity_bucket_frame = liquidity_bucket_frame.reindex(index=close.index, columns=close.columns)
        liquidity_z_20 = _cross_sectional_zscore_frame(adv20)
        liquidity_ratio_20_60 = adv20.div(adv60.replace(0, np.nan)).fillna(1.0)
        liquidity_bucket_mean_ret_20 = _dynamic_group_mean_feature(ret_20, liquidity_bucket_frame, list(range(liquidity_bucket_count)))
        liquidity_bucket_mean_ma_20_gap = _dynamic_group_mean_feature(ma_20_gap, liquidity_bucket_frame, list(range(liquidity_bucket_count)))
        features["liquidity_rank_20"] = liquidity_rank_20
        features["liquidity_z_20"] = liquidity_z_20
        features["liquidity_ratio_20_60"] = liquidity_ratio_20_60
        features["liquidity_rel_ret_20"] = ret_20.sub(liquidity_bucket_mean_ret_20, axis=0).fillna(0.0)
        features["liquidity_rel_ma_20_gap"] = ma_20_gap.sub(liquidity_bucket_mean_ma_20_gap, axis=0).fillna(0.0)
        for bucket in range(liquidity_bucket_count):
            bucket_mask = (liquidity_bucket_frame == float(bucket)).astype(float)
            features[f"liquidity_bucket_{bucket}"] = bucket_mask.fillna(0.0)
    if dynamic_graph_layer:
        features.update(
            build_dynamic_graph_feature_frames(
                close=close,
                ret_5=ret_5,
                ret_20=ret_20,
                ma_20_gap=ma_20_gap,
                amount_ratio_5_20=amount_ratio_5_20,
                liquidity_rank_20=liquidity_rank_20,
                industry_map=industry_map,
                style_map=style_map,
                top_k=dynamic_graph_top_k,
                temperature=dynamic_graph_temperature,
                industry_boost=dynamic_graph_industry_boost,
                style_boost=dynamic_graph_style_boost,
            )
        )
    if short_alpha_features:
        prev_close = close.shift(1)
        prev_high_20 = high.rolling(20).max().shift(1)
        prev_high_60 = high.rolling(60).max().shift(1)
        benchmark_momentum_2 = benchmark_close.pct_change(2).fillna(0.0)
        vol_std_10 = ret_1.rolling(10).std()
        vol_std_3 = ret_1.rolling(3).std()
        vol_std_60 = ret_1.rolling(60).std()
        gap_open_1 = open_df.div(prev_close.replace(0, np.nan)).sub(1.0).fillna(0.0)
        close_pos_in_range = close.sub(low).div(high.sub(low).replace(0, np.nan)).fillna(0.5)
        upper_shadow_ratio = high.sub(pd.concat([open_df, close], axis=0).groupby(level=0).max()).div(
            high.sub(low).replace(0, np.nan)
        ).fillna(0.0)
        lower_shadow_ratio = pd.concat([open_df, close], axis=0).groupby(level=0).min().sub(low).div(
            high.sub(low).replace(0, np.nan)
        ).fillna(0.0)
        breakout_distance_20 = close.div(prev_high_20.replace(0, np.nan)).sub(1.0).fillna(0.0)
        breakout_distance_60 = close.div(prev_high_60.replace(0, np.nan)).sub(1.0).fillna(0.0)
        breakout_intraday_high_20 = high.div(prev_high_20.replace(0, np.nan)).sub(1.0).fillna(0.0)
        compression_10_60 = vol_std_10.div(vol_std_60.replace(0, np.nan)).fillna(1.0)
        volatility_ratio_3_10 = vol_std_3.div(vol_std_10.replace(0, np.nan)).fillna(1.0)
        range_expansion_1_5 = range_pct.div(range_pct.rolling(5).mean().replace(0, np.nan)).fillna(1.0)
        body_strength_1_5 = intraday_body.abs().div(intraday_body.abs().rolling(5).mean().replace(0, np.nan)).fillna(1.0)
        volume_burst_1_5 = volume.div(volume.rolling(5).mean().replace(0, np.nan)).fillna(1.0)
        volume_burst_1_3 = volume.div(volume.rolling(3).mean().replace(0, np.nan)).fillna(1.0)
        amount_burst_1_5 = amount.div(amount.rolling(5).mean().replace(0, np.nan)).fillna(1.0)
        amount_burst_1_3 = amount.div(amount.rolling(3).mean().replace(0, np.nan)).fillna(1.0)
        signal_persistence_5 = ret_1.gt(0.0).rolling(5).mean().fillna(0.5)
        momentum_2 = close.pct_change(2).fillna(0.0)
        rel_momentum_2 = momentum_2.sub(benchmark_momentum_2, axis=0)
        momentum_3 = close.pct_change(3).fillna(0.0)
        rel_momentum_3 = momentum_3.sub(benchmark_close.pct_change(3).fillna(0.0), axis=0)
        features.update(
            {
                "gap_open_1": gap_open_1,
                "close_pos_in_range": close_pos_in_range,
                "upper_shadow_ratio": upper_shadow_ratio,
                "lower_shadow_ratio": lower_shadow_ratio,
                "breakout_distance_20": breakout_distance_20,
                "breakout_distance_60": breakout_distance_60,
                "breakout_intraday_high_20": breakout_intraday_high_20,
                "compression_10_60": compression_10_60,
                "volatility_ratio_3_10": volatility_ratio_3_10,
                "range_expansion_1_5": range_expansion_1_5,
                "body_strength_1_5": body_strength_1_5,
                "volume_burst_1_5": volume_burst_1_5,
                "volume_burst_1_3": volume_burst_1_3,
                "amount_burst_1_5": amount_burst_1_5,
                "amount_burst_1_3": amount_burst_1_3,
                "signal_persistence_5": signal_persistence_5,
                "momentum_2": momentum_2,
                "rel_momentum_2": rel_momentum_2,
                "momentum_3": momentum_3,
                "rel_momentum_3": rel_momentum_3,
            }
        )
    return features


def build_targets(
    close: pd.DataFrame,
    benchmark_close: pd.Series,
    horizons: tuple[int, ...],
    open_df: pd.DataFrame | None = None,
    benchmark_open: pd.Series | None = None,
    execution_mode: str = "close",
    breakout_event_horizon: int = 5,
    breakout_event_threshold: float = 0.08,
    breakout_event_pullback_limit: float = 0.03,
    breakout_event_task: bool = False,
    clean_breakout_event_task: bool = False,
) -> Dict[str, pd.DataFrame]:
    target_frames: Dict[str, pd.DataFrame] = {}
    mode = str(execution_mode or "close").lower()
    for horizon in horizons:
        if mode == "next_open":
            if open_df is None or benchmark_open is None:
                raise ValueError("next_open targets require open_df and benchmark_open.")
            stock_entry = open_df.shift(-1)
            stock_exit = open_df.shift(-(horizon + 1))
            bench_entry = benchmark_open.shift(-1)
            bench_exit = benchmark_open.shift(-(horizon + 1))
            stock_fwd = stock_exit.div(stock_entry).sub(1.0)
            bench_fwd = bench_exit.div(bench_entry).sub(1.0)
        else:
            stock_fwd = close.shift(-horizon).div(close).sub(1.0)
            bench_fwd = benchmark_close.shift(-horizon).div(benchmark_close).sub(1.0)
        target_frames[f"fwd_excess_{horizon}"] = stock_fwd.sub(bench_fwd, axis=0)
    if mode == "next_open":
        future_min = open_df.shift(-1).rolling(max(horizons)).min()
        target_frames["risk_downside_20"] = future_min.div(open_df.shift(-1)).sub(1.0)
    else:
        future_min = close.shift(-1).rolling(max(horizons)).min()
        target_frames["risk_downside_20"] = future_min.div(close).sub(1.0)
    event_horizon = max(int(breakout_event_horizon), 1)
    if breakout_event_task or clean_breakout_event_task:
        if mode == "next_open":
            if open_df is None:
                raise ValueError("Breakout event targets require open_df under next_open mode.")
            entry = open_df.shift(-1)
            future_opens = [open_df.shift(-(step + 1)) for step in range(1, event_horizon + 1)]
        else:
            entry = close
            future_opens = [close.shift(-step) for step in range(1, event_horizon + 1)]
        future_max = future_opens[0].copy()
        future_min_path = future_opens[0].copy()
        for future_frame in future_opens[1:]:
            future_max = future_max.combine(future_frame, np.fmax)
            future_min_path = future_min_path.combine(future_frame, np.fmin)
        max_gain = future_max.div(entry.replace(0, np.nan)).sub(1.0)
        max_pullback = future_min_path.div(entry.replace(0, np.nan)).sub(1.0)
        if breakout_event_task:
            target_frames[f"event_breakout_{event_horizon}"] = (max_gain >= float(breakout_event_threshold)).astype(float)
        if clean_breakout_event_task:
            clean_event = (
                (max_gain >= float(breakout_event_threshold))
                & (max_pullback >= -float(breakout_event_pullback_limit))
            )
            target_frames[f"event_clean_breakout_{event_horizon}"] = clean_event.astype(float)
    return target_frames


@dataclass
class SampleMeta:
    date: pd.Timestamp
    stock: str
    state_id: int | None = None
    liquidity_bucket: int | None = None
    structure_id: int | None = None


@dataclass
class StockSequenceCorpus:
    feature_names: List[str]
    target_names: List[str]
    samples: List[np.ndarray]
    train_targets: List[np.ndarray]
    raw_targets: List[np.ndarray]
    meta: List[SampleMeta]

    def build_index(self, start_date: pd.Timestamp | None = None, end_date: pd.Timestamp | None = None) -> List[int]:
        indices: List[int] = []
        for idx, item in enumerate(self.meta):
            if start_date is not None and item.date < pd.Timestamp(start_date):
                continue
            if end_date is not None and item.date > pd.Timestamp(end_date):
                continue
            indices.append(idx)
        return indices


def build_sequence_corpus(
    feature_frames: Dict[str, pd.DataFrame],
    train_target_frames: Dict[str, pd.DataFrame],
    raw_target_frames: Dict[str, pd.DataFrame],
    lookback_window: int,
    sample_dates: List[pd.Timestamp],
    min_adv20: float,
    amount_frame: pd.DataFrame,
    close_frame: pd.DataFrame,
    min_price: float,
    max_price: float,
    state_frame: pd.DataFrame | None = None,
    liquidity_bucket_frame: pd.DataFrame | None = None,
    structure_label_frame: pd.DataFrame | None = None,
    universe_membership_frame: pd.DataFrame | None = None,
    require_targets: bool = True,
) -> StockSequenceCorpus:
    feature_names = list(feature_frames.keys())
    target_names = list(raw_target_frames.keys())
    date_index = pd.Index(close_frame.index)
    stock_columns = list(close_frame.columns)
    sample_date_set = set(pd.to_datetime(sample_dates))

    adv20 = amount_frame.rolling(20).mean().reindex(index=date_index, columns=stock_columns)
    close_arr = close_frame.reindex(index=date_index, columns=stock_columns).to_numpy(dtype=np.float32, copy=False)
    adv20_arr = adv20.to_numpy(dtype=np.float32, copy=False)
    feature_cube = np.stack(
        [feature_frames[name].reindex(index=date_index, columns=stock_columns).to_numpy(dtype=np.float32, copy=False) for name in feature_names],
        axis=-1,
    )
    train_target_cube = np.stack(
        [train_target_frames[name].reindex(index=date_index, columns=stock_columns).to_numpy(dtype=np.float32, copy=False) for name in target_names],
        axis=-1,
    )
    raw_target_cube = np.stack(
        [raw_target_frames[name].reindex(index=date_index, columns=stock_columns).to_numpy(dtype=np.float32, copy=False) for name in target_names],
        axis=-1,
    )
    liquidity_bucket_arr = None
    state_id_arr = None
    if state_frame is not None and "state_id" in state_frame.columns:
        state_id_series = state_frame["state_id"].reindex(date_index).ffill().fillna(0).astype(float)
        state_id_arr = state_id_series.to_numpy(dtype=np.float32, copy=False)
    if liquidity_bucket_frame is not None:
        liquidity_bucket_arr = liquidity_bucket_frame.reindex(index=date_index, columns=stock_columns).to_numpy(dtype=np.float32, copy=False)
    structure_id_arr = None
    if structure_label_frame is not None:
        structure_id_frame = structure_label_frame.reindex(index=date_index, columns=stock_columns).apply(
            lambda col: col.map(STRUCTURE_LABEL_TO_ID)
        )
        structure_id_arr = structure_id_frame.to_numpy(dtype=np.float32, copy=False)
    universe_membership_arr = None
    if universe_membership_frame is not None:
        universe_membership_arr = (
            universe_membership_frame.reindex(index=date_index, columns=stock_columns).eq(True).to_numpy(dtype=bool, copy=False)
        )

    sample_positions = [
        end_pos
        for end_pos, dt in enumerate(date_index)
        if end_pos >= lookback_window - 1 and pd.Timestamp(dt) in sample_date_set
    ]

    price_ok = np.isfinite(close_arr) & (close_arr >= float(min_price)) & (close_arr <= float(max_price))
    adv_ok = np.isfinite(adv20_arr) & (adv20_arr >= float(min_adv20))
    target_ok = np.all(np.isfinite(raw_target_cube), axis=2) if require_targets else np.ones_like(price_ok, dtype=bool)
    eligible_mask = price_ok & adv_ok & target_ok
    if universe_membership_arr is not None:
        eligible_mask &= universe_membership_arr

    samples: List[np.ndarray] = []
    train_targets: List[np.ndarray] = []
    raw_targets: List[np.ndarray] = []
    meta: List[SampleMeta] = []

    for end_pos in sample_positions:
        dt = pd.Timestamp(date_index[end_pos])
        start_pos = end_pos - lookback_window + 1
        valid_stock_indices = np.flatnonzero(eligible_mask[end_pos])
        if valid_stock_indices.size == 0:
            continue
        for stock_idx in valid_stock_indices.tolist():
            seq = feature_cube[start_pos : end_pos + 1, stock_idx, :]
            if not np.isfinite(seq).any():
                continue
            seq = _zscore_2d(seq)
            samples.append(seq.astype(np.float32, copy=False))
            train_targets.append(train_target_cube[end_pos, stock_idx, :].astype(np.float32, copy=False))
            raw_targets.append(raw_target_cube[end_pos, stock_idx, :].astype(np.float32, copy=False))
            bucket = None
            state_id = None
            if state_id_arr is not None:
                state_val = float(state_id_arr[end_pos])
                state_id = int(state_val) if np.isfinite(state_val) else None
            if liquidity_bucket_arr is not None:
                bucket_val = float(liquidity_bucket_arr[end_pos, stock_idx])
                bucket = int(bucket_val) if np.isfinite(bucket_val) else None
            structure_id = None
            if structure_id_arr is not None:
                structure_val = float(structure_id_arr[end_pos, stock_idx])
                structure_id = int(structure_val) if np.isfinite(structure_val) else None
            meta.append(
                SampleMeta(
                    date=dt,
                    stock=stock_columns[stock_idx],
                    state_id=state_id,
                    liquidity_bucket=bucket,
                    structure_id=structure_id,
                )
            )

    return StockSequenceCorpus(
        feature_names=feature_names,
        target_names=target_names,
        samples=samples,
        train_targets=train_targets,
        raw_targets=raw_targets,
        meta=meta,
    )


class StockSequenceDataset(Dataset):
    def __init__(
        self,
        corpus: StockSequenceCorpus,
        indices: List[int] | None = None,
    ) -> None:
        self.corpus = corpus
        self.indices = list(indices) if indices is not None else list(range(len(corpus.meta)))
        self.feature_names = corpus.feature_names
        self.target_names = corpus.target_names
        self.meta: List[SampleMeta] = [corpus.meta[idx] for idx in self.indices]

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, idx: int):
        base_idx = self.indices[idx]
        x = torch.from_numpy(self.corpus.samples[base_idx])
        y = torch.from_numpy(self.corpus.train_targets[base_idx])
        y_raw = torch.from_numpy(self.corpus.raw_targets[base_idx])
        meta = self.corpus.meta[base_idx]
        state_id = -1 if meta.state_id is None else int(meta.state_id)
        liquidity_bucket = -1 if meta.liquidity_bucket is None else int(meta.liquidity_bucket)
        structure_id = -1 if meta.structure_id is None else int(meta.structure_id)
        return x, y, y_raw, state_id, liquidity_bucket, structure_id, meta.date, meta.stock


class SequenceOnlyDataset(Dataset):
    def __init__(
        self,
        corpus: StockSequenceCorpus,
        indices: List[int] | None = None,
    ) -> None:
        self.corpus = corpus
        self.indices = list(indices) if indices is not None else list(range(len(corpus.meta)))
        self.feature_names = corpus.feature_names
        self.meta: List[SampleMeta] = [corpus.meta[idx] for idx in self.indices]

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, idx: int):
        base_idx = self.indices[idx]
        x = torch.from_numpy(self.corpus.samples[base_idx])
        meta = self.corpus.meta[base_idx]
        return x, meta.date, meta.stock


class DateGroupedBatchSampler(BatchSampler):
    def __init__(self, meta: List[SampleMeta], batch_size: int, shuffle_dates: bool = True, random_seed: int = 7) -> None:
        self.batch_size = batch_size
        self.shuffle_dates = shuffle_dates
        self.random_seed = random_seed
        self._epoch = 0
        groups: Dict[pd.Timestamp, List[int]] = {}
        for idx, item in enumerate(meta):
            groups.setdefault(pd.Timestamp(item.date), []).append(idx)
        self.groups = groups
        self.dates = list(groups.keys())
        self._length = sum((len(indices) + batch_size - 1) // batch_size for indices in groups.values())

    def __iter__(self):
        rng = np.random.default_rng(self.random_seed + self._epoch)
        self._epoch += 1
        dates = list(self.dates)
        if self.shuffle_dates:
            rng.shuffle(dates)
        for dt in dates:
            indices = list(self.groups[dt])
            if self.shuffle_dates:
                rng.shuffle(indices)
            for start in range(0, len(indices), self.batch_size):
                yield indices[start : start + self.batch_size]

    def __len__(self) -> int:
        return self._length

    def state_dict(self) -> dict[str, int]:
        return {"epoch": int(self._epoch)}

    def load_state_dict(self, state: dict[str, int] | None) -> None:
        payload = dict(state or {})
        self._epoch = max(int(payload.get("epoch", 0) or 0), 0)
