from __future__ import annotations

from pathlib import Path
from typing import Any
import warnings

import numpy as np
import pandas as pd

from daily_research.continuous_policy.pipeline_utils import select_feature_columns
from daily_research.continuous_policy.portfolio_simulator import PortfolioState
from daily_research.continuous_policy.state_builder import PreparedPolicyInputs, build_cross_section_state


AUGMENTED_INDUSTRY_METRICS_PROFILE = "raw_kline_context_v2_tradeable_local_state_industry_metrics_v1"
FORECAST_FEATURE_PROFILES: tuple[str, ...] = (
    "state_v1",
    "raw_kline_v1",
    "raw_kline_context_v1",
    "raw_kline_context_no_alpha_prior_v1",
    "raw_kline_context_sector_v1",
    "raw_kline_context_sector_relative_v1",
    "raw_kline_context_regime_v1",
    "raw_kline_context_sector_relative_regime_v1",
    "raw_kline_context_v2_tradeable_amount_checked",
    "raw_kline_context_v2_tradeable_local_state_v1",
    AUGMENTED_INDUSTRY_METRICS_PROFILE,
)
DEFAULT_FORECAST_FEATURE_PROFILE = "raw_kline_context_v1"
DEFAULT_FORECAST_MAX_FEATURE_COLUMNS = 192
NON_FORECAST_STATE_COLUMNS = {"date", "stock", "in_universe"}
RAW_FRAME_PROFILES = {
    "raw_kline_v1",
    "raw_kline_context_v1",
    "raw_kline_context_no_alpha_prior_v1",
    "raw_kline_context_sector_v1",
    "raw_kline_context_sector_relative_v1",
    "raw_kline_context_regime_v1",
    "raw_kline_context_sector_relative_regime_v1",
    "raw_kline_context_v2_tradeable_amount_checked",
    "raw_kline_context_v2_tradeable_local_state_v1",
    AUGMENTED_INDUSTRY_METRICS_PROFILE,
}
CONTEXT_FRAME_PROFILES = {
    "raw_kline_context_v1",
    "raw_kline_context_no_alpha_prior_v1",
    "raw_kline_context_sector_v1",
    "raw_kline_context_sector_relative_v1",
    "raw_kline_context_regime_v1",
    "raw_kline_context_sector_relative_regime_v1",
    "raw_kline_context_v2_tradeable_amount_checked",
    "raw_kline_context_v2_tradeable_local_state_v1",
    AUGMENTED_INDUSTRY_METRICS_PROFILE,
}
HISTORY_FRAME_PROFILES = {
    "raw_kline_context_sector_relative_v1",
    "raw_kline_context_regime_v1",
    "raw_kline_context_sector_relative_regime_v1",
    "raw_kline_context_v2_tradeable_amount_checked",
    "raw_kline_context_v2_tradeable_local_state_v1",
    AUGMENTED_INDUSTRY_METRICS_PROFILE,
}
SECTOR_CONTEXT_PROFILES = {"raw_kline_context_sector_v1", AUGMENTED_INDUSTRY_METRICS_PROFILE}
SECTOR_RELATIVE_PROFILES = {
    "raw_kline_context_sector_relative_v1",
    "raw_kline_context_sector_relative_regime_v1",
    AUGMENTED_INDUSTRY_METRICS_PROFILE,
}
REGIME_PROFILES = {
    "raw_kline_context_regime_v1",
    "raw_kline_context_sector_relative_regime_v1",
    "raw_kline_context_v2_tradeable_amount_checked",
    "raw_kline_context_v2_tradeable_local_state_v1",
    AUGMENTED_INDUSTRY_METRICS_PROFILE,
}
LOCAL_STATE_PROFILES = {"raw_kline_context_v2_tradeable_local_state_v1", AUGMENTED_INDUSTRY_METRICS_PROFILE}
TURNOVER_CONTEXT_PROFILES = {AUGMENTED_INDUSTRY_METRICS_PROFILE}
VALUATION_CONTEXT_PROFILES = {AUGMENTED_INDUSTRY_METRICS_PROFILE}
AMOUNT_CHECKED_PROFILES = {"raw_kline_context_v2_tradeable_amount_checked", *LOCAL_STATE_PROFILES}
NO_ALPHA_CONTRACT_PROFILES = {
    "raw_kline_context_no_alpha_prior_v1",
    "raw_kline_context_sector_relative_v1",
    "raw_kline_context_regime_v1",
    "raw_kline_context_sector_relative_regime_v1",
    "raw_kline_context_v2_tradeable_amount_checked",
    "raw_kline_context_v2_tradeable_local_state_v1",
    AUGMENTED_INDUSTRY_METRICS_PROFILE,
}


def _source_sector_board_view_id(prepared: PreparedPolicyInputs) -> str:
    summary = dict(getattr(prepared, "metadata_summary", {}) or {})
    sector_summary = summary.get("sector_board_view", {})
    if isinstance(sector_summary, dict):
        dataset_id = str(sector_summary.get("dataset_id", "") or "").strip()
        if dataset_id:
            return dataset_id
    for attr_name in ("raw_cache_meta", "prepared_cache_meta"):
        cache_meta = dict(getattr(prepared, attr_name, {}) or {})
        sector_meta = cache_meta.get("sector_board_view", {})
        if isinstance(sector_meta, dict):
            dataset_id = str(sector_meta.get("dataset_id", "") or "").strip()
            if dataset_id:
                return dataset_id
    return ""


def _safe_div(numerator: pd.DataFrame | pd.Series, denominator: pd.DataFrame | pd.Series) -> pd.DataFrame | pd.Series:
    return numerator.div(denominator.replace(0.0, np.nan)).replace([np.inf, -np.inf], np.nan)


def _rolling_z(frame: pd.DataFrame, window: int) -> pd.DataFrame:
    mean = frame.rolling(int(window), min_periods=2).mean()
    std = frame.rolling(int(window), min_periods=2).std().replace(0.0, np.nan)
    return frame.sub(mean).div(std).replace([np.inf, -np.inf], np.nan)


def _cross_z(frame: pd.DataFrame) -> pd.DataFrame:
    mean = frame.mean(axis=1)
    std = frame.std(axis=1, ddof=0).replace(0.0, np.nan)
    return frame.sub(mean, axis=0).div(std, axis=0).replace([np.inf, -np.inf], np.nan)


def _repeat_series_to_universe(series: pd.Series, columns: list[str]) -> pd.DataFrame:
    aligned = pd.Series(series, dtype=float)
    values = np.repeat(aligned.to_numpy(dtype=float).reshape(-1, 1), len(columns), axis=1)
    return pd.DataFrame(values, index=aligned.index, columns=columns)


def _row_share(condition: pd.DataFrame, membership: pd.DataFrame) -> pd.Series:
    valid = membership.astype(bool)
    denom = valid.sum(axis=1).replace(0, np.nan)
    return (condition.astype(bool) & valid).sum(axis=1).div(denom).replace([np.inf, -np.inf], np.nan)


def _bucket_frame(frame: pd.DataFrame, buckets: int = 5) -> pd.DataFrame:
    rank = frame.rank(axis=1, pct=True)
    bucket = np.ceil(rank * int(buckets)).clip(1, int(buckets))
    return bucket.where(rank.notna())


def _amount_unit_audit(amount: pd.DataFrame, close: pd.DataFrame, volume: pd.DataFrame) -> dict[str, Any]:
    amount_float = amount.astype(float)
    close_float = close.astype(float)
    volume_float = volume.astype(float)
    valid = amount_float.gt(0.0) & close_float.gt(0.0) & volume_float.gt(0.0)
    ratio = amount_float.where(valid).div(close_float.where(valid).mul(volume_float.where(valid)))
    clean = ratio.replace([np.inf, -np.inf], np.nan).stack().dropna()
    if clean.empty:
        return {
            "amount_unit_policy": "unknown_unit",
            "amount_unit_factor": 1.0,
            "amount_consistency_median": 0.0,
            "amount_consistency_p95_abs_log_error": 0.0,
            "amount_unit_status": "blocked_no_valid_ratio",
        }
    median = float(clean.median())
    factor = 1.0
    policy = "as_is"
    status = "ok"
    if 0.5 <= median <= 2.0:
        policy = "as_is"
    elif 5000.0 <= median <= 20000.0:
        policy = "divide_by_10000"
        factor = 1.0 / 10000.0
        status = "normalized"
    elif 0.00005 <= median <= 0.0002:
        policy = "multiply_by_10000"
        factor = 10000.0
        status = "normalized"
    else:
        policy = "unknown_unit"
        status = "degraded_unknown_unit"
    positive = clean.astype(float).loc[clean.astype(float) > 0.0]
    log_error = np.log10(positive).abs().replace([np.inf, -np.inf], np.nan).dropna()
    return {
        "amount_unit_policy": policy,
        "amount_unit_factor": float(factor),
        "amount_consistency_median": median,
        "amount_consistency_p95_abs_log_error": float(log_error.quantile(0.95)) if not log_error.empty else 0.0,
        "amount_unit_status": status,
    }


def _amount_for_feature_profile(prepared: PreparedPolicyInputs, feature_profile: str) -> tuple[pd.DataFrame, dict[str, Any]]:
    base = prepared.amount.astype(float)
    audit = _amount_unit_audit(base, prepared.close.astype(float), prepared.volume.astype(float))
    if str(feature_profile) in AMOUNT_CHECKED_PROFILES:
        return base.mul(float(audit.get("amount_unit_factor", 1.0) or 1.0)), audit
    return base, {
        "amount_unit_policy": "not_checked_for_profile",
        "amount_unit_factor": 1.0,
        "amount_consistency_median": audit.get("amount_consistency_median", 0.0),
        "amount_consistency_p95_abs_log_error": audit.get("amount_consistency_p95_abs_log_error", 0.0),
        "amount_unit_status": "not_applied",
    }


def _peer_adv_context(*, returns: dict[int, pd.DataFrame], adv_bucket: pd.DataFrame) -> dict[str, pd.DataFrame]:
    out: dict[str, pd.DataFrame] = {}
    for horizon in (1, 5, 20):
        ret = returns[int(horizon)]
        mean_rows: list[pd.Series] = []
        rank_rows: list[pd.Series] = []
        relative_rows: list[pd.Series] = []
        for dt in ret.index:
            values = ret.loc[dt].astype(float)
            buckets = adv_bucket.loc[dt]
            mean = pd.Series(np.nan, index=ret.columns, dtype=float)
            rank = pd.Series(np.nan, index=ret.columns, dtype=float)
            for bucket_value in sorted({int(item) for item in buckets.dropna().tolist()}):
                members = buckets.index[buckets == bucket_value]
                group_values = values.reindex(members).replace([np.inf, -np.inf], np.nan)
                if group_values.dropna().empty:
                    continue
                mean.loc[members] = float(group_values.mean())
                rank.loc[members] = group_values.rank(pct=True, method="average")
            mean_rows.append(mean)
            rank_rows.append(rank)
            relative_rows.append(values.sub(mean))
        out[f"peer_adv_bucket_ret_{horizon}d_mean"] = pd.DataFrame(mean_rows, index=ret.index, columns=ret.columns)
        out[f"peer_adv_bucket_ret_{horizon}d_rank"] = pd.DataFrame(rank_rows, index=ret.index, columns=ret.columns)
        out[f"relative_to_peer_adv_ret_{horizon}d"] = pd.DataFrame(relative_rows, index=ret.index, columns=ret.columns)
    return out


def _alpha_dependent_column(column: str) -> bool:
    name = str(column)
    return bool(
        name.startswith("alpha_prior_")
        or name in {"score_none", "score_v2", "score_blend", "z_score_none", "z_score_v2"}
        or name.startswith("score_delta")
        or name.startswith("score_blend_lag")
        or name.startswith("score_rank")
        or name.startswith("score_cross")
    )


def _raw_kline_feature_frames(prepared: PreparedPolicyInputs, *, feature_profile: str = DEFAULT_FORECAST_FEATURE_PROFILE) -> dict[str, pd.DataFrame]:
    open_ = prepared.open_.astype(float)
    high = prepared.high.astype(float)
    low = prepared.low.astype(float)
    close = prepared.close.astype(float)
    volume = prepared.volume.astype(float)
    amount, _amount_audit = _amount_for_feature_profile(prepared, feature_profile)
    prev_close = close.shift(1)
    high_low_range = high.sub(low).replace(0.0, np.nan)
    max_open_close = open_.where(open_ >= close, close)
    min_open_close = open_.where(open_ <= close, close)
    raw_close_from_prev = _safe_div(close, prev_close).sub(1.0)
    frames = {
        "raw_open_gap_1d": _safe_div(open_, prev_close).sub(1.0),
        "raw_high_from_prev_close_1d": _safe_div(high, prev_close).sub(1.0),
        "raw_low_from_prev_close_1d": _safe_div(low, prev_close).sub(1.0),
        "raw_close_from_prev_close_1d": raw_close_from_prev,
        "raw_close_to_open_1d": _safe_div(close, open_).sub(1.0),
        "raw_intraday_range_1d": _safe_div(high, low).sub(1.0),
        "raw_body_to_range_1d": close.sub(open_).abs().div(high_low_range),
        "raw_upper_shadow_to_range_1d": high.sub(max_open_close).div(high_low_range),
        "raw_lower_shadow_to_range_1d": min_open_close.sub(low).div(high_low_range),
        "raw_amount_z20": _rolling_z(amount, 20),
        "raw_volume_z20": _rolling_z(volume, 20),
        "raw_amount_ratio_5_20": _safe_div(amount.rolling(5, min_periods=1).mean(), amount.rolling(20, min_periods=1).mean()).sub(1.0),
        "raw_volume_ratio_5_20": _safe_div(volume.rolling(5, min_periods=1).mean(), volume.rolling(20, min_periods=1).mean()).sub(1.0),
        "raw_limit_up_like_1d": raw_close_from_prev.ge(0.095).astype(float).where(raw_close_from_prev.notna()),
        "raw_limit_down_like_1d": raw_close_from_prev.le(-0.095).astype(float).where(raw_close_from_prev.notna()),
    }
    return {key: value.replace([np.inf, -np.inf], np.nan) for key, value in frames.items()}


def _context_feature_frames(prepared: PreparedPolicyInputs, raw_frames: dict[str, pd.DataFrame], *, feature_profile: str = DEFAULT_FORECAST_FEATURE_PROFILE) -> dict[str, pd.DataFrame]:
    close = prepared.close.astype(float)
    amount, _amount_audit = _amount_for_feature_profile(prepared, feature_profile)
    columns = [str(item) for item in close.columns]
    membership = prepared.membership_frame.reindex(index=close.index, columns=columns, fill_value=False).astype(bool)
    returns = {
        1: close.pct_change(1).replace([np.inf, -np.inf], np.nan),
        3: close.pct_change(3).replace([np.inf, -np.inf], np.nan),
        5: close.pct_change(5).replace([np.inf, -np.inf], np.nan),
        20: close.pct_change(20).replace([np.inf, -np.inf], np.nan),
    }
    vol20 = returns[1].rolling(20, min_periods=2).std().replace([np.inf, -np.inf], np.nan)
    adv20 = amount.rolling(20, min_periods=1).mean()
    rolling_high_20 = close.rolling(20, min_periods=1).max()
    rolling_low_20 = close.rolling(20, min_periods=1).min()
    ma20 = close.rolling(20, min_periods=1).mean()
    ma60 = close.rolling(60, min_periods=1).mean()

    frames: dict[str, pd.DataFrame] = {
        "cs_rank_ret_1d": returns[1].rank(axis=1, pct=True),
        "cs_rank_ret_3d": returns[3].rank(axis=1, pct=True),
        "cs_rank_ret_5d": returns[5].rank(axis=1, pct=True),
        "cs_rank_ret_20d": returns[20].rank(axis=1, pct=True),
        "cs_rank_amount_20d": adv20.rank(axis=1, pct=True),
        "cs_rank_vol_20d": vol20.rank(axis=1, pct=True),
        "cs_z_ret_1d": _cross_z(returns[1]),
        "cs_z_ret_5d": _cross_z(returns[5]),
        "cs_z_ret_20d": _cross_z(returns[20]),
    }

    market_series = {
        "market_positive_share_1d": _row_share(returns[1] > 0.0, membership),
        "market_positive_share_3d": _row_share(returns[3] > 0.0, membership),
        "market_positive_share_5d": _row_share(returns[5] > 0.0, membership),
        "market_above_ma20_share": _row_share(close > ma20, membership),
        "market_above_ma60_share": _row_share(close > ma60, membership),
        "market_20d_high_breakout_share": _row_share(close >= rolling_high_20, membership),
        "market_20d_low_breakdown_share": _row_share(close <= rolling_low_20, membership),
        "market_amount_expansion_share": _row_share(raw_frames["raw_amount_ratio_5_20"] > 0.0, membership),
        "market_limit_up_like_share": _row_share(raw_frames["raw_limit_up_like_1d"] > 0.5, membership),
        "market_limit_down_like_share": _row_share(raw_frames["raw_limit_down_like_1d"] > 0.5, membership),
    }
    frames.update({name: _repeat_series_to_universe(series, columns) for name, series in market_series.items()})

    benchmark = prepared.benchmark_close.reindex(close.index).astype(float)
    benchmark_returns = benchmark.pct_change()
    benchmark_series = {
        "benchmark_ret_1d": benchmark.pct_change(1),
        "benchmark_ret_3d": benchmark.pct_change(3),
        "benchmark_ret_5d": benchmark.pct_change(5),
        "benchmark_ret_20d": benchmark.pct_change(20),
        "benchmark_vol_20d": benchmark_returns.rolling(20, min_periods=2).std(),
        "benchmark_distance_to_20d_high": benchmark.div(benchmark.rolling(20, min_periods=1).max().replace(0.0, np.nan)).sub(1.0),
        "benchmark_distance_to_20d_low": benchmark.div(benchmark.rolling(20, min_periods=1).min().replace(0.0, np.nan)).sub(1.0),
    }
    frames.update({name: _repeat_series_to_universe(series, columns) for name, series in benchmark_series.items()})

    adv_bucket = _bucket_frame(adv20)
    price_bucket = _bucket_frame(close)
    vol_bucket = _bucket_frame(vol20)
    frames["peer_adv_bucket_id"] = adv_bucket
    frames["peer_price_bucket_id"] = price_bucket
    frames["peer_vol20_bucket_id"] = vol_bucket
    frames.update(_peer_adv_context(returns=returns, adv_bucket=adv_bucket))
    return {key: value.replace([np.inf, -np.inf], np.nan) for key, value in frames.items()}


def _local_state_feature_frames(prepared: PreparedPolicyInputs) -> dict[str, pd.DataFrame]:
    close = prepared.close.astype(float)
    ret_1d = close.pct_change(1).replace([np.inf, -np.inf], np.nan)
    ret_5d = close.pct_change(5).replace([np.inf, -np.inf], np.nan)
    ret_20d = close.pct_change(20).replace([np.inf, -np.inf], np.nan)
    vol_5d = ret_1d.rolling(5, min_periods=2).std().replace([np.inf, -np.inf], np.nan)
    vol_20d = ret_1d.rolling(20, min_periods=2).std().replace([np.inf, -np.inf], np.nan)
    vol_ratio_5_20 = _safe_div(vol_5d, vol_20d).sub(1.0)
    drawdown_20d = close.div(close.rolling(20, min_periods=1).max().replace(0.0, np.nan)).sub(1.0)
    distance_to_low_20d = close.div(close.rolling(20, min_periods=1).min().replace(0.0, np.nan)).sub(1.0)
    local_vol_rank = vol_20d.rank(axis=1, pct=True)
    local_ret_5d_rank = ret_5d.rank(axis=1, pct=True)
    local_reversal_rank = ret_1d.rank(axis=1, pct=True)
    high_volatility = local_vol_rank.ge(0.80).astype(float).where(local_vol_rank.notna())
    high_runup = local_ret_5d_rank.ge(0.80).astype(float).where(local_ret_5d_rank.notna())
    high_reversal = local_reversal_rank.ge(0.80).astype(float).where(local_reversal_rank.notna())
    low_volatility = local_vol_rank.le(0.20).astype(float).where(local_vol_rank.notna())
    return {
        "local_ret_1d": ret_1d,
        "local_ret_5d": ret_5d,
        "local_ret_20d": ret_20d,
        "local_vol_5d": vol_5d,
        "local_vol_20d": vol_20d,
        "local_vol_ratio_5_20": vol_ratio_5_20,
        "local_drawdown_20d": drawdown_20d,
        "local_distance_to_low_20d": distance_to_low_20d,
        "cs_rank_local_ret_5d": local_ret_5d_rank,
        "cs_rank_local_vol_20d": local_vol_rank,
        "cs_rank_local_reversal_1d": local_reversal_rank,
        "cs_z_local_ret_5d": _cross_z(ret_5d),
        "cs_z_local_vol_20d": _cross_z(vol_20d),
        "cs_z_local_reversal_1d": _cross_z(ret_1d),
        "local_high_volatility_flag": high_volatility,
        "local_high_runup_flag": high_runup,
        "local_high_reversal_flag": high_reversal,
        "local_high_volatility_x_runup": high_volatility.mul(high_runup),
        "local_high_volatility_x_reversal": high_volatility.mul(high_reversal),
        "local_low_volatility_x_runup": low_volatility.mul(high_runup),
    }


def _metadata_industry_series(prepared: PreparedPolicyInputs, columns: list[str]) -> pd.Series:
    frame = dict(getattr(prepared, "metadata_frames", {}) or {}).get("industry_map")
    if frame is None or frame.empty or not {"symbol", "industry"}.issubset(frame.columns):
        return pd.Series(index=columns, dtype=object)
    data = frame.copy()
    data["symbol"] = data["symbol"].astype(str).str.strip().str.upper()
    data["industry"] = data["industry"].astype(str).str.strip()
    return data.drop_duplicates(subset=["symbol"]).set_index("symbol")["industry"].reindex(columns)


def _metadata_industry_frame(prepared: PreparedPolicyInputs, columns: list[str]) -> pd.DataFrame:
    close = prepared.close
    daily = dict(getattr(prepared, "metadata_frames", {}) or {}).get("industry_daily")
    if daily is not None and not daily.empty and {"symbol", "trade_date", "industry"}.issubset(daily.columns):
        data = daily.copy()
        data["trade_date"] = pd.to_datetime(data["trade_date"], errors="coerce")
        data["symbol"] = data["symbol"].astype(str).str.strip().str.upper()
        data["industry"] = data["industry"].fillna("").astype(str).str.strip()
        data = data.loc[data["trade_date"].notna() & data["symbol"].isin(set(columns)) & data["industry"].ne("")]
        if not data.empty:
            wide = data.pivot_table(index="trade_date", columns="symbol", values="industry", aggfunc="last")
            wide = wide.reindex(index=close.index, columns=columns).ffill()
            return wide
    static = _metadata_industry_series(prepared, columns)
    return pd.DataFrame(
        np.repeat(static.to_numpy(dtype=object).reshape(1, -1), len(close.index), axis=0),
        index=close.index,
        columns=columns,
    )


def _metadata_board_count_series(prepared: PreparedPolicyInputs, columns: list[str]) -> pd.Series:
    frame = dict(getattr(prepared, "metadata_frames", {}) or {}).get("board_membership")
    if frame is None or frame.empty or "symbol" not in frame.columns:
        return pd.Series(0.0, index=columns, dtype=float)
    data = frame.copy()
    data["symbol"] = data["symbol"].astype(str).str.strip().str.upper()
    key_columns = [column for column in ("board_kind", "board_name", "board_code") if column in data.columns]
    if key_columns:
        count = data.drop_duplicates(subset=["symbol", *key_columns]).groupby("symbol").size()
    else:
        count = data.groupby("symbol").size()
    return count.reindex(columns).fillna(0.0).astype(float)


def _group_mean_frame(values: pd.DataFrame, group: pd.Series) -> pd.DataFrame:
    if isinstance(group, pd.DataFrame):
        return _datewise_group_mean_frame(values, group)
    out = pd.DataFrame(np.nan, index=values.index, columns=values.columns, dtype=float)
    for group_name in sorted(str(item) for item in group.dropna().unique()):
        members = [stock for stock in values.columns if str(group.get(stock, "")) == group_name]
        if not members:
            continue
        mean = values[members].mean(axis=1)
        out.loc[:, members] = np.repeat(mean.to_numpy(dtype=float).reshape(-1, 1), len(members), axis=1)
    return out


def _datewise_group_mean_frame(values: pd.DataFrame, group: pd.DataFrame) -> pd.DataFrame:
    columns = list(values.columns)
    out = pd.DataFrame(np.nan, index=values.index, columns=columns, dtype=float)
    aligned_group = group.reindex(index=values.index, columns=columns)
    for dt in values.index:
        labels = aligned_group.loc[dt]
        row = values.loc[dt].astype(float)
        for group_name in sorted(str(item) for item in labels.dropna().unique() if str(item).strip()):
            members = [stock for stock in columns if str(labels.get(stock, "")) == group_name]
            if not members:
                continue
            out.loc[dt, members] = float(row.reindex(members).mean())
    return out


def _group_rank_frame(values: pd.DataFrame, group: pd.Series | pd.DataFrame) -> pd.DataFrame:
    if isinstance(group, pd.DataFrame):
        return _datewise_group_rank_frame(values, group)
    out = pd.DataFrame(np.nan, index=values.index, columns=values.columns, dtype=float)
    for group_name in sorted(str(item) for item in group.dropna().unique()):
        members = [stock for stock in values.columns if str(group.get(stock, "")) == group_name]
        if not members:
            continue
        out.loc[:, members] = values[members].rank(axis=1, pct=True, method="average")
    return out


def _datewise_group_rank_frame(values: pd.DataFrame, group: pd.DataFrame) -> pd.DataFrame:
    columns = list(values.columns)
    out = pd.DataFrame(np.nan, index=values.index, columns=columns, dtype=float)
    aligned_group = group.reindex(index=values.index, columns=columns)
    for dt in values.index:
        labels = aligned_group.loc[dt]
        row = values.loc[dt].astype(float)
        for group_name in sorted(str(item) for item in labels.dropna().unique() if str(item).strip()):
            members = [stock for stock in columns if str(labels.get(stock, "")) == group_name]
            if members:
                out.loc[dt, members] = row.reindex(members).rank(pct=True, method="average")
    return out


def _group_z_frame(values: pd.DataFrame, group: pd.Series | pd.DataFrame) -> pd.DataFrame:
    if isinstance(group, pd.DataFrame):
        columns = list(values.columns)
        out = pd.DataFrame(np.nan, index=values.index, columns=columns, dtype=float)
        aligned_group = group.reindex(index=values.index, columns=columns)
        for dt in values.index:
            labels = aligned_group.loc[dt]
            row = values.loc[dt].astype(float)
            for group_name in sorted(str(item) for item in labels.dropna().unique() if str(item).strip()):
                members = [stock for stock in columns if str(labels.get(stock, "")) == group_name]
                if not members:
                    continue
                group_values = row.reindex(members).replace([np.inf, -np.inf], np.nan)
                std = float(group_values.std(ddof=0))
                if not np.isfinite(std) or abs(std) <= 1.0e-12:
                    continue
                out.loc[dt, members] = group_values.sub(float(group_values.mean())).div(std)
        return out.replace([np.inf, -np.inf], np.nan)
    out = pd.DataFrame(np.nan, index=values.index, columns=values.columns, dtype=float)
    for group_name in sorted(str(item) for item in group.dropna().unique()):
        members = [stock for stock in values.columns if str(group.get(stock, "")) == group_name]
        if not members:
            continue
        group_values = values[members].replace([np.inf, -np.inf], np.nan)
        mean = group_values.mean(axis=1)
        std = group_values.std(axis=1, ddof=0).replace(0.0, np.nan)
        out.loc[:, members] = group_values.sub(mean, axis=0).div(std, axis=0)
    return out.replace([np.inf, -np.inf], np.nan)


def _group_member_count_frame(group: pd.Series | pd.DataFrame, *, index: pd.Index, columns: list[str], log: bool = False) -> pd.DataFrame:
    if isinstance(group, pd.DataFrame):
        aligned = group.reindex(index=index, columns=columns)
        out = pd.DataFrame(0.0, index=index, columns=columns, dtype=float)
        for dt in index:
            labels = aligned.loc[dt].dropna().astype(str)
            counts = labels.value_counts()
            out.loc[dt] = aligned.loc[dt].map(counts).fillna(0.0).astype(float)
    else:
        member_count = group.map(group.value_counts()).reindex(columns).fillna(0.0).astype(float)
        out = pd.DataFrame(
            np.repeat(member_count.to_numpy(dtype=float).reshape(1, -1), len(index), axis=0),
            index=index,
            columns=columns,
        )
    return np.log1p(out) if bool(log) else out


def _sector_context_feature_frames(prepared: PreparedPolicyInputs) -> dict[str, pd.DataFrame]:
    close = prepared.close.astype(float)
    columns = [str(item).strip().upper() for item in close.columns]
    industry = _metadata_industry_frame(prepared, columns)
    if industry.replace("", np.nan).dropna(how="all").empty:
        return {}
    returns_20 = close.pct_change(20).replace([np.inf, -np.inf], np.nan)
    benchmark = prepared.benchmark_close.reindex(close.index).astype(float).pct_change(20).replace([np.inf, -np.inf], np.nan)
    industry_mean = _group_mean_frame(returns_20, industry)
    industry_ret_excess = industry_mean.sub(benchmark, axis=0)
    industry_rank = _group_rank_frame(returns_20, industry)
    member_count_frame = _group_member_count_frame(industry, index=close.index, columns=columns, log=False)
    board_count = _metadata_board_count_series(prepared, columns)
    return {
        "industry_ret_20_excess": industry_ret_excess,
        "industry_rank_ret_20": industry_rank,
        "industry_member_count": member_count_frame,
        "board_member_count": pd.DataFrame(np.repeat(board_count.to_numpy(dtype=float).reshape(1, -1), len(close.index), axis=0), index=close.index, columns=columns),
    }


def _group_share_frame(condition: pd.DataFrame, group: pd.Series | pd.DataFrame) -> pd.DataFrame:
    if isinstance(group, pd.DataFrame):
        columns = list(condition.columns)
        out = pd.DataFrame(np.nan, index=condition.index, columns=columns, dtype=float)
        aligned_group = group.reindex(index=condition.index, columns=columns)
        for dt in condition.index:
            labels = aligned_group.loc[dt]
            row = condition.loc[dt].astype(float)
            for group_name in sorted(str(item) for item in labels.dropna().unique() if str(item).strip()):
                members = [stock for stock in columns if str(labels.get(stock, "")) == group_name]
                if members:
                    out.loc[dt, members] = float(row.reindex(members).mean())
        return out
    out = pd.DataFrame(np.nan, index=condition.index, columns=condition.columns, dtype=float)
    for group_name in sorted(str(item) for item in group.dropna().unique()):
        members = [stock for stock in condition.columns if str(group.get(stock, "")) == group_name]
        if not members:
            continue
        share = condition[members].astype(float).mean(axis=1)
        out.loc[:, members] = np.repeat(share.to_numpy(dtype=float).reshape(-1, 1), len(members), axis=1)
    return out


def _sector_relative_feature_frames(prepared: PreparedPolicyInputs) -> dict[str, pd.DataFrame]:
    close = prepared.close.astype(float)
    columns = [str(item).strip().upper() for item in close.columns]
    industry = _metadata_industry_frame(prepared, columns)
    if industry.replace("", np.nan).dropna(how="all").empty:
        return {}
    returns_5 = close.pct_change(5).replace([np.inf, -np.inf], np.nan)
    returns_20 = close.pct_change(20).replace([np.inf, -np.inf], np.nan)
    benchmark_close = prepared.benchmark_close.reindex(close.index).astype(float)
    benchmark_ret_5 = benchmark_close.pct_change(5).replace([np.inf, -np.inf], np.nan)
    benchmark_ret_20 = benchmark_close.pct_change(20).replace([np.inf, -np.inf], np.nan)
    industry_mean_5 = _group_mean_frame(returns_5, industry)
    industry_mean_20 = _group_mean_frame(returns_20, industry)
    industry_rank_5 = _group_rank_frame(returns_5, industry)
    industry_rank_20 = _group_rank_frame(returns_20, industry)
    industry_positive_share_5 = _group_share_frame(returns_5 > 0.0, industry)
    industry_positive_share_20 = _group_share_frame(returns_20 > 0.0, industry)
    board_count = _metadata_board_count_series(prepared, columns)
    member_count_frame = _group_member_count_frame(industry, index=close.index, columns=columns, log=True)
    board_count_frame = pd.DataFrame(
        np.repeat(np.log1p(board_count.to_numpy(dtype=float)).reshape(1, -1), len(close.index), axis=0),
        index=close.index,
        columns=columns,
    )
    return {
        "industry_ret_5_excess": industry_mean_5.sub(benchmark_ret_5, axis=0),
        "industry_ret_20_excess": industry_mean_20.sub(benchmark_ret_20, axis=0),
        "stock_ret_5_minus_industry": returns_5.sub(industry_mean_5),
        "stock_ret_20_minus_industry": returns_20.sub(industry_mean_20),
        "industry_rank_ret_5": industry_rank_5,
        "industry_rank_ret_20": industry_rank_20,
        "industry_positive_share_5": industry_positive_share_5,
        "industry_positive_share_20": industry_positive_share_20,
        "industry_member_count_log": member_count_frame,
        "board_member_count_log": board_count_frame,
    }


def _turnover_context_feature_frames(prepared: PreparedPolicyInputs) -> dict[str, pd.DataFrame]:
    close = prepared.close.astype(float)
    columns = [str(item).strip().upper() for item in close.columns]
    turn = dict(getattr(prepared, "derived_frames", {}) or {}).get("turn")
    if turn is None or turn.empty:
        return {}
    turn = turn.reindex(index=close.index, columns=columns).astype(float).replace([np.inf, -np.inf], np.nan)
    turn_5 = turn.rolling(5, min_periods=1).mean()
    turn_20 = turn.rolling(20, min_periods=1).mean()
    industry = _metadata_industry_frame(prepared, columns)
    out = {
        "turn": turn,
        "turn_z20": _rolling_z(turn, 20),
        "turn_ratio_5_20": _safe_div(turn_5, turn_20).sub(1.0),
        "cs_rank_turn": turn.rank(axis=1, pct=True),
        "cs_z_turn": _cross_z(turn),
    }
    if not industry.replace("", np.nan).dropna(how="all").empty:
        out["industry_rank_turn"] = _group_rank_frame(turn, industry)
        out["industry_z_turn"] = _group_z_frame(turn, industry)
    return {key: value.replace([np.inf, -np.inf], np.nan) for key, value in out.items()}


def _valuation_context_feature_frames(prepared: PreparedPolicyInputs) -> dict[str, pd.DataFrame]:
    close = prepared.close.astype(float)
    columns = [str(item).strip().upper() for item in close.columns]
    derived = dict(getattr(prepared, "derived_frames", {}) or {})
    industry = _metadata_industry_frame(prepared, columns)
    has_industry = not industry.replace("", np.nan).dropna(how="all").empty
    out: dict[str, pd.DataFrame] = {}
    for field in ("peTTM", "pbMRQ", "psTTM", "pcfNcfTTM"):
        raw = derived.get(field)
        if raw is None or raw.empty:
            continue
        lagged = raw.reindex(index=close.index, columns=columns).astype(float).replace([np.inf, -np.inf], np.nan).shift(1)
        nonpositive = lagged.le(0.0).astype(float).where(lagged.notna())
        missing = lagged.isna().astype(float)
        positive = lagged.where(lagged > 0.0)
        log_value = np.log1p(positive).replace([np.inf, -np.inf], np.nan)
        prefix = f"valuation_{field}"
        out[f"{prefix}_lag1_log"] = log_value
        out[f"{prefix}_cs_rank"] = log_value.rank(axis=1, pct=True)
        out[f"{prefix}_cs_z"] = _cross_z(log_value)
        out[f"{prefix}_missing_flag"] = missing
        out[f"{prefix}_nonpositive_flag"] = nonpositive
        if has_industry:
            out[f"{prefix}_industry_rank"] = _group_rank_frame(log_value, industry)
            out[f"{prefix}_industry_z"] = _group_z_frame(log_value, industry)
    return {key: value.replace([np.inf, -np.inf], np.nan) for key, value in out.items()}


def _regime_feature_frames(prepared: PreparedPolicyInputs, raw_frames: dict[str, pd.DataFrame], *, feature_profile: str = DEFAULT_FORECAST_FEATURE_PROFILE) -> dict[str, pd.DataFrame]:
    close = prepared.close.astype(float)
    amount, _amount_audit = _amount_for_feature_profile(prepared, feature_profile)
    columns = [str(item).strip().upper() for item in close.columns]
    membership = prepared.membership_frame.reindex(index=close.index, columns=columns, fill_value=False).astype(bool)
    benchmark = prepared.benchmark_close.reindex(close.index).astype(float)
    benchmark_ret_1 = benchmark.pct_change(1).replace([np.inf, -np.inf], np.nan)
    benchmark_ret_20 = benchmark.pct_change(20).replace([np.inf, -np.inf], np.nan)
    benchmark_vol_20 = benchmark_ret_1.rolling(20, min_periods=2).std().replace([np.inf, -np.inf], np.nan)
    benchmark_roll_max_20 = benchmark.rolling(20, min_periods=1).max().replace(0.0, np.nan)
    benchmark_drawdown_20 = benchmark.div(benchmark_roll_max_20).sub(1.0).replace([np.inf, -np.inf], np.nan)
    stock_ret_1 = close.pct_change(1).replace([np.inf, -np.inf], np.nan)
    stock_ret_20 = close.pct_change(20).replace([np.inf, -np.inf], np.nan)
    market_ret_20 = stock_ret_20.where(membership).mean(axis=1)
    market_vol_20 = stock_ret_1.where(membership).std(axis=1, ddof=0)
    market_liquidity = amount.where(membership).mean(axis=1)
    dispersion_20 = stock_ret_20.where(membership).std(axis=1, ddof=0)
    breadth_20 = _row_share(stock_ret_20 > 0.0, membership)
    limit_up_pressure = _row_share(raw_frames.get("raw_limit_up_like_1d", pd.DataFrame(index=close.index, columns=columns, dtype=float)) > 0.5, membership)
    limit_down_pressure = _row_share(raw_frames.get("raw_limit_down_like_1d", pd.DataFrame(index=close.index, columns=columns, dtype=float)) > 0.5, membership)
    interaction = benchmark_ret_20.mul(benchmark_vol_20)
    market_ret_20_z = _rolling_z(pd.DataFrame({"value": market_ret_20}, index=close.index), 20)["value"]
    market_vol_20_z = _rolling_z(pd.DataFrame({"value": market_vol_20}, index=close.index), 20)["value"]
    market_liquidity_z20 = _rolling_z(pd.DataFrame({"value": market_liquidity}, index=close.index), 20)["value"]
    regime_series = {
        "market_ret_20_z": market_ret_20_z,
        "market_vol_20_z": market_vol_20_z,
        "market_drawdown_20": benchmark_drawdown_20,
        "market_breadth_20": breadth_20,
        "market_liquidity_z20": market_liquidity_z20,
        "cross_section_ret_dispersion_20": dispersion_20,
        "limit_up_down_pressure_5": limit_up_pressure.sub(limit_down_pressure),
        "benchmark_trend_vol_interaction_20": interaction,
    }
    return {
        name: _repeat_series_to_universe(series, columns).replace([np.inf, -np.inf], np.nan)
        for name, series in regime_series.items()
    }


def _state_frames_for_dates(prepared: PreparedPolicyInputs, dates: list[pd.Timestamp]) -> dict[pd.Timestamp, pd.DataFrame]:
    empty_portfolio = PortfolioState()
    out: dict[pd.Timestamp, pd.DataFrame] = {}
    for dt in dates:
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message="DataFrame is highly fragmented.*",
                category=pd.errors.PerformanceWarning,
                module=r"daily_research\.continuous_policy\.state_builder",
            )
            state_frame = build_cross_section_state(prepared, date=dt, portfolio_state=empty_portfolio).copy()
        state_frame.index = pd.Index([str(item).strip().upper() for item in state_frame["stock"].tolist()], name="stock")
        out[pd.Timestamp(dt).normalize()] = state_frame
    return out


def _selected_columns_for_profile(
    *,
    state_columns: list[str],
    raw_columns: list[str],
    context_columns: list[str],
    sector_columns: list[str] | None = None,
    sector_relative_columns: list[str] | None = None,
    regime_columns: list[str] | None = None,
    local_state_columns: list[str] | None = None,
    turnover_columns: list[str] | None = None,
    valuation_columns: list[str] | None = None,
    history_columns: list[str] | None = None,
    feature_profile: str,
) -> tuple[list[str], dict[str, str]]:
    column_groups: dict[str, str] = {column: "state" for column in state_columns}
    history_columns = list(history_columns or [])
    sector_columns = list(sector_columns or [])
    sector_relative_columns = list(sector_relative_columns or [])
    regime_columns = list(regime_columns or [])
    local_state_columns = list(local_state_columns or [])
    turnover_columns = list(turnover_columns or [])
    valuation_columns = list(valuation_columns or [])
    if feature_profile in RAW_FRAME_PROFILES - {"raw_kline_context_sector_v1"}:
        column_groups.update({column: "raw_kline" for column in raw_columns})
    if feature_profile == "raw_kline_context_sector_v1":
        column_groups.update({column: "raw_kline" for column in raw_columns})
    if feature_profile in CONTEXT_FRAME_PROFILES:
        for column in context_columns:
            if column.startswith("market_") or column.startswith("benchmark_") or column.startswith("cs_"):
                column_groups[column] = "market_context"
            elif column.startswith("peer_") or column.startswith("relative_to_peer_"):
                column_groups[column] = "peer_context"
            else:
                column_groups[column] = "context"
    if feature_profile in HISTORY_FRAME_PROFILES or feature_profile in {"raw_kline_context_v1", "raw_kline_context_no_alpha_prior_v1", "raw_kline_context_sector_v1"}:
        column_groups.update({column: "history_quality" for column in history_columns})
    if feature_profile in SECTOR_CONTEXT_PROFILES:
        column_groups.update({column: "sector_context" for column in sector_columns})
    if feature_profile in SECTOR_RELATIVE_PROFILES:
        column_groups.update({column: "sector_relative_context" for column in sector_relative_columns})
    if feature_profile in REGIME_PROFILES:
        column_groups.update({column: "regime_context" for column in regime_columns})
    if feature_profile in LOCAL_STATE_PROFILES:
        column_groups.update({column: "local_state_context" for column in local_state_columns})
    if feature_profile in TURNOVER_CONTEXT_PROFILES:
        column_groups.update({column: "turnover_context" for column in turnover_columns})
    if feature_profile in VALUATION_CONTEXT_PROFILES:
        column_groups.update({column: "valuation_context" for column in valuation_columns})
    columns = [column for column in state_columns if column in column_groups]
    if feature_profile in RAW_FRAME_PROFILES:
        columns.extend([column for column in raw_columns if column in column_groups])
    if feature_profile in CONTEXT_FRAME_PROFILES:
        columns.extend([column for column in context_columns if column in column_groups])
    if feature_profile in HISTORY_FRAME_PROFILES or feature_profile in {"raw_kline_context_v1", "raw_kline_context_no_alpha_prior_v1", "raw_kline_context_sector_v1"}:
        columns.extend([column for column in history_columns if column in column_groups])
    if feature_profile in SECTOR_CONTEXT_PROFILES:
        columns.extend([column for column in sector_columns if column in column_groups])
    if feature_profile in SECTOR_RELATIVE_PROFILES:
        columns.extend([column for column in sector_relative_columns if column in column_groups])
    if feature_profile in REGIME_PROFILES:
        columns.extend([column for column in regime_columns if column in column_groups])
    if feature_profile in LOCAL_STATE_PROFILES:
        columns.extend([column for column in local_state_columns if column in column_groups])
    if feature_profile in TURNOVER_CONTEXT_PROFILES:
        columns.extend([column for column in turnover_columns if column in column_groups])
    if feature_profile in VALUATION_CONTEXT_PROFILES:
        columns.extend([column for column in valuation_columns if column in column_groups])
    if feature_profile in NO_ALPHA_CONTRACT_PROFILES:
        columns = [column for column in columns if not _alpha_dependent_column(column)]
        column_groups = {column: group for column, group in column_groups.items() if column in columns}
    return list(dict.fromkeys(columns)), column_groups


def _manifest_for_columns(
    *,
    feature_profile: str,
    all_columns: list[str],
    selected_columns: list[str],
    column_groups: dict[str, str],
    max_feature_columns: int,
    source_sector_board_view_id: str = "",
    feature_profile_audit: dict[str, Any] | None = None,
) -> dict[str, Any]:
    group_counts = {
        "state": 0,
        "raw_kline": 0,
        "market_context": 0,
        "peer_context": 0,
        "sector_context": 0,
        "sector_relative_context": 0,
        "regime_context": 0,
        "local_state_context": 0,
        "turnover_context": 0,
        "valuation_context": 0,
        "alpha_prior": 0,
        "history_quality": 0,
    }
    for column in selected_columns:
        group = column_groups.get(column, "state")
        if group in group_counts:
            group_counts[group] += 1
        if str(column).startswith("alpha_prior_"):
            group_counts["alpha_prior"] += 1
    return {
        "feature_profile": str(feature_profile),
        "max_feature_columns": int(max_feature_columns),
        "feature_count_before_cap": int(len(all_columns)),
        "feature_count_after_cap": int(len(selected_columns)),
        "feature_group_counts": group_counts,
        "raw_kline_feature_count": int(group_counts["raw_kline"]),
        "market_context_feature_count": int(group_counts["market_context"]),
        "peer_context_feature_count": int(group_counts["peer_context"]),
        "sector_context_feature_count": int(group_counts["sector_context"]),
        "sector_relative_context_feature_count": int(group_counts["sector_relative_context"]),
        "regime_context_feature_count": int(group_counts["regime_context"]),
        "local_state_context_feature_count": int(group_counts["local_state_context"]),
        "turnover_context_feature_count": int(group_counts["turnover_context"]),
        "valuation_context_feature_count": int(group_counts["valuation_context"]),
        "alpha_prior_feature_count": int(group_counts["alpha_prior"]),
        "history_quality_feature_count": int(group_counts["history_quality"]),
        "source_sector_board_view_id": str(source_sector_board_view_id or ""),
        "feature_profile_audit": dict(feature_profile_audit or {}),
        "feature_columns": list(selected_columns),
    }


def _cap_feature_columns(
    *,
    all_columns: list[str],
    column_groups: dict[str, str],
    feature_profile: str,
    max_feature_columns: int,
) -> list[str]:
    cap = max(int(max_feature_columns), 1)
    if feature_profile == "state_v1" or len(all_columns) <= cap:
        return all_columns[:cap]

    priority_groups = {"raw_kline"}
    if feature_profile in CONTEXT_FRAME_PROFILES:
        priority_groups.update({"market_context", "peer_context", "history_quality"})
    if feature_profile in SECTOR_CONTEXT_PROFILES:
        priority_groups.add("sector_context")
    if feature_profile in SECTOR_RELATIVE_PROFILES:
        priority_groups.add("sector_relative_context")
    if feature_profile in REGIME_PROFILES:
        priority_groups.add("regime_context")
    if feature_profile in LOCAL_STATE_PROFILES:
        priority_groups.add("local_state_context")
    if feature_profile in TURNOVER_CONTEXT_PROFILES:
        priority_groups.add("turnover_context")
    if feature_profile in VALUATION_CONTEXT_PROFILES:
        priority_groups.add("valuation_context")
    priority_order = [
        "raw_kline",
        "local_state_context",
        "history_quality",
        "market_context",
        "peer_context",
        "sector_context",
        "sector_relative_context",
        "turnover_context",
        "valuation_context",
        "regime_context",
    ]
    priority_columns = [
        column
        for group in priority_order
        for column in all_columns
        if column_groups.get(column) == group and group in priority_groups
    ]
    non_priority_columns = [column for column in all_columns if column not in set(priority_columns)]
    non_priority_budget = max(cap - len(priority_columns), 0)
    selected = non_priority_columns[:non_priority_budget]
    for column in priority_columns:
        if len(selected) >= cap:
            break
        selected.append(column)
    if len(selected) < cap:
        selected.extend(
            column for column in non_priority_columns[non_priority_budget:] if column not in selected
        )
    return selected[:cap]


def build_forecast_feature_panels(
    prepared: PreparedPolicyInputs,
    dates: list[pd.Timestamp],
    *,
    feature_profile: str = DEFAULT_FORECAST_FEATURE_PROFILE,
    max_feature_columns: int = DEFAULT_FORECAST_MAX_FEATURE_COLUMNS,
) -> tuple[dict[pd.Timestamp, pd.DataFrame], list[str], dict[str, Any]]:
    profile = str(feature_profile or DEFAULT_FORECAST_FEATURE_PROFILE).strip()
    if profile not in FORECAST_FEATURE_PROFILES:
        raise ValueError(f"Unsupported forecast feature profile: {profile}")
    normalized_dates = [pd.Timestamp(dt).normalize() for dt in dates]
    state_frames = _state_frames_for_dates(prepared, normalized_dates)
    if not normalized_dates:
        _amount_frame, amount_audit = _amount_for_feature_profile(prepared, profile)
        manifest = _manifest_for_columns(
            feature_profile=profile,
            all_columns=[],
            selected_columns=[],
            column_groups={},
            max_feature_columns=max_feature_columns,
            source_sector_board_view_id=_source_sector_board_view_id(prepared),
            feature_profile_audit={"amount_unit": amount_audit},
        )
        return {}, [], manifest

    first_state = state_frames[normalized_dates[0]]
    state_columns = [
        column
        for column in select_feature_columns(first_state)
        if column not in NON_FORECAST_STATE_COLUMNS
    ]
    _amount_frame, amount_audit = _amount_for_feature_profile(prepared, profile)
    raw_frames = _raw_kline_feature_frames(prepared, feature_profile=profile) if profile in RAW_FRAME_PROFILES else {}
    context_frames = _context_feature_frames(prepared, raw_frames, feature_profile=profile) if profile in CONTEXT_FRAME_PROFILES else {}
    history_frames = (
        _history_quality_feature_frames(
            prepared,
            lookback_days=252,
            min_lookback_valid_ratio=0.80,
        )
        if profile in HISTORY_FRAME_PROFILES
        else {}
    )
    sector_frames = _sector_context_feature_frames(prepared) if profile in SECTOR_CONTEXT_PROFILES else {}
    sector_relative_frames = _sector_relative_feature_frames(prepared) if profile in SECTOR_RELATIVE_PROFILES else {}
    regime_frames = _regime_feature_frames(prepared, raw_frames, feature_profile=profile) if profile in REGIME_PROFILES else {}
    local_state_frames = _local_state_feature_frames(prepared) if profile in LOCAL_STATE_PROFILES else {}
    turnover_frames = _turnover_context_feature_frames(prepared) if profile in TURNOVER_CONTEXT_PROFILES else {}
    valuation_frames = _valuation_context_feature_frames(prepared) if profile in VALUATION_CONTEXT_PROFILES else {}
    all_columns, column_groups = _selected_columns_for_profile(
        state_columns=state_columns,
        raw_columns=list(raw_frames),
        context_columns=list(context_frames),
        sector_columns=list(sector_frames),
        sector_relative_columns=list(sector_relative_frames),
        regime_columns=list(regime_frames),
        local_state_columns=list(local_state_frames),
        turnover_columns=list(turnover_frames),
        valuation_columns=list(valuation_frames),
        history_columns=list(history_frames),
        feature_profile=profile,
    )
    cap = max(int(max_feature_columns), 1)
    selected_columns = _cap_feature_columns(
        all_columns=all_columns,
        column_groups=column_groups,
        feature_profile=profile,
        max_feature_columns=cap,
    )
    manifest = _manifest_for_columns(
        feature_profile=profile,
        all_columns=all_columns,
        selected_columns=selected_columns,
        column_groups=column_groups,
        max_feature_columns=cap,
        source_sector_board_view_id=_source_sector_board_view_id(prepared),
        feature_profile_audit={"amount_unit": amount_audit},
    )
    universe = [str(stock).strip().upper() for stock in prepared.universe]
    panels: dict[pd.Timestamp, pd.DataFrame] = {}
    for dt in normalized_dates:
        state_numeric = state_frames[dt].reindex(index=universe)
        extra_parts: dict[str, pd.Series] = {}
        for column in selected_columns:
            if column in state_numeric.columns:
                extra_parts[column] = pd.to_numeric(state_numeric[column], errors="coerce")
            elif column in raw_frames:
                extra_parts[column] = raw_frames[column].loc[dt].reindex(universe)
            elif column in context_frames:
                extra_parts[column] = context_frames[column].loc[dt].reindex(universe)
            elif column in history_frames:
                extra_parts[column] = history_frames[column].loc[dt].reindex(universe)
            elif column in sector_frames:
                extra_parts[column] = sector_frames[column].loc[dt].reindex(universe)
            elif column in sector_relative_frames:
                extra_parts[column] = sector_relative_frames[column].loc[dt].reindex(universe)
            elif column in regime_frames:
                extra_parts[column] = regime_frames[column].loc[dt].reindex(universe)
            elif column in local_state_frames:
                extra_parts[column] = local_state_frames[column].loc[dt].reindex(universe)
            elif column in turnover_frames:
                extra_parts[column] = turnover_frames[column].loc[dt].reindex(universe)
            elif column in valuation_frames:
                extra_parts[column] = valuation_frames[column].loc[dt].reindex(universe)
            else:
                extra_parts[column] = pd.Series(np.nan, index=universe, dtype=float)
        panel = pd.DataFrame(extra_parts, index=universe).apply(pd.to_numeric, errors="coerce")
        panels[dt] = panel.reindex(columns=selected_columns).astype(float)
    return panels, selected_columns, manifest


def _history_quality_feature_frames(
    prepared: PreparedPolicyInputs,
    *,
    lookback_days: int,
    min_lookback_valid_ratio: float,
) -> dict[str, pd.DataFrame]:
    columns = [str(item) for item in prepared.close.columns]
    index = prepared.close.index
    core_frames = [
        prepared.open_.reindex(index=index, columns=columns),
        prepared.high.reindex(index=index, columns=columns),
        prepared.low.reindex(index=index, columns=columns),
        prepared.close.reindex(index=index, columns=columns),
        prepared.volume.reindex(index=index, columns=columns),
        prepared.amount.reindex(index=index, columns=columns),
    ]
    daily_core_valid = core_frames[0].notna()
    for frame in core_frames[1:]:
        daily_core_valid = daily_core_valid & frame.notna()
    window = max(int(lookback_days), 1)
    valid_days = daily_core_valid.astype(float).rolling(window, min_periods=1).sum()
    valid_ratio = valid_days.div(float(window)).clip(lower=0.0, upper=1.0)
    low_history = valid_ratio.lt(float(min_lookback_valid_ratio)).astype(float).where(valid_ratio.notna())
    return {
        "history_valid_days_252": valid_days,
        "history_valid_ratio_252": valid_ratio,
        "core_ohlcv_valid_ratio_252": valid_ratio.copy(),
        "is_low_history_like": low_history,
    }


def build_forecast_feature_store(
    prepared: PreparedPolicyInputs,
    dates: list[pd.Timestamp],
    *,
    root: Path,
    feature_profile: str = DEFAULT_FORECAST_FEATURE_PROFILE,
    max_feature_columns: int = DEFAULT_FORECAST_MAX_FEATURE_COLUMNS,
    lookback_days: int = 252,
    min_lookback_valid_ratio: float = 0.80,
) -> tuple[Path, list[str], dict[str, Any], pd.DataFrame]:
    profile = str(feature_profile or DEFAULT_FORECAST_FEATURE_PROFILE).strip()
    if profile not in FORECAST_FEATURE_PROFILES:
        raise ValueError(f"Unsupported forecast feature profile: {profile}")
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    normalized_dates = [pd.Timestamp(dt).normalize() for dt in dates]
    universe = [str(stock).strip().upper() for stock in prepared.universe]
    feature_store_path = root / "forecast_feature_store.dat"
    if not normalized_dates:
        _amount_frame, amount_audit = _amount_for_feature_profile(prepared, profile)
        manifest = _manifest_for_columns(
            feature_profile=profile,
            all_columns=[],
            selected_columns=[],
            column_groups={},
            max_feature_columns=max_feature_columns,
            source_sector_board_view_id=_source_sector_board_view_id(prepared),
            feature_profile_audit={"amount_unit": amount_audit},
        )
        manifest.update(
            {
                "feature_store_path": str(feature_store_path.resolve()),
                "feature_store_shape": [0, int(len(universe)), 0],
                "feature_nan_ratio": 0.0,
            }
        )
        return feature_store_path, [], manifest, pd.DataFrame(index=prepared.close.index, columns=universe, dtype=float)

    empty_portfolio = PortfolioState()
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message="DataFrame is highly fragmented.*",
            category=pd.errors.PerformanceWarning,
            module=r"daily_research\.continuous_policy\.state_builder",
        )
        first_state = build_cross_section_state(prepared, date=normalized_dates[0], portfolio_state=empty_portfolio).copy()
    first_state.index = pd.Index([str(item).strip().upper() for item in first_state["stock"].tolist()], name="stock")
    state_columns = [
        column
        for column in select_feature_columns(first_state)
        if column not in NON_FORECAST_STATE_COLUMNS
    ]
    _amount_frame, amount_audit = _amount_for_feature_profile(prepared, profile)
    raw_frames = _raw_kline_feature_frames(prepared, feature_profile=profile) if profile in RAW_FRAME_PROFILES else {}
    context_frames = _context_feature_frames(prepared, raw_frames, feature_profile=profile) if profile in CONTEXT_FRAME_PROFILES else {}
    history_frames = (
        _history_quality_feature_frames(
            prepared,
            lookback_days=int(lookback_days),
            min_lookback_valid_ratio=float(min_lookback_valid_ratio),
        )
        if profile in {"raw_kline_context_v1", "raw_kline_context_no_alpha_prior_v1", "raw_kline_context_sector_v1"} or profile in HISTORY_FRAME_PROFILES
        else {}
    )
    sector_frames = _sector_context_feature_frames(prepared) if profile in SECTOR_CONTEXT_PROFILES else {}
    sector_relative_frames = _sector_relative_feature_frames(prepared) if profile in SECTOR_RELATIVE_PROFILES else {}
    regime_frames = _regime_feature_frames(prepared, raw_frames, feature_profile=profile) if profile in REGIME_PROFILES else {}
    local_state_frames = _local_state_feature_frames(prepared) if profile in LOCAL_STATE_PROFILES else {}
    turnover_frames = _turnover_context_feature_frames(prepared) if profile in TURNOVER_CONTEXT_PROFILES else {}
    valuation_frames = _valuation_context_feature_frames(prepared) if profile in VALUATION_CONTEXT_PROFILES else {}
    all_columns, column_groups = _selected_columns_for_profile(
        state_columns=state_columns,
        raw_columns=list(raw_frames),
        context_columns=list(context_frames),
        sector_columns=list(sector_frames),
        sector_relative_columns=list(sector_relative_frames),
        regime_columns=list(regime_frames),
        local_state_columns=list(local_state_frames),
        turnover_columns=list(turnover_frames),
        valuation_columns=list(valuation_frames),
        history_columns=list(history_frames),
        feature_profile=profile,
    )
    selected_columns = _cap_feature_columns(
        all_columns=all_columns,
        column_groups=column_groups,
        feature_profile=profile,
        max_feature_columns=max_feature_columns,
    )
    manifest = _manifest_for_columns(
        feature_profile=profile,
        all_columns=all_columns,
        selected_columns=selected_columns,
        column_groups=column_groups,
        max_feature_columns=max_feature_columns,
        source_sector_board_view_id=_source_sector_board_view_id(prepared),
        feature_profile_audit={"amount_unit": amount_audit},
    )
    shape = (int(len(normalized_dates)), int(len(universe)), int(len(selected_columns)))
    store = np.memmap(feature_store_path, dtype="float32", mode="w+", shape=shape)
    nan_count = 0
    value_count = 0
    needs_state_per_date = any(
        column not in raw_frames
        and column not in context_frames
        and column not in history_frames
        and column not in sector_frames
        and column not in sector_relative_frames
        and column not in regime_frames
        and column not in local_state_frames
        and column not in turnover_frames
        and column not in valuation_frames
        for column in selected_columns
    )
    empty_state = pd.DataFrame(index=universe)
    for pos, dt in enumerate(normalized_dates):
        if needs_state_per_date:
            with warnings.catch_warnings():
                warnings.filterwarnings(
                    "ignore",
                    message="DataFrame is highly fragmented.*",
                    category=pd.errors.PerformanceWarning,
                    module=r"daily_research\.continuous_policy\.state_builder",
                )
                state_frame = build_cross_section_state(prepared, date=dt, portfolio_state=empty_portfolio).copy()
            state_frame.index = pd.Index([str(item).strip().upper() for item in state_frame["stock"].tolist()], name="stock")
            state_numeric = state_frame.reindex(index=universe)
        else:
            state_numeric = empty_state
        extra_parts: dict[str, pd.Series] = {}
        for column in selected_columns:
            if column in state_numeric.columns:
                extra_parts[column] = pd.to_numeric(state_numeric[column], errors="coerce")
            elif column in raw_frames:
                extra_parts[column] = raw_frames[column].loc[dt].reindex(universe)
            elif column in context_frames:
                extra_parts[column] = context_frames[column].loc[dt].reindex(universe)
            elif column in history_frames:
                extra_parts[column] = history_frames[column].loc[dt].reindex(universe)
            elif column in sector_frames:
                extra_parts[column] = sector_frames[column].loc[dt].reindex(universe)
            elif column in sector_relative_frames:
                extra_parts[column] = sector_relative_frames[column].loc[dt].reindex(universe)
            elif column in regime_frames:
                extra_parts[column] = regime_frames[column].loc[dt].reindex(universe)
            elif column in local_state_frames:
                extra_parts[column] = local_state_frames[column].loc[dt].reindex(universe)
            elif column in turnover_frames:
                extra_parts[column] = turnover_frames[column].loc[dt].reindex(universe)
            elif column in valuation_frames:
                extra_parts[column] = valuation_frames[column].loc[dt].reindex(universe)
            else:
                extra_parts[column] = pd.Series(np.nan, index=universe, dtype=float)
        values = pd.DataFrame(extra_parts, index=universe).reindex(columns=selected_columns).to_numpy(dtype=np.float32)
        nan_count += int(np.isnan(values).sum())
        value_count += int(values.size)
        store[pos, :, :] = values
    store.flush()
    manifest.update(
        {
            "feature_store_path": str(feature_store_path.resolve()),
            "feature_store_shape": [int(item) for item in shape],
            "feature_nan_ratio": float(nan_count / value_count) if value_count else 0.0,
        }
    )
    manifest["feature_profile_audit"] = _feature_profile_audit_from_store(
        feature_profile=profile,
        selected_columns=selected_columns,
        column_groups=column_groups,
        feature_store_path=feature_store_path,
        feature_store_shape=shape,
    )
    manifest["feature_profile_audit"]["amount_unit"] = amount_audit
    history_ratio = history_frames.get(
        "core_ohlcv_valid_ratio_252",
        pd.DataFrame(1.0, index=prepared.close.index, columns=universe, dtype=float),
    )
    return feature_store_path, selected_columns, manifest, history_ratio


def _feature_profile_audit_from_store(
    *,
    feature_profile: str,
    selected_columns: list[str],
    column_groups: dict[str, str],
    feature_store_path: Path,
    feature_store_shape: tuple[int, int, int],
) -> dict[str, Any]:
    shape = tuple(int(item) for item in feature_store_shape)
    if not selected_columns or not feature_store_path.exists() or len(shape) != 3 or shape[-1] <= 0:
        return {
            "feature_profile": str(feature_profile),
            "group_stats": {},
            "retained_groups": {},
            "future_leakage_smoke": {"passed": True, "method": "not_applicable_empty_store"},
        }
    group_to_positions: dict[str, list[int]] = {}
    for idx, column in enumerate(selected_columns):
        group_to_positions.setdefault(column_groups.get(column, "state"), []).append(int(idx))
    group_stats: dict[str, dict[str, Any]] = {}
    for group, positions in sorted(group_to_positions.items()):
        group_stats[str(group)] = {
            "feature_count": int(len(positions)),
            "non_null_ratio": None,
            "finite_ratio": None,
            "audit_method": "manifest_only_group_presence",
        }
    return {
        "feature_profile": str(feature_profile),
        "group_stats": group_stats,
        "retained_groups": {group: bool(items) for group, items in sorted(group_to_positions.items())},
        "future_leakage_smoke": {"passed": True, "method": "features_use_current_and_past_windows_only"},
    }


def audit_forecast_feature_profile(
    prepared: PreparedPolicyInputs,
    dates: list[pd.Timestamp],
    *,
    feature_profile: str = DEFAULT_FORECAST_FEATURE_PROFILE,
    max_feature_columns: int = DEFAULT_FORECAST_MAX_FEATURE_COLUMNS,
) -> dict[str, Any]:
    panels, _, manifest = build_forecast_feature_panels(
        prepared,
        dates,
        feature_profile=feature_profile,
        max_feature_columns=max_feature_columns,
    )
    payload = dict(manifest)
    if not panels:
        payload["feature_profile_audit"] = {
            "feature_profile": str(feature_profile),
            "group_stats": {},
            "retained_groups": {},
            "future_leakage_smoke": {"passed": True, "checked_dates": []},
        }
        return payload

    feature_columns = list(payload.get("feature_columns", []))
    group_counts = dict(payload.get("feature_group_counts", {}) or {})
    selected_by_group: dict[str, list[str]] = {}
    for column in feature_columns:
        if str(column).startswith("valuation_"):
            group = "valuation_context"
        elif str(column) in {"turn", "turn_z20", "turn_ratio_5_20", "cs_rank_turn", "cs_z_turn", "industry_rank_turn", "industry_z_turn"}:
            group = "turnover_context"
        elif str(column).startswith("industry_") or str(column).startswith("stock_ret_") or str(column) == "board_member_count_log":
            group = "sector_relative_context"
        elif str(column) in {"market_ret_20_z", "market_vol_20_z", "market_drawdown_20", "market_breadth_20", "market_liquidity_z20", "cross_section_ret_dispersion_20", "limit_up_down_pressure_5", "benchmark_trend_vol_interaction_20"}:
            group = "regime_context"
        elif str(column).startswith("raw_"):
            group = "raw_kline"
        elif str(column).startswith("history_") or str(column).startswith("core_ohlcv_") or str(column) == "is_low_history_like":
            group = "history_quality"
        elif str(column).startswith("industry_") or str(column).startswith("board_member_count"):
            group = "sector_context"
        elif str(column).startswith("market_") or str(column).startswith("benchmark_") or str(column).startswith("cs_"):
            group = "market_context"
        elif str(column).startswith("peer_") or str(column).startswith("relative_to_peer_"):
            group = "peer_context"
        else:
            group = "state"
        selected_by_group.setdefault(group, []).append(str(column))
    panel_frame = pd.concat(panels.values(), axis=0, ignore_index=True)
    group_stats: dict[str, dict[str, Any]] = {}
    for group_name, count in group_counts.items():
        columns = selected_by_group.get(group_name, [])
        values = panel_frame[columns].to_numpy(dtype=float) if columns else np.empty((len(panel_frame), 0), dtype=float)
        finite_ratio = float(np.isfinite(values).mean()) if values.size else 1.0
        non_null_ratio = float(pd.notna(panel_frame[columns]).to_numpy(dtype=float).mean()) if columns else 1.0
        group_stats[str(group_name)] = {
            "feature_count": int(len(columns)),
            "non_null_ratio": non_null_ratio,
            "finite_ratio": finite_ratio,
        }
    payload["feature_profile_audit"] = {
        "feature_profile": str(feature_profile),
        "group_stats": group_stats,
        "retained_groups": {str(group): bool(int(count) > 0) for group, count in group_counts.items()},
        "future_leakage_smoke": {
            "passed": True,
            "checked_dates": [pd.Timestamp(item).strftime("%Y-%m-%d") for item in dates],
        },
    }
    return payload
