from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from daily_research.baseline.ml_alpha import build_ml_target
from daily_research.continuous_policy.state_builder import PreparedPolicyInputs


LABEL_HORIZONS: tuple[int, ...] = (1, 3, 5, 10, 20)


@dataclass(frozen=True)
class LifecycleLabelConfig:
    name: str
    min_hold_days: int
    open_entry_threshold: float
    open_signal_threshold: float
    add_quality_threshold: float
    reduce_quality_threshold: float
    exit_urgency_threshold: float
    catastrophic_exit_threshold: float
    gross_scale: float
    min_gross_target: float
    max_gross_target: float
    base_turnover_budget: float
    max_turnover_budget: float
    defensive_cash_bias: float
    reentry_cooldown_days: int
    hold_support_bonus: float
    cash_regime_sensitivity: float
    turnover_sensitivity: float
    profit_take_penalty: float


LABEL_CONFIGS: dict[str, LifecycleLabelConfig] = {
    "balanced_v2": LifecycleLabelConfig(
        name="balanced_v2",
        min_hold_days=2,
        open_entry_threshold=0.085,
        open_signal_threshold=0.025,
        add_quality_threshold=0.095,
        reduce_quality_threshold=0.115,
        exit_urgency_threshold=0.165,
        catastrophic_exit_threshold=0.290,
        gross_scale=0.185,
        min_gross_target=0.22,
        max_gross_target=0.94,
        base_turnover_budget=0.15,
        max_turnover_budget=0.90,
        defensive_cash_bias=0.00,
        reentry_cooldown_days=4,
        hold_support_bonus=0.08,
        cash_regime_sensitivity=0.12,
        turnover_sensitivity=0.08,
        profit_take_penalty=0.06,
    ),
    "swing_v2": LifecycleLabelConfig(
        name="swing_v2",
        min_hold_days=4,
        open_entry_threshold=0.095,
        open_signal_threshold=0.035,
        add_quality_threshold=0.105,
        reduce_quality_threshold=0.130,
        exit_urgency_threshold=0.180,
        catastrophic_exit_threshold=0.300,
        gross_scale=0.175,
        min_gross_target=0.20,
        max_gross_target=0.90,
        base_turnover_budget=0.12,
        max_turnover_budget=0.72,
        defensive_cash_bias=0.03,
        reentry_cooldown_days=5,
        hold_support_bonus=0.10,
        cash_regime_sensitivity=0.15,
        turnover_sensitivity=0.10,
        profit_take_penalty=0.08,
    ),
    "defensive_v2": LifecycleLabelConfig(
        name="defensive_v2",
        min_hold_days=1,
        open_entry_threshold=0.110,
        open_signal_threshold=0.045,
        add_quality_threshold=0.125,
        reduce_quality_threshold=0.100,
        exit_urgency_threshold=0.145,
        catastrophic_exit_threshold=0.240,
        gross_scale=0.155,
        min_gross_target=0.16,
        max_gross_target=0.82,
        base_turnover_budget=0.11,
        max_turnover_budget=0.68,
        defensive_cash_bias=0.08,
        reentry_cooldown_days=5,
        hold_support_bonus=0.08,
        cash_regime_sensitivity=0.18,
        turnover_sensitivity=0.12,
        profit_take_penalty=0.04,
    ),
    "holdcash_v3": LifecycleLabelConfig(
        name="holdcash_v3",
        min_hold_days=5,
        open_entry_threshold=0.102,
        open_signal_threshold=0.036,
        add_quality_threshold=0.112,
        reduce_quality_threshold=0.152,
        exit_urgency_threshold=0.190,
        catastrophic_exit_threshold=0.315,
        gross_scale=0.168,
        min_gross_target=0.18,
        max_gross_target=0.84,
        base_turnover_budget=0.10,
        max_turnover_budget=0.56,
        defensive_cash_bias=0.06,
        reentry_cooldown_days=6,
        hold_support_bonus=0.16,
        cash_regime_sensitivity=0.24,
        turnover_sensitivity=0.16,
        profit_take_penalty=0.12,
    ),
    "holdcash_v4": LifecycleLabelConfig(
        name="holdcash_v4",
        min_hold_days=6,
        open_entry_threshold=0.110,
        open_signal_threshold=0.040,
        add_quality_threshold=0.118,
        reduce_quality_threshold=0.145,
        exit_urgency_threshold=0.205,
        catastrophic_exit_threshold=0.325,
        gross_scale=0.162,
        min_gross_target=0.16,
        max_gross_target=0.82,
        base_turnover_budget=0.09,
        max_turnover_budget=0.48,
        defensive_cash_bias=0.08,
        reentry_cooldown_days=7,
        hold_support_bonus=0.18,
        cash_regime_sensitivity=0.30,
        turnover_sensitivity=0.18,
        profit_take_penalty=0.16,
    ),
    "holdcash_v5": LifecycleLabelConfig(
        name="holdcash_v5",
        min_hold_days=7,
        open_entry_threshold=0.108,
        open_signal_threshold=0.034,
        add_quality_threshold=0.122,
        reduce_quality_threshold=0.162,
        exit_urgency_threshold=0.215,
        catastrophic_exit_threshold=0.332,
        gross_scale=0.158,
        min_gross_target=0.18,
        max_gross_target=0.80,
        base_turnover_budget=0.08,
        max_turnover_budget=0.42,
        defensive_cash_bias=0.07,
        reentry_cooldown_days=8,
        hold_support_bonus=0.24,
        cash_regime_sensitivity=0.28,
        turnover_sensitivity=0.20,
        profit_take_penalty=0.22,
    ),
}


@dataclass(frozen=True)
class FuturePathMetrics:
    frames: dict[str, pd.DataFrame]
    benchmark_returns: dict[str, pd.Series]
    horizons: tuple[int, ...] = LABEL_HORIZONS


def resolve_label_config(value: str | LifecycleLabelConfig | None = None) -> LifecycleLabelConfig:
    if isinstance(value, LifecycleLabelConfig):
        return value
    key = str(value or "balanced_v2").strip().lower() or "balanced_v2"
    resolved = LABEL_CONFIGS.get(key)
    if resolved is None:
        available = ", ".join(sorted(LABEL_CONFIGS))
        raise KeyError(f"Unknown lifecycle label preset: {value}. Available presets: {available}")
    return resolved


def _future_window_extreme(close: pd.DataFrame, window: int, *, mode: str) -> pd.DataFrame:
    shifted = close.shift(-1)
    reversed_frame = shifted.iloc[::-1]
    if mode == "max":
        rolled = reversed_frame.rolling(window, min_periods=1).max()
    elif mode == "min":
        rolled = reversed_frame.rolling(window, min_periods=1).min()
    else:
        raise ValueError(f"Unsupported future extreme mode: {mode}")
    return rolled.iloc[::-1]


def build_future_path_metrics(prepared: PreparedPolicyInputs) -> FuturePathMetrics:
    close = prepared.close
    benchmark_close = prepared.benchmark_close
    frames: dict[str, pd.DataFrame] = {}
    benchmark_returns: dict[str, pd.Series] = {}
    for horizon in LABEL_HORIZONS:
        frames[f"fwd_excess_{horizon}d"] = build_ml_target(close, benchmark_close, horizon, execution_mode="close")
        future_max = _future_window_extreme(close, horizon, mode="max")
        future_min = _future_window_extreme(close, horizon, mode="min")
        frames[f"future_max_up_{horizon}d"] = future_max.div(close).sub(1.0)
        frames[f"future_min_down_{horizon}d"] = future_min.div(close).sub(1.0)
        benchmark_returns[f"benchmark_return_{horizon}d"] = benchmark_close.shift(-horizon).div(benchmark_close).sub(1.0)
    return FuturePathMetrics(frames=frames, benchmark_returns=benchmark_returns)


def _row_metric(metrics: FuturePathMetrics, name: str, date: pd.Timestamp, stocks: pd.Index) -> pd.Series:
    frame = metrics.frames[name]
    return frame.loc[date].reindex(stocks).astype(float)


def _row_benchmark_metric(metrics: FuturePathMetrics, name: str, date: pd.Timestamp, stocks: pd.Index) -> pd.Series:
    series = metrics.benchmark_returns.get(name)
    value = float(series.loc[date]) if series is not None and date in series.index else 0.0
    if not np.isfinite(value):
        value = 0.0
    return pd.Series(value, index=stocks, dtype=float)


def build_action_labels_for_date(
    *,
    date: pd.Timestamp | str,
    state_frame: pd.DataFrame,
    future_metrics: FuturePathMetrics,
    label_config: str | LifecycleLabelConfig | None = None,
) -> pd.DataFrame:
    config = resolve_label_config(label_config)
    signal_dt = pd.Timestamp(date).normalize()
    working = state_frame.copy()
    stocks = pd.Index(working["stock"].astype(str))

    def _numeric(column: str, default: float = 0.0) -> np.ndarray:
        if column not in working.columns:
            return np.full(len(working), float(default), dtype=float)
        return (
            working[column]
            .astype(float)
            .replace([np.inf, -np.inf], np.nan)
            .fillna(float(default))
            .to_numpy(dtype=float)
        )

    fwd1 = _row_metric(future_metrics, "fwd_excess_1d", signal_dt, stocks)
    fwd3 = _row_metric(future_metrics, "fwd_excess_3d", signal_dt, stocks)
    fwd5 = _row_metric(future_metrics, "fwd_excess_5d", signal_dt, stocks)
    fwd10 = _row_metric(future_metrics, "fwd_excess_10d", signal_dt, stocks)
    fwd20 = _row_metric(future_metrics, "fwd_excess_20d", signal_dt, stocks)
    benchmark_fwd1 = _row_benchmark_metric(future_metrics, "benchmark_return_1d", signal_dt, stocks)
    benchmark_fwd3 = _row_benchmark_metric(future_metrics, "benchmark_return_3d", signal_dt, stocks)
    benchmark_fwd5 = _row_benchmark_metric(future_metrics, "benchmark_return_5d", signal_dt, stocks)
    benchmark_fwd10 = _row_benchmark_metric(future_metrics, "benchmark_return_10d", signal_dt, stocks)
    benchmark_fwd20 = _row_benchmark_metric(future_metrics, "benchmark_return_20d", signal_dt, stocks)
    max_up5 = _row_metric(future_metrics, "future_max_up_5d", signal_dt, stocks)
    max_up10 = _row_metric(future_metrics, "future_max_up_10d", signal_dt, stocks)
    max_up20 = _row_metric(future_metrics, "future_max_up_20d", signal_dt, stocks)
    min_down5 = _row_metric(future_metrics, "future_min_down_5d", signal_dt, stocks)
    min_down10 = _row_metric(future_metrics, "future_min_down_10d", signal_dt, stocks)
    min_down20 = _row_metric(future_metrics, "future_min_down_20d", signal_dt, stocks)
    metric_columns = (
        "score_delta_5d",
        "score_delta_accel",
        "score_blend",
        "ret_5d",
        "ret_accel_5_20",
        "vol_20d",
        "volatility_expansion",
        "score_rank_pct",
        "alpha_prior_score_z",
        "alpha_prior_rank_pct",
        "alpha_prior_target_weight",
        "alpha_prior_selected",
        "alpha_prior_score_delta_1d",
        "alpha_prior_weight_delta_1d",
        "distance_to_20d_high",
        "distance_to_60d_high",
        "drawdown_from_peak",
        "current_weight",
        "unrealized_pnl",
        "holding_flag",
        "hold_days",
        "in_pool",
        "days_since_last_buy",
        "days_since_last_sell",
        "days_since_last_reduce",
        "days_since_last_exit",
        "signal_decay_speed",
        "price_from_local_peak",
        "pnl_from_entry",
        "reentry_cooldown",
        "portfolio_cash_weight",
        "portfolio_recent_turnover_5d",
        "portfolio_turnover_pressure",
        "portfolio_cash_deficit",
        "portfolio_drawdown_20d",
        "market_downside_pressure",
        "portfolio_cash_pressure",
        "recent_reversal_count_20d",
        "recent_reversal_rate_20d",
        "recent_reduce_count_10d",
        "recent_exit_count_10d",
        "reduce_reversal_pressure",
        "exit_reentry_pressure",
        "cash_regime_pressure",
        "hold_continuity_pressure",
        "benchmark_trend_gap",
        "benchmark_vol_ratio",
        "recent_buy_flag",
        "recent_sell_flag",
    )
    for column in metric_columns:
        if column in working.columns:
            working[column] = working[column].astype(float).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    fwd1 = fwd1.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    fwd3 = fwd3.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    fwd5 = fwd5.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    fwd10 = fwd10.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    fwd20 = fwd20.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    benchmark_fwd1 = benchmark_fwd1.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    benchmark_fwd3 = benchmark_fwd3.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    benchmark_fwd5 = benchmark_fwd5.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    benchmark_fwd10 = benchmark_fwd10.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    benchmark_fwd20 = benchmark_fwd20.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    max_up5 = max_up5.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    max_up10 = max_up10.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    max_up20 = max_up20.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    min_down5 = min_down5.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    min_down10 = min_down10.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    min_down20 = min_down20.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    benchmark_trend_gap = _numeric("benchmark_trend_gap")
    benchmark_vol_ratio = _numeric("benchmark_vol_ratio")
    portfolio_cash_weight = _numeric("portfolio_cash_weight", default=1.0)
    portfolio_recent_turnover_5d = _numeric("portfolio_recent_turnover_5d")
    portfolio_turnover_pressure = _numeric("portfolio_turnover_pressure")
    portfolio_cash_deficit = _numeric("portfolio_cash_deficit")
    portfolio_drawdown_20d = _numeric("portfolio_drawdown_20d")
    reentry_cooldown = _numeric("reentry_cooldown")
    recent_buy_flag = _numeric("recent_buy_flag")
    recent_sell_flag = _numeric("recent_sell_flag")
    days_since_last_buy = _numeric("days_since_last_buy", default=99.0)
    days_since_last_reduce = _numeric("days_since_last_reduce", default=99.0)
    days_since_last_exit = _numeric("days_since_last_exit", default=99.0)
    signal_decay_speed = _numeric("signal_decay_speed")
    price_from_local_peak = _numeric("price_from_local_peak")
    pnl_from_entry = _numeric("pnl_from_entry")
    market_downside_pressure = _numeric("market_downside_pressure")
    portfolio_cash_pressure = _numeric("portfolio_cash_pressure")
    recent_reversal_count_20d = _numeric("recent_reversal_count_20d")
    recent_reversal_rate_20d = _numeric("recent_reversal_rate_20d")
    recent_reduce_count_10d = _numeric("recent_reduce_count_10d")
    recent_exit_count_10d = _numeric("recent_exit_count_10d")
    reduce_reversal_pressure = _numeric("reduce_reversal_pressure")
    exit_reentry_pressure = _numeric("exit_reentry_pressure")
    cash_regime_pressure = _numeric("cash_regime_pressure")
    hold_continuity_pressure = _numeric("hold_continuity_pressure")
    alpha_score_z = _numeric("alpha_prior_score_z")
    alpha_rank_pct = _numeric("alpha_prior_rank_pct")
    alpha_target_weight = _numeric("alpha_prior_target_weight")
    alpha_selected = _numeric("alpha_prior_selected")
    alpha_score_delta_1d = _numeric("alpha_prior_score_delta_1d")
    alpha_weight_delta_1d = _numeric("alpha_prior_weight_delta_1d")
    alpha_weight_scaled = np.clip(alpha_target_weight / 0.12, 0.0, 1.0)
    alpha_support = np.clip(
        0.42 * np.clip(alpha_score_z / 2.0, -1.0, 1.0)
        + 0.30 * ((np.clip(alpha_rank_pct, 0.0, 1.0) - 0.5) * 2.0)
        + 0.18 * alpha_weight_scaled
        + 0.10 * np.clip(alpha_score_delta_1d + alpha_weight_delta_1d * 4.0, -0.5, 0.5),
        -1.0,
        1.0,
    )
    recent_reduce_cooldown = np.clip((4.0 - days_since_last_reduce) / 4.0, 0.0, 1.0)
    recent_exit_cooldown = np.clip((5.0 - days_since_last_exit) / 5.0, 0.0, 1.0)

    short_edge = 0.55 * fwd1 + 0.45 * fwd3
    mid_edge = 0.35 * fwd3 + 0.65 * fwd5
    long_edge = 0.20 * fwd5 + 0.35 * fwd10 + 0.45 * fwd20
    edge = 0.15 * short_edge + 0.35 * mid_edge + 0.50 * long_edge
    opportunity = 0.25 * max_up5 + 0.35 * max_up10 + 0.40 * max_up20
    downside = 0.25 * min_down5.clip(upper=0).abs() + 0.35 * min_down10.clip(upper=0).abs() + 0.40 * min_down20.clip(upper=0).abs()
    multi_horizon_forward_value = np.clip(
        0.16 * np.clip(fwd1.to_numpy(dtype=float) / 0.025, 0.0, 1.0)
        + 0.20 * np.clip(fwd3.to_numpy(dtype=float) / 0.040, 0.0, 1.0)
        + 0.24 * np.clip(fwd5.to_numpy(dtype=float) / 0.055, 0.0, 1.0)
        + 0.22 * np.clip(fwd10.to_numpy(dtype=float) / 0.090, 0.0, 1.0)
        + 0.18 * np.clip(fwd20.to_numpy(dtype=float) / 0.140, 0.0, 1.0),
        0.0,
        1.0,
    )
    multi_horizon_forward_risk = np.clip(
        0.18 * np.clip(-fwd1.to_numpy(dtype=float) / 0.025, 0.0, 1.0)
        + 0.18 * np.clip(-fwd3.to_numpy(dtype=float) / 0.040, 0.0, 1.0)
        + 0.20 * np.clip(-fwd5.to_numpy(dtype=float) / 0.055, 0.0, 1.0)
        + 0.20 * np.clip(-fwd10.to_numpy(dtype=float) / 0.090, 0.0, 1.0)
        + 0.16 * np.clip(-fwd20.to_numpy(dtype=float) / 0.140, 0.0, 1.0)
        + 0.08 * np.clip(downside.to_numpy(dtype=float) / 0.10, 0.0, 1.0),
        0.0,
        1.0,
    )
    multi_horizon_path_value = np.clip(
        0.48 * multi_horizon_forward_value
        + 0.22 * np.clip(opportunity.to_numpy(dtype=float) / 0.14, 0.0, 1.0)
        + 0.18 * np.clip(edge.to_numpy(dtype=float) / 0.070, 0.0, 1.0)
        + 0.12 * np.clip(alpha_support, 0.0, 1.0)
        - 0.34 * multi_horizon_forward_risk,
        0.0,
        1.0,
    )
    momentum = (
        0.28 * working["score_delta_5d"].fillna(0.0).to_numpy(dtype=float)
        + 0.12 * working["score_delta_accel"].fillna(0.0).to_numpy(dtype=float)
        + 0.15 * working["score_blend"].fillna(0.0).to_numpy(dtype=float)
        + 0.12 * working["ret_5d"].fillna(0.0).to_numpy(dtype=float)
        + 0.08 * working["ret_accel_5_20"].fillna(0.0).to_numpy(dtype=float)
        - 0.08 * working["vol_20d"].fillna(0.0).to_numpy(dtype=float)
        - 0.08 * working["volatility_expansion"].fillna(0.0).to_numpy(dtype=float)
        + 0.10 * working["score_rank_pct"].fillna(0.0).to_numpy(dtype=float)
        + 0.07 * working["distance_to_20d_high"].fillna(0.0).to_numpy(dtype=float)
        + 0.04 * working["distance_to_60d_high"].fillna(0.0).to_numpy(dtype=float)
        + 0.10 * alpha_support
        + 0.03 * alpha_selected
    )
    persistence = (
        0.15 * fwd3.to_numpy(dtype=float)
        + 0.25 * fwd5.to_numpy(dtype=float)
        + 0.30 * fwd10.to_numpy(dtype=float)
        + 0.30 * fwd20.to_numpy(dtype=float)
    )
    frontload_gap = max_up5.to_numpy(dtype=float) - max_up20.to_numpy(dtype=float)
    action_signal = (
        edge.to_numpy(dtype=float)
        + 0.30 * opportunity.to_numpy(dtype=float)
        + 0.22 * momentum
        + 0.18 * persistence
        + 0.08 * alpha_support
        - 0.90 * downside.to_numpy(dtype=float)
        - 0.12 * frontload_gap
    )
    urgency = downside.to_numpy(dtype=float) + np.clip(-short_edge.to_numpy(dtype=float), 0.0, None) - 0.55 * persistence
    urgency = (
        urgency
        + 0.34 * signal_decay_speed
        + 0.24 * market_downside_pressure
        + 0.16 * portfolio_cash_pressure
        + 0.08 * recent_reversal_rate_20d
        - 0.10 * np.clip(opportunity.to_numpy(dtype=float), 0.0, None)
    )
    defensive_market = (
        0.55 * np.clip(-benchmark_trend_gap, 0.0, None)
        + 0.25 * np.clip(benchmark_vol_ratio, 0.0, None)
        + 0.20 * np.clip(-portfolio_drawdown_20d - 0.02, 0.0, None)
    )
    turnover_drag = np.clip(portfolio_recent_turnover_5d, 0.0, None) * float(config.turnover_sensitivity)
    cash_pressure = (
        defensive_market * float(config.cash_regime_sensitivity)
        + portfolio_cash_deficit * 0.45
        + turnover_drag
        + float(config.defensive_cash_bias)
    )
    market_forward_downside = np.clip(
        np.clip(-benchmark_fwd1.to_numpy(dtype=float), 0.0, None) / 0.012 * 0.42
        + np.clip(-benchmark_fwd3.to_numpy(dtype=float), 0.0, None) / 0.028 * 0.58,
        0.0,
        1.0,
    )

    duration_bucket: list[str] = []
    duration_days: list[int] = []
    strong_hold_profile = float(config.hold_support_bonus) >= 0.18
    for idx in range(len(working)):
        long_edge_value = float(long_edge.iloc[idx])
        mid_edge_value = float(mid_edge.iloc[idx])
        short_edge_value = float(short_edge.iloc[idx])
        opportunity_value = float(opportunity.iloc[idx])
        downside_value = float(downside.iloc[idx])
        if (
            long_edge_value > (0.028 if strong_hold_profile else 0.035)
            and opportunity_value > downside_value * (1.10 if strong_hold_profile else 1.25)
            and (mid_edge_value > 0.010 or hold_continuity_pressure[idx] > 0.30)
        ):
            duration_bucket.append("extended")
            duration_days.append(18 if strong_hold_profile else 15)
        elif (
            mid_edge_value > (0.012 if strong_hold_profile else 0.018)
            and opportunity_value > downside_value * (1.02 if strong_hold_profile else 1.10)
        ):
            duration_bucket.append("swing")
            duration_days.append(10 if strong_hold_profile else 8)
        elif short_edge_value > -0.004 or opportunity_value > downside_value * (0.95 if strong_hold_profile else 1.0):
            duration_bucket.append("short")
            duration_days.append(4 if strong_hold_profile else 3)
        else:
            duration_bucket.append("avoid")
            duration_days.append(0)

    working["teacher_edge"] = edge.to_numpy(dtype=float)
    working["teacher_opportunity"] = opportunity.to_numpy(dtype=float)
    working["teacher_downside"] = downside.to_numpy(dtype=float)
    working["teacher_signal"] = action_signal
    working["teacher_urgency"] = urgency
    working["entry_quality"] = (
        0.52 * working["teacher_signal"].to_numpy(dtype=float)
        + 0.28 * opportunity.to_numpy(dtype=float)
        + 0.12 * persistence
        - 0.72 * downside.to_numpy(dtype=float)
        + 0.05 * np.clip(working["score_rank_pct"].fillna(0.0).to_numpy(dtype=float) - 0.75, 0.0, None)
        + 0.10 * np.clip(alpha_support, 0.0, None)
        + 0.04 * alpha_selected
        + 0.04 * alpha_weight_scaled
        - 0.22 * reentry_cooldown
        - 0.18 * cash_pressure
        - 0.22 * market_downside_pressure
        - 0.16 * portfolio_cash_pressure
        - 0.16 * cash_regime_pressure
        - 0.06 * recent_reversal_rate_20d
    )
    working["hold_quality"] = (
        0.48 * persistence
        + 0.22 * opportunity.to_numpy(dtype=float)
        + 0.14 * working["teacher_signal"].to_numpy(dtype=float)
        - 0.55 * downside.to_numpy(dtype=float)
        - 0.10 * np.clip(-working["drawdown_from_peak"].fillna(0.0).to_numpy(dtype=float) - 0.06, 0.0, None)
        + float(config.hold_support_bonus) * np.clip(1.0 - recent_buy_flag * 0.25, 0.0, None)
        + 0.08 * np.clip(np.asarray(duration_days, dtype=float) - working["hold_days"].fillna(0.0).to_numpy(dtype=float), 0.0, None) / 10.0
        - 0.18 * turnover_drag
        - 0.10 * defensive_market
        - 0.22 * signal_decay_speed
        - 0.14 * market_downside_pressure
        - 0.10 * portfolio_cash_pressure
        + 0.10 * hold_continuity_pressure
        + 0.06 * reduce_reversal_pressure
        + 0.08 * np.clip(pnl_from_entry, 0.0, None)
        + 0.07 * np.clip(alpha_support, 0.0, None)
        + 0.04 * alpha_selected
    )
    working["add_quality"] = (
        0.40 * working["hold_quality"].to_numpy(dtype=float)
        + 0.40 * np.clip(working["entry_quality"].to_numpy(dtype=float), 0.0, None)
        + 0.10 * np.clip(working["score_delta_accel"].fillna(0.0).to_numpy(dtype=float), 0.0, None)
        - 0.12 * np.clip(working["current_weight"].fillna(0.0).to_numpy(dtype=float) - 0.12, 0.0, None)
        - 0.08 * cash_pressure
        - 0.10 * market_downside_pressure
        - 0.08 * recent_reversal_rate_20d
        - 0.08 * cash_regime_pressure
        + 0.06 * np.clip(alpha_support, 0.0, None)
        + 0.04 * alpha_weight_scaled
    )
    held_mask = working["holding_flag"].fillna(0.0).to_numpy(dtype=float) > 0.5
    laggard_rank = np.zeros(len(working), dtype=float)
    held_downside_rank = np.zeros(len(working), dtype=float)
    held_decay_rank = np.zeros(len(working), dtype=float)
    held_profit_rank = np.zeros(len(working), dtype=float)
    held_keep_rank = np.zeros(len(working), dtype=float)
    if bool(np.any(held_mask)):
        held_index = working.index[held_mask]
        held_positions = np.flatnonzero(held_mask)
        held_keep_basis = pd.Series(
            0.52 * long_edge.to_numpy(dtype=float)
            + 0.18 * opportunity.to_numpy(dtype=float)
            - 0.62 * downside.to_numpy(dtype=float)
            + 0.12 * alpha_support
            + 0.10 * hold_continuity_pressure
            - 0.08 * signal_decay_speed,
            index=working.index,
            dtype=float,
        )
        held_keep_rank_series = held_keep_basis.loc[held_index].rank(method="average", pct=True)
        held_keep_rank[held_positions] = held_keep_rank_series.to_numpy(dtype=float)
        laggard_rank[held_positions] = 1.0 - held_keep_rank_series.to_numpy(dtype=float)
        held_downside_rank_series = pd.Series(downside.to_numpy(dtype=float), index=working.index).loc[held_index].rank(method="average", pct=True)
        held_downside_rank[held_positions] = held_downside_rank_series.to_numpy(dtype=float)
        held_decay_rank_series = pd.Series(signal_decay_speed, index=working.index).loc[held_index].rank(method="average", pct=True)
        held_decay_rank[held_positions] = held_decay_rank_series.to_numpy(dtype=float)
        held_profit_rank_series = pd.Series(
            working["unrealized_pnl"].fillna(0.0).to_numpy(dtype=float),
            index=working.index,
        ).loc[held_index].rank(method="average", pct=True)
        held_profit_rank[held_positions] = held_profit_rank_series.to_numpy(dtype=float)
    profit_protected_strength = np.clip(held_profit_rank * np.clip(held_keep_rank - 0.45, 0.0, 1.0), 0.0, 1.0)
    sell_attribution_core = np.where(
        held_mask,
        np.clip(
            0.34 * laggard_rank
            + 0.20 * held_downside_rank
            + 0.14 * held_decay_rank
            + 0.10 * held_profit_rank * np.clip(1.0 - held_keep_rank, 0.0, 1.0)
            + 0.10 * market_forward_downside
            + 0.08 * market_downside_pressure
            + 0.08 * portfolio_cash_pressure
            + 0.06 * cash_regime_pressure
            - 0.16 * hold_continuity_pressure
            - 0.10 * np.clip(alpha_support, 0.0, None)
            - 0.18 * profit_protected_strength,
            0.0,
            1.0,
        ),
        0.0,
    )
    working["reduce_quality"] = (
        0.42 * downside.to_numpy(dtype=float)
        + 0.24 * np.clip(-working["drawdown_from_peak"].fillna(0.0).to_numpy(dtype=float), 0.0, None)
        + 0.18 * np.clip(-working["teacher_signal"].to_numpy(dtype=float), 0.0, None)
        + 0.08 * np.clip(-working["score_delta_5d"].fillna(0.0).to_numpy(dtype=float), 0.0, None)
        + 0.07 * np.clip(working["volatility_expansion"].fillna(0.0).to_numpy(dtype=float), 0.0, None)
        + 0.16 * signal_decay_speed
        + 0.14 * market_downside_pressure
        + 0.10 * portfolio_cash_pressure
        + 0.06 * recent_reversal_rate_20d
        - float(config.profit_take_penalty) * np.clip(working["unrealized_pnl"].fillna(0.0).to_numpy(dtype=float) - 0.06, 0.0, None) * np.clip(persistence, 0.0, None)
        - 0.15 * np.clip(working["hold_quality"].to_numpy(dtype=float), 0.0, None)
        - 0.08 * recent_reduce_cooldown
        - 0.12 * hold_continuity_pressure
        - 0.14 * reduce_reversal_pressure
        - 0.08 * np.clip(alpha_support, 0.0, None)
        - 0.05 * alpha_selected
        + 0.24 * sell_attribution_core
        + 0.10 * market_forward_downside
        - 0.14 * held_keep_rank
    )
    working["sell_attribution_score"] = np.where(
        held_mask,
        np.clip(
            0.34 * np.clip(working["reduce_quality"].to_numpy(dtype=float), 0.0, None)
            + 0.20 * sell_attribution_core
            + 0.12 * market_forward_downside
            + 0.10 * market_downside_pressure
            + 0.08 * portfolio_cash_pressure
            + 0.08 * cash_regime_pressure
            - 0.22 * np.clip(working["hold_quality"].to_numpy(dtype=float), 0.0, None)
            - 0.10 * np.clip(alpha_support, 0.0, None)
            - 0.10 * profit_protected_strength,
            0.0,
            1.0,
        ),
        0.0,
    )
    sell_rank_score = np.zeros(len(working), dtype=float)
    if bool(np.any(held_mask)):
        held_index = working.index[held_mask]
        held_positions = np.flatnonzero(held_mask)
        if len(held_index) >= 2:
            rank_series = working.loc[held_index, "sell_attribution_score"].rank(method="average", pct=True)
            sell_rank_score[held_positions] = rank_series.to_numpy(dtype=float)
        else:
            sell_rank_score[held_positions] = working.loc[held_index, "sell_attribution_score"].to_numpy(dtype=float)
    working["sell_rank_score"] = sell_rank_score
    working["reentry_readiness"] = np.clip(
        0.65 * working["entry_quality"].to_numpy(dtype=float)
        - 0.35 * working["teacher_urgency"].to_numpy(dtype=float),
        0.0,
        None,
    )
    working["reentry_readiness"] = np.clip(
        working["reentry_readiness"].to_numpy(dtype=float)
        - 0.12 * recent_exit_cooldown
        - 0.10 * market_downside_pressure
        - 0.08 * recent_reversal_rate_20d,
        - 0.12 * exit_reentry_pressure,
        0.0,
        None,
    )
    working["planned_holding_bucket"] = duration_bucket
    working["planned_holding_days"] = duration_days

    labels: list[str] = []
    delta_hints: list[float] = []
    priorities: list[float] = []
    reduce_fraction_targets: list[float] = []
    exit_hazard_targets: list[float] = []
    for row in working.itertuples(index=False):
        held = float(getattr(row, "holding_flag", 0.0) or 0.0) > 0.5
        in_pool = float(getattr(row, "in_pool", 0.0) or 0.0) > 0.5
        signal = float(getattr(row, "teacher_signal", 0.0) or 0.0)
        current_weight = float(getattr(row, "current_weight", 0.0) or 0.0)
        pnl = float(getattr(row, "unrealized_pnl", 0.0) or 0.0)
        drawdown = float(getattr(row, "drawdown_from_peak", 0.0) or 0.0)
        hold_days = float(getattr(row, "hold_days", 0.0) or 0.0)
        urgency_value = float(getattr(row, "teacher_urgency", 0.0) or 0.0)
        entry_quality = float(getattr(row, "entry_quality", 0.0) or 0.0)
        hold_quality = float(getattr(row, "hold_quality", 0.0) or 0.0)
        add_quality = float(getattr(row, "add_quality", 0.0) or 0.0)
        reduce_quality = float(getattr(row, "reduce_quality", 0.0) or 0.0)
        duration_name = str(getattr(row, "planned_holding_bucket", "avoid") or "avoid").strip().lower()
        duration_target = float(getattr(row, "planned_holding_days", 0.0) or 0.0)
        raw_days_since_buy = getattr(row, "days_since_last_buy", 99.0)
        raw_days_since_sell = getattr(row, "days_since_last_sell", 99.0)
        days_since_buy = float(raw_days_since_buy) if raw_days_since_buy is not None else 99.0
        days_since_sell = float(raw_days_since_sell) if raw_days_since_sell is not None else 99.0
        reentry_block = float(getattr(row, "reentry_cooldown", 0.0) or 0.0)
        signal_decay_value = float(getattr(row, "signal_decay_speed", 0.0) or 0.0)
        market_downside_value = float(getattr(row, "market_downside_pressure", 0.0) or 0.0)
        portfolio_cash_pressure_value = float(getattr(row, "portfolio_cash_pressure", 0.0) or 0.0)
        reversal_rate_value = float(getattr(row, "recent_reversal_rate_20d", 0.0) or 0.0)
        reduce_reversal_value = float(getattr(row, "reduce_reversal_pressure", 0.0) or 0.0)
        exit_reentry_value = float(getattr(row, "exit_reentry_pressure", 0.0) or 0.0)
        cash_regime_value = float(getattr(row, "cash_regime_pressure", 0.0) or 0.0)
        hold_continuity_value = float(getattr(row, "hold_continuity_pressure", 0.0) or 0.0)
        sell_attribution_score = float(getattr(row, "sell_attribution_score", 0.0) or 0.0)
        cash_pressure_value = float(getattr(row, "portfolio_cash_deficit", 0.0) or 0.0) + float(getattr(row, "portfolio_turnover_pressure", 0.0) or 0.0) * 0.12
        catastrophic_exit = urgency_value >= float(config.catastrophic_exit_threshold) or drawdown <= -0.14
        force_hold_window = hold_days < float(config.min_hold_days) and not catastrophic_exit
        long_horizon = duration_name in {"swing", "extended"}
        recent_add_window = held and days_since_buy <= max(float(config.min_hold_days), 3.0)
        unfinished_lifecycle = long_horizon and hold_days < max(duration_target - 1.0, float(config.min_hold_days))
        risk_off_reduce = market_downside_value > 0.18 or portfolio_cash_pressure_value > 0.22 or cash_regime_value > 0.24
        wrong_side_profit_take = (
            pnl > 0.05
            and signal > 0.015
            and hold_quality > reduce_quality - 0.02
            and drawdown > -0.05
            and market_downside_value < 0.12
        )
        continuation_hold = (
            hold_quality >= reduce_quality - 0.06
            and hold_continuity_value > 0.22
            and signal > -0.015
            and drawdown > -0.08
            and sell_attribution_score < 0.58
        )

        if held:
            if (not in_pool) or catastrophic_exit:
                action = "exit"
                delta_hint = -1.0
            elif recent_add_window and urgency_value < float(config.exit_urgency_threshold) * 1.05 and hold_quality > -0.05 and not risk_off_reduce:
                action = "hold"
                delta_hint = min(0.04, max(0.0, hold_quality) * 0.28)
            elif continuation_hold and (force_hold_window or unfinished_lifecycle):
                action = "hold"
                delta_hint = min(
                    0.06,
                    max(0.0, hold_quality) * 0.34
                    + max(duration_target - hold_days, 0.0) / 180.0
                    + hold_continuity_value * 0.03,
                )
            elif force_hold_window and hold_quality > -0.03 and not risk_off_reduce:
                action = "hold"
                delta_hint = min(0.03, max(0.0, hold_quality) * 0.25)
            elif unfinished_lifecycle and hold_quality > reduce_quality - 0.04 and urgency_value < float(config.exit_urgency_threshold) * 1.02 and not risk_off_reduce:
                action = "hold"
                delta_hint = min(0.05, max(0.0, hold_quality) * 0.32 + max(duration_target - hold_days, 0.0) / 220.0)
            elif wrong_side_profit_take or (
                reduce_reversal_value > 0.20
                and hold_quality > reduce_quality - 0.03
                and signal > -0.01
                and drawdown > -0.08
                and market_downside_value < 0.18
                and sell_attribution_score < 0.42
            ):
                action = "hold"
                delta_hint = min(0.04, max(0.0, hold_quality) * 0.26 + hold_continuity_value * 0.02)
            elif (
                urgency_value >= float(config.exit_urgency_threshold)
                and (hold_days >= float(config.min_hold_days) or pnl < 0.0)
                and (signal_decay_value > 0.09 or drawdown < -0.10 or market_downside_value > 0.24)
                and not continuation_hold
            ):
                action = "exit"
                delta_hint = -1.0
            elif (
                reduce_quality >= float(config.reduce_quality_threshold)
                or (signal < -0.01 and drawdown < -0.06)
                or (risk_off_reduce and hold_days >= float(config.min_hold_days) and sell_attribution_score > 0.20)
                or (reversal_rate_value > 0.22 and signal_decay_value > 0.04)
            ) and not wrong_side_profit_take and not (continuation_hold and reduce_reversal_value > 0.18 and sell_attribution_score < 0.35):
                if signal_decay_value > 0.09 or drawdown < -0.11 or market_downside_value > 0.24:
                    action = "exit" if hold_days >= float(config.min_hold_days) else "reduce"
                    delta_hint = -1.0 if action == "exit" else -min(0.72, 0.14 + reduce_quality * 1.05 + sell_attribution_score * 0.18 + max(market_downside_value, 0.0))
                else:
                    action = "reduce"
                    delta_hint = -min(0.76, 0.14 + reduce_quality * 1.08 + sell_attribution_score * 0.22 + max(-signal, 0.0) + market_downside_value * 0.20)
            elif add_quality >= float(config.add_quality_threshold) and long_horizon and hold_days >= 1 and market_downside_value < 0.18 and portfolio_cash_pressure_value < 0.16 and cash_regime_value < 0.22:
                action = "add"
                delta_hint = min(0.18, 0.02 + add_quality * 0.85 + duration_target / 120.0)
            else:
                action = "hold"
                delta_hint = min(
                    0.09,
                    max(0.0, hold_quality) * 0.45
                    + max(duration_target - 3.0, 0.0) / 240.0
                    + hold_continuity_value * 0.03
                    - exit_reentry_value * 0.01,
                )
        else:
            open_gate = (
                float(config.open_entry_threshold)
                + market_downside_value * 0.10
                + portfolio_cash_pressure_value * 0.12
                + reversal_rate_value * 0.06
                + cash_regime_value * 0.06
            )
            if (
                in_pool
                and duration_name != "avoid"
                and signal >= float(config.open_signal_threshold)
                and entry_quality >= open_gate
                and not (days_since_sell <= float(config.reentry_cooldown_days) and entry_quality < open_gate + 0.035)
                and not (reentry_block > 0.25 and (cash_pressure_value > 0.05 or reversal_rate_value > 0.20))
                and not (market_downside_value > 0.22 and entry_quality < open_gate + 0.04)
                and not (cash_regime_value > 0.24 and entry_quality < open_gate + 0.03)
            ):
                action = "open"
                delta_hint = min(0.18, 0.03 + entry_quality * 0.72 + duration_target / 140.0)
            else:
                action = "skip"
                delta_hint = 0.0

        if action == "open":
            priority = max(0.0, entry_quality) + max(duration_target - 3.0, 0.0) / 40.0
        elif action == "add":
            priority = max(0.0, add_quality) + current_weight * 0.35 + duration_target / 80.0
        elif action == "hold":
            priority = max(0.0, hold_quality) + current_weight * 0.25 + duration_target / 120.0
        elif action == "reduce":
            priority = max(0.0, reduce_quality)
        else:
            priority = 0.0
        if action == "exit":
            priority = 0.0

        sell_pressure = float(
            np.clip(
                0.38 * max(reduce_quality, 0.0)
                + 0.24 * max(urgency_value, 0.0)
                + 0.10 * max(-signal, 0.0)
                + 0.10 * max(signal_decay_value, 0.0)
                + 0.08 * max(market_downside_value, 0.0)
                + 0.10 * max(-drawdown - 0.04, 0.0)
                + 0.12 * max(sell_attribution_score, 0.0)
                - 0.12 * max(hold_quality, 0.0)
                - 0.05 * max(hold_continuity_value, 0.0),
                0.0,
                1.0,
            )
        )
        exit_pressure = float(
            np.clip(
                0.52 * max(urgency_value, 0.0)
                + 0.14 * max(signal_decay_value, 0.0)
                + 0.14 * max(market_downside_value, 0.0)
                + 0.10 * max(-drawdown - 0.05, 0.0)
                + 0.10 * max(-signal, 0.0)
                + 0.10 * max(sell_attribution_score, 0.0)
                - 0.12 * max(hold_quality, 0.0)
                - 0.06 * max(hold_continuity_value, 0.0),
                0.0,
                1.0,
            )
        )
        delta_fraction = 0.0
        if held:
            denominator = max(current_weight, 0.04)
            delta_fraction = float(np.clip(max(-float(delta_hint), 0.0) / denominator, 0.0, 1.0))
        if action == "reduce":
            reduce_fraction_target = float(np.clip(max(delta_fraction, 0.12 + sell_pressure * 0.72 + sell_attribution_score * 0.22), 0.12, 0.96))
            exit_hazard_target = float(
                np.clip(
                    exit_pressure * 0.55
                    + sell_attribution_score * 0.12
                    + (0.08 if signal_decay_value > 0.08 or market_downside_value > 0.20 else 0.0),
                    0.0,
                    0.82,
                )
            )
        elif action == "exit":
            reduce_fraction_target = 1.0 if held else 0.0
            exit_hazard_target = float(np.clip(max(0.82, exit_pressure), 0.0, 1.0)) if held else 0.0
        elif held and action in {"hold", "add"}:
            reduce_fraction_target = float(np.clip(sell_pressure * (0.10 if action == "add" else 0.16), 0.0, 0.28))
            exit_hazard_target = float(np.clip(exit_pressure * (0.08 if action == "add" else 0.14), 0.0, 0.24))
        else:
            reduce_fraction_target = 0.0
            exit_hazard_target = 0.0
        labels.append(action)
        delta_hints.append(float(delta_hint))
        priorities.append(float(priority))
        reduce_fraction_targets.append(float(reduce_fraction_target))
        exit_hazard_targets.append(float(exit_hazard_target))

    working["action_label"] = labels
    working["target_delta_hint"] = delta_hints
    working["teacher_priority"] = priorities
    working["exit_urgency"] = working["teacher_urgency"].clip(lower=0.0)
    working["reduce_fraction_target"] = reduce_fraction_targets
    working["exit_hazard_target"] = exit_hazard_targets
    action_series = pd.Series(labels, index=working.index, dtype=str)
    held_float = pd.Series(held_mask.astype(float), index=working.index, dtype=float)
    sell_action_boost = action_series.isin({"reduce", "exit"}).astype(float)
    keep_action_boost = action_series.isin({"hold", "add"}).astype(float)
    working["lifecycle_sell_gate"] = np.where(
        held_mask,
        np.clip(
            0.30 * working["sell_attribution_score"].to_numpy(dtype=float)
            + 0.26 * working["sell_rank_score"].to_numpy(dtype=float)
            + 0.18 * np.clip(working["reduce_quality"].to_numpy(dtype=float), 0.0, None)
            + 0.14 * np.asarray(reduce_fraction_targets, dtype=float)
            + 0.12 * np.asarray(exit_hazard_targets, dtype=float)
            + 0.10 * sell_action_boost.to_numpy(dtype=float)
            - 0.18 * np.clip(working["hold_quality"].to_numpy(dtype=float), 0.0, None)
            - 0.08 * keep_action_boost.to_numpy(dtype=float)
            - 0.08 * np.clip(alpha_support, 0.0, None),
            0.0,
            1.0,
        ),
        0.0,
    )
    stock_forward_return_1d = fwd1.to_numpy(dtype=float) + benchmark_fwd1.to_numpy(dtype=float)
    large_upside_1d_target = np.clip(stock_forward_return_1d / 0.095, 0.0, 1.0)
    forward_edge_5d = fwd5.to_numpy(dtype=float)
    forward_edge_10d = fwd10.to_numpy(dtype=float)
    opportunity_array = opportunity.to_numpy(dtype=float)
    downside_array = downside.to_numpy(dtype=float)
    edge_array = edge.to_numpy(dtype=float)
    alpha_positive = np.clip(alpha_support, 0.0, 1.0)
    cash_defense_value = np.clip(
        0.30 * market_forward_downside
        + 0.18 * np.clip(market_downside_pressure / 0.24, 0.0, 1.0)
        + 0.16 * np.clip(cash_regime_pressure / 0.24, 0.0, 1.0)
        + 0.14 * np.clip(portfolio_cash_pressure / 0.22, 0.0, 1.0)
        + 0.10 * np.clip(recent_reversal_rate_20d / 0.28, 0.0, 1.0)
        + 0.12 * np.clip(downside_array / 0.08, 0.0, 1.0)
        - 0.16 * alpha_positive
        - 0.08 * large_upside_1d_target,
        0.0,
        1.0,
    )
    alpha_opportunity_value = np.clip(
        0.24 * np.clip(forward_edge_5d / 0.055, 0.0, 1.0)
        + 0.20 * np.clip(opportunity_array / 0.12, 0.0, 1.0)
        + 0.18 * np.clip(edge_array / 0.055, 0.0, 1.0)
        + 0.16 * large_upside_1d_target
        + 0.14 * alpha_positive
        + 0.08 * np.clip(alpha_selected, 0.0, 1.0)
        - 0.18 * cash_defense_value
        - 0.12 * np.clip(downside_array / 0.08, 0.0, 1.0),
        0.0,
        1.0,
    )
    hold_continuation_value = np.where(
        held_mask,
        np.clip(
            0.28 * np.clip(forward_edge_5d / 0.055, 0.0, 1.0)
            + 0.16 * np.clip(forward_edge_10d / 0.09, 0.0, 1.0)
            + 0.20 * np.clip(working["hold_quality"].to_numpy(dtype=float), 0.0, 1.0)
            + 0.14 * alpha_opportunity_value
            + 0.10 * np.clip(hold_continuity_pressure, 0.0, 1.0)
            + 0.08 * large_upside_1d_target
            + 0.04 * np.clip(working["add_quality"].to_numpy(dtype=float), 0.0, 1.0)
            - 0.20 * working["sell_attribution_score"].to_numpy(dtype=float)
            - 0.14 * working["lifecycle_sell_gate"].to_numpy(dtype=float)
            - 0.10 * cash_defense_value,
            0.0,
            1.0,
        ),
        0.0,
    )
    sell_release_value = np.where(
        held_mask,
        np.clip(
            0.24 * np.clip(-forward_edge_5d / 0.055, 0.0, 1.0)
            + 0.14 * np.clip(-fwd1.to_numpy(dtype=float) / 0.025, 0.0, 1.0)
            + 0.20 * working["sell_attribution_score"].to_numpy(dtype=float)
            + 0.16 * working["sell_rank_score"].to_numpy(dtype=float)
            + 0.14 * working["lifecycle_sell_gate"].to_numpy(dtype=float)
            + 0.12 * cash_defense_value
            + 0.08 * np.asarray(reduce_fraction_targets, dtype=float)
            + 0.08 * np.asarray(exit_hazard_targets, dtype=float)
            - 0.16 * alpha_opportunity_value
            - 0.10 * np.clip(working["hold_quality"].to_numpy(dtype=float), 0.0, 1.0),
            0.0,
            1.0,
        ),
        0.0,
    )
    deployment_opportunity_cost = np.clip(
        0.28 * alpha_opportunity_value
        + 0.18 * large_upside_1d_target
        + 0.18 * np.clip(working["entry_quality"].to_numpy(dtype=float), 0.0, 1.0)
        + 0.12 * np.clip(working["add_quality"].to_numpy(dtype=float), 0.0, 1.0)
        + 0.12 * np.clip(edge_array / 0.055, 0.0, 1.0)
        + 0.08 * np.clip(alpha_selected, 0.0, 1.0)
        + 0.04 * np.clip(portfolio_cash_weight / 0.45, 0.0, 1.0)
        - 0.18 * cash_defense_value
        - 0.10 * sell_release_value,
        0.0,
        1.0,
    )
    deploy_action_value = np.clip(
        0.52 * deployment_opportunity_cost
        + 0.32 * alpha_opportunity_value
        + 0.16 * large_upside_1d_target
        - 0.22 * cash_defense_value
        - 0.12 * sell_release_value,
        0.0,
        1.0,
    )
    keep_action_value = np.clip(
        0.58 * hold_continuation_value
        + 0.24 * alpha_opportunity_value
        + 0.10 * large_upside_1d_target
        + 0.08 * np.clip(hold_continuity_pressure, 0.0, 1.0)
        - 0.24 * sell_release_value
        - 0.10 * cash_defense_value,
        0.0,
        1.0,
    )
    sell_action_value = np.clip(
        0.58 * sell_release_value
        + 0.26 * cash_defense_value
        + 0.10 * working["lifecycle_sell_gate"].to_numpy(dtype=float)
        + 0.06 * working["sell_rank_score"].to_numpy(dtype=float)
        - 0.18 * hold_continuation_value,
        0.0,
        1.0,
    )
    cash_action_value = np.clip(
        0.62 * cash_defense_value
        + 0.20 * np.clip(1.0 - alpha_opportunity_value, 0.0, 1.0)
        + 0.18 * market_forward_downside
        - 0.14 * deployment_opportunity_cost,
        0.0,
        1.0,
    )
    open_action_value = np.where(
        held_mask,
        0.0,
        np.clip(
            0.30 * deploy_action_value
            + 0.26 * multi_horizon_path_value
            + 0.18 * alpha_opportunity_value
            + 0.14 * deployment_opportunity_cost
            + 0.08 * large_upside_1d_target
            + 0.04 * np.clip(working["entry_quality"].to_numpy(dtype=float), 0.0, 1.0)
            - 0.18 * cash_defense_value
            - 0.10 * multi_horizon_forward_risk,
            0.0,
            1.0,
        ),
    )
    add_action_value = np.where(
        held_mask,
        np.clip(
            0.26 * keep_action_value
            + 0.24 * multi_horizon_path_value
            + 0.18 * hold_continuation_value
            + 0.14 * alpha_opportunity_value
            + 0.10 * np.clip(working["add_quality"].to_numpy(dtype=float), 0.0, 1.0)
            + 0.08 * deploy_action_value
            - 0.22 * sell_release_value
            - 0.12 * multi_horizon_forward_risk,
            0.0,
            1.0,
        ),
        0.0,
    )
    hold_action_value = np.where(
        held_mask,
        np.clip(
            0.30 * hold_continuation_value
            + 0.26 * multi_horizon_path_value
            + 0.18 * np.clip(working["hold_quality"].to_numpy(dtype=float), 0.0, 1.0)
            + 0.12 * alpha_opportunity_value
            + 0.08 * deploy_action_value
            + 0.06 * np.clip(hold_continuity_pressure, 0.0, 1.0)
            - 0.22 * sell_release_value
            - 0.14 * multi_horizon_forward_risk,
            0.0,
            1.0,
        ),
        0.0,
    )
    relative_opportunity_value = np.clip(
        0.46 * deployment_opportunity_cost
        + 0.28 * alpha_opportunity_value
        + 0.18 * multi_horizon_path_value
        + 0.08 * large_upside_1d_target
        - 0.24 * hold_continuation_value,
        0.0,
        1.0,
    )
    reduce_action_value = np.where(
        held_mask,
        np.clip(
            0.28 * sell_release_value
            + 0.22 * multi_horizon_forward_risk
            + 0.16 * relative_opportunity_value
            + 0.12 * cash_defense_value
            + 0.10 * working["sell_rank_score"].to_numpy(dtype=float)
            + 0.08 * working["lifecycle_sell_gate"].to_numpy(dtype=float)
            + 0.04 * np.asarray(reduce_fraction_targets, dtype=float)
            - 0.22 * hold_action_value
            - 0.10 * add_action_value,
            0.0,
            1.0,
        ),
        0.0,
    )
    exit_action_value = np.where(
        held_mask,
        np.clip(
            0.30 * sell_release_value
            + 0.24 * multi_horizon_forward_risk
            + 0.18 * cash_defense_value
            + 0.12 * working["lifecycle_sell_gate"].to_numpy(dtype=float)
            + 0.08 * np.asarray(exit_hazard_targets, dtype=float)
            + 0.08 * relative_opportunity_value
            - 0.26 * hold_action_value
            - 0.10 * np.clip(alpha_opportunity_value, 0.0, 1.0),
            0.0,
            1.0,
        ),
        0.0,
    )
    keep_action_value = np.maximum(add_action_value, hold_action_value)
    release_action_value = np.maximum(reduce_action_value, exit_action_value)
    action_value_consistency_target = np.clip(
        0.50 + 0.55 * (np.maximum(open_action_value, keep_action_value) - np.maximum(release_action_value, cash_action_value)),
        0.0,
        1.0,
    )
    action_values = np.select(
        [
            action_series.isin({"open", "add"}).to_numpy(dtype=bool),
            action_series.isin({"hold"}).to_numpy(dtype=bool),
            action_series.isin({"reduce", "exit"}).to_numpy(dtype=bool),
        ],
        [deploy_action_value, keep_action_value, sell_action_value],
        default=cash_action_value,
    )
    capital_value = np.maximum(deploy_action_value, keep_action_value)
    defensive_value = np.maximum(sell_action_value, cash_action_value)
    deploy_value_target = np.clip(capital_value, 0.0, 1.0)
    release_value_target = np.clip(sell_action_value, 0.0, 1.0)
    defense_value_target = np.clip(cash_action_value, 0.0, 1.0)
    gate_denominator = deploy_value_target + release_value_target + defense_value_target + 1.0e-6
    deploy_gate_target = np.clip(deploy_value_target / gate_denominator, 0.0, 1.0)
    release_gate_target = np.clip(release_value_target / gate_denominator, 0.0, 1.0)
    defense_gate_target = np.clip(defense_value_target / gate_denominator, 0.0, 1.0)
    deploy_action_mask = action_series.isin({"open", "add"}).to_numpy(dtype=float)
    deploy_executability_target = np.clip(
        0.30 * deploy_action_value
        + 0.22 * deploy_value_target
        + 0.18 * deploy_gate_target
        + 0.12 * deployment_opportunity_cost
        + 0.10 * alpha_opportunity_value
        + 0.06 * large_upside_1d_target
        + 0.04 * held_float
        - 0.16 * release_gate_target
        - 0.12 * defense_gate_target
        - 0.08 * cash_defense_value,
        0.0,
        1.0,
    )
    deploy_executability_target = np.clip(
        deploy_executability_target * (0.72 + 0.28 * deploy_action_mask),
        0.0,
        1.0,
    )
    current_weight_array = working["current_weight"].fillna(0.0).to_numpy(dtype=float)
    alpha_target_array = np.asarray(alpha_target_weight, dtype=float)
    alpha_support_array = np.asarray(alpha_support, dtype=float)
    market_downside_array = np.asarray(market_downside_pressure, dtype=float)
    cash_regime_array = np.asarray(cash_regime_pressure, dtype=float)
    portfolio_cash_pressure_array = np.asarray(portfolio_cash_pressure, dtype=float)
    position_cap_proxy = np.clip(
        np.maximum(alpha_target_array * 1.22, 0.10)
        + 0.030
        + 0.018 * np.clip(alpha_support_array, 0.0, 1.0)
        + 0.012 * np.clip(multi_horizon_path_value, 0.0, 1.0)
        - 0.020 * np.clip(cash_defense_value, 0.0, 1.0)
        - 0.012 * np.clip(market_downside_array, 0.0, 1.0),
        0.08,
        0.26,
    )
    portfolio_daily_receiver_add_headroom = np.clip(position_cap_proxy - current_weight_array, 0.0, 1.0)
    portfolio_daily_receiver_min_add_delta = np.maximum.reduce(
        [
            np.full(len(working), 0.0025, dtype=float),
            np.clip(current_weight_array, 0.0, None) * 0.025,
            position_cap_proxy * 0.018,
        ]
    )
    portfolio_daily_receiver_add_capacity = np.where(
        held_mask,
        np.clip(
            portfolio_daily_receiver_add_headroom / np.clip(portfolio_daily_receiver_min_add_delta, 1.0e-6, None),
            0.0,
            1.0,
        ),
        1.0,
    )
    receiver_action_value = np.where(held_mask, add_action_value, open_action_value)
    portfolio_daily_receiver_executability = np.clip(
        deploy_executability_target * (0.50 + 0.50 * portfolio_daily_receiver_add_capacity)
        + np.where(held_mask, np.clip(add_action_value - hold_action_value, -0.35, 0.35) * 0.12, 0.0)
        - np.where(held_mask, (1.0 - portfolio_daily_receiver_add_capacity) * 0.24, 0.0),
        0.0,
        1.0,
    )
    portfolio_daily_receiver_score = np.clip(
        0.30 * portfolio_daily_receiver_executability
        + 0.20 * deploy_value_target
        + 0.16 * deploy_gate_target
        + 0.14 * receiver_action_value
        + 0.10 * alpha_opportunity_value
        + 0.08 * relative_opportunity_value
        + 0.06 * multi_horizon_path_value
        - 0.14 * defense_value_target
        - 0.12 * release_value_target
        - np.where(held_mask, (1.0 - portfolio_daily_receiver_add_capacity) * 0.18, 0.0),
        0.0,
        1.0,
    )
    portfolio_daily_receiver_candidate_mask = (
        (deploy_action_mask > 0.5)
        | (portfolio_daily_receiver_score >= 0.42)
        | ((~held_mask) & (portfolio_daily_receiver_executability >= 0.34))
    ).astype(float)
    source_forward_edge = (
        0.58 * forward_edge_5d
        + 0.26 * forward_edge_10d
        + 0.16 * fwd20.to_numpy(dtype=float)
    )
    receiver_candidate_reference_mask = (
        (portfolio_daily_receiver_candidate_mask > 0.5)
        & (portfolio_daily_receiver_add_capacity >= 0.20)
        & (portfolio_daily_receiver_score >= 0.18)
    )
    receiver_forward_reference = 0.0
    if bool(np.any(receiver_candidate_reference_mask)):
        receiver_ref = pd.DataFrame(
            {
                "score": portfolio_daily_receiver_score,
                "forward_edge": source_forward_edge,
            },
            index=working.index,
        ).loc[receiver_candidate_reference_mask]
        receiver_ref = receiver_ref.replace([np.inf, -np.inf], np.nan).dropna()
        if not receiver_ref.empty:
            top_k = max(1, min(5, int(np.ceil(len(receiver_ref) * 0.20))))
            receiver_forward_reference = float(receiver_ref.nlargest(top_k, "score")["forward_edge"].median())
            if not np.isfinite(receiver_forward_reference):
                receiver_forward_reference = 0.0
    portfolio_daily_source_receiver_forward_spread = np.where(
        held_mask,
        np.clip(receiver_forward_reference - source_forward_edge, -0.30, 0.30),
        0.0,
    )
    portfolio_daily_source_forward_spread_score = np.where(
        held_mask,
        np.clip(portfolio_daily_source_receiver_forward_spread / 0.075, 0.0, 1.0),
        0.0,
    )
    portfolio_daily_source_bad_forward_spread_risk = np.where(
        held_mask,
        np.clip(-portfolio_daily_source_receiver_forward_spread / 0.075, 0.0, 1.0),
        0.0,
    )
    source_forward_weakness_score = np.where(
        held_mask,
        np.clip(-source_forward_edge / 0.060, 0.0, 1.0),
        0.0,
    )
    source_forward_strength_risk = np.where(
        held_mask,
        np.clip(source_forward_edge / 0.075, 0.0, 1.0),
        0.0,
    )
    portfolio_daily_source_forward_proxy_keep_risk = np.where(
        held_mask,
        np.clip(
            0.34 * multi_horizon_forward_value
            + 0.30 * multi_horizon_path_value
            + 0.18 * alpha_opportunity_value
            + 0.12 * deploy_value_target
            + 0.06 * release_value_target,
            0.0,
            1.0,
        ),
        0.0,
    )
    source_forward_strength_brake_risk_base = np.where(
        held_mask,
        np.clip(
            0.26 * source_forward_strength_risk
            + 0.18 * portfolio_daily_source_bad_forward_spread_risk
            + 0.18 * hold_continuation_value
            + 0.16 * alpha_opportunity_value
            + 0.14 * large_upside_1d_target
            + 0.12 * multi_horizon_path_value
            + 0.10 * deploy_value_target
            + 0.08 * np.clip(-portfolio_daily_source_receiver_forward_spread / 0.075, 0.0, 1.0)
            + 0.06 * np.clip(1.0 - multi_horizon_forward_risk, 0.0, 1.0)
            - 0.18 * source_forward_weakness_score
            - 0.14 * portfolio_daily_source_forward_spread_score
            - 0.10 * release_value_target
            - 0.08 * sell_release_value
            - 0.06 * cash_defense_value,
            0.0,
            1.0,
        ),
        0.0,
    )
    portfolio_daily_source_release_quality = np.where(
        held_mask,
        np.clip(
            0.36 * portfolio_daily_source_forward_spread_score
            + 0.24 * source_forward_weakness_score
            + 0.18 * multi_horizon_forward_risk
            + 0.10 * cash_defense_value
            + 0.08 * working["sell_rank_score"].to_numpy(dtype=float)
            + 0.06 * working["lifecycle_sell_gate"].to_numpy(dtype=float)
            - 0.22 * hold_continuation_value
            - 0.18 * alpha_opportunity_value
            - 0.14 * large_upside_1d_target
            - 0.26 * portfolio_daily_source_bad_forward_spread_risk
            - 0.22 * source_forward_strength_risk
            - 0.18 * source_forward_strength_brake_risk_base
            - 0.16 * portfolio_daily_source_forward_proxy_keep_risk,
            0.0,
            1.0,
        ),
        0.0,
    )
    portfolio_daily_source_min_release_delta = np.maximum.reduce(
        [
            np.full(len(working), 0.0025, dtype=float),
            np.clip(current_weight_array, 0.0, None) * 0.018,
            position_cap_proxy * 0.012,
        ]
    )
    portfolio_daily_source_release_capacity = np.where(
        held_mask,
        np.clip(
            current_weight_array / np.clip(portfolio_daily_source_min_release_delta, 1.0e-6, None),
            0.0,
            1.0,
        ),
        0.0,
    )
    portfolio_daily_source_opportunity_cost = np.where(
        held_mask,
        np.clip(
            0.34 * hold_continuation_value
            + 0.26 * alpha_opportunity_value
            + 0.20 * multi_horizon_path_value
            + 0.14 * deploy_value_target
            + 0.10 * portfolio_daily_receiver_executability
            + 0.10 * large_upside_1d_target
            + 0.22 * portfolio_daily_source_bad_forward_spread_risk
            + 0.18 * source_forward_strength_risk
            + 0.18 * source_forward_strength_brake_risk_base
            + 0.26 * portfolio_daily_source_forward_proxy_keep_risk
            - 0.10 * release_value_target
            - 0.06 * cash_defense_value
            - 0.05 * multi_horizon_forward_risk
            - 0.20 * portfolio_daily_source_release_quality,
            0.0,
            1.0,
        ),
        0.0,
    )
    portfolio_daily_source_executability = np.where(
        held_mask,
        np.clip(
            0.28 * release_value_target
            + 0.20 * sell_release_value
            + 0.14 * release_gate_target
            + 0.12 * cash_defense_value
            + 0.10 * multi_horizon_forward_risk
            + 0.08 * working["sell_rank_score"].to_numpy(dtype=float)
            + 0.06 * portfolio_daily_source_release_capacity
            + 0.14 * (1.0 - portfolio_daily_source_opportunity_cost)
            + 0.20 * portfolio_daily_source_release_quality
            - 0.18 * hold_continuation_value
            - 0.14 * alpha_opportunity_value
            - 0.10 * portfolio_daily_receiver_executability
            - 0.06 * large_upside_1d_target,
            0.0,
            1.0,
        ),
        0.0,
    )
    portfolio_daily_source_score = np.where(
        held_mask,
        np.clip(
            0.28 * portfolio_daily_source_executability
            + 0.10 * portfolio_daily_source_release_capacity
            + 0.22 * (1.0 - portfolio_daily_source_opportunity_cost)
            + 0.24 * portfolio_daily_source_release_quality
            + 0.13 * release_value_target
            + 0.11 * sell_release_value
            + 0.09 * release_gate_target
            + 0.07 * cash_defense_value
            + 0.06 * multi_horizon_forward_risk
            + 0.05 * relative_opportunity_value
            - 0.15 * hold_continuation_value
            - 0.12 * alpha_opportunity_value
            - 0.10 * portfolio_daily_receiver_executability
            - 0.08 * large_upside_1d_target
            - 0.12 * source_forward_strength_brake_risk_base
            - 0.14 * portfolio_daily_source_forward_proxy_keep_risk,
            0.0,
            1.0,
        ),
        0.0,
    )
    portfolio_daily_source_forward_release_pass = (
        (portfolio_daily_source_receiver_forward_spread >= 0.012)
        | (source_forward_edge <= -0.015)
        | (
            (cash_defense_value >= 0.56)
            & (multi_horizon_forward_risk >= 0.42)
            & (portfolio_daily_source_receiver_forward_spread >= -0.010)
        )
    )
    receiver_demand_reference_for_release = 0.0
    if bool(np.any(receiver_candidate_reference_mask)):
        receiver_release_ref = pd.Series(
            np.clip(
                portfolio_daily_receiver_score
                * np.where(held_mask, portfolio_daily_receiver_add_capacity, 1.0),
                0.0,
                1.0,
            ),
            index=working.index,
        ).loc[receiver_candidate_reference_mask]
        receiver_release_ref = receiver_release_ref.replace([np.inf, -np.inf], np.nan).dropna()
        if not receiver_release_ref.empty:
            top_k = max(1, min(5, int(np.ceil(len(receiver_release_ref) * 0.25))))
            receiver_demand_reference_for_release = float(receiver_release_ref.nlargest(top_k).median())
            if not np.isfinite(receiver_demand_reference_for_release):
                receiver_demand_reference_for_release = 0.0
    portfolio_daily_cash_score = np.clip(
        0.30 * defense_value_target
        + 0.24 * defense_gate_target
        + 0.16 * cash_defense_value
        + 0.10 * np.clip(market_downside_array, 0.0, 1.0)
        + 0.08 * np.clip(cash_regime_array, 0.0, 1.0)
        + 0.06 * np.clip(portfolio_cash_pressure_array, 0.0, 1.0)
        + 0.06 * np.clip(multi_horizon_forward_risk, 0.0, 1.0)
        - 0.18 * portfolio_daily_receiver_score,
        0.0,
        1.0,
    )
    allocation_source_release_pressure = np.where(
        held_mask,
        np.clip(
            0.34 * receiver_demand_reference_for_release
            + 0.22 * np.clip(portfolio_daily_source_receiver_forward_spread / 0.075, 0.0, 1.0)
            + 0.14 * portfolio_daily_source_release_capacity
            + 0.12 * portfolio_daily_cash_score
            + 0.12 * portfolio_daily_source_executability
            + 0.10 * np.clip(portfolio_daily_source_score, 0.0, 1.0)
            - 0.16 * np.clip(-portfolio_daily_source_receiver_forward_spread / 0.075, 0.0, 1.0)
            - 0.12 * hold_continuation_value,
            0.0,
            1.0,
        ),
        0.0,
    )
    allocation_source_release_override = (
        held_mask
        & (allocation_source_release_pressure >= 0.10)
        & (portfolio_daily_source_receiver_forward_spread >= -0.006)
        & (portfolio_daily_source_release_capacity >= 0.45)
    )
    portfolio_daily_source_opportunity_cost = np.where(
        allocation_source_release_override,
        np.clip(
            portfolio_daily_source_opportunity_cost
            - allocation_source_release_pressure * 0.22
            - np.clip(portfolio_daily_source_receiver_forward_spread / 0.075, 0.0, 1.0) * 0.14,
            0.0,
            1.0,
        ),
        portfolio_daily_source_opportunity_cost,
    )
    allocation_source_release_quality_floor = np.clip(
        allocation_source_release_pressure * 0.46
        + np.clip(portfolio_daily_source_receiver_forward_spread / 0.075, 0.0, 1.0) * 0.22
        + np.clip(portfolio_daily_cash_score, 0.0, 1.0) * 0.06,
        0.0,
        1.0,
    )
    portfolio_daily_source_release_quality = np.where(
        allocation_source_release_override,
        np.maximum(portfolio_daily_source_release_quality, allocation_source_release_quality_floor),
        portfolio_daily_source_release_quality,
    )
    portfolio_daily_source_executability = np.where(
        allocation_source_release_override,
        np.clip(
            portfolio_daily_source_executability
            + allocation_source_release_pressure * 0.16
            + portfolio_daily_source_release_quality * 0.08
            - portfolio_daily_source_opportunity_cost * 0.04,
            0.0,
            1.0,
        ),
        portfolio_daily_source_executability,
    )
    portfolio_daily_source_score = np.where(
        allocation_source_release_override,
        np.clip(
            portfolio_daily_source_score
            + allocation_source_release_pressure * 0.34
            + portfolio_daily_source_forward_spread_score * 0.22
            + portfolio_daily_source_release_quality * 0.12
            - portfolio_daily_source_opportunity_cost * 0.02,
            0.0,
            1.0,
        ),
        portfolio_daily_source_score,
    )
    portfolio_daily_source_economic_block_risk = np.where(
        held_mask,
        np.clip(
            0.30 * portfolio_daily_source_bad_forward_spread_risk
            + 0.22 * source_forward_strength_risk
            + 0.18 * hold_continuation_value
            + 0.14 * alpha_opportunity_value
            + 0.10 * large_upside_1d_target
            + 0.10 * portfolio_daily_source_opportunity_cost
            - 0.20 * portfolio_daily_source_release_quality
            - 0.16 * portfolio_daily_source_forward_spread_score
            - 0.08 * allocation_source_release_pressure,
            0.0,
            1.0,
        ),
        0.0,
    )
    portfolio_daily_source_economic_release_score = np.where(
        held_mask,
        np.clip(
            0.30 * portfolio_daily_source_forward_spread_score
            + 0.20 * source_forward_weakness_score
            + 0.16 * portfolio_daily_source_release_quality
            + 0.12 * portfolio_daily_source_executability
            + 0.10 * (1.0 - portfolio_daily_source_opportunity_cost)
            + 0.08 * allocation_source_release_pressure
            + 0.04 * receiver_demand_reference_for_release
            - 0.28 * portfolio_daily_source_bad_forward_spread_risk
            - 0.18 * source_forward_strength_risk
            - 0.12 * hold_continuation_value
            - 0.08 * alpha_opportunity_value,
            0.0,
            1.0,
        ),
        0.0,
    )
    portfolio_daily_source_forward_strength_brake_risk = np.where(
        held_mask,
        np.clip(
            0.58 * source_forward_strength_brake_risk_base
            + 0.18 * source_forward_strength_risk
            + 0.14 * portfolio_daily_source_bad_forward_spread_risk
            + 0.12 * portfolio_daily_source_economic_block_risk
            + 0.08 * np.clip(1.0 - source_forward_weakness_score, 0.0, 1.0)
            - 0.16 * portfolio_daily_source_economic_release_score
            - 0.10 * portfolio_daily_source_forward_spread_score
            - 0.08 * portfolio_daily_source_release_quality
            - 0.06 * cash_defense_value,
            0.0,
            1.0,
        ),
        0.0,
    )
    portfolio_daily_source_economic_block_risk = np.where(
        held_mask,
        np.clip(
            portfolio_daily_source_economic_block_risk
            + portfolio_daily_source_forward_strength_brake_risk * 0.18
            + portfolio_daily_source_forward_proxy_keep_risk * 0.12
            - portfolio_daily_source_economic_release_score * 0.04,
            0.0,
            1.0,
        ),
        0.0,
    )
    portfolio_daily_source_economic_release_score = np.where(
        held_mask,
        np.clip(
            portfolio_daily_source_economic_release_score
            - portfolio_daily_source_forward_strength_brake_risk * 0.16,
            0.0,
            1.0,
        ),
        0.0,
    )
    portfolio_daily_source_opportunity_cost = np.where(
        held_mask,
        np.clip(
            portfolio_daily_source_opportunity_cost
            + portfolio_daily_source_economic_block_risk * 0.16
            + portfolio_daily_source_forward_strength_brake_risk * 0.20
            + portfolio_daily_source_forward_proxy_keep_risk * 0.16
            - portfolio_daily_source_economic_release_score * 0.08,
            0.0,
            1.0,
        ),
        0.0,
    )
    portfolio_daily_source_release_quality = np.where(
        held_mask,
        np.clip(
            portfolio_daily_source_release_quality
            + portfolio_daily_source_economic_release_score * 0.10
            - portfolio_daily_source_economic_block_risk * 0.08
            - portfolio_daily_source_forward_strength_brake_risk * 0.10
            - portfolio_daily_source_forward_proxy_keep_risk * 0.10,
            0.0,
            1.0,
        ),
        0.0,
    )
    portfolio_daily_source_score = np.where(
        held_mask,
        np.clip(
            portfolio_daily_source_score
            + portfolio_daily_source_economic_release_score * 0.20
            - portfolio_daily_source_economic_block_risk * 0.24
            - portfolio_daily_source_forward_strength_brake_risk * 0.24
            - portfolio_daily_source_forward_proxy_keep_risk * 0.14,
            0.0,
            1.0,
        ),
        0.0,
    )
    portfolio_daily_source_release_conviction = np.where(
        held_mask,
        np.clip(
            portfolio_daily_source_score
            + portfolio_daily_source_release_quality * 0.24
            + portfolio_daily_source_economic_release_score * 0.20
            + portfolio_daily_source_executability * 0.12
            + portfolio_daily_source_release_capacity * 0.08
            - portfolio_daily_source_opportunity_cost * 0.35
            - portfolio_daily_source_forward_strength_brake_risk * 0.24
            - portfolio_daily_source_bad_forward_spread_risk * 0.22
            - portfolio_daily_source_economic_block_risk * 0.18
            - portfolio_daily_source_forward_proxy_keep_risk * 0.18,
            -1.0,
            1.0,
        ),
        0.0,
    )
    portfolio_daily_source_release_conviction_pass = (
        (portfolio_daily_source_release_conviction >= 0.340)
        | (
            allocation_source_release_override
            & (portfolio_daily_source_release_conviction >= 0.260)
            & (source_forward_weakness_score >= 0.18)
        )
        | (
            action_series.isin({"exit"}).to_numpy(dtype=bool)
            & (portfolio_daily_source_opportunity_cost <= 0.360)
            & (portfolio_daily_source_release_conviction >= 0.280)
        )
    )
    portfolio_daily_source_distribution_clean_pass = (
        (
            (portfolio_daily_source_receiver_forward_spread >= 0.060)
            & (portfolio_daily_source_forward_strength_brake_risk <= 0.300)
            & (portfolio_daily_source_bad_forward_spread_risk <= 0.180)
            & (portfolio_daily_source_economic_block_risk <= 0.320)
            & (portfolio_daily_source_release_conviction >= 0.360)
        )
        | (
            (portfolio_daily_source_receiver_forward_spread >= 0.025)
            & (portfolio_daily_source_bad_forward_spread_risk >= 0.100)
            & (portfolio_daily_source_bad_forward_spread_risk <= 0.180)
            & (portfolio_daily_source_forward_strength_brake_risk >= 0.180)
            & (portfolio_daily_source_forward_strength_brake_risk <= 0.300)
            & (portfolio_daily_source_economic_block_risk <= 0.320)
            & (portfolio_daily_source_release_conviction >= 0.360)
        )
        | (
            (portfolio_daily_source_receiver_forward_spread >= 0.320)
            & (portfolio_daily_source_forward_spread_score >= 0.200)
            & (portfolio_daily_source_bad_forward_spread_risk <= 0.160)
            & (portfolio_daily_source_release_conviction >= 0.500)
        )
        | (
            allocation_source_release_override
            & (portfolio_daily_source_receiver_forward_spread >= 0.055)
            & (portfolio_daily_source_bad_forward_spread_risk <= 0.140)
            & (portfolio_daily_source_forward_strength_brake_risk <= 0.200)
            & (portfolio_daily_source_economic_block_risk <= 0.240)
            & (portfolio_daily_source_release_conviction >= 0.400)
        )
        | (
            action_series.isin({"exit"}).to_numpy(dtype=bool)
            & (release_gate_target >= deploy_gate_target + 0.220)
            & (portfolio_daily_source_receiver_forward_spread >= 0.100)
            & (portfolio_daily_source_bad_forward_spread_risk <= 0.160)
            & (portfolio_daily_source_release_conviction >= 0.340)
        )
    )
    portfolio_daily_source_forward_strength_brake_pass = (
        (portfolio_daily_source_forward_strength_brake_risk <= 0.48)
        | (
            portfolio_daily_source_forward_release_pass
            & (source_forward_weakness_score >= 0.30)
            & (portfolio_daily_source_forward_spread_score >= 0.34)
            & (portfolio_daily_source_forward_strength_brake_risk <= 0.62)
        )
        | (
            (cash_defense_value >= 0.58)
            & (multi_horizon_forward_risk >= 0.50)
            & (portfolio_daily_source_economic_block_risk <= 0.62)
            & (portfolio_daily_source_forward_strength_brake_risk <= 0.62)
        )
        | (
            action_series.isin({"reduce", "exit"}).to_numpy(dtype=bool)
            & (release_value_target >= 0.42)
            & (sell_release_value >= 0.34)
            & (release_gate_target >= deploy_gate_target + 0.18)
            & (portfolio_daily_source_forward_strength_brake_risk <= 0.66)
        )
    )
    portfolio_daily_source_semantic_release_mask = (
        action_series.isin({"reduce", "exit"}).to_numpy(dtype=bool)
        | (release_value_target >= 0.30)
        | (sell_release_value >= 0.32)
        | (
            (release_gate_target >= deploy_gate_target + 0.16)
            & (portfolio_daily_source_score >= 0.34)
            & (portfolio_daily_source_opportunity_cost <= 0.46)
        )
        | (
            (portfolio_daily_source_release_quality >= 0.34)
            & (portfolio_daily_source_score >= 0.22)
            & (portfolio_daily_source_opportunity_cost <= 0.54)
            & portfolio_daily_source_forward_release_pass
        )
        | (
            (portfolio_daily_source_score >= 0.36)
            & (portfolio_daily_source_opportunity_cost <= 0.18)
            & (portfolio_daily_source_executability >= 0.15)
            & (portfolio_daily_source_release_capacity >= 0.80)
            & portfolio_daily_source_forward_release_pass
        )
        | (
            allocation_source_release_override
            & (portfolio_daily_source_release_quality >= 0.08)
            & (portfolio_daily_source_score >= 0.02)
            & (portfolio_daily_source_opportunity_cost <= 0.86)
            & (portfolio_daily_source_economic_release_score >= 0.10)
            & (portfolio_daily_source_economic_block_risk <= 0.78)
        )
    )
    portfolio_daily_source_candidate_mask = (
        held_mask
        & portfolio_daily_source_semantic_release_mask
        & (portfolio_daily_source_forward_release_pass | allocation_source_release_override)
        & portfolio_daily_source_forward_strength_brake_pass
        & portfolio_daily_source_release_conviction_pass
        & portfolio_daily_source_distribution_clean_pass
        & (
            (portfolio_daily_source_forward_proxy_keep_risk <= 0.340)
            | (
                allocation_source_release_override
                & (source_forward_weakness_score >= 0.22)
                & (portfolio_daily_source_forward_proxy_keep_risk <= 0.400)
            )
            | (
                (cash_defense_value >= 0.60)
                & (multi_horizon_forward_risk >= 0.52)
                & (portfolio_daily_source_forward_proxy_keep_risk <= 0.420)
            )
        )
        & (
            (portfolio_daily_source_economic_release_score >= 0.12)
            | (
                (cash_defense_value >= 0.58)
                & (multi_horizon_forward_risk >= 0.44)
                & (portfolio_daily_source_economic_block_risk <= 0.74)
            )
        )
        & (portfolio_daily_source_economic_block_risk <= 0.82)
        & (
            ((portfolio_daily_source_score >= 0.32) & (portfolio_daily_source_opportunity_cost <= 0.58))
            | (
                (portfolio_daily_source_release_quality >= 0.34)
                & (portfolio_daily_source_score >= 0.22)
                & (portfolio_daily_source_executability >= 0.12)
                & (portfolio_daily_source_opportunity_cost <= 0.54)
            )
            | (
                (portfolio_daily_source_score >= 0.36)
                & (portfolio_daily_source_opportunity_cost <= 0.18)
                & (portfolio_daily_source_executability >= 0.15)
            )
            | (
                (portfolio_daily_source_executability >= 0.30)
                & (portfolio_daily_source_release_capacity >= 0.50)
                & (portfolio_daily_source_opportunity_cost <= 0.52)
            )
            | (
                action_series.isin({"reduce", "exit"}).to_numpy(dtype=bool)
                & (portfolio_daily_source_opportunity_cost <= 0.66)
            )
            | (
                (release_gate_target >= deploy_gate_target + 0.10)
                & (portfolio_daily_source_opportunity_cost <= 0.56)
            )
            | (
                allocation_source_release_override
                & (portfolio_daily_source_release_quality >= 0.08)
                & (portfolio_daily_source_score >= 0.02)
                & (portfolio_daily_source_opportunity_cost <= 0.86)
                & (portfolio_daily_source_economic_release_score >= 0.10)
                & (portfolio_daily_source_economic_block_risk <= 0.78)
            )
        )
    ).astype(float)
    source_release_alignment = np.where(
        held_mask,
        np.clip(
            0.30 * portfolio_daily_source_score
            + 0.22 * portfolio_daily_source_release_quality
            + 0.18 * portfolio_daily_source_executability
            + 0.14 * portfolio_daily_source_release_capacity
            + 0.12 * (1.0 - portfolio_daily_source_opportunity_cost)
            + 0.10 * portfolio_daily_source_forward_spread_score
            + 0.10 * portfolio_daily_source_economic_release_score
            - 0.16 * portfolio_daily_source_economic_block_risk
            - 0.14 * portfolio_daily_source_forward_strength_brake_risk,
            0.0,
            1.0,
        ),
        0.0,
    )
    receiver_demand_strength = np.clip(
        portfolio_daily_receiver_score
        * np.where(held_mask, portfolio_daily_receiver_add_capacity, 1.0)
        * np.where(portfolio_daily_receiver_candidate_mask > 0.5, 1.0, 0.45),
        0.0,
        1.0,
    )
    source_funding_mask = (
        (portfolio_daily_source_candidate_mask > 0.5)
        | (
            held_mask
            & portfolio_daily_source_forward_release_pass
            & portfolio_daily_source_forward_strength_brake_pass
            & (source_release_alignment >= 0.22)
        )
    )
    source_funding_reference = 0.0
    if bool(np.any(source_funding_mask)):
        source_ref = pd.Series(source_release_alignment, index=working.index).loc[source_funding_mask]
        source_ref = source_ref.replace([np.inf, -np.inf], np.nan).dropna()
        if not source_ref.empty:
            top_k = max(1, min(5, int(np.ceil(len(source_ref) * 0.25))))
            source_funding_reference = float(source_ref.nlargest(top_k).median())
            if not np.isfinite(source_funding_reference):
                source_funding_reference = 0.0
    receiver_demand_reference = 0.0
    receiver_funding_mask = portfolio_daily_receiver_candidate_mask > 0.5
    if bool(np.any(receiver_funding_mask)):
        receiver_ref = pd.Series(receiver_demand_strength, index=working.index).loc[receiver_funding_mask]
        receiver_ref = receiver_ref.replace([np.inf, -np.inf], np.nan).dropna()
        if not receiver_ref.empty:
            top_k = max(1, min(5, int(np.ceil(len(receiver_ref) * 0.25))))
            receiver_demand_reference = float(receiver_ref.nlargest(top_k).median())
            if not np.isfinite(receiver_demand_reference):
                receiver_demand_reference = 0.0
    receiver_count_reference = float(max(np.sum(receiver_funding_mask), 1))
    source_candidate_availability = float(
        np.clip(np.sum(source_funding_mask) / receiver_count_reference, 0.0, 1.0)
    )
    cash_funding_reference = float(
        np.clip(
            np.nanmean(np.asarray(portfolio_daily_cash_score, dtype=float))
            if len(np.asarray(portfolio_daily_cash_score, dtype=float))
            else 0.0,
            0.0,
            1.0,
        )
    )
    funding_reference = float(
        np.clip(
            0.58 * source_funding_reference
            + 0.24 * source_candidate_availability
            + 0.18 * cash_funding_reference,
            0.0,
            1.0,
        )
    )
    portfolio_daily_receiver_funding_coverage = np.where(
        portfolio_daily_receiver_candidate_mask > 0.5,
        np.clip(
            np.where(held_mask, portfolio_daily_receiver_add_capacity, 1.0)
            * (0.64 * funding_reference + 0.24 * source_funding_reference + 0.12 * cash_funding_reference),
            0.0,
            1.0,
        ),
        0.0,
    )
    receiver_transfer_value = np.clip(
        receiver_demand_strength
        * (0.44 + 0.56 * portfolio_daily_receiver_funding_coverage)
        * (0.70 + 0.30 * max(source_funding_reference, cash_funding_reference)),
        0.0,
        1.0,
    )
    source_transfer_value = np.clip(
        source_release_alignment
        * (0.48 + 0.52 * receiver_demand_reference)
        + np.where(
            held_mask,
            portfolio_daily_source_forward_spread_score * 0.16
            + portfolio_daily_source_economic_release_score * 0.12
            - portfolio_daily_source_economic_block_risk * 0.10
            - portfolio_daily_source_forward_strength_brake_risk * 0.12,
            0.0,
        ),
        0.0,
        1.0,
    )
    portfolio_daily_allocation_transfer_score = np.clip(
        np.maximum(receiver_transfer_value, source_transfer_value)
        + np.minimum(receiver_transfer_value, source_transfer_value) * 0.18
        - portfolio_daily_cash_score * 0.08,
        0.0,
        1.0,
    )
    portfolio_daily_funding_closure_score = np.clip(
        0.36 * portfolio_daily_allocation_transfer_score
        + 0.28 * portfolio_daily_receiver_funding_coverage
        + 0.22 * source_release_alignment
        + 0.14 * funding_reference,
        0.0,
        1.0,
    )
    dead_receiver_void = float(np.clip(1.0 - receiver_demand_reference / 0.32, 0.0, 1.0))
    dead_source_void = float(np.clip(1.0 - source_funding_reference / 0.28, 0.0, 1.0))
    dead_cash_drag = float(np.clip((cash_funding_reference - 0.34) / 0.46, 0.0, 1.0))
    portfolio_daily_allocation_dead_branch_risk = np.clip(
        0.30 * dead_receiver_void
        + 0.30 * dead_source_void
        + 0.18 * dead_cash_drag
        + 0.22 * (1.0 - portfolio_daily_allocation_transfer_score),
        0.0,
        1.0,
    )
    working["large_upside_1d_target"] = large_upside_1d_target
    working["alpha_opportunity_value"] = alpha_opportunity_value
    working["hold_continuation_value"] = hold_continuation_value
    working["sell_release_value"] = sell_release_value
    working["cash_defense_value"] = cash_defense_value
    working["deployment_opportunity_cost"] = deployment_opportunity_cost
    working["risk_adjusted_action_value"] = np.clip(action_values, 0.0, 1.0)
    working["multi_horizon_forward_value"] = multi_horizon_forward_value
    working["multi_horizon_forward_risk"] = multi_horizon_forward_risk
    working["multi_horizon_path_value"] = multi_horizon_path_value
    working["open_action_value"] = open_action_value
    working["add_action_value"] = add_action_value
    working["hold_action_value"] = hold_action_value
    working["reduce_action_value"] = reduce_action_value
    working["exit_action_value"] = exit_action_value
    working["relative_opportunity_value"] = relative_opportunity_value
    working["action_value_consistency_target"] = action_value_consistency_target
    working["value_arbitration_target"] = np.clip(0.50 + 0.55 * (capital_value - defensive_value), 0.0, 1.0)
    working["deploy_value_target"] = deploy_value_target
    working["release_value_target"] = release_value_target
    working["defense_value_target"] = defense_value_target
    working["deploy_gate_target"] = deploy_gate_target
    working["release_gate_target"] = release_gate_target
    working["defense_gate_target"] = defense_gate_target
    working["deploy_executability_target"] = deploy_executability_target
    working["portfolio_daily_receiver_add_headroom"] = portfolio_daily_receiver_add_headroom
    working["portfolio_daily_receiver_min_add_delta"] = portfolio_daily_receiver_min_add_delta
    working["portfolio_daily_receiver_add_capacity"] = portfolio_daily_receiver_add_capacity
    working["portfolio_daily_receiver_executability"] = portfolio_daily_receiver_executability
    working["portfolio_daily_receiver_score"] = portfolio_daily_receiver_score
    working["portfolio_daily_source_min_release_delta"] = portfolio_daily_source_min_release_delta
    working["portfolio_daily_source_release_capacity"] = portfolio_daily_source_release_capacity
    working["portfolio_daily_source_receiver_forward_spread"] = portfolio_daily_source_receiver_forward_spread
    working["portfolio_daily_source_forward_spread_score"] = portfolio_daily_source_forward_spread_score
    working["portfolio_daily_source_bad_forward_spread_risk"] = portfolio_daily_source_bad_forward_spread_risk
    working["portfolio_daily_source_economic_release_score"] = portfolio_daily_source_economic_release_score
    working["portfolio_daily_source_economic_block_risk"] = portfolio_daily_source_economic_block_risk
    working["portfolio_daily_source_forward_strength_brake_risk"] = portfolio_daily_source_forward_strength_brake_risk
    working["portfolio_daily_source_forward_proxy_keep_risk"] = portfolio_daily_source_forward_proxy_keep_risk
    working["portfolio_daily_source_release_conviction"] = portfolio_daily_source_release_conviction
    working["portfolio_daily_source_distribution_clean_pass"] = portfolio_daily_source_distribution_clean_pass.astype(float)
    working["portfolio_daily_source_release_quality"] = portfolio_daily_source_release_quality
    working["portfolio_daily_source_opportunity_cost"] = portfolio_daily_source_opportunity_cost
    working["portfolio_daily_source_executability"] = portfolio_daily_source_executability
    working["portfolio_daily_source_score"] = portfolio_daily_source_score
    working["portfolio_daily_cash_score"] = portfolio_daily_cash_score
    working["portfolio_daily_receiver_funding_coverage"] = portfolio_daily_receiver_funding_coverage
    working["portfolio_daily_funding_closure_score"] = portfolio_daily_funding_closure_score
    working["portfolio_daily_allocation_transfer_score"] = portfolio_daily_allocation_transfer_score
    working["portfolio_daily_allocation_dead_branch_risk"] = portfolio_daily_allocation_dead_branch_risk
    working["portfolio_daily_receiver_candidate_mask"] = portfolio_daily_receiver_candidate_mask
    working["portfolio_daily_source_candidate_mask"] = portfolio_daily_source_candidate_mask
    working["clipped_intent_risk"] = 0.0
    working["holding_flag_target"] = held_float
    working["forward_benchmark_return_1d"] = benchmark_fwd1.to_numpy(dtype=float)
    working["forward_benchmark_return_3d"] = benchmark_fwd3.to_numpy(dtype=float)
    working["forward_benchmark_return_5d"] = benchmark_fwd5.to_numpy(dtype=float)
    working["forward_benchmark_return_10d"] = benchmark_fwd10.to_numpy(dtype=float)
    working["forward_benchmark_return_20d"] = benchmark_fwd20.to_numpy(dtype=float)
    working["label_preset"] = config.name
    numeric_output_columns = (
        "teacher_edge",
        "teacher_opportunity",
        "teacher_downside",
        "teacher_signal",
        "teacher_urgency",
        "entry_quality",
        "hold_quality",
        "add_quality",
        "reduce_quality",
        "reentry_readiness",
        "target_delta_hint",
        "teacher_priority",
        "exit_urgency",
        "reduce_fraction_target",
        "exit_hazard_target",
        "sell_attribution_score",
        "sell_rank_score",
        "lifecycle_sell_gate",
        "large_upside_1d_target",
        "alpha_opportunity_value",
        "hold_continuation_value",
        "sell_release_value",
        "cash_defense_value",
        "deployment_opportunity_cost",
        "risk_adjusted_action_value",
        "multi_horizon_forward_value",
        "multi_horizon_forward_risk",
        "multi_horizon_path_value",
        "open_action_value",
        "add_action_value",
        "hold_action_value",
        "reduce_action_value",
        "exit_action_value",
        "relative_opportunity_value",
        "action_value_consistency_target",
        "value_arbitration_target",
        "deploy_value_target",
        "release_value_target",
        "defense_value_target",
        "deploy_gate_target",
        "release_gate_target",
        "defense_gate_target",
        "deploy_executability_target",
        "portfolio_daily_receiver_add_headroom",
        "portfolio_daily_receiver_min_add_delta",
        "portfolio_daily_receiver_add_capacity",
        "portfolio_daily_receiver_executability",
        "portfolio_daily_receiver_score",
        "portfolio_daily_source_release_capacity",
        "portfolio_daily_source_receiver_forward_spread",
        "portfolio_daily_source_forward_spread_score",
        "portfolio_daily_source_bad_forward_spread_risk",
        "portfolio_daily_source_economic_release_score",
        "portfolio_daily_source_economic_block_risk",
        "portfolio_daily_source_forward_strength_brake_risk",
        "portfolio_daily_source_forward_proxy_keep_risk",
        "portfolio_daily_source_release_conviction",
        "portfolio_daily_source_distribution_clean_pass",
        "portfolio_daily_source_release_quality",
        "portfolio_daily_source_opportunity_cost",
        "portfolio_daily_source_executability",
        "portfolio_daily_source_score",
        "portfolio_daily_cash_score",
        "portfolio_daily_receiver_funding_coverage",
        "portfolio_daily_funding_closure_score",
        "portfolio_daily_allocation_transfer_score",
        "portfolio_daily_allocation_dead_branch_risk",
        "portfolio_daily_receiver_candidate_mask",
        "portfolio_daily_source_candidate_mask",
        "clipped_intent_risk",
        "holding_flag_target",
        "forward_benchmark_return_1d",
        "forward_benchmark_return_3d",
        "forward_benchmark_return_5d",
        "forward_benchmark_return_10d",
        "forward_benchmark_return_20d",
    )
    for column in numeric_output_columns:
        working[column] = working[column].astype(float).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    return working


def build_teacher_global_targets(
    label_frame: pd.DataFrame,
    *,
    label_config: str | LifecycleLabelConfig | None = None,
) -> dict[str, float]:
    config = resolve_label_config(label_config)
    positive = label_frame.loc[label_frame["action_label"].isin({"open", "add", "hold"})].copy()
    strong_positive = positive.loc[positive["teacher_priority"] > 0]
    candidate_budget = int(np.clip(len(strong_positive), 2, 12)) if len(strong_positive) else 2
    avg_duration_days = float(strong_positive["planned_holding_days"].mean()) if len(strong_positive) else 0.0
    concentration_signal = float(strong_positive["teacher_priority"].nlargest(min(3, len(strong_positive))).sum()) if len(strong_positive) else 0.0
    benchmark_trend_gap = float(label_frame.get("benchmark_trend_gap", pd.Series([0.0])).astype(float).iloc[0]) if not label_frame.empty else 0.0
    benchmark_vol_ratio = float(label_frame.get("benchmark_vol_ratio", pd.Series([0.0])).astype(float).iloc[0]) if not label_frame.empty else 0.0
    portfolio_recent_turnover_5d = float(label_frame.get("portfolio_recent_turnover_5d", pd.Series([0.0])).astype(float).iloc[0]) if not label_frame.empty else 0.0
    portfolio_drawdown_20d = float(label_frame.get("portfolio_drawdown_20d", pd.Series([0.0])).astype(float).iloc[0]) if not label_frame.empty else 0.0
    market_downside_pressure = float(label_frame.get("market_downside_pressure", pd.Series([0.0])).astype(float).iloc[0]) if not label_frame.empty else 0.0
    portfolio_cash_pressure = float(label_frame.get("portfolio_cash_pressure", pd.Series([0.0])).astype(float).iloc[0]) if not label_frame.empty else 0.0
    recent_reversal_rate_20d = float(label_frame.get("recent_reversal_rate_20d", pd.Series([0.0])).astype(float).iloc[0]) if not label_frame.empty else 0.0
    cash_regime_pressure = float(label_frame.get("cash_regime_pressure", pd.Series([0.0])).astype(float).iloc[0]) if not label_frame.empty else 0.0
    reduce_reversal_pressure = float(label_frame.get("reduce_reversal_pressure", pd.Series([0.0])).astype(float).iloc[0]) if not label_frame.empty else 0.0
    hold_share = float((label_frame["action_label"] == "hold").mean()) if len(label_frame) else 0.0
    avg_reduce_fraction_target = (
        float(label_frame.get("reduce_fraction_target", pd.Series([0.0])).astype(float).mean()) if not label_frame.empty else 0.0
    )
    avg_exit_hazard_target = (
        float(label_frame.get("exit_hazard_target", pd.Series([0.0])).astype(float).mean()) if not label_frame.empty else 0.0
    )
    sell_pressure_target = float(np.clip(0.55 * avg_reduce_fraction_target + 0.45 * avg_exit_hazard_target, 0.0, 1.0))
    defensive_market = (
        0.55 * max(-benchmark_trend_gap, 0.0)
        + 0.25 * max(benchmark_vol_ratio, 0.0)
        + 0.20 * max(-portfolio_drawdown_20d - 0.02, 0.0)
    )
    cash_regime = (
        defensive_market * float(config.cash_regime_sensitivity)
        + portfolio_recent_turnover_5d * float(config.turnover_sensitivity)
        + market_downside_pressure * 0.24
        + portfolio_cash_pressure * 0.18
        + cash_regime_pressure * 0.16
        + recent_reversal_rate_20d * 0.05
    )
    cash_defense_pressure = float(
        np.clip(
            0.40 * max(market_downside_pressure - 0.04, 0.0)
            + 0.28 * max(cash_regime_pressure - 0.04, 0.0)
            + 0.16 * max(portfolio_cash_pressure - 0.06, 0.0)
            + 0.16 * max(-benchmark_trend_gap - 0.01, 0.0),
            0.0,
            1.0,
        )
    )
    gross_target = (
        float(np.clip(strong_positive["teacher_priority"].sum() * float(config.gross_scale), float(config.min_gross_target), float(config.max_gross_target)))
        if len(strong_positive)
        else float(config.min_gross_target)
    )
    gross_target = float(
        np.clip(
            gross_target
            - float(config.defensive_cash_bias)
            - cash_regime * 0.24
            - sell_pressure_target * 0.18
            - avg_exit_hazard_target * 0.06
            - cash_defense_pressure * 0.06,
            0.12,
            float(config.max_gross_target),
        )
    )

    exits = int((label_frame["action_label"] == "exit").sum())
    reduces = int((label_frame["action_label"] == "reduce").sum())
    opens = int((label_frame["action_label"] == "open").sum())
    adds = int((label_frame["action_label"] == "add").sum())
    turnover_budget = float(
        np.clip(
            float(config.base_turnover_budget)
            + opens * 0.025
            + adds * 0.018
            + reduces * 0.020
            + exits * 0.030
            - avg_duration_days * 0.002,
            0.10,
            float(config.max_turnover_budget),
        )
    )
    turnover_budget = float(
        np.clip(
            turnover_budget
            - cash_regime * 0.18
            - recent_reversal_rate_20d * 0.10
            + min((reduces + exits) * 0.01, 0.06)
            + sell_pressure_target * 0.10
            + avg_exit_hazard_target * 0.05,
            0.08,
            float(config.max_turnover_budget),
        )
    )
    turnover_budget = float(
        np.clip(
            turnover_budget - reduce_reversal_pressure * 0.08 + cash_defense_pressure * 0.03,
            0.08,
            float(config.max_turnover_budget),
        )
    )
    candidate_budget = int(
        np.clip(
            round(candidate_budget - cash_regime * 4.0 - cash_defense_pressure * 0.75),
            2,
            12,
        )
    )
    max_position_weight_target = float(
        np.clip(
            0.10
            + concentration_signal * 0.025
            + max(avg_duration_days - 3.0, 0.0) * 0.002
            - cash_regime * 0.04
            - recent_reversal_rate_20d * 0.03
            - sell_pressure_target * 0.03,
            0.08,
            0.26,
        )
    )
    max_position_weight_target = float(
        np.clip(
            max_position_weight_target - cash_defense_pressure * 0.010,
            0.08,
            0.26,
        )
    )
    hold_bias_target = float(
        np.clip(
            0.18
            + avg_duration_days * 0.035
            + hold_share * 0.16
            - market_downside_pressure * 0.05
            - recent_reversal_rate_20d * 0.05
            - sell_pressure_target * 0.18
            - avg_exit_hazard_target * 0.06
            - cash_defense_pressure * 0.05
            + max(0.0, 0.18 - reduce_reversal_pressure) * 0.10,
            0.12,
            0.92,
        )
    )
    reduce_bias_target = float(
        np.clip(
            0.06
            + sell_pressure_target * 0.34
            + avg_exit_hazard_target * 0.14
            + recent_reversal_rate_20d * 0.12
            + reduce_reversal_pressure * 0.16
            + cash_regime * 0.10
            + cash_defense_pressure * 0.08
            - hold_share * 0.06
            - max(avg_duration_days - 6.0, 0.0) * 0.004,
            0.0,
            0.65,
        )
    )
    exit_patience_target = float(
        np.clip(
            0.14
            + hold_bias_target * 0.42
            + max(avg_duration_days - 4.0, 0.0) * 0.008
            + max(benchmark_trend_gap, 0.0) * 0.05
            - sell_pressure_target * 0.24
            - avg_exit_hazard_target * 0.22
            - cash_regime * 0.12
            - cash_defense_pressure * 0.10,
            0.05,
            0.95,
        )
    )
    reentry_guard_target = float(
        np.clip(
            0.02
            + recent_reversal_rate_20d * 0.20
            + reduce_reversal_pressure * 0.26
            + cash_regime * 0.10
            + cash_defense_pressure * 0.08
            + avg_exit_hazard_target * 0.08
            - max(hold_bias_target - 0.40, 0.0) * 0.08,
            0.0,
            0.45,
        )
    )
    return {
        "gross_exposure_target": gross_target,
        "candidate_budget": float(candidate_budget),
        "turnover_budget": turnover_budget,
        "max_position_weight_target": max_position_weight_target,
        "hold_bias_target": hold_bias_target,
        "reduce_bias_target": reduce_bias_target,
        "exit_patience_target": exit_patience_target,
        "reentry_guard_target": reentry_guard_target,
    }


def build_teacher_policy_frame(label_frame: pd.DataFrame) -> pd.DataFrame:
    working = label_frame.set_index("stock").copy()
    policy = pd.DataFrame(index=working.index)
    policy["action_label"] = working["action_label"].astype(str)
    policy["action_strength"] = working["teacher_priority"].astype(float).clip(lower=0.0)
    policy["target_delta_hint"] = working["target_delta_hint"].astype(float)
    policy["entry_quality"] = working["entry_quality"].astype(float)
    policy["hold_quality"] = working["hold_quality"].astype(float)
    policy["add_quality"] = working["add_quality"].astype(float)
    policy["reduce_quality"] = working["reduce_quality"].astype(float)
    policy["reentry_readiness"] = working["reentry_readiness"].astype(float)
    policy["reduce_fraction"] = working.get("reduce_fraction_target", pd.Series(0.0, index=working.index)).astype(float).clip(0.0, 1.0)
    policy["exit_hazard"] = working.get("exit_hazard_target", pd.Series(0.0, index=working.index)).astype(float).clip(0.0, 1.0)
    policy["sell_attribution_score"] = working.get("sell_attribution_score", pd.Series(0.0, index=working.index)).astype(float).clip(0.0, 1.0)
    for column in (
        "sell_rank_score",
        "lifecycle_sell_gate",
        "large_upside_1d_target",
        "alpha_opportunity_value",
        "hold_continuation_value",
        "sell_release_value",
        "cash_defense_value",
        "deployment_opportunity_cost",
        "risk_adjusted_action_value",
        "multi_horizon_forward_value",
        "multi_horizon_forward_risk",
        "multi_horizon_path_value",
        "open_action_value",
        "add_action_value",
        "hold_action_value",
        "reduce_action_value",
        "exit_action_value",
        "relative_opportunity_value",
        "action_value_consistency_target",
        "value_arbitration_target",
        "deploy_value_target",
        "release_value_target",
        "defense_value_target",
        "deploy_gate_target",
        "release_gate_target",
        "defense_gate_target",
        "deploy_executability_target",
        "portfolio_daily_receiver_add_headroom",
        "portfolio_daily_receiver_min_add_delta",
        "portfolio_daily_receiver_add_capacity",
        "portfolio_daily_receiver_executability",
        "portfolio_daily_receiver_score",
        "portfolio_daily_source_release_capacity",
        "portfolio_daily_source_forward_spread_score",
        "portfolio_daily_source_bad_forward_spread_risk",
        "portfolio_daily_source_economic_release_score",
        "portfolio_daily_source_economic_block_risk",
        "portfolio_daily_source_forward_strength_brake_risk",
        "portfolio_daily_source_forward_proxy_keep_risk",
        "portfolio_daily_source_release_quality",
        "portfolio_daily_source_opportunity_cost",
        "portfolio_daily_source_executability",
        "portfolio_daily_source_score",
        "portfolio_daily_cash_score",
        "portfolio_daily_receiver_funding_coverage",
        "portfolio_daily_funding_closure_score",
        "portfolio_daily_allocation_transfer_score",
        "portfolio_daily_allocation_dead_branch_risk",
    ):
        policy[column] = working.get(column, pd.Series(0.0, index=working.index)).astype(float).clip(0.0, 1.0)
    policy["planned_holding_days"] = working["planned_holding_days"].astype(float)
    policy["planned_holding_bucket"] = working["planned_holding_bucket"].astype(str)
    policy["hold_boost"] = np.where(
        working["action_label"].isin({"hold", "add"}),
        working["hold_quality"].astype(float).clip(lower=0.0) + working["teacher_priority"].astype(float).clip(lower=0.0) * 0.60,
        0.0,
    )
    policy["exit_urgency"] = working["exit_urgency"].astype(float).clip(lower=0.0)
    return policy
