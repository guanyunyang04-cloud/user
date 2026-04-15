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
    for horizon in LABEL_HORIZONS:
        frames[f"fwd_excess_{horizon}d"] = build_ml_target(close, benchmark_close, horizon, execution_mode="close")
        future_max = _future_window_extreme(close, horizon, mode="max")
        future_min = _future_window_extreme(close, horizon, mode="min")
        frames[f"future_max_up_{horizon}d"] = future_max.div(close).sub(1.0)
        frames[f"future_min_down_{horizon}d"] = future_min.div(close).sub(1.0)
    return FuturePathMetrics(frames=frames)


def _row_metric(metrics: FuturePathMetrics, name: str, date: pd.Timestamp, stocks: pd.Index) -> pd.Series:
    frame = metrics.frames[name]
    return frame.loc[date].reindex(stocks).astype(float)


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
    recent_reduce_cooldown = np.clip((4.0 - days_since_last_reduce) / 4.0, 0.0, 1.0)
    recent_exit_cooldown = np.clip((5.0 - days_since_last_exit) / 5.0, 0.0, 1.0)

    short_edge = 0.55 * fwd1 + 0.45 * fwd3
    mid_edge = 0.35 * fwd3 + 0.65 * fwd5
    long_edge = 0.20 * fwd5 + 0.35 * fwd10 + 0.45 * fwd20
    edge = 0.15 * short_edge + 0.35 * mid_edge + 0.50 * long_edge
    opportunity = 0.25 * max_up5 + 0.35 * max_up10 + 0.40 * max_up20
    downside = 0.25 * min_down5.clip(upper=0).abs() + 0.35 * min_down10.clip(upper=0).abs() + 0.40 * min_down20.clip(upper=0).abs()
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
    )
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
                or (risk_off_reduce and hold_days >= float(config.min_hold_days))
                or (reversal_rate_value > 0.22 and signal_decay_value > 0.04)
            ) and not wrong_side_profit_take and not (continuation_hold and reduce_reversal_value > 0.18):
                if signal_decay_value > 0.09 or drawdown < -0.11 or market_downside_value > 0.24:
                    action = "exit" if hold_days >= float(config.min_hold_days) else "reduce"
                    delta_hint = -1.0 if action == "exit" else -min(0.65, 0.14 + reduce_quality * 1.15 + max(market_downside_value, 0.0))
                else:
                    action = "reduce"
                    delta_hint = -min(0.70, 0.16 + reduce_quality * 1.2 + max(-signal, 0.0) + market_downside_value * 0.25)
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
            reduce_fraction_target = float(np.clip(max(delta_fraction, 0.12 + sell_pressure * 0.88), 0.12, 0.96))
            exit_hazard_target = float(
                np.clip(
                    exit_pressure * 0.55
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
            0.34 * max(market_downside_pressure, 0.0)
            + 0.26 * max(cash_regime_pressure, 0.0)
            + 0.16 * max(portfolio_cash_pressure, 0.0)
            + 0.14 * max(-benchmark_trend_gap, 0.0)
            + 0.10 * sell_pressure_target
            + 0.08 * avg_exit_hazard_target,
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
            - cash_defense_pressure * 0.12,
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
            turnover_budget - reduce_reversal_pressure * 0.08 + cash_defense_pressure * 0.08,
            0.08,
            float(config.max_turnover_budget),
        )
    )
    candidate_budget = int(
        np.clip(
            round(candidate_budget - cash_regime * 4.0 - cash_defense_pressure * 2.0),
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
            max_position_weight_target - cash_defense_pressure * 0.025,
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
            - cash_defense_pressure * 0.14
            + max(0.0, 0.18 - reduce_reversal_pressure) * 0.10,
            0.12,
            0.92,
        )
    )
    return {
        "gross_exposure_target": gross_target,
        "candidate_budget": float(candidate_budget),
        "turnover_budget": turnover_budget,
        "max_position_weight_target": max_position_weight_target,
        "hold_bias_target": hold_bias_target,
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
    policy["planned_holding_days"] = working["planned_holding_days"].astype(float)
    policy["planned_holding_bucket"] = working["planned_holding_bucket"].astype(str)
    policy["hold_boost"] = np.where(
        working["action_label"].isin({"hold", "add"}),
        working["hold_quality"].astype(float).clip(lower=0.0) + working["teacher_priority"].astype(float).clip(lower=0.0) * 0.60,
        0.0,
    )
    policy["exit_urgency"] = working["exit_urgency"].astype(float).clip(lower=0.0)
    return policy
