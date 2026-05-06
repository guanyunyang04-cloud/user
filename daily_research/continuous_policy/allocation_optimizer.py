from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd


UNIFIED_ALLOCATION_COLUMNS: tuple[str, ...] = (
    "portfolio_daily_pre_unified_receiver_score",
    "portfolio_daily_pre_unified_source_score",
    "portfolio_daily_pre_unified_cash_score",
    "portfolio_daily_unified_receiver_score",
    "portfolio_daily_unified_source_score",
    "portfolio_daily_unified_cash_score",
    "portfolio_daily_source_positive_forward_penalty",
    "portfolio_daily_source_opportunity_cost_penalty",
    "portfolio_daily_source_strong_false_sell_penalty",
    "portfolio_daily_source_hard_negative_penalty",
    "portfolio_daily_source_tail_false_sell_penalty",
    "portfolio_daily_source_release_preference",
    "portfolio_daily_receiver_source_spread_reward",
    "portfolio_daily_transfer_regret_target",
    "portfolio_daily_unified_allocation_objective",
    "portfolio_daily_allocation_trade_quality_target",
    "portfolio_daily_allocation_cash_deployment_target",
    "portfolio_daily_allocation_risk_adjusted_return_target",
    "portfolio_daily_allocation_drawdown_control_target",
    "portfolio_daily_allocation_monthly_quality_target",
    "portfolio_daily_allocation_final_objective",
    "portfolio_daily_allocation_uncertainty_pressure_target",
    "portfolio_daily_allocation_tail_risk_control_target",
    "portfolio_daily_allocation_decision_focused_objective",
    "portfolio_daily_unified_receiver_candidate",
    "portfolio_daily_unified_source_candidate",
)


@dataclass(frozen=True)
class AllocationOptimizerConstraints:
    cash_reserve_target: float = 0.05
    turnover_limit: float = 0.60
    max_position_weight: float = 0.20
    transaction_cost_bps: float = 3.0
    slippage_bps: float = 7.0
    sell_tax_bps: float = 10.0
    min_trade_weight: float = 0.001


@dataclass(frozen=True)
class AllocationSolution:
    target_weight: pd.Series
    cash_after: float
    expected_turnover: float
    buy_turnover: float
    sell_turnover: float
    available_cash_to_deploy: float
    allocation_objective_value: float
    diagnostics: dict[str, float]


def _series(frame: pd.DataFrame, name: str, default: float = 0.0) -> pd.Series:
    if name in frame.columns:
        values = pd.to_numeric(frame[name], errors="coerce")
    else:
        values = pd.Series(float(default), index=frame.index, dtype=float)
    return values.replace([np.inf, -np.inf], np.nan).fillna(float(default)).astype(float)


def _clip_series(values: pd.Series, low: float = 0.0, high: float | None = 1.0) -> pd.Series:
    clean = values.astype(float)
    if high is None:
        clipped = np.maximum(clean, float(low))
    else:
        clipped = np.clip(clean, float(low), float(high))
    return pd.Series(clipped, index=values.index, dtype=float)


def _coalesce_series(frame: pd.DataFrame, names: tuple[str, ...], default: float = 0.0) -> pd.Series:
    out = pd.Series(float(default), index=frame.index, dtype=float)
    for name in names:
        if name not in frame.columns:
            continue
        values = _series(frame, name, default=np.nan)
        out = out.where(out.notna(), values)
        missing_or_default = out.isna() | (out == float(default))
        out = out.where(~missing_or_default, values.fillna(float(default)))
    return out.replace([np.inf, -np.inf], np.nan).fillna(float(default)).astype(float)


def _action_series(frame: pd.DataFrame) -> pd.Series:
    if "action_label" not in frame.columns:
        return pd.Series("", index=frame.index, dtype=object)
    return frame["action_label"].astype(str).str.lower()


def build_unified_allocation_problem(label_frame: pd.DataFrame) -> pd.DataFrame:
    """Build one source/receiver/cash listwise allocation surface.

    The surface is deliberately continuous and projection-friendly: upstream
    models can learn the scalar heads, while the final optimizer can still
    enforce cash, turnover, position-cap and cost constraints explicitly.
    """
    working = label_frame.copy()
    if working.empty:
        for column in UNIFIED_ALLOCATION_COLUMNS:
            working[column] = pd.Series(dtype=float)
        return working

    action = _action_series(working)
    current_weight = _clip_series(_series(working, "current_weight"), 0.0, 1.0)
    receiver_raw = _clip_series(_series(working, "portfolio_daily_receiver_score"), 0.0, 1.0)
    source_raw = _clip_series(_series(working, "portfolio_daily_source_score"), 0.0, 1.0)
    cash_raw = _clip_series(_series(working, "portfolio_daily_cash_score"), 0.0, 1.0)
    working["portfolio_daily_pre_unified_receiver_score"] = receiver_raw
    working["portfolio_daily_pre_unified_source_score"] = source_raw
    working["portfolio_daily_pre_unified_cash_score"] = cash_raw

    receiver_mask = (
        (_series(working, "portfolio_daily_receiver_candidate_mask") > 0.5)
        | action.isin({"open", "add"})
        | (receiver_raw > 0.05)
    )
    source_mask = (
        (_series(working, "portfolio_daily_source_candidate_mask") > 0.5)
        | action.isin({"reduce", "exit"})
        | ((source_raw > 0.05) & (current_weight > 1.0e-8))
    )
    if "portfolio_daily_receiver_executable_candidate" in working.columns:
        receiver_mask = receiver_mask & (_series(working, "portfolio_daily_receiver_executable_candidate") > 0.5)
    if "portfolio_daily_source_executable_candidate" in working.columns:
        source_mask = source_mask & (_series(working, "portfolio_daily_source_executable_candidate") > 0.5)
    receiver_exec = _clip_series(_series(working, "portfolio_daily_receiver_executability", default=1.0), 0.0, 1.0)
    source_exec = _clip_series(_series(working, "portfolio_daily_source_executability", default=1.0), 0.0, 1.0)
    receiver_headroom = _clip_series(
        _coalesce_series(
            working,
            (
                "portfolio_daily_receiver_add_headroom",
                "portfolio_daily_receiver_add_capacity",
                "portfolio_daily_receiver_min_add_delta",
            ),
            default=1.0,
        ),
        0.0,
        1.0,
    )
    receiver_headroom = receiver_headroom.where(current_weight > 1.0e-8, 1.0)
    source_capacity = _clip_series(_series(working, "portfolio_daily_source_release_capacity", default=1.0), 0.0, 1.0)
    source_release_quality = _clip_series(_series(working, "portfolio_daily_source_release_quality"), 0.0, 1.0)

    raw_spread = _series(working, "portfolio_daily_source_receiver_forward_spread", default=np.nan)
    if raw_spread.isna().all():
        raw_spread = (_series(working, "portfolio_daily_source_forward_spread_score") - 0.5) * 0.16
    receiver_source_spread_reward = _clip_series(raw_spread / 0.08, 0.0, 1.0)

    source_forward = _coalesce_series(
        working,
        (
            "portfolio_daily_source_forward_excess_5d",
            "source_forward_excess_5d",
            "forward_excess_5d",
        ),
        default=0.0,
    )
    predicted_positive_forward_penalty = _clip_series(
        _series(working, "portfolio_daily_source_positive_forward_penalty"),
        0.0,
        1.0,
    )
    positive_forward_penalty = _clip_series(
        np.maximum(source_forward / 0.08, predicted_positive_forward_penalty),
        0.0,
        1.0,
    )
    forward_proxy_keep_risk = _clip_series(_series(working, "portfolio_daily_source_forward_proxy_keep_risk"), 0.0, 1.0)
    forward_strength_brake = _clip_series(_series(working, "portfolio_daily_source_forward_strength_brake_risk"), 0.0, 1.0)
    source_opportunity_cost = _clip_series(_series(working, "portfolio_daily_source_opportunity_cost"), 0.0, 1.0)
    predicted_opportunity_cost_penalty = _clip_series(
        _series(working, "portfolio_daily_source_opportunity_cost_penalty"),
        0.0,
        1.0,
    )
    opportunity_cost_penalty = _clip_series(
        np.maximum(source_opportunity_cost, predicted_opportunity_cost_penalty),
        0.0,
        1.0,
    )
    bad_spread_risk = _clip_series(_series(working, "portfolio_daily_source_bad_forward_spread_risk"), 0.0, 1.0)
    economic_block_risk = _clip_series(_series(working, "portfolio_daily_source_economic_block_risk"), 0.0, 1.0)
    economic_release = _clip_series(_series(working, "portfolio_daily_source_economic_release_score"), 0.0, 1.0)
    transfer_score = _clip_series(_series(working, "portfolio_daily_allocation_transfer_score"), 0.0, 1.0)
    dead_branch_risk = _clip_series(_series(working, "portfolio_daily_allocation_dead_branch_risk"), 0.0, 1.0)
    receiver_forward = _coalesce_series(
        working,
        (
            "portfolio_daily_receiver_forward_excess_5d",
            "receiver_forward_excess_5d",
            "forward_excess_5d",
        ),
        default=0.0,
    )
    market_downside = _clip_series(_series(working, "market_downside_pressure"), 0.0, 1.0)
    cash_regime = _clip_series(_series(working, "cash_regime_pressure"), 0.0, 1.0)
    portfolio_cash_pressure = _clip_series(_series(working, "portfolio_cash_pressure"), 0.0, 1.0)
    multi_horizon_forward_risk = _clip_series(_series(working, "multi_horizon_forward_risk"), 0.0, 1.0)
    cash_defense_value = _clip_series(_series(working, "cash_defense_value"), 0.0, 1.0)
    portfolio_drawdown = _series(working, "portfolio_drawdown_20d")
    drawdown_pressure = _clip_series((-portfolio_drawdown - 0.02) / 0.10, 0.0, 1.0)
    forward_benchmark_1d = _series(working, "forward_benchmark_return_1d")
    forward_benchmark_3d = _series(working, "forward_benchmark_return_3d")
    benchmark_downside_timing = _clip_series(
        0.62 * np.clip(-forward_benchmark_1d / 0.025, 0.0, 1.0)
        + 0.38 * np.clip(-forward_benchmark_3d / 0.045, 0.0, 1.0),
        0.0,
        1.0,
    )
    benchmark_upside_timing = _clip_series(
        0.62 * np.clip(forward_benchmark_1d / 0.025, 0.0, 1.0)
        + 0.38 * np.clip(forward_benchmark_3d / 0.045, 0.0, 1.0),
        0.0,
        1.0,
    )
    risk_off_pressure = _clip_series(
        0.26 * market_downside
        + 0.22 * cash_regime
        + 0.20 * drawdown_pressure
        + 0.16 * multi_horizon_forward_risk
        + 0.10 * cash_defense_value
        + 0.06 * portfolio_cash_pressure,
        0.0,
        1.0,
    )
    strong_positive_forward_penalty = _clip_series((source_forward - 0.025) / 0.075, 0.0, 1.0)
    predicted_strong_false_sell_penalty = _clip_series(
        _series(working, "portfolio_daily_source_strong_false_sell_penalty"),
        0.0,
        1.0,
    )
    strong_false_raw = pd.Series(
        np.maximum.reduce(
            [
                _clip_series((source_forward - 0.055) / 0.075, 0.0, 1.0).to_numpy(dtype=float),
                predicted_strong_false_sell_penalty.to_numpy(dtype=float),
                _clip_series((positive_forward_penalty - 0.55) / 0.35, 0.0, 1.0).to_numpy(dtype=float),
            ]
        ),
        index=working.index,
        dtype=float,
    )
    strong_false_sell_penalty = _clip_series(
        strong_false_raw,
        0.0,
        1.0,
    )
    tail_false_sell_penalty = _clip_series((source_forward - 0.100) / 0.080, 0.0, 1.0)
    predicted_hard_negative_penalty = _clip_series(
        _series(working, "portfolio_daily_source_hard_negative_penalty"),
        0.0,
        1.0,
    )
    hard_negative_penalty = _clip_series(
        np.maximum(
            predicted_hard_negative_penalty,
            0.56 * strong_false_sell_penalty
            + 0.30 * tail_false_sell_penalty
            + 0.16 * opportunity_cost_penalty
            + 0.12 * forward_strength_brake
            + 0.10 * forward_proxy_keep_risk
            + 0.14 * positive_forward_penalty
            + 0.08 * bad_spread_risk,
        ),
        0.0,
        1.0,
    )
    positive_forward_penalty = _clip_series(
        np.maximum(
            positive_forward_penalty,
            0.54 * strong_positive_forward_penalty
            + 0.24 * hard_negative_penalty
            + 0.20 * forward_strength_brake
            + 0.16 * forward_proxy_keep_risk
            + 0.10 * bad_spread_risk,
        ),
        0.0,
        1.0,
    )
    source_distribution_spread_reward = _clip_series(
        receiver_source_spread_reward
        * (1.0 - 0.62 * strong_positive_forward_penalty)
        * (1.0 - 0.72 * hard_negative_penalty),
        0.0,
        1.0,
    )
    source_distribution_quality = _clip_series(
        0.30 * source_distribution_spread_reward
        + 0.22 * source_release_quality
        + 0.18 * economic_release
        + 0.12 * source_exec
        + 0.08 * source_capacity
        + 0.10 * np.clip(-source_forward / 0.08, 0.0, 1.0)
        - 0.28 * positive_forward_penalty
        - 0.24 * strong_positive_forward_penalty
        - 0.38 * hard_negative_penalty
        - 0.22 * opportunity_cost_penalty
        - 0.16 * bad_spread_risk
        - 0.14 * forward_strength_brake
        - 0.10 * economic_block_risk,
        0.0,
        1.0,
    )
    source_release_preference = _clip_series(
        0.26 * source_distribution_quality
        + 0.18 * source_release_quality
        + 0.16 * economic_release
        + 0.14 * source_exec
        + 0.10 * source_capacity
        + 0.18 * source_distribution_spread_reward
        + 0.12 * np.clip(-source_forward / 0.08, 0.0, 1.0)
        + 0.08 * transfer_score
        - 0.48 * hard_negative_penalty
        - 0.34 * positive_forward_penalty
        - 0.26 * opportunity_cost_penalty
        - 0.22 * strong_false_sell_penalty
        - 0.18 * tail_false_sell_penalty
        - 0.16 * forward_proxy_keep_risk
        - 0.14 * bad_spread_risk,
        0.0,
        1.0,
    )

    unified_receiver = _clip_series(
        receiver_raw * (0.36 + 0.42 * receiver_exec + 0.22 * receiver_headroom)
        + receiver_source_spread_reward * 0.10
        + transfer_score * 0.08
        - cash_raw * 0.04,
        0.0,
        1.0,
    )
    unified_receiver = unified_receiver.where(receiver_mask & (receiver_exec > 0.05) & (receiver_headroom > 1.0e-6), 0.0)

    unified_source = _clip_series(
        source_raw * 0.52
        + source_exec * 0.12
        + source_capacity * 0.10
        + source_release_quality * 0.10
        + economic_release * 0.12
        + source_distribution_quality * 0.14
        + source_release_preference * 0.18
        + source_distribution_spread_reward * 0.30
        - positive_forward_penalty * 0.58
        - strong_positive_forward_penalty * 0.36
        - hard_negative_penalty * 0.82
        - opportunity_cost_penalty * 0.44
        - bad_spread_risk * 0.18
        - economic_block_risk * 0.16
        - forward_strength_brake * 0.24
        - forward_proxy_keep_risk * 0.16,
        0.0,
        1.0,
    )
    unified_source = unified_source.where(
        source_mask
        & (source_exec > 0.05)
        & (current_weight > 1.0e-8)
        & (hard_negative_penalty < 0.72),
        0.0,
    )

    deploy_competition = _clip_series(
        0.34 * unified_receiver
        + 0.20 * source_distribution_spread_reward
        + 0.18 * transfer_score
        + 0.14 * receiver_raw
        + 0.14 * source_distribution_quality,
        0.0,
        1.0,
    )
    receiver_realized_deploy_proxy = _clip_series(
        receiver_mask.astype(float) * receiver_exec * receiver_headroom,
        0.0,
        1.0,
    )
    source_realized_release_proxy = _clip_series(
        source_mask.astype(float)
        * source_exec
        * source_capacity
        * source_release_preference
        * (1.0 - hard_negative_penalty),
        0.0,
        1.0,
    )
    receiver_forward_value = _clip_series(receiver_forward / 0.08, 0.0, 1.0)
    source_forward_release_value = _clip_series(-source_forward / 0.08, 0.0, 1.0)
    cash_timing_target = _clip_series(
        0.44 * risk_off_pressure
        + 0.50 * benchmark_downside_timing
        + 0.10 * positive_forward_penalty
        + 0.06 * opportunity_cost_penalty
        - 0.28 * benchmark_upside_timing
        - 0.18 * deploy_competition,
        0.0,
        1.0,
    )
    allocation_cash_deployment_target = _clip_series(
        0.28 * deploy_competition
        + 0.22 * portfolio_cash_pressure * (1.0 - risk_off_pressure)
        + 0.16 * receiver_realized_deploy_proxy
        + 0.14 * benchmark_upside_timing
        + 0.12 * receiver_forward_value
        + 0.10 * source_release_preference
        + 0.08 * source_realized_release_proxy
        - 0.34 * risk_off_pressure
        - 0.16 * dead_branch_risk,
        0.0,
        1.0,
    )
    unified_cash = _clip_series(
        cash_raw * 0.30
        + cash_timing_target * 0.50
        + dead_branch_risk * 0.10
        + positive_forward_penalty * 0.04
        + opportunity_cost_penalty * 0.03
        - deploy_competition * 0.20
        - allocation_cash_deployment_target * 0.18
        - transfer_score * 0.05,
        0.0,
        1.0,
    )
    defensive_cash_alignment = _clip_series(risk_off_pressure * unified_cash, 0.0, 1.0)
    deploy_cash_alignment = _clip_series((1.0 - risk_off_pressure) * (1.0 - unified_cash) * deploy_competition, 0.0, 1.0)
    transfer_regret_target = _clip_series(
        0.30 * unified_receiver
        + 0.24 * source_release_preference
        + 0.16 * source_distribution_spread_reward
        + 0.14 * transfer_score
        + 0.10 * deploy_cash_alignment
        + 0.06 * defensive_cash_alignment
        - 0.34 * hard_negative_penalty
        - 0.24 * positive_forward_penalty
        - 0.20 * opportunity_cost_penalty
        - 0.14 * tail_false_sell_penalty
        - 0.10 * dead_branch_risk,
        0.0,
        1.0,
    )
    allocation_risk_adjusted_return_target = _clip_series(
        0.24 * receiver_forward_value
        + 0.20 * receiver_source_spread_reward
        + 0.16 * source_forward_release_value
        + 0.14 * benchmark_upside_timing
        + 0.12 * unified_receiver
        + 0.08 * unified_source
        + 0.06 * transfer_score
        - 0.22 * risk_off_pressure
        - 0.20 * hard_negative_penalty
        - 0.16 * positive_forward_penalty
        - 0.14 * opportunity_cost_penalty
        - 0.10 * dead_branch_risk,
        0.0,
        1.0,
    )
    allocation_trade_quality_target = _clip_series(
        0.24 * receiver_realized_deploy_proxy
        + 0.20 * source_realized_release_proxy
        + 0.18 * receiver_source_spread_reward
        + 0.14 * source_release_preference
        + 0.10 * allocation_risk_adjusted_return_target
        + 0.08 * transfer_regret_target
        + 0.06 * transfer_score
        - 0.22 * hard_negative_penalty
        - 0.18 * positive_forward_penalty
        - 0.16 * opportunity_cost_penalty
        - 0.12 * dead_branch_risk,
        0.0,
        1.0,
    )
    allocation_drawdown_control_target = _clip_series(
        0.34 * risk_off_pressure
        + 0.24 * benchmark_downside_timing
        + 0.16 * drawdown_pressure
        + 0.12 * cash_defense_value
        + 0.08 * multi_horizon_forward_risk
        + 0.06 * dead_branch_risk
        - 0.18 * allocation_cash_deployment_target
        - 0.10 * benchmark_upside_timing,
        0.0,
        1.0,
    )
    allocation_monthly_quality_target = _clip_series(
        0.24 * allocation_risk_adjusted_return_target
        + 0.22 * allocation_trade_quality_target
        + 0.18 * benchmark_upside_timing
        + 0.14 * receiver_source_spread_reward
        + 0.12 * deploy_cash_alignment
        + 0.10 * (1.0 - drawdown_pressure)
        - 0.20 * hard_negative_penalty
        - 0.14 * positive_forward_penalty
        - 0.12 * opportunity_cost_penalty
        - 0.10 * dead_branch_risk,
        0.0,
        1.0,
    )
    objective = _clip_series(
        unified_receiver * 0.34
        + unified_source * 0.28
        + source_release_preference * 0.12
        + source_distribution_spread_reward * 0.16
        + transfer_score * 0.10
        + transfer_regret_target * 0.10
        + deploy_cash_alignment * 0.10
        + defensive_cash_alignment * 0.06
        - positive_forward_penalty * 0.22
        - strong_positive_forward_penalty * 0.16
        - hard_negative_penalty * 0.24
        - opportunity_cost_penalty * 0.18
        - dead_branch_risk * 0.08,
        0.0,
        1.0,
    )
    allocation_final_objective = _clip_series(
        0.22 * objective
        + 0.20 * allocation_trade_quality_target
        + 0.18 * allocation_risk_adjusted_return_target
        + 0.16 * allocation_cash_deployment_target
        + 0.12 * allocation_monthly_quality_target
        + 0.08 * transfer_regret_target
        + 0.04 * (1.0 - allocation_drawdown_control_target)
        - 0.18 * hard_negative_penalty
        - 0.12 * positive_forward_penalty
        - 0.10 * opportunity_cost_penalty
        - 0.08 * dead_branch_risk,
        0.0,
        1.0,
    )
    computed_uncertainty_pressure_target = _clip_series(
        0.26 * multi_horizon_forward_risk
        + 0.22 * market_downside
        + 0.18 * drawdown_pressure
        + 0.12 * dead_branch_risk
        + 0.10 * bad_spread_risk
        + 0.08 * economic_block_risk
        + 0.04 * cash_regime
        - 0.12 * receiver_source_spread_reward,
        0.0,
        1.0,
    )
    predicted_uncertainty_pressure_target = (
        _clip_series(_series(working, "portfolio_daily_allocation_uncertainty_pressure_target"), 0.0, 1.0)
        if "portfolio_daily_allocation_uncertainty_pressure_target" in working.columns
        else computed_uncertainty_pressure_target
    )
    allocation_uncertainty_pressure_target = _clip_series(
        pd.Series(
            np.maximum(
                computed_uncertainty_pressure_target.to_numpy(dtype=float),
                predicted_uncertainty_pressure_target.to_numpy(dtype=float),
            ),
            index=working.index,
        ),
        0.0,
        1.0,
    )
    computed_tail_risk_control_target = _clip_series(
        0.30 * drawdown_pressure
        + 0.24 * benchmark_downside_timing
        + 0.18 * multi_horizon_forward_risk
        + 0.12 * market_downside
        + 0.08 * hard_negative_penalty
        + 0.05 * tail_false_sell_penalty
        + 0.03 * cash_defense_value
        - 0.10 * benchmark_upside_timing
        - 0.06 * receiver_source_spread_reward,
        0.0,
        1.0,
    )
    predicted_tail_risk_control_target = (
        _clip_series(_series(working, "portfolio_daily_allocation_tail_risk_control_target"), 0.0, 1.0)
        if "portfolio_daily_allocation_tail_risk_control_target" in working.columns
        else computed_tail_risk_control_target
    )
    allocation_tail_risk_control_target = _clip_series(
        pd.Series(
            np.maximum(
                computed_tail_risk_control_target.to_numpy(dtype=float),
                predicted_tail_risk_control_target.to_numpy(dtype=float),
            ),
            index=working.index,
        ),
        0.0,
        1.0,
    )
    computed_decision_focused_objective = _clip_series(
        0.24 * allocation_final_objective
        + 0.18 * allocation_trade_quality_target
        + 0.16 * allocation_risk_adjusted_return_target
        + 0.14 * allocation_cash_deployment_target
        + 0.10 * source_release_preference
        + 0.08 * receiver_source_spread_reward
        + 0.06 * transfer_score
        + 0.05 * receiver_realized_deploy_proxy
        + 0.04 * source_realized_release_proxy
        - 0.20 * allocation_uncertainty_pressure_target
        - 0.18 * allocation_tail_risk_control_target
        - 0.12 * hard_negative_penalty
        - 0.08 * positive_forward_penalty,
        0.0,
        1.0,
    )
    predicted_decision_focused_objective = (
        _clip_series(_series(working, "portfolio_daily_allocation_decision_focused_objective"), 0.0, 1.0)
        if "portfolio_daily_allocation_decision_focused_objective" in working.columns
        else computed_decision_focused_objective
    )
    allocation_decision_focused_objective = _clip_series(
        0.68 * computed_decision_focused_objective + 0.32 * predicted_decision_focused_objective,
        0.0,
        1.0,
    )

    working["portfolio_daily_unified_receiver_score"] = unified_receiver
    working["portfolio_daily_unified_source_score"] = unified_source
    working["portfolio_daily_unified_cash_score"] = unified_cash
    working["portfolio_daily_source_positive_forward_penalty"] = positive_forward_penalty
    working["portfolio_daily_source_opportunity_cost_penalty"] = opportunity_cost_penalty
    working["portfolio_daily_source_strong_false_sell_penalty"] = strong_false_sell_penalty
    working["portfolio_daily_source_hard_negative_penalty"] = hard_negative_penalty
    working["portfolio_daily_source_tail_false_sell_penalty"] = tail_false_sell_penalty
    working["portfolio_daily_source_release_preference"] = source_release_preference
    working["portfolio_daily_receiver_source_spread_reward"] = receiver_source_spread_reward
    working["portfolio_daily_transfer_regret_target"] = transfer_regret_target
    working["portfolio_daily_unified_allocation_objective"] = objective
    working["portfolio_daily_allocation_trade_quality_target"] = allocation_trade_quality_target
    working["portfolio_daily_allocation_cash_deployment_target"] = allocation_cash_deployment_target
    working["portfolio_daily_allocation_risk_adjusted_return_target"] = allocation_risk_adjusted_return_target
    working["portfolio_daily_allocation_drawdown_control_target"] = allocation_drawdown_control_target
    working["portfolio_daily_allocation_monthly_quality_target"] = allocation_monthly_quality_target
    working["portfolio_daily_allocation_final_objective"] = allocation_final_objective
    working["portfolio_daily_allocation_uncertainty_pressure_target"] = allocation_uncertainty_pressure_target
    working["portfolio_daily_allocation_tail_risk_control_target"] = allocation_tail_risk_control_target
    working["portfolio_daily_allocation_decision_focused_objective"] = allocation_decision_focused_objective
    working["portfolio_daily_unified_receiver_candidate"] = (
        (unified_receiver > 0.0) & receiver_mask & (receiver_exec > 0.05)
    ).astype(float)
    working["portfolio_daily_unified_source_candidate"] = (
        (unified_source > 0.0)
        & source_mask
        & (source_exec > 0.05)
        & (current_weight > 1.0e-8)
        & (hard_negative_penalty < 0.72)
    ).astype(float)
    return working


def attach_unified_allocation_targets(label_frame: pd.DataFrame, *, replace_core_scores: bool = True) -> pd.DataFrame:
    working = build_unified_allocation_problem(label_frame)
    if replace_core_scores and not working.empty:
        working["portfolio_daily_receiver_score"] = working["portfolio_daily_unified_receiver_score"]
        working["portfolio_daily_source_score"] = working["portfolio_daily_unified_source_score"]
        working["portfolio_daily_cash_score"] = working["portfolio_daily_unified_cash_score"]
    return working


def _allocate_capped_budget(scores: pd.Series, capacities: pd.Series, budget: float) -> pd.Series:
    scores = _clip_series(scores, 0.0, None)
    capacities = _clip_series(capacities, 0.0, None)
    budget = float(max(budget, 0.0))
    allocation = pd.Series(0.0, index=scores.index, dtype=float)
    if budget <= 1.0e-12:
        return allocation
    remaining = budget
    active = (scores > 0.0) & (capacities > 0.0)
    while remaining > 1.0e-12 and bool(active.any()):
        local_scores = scores.where(active, 0.0)
        score_sum = float(local_scores.sum())
        if score_sum <= 1.0e-12:
            local_scores = capacities.where(active, 0.0)
            score_sum = float(local_scores.sum())
        if score_sum <= 1.0e-12:
            break
        proposed = local_scores / score_sum * remaining
        capped = pd.concat([proposed, capacities - allocation], axis=1).min(axis=1).clip(lower=0.0)
        allocation = allocation + capped.where(active, 0.0)
        used = float(capped.where(active, 0.0).sum())
        if used <= 1.0e-12:
            break
        remaining = max(0.0, remaining - used)
        active = active & ((capacities - allocation) > 1.0e-9)
    return allocation


def solve_semidifferentiable_allocation(
    allocation_problem: pd.DataFrame,
    *,
    constraints: AllocationOptimizerConstraints | None = None,
) -> AllocationSolution:
    constraints = constraints or AllocationOptimizerConstraints()
    problem = build_unified_allocation_problem(allocation_problem)
    if problem.empty:
        return AllocationSolution(
            target_weight=pd.Series(dtype=float),
            cash_after=1.0,
            expected_turnover=0.0,
            buy_turnover=0.0,
            sell_turnover=0.0,
            available_cash_to_deploy=0.0,
            allocation_objective_value=0.0,
            diagnostics={"constraint_violations": 0.0},
        )

    index = problem["stock"].astype(str) if "stock" in problem.columns else problem.index.astype(str)
    current = _clip_series(_series(problem, "current_weight"), 0.0, float(constraints.max_position_weight))
    current.index = index
    problem = problem.copy()
    problem.index = index

    max_position_weight = float(max(constraints.max_position_weight, 1.0e-8))
    turnover_limit = float(max(constraints.turnover_limit, 0.0))
    cash_reserve_target = float(np.clip(constraints.cash_reserve_target, 0.0, 1.0))
    current_cash = float(max(0.0, 1.0 - current.sum()))
    available_cash_to_deploy = float(max(0.0, current_cash - cash_reserve_target))
    uncertainty_pressure = _clip_series(_series(problem, "portfolio_daily_allocation_uncertainty_pressure_target"), 0.0, 1.0)
    tail_risk_control = _clip_series(_series(problem, "portfolio_daily_allocation_tail_risk_control_target"), 0.0, 1.0)
    decision_objective = _clip_series(_series(problem, "portfolio_daily_allocation_decision_focused_objective"), 0.0, 1.0)

    source_scores = _clip_series(
        0.46 * problem["portfolio_daily_unified_source_score"].astype(float)
        + 0.16 * problem["portfolio_daily_allocation_trade_quality_target"].astype(float)
        + 0.16 * decision_objective
        + 0.08 * problem["portfolio_daily_allocation_final_objective"].astype(float)
        + 0.06 * problem["portfolio_daily_allocation_risk_adjusted_return_target"].astype(float)
        + 0.04 * tail_risk_control * (1.0 - problem["portfolio_daily_source_hard_negative_penalty"].astype(float))
        - 0.10 * uncertainty_pressure
        - 0.16 * problem["portfolio_daily_source_hard_negative_penalty"].astype(float),
        0.0,
        1.0,
    )
    source_candidates = _series(problem, "portfolio_daily_unified_source_candidate") > 0.5
    source_capacity = current.where(source_candidates, 0.0).clip(lower=0.0)
    sell_budget = min(float(source_capacity.sum()), turnover_limit * 0.5)
    sells = _allocate_capped_budget(source_scores.where(source_candidates, 0.0), source_capacity, sell_budget)
    target = (current - sells).clip(lower=0.0, upper=max_position_weight)
    sell_turnover = float((current - target).clip(lower=0.0).sum())

    receiver_scores = _clip_series(
        0.50 * problem["portfolio_daily_unified_receiver_score"].astype(float)
        + 0.16 * problem["portfolio_daily_allocation_cash_deployment_target"].astype(float)
        + 0.18 * decision_objective
        + 0.08 * problem["portfolio_daily_allocation_final_objective"].astype(float)
        + 0.08 * problem["portfolio_daily_allocation_risk_adjusted_return_target"].astype(float)
        + 0.04 * problem["portfolio_daily_allocation_monthly_quality_target"].astype(float)
        - 0.24 * uncertainty_pressure
        - 0.20 * tail_risk_control,
        0.0,
        1.0,
    )
    receiver_candidates = _series(problem, "portfolio_daily_unified_receiver_candidate") > 0.5
    receiver_headroom = _clip_series(
        _coalesce_series(
            problem,
            (
                "portfolio_daily_receiver_add_headroom",
                "portfolio_daily_receiver_add_capacity",
                "portfolio_daily_receiver_min_add_delta",
            ),
            default=1.0,
        ),
        0.0,
        1.0,
    )
    receiver_headroom.index = index
    receiver_headroom = receiver_headroom.where(current > 1.0e-8, max_position_weight)
    buy_capacity = pd.concat(
        [
            (max_position_weight - target).clip(lower=0.0),
            receiver_headroom,
        ],
        axis=1,
    ).min(axis=1)
    buy_capacity = buy_capacity.where(receiver_candidates, 0.0)
    remaining_turnover = max(0.0, turnover_limit - sell_turnover)
    cost_buffer = (float(constraints.transaction_cost_bps) + float(constraints.slippage_bps)) / 10_000.0
    buy_budget = min(
        float(buy_capacity.sum()),
        remaining_turnover,
        max(0.0, sell_turnover + available_cash_to_deploy - cost_buffer * max(remaining_turnover, 0.0)),
    )
    if bool(receiver_candidates.any()):
        receiver_risk_brake = float(
            pd.concat(
                [
                    uncertainty_pressure.where(receiver_candidates, np.nan),
                    tail_risk_control.where(receiver_candidates, np.nan),
                ],
                axis=1,
            )
            .max(axis=1)
            .dropna()
            .mean()
        )
    else:
        receiver_risk_brake = float(pd.concat([uncertainty_pressure, tail_risk_control], axis=1).max(axis=1).mean())
    if not np.isfinite(receiver_risk_brake):
        receiver_risk_brake = 0.0
    buy_budget *= float(np.clip(1.0 - 0.82 * receiver_risk_brake, 0.0, 1.0))
    buys = _allocate_capped_budget(receiver_scores.where(receiver_candidates, 0.0), buy_capacity, buy_budget)
    target = (target + buys).clip(lower=0.0, upper=max_position_weight)
    buy_turnover = float((target - (current - sells)).clip(lower=0.0).sum())

    cost = (
        buy_turnover * (float(constraints.transaction_cost_bps) + float(constraints.slippage_bps))
        + sell_turnover * (float(constraints.transaction_cost_bps) + float(constraints.slippage_bps) + float(constraints.sell_tax_bps))
    ) / 10_000.0
    cash_after = float(max(0.0, 1.0 - target.sum() - cost))
    if cash_after + 1.0e-12 < cash_reserve_target and buy_turnover > 1.0e-12:
        shortfall = cash_reserve_target - cash_after
        scale = max(0.0, 1.0 - shortfall / max(buy_turnover, 1.0e-12))
        buys = buys * scale
        target = (current - sells + buys).clip(lower=0.0, upper=max_position_weight)
        buy_turnover = float(buys.sum())
        cost = (
            buy_turnover * (float(constraints.transaction_cost_bps) + float(constraints.slippage_bps))
            + sell_turnover * (float(constraints.transaction_cost_bps) + float(constraints.slippage_bps) + float(constraints.sell_tax_bps))
        ) / 10_000.0
        cash_after = float(max(0.0, 1.0 - target.sum() - cost))

    expected_turnover = float((target - current).abs().sum())
    objective = float(
        (
            problem["portfolio_daily_allocation_final_objective"].astype(float)
            * (sells.reindex(problem.index).fillna(0.0) + buys.reindex(problem.index).fillna(0.0))
        ).sum()
    )
    violations = 0.0
    violations += float(cash_after + 1.0e-9 < cash_reserve_target)
    violations += float(expected_turnover > turnover_limit + 1.0e-9)
    violations += float(target.max() > max_position_weight + 1.0e-9)
    violations += float((target < -1.0e-9).any())
    diagnostics = {
        "constraint_violations": violations,
        "cash_reserve_target": cash_reserve_target,
        "current_cash": current_cash,
        "transaction_cost_estimate": float(cost),
        "source_release_budget": float(sell_budget),
        "receiver_buy_budget": float(buy_budget),
        "max_position_weight": max_position_weight,
        "turnover_limit": turnover_limit,
    }
    return AllocationSolution(
        target_weight=target.astype(float),
        cash_after=cash_after,
        expected_turnover=expected_turnover,
        buy_turnover=buy_turnover,
        sell_turnover=sell_turnover,
        available_cash_to_deploy=available_cash_to_deploy,
        allocation_objective_value=objective,
        diagnostics=diagnostics,
    )


def build_unified_allocation_summary(label_frame: pd.DataFrame) -> dict[str, float]:
    if label_frame.empty:
        return {
            "portfolio_daily_unified_allocation_objective": 0.0,
            "portfolio_daily_unified_receiver_score": 0.0,
            "portfolio_daily_unified_source_score": 0.0,
            "portfolio_daily_unified_cash_score": 0.0,
            "portfolio_daily_source_positive_forward_penalty": 0.0,
            "portfolio_daily_source_opportunity_cost_penalty": 0.0,
            "portfolio_daily_source_strong_false_sell_penalty": 0.0,
            "portfolio_daily_source_hard_negative_penalty": 0.0,
            "portfolio_daily_source_tail_false_sell_penalty": 0.0,
            "portfolio_daily_source_release_preference": 0.0,
            "portfolio_daily_receiver_source_spread_reward": 0.0,
            "portfolio_daily_transfer_regret_target": 0.0,
            "portfolio_daily_allocation_trade_quality_target": 0.0,
            "portfolio_daily_allocation_cash_deployment_target": 0.0,
            "portfolio_daily_allocation_risk_adjusted_return_target": 0.0,
            "portfolio_daily_allocation_drawdown_control_target": 0.0,
            "portfolio_daily_allocation_monthly_quality_target": 0.0,
            "portfolio_daily_allocation_final_objective": 0.0,
            "portfolio_daily_allocation_uncertainty_pressure_target": 0.0,
            "portfolio_daily_allocation_tail_risk_control_target": 0.0,
            "portfolio_daily_allocation_decision_focused_objective": 0.0,
            "portfolio_daily_source_hard_negative_prevalence": 0.0,
            "portfolio_daily_source_hard_negative_selected_pressure": 0.0,
            "portfolio_daily_unified_receiver_candidate_count": 0.0,
            "portfolio_daily_unified_source_candidate_count": 0.0,
            "portfolio_daily_unified_constraint_violations": 0.0,
        }
    problem = build_unified_allocation_problem(label_frame)
    solution = solve_semidifferentiable_allocation(problem)
    source_candidate = problem["portfolio_daily_unified_source_candidate"].astype(float) > 0.5
    hard_negative = problem["portfolio_daily_source_hard_negative_penalty"].astype(float)
    hard_negative_selected_pressure = (
        float(hard_negative.where(source_candidate, np.nan).mean())
        if bool(source_candidate.any())
        else 0.0
    )
    return {
        "portfolio_daily_unified_allocation_objective": float(problem["portfolio_daily_unified_allocation_objective"].mean()),
        "portfolio_daily_unified_receiver_score": float(problem["portfolio_daily_unified_receiver_score"].mean()),
        "portfolio_daily_unified_source_score": float(problem["portfolio_daily_unified_source_score"].mean()),
        "portfolio_daily_unified_cash_score": float(problem["portfolio_daily_unified_cash_score"].mean()),
        "portfolio_daily_source_positive_forward_penalty": float(problem["portfolio_daily_source_positive_forward_penalty"].mean()),
        "portfolio_daily_source_opportunity_cost_penalty": float(problem["portfolio_daily_source_opportunity_cost_penalty"].mean()),
        "portfolio_daily_source_strong_false_sell_penalty": float(problem["portfolio_daily_source_strong_false_sell_penalty"].mean()),
        "portfolio_daily_source_hard_negative_penalty": float(problem["portfolio_daily_source_hard_negative_penalty"].mean()),
        "portfolio_daily_source_tail_false_sell_penalty": float(problem["portfolio_daily_source_tail_false_sell_penalty"].mean()),
        "portfolio_daily_source_release_preference": float(problem["portfolio_daily_source_release_preference"].mean()),
        "portfolio_daily_receiver_source_spread_reward": float(problem["portfolio_daily_receiver_source_spread_reward"].mean()),
        "portfolio_daily_transfer_regret_target": float(problem["portfolio_daily_transfer_regret_target"].mean()),
        "portfolio_daily_allocation_trade_quality_target": float(problem["portfolio_daily_allocation_trade_quality_target"].mean()),
        "portfolio_daily_allocation_cash_deployment_target": float(problem["portfolio_daily_allocation_cash_deployment_target"].mean()),
        "portfolio_daily_allocation_risk_adjusted_return_target": float(problem["portfolio_daily_allocation_risk_adjusted_return_target"].mean()),
        "portfolio_daily_allocation_drawdown_control_target": float(problem["portfolio_daily_allocation_drawdown_control_target"].mean()),
        "portfolio_daily_allocation_monthly_quality_target": float(problem["portfolio_daily_allocation_monthly_quality_target"].mean()),
        "portfolio_daily_allocation_final_objective": float(problem["portfolio_daily_allocation_final_objective"].mean()),
        "portfolio_daily_allocation_uncertainty_pressure_target": float(problem["portfolio_daily_allocation_uncertainty_pressure_target"].mean()),
        "portfolio_daily_allocation_tail_risk_control_target": float(problem["portfolio_daily_allocation_tail_risk_control_target"].mean()),
        "portfolio_daily_allocation_decision_focused_objective": float(problem["portfolio_daily_allocation_decision_focused_objective"].mean()),
        "portfolio_daily_source_hard_negative_prevalence": float((hard_negative >= 0.55).mean()),
        "portfolio_daily_source_hard_negative_selected_pressure": hard_negative_selected_pressure,
        "portfolio_daily_unified_receiver_candidate_count": float(problem["portfolio_daily_unified_receiver_candidate"].sum()),
        "portfolio_daily_unified_source_candidate_count": float(problem["portfolio_daily_unified_source_candidate"].sum()),
        "portfolio_daily_unified_expected_turnover": float(solution.expected_turnover),
        "portfolio_daily_unified_cash_after": float(solution.cash_after),
        "portfolio_daily_unified_constraint_violations": float(solution.diagnostics.get("constraint_violations", 0.0)),
        "portfolio_daily_unified_solution_objective": float(solution.allocation_objective_value),
    }


def summarize_unified_allocation_rows(rows: list[dict[str, Any]]) -> dict[str, float]:
    if not rows:
        return build_unified_allocation_summary(pd.DataFrame())
    frame = pd.DataFrame(rows).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    return {
        str(column): float(pd.to_numeric(frame[column], errors="coerce").fillna(0.0).mean())
        for column in frame.columns
    }
