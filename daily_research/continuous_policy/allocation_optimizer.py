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
    "portfolio_daily_receiver_source_spread_reward",
    "portfolio_daily_unified_allocation_objective",
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
    positive_forward_penalty = _clip_series(source_forward / 0.08, 0.0, 1.0)
    forward_proxy_keep_risk = _clip_series(_series(working, "portfolio_daily_source_forward_proxy_keep_risk"), 0.0, 1.0)
    forward_strength_brake = _clip_series(_series(working, "portfolio_daily_source_forward_strength_brake_risk"), 0.0, 1.0)
    opportunity_cost_penalty = _clip_series(_series(working, "portfolio_daily_source_opportunity_cost"), 0.0, 1.0)
    bad_spread_risk = _clip_series(_series(working, "portfolio_daily_source_bad_forward_spread_risk"), 0.0, 1.0)
    economic_block_risk = _clip_series(_series(working, "portfolio_daily_source_economic_block_risk"), 0.0, 1.0)
    economic_release = _clip_series(_series(working, "portfolio_daily_source_economic_release_score"), 0.0, 1.0)
    transfer_score = _clip_series(_series(working, "portfolio_daily_allocation_transfer_score"), 0.0, 1.0)
    dead_branch_risk = _clip_series(_series(working, "portfolio_daily_allocation_dead_branch_risk"), 0.0, 1.0)

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
        + receiver_source_spread_reward * 0.32
        - positive_forward_penalty * 0.45
        - opportunity_cost_penalty * 0.38
        - bad_spread_risk * 0.18
        - economic_block_risk * 0.16
        - forward_strength_brake * 0.20
        - forward_proxy_keep_risk * 0.16,
        0.0,
        1.0,
    )
    unified_source = unified_source.where(source_mask & (source_exec > 0.05) & (current_weight > 1.0e-8), 0.0)

    unified_cash = _clip_series(
        cash_raw * 0.62
        + dead_branch_risk * 0.20
        + positive_forward_penalty * 0.10
        + opportunity_cost_penalty * 0.08
        - transfer_score * 0.08
        - unified_receiver * 0.06,
        0.0,
        1.0,
    )
    objective = _clip_series(
        unified_receiver * 0.34
        + unified_source * 0.34
        + receiver_source_spread_reward * 0.16
        + transfer_score * 0.10
        + (1.0 - unified_cash) * 0.06
        - positive_forward_penalty * 0.16
        - opportunity_cost_penalty * 0.14
        - dead_branch_risk * 0.08,
        0.0,
        1.0,
    )

    working["portfolio_daily_unified_receiver_score"] = unified_receiver
    working["portfolio_daily_unified_source_score"] = unified_source
    working["portfolio_daily_unified_cash_score"] = unified_cash
    working["portfolio_daily_source_positive_forward_penalty"] = positive_forward_penalty
    working["portfolio_daily_source_opportunity_cost_penalty"] = opportunity_cost_penalty
    working["portfolio_daily_receiver_source_spread_reward"] = receiver_source_spread_reward
    working["portfolio_daily_unified_allocation_objective"] = objective
    working["portfolio_daily_unified_receiver_candidate"] = (
        (unified_receiver > 0.0) & receiver_mask & (receiver_exec > 0.05)
    ).astype(float)
    working["portfolio_daily_unified_source_candidate"] = (
        (unified_source > 0.0) & source_mask & (source_exec > 0.05) & (current_weight > 1.0e-8)
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

    source_scores = _clip_series(problem["portfolio_daily_unified_source_score"], 0.0, 1.0)
    source_candidates = _series(problem, "portfolio_daily_unified_source_candidate") > 0.5
    source_capacity = current.where(source_candidates, 0.0).clip(lower=0.0)
    sell_budget = min(float(source_capacity.sum()), turnover_limit * 0.5)
    sells = _allocate_capped_budget(source_scores.where(source_candidates, 0.0), source_capacity, sell_budget)
    target = (current - sells).clip(lower=0.0, upper=max_position_weight)
    sell_turnover = float((current - target).clip(lower=0.0).sum())

    receiver_scores = _clip_series(problem["portfolio_daily_unified_receiver_score"], 0.0, 1.0)
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
            problem["portfolio_daily_unified_allocation_objective"].astype(float)
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
            "portfolio_daily_receiver_source_spread_reward": 0.0,
            "portfolio_daily_unified_receiver_candidate_count": 0.0,
            "portfolio_daily_unified_source_candidate_count": 0.0,
            "portfolio_daily_unified_constraint_violations": 0.0,
        }
    problem = build_unified_allocation_problem(label_frame)
    solution = solve_semidifferentiable_allocation(problem)
    return {
        "portfolio_daily_unified_allocation_objective": float(problem["portfolio_daily_unified_allocation_objective"].mean()),
        "portfolio_daily_unified_receiver_score": float(problem["portfolio_daily_unified_receiver_score"].mean()),
        "portfolio_daily_unified_source_score": float(problem["portfolio_daily_unified_source_score"].mean()),
        "portfolio_daily_unified_cash_score": float(problem["portfolio_daily_unified_cash_score"].mean()),
        "portfolio_daily_source_positive_forward_penalty": float(problem["portfolio_daily_source_positive_forward_penalty"].mean()),
        "portfolio_daily_source_opportunity_cost_penalty": float(problem["portfolio_daily_source_opportunity_cost_penalty"].mean()),
        "portfolio_daily_receiver_source_spread_reward": float(problem["portfolio_daily_receiver_source_spread_reward"].mean()),
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
