from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class AllocationCoreV2Constraints:
    gross_target: float
    cash_reserve_target: float
    turnover_limit: float
    position_cap: float
    transaction_cost_rate: float = 0.0015
    available_cash_to_deploy: float | None = None
    activity_threshold: float = 1.0e-6


@dataclass
class AllocationCoreV2Result:
    target_weight: pd.Series
    cash_after: float
    buy_turnover: float
    sell_turnover: float
    receiver_target_count: int
    source_target_count: int
    cash_funded_deploy_amount: float
    source_funded_deploy_amount: float
    unused_receiver_headroom: float
    receiver_headroom_utilization: float
    source_release_required: bool
    underdeployment_reason: str
    diagnostics: dict[str, Any] = field(default_factory=dict)


def _numeric(frame: pd.DataFrame, names: tuple[str, ...], default: float = 0.0) -> pd.Series:
    for name in names:
        if name in frame.columns:
            return pd.to_numeric(frame[name], errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(default)
    return pd.Series(default, index=frame.index, dtype=float)


def _mask(frame: pd.DataFrame, names: tuple[str, ...]) -> pd.Series:
    return _numeric(frame, names, 0.0) > 0.5


def _ordered_allocate(
    *,
    amount: float,
    capacity: pd.Series,
    score: pd.Series,
    target_index: pd.Index,
) -> pd.Series:
    allocation = pd.Series(0.0, index=target_index, dtype=float)
    remaining = float(max(0.0, amount))
    active = capacity.clip(lower=0.0).astype(float)
    active = active[active > 1.0e-12]
    if remaining <= 1.0e-12 or active.empty:
        return allocation

    ordered_index = (
        pd.DataFrame(
            {
                "capacity": active,
                "score": score.reindex(active.index).fillna(0.0).astype(float).clip(lower=0.0),
                "stock_key": active.index.astype(str),
            }
        )
        .sort_values(["score", "capacity", "stock_key"], ascending=[False, False, True], kind="mergesort")
        .index
    )
    active = active.reindex(ordered_index)
    weights = score.reindex(ordered_index).fillna(0.0).astype(float).clip(lower=0.0)
    if float(weights.sum()) <= 1.0e-12:
        weights = pd.Series(1.0, index=ordered_index, dtype=float)

    while remaining > 1.0e-12 and not active.empty:
        current_weights = weights.reindex(active.index).fillna(0.0).clip(lower=0.0)
        if float(current_weights.sum()) <= 1.0e-12:
            current_weights = pd.Series(1.0, index=active.index, dtype=float)
        proposed = current_weights / float(current_weights.sum()) * remaining
        assigned = pd.concat([proposed.rename("proposed"), active.rename("capacity")], axis=1).min(axis=1)
        allocation.loc[assigned.index] = allocation.loc[assigned.index] + assigned
        assigned_sum = float(assigned.sum())
        if assigned_sum <= 1.0e-12:
            break
        remaining -= assigned_sum
        active = active - assigned
        active = active[active > 1.0e-12]
    return allocation


def solve_cash_funded_allocation_v2(
    frame: pd.DataFrame,
    constraints: AllocationCoreV2Constraints,
) -> AllocationCoreV2Result:
    """Cash-first deterministic allocation core for r53 research profiles."""
    if frame.empty:
        empty = pd.Series(dtype=float)
        return AllocationCoreV2Result(
            target_weight=empty,
            cash_after=1.0,
            buy_turnover=0.0,
            sell_turnover=0.0,
            receiver_target_count=0,
            source_target_count=0,
            cash_funded_deploy_amount=0.0,
            source_funded_deploy_amount=0.0,
            unused_receiver_headroom=0.0,
            receiver_headroom_utilization=0.0,
            source_release_required=False,
            underdeployment_reason="empty_universe",
            diagnostics={"target_sum_gap": 0.0},
        )

    working = frame.copy()
    if "stock" in working.columns:
        working.index = working["stock"].astype(str)
    working.index = working.index.astype(str)

    current = _numeric(working, ("current_weight", "weight", "previous_weight"), 0.0).clip(lower=0.0)
    position_cap = float(max(0.0, constraints.position_cap))
    gross_target = float(np.clip(constraints.gross_target, 0.0, 1.0))
    cash_reserve_target = float(np.clip(constraints.cash_reserve_target, 0.0, 1.0))
    turnover_limit = float(max(0.0, constraints.turnover_limit))
    transaction_cost_rate = float(max(0.0, constraints.transaction_cost_rate))
    stock_budget = min(gross_target, max(0.0, 1.0 - cash_reserve_target))
    current_gross = float(current.sum())
    inferred_cash_to_deploy = max(0.0, 1.0 - current_gross - cash_reserve_target)
    available_cash = (
        inferred_cash_to_deploy
        if constraints.available_cash_to_deploy is None
        else float(max(0.0, constraints.available_cash_to_deploy))
    )

    receiver_mask = _mask(
        working,
        (
            "receiver",
            "receiver_executable",
            "portfolio_daily_receiver_executable_candidate",
            "allocation_layer_receiver_executable_candidate",
        ),
    )
    source_mask = _mask(
        working,
        (
            "source",
            "source_executable",
            "portfolio_daily_source_executable_candidate",
            "allocation_layer_source_executable_candidate",
        ),
    )
    receiver_score = _numeric(
        working,
        ("receiver_score", "portfolio_daily_unified_receiver_score", "portfolio_daily_receiver_score"),
        0.0,
    )
    source_score = _numeric(
        working,
        ("source_score", "portfolio_daily_unified_source_score", "portfolio_daily_source_score"),
        0.0,
    )

    target = current.clip(lower=0.0, upper=position_cap).astype(float)
    receiver_headroom_initial = (position_cap - target).clip(lower=0.0).where(receiver_mask, 0.0)
    receiver_headroom_total = float(receiver_headroom_initial.sum())
    deployment_gap = max(0.0, stock_budget - float(target.sum()))
    turnover_remaining = turnover_limit

    cost_adjusted_cash = available_cash / max(1.0 + transaction_cost_rate, 1.0)
    cash_deploy_budget = min(
        deployment_gap,
        cost_adjusted_cash,
        receiver_headroom_total,
        turnover_remaining,
    )
    cash_buy = _ordered_allocate(
        amount=cash_deploy_budget,
        capacity=receiver_headroom_initial,
        score=receiver_score,
        target_index=target.index,
    )
    target = (target + cash_buy).clip(lower=0.0, upper=position_cap)
    cash_funded_deploy_amount = float(cash_buy.sum())
    turnover_remaining = max(0.0, turnover_remaining - cash_funded_deploy_amount)

    receiver_headroom_after_cash = (position_cap - target).clip(lower=0.0).where(receiver_mask, 0.0)
    remaining_receiver_demand = max(
        0.0,
        min(
            stock_budget - float(target.sum()),
            float(receiver_headroom_after_cash.sum()),
        ),
    )
    cash_limited = cash_funded_deploy_amount + 1.0e-12 < min(deployment_gap, receiver_headroom_total)
    source_capacity = current.clip(lower=0.0).where(source_mask & (current > constraints.activity_threshold), 0.0)
    source_release_required = bool(remaining_receiver_demand > 1.0e-12 and cash_limited and float(source_capacity.sum()) > 1.0e-12)
    source_funded_deploy_amount = 0.0

    if source_release_required and turnover_remaining > 1.0e-12:
        release_budget = min(
            remaining_receiver_demand,
            float(source_capacity.sum()),
            float(receiver_headroom_after_cash.sum()),
            turnover_remaining / 2.0,
        )
        source_sell = _ordered_allocate(
            amount=release_budget,
            capacity=source_capacity,
            score=source_score,
            target_index=target.index,
        )
        target = (target - source_sell).clip(lower=0.0, upper=position_cap)
        receiver_headroom_after_sell = (position_cap - target).clip(lower=0.0).where(receiver_mask, 0.0)
        source_buy = _ordered_allocate(
            amount=float(source_sell.sum()),
            capacity=receiver_headroom_after_sell,
            score=receiver_score,
            target_index=target.index,
        )
        target = (target + source_buy).clip(lower=0.0, upper=position_cap)
        source_funded_deploy_amount = float(min(source_sell.sum(), source_buy.sum()))
        turnover_remaining = max(0.0, turnover_remaining - float(source_sell.sum()) - float(source_buy.sum()))

    delta = target - current
    buy_turnover = float(delta.clip(lower=0.0).sum())
    sell_turnover = float((-delta.clip(upper=0.0)).sum())
    total_turnover = buy_turnover + sell_turnover
    cash_after = float(max(0.0, 1.0 - float(target.sum()) - transaction_cost_rate * total_turnover))
    receiver_threshold = max(float(constraints.activity_threshold), 1.0e-8)
    source_threshold = max(float(constraints.activity_threshold), 1.0e-8)
    receiver_targets = receiver_mask & (delta > receiver_threshold)
    source_targets = source_mask & (current > constraints.activity_threshold) & (delta < -source_threshold)
    unused_receiver_headroom = float((position_cap - target).clip(lower=0.0).where(receiver_mask, 0.0).sum())
    used_receiver_headroom = float((receiver_headroom_initial - (position_cap - target).clip(lower=0.0).where(receiver_mask, 0.0)).clip(lower=0.0).sum())
    receiver_headroom_utilization = (
        float(used_receiver_headroom / max(receiver_headroom_total, 1.0e-12))
        if receiver_headroom_total > 0.0
        else 0.0
    )
    target_sum = float(target.sum())
    target_sum_gap = max(0.0, stock_budget - target_sum)
    turnover_limited = bool(total_turnover >= turnover_limit - 1.0e-9 and target_sum_gap > 1.0e-8)
    if target_sum_gap <= 1.0e-8:
        underdeployment_reason = "none"
    elif turnover_limited:
        underdeployment_reason = "turnover_limited"
    elif unused_receiver_headroom <= 1.0e-8:
        underdeployment_reason = "receiver_headroom_insufficient"
    elif source_release_required and source_funded_deploy_amount <= 1.0e-8:
        underdeployment_reason = "source_release_failed"
    elif cash_limited:
        underdeployment_reason = "cash_limited"
    else:
        underdeployment_reason = "cash_funded_deployment_failed"

    diagnostics = {
        "stock_budget": stock_budget,
        "current_gross": current_gross,
        "target_weight_sum": target_sum,
        "target_sum_gap": target_sum_gap,
        "deployment_gap": deployment_gap,
        "available_cash_to_deploy": available_cash,
        "cash_funded_deploy_amount": cash_funded_deploy_amount,
        "source_funded_deploy_amount": source_funded_deploy_amount,
        "unused_receiver_headroom": unused_receiver_headroom,
        "receiver_headroom_total": receiver_headroom_total,
        "receiver_headroom_utilization": receiver_headroom_utilization,
        "source_release_required": source_release_required,
        "underdeployment_reason": underdeployment_reason,
        "turnover_limit": turnover_limit,
        "turnover_used": total_turnover,
    }
    return AllocationCoreV2Result(
        target_weight=target.reindex(working.index).fillna(0.0).astype(float),
        cash_after=cash_after,
        buy_turnover=buy_turnover,
        sell_turnover=sell_turnover,
        receiver_target_count=int(receiver_targets.sum()),
        source_target_count=int(source_targets.sum()),
        cash_funded_deploy_amount=cash_funded_deploy_amount,
        source_funded_deploy_amount=source_funded_deploy_amount,
        unused_receiver_headroom=unused_receiver_headroom,
        receiver_headroom_utilization=receiver_headroom_utilization,
        source_release_required=source_release_required,
        underdeployment_reason=underdeployment_reason,
        diagnostics=diagnostics,
    )
