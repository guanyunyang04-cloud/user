from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class ReleaseFirstAllocationConstraints:
    gross_target: float
    cash_reserve_target: float
    turnover_limit: float
    position_cap: float
    transaction_cost_rate: float = 0.0015
    available_cash_to_deploy: float | None = None
    cash_defense_intent: float = 0.0
    activity_threshold: float = 1.0e-6
    min_release_intent: float = 0.35


@dataclass
class ReleaseFirstAllocationResult:
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


def solve_release_first_allocation_v3(
    frame: pd.DataFrame,
    constraints: ReleaseFirstAllocationConstraints,
) -> ReleaseFirstAllocationResult:
    if frame.empty:
        empty = pd.Series(dtype=float)
        return ReleaseFirstAllocationResult(
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
            diagnostics={
                "release_first_source_intent_count": 0,
                "release_first_source_realized_count": 0,
                "release_first_rotation_amount": 0.0,
                "release_first_cash_buffer_amount": 0.0,
                "release_first_block_reason": "empty_universe",
                "target_sum_gap": 0.0,
                "turnover_used": 0.0,
            },
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
    cash_defense_intent = float(np.clip(constraints.cash_defense_intent, 0.0, 1.0))
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
    release_intent = _numeric(
        working,
        ("release_first_intent_score", "release_intent_score", "allocation_intent_release_score"),
        0.0,
    )
    release_quality = _numeric(
        working,
        ("portfolio_daily_source_release_quality", "reduce_quality", "sell_release_value"),
        0.0,
    )
    keep_risk = _numeric(
        working,
        ("portfolio_daily_source_forward_proxy_keep_risk", "portfolio_daily_source_forward_strength_brake_risk"),
        0.0,
    ).clip(0.0, 1.0)
    block_risk = _numeric(working, ("portfolio_daily_source_economic_block_risk",), 0.0).clip(0.0, 1.0)

    target = current.clip(lower=0.0, upper=position_cap).astype(float)
    held_source = source_mask & (current > float(constraints.activity_threshold))
    source_intent_mask = held_source & (release_intent >= float(constraints.min_release_intent))
    adjusted_source_score = (
        0.52 * release_intent.clip(0.0, 1.0)
        + 0.30 * source_score.clip(0.0, 1.0)
        + 0.18 * release_quality.clip(0.0, 1.0)
        - 0.58 * keep_risk
        - 0.58 * block_risk
    ).clip(lower=0.0)
    source_candidate = source_intent_mask & (keep_risk < 0.78) & (block_risk < 0.78) & (
        adjusted_source_score >= float(constraints.min_release_intent)
    )
    source_capacity = current.clip(lower=0.0).where(source_candidate, 0.0)

    receiver_headroom_initial = (position_cap - target).clip(lower=0.0).where(receiver_mask, 0.0)
    receiver_headroom_total = float(receiver_headroom_initial.sum())
    receiver_opportunity = float(receiver_score.where(receiver_mask, 0.0).clip(lower=0.0).max()) if receiver_mask.any() else 0.0
    deployment_gap = max(0.0, stock_budget - float(target.sum()))
    release_rotation_demand = (
        min(float(receiver_headroom_initial.sum()), max(float(source_capacity.sum()), 0.0))
        if receiver_opportunity > 0.02
        else 0.0
    )
    cash_buffer_demand = float(source_capacity.sum()) * min(cash_defense_intent * 0.45, 0.60)
    remaining_receiver_demand = max(0.0, min(deployment_gap, float(receiver_headroom_initial.sum())))
    release_demand = max(release_rotation_demand, cash_buffer_demand, remaining_receiver_demand)

    release_budget = min(release_demand, float(source_capacity.sum()), turnover_limit)
    source_sell = _ordered_allocate(
        amount=release_budget,
        capacity=source_capacity,
        score=adjusted_source_score,
        target_index=target.index,
    )
    target = (target - source_sell).clip(lower=0.0, upper=position_cap)
    release_amount = float(source_sell.sum())
    turnover_remaining = max(0.0, turnover_limit - release_amount)

    receiver_headroom_after_release = (position_cap - target).clip(lower=0.0).where(receiver_mask, 0.0)
    cash_buffer_amount = (
        release_amount
        if cash_defense_intent >= 0.75 and deployment_gap <= 1.0e-8
        else min(release_amount, release_amount * min(cash_defense_intent * 0.75, 0.80))
    )
    rotation_budget = min(
        max(0.0, release_amount - cash_buffer_amount),
        float(receiver_headroom_after_release.sum()),
        turnover_remaining,
    )
    source_funded_buy = _ordered_allocate(
        amount=rotation_budget,
        capacity=receiver_headroom_after_release,
        score=receiver_score,
        target_index=target.index,
    )
    target = (target + source_funded_buy).clip(lower=0.0, upper=position_cap)
    source_funded_deploy_amount = float(source_funded_buy.sum())
    release_cash_buffer_amount = float(max(0.0, release_amount - source_funded_deploy_amount))
    turnover_remaining = max(0.0, turnover_remaining - source_funded_deploy_amount)

    receiver_headroom_after_rotation = (position_cap - target).clip(lower=0.0).where(receiver_mask, 0.0)
    target_sum_after_rotation = float(target.sum())
    cash_deployment_gap = max(0.0, stock_budget - target_sum_after_rotation)
    cost_adjusted_cash = available_cash / max(1.0 + transaction_cost_rate, 1.0)
    cash_deploy_budget = (
        0.0
        if cash_defense_intent >= 0.75
        else min(
            cash_deployment_gap,
            cost_adjusted_cash,
            float(receiver_headroom_after_rotation.sum()),
            turnover_remaining,
        )
    )
    cash_buy = _ordered_allocate(
        amount=cash_deploy_budget,
        capacity=receiver_headroom_after_rotation,
        score=receiver_score,
        target_index=target.index,
    )
    target = (target + cash_buy).clip(lower=0.0, upper=position_cap)
    cash_funded_deploy_amount = float(cash_buy.sum())

    delta = target - current
    buy_turnover = float(delta.clip(lower=0.0).sum())
    sell_turnover = float((-delta.clip(upper=0.0)).sum())
    total_turnover = buy_turnover + sell_turnover
    cash_after = float(max(0.0, 1.0 - float(target.sum()) - transaction_cost_rate * total_turnover))
    receiver_threshold = max(float(constraints.activity_threshold), 1.0e-8)
    source_threshold = max(float(constraints.activity_threshold), 1.0e-8)
    receiver_targets = receiver_mask & (delta > receiver_threshold)
    source_targets = held_source & (delta < -source_threshold)
    unused_receiver_headroom = float((position_cap - target).clip(lower=0.0).where(receiver_mask, 0.0).sum())
    used_receiver_headroom = float(
        (receiver_headroom_initial - (position_cap - target).clip(lower=0.0).where(receiver_mask, 0.0))
        .clip(lower=0.0)
        .sum()
    )
    receiver_headroom_utilization = (
        float(used_receiver_headroom / max(receiver_headroom_total, 1.0e-12))
        if receiver_headroom_total > 0.0
        else 0.0
    )
    target_sum = float(target.sum())
    target_sum_gap = max(0.0, stock_budget - target_sum)
    source_release_required = bool(int(source_intent_mask.sum()) > 0)
    if int(source_intent_mask.sum()) > 0 and int(source_candidate.sum()) == 0:
        release_first_block_reason = "source_blocked_by_keep_or_economic_risk"
    elif int(held_source.sum()) == 0:
        release_first_block_reason = "no_held_source"
    elif release_amount <= 1.0e-12 and turnover_limit <= 1.0e-12:
        release_first_block_reason = "turnover_limited"
    elif release_amount <= 1.0e-12 and receiver_opportunity <= 0.02 and cash_defense_intent < 0.20:
        release_first_block_reason = "no_receiver_or_cash_defense"
    else:
        release_first_block_reason = "none"

    if target_sum_gap <= 1.0e-8:
        underdeployment_reason = "none"
    elif total_turnover >= turnover_limit - 1.0e-9 and target_sum_gap > 1.0e-8:
        underdeployment_reason = "turnover_limited"
    elif unused_receiver_headroom <= 1.0e-8 and target_sum_gap > 1.0e-8:
        underdeployment_reason = "receiver_headroom_insufficient"
    elif source_release_required and release_amount <= 1.0e-8:
        underdeployment_reason = "source_release_failed"
    elif cash_funded_deploy_amount <= 1.0e-8 and target_sum_gap > 1.0e-8:
        underdeployment_reason = "cash_limited"
    else:
        underdeployment_reason = "none"

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
        "release_first_source_intent_count": int(source_intent_mask.sum()),
        "release_first_source_realized_count": int(source_targets.sum()),
        "release_first_rotation_amount": float(source_funded_deploy_amount),
        "release_first_cash_buffer_amount": float(release_cash_buffer_amount),
        "release_first_block_reason": release_first_block_reason,
    }
    return ReleaseFirstAllocationResult(
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
