from __future__ import annotations

from collections import Counter
from typing import Any

import numpy as np
import pandas as pd


def _numeric(frame: pd.DataFrame, names: tuple[str, ...], default: float = 0.0) -> pd.Series:
    for name in names:
        if name in frame.columns:
            return pd.to_numeric(frame[name], errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(default)
    return pd.Series(float(default), index=frame.index, dtype=float)


def _strings(frame: pd.DataFrame, name: str, default: str = "") -> pd.Series:
    if name not in frame.columns:
        return pd.Series(str(default), index=frame.index, dtype=object)
    return frame[name].fillna(str(default)).astype(str)


def _allocation_metric(allocation_result: Any, key: str, default: float = 0.0) -> float:
    if allocation_result is None:
        return float(default)
    if isinstance(allocation_result, dict):
        if key in allocation_result:
            return float(allocation_result.get(key, default) or default)
        diagnostics = allocation_result.get("diagnostics", {})
        if isinstance(diagnostics, dict) and key in diagnostics:
            return float(diagnostics.get(key, default) or default)
        return float(default)
    diagnostics = getattr(allocation_result, "diagnostics", {})
    if isinstance(diagnostics, dict) and key in diagnostics:
        return float(diagnostics.get(key, default) or default)
    if hasattr(allocation_result, key):
        return float(getattr(allocation_result, key) or default)
    return float(default)


def build_release_flow_trace(
    policy_frame: pd.DataFrame,
    allocation_problem: pd.DataFrame | None = None,
    allocation_result: Any | None = None,
    *,
    deadband: float = 0.003,
    release_threshold: float = 0.30,
) -> dict[str, Any]:
    frame = (allocation_problem if allocation_problem is not None else policy_frame).copy()
    if len(frame.index) == 0:
        return {
            "held_count": 0,
            "source_executable_count": 0,
            "receiver_executable_count": 0,
            "held_negative_delta_count": 0,
            "receiver_positive_delta_count": 0,
            "release_score_above_threshold_count": 0,
            "release_action_hint_count": 0,
            "source_intent_without_realization_count": 0,
            "receiver_score_dead_count": 0,
            "target_delta_weight_conflict_count": 0,
            "release_block_reason_counts": {"empty": 1},
            "primary_blocker": "no_held_source",
        }

    current = _numeric(frame, ("current_weight", "weight", "previous_weight"), 0.0).clip(lower=0.0)
    target_delta = _numeric(frame, ("portfolio_daily_target_delta_intent", "target_delta_hint"), 0.0)
    target_weight = _numeric(frame, ("portfolio_daily_target_weight_intent", "portfolio_daily_target_weight"), np.nan)
    target_weight = target_weight.where(target_weight.notna(), current + target_delta)
    source_executable = _numeric(
        frame,
        ("portfolio_daily_source_executable_candidate", "source_executable", "source"),
        0.0,
    ) > 0.5
    receiver_executable = _numeric(
        frame,
        ("portfolio_daily_receiver_executable_candidate", "receiver_executable", "receiver"),
        0.0,
    ) > 0.5
    receiver_score = _numeric(
        frame,
        ("portfolio_daily_receiver_score", "portfolio_daily_unified_receiver_score", "receiver_score"),
        0.0,
    ).clip(0.0, 1.0)
    release_score = _numeric(
        frame,
        ("release_first_intent_score", "portfolio_daily_release_first_intent", "release_intent_score"),
        0.0,
    ).clip(0.0, 1.0)
    cashflow_mode = _numeric(frame, ("portfolio_cashflow_decision_v1_mode",), 0.0) > 0.5
    source_supply = _numeric(frame, ("portfolio_set_v5_source_supply",), 0.0).clip(lower=0.0)
    receiver_demand = _numeric(frame, ("portfolio_set_v5_receiver_demand",), 0.0).clip(lower=0.0)
    r69_mode = _numeric(frame, ("portfolio_set_v5_value_arbitration_mode",), 0.0) > 0.5
    r69_wrong_side = (_numeric(frame, ("portfolio_set_v5_r69_source_wrong_side_sell",), 0.0) > 0.5) & r69_mode
    r69_reversal_guarded = (_numeric(frame, ("portfolio_set_v5_r69_reversal_guarded",), 0.0) > 0.5) & r69_mode
    r69_defense = _numeric(frame, ("portfolio_set_v5_r69_defense_value",), 0.0).clip(lower=0.0)
    r69_cash_timing = _numeric(frame, ("portfolio_set_v5_r69_cash_timing_value",), 0.0).clip(lower=0.0)
    r69_spread = _numeric(frame, ("portfolio_set_v5_r69_receiver_source_spread_value",), 0.0)
    cashflow_source_intent = (
        (_numeric(frame, ("portfolio_daily_source_target_intent",), 0.0) > 0.5)
        | ((source_supply > float(deadband)) & (target_delta < -float(deadband)))
    )
    cashflow_receiver_intent = (
        (_numeric(frame, ("portfolio_daily_receiver_target_intent",), 0.0) > 0.5)
        | ((receiver_demand > float(deadband)) & (target_delta > float(deadband)))
    )
    action_hint = _strings(frame, "release_first_action_hint", "hold").str.lower()
    block_reason = _strings(frame, "release_first_block_reason", "none").replace("", "none")

    held = current > float(deadband)
    held_negative_delta = held & (target_delta < -float(deadband))
    receiver_positive_delta = receiver_executable & (target_delta > float(deadband))
    cashflow_active = bool(cashflow_mode.any())
    if cashflow_active:
        source_executable = source_executable | cashflow_source_intent | (source_supply > float(deadband))
        receiver_executable = receiver_executable | cashflow_receiver_intent | (receiver_demand > float(deadband))
        held_negative_delta = held_negative_delta | cashflow_source_intent
        receiver_positive_delta = receiver_positive_delta | cashflow_receiver_intent
    release_score_active = held & (release_score >= float(release_threshold))
    release_action_active = held & action_hint.isin({"reduce", "exit"})
    target_delta_weight_conflict = (
        ((target_weight - current) * target_delta < -float(deadband) ** 2)
        | ((target_delta < -float(deadband)) & (target_weight > current + float(deadband)))
        | ((target_delta > float(deadband)) & (target_weight < current - float(deadband)))
    )
    source_intent = int(max(_allocation_metric(allocation_result, "release_first_source_intent_count", 0.0), float(release_score_active.sum())))
    if cashflow_active:
        source_intent = int(max(source_intent, int(cashflow_source_intent.sum()), int((source_supply > float(deadband)).sum())))
    source_realized = int(_allocation_metric(allocation_result, "release_first_source_realized_count", 0.0))
    if cashflow_active and allocation_result is None:
        source_realized = int(max(source_realized, int(cashflow_source_intent.sum())))
    source_intent_without_realization = max(0, source_intent - source_realized)
    receiver_dead = int(((receiver_executable | (target_delta > float(deadband))) & (receiver_score <= 0.02)).sum())
    current_cash = max(0.0, 1.0 - float(current.sum()))
    target_cash = max(0.0, 1.0 - float(target_weight.sum()))
    cashflow_cash_conservation_gap = abs(target_cash - (current_cash + float(source_supply.sum()) - float(receiver_demand.sum())))

    blocker = "none"
    if int(held.sum()) == 0:
        blocker = "no_held_source"
    elif int(target_delta_weight_conflict.sum()) > 0:
        blocker = "target_delta_weight_conflict"
    elif source_intent_without_realization > 0 and receiver_dead > 0:
        blocker = "receiver_score_dead"
    elif int(source_executable.sum()) == 0:
        blocker = "source_executable_dead"
    elif int(receiver_executable.sum()) == 0 and source_intent_without_realization > 0:
        blocker = "receiver_executable_dead"
    elif int(held_negative_delta.sum()) == 0 and source_intent == 0:
        blocker = "no_held_negative_delta"

    reason_counts = Counter(block_reason.where(held, "not_held").astype(str))
    return {
        "held_count": int(held.sum()),
        "source_executable_count": int(source_executable.sum()),
        "receiver_executable_count": int(receiver_executable.sum()),
        "held_negative_delta_count": int(held_negative_delta.sum()),
        "receiver_positive_delta_count": int(receiver_positive_delta.sum()),
        "release_score_above_threshold_count": int(release_score_active.sum()),
        "release_action_hint_count": int(release_action_active.sum()),
        "source_intent_without_realization_count": int(source_intent_without_realization),
        "cashflow_decision_mode_count": int(cashflow_mode.sum()),
        "cashflow_decision_source_intent_count": int(cashflow_source_intent.sum()),
        "cashflow_decision_receiver_intent_count": int(cashflow_receiver_intent.sum()),
        "cashflow_decision_source_supply_sum": float(source_supply.sum()),
        "cashflow_decision_receiver_demand_sum": float(receiver_demand.sum()),
        "cashflow_decision_cash_conservation_gap": float(cashflow_cash_conservation_gap),
        "r69_source_wrong_side_sell_count": int((r69_wrong_side & cashflow_source_intent).sum()),
        "r69_reversal_guarded_count": int((r69_reversal_guarded & cashflow_source_intent).sum()),
        "r69_defense_value_mean": float(r69_defense.loc[cashflow_mode & r69_mode].mean()) if bool((cashflow_mode & r69_mode).any()) else 0.0,
        "r69_cash_timing_value_mean": float(r69_cash_timing.loc[cashflow_mode & r69_mode].mean()) if bool((cashflow_mode & r69_mode).any()) else 0.0,
        "r69_receiver_source_spread_value_mean": float(r69_spread.loc[cashflow_mode & r69_mode].mean()) if bool((cashflow_mode & r69_mode).any()) else 0.0,
        "receiver_score_dead_count": int(receiver_dead),
        "target_delta_weight_conflict_count": int(target_delta_weight_conflict.sum()),
        "release_block_reason_counts": dict(reason_counts),
        "primary_blocker": blocker,
    }
