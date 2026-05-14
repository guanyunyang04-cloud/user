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
    action_hint = _strings(frame, "release_first_action_hint", "hold").str.lower()
    block_reason = _strings(frame, "release_first_block_reason", "none").replace("", "none")

    held = current > float(deadband)
    held_negative_delta = held & (target_delta < -float(deadband))
    receiver_positive_delta = receiver_executable & (target_delta > float(deadband))
    release_score_active = held & (release_score >= float(release_threshold))
    release_action_active = held & action_hint.isin({"reduce", "exit"})
    target_delta_weight_conflict = (
        ((target_weight - current) * target_delta < -float(deadband) ** 2)
        | ((target_delta < -float(deadband)) & (target_weight > current + float(deadband)))
        | ((target_delta > float(deadband)) & (target_weight < current - float(deadband)))
    )
    source_intent = int(max(_allocation_metric(allocation_result, "release_first_source_intent_count", 0.0), float(release_score_active.sum())))
    source_realized = int(_allocation_metric(allocation_result, "release_first_source_realized_count", 0.0))
    source_intent_without_realization = max(0, source_intent - source_realized)
    receiver_dead = int(((receiver_executable | (target_delta > float(deadband))) & (receiver_score <= 0.02)).sum())

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
        "receiver_score_dead_count": int(receiver_dead),
        "target_delta_weight_conflict_count": int(target_delta_weight_conflict.sum()),
        "release_block_reason_counts": dict(reason_counts),
        "primary_blocker": blocker,
    }
