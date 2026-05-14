from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd


PORTFOLIO_CASHFLOW_DECISION_MODE_COLUMN = "portfolio_cashflow_decision_v1_mode"
PORTFOLIO_CASHFLOW_DECISION_REQUIRED_COLUMNS: tuple[str, ...] = (
    "current_weight",
    "portfolio_daily_target_weight_intent",
    "portfolio_daily_target_delta_intent",
    "portfolio_set_v5_source_supply",
    "portfolio_set_v5_receiver_demand",
    "portfolio_set_v5_cash_buffer_score",
)


@dataclass(frozen=True)
class PortfolioCashflowDecision:
    frame: pd.DataFrame
    target_weight: pd.Series
    target_delta: pd.Series
    source_supply: pd.Series
    receiver_demand: pd.Series
    source_candidate: pd.Series
    receiver_candidate: pd.Series
    source_target: pd.Series
    receiver_target: pd.Series
    action_label: pd.Series
    valid: bool
    diagnostics: dict[str, Any]


def _numeric(frame: pd.DataFrame, name: str, default: float = 0.0) -> pd.Series:
    if name not in frame.columns:
        return pd.Series(float(default), index=frame.index, dtype=float)
    return pd.to_numeric(frame[name], errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(float(default)).astype(float)


def _optional_numeric(frame: pd.DataFrame, name: str) -> pd.Series | None:
    if name not in frame.columns:
        return None
    return pd.to_numeric(frame[name], errors="coerce").replace([np.inf, -np.inf], np.nan).astype(float)


def portfolio_cashflow_decision_enabled(
    policy_frame: pd.DataFrame,
    global_targets: dict[str, Any] | None = None,
) -> bool:
    global_enabled = float((global_targets or {}).get(PORTFOLIO_CASHFLOW_DECISION_MODE_COLUMN, 0.0) or 0.0) > 0.5
    if global_enabled:
        return True
    if PORTFOLIO_CASHFLOW_DECISION_MODE_COLUMN not in policy_frame.columns:
        return False
    mode = pd.to_numeric(policy_frame[PORTFOLIO_CASHFLOW_DECISION_MODE_COLUMN], errors="coerce").fillna(0.0)
    return bool((mode > 0.5).any())


def normalize_portfolio_cashflow_decision(
    policy_frame: pd.DataFrame,
    *,
    current_weight: pd.Series | None = None,
    position_cap: float = 0.24,
    turnover_limit: float | None = None,
    deadband: float = 0.003,
    transaction_cost_rate: float = 0.0015,
    fail_closed: bool = True,
) -> PortfolioCashflowDecision:
    frame = policy_frame.copy()
    index = frame.index
    current = (
        current_weight.reindex(index).replace([np.inf, -np.inf], np.nan).fillna(0.0).astype(float)
        if current_weight is not None
        else _numeric(frame, "current_weight", 0.0)
    ).clip(lower=0.0)
    position_cap = float(max(0.0, position_cap))
    deadband = float(max(0.0, deadband))

    target_weight_raw = _optional_numeric(frame, "portfolio_daily_target_weight_intent")
    target_delta_raw = _optional_numeric(frame, "portfolio_daily_target_delta_intent")
    source_supply_raw = _optional_numeric(frame, "portfolio_set_v5_source_supply")
    receiver_demand_raw = _optional_numeric(frame, "portfolio_set_v5_receiver_demand")
    cash_buffer_score = _numeric(frame, "portfolio_set_v5_cash_buffer_score", 0.0).clip(0.0, 1.0)

    missing_columns = [
        name for name in PORTFOLIO_CASHFLOW_DECISION_REQUIRED_COLUMNS
        if name not in frame.columns and not (name == "current_weight" and current_weight is not None)
    ]
    target_weight = target_weight_raw.reindex(index) if target_weight_raw is not None else None
    target_delta = target_delta_raw.reindex(index) if target_delta_raw is not None else None
    if target_weight is None and target_delta is not None:
        target_weight = current + target_delta.fillna(0.0)
    if target_delta is None and target_weight is not None:
        target_delta = target_weight.fillna(current) - current
    if target_weight is None:
        target_weight = current.copy()
    if target_delta is None:
        target_delta = pd.Series(0.0, index=index, dtype=float)
    target_weight = target_weight.replace([np.inf, -np.inf], np.nan).fillna(current).astype(float)
    target_delta = target_delta.replace([np.inf, -np.inf], np.nan).fillna(0.0).astype(float)
    source_supply = (
        source_supply_raw.reindex(index).replace([np.inf, -np.inf], np.nan).fillna(0.0).astype(float)
        if source_supply_raw is not None
        else pd.Series(0.0, index=index, dtype=float)
    ).clip(lower=0.0)
    receiver_demand = (
        receiver_demand_raw.reindex(index).replace([np.inf, -np.inf], np.nan).fillna(0.0).astype(float)
        if receiver_demand_raw is not None
        else pd.Series(0.0, index=index, dtype=float)
    ).clip(lower=0.0)

    delta_from_weight = target_weight - current
    coherence_gap = (delta_from_weight - target_delta).abs()
    target_delta = delta_from_weight.where(delta_from_weight.abs() >= deadband, 0.0)
    target_weight = (current + target_delta).clip(lower=0.0, upper=position_cap)
    source_supply = source_supply.where(source_supply >= deadband, 0.0)
    receiver_demand = receiver_demand.where(receiver_demand >= deadband, 0.0)

    oracle_feasible = _numeric(frame, "portfolio_set_v5_oracle_feasible", 1.0) > 0.5
    source_active = source_supply > deadband
    receiver_active = receiver_demand > deadband
    source_candidate = (current > deadband) & (source_active | (target_delta < -deadband))
    receiver_headroom = (position_cap - current).clip(lower=0.0)
    receiver_candidate = (receiver_headroom > deadband) & (receiver_active | (target_delta > deadband))
    source_target = source_candidate & (target_delta < -deadband)
    receiver_target = receiver_candidate & (target_delta > deadband) & (~source_target)

    source_direction_conflict = source_active & (target_delta >= -deadband)
    receiver_direction_conflict = receiver_active & (target_delta <= deadband)
    overlap_conflict = source_target & receiver_target
    cap_violation_count = int(((target_weight > position_cap + 1.0e-8) | (target_weight < -1.0e-8)).sum())
    gross_violation_count = int(float(target_weight.sum()) > 1.0 + 1.0e-6)
    turnover = float((target_weight - current).abs().sum())
    turnover_violation_count = int(
        turnover_limit is not None and float(turnover_limit) >= 0.0 and turnover > float(turnover_limit) + 1.0e-6
    )
    current_cash = max(0.0, 1.0 - float(current.sum()))
    expected_cash_after = max(0.0, 1.0 - float(target_weight.sum()) - float(transaction_cost_rate) * turnover)
    flow_cash_after = max(
        0.0,
        current_cash + float(source_supply.sum()) - float(receiver_demand.sum()) - float(transaction_cost_rate) * turnover,
    )
    cash_conservation_gap = abs(expected_cash_after - flow_cash_after)
    target_weight_delta_coherence_gap_max = float(coherence_gap.max()) if len(coherence_gap) else 0.0
    conflict_count = int(
        source_direction_conflict.sum()
        + receiver_direction_conflict.sum()
        + overlap_conflict.sum()
        + cap_violation_count
        + gross_violation_count
        + turnover_violation_count
        + int((~oracle_feasible).sum())
    )
    invalid_reasons: list[str] = []
    if missing_columns:
        invalid_reasons.append("missing_required_columns")
    if not bool(oracle_feasible.all()):
        invalid_reasons.append("oracle_infeasible")
    if target_weight_delta_coherence_gap_max > 1.0e-6:
        invalid_reasons.append("target_delta_weight_conflict")
    if int(source_direction_conflict.sum()) > 0:
        invalid_reasons.append("source_direction_conflict")
    if int(receiver_direction_conflict.sum()) > 0:
        invalid_reasons.append("receiver_direction_conflict")
    if int(overlap_conflict.sum()) > 0:
        invalid_reasons.append("source_receiver_overlap")
    if cap_violation_count:
        invalid_reasons.append("position_cap_violation")
    if gross_violation_count:
        invalid_reasons.append("gross_violation")
    if turnover_violation_count:
        invalid_reasons.append("turnover_violation")
    if cash_conservation_gap > 1.0e-5:
        invalid_reasons.append("cash_conservation_gap")

    valid = len(invalid_reasons) == 0
    if not valid and fail_closed:
        target_delta = pd.Series(0.0, index=index, dtype=float)
        target_weight = current.clip(lower=0.0, upper=position_cap)
        source_supply = pd.Series(0.0, index=index, dtype=float)
        receiver_demand = pd.Series(0.0, index=index, dtype=float)
        source_candidate = pd.Series(False, index=index, dtype=bool)
        receiver_candidate = pd.Series(False, index=index, dtype=bool)
        source_target = pd.Series(False, index=index, dtype=bool)
        receiver_target = pd.Series(False, index=index, dtype=bool)
        turnover = 0.0
        expected_cash_after = max(0.0, 1.0 - float(target_weight.sum()))

    action = pd.Series("hold", index=index, dtype=object)
    action.loc[current <= deadband] = "skip"
    action.loc[(current <= deadband) & (target_delta > deadband)] = "open"
    action.loc[(current > deadband) & (target_delta > deadband)] = "add"
    action.loc[(current > deadband) & (target_delta < -deadband) & (target_weight > deadband)] = "reduce"
    action.loc[(current > deadband) & (target_delta < -deadband) & (target_weight <= deadband)] = "exit"

    source_capacity = current.clip(lower=deadband)
    release_intent = (source_supply / source_capacity).replace([np.inf, -np.inf], np.nan).fillna(0.0).clip(0.0, 1.0)
    receiver_score = _numeric(frame, "portfolio_daily_receiver_score", 0.0).clip(0.0, 1.0)
    receiver_score = pd.concat([receiver_score.rename("existing"), receiver_demand.rename("receiver_demand")], axis=1).max(axis=1)
    source_score = _numeric(frame, "portfolio_daily_source_score", 0.0).clip(0.0, 1.0)
    source_score = pd.concat([source_score.rename("existing"), release_intent.rename("release_intent")], axis=1).max(axis=1)

    frame["current_weight"] = current.astype(float)
    frame[PORTFOLIO_CASHFLOW_DECISION_MODE_COLUMN] = 1.0
    frame["portfolio_daily_target_weight_intent"] = target_weight.astype(float)
    frame["portfolio_daily_target_delta_intent"] = target_delta.astype(float)
    frame["target_delta_hint"] = target_delta.astype(float)
    frame["portfolio_set_v5_source_supply"] = source_supply.astype(float)
    frame["portfolio_set_v5_receiver_demand"] = receiver_demand.astype(float)
    frame["portfolio_set_v5_cash_buffer_score"] = cash_buffer_score.astype(float)
    frame["portfolio_daily_source_target_intent"] = source_target.astype(float)
    frame["portfolio_daily_receiver_target_intent"] = receiver_target.astype(float)
    frame["portfolio_daily_source_executable_candidate"] = source_candidate.astype(float)
    frame["portfolio_daily_receiver_executable_candidate"] = receiver_candidate.astype(float)
    frame["portfolio_daily_source_score"] = source_score.astype(float)
    frame["portfolio_daily_unified_source_score"] = source_score.astype(float)
    frame["portfolio_daily_receiver_score"] = receiver_score.astype(float)
    frame["portfolio_daily_unified_receiver_score"] = receiver_score.astype(float)
    frame["portfolio_daily_cash_score"] = cash_buffer_score.astype(float)
    frame["portfolio_daily_unified_cash_score"] = cash_buffer_score.astype(float)
    frame["portfolio_daily_release_first_intent"] = release_intent.astype(float)
    frame["release_first_intent_score"] = release_intent.astype(float)
    frame["release_first_intent_delta"] = (-source_supply).where(source_target, 0.0).astype(float)
    frame["release_first_action_hint"] = action.where(action.isin({"reduce", "exit"}), "hold").astype(str)
    block_reason = pd.Series("none", index=index, dtype=object)
    block_reason.loc[current <= deadband] = "not_held"
    if not valid:
        block_reason.loc[:] = "cashflow_contract_invalid"
    frame["release_first_block_reason"] = block_reason.astype(str)
    frame["portfolio_daily_source_release_quality"] = source_score.astype(float)
    frame["portfolio_daily_source_release_capacity"] = release_intent.astype(float)
    frame["portfolio_daily_source_release_preference"] = release_intent.astype(float)
    frame["portfolio_daily_source_executability"] = source_candidate.astype(float)
    frame["portfolio_daily_source_economic_release_score"] = release_intent.astype(float)
    frame["portfolio_daily_receiver_add_headroom"] = receiver_headroom.astype(float)
    frame["portfolio_daily_receiver_add_capacity"] = receiver_headroom.astype(float)
    frame["portfolio_daily_receiver_executability"] = receiver_candidate.astype(float)
    frame["action_label"] = action.astype(str)
    frame["portfolio_cashflow_decision_v1_valid"] = float(valid)
    frame["portfolio_cashflow_decision_v1_invalid_reason"] = "none" if valid else "|".join(invalid_reasons)
    frame["portfolio_cashflow_decision_v1_violation_count"] = float(conflict_count)
    frame["portfolio_cashflow_decision_v1_cash_conservation_gap"] = float(cash_conservation_gap)

    diagnostics = {
        "cashflow_decision_valid": float(valid),
        "cashflow_decision_invalid_reason": "none" if valid else "|".join(invalid_reasons),
        "cashflow_decision_missing_required_column_count": int(len(missing_columns)),
        "cashflow_decision_source_intent_count": int(source_target.sum()),
        "cashflow_decision_receiver_intent_count": int(receiver_target.sum()),
        "cashflow_decision_source_candidate_count": int(source_candidate.sum()),
        "cashflow_decision_receiver_candidate_count": int(receiver_candidate.sum()),
        "cashflow_decision_direction_conflict_count": int(source_direction_conflict.sum() + receiver_direction_conflict.sum()),
        "cashflow_decision_overlap_conflict_count": int(overlap_conflict.sum()),
        "cashflow_decision_target_delta_weight_coherence_gap_max": target_weight_delta_coherence_gap_max,
        "cashflow_decision_cash_conservation_gap": float(cash_conservation_gap),
        "cashflow_decision_turnover": float(turnover),
        "cashflow_decision_buy_turnover": float(target_delta.clip(lower=0.0).sum()),
        "cashflow_decision_sell_turnover": float((-target_delta.clip(upper=0.0)).sum()),
        "cashflow_decision_cash_after": float(expected_cash_after),
        "cashflow_decision_guarded_count": 0,
        "cashflow_decision_violation_count": int(conflict_count),
    }
    return PortfolioCashflowDecision(
        frame=frame,
        target_weight=target_weight.astype(float),
        target_delta=target_delta.astype(float),
        source_supply=source_supply.astype(float),
        receiver_demand=receiver_demand.astype(float),
        source_candidate=source_candidate.astype(bool),
        receiver_candidate=receiver_candidate.astype(bool),
        source_target=source_target.astype(bool),
        receiver_target=receiver_target.astype(bool),
        action_label=action.astype(str),
        valid=valid,
        diagnostics=diagnostics,
    )
