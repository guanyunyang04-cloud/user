from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from daily_research.continuous_policy.portfolio_cashflow_decision import (
    PORTFOLIO_CASHFLOW_DECISION_MODE_COLUMN,
    normalize_portfolio_cashflow_decision,
)
from daily_research.continuous_policy.model_seq_v3 import SEQUENCE_STEP_ORDER, resolve_sequence_columns
from daily_research.continuous_policy.model_v2 import _apply_matrix, _prepare_matrix, _split_indices
from daily_research.continuous_policy.training_contracts import TRAINER_BACKEND_FORMAL_PORTFOLIO_SET_V5
from daily_research.continuous_policy.training_runtime_acceleration import (
    autocast_context,
    configure_torch_training_acceleration,
    move_to_device,
)


PORTFOLIO_SET_V5_ARTIFACT_TYPE = "continuous_policy_torch_portfolio_set_v5"
PORTFOLIO_SET_V5_ARTIFACT_FILENAME = "continuous_policy_portfolio_set_v5_artifact.pt"
PORTFOLIO_SET_V5_DEFAULT_STRICT_GOLD_DATASET_ID = "continuous_policy_training_matrices__strict_train__36c234208d5f375ea1cccfc1"
PORTFOLIO_SET_V5_DFL_PG_V1_VERSION = "portfolio_set_v5_dfl_pg_v1"
PORTFOLIO_SET_V5_R69_INTERNAL_VERSION = "portfolio_set_v5_dfl_pg_v1_r69_value_arbitration"
PORTFOLIO_SET_V5_INTERNAL_VERSION = PORTFOLIO_SET_V5_R69_INTERNAL_VERSION

PORTFOLIO_SET_V5_OUTPUT_NAMES: tuple[str, ...] = (
    "target_weight",
    "target_delta",
    "source_supply_score",
    "receiver_demand_score",
    "cash_buffer_score",
    "release_intent",
    "reduce_quality",
    "exit_hazard",
)
PORTFOLIO_SET_V5_DECISION_TARGET_NAMES: tuple[str, ...] = (
    "source_supply",
    "receiver_demand",
    "cash_buffer",
    "target_weight",
    "turnover_budget",
    "risk_budget",
    "constraint_violation",
    "decision_value",
    "deploy_value",
    "release_value",
    "defense_value",
    "cash_timing_value",
    "source_opportunity_cost",
    "receiver_source_spread_value",
    "reversal_risk_penalty",
    "source_wrong_side_sell_penalty",
)
PORTFOLIO_SET_V5_LOSS_ALIASES: tuple[str, ...] = (
    "alpha_result_value_budget_split_v48",
    "portfolio_set_release_first_decision_v1",
    PORTFOLIO_SET_V5_DFL_PG_V1_VERSION,
    PORTFOLIO_SET_V5_R69_INTERNAL_VERSION,
    PORTFOLIO_SET_V5_INTERNAL_VERSION,
)
PORTFOLIO_SET_V5_LOSS_PROFILE_NAMES: tuple[str, ...] = PORTFOLIO_SET_V5_LOSS_ALIASES
PORTFOLIO_SET_V5_MAX_TRAIN_DAYS = 256
PORTFOLIO_SET_V5_MAX_STOCKS_PER_DAY = 3070


def resolve_portfolio_set_v5_loss_profile(profile_name: str | None) -> tuple[str, dict[str, dict[str, float]]]:
    name = str(profile_name or PORTFOLIO_SET_V5_INTERNAL_VERSION).strip() or PORTFOLIO_SET_V5_INTERNAL_VERSION
    if name not in PORTFOLIO_SET_V5_LOSS_ALIASES:
        raise ValueError(
            f"Unsupported portfolio-set v5 loss profile: {profile_name!r}. "
            "Only alpha_result_value_budget_split_v48 / portfolio_set_release_first_decision_v1 / "
            f"{PORTFOLIO_SET_V5_DFL_PG_V1_VERSION} / {PORTFOLIO_SET_V5_R69_INTERNAL_VERSION} are supported."
        )
    resolved_name = (
        PORTFOLIO_SET_V5_DFL_PG_V1_VERSION
        if name in {PORTFOLIO_SET_V5_DFL_PG_V1_VERSION, "portfolio_set_release_first_decision_v1"}
        else name
    )
    if name == PORTFOLIO_SET_V5_R69_INTERNAL_VERSION:
        resolved_name = PORTFOLIO_SET_V5_R69_INTERNAL_VERSION
    return resolved_name, {
        "multi_objective_loss_weights": {
            "action_total": 0.0,
            "duration_total": 0.0,
            "target_weight_closure_total": 1.0,
            "source_supply_total": 0.52,
            "receiver_demand_total": 0.52,
            "cash_buffer_total": 0.20,
            "target_delta_weight_coherence_total": 0.34,
            "release_flow_balance_total": 0.48,
            "underdeployment_high_cash_total": 0.36,
            "intent_translation_conflict_total": 0.34,
            "decision_oracle_total": 1.20,
            "pg_dfl_surrogate_total": 0.42,
            "constraint_violation_total": 0.60,
            "value_arbitration_total": 0.54 if resolved_name == PORTFOLIO_SET_V5_R69_INTERNAL_VERSION else 0.0,
            "cash_timing_value_total": 0.26 if resolved_name == PORTFOLIO_SET_V5_R69_INTERNAL_VERSION else 0.0,
            "reversal_guard_total": 0.34 if resolved_name == PORTFOLIO_SET_V5_R69_INTERNAL_VERSION else 0.0,
        }
    }


def _numeric_series(frame: pd.DataFrame, name: str, default: float = 0.0) -> pd.Series:
    if name not in frame.columns:
        return pd.Series(float(default), index=frame.index, dtype=float)
    return pd.to_numeric(frame[name], errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(float(default))


def _current_weight(frame: pd.DataFrame) -> pd.Series:
    for name in ("current_weight", "position_weight", "portfolio_weight", "weight"):
        if name in frame.columns:
            return _numeric_series(frame, name, 0.0).clip(0.0, 1.0)
    return pd.Series(0.0, index=frame.index, dtype=float)


def _holding_mask(frame: pd.DataFrame, current: pd.Series, deadband: float = 0.003) -> pd.Series:
    holding_flag = _numeric_series(frame, "holding_flag", 0.0) > 0.5
    holding_flag = holding_flag | (_numeric_series(frame, "holding_flag_target", 0.0) > 0.5)
    return holding_flag | (current > float(deadband))


def _max_numeric_columns(frame: pd.DataFrame, columns: tuple[str, ...], default: float = 0.0) -> pd.Series:
    values = [_numeric_series(frame, column, default).rename(column) for column in columns if column in frame.columns]
    if not values:
        return pd.Series(float(default), index=frame.index, dtype=float)
    return pd.concat(values, axis=1).max(axis=1).fillna(float(default))


def _ensure_features(frame: pd.DataFrame, feature_names: list[str]) -> pd.DataFrame:
    result = frame.copy()
    for name in feature_names:
        if name not in result.columns:
            result[name] = np.nan
    return result


def _date_column(frame: pd.DataFrame) -> str:
    if "date" in frame.columns:
        return "date"
    if "trade_date" in frame.columns:
        return "trade_date"
    raise ValueError("portfolio-set v5 requires a date or trade_date column.")


def project_portfolio_set_v5_cashflow_oracle(
    *,
    current_weight: torch.Tensor,
    source_score: torch.Tensor,
    receiver_score: torch.Tensor,
    cash_buffer_score: torch.Tensor | None = None,
    deploy_value: torch.Tensor | None = None,
    release_value: torch.Tensor | None = None,
    defense_value: torch.Tensor | None = None,
    cash_timing_value: torch.Tensor | None = None,
    source_opportunity_cost: torch.Tensor | None = None,
    receiver_source_spread_value: torch.Tensor | None = None,
    reversal_risk_penalty: torch.Tensor | None = None,
    source_wrong_side_sell_penalty: torch.Tensor | None = None,
    sample_mask: torch.Tensor | None = None,
    turnover_budget: torch.Tensor | None = None,
    risk_budget: torch.Tensor | None = None,
    transaction_cost: float = 0.001,
    position_cap: float = 0.24,
    deadband: float = 0.003,
) -> dict[str, torch.Tensor]:
    """Project scores into a long-only source/receiver/cash decision surface."""
    current = current_weight.float().clamp(0.0, 1.0)
    if current.ndim == 1:
        current = current.unsqueeze(0)
    mask = torch.ones_like(current, dtype=torch.bool) if sample_mask is None else sample_mask.bool()
    if mask.ndim == 1:
        mask = mask.unsqueeze(0)
    source = source_score.float()
    receiver = receiver_score.float()
    if source.ndim == 1:
        source = source.unsqueeze(0)
    if receiver.ndim == 1:
        receiver = receiver.unsqueeze(0)
    source = source.clamp(0.0, 1.0)
    receiver = receiver.clamp(0.0, 1.0)
    dtype = current.dtype
    device = current.device
    mask_f = mask.to(dtype=dtype)
    held = (current > float(deadband)).to(dtype=dtype) * mask_f
    headroom = (float(position_cap) - current).clamp_min(0.0) * mask_f
    cash_now = (1.0 - (current * mask_f).sum(dim=1)).clamp(0.0, 1.0)

    if cash_buffer_score is None:
        cash_seed = torch.zeros_like(current)
    else:
        cash_seed = cash_buffer_score.float()
        if cash_seed.ndim == 1:
            cash_seed = cash_seed.unsqueeze(0)
        cash_seed = cash_seed.clamp(0.0, 1.0)
    value_arbitration_mode = any(
        item is not None
        for item in (
            deploy_value,
            release_value,
            defense_value,
            cash_timing_value,
            source_opportunity_cost,
            receiver_source_spread_value,
            reversal_risk_penalty,
            source_wrong_side_sell_penalty,
        )
    )

    def _optional_value(value: torch.Tensor | None, default: float = 0.0) -> torch.Tensor:
        if value is None:
            return torch.full_like(current, float(default))
        result = value.float()
        if result.ndim == 1:
            result = result.unsqueeze(0)
        if result.shape[0] == 1 and current.shape[0] > 1:
            result = result.expand(current.shape[0], -1)
        return result.to(device=current.device, dtype=dtype).clamp(0.0, 1.0) * mask_f

    deploy = _optional_value(deploy_value)
    release = _optional_value(release_value)
    defense = _optional_value(defense_value)
    cash_timing = _optional_value(cash_timing_value)
    opportunity_cost = _optional_value(source_opportunity_cost)
    spread_value = _optional_value(receiver_source_spread_value)
    reversal_penalty = _optional_value(reversal_risk_penalty)
    wrong_side_penalty = _optional_value(source_wrong_side_sell_penalty)
    defense_pressure = torch.maximum(defense, cash_timing)
    source_utility = (
        source * (0.55 if value_arbitration_mode else 1.0)
        + release * 0.35
        + defense_pressure * 0.18
        + spread_value * 0.12
        - opportunity_cost * 0.42
        - reversal_penalty * 0.36
        - wrong_side_penalty * 0.52
    ).clamp(0.0, 1.0)
    receiver_utility = (
        receiver * (0.60 if value_arbitration_mode else 1.0)
        + deploy * 0.35
        + spread_value * 0.16
        - defense_pressure * 0.22
    ).clamp(0.0, 1.0)
    if value_arbitration_mode:
        source = source_utility
        receiver = receiver_utility
    source_capacity = current * source * held
    receiver_capacity = headroom * receiver
    if risk_budget is None:
        risk = (torch.maximum(cash_seed, defense_pressure) * mask_f).sum(dim=1) / mask_f.sum(dim=1).clamp_min(1.0)
    else:
        risk = risk_budget.float().flatten()
        if risk.numel() == 1 and current.shape[0] > 1:
            risk = risk.expand(current.shape[0])
    risk = risk.to(device=device, dtype=dtype).clamp(0.0, 1.0)
    if turnover_budget is None:
        turnover = (0.18 - 0.08 * risk).clamp(0.04, 0.24)
    else:
        turnover = turnover_budget.float().flatten()
        if turnover.numel() == 1 and current.shape[0] > 1:
            turnover = turnover.expand(current.shape[0])
        turnover = turnover.to(device=device, dtype=dtype).clamp(0.0, 0.50)
    desired_cash_floor = (0.04 + 0.20 * risk).clamp(0.02, 0.35)
    if value_arbitration_mode:
        desired_cash_floor = (desired_cash_floor + 0.18 * defense_pressure.mean(dim=1)).clamp(0.02, 0.55)
        reachable_cash_floor = cash_now + torch.minimum(source_capacity.sum(dim=1), turnover)
        cash_floor = torch.minimum(desired_cash_floor, reachable_cash_floor)
    else:
        cash_floor = torch.minimum(desired_cash_floor, cash_now)

    source_capacity_sum = source_capacity.sum(dim=1)
    receiver_capacity_sum = receiver_capacity.sum(dim=1)
    cash_shortfall = (cash_floor - cash_now).clamp_min(0.0)
    cash_available = (cash_now - cash_floor).clamp_min(0.0)
    receiver_signal = (receiver_capacity_sum > float(deadband)).to(dtype=dtype)
    source_signal = (source_capacity_sum > float(deadband)).to(dtype=dtype)
    rotation_release_need = torch.minimum(
        source_capacity_sum,
        torch.maximum(receiver_capacity_sum * 0.65, turnover * 0.20 * receiver_signal),
    )
    source_budget = torch.minimum(
        source_capacity_sum,
        torch.minimum(turnover, torch.maximum(rotation_release_need, cash_shortfall)),
    )
    cash_deploy_budget = torch.minimum(cash_available, turnover) * (0.45 + 0.35 * (1.0 - risk))
    no_spread_defense = torch.zeros_like(cash_deploy_budget, dtype=torch.bool)
    if value_arbitration_mode:
        defense_mean = defense_pressure.mean(dim=1)
        receiver_spread_mean = (
            (spread_value * (headroom > float(deadband)).to(dtype=dtype)).sum(dim=1)
            / (headroom > float(deadband)).to(dtype=dtype).sum(dim=1).clamp_min(1.0)
        )
        cash_deploy_budget = cash_deploy_budget * (1.0 - 0.70 * defense_mean).clamp(0.05, 1.0)
        no_spread_defense = (defense_mean > 0.55) & (receiver_spread_mean < 0.20)
        cash_deploy_budget = torch.where(no_spread_defense, torch.zeros_like(cash_deploy_budget), cash_deploy_budget)
    cash_deploy_budget = torch.minimum(cash_deploy_budget, cash_available).clamp_min(0.0)
    receiver_budget = torch.minimum(
        receiver_capacity_sum,
        torch.minimum(turnover, source_budget + cash_deploy_budget),
    )

    eps = torch.tensor(1.0e-8, dtype=dtype, device=device)
    source_rank_count = int(max(1, min(6, current.shape[1])))
    receiver_rank_count = int(max(1, min(8, current.shape[1])))
    source_rank_mask = torch.zeros_like(source_capacity, dtype=torch.bool)
    receiver_rank_mask = torch.zeros_like(receiver_capacity, dtype=torch.bool)
    source_positive = source_capacity > float(deadband)
    receiver_positive = receiver_capacity > float(deadband)
    if source_capacity.shape[1] > 0:
        source_k = min(source_rank_count, source_capacity.shape[1])
        source_top_idx = torch.topk(source_capacity, k=source_k, dim=1).indices
        source_rank_mask.scatter_(1, source_top_idx, True)
        source_rank_mask = source_rank_mask & source_positive
    if receiver_capacity.shape[1] > 0:
        receiver_score_rank = receiver_capacity * (~source_positive).to(dtype=dtype)
        receiver_k = min(receiver_rank_count, receiver_capacity.shape[1])
        receiver_top_idx = torch.topk(receiver_score_rank, k=receiver_k, dim=1).indices
        receiver_rank_mask.scatter_(1, receiver_top_idx, True)
        receiver_rank_mask = receiver_rank_mask & receiver_positive & (~source_rank_mask)
        missing_receiver = receiver_signal.bool() & (~receiver_rank_mask.any(dim=1))
        if bool(missing_receiver.any()):
            fallback_eligible = receiver_positive & (~source_rank_mask)
            fallback_scores = receiver_capacity * fallback_eligible.to(dtype=dtype)
            missing_receiver = missing_receiver & fallback_eligible.any(dim=1)
            fallback_idx = torch.argmax(fallback_scores, dim=1)
            receiver_rank_mask[missing_receiver, :] = False
            receiver_rank_mask[missing_receiver, fallback_idx[missing_receiver]] = True
    source_capacity_sparse = source_capacity * source_rank_mask.to(dtype=dtype)
    receiver_capacity_sparse = receiver_capacity * receiver_rank_mask.to(dtype=dtype)
    source_capacity_sparse_sum = source_capacity_sparse.sum(dim=1)
    receiver_capacity_sparse_sum = receiver_capacity_sparse.sum(dim=1)
    cash_shortfall_budget = torch.minimum(source_capacity_sparse_sum, torch.minimum(turnover, cash_shortfall))
    if value_arbitration_mode:
        defense_mean = defense_pressure.mean(dim=1)
        receiver_spread_mean = (
            (spread_value * receiver_rank_mask.to(dtype=dtype)).sum(dim=1)
            / receiver_rank_mask.to(dtype=dtype).sum(dim=1).clamp_min(1.0)
        )
        defense_cash_release_budget = torch.minimum(
            source_capacity_sparse_sum - cash_shortfall_budget,
            (turnover - cash_shortfall_budget).clamp_min(0.0)
            * defense_mean
            * (1.0 - receiver_spread_mean).clamp(0.0, 1.0)
            * (defense_mean > 0.55).to(dtype=dtype),
        ).clamp_min(0.0)
        deploy_competition = (receiver_capacity_sparse_sum > float(deadband)) & (receiver_spread_mean >= 0.20)
        defense_cap = torch.where(
            deploy_competition,
            (turnover - cash_shortfall_budget).clamp_min(0.0) * 0.55,
            (turnover - cash_shortfall_budget).clamp_min(0.0),
        )
        defense_cash_release_budget = torch.minimum(defense_cash_release_budget, defense_cap)
    else:
        defense_cash_release_budget = torch.zeros_like(cash_shortfall_budget)
    residual_source_capacity = (source_capacity_sparse_sum - cash_shortfall_budget - defense_cash_release_budget).clamp_min(0.0)
    residual_turnover = (turnover - cash_shortfall_budget - defense_cash_release_budget).clamp_min(0.0)
    pair_seed_gate = (~no_spread_defense).to(dtype=dtype)
    pair_seed_budget = torch.minimum(
        torch.minimum(residual_source_capacity, receiver_capacity_sparse_sum),
        residual_turnover * 0.35 * source_signal * receiver_signal * pair_seed_gate,
    )
    cash_buy_budget = torch.minimum(
        (receiver_capacity_sparse_sum - pair_seed_budget).clamp_min(0.0),
        torch.minimum(cash_deploy_budget, (residual_turnover - 2.0 * pair_seed_budget).clamp_min(0.0)),
    )
    source_budget = cash_shortfall_budget + defense_cash_release_budget + pair_seed_budget
    receiver_budget = pair_seed_budget + cash_buy_budget
    source_alloc = source_capacity_sparse * (source_budget / source_capacity_sparse_sum.clamp_min(eps)).unsqueeze(1)
    receiver_alloc = receiver_capacity_sparse * (receiver_budget / receiver_capacity_sparse_sum.clamp_min(eps)).unsqueeze(1)
    target_weight = (current - source_alloc + receiver_alloc).clamp(0.0, float(position_cap)) * mask_f
    target_delta = (target_weight - current) * mask_f
    source_used = source_alloc.sum(dim=1)
    receiver_used = receiver_alloc.sum(dim=1)
    cash_after = (cash_now + source_used - receiver_used).clamp(0.0, 1.0)
    turnover_used = source_used + receiver_used
    trade_cost_turnover = source_used + receiver_used
    cap_violation = (target_weight - float(position_cap)).clamp_min(0.0).sum(dim=1)
    floor_violation = (-target_weight).clamp_min(0.0).sum(dim=1)
    source_violation = (source_alloc - current).clamp_min(0.0).sum(dim=1)
    receiver_violation = (receiver_alloc - headroom).clamp_min(0.0).sum(dim=1)
    turnover_violation = (turnover_used - turnover).clamp_min(0.0)
    cash_violation = (cash_floor - cash_after).clamp_min(0.0)
    violation = cap_violation + floor_violation + source_violation + receiver_violation + turnover_violation + cash_violation
    decision_value = (
        (source_alloc * source).sum(dim=1)
        + (receiver_alloc * receiver).sum(dim=1)
        + cash_after * torch.maximum(risk, defense_pressure.mean(dim=1))
        - float(transaction_cost) * trade_cost_turnover
        - 2.0 * violation
    )
    if value_arbitration_mode:
        decision_value = decision_value - (
            (source_alloc * (opportunity_cost + reversal_penalty + wrong_side_penalty)).sum(dim=1)
            + (receiver_alloc * defense_pressure).sum(dim=1) * 0.25
        )
    return {
        "source_supply": source_alloc * mask_f,
        "receiver_demand": receiver_alloc * mask_f,
        "cash_buffer": cash_after,
        "target_weight": target_weight,
        "target_delta": target_delta,
        "turnover_budget": turnover,
        "risk_budget": risk,
        "constraint_violation": violation,
        "decision_value": decision_value,
        "turnover_used": turnover_used,
        "trade_cost_turnover": trade_cost_turnover,
        "cash_floor": cash_floor,
        "deploy_value": deploy,
        "release_value": release,
        "defense_value": defense_pressure,
        "cash_timing_value": cash_timing,
        "source_opportunity_cost": opportunity_cost,
        "receiver_source_spread_value": spread_value,
        "reversal_risk_penalty": reversal_penalty,
        "source_wrong_side_sell_penalty": wrong_side_penalty,
    }


def _oracle_numpy(
    *,
    current: np.ndarray,
    source_score: np.ndarray,
    receiver_score: np.ndarray,
    cash_score: np.ndarray,
    deploy_value: np.ndarray | None = None,
    release_value: np.ndarray | None = None,
    defense_value: np.ndarray | None = None,
    cash_timing_value: np.ndarray | None = None,
    source_opportunity_cost: np.ndarray | None = None,
    receiver_source_spread_value: np.ndarray | None = None,
    reversal_risk_penalty: np.ndarray | None = None,
    source_wrong_side_sell_penalty: np.ndarray | None = None,
    turnover_budget: float,
    risk_budget: float,
) -> dict[str, np.ndarray | float]:
    def _tensor_or_none(values: np.ndarray | None) -> torch.Tensor | None:
        if values is None:
            return None
        return torch.as_tensor(np.asarray(values, dtype=float)[None, :], dtype=torch.float32)

    oracle = project_portfolio_set_v5_cashflow_oracle(
        current_weight=torch.as_tensor(current[None, :], dtype=torch.float32),
        source_score=torch.as_tensor(source_score[None, :], dtype=torch.float32),
        receiver_score=torch.as_tensor(receiver_score[None, :], dtype=torch.float32),
        cash_buffer_score=torch.as_tensor(cash_score[None, :], dtype=torch.float32),
        deploy_value=_tensor_or_none(deploy_value),
        release_value=_tensor_or_none(release_value),
        defense_value=_tensor_or_none(defense_value),
        cash_timing_value=_tensor_or_none(cash_timing_value),
        source_opportunity_cost=_tensor_or_none(source_opportunity_cost),
        receiver_source_spread_value=_tensor_or_none(receiver_source_spread_value),
        reversal_risk_penalty=_tensor_or_none(reversal_risk_penalty),
        source_wrong_side_sell_penalty=_tensor_or_none(source_wrong_side_sell_penalty),
        turnover_budget=torch.as_tensor([float(turnover_budget)], dtype=torch.float32),
        risk_budget=torch.as_tensor([float(risk_budget)], dtype=torch.float32),
        sample_mask=torch.ones((1, len(current)), dtype=torch.bool),
    )
    return {
        "source_supply": oracle["source_supply"][0].detach().cpu().numpy().astype(float),
        "receiver_demand": oracle["receiver_demand"][0].detach().cpu().numpy().astype(float),
        "cash_buffer": float(oracle["cash_buffer"][0].detach().cpu()),
        "target_weight": oracle["target_weight"][0].detach().cpu().numpy().astype(float),
        "target_delta": oracle["target_delta"][0].detach().cpu().numpy().astype(float),
        "turnover_budget": float(oracle["turnover_budget"][0].detach().cpu()),
        "risk_budget": float(oracle["risk_budget"][0].detach().cpu()),
        "constraint_violation": float(oracle["constraint_violation"][0].detach().cpu()),
        "decision_value": float(oracle["decision_value"][0].detach().cpu()),
        "deploy_value": oracle["deploy_value"][0].detach().cpu().numpy().astype(float),
        "release_value": oracle["release_value"][0].detach().cpu().numpy().astype(float),
        "defense_value": oracle["defense_value"][0].detach().cpu().numpy().astype(float),
        "cash_timing_value": oracle["cash_timing_value"][0].detach().cpu().numpy().astype(float),
        "source_opportunity_cost": oracle["source_opportunity_cost"][0].detach().cpu().numpy().astype(float),
        "receiver_source_spread_value": oracle["receiver_source_spread_value"][0].detach().cpu().numpy().astype(float),
        "reversal_risk_penalty": oracle["reversal_risk_penalty"][0].detach().cpu().numpy().astype(float),
        "source_wrong_side_sell_penalty": oracle["source_wrong_side_sell_penalty"][0].detach().cpu().numpy().astype(float),
    }


def build_portfolio_set_v5_targets(sample_frame: pd.DataFrame, *, deadband: float = 0.003) -> pd.DataFrame:
    current = _current_weight(sample_frame)
    held = _holding_mask(sample_frame, current, deadband=deadband)
    raw_delta = _numeric_series(sample_frame, "portfolio_daily_target_delta_intent", 0.0)
    if "portfolio_daily_target_delta_intent" not in sample_frame.columns and "target_delta_hint" in sample_frame.columns:
        raw_delta = _numeric_series(sample_frame, "target_delta_hint", 0.0)
    source_score = _max_numeric_columns(
        sample_frame,
        (
            "portfolio_daily_source_score",
            "portfolio_daily_unified_source_score",
            "portfolio_daily_source_release_preference",
            "portfolio_daily_source_release_quality",
            "sell_release_value",
            "release_value_target",
            "release_gate_target",
        ),
    ).clip(0.0, 1.0)
    release_action = sample_frame.get("action_label", pd.Series("hold", index=sample_frame.index)).fillna("hold").astype(str).str.lower().isin({"reduce", "exit"}).astype(float)
    source_score = pd.concat([source_score.rename("source_score"), release_action.rename("release_action"), (-raw_delta.clip(upper=0.0) / current.clip(lower=float(deadband))).clip(0.0, 1.0).rename("negative_delta")], axis=1).max(axis=1)
    keep_risk = _numeric_series(sample_frame, "portfolio_daily_source_forward_proxy_keep_risk", 0.0).clip(0.0, 1.0)
    block_risk = _numeric_series(sample_frame, "portfolio_daily_source_economic_block_risk", 0.0).clip(0.0, 1.0)
    source_score = source_score.where(held & (keep_risk < 0.78) & (block_risk < 0.78), 0.0).clip(0.0, 1.0)
    deploy_value = _max_numeric_columns(
        sample_frame,
        (
            "deploy_value_target",
            "result_value_deploy_value_target",
            "result_value_alpha_opportunity_value",
            "portfolio_daily_receiver_score",
            "portfolio_daily_unified_receiver_score",
        ),
    ).clip(0.0, 1.0)
    release_value = _max_numeric_columns(
        sample_frame,
        (
            "release_value_target",
            "sell_release_value",
            "portfolio_daily_source_release_quality",
            "portfolio_daily_source_economic_release_score",
            "portfolio_daily_source_release_preference",
        ),
    ).clip(0.0, 1.0)
    defense_value = _max_numeric_columns(
        sample_frame,
        (
            "defense_value_target",
            "cash_defense_value",
            "portfolio_daily_allocation_drawdown_control_target",
            "portfolio_daily_allocation_tail_risk_control_target",
            "market_downside_pressure",
        ),
    ).clip(0.0, 1.0)
    cash_timing_value = _max_numeric_columns(
        sample_frame,
        (
            "budget_cash_timing_signal_target",
            "portfolio_daily_cash_score",
            "portfolio_daily_unified_cash_score",
            "cash_defense_value",
            "cash_regime_pressure",
        ),
    ).clip(0.0, 1.0)
    source_opportunity_cost = _max_numeric_columns(
        sample_frame,
        (
            "portfolio_daily_source_opportunity_cost",
            "portfolio_daily_source_opportunity_cost_penalty",
            "hold_continuation_value",
            "alpha_opportunity_value",
            "large_upside_1d_target",
            "portfolio_daily_source_forward_proxy_keep_risk",
            "portfolio_daily_source_economic_block_risk",
        ),
    ).clip(0.0, 1.0)
    receiver_forward = _numeric_series(sample_frame, "portfolio_daily_receiver_forward_excess_5d", 0.0)
    source_forward = _numeric_series(sample_frame, "portfolio_daily_source_forward_excess_5d", 0.0)
    receiver_source_spread_value = _max_numeric_columns(
        sample_frame,
        (
            "portfolio_daily_receiver_source_spread_reward",
            "portfolio_daily_source_forward_spread_score",
        ),
    ).clip(0.0, 1.0)
    receiver_source_spread_value = pd.concat(
        [
            receiver_source_spread_value.rename("spread_reward"),
            ((receiver_forward - source_forward) / 0.075).clip(0.0, 1.0).rename("forward_spread"),
        ],
        axis=1,
    ).max(axis=1).fillna(0.0).clip(0.0, 1.0)
    positive_forward_sell_penalty = _max_numeric_columns(
        sample_frame,
        (
            "portfolio_daily_source_positive_forward_penalty",
            "portfolio_daily_source_bad_forward_spread_risk",
            "portfolio_daily_source_forward_strength_brake_risk",
        ),
    ).clip(0.0, 1.0)
    positive_forward_sell_penalty = pd.concat(
        [
            positive_forward_sell_penalty.rename("positive_forward_penalty"),
            (source_forward / 0.075).clip(0.0, 1.0).rename("source_forward_positive"),
        ],
        axis=1,
    ).max(axis=1).fillna(0.0).clip(0.0, 1.0)
    reversal_risk_penalty = _max_numeric_columns(
        sample_frame,
        (
            "reduce_reversal_pressure",
            "recent_reversal_rate_20d",
            "reentry_guard_score",
            "portfolio_daily_source_forward_proxy_keep_risk",
        ),
    ).clip(0.0, 1.0)
    source_score = (
        0.52 * source_score
        + 0.30 * release_value
        + 0.16 * defense_value
        + 0.12 * receiver_source_spread_value
        - 0.36 * source_opportunity_cost
        - 0.42 * positive_forward_sell_penalty
        - 0.30 * reversal_risk_penalty
    ).where(held, 0.0).clip(0.0, 1.0)
    receiver_score = _max_numeric_columns(
        sample_frame,
        (
            "portfolio_daily_receiver_score",
            "portfolio_daily_unified_receiver_score",
            "alpha_opportunity_value",
            "deploy_value_target",
            "deploy_gate_target",
            "result_value_deploy_gate_target",
            "result_value_alpha_opportunity_value",
        ),
    ).clip(0.0, 1.0)
    receiver_action = sample_frame.get("action_label", pd.Series("hold", index=sample_frame.index)).fillna("hold").astype(str).str.lower().isin({"open", "add"}).astype(float)
    headroom = (0.24 - current).clip(lower=0.0)
    receiver_score = pd.concat([receiver_score.rename("receiver_score"), receiver_action.rename("receiver_action"), (raw_delta.clip(lower=0.0) / headroom.clip(lower=float(deadband))).clip(0.0, 1.0).rename("positive_delta")], axis=1).max(axis=1)
    receiver_score = (
        0.62 * receiver_score
        + 0.36 * deploy_value
        + 0.22 * receiver_source_spread_value
        - 0.14 * defense_value
        - 0.06 * cash_timing_value
    ).clip(0.0, 1.0)
    receiver_score = pd.concat(
        [
            receiver_score.rename("receiver_value_score"),
            ((deploy_value - defense_value.clip(upper=0.60) * 0.35 + receiver_source_spread_value * 0.25).clip(0.0, 1.0)).rename("deploy_floor"),
        ],
        axis=1,
    ).max(axis=1).fillna(0.0).clip(0.0, 1.0)
    receiver_score = receiver_score.where(headroom > float(deadband), 0.0).clip(0.0, 1.0)
    risk_budget = _max_numeric_columns(
        sample_frame,
        (
            "cash_defense_value",
            "portfolio_daily_cash_score",
            "cash_regime_pressure",
            "market_downside_pressure",
            "portfolio_daily_allocation_uncertainty_pressure_target",
            "portfolio_daily_allocation_tail_risk_control_target",
        ),
    ).clip(0.0, 1.0)
    turnover_budget = (
        0.18
        - 0.08 * risk_budget
        + 0.04 * _numeric_series(sample_frame, "portfolio_daily_allocation_trade_quality_target", 0.0).clip(0.0, 1.0)
    ).clip(0.04, 0.24)
    enriched = pd.DataFrame(index=sample_frame.index)
    day_col = _date_column(sample_frame)
    for _, group in sample_frame.groupby(sample_frame[day_col].astype(str), sort=True):
        idx = group.index
        decision = _oracle_numpy(
            current=current.loc[idx].to_numpy(dtype=float),
            source_score=source_score.loc[idx].to_numpy(dtype=float),
            receiver_score=receiver_score.loc[idx].to_numpy(dtype=float),
            cash_score=risk_budget.loc[idx].to_numpy(dtype=float),
            deploy_value=deploy_value.loc[idx].to_numpy(dtype=float),
            release_value=release_value.loc[idx].to_numpy(dtype=float),
            defense_value=defense_value.loc[idx].to_numpy(dtype=float),
            cash_timing_value=cash_timing_value.loc[idx].to_numpy(dtype=float),
            source_opportunity_cost=source_opportunity_cost.loc[idx].to_numpy(dtype=float),
            receiver_source_spread_value=receiver_source_spread_value.loc[idx].to_numpy(dtype=float),
            reversal_risk_penalty=reversal_risk_penalty.loc[idx].to_numpy(dtype=float),
            source_wrong_side_sell_penalty=positive_forward_sell_penalty.loc[idx].to_numpy(dtype=float),
            turnover_budget=float(turnover_budget.loc[idx].median()) if len(idx) else 0.12,
            risk_budget=float(np.maximum(risk_budget.loc[idx], defense_value.loc[idx]).median()) if len(idx) else 0.0,
        )
        source_supply = np.asarray(decision["source_supply"], dtype=float)
        receiver_demand = np.asarray(decision["receiver_demand"], dtype=float)
        target_weight = np.asarray(decision["target_weight"], dtype=float)
        target_delta = np.asarray(decision["target_delta"], dtype=float)
        current_values = current.loc[idx].to_numpy(dtype=float)
        headroom_values = (0.24 - current.loc[idx]).clip(lower=0.0).to_numpy(dtype=float)
        source_ratio = np.divide(source_supply, np.maximum(current_values, float(deadband)), out=np.zeros_like(source_supply), where=current_values > float(deadband))
        receiver_ratio = np.divide(receiver_demand, np.maximum(headroom_values, float(deadband)), out=np.zeros_like(receiver_demand), where=headroom_values > float(deadband))
        enriched.loc[idx, "target_weight"] = target_weight
        enriched.loc[idx, "target_delta"] = target_delta
        enriched.loc[idx, "source_supply_score"] = np.clip(source_ratio, 0.0, 1.0)
        enriched.loc[idx, "receiver_demand_score"] = np.clip(receiver_ratio, 0.0, 1.0)
        enriched.loc[idx, "cash_buffer_score"] = float(decision["cash_buffer"])
        enriched.loc[idx, "release_intent"] = np.clip(source_ratio, 0.0, 1.0)
        enriched.loc[idx, "reduce_quality"] = source_score.loc[idx].to_numpy(dtype=float)
        enriched.loc[idx, "exit_hazard"] = ((source_supply >= current_values * 0.72) & (current_values > float(deadband))).astype(float)
        enriched.loc[idx, "source_supply"] = source_supply
        enriched.loc[idx, "receiver_demand"] = receiver_demand
        enriched.loc[idx, "cash_buffer"] = float(decision["cash_buffer"])
        enriched.loc[idx, "turnover_budget"] = float(decision["turnover_budget"])
        enriched.loc[idx, "risk_budget"] = float(decision["risk_budget"])
        enriched.loc[idx, "constraint_violation"] = float(decision["constraint_violation"])
        enriched.loc[idx, "decision_value"] = (
            source_supply * source_score.loc[idx].to_numpy(dtype=float)
            + receiver_demand * receiver_score.loc[idx].to_numpy(dtype=float)
        )
        enriched.loc[idx, "deploy_value"] = deploy_value.loc[idx].to_numpy(dtype=float)
        enriched.loc[idx, "release_value"] = release_value.loc[idx].to_numpy(dtype=float)
        enriched.loc[idx, "defense_value"] = defense_value.loc[idx].to_numpy(dtype=float)
        enriched.loc[idx, "cash_timing_value"] = cash_timing_value.loc[idx].to_numpy(dtype=float)
        enriched.loc[idx, "source_opportunity_cost"] = source_opportunity_cost.loc[idx].to_numpy(dtype=float)
        enriched.loc[idx, "receiver_source_spread_value"] = receiver_source_spread_value.loc[idx].to_numpy(dtype=float)
        enriched.loc[idx, "reversal_risk_penalty"] = reversal_risk_penalty.loc[idx].to_numpy(dtype=float)
        enriched.loc[idx, "source_wrong_side_sell_penalty"] = positive_forward_sell_penalty.loc[idx].to_numpy(dtype=float)
    enriched["target_delta"] = (enriched["target_weight"] - current).where(lambda s: s.abs() >= float(deadband), 0.0)
    enriched["target_weight"] = (current + enriched["target_delta"]).clip(0.0, 0.24)
    enriched["held_mask"] = held.astype(float)
    enriched["source_mask"] = (enriched["source_supply"] > float(deadband)).astype(float)
    enriched["receiver_mask"] = (enriched["receiver_demand"] > float(deadband)).astype(float)
    enriched["target_delta_weight_conflict"] = (
        ((enriched["target_weight"] - current) * enriched["target_delta"] < -(float(deadband) ** 2))
    ).astype(float)
    return enriched.astype(float)


def _resolve_static_and_sequence_columns(feature_names: list[str]) -> tuple[list[str], list[str], list[str]]:
    try:
        sequence_bases, sequence_columns = resolve_sequence_columns(feature_names)
    except ValueError:
        return list(feature_names), [], []
    static_columns = [name for name in feature_names if name not in set(sequence_columns)]
    if not static_columns:
        static_columns = list(feature_names)
        sequence_bases = []
        sequence_columns = []
    return static_columns, sequence_bases, sequence_columns


def _sequence_column_name(base_name: str, step: int) -> str:
    return str(base_name) if int(step) == 0 else f"{base_name}_lag{int(step)}"


def _build_sequence_array(frame: pd.DataFrame, sequence_bases: list[str], *, fill: np.ndarray, means: np.ndarray, stds: np.ndarray) -> np.ndarray:
    if not sequence_bases:
        return np.zeros((len(frame), 1, 1), dtype=np.float32)
    steps = list(SEQUENCE_STEP_ORDER)
    matrices: list[np.ndarray] = []
    offset = 0
    for step in steps:
        cols = [_sequence_column_name(base, step) for base in sequence_bases]
        raw = frame.reindex(columns=cols).replace([np.inf, -np.inf], np.nan).to_numpy(dtype=np.float32)
        step_fill = fill[offset : offset + len(cols)]
        step_means = means[offset : offset + len(cols)]
        step_stds = stds[offset : offset + len(cols)]
        raw = np.where(np.isfinite(raw), raw, step_fill)
        raw = (raw - step_means) / step_stds
        matrices.append(raw.astype(np.float32))
        offset += len(cols)
    return np.stack(matrices, axis=1).astype(np.float32)


def _daily_lookup(daily_frame: pd.DataFrame) -> dict[str, pd.Series]:
    day_col = _date_column(daily_frame)
    return {str(row[day_col]): row for _, row in daily_frame.iterrows()}


class PortfolioSetDayDataset(Dataset[dict[str, np.ndarray]]):
    def __init__(
        self,
        *,
        sample_frame: pd.DataFrame,
        daily_frame: pd.DataFrame,
        static_feature_names: list[str],
        sequence_bases: list[str],
        daily_feature_names: list[str],
        targets: pd.DataFrame,
        static_fill: np.ndarray,
        static_means: np.ndarray,
        static_stds: np.ndarray,
        sequence_fill: np.ndarray,
        sequence_means: np.ndarray,
        sequence_stds: np.ndarray,
        daily_fill: np.ndarray,
        daily_means: np.ndarray,
        daily_stds: np.ndarray,
        max_stocks_per_day: int = PORTFOLIO_SET_V5_MAX_STOCKS_PER_DAY,
    ) -> None:
        self.sample_frame = sample_frame.copy()
        self.targets = targets.copy()
        self.daily_by_date = _daily_lookup(daily_frame)
        self.static_feature_names = list(static_feature_names)
        self.sequence_bases = list(sequence_bases)
        self.daily_feature_names = list(daily_feature_names)
        self.static_fill = np.asarray(static_fill, dtype=np.float32)
        self.static_means = np.asarray(static_means, dtype=np.float32)
        self.static_stds = np.asarray(static_stds, dtype=np.float32)
        self.sequence_fill = np.asarray(sequence_fill, dtype=np.float32)
        self.sequence_means = np.asarray(sequence_means, dtype=np.float32)
        self.sequence_stds = np.asarray(sequence_stds, dtype=np.float32)
        self.daily_fill = np.asarray(daily_fill, dtype=np.float32)
        self.daily_means = np.asarray(daily_means, dtype=np.float32)
        self.daily_stds = np.asarray(daily_stds, dtype=np.float32)
        self.max_stocks_per_day = max(1, int(max_stocks_per_day or 1))
        date_col = _date_column(self.sample_frame)
        self.date_col = date_col
        self.dates = sorted(str(item) for item in self.sample_frame[date_col].astype(str).unique())

    def __len__(self) -> int:
        return len(self.dates)

    def __getitem__(self, index: int) -> dict[str, np.ndarray]:
        date_text = self.dates[int(index)]
        day = self.sample_frame[self.sample_frame[self.date_col].astype(str) == date_text].copy()
        if len(day) > self.max_stocks_per_day:
            day = day.head(self.max_stocks_per_day).copy()
        target = self.targets.loc[day.index]
        static = day.reindex(columns=self.static_feature_names).replace([np.inf, -np.inf], np.nan).to_numpy(dtype=np.float32)
        static = np.where(np.isfinite(static), static, self.static_fill)
        static = ((static - self.static_means) / self.static_stds).astype(np.float32)
        sequence = _build_sequence_array(
            day,
            self.sequence_bases,
            fill=self.sequence_fill,
            means=self.sequence_means,
            stds=self.sequence_stds,
        )
        daily_row = self.daily_by_date.get(date_text)
        if daily_row is None:
            daily_raw = np.full(len(self.daily_feature_names), np.nan, dtype=np.float32)
        else:
            daily_raw = pd.to_numeric(daily_row.reindex(self.daily_feature_names), errors="coerce").to_numpy(dtype=np.float32)
        daily_raw = np.where(np.isfinite(daily_raw), daily_raw, self.daily_fill)
        daily = ((daily_raw - self.daily_means) / self.daily_stds).astype(np.float32)
        return {
            "static_x": static,
            "sequence_x": sequence,
            "daily_x": daily,
            "current_weight": _current_weight(day).to_numpy(dtype=np.float32),
            "target_y": target.reindex(columns=PORTFOLIO_SET_V5_OUTPUT_NAMES).to_numpy(dtype=np.float32),
            "decision_target_y": target.reindex(columns=PORTFOLIO_SET_V5_DECISION_TARGET_NAMES).to_numpy(dtype=np.float32),
            "held_mask": target["held_mask"].to_numpy(dtype=np.float32),
            "source_mask": target["source_mask"].to_numpy(dtype=np.float32),
            "receiver_mask": target["receiver_mask"].to_numpy(dtype=np.float32),
        }


def collate_portfolio_set_days(batch: list[dict[str, np.ndarray]]) -> dict[str, torch.Tensor]:
    max_items = max(int(item["static_x"].shape[0]) for item in batch)
    batch_size = len(batch)
    static_dim = int(batch[0]["static_x"].shape[1])
    sequence_steps = int(batch[0]["sequence_x"].shape[1])
    sequence_dim = int(batch[0]["sequence_x"].shape[2])
    output_dim = len(PORTFOLIO_SET_V5_OUTPUT_NAMES)
    decision_output_dim = len(PORTFOLIO_SET_V5_DECISION_TARGET_NAMES)
    static_x = np.zeros((batch_size, max_items, static_dim), dtype=np.float32)
    sequence_x = np.zeros((batch_size, max_items, sequence_steps, sequence_dim), dtype=np.float32)
    target_y = np.zeros((batch_size, max_items, output_dim), dtype=np.float32)
    decision_target_y = np.zeros((batch_size, max_items, decision_output_dim), dtype=np.float32)
    current_weight = np.zeros((batch_size, max_items), dtype=np.float32)
    held_mask = np.zeros((batch_size, max_items), dtype=np.float32)
    source_mask = np.zeros((batch_size, max_items), dtype=np.float32)
    receiver_mask = np.zeros((batch_size, max_items), dtype=np.float32)
    sample_mask = np.zeros((batch_size, max_items), dtype=bool)
    daily_x = np.stack([item["daily_x"] for item in batch]).astype(np.float32)
    for batch_idx, item in enumerate(batch):
        size = int(item["static_x"].shape[0])
        static_x[batch_idx, :size] = item["static_x"]
        sequence_x[batch_idx, :size] = item["sequence_x"]
        target_y[batch_idx, :size] = item["target_y"]
        decision_target_y[batch_idx, :size] = item["decision_target_y"]
        current_weight[batch_idx, :size] = item["current_weight"]
        held_mask[batch_idx, :size] = item["held_mask"]
        source_mask[batch_idx, :size] = item["source_mask"]
        receiver_mask[batch_idx, :size] = item["receiver_mask"]
        sample_mask[batch_idx, :size] = True
    return {
        "static_x": torch.as_tensor(static_x, dtype=torch.float32),
        "sequence_x": torch.as_tensor(sequence_x, dtype=torch.float32),
        "daily_x": torch.as_tensor(daily_x, dtype=torch.float32),
        "target_y": torch.as_tensor(target_y, dtype=torch.float32),
        "decision_target_y": torch.as_tensor(decision_target_y, dtype=torch.float32),
        "current_weight": torch.as_tensor(current_weight, dtype=torch.float32),
        "held_mask": torch.as_tensor(held_mask, dtype=torch.float32),
        "source_mask": torch.as_tensor(source_mask, dtype=torch.float32),
        "receiver_mask": torch.as_tensor(receiver_mask, dtype=torch.float32),
        "sample_mask": torch.as_tensor(sample_mask, dtype=torch.bool),
    }


class TemporalEncoder(nn.Module):
    def __init__(self, *, input_dim: int, model_dim: int, layers: int, dropout: float) -> None:
        super().__init__()
        self.input_dim = int(input_dim)
        self.model_dim = int(model_dim)
        self.input_proj = nn.Linear(int(input_dim), int(model_dim))
        self.gru = nn.GRU(
            input_size=int(model_dim),
            hidden_size=int(model_dim),
            num_layers=max(1, int(layers)),
            batch_first=True,
            dropout=float(dropout) if int(layers) > 1 else 0.0,
        )
        self.norm = nn.LayerNorm(int(model_dim))

    def forward(self, sequence_x: torch.Tensor) -> torch.Tensor:
        bsz, count, steps, dim = sequence_x.shape
        encoded = self.input_proj(sequence_x.reshape(bsz * count, steps, dim))
        _, hidden = self.gru(encoded)
        return self.norm(hidden[-1].reshape(bsz, count, self.model_dim))


class LatentSetEncoder(nn.Module):
    def __init__(self, *, model_dim: int, latent_count: int, heads: int, layers: int, dropout: float) -> None:
        super().__init__()
        self.latents = nn.Parameter(torch.randn(1, max(1, int(latent_count)), int(model_dim)) * 0.02)
        self.cross_attn = nn.MultiheadAttention(int(model_dim), max(1, int(heads)), batch_first=True, dropout=float(dropout))
        self.self_layers = nn.ModuleList(
            [
                nn.TransformerEncoderLayer(
                    d_model=int(model_dim),
                    nhead=max(1, int(heads)),
                    dim_feedforward=int(model_dim) * 4,
                    dropout=float(dropout),
                    batch_first=True,
                    activation="gelu",
                    norm_first=True,
                )
                for _ in range(max(1, int(layers)))
            ]
        )
        self.norm = nn.LayerNorm(int(model_dim))

    def forward(self, tokens: torch.Tensor, sample_mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        batch_size = int(tokens.shape[0])
        latents = self.latents.expand(batch_size, -1, -1)
        key_padding_mask = ~sample_mask.bool()
        latents, _ = self.cross_attn(latents, tokens, tokens, key_padding_mask=key_padding_mask)
        for layer in self.self_layers:
            latents = layer(latents)
        context = self.norm(latents.mean(dim=1))
        return latents, context


class PortfolioSetPolicyNetV5(nn.Module):
    def __init__(
        self,
        *,
        static_input_dim: int,
        sequence_input_dim: int,
        daily_input_dim: int,
        model_dim: int = 192,
        temporal_layers: int = 2,
        cross_layers: int = 2,
        latent_count: int = 64,
        dropout: float = 0.18,
    ) -> None:
        super().__init__()
        self.model_dim = int(model_dim)
        self.latent_count = int(latent_count)
        self.static_encoder = nn.Sequential(
            nn.Linear(int(static_input_dim), int(model_dim)),
            nn.GELU(),
            nn.Dropout(float(dropout)),
            nn.Linear(int(model_dim), int(model_dim)),
            nn.GELU(),
        )
        self.temporal_encoder = TemporalEncoder(
            input_dim=int(sequence_input_dim),
            model_dim=int(model_dim),
            layers=int(temporal_layers),
            dropout=float(dropout),
        )
        self.daily_encoder = nn.Sequential(
            nn.Linear(int(daily_input_dim), int(model_dim)),
            nn.GELU(),
            nn.Dropout(float(dropout)),
            nn.Linear(int(model_dim), int(model_dim)),
            nn.GELU(),
        )
        self.stock_fusion = nn.Sequential(
            nn.Linear(int(model_dim) * 2, int(model_dim)),
            nn.GELU(),
            nn.Dropout(float(dropout)),
        )
        self.set_encoder = LatentSetEncoder(
            model_dim=int(model_dim),
            latent_count=int(latent_count),
            heads=4,
            layers=int(cross_layers),
            dropout=float(dropout),
        )
        self.stock_context = nn.Sequential(
            nn.Linear(int(model_dim) * 3, int(model_dim)),
            nn.GELU(),
            nn.Dropout(float(dropout)),
            nn.Linear(int(model_dim), int(model_dim)),
            nn.GELU(),
        )
        self.head = nn.Linear(int(model_dim), len(PORTFOLIO_SET_V5_OUTPUT_NAMES))

    def forward(
        self,
        static_x: torch.Tensor,
        sequence_x: torch.Tensor,
        daily_x: torch.Tensor,
        sample_mask: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        static_hidden = self.static_encoder(static_x)
        temporal_hidden = self.temporal_encoder(sequence_x)
        stock_hidden = self.stock_fusion(torch.cat([static_hidden, temporal_hidden], dim=-1))
        _, set_context = self.set_encoder(stock_hidden, sample_mask)
        daily_context = self.daily_encoder(daily_x)
        repeated_set = set_context.unsqueeze(1).expand(-1, stock_hidden.shape[1], -1)
        repeated_daily = daily_context.unsqueeze(1).expand(-1, stock_hidden.shape[1], -1)
        stock_context = self.stock_context(torch.cat([stock_hidden, repeated_set, repeated_daily], dim=-1))
        return {"raw": self.head(stock_context)}


@dataclass
class TorchPortfolioSetV5Artifact:
    feature_names: list[str]
    daily_feature_names: list[str]
    static_feature_names: list[str]
    sequence_bases: list[str]
    feature_fill_values: np.ndarray
    feature_means: np.ndarray
    feature_stds: np.ndarray
    sequence_fill_values: np.ndarray
    sequence_means: np.ndarray
    sequence_stds: np.ndarray
    daily_fill_values: np.ndarray
    daily_means: np.ndarray
    daily_stds: np.ndarray
    train_summary: dict[str, Any]
    training_diagnostics: dict[str, Any]
    training_contract: dict[str, Any]
    trained_at: str
    model_config: dict[str, Any] = field(default_factory=dict)
    model_state_dict: dict[str, Any] | None = None
    global_target_defaults: dict[str, float] = field(default_factory=dict)

    def save(self, path: Path) -> Path:
        path.parent.mkdir(parents=True, exist_ok=True)
        model_config = dict(self.model_config or {})
        model_config.setdefault("portfolio_set_v5_internal_version", PORTFOLIO_SET_V5_INTERNAL_VERSION)
        payload = {
            "artifact_type": PORTFOLIO_SET_V5_ARTIFACT_TYPE,
            "feature_names": list(self.feature_names),
            "daily_feature_names": list(self.daily_feature_names),
            "static_feature_names": list(self.static_feature_names),
            "sequence_bases": list(self.sequence_bases),
            "feature_fill_values": self.feature_fill_values.astype(np.float32).tolist(),
            "feature_means": self.feature_means.astype(np.float32).tolist(),
            "feature_stds": self.feature_stds.astype(np.float32).tolist(),
            "sequence_fill_values": self.sequence_fill_values.astype(np.float32).tolist(),
            "sequence_means": self.sequence_means.astype(np.float32).tolist(),
            "sequence_stds": self.sequence_stds.astype(np.float32).tolist(),
            "daily_fill_values": self.daily_fill_values.astype(np.float32).tolist(),
            "daily_means": self.daily_means.astype(np.float32).tolist(),
            "daily_stds": self.daily_stds.astype(np.float32).tolist(),
            "train_summary": self.train_summary,
            "training_diagnostics": self.training_diagnostics,
            "training_contract": self.training_contract,
            "trained_at": self.trained_at,
            "model_config": model_config,
            "model_state_dict": self.model_state_dict,
            "global_target_defaults": self.global_target_defaults,
        }
        torch.save(payload, path)
        return path


def load_torch_portfolio_set_v5_artifact(path: str | Path) -> TorchPortfolioSetV5Artifact:
    payload = torch.load(Path(path), map_location="cpu", weights_only=False)
    if str(payload.get("artifact_type", "") or "") != PORTFOLIO_SET_V5_ARTIFACT_TYPE:
        raise TypeError(f"Unsupported portfolio-set v5 artifact type: {payload.get('artifact_type')!r}")
    return TorchPortfolioSetV5Artifact(
        feature_names=list(payload.get("feature_names", []) or []),
        daily_feature_names=list(payload.get("daily_feature_names", []) or []),
        static_feature_names=list(payload.get("static_feature_names", []) or []),
        sequence_bases=list(payload.get("sequence_bases", []) or []),
        feature_fill_values=np.asarray(payload.get("feature_fill_values", []), dtype=np.float32),
        feature_means=np.asarray(payload.get("feature_means", []), dtype=np.float32),
        feature_stds=np.asarray(payload.get("feature_stds", []), dtype=np.float32),
        sequence_fill_values=np.asarray(payload.get("sequence_fill_values", []), dtype=np.float32),
        sequence_means=np.asarray(payload.get("sequence_means", []), dtype=np.float32),
        sequence_stds=np.asarray(payload.get("sequence_stds", []), dtype=np.float32),
        daily_fill_values=np.asarray(payload.get("daily_fill_values", []), dtype=np.float32),
        daily_means=np.asarray(payload.get("daily_means", []), dtype=np.float32),
        daily_stds=np.asarray(payload.get("daily_stds", []), dtype=np.float32),
        train_summary=dict(payload.get("train_summary", {}) or {}),
        training_diagnostics=dict(payload.get("training_diagnostics", {}) or {}),
        training_contract=dict(payload.get("training_contract", {}) or {}),
        trained_at=str(payload.get("trained_at", "") or ""),
        model_config=dict(payload.get("model_config", {}) or {}),
        model_state_dict=payload.get("model_state_dict"),
        global_target_defaults=dict(payload.get("global_target_defaults", {}) or {}),
    )


def _decode_raw(raw: torch.Tensor) -> dict[str, torch.Tensor]:
    return {
        "target_weight": torch.sigmoid(raw[..., 0]) * 0.24,
        "target_delta": torch.tanh(raw[..., 1]) * 0.18,
        "source_supply_score": torch.sigmoid(raw[..., 2]),
        "receiver_demand_score": torch.sigmoid(raw[..., 3]),
        "cash_buffer_score": torch.sigmoid(raw[..., 4]),
        "release_intent": torch.sigmoid(raw[..., 5]),
        "reduce_quality": torch.sigmoid(raw[..., 6]),
        "exit_hazard": torch.sigmoid(raw[..., 7]),
    }


def _masked_mean(value: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    weight = mask.to(dtype=value.dtype)
    while weight.ndim < value.ndim:
        weight = weight.unsqueeze(-1)
    return (value * weight).sum() / weight.sum().clamp_min(1.0)


def _decision_target_column(decision_target: torch.Tensor | None, index: int, fallback: torch.Tensor) -> torch.Tensor:
    if decision_target is None or decision_target.shape[-1] <= int(index):
        return fallback
    return decision_target[..., int(index)].to(device=fallback.device, dtype=fallback.dtype)


def _portfolio_set_loss(raw: torch.Tensor, batch: dict[str, torch.Tensor], weights: dict[str, float]) -> torch.Tensor:
    pred = _decode_raw(raw)
    target = batch["target_y"]
    decision_target = batch.get("decision_target_y")
    mask = batch["sample_mask"].float()
    current = batch["current_weight"].clamp(0.0, 1.0)
    target_weight = target[..., 0].clamp(0.0, 0.24)
    target_delta = target[..., 1].clamp(-0.18, 0.18)
    source_target = target[..., 2].clamp(0.0, 1.0)
    receiver_target = target[..., 3].clamp(0.0, 1.0)
    cash_target = target[..., 4].clamp(0.0, 1.0)
    release_target = target[..., 5].clamp(0.0, 1.0)
    reduce_target = target[..., 6].clamp(0.0, 1.0)
    exit_target = target[..., 7].clamp(0.0, 1.0)
    source_supply = pred["source_supply_score"] * pred["release_intent"] * (current > 0.003).float()
    receiver_demand = pred["receiver_demand_score"] * (current < 0.24 - 0.003).float()
    day_source = (source_supply * mask).sum(dim=1)
    day_receiver = (receiver_demand * mask).sum(dim=1)
    target_day_source = (source_target * release_target * mask).sum(dim=1)
    target_day_receiver = (receiver_target * mask).sum(dim=1)
    pred_delta_from_weight = pred["target_weight"] - current
    conflict = torch.relu(-(pred_delta_from_weight * pred["target_delta"]) - (0.003 ** 2))
    zero_target = torch.zeros_like(current)
    oracle_source = _decision_target_column(decision_target, 0, source_target * current).clamp(0.0, 1.0)
    oracle_receiver = _decision_target_column(
        decision_target,
        1,
        receiver_target * (0.24 - current).clamp_min(0.0),
    ).clamp(0.0, 1.0)
    oracle_cash = _decision_target_column(decision_target, 2, cash_target).clamp(0.0, 1.0)
    oracle_target_weight = _decision_target_column(decision_target, 3, target_weight).clamp(0.0, 0.24)
    target_value = (_decision_target_column(decision_target, 7, zero_target).clamp_min(0.0) * mask).sum(dim=1)
    deploy_value = _decision_target_column(decision_target, 8, zero_target).clamp(0.0, 1.0)
    release_value = _decision_target_column(decision_target, 9, zero_target).clamp(0.0, 1.0)
    defense_value = _decision_target_column(decision_target, 10, cash_target).clamp(0.0, 1.0)
    cash_timing_value = _decision_target_column(decision_target, 11, cash_target).clamp(0.0, 1.0)
    source_opportunity_cost = _decision_target_column(decision_target, 12, zero_target).clamp(0.0, 1.0)
    receiver_source_spread_value = _decision_target_column(decision_target, 13, zero_target).clamp(0.0, 1.0)
    reversal_risk_penalty = _decision_target_column(decision_target, 14, zero_target).clamp(0.0, 1.0)
    wrong_side_penalty = _decision_target_column(decision_target, 15, zero_target).clamp(0.0, 1.0)
    oracle = project_portfolio_set_v5_cashflow_oracle(
        current_weight=current,
        source_score=pred["source_supply_score"],
        receiver_score=pred["receiver_demand_score"],
        cash_buffer_score=pred["cash_buffer_score"],
        deploy_value=deploy_value,
        release_value=release_value,
        defense_value=defense_value,
        cash_timing_value=cash_timing_value,
        source_opportunity_cost=source_opportunity_cost,
        receiver_source_spread_value=receiver_source_spread_value,
        reversal_risk_penalty=reversal_risk_penalty,
        source_wrong_side_sell_penalty=wrong_side_penalty,
        sample_mask=batch["sample_mask"],
    )
    oracle_target_loss = (
        _masked_mean(nn.functional.smooth_l1_loss(oracle["target_weight"], oracle_target_weight, reduction="none"), mask)
        + _masked_mean(nn.functional.smooth_l1_loss(oracle["source_supply"], oracle_source, reduction="none"), mask)
        + _masked_mean(nn.functional.smooth_l1_loss(oracle["receiver_demand"], oracle_receiver, reduction="none"), mask)
        + nn.functional.smooth_l1_loss(oracle["cash_buffer"], (oracle_cash * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1.0))
    )
    value_gap = torch.relu(target_value - oracle["decision_value"]).mean()
    perturb_strength = 0.05
    plus_oracle = project_portfolio_set_v5_cashflow_oracle(
        current_weight=current,
        source_score=(pred["source_supply_score"] + perturb_strength * source_target).clamp(0.0, 1.0),
        receiver_score=(pred["receiver_demand_score"] + perturb_strength * receiver_target).clamp(0.0, 1.0),
        cash_buffer_score=pred["cash_buffer_score"],
        deploy_value=deploy_value,
        release_value=release_value,
        defense_value=defense_value,
        cash_timing_value=cash_timing_value,
        source_opportunity_cost=source_opportunity_cost,
        receiver_source_spread_value=receiver_source_spread_value,
        reversal_risk_penalty=reversal_risk_penalty,
        source_wrong_side_sell_penalty=wrong_side_penalty,
        sample_mask=batch["sample_mask"],
    )
    minus_oracle = project_portfolio_set_v5_cashflow_oracle(
        current_weight=current,
        source_score=(pred["source_supply_score"] - perturb_strength * source_target).clamp(0.0, 1.0),
        receiver_score=(pred["receiver_demand_score"] - perturb_strength * receiver_target).clamp(0.0, 1.0),
        cash_buffer_score=pred["cash_buffer_score"],
        deploy_value=deploy_value,
        release_value=release_value,
        defense_value=defense_value,
        cash_timing_value=cash_timing_value,
        source_opportunity_cost=source_opportunity_cost,
        receiver_source_spread_value=receiver_source_spread_value,
        reversal_risk_penalty=reversal_risk_penalty,
        source_wrong_side_sell_penalty=wrong_side_penalty,
        sample_mask=batch["sample_mask"],
    )
    pg_margin = plus_oracle["decision_value"] - minus_oracle["decision_value"]
    pg_surrogate = torch.relu(0.001 - pg_margin).mean()
    source_alloc = oracle["source_supply"]
    receiver_alloc = oracle["receiver_demand"]
    value_arbitration_loss = (
        _masked_mean(nn.functional.binary_cross_entropy_with_logits(raw[..., 2], (release_value * (1.0 - source_opportunity_cost) * (1.0 - wrong_side_penalty)).clamp(0.0, 1.0), reduction="none"), mask)
        + _masked_mean(nn.functional.binary_cross_entropy_with_logits(raw[..., 3], (deploy_value * (1.0 - defense_value)).clamp(0.0, 1.0), reduction="none"), mask)
        + _masked_mean(nn.functional.smooth_l1_loss(pred["reduce_quality"], release_value.clamp(0.0, 1.0), reduction="none"), mask)
        + _masked_mean(nn.functional.smooth_l1_loss(pred["exit_hazard"], (defense_value * release_value).clamp(0.0, 1.0), reduction="none"), mask)
    )
    cash_timing_loss = _masked_mean(
        nn.functional.smooth_l1_loss(pred["cash_buffer_score"], torch.maximum(defense_value, cash_timing_value), reduction="none"),
        mask,
    )
    reversal_guard_loss = (
        _masked_mean(source_alloc * (reversal_risk_penalty + wrong_side_penalty + source_opportunity_cost), mask)
        + _masked_mean(receiver_alloc * defense_value, mask) * 0.25
    )
    loss = (
        float(weights["target_weight_closure_total"]) * _masked_mean(nn.functional.smooth_l1_loss(pred["target_weight"], target_weight, reduction="none"), mask)
        + float(weights["source_supply_total"]) * _masked_mean(nn.functional.binary_cross_entropy_with_logits(raw[..., 2], source_target, reduction="none"), mask)
        + float(weights["receiver_demand_total"]) * _masked_mean(nn.functional.binary_cross_entropy_with_logits(raw[..., 3], receiver_target, reduction="none"), mask)
        + float(weights["cash_buffer_total"]) * _masked_mean(nn.functional.smooth_l1_loss(pred["cash_buffer_score"], cash_target, reduction="none"), mask)
        + float(weights["target_delta_weight_coherence_total"]) * _masked_mean(nn.functional.smooth_l1_loss(pred["target_delta"], target_delta, reduction="none") + conflict, mask)
        + float(weights["release_flow_balance_total"])
        * (
            nn.functional.smooth_l1_loss(day_source, target_day_source)
            + nn.functional.smooth_l1_loss(day_receiver, target_day_receiver)
            + torch.relu(day_source - day_receiver - pred["cash_buffer_score"].mean(dim=1) - 0.10).mean()
        )
        + float(weights["underdeployment_high_cash_total"]) * torch.relu(target_weight.sum(dim=1) - pred["target_weight"].sum(dim=1)).mean()
        + float(weights["intent_translation_conflict_total"]) * _masked_mean(conflict, mask)
        + float(weights.get("decision_oracle_total", 0.0)) * (oracle_target_loss + value_gap)
        + float(weights.get("pg_dfl_surrogate_total", 0.0)) * pg_surrogate
        + float(weights.get("constraint_violation_total", 0.0)) * oracle["constraint_violation"].mean()
        + float(weights.get("value_arbitration_total", 0.0)) * value_arbitration_loss
        + float(weights.get("cash_timing_value_total", 0.0)) * cash_timing_loss
        + float(weights.get("reversal_guard_total", 0.0)) * reversal_guard_loss
        + 0.10 * _masked_mean(nn.functional.binary_cross_entropy_with_logits(raw[..., 5], release_target, reduction="none"), mask)
        + 0.08 * _masked_mean(nn.functional.binary_cross_entropy_with_logits(raw[..., 6], reduce_target, reduction="none"), mask)
        + 0.08 * _masked_mean(nn.functional.binary_cross_entropy_with_logits(raw[..., 7], exit_target, reduction="none"), mask)
    )
    return loss


def portfolio_set_v5_decision_diagnostics(
    raw: torch.Tensor,
    batch: dict[str, torch.Tensor],
    weights: dict[str, float] | None = None,
) -> dict[str, float]:
    with torch.no_grad():
        pred = _decode_raw(raw)
        mask = batch["sample_mask"].float()
        current = batch["current_weight"].clamp(0.0, 1.0)
        target = batch.get("decision_target_y")
        zero_target = torch.zeros_like(current)
        cash_target = batch["target_y"][..., 4].clamp(0.0, 1.0)
        target_source = _decision_target_column(target, 0, zero_target).clamp(0.0, 1.0)
        target_receiver = _decision_target_column(target, 1, zero_target).clamp(0.0, 1.0)
        target_weight = _decision_target_column(target, 3, batch["target_y"][..., 0].clamp(0.0, 0.24)).clamp(0.0, 0.24)
        target_value = (_decision_target_column(target, 7, zero_target).clamp_min(0.0) * mask).sum(dim=1)
        deploy_value = _decision_target_column(target, 8, zero_target).clamp(0.0, 1.0)
        release_value = _decision_target_column(target, 9, zero_target).clamp(0.0, 1.0)
        defense_value = _decision_target_column(target, 10, cash_target).clamp(0.0, 1.0)
        cash_timing_value = _decision_target_column(target, 11, cash_target).clamp(0.0, 1.0)
        opportunity_cost = _decision_target_column(target, 12, zero_target).clamp(0.0, 1.0)
        spread_value = _decision_target_column(target, 13, zero_target).clamp(0.0, 1.0)
        reversal_penalty = _decision_target_column(target, 14, zero_target).clamp(0.0, 1.0)
        wrong_side_penalty = _decision_target_column(target, 15, zero_target).clamp(0.0, 1.0)
        oracle = project_portfolio_set_v5_cashflow_oracle(
            current_weight=current,
            source_score=pred["source_supply_score"],
            receiver_score=pred["receiver_demand_score"],
            cash_buffer_score=pred["cash_buffer_score"],
            deploy_value=deploy_value,
            release_value=release_value,
            defense_value=defense_value,
            cash_timing_value=cash_timing_value,
            source_opportunity_cost=opportunity_cost,
            receiver_source_spread_value=spread_value,
            reversal_risk_penalty=reversal_penalty,
            source_wrong_side_sell_penalty=wrong_side_penalty,
            sample_mask=batch["sample_mask"],
        )
        source_count = ((oracle["source_supply"] > 0.003) & batch["sample_mask"]).sum().item()
        receiver_count = ((oracle["receiver_demand"] > 0.003) & batch["sample_mask"]).sum().item()
        conflict = ((oracle["target_weight"] - current) * oracle["target_delta"] < -(0.003 ** 2)) & batch["sample_mask"]
        source_target_mask = (oracle["source_supply"] > 0.003) & batch["sample_mask"]
        deploy_release_spread = deploy_value - release_value
        source_wrong_side_share = (
            float(((wrong_side_penalty > 0.5) & source_target_mask).sum().item()) / max(float(source_count), 1.0)
        )
        defense_win_rate = (
            float(((oracle["cash_buffer"] >= oracle["cash_floor"] - 1.0e-6) & (defense_value.mean(dim=1) > 0.5)).sum().item())
            / max(float((defense_value.mean(dim=1) > 0.5).sum().item()), 1.0)
        )
        return {
            "paper_reproduction_pg_surrogate_loss": float(_portfolio_set_loss(raw, batch, weights or resolve_portfolio_set_v5_loss_profile(None)[1]["multi_objective_loss_weights"]).detach().cpu()),
            "decision_oracle_value_mean": float(oracle["decision_value"].mean().detach().cpu()),
            "decision_oracle_target_value_mean": float(target_value.mean().detach().cpu()),
            "decision_oracle_constraint_violation_mean": float(oracle["constraint_violation"].mean().detach().cpu()),
            "decision_oracle_source_l1": float(_masked_mean((oracle["source_supply"] - target_source).abs(), mask).detach().cpu()),
            "decision_oracle_receiver_l1": float(_masked_mean((oracle["receiver_demand"] - target_receiver).abs(), mask).detach().cpu()),
            "decision_oracle_target_weight_l1": float(_masked_mean((oracle["target_weight"] - target_weight).abs(), mask).detach().cpu()),
            "release_flow_source_target_count": float(source_count),
            "release_flow_receiver_target_count": float(receiver_count),
            "release_flow_intent_translation_conflict_count": float(conflict.sum().item()),
            "value_arbitration_deploy_release_spread_mean": float(_masked_mean(deploy_release_spread, mask).detach().cpu()),
            "value_arbitration_source_wrong_side_sell_share": float(source_wrong_side_share),
            "value_arbitration_cash_timing_decision_quality": float(_masked_mean(oracle["cash_buffer"].unsqueeze(1) * torch.maximum(defense_value, cash_timing_value), mask).detach().cpu()),
            "value_arbitration_reversal_risk_penalty": float(_masked_mean(oracle["source_supply"] * reversal_penalty, mask).detach().cpu()),
            "value_arbitration_defense_win_rate": float(defense_win_rate),
            "r69_reversal_guarded_count": float(((reversal_penalty > 0.5) & source_target_mask).sum().item()),
        }


def _merge_portfolio_set_v5_diagnostics(items: list[dict[str, float]]) -> dict[str, float]:
    if not items:
        return {}
    sum_keys = {
        "release_flow_source_target_count",
        "release_flow_receiver_target_count",
        "release_flow_intent_translation_conflict_count",
        "r69_reversal_guarded_count",
    }
    merged: dict[str, float] = {}
    keys = sorted({key for item in items for key in item})
    for key in keys:
        values = [float(item.get(key, 0.0) or 0.0) for item in items]
        merged[key] = float(sum(values) if key in sum_keys else np.mean(values))
    return merged


def _global_defaults(daily_frame: pd.DataFrame) -> dict[str, float]:
    defaults: dict[str, float] = {
        "release_first_allocation_v3_mode": 1.0,
        "allocation_intent_v2_mode": 1.0,
        "gross_exposure_target": 0.72,
        "candidate_budget": 8.0,
        "turnover_budget": 0.24,
        "max_position_weight_target": 0.24,
    }
    for key in ("gross_exposure_target", "candidate_budget", "turnover_budget", "max_position_weight_target"):
        if key in daily_frame.columns and len(daily_frame):
            defaults[key] = float(pd.to_numeric(daily_frame[key], errors="coerce").replace([np.inf, -np.inf], np.nan).dropna().median())
    return defaults


def _scenario_day_candidates(sample_frame: pd.DataFrame, *, deadband: float = 0.003) -> dict[str, list[str]]:
    date_col = _date_column(sample_frame)
    dates = sample_frame[date_col].astype(str)
    current = _current_weight(sample_frame)
    held = _holding_mask(sample_frame, current, deadband=deadband)
    action = sample_frame.get("action_label", pd.Series("hold", index=sample_frame.index)).fillna("hold").astype(str).str.lower()
    receiver_signal = _max_numeric_columns(
        sample_frame,
        (
            "portfolio_daily_receiver_score",
            "portfolio_daily_unified_receiver_score",
            "deploy_value_target",
            "result_value_deploy_value_target",
            "result_value_alpha_opportunity_value",
        ),
    )
    source_signal = _max_numeric_columns(
        sample_frame,
        (
            "portfolio_daily_source_score",
            "portfolio_daily_unified_source_score",
            "portfolio_daily_source_release_preference",
            "portfolio_daily_source_release_quality",
            "release_value_target",
            "sell_release_value",
        ),
    )
    risk_signal = _max_numeric_columns(
        sample_frame,
        (
            "cash_defense_value",
            "market_downside_pressure",
            "portfolio_daily_allocation_drawdown_control_target",
            "portfolio_daily_allocation_tail_risk_control_target",
            "cash_regime_pressure",
        ),
    )
    headroom = (0.24 - current).clip(lower=0.0)
    receiver_mask = (headroom > float(deadband)) & ((receiver_signal > 0.08) | action.isin({"open", "add"}))
    source_mask = held & ((source_signal > 0.02) | action.isin({"reduce", "exit"}))
    flat_receiver_mask = (current <= float(deadband)) & receiver_mask
    cash_only_mask = (current <= float(deadband)) & (receiver_signal <= 0.02) & (risk_signal <= 0.10)
    risk_off_mask = risk_signal > 0.35
    day_flags = pd.DataFrame(
        {
            "date": dates,
            "held_source": source_mask.astype(bool),
            "flat_receiver": flat_receiver_mask.astype(bool),
            "cash_only": cash_only_mask.astype(bool),
            "risk_off": risk_off_mask.astype(bool),
            "receiver": receiver_mask.astype(bool),
            "source": source_mask.astype(bool),
        }
    ).groupby("date", sort=True).any()
    day_flags["source_funded_rotation"] = day_flags["source"] & day_flags["receiver"]
    return {
        "held_source": day_flags.index[day_flags["held_source"]].astype(str).tolist(),
        "flat_receiver": day_flags.index[day_flags["flat_receiver"]].astype(str).tolist(),
        "cash_only": day_flags.index[day_flags["cash_only"]].astype(str).tolist(),
        "risk_off": day_flags.index[day_flags["risk_off"]].astype(str).tolist(),
        "source_funded_rotation": day_flags.index[day_flags["source_funded_rotation"]].astype(str).tolist(),
    }


def _select_train_days(sample_frame: pd.DataFrame, random_seed: int, max_days: int = PORTFOLIO_SET_V5_MAX_TRAIN_DAYS) -> tuple[pd.DataFrame, dict[str, int]]:
    date_col = _date_column(sample_frame)
    dates = sorted(str(item) for item in sample_frame[date_col].astype(str).unique())
    raw_day_count = len(dates)
    if raw_day_count <= int(max_days):
        return sample_frame.copy(), {"portfolio_set_v5_raw_train_day_count": raw_day_count, "portfolio_set_v5_train_day_count": raw_day_count}
    rng = np.random.default_rng(int(random_seed))
    selected_set: set[str] = set()
    scenario_days = _scenario_day_candidates(sample_frame)
    per_scenario_cap = max(4, int(max_days) // 8)
    for name in ("held_source", "flat_receiver", "cash_only", "risk_off", "source_funded_rotation"):
        candidates = np.asarray(scenario_days.get(name, []), dtype=object)
        if candidates.size == 0:
            continue
        take = min(int(per_scenario_cap), int(candidates.size), max(0, int(max_days) - len(selected_set)))
        if take <= 0:
            break
        selected_set.update(str(item) for item in rng.choice(candidates, size=take, replace=False).tolist())
    remaining = [item for item in dates if item not in selected_set]
    fill = max(0, int(max_days) - len(selected_set))
    if fill:
        selected_set.update(str(item) for item in rng.choice(np.asarray(remaining, dtype=object), size=min(fill, len(remaining)), replace=False).tolist())
    selected = sorted(selected_set)
    return sample_frame[sample_frame[date_col].astype(str).isin(selected)].copy(), {
        "portfolio_set_v5_raw_train_day_count": raw_day_count,
        "portfolio_set_v5_train_day_count": len(selected),
        "portfolio_set_v5_train_day_cap": int(max_days),
        "portfolio_set_v5_scenario_held_source_day_count": len(set(scenario_days.get("held_source", [])) & set(selected)),
        "portfolio_set_v5_scenario_flat_receiver_day_count": len(set(scenario_days.get("flat_receiver", [])) & set(selected)),
        "portfolio_set_v5_scenario_cash_only_day_count": len(set(scenario_days.get("cash_only", [])) & set(selected)),
        "portfolio_set_v5_scenario_risk_off_day_count": len(set(scenario_days.get("risk_off", [])) & set(selected)),
        "portfolio_set_v5_scenario_source_funded_rotation_day_count": len(set(scenario_days.get("source_funded_rotation", [])) & set(selected)),
    }


def _target_diagnostics(targets: pd.DataFrame) -> dict[str, float]:
    source = pd.to_numeric(targets.get("source_supply", pd.Series(dtype=float)), errors="coerce").fillna(0.0)
    receiver = pd.to_numeric(targets.get("receiver_demand", pd.Series(dtype=float)), errors="coerce").fillna(0.0)
    conflict = pd.to_numeric(targets.get("target_delta_weight_conflict", pd.Series(dtype=float)), errors="coerce").fillna(0.0)
    violation = pd.to_numeric(targets.get("constraint_violation", pd.Series(dtype=float)), errors="coerce").fillna(0.0)
    value = pd.to_numeric(targets.get("decision_value", pd.Series(dtype=float)), errors="coerce").fillna(0.0)
    deploy = pd.to_numeric(targets.get("deploy_value", pd.Series(dtype=float)), errors="coerce").fillna(0.0)
    release = pd.to_numeric(targets.get("release_value", pd.Series(dtype=float)), errors="coerce").fillna(0.0)
    defense = pd.to_numeric(targets.get("defense_value", pd.Series(dtype=float)), errors="coerce").fillna(0.0)
    cash_timing = pd.to_numeric(targets.get("cash_timing_value", pd.Series(dtype=float)), errors="coerce").fillna(0.0)
    reversal = pd.to_numeric(targets.get("reversal_risk_penalty", pd.Series(dtype=float)), errors="coerce").fillna(0.0)
    wrong_side = pd.to_numeric(targets.get("source_wrong_side_sell_penalty", pd.Series(dtype=float)), errors="coerce").fillna(0.0)
    source_mask = source > 0.003
    return {
        "decision_target_source_count": float((source > 0.003).sum()),
        "decision_target_receiver_count": float((receiver > 0.003).sum()),
        "decision_target_constraint_violation_mean": float(violation.mean()) if len(violation) else 0.0,
        "decision_target_value_mean": float(value.mean()) if len(value) else 0.0,
        "decision_target_intent_translation_conflict_count": float((conflict > 0.0).sum()),
        "r69_target_deploy_release_spread_mean": float((deploy - release).mean()) if len(deploy) else 0.0,
        "r69_target_defense_value_mean": float(defense.mean()) if len(defense) else 0.0,
        "r69_target_cash_timing_value_mean": float(cash_timing.mean()) if len(cash_timing) else 0.0,
        "r69_target_source_wrong_side_sell_share": (
            float((wrong_side.loc[source_mask] > 0.5).mean()) if bool(source_mask.any()) else 0.0
        ),
        "r69_target_reversal_risk_penalty_mean": (
            float(reversal.loc[source_mask].mean()) if bool(source_mask.any()) else 0.0
        ),
    }


def _make_model(artifact: TorchPortfolioSetV5Artifact) -> PortfolioSetPolicyNetV5:
    cfg = dict(artifact.model_config or {})
    model = PortfolioSetPolicyNetV5(
        static_input_dim=len(artifact.static_feature_names),
        sequence_input_dim=max(len(artifact.sequence_bases), 1),
        daily_input_dim=max(len(artifact.daily_feature_names), 1),
        model_dim=int(cfg.get("model_dim", 192)),
        temporal_layers=int(cfg.get("temporal_layers", 2)),
        cross_layers=int(cfg.get("cross_layers", 2)),
        latent_count=int(cfg.get("latent_count", 64)),
        dropout=float(cfg.get("dropout", 0.18)),
    )
    if artifact.model_state_dict:
        model.load_state_dict(artifact.model_state_dict, strict=True)
    return model


def _predict_outputs(artifact: TorchPortfolioSetV5Artifact, state_frame: pd.DataFrame, daily_features: dict[str, float]) -> pd.DataFrame:
    frame = _ensure_features(state_frame.copy(), artifact.feature_names)
    static_frame = _ensure_features(frame, artifact.static_feature_names)
    static_x = _apply_matrix(static_frame, artifact.static_feature_names, artifact.feature_fill_values, artifact.feature_means, artifact.feature_stds)
    sequence_x = _build_sequence_array(
        frame,
        artifact.sequence_bases,
        fill=artifact.sequence_fill_values,
        means=artifact.sequence_means,
        stds=artifact.sequence_stds,
    )
    daily_raw = np.asarray([float(daily_features.get(name, 0.0) or 0.0) for name in artifact.daily_feature_names], dtype=np.float32)
    if len(daily_raw) == 0:
        daily_raw = np.zeros(1, dtype=np.float32)
    daily_x = np.where(np.isfinite(daily_raw), daily_raw, artifact.daily_fill_values[: len(daily_raw)])
    daily_x = ((daily_x - artifact.daily_means[: len(daily_x)]) / artifact.daily_stds[: len(daily_x)]).astype(np.float32)
    model = _make_model(artifact).eval()
    with torch.no_grad():
        raw = model(
            torch.as_tensor(static_x[None, :, :], dtype=torch.float32),
            torch.as_tensor(sequence_x[None, :, :, :], dtype=torch.float32),
            torch.as_tensor(daily_x[None, :], dtype=torch.float32),
            torch.ones((1, len(frame)), dtype=torch.bool),
        )["raw"][0]
        decoded = _decode_raw(raw)
    return pd.DataFrame({name: tensor.detach().cpu().numpy().astype(float) for name, tensor in decoded.items()}, index=state_frame.index)


def _predict_cashflow_turnover_budget(artifact: TorchPortfolioSetV5Artifact) -> float:
    raw = float(dict(artifact.global_target_defaults or {}).get("turnover_budget", 0.08) or 0.08)
    return float(np.clip(min(raw, 0.08), 0.04, 0.08))


def _predict_value_arbitration_inputs(
    policy: pd.DataFrame,
    outputs: pd.DataFrame,
    current: pd.Series,
) -> dict[str, pd.Series]:
    deploy_value = _max_numeric_columns(
        policy,
        (
            "deploy_value_target",
            "result_value_deploy_value_target",
            "result_value_alpha_opportunity_value",
            "portfolio_daily_receiver_score",
            "portfolio_daily_unified_receiver_score",
            "alpha_opportunity_value",
        ),
    ).clip(0.0, 1.0)
    deploy_value = pd.concat(
        [deploy_value.rename("deploy_value"), outputs["receiver_demand_score"].clip(0.0, 1.0).rename("receiver_output")],
        axis=1,
    ).max(axis=1).fillna(0.0).clip(0.0, 1.0)
    release_value = _max_numeric_columns(
        policy,
        (
            "release_value_target",
            "sell_release_value",
            "portfolio_daily_source_release_quality",
            "portfolio_daily_source_economic_release_score",
            "portfolio_daily_source_release_preference",
        ),
    ).clip(0.0, 1.0)
    release_value = pd.concat(
        [release_value.rename("release_value"), outputs["source_supply_score"].clip(0.0, 1.0).rename("source_output")],
        axis=1,
    ).max(axis=1).where(current > 0.003, 0.0).fillna(0.0).clip(0.0, 1.0)
    defense_value = _max_numeric_columns(
        policy,
        (
            "defense_value_target",
            "cash_defense_value",
            "portfolio_daily_allocation_drawdown_control_target",
            "portfolio_daily_allocation_tail_risk_control_target",
            "market_downside_pressure",
            "portfolio_drawdown_20d",
        ),
    ).abs().clip(0.0, 1.0)
    cash_timing_value = _max_numeric_columns(
        policy,
        (
            "budget_cash_timing_signal_target",
            "portfolio_daily_cash_score",
            "portfolio_daily_unified_cash_score",
            "cash_defense_value",
            "cash_regime_pressure",
        ),
    ).clip(0.0, 1.0)
    cash_timing_value = pd.concat(
        [cash_timing_value.rename("cash_timing_value"), outputs["cash_buffer_score"].clip(0.0, 1.0).rename("cash_output")],
        axis=1,
    ).max(axis=1).fillna(0.0).clip(0.0, 1.0)
    opportunity_cost = _max_numeric_columns(
        policy,
        (
            "portfolio_daily_source_opportunity_cost",
            "portfolio_daily_source_opportunity_cost_penalty",
            "hold_continuation_value",
            "alpha_opportunity_value",
            "large_upside_1d_target",
            "portfolio_daily_source_forward_proxy_keep_risk",
            "portfolio_daily_source_economic_block_risk",
        ),
    ).where(current > 0.003, 0.0).fillna(0.0).clip(0.0, 1.0)
    receiver_forward = _numeric_series(policy, "portfolio_daily_receiver_forward_excess_5d", 0.0)
    source_forward = _numeric_series(policy, "portfolio_daily_source_forward_excess_5d", 0.0)
    spread_value = _max_numeric_columns(
        policy,
        (
            "portfolio_daily_receiver_source_spread_reward",
            "portfolio_daily_source_forward_spread_score",
        ),
    ).clip(0.0, 1.0)
    spread_value = pd.concat(
        [spread_value.rename("spread_value"), ((receiver_forward - source_forward) / 0.075).clip(0.0, 1.0).rename("forward_spread")],
        axis=1,
    ).max(axis=1).fillna(0.0).clip(0.0, 1.0)
    wrong_side_penalty = _max_numeric_columns(
        policy,
        (
            "portfolio_daily_source_positive_forward_penalty",
            "portfolio_daily_source_bad_forward_spread_risk",
            "portfolio_daily_source_forward_strength_brake_risk",
        ),
    ).clip(0.0, 1.0)
    wrong_side_penalty = pd.concat(
        [wrong_side_penalty.rename("wrong_side_penalty"), (source_forward / 0.075).clip(0.0, 1.0).rename("source_forward_positive")],
        axis=1,
    ).max(axis=1).where(current > 0.003, 0.0).fillna(0.0).clip(0.0, 1.0)
    reversal_penalty = _max_numeric_columns(
        policy,
        (
            "reduce_reversal_pressure",
            "recent_reversal_rate_20d",
            "reentry_guard_score",
            "portfolio_daily_source_forward_proxy_keep_risk",
        ),
    ).where(current > 0.003, 0.0).fillna(0.0).clip(0.0, 1.0)
    return {
        "deploy_value": deploy_value.astype(float),
        "release_value": release_value.astype(float),
        "defense_value": defense_value.astype(float),
        "cash_timing_value": cash_timing_value.astype(float),
        "source_opportunity_cost": opportunity_cost.astype(float),
        "receiver_source_spread_value": spread_value.astype(float),
        "reversal_risk_penalty": reversal_penalty.astype(float),
        "source_wrong_side_sell_penalty": wrong_side_penalty.astype(float),
    }


def predict_policy_portfolio_set_v5(
    artifact: TorchPortfolioSetV5Artifact,
    *,
    state_frame: pd.DataFrame,
    daily_features: dict[str, float],
) -> tuple[pd.DataFrame, dict[str, float]]:
    if state_frame.empty:
        raise ValueError("state_frame is empty.")
    policy = state_frame.copy()
    outputs = _predict_outputs(artifact, state_frame, daily_features)
    current = _current_weight(policy)
    source_score = outputs["source_supply_score"].clip(0.0, 1.0)
    receiver_score = outputs["receiver_demand_score"].clip(0.0, 1.0)
    value_inputs = _predict_value_arbitration_inputs(policy, outputs, current)
    cashflow_turnover_budget = _predict_cashflow_turnover_budget(artifact)
    oracle = project_portfolio_set_v5_cashflow_oracle(
        current_weight=torch.as_tensor(current.to_numpy(dtype=np.float32)[None, :], dtype=torch.float32),
        source_score=torch.as_tensor(source_score.to_numpy(dtype=np.float32)[None, :], dtype=torch.float32),
        receiver_score=torch.as_tensor(receiver_score.to_numpy(dtype=np.float32)[None, :], dtype=torch.float32),
        cash_buffer_score=torch.as_tensor(outputs["cash_buffer_score"].clip(0.0, 1.0).to_numpy(dtype=np.float32)[None, :], dtype=torch.float32),
        deploy_value=torch.as_tensor(value_inputs["deploy_value"].to_numpy(dtype=np.float32)[None, :], dtype=torch.float32),
        release_value=torch.as_tensor(value_inputs["release_value"].to_numpy(dtype=np.float32)[None, :], dtype=torch.float32),
        defense_value=torch.as_tensor(value_inputs["defense_value"].to_numpy(dtype=np.float32)[None, :], dtype=torch.float32),
        cash_timing_value=torch.as_tensor(value_inputs["cash_timing_value"].to_numpy(dtype=np.float32)[None, :], dtype=torch.float32),
        source_opportunity_cost=torch.as_tensor(value_inputs["source_opportunity_cost"].to_numpy(dtype=np.float32)[None, :], dtype=torch.float32),
        receiver_source_spread_value=torch.as_tensor(value_inputs["receiver_source_spread_value"].to_numpy(dtype=np.float32)[None, :], dtype=torch.float32),
        reversal_risk_penalty=torch.as_tensor(value_inputs["reversal_risk_penalty"].to_numpy(dtype=np.float32)[None, :], dtype=torch.float32),
        source_wrong_side_sell_penalty=torch.as_tensor(value_inputs["source_wrong_side_sell_penalty"].to_numpy(dtype=np.float32)[None, :], dtype=torch.float32),
        sample_mask=torch.ones((1, len(policy)), dtype=torch.bool),
        turnover_budget=torch.as_tensor([cashflow_turnover_budget], dtype=torch.float32),
        risk_budget=torch.as_tensor([float(np.clip(pd.concat([outputs["cash_buffer_score"], value_inputs["defense_value"], value_inputs["cash_timing_value"]], axis=1).max(axis=1).mean(), 0.0, 1.0))], dtype=torch.float32),
    )
    target_weight = pd.Series(oracle["target_weight"][0].detach().cpu().numpy().astype(float), index=policy.index)
    target_delta = (target_weight - current).clip(-0.18, 0.18)
    target_delta = target_delta.where(target_delta.abs() >= 0.003, 0.0)
    target_weight = (current + target_delta).clip(0.0, 0.24)
    source_supply = pd.Series(oracle["source_supply"][0].detach().cpu().numpy().astype(float), index=policy.index)
    receiver_demand = pd.Series(oracle["receiver_demand"][0].detach().cpu().numpy().astype(float), index=policy.index)
    source_capacity = current.clip(lower=0.003)
    release_intent = (source_supply / source_capacity).replace([np.inf, -np.inf], np.nan).fillna(0.0).clip(0.0, 1.0)
    release_action = pd.Series("hold", index=policy.index, dtype=object)
    release_action.loc[(current > 0.003) & (target_delta < -0.003) & (target_weight > 0.003)] = "reduce"
    release_action.loc[(current > 0.003) & (target_delta < -0.003) & (target_weight <= 0.003)] = "exit"
    release_block = pd.Series("none", index=policy.index, dtype=object)
    release_block.loc[current <= 0.003] = "not_held"
    release_block.loc[(current > 0.003) & (source_score < 0.02) & (target_delta >= -0.003)] = "source_score_dead"
    policy["portfolio_daily_target_weight_intent"] = target_weight.astype(float)
    policy["portfolio_daily_target_delta_intent"] = target_delta.astype(float)
    policy["target_delta_hint"] = target_delta.astype(float)
    policy["portfolio_daily_release_first_intent"] = release_intent.astype(float)
    policy["release_first_action_hint"] = release_action.astype(str)
    policy["release_first_block_reason"] = release_block.astype(str)
    policy["portfolio_daily_source_score"] = source_score.astype(float)
    policy["portfolio_daily_unified_source_score"] = source_score.astype(float)
    policy["portfolio_daily_source_executable_candidate"] = ((current > 0.003) & (source_supply > 0.003)).astype(float)
    policy["portfolio_daily_source_target_intent"] = ((current > 0.003) & (source_supply > 0.003) & (target_delta < -0.003)).astype(float)
    policy["portfolio_daily_source_release_quality"] = release_intent.astype(float)
    policy["portfolio_daily_source_release_capacity"] = release_intent.astype(float)
    policy["portfolio_daily_source_release_preference"] = release_intent.astype(float)
    policy["portfolio_daily_source_executability"] = policy["portfolio_daily_source_executable_candidate"].astype(float)
    policy["portfolio_daily_source_economic_release_score"] = release_intent.astype(float)
    policy["portfolio_daily_receiver_score"] = receiver_score.astype(float)
    policy["portfolio_daily_unified_receiver_score"] = receiver_score.astype(float)
    policy["portfolio_daily_receiver_executable_candidate"] = ((current < 0.24 - 0.003) & (receiver_demand > 0.003) & (target_delta > 0.003)).astype(float)
    policy["portfolio_daily_receiver_target_intent"] = policy["portfolio_daily_receiver_executable_candidate"].astype(float)
    policy["portfolio_daily_receiver_add_headroom"] = (0.24 - current).clip(lower=0.0).astype(float)
    policy["portfolio_daily_receiver_add_capacity"] = policy["portfolio_daily_receiver_add_headroom"].astype(float)
    policy["portfolio_daily_receiver_executability"] = policy["portfolio_daily_receiver_executable_candidate"].astype(float)
    policy["portfolio_daily_cash_score"] = outputs["cash_buffer_score"].clip(0.0, 1.0).astype(float)
    policy["portfolio_daily_unified_cash_score"] = outputs["cash_buffer_score"].clip(0.0, 1.0).astype(float)
    policy["portfolio_set_v5_r69_deploy_value"] = value_inputs["deploy_value"].astype(float)
    policy["portfolio_set_v5_r69_release_value"] = value_inputs["release_value"].astype(float)
    policy["portfolio_set_v5_r69_defense_value"] = value_inputs["defense_value"].astype(float)
    policy["portfolio_set_v5_r69_cash_timing_value"] = value_inputs["cash_timing_value"].astype(float)
    policy["portfolio_set_v5_r69_source_opportunity_cost"] = value_inputs["source_opportunity_cost"].astype(float)
    policy["portfolio_set_v5_r69_receiver_source_spread_value"] = value_inputs["receiver_source_spread_value"].astype(float)
    policy["portfolio_set_v5_r69_reversal_risk_penalty"] = value_inputs["reversal_risk_penalty"].astype(float)
    policy["portfolio_set_v5_r69_source_wrong_side_sell_penalty"] = value_inputs["source_wrong_side_sell_penalty"].astype(float)
    policy["portfolio_set_v5_r69_reversal_guarded"] = (
        (policy["portfolio_daily_source_target_intent"] > 0.5)
        & (policy["portfolio_set_v5_r69_reversal_risk_penalty"] > 0.5)
    ).astype(float)
    policy["portfolio_set_v5_r69_source_wrong_side_sell"] = (
        (policy["portfolio_daily_source_target_intent"] > 0.5)
        & (policy["portfolio_set_v5_r69_source_wrong_side_sell_penalty"] > 0.5)
    ).astype(float)
    policy["portfolio_set_v5_source_supply_score"] = source_score.astype(float)
    policy["portfolio_set_v5_receiver_demand_score"] = receiver_score.astype(float)
    policy["portfolio_set_v5_source_supply"] = source_supply.astype(float)
    policy["portfolio_set_v5_receiver_demand"] = receiver_demand.astype(float)
    policy["portfolio_set_v5_cash_buffer_score"] = float(oracle["cash_buffer"][0].detach().cpu())
    policy["portfolio_set_v5_turnover_budget"] = cashflow_turnover_budget
    policy["portfolio_set_v5_turnover_used"] = float(oracle["turnover_used"][0].detach().cpu())
    policy["portfolio_set_v5_oracle_constraint_violation"] = float(oracle["constraint_violation"][0].detach().cpu())
    policy["portfolio_set_v5_oracle_decision_value"] = float(oracle["decision_value"][0].detach().cpu())
    policy["portfolio_set_v5_oracle_feasible"] = float(float(oracle["constraint_violation"][0].detach().cpu()) <= 1.0e-6)
    source_target_count = max(float(policy["portfolio_daily_source_target_intent"].sum()), 1.0)
    defense_days = max(float((policy["portfolio_set_v5_r69_defense_value"] > 0.5).sum()), 1.0)
    policy["portfolio_set_v5_r69_deploy_release_spread_mean"] = float(
        (policy["portfolio_set_v5_r69_deploy_value"] - policy["portfolio_set_v5_r69_release_value"]).mean()
    )
    policy["portfolio_set_v5_r69_source_wrong_side_sell_share"] = float(policy["portfolio_set_v5_r69_source_wrong_side_sell"].sum() / source_target_count)
    policy["portfolio_set_v5_r69_cash_timing_decision_quality"] = float(
        float(oracle["cash_buffer"][0].detach().cpu())
        * max(float(policy["portfolio_set_v5_r69_defense_value"].mean()), float(policy["portfolio_set_v5_r69_cash_timing_value"].mean()))
    )
    policy["portfolio_set_v5_r69_reversal_risk_penalty_mean"] = float(
        (policy["portfolio_set_v5_source_supply"] * policy["portfolio_set_v5_r69_reversal_risk_penalty"]).mean()
    )
    policy["portfolio_set_v5_r69_defense_win_rate"] = float(
        ((policy["portfolio_set_v5_cash_buffer_score"] >= float(oracle["cash_floor"][0].detach().cpu()) - 1.0e-6) & (policy["portfolio_set_v5_r69_defense_value"] > 0.5)).sum()
        / defense_days
    )
    policy["portfolio_set_v5_target_delta_weight_conflict_count"] = int((((target_weight - current) * target_delta) < -(0.003 ** 2)).sum())
    add_mask = (current > 0.0) & (target_delta > 0.003)
    open_mask = (current <= 0.0) & (target_delta > 0.003)
    reduce_mask = (current > 0.0) & (target_delta < -0.003) & (target_weight > 0.003)
    exit_mask = (current > 0.0) & (target_delta < -0.003) & (target_weight <= 0.003)
    policy["action_label"] = "hold"
    policy.loc[open_mask, "action_label"] = "open"
    policy.loc[add_mask, "action_label"] = "add"
    policy.loc[reduce_mask, "action_label"] = "reduce"
    policy.loc[exit_mask, "action_label"] = "exit"
    policy[PORTFOLIO_CASHFLOW_DECISION_MODE_COLUMN] = 1.0
    cashflow_decision = normalize_portfolio_cashflow_decision(
        policy,
        current_weight=current,
        position_cap=0.24,
        turnover_limit=cashflow_turnover_budget,
        fail_closed=True,
    )
    policy = cashflow_decision.frame
    global_targets = dict(artifact.global_target_defaults or {})
    global_targets.update(
        {
            PORTFOLIO_CASHFLOW_DECISION_MODE_COLUMN: 1.0,
            "release_first_allocation_v3_mode": 1.0,
            "allocation_intent_v2_mode": 1.0,
            "target_weight_intent_mode": 1.0,
        }
    )
    return policy, {key: float(value) for key, value in global_targets.items()}


def fit_policy_models_portfolio_set_v5(
    *,
    sample_frame: pd.DataFrame,
    daily_frame: pd.DataFrame,
    feature_names: list[str],
    daily_feature_names: list[str],
    run_root: Path,
    random_seed: int = 7,
    train_summary: dict[str, Any] | None = None,
    trained_at: str = "",
    training_contract: dict[str, Any] | None = None,
    epochs: int = 12,
    min_epochs: int = 8,
    batch_size: int = 1,
    learning_rate: float = 5.0e-5,
    model_dim: int = 192,
    temporal_layers: int = 2,
    cross_layers: int = 2,
    latent_count: int = 64,
    dropout: float = 0.18,
    early_stop_patience: int = 10,
    resume_mode: str = "strict",
    loss_profile: str = PORTFOLIO_SET_V5_INTERNAL_VERSION,
    progress_sink: Any | None = None,
) -> TorchPortfolioSetV5Artifact:
    if sample_frame.empty or daily_frame.empty:
        raise ValueError("formal_torch_portfolio_set_v5 received empty training data.")
    contract = dict(training_contract or {})
    if str(contract.get("trainer_backend", "") or "") != TRAINER_BACKEND_FORMAL_PORTFOLIO_SET_V5:
        raise ValueError("fit_policy_models_portfolio_set_v5 requires the formal_torch_portfolio_set_v5 training contract.")
    resolved_loss_profile, loss_config = resolve_portfolio_set_v5_loss_profile(loss_profile)
    if not torch.cuda.is_available() and bool(contract.get("gpu_required", False)):
        raise RuntimeError("continuous_policy formal_torch_portfolio_set_v5 requires CUDA, but torch.cuda.is_available() is False.")
    device = torch.device("cuda" if bool(contract.get("gpu_required", False)) else "cpu")
    runtime = configure_torch_training_acceleration(device, cvxpy_layers_enabled=False)
    torch.manual_seed(int(random_seed))
    np.random.seed(int(random_seed))
    run_root.mkdir(parents=True, exist_ok=True)
    train_frame, day_diagnostics = _select_train_days(sample_frame, int(random_seed))
    targets = build_portfolio_set_v5_targets(train_frame)
    target_diagnostics = _target_diagnostics(targets)
    static_feature_names, sequence_bases, sequence_columns = _resolve_static_and_sequence_columns(feature_names)
    static_matrix, static_fill, static_means, static_stds = _prepare_matrix(_ensure_features(train_frame, static_feature_names), static_feature_names)
    if sequence_columns:
        _, sequence_fill, sequence_means, sequence_stds = _prepare_matrix(_ensure_features(train_frame, sequence_columns), sequence_columns)
    else:
        sequence_fill = np.zeros(1, dtype=np.float32)
        sequence_means = np.zeros(1, dtype=np.float32)
        sequence_stds = np.ones(1, dtype=np.float32)
    _, daily_fill, daily_means, daily_stds = _prepare_matrix(_ensure_features(daily_frame, daily_feature_names), daily_feature_names)
    dataset = PortfolioSetDayDataset(
        sample_frame=train_frame,
        daily_frame=daily_frame,
        static_feature_names=static_feature_names,
        sequence_bases=sequence_bases,
        daily_feature_names=daily_feature_names,
        targets=targets,
        static_fill=static_fill,
        static_means=static_means,
        static_stds=static_stds,
        sequence_fill=sequence_fill,
        sequence_means=sequence_means,
        sequence_stds=sequence_stds,
        daily_fill=daily_fill,
        daily_means=daily_means,
        daily_stds=daily_stds,
    )
    train_idx, val_idx = _split_indices(len(dataset), int(random_seed))
    train_subset = torch.utils.data.Subset(dataset, train_idx.tolist())
    val_subset = torch.utils.data.Subset(dataset, val_idx.tolist())
    loader = DataLoader(
        train_subset,
        batch_size=max(1, min(int(batch_size or 1), 2)),
        shuffle=True,
        collate_fn=collate_portfolio_set_days,
        pin_memory=runtime.pin_memory,
    )
    val_loader = DataLoader(
        val_subset,
        batch_size=1,
        shuffle=False,
        collate_fn=collate_portfolio_set_days,
        pin_memory=runtime.pin_memory,
    )
    if progress_sink is not None:
        progress_sink.emit("train_dataframe_ready", train_sample_rows=int(len(train_frame)), portfolio_set_v5=True)
        progress_sink.emit("train_dataloader_ready", batch_count=int(len(loader)), data_loader_pin_memory=runtime.pin_memory)
    model = PortfolioSetPolicyNetV5(
        static_input_dim=len(static_feature_names),
        sequence_input_dim=max(len(sequence_bases), 1),
        daily_input_dim=max(len(daily_feature_names), 1),
        model_dim=int(model_dim),
        temporal_layers=int(temporal_layers),
        cross_layers=int(cross_layers),
        latent_count=int(latent_count),
        dropout=float(dropout),
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(learning_rate), weight_decay=1.0e-4)
    scaler = torch.amp.GradScaler("cuda", enabled=runtime.amp_enabled)
    weights = loss_config["multi_objective_loss_weights"]
    best_loss = float("inf")
    best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
    best_epoch = 0
    completed_epochs = 0
    progress_event_count = 0
    last_train_decision_diagnostics: dict[str, float] = {}
    last_val_decision_diagnostics: dict[str, float] = {}
    started = time.monotonic()
    for epoch in range(1, max(int(epochs or 0), 1) + 1):
        epoch_started = time.monotonic()
        model.train()
        train_loss_sum = 0.0
        train_count = 0
        train_diagnostic_items: list[dict[str, float]] = []
        for batch in loader:
            batch = move_to_device(batch, device, non_blocking=runtime.non_blocking_transfer)
            optimizer.zero_grad(set_to_none=True)
            with autocast_context(runtime):
                raw = model(batch["static_x"], batch["sequence_x"], batch["daily_x"], batch["sample_mask"])["raw"]
                loss = _portfolio_set_loss(raw, batch, weights)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            train_loss_sum += float(loss.detach().cpu())
            train_count += 1
            train_diagnostic_items.append(portfolio_set_v5_decision_diagnostics(raw.detach(), batch, weights))
        last_train_decision_diagnostics = _merge_portfolio_set_v5_diagnostics(train_diagnostic_items)
        model.eval()
        val_losses: list[float] = []
        val_diagnostic_items: list[dict[str, float]] = []
        with torch.no_grad():
            for batch in val_loader:
                batch = move_to_device(batch, device, non_blocking=runtime.non_blocking_transfer)
                raw = model(batch["static_x"], batch["sequence_x"], batch["daily_x"], batch["sample_mask"])["raw"]
                val_losses.append(float(_portfolio_set_loss(raw, batch, weights).detach().cpu()))
                val_diagnostic_items.append(portfolio_set_v5_decision_diagnostics(raw.detach(), batch, weights))
        last_val_decision_diagnostics = _merge_portfolio_set_v5_diagnostics(val_diagnostic_items)
        val_loss = float(np.mean(val_losses)) if val_losses else float(train_loss_sum / max(train_count, 1))
        completed_epochs = epoch
        if val_loss < best_loss:
            best_loss = val_loss
            best_epoch = epoch
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
        if progress_sink is not None:
            progress_sink.emit(
                "train_epoch_complete",
                epoch=epoch,
                completed_epochs=completed_epochs,
                train_loss=train_loss_sum / max(train_count, 1),
                validation_loss=val_loss,
                epoch_seconds=round(time.monotonic() - epoch_started, 3),
                amp_enabled=runtime.amp_enabled,
                data_loader_pin_memory=runtime.pin_memory,
                non_blocking_transfer=runtime.non_blocking_transfer,
                portfolio_set_v5=True,
                paper_reproduction_metrics=last_train_decision_diagnostics,
                decision_oracle_metrics=last_val_decision_diagnostics,
            )
            progress_event_count += 1
        if epoch >= int(min_epochs or 0) and epoch - best_epoch >= int(early_stop_patience or 0):
            break
    model.load_state_dict(best_state, strict=True)
    diagnostics = {
        "trainer_backend": TRAINER_BACKEND_FORMAL_PORTFOLIO_SET_V5,
        "loss_profile": resolved_loss_profile,
        "status": "portfolio_set_v5_complete",
        "portfolio_set_v5_internal_version": PORTFOLIO_SET_V5_INTERNAL_VERSION,
        "paper_reproduction_metrics": {
            "method": "pg_dfl_surrogate",
            "surrogate": "positive_negative_score_perturbation",
            "oracle": "torch_long_only_cashflow_value_arbitration_projection",
            **last_train_decision_diagnostics,
        },
        "decision_oracle_metrics": {
            "position_cap": 0.24,
            "long_only": True,
            "source_receiver_cash_conservation": True,
            "value_arbitration": resolved_loss_profile == PORTFOLIO_SET_V5_R69_INTERNAL_VERSION,
            **last_val_decision_diagnostics,
        },
        "release_flow_metrics": {
            "target_source_count": target_diagnostics["decision_target_source_count"],
            "target_receiver_count": target_diagnostics["decision_target_receiver_count"],
            "target_intent_translation_conflict_count": target_diagnostics["decision_target_intent_translation_conflict_count"],
            "source_recall_observed_count": target_diagnostics["decision_target_source_count"],
            "receiver_recall_observed_count": target_diagnostics["decision_target_receiver_count"],
            "receiver_source_balance_ratio": float(
                target_diagnostics["decision_target_receiver_count"]
                / max(target_diagnostics["decision_target_source_count"], 1.0)
            ),
            "r69_target_source_wrong_side_sell_share": target_diagnostics["r69_target_source_wrong_side_sell_share"],
            "r69_target_reversal_risk_penalty_mean": target_diagnostics["r69_target_reversal_risk_penalty_mean"],
            "r69_target_cash_timing_value_mean": target_diagnostics["r69_target_cash_timing_value_mean"],
        },
        "device": str(device),
        "gpu_acceleration": runtime.to_diagnostics(),
        "amp_enabled": runtime.amp_enabled,
        "data_loader_pin_memory": runtime.pin_memory,
        "non_blocking_transfer": runtime.non_blocking_transfer,
        "completed_epochs": int(completed_epochs),
        "best_epoch": int(best_epoch),
        "best_validation_loss": float(best_loss),
        "train_sample_rows": int(len(train_frame)),
        "raw_train_sample_rows": int(len(sample_frame)),
        "train_day_count": int(len(dataset)),
        "portfolio_set_v5_latent_count": int(latent_count),
        "portfolio_set_v5_uses_latent_attention": True,
        "portfolio_set_v5_full_self_attention": False,
        "portfolio_set_v5_shadow_only": True,
        "supports_release_first_allocation_v3_mode": True,
        "supports_portfolio_cashflow_decision_v1_mode": True,
        "supports_portfolio_set_v5_r69_value_arbitration": resolved_loss_profile == PORTFOLIO_SET_V5_R69_INTERNAL_VERSION,
        "progress_event_count": int(progress_event_count),
        "train_seconds": round(time.monotonic() - started, 3),
        **target_diagnostics,
        **day_diagnostics,
    }
    if progress_sink is not None:
        progress_sink.emit("training_complete", completed_epochs=completed_epochs, best_epoch=best_epoch, portfolio_set_v5=True)
    artifact = TorchPortfolioSetV5Artifact(
        feature_names=list(feature_names),
        daily_feature_names=list(daily_feature_names),
        static_feature_names=list(static_feature_names),
        sequence_bases=list(sequence_bases),
        feature_fill_values=static_fill,
        feature_means=static_means,
        feature_stds=static_stds,
        sequence_fill_values=sequence_fill,
        sequence_means=sequence_means,
        sequence_stds=sequence_stds,
        daily_fill_values=daily_fill if len(daily_fill) else np.zeros(1, dtype=np.float32),
        daily_means=daily_means if len(daily_means) else np.zeros(1, dtype=np.float32),
        daily_stds=daily_stds if len(daily_stds) else np.ones(1, dtype=np.float32),
        train_summary=dict(train_summary or {}),
        training_diagnostics=diagnostics,
        training_contract=contract,
        trained_at=str(trained_at or ""),
        model_config={
            "model_dim": int(model_dim),
            "temporal_layers": int(temporal_layers),
            "cross_layers": int(cross_layers),
            "latent_count": int(latent_count),
            "dropout": float(dropout),
            "portfolio_set_v5_internal_version": PORTFOLIO_SET_V5_INTERNAL_VERSION,
        },
        model_state_dict={key: value.detach().cpu() for key, value in model.state_dict().items()},
        global_target_defaults=_global_defaults(daily_frame),
    )
    artifact.save(run_root / PORTFOLIO_SET_V5_ARTIFACT_FILENAME)
    return artifact
