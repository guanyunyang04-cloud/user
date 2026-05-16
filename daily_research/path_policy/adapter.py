from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd


PATH_POLICY_MODE_COLUMN = "alpha_path20_neural_policy_v1_mode"
PATH_POLICY_TARGET_WEIGHT_COLUMN = "alpha_path20_target_weight"


@dataclass(frozen=True)
class ProjectionDiagnostics:
    raw_sum: float
    projected_sum: float
    cash_weight: float
    clipped_count: int
    masked_count: int
    scaled_down: float
    turnover_budget: float
    projected_turnover: float
    turnover_scaled_down: float
    target_count: int

    def to_dict(self) -> dict[str, float | int]:
        return {
            "path_policy_raw_target_weight_sum": float(self.raw_sum),
            "path_policy_projected_target_weight_sum": float(self.projected_sum),
            "path_policy_cash_weight": float(self.cash_weight),
            "path_policy_clipped_count": int(self.clipped_count),
            "path_policy_masked_count": int(self.masked_count),
            "path_policy_scaled_down": float(self.scaled_down),
            "path_policy_turnover_budget": float(self.turnover_budget),
            "path_policy_projected_turnover": float(self.projected_turnover),
            "path_policy_turnover_scaled_down": float(self.turnover_scaled_down),
            "path_policy_target_count": int(self.target_count),
        }


def _numeric(frame: pd.DataFrame, column: str, default: float = 0.0) -> pd.Series:
    if column not in frame.columns:
        return pd.Series(float(default), index=frame.index, dtype=float)
    return pd.to_numeric(frame[column], errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(float(default)).astype(float)


def project_target_weights(
    raw_target_weight: pd.Series,
    *,
    current_weight: pd.Series | None = None,
    tradable_mask: pd.Series | None = None,
    max_position_weight: float = 0.20,
    max_gross_exposure: float = 0.95,
    max_positions: int = 50,
    turnover_budget: float = 1.00,
) -> tuple[pd.Series, ProjectionDiagnostics]:
    index = raw_target_weight.index.map(str)
    raw = raw_target_weight.copy().astype(float)
    raw.index = index
    raw = raw.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    current = (
        current_weight.reindex(index).replace([np.inf, -np.inf], np.nan).fillna(0.0).astype(float)
        if current_weight is not None
        else pd.Series(0.0, index=index, dtype=float)
    )
    if tradable_mask is not None:
        tradable = tradable_mask.reindex(index).fillna(False).astype(bool)
    else:
        tradable = pd.Series(True, index=index, dtype=bool)
    raw_sum = float(raw.clip(lower=0.0).sum())
    clipped = raw.clip(lower=0.0, upper=float(max_position_weight))
    clipped_count = int(((raw < 0.0) | (raw > float(max_position_weight))).sum())
    masked_count = int(((~tradable) & ((clipped - current).abs() > 1.0e-12)).sum())
    locked = current.where((~tradable) & (current > 1.0e-12), 0.0).clip(lower=0.0)
    tradable_target = clipped.where(tradable, 0.0)
    if max_positions > 0:
        locked_names = set(locked[locked > 1.0e-12].index)
        remaining_slots = max(int(max_positions) - len(locked_names), 0)
        positive_tradable = tradable_target[tradable_target > 1.0e-12].sort_values(ascending=False)
        keep_tradable = set(positive_tradable.head(remaining_slots).index) if remaining_slots > 0 else set()
        tradable_target = tradable_target.where(tradable_target.index.isin(keep_tradable), 0.0)
    projected = (locked + tradable_target).clip(lower=0.0)
    projected_sum = float(projected.sum())
    scaled_down = 0.0
    max_gross = float(np.clip(max_gross_exposure, 0.0, 1.0))
    locked_sum = float(locked.sum())
    tradable_sum = float(tradable_target.sum())
    if projected_sum > max_gross and tradable_sum > 1.0e-12:
        available = max(float(max_gross - locked_sum), 0.0)
        tradable_target = tradable_target / tradable_sum * available
        projected = (locked + tradable_target).clip(lower=0.0)
        scaled_down = 1.0
        projected_sum = float(projected.sum())
    turnover_scaled_down = 0.0
    turnover_budget_value = float(max(turnover_budget, 0.0))
    projected_turnover = float((projected - current).abs().sum())
    if turnover_budget_value > 0.0 and projected_turnover > turnover_budget_value + 1.0e-12:
        scale = turnover_budget_value / max(projected_turnover, 1.0e-12)
        projected = (current + (projected - current) * scale).clip(lower=0.0, upper=float(max_position_weight))
        turnover_scaled_down = 1.0
        projected_sum = float(projected.sum())
        projected_turnover = float((projected - current).abs().sum())
    diagnostics = ProjectionDiagnostics(
        raw_sum=raw_sum,
        projected_sum=projected_sum,
        cash_weight=float(max(0.0, 1.0 - projected_sum)),
        clipped_count=clipped_count,
        masked_count=masked_count,
        scaled_down=scaled_down,
        turnover_budget=turnover_budget_value,
        projected_turnover=projected_turnover,
        turnover_scaled_down=turnover_scaled_down,
        target_count=int((projected > 1.0e-12).sum()),
    )
    return projected.astype(float), diagnostics


def build_path_policy_frame(
    base_frame: pd.DataFrame,
    *,
    raw_target_weight: pd.Series,
    current_weight: pd.Series | None = None,
    tradable_mask: pd.Series | None = None,
    max_position_weight: float = 0.20,
    max_gross_exposure: float = 0.95,
    max_positions: int = 50,
    turnover_budget: float = 1.00,
    source_label: str = "alpha_path20",
) -> tuple[pd.DataFrame, dict[str, Any]]:
    working = base_frame.copy()
    if "stock" in working.columns:
        working.index = working["stock"].astype(str)
    else:
        working.index = working.index.map(str)
        working["stock"] = working.index
    current = (
        current_weight.reindex(working.index).replace([np.inf, -np.inf], np.nan).fillna(0.0).astype(float)
        if current_weight is not None
        else _numeric(working, "current_weight", 0.0)
    )
    target, diagnostics = project_target_weights(
        raw_target_weight.reindex(working.index).fillna(0.0).astype(float),
        current_weight=current,
        tradable_mask=tradable_mask.reindex(working.index).fillna(True).astype(bool) if tradable_mask is not None else None,
        max_position_weight=max_position_weight,
        max_gross_exposure=max_gross_exposure,
        max_positions=max_positions,
        turnover_budget=turnover_budget,
    )
    delta = (target - current).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    action = pd.Series("hold", index=working.index, dtype=object)
    action.loc[(current <= 1.0e-8) & (target <= 1.0e-8)] = "skip"
    action.loc[(current <= 1.0e-8) & (delta > 1.0e-8)] = "open"
    action.loc[(current > 1.0e-8) & (delta > 1.0e-8)] = "add"
    action.loc[(current > 1.0e-8) & (delta < -1.0e-8) & (target > 1.0e-8)] = "reduce"
    action.loc[(current > 1.0e-8) & (target <= 1.0e-8)] = "exit"
    source_supply = (-delta.clip(upper=0.0)).clip(lower=0.0)
    receiver_demand = delta.clip(lower=0.0)
    working[PATH_POLICY_MODE_COLUMN] = 1.0
    working[PATH_POLICY_TARGET_WEIGHT_COLUMN] = target.astype(float)
    working["portfolio_daily_target_weight"] = target.astype(float)
    working["portfolio_daily_target_weight_intent"] = target.astype(float)
    working["portfolio_daily_target_delta_intent"] = delta.astype(float)
    working["target_delta_hint"] = delta.astype(float)
    working["action_label"] = action.astype(str)
    working["action_strength"] = delta.abs().clip(0.0, 1.0).astype(float)
    working["path_policy_source_supply_derived"] = source_supply.astype(float)
    working["path_policy_receiver_demand_derived"] = receiver_demand.astype(float)
    working["portfolio_daily_receiver_executable_candidate"] = ((receiver_demand > 1.0e-8) & (target <= float(max_position_weight) + 1.0e-8)).astype(float)
    working["portfolio_daily_source_executable_candidate"] = ((source_supply > 1.0e-8) & (current > 1.0e-8)).astype(float)
    working["portfolio_daily_receiver_target_intent"] = (receiver_demand > 1.0e-8).astype(float)
    working["portfolio_daily_source_target_intent"] = (source_supply > 1.0e-8).astype(float)
    working["portfolio_daily_receiver_score"] = receiver_demand.astype(float)
    working["portfolio_daily_source_score"] = source_supply.astype(float)
    working["path_policy_cash_weight"] = diagnostics.cash_weight
    working["path_policy_source_label"] = str(source_label)
    global_targets: dict[str, Any] = {
        "decision_core_version": "alpha_path20_neural_policy_v1",
        "gross_exposure_target": float(max_gross_exposure),
        "max_position_weight_target": float(max_position_weight),
        "turnover_budget": float(turnover_budget),
        "cash_reserve_target": float(max(0.0, 1.0 - float(max_gross_exposure))),
        **diagnostics.to_dict(),
    }
    return working, global_targets
