from __future__ import annotations

import numpy as np
import pandas as pd


def _numeric(frame: pd.DataFrame, column: str, default: float = 0.0) -> pd.Series:
    if column not in frame.columns:
        return pd.Series(float(default), index=frame.index, dtype=float)
    return pd.to_numeric(frame[column], errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(float(default))


def _max_columns(frame: pd.DataFrame, columns: tuple[str, ...], default: float = 0.0) -> pd.Series:
    values = [_numeric(frame, column, default).rename(column) for column in columns if column in frame.columns]
    if not values:
        return pd.Series(float(default), index=frame.index, dtype=float)
    return pd.concat(values, axis=1).max(axis=1).fillna(float(default))


def build_core_v4_release_first_targets(
    sample_frame: pd.DataFrame,
    *,
    deadband: float = 0.003,
    position_cap: float = 0.24,
) -> pd.DataFrame:
    result = pd.DataFrame(index=sample_frame.index)
    raw_current = _numeric(sample_frame, "current_weight", 0.0)
    if "current_weight" not in sample_frame.columns:
        raw_current = _numeric(sample_frame, "position_weight", 0.0)
    holding_flag = _numeric(sample_frame, "holding_flag", 0.0) > 0.5
    release_capacity = _numeric(sample_frame, "portfolio_daily_source_release_capacity", 0.0).clip(lower=0.0)
    min_release_delta = _numeric(sample_frame, "portfolio_daily_source_min_release_delta", 0.0).clip(lower=0.0)
    current = pd.concat(
        [
            raw_current.clip(lower=0.0).rename("current_weight"),
            release_capacity.where(holding_flag, 0.0).rename("release_capacity"),
            min_release_delta.where(holding_flag, 0.0).rename("min_release_delta"),
        ],
        axis=1,
    ).max(axis=1).clip(lower=0.0, upper=float(position_cap))
    action = sample_frame.get("action_label", pd.Series("hold", index=sample_frame.index)).fillna("hold").astype(str).str.lower()
    raw_delta = _numeric(sample_frame, "portfolio_daily_target_delta_intent", 0.0)
    if "portfolio_daily_target_delta_intent" not in sample_frame.columns and "target_delta_hint" in sample_frame.columns:
        raw_delta = _numeric(sample_frame, "target_delta_hint", 0.0)

    release_pressure = pd.concat(
        [
            (-raw_delta.clip(upper=0.0) / current.clip(lower=float(deadband))).clip(0.0, 1.0).rename("negative_delta"),
            action.isin({"reduce", "exit"}).astype(float).rename("action_release"),
            _max_columns(
                sample_frame,
                (
                    "portfolio_daily_source_release_preference",
                    "result_value_release_gate_target",
                    "result_value_sell_release_value",
                    "sell_release_value",
                ),
            ).clip(0.0, 1.0).rename("release_value"),
        ],
        axis=1,
    ).max(axis=1)
    keep_risk = _numeric(sample_frame, "portfolio_daily_source_forward_proxy_keep_risk", 0.0).clip(0.0, 1.0)
    block_risk = _numeric(sample_frame, "portfolio_daily_source_economic_block_risk", 0.0).clip(0.0, 1.0)
    held = holding_flag | (current > float(deadband))
    release_allowed = held & (release_pressure > 0.0) & (keep_risk < 0.78) & (block_risk < 0.78)
    release_blocked = held & (release_pressure > 0.0) & ((keep_risk >= 0.78) | (block_risk >= 0.78))
    existing_release = (-raw_delta.clip(upper=0.0)).clip(lower=0.0)
    release_size = pd.concat(
        [
            existing_release.rename("existing_release"),
            (current * release_pressure * 0.45).rename("pressure_release"),
        ],
        axis=1,
    ).max(axis=1)
    release_size = release_size.clip(lower=0.0, upper=current)

    target_delta = raw_delta.clip(lower=-float(position_cap), upper=float(position_cap))
    target_delta = target_delta.where(~release_allowed, -release_size)
    target_delta = target_delta.where(~(release_blocked & (target_delta < 0.0)), 0.0)
    target_delta = target_delta.where(held | (target_delta >= 0.0), 0.0)

    headroom = (float(position_cap) - current).clip(lower=0.0)
    receiver_pressure = pd.concat(
        [
            raw_delta.clip(lower=0.0).rename("positive_delta"),
            action.isin({"open", "add"}).astype(float).rename("action_receiver"),
            _max_columns(
                sample_frame,
                (
                    "portfolio_daily_receiver_score",
                    "portfolio_daily_unified_receiver_score",
                    "result_value_deploy_gate_target",
                    "result_value_alpha_opportunity_value",
                ),
            ).clip(0.0, 1.0).rename("receiver_value"),
        ],
        axis=1,
    ).max(axis=1)
    receiver_support = ((headroom > float(deadband)) & (receiver_pressure > 0.0)).astype(float)
    receiver_delta = pd.concat(
        [
            raw_delta.clip(lower=0.0).rename("raw_positive_delta"),
            (headroom * receiver_pressure * 0.70).rename("pressure_deploy"),
        ],
        axis=1,
    ).max(axis=1).clip(lower=0.0, upper=headroom)
    target_delta = target_delta.where(~(receiver_support > 0.0), receiver_delta)
    target_delta = target_delta.where(target_delta.abs() >= float(deadband), 0.0)
    target_weight = (current + target_delta).clip(lower=0.0, upper=float(position_cap))
    target_delta = target_weight - current

    release_intent = release_pressure.where(release_allowed, 0.0).clip(0.0, 1.0)
    source_score = pd.concat(
        [
            release_intent.rename("release_intent"),
            _numeric(sample_frame, "portfolio_daily_source_score", 0.0).rename("source_score"),
            _numeric(sample_frame, "portfolio_daily_unified_source_score", 0.0).rename("unified_source_score"),
        ],
        axis=1,
    ).max(axis=1).where(held, 0.0).clip(0.0, 1.0)
    receiver_score = pd.concat(
        [
            receiver_pressure.rename("receiver_pressure"),
            _numeric(sample_frame, "portfolio_daily_receiver_score", 0.0).rename("receiver_score"),
            _numeric(sample_frame, "portfolio_daily_unified_receiver_score", 0.0).rename("unified_receiver_score"),
        ],
        axis=1,
    ).max(axis=1).where(receiver_support > 0.0, 0.0).clip(0.0, 1.0)

    result["target_weight"] = target_weight.astype(float)
    result["target_delta"] = target_delta.astype(float)
    result["receiver_score"] = receiver_score.astype(float)
    result["receiver_support"] = receiver_support.astype(float)
    result["release_intent"] = release_intent.astype(float)
    result["source_score"] = source_score.astype(float)
    result["source_release_quality"] = release_pressure.where(held, 0.0).clip(0.0, 1.0).astype(float)
    result["source_economic_block_risk"] = block_risk.astype(float)
    result["reduce_quality"] = release_intent.astype(float)
    result["exit_hazard"] = (
        ((action == "exit").astype(float) * 0.70)
        + (release_intent * ((release_size >= current * 0.70) & held).astype(float) * 0.30)
    ).clip(0.0, 1.0).astype(float)
    return result
