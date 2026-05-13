from __future__ import annotations

import numpy as np
import pandas as pd


def derive_cash_timing_intent(
    *,
    risk_score: float,
    deploy_score: float,
    benchmark_downside: float,
    alpha_alignment: float,
) -> float:
    raw = (
        0.46 * float(risk_score)
        + 0.28 * max(float(risk_score) - float(deploy_score), 0.0)
        + 0.26 * float(benchmark_downside)
        - 0.24 * float(deploy_score)
        - 0.18 * float(alpha_alignment)
    )
    return float(np.clip(raw, 0.0, 1.0))


def derive_release_intent_from_target_delta(
    policy_frame: pd.DataFrame,
    *,
    deadband: float = 0.003,
) -> pd.DataFrame:
    result = pd.DataFrame(index=policy_frame.index)
    current_weight = pd.to_numeric(policy_frame.get("current_weight", 0.0), errors="coerce").fillna(0.0)
    target_delta = pd.to_numeric(
        policy_frame.get("portfolio_daily_target_delta_intent", 0.0),
        errors="coerce",
    ).fillna(0.0)
    exit_hazard = pd.to_numeric(policy_frame.get("exit_hazard", 0.0), errors="coerce").fillna(0.0)
    reduce_quality = pd.to_numeric(policy_frame.get("reduce_quality", 0.0), errors="coerce").fillna(0.0)

    held = current_weight > float(deadband)
    release_size = (-target_delta).clip(lower=0.0)
    release_ratio = release_size / current_weight.clip(lower=float(deadband))
    release_score = (
        0.62 * release_ratio.clip(lower=0.0, upper=1.0)
        + 0.22 * exit_hazard.clip(0.0, 1.0)
        + 0.16 * reduce_quality.clip(0.0, 1.0)
    ).where(held, 0.0)

    action = pd.Series("hold", index=policy_frame.index, dtype=object)
    reduce_mask = held & (target_delta < -float(deadband))
    exit_mask = reduce_mask & ((exit_hazard >= 0.62) | (release_size >= current_weight * 0.72))
    action.loc[reduce_mask] = "reduce"
    action.loc[exit_mask] = "exit"

    result["release_intent_score"] = release_score.astype(float)
    result["release_intent_action"] = action
    result["release_intent_delta"] = (-release_size).where(held, 0.0).astype(float)
    return result


def derive_release_first_intent(
    policy_frame: pd.DataFrame,
    *,
    deadband: float = 0.003,
    min_intent: float = 0.35,
) -> pd.DataFrame:
    result = pd.DataFrame(index=policy_frame.index)
    current_weight = pd.to_numeric(policy_frame.get("current_weight", 0.0), errors="coerce").fillna(0.0)
    target_delta = pd.to_numeric(
        policy_frame.get("portfolio_daily_target_delta_intent", 0.0),
        errors="coerce",
    ).fillna(0.0)
    source_score = pd.to_numeric(
        policy_frame.get(
            "portfolio_daily_source_score",
            policy_frame.get("source_score", 0.0),
        ),
        errors="coerce",
    ).fillna(0.0)
    release_quality = pd.to_numeric(
        policy_frame.get(
            "portfolio_daily_source_release_quality",
            policy_frame.get("reduce_quality", policy_frame.get("sell_release_value", 0.0)),
        ),
        errors="coerce",
    ).fillna(0.0)
    exit_hazard = pd.to_numeric(policy_frame.get("exit_hazard", 0.0), errors="coerce").fillna(0.0)
    keep_risk = pd.to_numeric(
        policy_frame.get("portfolio_daily_source_forward_proxy_keep_risk", 0.0),
        errors="coerce",
    ).fillna(0.0)
    block_risk = pd.to_numeric(
        policy_frame.get("portfolio_daily_source_economic_block_risk", 0.0),
        errors="coerce",
    ).fillna(0.0)
    receiver_score = pd.to_numeric(
        policy_frame.get("portfolio_daily_receiver_score", policy_frame.get("receiver_score", 0.0)),
        errors="coerce",
    ).fillna(0.0)

    held = current_weight > float(deadband)
    release_size = (-target_delta).clip(lower=0.0)
    release_ratio = release_size / current_weight.clip(lower=float(deadband))
    raw_score = (
        0.32 * source_score.clip(0.0, 1.0)
        + 0.28 * release_quality.clip(0.0, 1.0)
        + 0.22 * release_ratio.clip(0.0, 1.0)
        + 0.12 * exit_hazard.clip(0.0, 1.0)
        + 0.06 * receiver_score.clip(0.0, 1.0)
        - 0.62 * keep_risk.clip(0.0, 1.0)
        - 0.62 * block_risk.clip(0.0, 1.0)
    )
    release_score = raw_score.clip(lower=0.0, upper=1.0).where(held, 0.0)

    block_reason = pd.Series("none", index=policy_frame.index, dtype=object)
    block_reason.loc[~held] = "not_held"
    block_reason.loc[held & (keep_risk >= 0.78)] = "keep_risk"
    block_reason.loc[held & (keep_risk < 0.78) & (block_risk >= 0.78)] = "economic_block"
    block_reason.loc[held & (keep_risk < 0.78) & (block_risk < 0.78) & (release_score < float(min_intent))] = "below_threshold"

    action = pd.Series("hold", index=policy_frame.index, dtype=object)
    release_mask = held & (release_score >= float(min_intent)) & (block_reason == "none")
    exit_mask = release_mask & ((exit_hazard >= 0.62) | (release_ratio >= 0.72))
    action.loc[release_mask] = "reduce"
    action.loc[exit_mask] = "exit"

    fallback_release = current_weight * release_score.clip(0.0, 1.0) * 0.45
    intended_release = pd.concat([release_size.rename("target_delta_release"), fallback_release.rename("score_release")], axis=1).max(axis=1)
    intended_release = intended_release.clip(lower=0.0, upper=current_weight.clip(lower=0.0))
    result["release_first_intent_score"] = release_score.astype(float)
    result["release_first_intent_delta"] = (-intended_release).where(release_mask, 0.0).astype(float)
    result["release_first_action_hint"] = action
    result["release_first_block_reason"] = block_reason
    return result
