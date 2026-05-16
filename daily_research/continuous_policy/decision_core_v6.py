from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

from torch.utils.data import DataLoader

from daily_research.continuous_policy.model_portfolio_set_v5 import (
    PORTFOLIO_SET_V5_ARTIFACT_FILENAME,
    PORTFOLIO_SET_V5_DECISION_TARGET_NAMES,
    PORTFOLIO_SET_V5_DEFAULT_STRICT_GOLD_DATASET_ID,
    PORTFOLIO_SET_V5_OUTPUT_NAMES,
    PortfolioSetDayDataset,
    PortfolioSetPolicyNetV5,
    _apply_matrix,
    _build_sequence_array,
    _current_weight,
    _date_column,
    _decode_raw,
    _ensure_features,
    _global_defaults,
    _masked_mean,
    _make_model,
    _predict_cashflow_turnover_budget,
    _predict_outputs,
    _prepare_matrix,
    _resolve_static_and_sequence_columns,
    _select_train_days,
    _split_indices,
    autocast_context,
    collate_portfolio_set_days,
    configure_torch_training_acceleration,
    move_to_device,
    project_portfolio_set_v5_cashflow_oracle,
)
from daily_research.continuous_policy.portfolio_cashflow_decision import (
    PORTFOLIO_CASHFLOW_DECISION_MODE_COLUMN,
    normalize_portfolio_cashflow_decision,
)
from daily_research.continuous_policy.portfolio_decision_features import (
    PORTFOLIO_DECISION_FEATURE_BUNDLE_MODE_COLUMN,
    PORTFOLIO_DECISION_FEATURE_CONTRACT_BLOCKER_COLUMN,
    PORTFOLIO_DECISION_FEATURE_CONTRACT_BLOCKER_COUNT_COLUMN,
    PORTFOLIO_DECISION_FEATURE_CONTRACT_DEGRADED_COUNT_COLUMN,
    PORTFOLIO_DECISION_FEATURE_CONTRACT_DEGRADED_REASON_COLUMN,
    PORTFOLIO_DECISION_FEATURE_CONTRACT_MISSING_COUNT_COLUMN,
    PORTFOLIO_DECISION_FEATURE_CONTRACT_NEUTRAL_DEFAULT_COUNT_COLUMN,
    PORTFOLIO_DECISION_FEATURE_CONTRACT_SEVERITY_COLUMN,
    R74_HIGH_VALUE_LAKE_FEATURES,
    attach_portfolio_decision_features,
)
from daily_research.continuous_policy.training_contracts import TRAINER_BACKEND_FORMAL_DECISION_CORE_V6


DECISION_CORE_V6_VERSION = "v6"
DECISION_CORE_V6_ARTIFACT_TYPE = "continuous_policy_decision_core_v6_artifact"
DECISION_CORE_V6_ARTIFACT_FILENAME = "continuous_policy_decision_core_v6_artifact.pt"
DECISION_CORE_V6_PROFILE = "portfolio_decision_core_v6_longrun_candidate"
DECISION_CORE_V6_STRICT_GOLD_DATASET_ID = PORTFOLIO_SET_V5_DEFAULT_STRICT_GOLD_DATASET_ID
DECISION_CORE_V6_STANDARD_COLUMNS: tuple[str, ...] = (
    "decision_core_version",
    "source_supply",
    "receiver_demand",
    "cash_reserve",
    "target_weight",
    "target_delta",
    "intent",
    "reason_code",
    "contract_status",
    "contract_blocker_reason",
)
DECISION_CORE_V6_LOSS_PROFILE_NAMES: tuple[str, ...] = (DECISION_CORE_V6_PROFILE,)
DECISION_CORE_V6_POSITION_CAP = 0.24
DECISION_CORE_V6_DEADBAND = 0.003
DECISION_CORE_V6_MIN_RELEASE_UTILITY = 0.48
DECISION_CORE_V6_MIN_RELEASE_SPREAD = 0.06
DECISION_CORE_V6_MIN_RELEASE_EVIDENCE = 0.22


@dataclass(frozen=True)
class DecisionFrameV6:
    frame: pd.DataFrame
    diagnostics: dict[str, Any]


@dataclass
class TorchDecisionCoreV6Artifact:
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
        model_config.setdefault("decision_core_version", DECISION_CORE_V6_VERSION)
        model_config.setdefault("decision_core_profile", DECISION_CORE_V6_PROFILE)
        payload = {
            "artifact_type": DECISION_CORE_V6_ARTIFACT_TYPE,
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


def load_decision_core_v6_artifact(path: str | Path) -> TorchDecisionCoreV6Artifact:
    payload = torch.load(Path(path), map_location="cpu", weights_only=False)
    if str(payload.get("artifact_type", "") or "") != DECISION_CORE_V6_ARTIFACT_TYPE:
        raise TypeError(f"Unsupported decision-core v6 artifact type: {payload.get('artifact_type')!r}")
    return TorchDecisionCoreV6Artifact(
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


def decision_core_v6_profile_name(value: str | None = None) -> str:
    text = str(value or DECISION_CORE_V6_PROFILE).strip()
    if text != DECISION_CORE_V6_PROFILE:
        raise ValueError(
            f"Unsupported v6 decision-core profile: {value!r}. "
            f"Use {DECISION_CORE_V6_PROFILE!r}; r69/r71/r74 aliases are historical only."
        )
    return DECISION_CORE_V6_PROFILE


def _numeric(frame: pd.DataFrame, name: str, default: float = 0.0) -> pd.Series:
    if name not in frame.columns:
        return pd.Series(float(default), index=frame.index, dtype=float)
    return pd.to_numeric(frame[name], errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(float(default)).astype(float)


def _max_cols(frame: pd.DataFrame, names: tuple[str, ...], default: float = 0.0) -> pd.Series:
    values = [_numeric(frame, name, default).rename(name) for name in names if name in frame.columns]
    if not values:
        return pd.Series(float(default), index=frame.index, dtype=float)
    return pd.concat(values, axis=1).max(axis=1).fillna(float(default)).astype(float)


def _clip01(series: pd.Series, default: float = 0.0) -> pd.Series:
    return pd.to_numeric(series, errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(float(default)).clip(0.0, 1.0).astype(float)


def _positive_pressure(series: pd.Series, scale: float) -> pd.Series:
    return _clip01(pd.to_numeric(series, errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(0.0) / max(float(scale), 1.0e-8))


def _negative_pressure(series: pd.Series, scale: float) -> pd.Series:
    return _clip01(-pd.to_numeric(series, errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(0.0) / max(float(scale), 1.0e-8))


def _rank_pct_fallback(frame: pd.DataFrame, *, rank_name: str, value_name: str) -> pd.Series:
    raw = pd.to_numeric(frame.get(rank_name, pd.Series(np.nan, index=frame.index)), errors="coerce").replace([np.inf, -np.inf], np.nan)
    values = pd.to_numeric(frame.get(value_name, pd.Series(0.0, index=frame.index)), errors="coerce").replace([np.inf, -np.inf], np.nan)
    fallback = values.rank(pct=True, method="average").fillna(0.5) if values.notna().any() else pd.Series(0.5, index=frame.index)
    return raw.fillna(fallback).clip(0.0, 1.0).astype(float)


def _v6_objective_components(
    policy: pd.DataFrame,
    *,
    source_score: pd.Series,
    receiver_score: pd.Series,
    cash_score: pd.Series,
    current: pd.Series,
    deadband: float = DECISION_CORE_V6_DEADBAND,
) -> dict[str, pd.Series]:
    index = policy.index
    held = current > float(deadband)
    headroom = (DECISION_CORE_V6_POSITION_CAP - current).clip(lower=0.0)
    score_rank = _rank_pct_fallback(policy, rank_name="score_rank_pct", value_name="score_blend")
    alpha_coverage = _clip01(_numeric(policy, "alpha_prior_coverage", 0.0))
    alpha_rank = _clip01(_numeric(policy, "alpha_prior_rank_pct", 0.5), default=0.5) * alpha_coverage
    ret_5d = _numeric(policy, "ret_5d", 0.0)
    ret_20d = _numeric(policy, "ret_20d", 0.0)
    ret5_pos = _positive_pressure(ret_5d, 0.08)
    ret20_pos = _positive_pressure(ret_20d, 0.16)
    ret5_neg = _negative_pressure(ret_5d, 0.08)
    ret20_neg = _negative_pressure(ret_20d, 0.16)
    score_delta_pos = _positive_pressure(_numeric(policy, "score_delta_5d", 0.0), 0.75)
    score_delta_neg = _negative_pressure(_numeric(policy, "score_delta_5d", 0.0), 0.75)
    local_peak_gap = _positive_pressure(-_numeric(policy, "distance_to_20d_high", 0.0), 0.18)
    near_high = (1.0 - local_peak_gap).clip(0.0, 1.0)
    drawdown_pressure = _positive_pressure(-_numeric(policy, "drawdown_from_peak", 0.0), 0.16)
    loss_pressure = _negative_pressure(_numeric(policy, "unrealized_pnl", 0.0), 0.18)
    pnl_strength = _positive_pressure(_numeric(policy, "unrealized_pnl", 0.0), 0.18)

    cash_regime = _clip01(
        pd.concat(
            [
                _numeric(policy, "cash_regime_pressure", 0.0).mul(2.2),
                _numeric(policy, "market_downside_pressure", 0.0).mul(3.0),
                _negative_pressure(_numeric(policy, "portfolio_recent_return_20d", 0.0), 0.08),
                _negative_pressure(_numeric(policy, "benchmark_trend_gap", 0.0), 0.06),
                _numeric(policy, "recent_reversal_rate_20d", 0.0).mul(2.5),
                _numeric(policy, "portfolio_cash_pressure", 0.0).mul(1.8),
            ],
            axis=1,
        ).max(axis=1)
    )
    low_score = (1.0 - score_rank).clip(0.0, 1.0)
    low_alpha = (1.0 - alpha_rank).clip(0.0, 1.0) * alpha_coverage
    source_weakness = (
        low_score * 0.34
        + low_alpha * 0.08
        + ret5_neg * 0.16
        + ret20_neg * 0.10
        + score_delta_neg * 0.10
        + local_peak_gap * 0.10
        + drawdown_pressure * 0.07
        + loss_pressure * 0.05
        + cash_regime * low_score * 0.12
    ).clip(0.0, 1.0)
    source_floor_gate = held & (
        (low_score >= 0.55)
        | (ret5_neg >= 0.25)
        | (ret20_neg >= 0.25)
        | (score_delta_neg >= 0.28)
        | (local_peak_gap >= 0.40)
    )
    source_weakness = source_weakness.where(~source_floor_gate, np.maximum(source_weakness, 0.24 + 0.20 * cash_regime))
    keep_strength = (
        score_rank * 0.36
        + alpha_rank * 0.14
        + ret5_pos * 0.18
        + ret20_pos * 0.12
        + score_delta_pos * 0.08
        + near_high * 0.07
        + pnl_strength * 0.05
    ).clip(0.0, 1.0)
    recent_buy_protection = _clip01(
        _numeric(policy, "recent_buy_flag", 0.0).mul(0.65)
        + _numeric(policy, "holding_age_short", 0.0).mul(0.35)
        + _numeric(policy, "exit_reentry_pressure", 0.0).mul(0.35)
    )
    keep_strength = pd.concat(
        [
            keep_strength.rename("keep_strength"),
            recent_buy_protection.rename("recent_buy_protection"),
            _numeric(policy, "hold_continuity_pressure", 0.0).clip(0.0, 1.0).rename("hold_continuity"),
        ],
        axis=1,
    ).max(axis=1).clip(0.0, 1.0)
    source_base = _clip01(source_score)
    release_conviction = _numeric(policy, "portfolio_daily_source_release_conviction", 0.0).clip(0.0, 1.0)
    release_quality = _numeric(policy, "portfolio_daily_source_release_quality", 0.0).clip(0.0, 1.0)
    source_forward_spread = _numeric(policy, "portfolio_daily_source_forward_spread_score", 0.0).clip(0.0, 1.0)
    source_bad_forward_spread = _numeric(policy, "portfolio_daily_source_bad_forward_spread_risk", 0.0).clip(0.0, 1.0)
    release_evidence = pd.concat(
        [
            source_base.rename("model_or_label_source"),
            release_conviction.rename("release_conviction"),
            release_quality.mul(0.85).rename("release_quality"),
            source_forward_spread.rename("source_forward_spread"),
            _numeric(policy, "portfolio_daily_source_economic_release_score", 0.0)
            .clip(0.0, 1.0)
            .rename("economic_release"),
        ],
        axis=1,
    ).max(axis=1).where(held, 0.0).fillna(0.0).clip(0.0, 1.0)
    source_shaped = pd.concat(
        [
            source_weakness.mul((0.12 + 0.88 * release_evidence).clip(0.0, 1.0)).rename("source_weakness_with_evidence"),
            source_base.mul((1.0 - 0.90 * keep_strength).clip(0.05, 1.0)).rename("model_source_after_keep"),
            (
                source_forward_spread
                * (0.42 + 0.42 * source_weakness).clip(0.0, 0.84)
                * (1.0 - 0.65 * keep_strength).clip(0.10, 1.0)
            ).rename("forward_spread_source"),
        ],
        axis=1,
    ).max(axis=1)
    strong_keep = held & (keep_strength >= 0.68) & (source_weakness < 0.46)
    source_shaped = pd.Series(
        np.where(strong_keep, np.minimum(source_shaped, 0.06 + 0.18 * source_weakness), source_shaped),
        index=index,
        dtype=float,
    ).where(held, 0.0).fillna(0.0).clip(0.0, 1.0)

    receiver_quality = (
        score_rank * 0.38
        + alpha_rank * 0.15
        + ret5_pos * 0.16
        + ret20_pos * 0.10
        + score_delta_pos * 0.08
        + near_high * 0.08
        + _positive_pressure(_numeric(policy, "score_delta_1d", 0.0), 0.35) * 0.05
    ).clip(0.0, 1.0)
    receiver_base = _clip01(receiver_score)
    receiver_shaped = pd.concat(
        [
            receiver_base.mul((0.45 + 0.65 * receiver_quality).clip(0.15, 1.0)).rename("model_receiver_after_quality"),
            receiver_quality.mul((1.0 - 0.50 * cash_regime).clip(0.10, 1.0)).rename("receiver_quality_floor"),
        ],
        axis=1,
    ).max(axis=1).where(headroom > float(deadband), 0.0).fillna(0.0).clip(0.0, 1.0)
    defense_value = pd.concat(
        [
            _clip01(cash_score).rename("model_cash"),
            cash_regime.rename("cash_regime"),
            _numeric(policy, "portfolio_drawdown_20d", 0.0).mul(-3.0).rename("portfolio_drawdown"),
        ],
        axis=1,
    ).max(axis=1).fillna(0.0).clip(0.0, 1.0)
    receiver_top = float(receiver_quality.where(headroom > float(deadband)).max(skipna=True)) if len(receiver_quality) else 0.0
    receiver_top = receiver_top if np.isfinite(receiver_top) else 0.0
    source_spread = pd.Series(receiver_top, index=index, dtype=float).sub(keep_strength).clip(0.0, 1.0).where(held, 0.0)
    receiver_spread = pd.concat(
        [
            _numeric(policy, "portfolio_decision_r74_receiver_source_spread_value", 0.0).clip(0.0, 1.0).rename("lake_spread"),
            receiver_quality.where(headroom > float(deadband), 0.0).rename("receiver_quality"),
            source_spread.rename("source_rotation_spread"),
            source_forward_spread.rename("source_forward_spread"),
        ],
        axis=1,
    ).max(axis=1).fillna(0.0).clip(0.0, 1.0)
    source_opportunity_cost = pd.concat(
        [
            keep_strength.rename("keep_strength"),
            ret5_pos.rename("ret5_pos"),
            ret20_pos.rename("ret20_pos"),
            recent_buy_protection.rename("recent_buy_protection"),
            _numeric(policy, "portfolio_daily_source_forward_proxy_keep_risk", 0.0)
            .clip(0.0, 1.0)
            .rename("forward_proxy_keep_risk"),
            _numeric(policy, "portfolio_daily_source_positive_forward_penalty", 0.0)
            .clip(0.0, 1.0)
            .rename("positive_forward_penalty"),
            source_bad_forward_spread.rename("bad_forward_spread"),
            _numeric(policy, "portfolio_daily_source_opportunity_cost_penalty", 0.0)
            .clip(0.0, 1.0)
            .rename("opportunity_cost_penalty"),
            _numeric(policy, "portfolio_daily_source_strong_false_sell_penalty", 0.0)
            .clip(0.0, 1.0)
            .rename("strong_false_sell_penalty"),
            _numeric(policy, "portfolio_daily_source_tail_false_sell_penalty", 0.0)
            .clip(0.0, 1.0)
            .rename("tail_false_sell_penalty"),
        ],
        axis=1,
    ).max(axis=1).where(held, 0.0).fillna(0.0).clip(0.0, 1.0)
    false_sell_penalty = pd.concat(
        [
            _numeric(policy, "portfolio_daily_source_strong_false_sell_penalty", 0.0)
            .clip(0.0, 1.0)
            .rename("strong_false_sell"),
            _numeric(policy, "portfolio_daily_source_hard_negative_penalty", 0.0)
            .clip(0.0, 1.0)
            .rename("hard_negative"),
            _numeric(policy, "portfolio_daily_source_tail_false_sell_penalty", 0.0)
            .clip(0.0, 1.0)
            .rename("tail_false_sell"),
            _numeric(policy, "portfolio_daily_source_positive_forward_penalty", 0.0)
            .clip(0.0, 1.0)
            .mul(0.85)
            .rename("positive_forward"),
            source_bad_forward_spread.mul(0.90).rename("bad_forward_spread"),
        ],
        axis=1,
    ).max(axis=1).where(held, 0.0).fillna(0.0).clip(0.0, 1.0)
    wrong_side_penalty = pd.concat(
        [
            _numeric(policy, "portfolio_decision_r74_source_quality_penalty", 0.0).clip(0.0, 1.0).rename("lake_source_penalty"),
            source_opportunity_cost.rename("source_opportunity_cost"),
            false_sell_penalty.rename("false_sell_penalty"),
        ],
        axis=1,
    ).max(axis=1).fillna(0.0).clip(0.0, 1.0)
    reversal_penalty = pd.concat(
        [
            recent_buy_protection.rename("recent_buy_protection"),
            _numeric(policy, "reduce_reversal_pressure", 0.0).clip(0.0, 1.0).rename("reduce_reversal_pressure"),
            _numeric(policy, "portfolio_daily_source_forward_strength_brake_risk", 0.0)
            .clip(0.0, 1.0)
            .mul(0.70)
            .rename("forward_strength_brake"),
            false_sell_penalty.mul(0.45).rename("false_sell_penalty"),
        ],
        axis=1,
    ).max(axis=1).where(held, 0.0).fillna(0.0).clip(0.0, 1.0)
    release_utility = (
        source_weakness * 0.52
        + release_conviction * 0.28
        + cash_regime * 0.14
        + receiver_spread * 0.16
        + source_forward_spread * 0.28
        - source_opportunity_cost * 0.38
        - wrong_side_penalty * 0.28
        - reversal_penalty * 0.20
        - source_bad_forward_spread * 0.20
        - keep_strength * 0.18
    ).where(held, 0.0).fillna(0.0).clip(0.0, 1.0)
    weakness_release_gate = (
        (source_weakness >= 0.58)
        & (pd.concat([receiver_spread.rename("receiver_spread"), source_forward_spread.rename("source_forward")], axis=1).max(axis=1) >= DECISION_CORE_V6_MIN_RELEASE_SPREAD)
        & (keep_strength <= 0.58)
    )
    model_release_gate = (
        (source_base >= 0.80)
        & (source_weakness >= 0.30)
        & (keep_strength <= 0.50)
    )
    defensive_release_gate = (
        (cash_regime >= 0.72)
        & (source_weakness >= 0.48)
        & (keep_strength <= 0.55)
        & (receiver_spread >= 0.03)
    )
    release_gate = held & (
        ((release_utility >= DECISION_CORE_V6_MIN_RELEASE_UTILITY) & (release_evidence >= DECISION_CORE_V6_MIN_RELEASE_EVIDENCE))
        | model_release_gate
        | weakness_release_gate
        | defensive_release_gate
    )
    high_false_sell_risk = (
        pd.concat(
            [
                source_opportunity_cost.rename("source_opportunity"),
                wrong_side_penalty.rename("wrong_side"),
                reversal_penalty.rename("reversal"),
                false_sell_penalty.rename("false_sell"),
            ],
            axis=1,
        ).max(axis=1)
        >= 0.72
    )
    release_gate = release_gate & (
        (~high_false_sell_risk)
        | ((source_weakness >= 0.66) & (keep_strength <= 0.42))
        | ((release_evidence >= 0.74) & (receiver_spread >= 0.25) & (keep_strength <= 0.58))
        | ((source_forward_spread >= 0.34) & (source_weakness >= 0.50) & (keep_strength <= 0.52))
    )
    source_shaped = source_shaped.where(release_gate, 0.0)
    release_value = source_weakness.where(release_gate, 0.0)
    return {
        "source_score": source_shaped,
        "receiver_score": receiver_shaped,
        "cash_score": defense_value,
        "deploy_value": receiver_quality.where(headroom > float(deadband), 0.0).fillna(0.0).clip(0.0, 1.0),
        "release_value": release_value.where(held, 0.0).fillna(0.0).clip(0.0, 1.0),
        "defense_value": defense_value,
        "cash_timing_value": defense_value,
        "receiver_source_spread_value": receiver_spread,
        "source_opportunity_cost": source_opportunity_cost,
        "source_wrong_side_sell_penalty": wrong_side_penalty,
        "reversal_risk_penalty": reversal_penalty,
        "release_utility": release_utility,
        "release_gate": release_gate.astype(float),
        "source_release_evidence": release_evidence,
        "source_false_sell_penalty": false_sell_penalty,
        "source_forward_spread_evidence": source_forward_spread.where(held, 0.0).fillna(0.0).clip(0.0, 1.0),
        "source_weakness": source_weakness.where(held, 0.0).fillna(0.0).clip(0.0, 1.0),
        "source_keep_strength": keep_strength.where(held, 0.0).fillna(0.0).clip(0.0, 1.0),
        "receiver_quality": receiver_quality.where(headroom > float(deadband), 0.0).fillna(0.0).clip(0.0, 1.0),
    }


def _safe_ratio(numer: float, denom: float) -> float:
    return float(numer / denom) if float(denom) > 0.0 else 0.0


DECISION_CORE_V6_SCORE_FAMILY: tuple[str, ...] = (
    "score_blend",
    "score_rank_pct",
    "score_v2",
    "score_none",
    "alpha_prior_score_z",
    "alpha_prior_rank_pct",
)
DECISION_CORE_V6_MEMBERSHIP_FAMILY: tuple[str, ...] = ("in_pool", "current_weight", "holding_flag")
DECISION_CORE_V6_PRICE_FAMILY: tuple[str, ...] = (
    "close",
    "open",
    "high",
    "low",
    "current_price",
    "price",
    "price_rank",
    "ma20_gap",
    "ma60_gap",
    "distance_to_20d_high",
    "distance_to_60d_high",
    "distance_to_20d_low",
)
DECISION_CORE_V6_RETURN_VOL_FAMILY: tuple[str, ...] = (
    "ret_1d",
    "ret_3d",
    "ret_5d",
    "ret_10d",
    "ret_20d",
    "vol_5d",
    "vol_20d",
    "ret_accel_5_20",
    "volatility_expansion",
    "portfolio_return_vol_20d",
)
DECISION_CORE_V6_DIAGNOSTIC_ONLY_MISSING: tuple[str, ...] = (
    "z_volatility_contraction",
    "z_volume_contraction",
)


def _feature_family_available(frame: pd.DataFrame, names: tuple[str, ...]) -> pd.Series:
    available = pd.Series(False, index=frame.index, dtype=bool)
    for name in names:
        if name not in frame.columns:
            continue
        values = frame[name]
        numeric = pd.to_numeric(values, errors="coerce")
        if numeric.notna().any():
            present = numeric.replace([np.inf, -np.inf], np.nan).notna()
        else:
            present = values.notna() & values.astype(str).str.strip().ne("")
        available = available | present.astype(bool)
    return available


def _append_contract_reason(reasons: pd.Series, mask: pd.Series, reason: str) -> pd.Series:
    if not bool(mask.any()):
        return reasons
    updated = reasons.copy()
    empty = updated.eq("none") | updated.eq("")
    updated.loc[mask & empty] = reason
    updated.loc[mask & (~empty) & (~updated.str.contains(reason, regex=False))] = (
        updated.loc[mask & (~empty) & (~updated.str.contains(reason, regex=False))] + ";" + reason
    )
    return updated


def attach_v6_feature_contract(frame: pd.DataFrame) -> pd.DataFrame:
    enriched = attach_portfolio_decision_features(frame.copy(), mode="predict")
    if frame.empty:
        enriched["decision_core_v6_feature_contract_blocker_count"] = 1.0
        enriched["decision_core_v6_feature_contract_degraded_count"] = 0.0
        enriched["decision_core_v6_feature_contract_neutral_fallback_count"] = 0.0
        enriched["decision_core_v6_feature_contract_diagnostic_missing_count"] = 0.0
        enriched["decision_core_v6_feature_contract_status"] = "blocker"
        enriched["decision_core_v6_feature_contract_blocker_reason"] = "empty_frame"
        enriched["decision_core_v6_feature_contract_degraded_reason"] = "none"
        enriched["decision_core_v6_feature_contract_neutral_fallback_reason"] = "none"
        enriched["decision_core_v6_feature_contract_diagnostic_missing_reason"] = "none"
        enriched["decision_core_v6_feature_contract_degraded_feature_list"] = ""
        return enriched
    missing_high_value = [name for name in R74_HIGH_VALUE_LAKE_FEATURES if name not in frame.columns]
    score_available = _feature_family_available(frame, DECISION_CORE_V6_SCORE_FAMILY)
    membership_available = _feature_family_available(frame, DECISION_CORE_V6_MEMBERSHIP_FAMILY)
    price_available = _feature_family_available(frame, DECISION_CORE_V6_PRICE_FAMILY)
    return_vol_available = _feature_family_available(frame, DECISION_CORE_V6_RETURN_VOL_FAMILY)

    blocker_reason = pd.Series("none", index=enriched.index, dtype=object)
    blocker_reason = _append_contract_reason(blocker_reason, ~score_available, "missing_score_family")
    blocker_reason = _append_contract_reason(blocker_reason, ~membership_available, "missing_membership_family")
    legacy_blocker = enriched[PORTFOLIO_DECISION_FEATURE_CONTRACT_BLOCKER_COLUMN].fillna("none").astype(str)
    legacy_blocker_mask = legacy_blocker.ne("none") & legacy_blocker.ne("")
    for legacy_reason in sorted(str(value) for value in legacy_blocker.loc[legacy_blocker_mask].unique()):
        blocker_reason = _append_contract_reason(
            blocker_reason,
            legacy_blocker_mask & legacy_blocker.eq(legacy_reason),
            f"legacy_{legacy_reason}",
        )
    blocker_count = (~score_available).astype(float) + (~membership_available).astype(float)
    blocker_count = blocker_count + legacy_blocker_mask.astype(float)

    degraded_reason = pd.Series("none", index=enriched.index, dtype=object)
    degraded_reason = _append_contract_reason(degraded_reason, ~price_available, "missing_price_family")
    degraded_reason = _append_contract_reason(
        degraded_reason,
        ~return_vol_available,
        "missing_core_return_volatility_family",
    )
    degraded_count = (~price_available).astype(float) + (~return_vol_available).astype(float)

    diagnostic_missing = [name for name in missing_high_value if name in DECISION_CORE_V6_DIAGNOSTIC_ONLY_MISSING]
    neutral_missing = [name for name in missing_high_value if name not in DECISION_CORE_V6_DIAGNOSTIC_ONLY_MISSING]
    neutral_count = pd.Series(float(len(neutral_missing)), index=enriched.index, dtype=float)
    diagnostic_count = pd.Series(float(len(diagnostic_missing)), index=enriched.index, dtype=float)
    neutral_reason = pd.Series("none", index=enriched.index, dtype=object)
    diagnostic_reason = pd.Series("none", index=enriched.index, dtype=object)
    if missing_high_value:
        if neutral_missing:
            neutral_reason = pd.Series(
                "feature_family_neutral_fallback_allowed:" + ",".join(sorted(neutral_missing)),
                index=enriched.index,
                dtype=object,
            )
        if diagnostic_missing:
            diagnostic_reason = pd.Series(
                "diagnostic_only_missing:" + ",".join(sorted(diagnostic_missing)),
                index=enriched.index,
                dtype=object,
            )
    severity = pd.Series("ok", index=enriched.index, dtype=object)
    severity.loc[degraded_count > 0.0] = "degraded"
    severity.loc[blocker_count > 0.0] = "blocker"
    enriched["decision_core_v6_feature_contract_blocker_count"] = blocker_count.astype(float)
    enriched["decision_core_v6_feature_contract_degraded_count"] = degraded_count.astype(float)
    enriched["decision_core_v6_feature_contract_neutral_fallback_count"] = neutral_count.astype(float)
    enriched["decision_core_v6_feature_contract_diagnostic_missing_count"] = diagnostic_count.astype(float)
    enriched["decision_core_v6_feature_contract_status"] = severity.astype(str)
    enriched["decision_core_v6_feature_contract_blocker_reason"] = blocker_reason.astype(str)
    enriched["decision_core_v6_feature_contract_degraded_reason"] = degraded_reason.astype(str)
    enriched["decision_core_v6_feature_contract_neutral_fallback_reason"] = neutral_reason.astype(str)
    enriched["decision_core_v6_feature_contract_diagnostic_missing_reason"] = diagnostic_reason.astype(str)
    enriched["decision_core_v6_feature_contract_degraded_feature_list"] = degraded_reason.where(
        degraded_reason.ne("none"),
        "",
    ).astype(str)
    return enriched


def _project_v6_decision_day(
    policy: pd.DataFrame,
    *,
    deadband: float = DECISION_CORE_V6_DEADBAND,
) -> pd.DataFrame:
    if policy.empty:
        return pd.DataFrame(index=policy.index)
    policy = attach_v6_feature_contract(policy.copy())
    current = _current_weight(policy)
    held = current > float(deadband)
    headroom = (DECISION_CORE_V6_POSITION_CAP - current).clip(lower=0.0)
    action = policy.get("action_label", pd.Series("hold", index=policy.index)).fillna("hold").astype(str).str.lower()
    raw_delta = _numeric(policy, "portfolio_daily_target_delta_intent", np.nan)
    if raw_delta.isna().all() and "target_delta_hint" in policy.columns:
        raw_delta = _numeric(policy, "target_delta_hint", 0.0)
    raw_delta = raw_delta.fillna(0.0).clip(-0.18, 0.18)

    source_score = _max_cols(
        policy,
        (
            "portfolio_decision_r74_release_value",
            "portfolio_daily_source_score",
            "portfolio_daily_unified_source_score",
            "portfolio_daily_source_release_quality",
            "portfolio_daily_source_economic_release_score",
            "portfolio_daily_source_release_preference",
            "release_value_target",
            "sell_release_value",
        ),
    ).clip(0.0, 1.0)
    receiver_score = _max_cols(
        policy,
        (
            "portfolio_decision_r74_deploy_value",
            "portfolio_daily_receiver_score",
            "portfolio_daily_unified_receiver_score",
            "deploy_value_target",
            "result_value_deploy_value_target",
            "result_value_alpha_opportunity_value",
            "alpha_opportunity_value",
        ),
    ).clip(0.0, 1.0)
    cash_score = _max_cols(
        policy,
        (
            "portfolio_decision_r74_cash_defense_value",
            "budget_cash_timing_signal_target",
            "portfolio_daily_cash_score",
            "portfolio_daily_unified_cash_score",
            "cash_defense_value",
            "defense_value_target",
            "market_downside_pressure",
        ),
    ).clip(0.0, 1.0)
    source_score = pd.concat(
        [
            source_score.rename("source_score"),
            action.isin({"reduce", "exit"}).astype(float).rename("source_action"),
            (-raw_delta.clip(upper=0.0) / current.clip(lower=float(deadband))).clip(0.0, 1.0).rename("negative_delta"),
        ],
        axis=1,
    ).max(axis=1).where(held, 0.0).fillna(0.0).clip(0.0, 1.0)
    receiver_score = pd.concat(
        [
            receiver_score.rename("receiver_score"),
            action.isin({"open", "add"}).astype(float).rename("receiver_action"),
            (raw_delta.clip(lower=0.0) / headroom.clip(lower=float(deadband))).clip(0.0, 1.0).rename("positive_delta"),
        ],
        axis=1,
    ).max(axis=1).where(headroom > float(deadband), 0.0).fillna(0.0).clip(0.0, 1.0)
    components = _v6_objective_components(
        policy,
        source_score=source_score,
        receiver_score=receiver_score,
        cash_score=cash_score,
        current=current,
        deadband=deadband,
    )
    source_score = components["source_score"]
    receiver_score = components["receiver_score"]
    cash_score = components["cash_score"]
    turnover_budget = min(float(_global_defaults(policy).get("turnover_budget", 0.08) or 0.08), 0.08)
    oracle = project_portfolio_set_v5_cashflow_oracle(
        current_weight=torch.as_tensor(current.to_numpy(dtype=np.float32)[None, :], dtype=torch.float32),
        source_score=torch.as_tensor(source_score.to_numpy(dtype=np.float32)[None, :], dtype=torch.float32),
        receiver_score=torch.as_tensor(receiver_score.to_numpy(dtype=np.float32)[None, :], dtype=torch.float32),
        cash_buffer_score=torch.as_tensor(cash_score.to_numpy(dtype=np.float32)[None, :], dtype=torch.float32),
        deploy_value=torch.as_tensor(components["deploy_value"].to_numpy(dtype=np.float32)[None, :], dtype=torch.float32),
        release_value=torch.as_tensor(components["release_value"].to_numpy(dtype=np.float32)[None, :], dtype=torch.float32),
        defense_value=torch.as_tensor(components["defense_value"].to_numpy(dtype=np.float32)[None, :], dtype=torch.float32),
        cash_timing_value=torch.as_tensor(components["cash_timing_value"].to_numpy(dtype=np.float32)[None, :], dtype=torch.float32),
        receiver_source_spread_value=torch.as_tensor(components["receiver_source_spread_value"].to_numpy(dtype=np.float32)[None, :], dtype=torch.float32),
        source_opportunity_cost=torch.as_tensor(components["source_opportunity_cost"].to_numpy(dtype=np.float32)[None, :], dtype=torch.float32),
        reversal_risk_penalty=torch.as_tensor(components["reversal_risk_penalty"].to_numpy(dtype=np.float32)[None, :], dtype=torch.float32),
        source_wrong_side_sell_penalty=torch.as_tensor(components["source_wrong_side_sell_penalty"].to_numpy(dtype=np.float32)[None, :], dtype=torch.float32),
        turnover_budget=torch.as_tensor([turnover_budget], dtype=torch.float32),
        risk_budget=torch.as_tensor([float(components["defense_value"].mean()) if len(cash_score) else 0.0], dtype=torch.float32),
        sample_mask=torch.ones((1, len(policy)), dtype=torch.bool),
    )
    target_weight = pd.Series(oracle["target_weight"][0].detach().cpu().numpy().astype(float), index=policy.index)
    target_delta = (target_weight - current).clip(-0.18, 0.18)
    target_delta = target_delta.where(target_delta.abs() >= float(deadband), 0.0)
    target_weight = (current + target_delta).clip(0.0, DECISION_CORE_V6_POSITION_CAP)
    source_supply = pd.Series(oracle["source_supply"][0].detach().cpu().numpy().astype(float), index=policy.index)
    receiver_demand = pd.Series(oracle["receiver_demand"][0].detach().cpu().numpy().astype(float), index=policy.index)
    result = pd.DataFrame(index=policy.index)
    result["target_weight"] = target_weight.astype(float)
    result["target_delta"] = target_delta.astype(float)
    result["source_supply_score"] = source_score.astype(float)
    result["receiver_demand_score"] = receiver_score.astype(float)
    result["cash_buffer_score"] = cash_score.astype(float)
    result["source_weakness"] = components["source_weakness"].astype(float)
    result["source_keep_strength"] = components["source_keep_strength"].astype(float)
    result["receiver_quality"] = components["receiver_quality"].astype(float)
    result["deploy_value"] = components["deploy_value"].astype(float)
    result["release_value"] = components["release_value"].astype(float)
    result["defense_value"] = components["defense_value"].astype(float)
    result["cash_timing_value"] = components["cash_timing_value"].astype(float)
    result["source_opportunity_cost"] = components["source_opportunity_cost"].astype(float)
    result["reversal_risk_penalty"] = components["reversal_risk_penalty"].astype(float)
    result["source_wrong_side_sell_penalty"] = components["source_wrong_side_sell_penalty"].astype(float)
    result["receiver_source_spread_value"] = components["receiver_source_spread_value"].astype(float)
    result["release_utility"] = components["release_utility"].astype(float)
    result["release_gate"] = components["release_gate"].astype(float)
    result["source_release_evidence"] = components["source_release_evidence"].astype(float)
    result["source_false_sell_penalty"] = components["source_false_sell_penalty"].astype(float)
    result["release_intent"] = (source_supply / current.clip(lower=float(deadband))).replace([np.inf, -np.inf], np.nan).fillna(0.0).clip(0.0, 1.0)
    result["reduce_quality"] = source_score.astype(float)
    result["exit_hazard"] = source_score.where(target_weight <= float(deadband), 0.0).astype(float)
    result["source_supply"] = source_supply.astype(float)
    result["receiver_demand"] = receiver_demand.astype(float)
    result["cash_buffer"] = float(oracle["cash_buffer"][0].detach().cpu())
    result["cash_reserve"] = result["cash_buffer"]
    result["turnover_budget"] = float(oracle["turnover_budget"][0].detach().cpu())
    result["risk_budget"] = float(oracle["risk_budget"][0].detach().cpu())
    result["constraint_violation"] = float(oracle["constraint_violation"][0].detach().cpu())
    result["decision_value"] = float(oracle["decision_value"][0].detach().cpu())
    result["target_delta_weight_conflict"] = (((target_weight - current) - target_delta).abs() > 1.0e-6).astype(float)
    result["held_mask"] = held.astype(float)
    result["source_mask"] = (result["source_supply"] > float(deadband)).astype(float)
    result["receiver_mask"] = (result["receiver_demand"] > float(deadband)).astype(float)
    result["decision_core_version"] = DECISION_CORE_V6_VERSION
    result["contract_status"] = policy["decision_core_v6_feature_contract_status"].astype(str)
    result["contract_blocker_reason"] = policy["decision_core_v6_feature_contract_blocker_reason"].astype(str)
    result["intent"] = "hold"
    result.loc[(current <= float(deadband)) & (target_delta > float(deadband)), "intent"] = "open"
    result.loc[(current > float(deadband)) & (target_delta > float(deadband)), "intent"] = "add"
    result.loc[(current > float(deadband)) & (target_delta < -float(deadband)) & (target_weight > float(deadband)), "intent"] = "reduce"
    result.loc[(current > float(deadband)) & (target_delta < -float(deadband)) & (target_weight <= float(deadband)), "intent"] = "exit"
    result["reason_code"] = "neutral"
    result.loc[result["intent"].isin(["open", "add"]), "reason_code"] = "receiver_demand"
    result.loc[result["intent"].isin(["reduce", "exit"]), "reason_code"] = "source_supply"
    result.loc[cash_score >= pd.concat([source_score, receiver_score], axis=1).max(axis=1), "reason_code"] = "cash_reserve"
    for name in PORTFOLIO_SET_V5_DECISION_TARGET_NAMES:
        if name not in result.columns:
            result[name] = 0.0
    return result


def build_decision_core_v6_targets(sample_frame: pd.DataFrame, *, deadband: float = DECISION_CORE_V6_DEADBAND) -> pd.DataFrame:
    if sample_frame.empty:
        return _project_v6_decision_day(sample_frame, deadband=deadband)
    try:
        date_col = _date_column(sample_frame)
    except ValueError:
        return _project_v6_decision_day(sample_frame, deadband=deadband)

    pieces = [
        _project_v6_decision_day(day_frame, deadband=deadband)
        for _, day_frame in sample_frame.groupby(sample_frame[date_col].astype(str), sort=True)
    ]
    if not pieces:
        return _project_v6_decision_day(sample_frame, deadband=deadband)
    return pd.concat(pieces, axis=0).reindex(sample_frame.index)


def build_decision_frame_v6(
    *,
    state_frame: pd.DataFrame,
    outputs: pd.DataFrame,
    turnover_budget: float,
) -> DecisionFrameV6:
    if state_frame.empty:
        raise ValueError("state_frame is empty.")
    policy = attach_v6_feature_contract(state_frame.copy())
    current = _current_weight(policy)
    source_score = outputs["source_supply_score"].clip(0.0, 1.0)
    receiver_score = outputs["receiver_demand_score"].clip(0.0, 1.0)
    cash_score = outputs["cash_buffer_score"].clip(0.0, 1.0)
    deploy_value = pd.concat(
        [
            receiver_score.rename("receiver_output"),
            _numeric(policy, "portfolio_decision_r74_deploy_value", 0.0).rename("lake_deploy"),
            _numeric(policy, "portfolio_daily_receiver_score", 0.0).rename("receiver_feature"),
            _numeric(policy, "portfolio_daily_unified_receiver_score", 0.0).rename("receiver_unified"),
        ],
        axis=1,
    ).max(axis=1).fillna(0.0).clip(0.0, 1.0)
    release_value = pd.concat(
        [
            source_score.rename("source_output"),
            _numeric(policy, "portfolio_decision_r74_release_value", 0.0).rename("lake_release"),
            _numeric(policy, "portfolio_daily_source_release_quality", 0.0).rename("source_feature"),
            _numeric(policy, "portfolio_daily_unified_source_score", 0.0).rename("source_unified"),
        ],
        axis=1,
    ).max(axis=1).where(current > DECISION_CORE_V6_DEADBAND, 0.0).fillna(0.0).clip(0.0, 1.0)
    defense_value = pd.concat(
        [
            cash_score.rename("cash_output"),
            _numeric(policy, "portfolio_decision_r74_cash_defense_value", 0.0).rename("lake_cash_defense"),
            _numeric(policy, "cash_defense_value", 0.0).rename("cash_defense"),
            _numeric(policy, "market_downside_pressure", 0.0).rename("market_downside"),
        ],
        axis=1,
    ).max(axis=1).fillna(0.0).clip(0.0, 1.0)
    receiver_spread = _numeric(policy, "portfolio_decision_r74_receiver_source_spread_value", 0.0).clip(0.0, 1.0)
    source_penalty = _numeric(policy, "portfolio_decision_r74_source_quality_penalty", 0.0).clip(0.0, 1.0)
    components = _v6_objective_components(
        policy,
        source_score=source_score,
        receiver_score=receiver_score,
        cash_score=cash_score,
        current=current,
    )
    source_score = components["source_score"]
    receiver_score = components["receiver_score"]
    cash_score = components["cash_score"]
    deploy_value = components["deploy_value"]
    release_value = components["release_value"]
    defense_value = components["defense_value"]
    receiver_spread = components["receiver_source_spread_value"]
    source_penalty = components["source_wrong_side_sell_penalty"]
    oracle = project_portfolio_set_v5_cashflow_oracle(
        current_weight=torch.as_tensor(current.to_numpy(dtype=np.float32)[None, :], dtype=torch.float32),
        source_score=torch.as_tensor(source_score.to_numpy(dtype=np.float32)[None, :], dtype=torch.float32),
        receiver_score=torch.as_tensor(receiver_score.to_numpy(dtype=np.float32)[None, :], dtype=torch.float32),
        cash_buffer_score=torch.as_tensor(cash_score.to_numpy(dtype=np.float32)[None, :], dtype=torch.float32),
        deploy_value=torch.as_tensor(deploy_value.to_numpy(dtype=np.float32)[None, :], dtype=torch.float32),
        release_value=torch.as_tensor(release_value.to_numpy(dtype=np.float32)[None, :], dtype=torch.float32),
        defense_value=torch.as_tensor(defense_value.to_numpy(dtype=np.float32)[None, :], dtype=torch.float32),
        cash_timing_value=torch.as_tensor(defense_value.to_numpy(dtype=np.float32)[None, :], dtype=torch.float32),
        receiver_source_spread_value=torch.as_tensor(receiver_spread.to_numpy(dtype=np.float32)[None, :], dtype=torch.float32),
        source_opportunity_cost=torch.as_tensor(components["source_opportunity_cost"].to_numpy(dtype=np.float32)[None, :], dtype=torch.float32),
        reversal_risk_penalty=torch.as_tensor(components["reversal_risk_penalty"].to_numpy(dtype=np.float32)[None, :], dtype=torch.float32),
        source_wrong_side_sell_penalty=torch.as_tensor(source_penalty.to_numpy(dtype=np.float32)[None, :], dtype=torch.float32),
        turnover_budget=torch.as_tensor([float(np.clip(turnover_budget, 0.04, 0.08))], dtype=torch.float32),
        risk_budget=torch.as_tensor([float(defense_value.mean()) if len(defense_value) else 0.0], dtype=torch.float32),
        sample_mask=torch.ones((1, len(policy)), dtype=torch.bool),
    )
    target_weight = pd.Series(oracle["target_weight"][0].detach().cpu().numpy().astype(float), index=policy.index)
    target_delta = (target_weight - current).clip(-0.18, 0.18)
    target_delta = target_delta.where(target_delta.abs() >= DECISION_CORE_V6_DEADBAND, 0.0)
    target_weight = (current + target_delta).clip(0.0, DECISION_CORE_V6_POSITION_CAP)
    source_supply = pd.Series(oracle["source_supply"][0].detach().cpu().numpy().astype(float), index=policy.index)
    receiver_demand = pd.Series(oracle["receiver_demand"][0].detach().cpu().numpy().astype(float), index=policy.index)
    cash_reserve = float(oracle["cash_buffer"][0].detach().cpu())
    frame = policy.copy()
    frame["decision_core_version"] = DECISION_CORE_V6_VERSION
    frame["source_supply"] = source_supply.astype(float)
    frame["receiver_demand"] = receiver_demand.astype(float)
    frame["cash_reserve"] = cash_reserve
    frame["target_weight"] = target_weight.astype(float)
    frame["target_delta"] = target_delta.astype(float)
    frame["intent"] = "hold"
    frame.loc[(current <= DECISION_CORE_V6_DEADBAND) & (target_delta > DECISION_CORE_V6_DEADBAND), "intent"] = "open"
    frame.loc[(current > DECISION_CORE_V6_DEADBAND) & (target_delta > DECISION_CORE_V6_DEADBAND), "intent"] = "add"
    frame.loc[
        (current > DECISION_CORE_V6_DEADBAND) & (target_delta < -DECISION_CORE_V6_DEADBAND) & (target_weight > DECISION_CORE_V6_DEADBAND),
        "intent",
    ] = "reduce"
    frame.loc[
        (current > DECISION_CORE_V6_DEADBAND) & (target_delta < -DECISION_CORE_V6_DEADBAND) & (target_weight <= DECISION_CORE_V6_DEADBAND),
        "intent",
    ] = "exit"
    frame["reason_code"] = "neutral"
    frame.loc[frame["intent"].isin(["open", "add"]), "reason_code"] = "receiver_demand"
    frame.loc[frame["intent"].isin(["reduce", "exit"]), "reason_code"] = "source_supply"
    frame.loc[defense_value >= pd.concat([deploy_value, release_value], axis=1).max(axis=1), "reason_code"] = "cash_reserve"
    frame["contract_status"] = frame["decision_core_v6_feature_contract_status"].astype(str)
    frame["contract_blocker_reason"] = frame["decision_core_v6_feature_contract_blocker_reason"].astype(str)
    frame["decision_core_v6_source_supply"] = source_supply.astype(float)
    frame["decision_core_v6_receiver_demand"] = receiver_demand.astype(float)
    frame["decision_core_v6_cash_reserve"] = cash_reserve
    frame["decision_core_v6_target_weight"] = target_weight.astype(float)
    frame["decision_core_v6_target_delta"] = target_delta.astype(float)
    frame["decision_core_v6_intent"] = frame["intent"].astype(str)
    frame["decision_core_v6_reason_code"] = frame["reason_code"].astype(str)
    frame["decision_core_v6_contract_status"] = frame["contract_status"].astype(str)
    frame["decision_core_v6_contract_blocker_reason"] = frame["contract_blocker_reason"].astype(str)
    frame["portfolio_daily_target_weight_intent"] = target_weight.astype(float)
    frame["portfolio_daily_target_delta_intent"] = target_delta.astype(float)
    frame["target_delta_hint"] = target_delta.astype(float)
    frame["portfolio_set_v5_source_supply"] = source_supply.astype(float)
    frame["portfolio_set_v5_receiver_demand"] = receiver_demand.astype(float)
    frame["portfolio_set_v5_source_supply_score"] = source_score.astype(float)
    frame["portfolio_set_v5_receiver_demand_score"] = receiver_score.astype(float)
    frame["portfolio_set_v5_cash_buffer_score"] = cash_reserve
    frame["decision_core_v6_source_weakness"] = components["source_weakness"].astype(float)
    frame["decision_core_v6_source_keep_strength"] = components["source_keep_strength"].astype(float)
    frame["decision_core_v6_receiver_quality"] = components["receiver_quality"].astype(float)
    frame["decision_core_v6_deploy_value"] = components["deploy_value"].astype(float)
    frame["decision_core_v6_release_value"] = components["release_value"].astype(float)
    frame["decision_core_v6_defense_value"] = components["defense_value"].astype(float)
    frame["decision_core_v6_cash_timing_value"] = components["cash_timing_value"].astype(float)
    frame["decision_core_v6_source_opportunity_cost"] = components["source_opportunity_cost"].astype(float)
    frame["decision_core_v6_reversal_risk_penalty"] = components["reversal_risk_penalty"].astype(float)
    frame["decision_core_v6_source_wrong_side_sell_penalty"] = source_penalty.astype(float)
    frame["decision_core_v6_receiver_source_spread_value"] = receiver_spread.astype(float)
    frame["decision_core_v6_release_utility"] = components["release_utility"].astype(float)
    frame["decision_core_v6_release_gate"] = components["release_gate"].astype(float)
    frame["decision_core_v6_source_release_evidence"] = components["source_release_evidence"].astype(float)
    frame["decision_core_v6_source_false_sell_penalty"] = components["source_false_sell_penalty"].astype(float)
    frame["portfolio_set_v5_turnover_budget"] = float(oracle["turnover_budget"][0].detach().cpu())
    frame["portfolio_set_v5_turnover_used"] = float(oracle["turnover_used"][0].detach().cpu())
    frame["portfolio_set_v5_oracle_constraint_violation"] = float(oracle["constraint_violation"][0].detach().cpu())
    frame["portfolio_set_v5_oracle_cash_floor_violation"] = float(oracle["cash_floor_violation"][0].detach().cpu())
    frame["portfolio_set_v5_oracle_turnover_violation"] = float(oracle["turnover_violation"][0].detach().cpu())
    frame["portfolio_set_v5_oracle_source_capacity_violation"] = float(oracle["source_capacity_violation"][0].detach().cpu())
    frame["portfolio_set_v5_oracle_receiver_capacity_violation"] = float(oracle["receiver_capacity_violation"][0].detach().cpu())
    frame["portfolio_set_v5_oracle_decision_value"] = float(oracle["decision_value"][0].detach().cpu())
    frame["portfolio_set_v5_oracle_feasible"] = float(float(oracle["constraint_violation"][0].detach().cpu()) <= 1.0e-6)
    frame["portfolio_daily_source_score"] = source_score.astype(float)
    frame["portfolio_daily_unified_source_score"] = source_score.astype(float)
    frame["portfolio_daily_receiver_score"] = receiver_score.astype(float)
    frame["portfolio_daily_unified_receiver_score"] = receiver_score.astype(float)
    frame["portfolio_daily_cash_score"] = cash_score.astype(float)
    frame["portfolio_daily_unified_cash_score"] = cash_score.astype(float)
    frame[PORTFOLIO_CASHFLOW_DECISION_MODE_COLUMN] = 1.0
    frame["portfolio_daily_release_first_intent"] = (
        source_supply / current.clip(lower=DECISION_CORE_V6_DEADBAND)
    ).replace([np.inf, -np.inf], np.nan).fillna(0.0).clip(0.0, 1.0)
    frame["release_first_intent_score"] = frame["portfolio_daily_release_first_intent"].astype(float)
    frame["release_first_intent_delta"] = (-source_supply).where(target_delta < -DECISION_CORE_V6_DEADBAND, 0.0).astype(float)
    frame["release_first_action_hint"] = "hold"
    frame.loc[frame["intent"].isin(["reduce", "exit"]), "release_first_action_hint"] = frame.loc[
        frame["intent"].isin(["reduce", "exit"]),
        "intent",
    ].astype(str)
    frame["release_first_block_reason"] = "none"
    frame.loc[current <= DECISION_CORE_V6_DEADBAND, "release_first_block_reason"] = "not_held"
    frame["portfolio_daily_source_release_quality"] = frame["portfolio_daily_release_first_intent"].astype(float)
    frame["portfolio_daily_source_release_capacity"] = frame["portfolio_daily_release_first_intent"].astype(float)
    frame["portfolio_daily_source_release_preference"] = frame["portfolio_daily_release_first_intent"].astype(float)
    frame["portfolio_daily_source_economic_release_score"] = frame["portfolio_daily_release_first_intent"].astype(float)
    frame["portfolio_daily_receiver_add_headroom"] = (DECISION_CORE_V6_POSITION_CAP - current).clip(lower=0.0).astype(float)
    cashflow_decision = normalize_portfolio_cashflow_decision(
        frame,
        current_weight=current,
        position_cap=DECISION_CORE_V6_POSITION_CAP,
        turnover_limit=float(oracle["turnover_budget"][0].detach().cpu()),
        fail_closed=True,
    )
    frame = cashflow_decision.frame
    frame["decision_core_v6_contract_valid"] = float(cashflow_decision.valid)
    frame["decision_core_v6_oracle_constraint_violation"] = float(oracle["constraint_violation"][0].detach().cpu())
    frame["decision_core_v6_cashflow_cash_conservation_gap"] = float(
        cashflow_decision.diagnostics.get("cashflow_decision_cash_conservation_gap", 0.0)
    )
    diagnostics = {
        "decision_core_version": DECISION_CORE_V6_VERSION,
        "decision_oracle_constraint_violation": float(oracle["constraint_violation"][0].detach().cpu()),
        "cashflow_decision_valid": float(cashflow_decision.valid),
        "feature_contract_blocker_rate": _safe_ratio(
            float((frame["decision_core_v6_feature_contract_blocker_count"] > 0.0).sum()),
            float(len(frame)),
        ),
        "feature_contract_degraded_rate": _safe_ratio(
            float((frame["decision_core_v6_feature_contract_degraded_count"] > 0.0).sum()),
            float(len(frame)),
        ),
    }
    return DecisionFrameV6(frame=frame, diagnostics=diagnostics)


def _make_v6_model(artifact: TorchDecisionCoreV6Artifact) -> PortfolioSetPolicyNetV5:
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


def _predict_v6_outputs(
    artifact: TorchDecisionCoreV6Artifact,
    state_frame: pd.DataFrame,
    daily_features: dict[str, float],
) -> pd.DataFrame:
    frame = _ensure_features(state_frame.copy(), artifact.feature_names)
    static_frame = _ensure_features(frame, artifact.static_feature_names)
    static_x = _apply_matrix(
        static_frame,
        artifact.static_feature_names,
        artifact.feature_fill_values,
        artifact.feature_means,
        artifact.feature_stds,
    )
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
    model = _make_v6_model(artifact).eval()
    with torch.no_grad():
        raw = model(
            torch.as_tensor(static_x[None, :, :], dtype=torch.float32),
            torch.as_tensor(sequence_x[None, :, :, :], dtype=torch.float32),
            torch.as_tensor(daily_x[None, :], dtype=torch.float32),
            torch.ones((1, len(frame)), dtype=torch.bool),
        )["raw"][0]
        decoded = _decode_raw(raw)
    return pd.DataFrame({name: tensor.detach().cpu().numpy().astype(float) for name, tensor in decoded.items()}, index=state_frame.index)


def predict_policy_decision_core_v6(
    artifact: TorchDecisionCoreV6Artifact,
    *,
    state_frame: pd.DataFrame,
    daily_features: dict[str, float],
) -> tuple[pd.DataFrame, dict[str, float]]:
    if state_frame.empty:
        raise ValueError("state_frame is empty.")
    outputs = _predict_v6_outputs(artifact, state_frame, daily_features)
    decision = build_decision_frame_v6(
        state_frame=state_frame,
        outputs=outputs,
        turnover_budget=_predict_cashflow_turnover_budget(artifact),  # type: ignore[arg-type]
    )
    global_targets = dict(artifact.global_target_defaults or {})
    global_targets.update(
        {
            "decision_core_version": DECISION_CORE_V6_VERSION,
            "decision_core_v6_mode": 1.0,
            "release_first_allocation_v3_mode": 1.0,
            "allocation_intent_v2_mode": 1.0,
            PORTFOLIO_CASHFLOW_DECISION_MODE_COLUMN: 1.0,
            PORTFOLIO_DECISION_FEATURE_BUNDLE_MODE_COLUMN: 1.0,
            "feature_contract_blocker_rate": decision.diagnostics["feature_contract_blocker_rate"],
            "feature_contract_degraded_rate": decision.diagnostics["feature_contract_degraded_rate"],
        }
    )
    return decision.frame, global_targets


def fit_policy_models_decision_core_v6(
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
    epochs: int = 32,
    min_epochs: int = 32,
    batch_size: int = 1,
    learning_rate: float = 5.0e-5,
    model_dim: int = 192,
    temporal_layers: int = 2,
    cross_layers: int = 2,
    latent_count: int = 64,
    dropout: float = 0.18,
    early_stop_patience: int = 10,
    resume_mode: str = "strict",
    decision_profile: str = DECISION_CORE_V6_PROFILE,
    progress_sink: Any | None = None,
) -> TorchDecisionCoreV6Artifact:
    if sample_frame.empty or daily_frame.empty:
        raise ValueError("formal_torch_decision_core_v6 received empty training data.")
    decision_core_v6_profile_name(decision_profile)
    contract = dict(training_contract or {})
    if str(contract.get("trainer_backend", "") or "") != TRAINER_BACKEND_FORMAL_DECISION_CORE_V6:
        raise ValueError("fit_policy_models_decision_core_v6 requires the formal_torch_decision_core_v6 training contract.")
    if not torch.cuda.is_available() and bool(contract.get("gpu_required", False)):
        raise RuntimeError("continuous_policy formal_torch_decision_core_v6 requires CUDA, but torch.cuda.is_available() is False.")
    device = torch.device("cuda" if bool(contract.get("gpu_required", False)) else "cpu")
    runtime = configure_torch_training_acceleration(device, cvxpy_layers_enabled=False)
    torch.manual_seed(int(random_seed))
    np.random.seed(int(random_seed))
    run_root.mkdir(parents=True, exist_ok=True)
    train_frame, day_diagnostics = _select_train_days(sample_frame, int(random_seed))
    targets = build_decision_core_v6_targets(train_frame)
    target_diagnostics = _decision_core_v6_target_diagnostics(targets)
    static_feature_names, sequence_bases, sequence_columns = _resolve_static_and_sequence_columns(feature_names)
    _, static_fill, static_means, static_stds = _prepare_matrix(_ensure_features(train_frame, static_feature_names), static_feature_names)
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
    best_loss = float("inf")
    best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
    best_epoch = 0
    completed_epochs = 0
    progress_event_count = 0
    started = pd.Timestamp.now()
    for epoch in range(1, max(int(epochs or 0), 1) + 1):
        model.train()
        train_loss_sum = 0.0
        train_count = 0
        for batch in loader:
            batch = move_to_device(batch, device, non_blocking=runtime.non_blocking_transfer)
            optimizer.zero_grad(set_to_none=True)
            with autocast_context(runtime):
                raw = model(batch["static_x"], batch["sequence_x"], batch["daily_x"], batch["sample_mask"])["raw"]
                loss = _decision_core_v6_loss(raw, batch)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            train_loss_sum += float(loss.detach().cpu())
            train_count += 1
        model.eval()
        val_losses: list[float] = []
        with torch.no_grad():
            for batch in val_loader:
                batch = move_to_device(batch, device, non_blocking=runtime.non_blocking_transfer)
                raw = model(batch["static_x"], batch["sequence_x"], batch["daily_x"], batch["sample_mask"])["raw"]
                val_losses.append(float(_decision_core_v6_loss(raw, batch).detach().cpu()))
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
                decision_core_version=DECISION_CORE_V6_VERSION,
            )
            progress_event_count += 1
        if epoch >= int(min_epochs or 0) and epoch - best_epoch >= int(early_stop_patience or 0):
            break
    model.load_state_dict(best_state, strict=True)
    diagnostics = {
        "trainer_backend": TRAINER_BACKEND_FORMAL_DECISION_CORE_V6,
        "loss_profile": DECISION_CORE_V6_PROFILE,
        "decision_profile": DECISION_CORE_V6_PROFILE,
        "status": "decision_core_v6_complete",
        "decision_core_version": DECISION_CORE_V6_VERSION,
        "artifact_schema": DECISION_CORE_V6_ARTIFACT_TYPE,
        "shadow_only": True,
        "training_dataset_id": str((train_summary or {}).get("training_dataset_cache", {}).get("dataset_id", "")),
        "decision_oracle_metrics": {
            "position_cap": DECISION_CORE_V6_POSITION_CAP,
            "long_only": True,
            "source_receiver_cash_conservation": True,
            "decision_oracle_constraint_violation_mean": target_diagnostics["decision_target_constraint_violation_mean"],
        },
        "paper_reproduction_metrics": {
            "method": "decision_frame_v6_target_imitation",
            "oracle": "unified_long_only_cashflow_projection",
        },
        "release_flow_metrics": {
            "target_source_count": target_diagnostics["decision_target_source_count"],
            "target_receiver_count": target_diagnostics["decision_target_receiver_count"],
            "target_intent_translation_conflict_count": target_diagnostics["decision_target_intent_translation_conflict_count"],
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
        "validation_day_count": int(len(val_subset)),
        "decision_core_v6_standard_columns": list(DECISION_CORE_V6_STANDARD_COLUMNS),
        "supports_portfolio_cashflow_decision_v1_mode": True,
        "supports_decision_core_v6": True,
        "progress_event_count": int(progress_event_count),
        "train_seconds": max(0.0, (pd.Timestamp.now() - started).total_seconds()),
        **target_diagnostics,
        **day_diagnostics,
    }
    if progress_sink is not None:
        progress_sink.emit("training_complete", completed_epochs=completed_epochs, best_epoch=best_epoch, decision_core_version=DECISION_CORE_V6_VERSION)
    artifact = TorchDecisionCoreV6Artifact(
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
            "decision_core_version": DECISION_CORE_V6_VERSION,
            "decision_core_profile": DECISION_CORE_V6_PROFILE,
        },
        model_state_dict={key: value.detach().cpu() for key, value in model.state_dict().items()},
        global_target_defaults=_global_defaults(daily_frame),
    )
    artifact.save(run_root / DECISION_CORE_V6_ARTIFACT_FILENAME)
    return artifact


def _decision_core_v6_loss(raw: torch.Tensor, batch: dict[str, torch.Tensor]) -> torch.Tensor:
    pred = _decode_raw(raw)
    target = batch["target_y"]
    decision = batch.get("decision_target_y")
    mask = batch["sample_mask"].float()
    current = batch["current_weight"].clamp(0.0, 1.0)
    target_weight = target[..., 0].clamp(0.0, DECISION_CORE_V6_POSITION_CAP)
    target_delta = target[..., 1].clamp(-0.18, 0.18)
    cash_target = target[..., 4].clamp(0.0, 1.0)
    zero_target = torch.zeros_like(current)
    if decision is not None and decision.shape[-1] >= 4:
        flow_source_target = decision[..., 0].to(device=current.device, dtype=current.dtype).clamp(0.0, 1.0)
        flow_receiver_target = decision[..., 1].to(device=current.device, dtype=current.dtype).clamp(0.0, 1.0)
        flow_cash_target = decision[..., 2].to(device=current.device, dtype=current.dtype).clamp(0.0, 1.0)
        oracle_target_weight = decision[..., 3].to(device=current.device, dtype=current.dtype).clamp(
            0.0,
            DECISION_CORE_V6_POSITION_CAP,
        )
        source_opportunity_cost = decision[..., 12].to(device=current.device, dtype=current.dtype).clamp(0.0, 1.0)
        receiver_source_spread = decision[..., 13].to(device=current.device, dtype=current.dtype).clamp(0.0, 1.0)
        reversal_risk = decision[..., 14].to(device=current.device, dtype=current.dtype).clamp(0.0, 1.0)
        wrong_side_penalty = decision[..., 15].to(device=current.device, dtype=current.dtype).clamp(0.0, 1.0)
    else:
        flow_source_target = target[..., 2].clamp(0.0, 1.0) * current
        flow_receiver_target = target[..., 3].clamp(0.0, 1.0) * (
            DECISION_CORE_V6_POSITION_CAP - current
        ).clamp_min(0.0)
        flow_cash_target = cash_target
        oracle_target_weight = target_weight
        source_opportunity_cost = zero_target
        receiver_source_spread = zero_target
        reversal_risk = zero_target
        wrong_side_penalty = zero_target
    source_supply = pred["source_supply_score"] * pred["release_intent"] * (current > DECISION_CORE_V6_DEADBAND).float()
    receiver_demand = pred["receiver_demand_score"] * (current < DECISION_CORE_V6_POSITION_CAP - DECISION_CORE_V6_DEADBAND).float()
    loss = _masked_mse(pred["target_weight"], target_weight, mask)
    loss = loss + 0.75 * _masked_mse(pred["target_delta"], target_delta, mask)
    loss = loss + 2.20 * _masked_mse(source_supply, flow_source_target, mask)
    loss = loss + 2.20 * _masked_mse(receiver_demand, flow_receiver_target, mask)
    loss = loss + 0.50 * _masked_mse(pred["cash_buffer_score"], flow_cash_target, mask)
    if decision is not None and decision.shape[-1] >= 4:
        loss = loss + 0.50 * _masked_mse(pred["target_weight"], oracle_target_weight, mask)
    source_active = (source_supply > DECISION_CORE_V6_DEADBAND).float()
    receiver_active = (receiver_demand > DECISION_CORE_V6_DEADBAND).float()
    loss = loss + 0.35 * _masked_mean(source_supply * torch.maximum(source_opportunity_cost, wrong_side_penalty), mask)
    loss = loss + 0.22 * _masked_mean(source_supply * reversal_risk, mask)
    loss = loss + 0.18 * _masked_mean(receiver_demand * (1.0 - receiver_source_spread).clamp(0.0, 1.0), mask)
    loss = loss + 0.12 * _masked_mean(source_active * (1.0 - (flow_source_target > DECISION_CORE_V6_DEADBAND).float()), mask)
    loss = loss + 0.08 * _masked_mean(receiver_active * (1.0 - (flow_receiver_target > DECISION_CORE_V6_DEADBAND).float()), mask)
    coherence = ((current + pred["target_delta"]).clamp(0.0, DECISION_CORE_V6_POSITION_CAP) - pred["target_weight"]).abs()
    loss = loss + 0.20 * (coherence * mask).sum() / mask.sum().clamp_min(1.0)
    return loss


def _masked_mse(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    return (((pred - target) ** 2) * mask).sum() / mask.sum().clamp_min(1.0)


def _decision_core_v6_target_diagnostics(targets: pd.DataFrame) -> dict[str, float]:
    source = pd.to_numeric(targets.get("source_supply", pd.Series(dtype=float)), errors="coerce").fillna(0.0)
    receiver = pd.to_numeric(targets.get("receiver_demand", pd.Series(dtype=float)), errors="coerce").fillna(0.0)
    violation = pd.to_numeric(targets.get("constraint_violation", pd.Series(dtype=float)), errors="coerce").fillna(0.0)
    conflict = pd.to_numeric(targets.get("target_delta_weight_conflict", pd.Series(dtype=float)), errors="coerce").fillna(0.0)
    blocker = targets.get("contract_status", pd.Series("", index=targets.index)).astype(str).eq("blocker")
    degraded = targets.get("contract_status", pd.Series("", index=targets.index)).astype(str).eq("degraded")
    return {
        "decision_target_source_count": float((source > DECISION_CORE_V6_DEADBAND).sum()),
        "decision_target_receiver_count": float((receiver > DECISION_CORE_V6_DEADBAND).sum()),
        "decision_target_constraint_violation_mean": float(violation.mean()) if len(violation) else 0.0,
        "decision_target_intent_translation_conflict_count": float((conflict > 0.0).sum()),
        "feature_contract_blocker_rate": float(blocker.mean()) if len(blocker) else 0.0,
        "feature_contract_degraded_rate": float(degraded.mean()) if len(degraded) else 0.0,
        "feature_contract_blocker_count": float(blocker.sum()),
        "feature_contract_degraded_count": float(degraded.sum()),
    }
