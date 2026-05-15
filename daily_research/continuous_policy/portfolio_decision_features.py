from __future__ import annotations

from typing import Any, Iterable

import numpy as np
import pandas as pd


PORTFOLIO_DECISION_FEATURE_BUNDLE_VERSION = "portfolio_decision_feature_bundle_v1"
PORTFOLIO_DECISION_FEATURE_BUNDLE_MODE_COLUMN = "portfolio_decision_feature_bundle_v1_mode"
PORTFOLIO_DECISION_FEATURE_CONTRACT_MISSING_COUNT_COLUMN = (
    "portfolio_decision_feature_contract_missing_count"
)
PORTFOLIO_DECISION_FEATURE_CONTRACT_NEUTRAL_DEFAULT_COUNT_COLUMN = (
    "portfolio_decision_feature_contract_neutral_default_count"
)
PORTFOLIO_DECISION_FEATURE_CONTRACT_DEGRADED_COUNT_COLUMN = (
    "portfolio_decision_feature_contract_degraded_count"
)
PORTFOLIO_DECISION_FEATURE_CONTRACT_BLOCKER_COUNT_COLUMN = (
    "portfolio_decision_feature_contract_blocker_count"
)
PORTFOLIO_DECISION_FEATURE_CONTRACT_SEVERITY_COLUMN = "portfolio_decision_feature_contract_severity"
PORTFOLIO_DECISION_FEATURE_CONTRACT_BLOCKER_COLUMN = "portfolio_decision_feature_contract_blocker"
PORTFOLIO_DECISION_FEATURE_CONTRACT_DEGRADED_REASON_COLUMN = (
    "portfolio_decision_feature_contract_degraded_reason"
)


R74_HIGH_VALUE_LAKE_FEATURES: tuple[str, ...] = (
    "adv20",
    "amount",
    "volume",
    "z_drawdown_20",
    "z_volatility_20",
    "z_vol_ratio_5_20",
    "z_price_volume_divergence",
    "z_breakout_volume",
    "z_volatility_contraction",
    "z_volume_contraction",
)
R74_CRITICAL_SCORE_FEATURES: tuple[str, ...] = ("score_blend", "score_rank_pct", "score_v2", "score_none")
R74_CRITICAL_MEMBERSHIP_FEATURES: tuple[str, ...] = ("in_pool",)
R74_CRITICAL_PRICE_FEATURES: tuple[str, ...] = ("close", "open", "high", "low")
R74_CORE_BEHAVIOR_FEATURES: tuple[str, ...] = ("ret_1d", "ret_3d", "ret_5d", "vol_20d")


DECISION_ORACLE_INPUT_COLUMNS: tuple[str, ...] = (
    "alpha_opportunity_value",
    "deploy_value_target",
    "result_value_deploy_value_target",
    "result_value_alpha_opportunity_value",
    "portfolio_daily_receiver_score",
    "portfolio_daily_unified_receiver_score",
    "portfolio_daily_receiver_source_spread_reward",
    "portfolio_daily_source_forward_spread_score",
    "release_value_target",
    "sell_release_value",
    "portfolio_daily_source_release_quality",
    "portfolio_daily_source_economic_release_score",
    "portfolio_daily_source_release_preference",
    "cash_defense_value",
    "defense_value_target",
    "budget_cash_timing_signal_target",
    "portfolio_daily_cash_score",
    "portfolio_daily_unified_cash_score",
    "cash_regime_pressure",
    "market_downside_pressure",
    "portfolio_daily_source_opportunity_cost",
    "portfolio_daily_source_opportunity_cost_penalty",
    "hold_continuation_value",
    "large_upside_1d_target",
    "portfolio_daily_source_forward_proxy_keep_risk",
    "portfolio_daily_source_economic_block_risk",
    "portfolio_daily_source_positive_forward_penalty",
    "portfolio_daily_source_bad_forward_spread_risk",
    "portfolio_daily_source_forward_strength_brake_risk",
    "portfolio_daily_receiver_forward_excess_5d",
    "portfolio_daily_source_forward_excess_5d",
    "forward_excess_1d",
    "forward_excess_3d",
    "forward_excess_5d",
    "portfolio_daily_crowding_penalty",
)


STATE_BUILDER_FEATURE_INPUTS: tuple[str, ...] = (
    "score_none",
    "score_v2",
    "score_blend",
    "z_score_none",
    "z_score_v2",
    "adv20_rank",
    "price_rank",
    "ma20_gap",
    "ma60_gap",
    "volume_rank",
    "ret_1d",
    "ret_3d",
    "ret_5d",
    "ret_10d",
    "ret_20d",
    "vol_5d",
    "vol_20d",
    "score_delta_1d",
    "score_delta_5d",
    "score_delta_accel",
    "ret_accel_5_20",
    "volume_ratio_5_20",
    "distance_to_20d_high",
    "distance_to_60d_high",
    "distance_to_20d_low",
    "volatility_expansion",
    "adv_ratio_5_20",
    "alpha_prior_score_z",
    "alpha_prior_rank_pct",
    "alpha_prior_target_weight",
    "alpha_prior_selected",
    "alpha_prior_score_delta_1d",
    "alpha_prior_score_delta_5d",
    "alpha_prior_weight_delta_1d",
    "alpha_prior_coverage",
)


LABEL_BUILDER_DECISION_COLUMNS: tuple[str, ...] = (
    "alpha_opportunity_value",
    "hold_continuation_value",
    "sell_release_value",
    "cash_defense_value",
    "deploy_value_target",
    "release_value_target",
    "defense_value_target",
    "deploy_gate_target",
    "release_gate_target",
    "defense_gate_target",
    "portfolio_daily_receiver_score",
    "portfolio_daily_source_score",
    "portfolio_daily_cash_score",
    "portfolio_daily_receiver_source_spread_reward",
    "portfolio_daily_source_forward_spread_score",
    "portfolio_daily_source_forward_strength_brake_risk",
    "portfolio_daily_source_forward_proxy_keep_risk",
)


HIGH_VALUE_FEATURE_HINTS: tuple[str, ...] = (
    "score",
    "alpha",
    "ret",
    "return",
    "vol",
    "adv",
    "volume",
    "amount",
    "distance",
    "momentum",
    "liquidity",
    "rank",
    "drawdown",
)


def _numeric_series(frame: pd.DataFrame, name: str, default: float = 0.0) -> pd.Series:
    if name not in frame.columns:
        return pd.Series(float(default), index=frame.index, dtype=float)
    return pd.to_numeric(frame[name], errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(float(default))


def _clip01(value: pd.Series | np.ndarray | float) -> pd.Series:
    if isinstance(value, pd.Series):
        return value.replace([np.inf, -np.inf], np.nan).fillna(0.0).clip(0.0, 1.0)
    return pd.Series(value).replace([np.inf, -np.inf], np.nan).fillna(0.0).clip(0.0, 1.0)


def _rank_pct(values: pd.Series) -> pd.Series:
    clean = pd.to_numeric(values, errors="coerce").replace([np.inf, -np.inf], np.nan)
    if clean.notna().sum() <= 1:
        return pd.Series(0.0, index=values.index, dtype=float)
    return clean.rank(pct=True, method="average").fillna(0.0).clip(0.0, 1.0)


def _rank_or_scaled_feature(frame: pd.DataFrame, name: str, default: float = 0.5) -> pd.Series:
    if name not in frame.columns:
        return pd.Series(float(default), index=frame.index, dtype=float)
    values = pd.to_numeric(frame[name], errors="coerce").replace([np.inf, -np.inf], np.nan)
    if values.notna().sum() <= 1:
        scaled = values.fillna(float(default))
        if float(scaled.abs().max() if len(scaled) else 0.0) > 1.0:
            return pd.Series(float(default), index=frame.index, dtype=float)
        return scaled.fillna(float(default)).clip(0.0, 1.0)
    return _rank_pct(values).fillna(float(default)).clip(0.0, 1.0)


def _z_positive(frame: pd.DataFrame, name: str, scale: float = 3.0) -> pd.Series:
    return (_numeric_series(frame, name, 0.0).clip(lower=0.0, upper=float(scale)) / float(scale)).clip(0.0, 1.0)


def _z_abs(frame: pd.DataFrame, name: str, scale: float = 3.0) -> pd.Series:
    return (_numeric_series(frame, name, 0.0).abs().clip(upper=float(scale)) / float(scale)).clip(0.0, 1.0)


def _max_columns(frame: pd.DataFrame, names: Iterable[str], default: float = 0.0) -> pd.Series:
    values = [_numeric_series(frame, name, default).rename(name) for name in names if name in frame.columns]
    if not values:
        return pd.Series(float(default), index=frame.index, dtype=float)
    return pd.concat(values, axis=1).max(axis=1).fillna(float(default))


def _assign_decision_column(result: pd.DataFrame, name: str, values: pd.Series) -> None:
    clean = pd.to_numeric(values.reindex(result.index), errors="coerce").replace([np.inf, -np.inf], np.nan)
    if name in result.columns:
        existing = pd.to_numeric(result[name], errors="coerce").replace([np.inf, -np.inf], np.nan)
        result[name] = existing.combine_first(clean).fillna(0.0).astype(float)
    else:
        result[name] = clean.fillna(0.0).astype(float)


def attach_portfolio_decision_features(
    frame: pd.DataFrame,
    *,
    mode: str = "predict",
    neutral_default: float = 0.0,
) -> pd.DataFrame:
    """Attach non-leaking decision features consumed by portfolio-set v5 r69/r71 oracle inputs."""
    if frame.empty:
        out = frame.copy()
        out[PORTFOLIO_DECISION_FEATURE_BUNDLE_MODE_COLUMN] = 1.0
        out[PORTFOLIO_DECISION_FEATURE_CONTRACT_MISSING_COUNT_COLUMN] = 0.0
        out[PORTFOLIO_DECISION_FEATURE_CONTRACT_NEUTRAL_DEFAULT_COUNT_COLUMN] = 0.0
        out[PORTFOLIO_DECISION_FEATURE_CONTRACT_DEGRADED_COUNT_COLUMN] = 0.0
        out[PORTFOLIO_DECISION_FEATURE_CONTRACT_BLOCKER_COUNT_COLUMN] = 1.0
        out[PORTFOLIO_DECISION_FEATURE_CONTRACT_SEVERITY_COLUMN] = "blocker"
        out[PORTFOLIO_DECISION_FEATURE_CONTRACT_BLOCKER_COLUMN] = "empty_frame"
        out[PORTFOLIO_DECISION_FEATURE_CONTRACT_DEGRADED_REASON_COLUMN] = "empty_frame"
        return out

    result = frame.copy()
    has_score_signal = any(name in result.columns for name in R74_CRITICAL_SCORE_FEATURES)
    has_membership_signal = any(name in result.columns for name in R74_CRITICAL_MEMBERSHIP_FEATURES)
    missing_price_inputs = [name for name in R74_CRITICAL_PRICE_FEATURES if name not in result.columns]
    missing_core_inputs = [name for name in R74_CORE_BEHAVIOR_FEATURES if name not in result.columns]
    missing_high_value_inputs = [name for name in R74_HIGH_VALUE_LAKE_FEATURES if name not in result.columns]
    missing_inputs = [
        name
        for name in ("score_blend", "score_rank_pct", "ret_3d", "ret_5d", "vol_20d", "in_pool")
        if name not in result.columns
    ]
    neutral_default_count = len(missing_inputs)
    blocker_reasons: list[str] = []
    if not has_score_signal:
        blocker_reasons.append("missing_score_signal")
    if not has_membership_signal:
        blocker_reasons.append("missing_membership_signal")
    degraded_reasons: list[str] = []
    if missing_price_inputs:
        degraded_reasons.append("missing_price_ohlc")
    if "volume" not in result.columns and "volume_rank" not in result.columns:
        degraded_reasons.append("missing_volume_signal")
    if "amount" not in result.columns and "adv20" not in result.columns and "adv20_rank" not in result.columns:
        degraded_reasons.append("missing_liquidity_signal")
    if missing_core_inputs:
        degraded_reasons.append("missing_core_return_volatility")
    missing_high_value_degraded = [
        name
        for name in missing_high_value_inputs
        if name in {"z_drawdown_20", "z_volatility_20", "z_vol_ratio_5_20", "z_price_volume_divergence", "z_breakout_volume"}
    ]
    if missing_high_value_degraded:
        degraded_reasons.append("missing_r74_high_value_features")

    current = _numeric_series(result, "current_weight", 0.0).clip(0.0, 1.0)
    held = ((_numeric_series(result, "holding_flag", 0.0) > 0.5) | (current > 0.003)).astype(float)
    in_pool = (_numeric_series(result, "in_pool", 1.0) > 0.5).astype(float)
    headroom = (0.24 - current).clip(lower=0.0)

    score_blend = _numeric_series(result, "score_blend", float(neutral_default))
    score_rank = (
        _numeric_series(result, "score_rank_pct", 0.0).clip(0.0, 1.0)
        if "score_rank_pct" in result.columns
        else _rank_pct(score_blend)
    )
    score_z = (0.5 + score_blend.clip(-3.0, 3.0) / 6.0).clip(0.0, 1.0)
    alpha_prior = _max_columns(
        result,
        ("alpha_prior_rank_pct", "alpha_prior_selected", "alpha_prior_coverage"),
        default=0.0,
    ).clip(0.0, 1.0)
    if "alpha_prior_score_z" in result.columns:
        alpha_prior = pd.concat(
            [
                alpha_prior.rename("alpha_prior"),
                (0.5 + _numeric_series(result, "alpha_prior_score_z", 0.0).clip(-3.0, 3.0) / 6.0).rename("alpha_prior_score_z"),
            ],
            axis=1,
        ).max(axis=1).clip(0.0, 1.0)

    ret_1d = _numeric_series(result, "ret_1d", 0.0)
    ret_3d = _numeric_series(result, "ret_3d", 0.0)
    ret_5d = _numeric_series(result, "ret_5d", 0.0)
    ret_accel = _numeric_series(result, "ret_accel_5_20", 0.0)
    score_delta_1d = _numeric_series(result, "score_delta_1d", 0.0)
    score_delta_5d = _numeric_series(result, "score_delta_5d", 0.0)
    momentum_value = (
        0.24 * (ret_1d / 0.035).clip(-1.0, 1.0)
        + 0.32 * (ret_3d / 0.055).clip(-1.0, 1.0)
        + 0.28 * (ret_5d / 0.075).clip(-1.0, 1.0)
        + 0.10 * (ret_accel / 0.05).clip(-1.0, 1.0)
        + 0.06 * (score_delta_1d + score_delta_5d / 5.0).clip(-1.0, 1.0)
    )
    momentum_positive = ((momentum_value + 1.0) / 2.0).clip(0.0, 1.0)
    momentum_negative = ((-momentum_value).clip(0.0, 1.0)).clip(0.0, 1.0)

    liquidity_value = _max_columns(
        result,
        ("adv20_rank", "volume_rank", "adv_ratio_5_20", "volume_ratio_5_20"),
        default=0.5,
    )
    liquidity_value = (0.5 + (liquidity_value - 0.5).clip(-1.0, 1.0)).clip(0.0, 1.0)
    r74_adv20_liquidity = _rank_or_scaled_feature(result, "adv20", default=0.5)
    r74_amount_liquidity = _rank_or_scaled_feature(result, "amount", default=0.5)
    r74_volume_liquidity = _rank_or_scaled_feature(result, "volume", default=0.5)
    r74_liquidity_quality = pd.concat(
        [
            liquidity_value.rename("legacy_liquidity"),
            r74_adv20_liquidity.rename("adv20"),
            r74_amount_liquidity.rename("amount"),
            r74_volume_liquidity.rename("volume"),
        ],
        axis=1,
    ).max(axis=1).fillna(0.5).clip(0.0, 1.0)
    r74_volume_contraction = _z_positive(result, "z_volume_contraction")
    r74_volatility_contraction = _z_positive(result, "z_volatility_contraction")
    r74_drawdown_pressure = pd.concat(
        [
            _z_positive(result, "z_drawdown_20").rename("z_drawdown_20"),
            (-_numeric_series(result, "portfolio_drawdown_20d", 0.0) / 0.10).clip(0.0, 1.0).rename("portfolio_drawdown_20d"),
        ],
        axis=1,
    ).max(axis=1).fillna(0.0).clip(0.0, 1.0)
    r74_volatility_pressure = pd.concat(
        [
            _z_positive(result, "z_volatility_20").rename("z_volatility_20"),
            _z_positive(result, "z_vol_ratio_5_20").rename("z_vol_ratio_5_20"),
            _numeric_series(result, "volatility_expansion", 0.0).clip(0.0, 1.0).rename("volatility_expansion"),
        ],
        axis=1,
    ).max(axis=1).fillna(0.0).clip(0.0, 1.0)
    r74_price_volume_divergence = _z_abs(result, "z_price_volume_divergence")
    r74_breakout_volume = _z_positive(result, "z_breakout_volume")
    vol_20d = _numeric_series(result, "vol_20d", 0.0).abs()
    vol_pressure = pd.concat(
        [
            (vol_20d / 0.08).rename("vol_20d"),
            _numeric_series(result, "volatility_expansion", 0.0).clip(0.0, 1.0).rename("volatility_expansion"),
        ],
        axis=1,
    ).max(axis=1).clip(0.0, 1.0)
    distance_to_high = _numeric_series(result, "distance_to_20d_high", 0.0)
    distance_to_low = _numeric_series(result, "distance_to_20d_low", 0.0)
    breakout_room = ((distance_to_high + 0.12) / 0.12).clip(0.0, 1.0)
    local_rebound = ((distance_to_low + 0.02) / 0.12).clip(0.0, 1.0)

    market_pressure = _max_columns(
        result,
        ("market_downside_pressure", "cash_regime_pressure", "portfolio_cash_pressure"),
        default=0.0,
    ).clip(0.0, 1.0)
    portfolio_drawdown = (-_numeric_series(result, "portfolio_drawdown_20d", 0.0) / 0.10).clip(0.0, 1.0)
    cash_defense = (
        0.54 * market_pressure
        + 0.20 * portfolio_drawdown
        + 0.16 * vol_pressure
        + 0.10 * _numeric_series(result, "portfolio_cash_deficit", 0.0).clip(0.0, 1.0)
    ).clip(0.0, 1.0)
    r74_cash_defense = (
        0.36 * market_pressure
        + 0.22 * r74_drawdown_pressure
        + 0.20 * r74_volatility_pressure
        + 0.12 * r74_volume_contraction
        + 0.10 * (1.0 - r74_liquidity_quality).clip(0.0, 1.0)
        - 0.08 * r74_breakout_volume
    ).clip(0.0, 1.0)

    reentry_pressure = _numeric_series(result, "reentry_cooldown", 0.0).clip(0.0, 1.0)
    alpha_opportunity = (
        0.42 * score_rank
        + 0.18 * score_z
        + 0.14 * alpha_prior
        + 0.14 * momentum_positive
        + 0.07 * liquidity_value
        + 0.05 * breakout_room
        - 0.10 * cash_defense
        - 0.08 * reentry_pressure
    ).where(in_pool > 0.5, 0.0).clip(0.0, 1.0)

    deploy_floor = ((score_rank - 0.62) / 0.34).clip(0.0, 1.0) * (1.0 - 0.55 * cash_defense).clip(0.0, 1.0)
    deploy_value = pd.concat(
        [
            (
                0.70 * alpha_opportunity
                + 0.16 * momentum_positive
                + 0.08 * liquidity_value
                + 0.06 * local_rebound
                - 0.24 * cash_defense
                - 0.10 * vol_pressure
            ).rename("deploy_value"),
            deploy_floor.rename("deploy_floor"),
        ],
        axis=1,
    ).max(axis=1).where((in_pool > 0.5) & (headroom > 0.003), 0.0).clip(0.0, 1.0)
    r74_deploy_value = (
        0.44 * alpha_opportunity
        + 0.18 * momentum_positive
        + 0.14 * r74_liquidity_quality
        + 0.12 * r74_breakout_volume
        + 0.06 * local_rebound
        - 0.22 * r74_cash_defense
        - 0.14 * r74_volatility_pressure
        - 0.10 * r74_price_volume_divergence
    ).where((in_pool > 0.5) & (headroom > 0.003), 0.0).clip(0.0, 1.0)

    unrealized_pnl = _numeric_series(result, "unrealized_pnl", 0.0)
    drawdown_from_peak = _numeric_series(result, "drawdown_from_peak", 0.0)
    signal_decay = _numeric_series(result, "signal_decay_speed", 0.0).clip(0.0, 1.0)
    hold_continuity = _numeric_series(result, "hold_continuity_pressure", 0.0).clip(0.0, 1.0)
    held_quality = (
        0.56 * alpha_opportunity
        + 0.16 * momentum_positive
        + 0.12 * np.maximum(unrealized_pnl, 0.0).clip(0.0, 0.12) / 0.12
        + 0.10 * hold_continuity
        + 0.06 * liquidity_value
    ).clip(0.0, 1.0)
    source_weakness = (1.0 - alpha_opportunity).clip(0.0, 1.0)
    drawdown_release = (-drawdown_from_peak / 0.10).clip(0.0, 1.0)
    release_value = (
        0.42 * source_weakness
        + 0.24 * drawdown_release
        + 0.16 * momentum_negative
        + 0.12 * signal_decay
        + 0.06 * cash_defense
    ).where(held > 0.5, 0.0).clip(0.0, 1.0)
    r74_source_quality_penalty = pd.concat(
        [
            held_quality.rename("held_quality"),
            momentum_positive.rename("momentum_positive"),
            (r74_breakout_volume * 0.85).rename("breakout_volume"),
            (1.0 - r74_liquidity_quality).clip(0.0, 1.0).mul(0.55).rename("illiquidity"),
        ],
        axis=1,
    ).max(axis=1).where(held > 0.5, 0.0).clip(0.0, 1.0)
    r74_release_value = (
        0.38 * release_value
        + 0.22 * drawdown_release
        + 0.16 * r74_cash_defense
        + 0.12 * momentum_negative
        + 0.12 * signal_decay
        - 0.24 * r74_source_quality_penalty
    ).where(held > 0.5, 0.0).clip(0.0, 1.0)
    opportunity_cost = pd.concat(
        [
            held_quality.rename("held_quality"),
            (np.maximum(unrealized_pnl, 0.0).clip(0.0, 0.12) / 0.12).rename("positive_pnl"),
            momentum_positive.rename("momentum_positive"),
        ],
        axis=1,
    ).max(axis=1).where(held > 0.5, 0.0).clip(0.0, 1.0)

    held_reference = float(held_quality.loc[held > 0.5].median()) if bool((held > 0.5).any()) else 0.0
    receiver_spread = (
        (deploy_value - held_reference + (0.18 if held_reference > 0.0 else 0.0)) / 0.55
    ).clip(0.0, 1.0)
    r74_receiver_spread = (
        (
            r74_deploy_value
            - held_reference
            + (0.16 if held_reference > 0.0 else 0.0)
            - 0.18 * r74_cash_defense
            - 0.10 * r74_price_volume_divergence
        )
        / 0.50
    ).clip(0.0, 1.0)
    if held_reference <= 0.0:
        receiver_spread = pd.concat(
            [receiver_spread.rename("spread"), deploy_value.rename("cash_funded_deploy")],
            axis=1,
        ).max(axis=1).clip(0.0, 1.0)
    source_spread = ((held_reference - held_quality + 0.08) / 0.45).where(held > 0.5, 0.0).clip(0.0, 1.0)
    source_positive_forward_penalty = pd.concat(
        [
            opportunity_cost.rename("opportunity_cost"),
            momentum_positive.rename("momentum_positive"),
            alpha_opportunity.rename("alpha_opportunity"),
        ],
        axis=1,
    ).max(axis=1).where(held > 0.5, 0.0).clip(0.0, 1.0)
    source_block_risk = (opportunity_cost - release_value + 0.15).clip(0.0, 1.0).where(held > 0.5, 0.0)
    source_bad_spread_risk = pd.concat(
        [
            source_block_risk.rename("source_block_risk"),
            (held_quality - release_value + 0.10).clip(0.0, 1.0).rename("held_quality_over_release"),
        ],
        axis=1,
    ).max(axis=1).where(held > 0.5, 0.0).clip(0.0, 1.0)
    source_score = (release_value * (1.0 - 0.45 * source_block_risk) + source_spread * 0.25).where(
        held > 0.5,
        0.0,
    ).clip(0.0, 1.0)
    receiver_score = (
        0.68 * deploy_value
        + 0.22 * receiver_spread
        + 0.10 * alpha_opportunity
        - 0.14 * cash_defense
    ).where((in_pool > 0.5) & (headroom > 0.003), 0.0).clip(0.0, 1.0)
    cash_score = cash_defense.clip(0.0, 1.0)
    expected_forward = (
        (alpha_opportunity - 0.50) * 0.075
        + (momentum_positive - 0.50) * 0.035
        - cash_defense * 0.020
    ).clip(-0.12, 0.12)
    source_forward_proxy = expected_forward.where(held > 0.5, 0.0)
    receiver_forward_proxy = expected_forward.where((in_pool > 0.5) & (headroom > 0.003), 0.0)
    crowding_penalty = (vol_pressure * 0.20 + (1.0 - liquidity_value).clip(0.0, 1.0) * 0.18).clip(0.0, 1.0)
    r74_crowding_penalty = (
        0.26 * r74_volatility_pressure
        + 0.22 * r74_price_volume_divergence
        + 0.20 * (1.0 - r74_liquidity_quality).clip(0.0, 1.0)
        + 0.12 * r74_volume_contraction
        - 0.08 * r74_volatility_contraction
    ).clip(0.0, 1.0)

    assignments = {
        "alpha_opportunity_value": alpha_opportunity,
        "result_value_alpha_opportunity_value": alpha_opportunity,
        "deploy_value_target": deploy_value,
        "result_value_deploy_value_target": deploy_value,
        "deploy_gate_target": deploy_value,
        "deploy_executability_target": (deploy_value * liquidity_value).clip(0.0, 1.0),
        "cash_defense_value": cash_defense,
        "defense_value_target": cash_defense,
        "defense_gate_target": cash_defense,
        "budget_cash_timing_signal_target": cash_score,
        "release_value_target": release_value,
        "sell_release_value": release_value,
        "release_gate_target": release_value,
        "hold_continuation_value": held_quality.where(held > 0.5, 0.0).clip(0.0, 1.0),
        "large_upside_1d_target": (alpha_opportunity * momentum_positive).clip(0.0, 1.0),
        "portfolio_daily_receiver_score": receiver_score,
        "portfolio_daily_unified_receiver_score": receiver_score,
        "portfolio_daily_receiver_add_headroom": headroom,
        "portfolio_daily_receiver_add_capacity": headroom,
        "portfolio_daily_receiver_executability": (receiver_score * liquidity_value).clip(0.0, 1.0),
        "portfolio_daily_receiver_source_spread_reward": receiver_spread,
        "portfolio_daily_source_score": source_score,
        "portfolio_daily_unified_source_score": source_score,
        "portfolio_daily_source_release_quality": release_value,
        "portfolio_daily_source_release_capacity": current.where(held > 0.5, 0.0).clip(0.0, 1.0),
        "portfolio_daily_source_release_preference": source_score,
        "portfolio_daily_source_economic_release_score": source_score,
        "portfolio_daily_source_economic_block_risk": source_block_risk,
        "portfolio_daily_source_forward_spread_score": source_spread,
        "portfolio_daily_source_forward_strength_brake_risk": source_positive_forward_penalty,
        "portfolio_daily_source_forward_proxy_keep_risk": opportunity_cost,
        "portfolio_daily_source_positive_forward_penalty": source_positive_forward_penalty,
        "portfolio_daily_source_bad_forward_spread_risk": source_bad_spread_risk,
        "portfolio_daily_source_opportunity_cost": opportunity_cost,
        "portfolio_daily_source_opportunity_cost_penalty": opportunity_cost,
        "portfolio_daily_source_executability": source_score,
        "portfolio_daily_cash_score": cash_score,
        "portfolio_daily_unified_cash_score": cash_score,
        "portfolio_daily_crowding_penalty": crowding_penalty,
        "portfolio_daily_receiver_forward_excess_5d": receiver_forward_proxy,
        "portfolio_daily_source_forward_excess_5d": source_forward_proxy,
        "forward_excess_1d": (expected_forward * 0.35).clip(-0.06, 0.06),
        "forward_excess_3d": (expected_forward * 0.70).clip(-0.09, 0.09),
        "forward_excess_5d": expected_forward,
        "portfolio_decision_alpha_opportunity_value": alpha_opportunity,
        "portfolio_decision_deploy_value": deploy_value,
        "portfolio_decision_release_value": release_value,
        "portfolio_decision_defense_value": cash_defense,
        "portfolio_decision_receiver_source_spread_value": receiver_spread,
        "portfolio_decision_r74_liquidity_quality": r74_liquidity_quality,
        "portfolio_decision_r74_drawdown_pressure": r74_drawdown_pressure,
        "portfolio_decision_r74_volatility_pressure": r74_volatility_pressure,
        "portfolio_decision_r74_volume_contraction": r74_volume_contraction,
        "portfolio_decision_r74_price_volume_divergence": r74_price_volume_divergence,
        "portfolio_decision_r74_breakout_volume": r74_breakout_volume,
        "portfolio_decision_r74_cash_defense_value": r74_cash_defense,
        "portfolio_decision_r74_deploy_value": r74_deploy_value,
        "portfolio_decision_r74_release_value": r74_release_value,
        "portfolio_decision_r74_source_quality_penalty": r74_source_quality_penalty,
        "portfolio_decision_r74_receiver_source_spread_value": r74_receiver_spread,
        "portfolio_decision_r74_crowding_liquidity_penalty": r74_crowding_penalty,
    }
    for name, values in assignments.items():
        _assign_decision_column(result, name, pd.Series(values, index=result.index, dtype=float))

    blocker = "none" if not blocker_reasons else "feature_contract_blocker:" + ",".join(blocker_reasons)
    degraded_reason = "none" if not degraded_reasons else "feature_contract_degraded:" + ",".join(sorted(set(degraded_reasons)))
    blocker_count = float(len(blocker_reasons))
    degraded_count = float(len(sorted(set(degraded_reasons))))
    if blocker_count > 0.0:
        severity = "blocker"
    elif degraded_count > 0.0:
        severity = "degraded"
    elif neutral_default_count > 0:
        severity = "neutral_default"
    else:
        severity = "ok"
    result[PORTFOLIO_DECISION_FEATURE_BUNDLE_MODE_COLUMN] = 1.0
    result[PORTFOLIO_DECISION_FEATURE_CONTRACT_MISSING_COUNT_COLUMN] = float(len(missing_inputs))
    result[PORTFOLIO_DECISION_FEATURE_CONTRACT_NEUTRAL_DEFAULT_COUNT_COLUMN] = float(neutral_default_count)
    result[PORTFOLIO_DECISION_FEATURE_CONTRACT_DEGRADED_COUNT_COLUMN] = degraded_count
    result[PORTFOLIO_DECISION_FEATURE_CONTRACT_BLOCKER_COUNT_COLUMN] = blocker_count
    result[PORTFOLIO_DECISION_FEATURE_CONTRACT_SEVERITY_COLUMN] = severity
    result[PORTFOLIO_DECISION_FEATURE_CONTRACT_BLOCKER_COLUMN] = blocker
    result[PORTFOLIO_DECISION_FEATURE_CONTRACT_DEGRADED_REASON_COLUMN] = degraded_reason
    result["portfolio_decision_feature_bundle_version"] = PORTFOLIO_DECISION_FEATURE_BUNDLE_VERSION
    result["portfolio_decision_feature_bundle_mode"] = str(mode or "predict")
    return result


def _frame_feature_report(
    *,
    name: str,
    frame: pd.DataFrame | pd.Series,
    membership_frame: pd.DataFrame | None,
    sample_dates: Iterable[Any] | None,
) -> dict[str, Any]:
    if isinstance(frame, pd.Series):
        working = frame.to_frame(name)
    else:
        working = frame.copy()
    if sample_dates is not None:
        sample_date_list = list(sample_dates)
    else:
        sample_date_list = []
    if sample_date_list:
        dates = pd.to_datetime(pd.Index(sample_date_list)).normalize()
        working = working.loc[pd.to_datetime(working.index).normalize().isin(dates)]
    values = working.replace([np.inf, -np.inf], np.nan)
    numeric = values.apply(pd.to_numeric, errors="coerce")
    total_cells = int(numeric.shape[0] * numeric.shape[1])
    finite_mask = np.isfinite(numeric.to_numpy(dtype=float, copy=True)) if total_cells else np.zeros((0, 0), dtype=bool)
    finite_cells = int(finite_mask.sum()) if total_cells else 0
    coverage = float(finite_cells / total_cells) if total_cells else 0.0
    variance = numeric.var(axis=1, skipna=True).replace([np.inf, -np.inf], np.nan)
    membership_valid_ratio = 0.0
    if membership_frame is not None and isinstance(frame, pd.DataFrame) and not frame.empty:
        aligned_membership = membership_frame.reindex(index=working.index, columns=working.columns).fillna(False).astype(bool)
        membership_total = int(aligned_membership.to_numpy(dtype=bool).sum())
        if membership_total > 0:
            membership_valid_ratio = float((np.isfinite(numeric.to_numpy(dtype=float)) & aligned_membership.to_numpy(dtype=bool)).sum() / membership_total)
    return {
        "feature": str(name),
        "row_count": int(numeric.shape[0]),
        "column_count": int(numeric.shape[1]),
        "finite_cell_count": finite_cells,
        "finite_ratio": coverage,
        "membership_valid_ratio": membership_valid_ratio,
        "cross_section_variance_mean": float(variance.mean()) if len(variance.dropna()) else 0.0,
        "low_variance": bool(coverage > 0.0 and (float(variance.mean()) if len(variance.dropna()) else 0.0) <= 1.0e-12),
    }


def build_lake_feature_utilization_report(
    prepared: Any,
    *,
    used_feature_names: Iterable[str] | None = None,
    daily_feature_names: Iterable[str] | None = None,
    sample_dates: Iterable[Any] | None = None,
) -> dict[str, Any]:
    used = {str(name) for name in (used_feature_names or []) if str(name)}
    daily_used = {str(name) for name in (daily_feature_names or []) if str(name)}
    state_inputs = set(STATE_BUILDER_FEATURE_INPUTS)
    label_inputs = set(LABEL_BUILDER_DECISION_COLUMNS)
    oracle_inputs = set(DECISION_ORACLE_INPUT_COLUMNS)
    r74_inputs = set(R74_HIGH_VALUE_LAKE_FEATURES)

    panels: dict[str, pd.DataFrame | pd.Series] = {}
    for name in ("score_none", "score_v2", "score_blend", "close", "open", "high", "low", "volume", "amount"):
        value = getattr(prepared, "open_", None) if name == "open" else getattr(prepared, name, None)
        if isinstance(value, (pd.DataFrame, pd.Series)):
            panels[name] = value
    for mapping_name in ("feature_frames", "derived_frames", "market_features"):
        mapping = getattr(prepared, mapping_name, {}) or {}
        if isinstance(mapping, dict):
            for name, frame in mapping.items():
                if isinstance(frame, (pd.DataFrame, pd.Series)):
                    panels[str(name)] = frame

    membership = getattr(prepared, "membership_frame", None)
    reports: list[dict[str, Any]] = []
    for name in sorted(panels):
        report = _frame_feature_report(
            name=name,
            frame=panels[name],
            membership_frame=membership if isinstance(membership, pd.DataFrame) else None,
            sample_dates=sample_dates,
        )
        report["used_by_model_feature_matrix"] = bool(name in used)
        report["used_by_daily_feature_matrix"] = bool(name in daily_used)
        report["used_by_state_builder"] = bool(name in state_inputs or str(name).startswith("alpha_prior_"))
        report["used_by_label_builder"] = bool(name in label_inputs)
        report["used_by_v5_oracle"] = bool(name in oracle_inputs)
        report["used_by_r71_diagnostics"] = bool("regret" in name or "spread" in name or "crowding" in name)
        report["used_by_r74_behavior_quality"] = bool(name in r74_inputs)
        report["actually_used"] = bool(
            report["used_by_model_feature_matrix"]
            or report["used_by_daily_feature_matrix"]
            or report["used_by_state_builder"]
            or report["used_by_label_builder"]
            or report["used_by_v5_oracle"]
            or report["used_by_r71_diagnostics"]
            or report["used_by_r74_behavior_quality"]
        )
        report["high_value_hint"] = bool(any(token in name.lower() for token in HIGH_VALUE_FEATURE_HINTS))
        if name in {"close", "open", "high", "low", "volume", "amount", "score_blend", "score_v2", "score_none"} and float(report["finite_ratio"]) < 0.80:
            report["feature_contract_severity"] = "blocker"
        elif bool(report["low_variance"]) or (name in r74_inputs and float(report["finite_ratio"]) < 0.80):
            report["feature_contract_severity"] = "degraded"
        else:
            report["feature_contract_severity"] = "ok"
        reports.append(report)

    available = [item["feature"] for item in reports if float(item["finite_ratio"]) > 0.0]
    actually_used = [item["feature"] for item in reports if bool(item["actually_used"])]
    unused_high_value = [
        item["feature"]
        for item in reports
        if bool(item["high_value_hint"])
        and float(item["finite_ratio"]) >= 0.80
        and not bool(item["actually_used"])
        and not bool(item["low_variance"])
    ]
    missing_or_low_variance = [
        item["feature"]
        for item in reports
        if float(item["finite_ratio"]) <= 0.0 or bool(item["low_variance"])
    ]
    high_value_used = [
        item["feature"]
        for item in reports
        if bool(item["high_value_hint"]) and bool(item["actually_used"]) and float(item["finite_ratio"]) > 0.0
    ]
    blocker_features = [item["feature"] for item in reports if item.get("feature_contract_severity") == "blocker"]
    degraded_features = [item["feature"] for item in reports if item.get("feature_contract_severity") == "degraded"]
    missing_high_value_features = [name for name in R74_HIGH_VALUE_LAKE_FEATURES if name not in panels]
    return {
        "version": PORTFOLIO_DECISION_FEATURE_BUNDLE_VERSION,
        "status": "ok",
        "data_source": str(getattr(prepared, "data_source", "") or ""),
        "pool_name": str(getattr(prepared, "pool_name", "") or ""),
        "start_date": str(getattr(prepared, "start_date", "") or ""),
        "end_date": str(getattr(prepared, "end_date", "") or ""),
        "feature_panel_count": int(len(reports)),
        "available_feature_count": int(len(available)),
        "actually_used_feature_count": int(len(actually_used)),
        "unused_high_value_feature_count": int(len(unused_high_value)),
        "missing_or_low_variance_feature_count": int(len(missing_or_low_variance)),
        "decision_feature_high_value_used_count": int(len(high_value_used)),
        "decision_feature_high_value_unused_count": int(len(unused_high_value)),
        "decision_feature_degraded_count": int(len(degraded_features) + len(missing_high_value_features)),
        "decision_feature_blocker_count": int(len(blocker_features)),
        "available_features": available,
        "actually_used_features": actually_used,
        "high_value_used_features": high_value_used[:80],
        "unused_high_value_features": unused_high_value[:80],
        "missing_or_low_variance_features": missing_or_low_variance[:80],
        "missing_high_value_features": missing_high_value_features[:80],
        "feature_contract_degraded_features": (degraded_features + missing_high_value_features)[:80],
        "feature_contract_blocker_features": blocker_features[:80],
        "feature_reports": reports,
    }
