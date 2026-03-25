from __future__ import annotations

from dataclasses import replace
from typing import Dict

from daily_research.baseline.config import ResearchConfig


def _merge_dict(base: dict, updates: dict) -> dict:
    merged = dict(base)
    merged.update(updates)
    return merged


def _build_up_low_breakout_config(base_config: ResearchConfig) -> ResearchConfig:
    group_weights = _merge_dict(
        base_config.factor_group_weights,
        {
            "trend": 0.20,
            "volume": 0.25,
            "volatility": 0.25,
            "structure": 0.30,
        },
    )
    factor_weights = _merge_dict(
        base_config.factor_weights,
        {
            "mom_5": 0.10,
            "mom_20": 0.00,
            "mom_60": 0.00,
            "ma_gap_10": 0.00,
            "ma_gap_20_60": 0.00,
            "trend_slope_20": 0.00,
            "breakout_20": 0.60,
            "vol_ratio_5_20": 0.00,
            "breakout_volume": 0.00,
            "volume_contraction": 0.35,
            "price_volume_divergence": 0.25,
            "atr_14_pct": 0.20,
            "volatility_20": 0.35,
            "volatility_contraction": 0.20,
            "range_position_20": 0.40,
            "drawdown_20": 0.40,
            "close_strength": 0.60,
            "body_strength": 0.00,
            "up_day_ratio_10": 0.35,
            "trend_streak": 0.25,
        },
    )
    return replace(
        base_config,
        factor_group_weights=group_weights,
        factor_weights=factor_weights,
    )


def _build_up_low_breakout_v2_config(base_config: ResearchConfig) -> ResearchConfig:
    group_weights = _merge_dict(
        base_config.factor_group_weights,
        {
            "trend": 0.05,
            "volume": 0.32,
            "volatility": 0.33,
            "structure": 0.30,
        },
    )
    factor_weights = _merge_dict(
        base_config.factor_weights,
        {
            "mom_5": 0.00,
            "mom_20": 0.00,
            "mom_60": 0.00,
            "ma_gap_10": 0.00,
            "ma_gap_20_60": 0.00,
            "trend_slope_20": 0.00,
            "breakout_20": 0.25,
            "vol_ratio_5_20": 0.00,
            "breakout_volume": 0.00,
            "volume_contraction": 0.45,
            "price_volume_divergence": 0.35,
            "atr_14_pct": 0.20,
            "volatility_20": 0.40,
            "volatility_contraction": 0.25,
            "range_position_20": 0.25,
            "drawdown_20": 0.25,
            "close_strength": 0.75,
            "body_strength": 0.00,
            "up_day_ratio_10": 0.15,
            "trend_streak": 0.10,
        },
    )
    return replace(
        base_config,
        factor_group_weights=group_weights,
        factor_weights=factor_weights,
    )


def _build_up_high_momentum_config(base_config: ResearchConfig) -> ResearchConfig:
    group_weights = _merge_dict(
        base_config.factor_group_weights,
        {
            "trend": 0.40,
            "volume": 0.20,
            "volatility": 0.10,
            "structure": 0.30,
        },
    )
    factor_weights = _merge_dict(
        base_config.factor_weights,
        {
            "mom_5": 0.00,
            "mom_20": 0.25,
            "mom_60": 0.45,
            "ma_gap_10": 0.00,
            "ma_gap_20_60": 0.50,
            "trend_slope_20": 0.20,
            "breakout_20": 0.00,
            "vol_ratio_5_20": 0.00,
            "breakout_volume": 0.00,
            "volume_contraction": 0.20,
            "price_volume_divergence": 0.20,
            "atr_14_pct": 0.00,
            "volatility_20": 0.00,
            "volatility_contraction": 0.10,
            "range_position_20": 0.00,
            "drawdown_20": 0.00,
            "close_strength": 0.35,
            "body_strength": 0.15,
            "up_day_ratio_10": 0.25,
            "trend_streak": 0.25,
        },
    )
    return replace(
        base_config,
        factor_group_weights=group_weights,
        factor_weights=factor_weights,
    )


def _build_down_low_rebound_config(base_config: ResearchConfig) -> ResearchConfig:
    group_weights = _merge_dict(
        base_config.factor_group_weights,
        {
            "trend": 0.08,
            "volume": 0.18,
            "volatility": 0.38,
            "structure": 0.26,
            "signal": 0.10,
        },
    )
    factor_weights = _merge_dict(
        base_config.factor_weights,
        {
            "mom_5": 0.12,
            "mom_20": -0.22,
            "mom_60": -0.08,
            "ma_gap_10": -0.08,
            "ma_gap_20_60": -0.05,
            "trend_slope_20": 0.00,
            "breakout_20": 0.00,
            "vol_ratio_5_20": 0.00,
            "breakout_volume": 0.00,
            "volume_contraction": 0.30,
            "price_volume_divergence": 0.22,
            "atr_14_pct": 0.35,
            "volatility_20": 0.48,
            "volatility_contraction": 0.32,
            "range_position_20": -0.32,
            "drawdown_20": -0.22,
            "close_strength": 0.48,
            "body_strength": 0.22,
            "up_day_ratio_10": -0.08,
            "trend_streak": -0.12,
            "kama_gap": -0.08,
            "kama_slope": 0.05,
            "long_regime_flag": 0.00,
            "mbuy_flag": 0.18,
            "zjtp_flag": 0.08,
            "hcw_flag": 0.12,
        },
    )
    return replace(
        base_config,
        factor_group_weights=group_weights,
        factor_weights=factor_weights,
    )


def _build_down_high_reversal_config(base_config: ResearchConfig) -> ResearchConfig:
    group_weights = _merge_dict(
        base_config.factor_group_weights,
        {
            "trend": 0.05,
            "volume": 0.14,
            "volatility": 0.26,
            "structure": 0.20,
            "signal": 0.35,
        },
    )
    factor_weights = _merge_dict(
        base_config.factor_weights,
        {
            "mom_5": 0.18,
            "mom_20": -0.18,
            "mom_60": -0.08,
            "ma_gap_10": -0.12,
            "ma_gap_20_60": -0.10,
            "trend_slope_20": 0.00,
            "breakout_20": 0.00,
            "vol_ratio_5_20": 0.00,
            "breakout_volume": 0.00,
            "volume_contraction": 0.12,
            "price_volume_divergence": 0.18,
            "atr_14_pct": 0.22,
            "volatility_20": 0.24,
            "volatility_contraction": 0.18,
            "range_position_20": -0.38,
            "drawdown_20": -0.32,
            "close_strength": 0.60,
            "body_strength": 0.32,
            "up_day_ratio_10": -0.12,
            "trend_streak": -0.18,
            "kama_gap": -0.12,
            "kama_slope": 0.08,
            "long_regime_flag": 0.00,
            "mbuy_flag": 0.40,
            "zjtp_flag": 0.32,
            "hcw_flag": 0.22,
        },
    )
    return replace(
        base_config,
        factor_group_weights=group_weights,
        factor_weights=factor_weights,
    )


def _build_up_low_breakout_v3_config(base_config: ResearchConfig) -> ResearchConfig:
    group_weights = _merge_dict(
        base_config.factor_group_weights,
        {
            "trend": 0.05,
            "volume": 0.26,
            "volatility": 0.27,
            "structure": 0.24,
            "signal": 0.18,
        },
    )
    factor_weights = _merge_dict(
        base_config.factor_weights,
        {
            "mom_5": 0.00,
            "mom_20": 0.00,
            "mom_60": 0.00,
            "ma_gap_10": 0.00,
            "ma_gap_20_60": 0.00,
            "trend_slope_20": 0.00,
            "breakout_20": 0.25,
            "vol_ratio_5_20": 0.00,
            "breakout_volume": 0.00,
            "volume_contraction": 0.42,
            "price_volume_divergence": 0.33,
            "atr_14_pct": 0.20,
            "volatility_20": 0.38,
            "volatility_contraction": 0.22,
            "range_position_20": 0.22,
            "drawdown_20": 0.22,
            "close_strength": 0.70,
            "body_strength": 0.00,
            "up_day_ratio_10": 0.18,
            "trend_streak": 0.12,
            "kama_gap": 0.18,
            "kama_slope": 0.16,
            "long_regime_flag": 0.10,
            "mbuy_flag": 0.22,
            "zjtp_flag": 0.12,
            "hcw_flag": 0.22,
        },
    )
    return replace(
        base_config,
        factor_group_weights=group_weights,
        factor_weights=factor_weights,
    )


def build_state_configs(
    base_config: ResearchConfig,
    profile_name: str,
) -> Dict[str, ResearchConfig]:
    profile_name = str(profile_name or "").strip().lower()
    if not profile_name or profile_name == "none":
        return {}

    if profile_name == "up_low_breakout_v1":
        return {
            "trend_up_low_vol": _build_up_low_breakout_config(base_config),
        }

    if profile_name == "up_low_breakout_v2":
        return {
            "trend_up_low_vol": _build_up_low_breakout_v2_config(base_config),
        }

    if profile_name == "up_low_breakout_v3":
        return {
            "trend_up_low_vol": _build_up_low_breakout_v3_config(base_config),
        }

    if profile_name == "up_dual_v1":
        return {
            "trend_up_low_vol": _build_up_low_breakout_config(base_config),
            "trend_up_high_vol": _build_up_high_momentum_config(base_config),
        }

    if profile_name == "up_dual_v2":
        return {
            "trend_up_low_vol": _build_up_low_breakout_v2_config(base_config),
            "trend_up_high_vol": _build_up_high_momentum_config(base_config),
        }

    if profile_name == "upv2_downlow_rebound_v1":
        return {
            "trend_up_low_vol": _build_up_low_breakout_v2_config(base_config),
            "trend_down_low_vol": _build_down_low_rebound_config(base_config),
        }

    if profile_name == "upv2_downdual_reversal_v1":
        return {
            "trend_up_low_vol": _build_up_low_breakout_v2_config(base_config),
            "trend_down_low_vol": _build_down_low_rebound_config(base_config),
            "trend_down_high_vol": _build_down_high_reversal_config(base_config),
        }

    raise ValueError(f"Unsupported state alpha profile: {profile_name}")
