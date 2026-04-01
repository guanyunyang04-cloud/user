from __future__ import annotations

from typing import Any

import pandas as pd

from daily_research.baseline.regime import normalize_regime_state_selector, resolve_regime_label_series


SOFT_STATE_PROFILE_SPECS: dict[str, dict[str, Any]] = {
    "quadrant_guard_v1": {
        "selector": "quadrant",
        "gross_map": {
            "trend_up_low_vol": 1.00,
            "trend_up_high_vol": 0.95,
            "trend_down_low_vol": 0.75,
            "trend_down_high_vol": 0.55,
        },
        "description": "Quadrant-aware cash buffer: keep full risk in favorable states and soften exposure in weaker quadrants.",
    },
    "trend_guard_v1": {
        "selector": "trend_bucket",
        "gross_map": {
            "trend_up": 1.00,
            "trend_flat": 0.82,
            "trend_down": 0.64,
        },
        "description": "Trend-bucket soft sizing: retain full exposure in uptrends and hold partial cash in flat/down trends.",
    },
    "market_state_guard_v1": {
        "selector": "market_state",
        "gross_map": {
            "trend_up_vol_low": 1.00,
            "trend_up_vol_mid": 0.96,
            "trend_up_vol_high": 0.88,
            "trend_flat_vol_low": 0.82,
            "trend_flat_vol_mid": 0.68,
            "trend_flat_vol_high": 0.52,
            "trend_down_vol_low": 0.72,
            "trend_down_vol_mid": 0.54,
            "trend_down_vol_high": 0.34,
        },
        "description": "Full state soft sizing: richer trend/vol buckets keep the bridge invested while reserving cash in adverse states.",
    },
}


def parse_state_value_map(raw: str | None) -> dict[str, float]:
    mapping: dict[str, float] = {}
    if raw in {None, ""}:
        return mapping
    for item in str(raw).split(","):
        chunk = str(item).strip()
        if not chunk:
            continue
        key, sep, value = chunk.partition(":")
        if not sep:
            raise ValueError(
                f"Invalid soft-state map item {chunk!r}. Expected label:value, e.g. trend_up_low_vol:1.0"
            )
        label = str(key).strip().lower()
        if not label:
            raise ValueError(f"Invalid soft-state map item {chunk!r}: empty label.")
        parsed = float(value)
        if parsed < 0.0 or parsed > 1.0:
            raise ValueError(
                f"Soft-state gross exposure must be within [0, 1], got {parsed} for label {label!r}."
            )
        mapping[label] = parsed
    return mapping


def resolve_soft_state_profile(
    *,
    profile: str | None,
    selector: str | None = None,
    gross_map_raw: str | None = None,
) -> dict[str, Any]:
    normalized_profile = str(profile or "").strip().lower()
    custom_gross_map = parse_state_value_map(gross_map_raw)
    if normalized_profile in {"", "off", "none"} and not custom_gross_map:
        return {
            "enabled": False,
            "profile": "off",
            "selector": normalize_regime_state_selector(selector or "quadrant"),
            "gross_map": {},
            "description": "disabled",
        }

    if normalized_profile in {"", "off", "none"}:
        resolved_profile = "custom"
        base_spec: dict[str, Any] = {"selector": selector or "quadrant", "gross_map": {}}
    else:
        if normalized_profile not in SOFT_STATE_PROFILE_SPECS:
            raise ValueError(
                f"Unsupported soft-state profile: {profile}. Expected one of {sorted(SOFT_STATE_PROFILE_SPECS)}."
            )
        resolved_profile = normalized_profile
        base_spec = SOFT_STATE_PROFILE_SPECS[normalized_profile]

    resolved_selector = normalize_regime_state_selector(selector or base_spec.get("selector", "quadrant"))
    gross_map = {
        str(label).strip().lower(): float(value)
        for label, value in dict(base_spec.get("gross_map", {})).items()
    }
    if custom_gross_map:
        gross_map = custom_gross_map

    if not gross_map:
        return {
            "enabled": False,
            "profile": resolved_profile,
            "selector": resolved_selector,
            "gross_map": {},
            "description": "disabled",
        }

    return {
        "enabled": True,
        "profile": resolved_profile,
        "selector": resolved_selector,
        "gross_map": gross_map,
        "description": str(base_spec.get("description", "")),
    }


def build_soft_state_scale_series(
    regime_state: pd.DataFrame,
    *,
    selector: str,
    gross_map: dict[str, float],
    default_gross: float = 1.0,
) -> pd.Series:
    label_series = resolve_regime_label_series(regime_state, selector).astype("string").str.lower()
    scale_series = pd.Series(float(default_gross), index=label_series.index, dtype=float)
    for label, gross in gross_map.items():
        scale_series.loc[label_series.eq(str(label).strip().lower()).fillna(False)] = float(gross)
    scale_series.loc[label_series.isna()] = float(default_gross)
    return scale_series.clip(lower=0.0, upper=1.0)


def apply_soft_state_sizing(
    target_weights: pd.DataFrame,
    regime_state: pd.DataFrame,
    *,
    profile_meta: dict[str, Any],
    default_gross: float = 1.0,
) -> tuple[pd.DataFrame, pd.Series, dict[str, Any]]:
    if not bool(profile_meta.get("enabled")):
        scale_series = pd.Series(float(default_gross), index=target_weights.index, dtype=float)
        return target_weights.copy(), scale_series, {
            "soft_state_profile": str(profile_meta.get("profile", "off")),
            "soft_state_enabled": False,
            "soft_state_selector": str(profile_meta.get("selector", "quadrant")),
            "soft_state_gross_map": {},
            "soft_state_scale_mean": float(scale_series.mean()) if not scale_series.empty else 1.0,
            "soft_state_scale_min": float(scale_series.min()) if not scale_series.empty else 1.0,
            "soft_state_scale_max": float(scale_series.max()) if not scale_series.empty else 1.0,
            "soft_state_scale_active_ratio": 0.0,
        }

    scale_series = build_soft_state_scale_series(
        regime_state,
        selector=str(profile_meta["selector"]),
        gross_map=dict(profile_meta["gross_map"]),
        default_gross=default_gross,
    ).reindex(target_weights.index).fillna(float(default_gross))
    scaled = target_weights.mul(scale_series.astype(float), axis=0).fillna(0.0)
    active_ratio = float(scale_series.lt(float(default_gross) - 1e-12).mean()) if len(scale_series) > 0 else 0.0
    meta = {
        "soft_state_profile": str(profile_meta.get("profile", "custom")),
        "soft_state_enabled": True,
        "soft_state_selector": str(profile_meta["selector"]),
        "soft_state_gross_map": dict(profile_meta["gross_map"]),
        "soft_state_scale_mean": float(scale_series.mean()) if not scale_series.empty else float(default_gross),
        "soft_state_scale_min": float(scale_series.min()) if not scale_series.empty else float(default_gross),
        "soft_state_scale_max": float(scale_series.max()) if not scale_series.empty else float(default_gross),
        "soft_state_scale_active_ratio": active_ratio,
    }
    if str(profile_meta.get("description", "")).strip():
        meta["soft_state_description"] = str(profile_meta["description"])
    return scaled, scale_series, meta
