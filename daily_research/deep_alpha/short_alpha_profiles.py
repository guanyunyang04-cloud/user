from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ShortAlphaProfile:
    name: str
    description: str
    state_context: bool = False
    liquidity_context: bool = False
    ranking_loss_weight: float = 0.0
    listwise_loss_weight: float = 0.0
    listwise_temperature: float = 0.35
    prediction_horizons: str = "5,10,20"
    task_loss_weights: str = "5:0.2,10:0.3,20:0.5,downside:0.35"
    score_horizon_weights: str = "5:0.2,10:0.3,20:0.5"
    short_alpha_features: bool = False
    breakout_event_horizon: int = 5
    breakout_event_threshold: float = 0.08
    breakout_event_pullback_limit: float = 0.03
    breakout_event_loss_weight: float = 0.0
    clean_breakout_event_loss_weight: float = 0.0


PROFILE_REGISTRY: dict[str, ShortAlphaProfile] = {
    "baseline_current": ShortAlphaProfile(
        name="baseline_current",
        description="Current dynamic_graph_v1 liquid500 recent-formal baseline.",
    ),
    "state_context_v1": ShortAlphaProfile(
        name="state_context_v1",
        description="Turn on learned market-state context without changing targets or inputs.",
        state_context=True,
    ),
    "state_liquidity_listwise_v1": ShortAlphaProfile(
        name="state_liquidity_listwise_v1",
        description="Joint structure-capacity probe: state + liquidity context with light ranking/listwise losses.",
        state_context=True,
        liquidity_context=True,
        ranking_loss_weight=0.02,
        listwise_loss_weight=0.05,
        listwise_temperature=0.35,
    ),
    "short_target_v1": ShortAlphaProfile(
        name="short_target_v1",
        description="Short-line target probe: 1/2/3/5-day excess targets plus breakout-event auxiliary labels.",
        prediction_horizons="1,2,3,5",
        task_loss_weights="1:0.30,2:0.25,3:0.25,5:0.20,downside:0.30",
        score_horizon_weights="1:0.35,2:0.25,3:0.20,5:0.20",
        breakout_event_horizon=5,
        breakout_event_threshold=0.08,
        breakout_event_pullback_limit=0.03,
        breakout_event_loss_weight=0.20,
        clean_breakout_event_loss_weight=0.20,
    ),
    "short_input_v1": ShortAlphaProfile(
        name="short_input_v1",
        description="Short-line input probe: add breakout/compression/energy candidate features while keeping current targets.",
        short_alpha_features=True,
    ),
    "short_combo_v1": ShortAlphaProfile(
        name="short_combo_v1",
        description="Combined short-line probe: short targets plus new breakout/compression/energy inputs.",
        prediction_horizons="1,2,3,5",
        task_loss_weights="1:0.30,2:0.25,3:0.25,5:0.20,downside:0.30",
        score_horizon_weights="1:0.35,2:0.25,3:0.20,5:0.20",
        short_alpha_features=True,
        breakout_event_horizon=5,
        breakout_event_threshold=0.08,
        breakout_event_pullback_limit=0.03,
        breakout_event_loss_weight=0.20,
        clean_breakout_event_loss_weight=0.20,
    ),
}

PROFILE_ALIASES: dict[str, str] = {
    "default": "baseline_current",
    "baseline": "baseline_current",
    "state": "state_context_v1",
    "capacity": "state_liquidity_listwise_v1",
    "target": "short_target_v1",
    "input": "short_input_v1",
    "combo": "short_combo_v1",
}

DEFAULT_SHORT_ALPHA_PROFILE = "baseline_current"


def resolve_profile_name(name: str) -> str:
    normalized = str(name or "").strip()
    if not normalized:
        raise KeyError("Empty short-alpha profile name.")
    return PROFILE_ALIASES.get(normalized, normalized)


def get_profile(name: str) -> ShortAlphaProfile:
    resolved = resolve_profile_name(name)
    if resolved not in PROFILE_REGISTRY:
        available = ", ".join(sorted(PROFILE_REGISTRY))
        aliases = ", ".join(f"{alias}->{target}" for alias, target in sorted(PROFILE_ALIASES.items()))
        raise KeyError(f"Unknown short-alpha profile: {name}. Available: {available}. Aliases: {aliases}")
    return PROFILE_REGISTRY[resolved]


def list_profile_lines() -> list[str]:
    lines = [f"default={DEFAULT_SHORT_ALPHA_PROFILE}", "profiles:"]
    for key in sorted(PROFILE_REGISTRY):
        profile = PROFILE_REGISTRY[key]
        lines.append(f"- {profile.name}: {profile.description}")
    alias_text = ", ".join(f"{alias}->{target}" for alias, target in sorted(PROFILE_ALIASES.items()))
    lines.append(f"aliases: {alias_text}")
    return lines
