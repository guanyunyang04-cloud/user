from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class DynamicGraphProfile:
    name: str
    description: str
    dynamic_graph_layer: bool
    top_k: int
    temperature: float
    industry_boost: float
    style_boost: float


PROFILE_REGISTRY: dict[str, DynamicGraphProfile] = {
    "plain_baseline": DynamicGraphProfile(
        name="plain_baseline",
        description="Patch-transformer plain baseline under the strict liquid800 formal protocol.",
        dynamic_graph_layer=False,
        top_k=0,
        temperature=0.35,
        industry_boost=0.0,
        style_boost=0.0,
    ),
    "dynamic_graph_v1": DynamicGraphProfile(
        name="dynamic_graph_v1",
        description="Current dynamic_graph winner: top-k 8, temperature 0.35, industry boost 0.15, style boost 0.05.",
        dynamic_graph_layer=True,
        top_k=8,
        temperature=0.35,
        industry_boost=0.15,
        style_boost=0.05,
    ),
    "dynamic_graph_topk4": DynamicGraphProfile(
        name="dynamic_graph_topk4",
        description="Concentrated graph ablation with top-k 4.",
        dynamic_graph_layer=True,
        top_k=4,
        temperature=0.35,
        industry_boost=0.15,
        style_boost=0.05,
    ),
    "dynamic_graph_topk12": DynamicGraphProfile(
        name="dynamic_graph_topk12",
        description="Wider graph ablation with top-k 12.",
        dynamic_graph_layer=True,
        top_k=12,
        temperature=0.35,
        industry_boost=0.15,
        style_boost=0.05,
    ),
    "dynamic_graph_no_industry_boost": DynamicGraphProfile(
        name="dynamic_graph_no_industry_boost",
        description="Ablation removing the industry prior while keeping style prior.",
        dynamic_graph_layer=True,
        top_k=8,
        temperature=0.35,
        industry_boost=0.0,
        style_boost=0.05,
    ),
    "dynamic_graph_no_style_boost": DynamicGraphProfile(
        name="dynamic_graph_no_style_boost",
        description="Ablation removing the style prior while keeping the industry prior.",
        dynamic_graph_layer=True,
        top_k=8,
        temperature=0.35,
        industry_boost=0.15,
        style_boost=0.0,
    ),
    "dynamic_graph_no_priors": DynamicGraphProfile(
        name="dynamic_graph_no_priors",
        description="Ablation removing both industry and style priors.",
        dynamic_graph_layer=True,
        top_k=8,
        temperature=0.35,
        industry_boost=0.0,
        style_boost=0.0,
    ),
}

PROFILE_ALIASES: dict[str, str] = {
    "default": "dynamic_graph_v1",
    "winner": "dynamic_graph_v1",
    "plain": "plain_baseline",
}

DEFAULT_DYNAMIC_GRAPH_PROFILE = "dynamic_graph_v1"


def resolve_profile_name(name: str) -> str:
    normalized = str(name or "").strip()
    if not normalized:
        raise KeyError("Empty dynamic graph profile name.")
    return PROFILE_ALIASES.get(normalized, normalized)


def get_profile(name: str) -> DynamicGraphProfile:
    resolved = resolve_profile_name(name)
    if resolved not in PROFILE_REGISTRY:
        available = ", ".join(sorted(PROFILE_REGISTRY))
        aliases = ", ".join(f"{alias}->{target}" for alias, target in sorted(PROFILE_ALIASES.items()))
        raise KeyError(f"Unknown dynamic graph profile: {name}. Available: {available}. Aliases: {aliases}")
    return PROFILE_REGISTRY[resolved]


def list_profile_lines() -> list[str]:
    lines = [f"default={DEFAULT_DYNAMIC_GRAPH_PROFILE}", "profiles:"]
    for key in sorted(PROFILE_REGISTRY):
        profile = PROFILE_REGISTRY[key]
        lines.append(f"- {profile.name}: {profile.description}")
    alias_text = ", ".join(f"{alias}->{target}" for alias, target in sorted(PROFILE_ALIASES.items()))
    lines.append(f"aliases: {alias_text}")
    return lines
