from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ArchitectureProfile:
    name: str
    description: str
    category: str
    hidden_dim: int = 96
    encoder_family: str = "patch_transformer"
    patch_len: int = 5
    transformer_heads: int = 4
    transformer_layers: int = 2
    dynamic_graph_layer: bool = True
    dynamic_graph_top_k: int = 8
    dynamic_graph_temperature: float = 0.35
    dynamic_graph_industry_boost: float = 0.15
    dynamic_graph_style_boost: float = 0.05
    relation_layer: bool = False
    state_context: bool = False
    liquidity_context: bool = False
    structure_context: bool = False
    aux_structure_task: bool = False
    aux_structure_loss_weight: float = 0.10
    aux_structure_label_smoothing: float = 0.05
    structure_prototype_task: bool = False
    structure_prototype_loss_weight: float = 0.05
    structure_prototype_temperature: float = 0.20


PROFILE_REGISTRY: dict[str, ArchitectureProfile] = {
    "baseline_current": ArchitectureProfile(
        name="baseline_current",
        description="Current liquid500 recent-formal winner: patch_transformer + dynamic_graph_v1.",
        category="baseline",
    ),
    "capacity_small_h64": ArchitectureProfile(
        name="capacity_small_h64",
        description="Lower model complexity: hidden_dim 64 with the rest of the winner protocol unchanged.",
        category="complexity",
        hidden_dim=64,
    ),
    "capacity_large_h160": ArchitectureProfile(
        name="capacity_large_h160",
        description="Higher model complexity: hidden_dim 160 with the rest of the winner protocol unchanged.",
        category="complexity",
        hidden_dim=160,
    ),
    "depth_shallow_l1": ArchitectureProfile(
        name="depth_shallow_l1",
        description="Shallower network: transformer_layers 1.",
        category="depth",
        transformer_layers=1,
    ),
    "depth_deep_l4": ArchitectureProfile(
        name="depth_deep_l4",
        description="Deeper network: transformer_layers 4.",
        category="depth",
        transformer_layers=4,
    ),
    "encoder_transformer_v1": ArchitectureProfile(
        name="encoder_transformer_v1",
        description="Change backbone structure from patch_transformer to vanilla transformer.",
        category="encoder",
        encoder_family="transformer",
    ),
    "encoder_mamba_v1": ArchitectureProfile(
        name="encoder_mamba_v1",
        description="Change backbone structure from patch_transformer to mamba.",
        category="encoder",
        encoder_family="mamba",
    ),
    "graph_off_plain": ArchitectureProfile(
        name="graph_off_plain",
        description="Remove dynamic graph and relation modules to test plain backbone structure.",
        category="graph",
        dynamic_graph_layer=False,
    ),
    "graph_relation_only": ArchitectureProfile(
        name="graph_relation_only",
        description="Disable dynamic graph but keep lightweight relation_layer features.",
        category="graph",
        dynamic_graph_layer=False,
        relation_layer=True,
    ),
    "graph_topk4": ArchitectureProfile(
        name="graph_topk4",
        description="Reduce dynamic graph fan-out from top-k 8 to top-k 4.",
        category="graph",
        dynamic_graph_top_k=4,
    ),
    "state_context_only": ArchitectureProfile(
        name="state_context_only",
        description="Turn on learned market-state context without touching losses or targets.",
        category="context",
        state_context=True,
    ),
    "state_liquidity_context": ArchitectureProfile(
        name="state_liquidity_context",
        description="Turn on state + liquidity context only, keeping the winner losses unchanged.",
        category="context",
        state_context=True,
        liquidity_context=True,
    ),
    "structure_context_only": ArchitectureProfile(
        name="structure_context_only",
        description="Turn on structure-label context embeddings only.",
        category="structure",
        structure_context=True,
    ),
    "structure_aux_task_v1": ArchitectureProfile(
        name="structure_aux_task_v1",
        description="Add the auxiliary structure-classification task as a light regularizer.",
        category="structure",
        aux_structure_task=True,
    ),
    "structure_prototype_task_v1": ArchitectureProfile(
        name="structure_prototype_task_v1",
        description="Add the structure-prototype regularization task without changing the main losses.",
        category="structure",
        structure_prototype_task=True,
    ),
}

PROFILE_ALIASES: dict[str, str] = {
    "default": "baseline_current",
    "baseline": "baseline_current",
    "small": "capacity_small_h64",
    "large": "capacity_large_h160",
    "shallow": "depth_shallow_l1",
    "deep": "depth_deep_l4",
    "transformer": "encoder_transformer_v1",
    "mamba": "encoder_mamba_v1",
    "plain": "graph_off_plain",
    "relation": "graph_relation_only",
    "topk4": "graph_topk4",
    "state": "state_context_only",
    "state_liq": "state_liquidity_context",
    "struct_ctx": "structure_context_only",
    "struct_aux": "structure_aux_task_v1",
    "struct_proto": "structure_prototype_task_v1",
}

DEFAULT_ARCHITECTURE_PROFILE = "baseline_current"


def resolve_profile_name(name: str) -> str:
    normalized = str(name or "").strip()
    if not normalized:
        raise KeyError("Empty architecture profile name.")
    return PROFILE_ALIASES.get(normalized, normalized)


def get_profile(name: str) -> ArchitectureProfile:
    resolved = resolve_profile_name(name)
    if resolved not in PROFILE_REGISTRY:
        available = ", ".join(sorted(PROFILE_REGISTRY))
        aliases = ", ".join(f"{alias}->{target}" for alias, target in sorted(PROFILE_ALIASES.items()))
        raise KeyError(f"Unknown architecture profile: {name}. Available: {available}. Aliases: {aliases}")
    return PROFILE_REGISTRY[resolved]


def list_profile_lines() -> list[str]:
    lines = [f"default={DEFAULT_ARCHITECTURE_PROFILE}", "profiles:"]
    for key in sorted(PROFILE_REGISTRY):
        profile = PROFILE_REGISTRY[key]
        lines.append(f"- {profile.name} [{profile.category}]: {profile.description}")
    alias_text = ", ".join(f"{alias}->{target}" for alias, target in sorted(PROFILE_ALIASES.items()))
    lines.append(f"aliases: {alias_text}")
    return lines
