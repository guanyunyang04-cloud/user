import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_ROOT = PROJECT_ROOT / "daily_research" / "output"
DEFAULT_LATEST_MANIFEST_PATH = OUTPUT_ROOT / "deep_alpha_family_epoch_budget_latest.json"
DEFAULT_FALLBACK_EPOCH_BUDGET = 8


@dataclass(frozen=True)
class FamilyBudgetSpec:
    family_key: str
    description: str
    runner_kind: str
    profile_name: str
    fallback_epochs: int = DEFAULT_FALLBACK_EPOCH_BUDGET


FAMILY_BUDGET_SPECS: dict[str, FamilyBudgetSpec] = {
    "baseline": FamilyBudgetSpec(
        family_key="baseline",
        description="Execution-first baseline architecture family.",
        runner_kind="architecture",
        profile_name="baseline_current",
    ),
    "structure": FamilyBudgetSpec(
        family_key="structure",
        description="Structure-context architecture family.",
        runner_kind="architecture",
        profile_name="structure_context_only",
    ),
    "short_alpha": FamilyBudgetSpec(
        family_key="short_alpha",
        description="Short-alpha family under state+liquidity listwise challenger.",
        runner_kind="short_alpha",
        profile_name="state_liquidity_listwise_v1",
    ),
    "dynamic_graph": FamilyBudgetSpec(
        family_key="dynamic_graph",
        description="Dynamic-graph family under the current no-priors challenger.",
        runner_kind="dynamic_graph",
        profile_name="dynamic_graph_no_priors",
    ),
}


def get_family_budget_spec(family_key: str) -> FamilyBudgetSpec:
    normalized = str(family_key or "").strip().lower()
    if normalized not in FAMILY_BUDGET_SPECS:
        available = ", ".join(sorted(FAMILY_BUDGET_SPECS))
        raise KeyError(f"Unknown family budget spec: {family_key}. Available: {available}")
    return FAMILY_BUDGET_SPECS[normalized]


def list_family_budget_lines() -> list[str]:
    lines = ["families:"]
    for key in sorted(FAMILY_BUDGET_SPECS):
        spec = FAMILY_BUDGET_SPECS[key]
        lines.append(f"- {spec.family_key}: {spec.profile_name} [{spec.runner_kind}] {spec.description}")
    return lines


def load_family_epoch_budget_manifest(path: str | Path | None = None) -> dict[str, Any]:
    manifest_path = DEFAULT_LATEST_MANIFEST_PATH if path in {None, ""} else Path(path)
    if not manifest_path.exists():
        return {}
    with open(manifest_path, "r", encoding="utf-8") as f:
        payload = json.load(f)
    if not isinstance(payload, dict):
        raise ValueError(f"Family epoch budget manifest is not a JSON object: {manifest_path}")
    return payload


def resolve_epoch_budget_for_family(
    family_key: str,
    *,
    manifest: dict[str, Any] | None = None,
    manifest_path: str | Path | None = None,
    fallback_epochs: int | None = None,
) -> int:
    payload = manifest if manifest is not None else load_family_epoch_budget_manifest(manifest_path)
    families = payload.get("families", {}) if isinstance(payload, dict) else {}
    family_payload = families.get(str(family_key), {}) if isinstance(families, dict) else {}
    recommended = family_payload.get("recommended_epoch_budget")
    if recommended is not None:
        value = int(recommended)
        if value > 0:
            return value
    spec = get_family_budget_spec(family_key)
    base = spec.fallback_epochs if fallback_epochs is None else int(fallback_epochs)
    return max(int(base), 1)

