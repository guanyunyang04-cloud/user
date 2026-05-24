from __future__ import annotations

import re
from pathlib import Path
from typing import Iterable


REGISTRY_PATH = Path("daily_research/brain/references/evidence_registry.json")
REFERENCE_ROOTS = (Path("daily_research/brain/references"),)
REFERENCE_FILE_PATTERNS = (
    re.compile(r"^(r\d+[a-z]?|gpu-runtime)-.+\.md$"),
    re.compile(r"^(alpha_path20|path20|path_policy|alpha_multi_horizon)_.+\.md$"),
    re.compile(r"^data_lake_.+\.md$"),
    re.compile(r"^tdx_free_data_platform_.+\.md$"),
    re.compile(r"^execution_.+\.md$"),
    re.compile(r"^(brain_native|brain_system|api_agent)_.+\.md$"),
)
STUDY_TAG_PATTERN = re.compile(
    r"\b(?:self_opt_study|protocol|path20|alpha_path20|mh_utility|mh_short|mh_mid|mh_long|alpha_multi_horizon)_[A-Za-z0-9_]+"
)


def is_reference_file(path: Path) -> bool:
    return (
        path.is_file()
        and (
            any(pattern.match(path.name) for pattern in REFERENCE_FILE_PATTERNS)
            or path.name.startswith("r")
        )
    )


def owns_reference_path(path: Path, workspace_root: Path) -> bool:
    normalized = path if path.is_absolute() else workspace_root / path
    for root_rel in REFERENCE_ROOTS:
        root = workspace_root / root_rel
        try:
            normalized.relative_to(root)
        except ValueError:
            continue
        return True
    return False


def workflow_from(path: Path, text: str) -> str:
    lower = f"{path.name}\n{text}".lower()
    if path.name.startswith(("brain_native", "brain_system", "api_agent")):
        return "brain"
    if "brain maintenance" in lower or "brain-skill" in lower or "brain_skill" in lower:
        return "brain"
    if "brain / workflow" in lower or "workflow maintenance" in lower:
        return "brain"
    if "tdx-free" in lower or "tdx_free" in lower or "data platform" in lower:
        return "research_data_lake"
    if path.name.startswith("execution_") or "daily_research.execution" in lower or "execution console" in lower:
        return "execution"
    if "path_policy" in lower or "alpha_path20" in lower or "alpha_multi_horizon" in lower or "mh_utility" in lower:
        return "path_policy"
    if "continuous_policy" in lower or "core_v4" in lower or "release_first" in lower:
        return "continuous_policy"
    if "data lake" in lower or "research_data_lake" in lower:
        return "research_data_lake"
    if "brain" in lower:
        return "brain"
    return "daily_research"


def extra_tags(path: Path, text: str, workflow: str) -> list[str]:
    haystack = f"{path.name}\n{text[:800]}".lower()
    tags: list[str] = []
    if workflow == "brain":
        tags.append("brain")
        return _dedupe(tags)
    is_portfolio_set_v5 = "portfolio-set" in haystack or "portfolio_set" in haystack or "portfolio set" in haystack
    if "tdx-free" in haystack or "tdx_free" in haystack or "data platform" in haystack:
        tags.extend(["data_lake", "data_platform"])
        return _dedupe(tags)
    if workflow == "execution":
        tags.append("execution")
        if "signal" in haystack:
            tags.append("signal_panel")
        if "trade" in haystack:
            tags.append("trade_plan")
        if "data refresh" in haystack or "data_refresh" in haystack:
            tags.append("data_platform")
        return _dedupe(tags)
    if "data lake" in haystack or "research_data_lake" in haystack:
        tags.append("data_lake")
    if "path_policy" in haystack or "alpha_path20" in haystack or "alpha_multi_horizon" in haystack or "mh_utility" in haystack:
        tags.append("path_policy")
    if "gpu" in haystack or "cuda" in haystack:
        tags.append("gpu")
    if not is_portfolio_set_v5 and ("core-v4" in haystack or "core_v4" in haystack):
        tags.append("core_v4")
    if is_portfolio_set_v5:
        tags.append("portfolio_set_v5")
    if "brain-skill" in haystack or "brain skill" in haystack:
        tags.append("brain")
    return _dedupe(tags)


def study_tags(text: str) -> list[str]:
    return _dedupe(STUDY_TAG_PATTERN.findall(text))


def _dedupe(items: Iterable[str]) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for item in items:
        text = str(item or "").strip()
        if not text or text in seen:
            continue
        seen.add(text)
        out.append(text)
    return out
