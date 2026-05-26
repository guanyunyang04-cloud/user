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
RUN_TAG_PATTERN = re.compile(
    r"\b(?:self_opt_study|protocol|path20|alpha_path20|mh_utility|mh_short|mh_mid|mh_long|mh_out|mh_grid|mh25)_[A-Za-z0-9_]+"
)
RESEARCH_PROGRAM_PATTERN = re.compile(r"\balpha_multi_horizon_utility_policy_v\d+\b")
EXPLICIT_STUDY_FAMILY_PATTERN = re.compile(r"(?:Study family|study_family)\s*:\s*`?([A-Za-z0-9_]+)`?", re.IGNORECASE)
CODE_TOKEN_PATTERN = re.compile(r"`([A-Za-z][A-Za-z0-9_]*(?:_[A-Za-z0-9]+)+)`")
RESEARCH_POINTER_PATTERN = re.compile(r"^alpha_[A-Za-z0-9_]+_policy_v\d+$")
RUN_INSTANCE_MARKER_PATTERN = re.compile(
    r"(?:20\d{6}|seed\d+|\br\d+[a-z]?\b|protocol_|study_|smoke_|dryrun_|fullgrid|daily\d|liquid\d|h\d)",
    re.IGNORECASE,
)
RUN_TAG_SECTION_HEADINGS = (
    "run tags",
    "run tag",
)
STATUS_FAMILY_MAP = (
    ("stage25_completed", "stage25_stability_calibration"),
    ("stage2_completed", "stage2_horizon_grid_calibration"),
    ("stage1_completed", "stage1_output_aux_grid"),
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


def _candidate_run_tokens(text: str) -> list[str]:
    candidates = [*RUN_TAG_PATTERN.findall(text), *_section_code_tokens(text, RUN_TAG_SECTION_HEADINGS)]
    research_program_tokens = set(RESEARCH_PROGRAM_PATTERN.findall(text))
    return _dedupe(tag for tag in candidates if tag not in research_program_tokens)


def research_programs(text: str) -> list[str]:
    programs = RESEARCH_PROGRAM_PATTERN.findall(text)
    lower = text.lower()
    if (
        "alpha multi-horizon" in lower
        or "alpha_multi_horizon" in lower
        or any(tag.startswith(("mh_", "mh25_")) for tag in _candidate_run_tokens(text))
    ):
        programs.append("alpha_multi_horizon_utility_policy_v1")
    return _dedupe(programs)


def run_tags(text: str) -> list[str]:
    programs = set(research_programs(text))
    return _dedupe(tag for tag in _candidate_run_tokens(text) if tag not in programs and _is_run_instance_tag(tag))


def study_families(text: str) -> list[str]:
    explicit = _dedupe(EXPLICIT_STUDY_FAMILY_PATTERN.findall(text))
    if explicit:
        return explicit
    status_family = _family_from_status(text)
    if status_family:
        return [status_family]
    title = _document_title(text)
    if "stage 2.5" in title or "stage25" in title:
        return ["stage25_stability_calibration"]
    if "stage 2" in title and ("calibration" in title or "horizon-grid" in title or "horizon grid" in title):
        return ["stage2_horizon_grid_calibration"]
    if "stage 1" in title and (
        "output/aux" in title
        or "output/loss" in title
        or "output aux" in title
        or "output-aux" in title
    ):
        return ["stage1_output_aux_grid"]
    return []


def _section_code_tokens(text: str, heading_keywords: Iterable[str]) -> list[str]:
    keywords = [keyword.lower() for keyword in heading_keywords]
    tokens: list[str] = []
    in_section = False
    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith("#"):
            heading = stripped.lstrip("#").strip().lower()
            in_section = any(keyword in heading for keyword in keywords)
            continue
        if not in_section:
            continue
        tokens.extend(CODE_TOKEN_PATTERN.findall(stripped))
    return _dedupe(tokens)


def _family_from_status(text: str) -> str:
    for line in text.splitlines()[:24]:
        lowered = line.lower()
        if "status" not in lowered:
            continue
        for status_token, family in STATUS_FAMILY_MAP:
            if status_token in lowered:
                return family
    return ""


def _document_title(text: str) -> str:
    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith("#"):
            return stripped.lstrip("#").strip().lower()
    return ""


def _is_run_instance_tag(tag: str) -> bool:
    normalized = str(tag or "").strip()
    if not normalized:
        return False
    if RESEARCH_POINTER_PATTERN.match(normalized):
        return False
    return bool(RUN_INSTANCE_MARKER_PATTERN.search(normalized))


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
