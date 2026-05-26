from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

from tools.brain.evidence_registry import load_evidence_registry


@dataclass(frozen=True)
class RouteCandidate:
    brain_id: str
    score: int
    matched_terms: list[str]
    sources: list[str]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class RouteTarget:
    id: str
    kind: str
    domain: str
    bootstrap_aliases: list[str]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


WORKSPACE_TARGET = RouteTarget(
    id="workspace",
    kind="workspace",
    domain="workspace_governance",
    bootstrap_aliases=["workspace", "workspace_root", "workspace_governance"],
)


def _child_target(brain_id: str) -> RouteTarget:
    return RouteTarget(
        id=brain_id,
        kind="child",
        domain=brain_id,
        bootstrap_aliases=[brain_id],
    )


def _ambiguous_target() -> RouteTarget:
    return RouteTarget(id="", kind="ambiguous", domain="", bootstrap_aliases=[])


ROUTE_TERMS: dict[str, tuple[str, ...]] = {
    "workspace_governance": (
        "主脑",
        "分脑",
        "脑区",
        "brain",
        "capsule",
        "接管",
        "治理",
        "main-only",
        "branch",
        "worktree",
        "主分脑",
    ),
    "daily_research": (
        "daily_research",
        "daily_research/",
        "daily_research\\",
        "path20",
        "path_policy",
        "alpha_path20",
        "alpha_multi_horizon",
        "alpha_multi_horizon_utility_policy_v1",
        "multi-horizon",
        "multi horizon",
        "horizon root cause",
        "root-cause audit",
        "root cause audit",
        "根因审计",
        "长短周期",
        "效用排序",
        "forecast",
        "utility policy",
        "continuous_policy",
        "deep_alpha",
        "study",
        "training",
        "data lake",
        "active_execution_strategy",
        "active manifest",
        "执行端",
        "交易计划",
        "数据刷新",
        "数据集",
        "模型页",
        "production signal",
        "signal panel",
    ),
    "t0_project": (
        "t0",
        "盘中",
        "intraday",
        "rl",
        "reinforcement",
    ),
    "daily_stock_analysis-main": (
        "daily_stock_analysis",
        "多市场",
        "stock analysis product",
    ),
}

PROJECT_PATH_TERMS: dict[str, tuple[str, ...]] = {
    "daily_research": ("daily_research/", "daily_research\\"),
    "t0_project": ("t0_project/", "t0_project\\"),
    "daily_stock_analysis-main": ("daily_stock_analysis-main/", "daily_stock_analysis-main\\", "daily_stock_analysis/"),
}

DAILY_RESEARCH_REGISTRY_WORKFLOWS = {
    "path_policy",
    "execution",
    "continuous_policy",
    "research_data_lake",
    "daily_research",
}


def _matches(task: str, terms: tuple[str, ...]) -> list[str]:
    lower = task.lower()
    return [term for term in terms if term.lower() in lower or term in task]


def _project_path_matches(task: str) -> list[RouteCandidate]:
    candidates: list[RouteCandidate] = []
    for brain_id, terms in PROJECT_PATH_TERMS.items():
        matches = _matches(task, terms)
        if not matches:
            continue
        candidates.append(RouteCandidate(brain_id=brain_id, score=100 + len(matches), matched_terms=matches, sources=["path_match"]))
    return candidates


def _term_matches(task: str) -> list[RouteCandidate]:
    candidates: list[RouteCandidate] = []
    for brain_id, terms in ROUTE_TERMS.items():
        matches = _matches(task, terms)
        if not matches:
            continue
        candidates.append(RouteCandidate(brain_id=brain_id, score=len(matches), matched_terms=matches, sources=["term_match"]))
    return candidates


def _registry_daily_research_match(task: str) -> RouteCandidate | None:
    text = str(task or "").strip()
    if not text:
        return None
    try:
        payload = load_evidence_registry()
    except Exception:
        return None
    exact_terms: list[str] = []
    for record in payload.get("records", []) or []:
        if not isinstance(record, dict):
            continue
        path = str(record.get("path", "") or "").replace("\\", "/")
        workflow = str(record.get("workflow", "") or "")
        if not path.startswith("daily_research/brain/references/") or workflow not in DAILY_RESEARCH_REGISTRY_WORKFLOWS:
            continue
        values = [
            str(record.get("id", "") or ""),
            str(record.get("path", "") or ""),
            *[str(item) for item in record.get("research_programs", []) or []],
            *[str(item) for item in record.get("study_families", []) or []],
            *[str(item) for item in record.get("run_tags", []) or []],
            *[str(item) for item in record.get("dataset_ids", []) or []],
        ]
        for value in values:
            if value and value.lower() in text.lower():
                exact_terms.append(value)
    if not exact_terms:
        return None
    return RouteCandidate(
        brain_id="daily_research",
        score=80 + len(set(exact_terms)),
        matched_terms=sorted(set(exact_terms)),
        sources=["registry_exact_match"],
    )


def _merge_candidates(candidates: list[RouteCandidate]) -> list[RouteCandidate]:
    merged: dict[str, dict[str, Any]] = {}
    for candidate in candidates:
        item = merged.setdefault(candidate.brain_id, {"score": 0, "matched_terms": [], "sources": []})
        item["score"] += int(candidate.score)
        item["matched_terms"].extend(candidate.matched_terms)
        item["sources"].extend(candidate.sources)
    return [
        RouteCandidate(
            brain_id=brain_id,
            score=int(payload["score"]),
            matched_terms=sorted(set(payload["matched_terms"])),
            sources=sorted(set(payload["sources"])),
        )
        for brain_id, payload in merged.items()
    ]


def route_task_to_brain(task: str) -> dict[str, Any]:
    text = str(task or "").strip()
    raw_candidates = [*_project_path_matches(text), *_term_matches(text)]
    registry_candidate = _registry_daily_research_match(text)
    if registry_candidate is not None:
        raw_candidates.append(registry_candidate)
    candidates = _merge_candidates(raw_candidates)
    candidates = sorted(candidates, key=lambda item: (-item.score, item.brain_id))
    positive_children = [item for item in candidates if item.brain_id != "workspace_governance" and item.score > 0]
    selected = "workspace"
    target = WORKSPACE_TARGET
    status = "selected"
    reason = "no project path, route terms, or registry evidence matched; default to workspace governance"

    if len(positive_children) > 1:
        selected = ""
        target = _ambiguous_target()
        status = "ambiguous"
        reason = "multiple child brains matched; main brain must not default to daily_research"
    elif positive_children:
        selected = positive_children[0].brain_id
        target = _child_target(selected)
        reason = f"task matched {selected} terms: {', '.join(positive_children[0].matched_terms)}"
    elif any(item.brain_id == "workspace_governance" for item in candidates):
        reason = "task matched workspace governance terms"

    return {
        "status": status,
        "selected_brain_id": selected,
        "target": target.to_dict(),
        "candidates": [item.to_dict() for item in candidates],
        "routing_sources": sorted({source for item in candidates for source in item.sources}),
        "reason": reason,
    }
