from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


@dataclass(frozen=True)
class RouteCandidate:
    brain_id: str
    score: int
    matched_terms: list[str]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


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
        "path20",
        "continuous_policy",
        "deep_alpha",
        "study",
        "training",
        "data lake",
        "active_execution_strategy",
        "active manifest",
        "daily_research",
        "daily_research/",
        "daily_research\\",
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


def _matches(task: str, terms: tuple[str, ...]) -> list[str]:
    lower = task.lower()
    return [term for term in terms if term.lower() in lower or term in task]


def route_task_to_brain(task: str) -> dict[str, Any]:
    text = str(task or "").strip()
    candidates = [
        RouteCandidate(brain_id=brain_id, score=len(matches), matched_terms=matches)
        for brain_id, terms in ROUTE_TERMS.items()
        if (matches := _matches(text, terms))
    ]
    candidates = sorted(candidates, key=lambda item: (-item.score, item.brain_id))
    positive_children = [item for item in candidates if item.brain_id != "workspace_governance" and item.score > 0]
    selected = "workspace_governance"
    status = "selected"
    reason = "no project-specific route matched; default to workspace governance"

    if len(positive_children) > 1:
        selected = ""
        status = "ambiguous"
        reason = "multiple child brains matched; main brain must not default to daily_research"
    elif positive_children:
        selected = positive_children[0].brain_id
        reason = f"task matched {selected} terms: {', '.join(positive_children[0].matched_terms)}"
    elif any(item.brain_id == "workspace_governance" for item in candidates):
        reason = "task matched workspace governance terms"

    return {
        "status": status,
        "selected_brain_id": selected,
        "candidates": [item.to_dict() for item in candidates],
        "reason": reason,
    }
