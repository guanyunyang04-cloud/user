from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Any

from daily_research.tools.brain_evidence_registry import query_evidence_registry
from daily_research.tools.brain_platform import (
    WORKSPACE_ROOT,
    build_workflow_guide,
    build_workflow_state,
    load_workflow_registry,
    read_text,
    resolve_bootstrap,
    select_workflow_for_task,
)
from daily_research.tools.brain_rules import run_brain_rules
from daily_research.tools.frontier_scanner import build_frontier_report


KEY_TERMS = (
    "continuous_policy",
    "data lake",
    "training",
    "study",
    "brain",
    "active artifact",
    "r61",
    "r62",
    "r63",
)


def _git_status() -> dict[str, Any]:
    result = subprocess.run(
        ["git", "status", "--short", "--branch"],
        cwd=str(WORKSPACE_ROOT),
        capture_output=True,
        text=True,
        encoding="utf-8",
        check=False,
    )
    lines = [line for line in (result.stdout or "").splitlines() if line.strip()]
    branch = lines[0] if lines else ""
    return {
        "branch_line": branch,
        "on_main": branch.startswith("## main"),
        "dirty_paths": [line for line in lines[1:]],
        "returncode": result.returncode,
    }


def _text_excerpt(path: str, *, max_lines: int = 18) -> list[str]:
    try:
        lines = read_text(path).splitlines()
    except Exception:
        return []
    out: list[str] = []
    for line in lines:
        stripped = line.strip()
        if not stripped:
            continue
        if stripped.startswith("#") or stripped.startswith("-"):
            out.append(stripped)
        if len(out) >= max_lines:
            break
    return out


def _task_terms(task: str) -> list[str]:
    lower = str(task or "").lower()
    terms = [term for term in KEY_TERMS if term in lower]
    for token in lower.replace("/", " ").replace("_", " ").split():
        if token.startswith("r") and token[1:].isdigit():
            terms.append(token)
    return sorted(set(terms))


def build_task_capsule(*, child: str = "daily_research", task: str = "", workflow: str = "brain_handoff", study_tag: str = "") -> dict[str, Any]:
    bootstrap = resolve_bootstrap(child).to_dict()
    registry = load_workflow_registry()
    selection = select_workflow_for_task(task) if workflow == "auto" else {"selected_workflow": workflow, "reason": "explicit workflow requested"}
    selected_workflow = str(selection.get("selected_workflow", "") or "brain_handoff")
    workflow_id = selected_workflow if selected_workflow in registry else "brain_handoff"
    workflow_state = build_workflow_state(workflow_id, study_tag=study_tag or None).to_dict()
    workflow_guide = build_workflow_guide(workflow_id)
    git = _git_status()
    rules = run_brain_rules(has_explicit_study_tag=bool(study_tag))
    frontier_report = build_frontier_report()
    terms = _task_terms(task)
    evidence_matches: list[dict[str, Any]] = []
    for term in terms or ["r63"]:
        evidence_matches.extend(query_evidence_registry(term).get("matches", [])[:5])
    seen_paths: set[str] = set()
    related_references: list[dict[str, Any]] = []
    for match in evidence_matches:
        if not isinstance(match, dict):
            continue
        path = str(match.get("path", "") or "")
        if not path or path in seen_paths:
            continue
        seen_paths.add(path)
        related_references.append(
            {
                "id": match.get("id", ""),
                "path": path,
                "verdict": match.get("verdict", ""),
                "blockers": match.get("blockers", []),
                "dataset_ids": match.get("dataset_ids", []),
            }
        )
    return {
        "schema_version": 1,
        "child": child,
        "task": task,
        "workflow": workflow_id,
        "selected_workflow": workflow_id,
        "workflow_selection": selection,
        "workflow_guide": workflow_guide,
        "required_checklist": workflow_guide.get("checklist", []),
        "stop_conditions": workflow_guide.get("stop_conditions", []),
        "study_tag": study_tag,
        "git": git,
        "active_artifact_guard": {
            "path": "daily_research/output/active_execution_strategy.json",
            "status": "clean" if not any(f["code"] == "active_artifact_diff" for f in rules["findings"]) else "dirty",
        },
        "brain_boot_order": bootstrap.get("boot_order", []),
        "fast_handoff_paths": bootstrap.get("child_fast_handoff_paths", []),
        "state_summary": _text_excerpt("daily_research/brain/state_center.md"),
        "hard_rules": _text_excerpt("daily_research/brain/knowledge_center.md"),
        "forbidden_actions": workflow_state.get("registry_entry", {}).get("forbidden_actions", []),
        "related_references": related_references,
        "frontier_report": frontier_report,
        "facts": [
            "daily_research brain is the project fact, state, governance, and evidence source",
            "active_execution_strategy.json is guarded and must not be changed without explicit promotion decision",
            "workflow registry describes operating workflows but does not replace brain truth",
        ],
        "inferences": [
            "Use explicit evidence references before interpreting latest artifacts",
            "Treat smoke, dry-run, failed trial, and realtime-tail labels as non-completed evidence unless proven otherwise",
            "If frontier_report.brain_may_be_stale is true, reconcile explicit output tags before answering current-state questions",
        ],
        "assumptions": [
            "Work remains on main",
            "Current task is research infrastructure unless the user explicitly requests promotion/live changes",
        ],
        "next_allowed_actions": workflow_state.get("next_allowed_actions", []),
        "validation_commands": [
            "git diff -- daily_research/output/active_execution_strategy.json",
            "git diff --check",
            "C:/Users/ASUS/miniconda3/envs/yolos/python.exe daily_research/tools/doc_guard.py check",
            "C:/Users/ASUS/miniconda3/envs/yolos/python.exe daily_research/tools/brain_integrity_check.py --json",
        ],
        "rule_report": rules,
    }
