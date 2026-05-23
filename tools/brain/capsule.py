from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Any

from tools.brain.adapters import daily_research as daily_research_adapter
from tools.brain.evidence_registry import query_evidence_registry
from tools.brain.platform import (
    PYTHON_EXECUTABLE,
    WORKSPACE_ROOT,
    build_workflow_guide,
    build_workflow_state,
    child_brain_ids,
    load_workflow_registry,
    read_text,
    resolve_bootstrap,
    select_workflow_for_task,
)
from tools.brain.routing import route_task_to_brain
from tools.brain.rules import run_brain_rules


PROJECT_TERM_DEFAULTS = {
    "daily_research": ("r63",),
    "workspace_governance": ("brain",),
}


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


def _task_terms(task: str, selected_brain_id: str) -> list[str]:
    lower = str(task or "").lower()
    terms = []
    for token in ("continuous_policy", "data lake", "training", "study", "brain", "active artifact", "path20"):
        if token in lower:
            terms.append(token)
    for token in lower.replace("/", " ").replace("_", " ").split():
        if token.startswith("r") and token[1:].isdigit():
            terms.append(token)
    if not terms:
        terms.extend(PROJECT_TERM_DEFAULTS.get(selected_brain_id, ()))
    return sorted(set(terms))


def _related_references(task: str, selected_brain_id: str) -> list[dict[str, Any]]:
    evidence_matches: list[dict[str, Any]] = []
    for term in _task_terms(task, selected_brain_id):
        evidence_matches.extend(query_evidence_registry(term).get("matches", [])[:5])
    seen_paths: set[str] = set()
    out: list[dict[str, Any]] = []
    for match in evidence_matches:
        if not isinstance(match, dict):
            continue
        path = str(match.get("path", "") or "")
        if not path or path in seen_paths:
            continue
        seen_paths.add(path)
        out.append(
            {
                "id": match.get("id", ""),
                "path": path,
                "verdict": match.get("verdict", ""),
                "blockers": match.get("blockers", []),
                "dataset_ids": match.get("dataset_ids", []),
            }
        )
    return out


def _main_context() -> dict[str, Any]:
    bootstrap = resolve_bootstrap(None).to_dict()
    return {
        "brain_id": "workspace",
        "main_manifest": bootstrap.get("main_manifest", ""),
        "main_entrypoint": bootstrap.get("main_entrypoint", ""),
        "boot_order": bootstrap.get("boot_order", []),
        "git": _git_status(),
        "branch_policy": {
            "expected_branch": "main",
            "mutation_requires_main": True,
            "exception_requires_user_authorization": True,
        },
        "global_boundaries": [
            "main brain is the agent entrypoint",
            "child brains hold project facts only after routing",
            "generic skills yield to brain safety boundaries",
        ],
        "summary": _text_excerpt("brain/state_center.md"),
        "hard_rules": _text_excerpt("brain/knowledge_center.md"),
    }


def _child_context(selected_brain_id: str, task: str, workflow_id: str, study_tag: str) -> dict[str, Any]:
    bootstrap = resolve_bootstrap(selected_brain_id).to_dict()
    context: dict[str, Any] = {
        "brain_id": selected_brain_id,
        "child_manifest": bootstrap.get("child_manifest", ""),
        "child_entrypoint": bootstrap.get("child_entrypoint", ""),
        "boot_order": bootstrap.get("child_boot_order", []),
        "fast_handoff_paths": bootstrap.get("child_fast_handoff_paths", []),
        "write_routes": bootstrap.get("child_write_routes", {}),
        "related_references": _related_references(task, selected_brain_id),
    }
    if selected_brain_id == "daily_research":
        context.update(daily_research_adapter.child_context_additions())
    return context


def build_task_capsule(*, task: str = "", workflow: str = "brain_handoff", study_tag: str = "") -> dict[str, Any]:
    routing = route_task_to_brain(task)
    selected_brain_id = str(routing.get("selected_brain_id", "") or "")
    selection = select_workflow_for_task(task) if workflow == "auto" else {"selected_workflow": workflow, "reason": "explicit workflow requested"}
    workflow_id = str(selection.get("selected_workflow", "") or "brain_handoff")
    child_registry_id = selected_brain_id if selected_brain_id in child_brain_ids() else None
    registry = load_workflow_registry(child_registry_id)
    if workflow_id not in registry:
        workflow_id = "brain_handoff"
    workflow_state = build_workflow_state(workflow_id, study_tag=study_tag or None, child_brain=child_registry_id).to_dict()
    workflow_guide = build_workflow_guide(workflow_id, child_brain=child_registry_id)
    rules = run_brain_rules(has_explicit_study_tag=bool(study_tag))
    guards: dict[str, Any] = {
        "rule_report": rules,
        "validation_commands": [
            f"git diff -- {daily_research_adapter.ACTIVE_ARTIFACT.as_posix()}",
            "git diff --check",
            f"{PYTHON_EXECUTABLE} -m tools.brain.doc_guard check",
            f"{PYTHON_EXECUTABLE} -m tools.brain.integrity_check --json",
        ],
    }
    if selected_brain_id == "daily_research":
        guards.update(daily_research_adapter.capsule_guard_additions(rules))

    payload: dict[str, Any] = {
        "schema_version": 2,
        "task": task,
        "workflow": workflow_id,
        "workflow_selection": selection,
        "workflow_guide": workflow_guide,
        "required_checklist": workflow_guide.get("checklist", []),
        "stop_conditions": workflow_guide.get("stop_conditions", []),
        "study_tag": study_tag,
        "main_context": _main_context(),
        "routing": routing,
        "guards": guards,
        "forbidden_actions": workflow_state.get("registry_entry", {}).get("forbidden_actions", []),
        "next_allowed_actions": workflow_state.get("next_allowed_actions", []),
        "assumptions": [
            "Work remains on main unless the user explicitly changes branch policy",
            "No child brain is loaded until main-brain routing selects one",
        ],
    }
    if selected_brain_id in child_brain_ids() and routing.get("status") == "selected":
        payload["child_context"] = _child_context(selected_brain_id, task, workflow_id, study_tag)
    return payload
