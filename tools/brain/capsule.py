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
from tools.brain.runtime_context import (
    compact_child_context,
    compact_main_context,
    deep_dive_commands,
    normalize_verbosity,
    summarize_frontier,
    summary_budget,
)


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
                "tags": match.get("tags", []),
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


def _skill_sync_status() -> dict[str, Any]:
    try:
        from tools.brain.skill_install import build_sync_plan

        plan = build_sync_plan()
        return {
            "status": "ok",
            "all_in_sync": bool(plan.get("all_in_sync")),
            "skills": plan.get("skills", []),
        }
    except Exception as exc:
        return {"status": "error", "error": str(exc), "all_in_sync": False, "skills": []}


def _preflight_blockers(*, routing: dict[str, Any], main_context: dict[str, Any], intent: str, guards: dict[str, Any]) -> list[str]:
    blockers: list[str] = []
    normalized_intent = str(intent or "read")
    if normalized_intent in {"mutate", "long_task", "writeback"} and not bool(main_context.get("git", {}).get("on_main")):
        blockers.append("not_on_main_for_mutation")
    if routing.get("status") == "ambiguous":
        blockers.append("ambiguous_routing")
    if str(routing.get("selected_brain_id", "") or "") and routing.get("status") != "selected":
        blockers.append("missing_child_brain")
    active_guard_status = str(guards.get("active_artifact_guard", {}).get("status", "") or "")
    if active_guard_status and active_guard_status != "clean":
        blockers.append("active_artifact_dirty")
    skill_status = _skill_sync_status()
    if not bool(skill_status.get("all_in_sync")):
        blockers.append("stale_or_missing_global_skill")
    return sorted(set(blockers))


def build_task_capsule(
    *,
    task: str = "",
    workflow: str = "brain_handoff",
    study_tag: str = "",
    intent: str = "read",
    verbosity: str = "lite",
) -> dict[str, Any]:
    context_profile = normalize_verbosity(verbosity)
    routing = route_task_to_brain(task)
    selected_brain_id = str(routing.get("selected_brain_id", "") or "")
    if workflow == "auto" and str(intent or "") == "long_task":
        selection = {"task": task, "selected_workflow": "long_task", "reason": "long_task intent requested"}
    else:
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
        if "frontier_report" in guards:
            guards["frontier_report"] = summarize_frontier(guards["frontier_report"], profile=context_profile)

    raw_main_context = _main_context()
    main_context = compact_main_context(raw_main_context, profile=context_profile)
    skill_status = _skill_sync_status()
    preflight_blockers = _preflight_blockers(routing=routing, main_context=main_context, intent=intent, guards=guards)
    mutation_allowed = str(intent or "read") not in {"mutate", "long_task", "writeback"} or "not_on_main_for_mutation" not in preflight_blockers
    payload: dict[str, Any] = {
        "schema_version": 2,
        "task": task,
        "context_profile": context_profile,
        "summary_budget": summary_budget(context_profile),
        "available_deep_dive_commands": deep_dive_commands(selected_brain_id=selected_brain_id),
        "intent": str(intent or "read"),
        "mutation_allowed": bool(mutation_allowed),
        "preflight_blockers": preflight_blockers,
        "global_skill_sync": skill_status,
        "workflow": workflow_id,
        "workflow_selection": selection,
        "workflow_guide": workflow_guide,
        "required_checklist": workflow_guide.get("checklist", []),
        "stop_conditions": workflow_guide.get("stop_conditions", []),
        "study_tag": study_tag,
        "main_context": main_context,
        "routing": routing,
        "guards": guards,
        "forbidden_actions": workflow_state.get("registry_entry", {}).get("forbidden_actions", []),
        "next_allowed_actions": workflow_state.get("next_allowed_actions", []),
        "assumptions": [
            "Work remains on main unless the user explicitly changes branch policy",
            "No child brain is loaded until main-brain routing selects one",
        ],
    }
    long_task_contract = workflow_state.get("registry_entry", {}).get("long_task_contract", {})
    if str(intent or "") == "long_task" and isinstance(long_task_contract, dict) and long_task_contract:
        payload["long_task_contract"] = dict(long_task_contract)
    if selected_brain_id in child_brain_ids() and routing.get("status") == "selected":
        raw_child_context = _child_context(selected_brain_id, task, workflow_id, study_tag)
        payload["child_context"] = compact_child_context(raw_child_context, profile=context_profile)
    return payload
