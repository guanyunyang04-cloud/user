from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Any

from tools.brain.adapters import daily_research as daily_research_adapter
from tools.brain.evidence_registry import query_evidence_registry
from tools.brain.agent_meta import (
    AGENT_META_IMPLEMENTATION_APPROVAL_POLICY,
    AGENT_META_PROPOSAL_CREATION_POLICY,
    AGENT_META_REQUIRED_PASSES,
    analyze_agent_meta_signals,
)
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
from tools.brain.project_profiles import load_project_profile
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
            "local skills own general methods; brain supplies project facts, routing, guards, evidence, and writeback routes",
        ],
        "summary": _text_excerpt("brain/state_center.md"),
        "hard_rules": _text_excerpt("brain/knowledge_center.md"),
    }


def _child_context(selected_brain_id: str, task: str, workflow_id: str, run_tag: str) -> dict[str, Any]:
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


def _workspace_context(main_context: dict[str, Any]) -> dict[str, Any]:
    return {
        "brain_id": "workspace",
        "workspace_manifest": main_context.get("main_manifest", ""),
        "workspace_entrypoint": main_context.get("main_entrypoint", ""),
        "write_routes": {
            "workspace_state": "brain/state_center.md",
            "workspace_operations": "brain/operations_center.md",
            "workspace_governance": "brain/governance_layer.md",
        },
    }


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
    if normalized_intent in {"mutate", "writeback"} and not bool(main_context.get("git", {}).get("on_main")):
        blockers.append("not_on_main_for_mutation")
    if routing.get("status") == "ambiguous":
        blockers.append("ambiguous_routing")
    target = routing.get("target", {})
    if routing.get("status") == "selected":
        target_kind = str(target.get("kind", "") if isinstance(target, dict) else "")
        target_id = str(target.get("id", "") if isinstance(target, dict) else "")
        if target_kind not in {"workspace", "child"} or not target_id:
            blockers.append("unbootstrapable_route_target")
        else:
            try:
                resolve_bootstrap(target_id)
            except Exception:
                blockers.append("unbootstrapable_route_target")
    if str(routing.get("selected_brain_id", "") or "") and routing.get("status") != "selected":
        blockers.append("missing_child_brain")
    active_guard_status = str(guards.get("active_artifact_guard", {}).get("status", "") or "")
    if active_guard_status and active_guard_status != "clean":
        blockers.append("active_artifact_dirty")
    skill_status = _skill_sync_status()
    if not bool(skill_status.get("all_in_sync")):
        blockers.append("stale_or_missing_global_skill")
    return sorted(set(blockers))


def _dedupe_text(items: list[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for item in items:
        text = str(item or "").strip()
        if not text or text in seen:
            continue
        out.append(text)
        seen.add(text)
    return out


def _task_has_long_task_signal(task: str) -> bool:
    lower = str(task or "").lower()
    signals = (
        "长训练",
        "长任务",
        "训练轮询",
        "启动训练",
        "long training",
        "long-running",
        "long running",
        "long task",
        "training",
        "wait-process",
        "eta",
        "pid",
    )
    return any(signal in lower for signal in signals)


def _agent_meta_commands() -> dict[str, str]:
    return {
        "agent_meta_audit": f"{PYTHON_EXECUTABLE} brain/skills/workspace-brain/scripts/brain_runtime.py agent-meta-audit --cwd . --mode compact",
        "reflection_template": f"{PYTHON_EXECUTABLE} brain/skills/workspace-brain/scripts/brain_runtime.py reflection-template --json",
        "review_trace": f'{PYTHON_EXECUTABLE} brain/skills/workspace-brain/scripts/brain_runtime.py review --cwd . --trace-json "<trace.json>" --json',
        "proposal": (
            f"{PYTHON_EXECUTABLE} brain/skills/workspace-brain/scripts/brain_runtime.py "
            'proposal --cwd . --title "<short title>" --trigger "<fact>" '
            '--evidence "<path or observation>" --recommendation "<change proposal>" '
            "--severity info --owner-brain workspace --writeback-target brain/references/"
        ),
    }


def _build_agent_review(*, workflow_id: str, agent_meta_review: dict[str, Any]) -> dict[str, Any]:
    reason_codes: list[str] = []
    if workflow_id in {
        "brain_maintenance",
        "brain_architecture_refactor",
        "brain_writeback_verified",
    }:
        reason_codes.append("workflow_completion_review")
    if agent_meta_review.get("status") != "clear":
        reason_codes.append("agent_meta_opportunity")
        reason_codes.append("closure_meta_review")
    commands = _agent_meta_commands()
    closure_meta_review_required = bool(reason_codes)
    return {
        "before_final_required": bool(reason_codes),
        "reason_codes": _dedupe_text(reason_codes),
        "trace_template_command": commands["reflection_template"],
        "review_command": commands["review_trace"],
        "proposal_command": commands["proposal"],
        "closure_meta_review_required": closure_meta_review_required,
        "closure_review_command": commands["review_trace"],
    }


def build_task_capsule(
    *,
    task: str = "",
    workflow: str = "brain_handoff",
    run_tag: str = "",
    intent: str = "read",
    verbosity: str = "lite",
) -> dict[str, Any]:
    context_profile = normalize_verbosity(verbosity)
    routing = route_task_to_brain(task)
    target = routing.get("target", {}) if isinstance(routing.get("target", {}), dict) else {}
    target_id = str(target.get("id", "") or str(routing.get("selected_brain_id", "") or ""))
    target_kind = str(target.get("kind", "") or ("child" if target_id in child_brain_ids() else "workspace"))
    workflow_domain = str(target.get("domain", "") or ("workspace_governance" if target_kind == "workspace" else target_id))
    selected_brain_id = target_id if target_kind == "child" else ""
    project_profile = load_project_profile(selected_brain_id if selected_brain_id else "workspace")
    guard_profile = project_profile.get("guard_profile", {}) if isinstance(project_profile.get("guard_profile"), dict) else {}
    verification_profile = (
        project_profile.get("verification_profile", {}) if isinstance(project_profile.get("verification_profile"), dict) else {}
    )
    selection = (
        select_workflow_for_task(task, intent=str(intent or "read"))
        if workflow == "auto"
        else {"selected_workflow": workflow, "reason": "explicit workflow requested"}
    )
    workflow_id = str(selection.get("selected_workflow", "") or "brain_handoff")
    child_registry_id = selected_brain_id if target_kind == "child" and selected_brain_id in child_brain_ids() else None
    registry = load_workflow_registry(child_registry_id)
    if workflow_id not in registry:
        workflow_id = "brain_handoff"
    workflow_state = build_workflow_state(workflow_id, run_tag=run_tag or None, child_brain=child_registry_id).to_dict()
    workflow_guide = build_workflow_guide(workflow_id, child_brain=child_registry_id)
    capability_hints = list(workflow_guide.get("capability_hints", []) or [])
    risk_signals = list(workflow_guide.get("risk_signals", []) or [])
    verification_hints = list(workflow_guide.get("verification_hints", []) or [])
    if _task_has_long_task_signal(task):
        capability_hints.append(
            f"agent_run: use {PYTHON_EXECUTABLE} -m tools.brain.agent_run paths/register/launch/status with --project-id and --run-id so PID/log/progress stay in the selected project namespace; long_task_monitor status/wait-once also requires project/run identity and an explicit lease for cross-project reads"
        )
        risk_signals.append("long_task_without_pid_log_progress_or_eta")
        verification_hints.append("for long jobs, report project_id, run_id, PID status, elapsed time, progress, ETA, log tail, artifact mtime, and next decision after each wait window")
    rules = (
        run_brain_rules(has_explicit_run_tag=bool(run_tag))
        if bool(guard_profile.get("brain_rules"))
        else {"status": "ok", "error_count": 0, "warning_count": 0, "findings": []}
    )
    guards: dict[str, Any] = {
        "rule_report": rules,
        "validation_commands": list(verification_profile.get("always_commands", []) or []),
    }
    if selected_brain_id == "daily_research" and bool(guard_profile.get("active_artifact_guard")):
        guards.update(daily_research_adapter.capsule_guard_additions(rules))
        if "frontier_report" in guards and bool(guard_profile.get("frontier_report")):
            guards["frontier_report"] = summarize_frontier(guards["frontier_report"], profile=context_profile)
        elif "frontier_report" in guards:
            guards.pop("frontier_report", None)

    raw_main_context = _main_context()
    main_context = compact_main_context(raw_main_context, profile=context_profile)
    skill_status = _skill_sync_status()
    preflight_blockers = _preflight_blockers(routing=routing, main_context=main_context, intent=intent, guards=guards)
    mutation_allowed = str(intent or "read") not in {"mutate", "writeback"} or "not_on_main_for_mutation" not in preflight_blockers
    risk_signals.extend(preflight_blockers)
    for finding in guards.get("rule_report", {}).get("findings", []):
        if isinstance(finding, dict) and str(finding.get("severity", "") or "") == "warning":
            risk_signals.append(str(finding.get("code", "") or ""))
    payload: dict[str, Any] = {
        "schema_version": 4,
        "task": task,
        "context_profile": context_profile,
        "summary_budget": summary_budget(context_profile),
        "available_deep_dive_commands": deep_dive_commands(
            selected_brain_id=target_id,
            target_kind=target_kind,
            candidate_brain_ids=[
                str(item.get("brain_id", "") or "")
                for item in routing.get("candidate_summary", [])
                if isinstance(item, dict) and str(item.get("brain_id", "") or "") in child_brain_ids()
            ],
        ),
        "intent": str(intent or "read"),
        "mutation_allowed": bool(mutation_allowed),
        "preflight_blockers": preflight_blockers,
        "target_kind": target_kind,
        "workflow_domain": workflow_domain,
        "global_skill_sync": skill_status,
        "workflow": workflow_id,
        "workflow_selection": selection,
        "workflow_guide": workflow_guide,
        "required_checklist": workflow_guide.get("checklist", []),
        "capability_hints": _dedupe_text(capability_hints),
        "risk_signals": _dedupe_text(risk_signals),
        "verification_hints": _dedupe_text(verification_hints),
        "stop_conditions": workflow_guide.get("stop_conditions", []),
        "run_tag": run_tag,
        "main_context": main_context,
        "routing": routing,
        "agent_selected_brain_id": selected_brain_id if routing.get("status") == "selected" else "",
        "selection_reason": routing.get("reason", ""),
        "routing_evidence": {
            "status": routing.get("status", ""),
            "confidence": routing.get("confidence", ""),
            "decision_required": bool(routing.get("decision_required")),
            "candidate_summary": routing.get("candidate_summary", []),
            "workspace_governance_signal": routing.get("workspace_governance_signal", {}),
        },
        "project_profile": project_profile,
        "guards": guards,
        "next_allowed_actions": workflow_state.get("next_allowed_actions", []),
        "assumptions": [
            "Work remains on main unless the user explicitly changes branch policy",
            "No child brain is loaded until main-brain routing selects one",
        ],
    }
    if target_kind == "workspace" and routing.get("status") == "selected":
        payload["workspace_context"] = _workspace_context(main_context)
    if target_kind == "child" and selected_brain_id in child_brain_ids() and routing.get("status") == "selected":
        raw_child_context = _child_context(selected_brain_id, task, workflow_id, run_tag)
        payload["child_context"] = compact_child_context(raw_child_context, profile=context_profile)
    agent_meta_review = analyze_agent_meta_signals(
        task=task,
        capsule_context={
            "target_kind": target_kind,
            "workflow_domain": workflow_domain,
            "workflow": workflow_id,
            "routing": routing,
            "guards": guards,
            "preflight_blockers": preflight_blockers,
        },
    )
    payload["agent_meta"] = {
        "actor": "agent",
        "substrate": "brain",
        "tool_role": "sensor",
        "authority": "propose_only",
        "proposal_creation_policy": AGENT_META_PROPOSAL_CREATION_POLICY,
        "implementation_approval_policy": AGENT_META_IMPLEMENTATION_APPROVAL_POLICY,
        "required_passes": list(AGENT_META_REQUIRED_PASSES),
        "review": agent_meta_review,
        "commands": _agent_meta_commands(),
    }
    payload["agent_review"] = _build_agent_review(workflow_id=workflow_id, agent_meta_review=agent_meta_review)
    return payload
