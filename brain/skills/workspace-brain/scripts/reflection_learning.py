from __future__ import annotations

from typing import Any


TRACE_EVENT_TYPES = (
    "tool_failure",
    "blocker_detected",
    "manual_workaround",
    "user_nudge",
    "step_skipped",
    "verification_failed",
    "verification_passed",
    "completion_claim",
    "route_or_selector_mismatch",
    "skill_sync_gap",
)


def reflection_template() -> dict[str, Any]:
    return {
        "schema_version": 1,
        "task": "",
        "planned_steps": [
            {
                "id": "step_id",
                "title": "short planned action",
                "expected_outcome": "observable result",
                "required": True,
            }
        ],
        "events": [
            {
                "type": "tool_failure",
                "step_id": "step_id",
                "summary": "what happened",
                "evidence": "path, command output, or observation",
                "command": "",
                "returncode": 0,
                "artifact_path": "",
            }
        ],
        "final_state": {
            "completed": False,
            "skipped_steps": [],
            "unresolved_blockers": [],
            "user_nudges": [],
            "verification": [],
        },
    }


def _list_of_dicts(value: Any) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        return []
    return [item for item in value if isinstance(item, dict)]


def _string_list(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    return [str(item) for item in value if str(item or "").strip()]


def normalize_trace(raw: dict[str, Any]) -> dict[str, Any]:
    trace = raw if isinstance(raw, dict) else {}
    steps: list[dict[str, Any]] = []
    for item in _list_of_dicts(trace.get("planned_steps")):
        step_id = str(item.get("id", "") or "").strip()
        if not step_id:
            continue
        steps.append(
            {
                "id": step_id,
                "title": str(item.get("title", "") or ""),
                "expected_outcome": str(item.get("expected_outcome", "") or ""),
                "required": bool(item.get("required", False)),
            }
        )
    events: list[dict[str, Any]] = []
    for item in _list_of_dicts(trace.get("events")):
        event_type = str(item.get("type", "") or "").strip()
        if event_type not in TRACE_EVENT_TYPES:
            continue
        events.append(
            {
                "type": event_type,
                "step_id": str(item.get("step_id", "") or ""),
                "summary": str(item.get("summary", "") or ""),
                "evidence": str(item.get("evidence", "") or ""),
                "command": str(item.get("command", "") or ""),
                "returncode": item.get("returncode", ""),
                "artifact_path": str(item.get("artifact_path", "") or ""),
            }
        )
    final_state_raw = trace.get("final_state") if isinstance(trace.get("final_state"), dict) else {}
    return {
        "schema_version": 1,
        "task": str(trace.get("task", "") or ""),
        "planned_steps": steps,
        "events": events,
        "final_state": {
            "completed": bool(final_state_raw.get("completed", False)),
            "skipped_steps": _string_list(final_state_raw.get("skipped_steps")),
            "unresolved_blockers": _string_list(final_state_raw.get("unresolved_blockers")),
            "user_nudges": _string_list(final_state_raw.get("user_nudges")),
            "verification": _list_of_dicts(final_state_raw.get("verification")),
        },
    }


def _candidate(
    *,
    lesson: str,
    root_cause: str,
    target_layer: str,
    recommended_change: str,
    supporting_events: list[dict[str, Any]],
    suggested_tests: list[str],
    anti_overfit_check: str,
    confidence: str,
) -> dict[str, Any]:
    return {
        "lesson": lesson,
        "root_cause": root_cause,
        "target_layer": target_layer,
        "recommended_change": recommended_change,
        "suggested_tests": suggested_tests,
        "anti_overfit_check": anti_overfit_check,
        "confidence": confidence,
        "supporting_events": supporting_events,
        "requires_user_confirmation": True,
    }


def _events_of_type(events: list[dict[str, Any]], event_type: str) -> list[dict[str, Any]]:
    return [event for event in events if event.get("type") == event_type]


def _event_text(event: dict[str, Any]) -> str:
    return " ".join(str(event.get(key, "") or "") for key in ("summary", "evidence", "command")).lower()


def _is_one_off_environment_failure(events: list[dict[str, Any]], final_state: dict[str, Any]) -> bool:
    failure_events = _events_of_type(events, "tool_failure")
    if not failure_events:
        return False
    if final_state.get("completed"):
        return False
    if _events_of_type(events, "manual_workaround") or _events_of_type(events, "user_nudge"):
        return False
    text = "\n".join(_event_text(event) for event in failure_events)
    return "one_off_environment_failure" in text or "network hiccup" in text or "transient" in text


def analyze_trace(raw_trace: dict[str, Any]) -> dict[str, Any]:
    trace = normalize_trace(raw_trace)
    steps = trace["planned_steps"]
    events = trace["events"]
    final_state = trace["final_state"]
    candidates: list[dict[str, Any]] = []

    if _is_one_off_environment_failure(events, final_state):
        return {
            "status": "ok",
            "analysis_mode": "structured_trace",
            "reflection_review_required": True,
            "evidence_gaps": [],
            "learning_candidates": [],
            "next_actions": ["no_persistent_learning"],
        }

    required_step_ids = {step["id"] for step in steps if step.get("required")}
    skipped_step_ids = set(final_state.get("skipped_steps", []))
    skipped_step_ids.update(event.get("step_id", "") for event in _events_of_type(events, "step_skipped"))
    missed_required = sorted(step_id for step_id in required_step_ids if step_id in skipped_step_ids)
    if final_state.get("completed") and missed_required:
        support = [
            event
            for event in events
            if event.get("step_id") in missed_required and event.get("type") in {"step_skipped", "user_nudge", "completion_claim"}
        ]
        if not support:
            support = [{"type": "step_skipped", "step_id": step_id, "summary": "required planned step skipped"} for step_id in missed_required]
        candidates.append(
            _candidate(
                lesson="Required planned steps must be launched, verified, or explicitly reported as blocked before completion.",
                root_cause="completion was claimed while required planned steps were skipped",
                target_layer="execution_completion_gate",
                recommended_change="tighten the runtime completion checklist to compare required planned steps against final state before final response",
                supporting_events=support,
                suggested_tests=["review trace with skipped required training step returns execution_completion_gate"],
                anti_overfit_check="only applies when a required planned step is skipped or contradicted by user nudge, not to optional exploratory steps",
                confidence="high",
            )
        )

    tool_failures = _events_of_type(events, "tool_failure")
    manual_workarounds = _events_of_type(events, "manual_workaround")
    if tool_failures and manual_workarounds:
        candidates.append(
            _candidate(
                lesson="A tool failure followed by a manual workaround is a tool contract gap worth making explicit.",
                root_cause="the planned tool path could not complete and the agent bypassed it manually",
                target_layer="tool_contract_gap",
                recommended_change="add a regression test or contract check for the failed tool path before relying on manual workaround again",
                supporting_events=[*tool_failures, *manual_workarounds],
                suggested_tests=["review trace with tool_failure followed by manual_workaround returns tool_contract_gap"],
                anti_overfit_check="do not apply when the manual action was the planned primary path or the failure was a one-off environment issue",
                confidence="high",
            )
        )

    if _events_of_type(events, "route_or_selector_mismatch"):
        candidates.append(
            _candidate(
                lesson="Routing and workflow mismatches should be learned from structured capsule evidence.",
                root_cause="selected workflow or brain route did not match the task evidence",
                target_layer="workflow_selector",
                recommended_change="tighten brain-native selector rules while keeping generic method terms with local skills",
                supporting_events=_events_of_type(events, "route_or_selector_mismatch"),
                suggested_tests=["review trace with route_or_selector_mismatch returns workflow_selector"],
                anti_overfit_check="only apply when trace contains explicit route or selector mismatch evidence",
                confidence="high",
            )
        )

    if _events_of_type(events, "skill_sync_gap"):
        candidates.append(
            _candidate(
                lesson="Skill sync gaps need a runtime learning proposal when they affect behavior.",
                root_cause="canonical and installed workspace skill were out of sync during execution",
                target_layer="skill",
                recommended_change="make skill sync visible in capsule and final verification when canonical skill changes",
                supporting_events=_events_of_type(events, "skill_sync_gap"),
                suggested_tests=["review trace with skill_sync_gap returns skill target layer"],
                anti_overfit_check="only apply when the sync gap is observed in execution, not from unrelated installed skill checks",
                confidence="medium",
            )
        )

    if _events_of_type(events, "verification_failed") and final_state.get("completed"):
        candidates.append(
            _candidate(
                lesson="Completion after failed verification needs a tests guard.",
                root_cause="final state was completed despite a failed verification event",
                target_layer="tests_guard",
                recommended_change="require failed verification to be resolved or explicitly marked as blocked before completion",
                supporting_events=_events_of_type(events, "verification_failed"),
                suggested_tests=["review trace with verification_failed and completed final state returns tests_guard"],
                anti_overfit_check="do not apply to failures that were followed by a passing verification event for the same check",
                confidence="medium",
            )
        )

    unique: list[dict[str, Any]] = []
    seen: set[tuple[str, str]] = set()
    for candidate in candidates:
        key = (str(candidate["target_layer"]), str(candidate["lesson"]))
        if key in seen:
            continue
        seen.add(key)
        unique.append(candidate)

    return {
        "status": "ok",
        "analysis_mode": "structured_trace",
        "reflection_review_required": True,
        "evidence_gaps": [],
        "learning_candidates": unique,
        "next_actions": ["create_runtime_learning_proposal"] if unique else ["no_runtime_learning_needed"],
    }


def analyze_freeform(*, task: str = "", observation: str = "", test_output: str = "") -> dict[str, Any]:
    text = "\n".join([str(task or ""), str(observation or ""), str(test_output or "")]).lower()
    evidence = observation or task or test_output
    candidates: list[dict[str, Any]] = []

    def add(
        *,
        lesson: str,
        root_cause: str,
        target_layer: str,
        recommended_change: str,
        suggested_tests: list[str],
    ) -> None:
        candidates.append(
            _candidate(
                lesson=lesson,
                root_cause=root_cause,
                target_layer=target_layer,
                recommended_change=recommended_change,
                supporting_events=[{"type": "freeform_observation", "step_id": "", "summary": evidence}],
                suggested_tests=suggested_tests,
                anti_overfit_check="confirm with a structured trace before changing hot-path behavior",
                confidence="low",
            )
        )

    if any(term in text for term in ("明明写", "为什么还是", "没有执行", "又错", "重复", "用户指出")):
        add(
            lesson="User correction indicates the executable hot path may be missing the intended runtime behavior.",
            root_cause="freeform observation reported a repeated failure or user correction",
            target_layer="capsule_contract",
            recommended_change="collect a structured trace and add the missing behavior to the capsule, skill, or test guard",
            suggested_tests=["freeform user correction returns low-confidence runtime learning candidate"],
        )
    if any(term in text for term in ("writeback-plan", "path_policy/studies", "continuous_policy/studies", "证据域", "手工绕过", "错路由")):
        add(
            lesson="Evidence resolver domain mismatch should be captured as a tool contract gap.",
            root_cause="freeform observation reported evidence domain mismatch or manual workaround",
            target_layer="study_evidence_resolver",
            recommended_change="confirm with trace and cover path_policy plus continuous_policy study tags with regression tests",
            suggested_tests=["freeform evidence domain mismatch returns study_evidence_resolver candidate"],
        )
    if any(term in text for term in ("start-sleep", "wait-process", "长任务", "轮询", "eta")):
        add(
            lesson="Long-task polling discipline should be verified with structured execution evidence.",
            root_cause="freeform observation reported long-task monitor or ETA behavior gap",
            target_layer="tests_guard",
            recommended_change="collect a trace for long-task polling and guard PID/log/progress/ETA reporting",
            suggested_tests=["freeform long-task observation returns tests_guard candidate"],
        )
    if any(term in text for term in ("计划包含训练但 agent 结束任务", "需要用户提示继续实施计划", "训练没启动", "long task step skipped", "training step skipped", "ended before training")):
        add(
            lesson="Planned long-running steps should be checked against final completion state.",
            root_cause="freeform observation reported completion before a planned training or long-running step",
            target_layer="execution_completion_gate",
            recommended_change="use structured trace to compare required planned steps with final state",
            suggested_tests=["trace skipped required training step returns execution_completion_gate"],
        )
    if any(term in text for term in ("selected brain_handoff for a brain rule mutation", "brain rule mutation", "workflow selector", "workflow_selector", "workflow mismatch", "wrong workflow", "selected wrong workflow", "capsule workflow mismatch", "completion_review=false", "completion_review_required=false", "路由误判", "工作流误判", "选错 workflow", "脑区规则被 handoff", "selector 漏判", "规则修改没有触发")):
        add(
            lesson="Brain-native selector mismatches should be confirmed from capsule trace evidence.",
            root_cause="freeform observation reported workflow or route mismatch",
            target_layer="workflow_selector",
            recommended_change="tighten brain-native workflow selector rules while keeping generic method terms owned by local skills",
            suggested_tests=["freeform workflow mismatch returns workflow_selector candidate"],
        )
    if any(term in text for term in ("stale skill", "global skill", "stale_or_missing_global_skill", "skill sync", "skill_install", "skill out of sync", "out of sync", "not in sync", "技能不同步", "skill 不同步", "未同步", "安装版落后", "同步失败")):
        add(
            lesson="Skill sync gaps should be learned only when they affect execution behavior.",
            root_cause="freeform observation reported a canonical or installed skill sync gap",
            target_layer="skill",
            recommended_change="confirm with a trace and sync canonical workspace-brain skill after edits",
            suggested_tests=["freeform skill sync observation returns skill candidate"],
        )

    unique: list[dict[str, Any]] = []
    seen: set[tuple[str, str]] = set()
    for candidate in candidates:
        key = (str(candidate["target_layer"]), str(candidate["lesson"]))
        if key in seen:
            continue
        seen.add(key)
        unique.append(candidate)

    return {
        "status": "ok",
        "analysis_mode": "freeform_fallback",
        "reflection_review_required": True,
        "evidence_gaps": ["structured_trace_missing"],
        "learning_candidates": unique,
        "next_actions": ["create_runtime_learning_proposal"] if unique else ["no_runtime_learning_needed"],
    }


def build_proposal_payload(
    *,
    proposal_id: str,
    title: str,
    created_at: str,
    status: str,
    severity: str,
    owner_brain: str,
    writeback_target: str,
    target_layer: str,
    related_task: str,
    trigger: str,
    evidence: str,
    recommendation: str,
    requires_user_confirmation: bool,
    suggested_tests: list[str],
    lesson: str = "",
    root_cause: str = "",
    supporting_events: list[dict[str, Any]] | None = None,
    anti_overfit_check: str = "",
) -> dict[str, Any]:
    return {
        "schema_version": 2,
        "proposal_id": proposal_id,
        "title": title,
        "created_at": created_at,
        "authority": "requires_user_confirmation",
        "status": status,
        "severity": severity,
        "owner_brain": owner_brain,
        "writeback_target": writeback_target,
        "target_layer": target_layer,
        "related_task": related_task,
        "requires_user_confirmation": bool(requires_user_confirmation),
        "facts": [trigger],
        "inferences": [],
        "assumptions": [],
        "trigger_evidence": [evidence],
        "recommendation": recommendation,
        "lesson": lesson or recommendation or trigger,
        "root_cause": root_cause or "runtime learning proposal was created from reported execution evidence",
        "supporting_events": list(supporting_events or [{"type": "proposal_input", "step_id": "", "summary": evidence}]),
        "anti_overfit_check": anti_overfit_check or "confirm the pattern with execution evidence before changing hot-path behavior",
        "suggested_write_routes": [writeback_target, "brain/knowledge_center.md"],
        "suggested_tests": list(suggested_tests or []),
        "risks": ["Core brain or skill changes require explicit confirmation."],
    }
