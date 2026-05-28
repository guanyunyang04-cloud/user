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
    "evidence_quality_gap",
    "budget_reliability_gap",
    "rule_not_enforced",
    "learning_opportunity_missed",
    "domain_guard_gap",
    "meta_question_missed",
    "human_feedback_overrode_tool_clear",
    "closure_boundary_misread",
    "research_scope_grade_mismatch",
    "long_task_poll",
)

TRACE_OPTIONAL_EVENT_FIELDS = (
    "owner_brain",
    "target_layer",
    "artifact_path",
    "evidence_grade",
    "confidence",
    "run_tag",
    "pid",
    "pid_alive",
    "child_pids",
    "poll_window_seconds",
    "progress_path",
    "stdout_path",
    "stderr_path",
    "artifact_dir",
    "artifact_mtime",
    "eta_status",
    "eta_at",
    "eta_no_eta_reason",
    "progress_percent",
    "decision",
    "final_verification",
)


def reflection_template() -> dict[str, Any]:
    return {
        "schema_version": 1,
        "task": "",
        "agent_meta_passes": [
            {
                "phase": "task_start",
                "status": "pending",
                "notes": "agent checks whether the task reveals a reusable rule, evidence gap, or actor-boundary issue",
            },
            {
                "phase": "decision_boundary",
                "status": "pending",
                "notes": "agent checks whether a major route, workaround, or conclusion needs a learning proposal",
            },
            {
                "phase": "before_final",
                "status": "pending",
                "notes": "agent checks whether final claims, skipped work, or verification gaps should update the brain-mediated protocol",
            },
        ],
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
        "closure_review": {
            "status": "pending",
            "notes": "optional closure-boundary review for meta-questions about frame, criteria, method, authority, evaluation, or learning salience",
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
        event = {
            "type": event_type,
            "step_id": str(item.get("step_id", "") or ""),
            "summary": str(item.get("summary", "") or ""),
            "evidence": str(item.get("evidence", "") or ""),
            "command": str(item.get("command", "") or ""),
            "returncode": item.get("returncode", ""),
        }
        for key in TRACE_OPTIONAL_EVENT_FIELDS:
            if key in item:
                event[key] = str(item.get(key, "") or "")
        if "artifact_path" not in event:
            event["artifact_path"] = str(item.get("artifact_path", "") or "")
        events.append(event)
    final_state_raw = trace.get("final_state") if isinstance(trace.get("final_state"), dict) else {}
    closure_review_raw = trace.get("closure_review") if isinstance(trace.get("closure_review"), dict) else {}
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
        "closure_review": {
            "status": str(closure_review_raw.get("status", "") or ""),
            "notes": str(closure_review_raw.get("notes", "") or ""),
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


def _meta_question_candidate(
    *,
    meta_layer: str,
    object_level_issue: str,
    meta_question: str,
    why_it_matters: str,
    confidence: str,
    supporting_events: list[dict[str, Any]],
) -> dict[str, Any]:
    return {
        "meta_layer": meta_layer,
        "object_level_issue": object_level_issue,
        "meta_question": meta_question,
        "why_it_matters": why_it_matters,
        "ask_user_for_evolution": True,
        "confidence": confidence,
        "supporting_events": supporting_events,
    }


def _events_of_type(events: list[dict[str, Any]], event_type: str) -> list[dict[str, Any]]:
    return [event for event in events if event.get("type") == event_type]


def _event_text(event: dict[str, Any]) -> str:
    return " ".join(str(event.get(key, "") or "") for key in ("summary", "evidence", "command")).lower()


def _meta_candidates_from_events(events: list[dict[str, Any]]) -> list[dict[str, Any]]:
    candidates: list[dict[str, Any]] = []

    for event in _events_of_type(events, "human_feedback_overrode_tool_clear"):
        summary = str(event.get("summary", "") or "tool/audit clear conflicted with human feedback")
        candidates.append(
            _meta_question_candidate(
                meta_layer="evaluation",
                object_level_issue=summary,
                meta_question="When a tool or audit says clear, what higher-level evidence can override that clear signal?",
                why_it_matters="工具 clear 只是传感器结论；若人类反馈指出闭合判断不成立，评价机制本身需要被重新审视。",
                confidence=str(event.get("confidence", "") or "high"),
                supporting_events=[event],
            )
        )

    for event in _events_of_type(events, "meta_question_missed"):
        summary = str(event.get("summary", "") or "agent corrected object-level behavior but missed the meta-question")
        candidates.append(
            _meta_question_candidate(
                meta_layer="learning_salience",
                object_level_issue=summary,
                meta_question="Did the correction reveal a reusable flaw in how the agent frames, evaluates, or closes tasks?",
                why_it_matters="这不是一次性错误；它暴露了可泛化的认知缺陷发现能力，需要在闭合前主动识别并征求是否沉淀。",
                confidence=str(event.get("confidence", "") or "high"),
                supporting_events=[event],
            )
        )

    for event in _events_of_type(events, "closure_boundary_misread"):
        summary = str(event.get("summary", "") or "agent misread the closure boundary")
        text = _event_text(event)
        target_layer = str(event.get("target_layer", "") or "").lower()
        if "authority" in target_layer or any(term in text for term in ("handoff", "explicit plan", "user", "用户", "clear", "old context", "旧上下文")):
            candidates.append(
                _meta_question_candidate(
                    meta_layer="authority",
                    object_level_issue=summary,
                    meta_question="Which instruction or evidence source should control the closure judgment when context sources conflict?",
                    why_it_matters="权威排序错误会跨任务复发：agent 可能持续把旧上下文、工具输出或保守推断放到用户明确意图之上。",
                    confidence=str(event.get("confidence", "") or "high"),
                    supporting_events=[event],
                )
            )
        elif any(term in text for term in ("criterion", "success", "完成标准", "标准")):
            candidates.append(
                _meta_question_candidate(
                    meta_layer="criterion",
                    object_level_issue=summary,
                    meta_question="What success criterion is being used to decide this task is complete, and does the user share it?",
                    why_it_matters="完成标准错位会让对象层工作看似完成，但实际没有回答用户要闭合的问题。",
                    confidence=str(event.get("confidence", "") or "medium"),
                    supporting_events=[event],
                )
            )
        else:
            candidates.append(
                _meta_question_candidate(
                    meta_layer="frame",
                    object_level_issue=summary,
                    meta_question="Is the agent closing the object-level task while missing a higher-level question about the task frame?",
                    why_it_matters="框架误读会让 agent 用局部修补替代对问题本身的重构判断。",
                    confidence=str(event.get("confidence", "") or "medium"),
                    supporting_events=[event],
                )
            )

    for event in _events_of_type(events, "research_scope_grade_mismatch"):
        summary = str(event.get("summary", "") or "research conclusion exceeded the supporting evidence scope or grade")
        candidates.append(
            _meta_question_candidate(
                meta_layer="criterion",
                object_level_issue=summary,
                meta_question="Does the evidence scope and grade actually support the conclusion level being claimed?",
                why_it_matters="研究结论的完成标准不是指标好看，而是 evidence scope/grade 足以支撑结论层级；否则会把诊断线索误写成证据级结论。",
                confidence=str(event.get("confidence", "") or "high"),
                supporting_events=[event],
            )
        )

    unique: list[dict[str, Any]] = []
    seen: set[tuple[str, str]] = set()
    for candidate in candidates:
        key = (str(candidate["meta_layer"]), str(candidate["object_level_issue"]))
        if key in seen:
            continue
        seen.add(key)
        unique.append(candidate)
    return unique


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
            "meta_question_candidates": [],
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
                lesson="Skill sync gaps need an agent learning proposal when they affect behavior.",
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

    if _events_of_type(events, "budget_reliability_gap"):
        support = _events_of_type(events, "budget_reliability_gap")
        candidates.append(
            _candidate(
                lesson="Low-budget training evidence must be downgraded before it is used as a model-quality conclusion.",
                root_cause="training evidence showed insufficient budget, boundary best epoch, max epochs reached, single seed, or missing convergence evidence",
                target_layer=str(support[0].get("target_layer", "") or "experiment_governance"),
                recommended_change="require smoke/scout/evidence/promotion-grade labels and validation convergence checks before model-quality conclusions",
                supporting_events=support,
                suggested_tests=["review trace with budget_reliability_gap returns experiment_governance"],
                anti_overfit_check="only applies when trace or domain evidence explicitly reports low training budget or reliability gaps",
                confidence=str(support[0].get("confidence", "") or "high"),
            )
        )

    if _events_of_type(events, "evidence_quality_gap"):
        support = _events_of_type(events, "evidence_quality_gap")
        candidates.append(
            _candidate(
                lesson="Evidence quality gaps should be routed to the owning brain before conclusions are reused.",
                root_cause="structured trace reported a conclusion backed by incomplete, stale, or weak evidence",
                target_layer=str(support[0].get("target_layer", "") or "evidence_governance"),
                recommended_change="add an evidence grade, writeback route, or verification check for this evidence class",
                supporting_events=support,
                suggested_tests=["review trace with evidence_quality_gap returns evidence governance candidate"],
                anti_overfit_check="do not apply to ordinary uncertainty unless the trace reports a concrete evidence quality gap",
                confidence=str(support[0].get("confidence", "") or "high"),
            )
        )

    if _events_of_type(events, "research_scope_grade_mismatch"):
        support = _events_of_type(events, "research_scope_grade_mismatch")
        candidates.append(
            _candidate(
                lesson="Research conclusions require scope-grade alignment before final wording.",
                root_cause="a research/model conclusion was at risk of exceeding the supporting evidence scope, grade, or gate status",
                target_layer=str(support[0].get("target_layer", "") or "research_conclusion_gate"),
                recommended_change=(
                    "before closing research conclusions, verify universe_scope, dataset/pool binding, seed count, budget class, "
                    "gate result, and diagnostic-vs-evidence-grade status; downgrade wording when the evidence does not support the claimed level"
                ),
                supporting_events=support,
                suggested_tests=["review trace with research_scope_grade_mismatch returns research_conclusion_gate"],
                anti_overfit_check="only applies to research/model verdicts with explicit scope, grade, gate, or conclusion-level mismatch evidence",
                confidence=str(support[0].get("confidence", "") or "high"),
            )
        )

    if _events_of_type(events, "rule_not_enforced"):
        support = _events_of_type(events, "rule_not_enforced")
        candidates.append(
            _candidate(
                lesson="Rules written in docs must become executable through capsule, workflow, guard, skill, or tests.",
                root_cause="structured trace reported a documented rule that did not affect the runtime hot path",
                target_layer=str(support[0].get("target_layer", "") or "capsule_contract"),
                recommended_change="route the rule to an executable hot-path contract and add a regression test",
                supporting_events=support,
                suggested_tests=["review trace with rule_not_enforced returns capsule contract candidate"],
                anti_overfit_check="only applies when a specific documented rule and missed runtime behavior are both observed",
                confidence=str(support[0].get("confidence", "") or "high"),
            )
        )

    if _events_of_type(events, "learning_opportunity_missed"):
        support = _events_of_type(events, "learning_opportunity_missed")
        candidates.append(
            _candidate(
                lesson="Missed learning opportunities should be surfaced by the agent meta protocol instead of relying on user reminders.",
                root_cause="structured trace reported a situation where the agent should have proposed learning but did not",
                target_layer=str(support[0].get("target_layer", "") or "agent_meta_protocol"),
                recommended_change="add or tune the agent meta protocol and proposal prompt for this class of missed opportunity",
                supporting_events=support,
                suggested_tests=["review trace with learning_opportunity_missed returns agent_meta_protocol candidate"],
                anti_overfit_check="only apply when a user correction or structured trace identifies a repeatable learning opportunity",
                confidence=str(support[0].get("confidence", "") or "high"),
            )
        )

    if _events_of_type(events, "domain_guard_gap"):
        support = _events_of_type(events, "domain_guard_gap")
        candidates.append(
            _candidate(
                lesson="Domain-specific guard gaps should be owned by the routed child brain.",
                root_cause="structured trace reported a missing domain guard or enforcement hook",
                target_layer=str(support[0].get("target_layer", "") or "domain_guard"),
                recommended_change="add a child-brain domain hook, guard, or regression test for the observed gap",
                supporting_events=support,
                suggested_tests=["review trace with domain_guard_gap returns domain guard candidate"],
                anti_overfit_check="only applies when the task is routed to a concrete domain and the gap is domain-specific",
                confidence=str(support[0].get("confidence", "") or "high"),
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

    meta_question_candidates = _meta_candidates_from_events(events)
    next_actions: list[str] = []
    if unique:
        next_actions.append("create_agent_learning_proposal")
    if meta_question_candidates:
        next_actions.append("ask_user_for_evolution")
    if not next_actions:
        next_actions.append("no_agent_learning_needed")

    return {
        "status": "ok",
        "analysis_mode": "structured_trace",
        "reflection_review_required": True,
        "evidence_gaps": [],
        "learning_candidates": unique,
        "meta_question_candidates": meta_question_candidates,
        "next_actions": next_actions,
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
            suggested_tests=["freeform user correction returns low-confidence agent learning candidate"],
        )
    if any(term in text for term in ("writeback-plan", "path_policy/studies", "continuous_policy/studies", "证据域", "手工绕过", "错路由")):
        add(
            lesson="Evidence resolver domain mismatch should be captured as a tool contract gap.",
            root_cause="freeform observation reported evidence domain mismatch or manual workaround",
            target_layer="run_evidence_resolver",
            recommended_change="confirm with trace and cover path_policy plus continuous_policy run tags with regression tests",
            suggested_tests=["freeform evidence domain mismatch returns run_evidence_resolver candidate"],
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

    meta_question_candidates: list[dict[str, Any]] = []
    if any(term in text for term in ("元问题", "元认知", "问题本身", "任务结束前", "工具 clear", "audit clear", "评价机制", "完成标准")):
        meta_question_candidates.append(
            _meta_question_candidate(
                meta_layer="learning_salience",
                object_level_issue=evidence,
                meta_question="Is this observation about the object-level task, or about how the agent framed, evaluated, or closed the task?",
                why_it_matters="用户把问题提升到元层时，agent 需要识别可泛化的认知缺陷，而不是只修正对象层回答。",
                confidence="low",
                supporting_events=[{"type": "freeform_observation", "step_id": "", "summary": evidence}],
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
        "analysis_mode": "freeform_fallback",
        "reflection_review_required": True,
        "evidence_gaps": ["structured_trace_missing"],
        "learning_candidates": unique,
        "meta_question_candidates": meta_question_candidates,
        "next_actions": (
            [*("create_agent_learning_proposal" for _ in unique[:1]), *("ask_user_for_evolution" for _ in meta_question_candidates[:1])]
            or ["no_agent_learning_needed"]
        ),
    }


def analyze_agent_meta_signals(
    *,
    task: str = "",
    capsule_context: dict[str, Any] | None = None,
    trace: dict[str, Any] | None = None,
    observations: Any = None,
) -> dict[str, Any]:
    from tools.brain.agent_meta import analyze_agent_meta_signals as _analyze_agent_meta_signals

    return _analyze_agent_meta_signals(
        task=task,
        capsule_context=capsule_context or {},
        trace=trace,
        observations=observations,
    )


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
        "schema_version": 3,
        "proposal_id": proposal_id,
        "title": title,
        "created_at": created_at,
        "authority": "requires_user_confirmation",
        "status": status,
        "lifecycle_status": status,
        "severity": severity,
        "owner_brain": owner_brain,
        "writeback_target": writeback_target,
        "writeback_route": writeback_target,
        "target_layer": target_layer,
        "verification_required": bool(requires_user_confirmation),
        "detector_id": "manual_proposal",
        "source_signal": "manual_agent_learning",
        "related_task": related_task,
        "requires_user_confirmation": bool(requires_user_confirmation),
        "facts": [trigger],
        "inferences": [],
        "assumptions": [],
        "trigger_evidence": [evidence],
        "recommendation": recommendation,
        "lesson": lesson or recommendation or trigger,
        "root_cause": root_cause or "agent learning proposal was created from reported execution evidence",
        "supporting_events": list(supporting_events or [{"type": "proposal_input", "step_id": "", "summary": evidence}]),
        "anti_overfit_check": anti_overfit_check or "confirm the pattern with execution evidence before changing hot-path behavior",
        "suggested_write_routes": [writeback_target, "brain/knowledge_center.md"],
        "suggested_tests": list(suggested_tests or []),
        "risks": ["Core brain or skill changes require explicit confirmation."],
    }
