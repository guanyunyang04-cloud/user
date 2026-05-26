from __future__ import annotations

from typing import Any, Mapping


AGENT_META_AUTHORITY = "propose_only"
AGENT_META_CLEAR_ACTION = "no_learning_needed"
AGENT_META_REQUIRED_PASSES = ["task_start", "decision_boundary", "before_final"]


def clear_agent_meta_review() -> dict[str, Any]:
    return {
        "status": "clear",
        "authority": AGENT_META_AUTHORITY,
        "signals": [],
        "learning_opportunities": [],
        "next_actions": [AGENT_META_CLEAR_ACTION],
    }


def _text_blob(*values: Any) -> str:
    parts: list[str] = []
    for value in values:
        if isinstance(value, str):
            parts.append(value)
        elif isinstance(value, Mapping):
            parts.extend(str(item) for item in value.values())
        elif isinstance(value, list):
            parts.extend(str(item) for item in value)
        elif value is not None:
            parts.append(str(value))
    return "\n".join(parts).lower()


def _dedupe_strings(items: list[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for item in items:
        text = str(item or "").strip()
        if not text or text in seen:
            continue
        seen.add(text)
        out.append(text)
    return out


def _dedupe_opportunities(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    seen: set[tuple[str, str, str]] = set()
    out: list[dict[str, Any]] = []
    for item in items:
        key = (
            str(item.get("owner_brain", "") or ""),
            str(item.get("target_layer", "") or ""),
            str(item.get("source_signal", "") or item.get("recommended_action", "") or ""),
        )
        if key in seen:
            continue
        seen.add(key)
        out.append(item)
    return out


def _opportunity(
    *,
    target_layer: str,
    owner_brain: str,
    confidence: str,
    recommended_action: str,
    writeback_route: str,
    source_signal: str,
    detector_id: str,
    verification_required: bool = True,
) -> dict[str, Any]:
    return {
        "target_layer": target_layer,
        "owner_brain": owner_brain,
        "confidence": confidence,
        "recommended_action": recommended_action,
        "writeback_route": writeback_route,
        "verification_required": bool(verification_required),
        "detector_id": detector_id,
        "source_signal": source_signal,
    }


def _events(trace: Mapping[str, Any] | None) -> list[dict[str, Any]]:
    if not isinstance(trace, Mapping):
        return []
    events = trace.get("events", [])
    if not isinstance(events, list):
        return []
    return [event for event in events if isinstance(event, dict)]


def _event_opportunity(event: Mapping[str, Any]) -> dict[str, Any] | None:
    event_type = str(event.get("type", "") or "").strip()
    owner_brain = str(event.get("owner_brain", "") or "workspace")
    target_layer = str(event.get("target_layer", "") or "")
    if event_type == "budget_reliability_gap":
        return _opportunity(
            target_layer=target_layer or "experiment_governance",
            owner_brain=owner_brain or "daily_research",
            confidence=str(event.get("confidence", "") or "high"),
            recommended_action="downgrade low-budget training evidence to smoke_only/scout_only before using it for model-quality conclusions",
            writeback_route="daily_research/brain/knowledge_center.md",
            source_signal=event_type,
            detector_id="trace_event_detector",
        )
    if event_type == "evidence_quality_gap":
        return _opportunity(
            target_layer=target_layer or "evidence_governance",
            owner_brain=owner_brain,
            confidence=str(event.get("confidence", "") or "high"),
            recommended_action="route the evidence quality gap to the owning brain docs, guard, or tests before reusing the conclusion",
            writeback_route="brain/governance_layer.md" if owner_brain == "workspace" else f"{owner_brain}/brain/knowledge_center.md",
            source_signal=event_type,
            detector_id="trace_event_detector",
        )
    if event_type == "learning_opportunity_missed":
        return _opportunity(
            target_layer=target_layer or "agent_meta_protocol",
            owner_brain=owner_brain,
            confidence=str(event.get("confidence", "") or "high"),
            recommended_action="make the missed learning trigger part of the agent meta protocol and verify it with a capsule or review test",
            writeback_route="brain/governance_layer.md",
            source_signal=event_type,
            detector_id="trace_event_detector",
        )
    if event_type == "rule_not_enforced":
        return _opportunity(
            target_layer=target_layer or "capsule_contract",
            owner_brain=owner_brain,
            confidence=str(event.get("confidence", "") or "high"),
            recommended_action="add an executable hot-path contract so the documented rule affects future agent behavior",
            writeback_route="brain/governance_layer.md",
            source_signal=event_type,
            detector_id="trace_event_detector",
        )
    if event_type == "domain_guard_gap":
        return _opportunity(
            target_layer=target_layer or "domain_guard",
            owner_brain=owner_brain,
            confidence=str(event.get("confidence", "") or "high"),
            recommended_action="add a domain guard or regression test for the observed gap",
            writeback_route="brain/governance_layer.md" if owner_brain == "workspace" else f"{owner_brain}/brain/operations_center.md",
            source_signal=event_type,
            detector_id="trace_event_detector",
        )
    return None


def _merge_domain_payload(
    *,
    signals: list[str],
    opportunities: list[dict[str, Any]],
    actions: list[str],
    domain_payload: Mapping[str, Any],
) -> None:
    signals.extend(str(item) for item in domain_payload.get("signals", []) or [])
    opportunities.extend(item for item in domain_payload.get("learning_opportunities", []) or [] if isinstance(item, dict))
    actions.extend(str(item) for item in domain_payload.get("next_actions", []) or [])


def _daily_research_applies(task: str, capsule_context: Mapping[str, Any]) -> bool:
    text = _text_blob(task, capsule_context)
    if str(capsule_context.get("workflow_domain", "") or "") == "daily_research":
        return True
    routing = capsule_context.get("routing", {})
    if isinstance(routing, Mapping) and str(routing.get("selected_brain_id", "") or "") == "daily_research":
        return True
    terms = (
        "daily_research",
        "multi-horizon",
        "multi horizon",
        "path_policy",
        "path20",
        "training",
        "实验",
        "模型",
        "训练",
        "证据",
    )
    return any(term in text for term in terms)


def analyze_agent_meta_signals(
    *,
    task: str = "",
    capsule_context: Mapping[str, Any] | None = None,
    trace: Mapping[str, Any] | None = None,
    observations: Any = None,
) -> dict[str, Any]:
    context = capsule_context if isinstance(capsule_context, Mapping) else {}
    text = _text_blob(task, observations)
    signals: list[str] = []
    opportunities: list[dict[str, Any]] = []
    actions: list[str] = []

    if (
        "脑区只是载体" in text
        or ("没有思考能力" in text and "agent" in text)
        or ("agent" in text and "元能力" in text and ("脑区" in text or "brain" in text))
        or ("brain" in text and "substrate" in text and "agent" in text)
    ):
        signals.append("actor_boundary_mismatch")
        opportunities.append(
            _opportunity(
                target_layer="agent_meta_protocol",
                owner_brain="workspace",
                confidence="high",
                recommended_action="rewrite the hot-path contract so the agent owns meta capability while the brain persists, distributes, and verifies the protocol",
                writeback_route="brain/governance_layer.md",
                source_signal="actor_boundary_mismatch",
                detector_id="actor_boundary_detector",
            )
        )

    learning_terms = (
        "应该学会",
        "为什么没提示",
        "没提示",
        "以后都要",
        "自动发现",
        "自进化",
        "智能自进化",
        "should learn",
        "why didn't",
        "always in future",
    )
    if any(term in text for term in learning_terms):
        signals.extend(["user_correction", "learning_opportunity_missed"])
        opportunities.append(
            _opportunity(
                target_layer="agent_meta_protocol",
                owner_brain="workspace",
                confidence="medium",
                recommended_action="create an agent learning proposal so future agents run the required meta pass instead of waiting for a tool prompt",
                writeback_route="brain/governance_layer.md",
                source_signal="learning_opportunity_missed",
                detector_id="user_correction_detector",
            )
        )

    if any(term in text for term in ("规则写了", "明明写", "没有执行", "未进入热路径", "rule not enforced")):
        signals.append("rule_not_enforced")
        opportunities.append(
            _opportunity(
                target_layer="capsule_contract",
                owner_brain="workspace",
                confidence="medium",
                recommended_action="verify whether the written rule is represented in capsule, guard, workflow, skill, or tests",
                writeback_route="brain/governance_layer.md",
                source_signal="rule_not_enforced",
                detector_id="hot_path_rule_detector",
            )
        )

    for event in _events(trace):
        event_type = str(event.get("type", "") or "").strip()
        opportunity = _event_opportunity(event)
        if opportunity is None:
            continue
        if event_type == "budget_reliability_gap":
            signals.append("low_budget_evidence_pollution")
        else:
            signals.append(event_type)
        opportunities.append(opportunity)

    if _daily_research_applies(task, context):
        try:
            from tools.brain.adapters.daily_research import detect_low_budget_evidence

            domain_payload = detect_low_budget_evidence(task=task)
            if domain_payload.get("status") != "clear":
                _merge_domain_payload(
                    signals=signals,
                    opportunities=opportunities,
                    actions=actions,
                    domain_payload=domain_payload,
                )
        except Exception:
            pass

    signals = _dedupe_strings(signals)
    opportunities = _dedupe_opportunities(opportunities)
    actions = _dedupe_strings([*actions, *("create_proposal" for _ in opportunities)])
    if not signals and not opportunities:
        return clear_agent_meta_review()
    return {
        "status": "opportunity",
        "authority": AGENT_META_AUTHORITY,
        "signals": signals,
        "learning_opportunities": opportunities,
        "next_actions": actions or ["create_proposal"],
    }
