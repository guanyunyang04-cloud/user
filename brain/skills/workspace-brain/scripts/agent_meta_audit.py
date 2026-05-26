from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from typing import Any

from agent_learning import find_brain_root, load_learning_index, proposal_status_counts


def _parse_json_stdout(result: subprocess.CompletedProcess[str]) -> dict[str, Any]:
    try:
        payload = json.loads(result.stdout or "")
    except Exception:
        return {}
    return payload if isinstance(payload, dict) else {}


def agent_meta_audit(cwd: Path, *, mode: str = "compact") -> dict[str, Any]:
    workspace = (find_brain_root(cwd.resolve()) or cwd.resolve()).resolve()
    if str(workspace) not in sys.path:
        sys.path.insert(0, str(workspace))
    index = load_learning_index(workspace)
    proposals = list(index.get("proposals", []) or [])
    status_counts = proposal_status_counts(proposals)
    actionable_items: list[dict[str, Any]] = []
    proposed_count = status_counts.get("proposed", 0)
    approved_count = status_counts.get("approved", 0)
    pending_approval_statuses = ("proposed", "approved")
    pending_approval_count = sum(status_counts.get(status, 0) for status in pending_approval_statuses)
    if pending_approval_count:
        actionable_items.append(
            {
                "type": "agent_learning_pending_approval",
                "severity": "info",
                "summary": f"{pending_approval_count} agent learning proposal(s) need user-visible approval or follow-up",
                "recommended_action": (
                    "surface proposed or approved agent learning items to the user, then mark approved/"
                    "implemented/verified/rejected/superseded as appropriate"
                ),
            }
        )

    command = [
        sys.executable,
        "-m",
        "tools.brain.workflow",
        "capsule",
        "--task",
        "agent meta protocol contract audit",
        "--workflow",
        "auto",
        "--intent",
        "read",
        "--verbosity",
        "lite",
        "--json",
    ]
    sample = subprocess.run(command, cwd=str(workspace), capture_output=True, text=True, encoding="utf-8", check=False)
    sample_capsule = _parse_json_stdout(sample)
    agent_meta = sample_capsule.get("agent_meta", {}) if isinstance(sample_capsule, dict) else {}
    review = agent_meta.get("review", {}) if isinstance(agent_meta, dict) else {}
    agent_meta_contract = {
        "status": "ok"
        if sample.returncode == 0
        and isinstance(agent_meta, dict)
        and agent_meta.get("actor") == "agent"
        and agent_meta.get("substrate") == "brain"
        and agent_meta.get("tool_role") == "sensor"
        and isinstance(review, dict)
        else "warning",
        "has_agent_meta": isinstance(agent_meta, dict),
        "review_status": str(review.get("status", "") if isinstance(review, dict) else ""),
    }
    if sample.returncode != 0:
        agent_meta_contract["returncode"] = sample.returncode
        agent_meta_contract["stderr_tail"] = (sample.stderr or "")[-1000:]
    if agent_meta_contract.get("status") != "ok":
        actionable_items.append(
            {
                "type": "agent_meta_contract",
                "severity": "warning",
                "summary": "capsule agent_meta contract is not confirmed",
                "recommended_action": "run capsule contract tests before relying on agent learning prompts",
            }
        )

    daily_quality: dict[str, Any] = {"status": "unavailable"}
    try:
        from tools.brain.adapters.daily_research import detect_low_budget_evidence

        daily_quality = detect_low_budget_evidence(task="审阅 multi-horizon 低预算实验是否可作模型质量结论")
    except Exception as exc:
        daily_quality = {"status": "error", "error": str(exc)}
    if daily_quality.get("status") != "clear":
        actionable_items.append(
            {
                "type": "daily_research_evidence_quality",
                "severity": "info",
                "summary": "daily_research low-budget evidence discipline should be visible when reviewing model-quality conclusions",
                "recommended_action": "downgrade low-budget runs to smoke_only/scout_only unless promotion-grade evidence exists",
            }
        )

    payload: dict[str, Any] = {
        "status": "ok",
        "mode": str(mode or "compact"),
        "agent_learning": {
            "index_schema_version": index.get("schema_version", 1),
            "proposal_count": len(proposals),
            "status_counts": status_counts,
            "proposed_count": proposed_count,
            "approved_count": approved_count,
            "pending_approval_statuses": list(pending_approval_statuses),
            "pending_approval_count": pending_approval_count,
        },
        "agent_meta_contract": agent_meta_contract,
        "daily_research_evidence_quality": {
            "status": daily_quality.get("status", "unknown"),
            "signals": list(daily_quality.get("signals", []) or []),
            "evidence_grade": str(daily_quality.get("evidence_grade", "") or ""),
        },
        "actionable_items": actionable_items,
        "next_actions": ["review_actionable_items"] if actionable_items else ["no_meta_audit_action_needed"],
    }
    if str(mode or "compact").lower() == "full":
        payload["agent_learning"]["proposals"] = proposals
        payload["daily_research_evidence_quality"]["learning_opportunities"] = list(daily_quality.get("learning_opportunities", []) or [])
    return payload
