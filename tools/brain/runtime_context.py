from __future__ import annotations

from typing import Any


VERBOSITY_LEVELS = ("lite", "standard", "full")


def normalize_verbosity(value: str | None) -> str:
    text = str(value or "lite").strip().lower()
    return text if text in VERBOSITY_LEVELS else "lite"


def summary_budget(profile: str) -> dict[str, int]:
    normalized = normalize_verbosity(profile)
    if normalized == "full":
        return {"main_summary_lines": 18, "child_summary_lines": 18, "related_references": 5, "frontier_tags": 12}
    if normalized == "standard":
        return {"main_summary_lines": 8, "child_summary_lines": 8, "related_references": 5, "frontier_tags": 5}
    return {"main_summary_lines": 0, "child_summary_lines": 0, "related_references": 3, "frontier_tags": 3}


def deep_dive_commands(*, selected_brain_id: str = "") -> list[str]:
    commands = [
        'C:/Users/ASUS/miniconda3/envs/yolos/python.exe -m tools.brain.workflow current-frontier --json',
        'C:/Users/ASUS/miniconda3/envs/yolos/python.exe -m tools.brain.workflow query --q "<tag|dataset|r-id>" --json',
        "C:/Users/ASUS/miniconda3/envs/yolos/python.exe brain/skills/workspace-brain/scripts/brain_runtime.py health --mode full --cwd .",
        'C:/Users/ASUS/miniconda3/envs/yolos/python.exe -m tools.brain.workflow capsule --task "<task>" --workflow auto --intent read --verbosity full --json',
    ]
    if selected_brain_id:
        commands.append(f"C:/Users/ASUS/miniconda3/envs/yolos/python.exe -m tools.brain.workflow bootstrap --brain {selected_brain_id} --json")
    return commands


def compact_text_lines(lines: Any, *, limit: int) -> list[str]:
    if limit <= 0 or not isinstance(lines, list):
        return []
    return [str(item) for item in lines[:limit] if str(item).strip()]


def compact_reference(record: dict[str, Any]) -> dict[str, Any]:
    verdict = str(record.get("verdict", "") or "")
    return {
        "id": str(record.get("id", "") or ""),
        "path": str(record.get("path", "") or ""),
        "tags": list(record.get("tags", []) or []),
        "short_verdict": verdict[:240],
    }


def compact_references(records: Any, *, limit: int, full: bool) -> list[dict[str, Any]]:
    if not isinstance(records, list):
        return []
    selected = [item for item in records if isinstance(item, dict)][:limit]
    return selected if full else [compact_reference(item) for item in selected]


def summarize_frontier(report: Any, *, profile: str) -> dict[str, Any]:
    if not isinstance(report, dict):
        return {}
    if normalize_verbosity(profile) == "full":
        return dict(report)
    budget = summary_budget(profile)
    unregistered = list(report.get("unregistered_latest_tags", []) or [])
    warnings = list(report.get("warnings", []) or [])
    return {
        "brain_may_be_stale": bool(report.get("brain_may_be_stale")),
        "warnings": warnings,
        "warning_count": len(warnings),
        "unregistered_latest_count": len(unregistered),
        "unregistered_latest_tags": unregistered[: budget["frontier_tags"]],
        "latest_brain_reference_time": str(report.get("latest_brain_reference_time", "") or ""),
        "guidance": str(report.get("guidance", "") or ""),
    }


def compact_main_context(context: dict[str, Any], *, profile: str) -> dict[str, Any]:
    normalized = normalize_verbosity(profile)
    if normalized == "full":
        return context
    budget = summary_budget(normalized)
    compacted = {
        "brain_id": context.get("brain_id", ""),
        "main_manifest": context.get("main_manifest", ""),
        "main_entrypoint": context.get("main_entrypoint", ""),
        "git": context.get("git", {}),
        "branch_policy": context.get("branch_policy", {}),
        "global_boundaries": context.get("global_boundaries", []),
    }
    if normalized == "standard":
        compacted["summary"] = compact_text_lines(context.get("summary", []), limit=budget["main_summary_lines"])
        compacted["hard_rules"] = compact_text_lines(context.get("hard_rules", []), limit=budget["main_summary_lines"])
    return compacted


def compact_child_context(context: dict[str, Any], *, profile: str) -> dict[str, Any]:
    normalized = normalize_verbosity(profile)
    if normalized == "full":
        return context
    budget = summary_budget(normalized)
    compacted: dict[str, Any] = {
        "brain_id": context.get("brain_id", ""),
        "child_manifest": context.get("child_manifest", ""),
        "child_entrypoint": context.get("child_entrypoint", ""),
        "fast_handoff_paths": context.get("fast_handoff_paths", []),
        "write_routes": context.get("write_routes", {}),
        "related_references": compact_references(
            context.get("related_references", []),
            limit=budget["related_references"],
            full=False,
        ),
    }
    if normalized == "standard":
        compacted["state_summary"] = compact_text_lines(context.get("state_summary", []), limit=budget["child_summary_lines"])
        compacted["hard_rules"] = compact_text_lines(context.get("hard_rules", []), limit=budget["child_summary_lines"])
    return compacted


def compact_catalog_health(catalog: dict[str, Any]) -> dict[str, Any]:
    acknowledged_statuses = {
        "cache_legacy",
        "non_truth_tooling",
        "external_or_inactive_missing_manifest",
    }
    non_truth = list(catalog.get("non_truth_brains", []) or [])
    acknowledged = [
        item
        for item in non_truth
        if isinstance(item, dict) and str(item.get("status", "") or "") in acknowledged_statuses
    ]
    actionable = [
        item
        for item in non_truth
        if isinstance(item, dict) and str(item.get("status", "") or "") not in acknowledged_statuses
    ]
    return {
        "status": catalog.get("status", "unknown"),
        "brain_count": int(catalog.get("brain_count", 0) or 0),
        "generated_at": str(catalog.get("generated_at", "") or ""),
        "acknowledged_info": acknowledged,
        "acknowledged_info_count": len(acknowledged),
        "actionable_warnings": actionable,
        "actionable_warning_count": len(actionable),
    }


def compact_frontier_health(frontier: dict[str, Any]) -> dict[str, Any]:
    unregistered = list(frontier.get("unregistered_latest_tags", []) or [])
    return {
        "status": frontier.get("status", "unknown"),
        "brain_may_be_stale": bool(frontier.get("brain_may_be_stale")),
        "warnings": list(frontier.get("warnings", []) or []),
        "unregistered_latest_count": len(unregistered),
        "unregistered_latest_tags": unregistered[:3],
    }
