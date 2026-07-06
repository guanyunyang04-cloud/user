from __future__ import annotations

from pathlib import Path
from typing import Any

from tools.brain.object_registry import (
    changed_paths_from_git,
    match_objects_for_paths,
    match_objects_for_task,
    registry_validation_commands,
    registry_writeback_targets,
)
from tools.brain.routing import route_task_to_brain


PYTHON = "C:/Users/ASUS/miniconda3/envs/yolos/python.exe"


def _dedupe(items: list[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for item in items:
        text = str(item or "").strip()
        if not text or text in seen:
            continue
        seen.add(text)
        out.append(text)
    return out


def _normalize_path(path: str | Path) -> str:
    return str(path or "").strip().replace("\\", "/").strip("/")


def _is_brain_path(path: str) -> bool:
    normalized = _normalize_path(path)
    parts = normalized.split("/")
    return "brain" in parts or normalized.startswith("tools/brain/")


def _is_tools_brain_path(path: str) -> bool:
    return _normalize_path(path).startswith("tools/brain/")


def _is_qdp_active_path(path: str) -> bool:
    normalized = _normalize_path(path)
    return normalized.startswith("quant_data_platform/data/qdp_v2/active/") or normalized.startswith(
        "quant_data_platform/data/qdp_v2/datasets/"
    )


def _is_active_execution_artifact(path: str) -> bool:
    return _normalize_path(path) == "daily_research/output/active_execution_strategy.json"


def _route_key(route: dict[str, Any]) -> tuple[str, str, str]:
    return (
        str(route.get("object", "") or ""),
        str(route.get("owner", "") or ""),
        str(route.get("mode", "") or ""),
    )


def _merge_object_routes(routes: list[dict[str, Any]]) -> list[dict[str, str]]:
    merged: dict[tuple[str, str, str], dict[str, str]] = {}
    for route in routes:
        object_id = str(route.get("object", "") or route.get("object_id", "") or "").strip()
        owner = str(route.get("owner", "") or "").strip()
        mode = str(route.get("mode", "") or route.get("default_mode", "") or "read_only").strip()
        reason = str(route.get("reason", "") or route.get("route_reason", "") or "").strip()
        if not object_id or not owner:
            continue
        key = (object_id, owner, mode)
        merged[key] = {"object": object_id, "owner": owner, "mode": mode, "reason": reason}
    return [merged[key] for key in sorted(merged)]


def _object_route_from_match(match: Any, *, mode: str | None = None) -> dict[str, str]:
    selected_mode = mode or ("write" if getattr(match, "matched_paths", []) or getattr(match, "matched_write_terms", []) else getattr(match, "default_mode", "read_only"))
    return {
        "object": str(match.object_id),
        "owner": str(match.owner),
        "mode": str(selected_mode),
        "reason": str(match.route_reason or ""),
    }


def _reason_codes(*, task: str, paths: list[str], object_ids: list[str], routing: dict[str, Any]) -> list[str]:
    codes: list[str] = []
    if task:
        codes.append("task_routing_considered")
    if paths:
        codes.append("changed_paths_considered")
    if any(_is_brain_path(path) for path in paths):
        codes.append("brain_surface_changed")
    if any(_is_tools_brain_path(path) for path in paths):
        codes.append("brain_tooling_changed")
    if any(_is_qdp_active_path(path) for path in paths) or "qdp_v2_active_data_base" in object_ids:
        codes.append("qdp_active_data_base_touched")
    if any(_is_active_execution_artifact(path) for path in paths) or "active_execution_artifact" in object_ids:
        codes.append("active_execution_artifact_touched")
    if {"sequence_training_pack", "model_research_evidence"}.intersection(object_ids):
        codes.append("daily_research_research_surface_touched")
    if routing.get("supporting_brain_ids"):
        codes.append("cross_project_orchestration")
    return _dedupe(codes)


def _validation_commands(*, paths: list[str], object_ids: list[str]) -> list[str]:
    commands: list[str] = []
    if paths:
        commands.append("git diff --check")
    commands.extend(registry_validation_commands(object_ids))
    if any(_is_tools_brain_path(path) for path in paths):
        commands.append(f"{PYTHON} -m pytest tools/brain/tests -q")
    if any(_is_brain_path(path) for path in paths) or "brain_sync_surface" in object_ids:
        commands.extend(
            [
                f"{PYTHON} -m tools.brain.doc_guard check --scope changed",
                f"{PYTHON} -m tools.brain.integrity_check --json",
                f"{PYTHON} -m tools.brain.brain_sync_audit --json",
            ]
        )
    return _dedupe(commands)


def build_closure_check(task: str = "", *, paths: list[str] | None = None) -> dict[str, Any]:
    task_text = str(task or "").strip()
    explicit_paths = paths is not None
    changed_paths = [_normalize_path(path) for path in (paths if explicit_paths else changed_paths_from_git())]
    changed_paths = _dedupe([path for path in changed_paths if path])

    routing = route_task_to_brain(task_text) if task_text else {}
    route_objects = list(routing.get("object_routes", []) or [])
    task_matches = match_objects_for_task(task_text) if task_text else []
    path_matches = match_objects_for_paths(changed_paths) if changed_paths else []

    match_routes = [_object_route_from_match(match) for match in task_matches]
    match_routes.extend(_object_route_from_match(match, mode="write") for match in path_matches)
    object_routes = _merge_object_routes([*route_objects, *match_routes])

    object_ids = _dedupe([str(item.get("object", "") or "") for item in object_routes])
    write_object_ids = _dedupe([str(item.get("object", "") or "") for item in object_routes if item.get("mode") == "write"])
    writeback_targets = _dedupe(
        [
            *[str(item) for item in routing.get("writeback_targets", []) or []],
            *registry_writeback_targets(write_object_ids),
        ]
    )
    reason_codes = _reason_codes(task=task_text, paths=changed_paths, object_ids=object_ids, routing=routing)
    brain_sync_required = bool(writeback_targets or "brain_surface_changed" in reason_codes or "brain_tooling_changed" in reason_codes)
    active_artifact_guard_required = "active_execution_artifact_touched" in reason_codes

    return {
        "schema_version": 1,
        "task": task_text,
        "changed_paths": changed_paths,
        "path_source": "explicit" if explicit_paths else "git_status",
        "routing": {
            "selected_brain_id": str(routing.get("selected_brain_id", "") or ""),
            "primary_brain_id": str(routing.get("primary_brain_id", "") or ""),
            "supporting_brain_ids": list(routing.get("supporting_brain_ids", []) or []),
        } if routing else {},
        "object_routes": object_routes,
        "writeback_targets": writeback_targets,
        "brain_sync_required": brain_sync_required,
        "active_artifact_guard_required": active_artifact_guard_required,
        "reason_codes": reason_codes,
        "validation_commands": _validation_commands(paths=changed_paths, object_ids=write_object_ids),
    }
