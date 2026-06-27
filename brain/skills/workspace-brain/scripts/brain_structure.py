from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Any

from agent_learning import find_brain_root


def _load_json_file(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8-sig"))
    except Exception:
        return {}
    return payload if isinstance(payload, dict) else {}


def _line_count(path: Path) -> int:
    try:
        return len(path.read_text(encoding="utf-8-sig").splitlines())
    except Exception:
        return 0


def _read_text(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8-sig")
    except Exception:
        return ""


def _relative_path(workspace: Path, path: Path) -> str:
    try:
        return path.resolve().relative_to(workspace.resolve()).as_posix()
    except Exception:
        return path.as_posix()


def _git_ls_files(workspace: Path, pathspec: str = "") -> list[str]:
    command = ["git", "ls-files"]
    if pathspec:
        command.append(pathspec)
    result = subprocess.run(
        command,
        cwd=str(workspace),
        capture_output=True,
        text=True,
        encoding="utf-8",
        check=False,
    )
    if result.returncode != 0:
        return []
    return [line.strip().replace("\\", "/") for line in result.stdout.splitlines() if line.strip()]


def brain_structure_audit(cwd: Path, *, mode: str = "compact") -> dict[str, Any]:
    workspace = (find_brain_root(cwd.resolve()) or cwd.resolve()).resolve()
    manifest = _load_json_file(workspace / "brain" / "brain_manifest.json")
    contract = manifest.get("brain_structure_contract") if isinstance(manifest.get("brain_structure_contract"), dict) else {}
    configured_hot_paths = contract.get("hot_path_files")
    hot_paths = (
        [str(path) for path in configured_hot_paths if str(path).strip()]
        if isinstance(configured_hot_paths, list)
        else [
            "brain/skills/workspace-brain/SKILL.md",
            "daily_research/brain/state_center.md",
            "daily_research/brain/operations_center.md",
        ]
    )
    hot_path_files: dict[str, Any] = {}
    blocked_items: list[dict[str, Any]] = []
    warning_items: list[dict[str, Any]] = []
    legacy_terms = ("当前优先级", "当前边界", "规则清单", "固定读取", "审批", "禁止")
    for rel_path in hot_paths:
        path = workspace / rel_path
        text = _read_text(path)
        lines = _line_count(path)
        structural_signals: list[str] = []
        if any(term in text for term in legacy_terms):
            structural_signals.append("legacy_global_rule_terms")
        if rel_path.endswith(".md") and "SKILL.md" not in rel_path:
            if "object `" not in text and "## Object" not in text:
                structural_signals.append("missing_object_interface")
            if "procedure `" not in text and "## Procedures" not in text and "Procedure Entries" not in text:
                structural_signals.append("missing_procedure_interface")
        if rel_path.endswith("SKILL.md") and "Multi-Paradigm Brain" not in text:
            structural_signals.append("missing_multi_paradigm_skill_contract")
        hot_path_files[rel_path] = {
            "path": rel_path,
            "line_count": lines,
            "line_count_policy": contract.get("line_count_policy", "diagnostic_only_not_blocking"),
            "structural_signals": structural_signals,
            "status": "ok" if not structural_signals else "warning",
        }
        for signal in structural_signals:
            warning_items.append(
                {
                    "type": signal,
                    "path": rel_path,
                    "summary": f"{rel_path} has structure signal: {signal}",
                }
            )

    tracked_skill_paths = _git_ls_files(workspace, "brain/skills/workspace-brain")
    tracked_non_source = [
        path
        for path in tracked_skill_paths
        if "__pycache__" in path or path.endswith(".pyc") or path.endswith(".pyo")
    ]
    local_non_source: list[str] = []
    skill_root = workspace / "brain" / "skills" / "workspace-brain"
    if skill_root.exists():
        for path in skill_root.rglob("*"):
            if "__pycache__" in path.parts or path.suffix in {".pyc", ".pyo"}:
                local_non_source.append(_relative_path(workspace, path))
    if tracked_non_source or local_non_source:
        blocked_items.append(
            {
                "type": "skill_non_source_files",
                "summary": "workspace-brain skill source contains cache or compiled files",
                "tracked_paths": tracked_non_source,
                "local_paths": local_non_source,
            }
        )

    runtime_text = (workspace / "brain" / "skills" / "workspace-brain" / "scripts" / "brain_runtime.py").read_text(
        encoding="utf-8-sig"
    )
    legacy_entrypoints = {
        "removed_runtime_capsule_command": "sub.add_parser(\"capsule\")" not in runtime_text and "args.command == \"capsule\"" not in runtime_text,
        "unsupported_legacy_entries": [],
    }
    if not legacy_entrypoints["removed_runtime_capsule_command"]:
        blocked_items.append(
            {
                "type": "unsupported_legacy_entrypoint",
                "summary": "brain_runtime.py still exposes legacy capsule command",
                "path": "brain/skills/workspace-brain/scripts/brain_runtime.py",
            }
        )
    tests = []
    for rel_path in _git_ls_files(workspace, "tools/brain/tests"):
        if rel_path.endswith(".py"):
            lines = _line_count(workspace / rel_path)
            tests.append({"path": rel_path, "line_count": lines})
            if lines > 650:
                warning_items.append(
                    {
                        "type": "large_test_file",
                        "path": rel_path,
                        "summary": f"{rel_path} has {lines} lines; prefer behavior/schema tests over wording locks when changing it next",
                    }
                )

    structure_status = "blocked" if blocked_items else ("warning" if warning_items else "ok")
    brain_structure = {
        "status": structure_status,
        "blocked_count": len(blocked_items),
        "warning_count": len(warning_items),
        "hot_path_files": hot_path_files,
        "legacy_entrypoints": legacy_entrypoints,
        "tracked_non_source_files": tracked_non_source,
        "local_non_source_files": local_non_source,
        "blocked_items": blocked_items,
        "warning_items": warning_items,
    }
    payload: dict[str, Any] = {
        "status": "ok",
        "mode": str(mode or "compact"),
        "brain_structure": brain_structure,
        "next_actions": ["fix_blocked_structure_items"] if blocked_items else (["review_warning_items"] if warning_items else ["no_structure_action_needed"]),
    }
    if str(mode or "compact").lower() == "full":
        payload["brain_structure"]["tests"] = tests
        payload["brain_structure"]["contract"] = contract
    return payload
