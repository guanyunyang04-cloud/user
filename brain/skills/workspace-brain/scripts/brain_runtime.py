from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

sys.dont_write_bytecode = True
SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from agent_learning import create_proposal as create_learning_proposal
from agent_learning import list_proposals as list_learning_proposals
from agent_learning import mark_proposal as mark_learning_proposal
from agent_meta_audit import agent_meta_audit
from brain_burden import brain_burden_audit as run_brain_burden_audit
from reflection_learning import analyze_freeform, analyze_trace, reflection_template


CORE_DOCS = {
    "identity_layer.md": "# {title} 身份层\n\n- 本项目脑区由 `workspace-brain` runtime 初始化。\n- 本文件保存项目身份、目标和边界。\n",
    "state_center.md": "# {title} 状态中枢\n\n- 当前状态：新建脑区，等待首次接管写入事实。\n- 默认分支纪律：repo-tracked mutation 优先在 `main` 分支执行。\n",
    "knowledge_center.md": "# {title} 知识中枢\n\n- 稳定事实、硬规则和可复用教训写入这里。\n- 长证据和过程细节下沉到 `brain/references/`。\n",
    "brain_architecture.md": "# {title} 脑区架构\n\n- 采用统一 7 模块核：identity、state、knowledge、architecture、operations、governance、episodic。\n- 共享结构以 workspace 主脑 manifest 为准；本文件只记录区域特化。\n",
    "operations_center.md": "# {title} 操作中枢\n\n- 接管入口：先运行 brain runtime detect，再用 workflow capsule 接管项目任务。\n",
    "governance_layer.md": "# {title} 治理层\n\n- 重大动作前区分事实、推断、假设和边界。\n- Agent learning 可自动创建低风险 proposed proposal；实现协议或行为改动仍需用户确认。\n",
    "episodic_memory.md": "# {title} 情景记忆\n\n- 时间顺序证据和长复盘写入这里或 `brain/references/`。\n",
}

CORE_MODULES = (
    ("identity_layer", "identity_layer.md"),
    ("state_center", "state_center.md"),
    ("knowledge_center", "knowledge_center.md"),
    ("brain_architecture", "brain_architecture.md"),
    ("operations_center", "operations_center.md"),
    ("governance_layer", "governance_layer.md"),
    ("episodic_memory", "episodic_memory.md"),
)


def _json_default(value: object) -> str:
    if isinstance(value, Path):
        return str(value)
    return str(value)


def _print_json(payload: dict[str, Any]) -> None:
    if hasattr(sys.stdout, "reconfigure"):
        try:
            sys.stdout.reconfigure(encoding="utf-8")
        except Exception:
            pass
    print(json.dumps(payload, ensure_ascii=False, indent=2, default=_json_default))


def _run_git(cwd: Path, *args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["git", *args],
        cwd=str(cwd),
        capture_output=True,
        text=True,
        encoding="utf-8",
        check=False,
    )


def _output_text(value: object) -> str:
    if value is None:
        return ""
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return str(value)


def _run_command(cwd: Path, command: list[str], *, timeout_sec: float | None = None) -> dict[str, Any]:
    try:
        result = subprocess.run(
            command,
            cwd=str(cwd),
            capture_output=True,
            text=True,
            encoding="utf-8",
            check=False,
            timeout=timeout_sec,
        )
    except subprocess.TimeoutExpired as exc:
        stdout = _output_text(exc.output)
        stderr = _output_text(exc.stderr)
        return {
            "command": command,
            "returncode": 124,
            "ok": False,
            "timed_out": True,
            "timeout_sec": timeout_sec,
            "stdout": stdout,
            "stderr": stderr,
            "stdout_tail": stdout[-4000:],
            "stderr_tail": stderr[-2000:],
        }
    return {
        "command": command,
        "returncode": result.returncode,
        "ok": result.returncode == 0,
        "timed_out": False,
        "timeout_sec": timeout_sec,
        "stdout": result.stdout or "",
        "stderr": result.stderr or "",
        "stdout_tail": (result.stdout or "")[-4000:],
        "stderr_tail": (result.stderr or "")[-2000:],
    }


def _parse_json_stdout(result: dict[str, Any]) -> dict[str, Any]:
    try:
        payload = json.loads(str(result.get("stdout", result.get("stdout_tail", "")) or ""))
    except Exception:
        return {}
    return payload if isinstance(payload, dict) else {}


def _load_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8-sig"))
    except Exception:
        return {}
    return payload if isinstance(payload, dict) else {}


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8", newline="\n")


def _git_state(cwd: Path) -> dict[str, Any]:
    root_result = _run_git(cwd, "rev-parse", "--show-toplevel")
    is_git = root_result.returncode == 0
    if not is_git:
        return {"is_git_repo": False, "git_root": "", "branch": "", "on_main": False, "dirty_paths": []}
    git_root = Path((root_result.stdout or "").strip())
    branch_result = _run_git(git_root, "branch", "--show-current")
    status_result = _run_git(git_root, "status", "--short", "--untracked-files=all")
    branch = (branch_result.stdout or "").strip()
    return {
        "is_git_repo": True,
        "git_root": str(git_root.resolve()),
        "branch": branch,
        "on_main": branch == "main",
        "dirty_paths": [line for line in (status_result.stdout or "").splitlines() if line.strip()],
    }


def _find_brain_root(cwd: Path) -> Path | None:
    for candidate in [cwd, *cwd.parents]:
        manifest = candidate / "brain" / "brain_manifest.json"
        if manifest.exists():
            return candidate
    return None


def _find_main_workspace_root(cwd: Path) -> Path | None:
    for candidate in [cwd, *cwd.parents]:
        manifest = candidate / "brain" / "brain_manifest.json"
        if not manifest.exists():
            continue
        payload = _load_json(manifest)
        if payload.get("brain_type") == "main":
            return candidate
    return None


def detect(cwd: Path) -> dict[str, Any]:
    resolved = cwd.resolve()
    brain_root = _find_brain_root(resolved)
    tools_workflow = (brain_root / "tools" / "brain" / "workflow.py") if brain_root else resolved / "tools" / "brain" / "workflow.py"
    git = _git_state(resolved)
    has_brain = brain_root is not None
    return {
        "status": "ok",
        "cwd": str(resolved),
        "has_brain": has_brain,
        "brain_root": str(brain_root.resolve()) if brain_root else "",
        "brain_manifest": str((brain_root / "brain" / "brain_manifest.json").resolve()) if brain_root else "",
        "has_brain_tools": bool(tools_workflow.exists()),
        "brain_tools_workflow": str(tools_workflow.resolve()) if tools_workflow.exists() else "",
        "git": git,
        "next_actions": ["inspect_goal_and_relevant_context"] if has_brain else ["init_brain_available"],
    }


def _summary_counts(payload: dict[str, Any]) -> dict[str, Any]:
    findings = payload.get("findings", [])
    if not isinstance(findings, list):
        findings = []
    return {
        "status": str(payload.get("status", "") or ""),
        "error_count": int(payload.get("error_count", 0) or 0),
        "warning_count": int(payload.get("warning_count", 0) or 0),
        "warning_codes": [
            str(item.get("code", "") or "")
            for item in findings
            if isinstance(item, dict) and str(item.get("severity", "") or "") == "warning"
        ],
    }


def _manifest_summary(detected: dict[str, Any]) -> dict[str, Any]:
    raw_path = str(detected.get("brain_manifest", "") or "")
    manifest_path = Path(raw_path) if raw_path else None
    manifest = _load_json(manifest_path) if manifest_path is not None else {}
    read_order = manifest.get("read_order", [])
    child_brains = manifest.get("child_brains", [])
    return {
        "exists": bool(detected.get("has_brain")),
        "path": str(manifest_path) if manifest_path is not None else "",
        "brain_type": str(manifest.get("brain_type", "") or ""),
        "brain_id": str(manifest.get("brain_id", "") or manifest.get("id", "") or "workspace"),
        "entrypoint": str(manifest.get("entrypoint", "") or ""),
        "read_order_count": len(read_order) if isinstance(read_order, list) else 0,
        "child_brain_count": len(child_brains) if isinstance(child_brains, list) else 0,
    }


def _takeover_health(cwd: Path) -> dict[str, Any]:
    detected = detect(cwd)
    workspace = Path(detected["brain_root"] or cwd.resolve()).resolve()
    manifest = _manifest_summary(detected)
    git = detected.get("git", {}) if isinstance(detected.get("git"), dict) else {}
    next_actions: list[str] = []
    if not detected["has_brain"]:
        next_actions.append("init_brain")
    elif detected["has_brain"] and not detected["has_brain_tools"]:
        next_actions.append("register_brain")
    else:
        next_actions.append("inspect_goal_and_relevant_context")
    return {
        "status": "ok" if detected["status"] == "ok" else "warning",
        "mode": "compact",
        "health_level": "takeover",
        "detect": {
            "status": detected.get("status", ""),
            "cwd": detected.get("cwd", ""),
            "has_brain": bool(detected.get("has_brain")),
            "brain_root": detected.get("brain_root", ""),
            "has_brain_tools": bool(detected.get("has_brain_tools")),
            "brain_tools_workflow": detected.get("brain_tools_workflow", ""),
            "git": git,
        },
        "manifest": manifest,
        "capsule_command": (
            'C:/Users/ASUS/miniconda3/envs/yolos/python.exe -m tools.brain.workflow capsule '
            '--task "<task>" --workflow auto --intent read --verbosity lite --json'
        ),
        "route_command": 'C:/Users/ASUS/miniconda3/envs/yolos/python.exe -m tools.brain.workflow route --task "<task>" --json',
        "bootstrap_command": "C:/Users/ASUS/miniconda3/envs/yolos/python.exe -m tools.brain.workflow bootstrap --brain <workspace|child_brain_id> --json",
        "full_health_command": "C:/Users/ASUS/miniconda3/envs/yolos/python.exe brain/skills/workspace-brain/scripts/brain_runtime.py health --cwd . --mode full --timeout-sec 60",
        "deferred_checks": [
            "skill_sync",
            "doc_guard",
            "integrity_check",
            "brain_catalog",
            "daily_research_frontier",
            "optional_project_checks",
        ],
        "project_checks_note": "Project-specific checks are selected by agent judgment from the current goal, risk, and changed files.",
        "workspace": str(workspace),
        "next_actions": next_actions,
    }


def health(cwd: Path, *, mode: str = "compact", timeout_sec: float = 60.0) -> dict[str, Any]:
    if str(mode or "compact").lower() == "compact":
        return _takeover_health(cwd)

    detected = detect(cwd)
    workspace = Path(detected["brain_root"] or cwd.resolve()).resolve()
    if detected["has_brain"] and not detected["has_brain_tools"]:
        payload = {
            "status": "ok",
            "mode": "full",
            "health_level": "deep",
            "detect": detected,
            "skill_sync": {"status": "skipped", "all_in_sync": False, "skills": []},
            "doc_guard": {"status": "skipped", "returncode": 0, "stdout_tail": "", "stderr_tail": ""},
            "integrity": {"status": "ok", "error_count": 0, "warning_count": 0, "warning_codes": []},
            "catalog": {"status": "standalone", "brain_count": 1, "generated_at": "", "non_truth_brains": []},
            "frontier": {
                "status": "skipped",
                "brain_may_be_stale": False,
                "warnings": [],
                "unregistered_latest_run_tags": [],
                "unregistered_latest_run_details": [],
            },
            "next_actions": ["register_brain"],
        }
        return payload
    skill_sync_result = _run_command(
        workspace,
        [sys.executable, "-m", "tools.brain.skill_install", "--check"],
        timeout_sec=timeout_sec,
    )
    integrity_result = _run_command(
        workspace,
        [sys.executable, "-m", "tools.brain.integrity_check", "--json"],
        timeout_sec=timeout_sec,
    )
    doc_guard_result = _run_command(
        workspace,
        [sys.executable, "-m", "tools.brain.doc_guard", "check"],
        timeout_sec=timeout_sec,
    )
    frontier_result = _run_command(
        workspace,
        [sys.executable, "-m", "tools.brain.workflow", "current-frontier", "--json"],
        timeout_sec=timeout_sec,
    )
    skill_sync = _parse_json_stdout(skill_sync_result)
    integrity = _parse_json_stdout(integrity_result)
    frontier = _parse_json_stdout(frontier_result)

    catalog_summary: dict[str, Any] = {"status": "unavailable"}
    frontier_summary: dict[str, Any] = {"status": "unavailable"}
    try:
        if str(workspace) not in sys.path:
            sys.path.insert(0, str(workspace))
        from tools.brain.platform import build_brain_catalog

        catalog = build_brain_catalog()
        non_truth_statuses = {
            "cache_legacy",
            "non_truth_tooling",
            "external_or_inactive_missing_manifest",
            "discovered_untracked",
            "missing_manifest",
        }
        catalog_summary = {
            "status": "ok",
            "brain_count": len(catalog.get("brains", [])),
            "generated_at": catalog.get("generated_at", ""),
            "non_truth_brains": [
                {
                    "brain_id": item.get("brain_id", ""),
                    "status": item.get("status", ""),
                    "root": item.get("root", ""),
                }
                for item in catalog.get("brains", [])
                if isinstance(item, dict) and item.get("status") in non_truth_statuses
            ],
        }
    except Exception as exc:
        catalog_summary = {"status": "error", "error": str(exc)}

    if frontier_result.get("timed_out"):
        frontier_summary = {
            "status": "timeout",
            "timed_out": True,
            "timeout_sec": frontier_result.get("timeout_sec"),
            "brain_may_be_stale": False,
            "warnings": ["frontier_check_timed_out"],
            "unregistered_latest_run_tags": [],
            "unregistered_latest_run_details": [],
            "stderr_tail": frontier_result.get("stderr_tail", ""),
        }
    elif frontier_result["returncode"] != 0:
        frontier_summary = {
            "status": "failed",
            "timed_out": False,
            "timeout_sec": frontier_result.get("timeout_sec"),
            "brain_may_be_stale": False,
            "warnings": ["frontier_check_failed"],
            "unregistered_latest_run_tags": [],
            "unregistered_latest_run_details": [],
            "stderr_tail": frontier_result.get("stderr_tail", ""),
        }
    else:
        frontier_summary = {
            "status": "warning" if frontier.get("brain_may_be_stale") else "ok",
            "timed_out": False,
            "timeout_sec": frontier_result.get("timeout_sec"),
            "brain_may_be_stale": bool(frontier.get("brain_may_be_stale")),
            "warnings": list(frontier.get("warnings", []) or []),
            "unregistered_latest_run_tags": list(frontier.get("unregistered_latest_run_tags", []) or []),
            "unregistered_latest_run_details": list(frontier.get("unregistered_latest_run_details", []) or []),
        }

    next_actions: list[str] = []
    if not detected["has_brain"]:
        next_actions.append("init_brain")
    if not skill_sync.get("all_in_sync", False):
        next_actions.append("run_skill_install")
    integrity_summary = _summary_counts(integrity)
    if integrity_result.get("timed_out"):
        next_actions.append("rerun_integrity_with_more_time")
    if integrity_summary["error_count"]:
        next_actions.append("fix_integrity_errors")
    if doc_guard_result.get("timed_out"):
        next_actions.append("rerun_doc_guard_with_more_time")
    if skill_sync_result.get("timed_out"):
        next_actions.append("rerun_skill_sync_with_more_time")
    if frontier_result.get("timed_out"):
        next_actions.append("rerun_frontier_with_more_time")
    if frontier_summary.get("brain_may_be_stale"):
        next_actions.append("review_frontier_reconciliation")
    if not next_actions:
        next_actions.append("inspect_goal_and_relevant_context")

    timed_out = any(bool(result.get("timed_out")) for result in (skill_sync_result, integrity_result, doc_guard_result, frontier_result))
    payload = {
        "status": "ok" if detected["status"] == "ok" and integrity_result["returncode"] == 0 and not timed_out else "warning",
        "mode": "full",
        "health_level": "deep",
        "detect": detected,
        "skill_sync": {
            "status": "ok" if skill_sync_result["returncode"] == 0 else "failed",
            "all_in_sync": bool(skill_sync.get("all_in_sync", False)),
            "skills": list(skill_sync.get("skills", []) or []),
            "timed_out": bool(skill_sync_result.get("timed_out")),
            "timeout_sec": skill_sync_result.get("timeout_sec"),
        },
        "doc_guard": {
            "status": "ok" if doc_guard_result["returncode"] == 0 else "failed",
            "returncode": doc_guard_result["returncode"],
            "timed_out": bool(doc_guard_result.get("timed_out")),
            "timeout_sec": doc_guard_result.get("timeout_sec"),
            "stdout_tail": doc_guard_result["stdout_tail"],
            "stderr_tail": doc_guard_result["stderr_tail"],
        },
        "integrity": {
            **integrity_summary,
            "timed_out": bool(integrity_result.get("timed_out")),
            "timeout_sec": integrity_result.get("timeout_sec"),
        },
        "catalog": catalog_summary,
        "frontier": frontier_summary,
        "next_actions": next_actions,
    }
    return payload


def _title_from_brain_id(brain_id: str) -> str:
    return " ".join(part.capitalize() for part in str(brain_id or "project").replace("-", "_").split("_") if part)


def _infer_brain_id(cwd: Path) -> str:
    raw = cwd.resolve().name.strip().lower()
    normalized = re.sub(r"[^a-z0-9_-]+", "_", raw).strip("_-")
    return normalized or "project_brain"


def _local_core_path(filename: str, *, body_root: str = "") -> str:
    prefix = str(body_root or "").strip().strip("/\\")
    path = f"brain/{filename}" if not prefix or prefix == "." else f"{prefix}/brain/{filename}"
    return path.replace("\\", "/")


def _build_project_manifest(brain_id: str, *, body_root: str = ".") -> dict[str, Any]:
    modules = [{"id": module_id, "paths": [_local_core_path(filename, body_root=body_root)]} for module_id, filename in CORE_MODULES]
    return {
        "brain_type": "project",
        "brain_id": brain_id,
        "attach_status": "standalone_unattached",
        "entrypoint": _local_core_path("identity_layer.md", body_root=body_root),
        "body_root": body_root,
        "regional_specialization": {
            "role": "project_cortex",
            "priority_regions": ["orientation_system", "long_term_memory", "executive_control", "sensorimotor_loop"],
            "focus_modules": ["state_center", "knowledge_center", "operations_center"],
            "body_entry_priority": [body_root],
        },
        "read_order": [_local_core_path(filename, body_root=body_root) for _, filename in CORE_MODULES],
        "identity_path": _local_core_path("identity_layer.md", body_root=body_root),
        "state_path": _local_core_path("state_center.md", body_root=body_root),
        "knowledge_path": _local_core_path("knowledge_center.md", body_root=body_root),
        "operations_path": _local_core_path("operations_center.md", body_root=body_root),
        "governance_path": _local_core_path("governance_layer.md", body_root=body_root),
        "write_routes": {
            "identity": _local_core_path("identity_layer.md", body_root=body_root),
            "state": _local_core_path("state_center.md", body_root=body_root),
            "knowledge": _local_core_path("knowledge_center.md", body_root=body_root),
            "operations": _local_core_path("operations_center.md", body_root=body_root),
            "governance": _local_core_path("governance_layer.md", body_root=body_root),
            "episodic": _local_core_path("episodic_memory.md", body_root=body_root),
            "brain_structure": _local_core_path("brain_architecture.md", body_root=body_root),
            "references": (f"{body_root}/brain/references/" if body_root and body_root != "." else "brain/references/").replace("\\", "/"),
        },
        "body_map": {"project": [body_root]},
        "modules": modules,
        "routing_hints": {
            "aliases": [brain_id],
            "terms": [],
            "path_prefixes": [] if body_root == "." else [body_root],
        },
        "handoff_contract": {
            "derive_entry_sequence_from_shared_contract": False,
            "entry_sequence": [_local_core_path(filename, body_root=body_root) for _, filename in CORE_MODULES],
            "body_entry_priority": ["project"],
            "principle": "Attach to the project brain first, then enter the project body through body_map.",
        },
    }


def init_brain(cwd: Path, brain_id: str | None = None) -> dict[str, Any]:
    resolved = cwd.resolve()
    resolved.mkdir(parents=True, exist_ok=True)
    resolved_brain_id = str(brain_id or "").strip() or _infer_brain_id(resolved)
    git = _git_state(resolved)
    if git["is_git_repo"] and not git["on_main"]:
        return {
            "status": "blocked",
            "blockers": ["not_on_main_for_mutation"],
            "mutation_allowed": False,
            "git": git,
        }
    brain_dir = resolved / "brain"
    manifest_path = brain_dir / "brain_manifest.json"
    title = _title_from_brain_id(resolved_brain_id)
    brain_dir.mkdir(parents=True, exist_ok=True)
    (brain_dir / "references").mkdir(parents=True, exist_ok=True)
    for filename, template in CORE_DOCS.items():
        path = brain_dir / filename
        if not path.exists():
            path.write_text(template.format(title=title), encoding="utf-8", newline="\n")
    if not manifest_path.exists():
        _write_json(manifest_path, _build_project_manifest(resolved_brain_id))
    return {
        "status": "ok",
        "mutation_allowed": True,
        "brain_id": resolved_brain_id,
        "brain_manifest": str(manifest_path.resolve()),
        "created_paths": [str((brain_dir / name).resolve()) for name in [*CORE_DOCS, "brain_manifest.json"]],
    }


def _workspace_relative(workspace_root: Path, path: Path) -> str:
    return path.resolve().relative_to(workspace_root.resolve()).as_posix()


def _reference_count(root: Path) -> int:
    references = root / "references"
    if not references.exists():
        return 0
    return sum(1 for item in references.iterdir() if item.is_file())


def _catalog_record(
    *,
    brain_id: str,
    root: str,
    manifest_path: str,
    status: str,
    body_root: str,
    last_guard_status: str,
) -> dict[str, Any]:
    return {
        "brain_id": brain_id,
        "root": root,
        "manifest_path": manifest_path,
        "status": status,
        "body_root": body_root,
        "references_path": f"{root}/references" if root else "",
        "references_count": _reference_count(Path(root)) if root else 0,
        "language_policy": "zh_semantic_en_identifiers_v1",
        "last_guard_status": last_guard_status,
    }


def _rebuild_catalog(workspace_root: Path, main_manifest: dict[str, Any]) -> dict[str, Any]:
    existing_catalog = _load_json(workspace_root / "brain" / "brain_catalog.json")
    preserved = []
    for item in existing_catalog.get("brains", []) if isinstance(existing_catalog.get("brains"), list) else []:
        if not isinstance(item, dict):
            continue
        status = str(item.get("status", "") or "")
        if status in {"canonical_root", "attached"}:
            continue
        preserved.append(item)
    records = [
        _catalog_record(
            brain_id="workspace_root",
            root="brain",
            manifest_path="brain/brain_manifest.json",
            status="canonical_root",
            body_root=".",
            last_guard_status="ok",
        )
    ]
    for child in main_manifest.get("child_brains", []) if isinstance(main_manifest.get("child_brains"), list) else []:
        if not isinstance(child, dict):
            continue
        child_id = str(child.get("id", "") or "").strip()
        child_path = str(child.get("path", "") or "").strip().replace("\\", "/")
        body_root = str(child.get("body_root", "") or "").strip().replace("\\", "/")
        if not child_id or not child_path:
            continue
        root = str(Path(child_path).parent).replace("\\", "/")
        records.append(
            _catalog_record(
                brain_id=child_id,
                root=root,
                manifest_path=child_path,
                status="attached" if str(child.get("attach_status", "") or "").startswith("attached") else "discovered_untracked",
                body_root=body_root or str(Path(root).parent).replace("\\", "/"),
                last_guard_status="ok",
            )
        )
    seen = {str(item.get("brain_id", "")) for item in records}
    records.extend(item for item in preserved if str(item.get("brain_id", "")) not in seen)
    return {
        "schema_version": 1,
        "generated_at": datetime.now().date().isoformat(),
        "language_policy": "zh_semantic_en_identifiers_v1",
        "brains": records,
    }


def _attached_manifest(base_manifest: dict[str, Any], *, brain_id: str, body_root: str) -> dict[str, Any]:
    manifest = _build_project_manifest(brain_id, body_root=body_root)
    regional = base_manifest.get("regional_specialization") if isinstance(base_manifest.get("regional_specialization"), dict) else {}
    routing_hints = base_manifest.get("routing_hints") if isinstance(base_manifest.get("routing_hints"), dict) else {}
    body_priority = list(regional.get("body_entry_priority", []) or [])
    if not body_priority or body_priority == ["."]:
        body_priority = [body_root]
    manifest.update(
        {
            "brain_type": "sub_brain",
            "parent_brain": "brain/brain_manifest.json",
            "attach_status": "attached_to_main_brain",
            "shared_contract_source": "brain/brain_manifest.json#shared_regional_brain_contract",
            "regional_specialization": {
                "role": str(regional.get("role", "") or "project_cortex"),
                "priority_regions": list(regional.get("priority_regions", []) or ["orientation_system", "long_term_memory", "executive_control", "sensorimotor_loop"]),
                "focus_modules": list(regional.get("focus_modules", []) or ["state_center", "knowledge_center", "operations_center"]),
                "body_entry_priority": body_priority,
            },
            "routing_hints": {
                "aliases": sorted(set([brain_id, body_root, *list(routing_hints.get("aliases", []) or [])])),
                "terms": sorted(set(str(item) for item in routing_hints.get("terms", []) or [] if str(item).strip())),
                "path_prefixes": sorted(set([body_root, *[str(item) for item in routing_hints.get("path_prefixes", []) or [] if str(item).strip()]])),
            },
            "handoff_contract": {
                "derive_entry_sequence_from_shared_contract": True,
                "body_entry_priority": ["project"],
                "principle": "Attach to the brain first, then read identity, state, knowledge, operations, governance, and enter body_map.",
            },
        }
    )
    manifest.pop("read_order", None)
    return manifest


def register_brain(cwd: Path, *, workspace_root: Path | None = None, brain_id: str | None = None) -> dict[str, Any]:
    project_root = cwd.resolve()
    workspace = (workspace_root.resolve() if workspace_root is not None else (_find_main_workspace_root(project_root.parent) or _find_main_workspace_root(project_root)))
    if workspace is None:
        return {"status": "error", "error": "workspace_main_brain_not_found", "project_root": str(project_root)}
    if project_root == workspace:
        return {"status": "error", "error": "cannot_register_workspace_root_as_child", "workspace_root": str(workspace)}
    try:
        body_root = _workspace_relative(workspace, project_root)
    except ValueError:
        return {
            "status": "blocked",
            "blockers": ["project_outside_workspace"],
            "workspace_root": str(workspace),
            "project_root": str(project_root),
        }

    git = _git_state(workspace)
    if git["is_git_repo"] and not git["on_main"]:
        return {
            "status": "blocked",
            "blockers": ["not_on_main_for_mutation"],
            "mutation_allowed": False,
            "git": git,
        }

    child_manifest_path = project_root / "brain" / "brain_manifest.json"
    if not child_manifest_path.exists():
        return {"status": "error", "error": "project_brain_manifest_missing", "project_root": str(project_root)}
    main_manifest_path = workspace / "brain" / "brain_manifest.json"
    main_manifest = _load_json(main_manifest_path)
    if main_manifest.get("brain_type") != "main":
        return {"status": "error", "error": "workspace_manifest_is_not_main", "workspace_root": str(workspace)}
    base_manifest = _load_json(child_manifest_path)
    resolved_brain_id = str(brain_id or base_manifest.get("brain_id") or "").strip() or _infer_brain_id(project_root)
    attached_manifest = _attached_manifest(base_manifest, brain_id=resolved_brain_id, body_root=body_root)
    _write_json(child_manifest_path, attached_manifest)

    child_ref = {
        "id": resolved_brain_id,
        "path": f"{body_root}/brain/brain_manifest.json",
        "role": attached_manifest["regional_specialization"]["role"],
        "regional_role": attached_manifest["regional_specialization"]["role"],
        "control_level": "managed_by_main_brain",
        "body_root": body_root,
        "entrypoint": f"{body_root}/brain/identity_layer.md",
        "attach_status": "attached",
    }
    children = main_manifest.get("child_brains")
    if not isinstance(children, list):
        children = []
    replaced = False
    updated_children: list[dict[str, Any]] = []
    for item in children:
        if not isinstance(item, dict):
            continue
        if item.get("id") == resolved_brain_id or item.get("body_root") == body_root:
            updated_children.append(child_ref)
            replaced = True
        else:
            updated_children.append(item)
    if not replaced:
        updated_children.append(child_ref)
    main_manifest["child_brains"] = updated_children
    _write_json(main_manifest_path, main_manifest)
    catalog = _rebuild_catalog(workspace, main_manifest)
    catalog_path = workspace / "brain" / "brain_catalog.json"
    _write_json(catalog_path, catalog)
    return {
        "status": "ok",
        "mutation_allowed": True,
        "registered": not replaced,
        "updated_existing": replaced,
        "brain_id": resolved_brain_id,
        "body_root": body_root,
        "child_manifest": str(child_manifest_path.resolve()),
        "workspace_manifest": str(main_manifest_path.resolve()),
        "catalog_path": str(catalog_path.resolve()),
        "child_ref": child_ref,
    }


def review(
    cwd: Path,
    *,
    task: str = "",
    observation: str = "",
    test_output: str = "",
    trace_json: Path | None = None,
) -> dict[str, Any]:
    if trace_json is not None:
        payload = json.loads(trace_json.read_text(encoding="utf-8-sig"))
        return analyze_trace(payload if isinstance(payload, dict) else {})
    return analyze_freeform(task=task, observation=observation, test_output=test_output)


def proposal(
    cwd: Path,
    title: str,
    trigger: str,
    evidence: str,
    recommendation: str,
    *,
    severity: str = "info",
    owner_brain: str = "",
    writeback_target: str = "brain/references/",
    requires_user_confirmation: bool = True,
    target_layer: str = "brain_docs",
    status: str = "proposed",
    related_task: str = "",
    suggested_tests: list[str] | None = None,
    lesson: str = "",
    root_cause: str = "",
    supporting_events: list[dict[str, Any]] | None = None,
    anti_overfit_check: str = "",
) -> dict[str, Any]:
    return create_learning_proposal(
        cwd,
        title=title,
        trigger=trigger,
        evidence=evidence,
        recommendation=recommendation,
        severity=severity,
        owner_brain_name=owner_brain,
        writeback_target=writeback_target,
        requires_user_confirmation=requires_user_confirmation,
        target_layer=target_layer,
        status=status,
        related_task=related_task,
        suggested_tests=suggested_tests,
        lesson=lesson,
        root_cause=root_cause,
        supporting_events=supporting_events,
        anti_overfit_check=anti_overfit_check,
    )


def list_proposals(cwd: Path) -> dict[str, Any]:
    return list_learning_proposals(cwd)


def mark_proposal(cwd: Path, proposal_id: str, status: str) -> dict[str, Any]:
    return mark_learning_proposal(cwd, proposal_id, status)


def meta_audit(cwd: Path, *, mode: str = "compact") -> dict[str, Any]:
    return agent_meta_audit(cwd, mode=mode)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Workspace brain runtime helper.")
    sub = parser.add_subparsers(dest="command", required=True)
    detect_parser = sub.add_parser("detect")
    detect_parser.add_argument("--cwd", default=".")
    init_parser = sub.add_parser("init")
    init_parser.add_argument("--cwd", default=".")
    init_parser.add_argument("--brain-id", default="")
    register_parser = sub.add_parser("register")
    register_parser.add_argument("--cwd", default=".")
    register_parser.add_argument("--workspace-root", default="")
    register_parser.add_argument("--brain-id", default="")
    attach_parser = sub.add_parser("attach")
    attach_parser.add_argument("--cwd", default=".")
    attach_parser.add_argument("--workspace-root", default="")
    attach_parser.add_argument("--brain-id", default="")
    health_parser = sub.add_parser("health")
    health_parser.add_argument("--cwd", default=".")
    health_parser.add_argument("--mode", choices=("compact", "full"), default="compact")
    health_parser.add_argument("--timeout-sec", type=float, default=60.0)
    template_parser = sub.add_parser("reflection-template")
    template_parser.add_argument("--json", action="store_true")
    proposal_parser = sub.add_parser("proposal")
    proposal_parser.add_argument("--cwd", default=".")
    proposal_parser.add_argument("--title", required=True)
    proposal_parser.add_argument("--trigger", required=True)
    proposal_parser.add_argument("--evidence", required=True)
    proposal_parser.add_argument("--recommendation", required=True)
    proposal_parser.add_argument("--severity", default="info")
    proposal_parser.add_argument("--owner-brain", default="")
    proposal_parser.add_argument("--writeback-target", default="brain/references/")
    proposal_parser.add_argument("--requires-user-confirmation", action="store_true", default=True)
    proposal_parser.add_argument("--target-layer", default="brain_docs")
    proposal_parser.add_argument("--status", default="proposed")
    proposal_parser.add_argument("--related-task", default="")
    proposal_parser.add_argument("--suggested-test", action="append", default=[])
    review_parser = sub.add_parser("review")
    review_parser.add_argument("--cwd", default=".")
    review_parser.add_argument("--task", default="")
    review_parser.add_argument("--observation", default="")
    review_parser.add_argument("--test-output", default="")
    review_parser.add_argument("--trace-json", default="")
    review_parser.add_argument("--json", action="store_true")
    meta_audit_parser = sub.add_parser("agent-meta-audit")
    meta_audit_parser.add_argument("--cwd", default=".")
    meta_audit_parser.add_argument("--mode", choices=("compact", "full"), default="compact")
    burden_audit_parser = sub.add_parser("brain-burden-audit")
    burden_audit_parser.add_argument("--cwd", default=".")
    burden_audit_parser.add_argument("--mode", choices=("compact", "full"), default="compact")
    list_parser = sub.add_parser("list-proposals")
    list_parser.add_argument("--cwd", default=".")
    mark_parser = sub.add_parser("mark-proposal")
    mark_parser.add_argument("--cwd", default=".")
    mark_parser.add_argument("--proposal-id", required=True)
    mark_parser.add_argument("--status", required=True)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    cwd = Path(str(getattr(args, "cwd", ".") or "."))
    if args.command == "detect":
        payload = detect(cwd)
    elif args.command == "init":
        payload = init_brain(cwd, str(args.brain_id or "") or None)
    elif args.command in {"register", "attach"}:
        workspace_root = Path(str(args.workspace_root)).resolve() if str(args.workspace_root or "").strip() else None
        payload = register_brain(cwd, workspace_root=workspace_root, brain_id=str(args.brain_id or "") or None)
    elif args.command == "health":
        payload = health(cwd, mode=str(args.mode or "compact"), timeout_sec=float(args.timeout_sec or 60.0))
    elif args.command == "reflection-template":
        payload = reflection_template()
    elif args.command == "proposal":
        payload = proposal(
            cwd,
            str(args.title),
            str(args.trigger),
            str(args.evidence),
            str(args.recommendation),
            severity=str(args.severity),
            owner_brain=str(args.owner_brain or ""),
            writeback_target=str(args.writeback_target),
            requires_user_confirmation=bool(args.requires_user_confirmation),
            target_layer=str(args.target_layer or "brain_docs"),
            status=str(args.status or "proposed"),
            related_task=str(args.related_task or ""),
            suggested_tests=list(args.suggested_test or []),
        )
    elif args.command == "review":
        payload = review(
            cwd,
            task=str(args.task or ""),
            observation=str(args.observation or ""),
            test_output=str(args.test_output or ""),
            trace_json=Path(str(args.trace_json)).resolve() if str(args.trace_json or "").strip() else None,
        )
    elif args.command == "agent-meta-audit":
        payload = meta_audit(cwd, mode=str(args.mode or "compact"))
    elif args.command == "brain-burden-audit":
        payload = run_brain_burden_audit(cwd, mode=str(args.mode or "compact"))
    elif args.command == "list-proposals":
        payload = list_proposals(cwd)
    elif args.command == "mark-proposal":
        payload = mark_proposal(cwd, str(args.proposal_id), str(args.status))
    else:
        raise ValueError(f"Unsupported command: {args.command}")
    _print_json(payload)
    return 0 if payload.get("status") != "error" else 2


if __name__ == "__main__":
    raise SystemExit(main())
