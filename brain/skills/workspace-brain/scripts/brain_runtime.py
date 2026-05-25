from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from reflection_learning import analyze_freeform, analyze_trace, build_proposal_payload, reflection_template


CORE_DOCS = {
    "identity_layer.md": "# {title} 身份层\n\n- 本项目脑区由 `workspace-brain` runtime 初始化。\n- 本文件保存项目身份、目标和边界。\n",
    "state_center.md": "# {title} 状态中枢\n\n- 当前状态：新建脑区，等待首次接管写入事实。\n- 默认分支纪律：repo-tracked mutation 优先在 `main` 分支执行。\n",
    "knowledge_center.md": "# {title} 知识中枢\n\n- 稳定事实、硬规则和可复用教训写入这里。\n- 长证据和过程细节下沉到 `brain/references/`。\n",
    "brain_architecture.md": "# {title} 脑区架构\n\n- 采用最小脑区结构：identity、state、knowledge、operations、governance、episodic。\n",
    "operations_center.md": "# {title} 操作中枢\n\n- 接管入口：先运行 brain runtime detect/capsule，再执行项目任务。\n",
    "governance_layer.md": "# {title} 治理层\n\n- 重大动作前区分事实、推断、假设和边界。\n- Runtime learning 默认只生成 proposal，核心治理写回需要用户确认。\n",
    "episodic_memory.md": "# {title} 情景记忆\n\n- 时间顺序证据和长复盘写入这里或 `brain/references/`。\n",
}


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


def _run_command(cwd: Path, command: list[str]) -> dict[str, Any]:
    result = subprocess.run(
        command,
        cwd=str(cwd),
        capture_output=True,
        text=True,
        encoding="utf-8",
        check=False,
    )
    return {
        "command": command,
        "returncode": result.returncode,
        "ok": result.returncode == 0,
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
        "next_actions": ["run_capsule"] if has_brain else ["init_brain_available"],
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


def _compact_health(full_payload: dict[str, Any]) -> dict[str, Any]:
    from tools.brain.runtime_context import compact_catalog_health, compact_frontier_health

    detect_payload = full_payload.get("detect", {}) if isinstance(full_payload.get("detect"), dict) else {}
    skill_sync = full_payload.get("skill_sync", {}) if isinstance(full_payload.get("skill_sync"), dict) else {}
    doc_guard = full_payload.get("doc_guard", {}) if isinstance(full_payload.get("doc_guard"), dict) else {}
    catalog = full_payload.get("catalog", {}) if isinstance(full_payload.get("catalog"), dict) else {}
    frontier = full_payload.get("frontier", {}) if isinstance(full_payload.get("frontier"), dict) else {}
    return {
        "status": full_payload.get("status", "unknown"),
        "mode": "compact",
        "detect": {
            "status": detect_payload.get("status", ""),
            "cwd": detect_payload.get("cwd", ""),
            "has_brain": bool(detect_payload.get("has_brain")),
            "has_brain_tools": bool(detect_payload.get("has_brain_tools")),
            "git": detect_payload.get("git", {}),
        },
        "skill_sync": {
            "status": skill_sync.get("status", "unknown"),
            "all_in_sync": bool(skill_sync.get("all_in_sync")),
        },
        "doc_guard_status": doc_guard.get("status", "unknown"),
        "integrity": full_payload.get("integrity", {}),
        "catalog": compact_catalog_health(catalog),
        "frontier": compact_frontier_health(frontier),
        "next_actions": list(full_payload.get("next_actions", []) or []),
    }


def health(cwd: Path, *, mode: str = "compact") -> dict[str, Any]:
    detected = detect(cwd)
    workspace = Path(detected["brain_root"] or cwd.resolve()).resolve()
    skill_sync_result = _run_command(
        workspace,
        [sys.executable, "-m", "tools.brain.skill_install", "--check"],
    )
    integrity_result = _run_command(
        workspace,
        [sys.executable, "-m", "tools.brain.integrity_check", "--json"],
    )
    doc_guard_result = _run_command(
        workspace,
        [sys.executable, "-m", "tools.brain.doc_guard", "check"],
    )
    skill_sync = _parse_json_stdout(skill_sync_result)
    integrity = _parse_json_stdout(integrity_result)

    catalog_summary: dict[str, Any] = {"status": "unavailable"}
    frontier_summary: dict[str, Any] = {"status": "unavailable"}
    try:
        if str(workspace) not in sys.path:
            sys.path.insert(0, str(workspace))
        from tools.brain.platform import build_brain_catalog
        from tools.brain.adapters.daily_research_frontier import build_frontier_report

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
        frontier = build_frontier_report(workspace_root=workspace)
        frontier_summary = {
            "status": "warning" if frontier.get("brain_may_be_stale") else "ok",
            "brain_may_be_stale": bool(frontier.get("brain_may_be_stale")),
            "warnings": list(frontier.get("warnings", []) or []),
            "unregistered_latest_tags": list(frontier.get("unregistered_latest_tags", []) or []),
            "unregistered_latest_output_details": list(frontier.get("unregistered_latest_output_details", []) or []),
        }
    except Exception as exc:
        catalog_summary = {"status": "error", "error": str(exc)}
        frontier_summary = {"status": "error", "error": str(exc)}

    next_actions: list[str] = []
    if not detected["has_brain"]:
        next_actions.append("init_brain")
    if not skill_sync.get("all_in_sync", False):
        next_actions.append("run_skill_install")
    integrity_summary = _summary_counts(integrity)
    if integrity_summary["error_count"]:
        next_actions.append("fix_integrity_errors")
    if frontier_summary.get("brain_may_be_stale"):
        next_actions.append("review_frontier_reconciliation")
    if not next_actions:
        next_actions.append("run_capsule")

    payload = {
        "status": "ok" if detected["status"] == "ok" and integrity_result["returncode"] == 0 else "warning",
        "mode": "full",
        "detect": detected,
        "skill_sync": {
            "status": "ok" if skill_sync_result["returncode"] == 0 else "failed",
            "all_in_sync": bool(skill_sync.get("all_in_sync", False)),
            "skills": list(skill_sync.get("skills", []) or []),
        },
        "doc_guard": {
            "status": "ok" if doc_guard_result["returncode"] == 0 else "failed",
            "returncode": doc_guard_result["returncode"],
            "stdout_tail": doc_guard_result["stdout_tail"],
            "stderr_tail": doc_guard_result["stderr_tail"],
        },
        "integrity": integrity_summary,
        "catalog": catalog_summary,
        "frontier": frontier_summary,
        "next_actions": next_actions,
    }
    return payload if str(mode or "compact").lower() == "full" else _compact_health(payload)


def _title_from_brain_id(brain_id: str) -> str:
    return " ".join(part.capitalize() for part in str(brain_id or "project").replace("-", "_").split("_") if part)


def init_brain(cwd: Path, brain_id: str) -> dict[str, Any]:
    resolved = cwd.resolve()
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
    title = _title_from_brain_id(brain_id)
    brain_dir.mkdir(parents=True, exist_ok=True)
    (brain_dir / "references").mkdir(parents=True, exist_ok=True)
    for filename, template in CORE_DOCS.items():
        path = brain_dir / filename
        if not path.exists():
            path.write_text(template.format(title=title), encoding="utf-8", newline="\n")
    manifest = {
        "brain_type": "project",
        "brain_id": brain_id,
        "entrypoint": "brain/identity_layer.md",
        "read_order": [
            "brain/identity_layer.md",
            "brain/state_center.md",
            "brain/knowledge_center.md",
            "brain/brain_architecture.md",
            "brain/operations_center.md",
            "brain/governance_layer.md",
        ],
        "write_routes": {
            "state": "brain/state_center.md",
            "knowledge": "brain/knowledge_center.md",
            "operations": "brain/operations_center.md",
            "governance": "brain/governance_layer.md",
            "episodic": "brain/episodic_memory.md",
            "references": "brain/references/",
        },
    }
    if not manifest_path.exists():
        manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8", newline="\n")
    return {
        "status": "ok",
        "mutation_allowed": True,
        "brain_id": brain_id,
        "brain_manifest": str(manifest_path.resolve()),
        "created_paths": [str((brain_dir / name).resolve()) for name in [*CORE_DOCS, "brain_manifest.json"]],
    }


def capsule(cwd: Path, task: str, intent: str, *, verbosity: str = "lite") -> dict[str, Any]:
    detected = detect(cwd)
    if detected["has_brain_tools"]:
        command = [
            sys.executable,
            "-m",
            "tools.brain.workflow",
            "capsule",
            "--task",
            task,
            "--workflow",
            "auto",
            "--intent",
            intent,
            "--verbosity",
            verbosity,
            "--json",
        ]
        result = subprocess.run(
            command,
            cwd=detected["brain_root"] or str(cwd.resolve()),
            capture_output=True,
            text=True,
            encoding="utf-8",
            check=False,
        )
        if result.returncode == 0:
            return json.loads(result.stdout)
        return {
            "status": "error",
            "error": result.stderr.strip(),
            "command": command,
            "detect": detected,
            "health_summary": health(cwd, mode="compact"),
        }
    mutation_allowed = intent != "mutate" or (not detected["git"]["is_git_repo"] or detected["git"]["on_main"])
    blockers = [] if mutation_allowed else ["not_on_main_for_mutation"]
    if not detected["has_brain"]:
        blockers.append("missing_brain")
    return {
        "schema_version": 1,
        "status": "minimal",
        "task": task,
        "intent": intent,
        "detect": detected,
        "mutation_allowed": mutation_allowed,
        "preflight_blockers": blockers,
        "next_actions": detected["next_actions"],
        "health_summary": health(cwd, mode="compact"),
    }


def _owner_brain(cwd: Path, fallback: str) -> str:
    brain_root = _find_brain_root(cwd.resolve())
    manifest = (brain_root or cwd.resolve()) / "brain" / "brain_manifest.json"
    if not manifest.exists():
        return fallback
    try:
        payload = json.loads(manifest.read_text(encoding="utf-8-sig"))
    except Exception:
        return fallback
    return str(payload.get("brain_id") or payload.get("brain_type") or fallback)


def _runtime_learning_dir(cwd: Path) -> Path:
    resolved = cwd.resolve()
    brain_root = _find_brain_root(resolved) or resolved
    return brain_root / "brain" / "output" / "runtime_learning"


def _index_path(cwd: Path) -> Path:
    return _runtime_learning_dir(cwd) / "runtime_learning_index.json"


def _load_learning_index(cwd: Path) -> dict[str, Any]:
    path = _index_path(cwd)
    if not path.exists():
        return {"schema_version": 1, "proposals": []}
    try:
        payload = json.loads(path.read_text(encoding="utf-8-sig"))
    except Exception:
        return {"schema_version": 1, "proposals": []}
    if not isinstance(payload, dict):
        return {"schema_version": 1, "proposals": []}
    proposals = payload.get("proposals", [])
    if not isinstance(proposals, list):
        proposals = []
    return {"schema_version": int(payload.get("schema_version", 1) or 1), "proposals": proposals}


def _write_learning_index(cwd: Path, payload: dict[str, Any]) -> Path:
    out_dir = _runtime_learning_dir(cwd)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "runtime_learning_index.json"
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8", newline="\n")
    return path


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
    resolved = cwd.resolve()
    out_dir = _runtime_learning_dir(resolved)
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    safe_title = "".join(ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in title.lower()).strip("_") or "proposal"
    proposal_id = f"{stamp}_{safe_title}"
    created_at = datetime.now().astimezone().isoformat(timespec="seconds")
    payload = build_proposal_payload(
        proposal_id=proposal_id,
        title=title,
        created_at=created_at,
        status=status,
        severity=severity,
        owner_brain=owner_brain or _owner_brain(resolved, "workspace"),
        writeback_target=writeback_target,
        target_layer=target_layer,
        related_task=related_task,
        trigger=trigger,
        evidence=evidence,
        recommendation=recommendation,
        requires_user_confirmation=requires_user_confirmation,
        suggested_tests=list(suggested_tests or []),
        lesson=lesson,
        root_cause=root_cause,
        supporting_events=supporting_events,
        anti_overfit_check=anti_overfit_check,
    )
    json_path = out_dir / f"{proposal_id}.json"
    md_path = out_dir / f"{proposal_id}.md"
    json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8", newline="\n")
    md_path.write_text(
        "\n".join(
            [
                f"# {title}",
                "",
                f"- Authority: `{payload['authority']}`",
                f"- Severity: `{payload['severity']}`",
                f"- Owner brain: `{payload['owner_brain']}`",
                f"- Writeback target: `{payload['writeback_target']}`",
                f"- Trigger: {trigger}",
                f"- Evidence: {evidence}",
                f"- Recommendation: {recommendation}",
                f"- Lesson: {payload['lesson']}",
                f"- Root cause: {payload['root_cause']}",
                "",
            ]
        ),
        encoding="utf-8",
        newline="\n",
    )
    index = _load_learning_index(resolved)
    proposals = [item for item in index.get("proposals", []) if isinstance(item, dict) and item.get("proposal_id") != proposal_id]
    proposals.append(
        {
            "proposal_id": proposal_id,
            "title": title,
            "status": status,
            "severity": severity,
            "target_layer": target_layer,
            "json_path": str(json_path.resolve()),
            "markdown_path": str(md_path.resolve()),
            "created_at": payload["created_at"],
        }
    )
    index["proposals"] = proposals
    _write_learning_index(resolved, index)
    return {
        "status": "ok",
        "proposal_id": proposal_id,
        "json_path": str(json_path.resolve()),
        "markdown_path": str(md_path.resolve()),
    }


def list_proposals(cwd: Path) -> dict[str, Any]:
    index = _load_learning_index(cwd)
    return {"status": "ok", "proposals": list(index.get("proposals", []) or [])}


def mark_proposal(cwd: Path, proposal_id: str, status: str) -> dict[str, Any]:
    normalized = str(status or "").strip().lower()
    if normalized not in {"proposed", "approved", "implemented", "rejected", "superseded"}:
        return {"status": "error", "error": f"unsupported proposal status: {status}"}
    index = _load_learning_index(cwd)
    proposals = list(index.get("proposals", []) or [])
    updated: dict[str, Any] | None = None
    for item in proposals:
        if isinstance(item, dict) and item.get("proposal_id") == proposal_id:
            item["status"] = normalized
            item["updated_at"] = datetime.now().astimezone().isoformat(timespec="seconds")
            updated = item
            json_path = Path(str(item.get("json_path", "")))
            if json_path.exists():
                try:
                    payload = json.loads(json_path.read_text(encoding="utf-8-sig"))
                    if isinstance(payload, dict):
                        payload["status"] = normalized
                        payload["updated_at"] = item["updated_at"]
                        json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8", newline="\n")
                except Exception:
                    pass
            break
    if updated is None:
        return {"status": "error", "error": f"proposal not found: {proposal_id}"}
    index["proposals"] = proposals
    _write_learning_index(cwd, index)
    return {"status": "ok", "proposal": updated}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Workspace brain runtime helper.")
    sub = parser.add_subparsers(dest="command", required=True)
    detect_parser = sub.add_parser("detect")
    detect_parser.add_argument("--cwd", default=".")
    init_parser = sub.add_parser("init")
    init_parser.add_argument("--cwd", default=".")
    init_parser.add_argument("--brain-id", required=True)
    capsule_parser = sub.add_parser("capsule")
    capsule_parser.add_argument("--cwd", default=".")
    capsule_parser.add_argument("--task", default="")
    capsule_parser.add_argument("--intent", choices=("read", "mutate", "writeback"), default="read")
    capsule_parser.add_argument("--verbosity", choices=("lite", "standard", "full"), default="lite")
    health_parser = sub.add_parser("health")
    health_parser.add_argument("--cwd", default=".")
    health_parser.add_argument("--mode", choices=("compact", "full"), default="compact")
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
        payload = init_brain(cwd, str(args.brain_id))
    elif args.command == "capsule":
        payload = capsule(cwd, str(args.task or ""), str(args.intent or "read"), verbosity=str(args.verbosity or "lite"))
    elif args.command == "health":
        payload = health(cwd, mode=str(args.mode or "compact"))
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
