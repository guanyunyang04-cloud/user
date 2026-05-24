from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any


CORE_DOCS = {
    "identity_layer.md": "# {title} 身份层\n\n- 本项目脑区由 `workspace-brain` runtime 初始化。\n- 本文件保存项目身份、目标和边界。\n",
    "state_center.md": "# {title} 状态中枢\n\n- 当前状态：新建脑区，等待首次接管写入事实。\n- 默认分支纪律：repo-tracked mutation 优先在 `main` 分支执行。\n",
    "knowledge_center.md": "# {title} 知识中枢\n\n- 稳定事实、硬规则和可复用教训写入这里。\n- 长证据和过程细节下沉到 `brain/references/`。\n",
    "brain_architecture.md": "# {title} 脑区架构\n\n- 采用最小脑区结构：identity、state、knowledge、operations、governance、episodic。\n",
    "operations_center.md": "# {title} 操作中枢\n\n- 接管入口：先运行 brain runtime detect/capsule，再执行项目任务。\n",
    "governance_layer.md": "# {title} 治理层\n\n- 重大动作前区分事实、推断、假设和边界。\n- 自进化默认只生成 proposal，核心治理写回需要用户确认。\n",
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


def capsule(cwd: Path, task: str, intent: str) -> dict[str, Any]:
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
        return {"status": "error", "error": result.stderr.strip(), "command": command, "detect": detected}
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
    }


def proposal(cwd: Path, title: str, trigger: str, evidence: str, recommendation: str) -> dict[str, Any]:
    resolved = cwd.resolve()
    brain_root = _find_brain_root(resolved) or resolved
    out_dir = brain_root / "brain" / "output" / "runtime_learning"
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    safe_title = "".join(ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in title.lower()).strip("_") or "proposal"
    payload = {
        "schema_version": 1,
        "title": title,
        "created_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "authority": "requires_user_confirmation",
        "facts": [trigger],
        "inferences": [],
        "assumptions": [],
        "trigger_evidence": [evidence],
        "recommendation": recommendation,
        "suggested_write_routes": ["brain/references/", "brain/knowledge_center.md"],
        "suggested_tests": [],
        "risks": ["Core brain or skill changes must not be applied without explicit confirmation."],
    }
    json_path = out_dir / f"{stamp}_{safe_title}.json"
    md_path = out_dir / f"{stamp}_{safe_title}.md"
    json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8", newline="\n")
    md_path.write_text(
        "\n".join(
            [
                f"# {title}",
                "",
                f"- Authority: `{payload['authority']}`",
                f"- Trigger: {trigger}",
                f"- Evidence: {evidence}",
                f"- Recommendation: {recommendation}",
                "",
            ]
        ),
        encoding="utf-8",
        newline="\n",
    )
    return {"status": "ok", "json_path": str(json_path.resolve()), "markdown_path": str(md_path.resolve())}


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
    capsule_parser.add_argument("--intent", choices=("read", "mutate", "long_task", "writeback"), default="read")
    proposal_parser = sub.add_parser("proposal")
    proposal_parser.add_argument("--cwd", default=".")
    proposal_parser.add_argument("--title", required=True)
    proposal_parser.add_argument("--trigger", required=True)
    proposal_parser.add_argument("--evidence", required=True)
    proposal_parser.add_argument("--recommendation", required=True)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    cwd = Path(str(getattr(args, "cwd", ".") or "."))
    if args.command == "detect":
        payload = detect(cwd)
    elif args.command == "init":
        payload = init_brain(cwd, str(args.brain_id))
    elif args.command == "capsule":
        payload = capsule(cwd, str(args.task or ""), str(args.intent or "read"))
    elif args.command == "proposal":
        payload = proposal(cwd, str(args.title), str(args.trigger), str(args.evidence), str(args.recommendation))
    else:
        raise ValueError(f"Unsupported command: {args.command}")
    _print_json(payload)
    return 0 if payload.get("status") != "error" else 2


if __name__ == "__main__":
    raise SystemExit(main())
