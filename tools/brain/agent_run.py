from __future__ import annotations

import argparse
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from tools.brain.long_task_monitor import build_status, validate_project_namespace
from tools.brain.project_profiles import load_project_profile


DEFAULT_WORKSPACE_ROOT = Path(__file__).resolve().parents[2]


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8-sig"))
    except Exception as exc:
        return {"_load_error": str(exc)}
    return payload if isinstance(payload, dict) else {"_load_error": "payload is not a JSON object"}


def _safe_run_id(run_id: str) -> str:
    run = str(run_id or "").strip().replace("\\", "/").strip("/")
    if not run or "/" in run or ".." in run:
        raise ValueError("run_id must be a single non-empty path segment")
    return run


def _workspace_path(path: str | Path, *, workspace_root: Path) -> Path:
    candidate = Path(str(path))
    return candidate if candidate.is_absolute() else workspace_root / candidate


def _replace_run_id(value: str, run_id: str) -> str:
    return str(value or "").replace("<run_id>", run_id).replace("\\", "/")


def build_run_paths(
    *,
    project_id: str,
    run_id: str,
    workspace_root: str | Path | None = None,
) -> dict[str, str]:
    project = str(project_id or "").strip()
    run = _safe_run_id(run_id)
    root = Path(workspace_root) if workspace_root is not None else DEFAULT_WORKSPACE_ROOT
    profile = load_project_profile(project)
    namespace = profile.get("process_namespace") if isinstance(profile.get("process_namespace"), dict) else {}
    run_root = _replace_run_id(str(namespace.get("run_path_template", "") or ""), run)
    if not run_root:
        body_root = str(profile.get("body_root", "") or project).replace("\\", "/").strip("/")
        run_root = f"{body_root}/output/agent_runs/{run}"
    paths = {
        "run_root": run_root,
        "pid_registry": _replace_run_id(str(namespace.get("pid_registry", "") or f"{run_root}/pid.json"), run),
        "stdout": _replace_run_id(str(namespace.get("stdout", "") or f"{run_root}/stdout.log"), run),
        "stderr": _replace_run_id(str(namespace.get("stderr", "") or f"{run_root}/stderr.log"), run),
        "progress": _replace_run_id(str(namespace.get("progress", "") or f"{run_root}/progress.json"), run),
        "summary": _replace_run_id(str(namespace.get("summary", "") or f"{run_root}/summary.json"), run),
    }
    namespace_check = validate_project_namespace(
        project_id=project,
        run_id=run,
        paths=list(paths.values()),
        workspace_root=root,
    )
    if namespace_check.get("status") != "ok":
        raise ValueError(str(namespace_check.get("reason") or "project_namespace_violation"))
    return paths


def _absolute_run_paths(paths: dict[str, str], *, workspace_root: Path) -> dict[str, Path]:
    return {key: _workspace_path(value, workspace_root=workspace_root) for key, value in paths.items()}


def register_run(
    *,
    project_id: str,
    run_id: str,
    pid: int,
    command: list[str] | None = None,
    run_tag: str = "",
    task: str = "",
    cwd: str | Path | None = None,
    status: str = "registered",
    workspace_root: str | Path | None = None,
) -> dict[str, Any]:
    project = str(project_id or "").strip()
    run = _safe_run_id(run_id)
    root = Path(workspace_root) if workspace_root is not None else DEFAULT_WORKSPACE_ROOT
    rel_paths = build_run_paths(project_id=project, run_id=run, workspace_root=root)
    abs_paths = _absolute_run_paths(rel_paths, workspace_root=root)
    abs_paths["run_root"].mkdir(parents=True, exist_ok=True)
    abs_paths["stdout"].parent.mkdir(parents=True, exist_ok=True)
    abs_paths["stderr"].parent.mkdir(parents=True, exist_ok=True)
    abs_paths["pid_registry"].parent.mkdir(parents=True, exist_ok=True)
    abs_paths["summary"].parent.mkdir(parents=True, exist_ok=True)
    abs_paths["stdout"].touch(exist_ok=True)
    abs_paths["stderr"].touch(exist_ok=True)
    payload = {
        "schema_version": 1,
        "project_id": project,
        "run_id": run,
        "run_tag": str(run_tag or ""),
        "task": str(task or ""),
        "pid": int(pid),
        "status": str(status or "registered"),
        "command": list(command or []),
        "cwd": str(cwd or root),
        "created_at": _utc_now(),
        "paths": rel_paths,
    }
    abs_paths["pid_registry"].write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    summary = {
        "schema_version": 1,
        "project_id": project,
        "run_id": run,
        "run_tag": str(run_tag or ""),
        "task": str(task or ""),
        "pid": int(pid),
        "status": str(status or "registered"),
        "created_at": payload["created_at"],
        "updated_at": payload["created_at"],
        "paths": rel_paths,
    }
    abs_paths["summary"].write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return {"schema_version": 1, "status": "ok", "run": payload, "summary": summary}


def launch_run(
    *,
    project_id: str,
    run_id: str,
    command: list[str],
    run_tag: str = "",
    task: str = "",
    cwd: str | Path | None = None,
    workspace_root: str | Path | None = None,
) -> dict[str, Any]:
    if not command:
        return {"schema_version": 1, "status": "blocked", "reason": "command_required"}
    root = Path(workspace_root) if workspace_root is not None else DEFAULT_WORKSPACE_ROOT
    rel_paths = build_run_paths(project_id=project_id, run_id=run_id, workspace_root=root)
    abs_paths = _absolute_run_paths(rel_paths, workspace_root=root)
    abs_paths["run_root"].mkdir(parents=True, exist_ok=True)
    stdout = abs_paths["stdout"].open("ab")
    stderr = abs_paths["stderr"].open("ab")
    try:
        process = subprocess.Popen(
            command,
            cwd=str(cwd or root),
            stdout=stdout,
            stderr=stderr,
            stdin=subprocess.DEVNULL,
        )
    finally:
        stdout.close()
        stderr.close()
    return register_run(
        project_id=project_id,
        run_id=run_id,
        pid=int(process.pid),
        command=command,
        run_tag=run_tag,
        task=task,
        cwd=cwd or root,
        status="launched",
        workspace_root=root,
    )


def list_runs(
    *,
    project_id: str,
    workspace_root: str | Path | None = None,
    limit: int = 20,
) -> dict[str, Any]:
    project = str(project_id or "").strip()
    root = Path(workspace_root) if workspace_root is not None else DEFAULT_WORKSPACE_ROOT
    profile = load_project_profile(project)
    namespace = profile.get("process_namespace") if isinstance(profile.get("process_namespace"), dict) else {}
    run_root_text = str(namespace.get("root", "") or "").replace("\\", "/").strip()
    run_root = _workspace_path(run_root_text, workspace_root=root)
    rows: list[dict[str, Any]] = []
    if run_root.exists():
        for item in run_root.iterdir():
            if not item.is_dir():
                continue
            summary = _load_json(item / "summary.json")
            pid_payload = _load_json(item / "pid.json")
            stat = item.stat()
            rows.append(
                {
                    "project_id": project,
                    "run_id": item.name,
                    "path": str(item.relative_to(root)).replace("\\", "/") if item.is_relative_to(root) else str(item),
                    "mtime": datetime.fromtimestamp(stat.st_mtime, tz=timezone.utc).isoformat(),
                    "pid": int((pid_payload.get("pid") or summary.get("pid") or 0) or 0),
                    "status": str(summary.get("status") or pid_payload.get("status") or ""),
                    "run_tag": str(summary.get("run_tag") or pid_payload.get("run_tag") or ""),
                    "task": str(summary.get("task") or pid_payload.get("task") or ""),
                    "has_pid": (item / "pid.json").exists(),
                    "has_summary": (item / "summary.json").exists(),
                    "has_stdout": (item / "stdout.log").exists(),
                    "has_stderr": (item / "stderr.log").exists(),
                    "has_progress": (item / "progress.json").exists(),
                }
            )
    rows.sort(key=lambda row: str(row.get("mtime") or ""), reverse=True)
    return {
        "schema_version": 1,
        "status": "ok",
        "project_id": project,
        "run_root": run_root_text,
        "run_count": len(rows),
        "runs": rows[: max(0, int(limit))],
    }


def run_status(
    *,
    project_id: str,
    run_id: str,
    workspace_root: str | Path | None = None,
) -> dict[str, Any]:
    project = str(project_id or "").strip()
    run = _safe_run_id(run_id)
    root = Path(workspace_root) if workspace_root is not None else DEFAULT_WORKSPACE_ROOT
    rel_paths = build_run_paths(project_id=project, run_id=run, workspace_root=root)
    abs_paths = _absolute_run_paths(rel_paths, workspace_root=root)
    pid_payload = _load_json(abs_paths["pid_registry"])
    pid = int((pid_payload.get("pid") or 0) or 0)
    payload = build_status(
        pid=pid,
        project_id=project,
        run_id=run,
        progress_path=abs_paths["progress"],
        stdout_path=abs_paths["stdout"],
        stderr_path=abs_paths["stderr"],
        artifact_dir=abs_paths["run_root"],
        workspace_root=root,
    )
    payload["run_paths"] = rel_paths
    payload["pid_registry"] = pid_payload
    return payload


def _print_payload(payload: dict[str, Any], *, as_json: bool) -> None:
    if as_json:
        print(json.dumps(payload, ensure_ascii=False, indent=2))
    else:
        print(json.dumps(payload, ensure_ascii=False))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Project-scoped agent run launcher and registry.")
    sub = parser.add_subparsers(dest="command", required=True)

    paths = sub.add_parser("paths", help="Print project-scoped run paths.")
    paths.add_argument("--project-id", required=True)
    paths.add_argument("--run-id", required=True)
    paths.add_argument("--workspace-root", default="")
    paths.add_argument("--json", action="store_true")

    register = sub.add_parser("register", help="Register an already-started PID under a project run namespace.")
    register.add_argument("--project-id", required=True)
    register.add_argument("--run-id", required=True)
    register.add_argument("--pid", type=int, required=True)
    register.add_argument("--run-tag", default="")
    register.add_argument("--task", default="")
    register.add_argument("--cwd", default="")
    register.add_argument("--command-text", default="")
    register.add_argument("--workspace-root", default="")
    register.add_argument("--json", action="store_true")

    launch = sub.add_parser("launch", help="Launch a command and register it under a project run namespace.")
    launch.add_argument("--project-id", required=True)
    launch.add_argument("--run-id", required=True)
    launch.add_argument("--run-tag", default="")
    launch.add_argument("--task", default="")
    launch.add_argument("--cwd", default="")
    launch.add_argument("--workspace-root", default="")
    launch.add_argument("--json", action="store_true")
    launch.add_argument("command_args", nargs=argparse.REMAINDER)

    list_parser = sub.add_parser("list", help="List recent runs for one project.")
    list_parser.add_argument("--project-id", required=True)
    list_parser.add_argument("--limit", type=int, default=20)
    list_parser.add_argument("--workspace-root", default="")
    list_parser.add_argument("--json", action="store_true")

    status_parser = sub.add_parser("status", help="Read status for one project run.")
    status_parser.add_argument("--project-id", required=True)
    status_parser.add_argument("--run-id", required=True)
    status_parser.add_argument("--workspace-root", default="")
    status_parser.add_argument("--json", action="store_true")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    try:
        if args.command == "paths":
            payload = {
                "schema_version": 1,
                "status": "ok",
                "project_id": str(args.project_id),
                "run_id": str(args.run_id),
                "paths": build_run_paths(
                    project_id=str(args.project_id),
                    run_id=str(args.run_id),
                    workspace_root=str(args.workspace_root or "") or None,
                ),
            }
        elif args.command == "register":
            command = str(args.command_text or "").split() if str(args.command_text or "").strip() else []
            payload = register_run(
                project_id=str(args.project_id),
                run_id=str(args.run_id),
                pid=int(args.pid),
                command=command,
                run_tag=str(args.run_tag or ""),
                task=str(args.task or ""),
                cwd=str(args.cwd or "") or None,
                workspace_root=str(args.workspace_root or "") or None,
            )
        elif args.command == "launch":
            command = list(args.command_args or [])
            if command and command[0] == "--":
                command = command[1:]
            payload = launch_run(
                project_id=str(args.project_id),
                run_id=str(args.run_id),
                command=command,
                run_tag=str(args.run_tag or ""),
                task=str(args.task or ""),
                cwd=str(args.cwd or "") or None,
                workspace_root=str(args.workspace_root or "") or None,
            )
        elif args.command == "list":
            payload = list_runs(
                project_id=str(args.project_id),
                limit=int(args.limit),
                workspace_root=str(args.workspace_root or "") or None,
            )
        else:
            payload = run_status(
                project_id=str(args.project_id),
                run_id=str(args.run_id),
                workspace_root=str(args.workspace_root or "") or None,
            )
    except Exception as exc:
        payload = {"schema_version": 1, "status": "blocked", "reason": "agent_run_error", "error": str(exc)}
    _print_payload(payload, as_json=bool(getattr(args, "json", False)))
    return 0 if payload.get("status") == "ok" else 2


if __name__ == "__main__":
    raise SystemExit(main())
