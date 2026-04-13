from __future__ import annotations

import argparse
import json
import sys

if __package__ in {None, ""}:
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from daily_research.deep_alpha.experiment_guardrails import resolve_project_python_executable
from daily_research.execution.app_runtime import append_event
from daily_research.execution.app_service import (
    build_doctor_payload,
    build_job_detail_payload,
    build_status_payload,
    launch_task_async,
    list_jobs_payload,
    list_tasks_payload,
    resume_task_async,
    resume_task_sync,
    run_task_sync,
    sanitize_passthrough_args,
    unlock_runtime,
)
from daily_research.execution.app_tasks import build_task_command, get_task_spec, list_task_lines
from daily_research.execution.entrypoint_utils import missing_runtime_dependency_error


def _print_payload(payload: object, *, as_json: bool) -> None:
    if as_json:
        print(json.dumps(payload, ensure_ascii=False, indent=2))
        return
    if isinstance(payload, dict):
        for key, value in payload.items():
            if isinstance(value, (dict, list)):
                print(f"{key}={json.dumps(value, ensure_ascii=False)}")
            else:
                print(f"{key}={value}")
        return
    if isinstance(payload, list):
        for item in payload:
            print(json.dumps(item, ensure_ascii=False))
        return
    print(str(payload))


def _handle_tasks(args: argparse.Namespace) -> int:
    if args.json:
        _print_payload(list_tasks_payload(), as_json=True)
        return 0
    print("\n".join(list_task_lines()))
    return 0


def _handle_status(args: argparse.Namespace) -> int:
    payload = build_status_payload(history_limit=args.history_limit)
    _print_payload(payload, as_json=args.json)
    return 0


def _handle_doctor(args: argparse.Namespace) -> int:
    payload = build_doctor_payload()
    _print_payload(payload, as_json=args.json)
    return 0 if payload.get("status") == "ok" else 1


def _handle_run(args: argparse.Namespace) -> int:
    task_name = str(args.task or "").strip()
    python_executable = resolve_project_python_executable(args.python_executable or sys.executable)
    passthrough_args = sanitize_passthrough_args(args.args)
    command = build_task_command(
        task_name=task_name,
        python_executable=python_executable,
        passthrough_args=passthrough_args,
    )
    if args.dry_run:
        _print_payload(
            {
                "task_name": task_name,
                "python_executable": python_executable,
                "command": command,
            },
            as_json=args.json,
        )
        return 0
    if args.background:
        result = launch_task_async(
            task_name=task_name,
            python_executable=python_executable,
            passthrough_args=passthrough_args,
            job_label=str(args.job_label or ""),
            force_unlock=bool(args.force_unlock),
        )
        _print_payload(result, as_json=args.json)
        return 0
    result = run_task_sync(
        task_name=task_name,
        python_executable=python_executable,
        passthrough_args=passthrough_args,
        job_label=str(args.job_label or ""),
        force_unlock=bool(args.force_unlock),
        echo_output=True,
    )
    _print_payload(result, as_json=args.json)
    return 0 if int(result.get("exit_code", 1)) == 0 else int(result.get("exit_code", 1))


def _handle_resume(args: argparse.Namespace) -> int:
    if args.background:
        result = resume_task_async(
            job_id=str(args.job_id or ""),
            job_label=str(args.job_label or ""),
            force_unlock=bool(args.force_unlock),
        )
        _print_payload(result, as_json=args.json)
        return 0
    result = resume_task_sync(
        job_id=str(args.job_id or ""),
        job_label=str(args.job_label or ""),
        force_unlock=bool(args.force_unlock),
        echo_output=True,
    )
    _print_payload(result, as_json=args.json)
    return 0 if int(result.get("exit_code", 1)) == 0 else int(result.get("exit_code", 1))


def _handle_tail(args: argparse.Namespace) -> int:
    payload = build_job_detail_payload(str(args.job_id), lines=args.lines)
    tail_payload = {
        "job_id": payload["job_id"],
        "task_name": payload["task_name"],
        "status": payload["status"],
        "stdout_tail": payload["stdout_tail"],
        "stderr_tail": payload["stderr_tail"],
    }
    _print_payload(tail_payload, as_json=args.json)
    return 0


def _handle_unlock(args: argparse.Namespace) -> int:
    payload = unlock_runtime(force=bool(args.force))
    _print_payload(payload, as_json=args.json)
    return 0


def _handle_jobs(args: argparse.Namespace) -> int:
    payload = list_jobs_payload(limit=args.limit)
    _print_payload(payload, as_json=args.json)
    return 0


def _handle_job(args: argparse.Namespace) -> int:
    payload = build_job_detail_payload(str(args.job_id), lines=args.lines)
    _print_payload(payload, as_json=args.json)
    return 0


def _handle_web(args: argparse.Namespace) -> int:
    try:
        from daily_research.execution.web_server import run_web_console
    except ModuleNotFoundError as exc:
        raise missing_runtime_dependency_error(
            exc,
            command_hint="python daily_research/execution/run_execution_app.py web --host 127.0.0.1 --port 8765",
        ) from exc

    run_web_console(
        host=str(args.host),
        port=int(args.port),
        reload=bool(args.reload),
    )
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="daily_research 执行侧统一应用入口。")
    subparsers = parser.add_subparsers(dest="command", required=True)

    tasks_parser = subparsers.add_parser("tasks", help="列出已注册的执行任务。")
    tasks_parser.add_argument("--json", action="store_true")

    status_parser = subparsers.add_parser("status", help="查看运行时状态、锁、manifest 和最近作业。")
    status_parser.add_argument("--history-limit", type=int, default=8)
    status_parser.add_argument("--json", action="store_true")

    doctor_parser = subparsers.add_parser("doctor", help="执行执行侧健康检查。")
    doctor_parser.add_argument("--json", action="store_true")

    jobs_parser = subparsers.add_parser("jobs", help="列出最近执行作业。")
    jobs_parser.add_argument("--limit", type=int, default=20)
    jobs_parser.add_argument("--json", action="store_true")

    job_parser = subparsers.add_parser("job", help="查看单个作业的完整元数据与日志尾部。")
    job_parser.add_argument("--job-id", required=True)
    job_parser.add_argument("--lines", type=int, default=80)
    job_parser.add_argument("--json", action="store_true")

    run_parser = subparsers.add_parser("run", help="在统一运行时下执行已注册任务。")
    run_parser.add_argument("--task", required=True)
    run_parser.add_argument("--job-label", default="")
    run_parser.add_argument("--python-executable", default="")
    run_parser.add_argument("--force-unlock", action="store_true")
    run_parser.add_argument("--dry-run", action="store_true")
    run_parser.add_argument("--background", action="store_true")
    run_parser.add_argument("--json", action="store_true")
    run_parser.add_argument("args", nargs=argparse.REMAINDER, help="跟在 `--` 之后的透传参数。")

    resume_parser = subparsers.add_parser("resume", help="按记录元数据恢复失败或阻塞的执行任务。")
    resume_parser.add_argument("--job-id", default="")
    resume_parser.add_argument("--job-label", default="")
    resume_parser.add_argument("--force-unlock", action="store_true")
    resume_parser.add_argument("--background", action="store_true")
    resume_parser.add_argument("--json", action="store_true")

    tail_parser = subparsers.add_parser("tail", help="查看记录作业的最新 stdout/stderr。")
    tail_parser.add_argument("--job-id", required=True)
    tail_parser.add_argument("--lines", type=int, default=40)
    tail_parser.add_argument("--json", action="store_true")

    unlock_parser = subparsers.add_parser("unlock", help="清理失效的 execution app 锁。")
    unlock_parser.add_argument("--force", action="store_true")
    unlock_parser.add_argument("--json", action="store_true")

    web_parser = subparsers.add_parser("web", help="启动本地执行 Web 控制台。")
    web_parser.add_argument("--host", default="127.0.0.1")
    web_parser.add_argument("--port", type=int, default=8765)
    web_parser.add_argument("--reload", action="store_true")

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    handlers = {
        "tasks": _handle_tasks,
        "status": _handle_status,
        "doctor": _handle_doctor,
        "jobs": _handle_jobs,
        "job": _handle_job,
        "run": _handle_run,
        "resume": _handle_resume,
        "tail": _handle_tail,
        "unlock": _handle_unlock,
        "web": _handle_web,
    }
    try:
        return handlers[str(args.command)](args)
    except Exception as exc:
        append_event("cli_error", command=str(args.command), detail=str(exc))
        print(f"execution_app_error={exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
