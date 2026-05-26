from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any


TASK_NAME = "DailyResearchDailyPlan"
TASK_DESCRIPTION = "Daily Research post-close daily plan runner"


def _workspace_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _python_executable() -> str:
    return sys.executable


def _query_task() -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["schtasks", "/Query", "/TN", TASK_NAME, "/FO", "LIST", "/V"],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        check=False,
    )


def _parse_status(stdout: str) -> dict[str, str]:
    payload: dict[str, str] = {}
    for line in str(stdout or "").splitlines():
        if ":" not in line:
            continue
        key, value = line.split(":", 1)
        payload[key.strip()] = value.strip()
    return payload


def scheduler_status() -> dict[str, Any]:
    if os.name != "nt":
        return {
            "status": "unsupported",
            "installed": False,
            "enabled": False,
            "task_name": TASK_NAME,
            "detail": "Windows Task Scheduler is only available on Windows.",
        }
    result = _query_task()
    if result.returncode != 0:
        return {
            "status": "missing",
            "installed": False,
            "enabled": False,
            "task_name": TASK_NAME,
            "detail": (result.stderr or result.stdout or "").strip(),
        }
    parsed = _parse_status(result.stdout)
    scheduled_task_state = parsed.get("Scheduled Task State") or parsed.get("计划任务状态") or ""
    status_text = parsed.get("Status") or parsed.get("状态") or ""
    enabled = "disabled" not in f"{scheduled_task_state} {status_text}".lower() and "禁用" not in f"{scheduled_task_state} {status_text}"
    return {
        "status": "ok",
        "installed": True,
        "enabled": enabled,
        "task_name": TASK_NAME,
        "next_run_time": parsed.get("Next Run Time") or parsed.get("下次运行时间") or "",
        "last_run_time": parsed.get("Last Run Time") or parsed.get("上次运行时间") or "",
        "last_result": parsed.get("Last Result") or parsed.get("上次运行结果") or "",
        "raw": parsed,
    }


def install_task(*, time_text: str = "15:45") -> dict[str, Any]:
    if os.name != "nt":
        return {"status": "unsupported", "installed": False, "task_name": TASK_NAME}
    command = (
        f'"{_python_executable()}" -m daily_research.execution.daily_plan_runner '
        "--mode post-close --json"
    )
    result = subprocess.run(
        [
            "schtasks",
            "/Create",
            "/TN",
            TASK_NAME,
            "/TR",
            command,
            "/SC",
            "DAILY",
            "/ST",
            str(time_text),
            "/F",
            "/RL",
            "LIMITED",
        ],
        cwd=str(_workspace_root()),
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        check=False,
    )
    payload = scheduler_status()
    payload.update(
        {
            "install_returncode": int(result.returncode),
            "install_stdout": result.stdout.strip(),
            "install_stderr": result.stderr.strip(),
            "command": command,
            "time": str(time_text),
        }
    )
    if result.returncode != 0:
        payload["status"] = "error"
    return payload


def uninstall_task() -> dict[str, Any]:
    if os.name != "nt":
        return {"status": "unsupported", "installed": False, "task_name": TASK_NAME}
    result = subprocess.run(
        ["schtasks", "/Delete", "/TN", TASK_NAME, "/F"],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        check=False,
    )
    return {
        "status": "ok" if result.returncode == 0 else "missing",
        "installed": False,
        "task_name": TASK_NAME,
        "returncode": int(result.returncode),
        "stdout": result.stdout.strip(),
        "stderr": result.stderr.strip(),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Manage Windows Task Scheduler for daily execution.")
    subparsers = parser.add_subparsers(dest="command", required=True)
    install = subparsers.add_parser("install")
    install.add_argument("--time", default="15:45")
    install.add_argument("--json", action="store_true")
    status = subparsers.add_parser("status")
    status.add_argument("--json", action="store_true")
    uninstall = subparsers.add_parser("uninstall")
    uninstall.add_argument("--json", action="store_true")
    return parser


def _print(payload: dict[str, Any], *, as_json: bool) -> None:
    print(json.dumps(payload, ensure_ascii=False, indent=2 if as_json else None))


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "install":
        payload = install_task(time_text=args.time)
    elif args.command == "uninstall":
        payload = uninstall_task()
    else:
        payload = scheduler_status()
    _print(payload, as_json=bool(getattr(args, "json", False)))
    return 0 if payload.get("status") in {"ok", "missing", "unsupported"} else 1


if __name__ == "__main__":
    raise SystemExit(main())
