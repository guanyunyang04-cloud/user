from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import psutil


def _now() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _kill_tree(proc: psutil.Process) -> None:
    children: list[psutil.Process] = []
    try:
        children = proc.children(recursive=True)
    except psutil.Error:
        children = []
    for item in children:
        try:
            item.kill()
        except psutil.Error:
            pass
    try:
        proc.kill()
    except psutil.Error:
        pass


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run a command with a system available-memory guard.")
    parser.add_argument(
        "--min-available-gb",
        type=float,
        default=1.0,
        help="Kill the command if system available physical memory stays below this threshold (default: 1.0 GiB; 0 disables).",
    )
    parser.add_argument("--interval-seconds", type=float, default=0.75)
    parser.add_argument("--consecutive-breaches", type=int, default=2)
    parser.add_argument("--log-json", default="")
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args(argv)
    command = list(args.command)
    if command and command[0] == "--":
        command = command[1:]
    if not command:
        raise SystemExit("memory_guard requires a command after --")

    log_path = Path(args.log_json) if args.log_json else Path("memory_guard_log.json")
    min_available_bytes = int(float(args.min_available_gb) * (1024**3))
    started_at = _now()
    proc = subprocess.Popen(command)
    ps_proc = psutil.Process(proc.pid)
    min_available = None
    available_breaches = 0
    status = "running"
    exit_code: int | None = None
    while True:
        exit_code = proc.poll()
        vm = psutil.virtual_memory()
        available = int(vm.available)
        if min_available is None or available < min_available:
            min_available = int(available)
        if min_available_bytes > 0 and available < min_available_bytes:
            available_breaches += 1
        else:
            available_breaches = 0
        _write_json(
            log_path,
            {
                "status": status,
                "started_at": started_at,
                "updated_at": _now(),
                "command": command,
                "pid": int(proc.pid),
                "min_available_gb": float(args.min_available_gb),
                "current_available_gb": available / (1024**3),
                "lowest_available_gb": (float(min_available) / (1024**3)) if min_available is not None else None,
                "available_breaches": int(available_breaches),
                "exit_code": exit_code,
            },
        )
        if exit_code is not None:
            status = "completed" if exit_code == 0 else "failed"
            break
        if available_breaches >= int(args.consecutive_breaches):
            status = "killed_low_available_memory"
            _kill_tree(ps_proc)
            exit_code = 137
            break
        time.sleep(max(float(args.interval_seconds), 0.1))

    _write_json(
        log_path,
        {
            "status": status,
            "started_at": started_at,
            "finished_at": _now(),
            "command": command,
            "pid": int(proc.pid),
            "min_available_gb": float(args.min_available_gb),
            "lowest_available_gb": (float(min_available) / (1024**3)) if min_available is not None else None,
            "exit_code": int(exit_code if exit_code is not None else 1),
        },
    )
    return int(exit_code if exit_code is not None else 1)


if __name__ == "__main__":
    sys.exit(main())
