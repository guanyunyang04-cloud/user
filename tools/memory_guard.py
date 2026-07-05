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


def _process_tree_rss(proc: psutil.Process) -> tuple[int, list[dict[str, Any]]]:
    total = 0
    rows: list[dict[str, Any]] = []
    processes = [proc]
    try:
        processes.extend(proc.children(recursive=True))
    except psutil.Error:
        pass
    for item in processes:
        try:
            mem = item.memory_info()
            rss = int(mem.rss)
            total += rss
            rows.append(
                {
                    "pid": int(item.pid),
                    "name": item.name(),
                    "rss": rss,
                    "rss_gb": rss / (1024**3),
                }
            )
        except psutil.Error:
            continue
    return total, rows


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
    parser = argparse.ArgumentParser(description="Run a command with a process-tree RSS memory guard.")
    parser.add_argument(
        "--max-rss-gb",
        type=float,
        default=0.0,
        help="Kill the command if process-tree RSS exceeds this threshold. 0 disables this guard.",
    )
    parser.add_argument(
        "--min-available-gb",
        type=float,
        default=0.0,
        help="Kill the command if system available physical memory stays below this threshold. 0 disables this guard.",
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
    limit_bytes = int(float(args.max_rss_gb) * (1024**3))
    min_available_bytes = int(float(args.min_available_gb) * (1024**3))
    started_at = _now()
    proc = subprocess.Popen(command)
    ps_proc = psutil.Process(proc.pid)
    peak_rss = 0
    min_available = None
    peak_rows: list[dict[str, Any]] = []
    breaches = 0
    available_breaches = 0
    status = "running"
    exit_code: int | None = None
    while True:
        exit_code = proc.poll()
        try:
            rss, rows = _process_tree_rss(ps_proc)
        except psutil.Error:
            rss, rows = 0, []
        vm = psutil.virtual_memory()
        available = int(vm.available)
        if min_available is None or available < min_available:
            min_available = int(available)
        if rss > peak_rss:
            peak_rss = int(rss)
            peak_rows = rows
        if limit_bytes > 0 and rss > limit_bytes:
            breaches += 1
        else:
            breaches = 0
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
                "limit_gb": float(args.max_rss_gb),
                "min_available_gb": float(args.min_available_gb),
                "current_rss_gb": rss / (1024**3),
                "peak_rss_gb": peak_rss / (1024**3),
                "current_available_gb": available / (1024**3),
                "lowest_available_gb": (float(min_available) / (1024**3)) if min_available is not None else None,
                "breaches": int(breaches),
                "available_breaches": int(available_breaches),
                "processes": rows,
                "peak_processes": peak_rows,
                "exit_code": exit_code,
            },
        )
        if exit_code is not None:
            status = "completed" if exit_code == 0 else "failed"
            break
        if breaches >= int(args.consecutive_breaches):
            status = "killed_memory_limit"
            _kill_tree(ps_proc)
            exit_code = 137
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
            "limit_gb": float(args.max_rss_gb),
            "min_available_gb": float(args.min_available_gb),
            "peak_rss_gb": peak_rss / (1024**3),
            "lowest_available_gb": (float(min_available) / (1024**3)) if min_available is not None else None,
            "peak_processes": peak_rows,
            "exit_code": int(exit_code if exit_code is not None else 1),
        },
    )
    return int(exit_code if exit_code is not None else 1)


if __name__ == "__main__":
    sys.exit(main())
