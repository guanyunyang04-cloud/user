from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from daily_research.deep_alpha.experiment_guardrails import resolve_project_python_executable


PROJECT_ROOT = Path(__file__).resolve().parents[2]
TOOLS_ROOT = PROJECT_ROOT / "daily_research" / "tools"
OUTPUT_ROOT = PROJECT_ROOT / "daily_research" / "output"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Monitor the independent recent-model protocol end-to-end without interrupting "
            "the current strongest-model training process."
        )
    )
    parser.add_argument("--python-executable", default=resolve_project_python_executable(sys.executable))
    parser.add_argument("--wait-pid", type=int, default=0)
    parser.add_argument("--poll-seconds", type=int, default=120)
    parser.add_argument("--recent-model-root-tag", default="short_alpha_recent_model_protocol_20260412_r1")
    parser.add_argument("--status-json", default=str(OUTPUT_ROOT / "recent_protocol_completion_monitor_20260410_r1.json"))
    parser.add_argument("--log-path", default=str(OUTPUT_ROOT / "recent_protocol_completion_monitor_20260410_r1.log"))
    return parser.parse_args()


def _timestamp() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def _write_log(log_path: Path, message: str) -> None:
    line = f"[{_timestamp()}] {message}\n"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as fh:
        fh.write(line)
    print(line, end="")


def _write_status(status_path: Path, payload: dict[str, Any]) -> None:
    status_path.parent.mkdir(parents=True, exist_ok=True)
    status_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _pid_exists(pid: int) -> bool:
    if pid <= 0:
        return False
    result = subprocess.run(
        ["tasklist", "/FI", f"PID eq {pid}"],
        capture_output=True,
        text=True,
        check=False,
    )
    text = (result.stdout or "") + "\n" + (result.stderr or "")
    return str(pid) in text


def _run(command: list[str], *, log_path: Path, status_path: Path, step: str) -> None:
    _write_log(log_path, f"START {step}: {' '.join(command)}")
    _write_status(status_path, {"stage": step, "running": True, "command": command, "updated_at": _timestamp()})
    subprocess.run(command, cwd=str(PROJECT_ROOT), check=True)
    _write_log(log_path, f"DONE {step}")


def main() -> None:
    args = parse_args()
    python_executable = str(args.python_executable)
    wait_pid = int(args.wait_pid or 0)
    poll_seconds = max(int(args.poll_seconds or 120), 30)
    recent_model_root_tag = str(args.recent_model_root_tag).strip()
    status_path = Path(args.status_json).expanduser().resolve()
    log_path = Path(args.log_path).expanduser().resolve()

    _write_log(log_path, "monitor boot")
    _write_status(
        status_path,
        {
            "stage": "boot",
            "running": True,
            "wait_pid": wait_pid,
            "recent_model_root_tag": recent_model_root_tag,
            "updated_at": _timestamp(),
        },
    )

    if wait_pid > 0:
        _write_log(log_path, f"waiting for existing strongest-model process pid={wait_pid} to exit")
        while _pid_exists(wait_pid):
            _write_status(
                status_path,
                {
                    "stage": "waiting_for_strongest_training",
                    "running": True,
                    "wait_pid": wait_pid,
                    "recent_model_root_tag": recent_model_root_tag,
                    "updated_at": _timestamp(),
                },
            )
            time.sleep(poll_seconds)
        _write_log(log_path, f"pid={wait_pid} exited")

    strongest_cmd = [
        python_executable,
        str(TOOLS_ROOT / "refresh_strongest_model_verdict.py"),
        "--python-executable",
        python_executable,
        "--recent-model-root-tag",
        recent_model_root_tag,
    ]
    _run(strongest_cmd, log_path=log_path, status_path=status_path, step="refresh_strongest_model_verdict")

    policy_v2_cmd = [
        python_executable,
        str(TOOLS_ROOT / "policy_v2_recent_eval.py"),
        "--python-executable",
        python_executable,
        "--recent-model-root-tag",
        recent_model_root_tag,
    ]
    _run(policy_v2_cmd, log_path=log_path, status_path=status_path, step="policy_v2_recent_eval")

    policy_v3_cmd = [
        python_executable,
        str(TOOLS_ROOT / "policy_v3_recent_eval.py"),
        "--python-executable",
        python_executable,
        "--recent-model-root-tag",
        recent_model_root_tag,
    ]
    _run(policy_v3_cmd, log_path=log_path, status_path=status_path, step="policy_v3_recent_eval")

    consistency_cmd = [python_executable, str(TOOLS_ROOT / "project_consistency_check.py")]
    _run(consistency_cmd, log_path=log_path, status_path=status_path, step="project_consistency_check")

    doc_guard_cmd = [python_executable, "-m", "tools.brain.doc_guard", "check"]
    _run(doc_guard_cmd, log_path=log_path, status_path=status_path, step="doc_guard_check")

    _write_status(
        status_path,
        {
            "stage": "completed",
            "running": False,
            "recent_model_root_tag": recent_model_root_tag,
            "updated_at": _timestamp(),
        },
    )
    _write_log(log_path, "monitor completed")


if __name__ == "__main__":
    main()
