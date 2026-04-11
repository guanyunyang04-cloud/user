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

from daily_research.deep_alpha.experiment_guardrails import resolve_project_python_executable, runtime_metadata


PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_ROOT = PROJECT_ROOT / "daily_research" / "output"
TOOLS_ROOT = PROJECT_ROOT / "daily_research" / "tools"
DEFAULT_PYTHON = resolve_project_python_executable(sys.executable)

FORMAL_ROOT_TAG = "short_alpha_policy_v2_family_formal_review_20260411_r1"
RECENT_ROOT_TAG = "short_alpha_policy_v2_family_recent_eval_20260411_r1"
STATUS_PATH = OUTPUT_ROOT / "short_alpha_policy_v2_family_followthrough_20260411_r1_status.json"
LOGICAL_ROOT = OUTPUT_ROOT / "short_alpha_policy_v2_family_followthrough_20260411_r1"
CONSTRAINED_PREFIX = "policy_v2_family"
CONSTRAINED_ROOT_TAG = "short_alpha_policy_v2_family_constrained_execution_review_20260411_r2"
CURRENT_FORMAL_RUN_DIR = (
    OUTPUT_ROOT / "short_alpha_short_horizon_expert_review_20260406_r2_fullbudget" / "runs" / "short_expert_monthly_v1"
)
EXISTING_POLICY_V2_RUN_DIR = OUTPUT_ROOT / "short_alpha_policy_v2_review_20260410_r1" / "runs" / "short_expert_policy_v2"
FORMAL_PID_PATH = OUTPUT_ROOT / FORMAL_ROOT_TAG / "formal_review_pid.txt"
FORMAL_SUMMARY_PATH = OUTPUT_ROOT / FORMAL_ROOT_TAG / "summary.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Wait for the formal policy_v2 family review, then run the foreground follow-through chain.")
    parser.add_argument("--python-executable", default=DEFAULT_PYTHON)
    return parser.parse_args()


def _write_status(payload: dict[str, Any]) -> None:
    STATUS_PATH.parent.mkdir(parents=True, exist_ok=True)
    STATUS_PATH.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _summary_json(root_tag: str) -> Path:
    return OUTPUT_ROOT / root_tag / "summary.json"


def _process_alive(pid: int) -> bool:
    try:
        if pid <= 0:
            return False
        result = subprocess.run(
            [
                "powershell",
                "-NoLogo",
                "-NoProfile",
                "-Command",
                f"$p = Get-Process -Id {pid} -ErrorAction SilentlyContinue; if ($p) {{ 'alive' }}",
            ],
            check=False,
            capture_output=True,
            text=True,
            cwd=str(PROJECT_ROOT),
        )
        return "alive" in (result.stdout or "")
    except Exception:
        return False


def _read_formal_pid() -> int | None:
    if not FORMAL_PID_PATH.exists():
        return None
    try:
        return int((FORMAL_PID_PATH.read_text(encoding="utf-8") or "").strip())
    except Exception:
        return None


def _wait_for_formal_summary(status: dict[str, Any], poll_seconds: int = 30) -> None:
    status["current_step"] = "wait_for_formal_family_review"
    status.setdefault("steps", []).append(
        {
            "name": "wait_for_formal_family_review",
            "status": "running",
            "started_at": datetime.now().isoformat(timespec="seconds"),
            "formal_summary_path": str(FORMAL_SUMMARY_PATH),
            "formal_pid_path": str(FORMAL_PID_PATH),
            **runtime_metadata(),
        }
    )
    _write_status(status)

    while True:
        if FORMAL_SUMMARY_PATH.exists():
            status["steps"][-1]["status"] = "completed"
            status["steps"][-1]["completed_at"] = datetime.now().isoformat(timespec="seconds")
            _write_status(status)
            return

        pid = _read_formal_pid()
        status["steps"][-1]["last_poll_at"] = datetime.now().isoformat(timespec="seconds")
        status["steps"][-1]["formal_pid"] = pid
        status["steps"][-1]["formal_pid_alive"] = bool(pid and _process_alive(pid))
        _write_status(status)

        if pid and _process_alive(pid):
            time.sleep(poll_seconds)
            continue

        raise RuntimeError(
            f"Formal review summary not found at {FORMAL_SUMMARY_PATH} and no live formal process is attached."
        )


def _run_step(name: str, command: list[str], status: dict[str, Any]) -> None:
    status["current_step"] = name
    status.setdefault("steps", []).append(
        {
            "name": name,
            "status": "running",
            "started_at": datetime.now().isoformat(timespec="seconds"),
            "command": command,
            **runtime_metadata(),
        }
    )
    _write_status(status)
    subprocess.run(command, check=True, cwd=str(PROJECT_ROOT))
    status["steps"][-1]["status"] = "completed"
    status["steps"][-1]["completed_at"] = datetime.now().isoformat(timespec="seconds")
    _write_status(status)


def _resolve_family_winner_run_dir(formal_summary: dict[str, Any]) -> Path:
    winner_name = str(formal_summary.get("family_formal_winner_profile_name", "") or "").strip()
    if not winner_name:
        raise RuntimeError("Formal family review did not produce family_formal_winner_profile_name.")
    if winner_name == "short_expert_policy_v2" and (EXISTING_POLICY_V2_RUN_DIR / "metrics.json").exists():
        return EXISTING_POLICY_V2_RUN_DIR
    candidate = OUTPUT_ROOT / FORMAL_ROOT_TAG / "runs" / winner_name
    if not (candidate / "metrics.json").exists():
        raise FileNotFoundError(f"Winner run_dir missing metrics.json: {candidate}")
    return candidate


def main() -> None:
    args = parse_args()
    python_executable = str(args.python_executable)
    LOGICAL_ROOT.mkdir(parents=True, exist_ok=True)
    status: dict[str, Any] = {
        "started_at": datetime.now().isoformat(timespec="seconds"),
        "current_step": "starting",
        "status": "running",
        "formal_root_tag": FORMAL_ROOT_TAG,
        "constrained_root_tag": CONSTRAINED_ROOT_TAG,
        "recent_root_tag": RECENT_ROOT_TAG,
        "steps": [],
        **runtime_metadata(),
    }
    _write_status(status)

    try:
        _wait_for_formal_summary(status)

        formal_summary = json.loads(_summary_json(FORMAL_ROOT_TAG).read_text(encoding="utf-8"))
        winner_name = str(formal_summary.get("family_formal_winner_profile_name", "") or "").strip()
        winner_run_dir = _resolve_family_winner_run_dir(formal_summary)
        status["family_formal_winner_profile_name"] = winner_name
        status["family_formal_winner_run_dir"] = str(winner_run_dir)
        _write_status(status)

        _run_step(
            "constrained_execution_review",
            [
                python_executable,
                str(TOOLS_ROOT / "policy_v2_family_constrained_execution_review.py"),
                "--family-formal-summary",
                str(_summary_json(FORMAL_ROOT_TAG)),
                "--output-root",
                str(OUTPUT_ROOT),
                "--root-tag",
                CONSTRAINED_ROOT_TAG,
                "--current-formal-run-dir",
                str(CURRENT_FORMAL_RUN_DIR),
            ],
            status,
        )

        _run_step(
            "recent_family_eval",
            [
                python_executable,
                str(TOOLS_ROOT / "policy_v2_family_recent_eval.py"),
                "--output-root",
                str(OUTPUT_ROOT),
                "--root-tag",
                RECENT_ROOT_TAG,
                "--python-executable",
                python_executable,
            ],
            status,
        )

        _run_step(
            "project_consistency_check",
            [python_executable, str(TOOLS_ROOT / "project_consistency_check.py")],
            status,
        )
        _run_step(
            "doc_guard_check",
            [python_executable, str(TOOLS_ROOT / "doc_guard.py"), "check"],
            status,
        )

        status["current_step"] = "completed"
        status["status"] = "completed"
        status["completed_at"] = datetime.now().isoformat(timespec="seconds")
        _write_status(status)
    except Exception as exc:
        status["current_step"] = "failed"
        status["status"] = "failed"
        status["failed_at"] = datetime.now().isoformat(timespec="seconds")
        status["error"] = str(exc)
        _write_status(status)
        raise


if __name__ == "__main__":
    main()
