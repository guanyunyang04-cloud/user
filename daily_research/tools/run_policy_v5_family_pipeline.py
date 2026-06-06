from __future__ import annotations

import argparse
import json
import subprocess
import sys
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

FORMAL_ROOT_TAG = "short_alpha_policy_v5_family_formal_review_20260412_r1"
CONSTRAINED_ROOT_TAG = "short_alpha_policy_v5_family_constrained_execution_review_20260412_r1"
RECENT_ROOT_TAG = "short_alpha_policy_v5_family_recent_eval_20260412_r1"
BREAKDOWN_ROOT_TAG = "short_alpha_policy_v5_family_formal_loss_breakdown_20260412_r1"
STATUS_PATH = OUTPUT_ROOT / "short_alpha_policy_v5_family_pipeline_20260412_r1_status.json"
RECENT_MODEL_ROOT_TAG = "short_alpha_recent_model_protocol_20260412_r1"
CURRENT_FORMAL_RUN_DIR = (
    OUTPUT_ROOT / "short_alpha_short_horizon_expert_review_20260406_r2_fullbudget" / "runs" / "short_expert_monthly_v1"
)
FAMILY_PROFILES = "short_expert_policy_v5a,short_expert_policy_v5b,short_expert_policy_v5c"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the policy_v5 family foreground pipeline under the current discipline.")
    parser.add_argument("--python-executable", default=DEFAULT_PYTHON)
    return parser.parse_args()


def _write_status(payload: dict[str, Any]) -> None:
    STATUS_PATH.parent.mkdir(parents=True, exist_ok=True)
    STATUS_PATH.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


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


def main() -> None:
    args = parse_args()
    python_executable = str(args.python_executable)
    status: dict[str, Any] = {
        "started_at": datetime.now().isoformat(timespec="seconds"),
        "current_step": "starting",
        "status": "running",
        "formal_root_tag": FORMAL_ROOT_TAG,
        "constrained_root_tag": CONSTRAINED_ROOT_TAG,
        "recent_root_tag": RECENT_ROOT_TAG,
        "recent_model_root_tag": RECENT_MODEL_ROOT_TAG,
        "breakdown_root_tag": BREAKDOWN_ROOT_TAG,
        "steps": [],
        **runtime_metadata(),
    }
    _write_status(status)

    try:
        _run_step(
            "formal_family_review",
            [
                python_executable,
                str(TOOLS_ROOT / "policy_v5_family_formal_review.py"),
                "--python-executable",
                python_executable,
                "--output-root",
                str(OUTPUT_ROOT),
                "--root-tag",
                FORMAL_ROOT_TAG,
            ],
            status,
        )

        _run_step(
            "constrained_execution_review",
            [
                python_executable,
                str(TOOLS_ROOT / "policy_v2_family_constrained_execution_review.py"),
                "--family-formal-summary",
                str(OUTPUT_ROOT / FORMAL_ROOT_TAG / "summary.json"),
                "--family-profiles",
                FAMILY_PROFILES,
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
                str(TOOLS_ROOT / "policy_v5_family_recent_eval.py"),
                "--output-root",
                str(OUTPUT_ROOT),
                "--root-tag",
                RECENT_ROOT_TAG,
                "--python-executable",
                python_executable,
                "--recent-model-root-tag",
                RECENT_MODEL_ROOT_TAG,
            ],
            status,
        )

        _run_step(
            "formal_loss_breakdown",
            [
                python_executable,
                str(TOOLS_ROOT / "policy_family_formal_loss_breakdown.py"),
                "--formal-summary",
                str(OUTPUT_ROOT / FORMAL_ROOT_TAG / "summary.json"),
                "--recent-summary",
                str(OUTPUT_ROOT / RECENT_ROOT_TAG / "summary.json"),
                "--constrained-summary",
                str(OUTPUT_ROOT / CONSTRAINED_ROOT_TAG / "summary.json"),
                "--focus-profiles",
                FAMILY_PROFILES,
                "--output-root",
                str(OUTPUT_ROOT),
                "--root-tag",
                BREAKDOWN_ROOT_TAG,
            ],
            status,
        )

        _run_step("project_consistency_check", [python_executable, str(TOOLS_ROOT / "project_consistency_check.py")], status)
        _run_step("doc_guard_check", [python_executable, "-m", "tools.brain.doc_guard", "check"], status)

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
