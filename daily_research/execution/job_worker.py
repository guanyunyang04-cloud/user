from __future__ import annotations

import argparse
import sys
from pathlib import Path

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from daily_research.execution import app_service
from daily_research.execution.app_runtime import append_event


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run one recorded execution app job in a detached worker.")
    parser.add_argument("--job-id", required=True)
    parser.add_argument("--force-unlock", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    job_id = str(args.job_id or "").strip()
    try:
        payload = app_service.run_recorded_job_sync(
            job_id=job_id,
            force_unlock=bool(args.force_unlock),
            echo_output=False,
        )
    except Exception as exc:
        append_event("detached_job_worker_error", job_id=job_id, detail=str(exc))
        print(f"detached_job_worker_error={exc}", file=sys.stderr, flush=True)
        return 1
    status = str(payload.get("status", "") or "")
    exit_code = int(payload.get("exit_code", 1))
    print(f"detached_job_worker_status={status}", flush=True)
    return 0 if exit_code == 0 else exit_code


if __name__ == "__main__":
    raise SystemExit(main())
