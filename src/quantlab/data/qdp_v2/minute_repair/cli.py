"""CLI for the targeted historical one-minute repair workflow."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from quantlab.data.core.json_io import json_safe

from .candidate import (
    DEFAULT_BATCH_CALENDAR_DAYS,
    DEFAULT_BATCH_TRADING_DAYS,
    DEFAULT_PRIORITY_RELATIVE_ERROR,
)
from .workflow import resume_targeted_minute_repair, run_targeted_minute_repair


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Download and evidence-gate only previously audited historical 1-minute anomalies."
    )
    parser.add_argument("--workspace-root", type=Path)
    parser.add_argument("--minimum-relative-error", type=float, default=DEFAULT_PRIORITY_RELATIVE_ERROR)
    parser.add_argument("--maximum-relative-error", type=float)
    parser.add_argument("--selection-mode", choices=("priority", "open"), default="priority")
    parser.add_argument("--max-calendar-days", type=int, default=DEFAULT_BATCH_CALENDAR_DAYS)
    parser.add_argument("--max-trading-days", type=int, default=DEFAULT_BATCH_TRADING_DAYS)
    parser.add_argument("--rpm", type=int, default=96)
    parser.add_argument("--workers", type=int, default=3)
    parser.add_argument("--resume-run-dir", type=Path)
    parser.add_argument("--apply", action="store_true")
    parser.add_argument("--json", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    if args.resume_run_dir:
        if not args.apply:
            raise ValueError("minute_repair_resume_requires_apply")
        result = resume_targeted_minute_repair(
            args.resume_run_dir,
            workspace_root=args.workspace_root,
            minimum_relative_error=float(args.minimum_relative_error),
            maximum_relative_error=args.maximum_relative_error,
            selection_mode=str(args.selection_mode),
            max_calendar_days=int(args.max_calendar_days),
            max_trading_days=int(args.max_trading_days),
            rpm=int(args.rpm),
            workers=int(args.workers),
        )
    else:
        result = run_targeted_minute_repair(
            workspace_root=args.workspace_root,
            minimum_relative_error=float(args.minimum_relative_error),
            maximum_relative_error=args.maximum_relative_error,
            selection_mode=str(args.selection_mode),
            max_calendar_days=int(args.max_calendar_days),
            max_trading_days=int(args.max_trading_days),
            rpm=int(args.rpm),
            workers=int(args.workers),
            apply=bool(args.apply),
        )
    print(json.dumps(json_safe(result), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
