"""CLI for the targeted historical one-minute repair workflow."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from quantlab.core.io import json_safe

from .candidate import (
    DEFAULT_BATCH_CALENDAR_DAYS,
    DEFAULT_BATCH_TRADING_DAYS,
    DEFAULT_PRIORITY_RELATIVE_ERROR,
)
from .workflow import resume_targeted_minute_repair, run_targeted_minute_repair


def _date_range(value: str) -> tuple[str, str]:
    try:
        start, end = str(value).split(":", 1)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("expected START_DATE:END_DATE") from exc
    if not start or not end:
        raise argparse.ArgumentTypeError("expected START_DATE:END_DATE")
    return start, end


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Download and evidence-gate only previously audited historical 1-minute anomalies."
    )
    parser.add_argument("--workspace-root", type=Path)
    parser.add_argument("--minimum-relative-error", type=float, default=DEFAULT_PRIORITY_RELATIVE_ERROR)
    parser.add_argument("--maximum-relative-error", type=float)
    parser.add_argument(
        "--selection-mode", choices=("priority", "open", "high-low", "all"), default="priority"
    )
    parser.add_argument("--max-calendar-days", type=int, default=DEFAULT_BATCH_CALENDAR_DAYS)
    parser.add_argument("--max-trading-days", type=int, default=DEFAULT_BATCH_TRADING_DAYS)
    parser.add_argument("--rpm", type=int, default=96)
    parser.add_argument("--workers", type=int, default=3)
    parser.add_argument("--resume-run-dir", type=Path)
    parser.add_argument("--local-minute-root", type=Path)
    parser.add_argument("--local-aggregate-seed", type=Path)
    parser.add_argument("--exclude-date-range", action="append", type=_date_range, default=[])
    parser.add_argument("--target-file", type=Path)
    parser.add_argument("--target-reason", action="append", default=[])
    parser.add_argument("--target-severity", action="append", default=[])
    parser.add_argument("--target-field", action="append", default=[])
    parser.add_argument("--maximum-targets", type=int)
    parser.add_argument("--apply", action="store_true")
    parser.add_argument("--json", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    if (
        args.target_reason
        or args.target_severity
        or args.target_field
        or args.maximum_targets is not None
    ) and not args.target_file:
        raise ValueError("minute_repair_target_filter_requires_target_file")
    if args.resume_run_dir:
        if not args.apply:
            raise ValueError("minute_repair_resume_requires_apply")
        if (args.resume_run_dir / "run_config.json").is_file():
            from .local_workflow import (
                apply_local_parquet_minute_repair_run,
                resume_local_parquet_minute_repair,
            )

            staged = args.resume_run_dir / "prepared" / "staged_replacements.json"
            if staged.is_file():
                result = resume_local_parquet_minute_repair(
                    args.resume_run_dir,
                    workspace_root=args.workspace_root,
                )
            else:
                result = apply_local_parquet_minute_repair_run(
                    args.resume_run_dir,
                    workspace_root=args.workspace_root,
                )
        else:
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
                target_file=args.target_file,
                target_reasons=tuple(args.target_reason),
                target_severities=tuple(args.target_severity),
                target_fields=tuple(args.target_field),
                maximum_targets=args.maximum_targets,
            )
    elif args.local_minute_root:
        from .local_workflow import run_local_parquet_minute_repair

        result = run_local_parquet_minute_repair(
            local_minute_root=args.local_minute_root,
            workspace_root=args.workspace_root,
            minimum_relative_error=float(args.minimum_relative_error),
            maximum_relative_error=args.maximum_relative_error,
            selection_mode=str(args.selection_mode),
            excluded_date_ranges=tuple(args.exclude_date_range),
            aggregate_seed_path=args.local_aggregate_seed,
            target_file=args.target_file,
            target_reasons=tuple(args.target_reason),
            target_severities=tuple(args.target_severity),
            target_fields=tuple(args.target_field),
            maximum_targets=args.maximum_targets,
            apply=bool(args.apply),
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
            target_file=args.target_file,
            target_reasons=tuple(args.target_reason),
            target_severities=tuple(args.target_severity),
            target_fields=tuple(args.target_field),
            maximum_targets=args.maximum_targets,
            apply=bool(args.apply),
        )
    print(json.dumps(json_safe(result), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
