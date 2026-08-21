"""Recent Market Repair: cli responsibilities."""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from .config import (
    DEFAULT_END_DATE,
    DEFAULT_START_DATE,
    DEFAULT_WORKERS,
)
from .entrypoints import (
    run_baostock_intraday_repair,
    run_recent_daily_repair,
    run_recent_intraday_repair,
)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "command",
        choices=("daily", "intraday", "baostock-intraday", "all"),
    )
    parser.add_argument("--workspace-root", default=str(Path.cwd()))
    parser.add_argument("--start-date", default=DEFAULT_START_DATE)
    parser.add_argument("--end-date", default=DEFAULT_END_DATE)
    parser.add_argument("--workers", type=int, default=DEFAULT_WORKERS)
    parser.add_argument("--no-resume", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    resume = not bool(args.no_resume)
    results: dict[str, Any] = {}
    if args.command in {"daily", "all"}:
        results["daily"] = run_recent_daily_repair(
            trade_date=args.end_date,
            workspace_root=args.workspace_root,
            workers=args.workers,
            resume=resume,
        )
    if args.command in {"intraday", "all"}:
        results["intraday"] = run_recent_intraday_repair(
            start_date=args.start_date,
            end_date=args.end_date,
            workspace_root=args.workspace_root,
            workers=args.workers,
            resume=resume,
        )
    if args.command in {"baostock-intraday", "all"}:
        results["baostock_intraday"] = run_baostock_intraday_repair(
            start_date=args.start_date,
            end_date=args.end_date,
            workspace_root=args.workspace_root,
            workers=args.workers,
            resume=resume,
        )
    print(json.dumps(results, ensure_ascii=False, sort_keys=True), flush=True)
    return 0
