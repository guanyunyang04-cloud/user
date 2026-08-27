"""Command-line entry points for the minute-v2 research pipeline."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from .builder import build_month, build_range, verify_dataset, verify_month
from .contracts import MinuteV2Config
from .pilot import audit_pilot_month
from .training import run_two_fold_baselines


def _print(value: Any) -> None:
    print(json.dumps(value, ensure_ascii=False, indent=2), flush=True)


def _config(args: argparse.Namespace) -> MinuteV2Config:
    return MinuteV2Config(
        maximum_trading_days_per_month=int(args.maximum_trading_days_per_month),
        processing_days_per_chunk=int(args.processing_days_per_chunk),
        duckdb_threads=int(args.duckdb_threads),
        memory_floor_gib=float(args.memory_floor_gib),
        duckdb_memory_limit_gib=float(args.duckdb_memory_limit_gib),
    )


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="quantlab research minute-v2")
    parser.add_argument("--workspace-root", default=str(Path.cwd()))
    subparsers = parser.add_subparsers(dest="command", required=True)
    month = subparsers.add_parser("build-month")
    month.add_argument("--year", type=int, required=True)
    month.add_argument("--month", type=int, choices=range(1, 13), required=True)
    month.add_argument("--output-root", default="")
    month.add_argument("--keep-base", action="store_true")
    month.add_argument("--force", action="store_true")
    month.add_argument("--maximum-trading-days-per-month", type=int, default=0)
    month.add_argument("--processing-days-per-chunk", type=int, default=1)
    month.add_argument("--duckdb-threads", type=int, default=4)
    month.add_argument("--memory-floor-gib", type=float, default=0.5)
    month.add_argument("--duckdb-memory-limit-gib", type=float, default=4.0)
    range_parser = subparsers.add_parser("build-range")
    range_parser.add_argument("--start-year", type=int, required=True)
    range_parser.add_argument("--end-year", type=int, required=True)
    range_parser.add_argument("--output-root", default="")
    range_parser.add_argument("--force", action="store_true")
    range_parser.add_argument("--maximum-trading-days-per-month", type=int, default=0)
    range_parser.add_argument("--processing-days-per-chunk", type=int, default=1)
    range_parser.add_argument("--duckdb-threads", type=int, default=4)
    range_parser.add_argument("--memory-floor-gib", type=float, default=0.5)
    range_parser.add_argument("--duckdb-memory-limit-gib", type=float, default=4.0)
    audit = subparsers.add_parser("audit-pilot")
    audit.add_argument("--manifest", required=True)
    audit.add_argument("--output", default="")
    verify = subparsers.add_parser("verify-month")
    verify.add_argument("--manifest", required=True)
    verify_dataset_parser = subparsers.add_parser("verify-dataset")
    verify_dataset_parser.add_argument("--manifest", required=True)
    train = subparsers.add_parser("train-baselines")
    train.add_argument("--dataset-root", required=True)
    train.add_argument("--output-root", required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    workspace = Path(args.workspace_root).resolve()
    if args.command == "build-month":
        _print(
            build_month(
                workspace,
                year=args.year,
                month=args.month,
                output_root=args.output_root or None,
                config=_config(args),
                keep_base=bool(args.keep_base),
                force=bool(args.force),
            )
        )
    elif args.command == "build-range":
        _print(
            build_range(
                workspace,
                start_year=args.start_year,
                end_year=args.end_year,
                output_root=args.output_root or None,
                config=_config(args),
                force=bool(args.force),
            )
        )
    elif args.command == "audit-pilot":
        _print(audit_pilot_month(args.manifest, output_path=args.output or None))
    elif args.command == "verify-month":
        _print(verify_month(args.manifest))
    elif args.command == "verify-dataset":
        _print(verify_dataset(args.manifest))
    elif args.command == "train-baselines":
        _print(
            run_two_fold_baselines(
                args.dataset_root,
                output_root=args.output_root,
            )
        )
    return 0


__all__ = ["build_arg_parser", "main"]
