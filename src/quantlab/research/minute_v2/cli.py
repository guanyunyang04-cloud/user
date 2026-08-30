"""Command-line entry points for the minute-v2 research pipeline."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from .builder import build_month, build_range, verify_dataset, verify_month
from .contracts import MinuteV2Config
from .pilot import audit_pilot_month
from .sampling import audit_candidate_recall_files
from .stage_one import run_stage_one_audit
from .training import run_two_fold_baselines


def _print(value: Any) -> None:
    print(json.dumps(value, ensure_ascii=False, indent=2), flush=True)


def _config(args: argparse.Namespace) -> MinuteV2Config:
    return MinuteV2Config(
        maximum_trading_days_per_month=int(args.maximum_trading_days_per_month),
        processing_days_per_chunk=int(args.processing_days_per_chunk),
        duckdb_threads=(
            args.duckdb_threads
            if str(args.duckdb_threads).strip().lower() == "auto"
            else int(args.duckdb_threads)
        ),
        memory_floor_gib=float(args.memory_floor_gib),
        duckdb_memory_limit_gib=(
            args.duckdb_memory_limit_gib
            if str(args.duckdb_memory_limit_gib).strip().lower() == "auto"
            else float(args.duckdb_memory_limit_gib)
        ),
        temp_directory=args.temp_directory or None,
        query_profile_path=args.query_profile_path or None,
        feature_storage=args.feature_storage,
    )


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="quantlab research minute-v2")
    parser.add_argument("--workspace-root", default=str(Path.cwd()))
    subparsers = parser.add_subparsers(dest="command", required=True)
    month = subparsers.add_parser("build-month")
    month.add_argument("--year", type=int, required=True)
    month.add_argument("--month", type=int, choices=range(1, 13), required=True)
    month.add_argument("--output-root", default="")
    month_base = month.add_mutually_exclusive_group()
    month_base.add_argument("--keep-base", dest="keep_base", action="store_true")
    month_base.add_argument("--drop-base", dest="keep_base", action="store_false")
    month.set_defaults(keep_base=True)
    month.add_argument("--force", action="store_true")
    month.add_argument("--maximum-trading-days-per-month", type=int, default=0)
    month.add_argument("--processing-days-per-chunk", type=int, default=1)
    month.add_argument("--duckdb-threads", default=4, help="integer or auto")
    month.add_argument("--memory-floor-gib", type=float, default=0.5)
    month.add_argument("--duckdb-memory-limit-gib", default="auto", help="GiB cap or auto")
    month.add_argument("--temp-directory", default="", help="DuckDB spill root")
    month.add_argument(
        "--query-profile-path",
        default="",
        help="write query timing JSON (use auto for month-local output)",
    )
    month.add_argument(
        "--feature-storage",
        choices=("split", "core", "full"),
        default="split",
        help=(
            "split=core base plus optional sidecar; core=core-only training profile; "
            "full=complete feature matrix in base"
        ),
    )
    range_parser = subparsers.add_parser("build-range")
    range_parser.add_argument("--start-year", type=int, required=True)
    range_parser.add_argument("--end-year", type=int, required=True)
    range_parser.add_argument("--output-root", default="")
    range_parser.add_argument("--force", action="store_true")
    range_base = range_parser.add_mutually_exclusive_group()
    range_base.add_argument("--keep-base", dest="keep_base", action="store_true")
    range_base.add_argument("--drop-base", dest="keep_base", action="store_false")
    range_parser.set_defaults(keep_base=True)
    range_parser.add_argument("--maximum-trading-days-per-month", type=int, default=0)
    range_parser.add_argument("--processing-days-per-chunk", type=int, default=1)
    range_parser.add_argument("--duckdb-threads", default=4, help="integer or auto")
    range_parser.add_argument("--memory-floor-gib", type=float, default=0.5)
    range_parser.add_argument("--duckdb-memory-limit-gib", default="auto", help="GiB cap or auto")
    range_parser.add_argument("--temp-directory", default="", help="DuckDB spill root")
    range_parser.add_argument(
        "--query-profile-path",
        default="",
        help="write query timing JSON (use auto for month-local output)",
    )
    range_parser.add_argument(
        "--feature-storage",
        choices=("split", "core", "full"),
        default="split",
        help=(
            "split=core base plus optional sidecar; core=core-only training profile; "
            "full=complete feature matrix in base"
        ),
    )
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
    recall = subparsers.add_parser(
        "audit-candidate-recall",
        help="audit candidate-gate recall from narrow Parquet artifacts",
    )
    recall.add_argument("--base", required=True)
    recall.add_argument("--candidates", required=True)
    recall.add_argument("--outcomes", required=True)
    recall.add_argument("--target", default="label_return_5m")
    recall.add_argument("--top-k", type=int, nargs="+", default=[1, 3, 5])
    recall.add_argument("--output", default="")
    stage_one = subparsers.add_parser(
        "stage-one-audit",
        help="build complete outcomes and audit the candidate gate for a narrow month",
    )
    stage_one.add_argument("--manifest", required=True)
    stage_one.add_argument("--output-directory", default="")
    stage_one.add_argument("--top-k", type=int, nargs="+", default=[1, 3, 5])
    stage_one.add_argument("--force", action="store_true")
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
                keep_base=bool(args.keep_base),
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
    elif args.command == "audit-candidate-recall":
        result = audit_candidate_recall_files(
            args.base,
            args.candidates,
            args.outcomes,
            target=args.target,
            top_k=args.top_k,
        )
        if args.output:
            output = Path(args.output).resolve()
            output.parent.mkdir(parents=True, exist_ok=True)
            output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
        _print(result)
    elif args.command == "stage-one-audit":
        _print(
            run_stage_one_audit(
                args.manifest,
                output_directory=args.output_directory or None,
                top_k=tuple(args.top_k),
                force=bool(args.force),
            )
        )
    return 0


__all__ = ["build_arg_parser", "main"]
