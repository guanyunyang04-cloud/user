from __future__ import annotations

import argparse
import json
from typing import Any

from .data import verify_current_data
from .ensemble import evaluate as evaluate_ensemble
from .minute import DEFAULT_PARTICIPATION_RATE, run_minute_baseline, run_minute_walk_forward
from .minute_v2.cli import main as minute_v2_main
from .sequence import evaluate as evaluate_sequence
from .sequence import train_all as train_all_sequence
from .sequence import train_fold as train_sequence_fold
from .tree import compare, evaluate, train_all, train_fold


def _print(value: Any) -> None:
    print(json.dumps(value, ensure_ascii=False, indent=2), flush=True)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Current daily research pipeline")
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("verify")
    train = subparsers.add_parser("train-tree")
    train.add_argument("--features", type=int, choices=(158, 183), required=True)
    train.add_argument("--fold", type=int, choices=range(1, 6))
    evaluate_parser = subparsers.add_parser("evaluate-tree")
    evaluate_parser.add_argument(
        "--features", type=int, choices=(158, 183), required=True
    )
    subparsers.add_parser("compare-tree")
    subparsers.add_parser("run-tree-comparison")
    sequence = subparsers.add_parser("train-sequence")
    sequence.add_argument("--fold", type=int, choices=range(1, 6))
    subparsers.add_parser("evaluate-sequence")
    subparsers.add_parser("evaluate-ensemble")
    minute = subparsers.add_parser("minute-baseline")
    minute.add_argument("--parquet", action="append", required=True)
    minute.add_argument("--train-end-date", required=True)
    minute.add_argument("--evaluation-start-date", required=True)
    minute.add_argument("--evaluation-end-date", default="")
    minute.add_argument("--output-dir", required=True)
    minute.add_argument("--top-k", type=int, default=5)
    minute.add_argument("--maximum-participation-rate", type=float, default=DEFAULT_PARTICIPATION_RATE)
    walk = subparsers.add_parser("minute-walk-forward")
    walk.add_argument("--parquet", action="append", required=True)
    walk.add_argument("--evaluation-start-year", type=int, required=True)
    walk.add_argument("--evaluation-end-year", type=int, required=True)
    walk.add_argument("--output-dir", required=True)
    walk.add_argument("--top-k", type=int, default=5)
    walk.add_argument("--maximum-participation-rate", type=float, default=DEFAULT_PARTICIPATION_RATE)
    minute_v2 = subparsers.add_parser("minute-v2")
    minute_v2.add_argument("args", nargs=argparse.REMAINDER)
    args = parser.parse_args(argv)

    if args.command == "verify":
        _print(verify_current_data())
    elif args.command == "train-tree":
        result = (
            train_fold(args.features, args.fold)
            if args.fold is not None
            else train_all(args.features)
        )
        _print(result)
    elif args.command == "evaluate-tree":
        _print(evaluate(args.features))
    elif args.command == "compare-tree":
        _print(compare())
    elif args.command == "train-sequence":
        result = (
            train_sequence_fold(args.fold)
            if args.fold is not None
            else train_all_sequence()
        )
        _print(result)
    elif args.command == "evaluate-sequence":
        _print(evaluate_sequence())
    elif args.command == "evaluate-ensemble":
        _print(evaluate_ensemble())
    elif args.command == "minute-baseline":
        _print(
            run_minute_baseline(
                args.parquet,
                train_end_date=args.train_end_date,
                evaluation_start_date=args.evaluation_start_date,
                evaluation_end_date=args.evaluation_end_date,
                output_dir=args.output_dir,
                top_k=args.top_k,
                maximum_participation_rate=args.maximum_participation_rate,
            )
        )
    elif args.command == "minute-walk-forward":
        _print(
            run_minute_walk_forward(
                args.parquet,
                evaluation_start_year=args.evaluation_start_year,
                evaluation_end_year=args.evaluation_end_year,
                output_dir=args.output_dir,
                top_k=args.top_k,
                maximum_participation_rate=args.maximum_participation_rate,
            )
        )
    elif args.command == "minute-v2":
        return int(minute_v2_main(list(args.args)) or 0)
    else:
        _print(verify_current_data())
        for feature_count in (158, 183):
            train_all(feature_count)
            _print(evaluate(feature_count))
        _print(compare())
    return 0


if __name__ == "__main__":
    main()
