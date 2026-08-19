from __future__ import annotations

import argparse
import json
from typing import Any

from .data import verify_current_data
from .ensemble import evaluate as evaluate_ensemble
from .sequence import evaluate as evaluate_sequence
from .sequence import train_all as train_all_sequence
from .sequence import train_fold as train_sequence_fold
from .tree import compare, evaluate, train_all, train_fold


def _print(value: Any) -> None:
    print(json.dumps(value, ensure_ascii=False, indent=2), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Current technical research pipeline")
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
    args = parser.parse_args()

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
    else:
        _print(verify_current_data())
        for feature_count in (158, 183):
            train_all(feature_count)
            _print(evaluate(feature_count))
        _print(compare())


if __name__ == "__main__":
    main()
