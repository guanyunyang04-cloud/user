"""Command-line interface for minute archive extraction and import."""

from __future__ import annotations

import argparse
import json

from quantlab.data.minute_archive.importer import import_year
from quantlab.data.minute_archive.reader import extract_to_parquet


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Read or formally import local minute ZIP archives")
    subparsers = parser.add_subparsers(dest="command", required=True)
    extract = subparsers.add_parser("extract")
    extract.add_argument("--archive", action="append", required=True)
    extract.add_argument("--symbol", action="append", required=True)
    extract.add_argument("--output", required=True)
    extract.add_argument("--start-date", default="")
    extract.add_argument("--end-date", default="")
    extract.add_argument("--retain-0930", action="store_true")
    extract.add_argument("--overwrite", action="store_true")
    formal = subparsers.add_parser("import-year")
    formal.add_argument("--archive", required=True)
    formal.add_argument("--year", type=int, required=True)
    formal.add_argument("--workspace-root", default="")
    formal.add_argument("--skip-sha256", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    if args.command == "extract":
        result = extract_to_parquet(
            args.archive,
            symbols=args.symbol,
            output_path=args.output,
            start_date=args.start_date,
            end_date=args.end_date,
            exclude_0930=not args.retain_0930,
            overwrite=args.overwrite,
        )
    else:
        result = import_year(
            args.archive,
            year=args.year,
            workspace_root_value=str(args.workspace_root or "") or None,
            compute_sha256=not args.skip_sha256,
        )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


__all__ = ["build_arg_parser", "main"]
