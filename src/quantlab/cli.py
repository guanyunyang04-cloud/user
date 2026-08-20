from __future__ import annotations

import argparse

from quantlab.data.cli import main as data_main
from quantlab.research.__main__ import main as research_main


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="quantlab")
    subparsers = parser.add_subparsers(dest="command", required=True)
    research = subparsers.add_parser("research", help="run daily research commands")
    research.add_argument("args", nargs=argparse.REMAINDER)
    data = subparsers.add_parser("data", help="run QDP data commands")
    data.add_argument("args", nargs=argparse.REMAINDER)
    args = parser.parse_args(argv)
    if args.command == "research":
        return int(research_main(list(args.args)) or 0)
    if args.command == "data":
        return int(data_main(list(args.args)) or 0)
    return 2
