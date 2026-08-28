from __future__ import annotations

import argparse
import sys

from quantlab.data.cli import main as data_main
from quantlab.research.__main__ import main as research_main


def main(argv: list[str] | None = None) -> int:
    values = list(sys.argv[1:] if argv is None else argv)
    # Nested command parsers own their option namespaces. Dispatch before
    # the outer parser sees those options, otherwise valid flags such as
    # minute-v2's --workspace-root are rejected at this level.
    if values and values[0] == "research":
        return int(research_main(values[1:]) or 0)
    if values and values[0] == "data":
        return int(data_main(values[1:]) or 0)

    parser = argparse.ArgumentParser(prog="quantlab")
    subparsers = parser.add_subparsers(dest="command", required=True)
    research = subparsers.add_parser("research", help="run daily research commands")
    research.add_argument("args", nargs=argparse.REMAINDER)
    data = subparsers.add_parser("data", help="run QDP data commands")
    data.add_argument("args", nargs=argparse.REMAINDER)
    parser.parse_args(values)
    return 2
