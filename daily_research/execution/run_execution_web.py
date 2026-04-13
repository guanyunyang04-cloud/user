from __future__ import annotations

import argparse
import sys
from pathlib import Path

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from daily_research.execution.entrypoint_utils import missing_runtime_dependency_error


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="启动 daily_research 执行侧本地 Web 控制台。")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--reload", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        from daily_research.execution.web_server import run_web_console
    except ModuleNotFoundError as exc:
        raise missing_runtime_dependency_error(
            exc,
            command_hint="python daily_research/execution/run_execution_web.py --host 127.0.0.1 --port 8765",
        ) from exc
    run_web_console(host=args.host, port=args.port, reload=args.reload)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
