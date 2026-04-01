from __future__ import annotations

import sys
from pathlib import Path

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from daily_research.execution.entrypoint_utils import inject_flag_arg
from daily_research.execution.run_trade_plan import main as run_trade_plan_main


def main():
    inject_flag_arg("--legacy-ml")
    run_trade_plan_main()


if __name__ == "__main__":
    main()
