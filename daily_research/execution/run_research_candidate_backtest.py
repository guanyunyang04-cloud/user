from __future__ import annotations

import sys
from pathlib import Path

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from daily_research.execution.entrypoint_utils import (
    bootstrap_execution_paths,
    ensure_default_pool_argument,
    has_arg,
    inject_default_arg,
    is_help_request,
)


def main():
    exec_dir = bootstrap_execution_paths(__file__)
    output_dir = exec_dir.parent / "output"

    inject_default_arg("--output-dir", str(output_dir))
    ensure_default_pool_argument()
    inject_default_arg("--regime-ma-window", "50")

    if not has_arg("--score-panel-csv") and not has_arg("--target-weight-panel-csv") and not is_help_request():
        raise ValueError(
            "Missing --score-panel-csv or --target-weight-panel-csv. Example: "
            "python daily_research/execution/run_research_candidate_backtest.py "
            "--target-weight-panel-csv daily_research/output/<run>/daily_target_weight_panel.csv "
            "--score-panel-csv daily_research/output/<run>/daily_score_panel.csv"
        )

    from daily_research.baseline.backtest_external_score_panel import main as backtest_external_score_panel_main

    backtest_external_score_panel_main()


if __name__ == "__main__":
    main()
