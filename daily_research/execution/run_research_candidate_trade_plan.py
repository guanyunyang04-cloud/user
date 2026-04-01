from __future__ import annotations

import sys
from pathlib import Path

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from daily_research.execution.entrypoint_utils import (
    bootstrap_execution_paths,
    ensure_default_pool_argument,
    ensure_execution_strategy_defaults,
    ensure_text_file_from_example,
    has_arg,
    inject_default_arg,
    is_help_request,
)


def main():
    exec_dir = bootstrap_execution_paths(__file__)
    positions_file = exec_dir / "current_positions.csv"
    example_file = exec_dir / "current_positions.example.csv"
    output_dir = exec_dir / "output" / "research_candidates"
    output_dir.mkdir(parents=True, exist_ok=True)

    ensure_text_file_from_example(
        positions_file,
        example_file,
        "stock,shares,cost_price\n",
    )

    inject_default_arg("--positions-file", str(positions_file))
    inject_default_arg("--output-dir", str(output_dir))
    inject_default_arg("--external-score-column", "latest_score")
    inject_default_arg("--external-target-weight-column", "target_weight")
    ensure_default_pool_argument()
    ensure_execution_strategy_defaults()

    if not has_arg("--external-score-csv") and not has_arg("--external-target-weight-csv") and not is_help_request():
        raise ValueError(
            "Missing --external-score-csv or --external-target-weight-csv. Example: "
            "python daily_research/execution/run_research_candidate_trade_plan.py "
            "--external-target-weight-csv daily_research/output/<run>/daily_target_weight_panel.csv "
            "--external-score-csv daily_research/output/<run>/daily_score_panel.csv"
        )

    from daily_research.baseline.generate_daily_trade_plan import main as generate_daily_trade_plan_main

    generate_daily_trade_plan_main()


if __name__ == "__main__":
    main()
