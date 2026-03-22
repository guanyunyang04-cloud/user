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
    inject_default_arg,
)


def main():
    exec_dir = bootstrap_execution_paths(__file__)

    positions_file = exec_dir / "current_positions.csv"
    example_file = exec_dir / "current_positions.example.csv"
    output_dir = exec_dir / "output"
    model_dir = exec_dir / "models"
    model_artifact = model_dir / "latest_ml_model.joblib"
    output_dir.mkdir(parents=True, exist_ok=True)
    model_dir.mkdir(parents=True, exist_ok=True)

    ensure_text_file_from_example(
        positions_file,
        example_file,
        "stock,shares,cost_price\n",
    )

    inject_default_arg("--positions-file", str(positions_file))
    inject_default_arg("--output-dir", str(output_dir))
    inject_default_arg("--model-artifact", str(model_artifact))
    ensure_default_pool_argument()
    ensure_execution_strategy_defaults()

    from daily_research.baseline.generate_daily_trade_plan import main as base_main

    base_main()


if __name__ == "__main__":
    main()
