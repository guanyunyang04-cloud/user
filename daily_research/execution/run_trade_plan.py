from __future__ import annotations

import sys
from pathlib import Path

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from daily_research.execution.entrypoint_utils import ensure_text_file_from_example
from daily_research.execution.high_profit_backend import run_legacy_profit_backend
from daily_research.execution.liquidity_universe import get_default_pool_file


def main():
    exec_dir = Path(__file__).resolve().parent
    positions_file = exec_dir / "current_positions.csv"
    example_file = exec_dir / "current_positions.example.csv"
    output_dir = exec_dir / "output"
    model_dir = exec_dir / "models"
    model_artifact = model_dir / "latest_ml_model.joblib"
    pool_file = get_default_pool_file()
    output_dir.mkdir(parents=True, exist_ok=True)
    model_dir.mkdir(parents=True, exist_ok=True)
    if not pool_file.exists():
        raise FileNotFoundError(
            f"Default liquid500 universe file not found: {pool_file}. "
            "Please run daily_research/execution/update_liquid_pool.py after close first."
        )

    ensure_text_file_from_example(
        positions_file,
        example_file,
        "stock,shares,cost_price\n",
    )

    run_legacy_profit_backend(
        __file__,
        "daily_research/baseline/generate_daily_trade_plan.py",
        [
            "--positions-file",
            str(positions_file),
            "--output-dir",
            str(output_dir),
            "--model-artifact",
            str(model_artifact),
            "--stocks-file",
            str(pool_file),
            "--regime-ma-window",
            "50",
        ],
    )


if __name__ == "__main__":
    main()
