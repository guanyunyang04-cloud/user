from __future__ import annotations

import sys
from pathlib import Path

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from daily_research.execution.high_profit_backend import run_legacy_profit_backend
from daily_research.execution.liquidity_universe import get_default_pool_file


def main():
    exec_dir = Path(__file__).resolve().parent
    models_dir = exec_dir / "models"
    models_dir.mkdir(parents=True, exist_ok=True)
    artifact_path = models_dir / "latest_ml_model.joblib"
    artifact_meta_path = models_dir / "latest_ml_model.json"
    pool_file = get_default_pool_file()
    if not pool_file.exists():
        raise FileNotFoundError(
            f"Default liquid500 universe file not found: {pool_file}. "
            "Please run daily_research/execution/update_liquid_pool.py after close first."
        )

    run_legacy_profit_backend(
        __file__,
        "daily_research/baseline/train_trade_model.py",
        [
            "--artifact-path",
            str(artifact_path),
            "--artifact-meta-path",
            str(artifact_meta_path),
            "--ml-model-family",
            "lgbm",
            "--stocks-file",
            str(pool_file),
            "--regime-ma-window",
            "50",
        ],
    )


if __name__ == "__main__":
    main()
