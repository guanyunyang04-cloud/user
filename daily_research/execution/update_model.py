from __future__ import annotations

import sys
from pathlib import Path

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from daily_research.execution.entrypoint_utils import (
    bootstrap_execution_paths,
    ensure_default_pool_argument,
    ensure_execution_strategy_defaults,
    inject_default_arg,
)


def main():
    exec_dir = bootstrap_execution_paths(__file__)

    models_dir = exec_dir / "models"
    models_dir.mkdir(parents=True, exist_ok=True)
    artifact_path = models_dir / "latest_ml_model.joblib"
    artifact_meta_path = models_dir / "latest_ml_model.json"

    inject_default_arg("--artifact-path", str(artifact_path))
    inject_default_arg("--artifact-meta-path", str(artifact_meta_path))
    ensure_default_pool_argument()
    ensure_execution_strategy_defaults()

    from daily_research.baseline.train_trade_model import main as base_main

    base_main()


if __name__ == "__main__":
    main()
