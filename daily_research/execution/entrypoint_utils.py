from __future__ import annotations

import sys
from pathlib import Path

EXECUTION_DEFAULT_ENHANCED_PROFILE = "up_low_breakout_v2"
EXECUTION_DEFAULT_STATE_ENSEMBLE_WEIGHTS = "trend_up_low_vol=ml:0.25,none:0.25,v2:0.50"
EXECUTION_DEFAULT_LGBM_N_ESTIMATORS = "520"


def has_arg(name: str) -> bool:
    for item in sys.argv[1:]:
        if item == name or item.startswith(name + "="):
            return True
    return False


def inject_default_arg(name: str, value: str) -> None:
    if not has_arg(name):
        sys.argv.extend([name, value])


def is_help_request() -> bool:
    return any(item in {"-h", "--help"} for item in sys.argv[1:])


def bootstrap_execution_paths(entry_file: str) -> Path:
    exec_dir = Path(entry_file).resolve().parent
    baseline_dir = exec_dir.parent / "baseline"
    project_root = exec_dir.parent.parent
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))
    if str(baseline_dir) not in sys.path:
        sys.path.insert(0, str(baseline_dir))
    return exec_dir


def ensure_default_pool_argument() -> None:
    if has_arg("--stocks") or has_arg("--stocks-file"):
        return
    from daily_research.execution.liquidity_universe import get_default_pool_file

    pool_file = get_default_pool_file()
    if not pool_file.exists() and not is_help_request():
        raise FileNotFoundError(
            f"Default liquid500 universe file not found: {pool_file}. "
            "Please run daily_research/execution/update_liquid_pool.py after close first."
        )
    inject_default_arg("--stocks-file", str(pool_file))


def ensure_execution_strategy_defaults() -> None:
    # Promote the current execution default from ma60 to the validated ma50 baseline.
    inject_default_arg("--regime-ma-window", "50")
    inject_default_arg("--enhanced-profile", EXECUTION_DEFAULT_ENHANCED_PROFILE)
    inject_default_arg("--ensemble-state-weights", EXECUTION_DEFAULT_STATE_ENSEMBLE_WEIGHTS)
    inject_default_arg("--lgbm-n-estimators", EXECUTION_DEFAULT_LGBM_N_ESTIMATORS)


def ensure_text_file_from_example(target: Path, example: Path, default_text: str) -> None:
    if target.exists():
        return
    if example.exists():
        target.write_text(example.read_text(encoding="utf-8"), encoding="utf-8")
        return
    target.write_text(default_text, encoding="utf-8")
