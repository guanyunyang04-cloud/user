from __future__ import annotations

import subprocess
import sys
from pathlib import Path

from daily_research.deep_alpha.experiment_guardrails import resolve_project_python_executable


LEGACY_PROFIT_SNAPSHOT_COMMIT = "e7d0f8d151c6667220f8ca5d0a6f98ab3b4b075d"
LEGACY_PROFIT_WORKTREE_NAME = "user_snapshot_codex_e7d0f8d"


def project_root_from_entry(entry_file: str) -> Path:
    return Path(entry_file).resolve().parents[2]


def ensure_legacy_profit_snapshot(project_root: Path) -> Path:
    snapshot_root = project_root.parent / LEGACY_PROFIT_WORKTREE_NAME
    marker = snapshot_root / "daily_research" / "baseline" / "train_trade_model.py"
    if marker.exists():
        return snapshot_root

    subprocess.run(
        [
            "git",
            "worktree",
            "add",
            "--detach",
            str(snapshot_root),
            LEGACY_PROFIT_SNAPSHOT_COMMIT,
        ],
        cwd=project_root,
        check=True,
    )
    return snapshot_root


def run_legacy_profit_backend(entry_file: str, script_relative_path: str, default_args: list[str]) -> None:
    project_root = project_root_from_entry(entry_file)
    snapshot_root = ensure_legacy_profit_snapshot(project_root)
    script_path = snapshot_root / script_relative_path
    if not script_path.exists():
        raise FileNotFoundError(f"Legacy profit backend script not found: {script_path}")

    cmd = [
        resolve_project_python_executable(sys.executable),
        str(script_path),
        *default_args,
        *sys.argv[1:],
    ]
    subprocess.run(cmd, cwd=project_root, check=True)
