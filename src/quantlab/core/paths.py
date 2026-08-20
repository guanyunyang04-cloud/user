from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


def workspace_root() -> Path:
    configured = os.environ.get("QUANTLAB_ROOT", "").strip()
    root = Path(configured) if configured else Path(__file__).resolve().parents[3]
    root = root.resolve()
    if not (root / "pyproject.toml").is_file():
        raise RuntimeError(f"quantlab workspace not found: {root}")
    return root


@dataclass(frozen=True)
class WorkspacePaths:
    root: Path
    data: Path
    market_data: Path
    research_data: Path
    runs: Path
    research: Path


def paths() -> WorkspacePaths:
    root = workspace_root()
    return WorkspacePaths(
        root=root,
        data=root / "data",
        market_data=root / "data" / "market",
        research_data=root / "data" / "research",
        runs=root / "runs",
        research=root / "research",
    )

