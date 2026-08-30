from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


def workspace_root(start: str | Path | None = None) -> Path:
    """Resolve the one workspace boundary used by data and research code."""

    if start is not None:
        return Path(start).resolve()
    # ``QDP_WORKSPACE_ROOT`` was the data-platform spelling before the
    # workspace resolver moved into ``quantlab.core``.  Keep it as a fallback
    # so existing scripts and scheduled jobs continue to resolve the same
    # workspace while new callers use the package-wide name.
    configured = os.environ.get("QUANTLAB_ROOT", "").strip()
    if not configured:
        configured = os.environ.get("QDP_WORKSPACE_ROOT", "").strip()
    if configured:
        return Path(configured).resolve()
    initial = Path.cwd().resolve()
    for candidate in (initial, *initial.parents):
        if _looks_like_workspace(candidate):
            return candidate
    bundled = Path(__file__).resolve().parents[3]
    if _looks_like_workspace(bundled):
        return bundled
    raise RuntimeError(f"quantlab workspace not found from: {initial}")


def _looks_like_workspace(path: Path) -> bool:
    return (path / "pyproject.toml").is_file() or (path / "AGENTS.md").is_file()


@dataclass(frozen=True)
class WorkspacePaths:
    root: Path
    data: Path
    market_data: Path
    research_data: Path
    runs: Path
    research: Path


def paths(root: str | Path | None = None) -> WorkspacePaths:
    root = workspace_root(root)
    return WorkspacePaths(
        root=root,
        data=root / "data",
        market_data=root / "data" / "market",
        research_data=root / "data" / "research",
        runs=root / "runs",
        research=root / "research",
    )
