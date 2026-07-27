from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class QdpPaths:
    workspace_root: Path
    data_dir: Path


def workspace_root(start: str | Path | None = None) -> Path:
    env_root = os.environ.get("QDP_WORKSPACE_ROOT", "").strip()
    if env_root:
        return Path(env_root).resolve()
    # A caller-supplied root is an exact isolation boundary.  In particular,
    # fixtures and recovery probes nested below the production workspace must
    # never inherit a parent brain or its active data pointers.
    if start is not None:
        return Path(start).resolve()

    initial = Path.cwd().resolve()
    for candidate in (initial, *initial.parents):
        if _looks_like_workspace(candidate):
            return candidate
    return Path(__file__).resolve().parents[4]


def _looks_like_workspace(path: Path) -> bool:
    if not (path / "quant_data_platform").is_dir():
        return False
    return any(
        (path / marker).exists()
        for marker in (".git", "AGENTS.md", "README.md", "daily_research")
    )


def project_root(root: str | Path | None = None) -> Path:
    ws = workspace_root(root)
    return (ws / "quant_data_platform").resolve()


def qdp_paths(root: str | Path | None = None) -> QdpPaths:
    ws = workspace_root(root)
    return QdpPaths(
        workspace_root=ws,
        data_dir=(ws / "quant_data_platform" / "data").resolve(),
    )
