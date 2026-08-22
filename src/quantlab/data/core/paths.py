from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class QdpPaths:
    workspace_root: Path
    data_dir: Path

    @property
    def qdp_v2_dir(self) -> Path:
        return self.data_dir / "qdp_v2"

    @property
    def runtime_dir(self) -> Path:
        return self.data_dir / "qdp_runtime"

    @property
    def runtime_archives_dir(self) -> Path:
        return self.data_dir / "qdp_runtime_archives"

    @property
    def private_dir(self) -> Path:
        return self.data_dir / "qdp_private"

    @property
    def event_packs_dir(self) -> Path:
        return self.data_dir / "event_packs"

    @property
    def source_archives_dir(self) -> Path:
        """Immutable external inputs kept outside the active QDP datasets."""

        return self.data_dir / "source_archives"


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
    return any(
        (path / marker).exists()
        for marker in (".git", "AGENTS.md", "README.md", "src")
    )


def project_root(root: str | Path | None = None) -> Path:
    # Kept as a compatibility name for callers that need the workspace root.
    return workspace_root(root)


def qdp_paths(root: str | Path | None = None) -> QdpPaths:
    ws = workspace_root(root)
    return QdpPaths(
        workspace_root=ws,
        data_dir=(ws / "data" / "qdp").resolve(),
    )
