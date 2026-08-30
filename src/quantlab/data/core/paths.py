from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from quantlab.core.paths import workspace_root


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


def project_root(root: str | Path | None = None) -> Path:
    # Kept as a compatibility name for callers that need the workspace root.
    return workspace_root(root)


def qdp_paths(root: str | Path | None = None) -> QdpPaths:
    ws = workspace_root(root)
    return QdpPaths(
        workspace_root=ws,
        data_dir=(ws / "data" / "qdp").resolve(),
    )
