from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from quant_data_platform.core.paths import qdp_paths


@dataclass(frozen=True)
class QdpV3Paths:
    root: Path
    raw: Path
    datasets: Path
    candidates: Path
    active: Path
    rollbacks: Path
    pins: Path
    audits: Path
    jobs: Path
    metadata: Path
    compatibility: Path
    trash: Path

    @property
    def active_manifest(self) -> Path:
        return self.active / "active.json"


def qdp_v3_paths(workspace_root: str | Path | None = None) -> QdpV3Paths:
    root = qdp_paths(workspace_root).data_dir / "qdp_v3"
    return QdpV3Paths(
        root=root,
        raw=root / "raw",
        datasets=root / "datasets",
        candidates=root / "candidates",
        active=root / "active",
        rollbacks=root / "rollbacks",
        pins=root / "pins",
        audits=root / "audits",
        jobs=root / "jobs",
        metadata=root / "metadata",
        compatibility=root / "compatibility",
        trash=root / ".trash",
    )


def ensure_qdp_v3_layout(workspace_root: str | Path | None = None) -> QdpV3Paths:
    paths = qdp_v3_paths(workspace_root)
    for path in (
        paths.raw,
        paths.datasets,
        paths.candidates,
        paths.active,
        paths.rollbacks,
        paths.pins,
        paths.audits,
        paths.jobs,
        paths.metadata,
        paths.compatibility,
        paths.trash,
    ):
        path.mkdir(parents=True, exist_ok=True)
    return paths
