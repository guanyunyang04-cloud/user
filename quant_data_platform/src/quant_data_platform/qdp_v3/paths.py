from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from quant_data_platform.core.paths import qdp_paths


@dataclass(frozen=True)
class QdpV3Paths:
    data_root: Path
    runtime_root: Path
    root: Path
    raw: Path
    raw_bundles: Path
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
    staging: Path
    runtime_index: Path

    @property
    def active_manifest(self) -> Path:
        return self.active / "active.json"


def qdp_v3_paths(workspace_root: str | Path | None = None) -> QdpV3Paths:
    configured_data_root = os.environ.get("QDP_DATA_ROOT", "").strip()
    data_root = (
        Path(configured_data_root).expanduser().resolve()
        if configured_data_root
        else qdp_paths(workspace_root).data_dir
    )
    root = data_root / "qdp_v3"
    configured_runtime_root = os.environ.get("QDP_RUNTIME_ROOT", "").strip()
    runtime_root = (
        Path(configured_runtime_root).expanduser().resolve()
        if configured_runtime_root
        else root
    )
    return QdpV3Paths(
        data_root=data_root,
        runtime_root=runtime_root,
        root=root,
        raw=root / "raw",
        raw_bundles=root / "raw_bundles",
        datasets=root / "datasets",
        candidates=root / "candidates",
        active=root / "active",
        rollbacks=root / "rollbacks",
        pins=root / "pins",
        audits=root / "audits",
        jobs=runtime_root / "jobs",
        metadata=root / "metadata",
        compatibility=root / "compatibility",
        trash=root / ".trash",
        staging=runtime_root / "jobs" / "staging",
        runtime_index=runtime_root / "runtime.sqlite3",
    )


def ensure_qdp_v3_layout(workspace_root: str | Path | None = None) -> QdpV3Paths:
    paths = qdp_v3_paths(workspace_root)
    for path in (
        paths.raw,
        paths.raw_bundles,
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
        paths.staging,
        paths.runtime_index.parent,
    ):
        path.mkdir(parents=True, exist_ok=True)
    return paths
