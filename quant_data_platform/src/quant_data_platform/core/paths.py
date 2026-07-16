from __future__ import annotations

import os
import json
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class QdpPaths:
    workspace_root: Path
    project_root: Path
    registry_dir: Path
    data_dir: Path
    audits_dir: Path
    memmap_dir: Path
    lake_root: Path
    legacy_registry_dir: Path

    @property
    def root_manifest(self) -> Path:
        return self.registry_dir / "root_manifest.json"

    @property
    def memmap_registry(self) -> Path:
        return self.registry_dir / "memmap_registry.json"

    @property
    def legacy_root_manifest(self) -> Path:
        return self.legacy_registry_dir / "root_manifest.json"

    @property
    def legacy_memmap_registry(self) -> Path:
        return self.legacy_registry_dir / "memmap_registry.json"


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
        if _is_main_brain_root(candidate):
            return candidate
    return Path(__file__).resolve().parents[4]


def _is_main_brain_root(path: Path) -> bool:
    manifest = path / "brain" / "brain_manifest.json"
    if not manifest.exists():
        return False
    try:
        payload = json.loads(manifest.read_text(encoding="utf-8"))
    except Exception:
        return False
    return str(dict(payload).get("brain_type", "") or "").strip().lower() == "main"


def project_root(root: str | Path | None = None) -> Path:
    ws = workspace_root(root)
    return (ws / "quant_data_platform").resolve()


def qdp_paths(root: str | Path | None = None) -> QdpPaths:
    ws = workspace_root(root)
    project = (ws / "quant_data_platform").resolve()
    data_dir = project / "data"
    return QdpPaths(
        workspace_root=ws,
        project_root=project,
        registry_dir=project / "registry",
        data_dir=data_dir,
        audits_dir=data_dir / "audits",
        memmap_dir=data_dir / "memmap",
        lake_root=data_dir / "lake",
        legacy_registry_dir=ws / "canonical_data" / "registry",
    )
