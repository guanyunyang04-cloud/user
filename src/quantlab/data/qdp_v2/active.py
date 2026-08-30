"""Read-only resolution of active QDP datasets."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from quantlab.data.qdp_v2.manifest import (
    DatasetManifest,
    active_manifest_path,
    dataset_manifest_path,
    qdp_v2_root,
    read_active_manifest,
    read_dataset_manifest,
    resolve_manifest_path,
)
from quantlab.data.qdp_v2.status import active_dataset_map


class ActiveDomainError(RuntimeError):
    """Raised when an active dataset cannot be resolved safely."""


@dataclass(frozen=True)
class ActiveDomain:
    root: Path
    active_path: Path
    active: dict[str, Any]
    domain: str
    dataset_id: str
    manifest_path: Path
    manifest: DatasetManifest

    @property
    def shard_paths(self) -> list[Path]:
        return [resolve_manifest_path(item.path, root=self.root) for item in self.manifest.shards]


def resolve_active_domain(
    domain: str,
    *,
    workspace_root: str | Path | None = None,
) -> ActiveDomain:
    requested = str(domain or "").strip()
    if not requested:
        raise ValueError("qdp_v2_active_domain_required")
    root = qdp_v2_root(workspace_root).resolve()
    active_path = active_manifest_path(root).resolve()
    active = read_active_manifest(root)
    dataset_id = active_dataset_map(active).get(requested, "")
    if not dataset_id:
        raise ActiveDomainError(f"qdp_v2_active_domain_missing:{requested}")
    manifest_path = dataset_manifest_path(root, requested, dataset_id).resolve()
    manifest = read_dataset_manifest(manifest_path)
    if manifest.domain != requested or manifest.dataset_id != dataset_id:
        raise ActiveDomainError(
            "qdp_v2_active_manifest_identity_mismatch:"
            f"expected={requested}:{dataset_id}:"
            f"actual={manifest.domain}:{manifest.dataset_id}"
        )
    return ActiveDomain(
        root=root,
        active_path=active_path,
        active=active,
        domain=requested,
        dataset_id=dataset_id,
        manifest_path=manifest_path,
        manifest=manifest,
    )


__all__ = ["ActiveDomain", "ActiveDomainError", "resolve_active_domain"]
