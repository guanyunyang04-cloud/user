from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

from .json_io import read_json, utc_now, write_json
from .paths import QdpPaths, qdp_paths

DEFAULT_ALIAS = "canonical_data_v1"


def load_root_manifest(paths: QdpPaths | None = None, *, fallback_legacy: bool = True) -> dict[str, Any]:
    resolved = paths or qdp_paths()
    payload = read_json(resolved.root_manifest)
    if payload or not fallback_legacy:
        return payload
    legacy = read_json(resolved.legacy_root_manifest)
    if legacy:
        legacy["registry_source"] = "legacy_canonical_data_fallback"
    return legacy


def load_memmap_registry(paths: QdpPaths | None = None, *, fallback_legacy: bool = True) -> dict[str, Any]:
    resolved = paths or qdp_paths()
    payload = read_json(resolved.memmap_registry)
    if payload or not fallback_legacy:
        return payload
    legacy = read_json(resolved.legacy_memmap_registry)
    if legacy:
        legacy["registry_source"] = "legacy_canonical_data_fallback"
    return legacy


def migrate_legacy_registry(paths: QdpPaths | None = None, *, write: bool = True) -> dict[str, Any]:
    resolved = paths or qdp_paths()
    root_manifest = read_json(resolved.legacy_root_manifest)
    memmap_registry = read_json(resolved.legacy_memmap_registry)
    if not root_manifest:
        raise FileNotFoundError(f"legacy_root_manifest_not_found: {resolved.legacy_root_manifest}")
    root_manifest = {
        **root_manifest,
        "workspace_root": str(resolved.workspace_root.as_posix()),
        "qdp_project_root": str(resolved.project_root.as_posix()),
        "canonical_memmap_registry": str(resolved.memmap_registry.as_posix()),
        "migrated_from": str(resolved.legacy_registry_dir.as_posix()),
        "registry_owner": "quant_data_platform",
        "updated_at": utc_now(),
    }
    if memmap_registry:
        memmap_registry = {
            **memmap_registry,
            "registry_owner": "quant_data_platform",
            "migrated_from": str(resolved.legacy_memmap_registry.as_posix()),
            "updated_at": utc_now(),
        }
    if write:
        write_json(resolved.root_manifest, root_manifest)
        if memmap_registry:
            write_json(resolved.memmap_registry, memmap_registry)
    return {
        "status": "written" if write else "dry_run",
        "root_manifest": str(resolved.root_manifest),
        "memmap_registry": str(resolved.memmap_registry),
        "canonical_dataset_id": str(root_manifest.get("canonical_dataset_id", "") or ""),
        "active_memmap_manifest": str(memmap_registry.get("active_manifest_json", "") or "") if memmap_registry else "",
    }


def write_root_manifest_update(
    updates: Mapping[str, Any],
    *,
    paths: QdpPaths | None = None,
) -> Path:
    resolved = paths or qdp_paths()
    payload = load_root_manifest(resolved)
    payload.update(dict(updates))
    payload["registry_owner"] = "quant_data_platform"
    payload["updated_at"] = utc_now()
    return write_json(resolved.root_manifest, payload)


def registry_status(paths: QdpPaths | None = None) -> dict[str, Any]:
    resolved = paths or qdp_paths()
    root_manifest = load_root_manifest(resolved)
    memmap_registry = load_memmap_registry(resolved)
    sharded_registry = read_json(resolved.registry_dir / "sharded_memmap_registry.json")
    sharded_status = dict(root_manifest.get("canonical_sharded_memmap_status", {}) or {})
    canonical_dataset_id = str(root_manifest.get("canonical_dataset_id", "") or "")
    active_memmap_source = str(memmap_registry.get("active_source_market_dataset_id", "") or "")
    active_memmap_matches = bool(canonical_dataset_id and active_memmap_source and canonical_dataset_id == active_memmap_source)
    return {
        "status": "ok" if root_manifest else "missing_root_manifest",
        "alias": str(root_manifest.get("alias", DEFAULT_ALIAS) or DEFAULT_ALIAS),
        "registry_root": str(resolved.registry_dir.as_posix()),
        "root_manifest_json": str(resolved.root_manifest.as_posix()),
        "root_manifest_exists": bool(resolved.root_manifest.exists()),
        "legacy_fallback_used": str(root_manifest.get("registry_source", "")) == "legacy_canonical_data_fallback",
        "canonical_dataset_id": canonical_dataset_id,
        "canonical_manifest": str(root_manifest.get("canonical_manifest", "") or ""),
        "canonical_start_date": str(root_manifest.get("canonical_start_date", "") or ""),
        "component_dataset_ids": dict(root_manifest.get("canonical_component_dataset_ids", {}) or {}),
        "memmap_registry_json": str(resolved.memmap_registry.as_posix()),
        "memmap_registry_exists": bool(resolved.memmap_registry.exists()),
        "active_memmap_manifest": str(memmap_registry.get("active_manifest_json", "") or ""),
        "active_memmap_source_market_dataset_id": active_memmap_source,
        "active_memmap_matches_canonical_bundle": active_memmap_matches,
        "active_memmap_state": "current" if active_memmap_matches else "stale_or_missing",
        "active_feature_profile": str(memmap_registry.get("active_feature_profile", "") or ""),
        "active_scope": str(memmap_registry.get("active_scope", "") or ""),
        "active_sample_count": int(memmap_registry.get("active_sample_count", 0) or 0),
        "sharded_memmap_registry_json": str((resolved.registry_dir / "sharded_memmap_registry.json").as_posix()),
        "sharded_memmap_registry_exists": bool((resolved.registry_dir / "sharded_memmap_registry.json").exists()),
        "latest_sharded_manifest": str(sharded_registry.get("latest_manifest_json", "") or sharded_status.get("latest_manifest_json", "") or ""),
        "latest_sharded_scope": str(sharded_registry.get("latest_scope", "") or sharded_status.get("latest_scope", "") or ""),
        "latest_sharded_status": str(sharded_status.get("status", "") or sharded_registry.get("status", "") or ""),
        "latest_sharded_stored_shard_count": int(sharded_status.get("latest_stored_shard_count", 0) or 0),
        "latest_sharded_planned_shard_count": int(sharded_status.get("latest_planned_shard_count", 0) or 0),
    }
