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
    canonical_manifest_path = Path(str(root_manifest.get("canonical_manifest", "") or "")) if str(root_manifest.get("canonical_manifest", "") or "").strip() else Path()
    canonical_manifest = read_json(canonical_manifest_path) if canonical_manifest_path and canonical_manifest_path.exists() and canonical_manifest_path.is_file() else {}
    memmap_registry = load_memmap_registry(resolved)
    sharded_registry = read_json(resolved.registry_dir / "sharded_memmap_registry.json")
    sharded_status = dict(root_manifest.get("canonical_sharded_memmap_status", {}) or {})
    canonical_dataset_id = str(root_manifest.get("canonical_dataset_id", "") or "")
    component_coverage = _active_component_coverage(root_manifest, paths=resolved)
    canonical_end_date = str(canonical_manifest.get("end_date", "") or root_manifest.get("canonical_end_date", "") or "")
    lagging_components = {
        domain: coverage
        for domain, coverage in component_coverage.items()
        if canonical_end_date
        and str(coverage.get("end_date", "") or "")
        and str(coverage.get("end_date", "") or "") < canonical_end_date
    }
    active_sharded_manifest = str(sharded_registry.get("active_manifest_json", "") or sharded_status.get("active_manifest_json", "") or "")
    active_sharded_payload = read_json(Path(active_sharded_manifest)) if active_sharded_manifest else {}
    active_sharded_source = str(
        active_sharded_payload.get("canonical_dataset_id", "")
        or sharded_status.get("latest_canonical_dataset_id", "")
        or ""
    )
    sharded_full_ready = bool(active_sharded_manifest and str(sharded_status.get("status", "")) == "sharded_full_ready")
    active_memmap_source = (
        active_sharded_source
        if sharded_full_ready and active_sharded_source
        else str(memmap_registry.get("active_source_market_dataset_id", "") or "")
    )
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
        "canonical_end_date": canonical_end_date,
        "component_dataset_ids": dict(root_manifest.get("canonical_component_dataset_ids", {}) or {}),
        "active_component_coverage": component_coverage,
        "active_component_lagging_to_canonical_end": lagging_components,
        "memmap_registry_json": str(resolved.memmap_registry.as_posix()),
        "memmap_registry_exists": bool(resolved.memmap_registry.exists()),
        "active_memmap_manifest": active_sharded_manifest or str(memmap_registry.get("active_manifest_json", "") or ""),
        "active_memmap_source_market_dataset_id": active_memmap_source,
        "active_memmap_matches_canonical_bundle": active_memmap_matches,
        "active_memmap_state": "current" if active_memmap_matches else "stale_or_missing",
        "active_feature_profile": str(sharded_status.get("latest_profile", "") or memmap_registry.get("active_feature_profile", "") or ""),
        "active_scope": str(sharded_status.get("latest_scope", "") or memmap_registry.get("active_scope", "") or ""),
        "active_sample_count": 0 if sharded_full_ready else int(memmap_registry.get("active_sample_count", 0) or 0),
        "sharded_memmap_registry_json": str((resolved.registry_dir / "sharded_memmap_registry.json").as_posix()),
        "sharded_memmap_registry_exists": bool((resolved.registry_dir / "sharded_memmap_registry.json").exists()),
        "latest_sharded_manifest": str(sharded_registry.get("latest_manifest_json", "") or sharded_status.get("latest_manifest_json", "") or ""),
        "latest_sharded_scope": str(sharded_registry.get("latest_scope", "") or sharded_status.get("latest_scope", "") or ""),
        "latest_sharded_status": str(sharded_status.get("status", "") or sharded_registry.get("status", "") or ""),
        "latest_sharded_stored_shard_count": int(sharded_status.get("latest_stored_shard_count", 0) or 0),
        "latest_sharded_planned_shard_count": int(sharded_status.get("latest_planned_shard_count", 0) or 0),
    }


def _active_component_coverage(root_manifest: Mapping[str, Any], *, paths: QdpPaths) -> dict[str, dict[str, Any]]:
    component_ids = dict(root_manifest.get("canonical_component_dataset_ids", {}) or {})
    if not component_ids:
        return {}
    try:
        from quant_data_platform.lake.catalog import ResearchDataLake

        lake_root = Path(str(root_manifest.get("primary_lake_root", "") or paths.lake_root))
        lake = ResearchDataLake(lake_root)
    except Exception as exc:
        return {
            str(domain): {"dataset_id": str(dataset_id), "status": "unavailable", "error": str(exc)}
            for domain, dataset_id in component_ids.items()
        }
    coverage: dict[str, dict[str, Any]] = {}
    for domain, dataset_id in component_ids.items():
        domain_text = str(domain)
        dataset_text = str(dataset_id or "")
        if not dataset_text:
            continue
        try:
            metadata = lake.describe_dataset(dataset_text)
            row_counts = dict(metadata.get("row_counts", {}) or {})
            coverage[domain_text] = {
                "dataset_id": dataset_text,
                "status": "ok",
                "dataset_kind": str(metadata.get("dataset_kind", "") or ""),
                "start_date": str(metadata.get("start_date", "") or ""),
                "end_date": str(metadata.get("end_date", "") or ""),
                "row_count": int(row_counts.get("silver_domain_data", 0) or row_counts.get("bronze_market_data", 0) or 0),
            }
        except Exception as exc:
            coverage[domain_text] = {"dataset_id": dataset_text, "status": "missing", "error": str(exc)}
    return coverage
