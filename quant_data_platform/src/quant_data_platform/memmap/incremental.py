from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from quant_data_platform.core.json_io import read_json, utc_now, write_json
from quant_data_platform.core.paths import QdpPaths, qdp_paths
from quant_data_platform.core.registry import load_root_manifest, write_root_manifest_update


TERMINAL_SHARD_STATUSES = {"completed", "empty", "no_coverage"}


@dataclass(frozen=True)
class IncrementalPlanConfig:
    base_manifest: str = ""
    rebuild_start_year: int = 0
    rebuild_end_year: int = 0
    recent_years: int = 1
    tag: str = ""

    def normalized(self) -> "IncrementalPlanConfig":
        return IncrementalPlanConfig(
            base_manifest=str(self.base_manifest or "").strip(),
            rebuild_start_year=max(int(self.rebuild_start_year or 0), 0),
            rebuild_end_year=max(int(self.rebuild_end_year or 0), 0),
            recent_years=max(int(self.recent_years or 1), 1),
            tag=str(self.tag or "").strip(),
        )


def freeze_sharded_memmap(
    paths: QdpPaths | None = None,
    *,
    manifest: str | Path | None = None,
    tag: str = "",
    write: bool = True,
) -> dict[str, Any]:
    resolved = paths or qdp_paths()
    manifest_path = _resolve_manifest_path(resolved, manifest)
    payload = _read_qdp_sharded_manifest(manifest_path)
    if str(payload.get("status", "")) != "completed":
        raise ValueError(f"cannot_freeze_incomplete_sharded_memmap: {manifest_path}")
    freeze_tag = _safe_name(tag or f"{payload.get('profile', 'profile')}_{payload.get('start_year', '')}_{payload.get('end_year', '')}")
    entry = {
        "status": "frozen",
        "artifact_type": "qdp_frozen_sharded_memmap",
        "freeze_tag": freeze_tag,
        "manifest_json": str(manifest_path.resolve()),
        "canonical_dataset_id": str(payload.get("canonical_dataset_id", "") or ""),
        "profile": str(payload.get("profile", "") or payload.get("feature_profile", "") or ""),
        "feature_schema_hash": str(payload.get("feature_schema_hash", "") or ""),
        "feature_count": int(payload.get("feature_count", 0) or 0),
        "start_year": int(payload.get("start_year", 0) or 0),
        "end_year": int(payload.get("end_year", 0) or 0),
        "years": [int(year) for year in list(payload.get("years", []) or [])],
        "symbol_count": int(payload.get("symbol_count", 0) or 0),
        "planned_shard_count": int(payload.get("planned_shard_count", 0) or 0),
        "stored_shard_count": int(payload.get("stored_shard_count", 0) or 0),
        "empty_shard_count": int(payload.get("empty_shard_count", 0) or 0),
        "frozen_at": utc_now(),
        "reuse_policy": "reuse_by_manifest_pointer_no_data_copy",
    }
    registry_path = resolved.registry_dir / "frozen_sharded_memmap_registry.json"
    registry = read_json(registry_path)
    entries = [
        dict(item)
        for item in list(registry.get("entries", []) or [])
        if str(dict(item).get("manifest_json", "") or "") != str(entry["manifest_json"])
    ]
    entries.append(entry)
    registry.update(
        {
            "schema_version": 1,
            "status": "ready",
            "active_frozen_manifest_json": str(entry["manifest_json"]),
            "latest_freeze_tag": freeze_tag,
            "entries": entries,
            "updated_at": utc_now(),
        }
    )
    if write:
        write_json(registry_path, registry)
        write_root_manifest_update(
            {
                "canonical_sharded_memmap_freeze": {
                    "status": "frozen_base_ready",
                    "active_frozen_manifest_json": str(entry["manifest_json"]),
                    "latest_freeze_tag": freeze_tag,
                    "canonical_dataset_id": str(entry["canonical_dataset_id"]),
                    "profile": str(entry["profile"]),
                    "feature_schema_hash": str(entry["feature_schema_hash"]),
                    "updated_at": utc_now(),
                }
            },
            paths=resolved,
        )
    return {**entry, "registry_json": str(registry_path.as_posix()), "write": bool(write)}


def plan_incremental_memmap(
    paths: QdpPaths | None = None,
    *,
    config: IncrementalPlanConfig | Mapping[str, Any] | None = None,
    write: bool = True,
) -> dict[str, Any]:
    resolved = paths or qdp_paths()
    cfg = _coerce_plan_config(config)
    base_path = _resolve_manifest_path(resolved, cfg.base_manifest or None)
    base = _read_qdp_sharded_manifest(base_path)
    root = load_root_manifest(resolved)
    target_dataset_id = str(root.get("canonical_dataset_id", "") or base.get("canonical_dataset_id", "") or "")
    base_start_year = int(base.get("start_year", 0) or min([int(y) for y in list(base.get("years", []) or [0])] or [0]))
    base_end_year = int(base.get("end_year", 0) or max([int(y) for y in list(base.get("years", []) or [0])] or [0]))
    rebuild_end = cfg.rebuild_end_year or base_end_year
    rebuild_start = cfg.rebuild_start_year or max(base_start_year, rebuild_end - cfg.recent_years + 1)
    rebuild_years = list(range(int(rebuild_start), int(rebuild_end) + 1))
    frozen_years = [int(year) for year in list(base.get("years", []) or []) if int(year) < int(rebuild_start)]
    tag = _safe_name(cfg.tag or f"{base.get('profile', 'profile')}_incremental_{rebuild_start}_{rebuild_end}")
    source_equivalence_required = bool(target_dataset_id and str(base.get("canonical_dataset_id", "") or "") != target_dataset_id)
    invalidation_reasons = []
    if source_equivalence_required:
        invalidation_reasons.append("canonical_dataset_id_changed_verify_frozen_year_source_equivalence")
    invalidation_reasons.append("current_tail_labels_within_horizon_may_change")
    payload = {
        "status": "planned",
        "artifact_type": "qdp_incremental_sharded_memmap_plan",
        "base_manifest_json": str(base_path.resolve()),
        "base_canonical_dataset_id": str(base.get("canonical_dataset_id", "") or ""),
        "target_canonical_dataset_id": target_dataset_id,
        "source_equivalence_required": source_equivalence_required,
        "profile": str(base.get("profile", "") or base.get("feature_profile", "") or ""),
        "feature_schema_hash": str(base.get("feature_schema_hash", "") or ""),
        "symbol_block_size": int(base.get("symbol_block_size", 0) or 0),
        "lookback_days": int(base.get("lookback_days", 0) or 0),
        "horizon": int(base.get("horizon", 0) or 0),
        "execution_mode": str(base.get("execution_mode", "") or ""),
        "base_start_year": base_start_year,
        "base_end_year": base_end_year,
        "frozen_years": frozen_years,
        "rebuild_years": rebuild_years,
        "rebuild_start_year": int(rebuild_start),
        "rebuild_end_year": int(rebuild_end),
        "invalidation_reasons": invalidation_reasons,
        "recommended_incremental_tag": tag,
        "recommended_build_command": (
            "qdp build-sharded-memmap "
            f"--profile {base.get('profile', '') or base.get('feature_profile', '')} "
            f"--start-year {int(rebuild_start)} --end-year {int(rebuild_end)} "
            f"--symbol-block-size {int(base.get('symbol_block_size', 300) or 300)} "
            f"--tag {tag}"
        ),
        "recommended_compose_command": (
            "qdp compose-sharded-memmap "
            f"--base-manifest \"{str(base_path.resolve())}\" "
            f"--overlay-manifest \"<incremental_manifest_json>\" "
            f"--tag {tag}_composite --activate"
        ),
        "created_at": utc_now(),
    }
    if write:
        plan_path = resolved.memmap_dir / "incremental" / f"{tag}_plan.json"
        write_json(plan_path, payload)
        payload["plan_json"] = str(plan_path.as_posix())
    return payload


def compose_sharded_memmap(
    paths: QdpPaths | None = None,
    *,
    base_manifest: str | Path | None = None,
    overlay_manifests: Sequence[str | Path] | None = None,
    tag: str = "",
    activate: bool = False,
    write: bool = True,
) -> dict[str, Any]:
    resolved = paths or qdp_paths()
    base_path = _resolve_manifest_path(resolved, base_manifest)
    base = _read_qdp_sharded_manifest(base_path)
    overlays = [_read_qdp_sharded_manifest(Path(item)) for item in list(overlay_manifests or [])]
    if not overlays:
        raise ValueError("overlay_manifest_required")
    for overlay in overlays:
        _assert_overlay_compatible(base, overlay)

    shard_by_key: dict[str, dict[str, Any]] = {}
    source_by_key: dict[str, str] = {}
    for shard in list(base.get("shards", []) or []):
        key = str(dict(shard).get("shard_key", "") or "")
        if not key:
            continue
        shard_by_key[key] = dict(shard)
        source_by_key[key] = str(base_path.resolve())
    for overlay_path, overlay in zip(list(overlay_manifests or []), overlays):
        overlay_source = str(Path(overlay_path).resolve())
        for shard in list(overlay.get("shards", []) or []):
            key = str(dict(shard).get("shard_key", "") or "")
            if not key:
                continue
            shard_by_key[key] = dict(shard)
            source_by_key[key] = overlay_source

    shards = [shard_by_key[key] for key in sorted(shard_by_key, key=_shard_sort_key)]
    stored = [item for item in shards if str(item.get("status", "") or "") == "completed"]
    empty = [item for item in shards if str(item.get("status", "") or "") in {"empty", "no_coverage"}]
    failed = [item for item in shards if str(item.get("status", "") or "") not in TERMINAL_SHARD_STATUSES]
    years = sorted({int(item.get("year", 0) or _year_from_key(str(item.get("shard_key", "") or "")) or 0) for item in shards})
    years = [year for year in years if year > 0]
    root = load_root_manifest(resolved)
    target_dataset_id = str(root.get("canonical_dataset_id", "") or overlays[-1].get("canonical_dataset_id", "") or base.get("canonical_dataset_id", "") or "")
    base_dataset_id = str(base.get("canonical_dataset_id", "") or "")
    overlay_dataset_ids = sorted({str(item.get("canonical_dataset_id", "") or "") for item in overlays if str(item.get("canonical_dataset_id", "") or "")})
    source_equivalence_required = bool(target_dataset_id and base_dataset_id and base_dataset_id != target_dataset_id)
    composite_tag = _safe_name(tag or f"{base.get('profile', 'profile')}_composite_{utc_now().replace(':', '').replace('-', '')}")
    out_root = resolved.memmap_dir / "sharded" / "composites" / composite_tag
    manifest_path = out_root / "sharded_memmap_manifest.json"
    composite = {
        **{key: value for key, value in base.items() if key not in {"shards", "manifest_json", "created_at", "status", "scope"}},
        "artifact_type": "qdp_sharded_memmap",
        "schema_version": 1,
        "status": "completed" if not failed else "partial",
        "scope": "full_canonical_incremental_composite",
        "canonical_dataset_id": target_dataset_id,
        "base_manifest_json": str(base_path.resolve()),
        "overlay_manifest_jsons": [str(Path(item).resolve()) for item in list(overlay_manifests or [])],
        "base_canonical_dataset_id": base_dataset_id,
        "overlay_canonical_dataset_ids": overlay_dataset_ids,
        "source_equivalence_required": source_equivalence_required,
        "source_reuse_contract": {
            "policy": "overlay_replaces_matching_shard_key_and_reuses_unmodified_base_shards_by_pointer",
            "base_manifest_json": str(base_path.resolve()),
            "source_equivalence_required": source_equivalence_required,
        },
        "years": years,
        "start_year": min(years) if years else int(base.get("start_year", 0) or 0),
        "end_year": max(years) if years else int(base.get("end_year", 0) or 0),
        "planned_shard_count": int(len(shards)),
        "processed_shard_count": int(len(shards)),
        "stored_shard_count": int(len(stored)),
        "empty_shard_count": int(len(empty)),
        "failed_shard_count": int(len(failed)),
        "shards": shards,
        "shard_source_manifest_by_key": source_by_key,
        "created_at": utc_now(),
        "manifest_json": str(manifest_path.resolve()),
    }
    if write:
        write_json(manifest_path, composite)
        _register_composite_manifest(resolved, composite, activate=bool(activate))
    return composite


def _resolve_manifest_path(paths: QdpPaths, manifest: str | Path | None = None) -> Path:
    if manifest:
        path = Path(manifest)
        if path.exists():
            return path
        raise FileNotFoundError(f"sharded_manifest_not_found: {path}")
    sharded_registry = read_json(paths.registry_dir / "sharded_memmap_registry.json")
    root_manifest = load_root_manifest(paths)
    raw = str(
        sharded_registry.get("active_manifest_json", "")
        or dict(root_manifest.get("canonical_sharded_memmap_status", {}) or {}).get("active_manifest_json", "")
        or sharded_registry.get("latest_manifest_json", "")
        or ""
    ).strip()
    if not raw:
        raise FileNotFoundError("active_sharded_manifest_not_found")
    path = Path(raw)
    if not path.exists():
        raise FileNotFoundError(f"sharded_manifest_not_found: {path}")
    return path


def _read_qdp_sharded_manifest(path: str | Path) -> dict[str, Any]:
    resolved = Path(path)
    payload = read_json(resolved)
    if str(payload.get("artifact_type", "")) != "qdp_sharded_memmap":
        raise ValueError(f"not_qdp_sharded_memmap: {resolved}")
    return payload


def _assert_overlay_compatible(base: Mapping[str, Any], overlay: Mapping[str, Any]) -> None:
    checks = [
        "profile",
        "feature_schema_hash",
        "feature_count",
        "symbol_block_size",
        "lookback_days",
        "horizon",
        "execution_mode",
    ]
    mismatches = []
    for key in checks:
        if str(base.get(key, "") or "") != str(overlay.get(key, "") or ""):
            mismatches.append(key)
    if mismatches:
        raise ValueError(f"incompatible_overlay_manifest: {','.join(mismatches)}")


def _register_composite_manifest(paths: QdpPaths, manifest: Mapping[str, Any], *, activate: bool) -> None:
    registry_path = paths.registry_dir / "sharded_memmap_registry.json"
    registry = read_json(registry_path)
    entries = [
        item
        for item in list(registry.get("entries", []) or [])
        if str(dict(item).get("manifest_json", "") or "") != str(manifest.get("manifest_json", "") or "")
    ]
    entry = {
        "manifest_json": str(manifest.get("manifest_json", "") or ""),
        "status": str(manifest.get("status", "") or ""),
        "scope": str(manifest.get("scope", "") or ""),
        "canonical_dataset_id": str(manifest.get("canonical_dataset_id", "") or ""),
        "profile": str(manifest.get("profile", "") or ""),
        "feature_schema_hash": str(manifest.get("feature_schema_hash", "") or ""),
        "stored_shard_count": int(manifest.get("stored_shard_count", 0) or 0),
        "processed_shard_count": int(manifest.get("processed_shard_count", 0) or 0),
        "empty_shard_count": int(manifest.get("empty_shard_count", 0) or 0),
        "planned_shard_count": int(manifest.get("planned_shard_count", 0) or 0),
        "registered_at": utc_now(),
    }
    entries.append(entry)
    registry.update(
        {
            "schema_version": 1,
            "status": "ready",
            "entries": entries,
            "latest_manifest_json": str(manifest.get("manifest_json", "") or ""),
            "latest_scope": str(manifest.get("scope", "") or ""),
            "updated_at": utc_now(),
        }
    )
    if activate and str(manifest.get("status", "")) == "completed":
        registry["active_manifest_json"] = str(manifest.get("manifest_json", "") or "")
    write_json(registry_path, registry)
    status = {
        "status": "sharded_full_ready" if activate and str(manifest.get("status", "")) == "completed" else "sharded_composite_ready",
        "latest_manifest_json": str(manifest.get("manifest_json", "") or ""),
        "latest_scope": str(manifest.get("scope", "") or ""),
        "latest_stored_shard_count": int(manifest.get("stored_shard_count", 0) or 0),
        "latest_processed_shard_count": int(manifest.get("processed_shard_count", 0) or 0),
        "latest_empty_shard_count": int(manifest.get("empty_shard_count", 0) or 0),
        "latest_planned_shard_count": int(manifest.get("planned_shard_count", 0) or 0),
        "latest_profile": str(manifest.get("profile", "") or ""),
        "latest_canonical_dataset_id": str(manifest.get("canonical_dataset_id", "") or ""),
        "updated_at": utc_now(),
    }
    if activate and str(manifest.get("status", "")) == "completed":
        status["active_manifest_json"] = str(manifest.get("manifest_json", "") or "")
    write_root_manifest_update({"canonical_sharded_memmap_status": status}, paths=paths)


def _coerce_plan_config(config: IncrementalPlanConfig | Mapping[str, Any] | None) -> IncrementalPlanConfig:
    if isinstance(config, IncrementalPlanConfig):
        return config.normalized()
    if isinstance(config, Mapping):
        return IncrementalPlanConfig(**dict(config)).normalized()
    return IncrementalPlanConfig().normalized()


def _shard_sort_key(key: str) -> tuple[int, int, str]:
    year = _year_from_key(key)
    block = 0
    for part in str(key).split("/"):
        if part.startswith("block="):
            try:
                block = int(part.split("=", 1)[1])
            except ValueError:
                block = 0
    return (year, block, str(key))


def _year_from_key(key: str) -> int:
    for part in str(key).split("/"):
        if part.startswith("year="):
            try:
                return int(part.split("=", 1)[1])
            except ValueError:
                return 0
    return 0


def _safe_name(value: str) -> str:
    return "".join(ch if ch.isalnum() or ch in {"_", "-", "."} else "_" for ch in str(value or "").strip()) or "incremental"
