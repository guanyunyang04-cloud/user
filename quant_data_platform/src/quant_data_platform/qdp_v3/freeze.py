from __future__ import annotations

from pathlib import Path
from typing import Any

from quant_data_platform.core.json_io import read_json
from quant_data_platform.qdp_v2.manifest import dataset_manifest_for_id as v2_manifest_for_id
from quant_data_platform.qdp_v2.manifest import qdp_v2_root, read_dataset_manifest as read_v2_dataset_manifest
from quant_data_platform.qdp_v2.status import _active_dataset_refs
from quant_data_platform.qdp_v2.gc import _dataset_ancestor_closure
from quant_data_platform.qdp_v3.manifest import atomic_write_json, sha256_file, utc_now
from quant_data_platform.qdp_v3.paths import ensure_qdp_v3_layout


EXPECTED_V2_ACTIVE_SHA256 = "e56f72a6cba8bcf86055817f6a0ec5e7391271fb3c27b4d628c3abc62944051e"


def _all_v2_intraday_dataset_ids(root: Path) -> set[str]:
    ids: set[str] = set()
    for domain in ("market_intraday_1m", "market_intraday_5m"):
        domain_root = root / "datasets" / domain
        if not domain_root.exists():
            continue
        ids.update(path.name for path in domain_root.iterdir() if path.is_dir() and not path.name.startswith(".") and (path / "dataset.json").exists())
    return ids


def _all_existing_v2_dataset_ids(root: Path) -> set[str]:
    ids: set[str] = set()
    datasets_root = root / "datasets"
    if not datasets_root.exists():
        return ids
    for manifest_path in datasets_root.glob("*/*/dataset.json"):
        try:
            ids.add(read_v2_dataset_manifest(manifest_path).dataset_id)
        except Exception:
            # A malformed manifest is surfaced later as a missing/unreadable
            # freeze input; never infer that its directory is safe to delete.
            ids.add(manifest_path.parent.name)
    return ids


def freeze_v2(
    *,
    workspace_root: str | Path | None = None,
    hash_shards: bool = True,
    expected_active_sha256: str = EXPECTED_V2_ACTIVE_SHA256,
) -> dict[str, Any]:
    paths = ensure_qdp_v3_layout(workspace_root)
    v2_root = qdp_v2_root(workspace_root)
    active_path = v2_root / "active" / "active.json"
    if not active_path.exists():
        raise FileNotFoundError(f"qdp_v2_active_missing:{active_path}")
    before_sha = sha256_file(active_path)
    if expected_active_sha256 and before_sha.lower() != str(expected_active_sha256).lower():
        raise RuntimeError(f"qdp_v2_active_sha_unexpected:expected={expected_active_sha256} actual={before_sha}")
    active = read_json(active_path)
    checkpoint_path = paths.metadata / "v2_freeze_20260713.json"
    checkpoint = read_json(checkpoint_path)
    file_hashes = dict(checkpoint.get("file_hashes", {}) or {})
    datasets: list[dict[str, Any]] = []
    pinned_ids = {dataset_id for _, _, dataset_id in _active_dataset_refs(active)}
    pinned_ids.update(_all_v2_intraday_dataset_ids(v2_root))
    pinned_ids.update(_all_existing_v2_dataset_ids(v2_root))
    processed_since_checkpoint = 0
    pinned_ids = _dataset_ancestor_closure(v2_root, pinned_ids)
    missing_dataset_ids: list[str] = []
    missing_files: list[dict[str, str]] = []
    for dataset_id in sorted(pinned_ids):
        manifest_path = v2_manifest_for_id(v2_root, dataset_id)
        if manifest_path is None:
            missing_dataset_ids.append(dataset_id)
            continue
        manifest = read_v2_dataset_manifest(manifest_path)
        domain = manifest.domain
        manifest_key = str(manifest_path.resolve())
        file_hashes[manifest_key] = {
            "sha256": sha256_file(manifest_path),
            "size": int(manifest_path.stat().st_size),
            "kind": "dataset_manifest",
        }
        shard_records: list[dict[str, Any]] = []
        for shard in manifest.shards:
            shard_path = Path(shard.path)
            if not shard_path.is_absolute():
                shard_path = (v2_root / shard_path).resolve()
            if not shard_path.exists():
                missing_files.append({"dataset_id": dataset_id, "domain": domain, "path": str(shard_path), "kind": "shard"})
                shard_records.append({"path": str(shard_path), "sha256": "", "size": 0, "status": "missing"})
                continue
            key = str(shard_path.resolve())
            stat = shard_path.stat()
            existing = dict(file_hashes.get(key, {}) or {})
            if hash_shards and not (
                existing.get("sha256")
                and int(existing.get("size", -1)) == int(stat.st_size)
                and int(existing.get("mtime_ns", -1)) == int(stat.st_mtime_ns)
            ):
                existing["sha256"] = sha256_file(shard_path)
            existing.update(
                {
                    "size": int(stat.st_size),
                    "mtime_ns": int(stat.st_mtime_ns),
                    "kind": "shard",
                    "dataset_id": dataset_id,
                    "domain": domain,
                }
            )
            file_hashes[key] = existing
            shard_records.append({"path": key, "sha256": str(existing.get("sha256", "")), "size": int(stat.st_size)})
            processed_since_checkpoint += 1
            if processed_since_checkpoint >= 100:
                atomic_write_json(
                    checkpoint_path,
                    {
                        "manifest_version": 3,
                        "status": "hashing",
                        "v2_active_sha256": before_sha,
                        "file_hashes": file_hashes,
                        "updated_at": utc_now(),
                    },
                )
                processed_since_checkpoint = 0
        datasets.append(
            {
                "domain": domain,
                "dataset_id": dataset_id,
                "manifest_path": manifest_key,
                "manifest_sha256": file_hashes[manifest_key]["sha256"],
                "shards": shard_records,
            }
        )
    after_sha = sha256_file(active_path)
    if after_sha != before_sha:
        raise RuntimeError(f"qdp_v2_active_changed_during_freeze:before={before_sha} after={after_sha}")
    fully_hashed = (
        bool(hash_shards)
        and not missing_dataset_ids
        and not missing_files
        and bool(file_hashes)
        and all(str(item.get("sha256", "") or "") for item in file_hashes.values())
    )
    status = "blocked_missing_lineage" if missing_dataset_ids or missing_files else "complete" if fully_hashed else "metadata_only"
    payload = {
        "manifest_version": 3,
        "status": status,
        "frozen_at": utc_now(),
        "v2_root": str(v2_root.resolve()),
        "v2_active_path": str(active_path.resolve()),
        "v2_active_sha256": before_sha,
        "hash_shards": bool(fully_hashed),
        "datasets": datasets,
        "pinned_dataset_ids": sorted(pinned_ids),
        "missing_dataset_ids": missing_dataset_ids,
        "missing_files": missing_files,
        "file_hashes": file_hashes,
        "legacy_research_factor_state": "provisional_due_to_unverified_adjust_factor_semantics",
        "deletion_gc_forbidden_until": "v3_first_publish_and_one_successful_incremental_update",
    }
    atomic_write_json(checkpoint_path, payload)
    pin_path = v2_root / "pins" / "qdp_v3_m0_freeze.json"
    atomic_write_json(
        pin_path,
        {
            "version": 1,
            "pin_name": "qdp_v3_m0_freeze",
            "created_at": utc_now(),
            "dataset_ids": sorted(pinned_ids),
            "missing_dataset_ids": missing_dataset_ids,
            "reason": "Preserve v2 active, all 1m/5m evidence, and rollback base until v3 is proven.",
            "active_sha256": before_sha,
        },
    )
    return {
        "status": payload["status"],
        "v2_active_sha256": before_sha,
        "dataset_count": len(datasets),
        "pinned_dataset_count": len(pinned_ids),
        "missing_dataset_count": len(missing_dataset_ids),
        "missing_file_count": len(missing_files),
        "hashed_file_count": sum(bool(item.get("sha256")) for item in file_hashes.values()),
        "freeze_manifest": str(checkpoint_path.resolve()),
        "pin_manifest": str(pin_path.resolve()),
    }
