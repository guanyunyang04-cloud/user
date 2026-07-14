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
AUTHORIZED_LINEAGE_GAP_STATUS = "complete_with_authorized_lineage_gap"


def _canonical_contract_sha256(payload: dict[str, Any]) -> str:
    import hashlib
    import json

    encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _missing_lineage_references(root: Path, missing_dataset_ids: list[str]) -> list[dict[str, Any]]:
    """Preserve compact evidence showing how each unavailable ancestor was referenced.

    Old v2 manifests can repeat the same source path once per shard.  The
    substitute freeze proof needs the relationship and a bounded sample, not a
    second multi-megabyte copy of every repeated path.
    """

    missing = set(missing_dataset_ids)
    grouped: dict[tuple[str, str], dict[str, Any]] = {}
    for manifest_path in sorted((root / "datasets").glob("*/*/dataset.json")):
        manifest = read_v2_dataset_manifest(manifest_path)
        direct_fields: dict[str, list[str]] = {}
        for key, value in dict(manifest.source or {}).items():
            dataset_id = str(value or "")
            if str(key).endswith("dataset_id") and dataset_id in missing:
                direct_fields.setdefault(dataset_id, []).append(str(key))
        shard_counts: dict[str, int] = {}
        shard_samples: dict[str, list[str]] = {}
        for shard in manifest.shards:
            source_path = str(shard.source_path or "")
            if not source_path:
                continue
            for dataset_id in missing:
                if dataset_id not in source_path:
                    continue
                shard_counts[dataset_id] = shard_counts.get(dataset_id, 0) + 1
                samples = shard_samples.setdefault(dataset_id, [])
                if len(samples) < 3:
                    samples.append(source_path)
        for dataset_id in sorted(set(direct_fields) | set(shard_counts)):
            grouped[(dataset_id, manifest.dataset_id)] = {
                "missing_dataset_id": dataset_id,
                "referencing_dataset_id": manifest.dataset_id,
                "referencing_domain": manifest.domain,
                "referencing_manifest": str(manifest_path.resolve()),
                "source_fields": sorted(direct_fields.get(dataset_id, [])),
                "shard_source_path_reference_count": int(shard_counts.get(dataset_id, 0)),
                "shard_source_path_samples": shard_samples.get(dataset_id, []),
            }
    return [grouped[key] for key in sorted(grouped)]


def validate_v2_freeze_proof(payload: dict[str, Any]) -> dict[str, Any]:
    """Validate either a complete v2 freeze or an explicit gap substitute.

    The exception does not claim that missing ancestors were reconstructed. It
    proves the active pointer and every extant manifest/shard byte, records the
    exact gaps, and keeps deletion GC forbidden.  A caller must not infer more
    than that contract states.
    """

    issues: list[str] = []
    missing_ids = sorted(str(item) for item in list(payload.get("missing_dataset_ids", []) or []))
    missing_files = list(payload.get("missing_files", []) or [])
    datasets = list(payload.get("datasets", []) or [])
    extant_hashed = bool(payload.get("extant_shards_fully_hashed", payload.get("hash_shards", False)))
    if not str(payload.get("v2_active_sha256", "") or ""):
        issues.append("v2_active_sha256_missing")
    if missing_files:
        issues.append("extant_file_missing")
    if not datasets:
        issues.append("extant_dataset_inventory_empty")
    for dataset in datasets:
        if not str(dataset.get("manifest_sha256", "") or ""):
            issues.append("extant_manifest_hash_missing")
            break
        if any(not str(shard.get("sha256", "") or "") for shard in list(dataset.get("shards", []) or [])):
            issues.append("extant_shard_hash_missing")
            break
    if not extant_hashed:
        issues.append("extant_shards_not_fully_hashed")

    status = str(payload.get("status", "") or "")
    proof_kind = ""
    if status == "complete":
        proof_kind = "complete_lineage"
        if missing_ids:
            issues.append("complete_status_has_missing_lineage")
    elif status == AUTHORIZED_LINEAGE_GAP_STATUS:
        proof_kind = "authorized_lineage_gap"
        contract = dict(payload.get("authorized_lineage_gap_contract", {}) or {})
        contract_hash = str(contract.pop("contract_sha256", "") or "")
        if not missing_ids:
            issues.append("authorized_gap_has_no_missing_lineage")
        if sorted(str(item) for item in list(contract.get("missing_dataset_ids", []) or [])) != missing_ids:
            issues.append("authorized_gap_dataset_ids_mismatch")
        if str(contract.get("v2_active_sha256", "") or "") != str(payload.get("v2_active_sha256", "") or ""):
            issues.append("authorized_gap_active_sha_mismatch")
        if not str(contract.get("authorization_note", "") or "").strip():
            issues.append("authorized_gap_note_missing")
        if str(contract.get("non_claim", "") or "") != "missing_ancestor_content_was_not_reconstructed":
            issues.append("authorized_gap_non_claim_missing")
        if contract_hash != _canonical_contract_sha256(contract):
            issues.append("authorized_gap_contract_hash_mismatch")
    else:
        issues.append("freeze_status_not_publishable")

    return {
        "status": "accepted" if not issues else "rejected",
        "acceptable": not issues,
        "proof_kind": proof_kind,
        "issue_count": len(issues),
        "issues": issues,
        "missing_dataset_count": len(missing_ids),
    }


def v2_freeze_proof_is_acceptable(payload: dict[str, Any]) -> bool:
    return bool(validate_v2_freeze_proof(payload).get("acceptable", False))


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
    accept_missing_lineage: bool = False,
    authorization_note: str = "",
) -> dict[str, Any]:
    if accept_missing_lineage and not str(authorization_note or "").strip():
        raise ValueError("qdp_v2_missing_lineage_authorization_note_required")
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
    extant_shards_fully_hashed = (
        bool(hash_shards)
        and not missing_files
        and bool(file_hashes)
        and all(str(item.get("sha256", "") or "") for item in file_hashes.values())
    )
    authorized_lineage_gap_contract: dict[str, Any] = {}
    if missing_dataset_ids and accept_missing_lineage and extant_shards_fully_hashed and not missing_files:
        contract_payload = {
            "contract_version": 1,
            "accepted_at": utc_now(),
            "authorization_actor": "user_authorized_operator",
            "authorization_note": str(authorization_note).strip(),
            "v2_active_sha256": before_sha,
            "missing_dataset_ids": sorted(missing_dataset_ids),
            "missing_lineage_references": _missing_lineage_references(v2_root, missing_dataset_ids),
            "proof_scope": "v2_active_pointer_and_all_extant_manifests_and_shards",
            "non_claim": "missing_ancestor_content_was_not_reconstructed",
            "release_boundary": "may_unblock_immutable_v3_rebuild_and_release_only_after_all_v3_semantic_gates_pass",
            "deletion_boundary": "v2_deletion_gc_remains_forbidden_until_v3_first_publish_and_one_successful_incremental_update",
        }
        authorized_lineage_gap_contract = {
            **contract_payload,
            "contract_sha256": _canonical_contract_sha256(contract_payload),
        }
    if missing_files:
        status = "blocked_missing_files"
    elif missing_dataset_ids:
        status = AUTHORIZED_LINEAGE_GAP_STATUS if authorized_lineage_gap_contract else "blocked_missing_lineage"
    else:
        status = "complete" if extant_shards_fully_hashed else "metadata_only"
    payload = {
        "manifest_version": 3,
        "status": status,
        "frozen_at": utc_now(),
        "v2_root": str(v2_root.resolve()),
        "v2_active_path": str(active_path.resolve()),
        "v2_active_sha256": before_sha,
        "hash_shards": bool(extant_shards_fully_hashed),
        "lineage_complete": not missing_dataset_ids,
        "extant_shards_fully_hashed": bool(extant_shards_fully_hashed),
        "datasets": datasets,
        "pinned_dataset_ids": sorted(pinned_ids),
        "missing_dataset_ids": missing_dataset_ids,
        "missing_files": missing_files,
        "authorized_lineage_gap_contract": authorized_lineage_gap_contract,
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
    proof_validation = validate_v2_freeze_proof(payload)
    return {
        "status": payload["status"],
        "v2_active_sha256": before_sha,
        "dataset_count": len(datasets),
        "pinned_dataset_count": len(pinned_ids),
        "missing_dataset_count": len(missing_dataset_ids),
        "missing_file_count": len(missing_files),
        "hashed_file_count": sum(bool(item.get("sha256")) for item in file_hashes.values()),
        "lineage_complete": not missing_dataset_ids,
        "extant_shards_fully_hashed": bool(extant_shards_fully_hashed),
        "authorized_lineage_gap": bool(authorized_lineage_gap_contract),
        "proof_validation": proof_validation,
        "freeze_manifest": str(checkpoint_path.resolve()),
        "pin_manifest": str(pin_path.resolve()),
    }
