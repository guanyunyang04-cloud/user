from __future__ import annotations

import shutil
from datetime import datetime, timezone
from functools import wraps
from pathlib import Path
from typing import Any, Callable, Iterable

from quant_data_platform.core.json_io import read_json
from quant_data_platform.qdp_v3.manifest import dataset_manifest_for_id, iter_dataset_manifests, read_dataset_manifest, utc_now
from quant_data_platform.qdp_v3.paths import ensure_qdp_v3_layout
from quant_data_platform.qdp_v3.release import _active_manifest_lock, read_candidate


def _serialize_gc_mutation(operation: Callable[..., dict[str, Any]]) -> Callable[..., dict[str, Any]]:
    @wraps(operation)
    def wrapped(*args: Any, **kwargs: Any) -> dict[str, Any]:
        if bool(kwargs.get("apply")) or bool(kwargs.get("purge_trash")):
            with _active_manifest_lock(kwargs.get("workspace_root")):
                return operation(*args, **kwargs)
        return operation(*args, **kwargs)

    return wrapped


def _dataset_ids_from_mapping(payload: dict[str, Any]) -> set[str]:
    ids: set[str] = set()
    for key in ("datasets", "dataset_ids"):
        value = payload.get(key, {})
        if isinstance(value, dict):
            ids.update(str(item) for item in value.values() if str(item))
        elif isinstance(value, list):
            ids.update(str(item) for item in value if str(item))
    active = payload.get("active")
    if isinstance(active, dict):
        ids.update(_dataset_ids_from_mapping(active))
    return ids


def _candidate_dataset_ids(candidate_id: str, *, paths: Any) -> set[str]:
    try:
        candidate = read_candidate(candidate_id, workspace_root=paths.root.parent.parent.parent)
    except Exception:
        candidate_path = paths.candidates / f"{candidate_id}.json"
        payload = read_json(candidate_path)
        return _dataset_ids_from_mapping(payload)
    return set(candidate.datasets.values())


def _root_dataset_ids(paths: Any) -> tuple[set[str], list[dict[str, Any]]]:
    roots: set[str] = set()
    sources: list[dict[str, Any]] = []
    active = read_json(paths.active_manifest)
    active_ids = _dataset_ids_from_mapping(active)
    roots.update(active_ids)
    if active_ids:
        sources.append({"kind": "active", "path": str(paths.active_manifest), "dataset_count": len(active_ids)})
    for candidate_path in sorted(paths.candidates.glob("*.json")):
        payload = read_json(candidate_path)
        ids = _dataset_ids_from_mapping(payload)
        roots.update(ids)
        sources.append({"kind": "candidate", "path": str(candidate_path), "dataset_count": len(ids)})
    for pin_path in sorted(paths.pins.glob("*.json")):
        payload = read_json(pin_path)
        ids = _dataset_ids_from_mapping(payload)
        candidate_id = str(payload.get("candidate_id", "") or "")
        if candidate_id:
            ids.update(_candidate_dataset_ids(candidate_id, paths=paths))
        roots.update(ids)
        sources.append({"kind": "pin", "path": str(pin_path), "dataset_count": len(ids)})
    for rollback_path in sorted(paths.rollbacks.glob("*.json")):
        ids = _dataset_ids_from_mapping(read_json(rollback_path))
        roots.update(ids)
        sources.append({"kind": "rollback", "path": str(rollback_path), "dataset_count": len(ids)})
    for audit_path in sorted(paths.audits.glob("**/*.json")):
        payload = read_json(audit_path)
        ids = _dataset_ids_from_mapping(payload)
        candidate_id = str(payload.get("candidate_id", "") or "")
        if candidate_id:
            ids.update(_candidate_dataset_ids(candidate_id, paths=paths))
        roots.update(ids)
        sources.append({"kind": "audit", "path": str(audit_path), "dataset_count": len(ids)})
    return roots, sources


def reachable_dataset_ids(paths: Any) -> tuple[set[str], list[dict[str, Any]], list[dict[str, Any]]]:
    roots, sources = _root_dataset_ids(paths)
    reachable: set[str] = set()
    findings: list[dict[str, Any]] = []
    queue = list(sorted(roots))
    while queue:
        dataset_id = queue.pop()
        if dataset_id in reachable:
            continue
        path = dataset_manifest_for_id(paths.root, dataset_id)
        if path is None:
            findings.append({"code": "gc_root_or_input_manifest_missing", "dataset_id": dataset_id})
            reachable.add(dataset_id)
            continue
        manifest = read_dataset_manifest(path)
        reachable.add(dataset_id)
        for input_ref in manifest.inputs:
            if input_ref.dataset_id not in reachable:
                queue.append(input_ref.dataset_id)
    return reachable, sources, findings


def _reachable_raw_hashes(paths: Any, reachable_datasets: set[str]) -> set[str]:
    hashes: set[str] = set()
    for dataset_id in reachable_datasets:
        path = dataset_manifest_for_id(paths.root, dataset_id)
        if path is not None:
            hashes.update(read_dataset_manifest(path).raw_content_hashes)
    for candidate_path in sorted(paths.candidates.glob("*.json")):
        payload = read_json(candidate_path)
        hashes.update(str(item.get("content_sha256", "") or "") for item in list(payload.get("raw_partitions", []) or []) if isinstance(item, dict))
    if paths.raw.exists():
        for latest_path in paths.raw.glob("*/*/latest.json"):
            latest = read_json(latest_path)
            hashes.add(str(latest.get("content_sha256", "") or ""))
    hashes.discard("")
    return hashes


def _raw_version_inventory(paths: Any, *, reachable_hashes: set[str], now: datetime, min_age_days: int) -> list[dict[str, Any]]:
    inventory: list[dict[str, Any]] = []
    if not paths.raw.exists():
        return inventory
    for receipt_path in sorted(paths.raw.glob("*/*/versions/*/receipt.json")):
        version_dir = receipt_path.parent
        receipt = read_json(receipt_path)
        content_hash = str(receipt.get("content_sha256", "") or version_dir.name)
        age = _age_days(version_dir, now=now)
        try:
            relative = version_dir.relative_to(paths.raw)
        except ValueError:
            continue
        inventory.append(
            {
                "raw_domain": str(receipt.get("raw_domain", "") or relative.parts[0]),
                "partition_field": str(receipt.get("partition_field", "") or ""),
                "partition_value": str(receipt.get("partition_value", "") or ""),
                "content_sha256": content_hash,
                "path": str(version_dir.resolve()),
                "relative_path": str(relative).replace("\\", "/"),
                "reachable": content_hash in reachable_hashes,
                "age_days": round(age, 3),
                "eligible": content_hash not in reachable_hashes and age >= int(min_age_days),
            }
        )
    return inventory


def _age_days(path: Path, *, now: datetime) -> float:
    modified = datetime.fromtimestamp(path.stat().st_mtime, tz=timezone.utc)
    return max(0.0, (now - modified).total_seconds() / 86_400.0)


@_serialize_gc_mutation
def collect_garbage(
    *,
    workspace_root: str | Path | None = None,
    apply: bool = False,
    yes: bool = False,
    min_age_days: int = 30,
    purge_trash: bool = False,
    trash_retention_days: int = 7,
) -> dict[str, Any]:
    if (apply or purge_trash) and not yes:
        raise ValueError("qdp_v3_gc_mutation_requires_yes")
    paths = ensure_qdp_v3_layout(workspace_root)
    reachable, root_sources, findings = reachable_dataset_ids(paths)
    reachable_raw_hashes = _reachable_raw_hashes(paths, reachable)
    now = datetime.now(timezone.utc)
    inventory: list[dict[str, Any]] = []
    for manifest_path in iter_dataset_manifests(paths.root):
        manifest = read_dataset_manifest(manifest_path)
        dataset_dir = manifest_path.parent
        age = _age_days(dataset_dir, now=now)
        inventory.append(
            {
                "dataset_id": manifest.dataset_id,
                "domain": manifest.domain,
                "path": str(dataset_dir.resolve()),
                "reachable": manifest.dataset_id in reachable,
                "age_days": round(age, 3),
                "eligible": manifest.dataset_id not in reachable and age >= int(min_age_days),
            }
        )
    eligible = [item for item in inventory if item["eligible"]]
    raw_inventory = _raw_version_inventory(paths, reachable_hashes=reachable_raw_hashes, now=now, min_age_days=int(min_age_days))
    raw_eligible = [item for item in raw_inventory if item["eligible"]]
    moved: list[dict[str, Any]] = []
    if apply:
        stamp = utc_now().replace(":", "").replace("-", "")
        for item in eligible:
            source = Path(str(item["path"]))
            try:
                source.resolve().relative_to(paths.datasets.resolve())
            except ValueError:
                findings.append({"code": "gc_path_outside_dataset_root", "path": str(source)})
                continue
            target = paths.trash / stamp / str(item["domain"]) / str(item["dataset_id"])
            target.parent.mkdir(parents=True, exist_ok=True)
            source.replace(target)
            moved.append({**item, "trash_path": str(target.resolve())})
        for item in raw_eligible:
            source = Path(str(item["path"]))
            try:
                source.resolve().relative_to(paths.raw.resolve())
            except ValueError:
                findings.append({"code": "gc_raw_path_outside_raw_root", "path": str(source)})
                continue
            target = paths.trash / stamp / "raw" / str(item["relative_path"])
            target.parent.mkdir(parents=True, exist_ok=True)
            source.replace(target)
            moved.append({**item, "trash_path": str(target.resolve()), "kind": "raw_version"})
    purged: list[str] = []
    if purge_trash and paths.trash.exists():
        for batch in sorted(path for path in paths.trash.iterdir() if path.is_dir()):
            if _age_days(batch, now=now) >= int(trash_retention_days):
                shutil.rmtree(batch)
                purged.append(str(batch.resolve()))
    return {
        "status": "applied" if apply or purge_trash else "dry_run",
        "root": str(paths.root.resolve()),
        "reachable_dataset_count": len(reachable),
        "dataset_count": len(inventory),
        "eligible_count": len(eligible),
        "raw_version_count": len(raw_inventory),
        "reachable_raw_hash_count": len(reachable_raw_hashes),
        "raw_eligible_count": len(raw_eligible),
        "moved_count": len(moved),
        "purged_batch_count": len(purged),
        "min_age_days": int(min_age_days),
        "trash_retention_days": int(trash_retention_days),
        "root_sources": root_sources,
        "findings": findings,
        "eligible": eligible,
        "raw_eligible": raw_eligible,
        "moved": moved,
        "purged": purged,
    }
