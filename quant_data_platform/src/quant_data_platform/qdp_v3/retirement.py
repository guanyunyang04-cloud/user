from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from quant_data_platform.core.json_io import read_json
from quant_data_platform.core.paths import qdp_paths
from quant_data_platform.qdp_v2.manifest import active_manifest_path as v2_active_manifest_path
from quant_data_platform.qdp_v2.manifest import dataset_manifest_for_id as v2_dataset_manifest_for_id
from quant_data_platform.qdp_v2.manifest import qdp_v2_root
from quant_data_platform.qdp_v3.constants import DOMAIN_MARKET_INTRADAY_5M, QUALITY_STRICT
from quant_data_platform.qdp_v3.manifest import (
    active_manifest_sha256,
    atomic_write_json,
    dataset_manifest_for_id,
    read_dataset_manifest,
    resolve_manifest_path,
    sha256_file,
    stable_hash,
    utc_now,
)
from quant_data_platform.qdp_v3.paths import ensure_qdp_v3_layout
from quant_data_platform.qdp_v3.release import latest_candidate_audit


V2_INTRADAY_RETIREMENT_DOMAINS = (
    "market_intraday_1m",
    "market_intraday_5m",
    "intraday_daily_features",
    "limit_intraday_features",
)


def _downstream_consumer_references(
    *,
    workspace_root: str | Path | None,
    target_dataset_ids: set[str],
) -> list[dict[str, Any]]:
    if not target_dataset_ids:
        return []
    workspace = qdp_paths(workspace_root).workspace_root.resolve()
    excluded_names = {".git", ".mypy_cache", ".pytest_cache", ".ruff_cache", "__pycache__", "node_modules"}
    readable_suffixes = {".json", ".jsonl", ".toml", ".yaml", ".yml"}
    matches: list[dict[str, Any]] = []
    for current, directory_names, file_names in os.walk(workspace):
        current_path = Path(current)
        relative_parts = current_path.relative_to(workspace).parts
        if relative_parts and relative_parts[0] in {"brain", "quant_data_platform"}:
            directory_names[:] = []
            continue
        directory_names[:] = [name for name in directory_names if name not in excluded_names]
        for file_name in file_names:
            path = current_path / file_name
            if path.suffix.lower() not in readable_suffixes or path.is_symlink():
                continue
            try:
                if path.stat().st_size > 32 * 1024 * 1024:
                    continue
                text = path.read_text(encoding="utf-8")
            except (OSError, UnicodeError):
                continue
            referenced = sorted(dataset_id for dataset_id in target_dataset_ids if dataset_id in text)
            if referenced:
                matches.append(
                    {
                        "path": path.relative_to(workspace).as_posix(),
                        "dataset_ids": referenced,
                    }
                )
                if len(matches) >= 100:
                    return matches
    return matches


def _running_qdp_jobs(paths: Any, target_dataset_ids: set[str]) -> list[dict[str, Any]]:
    running: list[dict[str, Any]] = []
    for path in sorted(paths.jobs.glob("*.json")):
        payload = read_json(path)
        status = str(payload.get("status", "") or "").lower()
        if status not in {"running", "pending", "publishing"}:
            continue
        serialized = str(payload)
        running.append(
            {
                "job_path": str(path.resolve()),
                "status": status,
                "references_retired_dataset": any(dataset_id in serialized for dataset_id in target_dataset_ids),
            }
        )
    return running


def _v2_retirement_inventory(
    *,
    root: Path,
    active: dict[str, Any],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[Path]]:
    active_manifests: list[dict[str, Any]] = []
    domain_inventory: list[dict[str, Any]] = []
    files: dict[Path, None] = {}
    active_datasets = {str(k): str(v) for k, v in dict(active.get("datasets", {}) or {}).items()}
    for domain in V2_INTRADAY_RETIREMENT_DOMAINS:
        dataset_id = active_datasets.get(domain, "")
        manifest_path = v2_dataset_manifest_for_id(root, dataset_id, domain) if dataset_id else None
        active_manifests.append(
            {
                "domain": domain,
                "dataset_id": dataset_id,
                "manifest_path": str(manifest_path.resolve()) if manifest_path is not None else "",
                "manifest_sha256": sha256_file(manifest_path) if manifest_path is not None else "",
                "manifest": read_json(manifest_path) if manifest_path is not None else {},
            }
        )
        domain_root = root / "datasets" / domain
        domain_files = sorted(
            path.resolve()
            for path in domain_root.rglob("*")
            if path.is_file() and path.suffix.lower() in {".parquet", ".pq", ".feather"}
        ) if domain_root.exists() else []
        for path in domain_files:
            try:
                path.relative_to(root.resolve())
            except ValueError as exc:
                raise RuntimeError(f"v2_retirement_path_outside_root:{path}") from exc
            files[path] = None
        domain_inventory.append(
            {
                "domain": domain,
                "active_dataset_id": dataset_id,
                "data_file_count": len(domain_files),
                "data_bytes": sum(path.stat().st_size for path in domain_files),
            }
        )
    return active_manifests, domain_inventory, sorted(files)


def retire_v2_intraday(
    *,
    expect_active_sha: str,
    workspace_root: str | Path | None = None,
    delete: bool = False,
    yes: bool = False,
) -> dict[str, Any]:
    paths = ensure_qdp_v3_layout(workspace_root)
    current_sha = active_manifest_sha256(workspace_root)
    blockers: list[dict[str, Any]] = []
    if str(expect_active_sha or "").strip().lower() != current_sha.lower():
        blockers.append(
            {
                "code": "active_compare_and_swap_failed",
                "expected": str(expect_active_sha or ""),
                "actual": current_sha,
            }
        )
    active = read_json(paths.active_manifest)
    if not active:
        blockers.append({"code": "qdp_v3_active_missing"})
    datasets = {str(k): str(v) for k, v in dict(active.get("datasets", {}) or {}).items()}
    if "market_intraday_1m" in datasets:
        blockers.append({"code": "qdp_v3_active_still_contains_1m"})
    five_id = datasets.get(DOMAIN_MARKET_INTRADAY_5M, "")
    five_path = dataset_manifest_for_id(paths.root, five_id, DOMAIN_MARKET_INTRADAY_5M) if five_id else None
    five_manifest = read_dataset_manifest(five_path) if five_path is not None else None
    if five_manifest is None:
        blockers.append({"code": "qdp_v3_active_5m_manifest_missing"})
    else:
        coverage = dict(five_manifest.coverage or {})
        rate = float(coverage.get("strict_coverage_rate", 0.0) or 0.0)
        five_watermark = str(coverage.get("watermark", "") or "")
        daily_watermark = str(dict(active.get("coverage", {}) or {}).get("market_daily_watermark", "") or "")
        if five_manifest.quality_tier != QUALITY_STRICT or rate < 0.9995:
            blockers.append(
                {
                    "code": "qdp_v3_active_5m_coverage_not_proven",
                    "quality_tier": five_manifest.quality_tier,
                    "strict_coverage_rate": rate,
                }
            )
        if not five_watermark or five_watermark != daily_watermark:
            blockers.append(
                {
                    "code": "qdp_v3_active_5m_watermark_not_equal_daily",
                    "market_intraday_5m_watermark": five_watermark,
                    "market_daily_watermark": daily_watermark,
                }
            )
        for shard in five_manifest.shards:
            shard_path = resolve_manifest_path(shard.path, root=paths.root)
            if not shard_path.exists():
                blockers.append({"code": "qdp_v3_active_5m_shard_missing", "path": str(shard_path)})
                continue
            actual = sha256_file(shard_path)
            if actual != shard.sha256:
                blockers.append(
                    {
                        "code": "qdp_v3_active_5m_shard_hash_mismatch",
                        "path": str(shard_path),
                        "expected": shard.sha256,
                        "actual": actual,
                    }
                )
    candidate_id = str(active.get("candidate_id", "") or "")
    audit = latest_candidate_audit(candidate_id, workspace_root=workspace_root) if candidate_id else {}
    if not (audit.get("status") == "passed" and audit.get("mode") == "semantic"):
        blockers.append({"code": "qdp_v3_active_semantic_audit_missing_or_failed", "candidate_id": candidate_id})
    diff_path = paths.audits / candidate_id / "v2_v3_diff.json" if candidate_id else paths.audits / "missing"
    if not diff_path.exists():
        blockers.append({"code": "v2_v3_diff_report_missing", "path": str(diff_path.resolve())})

    v2_root = qdp_v2_root(workspace_root)
    v2_active_path = v2_active_manifest_path(v2_root)
    v2_active = read_json(v2_active_path)
    if not v2_active:
        blockers.append({"code": "qdp_v2_active_missing"})
    target_ids = {
        str(dataset_id)
        for domain, dataset_id in dict(v2_active.get("datasets", {}) or {}).items()
        if str(domain) in V2_INTRADAY_RETIREMENT_DOMAINS and str(dataset_id)
    }
    running = _running_qdp_jobs(paths, target_ids)
    if running:
        blockers.append({"code": "qdp_jobs_still_running", "jobs": running})
    downstream_consumers = _downstream_consumer_references(
        workspace_root=workspace_root,
        target_dataset_ids=target_ids,
    )
    if downstream_consumers:
        blockers.append(
            {
                "code": "downstream_consumers_still_reference_v2_intraday",
                "references": downstream_consumers,
            }
        )
    active_manifests, domain_inventory, data_files = _v2_retirement_inventory(root=v2_root, active=v2_active)
    missing_active_manifests = [item["domain"] for item in active_manifests if not item["manifest"]]
    if missing_active_manifests:
        blockers.append(
            {
                "code": "qdp_v2_active_intraday_manifest_missing",
                "domains": missing_active_manifests,
            }
        )
    if delete and not yes:
        blockers.append({"code": "physical_delete_requires_yes"})

    status_path = paths.metadata / "v2_intraday_retirement_status.json"
    if blockers:
        payload = {
            "status": "published_cleanup_blocked" if active else "blocked",
            "active_sha256": current_sha,
            "blocker_count": len(blockers),
            "blockers": blockers,
            "checked_at": utc_now(),
        }
        atomic_write_json(status_path, payload)
        return {**payload, "status_path": str(status_path.resolve())}

    file_inventory = [
        {
            "path": path.relative_to(v2_root.resolve()).as_posix(),
            "bytes": path.stat().st_size,
        }
        for path in data_files
    ]
    retirement_payload = {
        "contract": "qdp_v2_intraday_physical_retirement_v1",
        "status": "ready_to_delete" if not delete else "deleting",
        "created_at": utc_now(),
        "qdp_v3_active_sha256": current_sha,
        "qdp_v3_candidate_id": candidate_id,
        "qdp_v3_replacement_dataset_id": five_id,
        "qdp_v3_semantic_audit_path": str(diff_path.parent.resolve()),
        "v2_v3_diff_path": str(diff_path.resolve()),
        "qdp_v2_active_path": str(v2_active_path.resolve()),
        "qdp_v2_active_sha256": sha256_file(v2_active_path),
        "qdp_v2_active": v2_active,
        "active_dataset_manifests": active_manifests,
        "domain_inventory": domain_inventory,
        "file_count": len(file_inventory),
        "total_bytes": sum(int(item["bytes"]) for item in file_inventory),
        "files": file_inventory,
        "replacement_policy": {
            "market_intraday_5m": five_id,
            "market_intraday_1m": "retired_without_replacement",
            "intraday_daily_features": "owned_by_daily_research_if_rebuilt",
            "limit_intraday_features": "owned_by_daily_research_if_rebuilt",
        },
        "downstream_consumer_scan": {
            "status": "passed",
            "reference_count": 0,
            "scanned_machine_readable_suffixes": [".json", ".jsonl", ".toml", ".yaml", ".yml"],
        },
    }
    retirement_id = stable_hash(retirement_payload, length=24)
    retirement_path = paths.metadata / f"retirement_manifest__{retirement_id}.json"
    existing = read_json(retirement_path)
    if existing and existing.get("qdp_v3_active_sha256") != current_sha:
        raise RuntimeError(f"retirement_manifest_contract_conflict:{retirement_path}")
    if not existing:
        atomic_write_json(retirement_path, retirement_payload)
    if not delete:
        payload = {
            "status": "ready_to_delete",
            "active_sha256": current_sha,
            "file_count": len(data_files),
            "total_bytes": retirement_payload["total_bytes"],
            "retirement_manifest": str(retirement_path.resolve()),
        }
        atomic_write_json(status_path, payload)
        return payload

    deleted_bytes = 0
    deleted_count = 0
    for path in data_files:
        if not path.exists():
            continue
        size = path.stat().st_size
        path.unlink()
        deleted_count += 1
        deleted_bytes += size
    remaining = [str(path) for path in data_files if path.exists()]
    active_sha_after_delete = active_manifest_sha256(workspace_root)
    from quant_data_platform.qdp_v3.audit import audit_candidate

    post_delete_audit = audit_candidate(
        candidate_id=candidate_id,
        mode="quick",
        workspace_root=workspace_root,
        write_report=False,
    )
    post_delete_consumers = _downstream_consumer_references(
        workspace_root=workspace_root,
        target_dataset_ids=target_ids,
    )
    if (
        remaining
        or active_sha_after_delete != current_sha
        or post_delete_audit.get("status") != "passed"
        or post_delete_consumers
    ):
        failure = {
            "status": "published_cleanup_blocked",
            "active_sha256": active_sha_after_delete,
            "expected_active_sha256": current_sha,
            "remaining_file_count": len(remaining),
            "remaining_file_sample": remaining[:20],
            "post_delete_qdp_v3_check_status": post_delete_audit.get("status"),
            "post_delete_qdp_v3_check_blocker_count": post_delete_audit.get("blocker_count"),
            "post_delete_consumer_reference_count": len(post_delete_consumers),
            "post_delete_consumer_reference_sample": post_delete_consumers[:20],
            "deleted_file_count": deleted_count,
            "deleted_bytes": deleted_bytes,
            "retirement_manifest": str(retirement_path.resolve()),
        }
        atomic_write_json(status_path, failure)
        return failure
    completion = {
        "contract": "qdp_v2_intraday_physical_retirement_completion_v1",
        "retirement_manifest": str(retirement_path.resolve()),
        "retirement_manifest_sha256": sha256_file(retirement_path),
        "status": "retired",
        "retired_at": utc_now(),
        "deleted_file_count": deleted_count,
        "deleted_bytes": deleted_bytes,
        "post_delete_validation": {
            "active_sha256_unchanged": True,
            "qdp_v3_quick_check_status": post_delete_audit.get("status"),
            "qdp_v3_quick_check_blocker_count": post_delete_audit.get("blocker_count"),
            "downstream_consumer_reference_count": len(post_delete_consumers),
            "remaining_v2_intraday_data_file_count": len(remaining),
        },
    }
    completion_path = paths.metadata / f"retirement_completion__{retirement_id}.json"
    atomic_write_json(completion_path, completion)
    status = {
        "status": "retired",
        "active_sha256": current_sha,
        "deleted_file_count": deleted_count,
        "deleted_bytes": deleted_bytes,
        "retirement_manifest": str(retirement_path.resolve()),
        "retirement_completion": str(completion_path.resolve()),
    }
    atomic_write_json(status_path, status)
    return status
