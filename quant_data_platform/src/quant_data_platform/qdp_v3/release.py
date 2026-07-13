from __future__ import annotations

import json
import os
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field
from functools import wraps
from pathlib import Path
from typing import Any, Callable, Iterator, Mapping

from quant_data_platform.core.json_io import read_json
from quant_data_platform.qdp_v2.manifest import (
    active_manifest_path as v2_active_manifest_path,
    dataset_manifest_for_id as v2_dataset_manifest_for_id,
    qdp_v2_root,
)
from quant_data_platform.qdp_v3.constants import MANIFEST_VERSION, QUALITY_STRICT, STRICT_RELEASE_DOMAINS
from quant_data_platform.qdp_v3.manifest import (
    active_manifest_sha256,
    atomic_write_json,
    canonical_json_bytes,
    dataset_manifest_for_id,
    manifest_sha256,
    read_dataset_manifest,
    sha256_file,
    stable_hash,
    utc_now,
)
from quant_data_platform.qdp_v3.paths import ensure_qdp_v3_layout, qdp_v3_paths


@contextmanager
def _active_manifest_lock(workspace_root: str | Path | None = None) -> Iterator[None]:
    """Serialize the SHA comparison and active-manifest replacement.

    The OS releases this advisory lock automatically if the process exits, so
    a crashed publisher cannot leave a stale lock-file sentinel behind.
    """

    paths = ensure_qdp_v3_layout(workspace_root)
    lock_path = paths.root / ".active_manifest.lock"
    with lock_path.open("a+b") as handle:
        handle.seek(0, os.SEEK_END)
        if handle.tell() == 0:
            handle.write(b"\0")
            handle.flush()
        handle.seek(0)
        try:
            if os.name == "nt":
                import msvcrt

                msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl

                fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as exc:
            raise RuntimeError(f"qdp_v3_active_manifest_lock_held:{lock_path}") from exc
        try:
            yield
        finally:
            handle.seek(0)
            if os.name == "nt":
                import msvcrt

                msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                import fcntl

                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def _serialized_active_change(operation: Callable[..., dict[str, Any]]) -> Callable[..., dict[str, Any]]:
    @wraps(operation)
    def wrapped(*args: Any, **kwargs: Any) -> dict[str, Any]:
        with _active_manifest_lock(kwargs.get("workspace_root")):
            return operation(*args, **kwargs)

    return wrapped


@dataclass(frozen=True)
class CandidateManifestV3:
    candidate_id: str
    datasets: dict[str, str]
    dataset_manifest_sha256: dict[str, str]
    quality_tiers: dict[str, str]
    blockers: list[dict[str, Any]]
    coverage: dict[str, Any]
    build: dict[str, Any]
    raw_partitions: list[dict[str, Any]] = field(default_factory=list)
    quarantine: list[dict[str, Any]] = field(default_factory=list)
    created_at: str = field(default_factory=utc_now)
    manifest_version: int = MANIFEST_VERSION
    status: str = "candidate"

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any]) -> "CandidateManifestV3":
        return cls(
            candidate_id=str(payload.get("candidate_id", "") or ""),
            datasets={str(k): str(v) for k, v in dict(payload.get("datasets", {}) or {}).items()},
            dataset_manifest_sha256={str(k): str(v) for k, v in dict(payload.get("dataset_manifest_sha256", {}) or {}).items()},
            quality_tiers={str(k): str(v) for k, v in dict(payload.get("quality_tiers", {}) or {}).items()},
            blockers=[dict(item) for item in list(payload.get("blockers", []) or []) if isinstance(item, Mapping)],
            coverage=dict(payload.get("coverage", {}) or {}),
            build=dict(payload.get("build", {}) or {}),
            raw_partitions=[dict(item) for item in list(payload.get("raw_partitions", []) or []) if isinstance(item, Mapping)],
            quarantine=[dict(item) for item in list(payload.get("quarantine", []) or []) if isinstance(item, Mapping)],
            created_at=str(payload.get("created_at", "") or utc_now()),
            manifest_version=int(payload.get("manifest_version", 0) or 0),
            status=str(payload.get("status", "") or "candidate"),
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def candidate_path(candidate_id: str, *, workspace_root: str | Path | None = None) -> Path:
    return qdp_v3_paths(workspace_root).candidates / f"{candidate_id}.json"


def candidate_id_for(
    *,
    datasets: Mapping[str, str],
    dataset_manifest_sha256: Mapping[str, str],
    build: Mapping[str, Any],
) -> str:
    seed = {
        "datasets": dict(datasets),
        "dataset_manifest_sha256": dict(dataset_manifest_sha256),
        "build": dict(build),
    }
    return f"candidate__{stable_hash(seed, length=24)}"


def write_candidate(
    *,
    datasets: Mapping[str, str],
    dataset_manifest_sha256: Mapping[str, str],
    quality_tiers: Mapping[str, str],
    blockers: list[dict[str, Any]],
    coverage: Mapping[str, Any],
    build: Mapping[str, Any],
    raw_partitions: list[dict[str, Any]],
    quarantine: list[dict[str, Any]],
    workspace_root: str | Path | None = None,
) -> CandidateManifestV3:
    ensure_qdp_v3_layout(workspace_root)
    candidate_id = candidate_id_for(datasets=datasets, dataset_manifest_sha256=dataset_manifest_sha256, build=build)
    path = candidate_path(candidate_id, workspace_root=workspace_root)
    if path.exists():
        return read_candidate(candidate_id, workspace_root=workspace_root)
    candidate = CandidateManifestV3(
        candidate_id=candidate_id,
        datasets={str(k): str(v) for k, v in dict(datasets).items()},
        dataset_manifest_sha256={str(k): str(v) for k, v in dict(dataset_manifest_sha256).items()},
        quality_tiers={str(k): str(v) for k, v in dict(quality_tiers).items()},
        blockers=list(blockers),
        coverage=dict(coverage),
        build=dict(build),
        raw_partitions=list(raw_partitions),
        quarantine=list(quarantine),
    )
    atomic_write_json(path, candidate.to_dict())
    return candidate


def read_candidate(candidate_id_or_path: str | Path, *, workspace_root: str | Path | None = None) -> CandidateManifestV3:
    candidate = Path(candidate_id_or_path)
    path = candidate if candidate.exists() else candidate_path(str(candidate_id_or_path), workspace_root=workspace_root)
    payload = read_json(path)
    if not payload:
        raise FileNotFoundError(f"candidate_missing:{candidate_id_or_path}")
    result = CandidateManifestV3.from_mapping(payload)
    if result.manifest_version != MANIFEST_VERSION:
        raise RuntimeError(f"candidate_manifest_version_mismatch:{result.manifest_version}")
    return result


def validate_candidate_graph(candidate: CandidateManifestV3, *, workspace_root: str | Path | None = None) -> list[dict[str, Any]]:
    root = qdp_v3_paths(workspace_root).root
    findings: list[dict[str, Any]] = []
    visited: dict[str, str] = {}

    def visit(*, domain: str, dataset_id: str, expected_sha: str, parent_dataset_id: str = "") -> None:
        prior_sha = visited.get(dataset_id)
        if prior_sha is not None:
            if prior_sha != expected_sha:
                findings.append(
                    {
                        "code": "dataset_input_hash_contract_conflict",
                        "domain": domain,
                        "dataset_id": dataset_id,
                        "first_expected": prior_sha,
                        "conflicting_expected": expected_sha,
                        "parent_dataset_id": parent_dataset_id,
                    }
                )
            return
        visited[dataset_id] = expected_sha
        path = dataset_manifest_for_id(root, dataset_id, domain)
        if path is None:
            findings.append(
                {
                    "code": "candidate_dataset_manifest_missing" if not parent_dataset_id else "dataset_input_missing",
                    "domain": domain,
                    "dataset_id": dataset_id,
                    "parent_dataset_id": parent_dataset_id,
                }
            )
            return
        actual_sha = manifest_sha256(path)
        if actual_sha != expected_sha:
            findings.append(
                {
                    "code": "candidate_dataset_manifest_hash_mismatch" if not parent_dataset_id else "dataset_input_hash_mismatch",
                    "domain": domain,
                    "dataset_id": dataset_id,
                    "expected": expected_sha,
                    "actual": actual_sha,
                    "parent_dataset_id": parent_dataset_id,
                }
            )
        manifest = read_dataset_manifest(path)
        if manifest.dataset_id != dataset_id or manifest.domain != domain:
            findings.append(
                {
                    "code": "dataset_manifest_identity_mismatch",
                    "domain": domain,
                    "dataset_id": dataset_id,
                    "manifest_domain": manifest.domain,
                    "manifest_dataset_id": manifest.dataset_id,
                    "parent_dataset_id": parent_dataset_id,
                }
            )
        for input_ref in manifest.inputs:
            visit(
                domain=input_ref.domain,
                dataset_id=input_ref.dataset_id,
                expected_sha=input_ref.manifest_sha256,
                parent_dataset_id=dataset_id,
            )

    for domain, dataset_id in candidate.datasets.items():
        visit(
            domain=domain,
            dataset_id=dataset_id,
            expected_sha=str(candidate.dataset_manifest_sha256.get(domain, "") or ""),
        )
    return findings


def latest_candidate_audit(candidate_id: str, *, workspace_root: str | Path | None = None) -> dict[str, Any]:
    audit_dir = qdp_v3_paths(workspace_root).audits / str(candidate_id)
    paths = sorted(audit_dir.glob("*.json"), key=lambda item: item.stat().st_mtime_ns) if audit_dir.exists() else []
    return read_json(paths[-1]) if paths else {}


@_serialized_active_change
def publish_candidate(
    *,
    candidate_id: str,
    expect_active_sha: str,
    workspace_root: str | Path | None = None,
    require_semantic_audit: bool = True,
) -> dict[str, Any]:
    paths = ensure_qdp_v3_layout(workspace_root)
    candidate = read_candidate(candidate_id, workspace_root=workspace_root)
    current_sha = active_manifest_sha256(workspace_root)
    expected = str(expect_active_sha or "none").strip().lower()
    if expected != current_sha.lower():
        raise RuntimeError(f"active_compare_and_swap_failed:expected={expected} actual={current_sha}")
    graph_findings = validate_candidate_graph(candidate, workspace_root=workspace_root)
    blockers = list(candidate.blockers) + graph_findings
    freeze = read_json(paths.metadata / "v2_freeze_20260713.json")
    if freeze.get("status") != "complete" or not bool(freeze.get("hash_shards", False)):
        blockers.append({"code": "qdp_v2_full_freeze_proof_missing"})
    for domain in STRICT_RELEASE_DOMAINS:
        if domain not in candidate.datasets:
            blockers.append({"code": "strict_release_domain_missing", "domain": domain})
        elif candidate.quality_tiers.get(domain) != QUALITY_STRICT:
            blockers.append(
                {
                    "code": "strict_release_domain_not_strict",
                    "domain": domain,
                    "quality_tier": candidate.quality_tiers.get(domain, ""),
                }
            )
    audit = latest_candidate_audit(candidate.candidate_id, workspace_root=workspace_root)
    if require_semantic_audit and not (
        audit.get("status") == "passed" and audit.get("mode") == "semantic" and audit.get("candidate_id") == candidate.candidate_id
    ):
        blockers.append({"code": "semantic_candidate_audit_missing_or_failed", "candidate_id": candidate.candidate_id})
    if blockers:
        return {
            "status": "blocked",
            "candidate_id": candidate.candidate_id,
            "active_sha256": current_sha,
            "blocker_count": len(blockers),
            "blockers": blockers,
        }
    previous = read_json(paths.active_manifest)
    if previous:
        rollback_path = paths.rollbacks / f"{utc_now().replace(':', '').replace('-', '')}__{current_sha}.json"
        atomic_write_json(
            rollback_path,
            {
                "manifest_version": MANIFEST_VERSION,
                "captured_at": utc_now(),
                "active_sha256": current_sha,
                "active": previous,
                "replaced_by_candidate": candidate.candidate_id,
            },
        )
    candidate_file = candidate_path(candidate.candidate_id, workspace_root=workspace_root)
    active_payload = {
        "manifest_version": MANIFEST_VERSION,
        "candidate_id": candidate.candidate_id,
        "candidate_manifest_sha256": manifest_sha256(candidate_file),
        "datasets": candidate.datasets,
        "dataset_manifest_sha256": candidate.dataset_manifest_sha256,
        "quality_tiers": candidate.quality_tiers,
        "coverage": candidate.coverage,
        "previous_active_sha256": current_sha,
        "published_at": utc_now(),
    }
    atomic_write_json(paths.active_manifest, active_payload)
    new_sha = active_manifest_sha256(workspace_root)
    return {
        "status": "published",
        "candidate_id": candidate.candidate_id,
        "previous_active_sha256": current_sha,
        "active_sha256": new_sha,
        "active_manifest": str(paths.active_manifest.resolve()),
    }


@_serialized_active_change
def rollback_active(
    *,
    expect_active_sha: str,
    workspace_root: str | Path | None = None,
) -> dict[str, Any]:
    paths = ensure_qdp_v3_layout(workspace_root)
    current_sha = active_manifest_sha256(workspace_root)
    if str(expect_active_sha).strip().lower() != current_sha.lower():
        raise RuntimeError(f"active_compare_and_swap_failed:expected={expect_active_sha} actual={current_sha}")
    records = sorted(paths.rollbacks.glob("*.json"), key=lambda item: item.stat().st_mtime_ns, reverse=True)
    if not records:
        raise RuntimeError("rollback_manifest_missing")
    record_path = records[0]
    record = read_json(record_path)
    prior = dict(record.get("active", {}) or {})
    if not prior:
        raise RuntimeError(f"rollback_manifest_has_no_active_payload:{record_path}")
    current = read_json(paths.active_manifest)
    archive = paths.rollbacks / f"rollback_from_{utc_now().replace(':', '').replace('-', '')}__{current_sha}.json"
    atomic_write_json(
        archive,
        {
            "manifest_version": MANIFEST_VERSION,
            "captured_at": utc_now(),
            "active_sha256": current_sha,
            "active": current,
            "rollback_source": str(record_path.name),
        },
    )
    atomic_write_json(paths.active_manifest, prior)
    return {
        "status": "rolled_back",
        "restored_from": str(record_path.resolve()),
        "previous_active_sha256": current_sha,
        "active_sha256": active_manifest_sha256(workspace_root),
    }


def _manifest_diff_summary(path: Path | None) -> dict[str, Any]:
    if path is None or not path.exists():
        return {}
    payload = read_json(path)
    coverage = dict(payload.get("coverage", {}) or {})
    schema = [str(item.get("name", "")) for item in list(payload.get("schema", []) or []) if isinstance(item, Mapping)]
    return {
        "row_count": int(payload.get("row_count", 0) or 0),
        "start_date": str(payload.get("start_date", "") or coverage.get("start_date", "") or ""),
        "end_date": str(payload.get("end_date", "") or coverage.get("end_date", "") or ""),
        "primary_key": [str(item) for item in list(payload.get("primary_key", []) or [])],
        "schema_columns": schema,
        "quality_tier": str(payload.get("quality_tier", "") or ""),
        "manifest_path": str(path.resolve()),
    }


def _active_diff_context(workspace_root: str | Path | None) -> tuple[str, dict[str, Any], str, Path | None]:
    v3_path = qdp_v3_paths(workspace_root).active_manifest
    if v3_path.exists():
        return "v3", read_json(v3_path), sha256_file(v3_path), v3_path
    v2_root = qdp_v2_root(workspace_root)
    v2_path = v2_active_manifest_path(v2_root)
    if v2_path.exists():
        return "v2", read_json(v2_path), sha256_file(v2_path), v2_path
    return "none", {}, "none", None


def diff_candidate(
    candidate_id: str,
    *,
    workspace_root: str | Path | None = None,
    against: str = "active",
) -> dict[str, Any]:
    if str(against).strip().lower() != "active":
        raise ValueError(f"unsupported_candidate_diff_target:{against}")
    candidate = read_candidate(candidate_id, workspace_root=workspace_root)
    active_generation, active, comparison_sha, active_path = _active_diff_context(workspace_root)
    active_datasets = {str(k): str(v) for k, v in dict(active.get("datasets", {}) or {}).items()}
    domains = sorted(set(active_datasets) | set(candidate.datasets))
    rows: list[dict[str, Any]] = []
    v3_root = qdp_v3_paths(workspace_root).root
    legacy_root = qdp_v2_root(workspace_root)
    for domain in domains:
        before = active_datasets.get(domain, "")
        after = candidate.datasets.get(domain, "")
        active_manifest = (
            dataset_manifest_for_id(v3_root, before, domain)
            if before and active_generation == "v3"
            else v2_dataset_manifest_for_id(legacy_root, before, domain)
            if before and active_generation == "v2"
            else None
        )
        candidate_manifest = dataset_manifest_for_id(v3_root, after, domain) if after else None
        before_summary = _manifest_diff_summary(active_manifest)
        after_summary = _manifest_diff_summary(candidate_manifest)
        before_schema = set(before_summary.get("schema_columns", []) or [])
        after_schema = set(after_summary.get("schema_columns", []) or [])
        row_count_delta = (
            int(after_summary["row_count"]) - int(before_summary["row_count"])
            if before_summary and after_summary
            else None
        )
        rows.append(
            {
                "domain": domain,
                "active_dataset_id": before,
                "candidate_dataset_id": after,
                "change": "unchanged" if before == after else "added" if not before else "removed" if not after else "replaced",
                "active": before_summary,
                "candidate": after_summary,
                "row_count_delta": row_count_delta,
                "primary_key_changed": bool(before_summary and after_summary and before_summary.get("primary_key") != after_summary.get("primary_key")),
                "schema_added": sorted(after_schema - before_schema),
                "schema_removed": sorted(before_schema - after_schema),
            }
        )
    return {
        "status": "ok",
        "candidate_id": candidate.candidate_id,
        "against": f"{active_generation}_active" if active_generation != "none" else "no_active",
        "active_generation": active_generation,
        "active_path": str(active_path.resolve()) if active_path else "",
        "active_sha256": comparison_sha,
        "changes": rows,
        "changed_count": sum(item["change"] != "unchanged" for item in rows),
    }
