"""Runtime Archive: seal responsibilities."""

from __future__ import annotations

import hashlib
import json
import math
import random
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import pandas as pd

from quantlab.data.qdp_v2.manifest import atomic_write_json, utc_now
from quantlab.data.qdp_v2.research_event_update import (
    _assert_credential_free,
    _sha256,
)

from .archive import (
    _create_tar_zst,
    restore_archive,
    verify_archive,
)
from .cleanup import (
    _delete_exact_sources,
    _remove_empty_parents,
)
from .config import (
    ARCHIVE_VERSION,
    DEFAULT_CLUSTER_BYTES,
    DEFAULT_SELECTIONS,
    ArchiveUnit,
    RuntimeArchiveError,
)
from .discovery import (
    _archive_root,
    _safe_name,
    _unit_directory,
    _within,
    _workspace,
    discover_units,
)
from .ledger import (
    _atomic_parquet,
    _state_snapshot,
    build_ledger,
)


def seal_unit(
    unit: ArchiveUnit,
    *,
    workspace_root: str | Path | None = None,
) -> dict[str, Any]:
    workspace = _workspace(workspace_root)
    directory = _unit_directory(workspace, unit)
    archive_path = directory / "raw.tar.zst"
    ledger_path = directory / "ledger.parquet"
    manifest_path = directory / "manifest.json"
    ledger = build_ledger(unit)
    if manifest_path.is_file() and ledger_path.is_file() and archive_path.is_file():
        try:
            existing = dict(json.loads(manifest_path.read_text(encoding="utf-8")))
            saved = pd.read_parquet(ledger_path)
            compare_columns = ["relative_path", "file_size", "mtime_ns", "sha256"]
            current_records = ledger.loc[:, compare_columns].to_dict("records")
            saved_records = saved.loc[:, compare_columns].to_dict("records")
            if (
                existing.get("status") == "sealed"
                and existing.get("verified") is True
                and current_records == saved_records
                and _sha256(ledger_path) == str(existing.get("ledger_sha256", ""))
                and _sha256(archive_path) == str(existing.get("archive_sha256", ""))
            ):
                return {
                    **existing,
                    "manifest_path": str(manifest_path),
                    "reused": True,
                }
        except (OSError, ValueError, KeyError):
            pass
        raise RuntimeArchiveError(f"runtime_archive_existing_unit_differs_restore_before_reseal:{unit.key}")
    _atomic_parquet(ledger, ledger_path)
    _create_tar_zst(unit, ledger, archive_path)
    verification = verify_archive(archive_path, ledger)
    state_snapshot = _state_snapshot(workspace, unit.workflow)
    logical_bytes = int(ledger["file_size"].sum())
    manifest = {
        "archive_version": ARCHIVE_VERSION,
        "status": "sealed",
        "workflow": unit.workflow,
        "domain": unit.domain,
        "year": unit.year,
        "unit_key": unit.key,
        "workflow_root": str(unit.workflow_root),
        "archive_path": str(archive_path),
        "archive_sha256": _sha256(archive_path),
        "archive_bytes": archive_path.stat().st_size,
        "ledger_path": str(ledger_path),
        "ledger_sha256": _sha256(ledger_path),
        "ledger_semantics": "file_inventory_with_best_effort_sidecar_request_metadata",
        "state_snapshot": state_snapshot,
        "source_file_count": len(ledger),
        "source_logical_bytes": logical_bytes,
        "estimated_source_allocated_bytes_1mib": int(
            sum(math.ceil(int(size) / DEFAULT_CLUSTER_BYTES) * DEFAULT_CLUSTER_BYTES for size in ledger["file_size"])
        ),
        "verified": bool(verification["verified"]),
        "verified_file_count": int(verification["verified_file_count"]),
        "verified_logical_bytes": int(verification["verified_logical_bytes"]),
        "restore": {
            "format": "tar+zstd",
            "member_base": "workflow_root",
            "overwrite_default": False,
        },
        "created_at": utc_now(),
    }
    _assert_credential_free(manifest)
    atomic_write_json(manifest_path, manifest)
    return {**manifest, "manifest_path": str(manifest_path), "reused": False}


def verify_unit_manifest(manifest_path: str | Path) -> dict[str, Any]:
    path = Path(manifest_path).resolve()
    manifest = dict(json.loads(path.read_text(encoding="utf-8")))
    archive_path = Path(str(manifest["archive_path"])).resolve()
    ledger_path = Path(str(manifest["ledger_path"])).resolve()
    if _sha256(archive_path) != str(manifest.get("archive_sha256", "")) or _sha256(ledger_path) != str(
        manifest.get("ledger_sha256", "")
    ):
        raise RuntimeArchiveError(f"runtime_archive_manifest_hash_mismatch:{path}")
    ledger = pd.read_parquet(ledger_path)
    verification = verify_archive(archive_path, ledger)
    return {
        "status": "ok",
        "manifest_path": str(path),
        "unit_key": manifest.get("unit_key", ""),
        **verification,
    }


def _sample_restore(
    workspace: Path,
    manifests: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    if not manifests:
        raise RuntimeArchiveError("runtime_archive_sample_restore_no_units")
    natural_year = [item for item in manifests if str(item.get("year", "")) != "multi"]
    choices = natural_year or list(manifests)
    seed = hashlib.sha256("|".join(sorted(str(item["archive_sha256"]) for item in choices)).encode("ascii")).digest()
    selected = random.Random(seed).choice(choices)
    check_root = _within(
        _archive_root(workspace)
        / "_restore_checks"
        / f"{_safe_name(str(selected['workflow']))}_{_safe_name(str(selected['domain']))}_{_safe_name(str(selected['year']))}",
        _archive_root(workspace),
    )
    if check_root.exists():
        existing = [path for path in check_root.rglob("*") if path.is_file()]
        for path in existing:
            _within(path, check_root).unlink()
        _remove_empty_parents(existing, stop=check_root)
        try:
            check_root.rmdir()
        except OSError:
            pass
    result = restore_archive(
        str(selected["archive_path"]),
        str(selected["ledger_path"]),
        target_root=check_root,
    )
    restored_files = [path for path in check_root.rglob("*") if path.is_file()]
    for path in restored_files:
        _within(path, check_root).unlink()
    _remove_empty_parents(restored_files, stop=check_root)
    try:
        check_root.rmdir()
    except OSError as exc:
        raise RuntimeArchiveError(f"runtime_archive_restore_check_cleanup:{check_root}") from exc
    return {
        **result,
        "status": "restored_and_hash_verified",
        "sample_unit_key": selected["unit_key"],
        "cleanup": "exact_files_deleted_after_verification",
    }


def seal_workflows(
    *,
    workspace_root: str | Path | None = None,
    workflows: Sequence[str] = (),
    delete_expanded: bool = False,
    yes: bool = False,
) -> dict[str, Any]:
    workspace = _workspace(workspace_root)
    if delete_expanded and not yes:
        raise RuntimeArchiveError("delete_expanded_requires_yes")
    units = discover_units(workspace_root=workspace, workflows=workflows)
    if not units:
        return {
            "status": "nothing_to_seal",
            "workflows": list(workflows),
            "archive_root": str(_archive_root(workspace)),
        }
    manifests: list[dict[str, Any]] = []
    for unit in units:
        manifests.append(seal_unit(unit, workspace_root=workspace))
    sample = _sample_restore(workspace, manifests)
    deletion: list[dict[str, Any]] = []
    if delete_expanded:
        unit_map = {unit.key: unit for unit in units}
        for manifest in manifests:
            unit = unit_map[str(manifest["unit_key"])]
            ledger = pd.read_parquet(str(manifest["ledger_path"]))
            deletion.append(
                {
                    "unit_key": unit.key,
                    **_delete_exact_sources(unit, ledger),
                }
            )
    source_files = sum(int(item["source_file_count"]) for item in manifests)
    source_logical = sum(int(item["source_logical_bytes"]) for item in manifests)
    source_allocated = sum(int(item["estimated_source_allocated_bytes_1mib"]) for item in manifests)
    archive_bytes = sum(int(item["archive_bytes"]) for item in manifests)
    payload = {
        "status": "sealed_and_deleted" if delete_expanded else "sealed",
        "archive_version": ARCHIVE_VERSION,
        "archive_root": str(_archive_root(workspace)),
        "workflows": sorted({unit.workflow for unit in units}),
        "unit_count": len(units),
        "source_file_count": source_files,
        "source_logical_bytes": source_logical,
        "estimated_source_allocated_bytes_1mib": source_allocated,
        "archive_bytes": archive_bytes,
        "estimated_allocated_bytes_reclaimed_1mib": max(
            0,
            source_allocated
            - sum(
                math.ceil(int(item["archive_bytes"]) / DEFAULT_CLUSTER_BYTES) * DEFAULT_CLUSTER_BYTES
                for item in manifests
            ),
        )
        if delete_expanded
        else 0,
        "manifests": [str(item["manifest_path"]) for item in manifests],
        "sample_restore": sample,
        "deletion": deletion,
        "delete_expanded": delete_expanded,
        "completed_at": utc_now(),
    }
    _assert_credential_free(payload)
    receipt = _archive_root(workspace) / "receipts" / f"cleanup_{utc_now().replace(':', '').replace('-', '')}.json"
    atomic_write_json(receipt, payload)
    return {**payload, "cleanup_receipt": str(receipt)}


def seal_completed_workflow(
    workflow: str,
    *,
    workspace_root: str | Path | None = None,
) -> dict[str, Any]:
    """Seal and remove expanded evidence after a workflow reaches its terminal state.

    Verification and a deterministic sample restore complete before exact source
    files are deleted.  An existing unit with different contents is never
    overwritten implicitly.
    """

    if workflow not in DEFAULT_SELECTIONS:
        raise RuntimeArchiveError(f"unknown_archive_workflow:{workflow}")
    return seal_workflows(
        workspace_root=workspace_root,
        workflows=(workflow,),
        delete_expanded=True,
        yes=True,
    )
