from __future__ import annotations

import hashlib
import json
import os
import time
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from quant_data_platform.core.json_io import json_safe, read_json
from quant_data_platform.core.paths import qdp_paths
from quant_data_platform.qdp_v2.duckdb_resources import (
    GuardedDuckDbConnection,
    open_guarded_duckdb,
)
from quant_data_platform.qdp_v2.manifest import (
    DatasetManifest,
    ShardManifestEntry,
    active_manifest_path,
    atomic_write_json,
    dataset_manifest_path,
    qdp_v2_root,
    read_dataset_manifest,
    resolve_manifest_path,
    schema_hash,
)
from quant_data_platform.qdp_v2.status import active_dataset_map


_DATE_COLUMNS = ("trade_date", "date", "query_date", "report_date", "ann_date")


class QdpV2RepairError(RuntimeError):
    """Raised when an in-place active dataset repair cannot be proven safe."""


class QdpV2RepairConflictError(QdpV2RepairError):
    """Raised when active/manifest bytes change during a repair transaction."""


@dataclass(frozen=True)
class ActiveDomain:
    root: Path
    active_path: Path
    active_sha256: str
    domain: str
    dataset_id: str
    manifest_path: Path
    manifest_sha256: str
    manifest: DatasetManifest

    @property
    def dataset_dir(self) -> Path:
        return self.manifest_path.parent

    @property
    def shard_paths(self) -> tuple[Path, ...]:
        return tuple(
            resolve_manifest_path(item.path, root=self.root).resolve()
            for item in self.manifest.shards
        )


@dataclass(frozen=True)
class _PreparedParquetInfo:
    source_path: Path
    row_count: int
    start_date: str
    end_date: str


@dataclass(frozen=True)
class _InstalledParquet:
    source_path: Path
    target_path: Path
    row_count: int
    start_date: str
    end_date: str
    file_sha256: str
    created: bool


def resolve_active_domain(
    domain: str,
    *,
    workspace_root: str | Path | None = None,
) -> ActiveDomain:
    """Resolve one domain from the current v2 active pointer without scanning data."""

    requested = str(domain or "").strip()
    if not requested:
        raise ValueError("qdp_v2_repair_domain_required")
    root = qdp_v2_root(workspace_root).resolve()
    active_path = active_manifest_path(root).resolve()
    active_sha, active = _stable_json_read(active_path, "active")
    dataset_id = active_dataset_map(active).get(requested, "")
    if not dataset_id:
        raise QdpV2RepairError(f"qdp_v2_repair_active_domain_missing:{requested}")
    manifest_path = dataset_manifest_path(root, requested, dataset_id).resolve()
    manifest_sha, _ = _stable_json_read(manifest_path, "dataset_manifest")
    manifest = read_dataset_manifest(manifest_path)
    if manifest.domain != requested or manifest.dataset_id != dataset_id:
        raise QdpV2RepairError(
            "qdp_v2_repair_manifest_identity_mismatch:"
            f"expected={requested}:{dataset_id}:"
            f"actual={manifest.domain}:{manifest.dataset_id}"
        )
    return ActiveDomain(
        root=root,
        active_path=active_path,
        active_sha256=active_sha,
        domain=requested,
        dataset_id=dataset_id,
        manifest_path=manifest_path,
        manifest_sha256=manifest_sha,
        manifest=manifest,
    )


def replace_active_table_from_parquet(
    domain: str,
    prepared_parquet_path: str | Path,
    *,
    reason: str,
    workspace_root: str | Path | None = None,
    primary_key: Sequence[str] | None = None,
    contract_version: str = "",
    source_updates: Mapping[str, Any] | None = None,
    quality_updates: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Replace every shard in one active table, including schema migrations.

    Auxiliary tables are intentionally small single-table domains whose
    contract can evolve in place.  The dataset manifest remains the commit
    point and the active dataset id is never changed.  Old shards are removed
    only after the new manifest is durable.
    """

    repair_reason = _required_reason(reason)
    context = resolve_active_domain(domain, workspace_root=workspace_root)
    prepared = _resolve_prepared_parquet_path(
        prepared_parquet_path,
        workspace_root=workspace_root,
    )
    parquet = pq.ParquetFile(prepared)
    arrow_schema = parquet.schema_arrow
    row_count = int(parquet.metadata.num_rows)
    if row_count <= 0:
        raise QdpV2RepairError(f"qdp_v2_repair_frame_empty:{prepared}")

    keys = [str(item) for item in (primary_key or context.manifest.primary_key)]
    if not keys:
        raise QdpV2RepairError("qdp_v2_repair_primary_key_missing")
    missing_keys = sorted(set(keys).difference(arrow_schema.names))
    if missing_keys:
        raise QdpV2RepairError(
            f"qdp_v2_repair_primary_key_columns_missing:{missing_keys}"
        )

    date_column = next(
        (item for item in _DATE_COLUMNS if item in arrow_schema.names),
        "",
    )
    start_date = ""
    end_date = ""
    with _open_repair_duckdb(context) as connection:
        quoted_keys = ",".join(_quote_identifier(item) for item in keys)
        null_predicate = " OR ".join(
            f"{_quote_identifier(item)} IS NULL" for item in keys
        )
        duplicate = connection.execute(
            "SELECT 1 FROM read_parquet(?) "
            f"WHERE NOT ({null_predicate}) GROUP BY {quoted_keys} "
            "HAVING count(*) > 1 LIMIT 1",
            [str(prepared)],
        ).fetchone()
        null_key = connection.execute(
            "SELECT 1 FROM read_parquet(?) "
            f"WHERE {null_predicate} LIMIT 1",
            [str(prepared)],
        ).fetchone()
        if null_key:
            raise QdpV2RepairError("qdp_v2_repair_primary_key_null")
        if duplicate:
            raise QdpV2RepairError("qdp_v2_repair_primary_key_duplicate")
        if date_column:
            quoted_date = _quote_identifier(date_column)
            row = connection.execute(
                "SELECT count(*) FILTER (WHERE try_cast("
                f"{quoted_date} AS DATE) IS NULL), "
                f"strftime(min(try_cast({quoted_date} AS DATE)), '%Y-%m-%d'), "
                f"strftime(max(try_cast({quoted_date} AS DATE)), '%Y-%m-%d') "
                "FROM read_parquet(?)",
                [str(prepared)],
            ).fetchone()
            if row is None or int(row[0] or 0) != 0 or not row[1] or not row[2]:
                raise QdpV2RepairError(
                    f"qdp_v2_repair_date_value_invalid:{prepared}"
                )
            start_date, end_date = str(row[1]), str(row[2])

    info = _PreparedParquetInfo(
        source_path=prepared,
        row_count=row_count,
        start_date=start_date,
        end_date=end_date,
    )
    installed = _install_prepared_parquet(
        info,
        target_dir=_active_shard_directory(context),
        target_name=lambda digest: f"auxiliary_replace_{digest[:24]}.parquet",
    )
    created = installed.created
    try:
        new_schema = _manifest_schema_from_arrow(arrow_schema)
        new_schema_hash = schema_hash(new_schema)
        entry = ShardManifestEntry(
            path=_path_for_manifest(installed.target_path, root=context.root),
            row_count=row_count,
            start_date=start_date,
            end_date=end_date,
            status="stored",
            file_size=installed.target_path.stat().st_size,
            schema_hash=new_schema_hash,
            source_path=str(prepared),
            content_key=f"auxiliary-replace:{installed.file_sha256[:24]}",
            metadata={
                "repair_reason": repair_reason,
                "repair_file_sha256": installed.file_sha256,
                "replaces": [item.path for item in context.manifest.shards],
            },
        )
        payload = context.manifest.to_dict()
        payload.update(
            {
                "dataset_id": context.dataset_id,
                "domain": context.domain,
                "primary_key": keys,
                "start_date": start_date,
                "end_date": end_date,
                "row_count": row_count,
                "schema": new_schema,
                "schema_hash": new_schema_hash,
                "shards": [entry.to_dict()],
            }
        )
        if contract_version:
            payload["contract_version"] = str(contract_version)
        source = dict(payload.get("source", {}) or {})
        source.update(dict(source_updates or {}))
        source.update(
            {
                "last_repair_action": "replace_active_table_from_parquet",
                "last_repair_reason": repair_reason,
                "last_repair_at": _utc_now(),
            }
        )
        payload["source"] = source
        quality = dict(payload.get("quality", {}) or {})
        quality.update(dict(quality_updates or {}))
        payload["quality"] = quality
        notes = [str(item) for item in list(payload.get("notes", []) or [])]
        note = f"replace_active_table_from_parquet:{repair_reason}"
        if note not in notes:
            notes.append(note)
        payload["notes"] = notes

        _assert_context_unchanged(context)
        _commit_manifest(context, payload)
        old_paths = [
            resolve_manifest_path(item.path, root=context.root).resolve()
            for item in context.manifest.shards
        ]
        deleted: list[str] = []
        for old_path in old_paths:
            if old_path == installed.target_path:
                continue
            old_path.unlink(missing_ok=True)
            deleted.append(str(old_path))
        _append_repair_log(
            workspace_root,
            {
                "action": "replace_active_table_from_parquet",
                "domain": context.domain,
                "dataset_id": context.dataset_id,
                "reason": repair_reason,
                "manifest_sha256_before": context.manifest_sha256,
                "manifest_sha256_after": _sha256_file(context.manifest_path),
                "new_shard_path": str(installed.target_path),
                "new_shard_sha256": installed.file_sha256,
                "deleted_shard_paths": deleted,
                "row_count": row_count,
            },
        )
        return {
            "status": "replaced",
            "domain": context.domain,
            "dataset_id": context.dataset_id,
            "row_count": row_count,
            "start_date": start_date,
            "end_date": end_date,
            "new_shard_path": str(installed.target_path),
            "deleted_shard_paths": deleted,
        }
    except BaseException:
        if created and _sha256_file(context.manifest_path) == context.manifest_sha256:
            installed.target_path.unlink(missing_ok=True)
        raise


def update_active_manifest_metadata(
    domain: str,
    *,
    reason: str,
    workspace_root: str | Path | None = None,
    source_updates: Mapping[str, Any] | None = None,
    quality_updates: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Atomically update active-domain metadata without touching its shards."""

    repair_reason = _required_reason(reason)
    context = resolve_active_domain(domain, workspace_root=workspace_root)
    payload = context.manifest.to_dict()
    source = dict(payload.get("source", {}) or {})
    source.update(dict(source_updates or {}))
    source.update(
        {
            "last_metadata_update_reason": repair_reason,
            "last_metadata_update_at": _utc_now(),
        }
    )
    payload["source"] = source
    quality = dict(payload.get("quality", {}) or {})
    quality.update(dict(quality_updates or {}))
    payload["quality"] = quality
    _assert_context_unchanged(context)
    _commit_manifest(context, payload)
    _append_repair_log(
        workspace_root,
        {
            "action": "update_active_manifest_metadata",
            "domain": context.domain,
            "dataset_id": context.dataset_id,
            "reason": repair_reason,
            "manifest_sha256_before": context.manifest_sha256,
            "manifest_sha256_after": _sha256_file(context.manifest_path),
            "row_count": context.manifest.row_count,
        },
    )
    return {
        "status": "metadata_updated",
        "domain": context.domain,
        "dataset_id": context.dataset_id,
        "row_count": context.manifest.row_count,
    }


def replace_active_shards(
    domain: str,
    replacement_frames: Sequence[pd.DataFrame] | Mapping[str | Path, pd.DataFrame] | pd.DataFrame,
    replaced_shard_paths: Sequence[str | Path],
    reason: str,
    *,
    workspace_root: str | Path | None = None,
) -> dict[str, Any]:
    """Replace selected active shards without changing dataset_id or active.json."""

    repair_reason = _required_reason(reason)
    context = resolve_active_domain(domain, workspace_root=workspace_root)
    old_entries = _resolve_selected_entries(context, replaced_shard_paths)
    frames = _pair_replacement_frames(
        replacement_frames,
        replaced_shard_paths=replaced_shard_paths,
        selected_entries=old_entries,
        context=context,
    )
    reference_schema = _reference_schema(context)
    prepared: list[tuple[ShardManifestEntry, Path, bool]] = []
    created_paths: list[Path] = []
    committed = False
    try:
        for old_entry, frame in zip(old_entries, frames, strict=True):
            old_path = resolve_manifest_path(old_entry.path, root=context.root).resolve()
            _assert_deletable_dataset_shard(context, old_path)
            validated = _validate_frame(frame, context=context)
            content_hash = frame_content_sha256(validated)
            old_token = hashlib.sha256(str(old_path).encode("utf-8")).hexdigest()[:8]
            target = old_path.with_name(
                f"repair_replace_{old_token}_{content_hash[:16]}.parquet"
            )
            created = _write_and_validate_parquet(
                target,
                validated,
                reference_schema=reference_schema,
                expected_content_hash=content_hash,
            )
            if created:
                created_paths.append(target)
            start_date, end_date = _frame_date_range(validated, context.manifest)
            prepared.append(
                (
                    _new_shard_entry(
                        context,
                        target,
                        validated,
                        start_date=start_date,
                        end_date=end_date,
                        content_hash=content_hash,
                        reason=repair_reason,
                        old_entry=old_entry,
                    ),
                    target,
                    created,
                )
            )
        old_keys = {_manifest_path_key(item.path, context.root) for item in old_entries}
        retained = [
            item
            for item in context.manifest.shards
            if _manifest_path_key(item.path, context.root) not in old_keys
        ]
        _validate_cross_shard_primary_keys(
            context,
            new_paths=[item[1] for item in prepared],
            retained_entries=retained,
        )
        replacement_by_key = {
            _manifest_path_key(old.path, context.root): new
            for old, (new, _, _) in zip(old_entries, prepared, strict=True)
        }
        updated_entries: list[ShardManifestEntry] = []
        for entry in context.manifest.shards:
            key = _manifest_path_key(entry.path, context.root)
            updated_entries.append(replacement_by_key.get(key, entry))
        _assert_context_unchanged(context)
        updated_payload = _updated_manifest_payload(
            context,
            updated_entries,
            reason=repair_reason,
            action="replace_active_shards",
        )
        _commit_manifest(context, updated_payload)
        committed = True
        deleted_paths: list[str] = []
        for old_entry in old_entries:
            old_path = resolve_manifest_path(old_entry.path, root=context.root).resolve()
            _assert_deletable_dataset_shard(context, old_path)
            old_path.unlink()
            deleted_paths.append(str(old_path))
        after_sha = _sha256_file(context.manifest_path)
        _append_repair_log(
            workspace_root,
            {
                "action": "replace_active_shards",
                "domain": context.domain,
                "dataset_id": context.dataset_id,
                "reason": repair_reason,
                "active_sha256": context.active_sha256,
                "manifest_sha256_before": context.manifest_sha256,
                "manifest_sha256_after": after_sha,
                "new_shard_paths": [str(item[1]) for item in prepared],
                "deleted_shard_paths": deleted_paths,
                "row_count": sum(item[0].row_count for item in prepared),
            },
        )
        return {
            "status": "replaced",
            "domain": context.domain,
            "dataset_id": context.dataset_id,
            "manifest_path": str(context.manifest_path),
            "manifest_sha256_before": context.manifest_sha256,
            "manifest_sha256_after": after_sha,
            "active_sha256": context.active_sha256,
            "new_shard_paths": [str(item[1]) for item in prepared],
            "deleted_shard_paths": deleted_paths,
            "replacement_count": len(prepared),
        }
    except BaseException:
        if not committed:
            for path in created_paths:
                path.unlink(missing_ok=True)
        raise


def append_active_shard(
    domain: str,
    frame: pd.DataFrame,
    reason: str,
    *,
    workspace_root: str | Path | None = None,
) -> dict[str, Any]:
    """Append one validated shard to the current active dataset, idempotently."""

    repair_reason = _required_reason(reason)
    context = resolve_active_domain(domain, workspace_root=workspace_root)
    validated = _validate_frame(frame, context=context)
    reference_schema = _reference_schema(context)
    content_hash = frame_content_sha256(validated)
    shard_dir = _active_shard_directory(context)
    target = shard_dir / f"repair_append_{content_hash[:24]}.parquet"
    target_key = _manifest_path_key(target, context.root)
    for entry in context.manifest.shards:
        if _manifest_path_key(entry.path, context.root) == target_key:
            _validate_existing_parquet(
                target,
                expected_frame=validated,
                reference_schema=reference_schema,
                expected_content_hash=content_hash,
            )
            return {
                "status": "reused",
                "domain": context.domain,
                "dataset_id": context.dataset_id,
                "manifest_path": str(context.manifest_path),
                "manifest_sha256": context.manifest_sha256,
                "active_sha256": context.active_sha256,
                "shard_path": str(target),
            }
    created = _write_and_validate_parquet(
        target,
        validated,
        reference_schema=reference_schema,
        expected_content_hash=content_hash,
    )
    committed = False
    try:
        _validate_cross_shard_primary_keys(
            context,
            new_paths=[target],
            retained_entries=list(context.manifest.shards),
        )
        start_date, end_date = _frame_date_range(validated, context.manifest)
        new_entry = _new_shard_entry(
            context,
            target,
            validated,
            start_date=start_date,
            end_date=end_date,
            content_hash=content_hash,
            reason=repair_reason,
            old_entry=None,
        )
        updated_entries = [*context.manifest.shards, new_entry]
        _assert_context_unchanged(context)
        updated_payload = _updated_manifest_payload(
            context,
            updated_entries,
            reason=repair_reason,
            action="append_active_shard",
        )
        _commit_manifest(context, updated_payload)
        committed = True
        after_sha = _sha256_file(context.manifest_path)
        _append_repair_log(
            workspace_root,
            {
                "action": "append_active_shard",
                "domain": context.domain,
                "dataset_id": context.dataset_id,
                "reason": repair_reason,
                "active_sha256": context.active_sha256,
                "manifest_sha256_before": context.manifest_sha256,
                "manifest_sha256_after": after_sha,
                "new_shard_paths": [str(target)],
                "deleted_shard_paths": [],
                "row_count": len(validated),
            },
        )
        return {
            "status": "appended",
            "domain": context.domain,
            "dataset_id": context.dataset_id,
            "manifest_path": str(context.manifest_path),
            "manifest_sha256_before": context.manifest_sha256,
            "manifest_sha256_after": after_sha,
            "active_sha256": context.active_sha256,
            "shard_path": str(target),
            "row_count": len(validated),
        }
    except BaseException:
        if created and not committed:
            target.unlink(missing_ok=True)
        raise


def replace_active_shard_from_parquet(
    domain: str,
    prepared_parquet_path: str | Path,
    replaced_shard_path: str | Path,
    reason: str,
    *,
    workspace_root: str | Path | None = None,
) -> dict[str, Any]:
    """Replace one active shard from an already prepared Parquet file."""

    return replace_active_shards_from_parquet(
        domain,
        [(replaced_shard_path, prepared_parquet_path)],
        reason,
        workspace_root=workspace_root,
    )


def replace_active_shards_from_parquet(
    domain: str,
    replacements: (
        Mapping[str | Path, str | Path]
        | Sequence[tuple[str | Path, str | Path]]
    ),
    reason: str,
    *,
    workspace_root: str | Path | None = None,
) -> dict[str, Any]:
    """Replace active shards from Parquet paths without loading them into pandas."""

    repair_reason = _required_reason(reason)
    context = resolve_active_domain(domain, workspace_root=workspace_root)
    pairs = _replacement_path_pairs(replacements)
    replaced_paths = [item[0] for item in pairs]
    old_entries = _resolve_selected_entries(context, replaced_paths)
    source_paths = [
        _resolve_prepared_parquet_path(item[1], workspace_root=workspace_root)
        for item in pairs
    ]
    reference_schema = _reference_schema(context)
    source_info = _inspect_prepared_parquets(
        context,
        source_paths,
        reference_schema=reference_schema,
    )
    installed: list[_InstalledParquet] = []
    created_paths: list[Path] = []
    committed = False
    try:
        for old_entry, info in zip(old_entries, source_info, strict=True):
            old_path = resolve_manifest_path(old_entry.path, root=context.root).resolve()
            _assert_deletable_dataset_shard(context, old_path)
            old_token = hashlib.sha256(
                str(old_path).encode("utf-8")
            ).hexdigest()[:8]
            item = _install_prepared_parquet(
                info,
                target_dir=old_path.parent,
                target_name=lambda digest, token=old_token: (
                    f"repair_replace_{token}_{digest[:16]}.parquet"
                ),
            )
            installed.append(item)
            if item.created:
                created_paths.append(item.target_path)
        _assert_unique_installed_targets(installed)
        old_keys = {_manifest_path_key(item.path, context.root) for item in old_entries}
        retained = [
            item
            for item in context.manifest.shards
            if _manifest_path_key(item.path, context.root) not in old_keys
        ]
        _validate_parquet_batch_primary_keys(
            context,
            new_paths=[item.target_path for item in installed],
            retained_entries=retained,
            new_start=min(item.start_date for item in installed),
            new_end=max(item.end_date for item in installed),
        )
        replacement_by_key = {
            _manifest_path_key(old.path, context.root): _new_parquet_shard_entry(
                context,
                item,
                reason=repair_reason,
                old_entry=old,
            )
            for old, item in zip(old_entries, installed, strict=True)
        }
        updated_entries = [
            replacement_by_key.get(
                _manifest_path_key(entry.path, context.root), entry
            )
            for entry in context.manifest.shards
        ]
        _assert_context_unchanged(context)
        updated_payload = _updated_manifest_payload(
            context,
            updated_entries,
            reason=repair_reason,
            action="replace_active_shards_from_parquet",
        )
        _commit_manifest(context, updated_payload)
        committed = True
        deleted_paths: list[str] = []
        for old_entry in old_entries:
            old_path = resolve_manifest_path(old_entry.path, root=context.root).resolve()
            _assert_deletable_dataset_shard(context, old_path)
            old_path.unlink()
            deleted_paths.append(str(old_path))
        after_sha = _sha256_file(context.manifest_path)
        _append_repair_log(
            workspace_root,
            {
                "action": "replace_active_shards_from_parquet",
                "domain": context.domain,
                "dataset_id": context.dataset_id,
                "reason": repair_reason,
                "active_sha256": context.active_sha256,
                "manifest_sha256_before": context.manifest_sha256,
                "manifest_sha256_after": after_sha,
                "new_shard_paths": [str(item.target_path) for item in installed],
                "new_shard_sha256": [item.file_sha256 for item in installed],
                "deleted_shard_paths": deleted_paths,
                "row_count": sum(item.row_count for item in installed),
            },
        )
        return {
            "status": "replaced",
            "domain": context.domain,
            "dataset_id": context.dataset_id,
            "manifest_path": str(context.manifest_path),
            "manifest_sha256_before": context.manifest_sha256,
            "manifest_sha256_after": after_sha,
            "active_sha256": context.active_sha256,
            "new_shard_paths": [str(item.target_path) for item in installed],
            "deleted_shard_paths": deleted_paths,
            "replacement_count": len(installed),
            "row_count": sum(item.row_count for item in installed),
        }
    except BaseException:
        if not committed:
            for path in created_paths:
                path.unlink(missing_ok=True)
        raise


def bulk_append_active_shards_from_parquet(
    domain: str,
    prepared_parquet_paths: Sequence[str | Path],
    reason: str,
    *,
    workspace_root: str | Path | None = None,
) -> dict[str, Any]:
    """Append prepared Parquet shards with one batch primary-key validation."""

    repair_reason = _required_reason(reason)
    if not prepared_parquet_paths:
        raise ValueError("qdp_v2_repair_prepared_parquet_paths_required")
    context = resolve_active_domain(domain, workspace_root=workspace_root)
    source_paths = [
        _resolve_prepared_parquet_path(item, workspace_root=workspace_root)
        for item in prepared_parquet_paths
    ]
    _assert_unique_prepared_paths(source_paths)
    reference_schema = _reference_schema(context)
    source_info = _inspect_prepared_parquets(
        context,
        source_paths,
        reference_schema=reference_schema,
    )
    shard_dir = _active_shard_directory(context)
    installed: list[_InstalledParquet] = []
    created_paths: list[Path] = []
    committed = False
    try:
        for info in source_info:
            item = _install_prepared_parquet(
                info,
                target_dir=shard_dir,
                target_name=lambda digest: f"repair_append_{digest[:24]}.parquet",
            )
            installed.append(item)
            if item.created:
                created_paths.append(item.target_path)
        _assert_unique_installed_targets(installed)
        active_by_key = {
            _manifest_path_key(entry.path, context.root): entry
            for entry in context.manifest.shards
        }
        pending: list[_InstalledParquet] = []
        reused: list[_InstalledParquet] = []
        for item in installed:
            key = _manifest_path_key(item.target_path, context.root)
            existing = active_by_key.get(key)
            if existing is None:
                pending.append(item)
                continue
            if int(existing.row_count) != item.row_count:
                raise QdpV2RepairError(
                    f"qdp_v2_repair_active_shard_row_count_mismatch:{item.target_path}"
                )
            reused.append(item)
        if not pending:
            return {
                "status": "reused",
                "domain": context.domain,
                "dataset_id": context.dataset_id,
                "manifest_path": str(context.manifest_path),
                "manifest_sha256": context.manifest_sha256,
                "active_sha256": context.active_sha256,
                "shard_paths": [str(item.target_path) for item in reused],
                "appended_count": 0,
                "reused_count": len(reused),
                "row_count": 0,
            }
        _validate_parquet_batch_primary_keys(
            context,
            new_paths=[item.target_path for item in pending],
            retained_entries=list(context.manifest.shards),
            new_start=min(item.start_date for item in pending),
            new_end=max(item.end_date for item in pending),
        )
        new_entries = [
            _new_parquet_shard_entry(
                context,
                item,
                reason=repair_reason,
                old_entry=None,
            )
            for item in pending
        ]
        updated_entries = [*context.manifest.shards, *new_entries]
        _assert_context_unchanged(context)
        updated_payload = _updated_manifest_payload(
            context,
            updated_entries,
            reason=repair_reason,
            action="bulk_append_active_shards_from_parquet",
        )
        _commit_manifest(context, updated_payload)
        committed = True
        after_sha = _sha256_file(context.manifest_path)
        _append_repair_log(
            workspace_root,
            {
                "action": "bulk_append_active_shards_from_parquet",
                "domain": context.domain,
                "dataset_id": context.dataset_id,
                "reason": repair_reason,
                "active_sha256": context.active_sha256,
                "manifest_sha256_before": context.manifest_sha256,
                "manifest_sha256_after": after_sha,
                "new_shard_paths": [str(item.target_path) for item in pending],
                "new_shard_sha256": [item.file_sha256 for item in pending],
                "deleted_shard_paths": [],
                "row_count": sum(item.row_count for item in pending),
            },
        )
        return {
            "status": "appended",
            "domain": context.domain,
            "dataset_id": context.dataset_id,
            "manifest_path": str(context.manifest_path),
            "manifest_sha256_before": context.manifest_sha256,
            "manifest_sha256_after": after_sha,
            "active_sha256": context.active_sha256,
            "shard_paths": [str(item.target_path) for item in installed],
            "appended_count": len(pending),
            "reused_count": len(reused),
            "row_count": sum(item.row_count for item in pending),
        }
    except BaseException:
        if not committed:
            for path in created_paths:
                path.unlink(missing_ok=True)
        raise


def mutate_active_shards_from_parquet(
    domain: str,
    *,
    replacements: (
        Mapping[str | Path, str | Path]
        | Sequence[tuple[str | Path, str | Path]]
    ) = (),
    removals: Sequence[str | Path] = (),
    appends: Sequence[str | Path] = (),
    reason: str,
    workspace_root: str | Path | None = None,
    primary_keys_prevalidated: bool = False,
    expected_mutation_id: str = "",
) -> dict[str, Any]:
    """Atomically replace, remove, and append active Parquet shards.

    The dataset manifest is the sole commit point.  Prepared files are copied
    to content-addressed names and fully validated before the manifest CAS;
    superseded files are not deleted until after that commit.  A deterministic
    mutation id is retained in the dataset manifest so replaying the exact
    request after a crash is a no-op (apart from finishing safe orphan cleanup).
    ``active.json`` and the active ``dataset_id`` are never rewritten.

    ``primary_keys_prevalidated`` requires a bound ``expected_mutation_id`` and
    at least one replacement or append.  It lets a domain-specific validator
    reuse its stronger, partition-aware primary-key proof instead of
    materializing the same large key set a second time.
    """

    repair_reason = _required_reason(reason)
    replacement_pairs = _optional_replacement_path_pairs(replacements)
    removal_paths = list(removals)
    append_paths = list(appends)
    if not replacement_pairs and not removal_paths and not append_paths:
        raise ValueError("qdp_v2_repair_shard_mutation_empty")
    expected_mutation = str(expected_mutation_id or "").strip()
    if primary_keys_prevalidated and (
        not expected_mutation or not (replacement_pairs or append_paths)
    ):
        raise ValueError(
            "qdp_v2_repair_prevalidated_keys_require_bound_new_shards"
        )

    context = resolve_active_domain(domain, workspace_root=workspace_root)
    replacement_old_paths = [
        _resolve_mutation_shard_path(context, item[0])
        for item in replacement_pairs
    ]
    resolved_removal_paths = [
        _resolve_mutation_shard_path(context, item) for item in removal_paths
    ]
    _assert_unique_mutation_old_paths(
        context,
        replacement_old_paths=replacement_old_paths,
        removal_paths=resolved_removal_paths,
    )

    replacement_source_paths = [
        _resolve_prepared_parquet_path(item[1], workspace_root=workspace_root)
        for item in replacement_pairs
    ]
    append_source_paths = [
        _resolve_prepared_parquet_path(item, workspace_root=workspace_root)
        for item in append_paths
    ]
    all_source_paths = [*replacement_source_paths, *append_source_paths]
    if all_source_paths:
        _assert_unique_prepared_paths(all_source_paths)

    reference_schema = _reference_schema(context)
    source_info = _inspect_prepared_parquets(
        context,
        all_source_paths,
        reference_schema=reference_schema,
    ) if all_source_paths else []
    replacement_info = source_info[: len(replacement_pairs)]
    append_info = source_info[len(replacement_pairs) :]

    replacement_installed: list[_InstalledParquet] = []
    append_installed: list[_InstalledParquet] = []
    created_paths: list[Path] = []
    committed = False
    try:
        for old_path, info in zip(
            replacement_old_paths, replacement_info, strict=True
        ):
            old_token = hashlib.sha256(
                _manifest_path_key(old_path, context.root).encode("utf-8")
            ).hexdigest()[:8]
            item = _install_prepared_parquet(
                info,
                target_dir=old_path.parent,
                target_name=lambda digest, token=old_token: (
                    f"repair_mutate_replace_{token}_{digest[:16]}.parquet"
                ),
            )
            replacement_installed.append(item)
            if item.created:
                created_paths.append(item.target_path)

        append_dir = _active_shard_directory(context) if append_info else None
        for info in append_info:
            assert append_dir is not None
            item = _install_prepared_parquet(
                info,
                target_dir=append_dir,
                target_name=lambda digest: (
                    f"repair_mutate_append_{digest[:24]}.parquet"
                ),
            )
            append_installed.append(item)
            if item.created:
                created_paths.append(item.target_path)

        all_installed = [*replacement_installed, *append_installed]
        _assert_unique_installed_targets(all_installed)
        mutation_id = _shard_mutation_id(
            context,
            replacement_old_paths=replacement_old_paths,
            replacement_items=replacement_installed,
            removal_paths=resolved_removal_paths,
            append_items=append_installed,
        )
        if expected_mutation and mutation_id != expected_mutation:
            raise QdpV2RepairError(
                "qdp_v2_repair_expected_mutation_id_mismatch"
            )

        if _manifest_has_shard_mutation(context.manifest, mutation_id):
            # The commit occurred in an earlier attempt.  Preserve/recreate any
            # referenced content-addressed targets and only finish deletion of
            # now-unreferenced old files.
            committed = True
            _validate_replayed_shard_mutation(
                context,
                replacement_old_paths=replacement_old_paths,
                removal_paths=resolved_removal_paths,
                installed=all_installed,
                reference_schema=reference_schema,
            )
            old_paths = [*replacement_old_paths, *resolved_removal_paths]
            cleanup = _cleanup_committed_old_shards(
                context,
                old_paths=old_paths,
                expected_sha256=_manifest_shard_mutation_old_hashes(
                    context.manifest, mutation_id
                ),
            )
            return {
                "status": "reused",
                "domain": context.domain,
                "dataset_id": context.dataset_id,
                "mutation_id": mutation_id,
                "manifest_path": str(context.manifest_path),
                "manifest_sha256": context.manifest_sha256,
                "active_sha256": context.active_sha256,
                "new_shard_paths": [
                    str(item.target_path) for item in all_installed
                ],
                "deleted_shard_paths": cleanup["deleted"],
                "retained_old_shard_paths": cleanup["retained"],
                "replacement_count": len(replacement_installed),
                "removal_count": len(resolved_removal_paths),
                "appended_count": 0,
                "reused_append_count": len(append_installed),
            }

        selected_entries = _resolve_selected_entries(
            context,
            [*replacement_old_paths, *resolved_removal_paths],
        ) if replacement_old_paths or resolved_removal_paths else []
        replacement_entries = selected_entries[: len(replacement_old_paths)]
        removal_entries = selected_entries[len(replacement_old_paths) :]
        selected_hashes = _validate_selected_old_shards(
            context,
            selected_entries,
            reference_schema=reference_schema,
        )

        selected_keys = {
            _manifest_path_key(item.path, context.root)
            for item in selected_entries
        }
        retained_entries = [
            item
            for item in context.manifest.shards
            if _manifest_path_key(item.path, context.root) not in selected_keys
        ]
        retained_by_key = {
            _manifest_path_key(item.path, context.root): item
            for item in retained_entries
        }
        pending_appends: list[_InstalledParquet] = []
        reused_appends: list[_InstalledParquet] = []
        for item in append_installed:
            existing = retained_by_key.get(
                _manifest_path_key(item.target_path, context.root)
            )
            if existing is None:
                pending_appends.append(item)
                continue
            _validate_installed_against_entry(
                item,
                existing,
                reference_schema=reference_schema,
            )
            reused_appends.append(item)

        new_items = [*replacement_installed, *pending_appends]
        if new_items:
            if not primary_keys_prevalidated:
                _validate_parquet_batch_primary_keys(
                    context,
                    new_paths=[item.target_path for item in new_items],
                    retained_entries=retained_entries,
                    new_start=min(item.start_date for item in new_items),
                    new_end=max(item.end_date for item in new_items),
                )
        elif not selected_entries:
            # Pure append replay where the content-addressed shard was already
            # active before this API was called: there is no manifest change.
            return {
                "status": "reused",
                "domain": context.domain,
                "dataset_id": context.dataset_id,
                "mutation_id": mutation_id,
                "manifest_path": str(context.manifest_path),
                "manifest_sha256": context.manifest_sha256,
                "active_sha256": context.active_sha256,
                "new_shard_paths": [
                    str(item.target_path) for item in append_installed
                ],
                "deleted_shard_paths": [],
                "retained_old_shard_paths": [],
                "replacement_count": 0,
                "removal_count": 0,
                "appended_count": 0,
                "reused_append_count": len(reused_appends),
            }

        replacement_by_key = {
            _manifest_path_key(old.path, context.root): _new_parquet_shard_entry(
                context,
                item,
                reason=repair_reason,
                old_entry=old,
            )
            for old, item in zip(
                replacement_entries, replacement_installed, strict=True
            )
        }
        removal_keys = {
            _manifest_path_key(item.path, context.root)
            for item in removal_entries
        }
        updated_entries: list[ShardManifestEntry] = []
        for entry in context.manifest.shards:
            key = _manifest_path_key(entry.path, context.root)
            if key in replacement_by_key:
                updated_entries.append(replacement_by_key[key])
            elif key not in removal_keys:
                updated_entries.append(entry)
        updated_entries.extend(
            _new_parquet_shard_entry(
                context,
                item,
                reason=repair_reason,
                old_entry=None,
            )
            for item in pending_appends
        )

        _validate_updated_shard_paths(
            context,
            updated_entries,
            installed=all_installed,
            reference_schema=reference_schema,
        )
        _assert_context_unchanged(context)
        updated_payload = _updated_manifest_payload(
            context,
            updated_entries,
            reason=repair_reason,
            action="mutate_active_shards_from_parquet",
        )
        _record_shard_mutation(
            updated_payload,
            mutation_id,
            old_sha256=selected_hashes,
        )
        _commit_manifest(context, updated_payload)
        committed = True

        old_paths = [
            resolve_manifest_path(item.path, root=context.root).resolve()
            for item in selected_entries
        ]
        cleanup = _cleanup_committed_old_shards(
            context,
            old_paths=old_paths,
            expected_sha256=selected_hashes,
        )
        after_sha = _sha256_file(context.manifest_path)
        log_error = ""
        try:
            _append_repair_log(
                workspace_root,
                {
                    "action": "mutate_active_shards_from_parquet",
                    "domain": context.domain,
                    "dataset_id": context.dataset_id,
                    "mutation_id": mutation_id,
                    "reason": repair_reason,
                    "active_sha256": context.active_sha256,
                    "manifest_sha256_before": context.manifest_sha256,
                    "manifest_sha256_after": after_sha,
                    "new_shard_paths": [
                        str(item.target_path) for item in new_items
                    ],
                    "new_shard_sha256": [
                        item.file_sha256 for item in new_items
                    ],
                    "deleted_shard_paths": cleanup["deleted"],
                    "retained_old_shard_paths": cleanup["retained"],
                    "row_count": sum(item.row_count for item in new_items),
                },
            )
        except BaseException as exc:  # the manifest transaction is committed
            log_error = f"{type(exc).__name__}:{exc}"
        return {
            "status": "mutated",
            "domain": context.domain,
            "dataset_id": context.dataset_id,
            "mutation_id": mutation_id,
            "manifest_path": str(context.manifest_path),
            "manifest_sha256_before": context.manifest_sha256,
            "manifest_sha256_after": after_sha,
            "active_sha256": context.active_sha256,
            "new_shard_paths": [str(item.target_path) for item in new_items],
            "deleted_shard_paths": cleanup["deleted"],
            "retained_old_shard_paths": cleanup["retained"],
            "replacement_count": len(replacement_installed),
            "removal_count": len(removal_entries),
            "appended_count": len(pending_appends),
            "reused_append_count": len(reused_appends),
            "manifest_row_count": sum(item.row_count for item in updated_entries),
            "repair_log_error": log_error,
        }
    except BaseException:
        if not committed:
            for path in created_paths:
                path.unlink(missing_ok=True)
        raise


def frame_content_sha256(frame: pd.DataFrame) -> str:
    data = frame.reset_index(drop=True)
    digest = hashlib.sha256()
    digest.update(
        json.dumps(
            {
                "columns": [str(item) for item in data.columns],
                "dtypes": [str(item) for item in data.dtypes],
                "rows": len(data),
            },
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    )
    if not data.empty:
        normalized = data.copy()
        for column in normalized.columns:
            values = normalized[column]
            if pd.api.types.is_datetime64_any_dtype(values):
                values = pd.to_datetime(values, errors="coerce").dt.strftime(
                    "%Y-%m-%dT%H:%M:%S.%f"
                )
            normalized[column] = values.astype("string").fillna("<QDP_NULL>")
        hashed = pd.util.hash_pandas_object(
            normalized, index=False, categorize=False
        ).to_numpy(dtype="uint64", copy=False)
        digest.update(hashed.tobytes())
    return digest.hexdigest()


def repair_log_path(workspace_root: str | Path | None = None) -> Path:
    return (
        qdp_paths(workspace_root).data_dir
        / "qdp_runtime"
        / "qdp_v2_repair_log.jsonl"
    ).resolve()


def _replacement_path_pairs(
    replacements: (
        Mapping[str | Path, str | Path]
        | Sequence[tuple[str | Path, str | Path]]
    ),
) -> list[tuple[str | Path, str | Path]]:
    pairs = (
        list(replacements.items())
        if isinstance(replacements, Mapping)
        else list(replacements)
    )
    if not pairs:
        raise ValueError("qdp_v2_repair_parquet_replacements_required")
    normalized: list[tuple[str | Path, str | Path]] = []
    for item in pairs:
        if not isinstance(item, (tuple, list)) or len(item) != 2:
            raise ValueError("qdp_v2_repair_parquet_replacement_pair_invalid")
        normalized.append((item[0], item[1]))
    return normalized


def _optional_replacement_path_pairs(
    replacements: (
        Mapping[str | Path, str | Path]
        | Sequence[tuple[str | Path, str | Path]]
    ),
) -> list[tuple[str | Path, str | Path]]:
    pairs = (
        list(replacements.items())
        if isinstance(replacements, Mapping)
        else list(replacements)
    )
    if not pairs:
        return []
    return _replacement_path_pairs(pairs)


def _resolve_mutation_shard_path(
    context: ActiveDomain,
    path: str | Path,
) -> Path:
    candidate = Path(path)
    resolved = (
        candidate.resolve()
        if candidate.is_absolute()
        else (context.root / candidate).resolve()
    )
    _assert_deletable_dataset_shard(context, resolved)
    return resolved


def _assert_unique_mutation_old_paths(
    context: ActiveDomain,
    *,
    replacement_old_paths: Sequence[Path],
    removal_paths: Sequence[Path],
) -> None:
    replacement_keys = [
        _manifest_path_key(item, context.root) for item in replacement_old_paths
    ]
    removal_keys = [
        _manifest_path_key(item, context.root) for item in removal_paths
    ]
    if len(replacement_keys) != len(set(replacement_keys)):
        raise ValueError("qdp_v2_repair_duplicate_replacement_path")
    if len(removal_keys) != len(set(removal_keys)):
        raise ValueError("qdp_v2_repair_duplicate_removal_path")
    overlap = set(replacement_keys).intersection(removal_keys)
    if overlap:
        raise ValueError("qdp_v2_repair_replacement_removal_overlap")


def _shard_mutation_id(
    context: ActiveDomain,
    *,
    replacement_old_paths: Sequence[Path],
    replacement_items: Sequence[_InstalledParquet],
    removal_paths: Sequence[Path],
    append_items: Sequence[_InstalledParquet],
) -> str:
    replacements = sorted(
        (
            _manifest_path_key(old_path, context.root),
            item.file_sha256,
        )
        for old_path, item in zip(
            replacement_old_paths, replacement_items, strict=True
        )
    )
    payload = {
        "version": 1,
        "domain": context.domain,
        "dataset_id": context.dataset_id,
        "replacements": replacements,
        "removals": sorted(
            _manifest_path_key(item, context.root) for item in removal_paths
        ),
        "appends": sorted(item.file_sha256 for item in append_items),
    }
    encoded = json.dumps(
        payload,
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return f"shard-mutation-v1:{hashlib.sha256(encoded).hexdigest()}"


def _manifest_has_shard_mutation(
    manifest: DatasetManifest,
    mutation_id: str,
) -> bool:
    records = dict(manifest.source.get("repair_shard_mutations", {}) or {})
    return mutation_id in records


def _record_shard_mutation(
    payload: dict[str, Any],
    mutation_id: str,
    *,
    old_sha256: Mapping[str, str],
) -> None:
    source = dict(payload.get("source", {}) or {})
    records = dict(source.get("repair_shard_mutations", {}) or {})
    records.setdefault(
        mutation_id,
        {
            "old_shard_sha256": {
                str(key): str(value)
                for key, value in sorted(old_sha256.items())
            }
        },
    )
    source["repair_shard_mutations"] = records
    source["last_repair_mutation_id"] = mutation_id
    payload["source"] = source


def _manifest_shard_mutation_old_hashes(
    manifest: DatasetManifest,
    mutation_id: str,
) -> dict[str, str]:
    records = dict(manifest.source.get("repair_shard_mutations", {}) or {})
    record = records.get(mutation_id, {})
    if not isinstance(record, Mapping):
        return {}
    values = record.get("old_shard_sha256", {})
    if not isinstance(values, Mapping):
        return {}
    return {str(key): str(value) for key, value in values.items()}


def _resolve_prepared_parquet_path(
    path: str | Path,
    *,
    workspace_root: str | Path | None,
) -> Path:
    candidate = Path(path).expanduser()
    if not candidate.is_absolute():
        candidate = qdp_paths(workspace_root).workspace_root / candidate
    resolved = candidate.resolve()
    if not resolved.is_file() or resolved.suffix.lower() != ".parquet":
        raise QdpV2RepairError(
            f"qdp_v2_repair_prepared_parquet_missing:{resolved}"
        )
    return resolved


def _assert_unique_prepared_paths(paths: Sequence[Path]) -> None:
    keys = [_manifest_path_key(path, Path.cwd()) for path in paths]
    if len(keys) != len(set(keys)):
        raise ValueError("qdp_v2_repair_duplicate_prepared_parquet_path")


def _inspect_prepared_parquets(
    context: ActiveDomain,
    paths: Sequence[Path],
    *,
    reference_schema: pa.Schema,
) -> list[_PreparedParquetInfo]:
    if not paths:
        raise ValueError("qdp_v2_repair_prepared_parquet_paths_required")
    _assert_unique_prepared_paths(paths)
    date_column = _date_column(context.manifest)
    expected_columns = [str(item.get("name", "")) for item in context.manifest.schema]
    expected_columns = [item for item in expected_columns if item]
    metadata_rows: list[tuple[Path, int]] = []
    for path in paths:
        parquet = pq.ParquetFile(path)
        row_count = int(parquet.metadata.num_rows)
        if row_count <= 0:
            raise QdpV2RepairError(f"qdp_v2_repair_frame_empty:{path}")
        schema = parquet.schema_arrow
        if not schema.equals(reference_schema, check_metadata=False):
            raise QdpV2RepairError(
                f"qdp_v2_repair_parquet_schema_mismatch:{path}"
            )
        if expected_columns and schema.names != expected_columns:
            raise QdpV2RepairError(
                "qdp_v2_repair_schema_columns_mismatch:"
                f"expected={expected_columns}:actual={schema.names}"
            )
        missing_keys = [
            item for item in context.manifest.primary_key if item not in schema.names
        ]
        if missing_keys:
            raise QdpV2RepairError(
                f"qdp_v2_repair_primary_key_columns_missing:{missing_keys}"
            )
        metadata_rows.append((path, row_count))

    if not date_column:
        return [
            _PreparedParquetInfo(
                source_path=path,
                row_count=row_count,
                start_date="",
                end_date="",
            )
            for path, row_count in metadata_rows
        ]

    results: list[_PreparedParquetInfo] = []
    quoted_date = _quote_identifier(date_column)
    parsed_date = f"try_cast({quoted_date} AS DATE)"
    with _open_repair_duckdb(context) as connection:
        for path, expected_rows in metadata_rows:
            row = connection.execute(
                "SELECT count(*) AS row_count, "
                f"count(*) FILTER (WHERE {quoted_date} IS NULL "
                f"OR {parsed_date} IS NULL) AS invalid_dates, "
                f"strftime(min({parsed_date}), '%Y-%m-%d') AS start_date, "
                f"strftime(max({parsed_date}), '%Y-%m-%d') AS end_date "
                f"FROM read_parquet({_sql_literal(str(path))})"
            ).fetchone()
            if row is None or int(row[0]) != expected_rows:
                raise QdpV2RepairError(
                    f"qdp_v2_repair_row_count_mismatch:{path}"
                )
            if int(row[1] or 0) != 0 or not row[2] or not row[3]:
                raise QdpV2RepairError(
                    f"qdp_v2_repair_date_value_invalid:{path}"
                )
            results.append(
                _PreparedParquetInfo(
                    source_path=path,
                    row_count=expected_rows,
                    start_date=str(row[2]),
                    end_date=str(row[3]),
                )
            )
    return results


def _open_repair_duckdb(context: ActiveDomain) -> GuardedDuckDbConnection:
    temp_dir = (context.root / "tmp").resolve()
    temp_dir.mkdir(parents=True, exist_ok=True)
    return open_guarded_duckdb(
        temp_directory=temp_dir,
        threads=max(1, min(os.cpu_count() or 1, 8)),
    )


def _install_prepared_parquet(
    info: _PreparedParquetInfo,
    *,
    target_dir: Path,
    target_name: Callable[[str], str],
) -> _InstalledParquet:
    target_dir.mkdir(parents=True, exist_ok=True)
    temporary = target_dir / (
        f".repair_prepared.{os.getpid()}.{uuid.uuid4().hex}.parquet.tmp"
    )
    digest = hashlib.sha256()
    try:
        with info.source_path.open("rb") as source, temporary.open("xb") as output:
            while True:
                chunk = source.read(8 * 1024 * 1024)
                if not chunk:
                    break
                digest.update(chunk)
                output.write(chunk)
            output.flush()
            os.fsync(output.fileno())
        file_sha256 = digest.hexdigest()
        target = target_dir / target_name(file_sha256)
        if target.suffix.lower() != ".parquet" or target.parent.resolve() != target_dir.resolve():
            raise QdpV2RepairError(
                f"qdp_v2_repair_invalid_prepared_target:{target}"
            )
        if target.exists():
            if _sha256_file(target) != file_sha256:
                raise QdpV2RepairError(
                    f"qdp_v2_repair_existing_target_hash_mismatch:{target}"
                )
            created = False
        else:
            temporary.replace(target)
            created = True
        parquet = pq.ParquetFile(target)
        if int(parquet.metadata.num_rows) != info.row_count:
            if created:
                target.unlink(missing_ok=True)
            raise QdpV2RepairError(
                f"qdp_v2_repair_row_count_mismatch:{target}"
            )
        return _InstalledParquet(
            source_path=info.source_path,
            target_path=target.resolve(),
            row_count=info.row_count,
            start_date=info.start_date,
            end_date=info.end_date,
            file_sha256=file_sha256,
            created=created,
        )
    finally:
        temporary.unlink(missing_ok=True)


def _assert_unique_installed_targets(items: Sequence[_InstalledParquet]) -> None:
    keys = [
        _manifest_path_key(item.target_path, Path.cwd())
        for item in items
    ]
    if len(keys) != len(set(keys)):
        raise QdpV2RepairError(
            "qdp_v2_repair_duplicate_prepared_parquet_content"
        )


def _validate_parquet_batch_primary_keys(
    context: ActiveDomain,
    *,
    new_paths: Sequence[Path],
    retained_entries: Sequence[ShardManifestEntry],
    new_start: str,
    new_end: str,
) -> None:
    if not new_paths:
        raise QdpV2RepairError("qdp_v2_repair_parquet_path_list_empty")
    primary_key = list(context.manifest.primary_key)
    if not primary_key:
        raise QdpV2RepairError("qdp_v2_repair_primary_key_missing")
    quoted_keys = ",".join(_quote_identifier(item) for item in primary_key)
    null_predicate = " OR ".join(
        f"{_quote_identifier(item)} IS NULL" for item in primary_key
    )
    retained = [
        resolve_manifest_path(item.path, root=context.root).resolve()
        for item in _entries_overlapping_range(
            context.manifest,
            retained_entries,
            new_start=new_start,
            new_end=new_end,
        )
    ]
    with _open_repair_duckdb(context) as connection:
        connection.execute(
            "CREATE TEMP TABLE qdp_repair_new_keys AS "
            f"SELECT {quoted_keys} FROM "
            f"read_parquet({_parquet_list_sql(new_paths)}, union_by_name=false)"
        )
        null_key = connection.execute(
            f"SELECT 1 FROM qdp_repair_new_keys WHERE {null_predicate} LIMIT 1"
        ).fetchone()
        if null_key:
            raise QdpV2RepairError("qdp_v2_repair_primary_key_null")
        duplicate = connection.execute(
            "SELECT 1 FROM qdp_repair_new_keys "
            f"GROUP BY {quoted_keys} HAVING count(*) > 1 LIMIT 1"
        ).fetchone()
        if duplicate:
            raise QdpV2RepairError(
                "qdp_v2_repair_new_shards_primary_key_overlap"
            )
        if retained:
            join = " AND ".join(
                f"n.{_quote_identifier(item)} = e.{_quote_identifier(item)}"
                for item in primary_key
            )
            overlap = connection.execute(
                "SELECT 1 FROM qdp_repair_new_keys n "
                f"JOIN read_parquet({_parquet_list_sql(retained)}, "
                f"union_by_name=false) e ON {join} LIMIT 1"
            ).fetchone()
            if overlap:
                raise QdpV2RepairError(
                    "qdp_v2_repair_retained_shard_primary_key_overlap"
                )


def _entries_overlapping_range(
    manifest: DatasetManifest,
    entries: Sequence[ShardManifestEntry],
    *,
    new_start: str,
    new_end: str,
) -> list[ShardManifestEntry]:
    date_column = _date_column(manifest)
    if not date_column or date_column not in manifest.primary_key:
        return list(entries)
    return [
        item
        for item in entries
        if not item.start_date
        or not item.end_date
        or (str(item.start_date) <= new_end and str(item.end_date) >= new_start)
    ]


def _new_parquet_shard_entry(
    context: ActiveDomain,
    item: _InstalledParquet,
    *,
    reason: str,
    old_entry: ShardManifestEntry | None,
) -> ShardManifestEntry:
    return ShardManifestEntry(
        path=_path_for_manifest(item.target_path, root=context.root),
        row_count=item.row_count,
        start_date=item.start_date,
        end_date=item.end_date,
        status="stored",
        file_size=item.target_path.stat().st_size,
        schema_hash=(
            old_entry.schema_hash if old_entry else context.manifest.schema_hash
        ),
        source_path=str(item.source_path),
        content_key=f"repair-file:{item.file_sha256[:24]}",
        metadata={
            "repair_reason": reason,
            "repair_file_sha256": item.file_sha256,
            "prepared_source_path": str(item.source_path),
            "replaces": str(old_entry.path) if old_entry else "",
        },
    )


def _validate_selected_old_shards(
    context: ActiveDomain,
    entries: Sequence[ShardManifestEntry],
    *,
    reference_schema: pa.Schema,
) -> dict[str, str]:
    hashes: dict[str, str] = {}
    for entry in entries:
        path = resolve_manifest_path(entry.path, root=context.root).resolve()
        _assert_deletable_dataset_shard(context, path)
        if not path.is_file():
            raise QdpV2RepairError(
                f"qdp_v2_repair_active_shard_missing:{path}"
            )
        size = path.stat().st_size
        if int(entry.file_size or 0) > 0 and size != int(entry.file_size):
            raise QdpV2RepairError(
                f"qdp_v2_repair_active_shard_file_size_mismatch:{path}"
            )
        parquet = pq.ParquetFile(path)
        if int(parquet.metadata.num_rows) != int(entry.row_count):
            raise QdpV2RepairError(
                f"qdp_v2_repair_active_shard_row_count_mismatch:{path}"
            )
        if not parquet.schema_arrow.equals(reference_schema, check_metadata=False):
            raise QdpV2RepairError(
                f"qdp_v2_repair_active_shard_schema_mismatch:{path}"
            )
        digest = _sha256_file(path)
        expected = str(
            entry.metadata.get("repair_file_sha256", "")
            or entry.metadata.get("file_sha256", "")
            or entry.metadata.get("sha256", "")
        )
        if expected and digest != expected:
            raise QdpV2RepairError(
                f"qdp_v2_repair_active_shard_hash_mismatch:{path}"
            )
        hashes[_manifest_path_key(path, context.root)] = digest
    return hashes


def _validate_installed_against_entry(
    item: _InstalledParquet,
    entry: ShardManifestEntry,
    *,
    reference_schema: pa.Schema,
) -> None:
    path = item.target_path.resolve()
    parquet = pq.ParquetFile(path)
    if int(parquet.metadata.num_rows) != item.row_count:
        raise QdpV2RepairError(f"qdp_v2_repair_row_count_mismatch:{path}")
    if int(entry.row_count) != item.row_count:
        raise QdpV2RepairError(
            f"qdp_v2_repair_active_shard_row_count_mismatch:{path}"
        )
    if not parquet.schema_arrow.equals(reference_schema, check_metadata=False):
        raise QdpV2RepairError(
            f"qdp_v2_repair_parquet_schema_mismatch:{path}"
        )
    if _sha256_file(path) != item.file_sha256:
        raise QdpV2RepairError(
            f"qdp_v2_repair_installed_target_hash_mismatch:{path}"
        )


def _validate_updated_shard_paths(
    context: ActiveDomain,
    entries: Sequence[ShardManifestEntry],
    *,
    installed: Sequence[_InstalledParquet],
    reference_schema: pa.Schema,
) -> None:
    if not entries:
        raise QdpV2RepairError("qdp_v2_repair_cannot_remove_all_shards")
    by_key: dict[str, ShardManifestEntry] = {}
    for entry in entries:
        path = resolve_manifest_path(entry.path, root=context.root).resolve()
        _assert_deletable_dataset_shard(context, path)
        key = _manifest_path_key(path, context.root)
        if key in by_key:
            raise QdpV2RepairError(
                f"qdp_v2_repair_duplicate_manifest_shard_path:{path}"
            )
        if not path.is_file():
            raise QdpV2RepairError(
                f"qdp_v2_repair_updated_shard_missing:{path}"
            )
        if int(entry.file_size or 0) > 0 and path.stat().st_size != int(
            entry.file_size
        ):
            raise QdpV2RepairError(
                f"qdp_v2_repair_updated_shard_file_size_mismatch:{path}"
            )
        by_key[key] = entry
    for item in installed:
        key = _manifest_path_key(item.target_path, context.root)
        entry = by_key.get(key)
        if entry is None:
            raise QdpV2RepairError(
                f"qdp_v2_repair_installed_target_not_in_manifest:{item.target_path}"
            )
        _validate_installed_against_entry(
            item,
            entry,
            reference_schema=reference_schema,
        )


def _validate_replayed_shard_mutation(
    context: ActiveDomain,
    *,
    replacement_old_paths: Sequence[Path],
    removal_paths: Sequence[Path],
    installed: Sequence[_InstalledParquet],
    reference_schema: pa.Schema,
) -> None:
    active_by_key = {
        _manifest_path_key(item.path, context.root): item
        for item in context.manifest.shards
    }
    for old_path in [*replacement_old_paths, *removal_paths]:
        if _manifest_path_key(old_path, context.root) in active_by_key:
            raise QdpV2RepairError(
                f"qdp_v2_repair_replayed_old_shard_still_active:{old_path}"
            )
    for item in installed:
        entry = active_by_key.get(
            _manifest_path_key(item.target_path, context.root)
        )
        if entry is None:
            raise QdpV2RepairError(
                f"qdp_v2_repair_replayed_target_not_active:{item.target_path}"
            )
        _validate_installed_against_entry(
            item,
            entry,
            reference_schema=reference_schema,
        )


def _cleanup_committed_old_shards(
    context: ActiveDomain,
    *,
    old_paths: Sequence[Path],
    expected_sha256: Mapping[str, str],
) -> dict[str, list[str]]:
    deleted: list[str] = []
    retained: list[str] = []
    try:
        current = resolve_active_domain(
            context.domain,
            workspace_root=context.root.parent.parent.parent,
        )
    except BaseException:
        return {
            "deleted": deleted,
            "retained": [str(item) for item in old_paths if item.exists()],
        }
    if current.dataset_id != context.dataset_id:
        return {
            "deleted": deleted,
            "retained": [str(item) for item in old_paths if item.exists()],
        }
    active_keys = {
        _manifest_path_key(item.path, current.root)
        for item in current.manifest.shards
    }
    for path in old_paths:
        resolved = path.resolve()
        if not resolved.exists():
            continue
        key = _manifest_path_key(resolved, context.root)
        expected = str(expected_sha256.get(key, ""))
        if key in active_keys or not expected:
            retained.append(str(resolved))
            continue
        try:
            _assert_deletable_dataset_shard(context, resolved)
            if _sha256_file(resolved) != expected:
                retained.append(str(resolved))
                continue
            resolved.unlink()
            deleted.append(str(resolved))
        except OSError:
            retained.append(str(resolved))
    return {"deleted": deleted, "retained": retained}


def _validate_frame(frame: pd.DataFrame, *, context: ActiveDomain) -> pd.DataFrame:
    if not isinstance(frame, pd.DataFrame) or frame.empty:
        raise QdpV2RepairError("qdp_v2_repair_frame_empty")
    data = frame.copy().reset_index(drop=True)
    expected_columns = [str(item.get("name", "")) for item in context.manifest.schema]
    expected_columns = [item for item in expected_columns if item]
    if expected_columns and list(data.columns) != expected_columns:
        raise QdpV2RepairError(
            "qdp_v2_repair_schema_columns_mismatch:"
            f"expected={expected_columns}:actual={list(data.columns)}"
        )
    primary_key = list(context.manifest.primary_key)
    if not primary_key or any(column not in data.columns for column in primary_key):
        raise QdpV2RepairError(
            f"qdp_v2_repair_primary_key_columns_missing:{primary_key}"
        )
    if data[primary_key].isna().any().any():
        raise QdpV2RepairError("qdp_v2_repair_primary_key_null")
    if data.duplicated(primary_key, keep=False).any():
        raise QdpV2RepairError("qdp_v2_repair_primary_key_duplicate")
    _frame_date_range(data, context.manifest)
    return data


def _reference_schema(context: ActiveDomain) -> pa.Schema:
    for path in context.shard_paths:
        if path.is_file():
            return pq.ParquetFile(path).schema_arrow
    raise QdpV2RepairError("qdp_v2_repair_reference_schema_missing")


def _write_and_validate_parquet(
    target: Path,
    frame: pd.DataFrame,
    *,
    reference_schema: pa.Schema,
    expected_content_hash: str,
) -> bool:
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        _validate_existing_parquet(
            target,
            expected_frame=frame,
            reference_schema=reference_schema,
            expected_content_hash=expected_content_hash,
        )
        return False
    temporary = target.with_name(
        f".{target.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp"
    )
    try:
        frame.to_parquet(
            temporary,
            index=False,
            engine="pyarrow",
            compression="zstd",
        )
        _validate_existing_parquet(
            temporary,
            expected_frame=frame,
            reference_schema=reference_schema,
            expected_content_hash=expected_content_hash,
        )
        _replace_file_with_retry(temporary, target)
    finally:
        _unlink_file_with_retry(temporary)
    return True


def _replace_file_with_retry(source: Path, target: Path, *, timeout_seconds: float = 5.0) -> None:
    deadline = time.monotonic() + max(0.0, float(timeout_seconds))
    while True:
        try:
            source.replace(target)
            return
        except PermissionError:
            if time.monotonic() >= deadline:
                raise
            time.sleep(0.05)


def _unlink_file_with_retry(path: Path, *, timeout_seconds: float = 5.0) -> None:
    deadline = time.monotonic() + max(0.0, float(timeout_seconds))
    while True:
        try:
            path.unlink(missing_ok=True)
            return
        except PermissionError:
            if time.monotonic() >= deadline:
                raise
            time.sleep(0.05)


def _validate_existing_parquet(
    path: Path,
    *,
    expected_frame: pd.DataFrame,
    reference_schema: pa.Schema,
    expected_content_hash: str,
) -> None:
    with path.open("rb") as handle:
        parquet = pq.ParquetFile(handle)
        if int(parquet.metadata.num_rows) != len(expected_frame):
            raise QdpV2RepairError(f"qdp_v2_repair_row_count_mismatch:{path}")
        if not parquet.schema_arrow.equals(reference_schema, check_metadata=False):
            raise QdpV2RepairError(f"qdp_v2_repair_parquet_schema_mismatch:{path}")
        persisted = parquet.read().to_pandas()
    if frame_content_sha256(persisted) != expected_content_hash:
        raise QdpV2RepairError(f"qdp_v2_repair_content_hash_mismatch:{path}")


def _validate_cross_shard_primary_keys(
    context: ActiveDomain,
    *,
    new_paths: Sequence[Path],
    retained_entries: Sequence[ShardManifestEntry],
) -> None:
    primary_key = list(context.manifest.primary_key)
    if not primary_key:
        raise QdpV2RepairError("qdp_v2_repair_primary_key_missing")
    new_sql = _parquet_list_sql(new_paths)
    quoted_keys = ",".join(_quote_identifier(item) for item in primary_key)
    retained = [
        resolve_manifest_path(item.path, root=context.root).resolve()
        for item in _overlapping_entries(context.manifest, retained_entries, new_paths)
    ]
    with _open_repair_duckdb(context) as connection:
        duplicate = connection.execute(
            f"SELECT 1 FROM read_parquet({new_sql}, union_by_name=true) "
            f"GROUP BY {quoted_keys} HAVING count(*) > 1 LIMIT 1"
        ).fetchone()
        if duplicate:
            raise QdpV2RepairError("qdp_v2_repair_new_shards_primary_key_overlap")
        if retained:
            retained_sql = _parquet_list_sql(retained)
            join = " AND ".join(
                f"n.{_quote_identifier(item)} = e.{_quote_identifier(item)}"
                for item in primary_key
            )
            overlap = connection.execute(
                f"SELECT 1 FROM read_parquet({new_sql}, union_by_name=true) n "
                f"JOIN read_parquet({retained_sql}, union_by_name=true) e ON {join} LIMIT 1"
            ).fetchone()
            if overlap:
                raise QdpV2RepairError(
                    "qdp_v2_repair_retained_shard_primary_key_overlap"
                )


def _overlapping_entries(
    manifest: DatasetManifest,
    entries: Sequence[ShardManifestEntry],
    new_paths: Sequence[Path],
) -> list[ShardManifestEntry]:
    date_column = _date_column(manifest)
    if not date_column:
        return list(entries)
    starts: list[str] = []
    ends: list[str] = []
    for path in new_paths:
        frame = pd.read_parquet(path, columns=[date_column], engine="pyarrow")
        start, end = _series_date_range(frame[date_column])
        starts.append(start)
        ends.append(end)
    new_start, new_end = min(starts), max(ends)
    return [
        item
        for item in entries
        if not item.start_date
        or not item.end_date
        or (str(item.start_date) <= new_end and str(item.end_date) >= new_start)
    ]


def _new_shard_entry(
    context: ActiveDomain,
    path: Path,
    frame: pd.DataFrame,
    *,
    start_date: str,
    end_date: str,
    content_hash: str,
    reason: str,
    old_entry: ShardManifestEntry | None,
) -> ShardManifestEntry:
    return ShardManifestEntry(
        path=_path_for_manifest(path, root=context.root),
        row_count=len(frame),
        start_date=start_date,
        end_date=end_date,
        status="stored",
        file_size=path.stat().st_size,
        schema_hash=(old_entry.schema_hash if old_entry else context.manifest.schema_hash),
        source_path=(str(old_entry.path) if old_entry else ""),
        content_key=f"repair:{content_hash[:24]}",
        metadata={
            "repair_reason": reason,
            "repair_content_sha256": content_hash,
            "replaces": str(old_entry.path) if old_entry else "",
        },
    )


def _updated_manifest_payload(
    context: ActiveDomain,
    entries: Sequence[ShardManifestEntry],
    *,
    reason: str,
    action: str,
) -> dict[str, Any]:
    if not entries:
        raise QdpV2RepairError("qdp_v2_repair_cannot_remove_all_shards")
    payload = context.manifest.to_dict()
    payload["dataset_id"] = context.dataset_id
    payload["domain"] = context.domain
    payload["shards"] = [item.to_dict() for item in entries]
    payload["row_count"] = sum(int(item.row_count) for item in entries)
    starts = [str(item.start_date) for item in entries if str(item.start_date)]
    ends = [str(item.end_date) for item in entries if str(item.end_date)]
    payload["start_date"] = min(starts) if starts else ""
    payload["end_date"] = max(ends) if ends else ""
    notes = [str(item) for item in list(payload.get("notes", []) or [])]
    repair_note = f"{action}:{reason}"
    if repair_note not in notes:
        notes.append(repair_note)
    payload["notes"] = notes
    source = dict(payload.get("source", {}) or {})
    source["last_repair_action"] = action
    source["last_repair_reason"] = reason
    source["last_repair_at"] = _utc_now()
    payload["source"] = source
    validated = DatasetManifest.from_mapping(payload)
    if validated.dataset_id != context.dataset_id or validated.domain != context.domain:
        raise QdpV2RepairError("qdp_v2_repair_updated_manifest_identity_changed")
    if validated.row_count != sum(item.row_count for item in validated.shards):
        raise QdpV2RepairError("qdp_v2_repair_updated_manifest_row_count_invalid")
    return payload


def _commit_manifest(context: ActiveDomain, payload: Mapping[str, Any]) -> None:
    _assert_context_unchanged(context)
    # Validate the exact payload before the atomic replace.  A validation read
    # after replacement could fail after the new manifest is already live;
    # callers would then mistake a committed transaction for a pre-commit
    # failure and remove shards referenced by that manifest.
    written = DatasetManifest.from_mapping(payload)
    if written.dataset_id != context.dataset_id or written.domain != context.domain:
        raise QdpV2RepairError("qdp_v2_repair_committed_manifest_identity_invalid")
    if written.row_count != sum(int(item.row_count) for item in written.shards):
        raise QdpV2RepairError("qdp_v2_repair_committed_manifest_row_count_invalid")
    atomic_write_json(context.manifest_path, payload)


def _resolve_selected_entries(
    context: ActiveDomain,
    requested_paths: Sequence[str | Path],
) -> list[ShardManifestEntry]:
    if not requested_paths:
        raise ValueError("qdp_v2_repair_replaced_shard_paths_required")
    by_key = {
        _manifest_path_key(item.path, context.root): item
        for item in context.manifest.shards
    }
    selected: list[ShardManifestEntry] = []
    seen: set[str] = set()
    for requested in requested_paths:
        key = _manifest_path_key(requested, context.root)
        if key in seen:
            raise ValueError(f"qdp_v2_repair_duplicate_replaced_path:{requested}")
        seen.add(key)
        entry = by_key.get(key)
        if entry is None:
            raise QdpV2RepairError(
                f"qdp_v2_repair_shard_not_active:{requested}"
            )
        selected.append(entry)
    return selected


def _pair_replacement_frames(
    replacement_frames: Sequence[pd.DataFrame] | Mapping[str | Path, pd.DataFrame] | pd.DataFrame,
    *,
    replaced_shard_paths: Sequence[str | Path],
    selected_entries: Sequence[ShardManifestEntry],
    context: ActiveDomain,
) -> list[pd.DataFrame]:
    if isinstance(replacement_frames, pd.DataFrame):
        frames = [replacement_frames]
    elif isinstance(replacement_frames, Mapping):
        mapped = {
            _manifest_path_key(key, context.root): value
            for key, value in replacement_frames.items()
        }
        frames = []
        for entry in selected_entries:
            key = _manifest_path_key(entry.path, context.root)
            if key not in mapped:
                raise ValueError(
                    f"qdp_v2_repair_replacement_frame_missing:{entry.path}"
                )
            frames.append(mapped[key])
    else:
        frames = list(replacement_frames)
    if len(frames) != len(replaced_shard_paths):
        raise ValueError(
            "qdp_v2_repair_replacement_count_mismatch:"
            f"frames={len(frames)}:paths={len(replaced_shard_paths)}"
        )
    return frames


def _active_shard_directory(context: ActiveDomain) -> Path:
    if context.manifest.shards:
        first = resolve_manifest_path(
            context.manifest.shards[0].path, root=context.root
        ).resolve()
        _assert_deletable_dataset_shard(context, first)
        return first.parent
    shard_dir = context.dataset_dir / "shards"
    shard_dir.mkdir(parents=True, exist_ok=True)
    return shard_dir


def _assert_deletable_dataset_shard(context: ActiveDomain, path: Path) -> None:
    resolved = path.resolve()
    dataset_dir = context.dataset_dir.resolve()
    try:
        resolved.relative_to(dataset_dir)
    except ValueError as exc:
        raise QdpV2RepairError(
            f"qdp_v2_repair_external_shard_mutation_refused:{resolved}"
        ) from exc
    if resolved == context.manifest_path or resolved.suffix.lower() != ".parquet":
        raise QdpV2RepairError(
            f"qdp_v2_repair_invalid_shard_target:{resolved}"
        )


def _frame_date_range(
    frame: pd.DataFrame, manifest: DatasetManifest
) -> tuple[str, str]:
    column = _date_column(manifest)
    if not column:
        return "", ""
    if column not in frame.columns:
        raise QdpV2RepairError("qdp_v2_repair_date_column_missing")
    return _series_date_range(frame[column])


def _series_date_range(series: pd.Series) -> tuple[str, str]:
    parsed = pd.to_datetime(series, errors="coerce")
    if parsed.isna().any():
        raise QdpV2RepairError("qdp_v2_repair_date_value_invalid")
    text = parsed.dt.strftime("%Y-%m-%d")
    return str(text.min()), str(text.max())


def _date_column(manifest: DatasetManifest) -> str:
    fields = [str(item.get("name", "")) for item in manifest.schema]
    fields.extend(manifest.primary_key)
    return next((item for item in _DATE_COLUMNS if item in fields), "")


def _assert_context_unchanged(context: ActiveDomain) -> None:
    if _sha256_file(context.active_path) != context.active_sha256:
        raise QdpV2RepairConflictError("qdp_v2_repair_active_changed")
    if _sha256_file(context.manifest_path) != context.manifest_sha256:
        raise QdpV2RepairConflictError("qdp_v2_repair_manifest_changed")


def _stable_json_read(path: Path, kind: str) -> tuple[str, dict[str, Any]]:
    if not path.is_file():
        raise FileNotFoundError(f"qdp_v2_repair_{kind}_missing:{path}")
    before = _sha256_file(path)
    payload = read_json(path)
    after = _sha256_file(path)
    if before != after:
        raise QdpV2RepairConflictError(
            f"qdp_v2_repair_{kind}_changed_while_reading"
        )
    return after, payload


def _append_repair_log(
    workspace_root: str | Path | None,
    payload: Mapping[str, Any],
) -> None:
    path = repair_log_path(workspace_root)
    path.parent.mkdir(parents=True, exist_ok=True)
    record = {"logged_at": _utc_now(), **dict(payload)}
    line = json.dumps(json_safe(record), ensure_ascii=False, sort_keys=True) + "\n"
    with path.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(line)
        handle.flush()
        os.fsync(handle.fileno())


def _manifest_path_key(path: str | Path, root: Path) -> str:
    candidate = Path(path)
    resolved = candidate.resolve() if candidate.is_absolute() else (root / candidate).resolve()
    return os.path.normcase(os.path.abspath(str(resolved)))


def _path_for_manifest(path: Path, *, root: Path) -> str:
    resolved = path.resolve()
    try:
        return resolved.relative_to(root.resolve()).as_posix()
    except ValueError:
        return str(resolved)


def _manifest_schema_from_arrow(schema: pa.Schema) -> list[dict[str, str]]:
    def sql_type(value: pa.DataType) -> str:
        if pa.types.is_boolean(value):
            return "BOOLEAN"
        if pa.types.is_integer(value):
            return "BIGINT"
        if pa.types.is_floating(value) or pa.types.is_decimal(value):
            return "DOUBLE"
        if pa.types.is_date(value) or pa.types.is_timestamp(value):
            return "TIMESTAMP"
        return "VARCHAR"

    return [
        {"name": field.name, "type": sql_type(field.type)}
        for field in schema
    ]


def _parquet_list_sql(paths: Sequence[Path]) -> str:
    if not paths:
        raise QdpV2RepairError("qdp_v2_repair_parquet_path_list_empty")
    return "[" + ",".join(_sql_literal(str(path)) for path in paths) + "]"


def _quote_identifier(value: str) -> str:
    return '"' + str(value).replace('"', '""') + '"'


def _sql_literal(value: str) -> str:
    return "'" + str(value).replace("'", "''") + "'"


def _required_reason(reason: str) -> str:
    text = str(reason or "").strip()
    if not text:
        raise ValueError("qdp_v2_repair_reason_required")
    return text


def _sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while True:
            chunk = handle.read(8 * 1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def _utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


__all__ = [
    "ActiveDomain",
    "QdpV2RepairConflictError",
    "QdpV2RepairError",
    "append_active_shard",
    "bulk_append_active_shards_from_parquet",
    "frame_content_sha256",
    "mutate_active_shards_from_parquet",
    "replace_active_table_from_parquet",
    "repair_log_path",
    "replace_active_shard_from_parquet",
    "replace_active_shards",
    "replace_active_shards_from_parquet",
    "resolve_active_domain",
]
