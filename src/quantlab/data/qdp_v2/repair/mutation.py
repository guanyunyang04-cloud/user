"""Repair: mutation responsibilities."""

from __future__ import annotations

import shutil
from collections.abc import Mapping, Sequence
from dataclasses import replace
from pathlib import Path
from typing import Any

import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

from quantlab.data.qdp_v2.active import resolve_active_domain
from quantlab.data.qdp_v2.manifest import (
    resolve_manifest_path,
)

from .common import (
    _time_token,
)
from .errors import QdpV2RepairError
from .validation import (
    _commit_manifest,
    _common_schema,
    _ensure_unique_paths,
    _entry_for_parquet,
    _install_prepared_files,
    _new_shard_path,
    _read_patch_request,
    _reference_schema,
    _remove_files,
    _remove_old_shards,
    _require_columns,
    _resolve_active_shard,
    _resolve_prepared_path,
    _updated_manifest,
    _validate_append_keys,
    _validate_parquet,
    _validate_primary_keys,
)


def replace_active_table_from_parquet(
    domain: str,
    prepared_parquet_path: str | Path | Sequence[str | Path],
    *,
    reason: str = "replace active table",
    workspace_root: str | Path | None = None,
    primary_key: Sequence[str] | None = None,
    contract_version: str = "",
    source_updates: Mapping[str, Any] | None = None,
    quality_updates: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    context = resolve_active_domain(domain, workspace_root=workspace_root)
    requested = (
        [prepared_parquet_path] if isinstance(prepared_parquet_path, (str, Path)) else list(prepared_parquet_path)
    )
    prepared = [_resolve_prepared_path(item, workspace_root) for item in requested]
    if not prepared:
        raise ValueError("qdp_v2_repair_prepared_parquet_paths_required")
    _ensure_unique_paths(prepared)
    schema = _common_schema(prepared)
    keys = [str(item) for item in (primary_key or context.manifest.primary_key)]
    _require_columns(schema.names, keys, "primary_key")
    _validate_primary_keys(prepared, keys)

    installed = _install_prepared_files(context, prepared, kind="replace")
    try:
        entries = [_entry_for_parquet(context, path, reason=reason) for path in installed]
        source = {**context.manifest.source, **dict(source_updates or {})}
        quality = {**context.manifest.quality, **dict(quality_updates or {})}
        manifest = _updated_manifest(
            context,
            entries,
            schema=schema,
            primary_key=keys,
            contract_version=contract_version or context.manifest.contract_version,
            source=source,
            quality=quality,
        )
        _commit_manifest(context, manifest)
    except Exception:
        _remove_files(installed)
        raise

    deleted, retained = _remove_old_shards(context, context.shard_paths)
    return {
        "status": "replaced",
        "domain": context.domain,
        "dataset_id": context.dataset_id,
        "row_count": manifest.row_count,
        "shard_count": len(manifest.shards),
        "installed_shard_paths": [str(item) for item in installed],
        "deleted_shard_paths": deleted,
        "retained_old_shard_paths": retained,
    }


def update_active_manifest_metadata(
    domain: str,
    *,
    reason: str = "update metadata",
    workspace_root: str | Path | None = None,
    source_updates: Mapping[str, Any] | None = None,
    quality_updates: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    del reason
    context = resolve_active_domain(domain, workspace_root=workspace_root)
    manifest = replace(
        context.manifest,
        source={**context.manifest.source, **dict(source_updates or {})},
        quality={**context.manifest.quality, **dict(quality_updates or {})},
    )
    _commit_manifest(context, manifest)
    return {
        "status": "metadata_updated",
        "domain": context.domain,
        "dataset_id": context.dataset_id,
        "row_count": manifest.row_count,
        "shard_count": len(manifest.shards),
    }


def append_active_shard(
    domain: str,
    frame: pd.DataFrame,
    reason: str = "append rows",
    *,
    workspace_root: str | Path | None = None,
) -> dict[str, Any]:
    context = resolve_active_domain(domain, workspace_root=workspace_root)
    if frame is None or frame.empty:
        return {
            "status": "nothing_to_append",
            "domain": context.domain,
            "dataset_id": context.dataset_id,
            "row_count": 0,
        }
    schema = _reference_schema(context)
    missing = sorted(set(schema.names).difference(frame.columns))
    extra = sorted(set(frame.columns).difference(schema.names))
    if missing or extra:
        raise QdpV2RepairError(f"qdp_v2_repair_frame_columns_mismatch:missing={missing}:extra={extra}")
    table = pa.Table.from_pandas(frame.loc[:, schema.names], schema=schema, preserve_index=False, safe=True)
    target = _new_shard_path(context, "append")
    temporary = target.with_suffix(target.suffix + ".tmp")
    target.parent.mkdir(parents=True, exist_ok=True)
    try:
        pq.write_table(table, temporary, compression="zstd")
        _validate_parquet(temporary, expected_schema=schema)
        temporary.replace(target)
        _validate_append_keys(context.shard_paths, target, context.manifest.primary_key)
        entry = _entry_for_parquet(context, target, reason=reason)
        manifest = _updated_manifest(
            context,
            [*context.manifest.shards, entry],
            schema=schema,
        )
        _commit_manifest(context, manifest)
    except Exception:
        temporary.unlink(missing_ok=True)
        target.unlink(missing_ok=True)
        raise
    return {
        "status": "appended",
        "domain": context.domain,
        "dataset_id": context.dataset_id,
        "appended_row_count": int(table.num_rows),
        "row_count": manifest.row_count,
        "shard_path": str(target),
    }


def bulk_append_active_shards_from_parquet(
    domain: str,
    prepared_parquet_paths: Sequence[str | Path],
    reason: str = "append prepared shards",
    *,
    workspace_root: str | Path | None = None,
) -> dict[str, Any]:
    context = resolve_active_domain(domain, workspace_root=workspace_root)
    prepared = [_resolve_prepared_path(item, workspace_root) for item in prepared_parquet_paths]
    if not prepared:
        return {
            "status": "nothing_to_append",
            "domain": context.domain,
            "dataset_id": context.dataset_id,
            "row_count": 0,
        }
    _ensure_unique_paths(prepared)
    schema = _reference_schema(context)
    for path in prepared:
        _validate_parquet(path, expected_schema=schema)
    installed = _install_prepared_files(context, prepared, kind="append")
    try:
        _validate_append_keys(context.shard_paths, installed, context.manifest.primary_key)
        entries = [_entry_for_parquet(context, path, reason=reason) for path in installed]
        manifest = _updated_manifest(
            context,
            [*context.manifest.shards, *entries],
            schema=schema,
        )
        _commit_manifest(context, manifest)
    except Exception:
        _remove_files(installed)
        raise
    return {
        "status": "appended",
        "domain": context.domain,
        "dataset_id": context.dataset_id,
        "appended_row_count": sum(item.row_count for item in entries),
        "row_count": manifest.row_count,
        "installed_shard_paths": [str(item) for item in installed],
    }


def mutate_active_shards_from_parquet(
    domain: str,
    *,
    replacements: Sequence[tuple[str | Path, str | Path]] = (),
    removals: Sequence[str | Path] = (),
    appends: Sequence[str | Path] = (),
    reason: str = "mutate active shards",
    workspace_root: str | Path | None = None,
    primary_keys_prevalidated: bool = False,
    allow_selected_schema_superset: bool = False,
    changed_columns: Sequence[str] = (),
    affected_date_range: Mapping[str, Any] | str | None = None,
) -> dict[str, Any]:
    del allow_selected_schema_superset, changed_columns, affected_date_range
    context = resolve_active_domain(domain, workspace_root=workspace_root)
    replacement_pairs = [
        (
            _resolve_active_shard(context, old),
            _resolve_prepared_path(new, workspace_root),
        )
        for old, new in replacements
    ]
    removal_paths = [_resolve_active_shard(context, item) for item in removals]
    append_paths = [_resolve_prepared_path(item, workspace_root) for item in appends]
    selected = [old for old, _ in replacement_pairs] + removal_paths
    if len(selected) != len(set(selected)):
        raise QdpV2RepairError("qdp_v2_repair_duplicate_selected_shard")
    prepared = [new for _, new in replacement_pairs] + append_paths
    _ensure_unique_paths(prepared)
    schema = _reference_schema(context)
    for path in prepared:
        _validate_parquet(path, expected_schema=schema)

    installed = _install_prepared_files(context, prepared, kind="mutate")
    installed_replacements = installed[: len(replacement_pairs)]
    installed_appends = installed[len(replacement_pairs) :]
    selected_set = {item.resolve() for item in selected}
    retained_entries = [
        entry
        for path, entry in zip(context.shard_paths, context.manifest.shards, strict=True)
        if path.resolve() not in selected_set
    ]
    new_entries = [
        _entry_for_parquet(context, path, reason=reason) for path in [*installed_replacements, *installed_appends]
    ]
    resulting_entries = [*retained_entries, *new_entries]
    resulting_paths = [resolve_manifest_path(item.path, root=context.root) for item in retained_entries] + [
        *installed_replacements,
        *installed_appends,
    ]
    try:
        if not primary_keys_prevalidated:
            _validate_primary_keys(resulting_paths, context.manifest.primary_key)
        manifest = _updated_manifest(
            context,
            resulting_entries,
            schema=schema,
        )
        _commit_manifest(context, manifest)
    except Exception:
        _remove_files(installed)
        raise

    deleted, retained_old = _remove_old_shards(context, selected)
    return {
        "status": "mutated",
        "domain": context.domain,
        "dataset_id": context.dataset_id,
        "row_count": manifest.row_count,
        "shard_count": len(manifest.shards),
        "installed_shard_paths": [str(item) for item in installed],
        "deleted_shard_paths": deleted,
        "retained_old_shard_paths": retained_old,
    }


def patch_active_cells(
    request_path: str | Path,
    *,
    reason: str = "patch cells",
    workspace_root: str | Path | None = None,
    apply: bool = False,
) -> dict[str, Any]:
    request = _read_patch_request(request_path, workspace_root)
    context = resolve_active_domain(str(request["domain"]), workspace_root=workspace_root)
    keys = list(context.manifest.primary_key)
    schema = _reference_schema(context)
    changes = [dict(item) for item in request["changes"]]
    tables: dict[Path, pa.Table] = {}
    changes_by_shard: dict[Path, list[tuple[int, str, pa.Scalar]]] = {}

    for index, change in enumerate(changes):
        if set(change) != {"key", "column", "expected", "value"}:
            raise QdpV2RepairError(f"qdp_v2_repair_patch_change_invalid:{index}")
        key = change["key"]
        if not isinstance(key, Mapping) or set(key) != set(keys):
            raise QdpV2RepairError(f"qdp_v2_repair_patch_key_invalid:{index}")
        column = str(change["column"])
        if column in keys or column not in schema.names:
            raise QdpV2RepairError(f"qdp_v2_repair_patch_column_invalid:{column}")
        matches: list[tuple[Path, int]] = []
        for shard_path in context.shard_paths:
            key_table = pq.read_table(shard_path, columns=keys)
            mask = None
            for key_name in keys:
                scalar = pa.scalar(key[key_name], type=schema.field(key_name).type)
                current = pc.equal(key_table[key_name], scalar)
                mask = current if mask is None else pc.and_(mask, current)
            indices = pc.indices_nonzero(mask).to_pylist() if mask is not None else []
            matches.extend((shard_path, int(row)) for row in indices)
        if len(matches) != 1:
            raise QdpV2RepairError(f"qdp_v2_repair_patch_match_count_invalid:{index}:{len(matches)}")
        shard_path, row_index = matches[0]
        table = tables.setdefault(shard_path, pq.read_table(shard_path))
        expected = pa.scalar(change["expected"], type=schema.field(column).type)
        actual = table[column][row_index]
        if actual.as_py() != expected.as_py():
            raise QdpV2RepairError(
                f"qdp_v2_repair_patch_expected_mismatch:{index}:expected={expected.as_py()!r}:actual={actual.as_py()!r}"
            )
        value = pa.scalar(change["value"], type=schema.field(column).type)
        changes_by_shard.setdefault(shard_path, []).append((row_index, column, value))

    plan = {
        "status": "would_patch" if not apply else "patching",
        "dry_run": not apply,
        "domain": context.domain,
        "dataset_id": context.dataset_id,
        "changed_columns": sorted({str(item["column"]) for item in changes}),
        "row_count_touched": sum(len(items) for items in changes_by_shard.values()),
        "shard_paths": [str(item) for item in changes_by_shard],
    }
    if not apply:
        return plan

    prepared_dir = context.root / "tmp" / f"patch_{_time_token()}"
    prepared_dir.mkdir(parents=True, exist_ok=True)
    replacements: list[tuple[Path, Path]] = []
    try:
        for number, (shard_path, edits) in enumerate(changes_by_shard.items()):
            table = tables[shard_path]
            columns = [table[column] for column in table.column_names]
            for row_index, column, value in edits:
                column_index = table.schema.get_field_index(column)
                values = columns[column_index].to_pylist()
                values[row_index] = value.as_py()
                columns[column_index] = pa.array(values, type=table.schema.field(column).type)
            updated = pa.Table.from_arrays(columns, schema=table.schema)
            target = prepared_dir / f"part_{number:04d}.parquet"
            pq.write_table(updated, target, compression="zstd")
            replacements.append((shard_path, target))
        result = mutate_active_shards_from_parquet(
            context.domain,
            replacements=replacements,
            reason=reason,
            workspace_root=workspace_root,
        )
        return {**plan, **result, "status": "patched", "dry_run": False}
    finally:
        shutil.rmtree(prepared_dir, ignore_errors=True)
