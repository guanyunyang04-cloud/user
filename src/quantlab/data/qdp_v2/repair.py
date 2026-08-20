from __future__ import annotations

import argparse
import json
import shutil
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

from quantlab.data.core.json_io import json_safe, read_json
from quantlab.data.core.paths import qdp_paths
from quantlab.data.qdp_v2.manifest import (
    DatasetManifest,
    ShardManifestEntry,
    _manifest_schema_from_arrow,
    active_manifest_path,
    dataset_manifest_path,
    path_for_manifest,
    qdp_v2_root,
    read_active_manifest,
    read_dataset_manifest,
    resolve_manifest_path,
    write_dataset_manifest,
)
from quantlab.data.qdp_v2.status import active_dataset_map

_DATE_COLUMNS = ("trade_date", "date", "datetime")


class QdpV2RepairError(RuntimeError):
    pass


@dataclass(frozen=True)
class ActiveDomain:
    root: Path
    active_path: Path
    active: dict[str, Any]
    domain: str
    dataset_id: str
    manifest_path: Path
    manifest: DatasetManifest

    @property
    def shard_paths(self) -> list[Path]:
        return [
            resolve_manifest_path(item.path, root=self.root)
            for item in self.manifest.shards
        ]


def resolve_active_domain(
    domain: str,
    *,
    workspace_root: str | Path | None = None,
) -> ActiveDomain:
    requested = str(domain or "").strip()
    if not requested:
        raise ValueError("qdp_v2_repair_domain_required")
    root = qdp_v2_root(workspace_root).resolve()
    active_path = active_manifest_path(root).resolve()
    active = read_active_manifest(root)
    dataset_id = active_dataset_map(active).get(requested, "")
    if not dataset_id:
        raise QdpV2RepairError(f"qdp_v2_repair_active_domain_missing:{requested}")
    manifest_path = dataset_manifest_path(root, requested, dataset_id).resolve()
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
        active=active,
        domain=requested,
        dataset_id=dataset_id,
        manifest_path=manifest_path,
        manifest=manifest,
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
        [prepared_parquet_path]
        if isinstance(prepared_parquet_path, (str, Path))
        else list(prepared_parquet_path)
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
        entries = [
            _entry_for_parquet(context, path, reason=reason)
            for path in installed
        ]
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
        raise QdpV2RepairError(
            f"qdp_v2_repair_frame_columns_mismatch:missing={missing}:extra={extra}"
        )
    table = pa.Table.from_pandas(
        frame.loc[:, schema.names], schema=schema, preserve_index=False, safe=True
    )
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
        entries = [
            _entry_for_parquet(context, path, reason=reason) for path in installed
        ]
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
        _entry_for_parquet(context, path, reason=reason)
        for path in [*installed_replacements, *installed_appends]
    ]
    resulting_entries = [*retained_entries, *new_entries]
    resulting_paths = [
        resolve_manifest_path(item.path, root=context.root)
        for item in retained_entries
    ] + [*installed_replacements, *installed_appends]
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
            raise QdpV2RepairError(
                f"qdp_v2_repair_patch_match_count_invalid:{index}:{len(matches)}"
            )
        shard_path, row_index = matches[0]
        table = tables.setdefault(shard_path, pq.read_table(shard_path))
        expected = pa.scalar(change["expected"], type=schema.field(column).type)
        actual = table[column][row_index]
        if actual.as_py() != expected.as_py():
            raise QdpV2RepairError(
                f"qdp_v2_repair_patch_expected_mismatch:{index}:"
                f"expected={expected.as_py()!r}:actual={actual.as_py()!r}"
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


def _read_patch_request(
    path: str | Path,
    workspace_root: str | Path | None,
) -> dict[str, Any]:
    candidate = Path(path).expanduser()
    if not candidate.is_absolute():
        candidate = qdp_paths(workspace_root).workspace_root / candidate
    payload = read_json(candidate.resolve())
    allowed = {"schema_version", "domain", "changes"}
    unknown = set(payload).difference(allowed)
    if unknown:
        raise QdpV2RepairError(f"qdp_v2_repair_patch_request_keys_invalid:{sorted(unknown)}")
    if not str(payload.get("domain", "") or "").strip():
        raise QdpV2RepairError("qdp_v2_repair_patch_domain_required")
    changes = payload.get("changes")
    if not isinstance(changes, list) or not changes or not all(
        isinstance(item, Mapping) for item in changes
    ):
        raise QdpV2RepairError("qdp_v2_repair_patch_changes_required")
    return dict(payload)


def _reference_schema(context: ActiveDomain) -> pa.Schema:
    if not context.shard_paths:
        raise QdpV2RepairError("qdp_v2_repair_active_shards_missing")
    return pq.read_schema(context.shard_paths[0])


def _common_schema(paths: Sequence[Path]) -> pa.Schema:
    schema = pq.read_schema(paths[0])
    for path in paths:
        _validate_parquet(path, expected_schema=schema)
    return schema


def _validate_parquet(path: Path, *, expected_schema: pa.Schema | None = None) -> None:
    if not path.is_file():
        raise FileNotFoundError(path)
    parquet = pq.ParquetFile(path)
    if parquet.metadata is None or int(parquet.metadata.num_rows) <= 0:
        raise QdpV2RepairError(f"qdp_v2_repair_parquet_empty:{path}")
    if expected_schema is not None and not parquet.schema_arrow.equals(
        expected_schema, check_metadata=False
    ):
        raise QdpV2RepairError(f"qdp_v2_repair_parquet_schema_mismatch:{path}")


def _require_columns(columns: Sequence[str], required: Sequence[str], label: str) -> None:
    missing = sorted(set(required).difference(columns))
    if missing:
        raise QdpV2RepairError(f"qdp_v2_repair_{label}_columns_missing:{missing}")


def _validate_primary_keys(paths: Sequence[Path], keys: Sequence[str]) -> None:
    if not paths or not keys:
        raise QdpV2RepairError("qdp_v2_repair_primary_key_missing")
    import duckdb

    columns = ",".join(_quote_identifier(item) for item in keys)
    sql_paths = _parquet_list_sql(paths)
    with duckdb.connect(":memory:") as connection:
        duplicate = connection.execute(
            f"SELECT 1 FROM read_parquet({sql_paths}, union_by_name=true) "
            f"GROUP BY {columns} HAVING count(*) > 1 LIMIT 1"
        ).fetchone()
    if duplicate is not None:
        raise QdpV2RepairError("qdp_v2_repair_primary_key_duplicate")


def _validate_append_keys(
    existing: Sequence[Path],
    additions: Path | Sequence[Path],
    keys: Sequence[str],
) -> None:
    new_paths = [additions] if isinstance(additions, Path) else list(additions)
    _validate_primary_keys(new_paths, keys)
    if not existing:
        return
    import duckdb

    condition = " AND ".join(
        f"n.{_quote_identifier(item)} = o.{_quote_identifier(item)}" for item in keys
    )
    with duckdb.connect(":memory:") as connection:
        overlap = connection.execute(
            f"SELECT 1 FROM read_parquet({_parquet_list_sql(new_paths)}, union_by_name=true) n "
            f"JOIN read_parquet({_parquet_list_sql(existing)}, union_by_name=true) o "
            f"ON {condition} LIMIT 1"
        ).fetchone()
    if overlap is not None:
        raise QdpV2RepairError("qdp_v2_repair_primary_key_overlap")


def _entry_for_parquet(
    context: ActiveDomain,
    path: Path,
    *,
    reason: str,
) -> ShardManifestEntry:
    parquet = pq.ParquetFile(path)
    date_column = _date_column(parquet.schema_arrow.names)
    start_date, end_date = _parquet_date_range(path, date_column)
    return ShardManifestEntry(
        path=path_for_manifest(path, root=context.root),
        row_count=int(parquet.metadata.num_rows),
        start_date=start_date,
        end_date=end_date,
        status="stored",
        file_size=int(path.stat().st_size),
        metadata={"reason": str(reason), "created_at": _utc_now()},
    )


def _updated_manifest(
    context: ActiveDomain,
    entries: Sequence[ShardManifestEntry],
    *,
    schema: pa.Schema,
    primary_key: Sequence[str] | None = None,
    contract_version: str | None = None,
    source: Mapping[str, Any] | None = None,
    quality: Mapping[str, Any] | None = None,
) -> DatasetManifest:
    ordered = sorted(
        list(entries), key=lambda item: (item.start_date, item.end_date, item.path)
    )
    if not ordered:
        raise QdpV2RepairError("qdp_v2_repair_resulting_shards_empty")
    starts = [item.start_date for item in ordered if item.start_date]
    ends = [item.end_date for item in ordered if item.end_date]
    return replace(
        context.manifest,
        primary_key=list(primary_key or context.manifest.primary_key),
        contract_version=contract_version or context.manifest.contract_version,
        start_date=min(starts) if starts else "",
        end_date=max(ends) if ends else "",
        row_count=sum(int(item.row_count) for item in ordered),
        shards=ordered,
        schema=_manifest_schema_from_arrow(schema),
        source=dict(source if source is not None else context.manifest.source),
        quality=dict(quality if quality is not None else context.manifest.quality),
    )


def _commit_manifest(context: ActiveDomain, manifest: DatasetManifest) -> None:
    write_dataset_manifest(context.root, manifest)
    current = read_dataset_manifest(context.manifest_path)
    if current.dataset_id != manifest.dataset_id or current.row_count != manifest.row_count:
        raise QdpV2RepairError("qdp_v2_repair_manifest_commit_validation_failed")
    for entry in current.shards:
        path = resolve_manifest_path(entry.path, root=context.root)
        if not path.is_file():
            raise QdpV2RepairError(f"qdp_v2_repair_committed_shard_missing:{path}")


def _install_prepared_files(
    context: ActiveDomain,
    prepared: Sequence[Path],
    *,
    kind: str,
) -> list[Path]:
    installed: list[Path] = []
    for number, source in enumerate(prepared):
        target = _new_shard_path(context, kind, number=number)
        temporary = target.with_suffix(target.suffix + ".tmp")
        target.parent.mkdir(parents=True, exist_ok=True)
        try:
            shutil.copy2(source, temporary)
            _validate_parquet(temporary)
            temporary.replace(target)
            installed.append(target)
        except Exception:
            temporary.unlink(missing_ok=True)
            _remove_files(installed)
            raise
    return installed


def _new_shard_path(context: ActiveDomain, kind: str, *, number: int = 0) -> Path:
    directory = context.manifest_path.parent / "shards"
    token = _time_token()
    candidate = directory / f"repair_{kind}_{token}_{number:04d}.parquet"
    suffix = 1
    while candidate.exists():
        candidate = directory / f"repair_{kind}_{token}_{number:04d}_{suffix}.parquet"
        suffix += 1
    return candidate


def _resolve_active_shard(context: ActiveDomain, value: str | Path) -> Path:
    path = Path(value)
    if not path.is_absolute():
        path = resolve_manifest_path(path, root=context.root)
    resolved = path.resolve()
    known = {item.resolve() for item in context.shard_paths}
    if resolved not in known:
        raise QdpV2RepairError(f"qdp_v2_repair_shard_not_active:{resolved}")
    return resolved


def _resolve_prepared_path(
    value: str | Path,
    workspace_root: str | Path | None,
) -> Path:
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = qdp_paths(workspace_root).workspace_root / path
    resolved = path.resolve()
    _validate_parquet(resolved)
    return resolved


def _ensure_unique_paths(paths: Sequence[Path]) -> None:
    resolved = [item.resolve() for item in paths]
    if len(resolved) != len(set(resolved)):
        raise QdpV2RepairError("qdp_v2_repair_duplicate_prepared_path")


def _remove_old_shards(
    context: ActiveDomain,
    paths: Sequence[Path],
) -> tuple[list[str], list[str]]:
    current = read_dataset_manifest(context.manifest_path)
    referenced = {
        resolve_manifest_path(item.path, root=context.root).resolve()
        for item in current.shards
    }
    dataset_root = context.manifest_path.parent.resolve()
    deleted: list[str] = []
    retained: list[str] = []
    for path in paths:
        resolved = path.resolve()
        if resolved in referenced:
            retained.append(str(resolved))
            continue
        try:
            resolved.relative_to(dataset_root)
        except ValueError:
            # Composite manifests may reference a shard owned by an older
            # dataset.  Removing that reference must not delete the owner's file.
            retained.append(str(resolved))
            continue
        try:
            resolved.unlink(missing_ok=True)
            deleted.append(str(resolved))
        except OSError:
            retained.append(str(resolved))
    return deleted, retained


def _remove_files(paths: Sequence[Path]) -> None:
    for path in paths:
        path.unlink(missing_ok=True)


def _date_column(columns: Sequence[str]) -> str:
    return next((item for item in _DATE_COLUMNS if item in columns), "")


def _parquet_date_range(path: Path, column: str) -> tuple[str, str]:
    if not column:
        return "", ""
    values = pq.read_table(path, columns=[column])[column]
    if len(values) == 0:
        return "", ""
    return str(pc.min(values).as_py()), str(pc.max(values).as_py())


def _parquet_list_sql(paths: Sequence[Path]) -> str:
    return "[" + ",".join(_sql_literal(str(item)) for item in paths) + "]"


def _quote_identifier(value: str) -> str:
    return '"' + str(value).replace('"', '""') + '"'


def _sql_literal(value: str) -> str:
    return "'" + str(value).replace("'", "''") + "'"


def _utc_now() -> str:
    return datetime.now(UTC).replace(microsecond=0).isoformat()


def _time_token() -> str:
    return datetime.now(UTC).strftime("%Y%m%dT%H%M%S_%f")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Repair active QDP data directly.")
    parser.add_argument("--workspace-root", type=Path)
    subparsers = parser.add_subparsers(dest="repair_command", required=True)

    patch = subparsers.add_parser("patch", help="Patch explicitly named cells.")
    patch.add_argument("--request", type=Path, required=True)
    patch.add_argument("--reason", default="patch cells")
    patch.add_argument("--apply", action="store_true")

    mutate = subparsers.add_parser("mutate", help="Replace, remove, or append shards.")
    mutate.add_argument("--domain", required=True)
    mutate.add_argument("--replace", action="append", default=[])
    mutate.add_argument("--remove", action="append", default=[])
    mutate.add_argument("--append", action="append", default=[])
    mutate.add_argument("--reason", default="mutate active shards")
    mutate.add_argument("--apply", action="store_true")

    replace_table = subparsers.add_parser("replace-table", help="Replace an active table.")
    replace_table.add_argument("--domain", required=True)
    replace_table.add_argument("--prepared", action="append", required=True)
    replace_table.add_argument("--primary-key", action="append", default=[])
    replace_table.add_argument("--contract-version", default="")
    replace_table.add_argument("--reason", default="replace active table")
    replace_table.add_argument("--apply", action="store_true")
    return parser


def _parse_replacements(values: Sequence[str]) -> list[tuple[str, str]]:
    pairs: list[tuple[str, str]] = []
    for value in values:
        if "=" not in value:
            raise ValueError(f"qdp_v2_repair_replace_invalid:{value}")
        old, new = value.split("=", 1)
        pairs.append((old, new))
    return pairs


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    workspace = str(args.workspace_root or "") or None
    if args.repair_command == "patch":
        result = patch_active_cells(
            args.request,
            reason=args.reason,
            workspace_root=workspace,
            apply=bool(args.apply),
        )
    elif args.repair_command == "mutate":
        if not args.apply:
            result = {
                "status": "would_mutate",
                "dry_run": True,
                "domain": args.domain,
                "replacements": _parse_replacements(args.replace),
                "removals": list(args.remove),
                "appends": list(args.append),
            }
        else:
            result = mutate_active_shards_from_parquet(
                args.domain,
                replacements=_parse_replacements(args.replace),
                removals=args.remove,
                appends=args.append,
                reason=args.reason,
                workspace_root=workspace,
            )
    else:
        if not args.apply:
            schema = _common_schema(
                [_resolve_prepared_path(item, workspace) for item in args.prepared]
            )
            result = {
                "status": "would_replace_table",
                "dry_run": True,
                "domain": args.domain,
                "prepared": list(args.prepared),
                "schema": _manifest_schema_from_arrow(schema),
            }
        else:
            result = replace_active_table_from_parquet(
                args.domain,
                args.prepared,
                reason=args.reason,
                workspace_root=workspace,
                primary_key=args.primary_key or None,
                contract_version=args.contract_version,
            )
    print(json.dumps(json_safe(result), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
