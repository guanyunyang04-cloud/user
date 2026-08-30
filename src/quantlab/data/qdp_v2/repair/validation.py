"""Repair: validation responsibilities."""

from __future__ import annotations

import shutil
from collections.abc import Mapping, Sequence
from dataclasses import replace
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq

from quantlab.core.io import read_optional_json
from quantlab.data.core.paths import qdp_paths
from quantlab.data.qdp_v2.active import ActiveDomain
from quantlab.data.qdp_v2.manifest import (
    DatasetManifest,
    ShardManifestEntry,
    _manifest_schema_from_arrow,
    path_for_manifest,
    read_dataset_manifest,
    resolve_manifest_path,
    write_dataset_manifest,
)

from .common import (
    _date_column,
    _parquet_date_range,
    _parquet_list_sql,
    _quote_identifier,
    _time_token,
    _utc_now,
)
from .errors import QdpV2RepairError


def _read_patch_request(
    path: str | Path,
    workspace_root: str | Path | None,
) -> dict[str, Any]:
    candidate = Path(path).expanduser()
    if not candidate.is_absolute():
        candidate = qdp_paths(workspace_root).workspace_root / candidate
    payload = read_optional_json(candidate.resolve())
    allowed = {"schema_version", "domain", "changes"}
    unknown = set(payload).difference(allowed)
    if unknown:
        raise QdpV2RepairError(f"qdp_v2_repair_patch_request_keys_invalid:{sorted(unknown)}")
    if not str(payload.get("domain", "") or "").strip():
        raise QdpV2RepairError("qdp_v2_repair_patch_domain_required")
    changes = payload.get("changes")
    if not isinstance(changes, list) or not changes or not all(isinstance(item, Mapping) for item in changes):
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
    if expected_schema is not None and not parquet.schema_arrow.equals(expected_schema, check_metadata=False):
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

    condition = " AND ".join(f"n.{_quote_identifier(item)} = o.{_quote_identifier(item)}" for item in keys)
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
    ordered = sorted(list(entries), key=lambda item: (item.start_date, item.end_date, item.path))
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
    referenced = {resolve_manifest_path(item.path, root=context.root).resolve() for item in current.shards}
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
