from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import pyarrow as pa

from quant_data_platform.core.json_io import json_safe, read_json
from quant_data_platform.core.paths import qdp_paths


ACTIVE_MANIFEST_VERSION = 2
EXPECTED_BAR_TIMES = tuple(
    f"{hour:02d}{minute:02d}00000"
    for hour, minute in (
        *[(9, minute) for minute in range(35, 60, 5)],
        *[(10, minute) for minute in range(0, 60, 5)],
        *[(11, minute) for minute in range(0, 31, 5)],
        *[(13, minute) for minute in range(5, 60, 5)],
        *[(14, minute) for minute in range(0, 60, 5)],
        (15, 0),
    )
)


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def qdp_v2_root(workspace_root: str | Path | None = None) -> Path:
    return qdp_paths(workspace_root).data_dir / "qdp_v2"


def active_manifest_path(root: str | Path | None = None) -> Path:
    resolved = Path(root) if root is not None else qdp_v2_root()
    return resolved / "active" / "active.json"


def dataset_manifest_path(root: str | Path, domain: str, dataset_id: str) -> Path:
    return Path(root) / "datasets" / str(domain) / str(dataset_id) / "dataset.json"


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

    return [{"name": field.name, "type": sql_type(field.type)} for field in schema]


_SCHEMA_TYPE_ALIASES: dict[str, pa.DataType] = {
    "BOOLEAN": pa.bool_(),
    "BOOL": pa.bool_(),
    "BIGINT": pa.int64(),
    "INT64": pa.int64(),
    "INTEGER": pa.int64(),
    "INT32": pa.int32(),
    "DOUBLE": pa.float64(),
    "FLOAT64": pa.float64(),
    "FLOAT": pa.float64(),
    "TIMESTAMP": pa.timestamp("ns"),
    "DATETIME64[NS]": pa.timestamp("ns"),
    "DATE32[DAY]": pa.date32(),
    "DATE": pa.date32(),
    "VARCHAR": pa.string(),
    "STRING": pa.string(),
    "OBJECT": pa.string(),
}


def schema_field_is_typed(field: Mapping[str, Any]) -> bool:
    return bool(str(field.get("name", "") or "")) and bool(
        str(field.get("type", "") or "").strip()
    )


def canonical_manifest_schema(
    schema: Sequence[Mapping[str, Any]],
) -> list[dict[str, str]] | None:
    fields = list(schema)
    if not fields or not all(schema_field_is_typed(item) for item in fields):
        return None
    try:
        return _manifest_schema_from_arrow(arrow_schema_from_manifest(fields))
    except ValueError:
        return None


def arrow_schema_from_manifest(schema: Sequence[Mapping[str, Any]]) -> pa.Schema:
    fields: list[pa.Field] = []
    for item in schema:
        name = str(item.get("name", "") or "")
        declared = str(item.get("type", "") or "").strip()
        if not name or not declared:
            raise ValueError("qdp_v2_manifest_schema_field_invalid")
        data_type = _SCHEMA_TYPE_ALIASES.get(declared.upper())
        if data_type is None:
            try:
                data_type = pa.type_for_alias(declared.lower())
            except ValueError as exc:
                raise ValueError(
                    f"qdp_v2_manifest_schema_type_unsupported:{name}:{declared}"
                ) from exc
        fields.append(pa.field(name, data_type))
    if not fields:
        raise ValueError("qdp_v2_manifest_schema_missing")
    return pa.schema(fields)


def atomic_write_json(path: str | Path, payload: Mapping[str, Any]) -> Path:
    resolved = Path(path)
    resolved.parent.mkdir(parents=True, exist_ok=True)
    temporary = resolved.with_name(f".{resolved.name}.{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(json_safe(dict(payload)), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(resolved)
    return resolved


def path_for_manifest(path: str | Path, *, root: str | Path) -> str:
    resolved = Path(path).resolve()
    root_path = Path(root).resolve()
    try:
        return resolved.relative_to(root_path).as_posix()
    except ValueError:
        return str(resolved)


def resolve_manifest_path(path: str | Path, *, root: str | Path) -> Path:
    candidate = Path(path)
    if candidate.is_absolute():
        return candidate
    return (Path(root) / candidate).resolve()


@dataclass(frozen=True)
class ShardManifestEntry:
    path: str
    row_count: int = 0
    start_date: str = ""
    end_date: str = ""
    status: str = "stored"
    file_size: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any]) -> "ShardManifestEntry":
        return cls(
            path=str(payload.get("path", "") or payload.get("file_path", "") or ""),
            row_count=int(payload.get("row_count", 0) or 0),
            start_date=str(payload.get("start_date", "") or ""),
            end_date=str(payload.get("end_date", "") or ""),
            status=str(payload.get("status", "") or "stored"),
            file_size=int(payload.get("file_size", 0) or payload.get("bytes", 0) or 0),
            metadata=dict(payload.get("metadata", {}) or {}),
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class DatasetManifest:
    dataset_id: str
    domain: str
    layer: str
    frequency: str
    contract_version: str
    primary_key: list[str]
    start_date: str
    end_date: str
    row_count: int
    shards: list[ShardManifestEntry]
    source: dict[str, Any]
    quality: dict[str, Any]
    created_at: str = field(default_factory=utc_now)
    schema: list[dict[str, str]] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any]) -> "DatasetManifest":
        return cls(
            dataset_id=str(payload.get("dataset_id", "") or ""),
            domain=str(payload.get("domain", "") or ""),
            layer=str(payload.get("layer", "") or ""),
            frequency=str(payload.get("frequency", "") or ""),
            contract_version=str(payload.get("contract_version", "") or ""),
            primary_key=[str(item) for item in list(payload.get("primary_key", []) or [])],
            start_date=str(payload.get("start_date", "") or ""),
            end_date=str(payload.get("end_date", "") or ""),
            row_count=int(payload.get("row_count", 0) or 0),
            shards=[
                ShardManifestEntry.from_mapping(item)
                for item in list(payload.get("shards", []) or [])
                if isinstance(item, Mapping)
            ],
            source=dict(payload.get("source", {}) or {}),
            quality=dict(payload.get("quality", {}) or {}),
            created_at=str(payload.get("created_at", "") or utc_now()),
            schema=[
                {str(key): str(value) for key, value in dict(item).items()}
                for item in list(payload.get("schema", []) or [])
                if isinstance(item, Mapping)
            ],
            notes=[str(item) for item in list(payload.get("notes", []) or [])],
        )

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["shards"] = [item.to_dict() for item in self.shards]
        return payload


def read_dataset_manifest(path: str | Path) -> DatasetManifest:
    return DatasetManifest.from_mapping(read_json(path))


def write_dataset_manifest(root: str | Path, manifest: DatasetManifest) -> Path:
    path = dataset_manifest_path(root, manifest.domain, manifest.dataset_id)
    return atomic_write_json(path, manifest.to_dict())


def read_active_manifest(root: str | Path) -> dict[str, Any]:
    return read_json(Path(root) / "active" / "active.json")


def write_active_manifest(root: str | Path, payload: Mapping[str, Any]) -> Path:
    active = dict(payload)
    active.setdefault("version", ACTIVE_MANIFEST_VERSION)
    active.setdefault("updated_at", utc_now())
    return atomic_write_json(Path(root) / "active" / "active.json", active)


def dataset_manifest_for_id(
    root: str | Path, dataset_id: str, domain_hint: str = ""
) -> Path | None:
    resolved = Path(root)
    if domain_hint:
        candidate = dataset_manifest_path(resolved, domain_hint, dataset_id)
        if candidate.exists():
            return candidate
    for candidate in sorted((resolved / "datasets").glob(f"*/{dataset_id}/dataset.json")):
        return candidate
    return None


def iter_dataset_manifests(root: str | Path) -> list[Path]:
    datasets_root = Path(root) / "datasets"
    if not datasets_root.exists():
        return []
    return sorted(datasets_root.glob("*/*/dataset.json"))
