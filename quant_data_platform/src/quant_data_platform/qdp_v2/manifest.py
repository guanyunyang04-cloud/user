from __future__ import annotations

import hashlib
import json
import os
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

from quant_data_platform.core.json_io import json_safe, read_json
from quant_data_platform.core.paths import qdp_paths


ACTIVE_MANIFEST_VERSION = 2


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def qdp_v2_root(workspace_root: str | Path | None = None) -> Path:
    return qdp_paths(workspace_root).data_dir / "qdp_v2"


def active_manifest_path(root: str | Path | None = None) -> Path:
    resolved = Path(root) if root is not None else qdp_v2_root()
    return resolved / "active" / "active.json"


def dataset_manifest_path(root: str | Path, domain: str, dataset_id: str) -> Path:
    return Path(root) / "datasets" / str(domain) / str(dataset_id) / "dataset.json"


def stable_hash(payload: Mapping[str, Any], *, length: int = 24) -> str:
    text = json.dumps(json_safe(dict(payload)), ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[: int(length)]


def schema_hash(schema: list[Mapping[str, Any]] | Mapping[str, Any] | None) -> str:
    if not schema:
        return ""
    payload: Any = schema
    if isinstance(schema, list):
        payload = [{str(k): str(v) for k, v in dict(item).items()} for item in schema]
    return stable_hash({"schema": payload}, length=32)


def atomic_write_json(path: str | Path, payload: Mapping[str, Any]) -> Path:
    resolved = Path(path)
    resolved.parent.mkdir(parents=True, exist_ok=True)
    tmp = resolved.with_name(f".{resolved.name}.{os.getpid()}.tmp")
    tmp.write_text(json.dumps(json_safe(dict(payload)), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    tmp.replace(resolved)
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
    schema_hash: str = ""
    source_path: str = ""
    content_key: str = ""
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
            schema_hash=str(payload.get("schema_hash", "") or ""),
            source_path=str(payload.get("source_path", "") or ""),
            content_key=str(payload.get("content_key", "") or ""),
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
    schema_hash: str
    shards: list[ShardManifestEntry]
    source: dict[str, Any]
    quality: dict[str, Any]
    created_at: str = field(default_factory=utc_now)
    schema: list[dict[str, str]] = field(default_factory=list)
    legacy: dict[str, Any] = field(default_factory=dict)
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
            schema_hash=str(payload.get("schema_hash", "") or ""),
            shards=[ShardManifestEntry.from_mapping(item) for item in list(payload.get("shards", []) or []) if isinstance(item, Mapping)],
            source=dict(payload.get("source", {}) or {}),
            quality=dict(payload.get("quality", {}) or {}),
            created_at=str(payload.get("created_at", "") or utc_now()),
            schema=[{str(k): str(v) for k, v in dict(item).items()} for item in list(payload.get("schema", []) or []) if isinstance(item, Mapping)],
            legacy=dict(payload.get("legacy", {}) or {}),
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


def dataset_manifest_for_id(root: str | Path, dataset_id: str, domain_hint: str = "") -> Path | None:
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
