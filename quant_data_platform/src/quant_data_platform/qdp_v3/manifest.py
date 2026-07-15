from __future__ import annotations

import hashlib
import json
import os
import time
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

from quant_data_platform.core.json_io import json_safe, read_json
from quant_data_platform.qdp_v3.constants import (
    LEGACY_MANIFEST_VERSIONS,
    MANIFEST_VERSION,
    QDP_V3_CONTRACT_VERSION,
    QUALITY_TIERS,
    SCHEMA_VERSION,
    SECURITY_IDENTITY_CONTRACT,
    TIMEZONE,
)
from quant_data_platform.qdp_v3.paths import qdp_v3_paths


_ATOMIC_REPLACE_ATTEMPTS = 20
_ATOMIC_REPLACE_DELAY_SECONDS = 0.02


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def canonical_json_bytes(payload: Mapping[str, Any]) -> bytes:
    return json.dumps(
        json_safe(dict(payload)),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def stable_hash(payload: Mapping[str, Any], *, length: int = 64) -> str:
    return hashlib.sha256(canonical_json_bytes(payload)).hexdigest()[: int(length)]


def sha256_file(path: str | Path, *, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while True:
            chunk = handle.read(int(chunk_size))
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def atomic_write_json(path: str | Path, payload: Mapping[str, Any]) -> Path:
    resolved = Path(path)
    resolved.parent.mkdir(parents=True, exist_ok=True)
    temp = resolved.with_name(f".{resolved.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp")
    text = json.dumps(json_safe(dict(payload)), ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    try:
        temp.write_text(text, encoding="utf-8")
        for attempt in range(_ATOMIC_REPLACE_ATTEMPTS):
            try:
                temp.replace(resolved)
                return resolved
            except PermissionError:
                if attempt + 1 >= _ATOMIC_REPLACE_ATTEMPTS:
                    raise
                time.sleep(_ATOMIC_REPLACE_DELAY_SECONDS * (attempt + 1))
    finally:
        temp.unlink(missing_ok=True)


def manifest_sha256(path: str | Path) -> str:
    return sha256_file(path)


def path_for_manifest(path: str | Path, *, root: str | Path) -> str:
    resolved = Path(path).resolve()
    root_path = Path(root).resolve()
    try:
        return resolved.relative_to(root_path).as_posix()
    except ValueError:
        return str(resolved)


def resolve_manifest_path(path: str | Path, *, root: str | Path) -> Path:
    candidate = Path(path)
    return candidate if candidate.is_absolute() else (Path(root) / candidate).resolve()


@dataclass(frozen=True)
class DatasetInputRef:
    domain: str
    dataset_id: str
    manifest_sha256: str

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any]) -> "DatasetInputRef":
        return cls(
            domain=str(payload.get("domain", "") or ""),
            dataset_id=str(payload.get("dataset_id", "") or ""),
            manifest_sha256=str(payload.get("manifest_sha256", "") or ""),
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ProviderEvidence:
    provider: str
    endpoint: str
    package_version: str
    wheel_sha256: str = ""
    request_range: dict[str, Any] = field(default_factory=dict)
    collected_at: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any]) -> "ProviderEvidence":
        return cls(
            provider=str(payload.get("provider", "") or ""),
            endpoint=str(payload.get("endpoint", "") or ""),
            package_version=str(payload.get("package_version", "") or ""),
            wheel_sha256=str(payload.get("wheel_sha256", "") or ""),
            request_range=dict(payload.get("request_range", {}) or {}),
            collected_at=str(payload.get("collected_at", "") or ""),
            metadata=dict(payload.get("metadata", {}) or {}),
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ShardManifestV3:
    path: str
    sha256: str
    row_count: int
    start_date: str = ""
    end_date: str = ""
    partition: dict[str, Any] = field(default_factory=dict)
    file_size: int = 0
    schema_hash: str = ""
    status: str = "stored"

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any]) -> "ShardManifestV3":
        return cls(
            path=str(payload.get("path", "") or ""),
            sha256=str(payload.get("sha256", "") or ""),
            row_count=int(payload.get("row_count", 0) or 0),
            start_date=str(payload.get("start_date", "") or ""),
            end_date=str(payload.get("end_date", "") or ""),
            partition=dict(payload.get("partition", {}) or {}),
            file_size=int(payload.get("file_size", 0) or 0),
            schema_hash=str(payload.get("schema_hash", "") or ""),
            status=str(payload.get("status", "") or "stored"),
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class DatasetManifestV3:
    dataset_id: str
    domain: str
    layer: str
    frequency: str
    primary_key: list[str]
    quality_tier: str
    start_date: str
    end_date: str
    row_count: int
    schema: list[dict[str, str]]
    schema_hash: str
    shards: list[ShardManifestV3]
    inputs: list[DatasetInputRef] = field(default_factory=list)
    provider_evidence: list[ProviderEvidence] = field(default_factory=list)
    raw_content_hashes: list[str] = field(default_factory=list)
    build: dict[str, Any] = field(default_factory=dict)
    quality_report: dict[str, Any] = field(default_factory=dict)
    blockers: list[dict[str, Any]] = field(default_factory=list)
    quarantine: list[dict[str, Any]] = field(default_factory=list)
    coverage: dict[str, Any] = field(default_factory=dict)
    units: dict[str, str] = field(default_factory=dict)
    created_at: str = field(default_factory=utc_now)
    manifest_version: int = MANIFEST_VERSION
    schema_version: str = SCHEMA_VERSION
    contract_version: str = QDP_V3_CONTRACT_VERSION
    security_identity_contract: str = SECURITY_IDENTITY_CONTRACT
    timezone: str = TIMEZONE

    def __post_init__(self) -> None:
        if int(self.manifest_version) not in {MANIFEST_VERSION, *LEGACY_MANIFEST_VERSIONS}:
            raise ValueError(f"qdp_v3_manifest_version_mismatch:{self.manifest_version}")
        if self.quality_tier not in QUALITY_TIERS:
            raise ValueError(f"invalid_quality_tier:{self.quality_tier}")
        if not self.dataset_id or not self.domain:
            raise ValueError("dataset_id_and_domain_are_required")
        if not self.primary_key:
            raise ValueError(f"primary_key_is_required:{self.domain}")
        if int(self.row_count) != sum(int(item.row_count) for item in self.shards):
            raise ValueError(
                f"manifest_row_count_mismatch:{self.dataset_id}:"
                f"manifest={self.row_count} shards={sum(int(item.row_count) for item in self.shards)}"
            )

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any]) -> "DatasetManifestV3":
        return cls(
            dataset_id=str(payload.get("dataset_id", "") or ""),
            domain=str(payload.get("domain", "") or ""),
            layer=str(payload.get("layer", "") or ""),
            frequency=str(payload.get("frequency", "") or ""),
            primary_key=[str(item) for item in list(payload.get("primary_key", []) or [])],
            quality_tier=str(payload.get("quality_tier", "") or ""),
            start_date=str(payload.get("start_date", "") or ""),
            end_date=str(payload.get("end_date", "") or ""),
            row_count=int(payload.get("row_count", 0) or 0),
            schema=[{str(k): str(v) for k, v in dict(item).items()} for item in list(payload.get("schema", []) or []) if isinstance(item, Mapping)],
            schema_hash=str(payload.get("schema_hash", "") or ""),
            shards=[ShardManifestV3.from_mapping(item) for item in list(payload.get("shards", []) or []) if isinstance(item, Mapping)],
            inputs=[DatasetInputRef.from_mapping(item) for item in list(payload.get("inputs", []) or []) if isinstance(item, Mapping)],
            provider_evidence=[ProviderEvidence.from_mapping(item) for item in list(payload.get("provider_evidence", []) or []) if isinstance(item, Mapping)],
            raw_content_hashes=[str(item) for item in list(payload.get("raw_content_hashes", []) or [])],
            build=dict(payload.get("build", {}) or {}),
            quality_report=dict(payload.get("quality_report", {}) or {}),
            blockers=[dict(item) for item in list(payload.get("blockers", []) or []) if isinstance(item, Mapping)],
            quarantine=[dict(item) for item in list(payload.get("quarantine", []) or []) if isinstance(item, Mapping)],
            coverage=dict(payload.get("coverage", {}) or {}),
            units={str(k): str(v) for k, v in dict(payload.get("units", {}) or {}).items()},
            created_at=str(payload.get("created_at", "") or utc_now()),
            manifest_version=int(payload.get("manifest_version", 0) or 0),
            schema_version=str(payload.get("schema_version", "") or ""),
            contract_version=str(payload.get("contract_version", "") or ""),
            security_identity_contract=str(payload.get("security_identity_contract", "") or ""),
            timezone=str(payload.get("timezone", "") or ""),
        )

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["shards"] = [item.to_dict() for item in self.shards]
        payload["inputs"] = [item.to_dict() for item in self.inputs]
        payload["provider_evidence"] = [item.to_dict() for item in self.provider_evidence]
        return payload


def dataset_id_for(domain: str, content_seed: Mapping[str, Any]) -> str:
    return f"{str(domain)}__{stable_hash({'domain': str(domain), 'content': dict(content_seed)}, length=24)}"


def schema_from_frame(frame: Any) -> list[dict[str, str]]:
    return [{"name": str(name), "dtype": str(dtype)} for name, dtype in zip(frame.columns, frame.dtypes)]


def schema_hash_for(schema: Sequence[Mapping[str, Any]]) -> str:
    return stable_hash({"schema": [dict(item) for item in schema]}, length=32)


def dataset_manifest_path(root: str | Path, domain: str, dataset_id: str) -> Path:
    return Path(root) / "datasets" / str(domain) / str(dataset_id) / "dataset.json"


def write_dataset_manifest(root: str | Path, manifest: DatasetManifestV3) -> Path:
    path = dataset_manifest_path(root, manifest.domain, manifest.dataset_id)
    payload = manifest.to_dict()
    if path.exists():
        existing = read_json(path)
        if canonical_json_bytes(existing) != canonical_json_bytes(payload):
            raise RuntimeError(f"immutable_dataset_manifest_conflict:{manifest.dataset_id}")
        return path
    return atomic_write_json(path, payload)


def read_dataset_manifest(path: str | Path) -> DatasetManifestV3:
    payload = read_json(path)
    if not payload:
        raise FileNotFoundError(str(path))
    return DatasetManifestV3.from_mapping(payload)


def dataset_manifest_for_id(root: str | Path, dataset_id: str, domain_hint: str = "") -> Path | None:
    root_path = Path(root)
    if domain_hint:
        candidate = dataset_manifest_path(root_path, domain_hint, dataset_id)
        if candidate.exists():
            return candidate
    for candidate in sorted((root_path / "datasets").glob(f"*/{dataset_id}/dataset.json")):
        return candidate
    return None


def iter_dataset_manifests(root: str | Path) -> list[Path]:
    datasets = Path(root) / "datasets"
    return sorted(datasets.glob("*/*/dataset.json")) if datasets.exists() else []


def active_manifest_sha256(workspace_root: str | Path | None = None) -> str:
    path = qdp_v3_paths(workspace_root).active_manifest
    return sha256_file(path) if path.exists() else "none"
