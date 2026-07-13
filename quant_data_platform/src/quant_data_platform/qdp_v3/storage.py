from __future__ import annotations

import hashlib
import json
import os
import re
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping

import pandas as pd

from quant_data_platform.core.json_io import read_json
from quant_data_platform.qdp_v3.manifest import atomic_write_json, path_for_manifest, sha256_file, stable_hash, utc_now
from quant_data_platform.qdp_v3.paths import ensure_qdp_v3_layout, qdp_v3_paths


_SAFE_PARTITION = re.compile(r"[^A-Za-z0-9_.=-]+")


@dataclass(frozen=True)
class RawPartitionRef:
    raw_domain: str
    partition_field: str
    partition_value: str
    content_sha256: str
    payload_path: Path
    receipt_path: Path
    row_count: int
    quality_tier: str
    revision_of: str = ""
    quality_assessment_path: Path | None = None
    quality_assessment_sha256: str = ""

    def to_dict(self, *, root: str | Path | None = None) -> dict[str, Any]:
        payload_path = str(self.payload_path)
        receipt_path = str(self.receipt_path)
        quality_assessment_path = str(self.quality_assessment_path) if self.quality_assessment_path else ""
        if root is not None:
            payload_path = path_for_manifest(self.payload_path, root=root)
            receipt_path = path_for_manifest(self.receipt_path, root=root)
            if self.quality_assessment_path:
                quality_assessment_path = path_for_manifest(self.quality_assessment_path, root=root)
        return {
            "raw_domain": self.raw_domain,
            "partition_field": self.partition_field,
            "partition_value": self.partition_value,
            "content_sha256": self.content_sha256,
            "payload_path": payload_path,
            "receipt_path": receipt_path,
            "row_count": int(self.row_count),
            "quality_tier": self.quality_tier,
            "revision_of": self.revision_of,
            "quality_assessment_path": quality_assessment_path,
            "quality_assessment_sha256": self.quality_assessment_sha256,
        }


def _safe_partition_value(value: str) -> str:
    normalized = _SAFE_PARTITION.sub("_", str(value or "").strip())
    if not normalized or normalized in {".", ".."}:
        raise ValueError(f"invalid_partition_value:{value!r}")
    return normalized


def frame_content_sha256(frame: pd.DataFrame) -> str:
    """Hash decompressed tabular content, schema, row order, and nulls."""

    data = frame.copy() if isinstance(frame, pd.DataFrame) else pd.DataFrame()
    digest = hashlib.sha256()
    digest.update(
        json.dumps(
            {
                "columns": [str(item) for item in data.columns],
                "dtypes": [str(item) for item in data.dtypes],
                "row_count": int(len(data)),
            },
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    )
    if data.empty:
        return digest.hexdigest()
    chunk_size = 50_000
    for start in range(0, len(data), chunk_size):
        chunk = data.iloc[start : start + chunk_size].copy()
        for column in chunk.columns:
            values = chunk[column]
            if pd.api.types.is_datetime64_any_dtype(values):
                values = pd.to_datetime(values, errors="coerce").dt.strftime("%Y-%m-%dT%H:%M:%S.%f")
            chunk[column] = values.astype("string").fillna("<QDP_NULL>")
        hashed = pd.util.hash_pandas_object(chunk, index=False, categorize=False)
        digest.update(hashed.to_numpy(dtype="uint64", copy=False).tobytes())
    return digest.hexdigest()


def atomic_write_parquet(path: str | Path, frame: pd.DataFrame, *, compression: str = "zstd") -> Path:
    resolved = Path(path)
    resolved.parent.mkdir(parents=True, exist_ok=True)
    temp = resolved.with_name(f".{resolved.name}.{os.getpid()}.tmp")
    try:
        frame.to_parquet(temp, index=False, engine="pyarrow", compression=compression)
        temp.replace(resolved)
    finally:
        if temp.exists():
            temp.unlink()
    return resolved


_QUALITY_ASSESSMENT_KEYS = {
    "quality_tier",
    "quality_report",
    "quality_note",
    "cross_check_content_sha256",
    "day_results",
    "quarantined_day_count",
    "stratum",
    "stratum_escalated",
    "inputs",
    "provider_errors",
    "error_report",
}


def _quality_assessment_payload(receipt: Mapping[str, Any]) -> dict[str, Any]:
    source = dict(receipt)
    return {key: source[key] for key in sorted(_QUALITY_ASSESSMENT_KEYS) if key in source}


def _write_quality_assessment(version_dir: Path, receipt: Mapping[str, Any]) -> tuple[Path | None, str]:
    quality = _quality_assessment_payload(receipt)
    if not quality:
        return None, ""
    assessment_id = stable_hash({"quality_assessment": quality})
    assessment_path = version_dir / "quality_assessments" / f"{assessment_id}.json"
    if not assessment_path.exists():
        atomic_write_json(
            assessment_path,
            {
                "raw_quality_assessment_version": 1,
                "assessment_id": assessment_id,
                "assessment": quality,
                "assessed_at": utc_now(),
            },
        )
    assessment_file_sha = sha256_file(assessment_path)
    latest_path = version_dir / "latest_quality.json"
    latest = read_json(latest_path)
    if latest.get("assessment_id") != assessment_id or latest.get("file_sha256") != assessment_file_sha:
        atomic_write_json(
            latest_path,
            {
                "raw_quality_assessment_version": 1,
                "assessment_id": assessment_id,
                "path": f"quality_assessments/{assessment_path.name}",
                "file_sha256": assessment_file_sha,
                "updated_at": utc_now(),
            },
        )
    return assessment_path, assessment_file_sha


def _latest_quality_assessment(receipt_path: Path) -> tuple[dict[str, Any], Path | None, str]:
    version_dir = receipt_path.parent
    latest = read_json(version_dir / "latest_quality.json")
    assessment_id = str(latest.get("assessment_id", "") or "")
    if not assessment_id:
        return {}, None, ""
    assessment_path = version_dir / "quality_assessments" / f"{assessment_id}.json"
    if not assessment_path.exists():
        raise RuntimeError(f"raw_quality_assessment_missing:{assessment_path}")
    expected_sha = str(latest.get("file_sha256", "") or "")
    actual_sha = sha256_file(assessment_path)
    if not expected_sha or actual_sha != expected_sha:
        raise RuntimeError(f"raw_quality_assessment_hash_mismatch:{assessment_path}")
    payload = read_json(assessment_path)
    if str(payload.get("assessment_id", "") or "") != assessment_id:
        raise RuntimeError(f"raw_quality_assessment_id_mismatch:{assessment_path}")
    return dict(payload.get("assessment", {}) or {}), assessment_path, actual_sha


def read_raw_receipt(ref_or_path: RawPartitionRef | str | Path) -> dict[str, Any]:
    receipt_path = ref_or_path.receipt_path if isinstance(ref_or_path, RawPartitionRef) else Path(ref_or_path)
    receipt = read_json(receipt_path)
    assessment, assessment_path, assessment_sha = _latest_quality_assessment(receipt_path)
    receipt.update(assessment)
    receipt["quality_assessment_path"] = str(assessment_path) if assessment_path else ""
    receipt["quality_assessment_sha256"] = assessment_sha
    return receipt


def write_raw_partition(
    *,
    raw_domain: str,
    partition_value: str,
    frame: pd.DataFrame,
    receipt: Mapping[str, Any],
    workspace_root: str | Path | None = None,
    partition_field: str = "query_date",
) -> tuple[RawPartitionRef, bool]:
    """Write an immutable, content-addressed raw partition.

    Returns ``(reference, created)``.  A provider revision creates a new
    version directory and records ``revision_of``; the previous bytes are
    never overwritten.
    """

    paths = ensure_qdp_v3_layout(workspace_root)
    value = _safe_partition_value(partition_value)
    field = _safe_partition_value(partition_field)
    partition_dir = paths.raw / str(raw_domain) / f"{field}={value}"
    partition_dir.mkdir(parents=True, exist_ok=True)
    staged_payload = partition_dir / f".payload.{os.getpid()}.{uuid.uuid4().hex}.parquet"
    try:
        atomic_write_parquet(staged_payload, frame)
        persisted_frame = pd.read_parquet(staged_payload, engine="pyarrow")
        content_sha = frame_content_sha256(persisted_frame)
    except BaseException:
        staged_payload.unlink(missing_ok=True)
        raise
    latest_path = partition_dir / "latest.json"
    previous = read_json(latest_path)
    previous_sha = str(previous.get("content_sha256", "") or "")
    version_dir = partition_dir / "versions" / content_sha
    payload_path = version_dir / "payload.parquet"
    receipt_path = version_dir / "receipt.json"
    quality_tier = str(dict(receipt).get("quality_tier", "") or "quarantined")
    revision_of = previous_sha if previous_sha and previous_sha != content_sha else ""
    created = False
    if not payload_path.exists() or not receipt_path.exists():
        if payload_path.exists() != receipt_path.exists():
            staged_payload.unlink(missing_ok=True)
            raise RuntimeError(f"incomplete_raw_version:{version_dir}")
        version_dir.mkdir(parents=True, exist_ok=True)
        staged_payload.replace(payload_path)
        payload = dict(receipt)
        payload.update(
            {
                "raw_manifest_version": 3,
                "raw_domain": str(raw_domain),
                "partition_field": field,
                "partition_value": str(partition_value),
                "content_sha256": content_sha,
                "parquet_sha256": sha256_file(payload_path),
                "row_count": int(len(persisted_frame)),
                "fields": [str(item) for item in persisted_frame.columns],
                "revision_of": revision_of,
                "stored_at": utc_now(),
            }
        )
        atomic_write_json(receipt_path, payload)
        created = True
    else:
        staged_payload.unlink(missing_ok=True)
    quality_assessment_path, quality_assessment_sha = _write_quality_assessment(version_dir, receipt)
    stored_receipt = read_json(receipt_path)
    if str(stored_receipt.get("content_sha256", "")) != content_sha:
        raise RuntimeError(f"raw_receipt_hash_mismatch:{receipt_path}")
    if not latest_path.exists() or previous_sha != content_sha:
        atomic_write_json(
            latest_path,
            {
                "raw_manifest_version": 3,
                "raw_domain": str(raw_domain),
                "partition_field": field,
                "partition_value": str(partition_value),
                "content_sha256": content_sha,
                "previous_content_sha256": revision_of,
                "version_path": path_for_manifest(version_dir, root=paths.root),
                "updated_at": utc_now(),
            },
        )
    return (
        RawPartitionRef(
            raw_domain=str(raw_domain),
            partition_field=field,
            partition_value=str(partition_value),
            content_sha256=content_sha,
            payload_path=payload_path,
            receipt_path=receipt_path,
            row_count=int(len(persisted_frame)),
            quality_tier=quality_tier,
            revision_of=revision_of,
            quality_assessment_path=quality_assessment_path,
            quality_assessment_sha256=quality_assessment_sha,
        ),
        created,
    )


def _ref_from_latest(latest_path: Path, *, root: Path) -> RawPartitionRef:
    latest = read_json(latest_path)
    if not latest:
        raise RuntimeError(f"raw_latest_invalid:{latest_path}")
    version_path = Path(str(latest.get("version_path", "") or ""))
    if not version_path.is_absolute():
        version_path = (root / version_path).resolve()
    receipt_path = version_path / "receipt.json"
    payload_path = version_path / "payload.parquet"
    receipt = read_raw_receipt(receipt_path)
    if not receipt or not payload_path.exists():
        raise RuntimeError(f"raw_partition_incomplete:{version_path}")
    expected = str(latest.get("content_sha256", "") or "")
    if str(receipt.get("content_sha256", "") or "") != expected:
        raise RuntimeError(f"raw_latest_receipt_mismatch:{latest_path}")
    return RawPartitionRef(
        raw_domain=str(latest.get("raw_domain", "") or ""),
        partition_field=str(latest.get("partition_field", "") or ""),
        partition_value=str(latest.get("partition_value", "") or ""),
        content_sha256=expected,
        payload_path=payload_path,
        receipt_path=receipt_path,
        row_count=int(receipt.get("row_count", 0) or 0),
        quality_tier=str(receipt.get("quality_tier", "") or "quarantined"),
        revision_of=str(receipt.get("revision_of", "") or ""),
        quality_assessment_path=Path(str(receipt.get("quality_assessment_path", ""))) if receipt.get("quality_assessment_path") else None,
        quality_assessment_sha256=str(receipt.get("quality_assessment_sha256", "") or ""),
    )


def iter_raw_partitions(
    raw_domain: str,
    *,
    workspace_root: str | Path | None = None,
    start_value: str = "",
    end_value: str = "",
) -> list[RawPartitionRef]:
    paths = qdp_v3_paths(workspace_root)
    domain_root = paths.raw / str(raw_domain)
    if not domain_root.exists():
        return []
    refs: list[RawPartitionRef] = []
    for latest_path in sorted(domain_root.glob("*/latest.json")):
        ref = _ref_from_latest(latest_path, root=paths.root)
        value = ref.partition_value
        if start_value and value < str(start_value):
            continue
        if end_value and value > str(end_value):
            continue
        refs.append(ref)
    return refs


def read_raw_partition(ref: RawPartitionRef, *, verify_hash: bool = True) -> pd.DataFrame:
    receipt = read_raw_receipt(ref)
    if verify_hash:
        expected_parquet = str(receipt.get("parquet_sha256", "") or "")
        if not expected_parquet or sha256_file(ref.payload_path) != expected_parquet:
            raise RuntimeError(f"raw_parquet_hash_mismatch:{ref.payload_path}")
    frame = pd.read_parquet(ref.payload_path, engine="pyarrow")
    if verify_hash and frame_content_sha256(frame) != ref.content_sha256:
        raise RuntimeError(f"raw_content_hash_mismatch:{ref.payload_path}")
    return frame


def raw_content_hashes(refs: Iterable[RawPartitionRef]) -> list[str]:
    return [str(ref.content_sha256) for ref in refs]
