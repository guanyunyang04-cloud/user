from __future__ import annotations

import hashlib
import json
import os
import re
import sqlite3
import uuid
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Iterable, Mapping

import pandas as pd
import pyarrow.parquet as pq

from quant_data_platform.core.json_io import read_json
from quant_data_platform.qdp_v3.manifest import atomic_write_json, path_for_manifest, sha256_file, stable_hash, utc_now
from quant_data_platform.qdp_v3.paths import ensure_qdp_v3_layout, qdp_v3_paths


_SAFE_PARTITION = re.compile(r"[^A-Za-z0-9_.=-]+")


@dataclass(frozen=True)
class BundleRowGroupRef:
    bundle_path: Path
    row_group: int
    row_count: int
    bundle_sha256: str
    source_schema_json: str = ""


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
    storage_kind: str = "legacy"
    bundle_segments: tuple[BundleRowGroupRef, ...] = ()
    embedded_receipt_json: str = ""
    catalog_path: Path | None = None

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
            "storage_kind": self.storage_kind,
            "bundle_segments": [
                {
                    "bundle_path": (
                        path_for_manifest(item.bundle_path, root=root)
                        if root is not None
                        else str(item.bundle_path)
                    ),
                    "row_group": int(item.row_group),
                    "row_count": int(item.row_count),
                    "bundle_sha256": item.bundle_sha256,
                }
                for item in self.bundle_segments
            ],
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
    if isinstance(ref_or_path, RawPartitionRef) and ref_or_path.embedded_receipt_json:
        try:
            receipt = json.loads(ref_or_path.embedded_receipt_json)
        except (TypeError, ValueError) as exc:
            raise RuntimeError("raw_bundle_receipt_json_invalid") from exc
        if not isinstance(receipt, dict):
            raise RuntimeError("raw_bundle_receipt_not_object")
        expected = str(ref_or_path.content_sha256)
        if str(receipt.get("content_sha256", "") or "") != expected:
            raise RuntimeError(
                f"raw_bundle_receipt_content_hash_mismatch:{ref_or_path.raw_domain}:"
                f"{ref_or_path.partition_field}={ref_or_path.partition_value}"
            )
        return dict(receipt)
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
    previous_bundle_ref: RawPartitionRef | None = None
    if not previous_sha:
        try:
            previous_bundle_ref = get_raw_partition(
                raw_domain,
                partition_field=field,
                partition_value=str(partition_value),
                workspace_root=workspace_root,
            )
            if previous_bundle_ref is not None:
                if previous_bundle_ref.storage_kind != "bundle":
                    raise RuntimeError(
                        "raw_partition_previous_without_latest_not_bundle:"
                        f"{raw_domain}:{field}={partition_value}"
                    )
                previous_receipt = read_raw_receipt(previous_bundle_ref)
                previous_identity = (
                    previous_bundle_ref.raw_domain,
                    previous_bundle_ref.partition_field,
                    previous_bundle_ref.partition_value,
                    str(previous_receipt.get("content_sha256", "") or ""),
                    int(previous_receipt.get("row_count", 0) or 0),
                )
                expected_previous_identity = (
                    str(raw_domain),
                    field,
                    str(partition_value),
                    previous_bundle_ref.content_sha256,
                    int(previous_bundle_ref.row_count),
                )
                if previous_identity != expected_previous_identity:
                    raise RuntimeError(
                        "raw_partition_previous_bundle_identity_mismatch:"
                        f"expected={expected_previous_identity}:actual={previous_identity}"
                    )
                previous_sha = previous_bundle_ref.content_sha256
                if previous_sha == content_sha:
                    staged_payload.unlink(missing_ok=True)
                    try:
                        partition_dir.rmdir()
                    except OSError:
                        pass
                    return previous_bundle_ref, False
        except BaseException:
            staged_payload.unlink(missing_ok=True)
            try:
                partition_dir.rmdir()
            except OSError:
                pass
            raise
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
                "dtypes": [str(item) for item in persisted_frame.dtypes],
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


@lru_cache(maxsize=256)
def _cached_file_sha256(path: str, size: int, modified_ns: int) -> str:
    del size, modified_ns
    return sha256_file(path)


def _verify_file_sha256(path: Path, expected: str, *, error_code: str) -> None:
    if not path.exists():
        raise RuntimeError(f"{error_code}_missing:{path}")
    stat = path.stat()
    actual = _cached_file_sha256(str(path), int(stat.st_size), int(stat.st_mtime_ns))
    if not expected or actual != expected:
        raise RuntimeError(f"{error_code}_hash_mismatch:{path}")


def _read_hash_sidecar(payload_path: Path) -> str:
    hash_path = payload_path.with_name(f"{payload_path.name}.sha256")
    if not hash_path.exists():
        raise RuntimeError(f"raw_bundle_catalog_hash_missing:{hash_path}")
    parts = hash_path.read_text(encoding="ascii").strip().split()
    if not parts or not re.fullmatch(r"[0-9a-fA-F]{64}", parts[0]):
        raise RuntimeError(f"raw_bundle_catalog_hash_invalid:{hash_path}")
    expected = parts[0].lower()
    _verify_file_sha256(payload_path, expected, error_code="raw_bundle_catalog")
    return expected


def _exported_catalog_candidates(metadata_root: Path) -> list[tuple[Path, Path]]:
    candidates: list[tuple[Path, Path]] = []
    if not metadata_root.exists():
        return candidates
    for index_path in metadata_root.rglob("raw_index.parquet"):
        receipts_path = index_path.with_name("raw_receipts.parquet")
        if receipts_path.exists():
            candidates.append((index_path, receipts_path))
    return sorted(
        candidates,
        key=lambda item: max(item[0].stat().st_mtime_ns, item[1].stat().st_mtime_ns),
        reverse=True,
    )


def _bundle_catalog_frames(
    raw_domain: str,
    *,
    workspace_root: str | Path | None,
    partition_key: str = "",
) -> tuple[pd.DataFrame, pd.DataFrame, Path, Path] | None:
    paths = qdp_v3_paths(workspace_root)
    runtime_error: BaseException | None = None
    if paths.runtime_index.exists():
        try:
            from quant_data_platform.qdp_v3.runtime_index import RuntimeIndex

            runtime_index = RuntimeIndex(
                workspace_root=workspace_root,
                database_path=paths.runtime_index,
            )
            partitions, receipts = runtime_index.raw_catalog_frames(
                str(raw_domain),
                partition_key=str(partition_key),
            )
            if not partitions.empty or not receipts.empty:
                return partitions, receipts, paths.runtime_index, paths.runtime_index
        except (OSError, RuntimeError, sqlite3.DatabaseError) as exc:
            runtime_error = exc

    export_error: BaseException | None = None
    for index_path, receipts_path in _exported_catalog_candidates(paths.metadata):
        try:
            _read_hash_sidecar(index_path)
            _read_hash_sidecar(receipts_path)
            partitions = pd.read_parquet(index_path, engine="pyarrow")
            receipts = pd.read_parquet(receipts_path, engine="pyarrow")
            partitions = partitions.loc[partitions["domain"].astype(str).eq(str(raw_domain))].copy()
            receipts = receipts.loc[receipts["domain"].astype(str).eq(str(raw_domain))].copy()
            if partition_key:
                partitions = partitions.loc[
                    partitions["partition_key"].astype(str).eq(str(partition_key))
                ].copy()
                receipts = receipts.loc[
                    receipts["partition_key"].astype(str).eq(str(partition_key))
                ].copy()
            if not partitions.empty or not receipts.empty:
                return partitions, receipts, index_path, receipts_path
        except (KeyError, OSError, RuntimeError, ValueError) as exc:
            export_error = exc

    if runtime_error is not None or export_error is not None:
        detail = export_error or runtime_error
        raise RuntimeError(f"raw_bundle_catalog_unavailable:{raw_domain}:{detail}") from detail
    return None


def _resolve_bundle_path(value: str, *, root: Path, bundle_root: Path) -> Path:
    path = Path(str(value))
    resolved = path.resolve() if path.is_absolute() else (root / path).resolve()
    try:
        resolved.relative_to(bundle_root.resolve())
    except ValueError as exc:
        raise RuntimeError(f"raw_bundle_path_outside_root:{resolved}") from exc
    return resolved


def _parse_source_schema(schema_json: str) -> tuple[list[str], list[str]]:
    if not schema_json:
        return [], []
    try:
        payload = json.loads(schema_json)
    except (TypeError, ValueError) as exc:
        raise RuntimeError("raw_bundle_source_schema_invalid") from exc
    if not isinstance(payload, Mapping):
        raise RuntimeError("raw_bundle_source_schema_not_object")
    columns = [str(value) for value in list(payload.get("columns", []) or [])]
    dtypes = [str(value) for value in list(payload.get("dtypes", []) or [])]
    if len(columns) != len(dtypes) or len(set(columns)) != len(columns):
        raise RuntimeError("raw_bundle_source_schema_width_mismatch")
    return columns, dtypes


def _latest_content_hash(receipts: pd.DataFrame, *, partition_key: str) -> str:
    payloads: dict[str, dict[str, Any]] = {}
    for row in receipts.itertuples(index=False):
        content_sha = str(row.source_content_sha256)
        try:
            payload = json.loads(str(row.receipt_json))
        except (TypeError, ValueError) as exc:
            raise RuntimeError(
                f"raw_bundle_receipt_json_invalid:{partition_key}:{content_sha}"
            ) from exc
        if not isinstance(payload, dict):
            raise RuntimeError(f"raw_bundle_receipt_not_object:{partition_key}:{content_sha}")
        if str(payload.get("content_sha256", "") or "") != content_sha:
            raise RuntimeError(
                f"raw_bundle_receipt_content_hash_mismatch:{partition_key}:{content_sha}"
            )
        payloads[content_sha] = payload
    predecessors = {
        str(payload.get("revision_of", "") or "")
        for payload in payloads.values()
        if str(payload.get("revision_of", "") or "")
    }
    tails = sorted(set(payloads) - predecessors)
    if len(tails) != 1:
        raise RuntimeError(
            f"raw_bundle_latest_revision_ambiguous:{partition_key}:{','.join(tails)}"
        )
    return tails[0]


def _index_bundle_partition_mappings(
    partitions: pd.DataFrame,
) -> dict[tuple[str, str], tuple[int, ...]]:
    """Group bundle mappings once for constant-time receipt resolution."""

    if partitions.empty:
        return {}
    key_frame = pd.DataFrame(
        {
            "partition_key": partitions["partition_key"].astype(str).to_numpy(),
            "source_content_sha256": (
                partitions["source_content_sha256"].astype(str).to_numpy()
            ),
        }
    )
    groups = key_frame.groupby(
        ["partition_key", "source_content_sha256"],
        sort=False,
        dropna=False,
    ).indices
    return {
        (str(partition_key), str(content_sha256)): tuple(
            int(position) for position in positions
        )
        for (partition_key, content_sha256), positions in groups.items()
    }


def _bundle_refs(
    raw_domain: str,
    *,
    workspace_root: str | Path | None,
    partition_key: str = "",
    content_sha256: str = "",
) -> list[RawPartitionRef]:
    catalog = _bundle_catalog_frames(
        raw_domain,
        workspace_root=workspace_root,
        partition_key=partition_key,
    )
    if catalog is None:
        return []
    partitions, receipts, index_path, receipts_path = catalog
    required_receipt_columns = {
        "domain",
        "partition_key",
        "row_count",
        "source_content_sha256",
        "quality_tier",
        "receipt_json",
    }
    required_partition_columns = {
        "domain",
        "partition_key",
        "row_count",
        "source_content_sha256",
        "bundle_path",
        "row_group",
        "status",
        "bundle_sha256",
    }
    if not required_receipt_columns.issubset(receipts.columns):
        missing = sorted(required_receipt_columns - set(receipts.columns))
        raise RuntimeError(f"raw_bundle_receipt_catalog_columns_missing:{missing}")
    if not required_partition_columns.issubset(partitions.columns):
        missing = sorted(required_partition_columns - set(partitions.columns))
        raise RuntimeError(f"raw_bundle_index_catalog_columns_missing:{missing}")
    if "source_schema_json" not in partitions.columns:
        partitions = partitions.assign(source_schema_json="")
    mapping_index = _index_bundle_partition_mappings(partitions)

    paths = qdp_v3_paths(workspace_root)
    refs: list[RawPartitionRef] = []
    for partition_key, receipt_group in receipts.groupby("partition_key", sort=True):
        key = str(partition_key)
        if content_sha256:
            requested_receipts = receipt_group.loc[
                receipt_group["source_content_sha256"].astype(str).eq(str(content_sha256))
            ]
            if requested_receipts.empty:
                continue
            if len(requested_receipts) != 1:
                raise RuntimeError(
                    f"raw_bundle_receipt_duplicate:{key}:{content_sha256}"
                )
            content_sha = str(content_sha256)
        else:
            content_sha = _latest_content_hash(receipt_group, partition_key=key)
        selected_receipt = receipt_group.loc[
            receipt_group["source_content_sha256"].astype(str).eq(content_sha)
        ]
        if len(selected_receipt) != 1:
            raise RuntimeError(f"raw_bundle_receipt_duplicate:{key}:{content_sha}")
        receipt_row = selected_receipt.iloc[0]
        try:
            receipt_payload = json.loads(str(receipt_row["receipt_json"]))
        except (TypeError, ValueError) as exc:
            raise RuntimeError(f"raw_bundle_receipt_json_invalid:{key}:{content_sha}") from exc
        field = str(receipt_payload.get("partition_field", "") or "")
        value = str(receipt_payload.get("partition_value", "") or "")
        if not field:
            field, separator, value = key.partition("=")
            if not separator:
                raise RuntimeError(f"raw_bundle_partition_key_invalid:{key}")
        if key != f"{field}={value}":
            raise RuntimeError(f"raw_bundle_partition_key_mismatch:{key}:{field}={value}")

        mapping_positions = mapping_index.get((key, content_sha))
        if mapping_positions is None:
            raise RuntimeError(f"raw_bundle_mapping_missing:{raw_domain}:{key}:{content_sha}")
        mapping_rows = partitions.iloc[list(mapping_positions)].copy()
        expected_rows = int(receipt_row["row_count"])
        if int(mapping_rows["row_count"].astype("int64").sum()) != expected_rows:
            raise RuntimeError(f"raw_bundle_mapping_row_count_mismatch:{raw_domain}:{key}")
        statuses = set(mapping_rows["status"].astype(str))
        if expected_rows == 0:
            if statuses != {"empty_success"} or len(mapping_rows) != 1:
                raise RuntimeError(f"raw_bundle_empty_mapping_invalid:{raw_domain}:{key}")
        elif statuses != {"bundled"}:
            raise RuntimeError(f"raw_bundle_mapping_status_invalid:{raw_domain}:{key}:{statuses}")

        segments: list[BundleRowGroupRef] = []
        schema_values = {
            str(value or "") for value in mapping_rows["source_schema_json"].tolist()
        }
        if len(schema_values) > 1:
            raise RuntimeError(f"raw_bundle_source_schema_conflict:{raw_domain}:{key}")
        schema_json = next(iter(schema_values), "")
        columns, dtypes = _parse_source_schema(schema_json)
        if columns:
            receipt_payload.setdefault("fields", columns)
            receipt_payload.setdefault("dtypes", dtypes)
        for row in mapping_rows.sort_values(["bundle_path", "row_group"]).itertuples(index=False):
            if str(row.status) == "empty_success":
                continue
            segments.append(
                BundleRowGroupRef(
                    bundle_path=_resolve_bundle_path(
                        str(row.bundle_path),
                        root=paths.root,
                        bundle_root=paths.raw_bundles,
                    ),
                    row_group=int(row.row_group),
                    row_count=int(row.row_count),
                    bundle_sha256=str(row.bundle_sha256),
                    source_schema_json=str(row.source_schema_json or ""),
                )
            )
        representative_path = segments[0].bundle_path if segments else index_path
        refs.append(
            RawPartitionRef(
                raw_domain=str(raw_domain),
                partition_field=field,
                partition_value=value,
                content_sha256=content_sha,
                payload_path=representative_path,
                receipt_path=receipts_path,
                row_count=expected_rows,
                quality_tier=str(receipt_row["quality_tier"] or "quarantined"),
                revision_of=str(receipt_payload.get("revision_of", "") or ""),
                storage_kind="bundle",
                bundle_segments=tuple(segments),
                embedded_receipt_json=json.dumps(
                    receipt_payload,
                    ensure_ascii=False,
                    sort_keys=True,
                    separators=(",", ":"),
                ),
                catalog_path=index_path,
            )
        )
    return refs


def iter_raw_partitions(
    raw_domain: str,
    *,
    workspace_root: str | Path | None = None,
    start_value: str = "",
    end_value: str = "",
) -> list[RawPartitionRef]:
    paths = qdp_v3_paths(workspace_root)
    domain_root = paths.raw / str(raw_domain)
    legacy_refs: list[RawPartitionRef] = []
    if domain_root.exists():
        for latest_path in sorted(domain_root.glob("*/latest.json")):
            legacy_refs.append(_ref_from_latest(latest_path, root=paths.root))
    legacy_keys = {(ref.partition_field, ref.partition_value) for ref in legacy_refs}
    bundled_refs = [
        ref
        for ref in _bundle_refs(raw_domain, workspace_root=workspace_root)
        if (ref.partition_field, ref.partition_value) not in legacy_keys
    ]
    refs: list[RawPartitionRef] = []
    for ref in sorted(
        [*legacy_refs, *bundled_refs],
        key=lambda item: (item.partition_value, item.partition_field),
    ):
        value = ref.partition_value
        if start_value and value < str(start_value):
            continue
        if end_value and value > str(end_value):
            continue
        refs.append(ref)
    return refs


def get_raw_partition(
    raw_domain: str,
    *,
    partition_field: str,
    partition_value: str,
    workspace_root: str | Path | None = None,
) -> RawPartitionRef | None:
    """Resolve one known raw partition directly without scanning its domain."""

    paths = qdp_v3_paths(workspace_root)
    field = _safe_partition_value(partition_field)
    value = _safe_partition_value(partition_value)
    latest_path = paths.raw / str(raw_domain) / f"{field}={value}" / "latest.json"
    if latest_path.exists():
        ref = _ref_from_latest(latest_path, root=paths.root)
    else:
        matches = [
            item
            for item in _bundle_refs(
                raw_domain,
                workspace_root=workspace_root,
                partition_key=f"{field}={partition_value}",
            )
            if item.partition_field == field and item.partition_value == str(partition_value)
        ]
        if not matches:
            return None
        if len(matches) != 1:
            raise RuntimeError(
                f"raw_bundle_direct_lookup_ambiguous:{raw_domain}:{field}={partition_value}"
            )
        ref = matches[0]
    if ref.partition_field != field or ref.partition_value != str(partition_value):
        raise RuntimeError(
            "raw_partition_direct_lookup_mismatch:"
            f"expected={field}={partition_value}:actual={ref.partition_field}={ref.partition_value}"
        )
    return ref


def get_raw_partition_version(
    raw_domain: str,
    *,
    partition_field: str,
    partition_value: str,
    content_sha256: str,
    workspace_root: str | Path | None = None,
) -> RawPartitionRef | None:
    """Resolve one immutable raw version from legacy files or bundle catalogs.

    Unlike :func:`get_raw_partition`, this lookup never substitutes the latest
    revision for the requested content hash.  During compaction both physical
    representations may coexist; they are treated as the same immutable
    version only when their identity and row-count metadata agree.
    """

    paths = qdp_v3_paths(workspace_root)
    field = _safe_partition_value(partition_field)
    value = _safe_partition_value(partition_value)
    content_sha = str(content_sha256 or "").strip().lower()
    if not re.fullmatch(r"[0-9a-f]{64}", content_sha):
        raise ValueError(f"invalid_raw_content_sha256:{content_sha256}")

    legacy_ref: RawPartitionRef | None = None
    version_dir = (
        paths.raw
        / str(raw_domain)
        / f"{field}={value}"
        / "versions"
        / content_sha
    )
    if version_dir.exists():
        receipt_path = version_dir / "receipt.json"
        payload_path = version_dir / "payload.parquet"
        if receipt_path.exists() != payload_path.exists():
            raise RuntimeError(f"raw_partition_version_incomplete:{version_dir}")
        if receipt_path.exists():
            receipt = read_raw_receipt(receipt_path)
            actual_identity = (
                str(receipt.get("raw_domain", "") or raw_domain),
                str(receipt.get("partition_field", "") or field),
                str(receipt.get("partition_value", "") or partition_value),
                str(receipt.get("content_sha256", "") or ""),
            )
            expected_identity = (
                str(raw_domain),
                field,
                str(partition_value),
                content_sha,
            )
            if actual_identity != expected_identity:
                raise RuntimeError(
                    "raw_partition_version_identity_mismatch:"
                    f"expected={expected_identity}:actual={actual_identity}"
                )
            assessment_path = str(receipt.get("quality_assessment_path", "") or "")
            legacy_ref = RawPartitionRef(
                raw_domain=str(raw_domain),
                partition_field=field,
                partition_value=str(partition_value),
                content_sha256=content_sha,
                payload_path=payload_path,
                receipt_path=receipt_path,
                row_count=int(receipt.get("row_count", 0) or 0),
                quality_tier=str(
                    receipt.get("quality_tier", "") or "quarantined"
                ),
                revision_of=str(receipt.get("revision_of", "") or ""),
                quality_assessment_path=(
                    Path(assessment_path) if assessment_path else None
                ),
                quality_assessment_sha256=str(
                    receipt.get("quality_assessment_sha256", "") or ""
                ),
            )

    bundled = _bundle_refs(
        raw_domain,
        workspace_root=workspace_root,
        partition_key=f"{field}={partition_value}",
        content_sha256=content_sha,
    )
    matching_bundles = [
        ref
        for ref in bundled
        if ref.partition_field == field
        and ref.partition_value == str(partition_value)
        and ref.content_sha256 == content_sha
    ]
    if len(matching_bundles) > 1:
        raise RuntimeError(
            "raw_partition_version_bundle_ambiguous:"
            f"{raw_domain}:{field}={partition_value}:{content_sha}"
        )
    bundle_ref = matching_bundles[0] if matching_bundles else None

    if legacy_ref is not None and bundle_ref is not None:
        legacy_receipt = read_raw_receipt(legacy_ref)
        bundle_receipt = read_raw_receipt(bundle_ref)
        legacy_identity = (
            legacy_ref.raw_domain,
            legacy_ref.partition_field,
            legacy_ref.partition_value,
            legacy_ref.content_sha256,
            int(legacy_ref.row_count),
            str(legacy_receipt.get("revision_of", "") or ""),
        )
        bundle_identity = (
            bundle_ref.raw_domain,
            bundle_ref.partition_field,
            bundle_ref.partition_value,
            bundle_ref.content_sha256,
            int(bundle_ref.row_count),
            str(bundle_receipt.get("revision_of", "") or ""),
        )
        if legacy_identity != bundle_identity:
            raise RuntimeError(
                "raw_partition_version_representation_conflict:"
                f"legacy={legacy_identity}:bundle={bundle_identity}"
            )
        return legacy_ref
    return legacy_ref or bundle_ref


def write_empty_raw_partition(
    *,
    raw_domain: str,
    partition_value: str,
    frame: pd.DataFrame,
    receipt: Mapping[str, Any],
    workspace_root: str | Path | None = None,
    partition_field: str = "query_date",
) -> tuple[RawPartitionRef, bool]:
    """Persist a successful empty response in the runtime raw catalog only."""

    if not isinstance(frame, pd.DataFrame) or not frame.empty:
        raise ValueError("index_only_raw_partition_requires_empty_frame")
    paths = ensure_qdp_v3_layout(workspace_root)
    field = _safe_partition_value(partition_field)
    value = _safe_partition_value(partition_value)
    normalized = frame.copy()
    content_sha = frame_content_sha256(normalized)
    latest = get_raw_partition(
        raw_domain,
        partition_field=field,
        partition_value=str(partition_value),
        workspace_root=workspace_root,
    )
    if latest is not None and latest.content_sha256 == content_sha:
        read_raw_receipt(latest)
        return latest, False
    if latest is not None:
        prior_same_content = get_raw_partition_version(
            raw_domain,
            partition_field=field,
            partition_value=str(partition_value),
            content_sha256=content_sha,
            workspace_root=workspace_root,
        )
        if prior_same_content is not None:
            raise RuntimeError(
                f"index_only_raw_content_reversion_unsupported:{raw_domain}:{field}={partition_value}"
            )

    revision_of = latest.content_sha256 if latest is not None else ""
    payload = dict(receipt)
    payload.update(
        {
            "raw_manifest_version": 4,
            "raw_domain": str(raw_domain),
            "partition_field": field,
            "partition_value": str(partition_value),
            "content_sha256": content_sha,
            "row_count": 0,
            "fields": [str(item) for item in normalized.columns],
            "dtypes": [str(item) for item in normalized.dtypes],
            "revision_of": revision_of,
            "storage_kind": "index_only_empty",
            "stored_at": utc_now(),
        }
    )
    provider = str(payload.get("provider", "") or "unknown")
    response_sha = str(payload.get("response_sha256", "") or content_sha)
    request_payload = payload.get("request", {})
    request_range = json.dumps(
        request_payload if isinstance(request_payload, Mapping) else str(request_payload),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    )
    quality_tier = str(payload.get("quality_tier", "") or "quarantined")
    source_schema_json = json.dumps(
        {
            "columns": [str(item) for item in normalized.columns],
            "dtypes": [str(item) for item in normalized.dtypes],
        },
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    from quant_data_platform.qdp_v3.runtime_index import RuntimeIndex

    runtime_index = RuntimeIndex(workspace_root=workspace_root)
    stored_receipt, created = runtime_index.record_index_only_empty(
        domain=str(raw_domain),
        partition_key=f"{field}={partition_value}",
        provider=provider,
        request_range=request_range,
        response_sha256=response_sha,
        source_content_sha256=content_sha,
        source_schema_json=source_schema_json,
        quality_tier=quality_tier,
        receipt=payload,
    )
    return (
        RawPartitionRef(
            raw_domain=str(raw_domain),
            partition_field=field,
            partition_value=str(partition_value),
            content_sha256=content_sha,
            payload_path=runtime_index.path,
            receipt_path=runtime_index.path,
            row_count=0,
            quality_tier=str(stored_receipt.get("quality_tier", "") or quality_tier),
            revision_of=str(stored_receipt.get("revision_of", "") or ""),
            storage_kind="bundle",
            bundle_segments=(),
            embedded_receipt_json=json.dumps(
                stored_receipt,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            ),
            catalog_path=runtime_index.path,
        ),
        created,
    )


def _schema_for_bundle_ref(ref: RawPartitionRef) -> tuple[list[str], list[str]]:
    schema_values = {
        segment.source_schema_json
        for segment in ref.bundle_segments
        if segment.source_schema_json
    }
    if len(schema_values) > 1:
        raise RuntimeError(
            f"raw_bundle_source_schema_conflict:{ref.raw_domain}:"
            f"{ref.partition_field}={ref.partition_value}"
        )
    if schema_values:
        return _parse_source_schema(next(iter(schema_values)))
    receipt = read_raw_receipt(ref)
    columns = [str(value) for value in list(receipt.get("fields", []) or [])]
    dtypes = [str(value) for value in list(receipt.get("dtypes", []) or [])]
    if dtypes and len(columns) != len(dtypes):
        raise RuntimeError(
            f"raw_bundle_receipt_schema_width_mismatch:{ref.raw_domain}:"
            f"{ref.partition_field}={ref.partition_value}"
        )
    return columns, dtypes


def _empty_frame_from_schema(columns: list[str], dtypes: list[str]) -> pd.DataFrame:
    if not columns:
        raise RuntimeError("raw_bundle_empty_source_schema_missing")
    if not dtypes:
        return pd.DataFrame(columns=columns)
    data: dict[str, pd.Series] = {}
    for column, dtype in zip(columns, dtypes, strict=True):
        try:
            data[column] = pd.Series(dtype=dtype)
        except (TypeError, ValueError) as exc:
            raise RuntimeError(f"raw_bundle_empty_dtype_invalid:{column}:{dtype}") from exc
    return pd.DataFrame(data, columns=columns)


def _validate_source_schema(
    frame: pd.DataFrame,
    *,
    columns: list[str],
    dtypes: list[str],
    ref: RawPartitionRef,
) -> None:
    actual_columns = [str(value) for value in frame.columns]
    if columns and actual_columns != columns:
        raise RuntimeError(
            f"raw_bundle_source_columns_mismatch:{ref.raw_domain}:"
            f"{ref.partition_field}={ref.partition_value}"
        )
    actual_dtypes = [str(value) for value in frame.dtypes]
    if dtypes and actual_dtypes != dtypes:
        raise RuntimeError(
            f"raw_bundle_source_dtypes_mismatch:{ref.raw_domain}:"
            f"{ref.partition_field}={ref.partition_value}:"
            f"expected={dtypes}:actual={actual_dtypes}"
        )


def _read_bundled_raw_partition(
    ref: RawPartitionRef,
    *,
    verify_hash: bool,
) -> pd.DataFrame:
    from quant_data_platform.qdp_v3.bundles import SOURCE_ROW_ORDINAL_COLUMN

    read_raw_receipt(ref)
    columns, dtypes = _schema_for_bundle_ref(ref)
    if int(ref.row_count) == 0:
        if ref.bundle_segments:
            raise RuntimeError(
                f"raw_bundle_empty_has_segments:{ref.raw_domain}:"
                f"{ref.partition_field}={ref.partition_value}"
            )
        frame = _empty_frame_from_schema(columns, dtypes)
    else:
        if not ref.bundle_segments:
            raise RuntimeError(
                f"raw_bundle_segments_missing:{ref.raw_domain}:"
                f"{ref.partition_field}={ref.partition_value}"
            )
        parquet_files: dict[Path, pq.ParquetFile] = {}
        frames: list[pd.DataFrame] = []
        for segment in ref.bundle_segments:
            if verify_hash:
                _verify_file_sha256(
                    segment.bundle_path,
                    segment.bundle_sha256,
                    error_code="raw_bundle",
                )
            parquet_file = parquet_files.get(segment.bundle_path)
            if parquet_file is None:
                if not segment.bundle_path.exists():
                    raise RuntimeError(f"raw_bundle_missing:{segment.bundle_path}")
                parquet_file = pq.ParquetFile(segment.bundle_path)
                parquet_files[segment.bundle_path] = parquet_file
            if segment.row_group < 0 or segment.row_group >= parquet_file.num_row_groups:
                raise RuntimeError(
                    f"raw_bundle_row_group_out_of_range:{segment.bundle_path}:"
                    f"{segment.row_group}:{parquet_file.num_row_groups}"
                )
            metadata_rows = int(
                parquet_file.metadata.row_group(segment.row_group).num_rows
            )
            if metadata_rows != int(segment.row_count):
                raise RuntimeError(
                    f"raw_bundle_row_group_count_mismatch:{segment.bundle_path}:"
                    f"{segment.row_group}:{metadata_rows}:{segment.row_count}"
                )
            segment_frame = parquet_file.read_row_group(segment.row_group).to_pandas()
            if len(segment_frame) != int(segment.row_count):
                raise RuntimeError(
                    f"raw_bundle_decoded_row_count_mismatch:{segment.bundle_path}:"
                    f"{segment.row_group}"
                )
            frames.append(segment_frame)
        frame = pd.concat(frames, ignore_index=True, sort=False)
        if SOURCE_ROW_ORDINAL_COLUMN in frame.columns:
            ordinal = pd.to_numeric(frame[SOURCE_ROW_ORDINAL_COLUMN], errors="coerce")
            if ordinal.isna().any() or ordinal.duplicated().any():
                raise RuntimeError(
                    f"raw_bundle_source_ordinal_invalid:{ref.raw_domain}:"
                    f"{ref.partition_field}={ref.partition_value}"
                )
            frame = frame.assign(**{SOURCE_ROW_ORDINAL_COLUMN: ordinal.astype("int64")})
            frame = frame.sort_values(SOURCE_ROW_ORDINAL_COLUMN, kind="stable").reset_index(drop=True)
            expected_ordinal = pd.Series(
                range(len(frame)),
                name=SOURCE_ROW_ORDINAL_COLUMN,
                dtype="int64",
            )
            if not frame[SOURCE_ROW_ORDINAL_COLUMN].equals(expected_ordinal):
                raise RuntimeError(
                    f"raw_bundle_source_ordinal_gap:{ref.raw_domain}:"
                    f"{ref.partition_field}={ref.partition_value}"
                )
            frame = frame.drop(columns=[SOURCE_ROW_ORDINAL_COLUMN])
        elif len(ref.bundle_segments) != 1:
            raise RuntimeError(
                f"raw_bundle_source_ordinal_missing:{ref.raw_domain}:"
                f"{ref.partition_field}={ref.partition_value}"
            )
    if len(frame) != int(ref.row_count):
        raise RuntimeError(
            f"raw_bundle_partition_row_count_mismatch:{ref.raw_domain}:"
            f"{ref.partition_field}={ref.partition_value}:{len(frame)}:{ref.row_count}"
        )
    _validate_source_schema(frame, columns=columns, dtypes=dtypes, ref=ref)
    if verify_hash and frame_content_sha256(frame) != ref.content_sha256:
        raise RuntimeError(
            f"raw_content_hash_mismatch:{ref.raw_domain}:"
            f"{ref.partition_field}={ref.partition_value}"
        )
    return frame


def read_raw_partition(ref: RawPartitionRef, *, verify_hash: bool = True) -> pd.DataFrame:
    if ref.storage_kind == "bundle":
        return _read_bundled_raw_partition(ref, verify_hash=verify_hash)
    receipt = read_raw_receipt(ref)
    if verify_hash:
        expected_parquet = str(receipt.get("parquet_sha256", "") or "")
        if not expected_parquet or sha256_file(ref.payload_path) != expected_parquet:
            raise RuntimeError(f"raw_parquet_hash_mismatch:{ref.payload_path}")
    frame = pd.read_parquet(ref.payload_path, engine="pyarrow")
    if len(frame) != int(ref.row_count):
        raise RuntimeError(f"raw_partition_row_count_mismatch:{ref.payload_path}")
    fields = [str(value) for value in list(receipt.get("fields", []) or [])]
    if fields and [str(value) for value in frame.columns] != fields:
        raise RuntimeError(f"raw_partition_columns_mismatch:{ref.payload_path}")
    if verify_hash and frame_content_sha256(frame) != ref.content_sha256:
        raise RuntimeError(f"raw_content_hash_mismatch:{ref.payload_path}")
    return frame


def raw_content_hashes(refs: Iterable[RawPartitionRef]) -> list[str]:
    return [str(ref.content_sha256) for ref in refs]
