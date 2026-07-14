from __future__ import annotations

import hashlib
import json
import os
import re
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from quant_data_platform.qdp_v3.manifest import path_for_manifest, sha256_file, stable_hash
from quant_data_platform.qdp_v3.paths import QdpV3Paths
from quant_data_platform.qdp_v3.runtime_index import RawPartitionIndexRecord, RuntimeIndex


MIB = 1024**2
GIB = 1024**3
DEFAULT_RAW_BUCKET_COUNT = 16
SOURCE_ROW_ORDINAL_COLUMN = "__qdp_source_row_ordinal__"
_SAFE_COMPONENT = re.compile(r"[^A-Za-z0-9_.=-]+")


@dataclass(frozen=True)
class BundleSizingPolicy:
    target_part_bytes: int = 384 * MIB
    max_part_bytes: int = 1 * GIB
    estimated_compression_ratio: float = 0.40

    def __post_init__(self) -> None:
        if int(self.target_part_bytes) <= 0:
            raise ValueError("bundle_target_part_bytes_must_be_positive")
        if int(self.max_part_bytes) < int(self.target_part_bytes):
            raise ValueError("bundle_max_part_bytes_below_target")
        if not 0 < float(self.estimated_compression_ratio) <= 1:
            raise ValueError("bundle_compression_ratio_out_of_range")

    def estimate_frame_bytes(self, frame: pd.DataFrame) -> int:
        if frame.empty:
            return 0
        memory_bytes = int(frame.memory_usage(index=False, deep=True).sum())
        estimate = int(memory_bytes * float(self.estimated_compression_ratio))
        return max(estimate, int(len(frame)) * max(1, len(frame.columns)))

    def rows_per_segment(self, frame: pd.DataFrame) -> int:
        estimate = self.estimate_frame_bytes(frame)
        if estimate <= self.target_part_bytes or frame.empty:
            return max(1, int(len(frame)))
        bytes_per_row = max(1.0, estimate / float(len(frame)))
        return max(1, int(self.target_part_bytes / bytes_per_row))


@dataclass(frozen=True)
class BundleSegment:
    domain: str
    partition_key: str
    provider: str
    request_range: str
    response_sha256: str
    source_content_sha256: str
    frame: pd.DataFrame
    segment_ordinal: int = 0


@dataclass(frozen=True)
class BundlePartRef:
    path: Path
    relative_path: str
    sha256: str
    row_count: int
    row_group_count: int
    year: str
    security_bucket: int
    created: bool


@dataclass(frozen=True)
class BundleWriteResult:
    parts: tuple[BundlePartRef, ...]
    indexed_rows: int

    @property
    def created_part_count(self) -> int:
        return sum(1 for part in self.parts if part.created)

    @property
    def reused_part_count(self) -> int:
        return sum(1 for part in self.parts if not part.created)


def stable_raw_bucket(value: object, *, bucket_count: int = DEFAULT_RAW_BUCKET_COUNT) -> int:
    if int(bucket_count) <= 0:
        raise ValueError("raw_bucket_count_must_be_positive")
    normalized = str(value or "").strip().upper()
    digest = hashlib.sha256(normalized.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], byteorder="big", signed=False) % int(bucket_count)


def split_bundle_segment(segment: BundleSegment, policy: BundleSizingPolicy) -> Iterable[BundleSegment]:
    rows_per_segment = policy.rows_per_segment(segment.frame)
    if segment.frame.empty or len(segment.frame) <= rows_per_segment:
        yield segment
        return
    ordinal = int(segment.segment_ordinal)
    for start in range(0, len(segment.frame), rows_per_segment):
        yield BundleSegment(
            domain=segment.domain,
            partition_key=segment.partition_key,
            provider=segment.provider,
            request_range=segment.request_range,
            response_sha256=segment.response_sha256,
            source_content_sha256=segment.source_content_sha256,
            frame=segment.frame.iloc[start : start + rows_per_segment].copy(),
            segment_ordinal=ordinal,
        )
        ordinal += 1


def _safe_component(value: str) -> str:
    safe = _SAFE_COMPONENT.sub("_", str(value).strip())
    if not safe or safe in {".", ".."}:
        raise ValueError(f"invalid_bundle_component:{value!r}")
    return safe


@dataclass
class _OpenBundle:
    key: tuple[str, str, int, str]
    temp_path: Path
    writer: pq.ParquetWriter
    estimated_bytes: int = 0
    row_count: int = 0
    mappings: list[BundleSegment] = field(default_factory=list)


class RawBundleWriter:
    """Stream immutable raw segments into bounded, content-addressed Parquet parts."""

    def __init__(
        self,
        *,
        paths: QdpV3Paths,
        runtime_index: RuntimeIndex,
        sizing: BundleSizingPolicy | None = None,
        bucket_count: int = DEFAULT_RAW_BUCKET_COUNT,
    ) -> None:
        self.paths = paths
        self.runtime_index = runtime_index
        self.sizing = sizing or BundleSizingPolicy()
        self.bucket_count = int(bucket_count)
        if self.bucket_count != DEFAULT_RAW_BUCKET_COUNT:
            raise ValueError(f"raw_bundle_bucket_count_must_be_{DEFAULT_RAW_BUCKET_COUNT}")
        self._open: dict[tuple[str, str, int, str], _OpenBundle] = {}
        self._parts: list[BundlePartRef] = []
        self._indexed_rows = 0
        self._prepared_sources: set[tuple[str, str, str]] = set()

    def add(self, *, year: str, security_bucket: int, segment: BundleSegment) -> None:
        source_key = (
            str(segment.domain),
            str(segment.partition_key),
            str(segment.source_content_sha256),
        )
        if source_key not in self._prepared_sources:
            self.runtime_index.delete_raw_partition_mappings(
                domain=source_key[0],
                partition_key=source_key[1],
                source_content_sha256=source_key[2],
            )
            self._prepared_sources.add(source_key)
        source_schema_json = json.dumps(
            {
                "columns": [str(column) for column in segment.frame.columns],
                "dtypes": [str(dtype) for dtype in segment.frame.dtypes],
            },
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        if segment.frame.empty:
            self.runtime_index.record_empty_success(
                domain=segment.domain,
                partition_key=segment.partition_key,
                provider=segment.provider,
                request_range=segment.request_range,
                response_sha256=segment.response_sha256,
                source_content_sha256=segment.source_content_sha256,
                source_schema_json=source_schema_json,
            )
            return
        for piece in split_bundle_segment(segment, self.sizing):
            if SOURCE_ROW_ORDINAL_COLUMN in piece.frame.columns:
                raise ValueError(
                    f"raw_bundle_reserved_column_collision:{SOURCE_ROW_ORDINAL_COLUMN}"
                )
            if not piece.frame.index.is_unique:
                raise ValueError("raw_bundle_source_index_not_unique")
            try:
                source_ordinals = [int(value) for value in piece.frame.index]
            except (TypeError, ValueError, OverflowError) as exc:
                raise ValueError("raw_bundle_source_index_not_integer") from exc
            if any(value < 0 for value in source_ordinals):
                raise ValueError("raw_bundle_source_index_negative")
            persisted = piece.frame.copy()
            persisted[SOURCE_ROW_ORDINAL_COLUMN] = source_ordinals
            table = pa.Table.from_pandas(persisted, preserve_index=False)
            schema_key = stable_hash({"schema": str(table.schema)}, length=24)
            key = (str(piece.domain), str(year), int(security_bucket), schema_key)
            estimate = self.sizing.estimate_frame_bytes(piece.frame)
            state = self._open.get(key)
            if state is not None and state.mappings and state.estimated_bytes + estimate > self.sizing.target_part_bytes:
                self._flush(key)
                state = None
            if state is None:
                state = self._new_state(key, table.schema)
                self._open[key] = state
            state.writer.write_table(table, row_group_size=max(1, len(piece.frame)))
            state.estimated_bytes += estimate
            state.row_count += int(len(piece.frame))
            state.mappings.append(piece)
            if state.estimated_bytes >= self.sizing.target_part_bytes:
                self._flush(key)

    def finish(self) -> BundleWriteResult:
        for key in sorted(tuple(self._open)):
            self._flush(key)
        return BundleWriteResult(parts=tuple(self._parts), indexed_rows=self._indexed_rows)

    def abort(self) -> None:
        for state in self._open.values():
            try:
                state.writer.close()
            finally:
                state.temp_path.unlink(missing_ok=True)
        self._open.clear()

    def _new_state(self, key: tuple[str, str, int, str], schema: pa.Schema) -> _OpenBundle:
        domain, year, bucket, _ = key
        directory = (
            self.paths.raw_bundles
            / _safe_component(domain)
            / f"year={_safe_component(year)}"
            / f"security_bucket={int(bucket):02d}"
        )
        directory.mkdir(parents=True, exist_ok=True)
        temp_path = directory / f".bundle.{os.getpid()}.{uuid.uuid4().hex}.tmp"
        writer = pq.ParquetWriter(temp_path, schema, compression="zstd")
        return _OpenBundle(key=key, temp_path=temp_path, writer=writer)

    def _flush(self, key: tuple[str, str, int, str]) -> None:
        state = self._open.pop(key)
        try:
            state.writer.close()
            actual_size = int(state.temp_path.stat().st_size)
            if actual_size > self.sizing.max_part_bytes:
                raise RuntimeError(
                    "raw_bundle_part_exceeds_max_bytes:"
                    f"{state.temp_path}:{actual_size}:{self.sizing.max_part_bytes}"
                )
            domain, year, bucket, schema_key = key
            identity = {
                "domain": domain,
                "year": year,
                "security_bucket": bucket,
                "schema": schema_key,
                "segments": [
                    {
                        "partition_key": item.partition_key,
                        "source_content_sha256": item.source_content_sha256,
                        "segment_ordinal": int(item.segment_ordinal),
                        "row_count": int(len(item.frame)),
                    }
                    for item in state.mappings
                ],
            }
            part_id = stable_hash(identity, length=32)
            final_path = state.temp_path.parent / f"part-{part_id}.parquet"
            staged_sha = sha256_file(state.temp_path)
            created = False
            if final_path.exists():
                if sha256_file(final_path) != staged_sha:
                    raise RuntimeError(f"raw_bundle_immutable_conflict:{final_path}")
                state.temp_path.unlink()
            else:
                state.temp_path.replace(final_path)
                created = True
            relative_path = path_for_manifest(final_path, root=self.paths.root)
            for row_group, segment in enumerate(state.mappings):
                self.runtime_index.record_raw_partition(
                    RawPartitionIndexRecord(
                        domain=domain,
                        partition_key=segment.partition_key,
                        provider=segment.provider,
                        request_range=segment.request_range,
                        row_count=int(len(segment.frame)),
                        response_sha256=segment.response_sha256,
                        source_content_sha256=segment.source_content_sha256,
                        bundle_path=relative_path,
                        row_group=row_group,
                        status="bundled",
                        bundle_sha256=staged_sha,
                        source_schema_json=json.dumps(
                            {
                                "columns": [str(column) for column in segment.frame.columns],
                                "dtypes": [str(dtype) for dtype in segment.frame.dtypes],
                            },
                            ensure_ascii=False,
                            sort_keys=True,
                            separators=(",", ":"),
                        ),
                    )
                )
                self._indexed_rows += int(len(segment.frame))
            self._parts.append(
                BundlePartRef(
                    path=final_path,
                    relative_path=relative_path,
                    sha256=staged_sha,
                    row_count=state.row_count,
                    row_group_count=len(state.mappings),
                    year=year,
                    security_bucket=bucket,
                    created=created,
                )
            )
        finally:
            state.temp_path.unlink(missing_ok=True)
