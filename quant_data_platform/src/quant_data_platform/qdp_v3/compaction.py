from __future__ import annotations

import json
import re
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping

import pandas as pd
import pyarrow.parquet as pq

from quant_data_platform.qdp_v3.bundles import (
    DEFAULT_RAW_BUCKET_COUNT,
    BundlePartRef,
    BundleSegment,
    BundleSizingPolicy,
    RawBundleWriter,
    stable_raw_bucket,
)
from quant_data_platform.qdp_v3.constants import (
    RAW_INDUSTRY_SNAPSHOT,
    RAW_INDEX_CONSTITUENTS,
    RAW_SECURITY_MASTER,
    RAW_TRADING_CALENDAR,
    RAW_TUSHARE_PROXY_NAMECHANGE,
    RAW_TUSHARE_PROXY_STOCK_BASIC,
    RAW_TUSHARE_PROXY_TRADE_CALENDAR,
)
from quant_data_platform.qdp_v3.manifest import sha256_file
from quant_data_platform.qdp_v3.paths import ensure_qdp_v3_layout
from quant_data_platform.qdp_v3.runtime_index import RawIndexExport, RuntimeIndex
from quant_data_platform.qdp_v3.storage import (
    RawPartitionRef,
    iter_raw_partitions,
    read_raw_partition,
    read_raw_receipt,
)


_YEAR_COLUMNS = (
    "trade_date",
    "query_date",
    "date",
    "cal_date",
    "trade_time",
    "datetime",
    "ann_date",
    "end_date",
    "dividOperateDate",
)
_IDENTITY_COLUMNS = ("security_id", "provider_symbol", "ts_code", "code", "symbol")
_YEAR_PATTERN = re.compile(r"(?:19|20)\d{2}")
_UNDATED_REFERENCE_DOMAINS = frozenset(
    {
        RAW_INDUSTRY_SNAPSHOT,
        RAW_INDEX_CONSTITUENTS,
        RAW_SECURITY_MASTER,
        RAW_TRADING_CALENDAR,
        RAW_TUSHARE_PROXY_NAMECHANGE,
        RAW_TUSHARE_PROXY_STOCK_BASIC,
        RAW_TUSHARE_PROXY_TRADE_CALENDAR,
    }
)


@dataclass(frozen=True)
class RawCompactionResult:
    raw_domain: str
    source_partition_count: int
    source_row_count: int
    empty_partition_count: int
    indexed_row_count: int
    parts: tuple[BundlePartRef, ...]
    raw_index_export: RawIndexExport | None = None
    deleted_source_partition_count: int = 0

    @property
    def created_part_count(self) -> int:
        return sum(1 for part in self.parts if part.created)

    @property
    def reused_part_count(self) -> int:
        return sum(1 for part in self.parts if not part.created)


def _fallback_year(ref: RawPartitionRef) -> str:
    match = _YEAR_PATTERN.search(str(ref.partition_value))
    return match.group(0) if match else "undated"


def _row_years(frame: pd.DataFrame, ref: RawPartitionRef) -> pd.Series:
    fallback = _fallback_year(ref)
    for column in _YEAR_COLUMNS:
        if column not in frame.columns:
            continue
        extracted = frame[column].astype("string").str.extract(r"((?:19|20)\d{2})", expand=False)
        if extracted.notna().any():
            return extracted.fillna(fallback).astype(str)
    return pd.Series([fallback] * len(frame), index=frame.index, dtype="string")


def _row_buckets(frame: pd.DataFrame, ref: RawPartitionRef) -> pd.Series:
    fallback = str(ref.partition_value)
    identity: pd.Series | None = None
    for column in _IDENTITY_COLUMNS:
        if column in frame.columns:
            identity = frame[column].astype("string").fillna("")
            break
    if identity is None:
        identity = pd.Series([fallback] * len(frame), index=frame.index, dtype="string")
    else:
        identity = identity.mask(identity.str.strip().eq(""), fallback)
    mapping = {
        value: stable_raw_bucket(value, bucket_count=DEFAULT_RAW_BUCKET_COUNT)
        for value in identity.drop_duplicates().tolist()
    }
    return identity.map(mapping).astype("int64")


def _is_intraday_5m_domain(raw_domain: str) -> bool:
    return "intraday_5m" in str(raw_domain).strip().lower()


def _segment_years(frame: pd.DataFrame, ref: RawPartitionRef) -> pd.Series:
    if ref.raw_domain in _UNDATED_REFERENCE_DOMAINS:
        return pd.Series(["undated"] * len(frame), index=frame.index, dtype="string")
    return _row_years(frame, ref)


def _segment_buckets(frame: pd.DataFrame, ref: RawPartitionRef) -> pd.Series:
    if _is_intraday_5m_domain(ref.raw_domain):
        return _row_buckets(frame, ref)
    return pd.Series([0] * len(frame), index=frame.index, dtype="int64")


def _receipt_provider(receipt: Mapping[str, Any]) -> str:
    for key in ("provider", "provider_name", "source", "historical_provider"):
        value = str(receipt.get(key, "") or "").strip()
        if value:
            return value
    return "unknown"


def _receipt_response_sha(receipt: Mapping[str, Any], ref: RawPartitionRef) -> str:
    for key in ("response_sha256", "raw_response_sha256", "content_sha256"):
        value = str(receipt.get(key, "") or "").strip()
        if value:
            return value
    return ref.content_sha256


def _request_range(ref: RawPartitionRef, receipt: Mapping[str, Any]) -> str:
    payload: dict[str, str] = {
        "partition_field": str(ref.partition_field),
        "partition_value": str(ref.partition_value),
    }
    for key in (
        "api_name",
        "endpoint",
        "query_date",
        "start_date",
        "end_date",
        "start_at",
        "end_at",
        "provider_symbol",
    ):
        value = str(receipt.get(key, "") or "").strip()
        if value:
            payload[key] = value
    return json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def iter_legacy_raw_versions(
    raw_domain: str,
    *,
    workspace_root: str | Path | None = None,
) -> list[RawPartitionRef]:
    """Return every immutable legacy version, not only each latest pointer."""

    paths = ensure_qdp_v3_layout(workspace_root)
    domain_root = paths.raw / str(raw_domain)
    if not domain_root.exists():
        return []
    refs: list[RawPartitionRef] = []
    for receipt_path in sorted(domain_root.glob("*/versions/*/receipt.json")):
        payload_path = receipt_path.parent / "payload.parquet"
        if not payload_path.exists():
            raise RuntimeError(f"raw_compaction_payload_missing:{payload_path}")
        receipt = read_raw_receipt(receipt_path)
        content_sha = str(receipt.get("content_sha256", "") or "")
        if not content_sha:
            raise RuntimeError(f"raw_compaction_content_hash_missing:{receipt_path}")
        assessment_path = str(receipt.get("quality_assessment_path", "") or "")
        refs.append(
            RawPartitionRef(
                raw_domain=str(receipt.get("raw_domain", raw_domain) or raw_domain),
                partition_field=str(receipt.get("partition_field", "query_date") or "query_date"),
                partition_value=str(receipt.get("partition_value", "") or ""),
                content_sha256=content_sha,
                payload_path=payload_path,
                receipt_path=receipt_path,
                row_count=int(receipt.get("row_count", 0) or 0),
                quality_tier=str(receipt.get("quality_tier", "quarantined") or "quarantined"),
                revision_of=str(receipt.get("revision_of", "") or ""),
                quality_assessment_path=Path(assessment_path) if assessment_path else None,
                quality_assessment_sha256=str(
                    receipt.get("quality_assessment_sha256", "") or ""
                ),
            )
        )
    return refs


def _segments_for_ref(ref: RawPartitionRef, frame: pd.DataFrame) -> Iterable[tuple[str, int, BundleSegment]]:
    receipt = read_raw_receipt(ref)
    common = {
        "domain": ref.raw_domain,
        "partition_key": f"{ref.partition_field}={ref.partition_value}",
        "provider": _receipt_provider(receipt),
        "request_range": _request_range(ref, receipt),
        "response_sha256": _receipt_response_sha(receipt, ref),
        "source_content_sha256": ref.content_sha256,
    }
    if frame.empty:
        yield (
            "undated" if ref.raw_domain in _UNDATED_REFERENCE_DOMAINS else _fallback_year(ref),
            stable_raw_bucket(ref.partition_value) if _is_intraday_5m_domain(ref.raw_domain) else 0,
            BundleSegment(frame=frame.copy(), **common),
        )
        return
    years = _segment_years(frame, ref)
    buckets = _segment_buckets(frame, ref)
    grouping = pd.DataFrame({"year": years, "bucket": buckets}, index=frame.index)
    for ordinal, ((year, bucket), indices) in enumerate(grouping.groupby(["year", "bucket"], sort=True).groups.items()):
        yield (
            str(year),
            int(bucket),
            BundleSegment(
                frame=frame.loc[indices].copy(),
                segment_ordinal=ordinal,
                **common,
            ),
        )


def verify_bundle_parts(parts: Iterable[BundlePartRef], *, runtime_index: RuntimeIndex) -> None:
    for part in parts:
        if not part.path.exists():
            raise RuntimeError(f"raw_bundle_missing:{part.path}")
        if sha256_file(part.path) != part.sha256:
            raise RuntimeError(f"raw_bundle_hash_mismatch:{part.path}")
        metadata = pq.read_metadata(part.path)
        if int(metadata.num_rows) != int(part.row_count):
            raise RuntimeError(f"raw_bundle_row_count_mismatch:{part.path}")
        if int(metadata.num_row_groups) != int(part.row_group_count):
            raise RuntimeError(f"raw_bundle_row_group_count_mismatch:{part.path}")
        records = runtime_index.raw_records_for_bundle(part.relative_path)
        expected_groups = set(range(part.row_group_count))
        actual_groups = {int(record["row_group"]) for record in records}
        if actual_groups != expected_groups:
            raise RuntimeError(f"raw_bundle_index_row_groups_mismatch:{part.path}")
        if sum(int(record["row_count"]) for record in records) != int(part.row_count):
            raise RuntimeError(f"raw_bundle_index_row_count_mismatch:{part.path}")
        if any(str(record["bundle_sha256"]) != part.sha256 for record in records):
            raise RuntimeError(f"raw_bundle_index_hash_mismatch:{part.path}")


def _verified_catalog_parts_for_legacy_sources(
    selected: list[RawPartitionRef],
    *,
    raw_domain: str,
    paths: Any,
    runtime_index: RuntimeIndex,
) -> tuple[tuple[BundlePartRef, ...], int, int] | None:
    """Use an already exported bundle catalog to retire unchanged legacy bytes."""

    if not selected:
        return None
    receipts = runtime_index.table_frame("raw_receipts")
    mappings = runtime_index.table_frame("raw_partitions")
    receipts = receipts.loc[receipts["domain"].astype(str).eq(str(raw_domain))]
    mappings = mappings.loc[mappings["domain"].astype(str).eq(str(raw_domain))]
    receipt_keys = {
        (str(row.partition_key), str(row.source_content_sha256)): row
        for row in receipts.itertuples(index=False)
    }
    mapping_groups = {
        (str(key), str(content_sha)): group.copy()
        for (key, content_sha), group in mappings.groupby(
            ["partition_key", "source_content_sha256"],
            sort=False,
        )
    }
    expected_keys = {
        (f"{ref.partition_field}={ref.partition_value}", ref.content_sha256)
        for ref in selected
    }
    if not expected_keys.issubset(receipt_keys) or not expected_keys.issubset(mapping_groups):
        return None

    source_rows = 0
    empty_count = 0
    selected_bundle_paths: set[str] = set()
    for ref in selected:
        key = f"{ref.partition_field}={ref.partition_value}"
        identity = (key, ref.content_sha256)
        receipt_row = receipt_keys[identity]
        current_receipt = read_raw_receipt(ref)
        try:
            indexed_receipt = json.loads(str(receipt_row.receipt_json))
        except (TypeError, ValueError) as exc:
            raise RuntimeError(f"raw_compaction_catalog_receipt_invalid:{raw_domain}:{key}") from exc
        if any(indexed_receipt.get(name) != value for name, value in current_receipt.items()):
            raise RuntimeError(f"raw_compaction_catalog_receipt_changed:{raw_domain}:{key}")
        expected_parquet_sha = str(current_receipt.get("parquet_sha256", "") or "")
        if not expected_parquet_sha or sha256_file(ref.payload_path) != expected_parquet_sha:
            raise RuntimeError(f"raw_compaction_source_parquet_changed:{ref.payload_path}")
        if int(receipt_row.row_count) != int(ref.row_count):
            raise RuntimeError(f"raw_compaction_catalog_receipt_row_count_mismatch:{raw_domain}:{key}")

        group = mapping_groups[identity]
        statuses = set(group["status"].astype(str))
        mapped_rows = int(group["row_count"].astype("int64").sum())
        if mapped_rows != int(ref.row_count):
            raise RuntimeError(f"raw_compaction_catalog_mapping_row_count_mismatch:{raw_domain}:{key}")
        if int(ref.row_count) == 0:
            if statuses != {"empty_success"} or len(group) != 1:
                raise RuntimeError(f"raw_compaction_catalog_empty_mapping_invalid:{raw_domain}:{key}")
            empty_count += 1
        else:
            if statuses != {"bundled"} or group["bundle_path"].astype(str).eq("").any():
                raise RuntimeError(f"raw_compaction_catalog_mapping_invalid:{raw_domain}:{key}")
            selected_bundle_paths.update(group["bundle_path"].astype(str))
        source_rows += int(ref.row_count)

    bundle_root = (paths.raw_bundles / str(raw_domain)).resolve()
    parts: list[BundlePartRef] = []
    for relative_path in sorted(selected_bundle_paths):
        path = (paths.root / relative_path).resolve()
        if not path.is_relative_to(bundle_root):
            raise RuntimeError(f"raw_compaction_catalog_bundle_outside_domain:{path}")
        group = mappings.loc[mappings["bundle_path"].astype(str).eq(relative_path)]
        bundle_hashes = set(group["bundle_sha256"].astype(str))
        if len(bundle_hashes) != 1:
            raise RuntimeError(f"raw_compaction_catalog_bundle_hash_conflict:{path}")
        expected_hash = next(iter(bundle_hashes))
        if not path.exists() or sha256_file(path) != expected_hash:
            raise RuntimeError(f"raw_compaction_catalog_bundle_hash_mismatch:{path}")
        metadata = pq.read_metadata(path)
        expected_groups = set(range(int(metadata.num_row_groups)))
        actual_groups = set(group["row_group"].astype("int64"))
        if actual_groups != expected_groups:
            raise RuntimeError(f"raw_compaction_catalog_bundle_row_groups_mismatch:{path}")
        if int(group["row_count"].astype("int64").sum()) != int(metadata.num_rows):
            raise RuntimeError(f"raw_compaction_catalog_bundle_rows_mismatch:{path}")
        components = Path(relative_path).parts
        year_component = next((item for item in components if item.startswith("year=")), "year=undated")
        bucket_component = next(
            (item for item in components if item.startswith("security_bucket=")),
            "security_bucket=00",
        )
        parts.append(
            BundlePartRef(
                path=path,
                relative_path=str(relative_path),
                sha256=expected_hash,
                row_count=int(metadata.num_rows),
                row_group_count=int(metadata.num_row_groups),
                year=year_component.split("=", 1)[1],
                security_bucket=int(bucket_component.split("=", 1)[1]),
                created=False,
            )
        )
    return tuple(parts), source_rows, empty_count


def compact_raw_partitions(
    refs: Iterable[RawPartitionRef],
    *,
    raw_domain: str,
    workspace_root: str | Path | None = None,
    runtime_index: RuntimeIndex | None = None,
    sizing: BundleSizingPolicy | None = None,
    export_index_dir: str | Path | None = None,
    delete_sources: bool = False,
    yes: bool = False,
) -> RawCompactionResult:
    """Verify and bundle immutable raw partitions, optionally retiring legacy files."""

    if delete_sources and not yes:
        raise ValueError("raw_compaction_source_deletion_requires_yes")
    if delete_sources and export_index_dir is None:
        raise ValueError("raw_compaction_source_deletion_requires_index_export")
    paths = ensure_qdp_v3_layout(workspace_root)
    index = runtime_index or RuntimeIndex(workspace_root=workspace_root)
    selected = sorted(
        (ref for ref in refs if ref.raw_domain == str(raw_domain)),
        key=lambda ref: (ref.partition_value, ref.content_sha256),
    )
    if delete_sources:
        verified = _verified_catalog_parts_for_legacy_sources(
            selected,
            raw_domain=str(raw_domain),
            paths=paths,
            runtime_index=index,
        )
        if verified is not None:
            parts, source_rows, empty_count = verified
            export = index.export_raw_index(export_index_dir)
            domain_root = (paths.raw / str(raw_domain)).resolve()
            partition_dirs = sorted({ref.receipt_path.resolve().parents[2] for ref in selected})
            for partition_dir in partition_dirs:
                if partition_dir.parent != domain_root:
                    raise RuntimeError(f"raw_compaction_delete_path_outside_domain:{partition_dir}")
            for partition_dir in partition_dirs:
                shutil.rmtree(partition_dir)
            return RawCompactionResult(
                raw_domain=str(raw_domain),
                source_partition_count=len(selected),
                source_row_count=source_rows,
                empty_partition_count=empty_count,
                indexed_row_count=source_rows,
                parts=parts,
                raw_index_export=export,
                deleted_source_partition_count=len(partition_dirs),
            )

    writer = RawBundleWriter(paths=paths, runtime_index=index, sizing=sizing)
    source_rows = 0
    empty_count = 0
    try:
        for ref in selected:
            frame = read_raw_partition(ref, verify_hash=True)
            if int(len(frame)) != int(ref.row_count):
                raise RuntimeError(f"raw_compaction_source_row_count_mismatch:{ref.payload_path}")
            source_rows += int(len(frame))
            empty_count += int(frame.empty)
            receipt = read_raw_receipt(ref)
            partition_key = f"{ref.partition_field}={ref.partition_value}"
            index.record_raw_receipt(
                domain=ref.raw_domain,
                partition_key=partition_key,
                provider=_receipt_provider(receipt),
                request_range=_request_range(ref, receipt),
                row_count=int(ref.row_count),
                response_sha256=_receipt_response_sha(receipt, ref),
                source_content_sha256=ref.content_sha256,
                quality_tier=ref.quality_tier,
                receipt=receipt,
            )
            for year, bucket, segment in _segments_for_ref(ref, frame):
                writer.add(year=year, security_bucket=bucket, segment=segment)
        write_result = writer.finish()
        if write_result.indexed_rows != source_rows:
            raise RuntimeError(
                f"raw_compaction_indexed_row_count_mismatch:{write_result.indexed_rows}:{source_rows}"
            )
        verify_bundle_parts(write_result.parts, runtime_index=index)
        export = index.export_raw_index(export_index_dir) if export_index_dir is not None else None
    except BaseException:
        writer.abort(remove_created_parts=True)
        index.restore_raw_domain_from_verified_export(str(raw_domain))
        raise
    deleted_partition_count = 0
    if delete_sources:
        receipt_rows = index.table_frame("raw_receipts")
        indexed_keys = {
            (str(row.domain), str(row.partition_key), str(row.source_content_sha256))
            for row in receipt_rows.itertuples(index=False)
        }
        expected_keys = {
            (
                ref.raw_domain,
                f"{ref.partition_field}={ref.partition_value}",
                ref.content_sha256,
            )
            for ref in selected
        }
        missing_receipts = sorted(expected_keys - indexed_keys)
        if missing_receipts:
            raise RuntimeError(
                f"raw_compaction_receipt_index_incomplete:{missing_receipts[:5]}"
            )
        domain_root = (paths.raw / str(raw_domain)).resolve()
        partition_dirs = sorted({ref.receipt_path.resolve().parents[2] for ref in selected})
        for partition_dir in partition_dirs:
            if partition_dir.parent != domain_root:
                raise RuntimeError(f"raw_compaction_delete_path_outside_domain:{partition_dir}")
        for partition_dir in partition_dirs:
            shutil.rmtree(partition_dir)
            deleted_partition_count += 1
    return RawCompactionResult(
        raw_domain=str(raw_domain),
        source_partition_count=len(selected),
        source_row_count=source_rows,
        empty_partition_count=empty_count,
        indexed_row_count=write_result.indexed_rows,
        parts=write_result.parts,
        raw_index_export=export,
        deleted_source_partition_count=deleted_partition_count,
    )


def compact_raw_domain(
    raw_domain: str,
    *,
    workspace_root: str | Path | None = None,
    runtime_index: RuntimeIndex | None = None,
    sizing: BundleSizingPolicy | None = None,
    export_index_dir: str | Path | None = None,
    include_revisions: bool = True,
    delete_sources: bool = False,
    yes: bool = False,
) -> RawCompactionResult:
    refs = (
        iter_legacy_raw_versions(raw_domain, workspace_root=workspace_root)
        if include_revisions
        else iter_raw_partitions(raw_domain, workspace_root=workspace_root)
    )
    return compact_raw_partitions(
        refs,
        raw_domain=raw_domain,
        workspace_root=workspace_root,
        runtime_index=runtime_index,
        sizing=sizing,
        export_index_dir=export_index_dir,
        delete_sources=delete_sources,
        yes=yes,
    )
