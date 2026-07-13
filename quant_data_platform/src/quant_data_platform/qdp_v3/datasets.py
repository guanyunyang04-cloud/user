from __future__ import annotations

import os
import shutil
import uuid
import hashlib
from pathlib import Path
from typing import Any, Iterable, Mapping

import pandas as pd

from quant_data_platform.qdp_v3.manifest import (
    DatasetInputRef,
    DatasetManifestV3,
    ProviderEvidence,
    ShardManifestV3,
    atomic_write_json,
    dataset_id_for,
    dataset_manifest_for_id,
    manifest_sha256,
    path_for_manifest,
    read_dataset_manifest,
    resolve_manifest_path,
    schema_from_frame,
    schema_hash_for,
    sha256_file,
)
from quant_data_platform.qdp_v3.quality import QualityReport
from quant_data_platform.qdp_v3.storage import atomic_write_parquet, frame_content_sha256


def dataset_input_ref(root: str | Path, manifest: DatasetManifestV3) -> DatasetInputRef:
    path = dataset_manifest_for_id(root, manifest.dataset_id, manifest.domain)
    if path is None:
        raise FileNotFoundError(f"dataset_manifest_missing:{manifest.dataset_id}")
    return DatasetInputRef(domain=manifest.domain, dataset_id=manifest.dataset_id, manifest_sha256=manifest_sha256(path))


def _partition_groups(frame: pd.DataFrame, *, date_column: str, partitioning: str) -> list[tuple[str, pd.DataFrame, str, str]]:
    if frame.empty or partitioning == "single" or not date_column or date_column not in frame.columns:
        start = str(frame[date_column].min())[:10] if not frame.empty and date_column in frame.columns else ""
        end = str(frame[date_column].max())[:10] if not frame.empty and date_column in frame.columns else ""
        return [("all", frame.copy(), start, end)]
    dates = frame[date_column].astype(str).str.slice(0, 10)
    if partitioning == "year":
        keys = dates.str.slice(0, 4)
    elif partitioning == "month":
        keys = dates.str.slice(0, 7).str.replace("-", "", regex=False)
    else:
        raise ValueError(f"unsupported_partitioning:{partitioning}")
    groups: list[tuple[str, pd.DataFrame, str, str]] = []
    for key, indices in keys.groupby(keys, sort=True).groups.items():
        part = frame.loc[indices].copy()
        part_dates = part[date_column].astype(str).str.slice(0, 10)
        groups.append((str(key), part, str(part_dates.min()), str(part_dates.max())))
    return groups


def write_dataset(
    *,
    root: str | Path,
    domain: str,
    frame: pd.DataFrame,
    layer: str,
    frequency: str,
    primary_key: list[str],
    quality_report: QualityReport,
    inputs: Iterable[DatasetInputRef] = (),
    provider_evidence: Iterable[ProviderEvidence] = (),
    raw_content_hashes: Iterable[str] = (),
    build: Mapping[str, Any] | None = None,
    coverage: Mapping[str, Any] | None = None,
    units: Mapping[str, str] | None = None,
    quarantine: Iterable[Mapping[str, Any]] = (),
    date_column: str = "trade_date",
    partitioning: str = "year",
) -> DatasetManifestV3:
    root_path = Path(root)
    data = frame.copy()
    missing_key = sorted(set(primary_key) - set(data.columns))
    if missing_key:
        raise ValueError(f"dataset_primary_key_columns_missing:{domain}:{missing_key}")
    if primary_key and not data.empty:
        data = data.sort_values(primary_key, kind="mergesort").reset_index(drop=True)
    schema = schema_from_frame(data)
    schema_hash = schema_hash_for(schema)
    input_list = list(inputs)
    evidence_list = list(provider_evidence)
    raw_hashes = sorted(set(str(item) for item in raw_content_hashes if str(item)))
    build_payload = dict(build or {})
    quality_payload = quality_report.to_dict()
    coverage_payload = dict(coverage or {})
    units_payload = {str(key): str(value) for key, value in dict(units or {}).items()}
    quarantine_list = [dict(item) for item in quarantine]
    content_hash = frame_content_sha256(data)
    seed = {
        "content_sha256": content_hash,
        "schema_hash": schema_hash,
        "layer": str(layer),
        "frequency": str(frequency),
        "primary_key": list(primary_key),
        "date_column": str(date_column),
        "partitioning": str(partitioning),
        "inputs": [item.to_dict() for item in input_list],
        "provider_evidence": [item.to_dict() for item in evidence_list],
        "raw_content_hashes": raw_hashes,
        "build": build_payload,
        "quality_report": quality_payload,
        "coverage": coverage_payload,
        "units": units_payload,
        "quarantine": quarantine_list,
    }
    dataset_id = dataset_id_for(domain, seed)
    existing_path = dataset_manifest_for_id(root_path, dataset_id, domain)
    if existing_path is not None:
        return read_dataset_manifest(existing_path)
    domain_dir = root_path / "datasets" / str(domain)
    target_dir = domain_dir / dataset_id
    if target_dir.exists():
        raise RuntimeError(f"incomplete_or_conflicting_dataset_dir:{target_dir}")
    temp_dir = domain_dir / f".{dataset_id}.{os.getpid()}.{uuid.uuid4().hex}.tmp"
    shards_dir = temp_dir / "shards"
    shards_dir.mkdir(parents=True, exist_ok=False)
    shard_entries: list[ShardManifestV3] = []
    try:
        groups = _partition_groups(data, date_column=date_column, partitioning=partitioning)
        for index, (partition_key, part, start_date, end_date) in enumerate(groups):
            safe_key = str(partition_key).replace("/", "_").replace("\\", "_")
            filename = f"part_{index:04d}_{safe_key}.parquet"
            temp_path = shards_dir / filename
            atomic_write_parquet(temp_path, part)
            final_path = target_dir / "shards" / filename
            shard_entries.append(
                ShardManifestV3(
                    path=path_for_manifest(final_path, root=root_path),
                    sha256=sha256_file(temp_path),
                    row_count=int(len(part)),
                    start_date=start_date,
                    end_date=end_date,
                    partition={"kind": partitioning, "value": partition_key},
                    file_size=int(temp_path.stat().st_size),
                    schema_hash=schema_hash,
                )
            )
        if data.empty:
            start_date = ""
            end_date = ""
        elif date_column and date_column in data.columns:
            normalized_dates = data[date_column].astype(str).str.slice(0, 10)
            start_date = str(normalized_dates.min())
            end_date = str(normalized_dates.max())
        else:
            start_date = ""
            end_date = ""
        manifest = DatasetManifestV3(
            dataset_id=dataset_id,
            domain=str(domain),
            layer=str(layer),
            frequency=str(frequency),
            primary_key=list(primary_key),
            quality_tier=quality_report.quality_tier,
            start_date=start_date,
            end_date=end_date,
            row_count=int(len(data)),
            schema=schema,
            schema_hash=schema_hash,
            shards=shard_entries,
            inputs=input_list,
            provider_evidence=evidence_list,
            raw_content_hashes=raw_hashes,
            build=build_payload,
            quality_report=quality_payload,
            blockers=[item.to_dict() for item in quality_report.blockers],
            quarantine=quarantine_list,
            coverage=coverage_payload,
            units=units_payload,
        )
        atomic_write_json(temp_dir / "dataset.json", manifest.to_dict())
        domain_dir.mkdir(parents=True, exist_ok=True)
        temp_dir.replace(target_dir)
        return manifest
    except BaseException:
        if temp_dir.exists():
            shutil.rmtree(temp_dir, ignore_errors=True)
        raise


def write_partitioned_dataset(
    *,
    root: str | Path,
    domain: str,
    partition_frames: Iterable[tuple[str, str, pd.DataFrame]],
    layer: str,
    frequency: str,
    primary_key: list[str],
    quality_report: QualityReport,
    partitioning: str,
    inputs: Iterable[DatasetInputRef] = (),
    provider_evidence: Iterable[ProviderEvidence] = (),
    raw_content_hashes: Iterable[str] = (),
    build: Mapping[str, Any] | None = None,
    coverage: Mapping[str, Any] | None = None,
    units: Mapping[str, str] | None = None,
    quarantine: Iterable[Mapping[str, Any]] = (),
    date_column: str = "trade_date",
) -> DatasetManifestV3:
    """Write a deterministic dataset one bounded partition at a time.

    The caller must yield ``(shard_key, partition_value, frame)`` in stable
    order. This keeps full-history intraday builds bounded by one symbol-month
    (or other caller-selected unit) rather than materializing the dataset.
    """

    root_path = Path(root)
    domain_dir = root_path / "datasets" / str(domain)
    staging = domain_dir / f".stream.{os.getpid()}.{uuid.uuid4().hex}.tmp"
    shards_dir = staging / "shards"
    shards_dir.mkdir(parents=True, exist_ok=False)
    schema: list[dict[str, str]] | None = None
    schema_hash = ""
    logical_digest = hashlib.sha256()
    staged: list[dict[str, Any]] = []
    row_count = 0
    dataset_start = ""
    dataset_end = ""
    try:
        for index, (shard_key, partition_value, source_frame) in enumerate(partition_frames):
            frame = source_frame.copy()
            missing_key = sorted(set(primary_key) - set(frame.columns))
            if missing_key:
                raise ValueError(f"dataset_primary_key_columns_missing:{domain}:{missing_key}")
            if primary_key and not frame.empty:
                frame = frame.sort_values(primary_key, kind="mergesort").reset_index(drop=True)
            current_schema = schema_from_frame(frame)
            if schema is None:
                schema = current_schema
                schema_hash = schema_hash_for(schema)
            elif current_schema != schema:
                raise ValueError(f"stream_dataset_schema_mismatch:{domain}:{shard_key}")
            safe_key = str(shard_key).replace("/", "_").replace("\\", "_").replace(":", "_")
            filename = f"part_{index:06d}_{safe_key}.parquet"
            temp_path = shards_dir / filename
            atomic_write_parquet(temp_path, frame)
            if frame.empty or not date_column or date_column not in frame.columns:
                part_start = ""
                part_end = ""
            else:
                dates = frame[date_column].astype(str).str.slice(0, 10)
                part_start = str(dates.min())
                part_end = str(dates.max())
                dataset_start = min(filter(None, (dataset_start, part_start)), default="")
                dataset_end = max(filter(None, (dataset_end, part_end)), default="")
            content_hash = frame_content_sha256(frame)
            logical_digest.update(f"{shard_key}\0{partition_value}\0{content_hash}\0{len(frame)}\n".encode("utf-8"))
            staged.append(
                {
                    "filename": filename,
                    "temp_path": temp_path,
                    "sha256": sha256_file(temp_path),
                    "row_count": int(len(frame)),
                    "start_date": part_start,
                    "end_date": part_end,
                    "partition_value": str(partition_value),
                    "source_partition": str(shard_key),
                    "file_size": int(temp_path.stat().st_size),
                }
            )
            row_count += int(len(frame))
        if schema is None:
            raise ValueError(f"stream_dataset_requires_at_least_one_partition:{domain}")
        input_list = list(inputs)
        evidence_list = list(provider_evidence)
        raw_hashes = sorted(set(str(item) for item in raw_content_hashes if str(item)))
        build_payload = dict(build or {})
        quality_payload = quality_report.to_dict()
        coverage_payload = dict(coverage or {})
        units_payload = {str(key): str(value) for key, value in dict(units or {}).items()}
        quarantine_list = [dict(item) for item in quarantine]
        seed = {
            "stream_content_sha256": logical_digest.hexdigest(),
            "schema_hash": schema_hash,
            "layer": str(layer),
            "frequency": str(frequency),
            "primary_key": list(primary_key),
            "date_column": str(date_column),
            "partitioning": str(partitioning),
            "inputs": [item.to_dict() for item in input_list],
            "provider_evidence": [item.to_dict() for item in evidence_list],
            "raw_content_hashes": raw_hashes,
            "build": build_payload,
            "quality_report": quality_payload,
            "coverage": coverage_payload,
            "units": units_payload,
            "quarantine": quarantine_list,
        }
        dataset_id = dataset_id_for(domain, seed)
        existing_path = dataset_manifest_for_id(root_path, dataset_id, domain)
        if existing_path is not None:
            shutil.rmtree(staging, ignore_errors=True)
            return read_dataset_manifest(existing_path)
        target_dir = domain_dir / dataset_id
        if target_dir.exists():
            raise RuntimeError(f"incomplete_or_conflicting_dataset_dir:{target_dir}")
        shard_entries = [
            ShardManifestV3(
                path=path_for_manifest(target_dir / "shards" / item["filename"], root=root_path),
                sha256=item["sha256"],
                row_count=item["row_count"],
                start_date=item["start_date"],
                end_date=item["end_date"],
                partition={
                    "kind": str(partitioning),
                    "value": item["partition_value"],
                    "source_partition": item["source_partition"],
                },
                file_size=item["file_size"],
                schema_hash=schema_hash,
            )
            for item in staged
        ]
        manifest = DatasetManifestV3(
            dataset_id=dataset_id,
            domain=str(domain),
            layer=str(layer),
            frequency=str(frequency),
            primary_key=list(primary_key),
            quality_tier=quality_report.quality_tier,
            start_date=dataset_start,
            end_date=dataset_end,
            row_count=row_count,
            schema=schema,
            schema_hash=schema_hash,
            shards=shard_entries,
            inputs=input_list,
            provider_evidence=evidence_list,
            raw_content_hashes=raw_hashes,
            build=build_payload,
            quality_report=quality_payload,
            blockers=[item.to_dict() for item in quality_report.blockers],
            quarantine=quarantine_list,
            coverage=coverage_payload,
            units=units_payload,
        )
        atomic_write_json(staging / "dataset.json", manifest.to_dict())
        domain_dir.mkdir(parents=True, exist_ok=True)
        staging.replace(target_dir)
        return manifest
    except BaseException:
        if staging.exists():
            shutil.rmtree(staging, ignore_errors=True)
        raise


def read_dataset_frame(root: str | Path, manifest: DatasetManifestV3, *, verify_hashes: bool = False) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for shard in manifest.shards:
        path = resolve_manifest_path(shard.path, root=root)
        if not path.exists():
            raise FileNotFoundError(f"dataset_shard_missing:{manifest.dataset_id}:{path}")
        if verify_hashes and sha256_file(path) != shard.sha256:
            raise RuntimeError(f"dataset_shard_hash_mismatch:{manifest.dataset_id}:{path}")
        frames.append(pd.read_parquet(path, engine="pyarrow"))
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=[item["name"] for item in manifest.schema])
