"""Historical Intraday External Archive Repair: install responsibilities."""

from __future__ import annotations

import hashlib
import os
import shutil
from collections.abc import Mapping, Sequence
from dataclasses import replace
from pathlib import Path
from typing import Any

import duckdb
import pyarrow.parquet as pq

from quantlab.data.qdp_v2.manifest import (
    DatasetManifest,
    ShardManifestEntry,
    dataset_manifest_for_id,
    path_for_manifest,
    qdp_v2_root,
    read_active_manifest,
    read_dataset_manifest,
    utc_now,
    write_active_manifest,
    write_dataset_manifest,
)
from quantlab.data.qdp_v2.recent_market_repair import (
    INTRADAY_DOMAIN,
)
from quantlab.data.qdp_v2.research_event_update import (
    _assert_credential_free,
    _sha256,
)
from quantlab.data.qdp_v2.status import active_dataset_map

from .config import (
    END_DATE,
    EXPECTED_CANDIDATE_DAYS,
    REPAIR_ID,
    SOURCE_NAME,
    ExternalArchiveRepairError,
)


def _dataset_id(input_dataset_id: str, bundle_paths: Sequence[Path]) -> str:
    digest = hashlib.sha256(input_dataset_id.encode("utf-8"))
    for path in bundle_paths:
        digest.update(_sha256(path).encode("ascii"))
    return f"{INTRADAY_DOMAIN}__{digest.hexdigest()[:24]}"


def _entry_for_installed(path: Path, *, root: Path, source_hash: str) -> ShardManifestEntry:
    parquet = pq.ParquetFile(path)
    connection = duckdb.connect()
    try:
        start_date, end_date = connection.execute(
            "SELECT min(trade_date),max(trade_date) FROM read_parquet(?,hive_partitioning=false)",
            [str(path)],
        ).fetchone()
    finally:
        connection.close()
    return ShardManifestEntry(
        path=path_for_manifest(path, root=root),
        row_count=int(parquet.metadata.num_rows),
        start_date=str(start_date),
        end_date=str(end_date),
        status="stored",
        file_size=path.stat().st_size,
        metadata={
            "source": SOURCE_NAME,
            "source_sha256": source_hash,
            "created_at": utc_now(),
        },
    )


def _install_immutable_version(
    workspace: Path,
    *,
    input_manifest: DatasetManifest,
    bundle_paths: Sequence[Path],
    validation: Mapping[str, Any],
) -> tuple[str, list[Path]]:
    root = qdp_v2_root(workspace)
    dataset_id = _dataset_id(input_manifest.dataset_id, bundle_paths)
    target_root = root / "datasets" / INTRADAY_DOMAIN / dataset_id / "shards"
    installed: list[Path] = []
    entries: list[ShardManifestEntry] = []
    for source_path in bundle_paths:
        year = source_path.parent.name
        target = target_root / "external_archive_v2" / year / "part-0000.parquet"
        source_hash = _sha256(source_path)
        if target.is_file():
            if _sha256(target) != source_hash:
                raise ExternalArchiveRepairError(f"installed_bundle_hash_conflict:{target}")
        else:
            target.parent.mkdir(parents=True, exist_ok=True)
            temporary = target.with_suffix(target.suffix + ".partial")
            temporary.unlink(missing_ok=True)
            shutil.copy2(source_path, temporary)
            if _sha256(temporary) != source_hash:
                temporary.unlink(missing_ok=True)
                raise ExternalArchiveRepairError(f"installed_bundle_copy_failed:{target}")
            os.replace(temporary, target)
        installed.append(target)
        entries.append(_entry_for_installed(target, root=root, source_hash=source_hash))
    new_manifest = replace(
        input_manifest,
        dataset_id=dataset_id,
        row_count=input_manifest.row_count + sum(item.row_count for item in entries),
        shards=[*input_manifest.shards, *entries],
        source={
            **input_manifest.source,
            "historical_external_archive_repair": REPAIR_ID,
            "historical_external_archive_provider": SOURCE_NAME,
            "historical_external_archive_checked_through": END_DATE,
            "historical_external_archive_eligibility_filter": False,
            "historical_external_archive_request_2026_count": 0,
        },
        quality={
            **input_manifest.quality,
            "historical_external_archive_candidate_days": EXPECTED_CANDIDATE_DAYS,
            "historical_external_archive_accepted_days": int(validation["decision_counts"]["accepted"]),
            "historical_external_archive_rejected_days": int(validation["decision_counts"]["rejected"]),
            "historical_external_archive_no_file_days": int(validation["decision_counts"]["no_file"]),
            "quality_liquidity_complete_pit_rows_2011_2025": int(validation["formal_quality_complete_pool_row_count"]),
            "external_archive_existing_keys_overwritten": 0,
            "external_archive_2026_rows_written": 0,
        },
        created_at=utc_now(),
        notes=[
            *input_manifest.notes,
            "External archive is used only to fill the frozen missing stock-day inventory.",
            "A 09:30 auction row is merged into 09:35 only for the exact canonical 49-row pattern.",
            "Daily OHLC must pass 2%; volume or amount must pass 5%; no bars are interpolated.",
            "Archive file presence never changes PIT universe membership.",
        ],
    )
    _assert_credential_free(new_manifest.to_dict())
    write_dataset_manifest(root, new_manifest)
    reread = read_dataset_manifest(dataset_manifest_for_id(root, dataset_id, INTRADAY_DOMAIN))
    if (
        reread.dataset_id != dataset_id
        or reread.row_count != new_manifest.row_count
        or len(reread.shards) != len(new_manifest.shards)
    ):
        raise ExternalArchiveRepairError("immutable_intraday_manifest_validation_failed")
    active = read_active_manifest(root)
    current = active_dataset_map(active)
    if current.get(INTRADAY_DOMAIN) != input_manifest.dataset_id:
        if current.get(INTRADAY_DOMAIN) == dataset_id:
            return dataset_id, installed
        raise ExternalArchiveRepairError("active_intraday_drifted_before_commit")
    active["datasets"] = {**current, INTRADAY_DOMAIN: dataset_id}
    active["updated_at"] = utc_now()
    write_active_manifest(root, active)
    if active_dataset_map(read_active_manifest(root)).get(INTRADAY_DOMAIN) != dataset_id:
        raise ExternalArchiveRepairError("active_intraday_atomic_switch_failed")
    return dataset_id, installed
