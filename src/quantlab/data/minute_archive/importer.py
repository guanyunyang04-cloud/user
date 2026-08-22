"""Transactional installation of a complete minute-archive year into QDP."""

from __future__ import annotations

import json
import os
import shutil
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from zipfile import ZipFile

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from quantlab.core.io import sha256_file, write_json
from quantlab.data.core.paths import qdp_paths, workspace_root
from quantlab.data.minute_archive.contracts import (
    AUCTION_TIME,
    BAR_SCHEMA,
    CONTINUOUS_TIMES,
    PRICE_MODE,
    SOURCE_NAME,
    MinuteArchiveError,
)
from quantlab.data.minute_archive.quality import (
    daily_evidence,
    parity_audit,
    qdp_daily_paths,
    validate_member_frame,
)
from quantlab.data.minute_archive.reader import (
    bar_table,
    list_members,
    read_member_year,
    standardize_frame,
)
from quantlab.data.qdp_v2.manifest import (
    DatasetManifest,
    ShardManifestEntry,
    _manifest_schema_from_arrow,
    read_active_manifest,
    utc_now,
    write_active_manifest,
)


class MonthlyWriters:
    def __init__(self, dataset_dir: Path) -> None:
        self.dataset_dir = dataset_dir
        self.writers: dict[str, pq.ParquetWriter] = {}
        self.paths: dict[str, Path] = {}
        self.rows: dict[str, int] = {}

    def write(self, frame: pd.DataFrame) -> None:
        if frame.empty:
            return
        months = frame["trade_date"].str.slice(0, 7).str.replace("-", "", regex=False)
        for month, selected in frame.groupby(months, sort=True):
            key = str(month)
            writer = self.writers.get(key)
            if writer is None:
                path = (
                    self.dataset_dir
                    / "shards"
                    / f"year={key[:4]}"
                    / f"month={key[4:]}"
                    / "part-0000.parquet"
                )
                path.parent.mkdir(parents=True, exist_ok=True)
                writer = pq.ParquetWriter(
                    path,
                    BAR_SCHEMA,
                    compression="zstd",
                    compression_level=6,
                    use_dictionary=True,
                )
                self.writers[key] = writer
                self.paths[key] = path
                self.rows[key] = 0
            writer.write_table(bar_table(selected), row_group_size=250_000)
            self.rows[key] += len(selected)

    def close(self) -> None:
        for writer in self.writers.values():
            writer.close()
        self.writers.clear()


@dataclass(frozen=True)
class ImportLayout:
    workspace: Path
    qdp_root: Path
    archive: Path
    year: int
    staging: Path
    staged_continuous: Path
    staged_auction: Path
    final_continuous: Path
    final_auction: Path
    quality_target: Path


@dataclass
class StreamedYear:
    continuous: MonthlyWriters
    auction: MonthlyWriters
    counts: pd.DataFrame
    shares: pd.DataFrame
    members: pd.DataFrame
    source_member_count: int
    share_inconsistent_days: int


def _import_layout(
    archive_path: str | Path,
    *,
    year: int,
    workspace_root_value: str | Path | None,
) -> ImportLayout:
    workspace = workspace_root(workspace_root_value)
    paths = qdp_paths(workspace)
    archive = Path(archive_path).resolve()
    if not archive.is_file():
        raise MinuteArchiveError(f"archive_missing:{archive}")
    qdp_root = paths.qdp_v2_dir
    staging = qdp_root / "tmp" / f"minute_import_{int(year)}"
    return ImportLayout(
        workspace=workspace,
        qdp_root=qdp_root,
        archive=archive,
        year=int(year),
        staging=staging,
        staged_continuous=staging / "market_intraday_1m" / "market_intraday_1m",
        staged_auction=staging / "market_opening_auction" / "market_opening_auction",
        final_continuous=qdp_root / "datasets" / "market_intraday_1m" / "market_intraday_1m",
        final_auction=qdp_root / "datasets" / "market_opening_auction" / "market_opening_auction",
        quality_target=paths.source_archives_dir / "minute" / "quality" / f"year={int(year)}",
    )


def _prepare_staging(layout: ImportLayout) -> None:
    if layout.final_continuous.exists() or layout.final_auction.exists() or layout.quality_target.exists():
        raise MinuteArchiveError("formal_minute_dataset_already_exists")
    if layout.staging.exists():
        shutil.rmtree(layout.staging)


def _stream_year(layout: ImportLayout) -> StreamedYear:
    members = list_members([layout.archive])
    if not members:
        raise MinuteArchiveError("archive_has_no_1m_members")
    continuous = MonthlyWriters(layout.staged_continuous)
    auction = MonthlyWriters(layout.staged_auction)
    member_audits: list[dict[str, Any]] = []
    count_parts: list[pd.DataFrame] = []
    share_parts: list[pd.DataFrame] = []
    share_inconsistent_days = 0
    try:
        with ZipFile(layout.archive) as zipped:
            for member in members:
                frame = read_member_year(
                    zipped,
                    zipped.getinfo(member.member),
                    symbol=member.symbol,
                    year=layout.year,
                )
                member_audits.append(validate_member_frame(frame, member=member, year=layout.year))
                if frame.empty:
                    continue
                variation = frame.groupby("trade_date")[["float_shares", "total_shares"]].nunique(
                    dropna=True
                )
                share_inconsistent_days += int((variation > 1).any(axis=1).sum())
                counts, shares = daily_evidence(frame, member.symbol)
                count_parts.append(counts)
                share_parts.append(shares)
                core = standardize_frame(frame, exclude_0930=False)
                continuous.write(core.loc[core["bar_time"].isin(CONTINUOUS_TIMES)])
                auction.write(core.loc[core["bar_time"].eq(AUCTION_TIME)])
    finally:
        continuous.close()
        auction.close()
    if not continuous.paths or not auction.paths or not count_parts:
        raise MinuteArchiveError("formal_import_empty")
    return StreamedYear(
        continuous=continuous,
        auction=auction,
        counts=pd.concat(count_parts, ignore_index=True),
        shares=pd.concat(share_parts, ignore_index=True),
        members=pd.DataFrame(member_audits),
        source_member_count=len(members),
        share_inconsistent_days=share_inconsistent_days,
    )


def _classify_sessions(counts: pd.DataFrame) -> dict[str, int | float]:
    complete = counts["continuous_bar_count"].eq(240)
    suspended = (
        (counts["continuous_bar_count"] == 1)
        & (counts["auction_row_count"] == 1)
        & counts["continuous_volume"].eq(0)
        & counts["continuous_amount"].eq(0)
        & counts["auction_volume"].eq(0)
        & counts["auction_amount"].eq(0)
    )
    counts["session_quality"] = np.select(
        [complete, suspended],
        ["complete_240", "zero_flow_suspension_placeholder"],
        default="unexplained_incomplete",
    )
    complete_days = int(complete.sum())
    suspended_days = int(suspended.sum())
    unexplained = int(len(counts) - complete_days - suspended_days)
    eligible = len(counts) - suspended_days
    return {
        "stock_day_count": len(counts),
        "continuous_full_240_days": complete_days,
        "continuous_full_240_rate": float(complete_days / len(counts)) if len(counts) else 0.0,
        "active_or_normal_session_full_240_rate": float(complete_days / eligible) if eligible else 0.0,
        "zero_flow_single_bar_suspension_placeholder_days": suspended_days,
        "unexplained_incomplete_stock_days": unexplained,
        "opening_auction_stock_days": int((counts["auction_row_count"] == 1).sum()),
        "opening_auction_missing_or_multiple_days": int((counts["auction_row_count"] != 1).sum()),
    }


def _write_quality_tables(streamed: StreamedYear, quality_dir: Path) -> None:
    if streamed.counts.duplicated(["symbol", "trade_date"]).any():
        raise MinuteArchiveError("daily_count_duplicate_keys")
    if streamed.shares.duplicated(["symbol", "trade_date"]).any():
        raise MinuteArchiveError("daily_share_evidence_duplicate_keys")
    quality_dir.mkdir(parents=True)
    streamed.counts.to_parquet(
        quality_dir / "daily_bar_counts.parquet",
        compression="zstd",
        index=False,
    )
    streamed.shares.to_parquet(
        quality_dir / "daily_share_candidates.parquet",
        compression="zstd",
        index=False,
    )
    streamed.members.to_parquet(
        quality_dir / "member_audit.parquet",
        compression="zstd",
        index=False,
    )


def _quality_summary(
    layout: ImportLayout,
    streamed: StreamedYear,
    quality_dir: Path,
) -> dict[str, Any]:
    sessions = _classify_sessions(streamed.counts)
    _write_quality_tables(streamed, quality_dir)
    parity = parity_audit(
        continuous_paths=list(streamed.continuous.paths.values()),
        auction_paths=list(streamed.auction.paths.values()),
        daily_paths=qdp_daily_paths(layout.workspace),
        year=layout.year,
        output_dir=quality_dir,
        final_quality_dir=layout.quality_target,
    )
    quality = {
        "source_member_count": streamed.source_member_count,
        "members_with_rows": int((streamed.members["row_count"] > 0).sum()),
        **sessions,
        "share_inconsistent_stock_days": int(streamed.share_inconsistent_days),
        **parity,
    }
    if (
        parity["duplicate_primary_keys"]
        or streamed.share_inconsistent_days
        or sessions["unexplained_incomplete_stock_days"]
    ):
        raise MinuteArchiveError(f"formal_import_quality_failed:{json.dumps(quality)}")
    return quality


def _date_bounds(counts: pd.DataFrame) -> dict[str, tuple[str, str]]:
    framed = counts.assign(month=counts["trade_date"].str.slice(0, 7).str.replace("-", "", regex=False))
    return {
        str(month): (str(group["trade_date"].min()), str(group["trade_date"].max()))
        for month, group in framed.groupby("month", sort=True)
    }


def _write_dataset_manifest(
    *,
    domain: str,
    staged_dataset: Path,
    final_dataset: Path,
    writers: MonthlyWriters,
    date_bounds: Mapping[str, tuple[str, str]],
    layout: ImportLayout,
    archive_sha256: str,
    quality: Mapping[str, Any],
) -> DatasetManifest:
    shards = [
        ShardManifestEntry(
            path=(final_dataset / path.relative_to(staged_dataset)).relative_to(layout.qdp_root).as_posix(),
            row_count=int(writers.rows[month]),
            start_date=date_bounds[month][0],
            end_date=date_bounds[month][1],
            file_size=path.stat().st_size,
        )
        for month, path in sorted(writers.paths.items())
    ]
    manifest = DatasetManifest(
        dataset_id=domain,
        domain=domain,
        layer="raw",
        frequency="opening_auction" if domain == "market_opening_auction" else "1m",
        contract_version=f"qdp_{domain}_raw",
        primary_key=["symbol", "trade_date", "bar_time"],
        start_date=min(value[0] for value in date_bounds.values()),
        end_date=max(value[1] for value in date_bounds.values()),
        row_count=sum(int(value) for value in writers.rows.values()),
        shards=shards,
        source={
            "provider": SOURCE_NAME,
            "source_archive": str(layout.archive.resolve()),
            "source_archive_sha256": archive_sha256,
            "price_mode": PRICE_MODE,
            "bar_label": "right_end",
        },
        quality={**dict(quality), "primary_key_unique": True},
        schema=_manifest_schema_from_arrow(pq.read_schema(next(iter(writers.paths.values())))),
        notes=[
            "09:30 opening auction is stored separately from continuous trading",
            "a zero-volume 09:30 row is a source placeholder, not proof of an auction transaction",
            "share fields from the source CSV are retained once per stock-day as quality evidence",
        ],
    )
    write_json(staged_dataset / "dataset.json", manifest.to_dict())
    return manifest


def _install(
    layout: ImportLayout,
    streamed: StreamedYear,
    *,
    quality: Mapping[str, Any],
    quality_dir: Path,
    archive_sha256: str,
) -> dict[str, Any]:
    bounds = _date_bounds(streamed.counts)
    continuous = _write_dataset_manifest(
        domain="market_intraday_1m",
        staged_dataset=layout.staged_continuous,
        final_dataset=layout.final_continuous,
        writers=streamed.continuous,
        date_bounds=bounds,
        layout=layout,
        archive_sha256=archive_sha256,
        quality=quality,
    )
    auction = _write_dataset_manifest(
        domain="market_opening_auction",
        staged_dataset=layout.staged_auction,
        final_dataset=layout.final_auction,
        writers=streamed.auction,
        date_bounds=bounds,
        layout=layout,
        archive_sha256=archive_sha256,
        quality=quality,
    )
    layout.final_continuous.parent.mkdir(parents=True, exist_ok=True)
    layout.final_auction.parent.mkdir(parents=True, exist_ok=True)
    os.replace(layout.staged_continuous, layout.final_continuous)
    os.replace(layout.staged_auction, layout.final_auction)
    layout.quality_target.parent.mkdir(parents=True, exist_ok=True)
    os.replace(quality_dir, layout.quality_target)
    audit = {
        "schema": "quantlab.minute_import_audit/1",
        "status": "ready",
        "created_at": utc_now(),
        "year": layout.year,
        "source_archive": str(layout.archive),
        "source_archive_sha256": archive_sha256,
        "continuous": continuous.to_dict(),
        "opening_auction": auction.to_dict(),
        "quality": dict(quality),
        "quality_directory": str(layout.quality_target),
    }
    write_json(layout.quality_target / "audit.json", audit)
    active = read_active_manifest(layout.qdp_root)
    active["datasets"] = {
        **dict(active.get("datasets", {}) or {}),
        "market_intraday_1m": continuous.dataset_id,
        "market_opening_auction": auction.dataset_id,
    }
    active["updated_at"] = utc_now()
    write_active_manifest(layout.qdp_root, active)
    return audit


def import_year(
    archive_path: str | Path,
    *,
    year: int,
    workspace_root_value: str | Path | None = None,
    compute_sha256: bool = True,
) -> dict[str, Any]:
    """Import one complete source year into an empty canonical QDP minute store."""

    layout = _import_layout(
        archive_path,
        year=int(year),
        workspace_root_value=workspace_root_value,
    )
    _prepare_staging(layout)
    try:
        streamed = _stream_year(layout)
        quality_dir = layout.staging / "quality"
        quality = _quality_summary(layout, streamed, quality_dir)
        archive_hash = sha256_file(layout.archive) if compute_sha256 else ""
        return _install(
            layout,
            streamed,
            quality=quality,
            quality_dir=quality_dir,
            archive_sha256=archive_hash,
        )
    finally:
        shutil.rmtree(layout.staging, ignore_errors=True)


__all__ = ["import_year"]
