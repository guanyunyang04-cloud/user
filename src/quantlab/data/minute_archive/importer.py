"""Transactional, partition-safe imports of local minute archives into QDP."""

from __future__ import annotations

import gc
import json
import os
import shutil
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any
from zipfile import ZipFile

import numpy as np
import pandas as pd
import psutil
import pyarrow as pa
import pyarrow.parquet as pq

from quantlab.core.io import read_json, sha256_file, write_json
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
    member_frame_audit,
    parity_audit,
    prepare_member_frame,
    qdp_daily_paths,
    repair_mislabeled_1300_as_1130,
    repair_zero_price_placeholders,
)
from quantlab.data.minute_archive.reader import bar_table, list_members, read_member_years
from quantlab.data.qdp_v2.manifest import (
    DatasetManifest,
    ShardManifestEntry,
    _manifest_schema_from_arrow,
    read_active_manifest,
    read_dataset_manifest,
    utc_now,
    write_active_manifest,
)

CONTINUOUS_DOMAIN = "market_intraday_1m"
AUCTION_DOMAIN = "market_opening_auction"
_MIB = 1024**2
_GIB = 1024**3


@dataclass(frozen=True)
class ImportMemoryPolicy:
    """A conservative RAM budget that still allows large Parquet row groups."""

    total_bytes: int
    available_at_start_bytes: int
    reserve_bytes: int
    writer_budget_bytes: int
    pressure_threshold_bytes: int
    continuous_flush_rows: int
    continuous_max_buffered_rows: int
    auction_flush_rows: int
    auction_max_buffered_rows: int

    def to_dict(self) -> dict[str, int]:
        return {str(key): int(value) for key, value in asdict(self).items()}


def _import_memory_policy() -> ImportMemoryPolicy:
    memory = psutil.virtual_memory()
    total = int(memory.total)
    available = int(memory.available)
    reserve = max(3 * _GIB, total // 4)
    free_headroom = max(0, available - reserve)
    writer_budget = min(_GIB, max(128 * _MIB, int(free_headroom * 0.35)))

    # Pandas object/string columns dominate the in-memory footprint. These row
    # estimates deliberately overstate the compact Parquet representation.
    estimated_row_bytes = 384
    continuous_max = min(
        2_000_000,
        max(100_000, int(writer_budget * 0.85) // estimated_row_bytes),
    )
    auction_max = min(
        500_000,
        max(25_000, int(writer_budget * 0.15) // estimated_row_bytes),
    )
    return ImportMemoryPolicy(
        total_bytes=total,
        available_at_start_bytes=available,
        reserve_bytes=reserve,
        writer_budget_bytes=writer_budget,
        pressure_threshold_bytes=reserve,
        continuous_flush_rows=min(250_000, max(50_000, continuous_max // 4)),
        continuous_max_buffered_rows=continuous_max,
        auction_flush_rows=min(100_000, max(10_000, auction_max // 4)),
        auction_max_buffered_rows=auction_max,
    )


class MonthlyWriters:
    """One Parquet writer per calendar month for a staged canonical dataset."""

    def __init__(
        self,
        dataset_dir: Path,
        *,
        flush_rows: int,
        max_buffered_rows: int,
    ) -> None:
        self.dataset_dir = dataset_dir
        self.flush_rows = int(flush_rows)
        self.max_buffered_rows = int(max_buffered_rows)
        self.writers: dict[str, pq.ParquetWriter] = {}
        self.paths: dict[str, Path] = {}
        self.rows: dict[str, int] = {}
        self.buffers: dict[str, list[pd.DataFrame]] = {}
        self.buffered_rows: dict[str, int] = {}

    def _writer(self, key: str) -> pq.ParquetWriter:
        writer = self.writers.get(key)
        if writer is not None:
            return writer
        path = self.dataset_dir / "shards" / f"year={key[:4]}" / f"month={key[4:]}" / "part-0000.parquet"
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
        return writer

    def _flush(self, key: str) -> None:
        parts = self.buffers.pop(key, [])
        if not parts:
            return
        selected = pd.concat(parts, ignore_index=True) if len(parts) > 1 else parts[0]
        self.buffered_rows.pop(key, None)
        self._writer(key).write_table(bar_table(selected), row_group_size=250_000)
        self.rows[key] += len(selected)

    def _enforce_memory_limit(self) -> None:
        total = sum(self.buffered_rows.values())
        while total > self.max_buffered_rows and self.buffered_rows:
            key = max(self.buffered_rows, key=self.buffered_rows.__getitem__)
            total -= self.buffered_rows[key]
            self._flush(key)

    @property
    def total_buffered_rows(self) -> int:
        return sum(self.buffered_rows.values())

    def flush_all(self) -> None:
        for key in list(self.buffers):
            self._flush(key)

    def write(self, frame: pd.DataFrame) -> None:
        if frame.empty:
            return
        months = frame["trade_date"].str.slice(0, 7).str.replace("-", "", regex=False)
        for month, selected in frame.groupby(months, sort=True):
            key = str(month)
            part = selected.reset_index(drop=True)
            self.buffers.setdefault(key, []).append(part)
            self.buffered_rows[key] = self.buffered_rows.get(key, 0) + len(part)
            if self.buffered_rows[key] >= self.flush_rows:
                self._flush(key)
        self._enforce_memory_limit()

    def close(self) -> None:
        self.flush_all()
        for writer in self.writers.values():
            writer.close()
        self.writers.clear()

    def year_paths(self, year: int) -> list[Path]:
        prefix = str(int(year))
        return [path for month, path in sorted(self.paths.items()) if month.startswith(prefix)]

    def year_rows(self, year: int) -> int:
        prefix = str(int(year))
        return sum(rows for month, rows in self.rows.items() if month.startswith(prefix))


class FrameWriter:
    """Append small audit frames to one Parquet file without retaining a year in RAM."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self.writer: pq.ParquetWriter | None = None
        self.schema: pa.Schema | None = None
        self.row_count = 0

    def write(self, frame: pd.DataFrame) -> None:
        if frame.empty:
            return
        table = pa.Table.from_pandas(frame, preserve_index=False)
        if self.writer is None:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self.schema = table.schema
            self.writer = pq.ParquetWriter(
                self.path,
                self.schema,
                compression="zstd",
                use_dictionary=True,
            )
        elif table.schema != self.schema:
            table = table.cast(self.schema)
        self.writer.write_table(table)
        self.row_count += len(frame)

    def close(self) -> None:
        if self.writer is not None:
            self.writer.close()
            self.writer = None


class AnnualEvidenceWriters:
    """Stream stock-day count and share evidence into separate annual files."""

    def __init__(self, quality_root: Path, years: Sequence[int]) -> None:
        self.counts = {
            int(year): FrameWriter(quality_root / f"year={int(year)}" / "daily_bar_counts.parquet") for year in years
        }
        self.shares = {
            int(year): FrameWriter(quality_root / f"year={int(year)}" / "daily_share_candidates.parquet")
            for year in years
        }
        self.repairs = {
            int(year): FrameWriter(quality_root / f"year={int(year)}" / "zero_price_placeholder_repairs.parquet")
            for year in years
        }
        self.time_repairs = {
            int(year): FrameWriter(quality_root / f"year={int(year)}" / "bar_time_repairs.parquet") for year in years
        }

    def write(self, year: int, counts: pd.DataFrame, shares: pd.DataFrame) -> None:
        self.counts[int(year)].write(counts)
        self.shares[int(year)].write(shares)

    def write_repairs(self, year: int, repairs: pd.DataFrame) -> None:
        self.repairs[int(year)].write(repairs)

    def write_time_repairs(self, year: int, repairs: pd.DataFrame) -> None:
        self.time_repairs[int(year)].write(repairs)

    def close(self) -> None:
        for writer in [
            *self.counts.values(),
            *self.shares.values(),
            *self.repairs.values(),
            *self.time_repairs.values(),
        ]:
            writer.close()


@dataclass(frozen=True)
class ImportLayout:
    workspace: Path
    qdp_root: Path
    archive: Path
    years: tuple[int, ...]
    staging: Path
    staged_continuous: Path
    staged_auction: Path
    staged_quality: Path
    final_continuous: Path
    final_auction: Path
    final_quality_root: Path

    def staged_quality_year(self, year: int) -> Path:
        return self.staged_quality / f"year={int(year)}"

    def final_quality_year(self, year: int) -> Path:
        return self.final_quality_root / f"year={int(year)}"


@dataclass(frozen=True)
class ExistingState:
    continuous: DatasetManifest | None
    auction: DatasetManifest | None
    years: tuple[int, ...]


@dataclass
class StreamedBatch:
    continuous: MonthlyWriters
    auction: MonthlyWriters
    evidence: AnnualEvidenceWriters
    member_audits: dict[int, list[dict[str, Any]]]
    source_member_count: int
    share_inconsistent_days: dict[int, int]
    memory_policy: dict[str, int]
    memory_pressure_flushes: int


def _normalize_years(years: Sequence[int]) -> tuple[int, ...]:
    selected = tuple(sorted({int(value) for value in years}))
    if not selected:
        raise MinuteArchiveError("import_years_empty")
    if selected[0] < 1990 or selected[-1] > 2100:
        raise MinuteArchiveError(f"import_year_out_of_range:{selected[0]}:{selected[-1]}")
    return selected


def _import_layout(
    archive_path: str | Path,
    *,
    years: Sequence[int],
    workspace_root_value: str | Path | None,
) -> ImportLayout:
    selected = _normalize_years(years)
    workspace = workspace_root(workspace_root_value)
    paths = qdp_paths(workspace)
    archive = Path(archive_path).resolve()
    if not archive.is_file():
        raise MinuteArchiveError(f"archive_missing:{archive}")
    qdp_root = paths.qdp_v2_dir
    staging = qdp_root / "tmp" / f"minute_import_{selected[0]}_{selected[-1]}_{os.getpid()}"
    return ImportLayout(
        workspace=workspace,
        qdp_root=qdp_root,
        archive=archive,
        years=selected,
        staging=staging,
        staged_continuous=staging / CONTINUOUS_DOMAIN,
        staged_auction=staging / AUCTION_DOMAIN,
        staged_quality=staging / "quality",
        final_continuous=qdp_root / "datasets" / CONTINUOUS_DOMAIN / CONTINUOUS_DOMAIN,
        final_auction=qdp_root / "datasets" / AUCTION_DOMAIN / AUCTION_DOMAIN,
        final_quality_root=paths.source_archives_dir / "minute" / "quality",
    )


def _year_from_shard_path(path: str) -> int:
    for part in Path(path).parts:
        if part.startswith("year="):
            try:
                return int(part.removeprefix("year="))
            except ValueError as exc:
                raise MinuteArchiveError(f"invalid_minute_shard_year:{path}") from exc
    raise MinuteArchiveError(f"minute_shard_year_missing:{path}")


def _manifest_years(manifest: DatasetManifest) -> tuple[int, ...]:
    return tuple(sorted({_year_from_shard_path(shard.path) for shard in manifest.shards}))


def _year_directories(dataset: Path) -> set[int]:
    result: set[int] = set()
    for path in (dataset / "shards").glob("year=*"):
        if not path.is_dir():
            continue
        try:
            result.add(int(path.name.removeprefix("year=")))
        except ValueError as exc:
            raise MinuteArchiveError(f"invalid_minute_year_directory:{path}") from exc
    return result


def _load_existing_state(layout: ImportLayout) -> ExistingState:
    continuous_path = layout.final_continuous / "dataset.json"
    auction_path = layout.final_auction / "dataset.json"
    if continuous_path.exists() != auction_path.exists():
        raise MinuteArchiveError("minute_dataset_manifest_pair_incomplete")
    if not continuous_path.exists():
        if _year_directories(layout.final_continuous) or _year_directories(layout.final_auction):
            raise MinuteArchiveError("minute_shards_exist_without_manifests")
        return ExistingState(continuous=None, auction=None, years=())
    continuous = read_dataset_manifest(continuous_path)
    auction = read_dataset_manifest(auction_path)
    if continuous.dataset_id != CONTINUOUS_DOMAIN or auction.dataset_id != AUCTION_DOMAIN:
        raise MinuteArchiveError("canonical_minute_dataset_id_mismatch")
    continuous_years = _manifest_years(continuous)
    auction_years = _manifest_years(auction)
    if continuous_years != auction_years:
        raise MinuteArchiveError("minute_manifest_years_mismatch")
    if set(continuous_years) != _year_directories(layout.final_continuous):
        raise MinuteArchiveError("continuous_manifest_directory_mismatch")
    if set(auction_years) != _year_directories(layout.final_auction):
        raise MinuteArchiveError("auction_manifest_directory_mismatch")
    missing_audits = [
        year for year in continuous_years if not (layout.final_quality_year(year) / "audit.json").is_file()
    ]
    if missing_audits:
        raise MinuteArchiveError(f"minute_year_audits_missing:{missing_audits}")
    return ExistingState(continuous=continuous, auction=auction, years=continuous_years)


def _safe_remove_staging(layout: ImportLayout) -> None:
    if not layout.staging.exists():
        return
    temporary_root = (layout.qdp_root / "tmp").resolve()
    resolved = layout.staging.resolve()
    if resolved == temporary_root:
        raise MinuteArchiveError("refuse_remove_qdp_tmp_root")
    try:
        resolved.relative_to(temporary_root)
    except ValueError as exc:
        raise MinuteArchiveError(f"unsafe_staging_path:{resolved}") from exc
    shutil.rmtree(resolved)


def _prepare_staging(layout: ImportLayout) -> None:
    _safe_remove_staging(layout)
    for year in layout.years:
        targets = (
            layout.final_continuous / "shards" / f"year={year}",
            layout.final_auction / "shards" / f"year={year}",
            layout.final_quality_year(year),
        )
        if any(path.exists() for path in targets):
            raise MinuteArchiveError(f"minute_year_target_already_exists:{year}")
    layout.staging.mkdir(parents=True)


def _empty_member_frame() -> pd.DataFrame:
    return pd.DataFrame(
        columns=[
            "timestamp",
            "open",
            "high",
            "low",
            "close",
            "volume",
            "amount",
            "price_change",
            "pct_change",
            "turnover_pct",
            "float_shares",
            "total_shares",
            "symbol",
        ]
    )


def _canonical_frame(frame: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "symbol",
        "trade_date",
        "bar_time",
        "open",
        "high",
        "low",
        "close",
        "volume",
        "amount",
    ]
    core = frame.loc[:, columns].copy()
    core["source"] = SOURCE_NAME
    core["adjusted_flag"] = PRICE_MODE
    return core


def _flush_if_memory_pressure(
    policy: ImportMemoryPolicy,
    *writers: MonthlyWriters,
) -> bool:
    if int(psutil.virtual_memory().available) >= policy.pressure_threshold_bytes:
        return False
    if not any(writer.total_buffered_rows for writer in writers):
        return False
    for writer in writers:
        writer.flush_all()
    pa.default_memory_pool().release_unused()
    gc.collect()
    return True


def _stream_years(layout: ImportLayout) -> StreamedBatch:
    members = list_members([layout.archive])
    if not members:
        raise MinuteArchiveError("archive_has_no_1m_members")
    memory_policy = _import_memory_policy()
    continuous = MonthlyWriters(
        layout.staged_continuous,
        flush_rows=memory_policy.continuous_flush_rows,
        max_buffered_rows=memory_policy.continuous_max_buffered_rows,
    )
    auction = MonthlyWriters(
        layout.staged_auction,
        flush_rows=memory_policy.auction_flush_rows,
        max_buffered_rows=memory_policy.auction_max_buffered_rows,
    )
    evidence = AnnualEvidenceWriters(layout.staged_quality, layout.years)
    member_audits: dict[int, list[dict[str, Any]]] = {year: [] for year in layout.years}
    inconsistent = {year: 0 for year in layout.years}
    memory_pressure_flushes = 0
    empty = _empty_member_frame()
    try:
        with ZipFile(layout.archive) as zipped:
            for member in members:
                memory_pressure_flushes += int(
                    _flush_if_memory_pressure(memory_policy, continuous, auction)
                )
                raw = read_member_years(
                    zipped,
                    zipped.getinfo(member.member),
                    symbol=member.symbol,
                    years=layout.years,
                )
                if raw.empty:
                    for year in layout.years:
                        member_audits[year].append(member_frame_audit(empty, member=member, year=year))
                    del raw
                    continue
                time_repairs = repair_mislabeled_1300_as_1130(raw, symbol=member.symbol)
                repairs = repair_zero_price_placeholders(raw, symbol=member.symbol)
                prepare_member_frame(
                    raw,
                    member=member,
                    year_label=f"{layout.years[0]}-{layout.years[-1]}",
                )
                year_keys = raw["trade_date"].str.slice(0, 4).astype("int16")
                counts, shares = daily_evidence(raw, member.symbol)
                variation = raw.groupby("trade_date")[["float_shares", "total_shares"]].nunique(dropna=True)
                inconsistent_dates = variation.index[(variation > 1).any(axis=1)]
                for year in layout.years:
                    frame = raw.loc[year_keys.eq(year)]
                    member_audits[year].append(member_frame_audit(frame, member=member, year=year))
                    prefix = f"{year}-"
                    year_counts = counts.loc[counts["trade_date"].str.startswith(prefix)]
                    year_shares = shares.loc[shares["trade_date"].str.startswith(prefix)]
                    evidence.write(year, year_counts, year_shares)
                    if not repairs.empty:
                        evidence.write_repairs(
                            year,
                            repairs.loc[repairs["trade_date"].str.startswith(prefix)],
                        )
                    if not time_repairs.empty:
                        evidence.write_time_repairs(
                            year,
                            time_repairs.loc[time_repairs["trade_date"].str.startswith(prefix)],
                        )
                    inconsistent[year] += sum(str(value).startswith(prefix) for value in inconsistent_dates)
                core = _canonical_frame(raw)
                continuous.write(core.loc[core["bar_time"].isin(CONTINUOUS_TIMES)])
                auction.write(core.loc[core["bar_time"].eq(AUCTION_TIME)])
                del (
                    core,
                    counts,
                    frame,
                    inconsistent_dates,
                    raw,
                    repairs,
                    shares,
                    time_repairs,
                    variation,
                    year_counts,
                    year_keys,
                    year_shares,
                )
                memory_pressure_flushes += int(
                    _flush_if_memory_pressure(memory_policy, continuous, auction)
                )
    finally:
        continuous.close()
        auction.close()
        evidence.close()
        pa.default_memory_pool().release_unused()
        gc.collect()
    for year in layout.years:
        if (
            not continuous.year_paths(year)
            or not auction.year_paths(year)
            or evidence.counts[year].row_count == 0
            or evidence.shares[year].row_count == 0
        ):
            raise MinuteArchiveError(f"formal_import_year_empty:{year}")
    return StreamedBatch(
        continuous=continuous,
        auction=auction,
        evidence=evidence,
        member_audits=member_audits,
        source_member_count=len(members),
        share_inconsistent_days=inconsistent,
        memory_policy=memory_policy.to_dict(),
        memory_pressure_flushes=memory_pressure_flushes,
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
        "active_or_normal_session_full_240_rate": (float(complete_days / eligible) if eligible else 0.0),
        "zero_flow_single_bar_suspension_placeholder_days": suspended_days,
        "unexplained_incomplete_stock_days": unexplained,
        "opening_auction_stock_days": int((counts["auction_row_count"] == 1).sum()),
        "opening_auction_missing_or_multiple_days": int((counts["auction_row_count"] != 1).sum()),
    }


def _quality_summary(
    layout: ImportLayout,
    streamed: StreamedBatch,
    *,
    year: int,
) -> dict[str, Any]:
    quality_dir = layout.staged_quality_year(year)
    counts = pd.read_parquet(quality_dir / "daily_bar_counts.parquet")
    shares = pd.read_parquet(quality_dir / "daily_share_candidates.parquet")
    if counts.duplicated(["symbol", "trade_date"]).any():
        raise MinuteArchiveError(f"daily_count_duplicate_keys:{year}")
    if shares.duplicated(["symbol", "trade_date"]).any():
        raise MinuteArchiveError(f"daily_share_evidence_duplicate_keys:{year}")
    members = pd.DataFrame(streamed.member_audits[year])
    members.to_parquet(quality_dir / "member_audit.parquet", compression="zstd", index=False)
    sessions = _classify_sessions(counts)
    session_exclusions = counts.loc[
        counts["continuous_bar_count"].ne(240) | counts["auction_row_count"].ne(1),
        [
            "symbol",
            "trade_date",
            "continuous_bar_count",
            "auction_row_count",
            "session_quality",
        ],
    ].copy()
    session_exclusions["exclusion_reason"] = "incomplete_or_noncanonical_session"
    session_exclusions_path = quality_dir / "session_feature_exclusions.parquet"
    session_exclusions.to_parquet(session_exclusions_path, compression="zstd", index=False)
    parity = parity_audit(
        continuous_paths=streamed.continuous.year_paths(year),
        auction_paths=streamed.auction.year_paths(year),
        daily_paths=qdp_daily_paths(layout.workspace),
        year=year,
        output_dir=quality_dir,
        final_quality_dir=layout.final_quality_year(year),
    )
    quality = {
        "source_member_count": streamed.source_member_count,
        "members_with_rows": int((members["row_count"] > 0).sum()),
        **sessions,
        "share_inconsistent_stock_days": int(streamed.share_inconsistent_days[year]),
        "zero_price_placeholder_repair_rows": int(streamed.evidence.repairs[year].row_count),
        "zero_price_placeholder_repairs_path": (
            str((layout.final_quality_year(year) / "zero_price_placeholder_repairs.parquet").resolve())
            if streamed.evidence.repairs[year].row_count
            else ""
        ),
        "bar_time_repair_rows": int(streamed.evidence.time_repairs[year].row_count),
        "bar_time_repairs_path": (
            str((layout.final_quality_year(year) / "bar_time_repairs.parquet").resolve())
            if streamed.evidence.time_repairs[year].row_count
            else ""
        ),
        "session_feature_exclusion_rows": len(session_exclusions),
        "session_feature_exclusions_path": str(
            (layout.final_quality_year(year) / session_exclusions_path.name).resolve()
        ),
        **parity,
    }
    quality["total_minute_feature_exclusion_rows"] = int(quality["minute_feature_exclusion_rows"]) + len(
        session_exclusions
    )
    quality["research_eligible_stock_days"] = len(counts) - len(session_exclusions)
    quality["status"] = "ready_with_exclusions" if quality["total_minute_feature_exclusion_rows"] else "clean"
    failed = bool(parity["duplicate_primary_keys"])
    if failed:
        raise MinuteArchiveError(f"formal_import_quality_failed:{json.dumps(quality)}")
    return quality


def _date_bounds(counts_path: Path) -> dict[str, tuple[str, str]]:
    counts = pd.read_parquet(counts_path, columns=["trade_date"])
    framed = counts.assign(month=counts["trade_date"].str.slice(0, 7).str.replace("-", "", regex=False))
    return {
        str(month): (str(group["trade_date"].min()), str(group["trade_date"].max()))
        for month, group in framed.groupby("month", sort=True)
    }


def _year_shards(
    *,
    year: int,
    writers: MonthlyWriters,
    staged_dataset: Path,
    final_dataset: Path,
    qdp_root: Path,
    bounds: Mapping[str, tuple[str, str]],
) -> list[ShardManifestEntry]:
    prefix = str(int(year))
    return [
        ShardManifestEntry(
            path=(final_dataset / path.relative_to(staged_dataset)).relative_to(qdp_root).as_posix(),
            row_count=int(writers.rows[month]),
            start_date=bounds[month][0],
            end_date=bounds[month][1],
            file_size=path.stat().st_size,
        )
        for month, path in sorted(writers.paths.items())
        if month.startswith(prefix)
    ]


def _archive_records(source: Mapping[str, Any], years: Sequence[int]) -> list[dict[str, Any]]:
    records = [dict(item) for item in list(source.get("archives", []) or []) if isinstance(item, Mapping)]
    if not records and source.get("source_archive"):
        records.append(
            {
                "path": str(source.get("source_archive", "")),
                "sha256": str(source.get("source_archive_sha256", "")),
                "years": [int(value) for value in years],
            }
        )
    return records


def _same_archive(record: Mapping[str, Any], archive: Path) -> bool:
    try:
        return Path(str(record.get("path", ""))).resolve() == archive.resolve()
    except (OSError, ValueError):
        return False


def _known_archive_hash(state: ExistingState, archive: Path) -> str:
    for manifest in (state.continuous, state.auction):
        if manifest is None:
            continue
        for record in _archive_records(manifest.source, state.years):
            if _same_archive(record, archive):
                return str(record.get("sha256", ""))
    return ""


def _merged_source(
    state: ExistingState,
    *,
    archive: Path,
    archive_hash: str,
    added_years: Sequence[int],
) -> dict[str, Any]:
    base = state.continuous.source if state.continuous is not None else {}
    records = _archive_records(base, state.years)
    matched = False
    for record in records:
        if _same_archive(record, archive):
            record["path"] = str(archive.resolve())
            record["sha256"] = archive_hash
            record["years"] = sorted({int(value) for value in list(record.get("years", []) or [])} | set(added_years))
            matched = True
    if not matched:
        records.append(
            {
                "path": str(archive.resolve()),
                "sha256": archive_hash,
                "years": sorted({int(value) for value in added_years}),
            }
        )
    records.sort(key=lambda item: (min(item.get("years", [9999])), str(item.get("path", ""))))
    return {
        "provider": SOURCE_NAME,
        "price_mode": PRICE_MODE,
        "bar_label": "right_end",
        "archives": records,
    }


def _read_existing_quality(layout: ImportLayout, years: Sequence[int]) -> dict[int, dict[str, Any]]:
    result: dict[int, dict[str, Any]] = {}
    for year in years:
        audit = read_json(layout.final_quality_year(year) / "audit.json")
        result[int(year)] = dict(audit.get("quality", {}) or {})
    return result


def _quality_rollup(
    layout: ImportLayout,
    qualities: Mapping[int, Mapping[str, Any]],
) -> dict[str, Any]:
    years = sorted(int(value) for value in qualities)
    coverage = [
        float(qualities[year]["daily_reference_coverage_rate"])
        for year in years
        if "daily_reference_coverage_rate" in qualities[year]
    ]
    return {
        "primary_key_unique": all(int(qualities[year].get("duplicate_primary_keys", 0) or 0) == 0 for year in years),
        "imported_years": years,
        "year_audits": {str(year): str((layout.final_quality_year(year) / "audit.json").resolve()) for year in years},
        "stock_day_count": sum(int(qualities[year].get("stock_day_count", 0) or 0) for year in years),
        "unexplained_incomplete_stock_days": sum(
            int(qualities[year].get("unexplained_incomplete_stock_days", 0) or 0) for year in years
        ),
        "minute_feature_exclusion_rows": sum(
            int(
                qualities[year].get(
                    "total_minute_feature_exclusion_rows",
                    qualities[year].get("minute_feature_exclusion_rows", 0),
                )
                or 0
            )
            for year in years
        ),
        "daily_reference_coverage_rate_min": min(coverage) if coverage else None,
        "append_contract": "transactional_year_partition_v1",
    }


def _merged_manifest(
    *,
    domain: str,
    existing: DatasetManifest | None,
    added_shards: Sequence[ShardManifestEntry],
    source: Mapping[str, Any],
    quality: Mapping[str, Any],
    schema_path: Path,
) -> DatasetManifest:
    shards = [*(existing.shards if existing is not None else []), *added_shards]
    if len({shard.path for shard in shards}) != len(shards):
        raise MinuteArchiveError(f"duplicate_minute_manifest_shards:{domain}")
    shards.sort(key=lambda item: item.path)
    return DatasetManifest(
        dataset_id=domain,
        domain=domain,
        layer="raw",
        frequency="opening_auction" if domain == AUCTION_DOMAIN else "1m",
        contract_version=f"qdp_{domain}_raw",
        primary_key=["symbol", "trade_date", "bar_time"],
        start_date=min(shard.start_date for shard in shards),
        end_date=max(shard.end_date for shard in shards),
        row_count=sum(int(shard.row_count) for shard in shards),
        shards=shards,
        source=dict(source),
        quality=dict(quality),
        created_at=existing.created_at if existing is not None else utc_now(),
        schema=(
            existing.schema
            if existing is not None and existing.schema
            else _manifest_schema_from_arrow(pq.read_schema(schema_path))
        ),
        notes=[
            "09:30 opening auction is stored separately from continuous trading",
            "zero-flow suspension placeholders are retained and explicitly audited",
            "share fields are retained once per stock-day as source quality evidence",
            "each calendar year is appended only after annual structural and daily parity audits",
        ],
    )


def _year_result(
    layout: ImportLayout,
    streamed: StreamedBatch,
    *,
    year: int,
    quality: Mapping[str, Any],
    archive_hash: str,
    continuous_shards: Sequence[ShardManifestEntry],
    auction_shards: Sequence[ShardManifestEntry],
) -> dict[str, Any]:
    result = {
        "schema": "quantlab.minute_import_audit/2",
        "status": "ready",
        "created_at": utc_now(),
        "year": int(year),
        "source_archive": str(layout.archive.resolve()),
        "source_archive_sha256": archive_hash,
        "continuous": {
            "row_count": streamed.continuous.year_rows(year),
            "start_date": min(shard.start_date for shard in continuous_shards),
            "end_date": max(shard.end_date for shard in continuous_shards),
            "shards": [shard.to_dict() for shard in continuous_shards],
        },
        "opening_auction": {
            "row_count": streamed.auction.year_rows(year),
            "start_date": min(shard.start_date for shard in auction_shards),
            "end_date": max(shard.end_date for shard in auction_shards),
            "shards": [shard.to_dict() for shard in auction_shards],
        },
        "quality": dict(quality),
        "quality_directory": str(layout.final_quality_year(year).resolve()),
    }
    write_json(layout.staged_quality_year(year) / "audit.json", result)
    return result


def _restore_json(path: Path, payload: Mapping[str, Any] | None) -> None:
    if payload is None:
        path.unlink(missing_ok=True)
    else:
        write_json(path, payload)


def _install(
    layout: ImportLayout,
    *,
    continuous_manifest: DatasetManifest,
    auction_manifest: DatasetManifest,
    year_results: Mapping[int, Mapping[str, Any]],
) -> None:
    continuous_manifest_path = layout.final_continuous / "dataset.json"
    auction_manifest_path = layout.final_auction / "dataset.json"
    active_path = layout.qdp_root / "active" / "active.json"
    old_continuous = read_json(continuous_manifest_path) if continuous_manifest_path.exists() else None
    old_auction = read_json(auction_manifest_path) if auction_manifest_path.exists() else None
    old_active = read_active_manifest(layout.qdp_root)
    moved: list[tuple[Path, Path]] = []
    try:
        for year in sorted(year_results):
            pairs = (
                (
                    layout.staged_continuous / "shards" / f"year={year}",
                    layout.final_continuous / "shards" / f"year={year}",
                ),
                (
                    layout.staged_auction / "shards" / f"year={year}",
                    layout.final_auction / "shards" / f"year={year}",
                ),
                (layout.staged_quality_year(year), layout.final_quality_year(year)),
            )
            for source, target in pairs:
                if target.exists():
                    raise MinuteArchiveError(f"minute_install_target_exists:{target}")
                target.parent.mkdir(parents=True, exist_ok=True)
                os.replace(source, target)
                moved.append((source, target))
        write_json(continuous_manifest_path, continuous_manifest.to_dict())
        write_json(auction_manifest_path, auction_manifest.to_dict())
        active = dict(old_active)
        active["datasets"] = {
            **dict(active.get("datasets", {}) or {}),
            CONTINUOUS_DOMAIN: CONTINUOUS_DOMAIN,
            AUCTION_DOMAIN: AUCTION_DOMAIN,
        }
        active["updated_at"] = utc_now()
        write_active_manifest(layout.qdp_root, active)
    except Exception:
        _restore_json(continuous_manifest_path, old_continuous)
        _restore_json(auction_manifest_path, old_auction)
        _restore_json(active_path, old_active)
        for source, target in reversed(moved):
            if target.exists() and not source.exists():
                source.parent.mkdir(parents=True, exist_ok=True)
                os.replace(target, source)
        raise


def _existing_year_results(layout: ImportLayout, years: Sequence[int]) -> dict[str, Any]:
    return {str(year): read_json(layout.final_quality_year(year) / "audit.json") for year in years}


def _monthly_checkpoint(
    writer: MonthlyWriters,
    *,
    staging: Path,
) -> dict[str, dict[str, Any]]:
    return {
        month: {
            "path": str(path.relative_to(staging).as_posix()),
            "row_count": int(writer.rows[month]),
            "file_size": path.stat().st_size,
        }
        for month, path in sorted(writer.paths.items())
    }


def _write_stream_checkpoint(layout: ImportLayout, streamed: StreamedBatch) -> Path:
    for year in layout.years:
        pd.DataFrame(streamed.member_audits[year]).to_parquet(
            layout.staged_quality_year(year) / "member_audit.parquet",
            compression="zstd",
            index=False,
        )
    checkpoint = {
        "schema": "quantlab.minute_stream_checkpoint/1",
        "status": "staged",
        "created_at": utc_now(),
        "archive": str(layout.archive.resolve()),
        "years": list(layout.years),
        "source_member_count": streamed.source_member_count,
        "share_inconsistent_days": {str(year): int(value) for year, value in streamed.share_inconsistent_days.items()},
        "memory_policy": streamed.memory_policy,
        "memory_pressure_flushes": streamed.memory_pressure_flushes,
        "continuous": _monthly_checkpoint(
            streamed.continuous,
            staging=layout.staging,
        ),
        "opening_auction": _monthly_checkpoint(
            streamed.auction,
            staging=layout.staging,
        ),
        "evidence_rows": {
            str(year): {
                "counts": streamed.evidence.counts[year].row_count,
                "shares": streamed.evidence.shares[year].row_count,
                "zero_price_repairs": streamed.evidence.repairs[year].row_count,
                "bar_time_repairs": streamed.evidence.time_repairs[year].row_count,
            }
            for year in layout.years
        },
    }
    path = layout.staging / "stream_checkpoint.json"
    write_json(path, checkpoint)
    return path


def _layout_from_staging(
    staging_path: str | Path,
    *,
    workspace_root_value: str | Path | None,
) -> tuple[ImportLayout, dict[str, Any]]:
    workspace = workspace_root(workspace_root_value)
    paths = qdp_paths(workspace)
    staging = Path(staging_path).resolve()
    temporary_root = (paths.qdp_v2_dir / "tmp").resolve()
    try:
        staging.relative_to(temporary_root)
    except ValueError as exc:
        raise MinuteArchiveError(f"unsafe_resume_staging_path:{staging}") from exc
    checkpoint = read_json(staging / "stream_checkpoint.json")
    if checkpoint.get("schema") != "quantlab.minute_stream_checkpoint/1":
        raise MinuteArchiveError("minute_stream_checkpoint_schema_invalid")
    archive = Path(str(checkpoint.get("archive", ""))).resolve()
    years = _normalize_years(list(checkpoint.get("years", []) or []))
    base = _import_layout(
        archive,
        years=years,
        workspace_root_value=workspace,
    )
    return (
        ImportLayout(
            workspace=base.workspace,
            qdp_root=base.qdp_root,
            archive=base.archive,
            years=base.years,
            staging=staging,
            staged_continuous=staging / CONTINUOUS_DOMAIN,
            staged_auction=staging / AUCTION_DOMAIN,
            staged_quality=staging / "quality",
            final_continuous=base.final_continuous,
            final_auction=base.final_auction,
            final_quality_root=base.final_quality_root,
        ),
        checkpoint,
    )


def _restore_monthly_writer(
    records: Mapping[str, Any],
    *,
    staging: Path,
    dataset_dir: Path,
) -> MonthlyWriters:
    writer = MonthlyWriters(
        dataset_dir,
        flush_rows=250_000,
        max_buffered_rows=2_000_000,
    )
    for month, value in sorted(records.items()):
        record = dict(value or {})
        path = (staging / str(record.get("path", ""))).resolve()
        try:
            path.relative_to(staging.resolve())
        except ValueError as exc:
            raise MinuteArchiveError(f"unsafe_checkpoint_file:{path}") from exc
        expected_rows = int(record.get("row_count", 0) or 0)
        expected_size = int(record.get("file_size", 0) or 0)
        if (
            not path.is_file()
            or path.stat().st_size != expected_size
            or pq.ParquetFile(path).metadata.num_rows != expected_rows
        ):
            raise MinuteArchiveError(f"minute_checkpoint_shard_mismatch:{path}")
        writer.paths[str(month)] = path
        writer.rows[str(month)] = expected_rows
    return writer


def _restore_streamed_batch(
    layout: ImportLayout,
    checkpoint: Mapping[str, Any],
) -> StreamedBatch:
    continuous = _restore_monthly_writer(
        dict(checkpoint.get("continuous", {}) or {}),
        staging=layout.staging,
        dataset_dir=layout.staged_continuous,
    )
    auction = _restore_monthly_writer(
        dict(checkpoint.get("opening_auction", {}) or {}),
        staging=layout.staging,
        dataset_dir=layout.staged_auction,
    )
    evidence = AnnualEvidenceWriters(layout.staged_quality, layout.years)
    evidence_rows = dict(checkpoint.get("evidence_rows", {}) or {})
    member_audits: dict[int, list[dict[str, Any]]] = {}
    for year in layout.years:
        rows = dict(evidence_rows.get(str(year), {}) or {})
        evidence.counts[year].row_count = int(rows.get("counts", 0) or 0)
        evidence.shares[year].row_count = int(rows.get("shares", 0) or 0)
        evidence.repairs[year].row_count = int(rows.get("zero_price_repairs", 0) or 0)
        evidence.time_repairs[year].row_count = int(rows.get("bar_time_repairs", 0) or 0)
        for item, path in (
            (evidence.counts[year], layout.staged_quality_year(year) / "daily_bar_counts.parquet"),
            (
                evidence.shares[year],
                layout.staged_quality_year(year) / "daily_share_candidates.parquet",
            ),
        ):
            if not path.is_file() or pq.ParquetFile(path).metadata.num_rows != item.row_count:
                raise MinuteArchiveError(f"minute_checkpoint_evidence_mismatch:{path}")
        member_path = layout.staged_quality_year(year) / "member_audit.parquet"
        member_audits[year] = pd.read_parquet(member_path).to_dict("records")
    return StreamedBatch(
        continuous=continuous,
        auction=auction,
        evidence=evidence,
        member_audits=member_audits,
        source_member_count=int(checkpoint.get("source_member_count", 0) or 0),
        share_inconsistent_days={
            int(year): int(value) for year, value in dict(checkpoint.get("share_inconsistent_days", {}) or {}).items()
        },
        memory_policy={
            str(key): int(value) for key, value in dict(checkpoint.get("memory_policy", {}) or {}).items()
        },
        memory_pressure_flushes=int(checkpoint.get("memory_pressure_flushes", 0) or 0),
    )


def _complete_staged_import(
    layout: ImportLayout,
    state: ExistingState,
    streamed: StreamedBatch,
    *,
    requested: Sequence[int],
    existing_requested: Sequence[int],
    compute_sha256: bool,
) -> dict[str, Any]:
    new_qualities = {year: _quality_summary(layout, streamed, year=year) for year in layout.years}
    known_hash = _known_archive_hash(state, layout.archive)
    archive_hash = known_hash or (sha256_file(layout.archive) if compute_sha256 else "")
    all_qualities = {
        **_read_existing_quality(layout, state.years),
        **new_qualities,
    }
    source = _merged_source(
        state,
        archive=layout.archive,
        archive_hash=archive_hash,
        added_years=layout.years,
    )
    bounds = {
        year: _date_bounds(layout.staged_quality_year(year) / "daily_bar_counts.parquet") for year in layout.years
    }
    continuous_by_year = {
        year: _year_shards(
            year=year,
            writers=streamed.continuous,
            staged_dataset=layout.staged_continuous,
            final_dataset=layout.final_continuous,
            qdp_root=layout.qdp_root,
            bounds=bounds[year],
        )
        for year in layout.years
    }
    auction_by_year = {
        year: _year_shards(
            year=year,
            writers=streamed.auction,
            staged_dataset=layout.staged_auction,
            final_dataset=layout.final_auction,
            qdp_root=layout.qdp_root,
            bounds=bounds[year],
        )
        for year in layout.years
    }
    rollup = _quality_rollup(layout, all_qualities)
    continuous_manifest = _merged_manifest(
        domain=CONTINUOUS_DOMAIN,
        existing=state.continuous,
        added_shards=[shard for year in layout.years for shard in continuous_by_year[year]],
        source=source,
        quality=rollup,
        schema_path=streamed.continuous.year_paths(layout.years[0])[0],
    )
    auction_manifest = _merged_manifest(
        domain=AUCTION_DOMAIN,
        existing=state.auction,
        added_shards=[shard for year in layout.years for shard in auction_by_year[year]],
        source=source,
        quality=rollup,
        schema_path=streamed.auction.year_paths(layout.years[0])[0],
    )
    year_results = {
        year: _year_result(
            layout,
            streamed,
            year=year,
            quality=new_qualities[year],
            archive_hash=archive_hash,
            continuous_shards=continuous_by_year[year],
            auction_shards=auction_by_year[year],
        )
        for year in layout.years
    }
    _install(
        layout,
        continuous_manifest=continuous_manifest,
        auction_manifest=auction_manifest,
        year_results=year_results,
    )
    return {
        "schema": "quantlab.minute_import_batch/1",
        "status": "ready",
        "requested_years": list(requested),
        "imported_years": list(layout.years),
        "skipped_years": list(existing_requested),
        "years": {
            **_existing_year_results(layout, existing_requested),
            **{str(year): result for year, result in year_results.items()},
        },
        "continuous": continuous_manifest.to_dict(),
        "opening_auction": auction_manifest.to_dict(),
        "runtime": {
            "memory_policy": streamed.memory_policy,
            "memory_pressure_flushes": streamed.memory_pressure_flushes,
        },
    }


def import_years(
    archive_path: str | Path,
    *,
    years: Sequence[int],
    workspace_root_value: str | Path | None = None,
    compute_sha256: bool = True,
) -> dict[str, Any]:
    """Append absent calendar years after one archive pass and annual audits."""

    requested = _normalize_years(years)
    initial_layout = _import_layout(
        archive_path,
        years=requested,
        workspace_root_value=workspace_root_value,
    )
    state = _load_existing_state(initial_layout)
    existing_requested = tuple(year for year in requested if year in state.years)
    missing = tuple(year for year in requested if year not in state.years)
    if not missing:
        return {
            "schema": "quantlab.minute_import_batch/1",
            "status": "unchanged",
            "requested_years": list(requested),
            "imported_years": [],
            "skipped_years": list(existing_requested),
            "years": _existing_year_results(initial_layout, requested),
        }
    layout = _import_layout(
        archive_path,
        years=missing,
        workspace_root_value=workspace_root_value,
    )
    _prepare_staging(layout)
    try:
        streamed = _stream_years(layout)
        _write_stream_checkpoint(layout, streamed)
        return _complete_staged_import(
            layout,
            state,
            streamed,
            requested=requested,
            existing_requested=existing_requested,
            compute_sha256=compute_sha256,
        )
    except Exception as exc:
        write_json(
            layout.staging / "failure.json",
            {
                "schema": "quantlab.minute_import_failure/1",
                "created_at": utc_now(),
                "archive": str(layout.archive),
                "years": list(layout.years),
                "error_type": type(exc).__name__,
                "error": str(exc),
            },
        )
        raise
    finally:
        if not (layout.staging / "failure.json").exists():
            _safe_remove_staging(layout)


def stage_years(
    archive_path: str | Path,
    *,
    years: Sequence[int],
    workspace_root_value: str | Path | None = None,
) -> dict[str, Any]:
    """Stream absent year partitions and stop at a durable audit checkpoint."""

    requested = _normalize_years(years)
    initial_layout = _import_layout(
        archive_path,
        years=requested,
        workspace_root_value=workspace_root_value,
    )
    state = _load_existing_state(initial_layout)
    missing = tuple(year for year in requested if year not in state.years)
    if not missing:
        return {
            "schema": "quantlab.minute_stage_batch/1",
            "status": "unchanged",
            "requested_years": list(requested),
            "staged_years": [],
            "staging": "",
        }
    layout = _import_layout(
        archive_path,
        years=missing,
        workspace_root_value=workspace_root_value,
    )
    _prepare_staging(layout)
    try:
        streamed = _stream_years(layout)
        checkpoint = _write_stream_checkpoint(layout, streamed)
        return {
            "schema": "quantlab.minute_stage_batch/1",
            "status": "staged",
            "requested_years": list(requested),
            "staged_years": list(missing),
            "staging": str(layout.staging.resolve()),
            "checkpoint": str(checkpoint.resolve()),
        }
    except Exception as exc:
        write_json(
            layout.staging / "failure.json",
            {
                "schema": "quantlab.minute_import_failure/1",
                "created_at": utc_now(),
                "archive": str(layout.archive),
                "years": list(layout.years),
                "error_type": type(exc).__name__,
                "error": str(exc),
                "stage_only": True,
            },
        )
        raise


def resume_staged_import(
    staging_path: str | Path,
    *,
    workspace_root_value: str | Path | None = None,
    compute_sha256: bool = True,
) -> dict[str, Any]:
    """Audit and install a completed streaming checkpoint in a fresh process."""

    layout, checkpoint = _layout_from_staging(
        staging_path,
        workspace_root_value=workspace_root_value,
    )
    state = _load_existing_state(layout)
    overlap = sorted(set(layout.years).intersection(state.years))
    if overlap:
        raise MinuteArchiveError(f"staged_years_already_installed:{overlap}")
    streamed = _restore_streamed_batch(layout, checkpoint)
    failure_path = layout.staging / "failure.json"
    failure_path.unlink(missing_ok=True)
    try:
        result = _complete_staged_import(
            layout,
            state,
            streamed,
            requested=layout.years,
            existing_requested=(),
            compute_sha256=compute_sha256,
        )
    except Exception as exc:
        write_json(
            failure_path,
            {
                "schema": "quantlab.minute_import_failure/1",
                "created_at": utc_now(),
                "archive": str(layout.archive),
                "years": list(layout.years),
                "error_type": type(exc).__name__,
                "error": str(exc),
                "resumed_from_checkpoint": True,
            },
        )
        raise
    _safe_remove_staging(layout)
    return result


def import_year(
    archive_path: str | Path,
    *,
    year: int,
    workspace_root_value: str | Path | None = None,
    compute_sha256: bool = True,
) -> dict[str, Any]:
    """Import one year; retained as the simple and backwards-compatible API."""

    result = import_years(
        archive_path,
        years=[int(year)],
        workspace_root_value=workspace_root_value,
        compute_sha256=compute_sha256,
    )
    return dict(result["years"][str(int(year))])


__all__ = [
    "import_year",
    "import_years",
    "resume_staged_import",
    "stage_years",
]
