from __future__ import annotations

import argparse
import hashlib
import heapq
import io
import json
import os
import re
import sqlite3
import tempfile
import time
import uuid
import zipfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Iterator, Mapping, Sequence

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from quant_data_platform.core.json_io import read_json
from quant_data_platform.core.paths import qdp_paths
from quant_data_platform.qdp_v2.duckdb_resources import open_guarded_duckdb
from quant_data_platform.qdp_v2.manifest import (
    atomic_write_json,
    dataset_manifest_for_id,
    qdp_v2_root,
    read_active_manifest,
    read_dataset_manifest,
    resolve_manifest_path,
)
from quant_data_platform.qdp_v2.status import active_dataset_map
from quant_data_platform.qdp_v2.repair import mutate_active_shards_from_parquet


DOMAIN = "market_intraday_5m"
DEFAULT_ARCHIVE_PATH = Path(r"H:\BaiduNetdiskDownload\量化数据\5分钟(2000-2025).zip")
REPAIR_PLAN_VERSION = 7
SOURCE_NAME = "external_quant_archive_5m_direct_repair"
BUCKET_COUNT = 16
DEFAULT_START_DATE = "2010-01-01"
DEFAULT_END_DATE = "2025-12-31"

EXPECTED_5M_BAR_ENDS = tuple(
    [
        f"{hour:02d}:{minute:02d}"
        for hour, minute in (
            *[(9, minute) for minute in range(35, 60, 5)],
            *[(10, minute) for minute in range(0, 60, 5)],
            *[(11, minute) for minute in range(0, 31, 5)],
            *[(13, minute) for minute in range(5, 60, 5)],
            *[(14, minute) for minute in range(0, 60, 5)],
            (15, 0),
        )
    ]
)
assert len(EXPECTED_5M_BAR_ENDS) == 48
CURRENT_RESEARCH_CODE_ALIASES: Mapping[str, str] = {
    "000022.SZ": "001872.SZ",
    "000043.SZ": "001914.SZ",
}

PATCH_COLUMNS = (
    "symbol",
    "trade_date",
    "bar_time",
    "open",
    "high",
    "low",
    "close",
    "volume",
    "amount",
    "source",
    "adjusted_flag",
)
PATCH_PRIMARY_KEY = ("symbol", "trade_date", "bar_time")
_NUMERIC_COLUMNS = ("open", "high", "low", "close", "volume", "amount")
_PRICE_COLUMNS = ("open", "high", "low", "close")
_MEMBER_PATTERN = re.compile(r"^(?P<exchange>sh|sz|bj)(?P<code>\d{6})\.csv$", re.IGNORECASE)
_NORMAL_SYMBOL_PATTERN = re.compile(r"^(?P<code>\d{6})\.(?P<exchange>SH|SZ|BJ)$")
_COLUMN_ALIASES: Mapping[str, tuple[str, ...]] = {
    "timestamp": ("日期", "时间", "datetime", "trade_time", "trade_datetime", "date", "time"),
    "open": ("开盘", "open"),
    "high": ("最高", "high"),
    "low": ("最低", "low"),
    "close": ("收盘", "close"),
    "volume": ("成交量(股)", "成交量", "volume", "vol"),
    "amount": ("成交额(元)", "成交额", "amount"),
}

_PATCH_ARROW_SCHEMA = pa.schema(
    [
        # Match the existing v2 Parquet physical schema.  Values are still
        # proven non-null by the planner/apply semantic checks.
        pa.field("symbol", pa.string()),
        pa.field("trade_date", pa.string()),
        pa.field("bar_time", pa.string()),
        pa.field("open", pa.float64()),
        pa.field("high", pa.float64()),
        pa.field("low", pa.float64()),
        pa.field("close", pa.float64()),
        pa.field("volume", pa.float64()),
        pa.field("amount", pa.float64()),
        pa.field("source", pa.string()),
        pa.field("adjusted_flag", pa.string()),
    ]
)
_BASE_DAY_SUMMARY_SCHEMA = pa.schema(
    [
        pa.field("symbol", pa.string()),
        pa.field("trade_date", pa.string()),
        pa.field("row_count", pa.int32()),
    ]
)


class IntradayRepairError(RuntimeError):
    """Raised when a direct-repair plan cannot be proven safe."""


class IntradayRepairResourceGuardError(IntradayRepairError):
    """Raised after available memory stays below the configured floor."""


@dataclass(frozen=True)
class ArchiveMember:
    member_name: str
    archive_symbol: str
    canonical_symbol: str
    crc32: str
    compressed_size: int
    uncompressed_size: int

    def fingerprint_record(self) -> dict[str, Any]:
        return {
            "member_name": self.member_name,
            "archive_symbol": self.archive_symbol,
            "canonical_symbol": self.canonical_symbol,
            "crc32": self.crc32,
            "compressed_size": int(self.compressed_size),
            "uncompressed_size": int(self.uncompressed_size),
        }


@dataclass(frozen=True)
class BaseSnapshot:
    root: Path
    active_path: Path
    active_sha256: str
    dataset_id: str
    manifest_path: Path
    manifest_sha256: str
    shard_paths: tuple[Path, ...]
    shard_stat_fingerprint: str


@dataclass(frozen=True)
class SymbolReceipt:
    symbol: str
    bucket: int
    accepted_days: int
    existing_days: int
    incomplete_existing_days: int
    rejected_days: int
    conflict_days: int
    patch_rows: int
    reject_sample_json: str
    content_fingerprint: str


class _MemoryGuard:
    def __init__(self, floor_gib: float, continuous_seconds: float) -> None:
        self.floor_bytes = max(0, int(float(floor_gib) * 1024**3))
        self.continuous_seconds = max(0.0, float(continuous_seconds))

    def check(self, label: str) -> None:
        if self.floor_bytes <= 0:
            return
        try:
            import psutil  # type: ignore
        except ImportError as exc:  # pragma: no cover - production dependency
            raise IntradayRepairResourceGuardError("psutil_required_for_memory_guard") from exc
        available = int(psutil.virtual_memory().available)
        if available >= self.floor_bytes:
            return
        started = time.monotonic()
        while available < self.floor_bytes:
            if time.monotonic() - started >= self.continuous_seconds:
                raise IntradayRepairResourceGuardError(
                    f"available_memory_below_floor:{label}:"
                    f"available={available}:floor={self.floor_bytes}:"
                    f"seconds={self.continuous_seconds}"
                )
            time.sleep(min(0.25, max(0.01, self.continuous_seconds / 10.0)))
            available = int(psutil.virtual_memory().available)


def _acquire_job_process_lock(lock_path: Path) -> Any:
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    handle = lock_path.open("a+b")
    if lock_path.stat().st_size == 0:
        handle.write(b"\0")
        handle.flush()
    handle.seek(0)
    try:
        if os.name == "nt":
            import msvcrt

            msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
        else:
            import fcntl  # pragma: no cover - Windows is the production platform.

            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError as exc:
        handle.close()
        raise IntradayRepairError(
            f"intraday_repair_job_lock_held:{lock_path}"
        ) from exc
    return handle


def _release_job_process_lock(handle: Any) -> None:
    try:
        handle.seek(0)
        if os.name == "nt":
            import msvcrt

            msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
        else:
            import fcntl  # pragma: no cover - Windows is the production platform.

            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
    finally:
        handle.close()


class _BucketBundleSink:
    """Write one bucket directly to year bundles with bounded Arrow buffers."""

    def __init__(
        self,
        *,
        output_root: Path,
        bucket: int,
        row_group_rows: int,
        dry_run: bool,
        guard: _MemoryGuard,
    ) -> None:
        self.output_root = output_root
        self.bucket = int(bucket)
        self.row_group_rows = int(row_group_rows)
        self.dry_run = bool(dry_run)
        self.guard = guard
        self._writers: dict[str, pq.ParquetWriter] = {}
        self._temporary_paths: dict[str, Path] = {}
        self._target_paths: dict[str, Path] = {}
        self._buffers: dict[str, list[pa.Table]] = {}
        self._buffer_rows: dict[str, int] = {}
        self._rows: dict[str, int] = {}
        self._days: dict[str, int] = {}
        self._finalized_targets: set[Path] = set()
        self._content_digest = hashlib.sha256()
        if not self.dry_run:
            self._clear_uncommitted_bucket_files()

    @property
    def content_fingerprint(self) -> str:
        return self._content_digest.hexdigest()

    @property
    def patch_row_count(self) -> int:
        return sum(self._rows.values())

    @property
    def patch_stock_day_count(self) -> int:
        return sum(self._days.values())

    def append_day(self, frame: pd.DataFrame, *, content_sha256: str) -> None:
        if len(frame) != 48:
            raise IntradayRepairError("intraday_repair_sink_requires_48_rows")
        year = str(frame["trade_date"].iloc[0])[:4]
        symbol = str(frame["symbol"].iloc[0])
        trade_date = str(frame["trade_date"].iloc[0])
        self._content_digest.update(
            (f"{symbol}\0{trade_date}\0{content_sha256}\0" + "48\n").encode("utf-8")
        )
        self._rows[year] = self._rows.get(year, 0) + 48
        self._days[year] = self._days.get(year, 0) + 1
        if self.dry_run:
            return
        table = pa.Table.from_pandas(
            frame.loc[:, PATCH_COLUMNS],
            schema=_PATCH_ARROW_SCHEMA,
            preserve_index=False,
            safe=True,
        )
        self._buffers.setdefault(year, []).append(table)
        self._buffer_rows[year] = self._buffer_rows.get(year, 0) + len(table)
        if self._buffer_rows[year] >= self.row_group_rows:
            self._flush_year(year)
        elif sum(self._buffer_rows.values()) >= 2 * self.row_group_rows:
            largest = max(self._buffer_rows, key=self._buffer_rows.__getitem__)
            self._flush_year(largest)

    def close_and_finalize(self) -> list[dict[str, Any]]:
        if self.dry_run:
            return [
                {
                    "year": year,
                    "bucket": self.bucket,
                    "path": f"patch/year={year}/bucket={self.bucket:02d}/part.parquet",
                    "row_count": int(self._rows[year]),
                    "stock_day_count": int(self._days[year]),
                    "file_size": 0,
                    "sha256": "",
                    "dry_run": True,
                }
                for year in sorted(self._rows)
            ]
        try:
            for year in sorted(self._buffers):
                self._flush_year(year)
            for writer in self._writers.values():
                writer.close()
            self._writers.clear()
            for year, temporary in sorted(self._temporary_paths.items()):
                self.guard.check(f"validate_bucket_bundle:{self.bucket}:{year}")
                metadata = pq.ParquetFile(temporary)
                try:
                    expected_rows = int(self._rows[year])
                    actual_rows = metadata.metadata.num_rows
                    schema_matches = metadata.schema_arrow.equals(
                        _PATCH_ARROW_SCHEMA,
                        check_metadata=False,
                    )
                finally:
                    metadata.close()
                if actual_rows != expected_rows:
                    raise IntradayRepairError(
                        f"intraday_repair_bundle_row_count_mismatch:{year}:{self.bucket}:"
                        f"expected={expected_rows}:actual={actual_rows}"
                    )
                if expected_rows % 48:
                    raise IntradayRepairError(
                        f"intraday_repair_bundle_not_whole_stock_days:{year}:{self.bucket}"
                    )
                if not schema_matches:
                    raise IntradayRepairError(
                        f"intraday_repair_bundle_schema_mismatch:{year}:{self.bucket}"
                    )
            for year, temporary in sorted(self._temporary_paths.items()):
                target = self._target_paths[year]
                _replace_file_with_retry(temporary, target)
                self._finalized_targets.add(target)
            records: list[dict[str, Any]] = []
            for year, target in sorted(self._target_paths.items()):
                records.append(
                    {
                        "year": year,
                        "bucket": self.bucket,
                        "path": target.relative_to(self.output_root).as_posix(),
                        "row_count": int(self._rows[year]),
                        "stock_day_count": int(self._days[year]),
                        "file_size": int(target.stat().st_size),
                        "sha256": _sha256_file(target),
                    }
                )
            return records
        except BaseException:
            self.abort()
            raise

    def abort(self) -> None:
        for writer in self._writers.values():
            try:
                writer.close()
            except Exception:
                pass
        self._writers.clear()
        for path in self._temporary_paths.values():
            _unlink_file_with_retry(path)
        for path in self._finalized_targets:
            _unlink_file_with_retry(path)
        self._finalized_targets.clear()

    def _flush_year(self, year: str) -> None:
        tables = self._buffers.pop(year, [])
        row_count = self._buffer_rows.pop(year, 0)
        if not tables:
            return
        self.guard.check(f"flush_bucket_bundle:{self.bucket}:{year}")
        writer = self._writer_for_year(year)
        combined = pa.concat_tables(tables)
        writer.write_table(combined, row_group_size=min(self.row_group_rows, len(combined)))
        if len(combined) != row_count:
            raise IntradayRepairError(
                f"intraday_repair_buffer_row_count_mismatch:{year}:{self.bucket}"
            )

    def _writer_for_year(self, year: str) -> pq.ParquetWriter:
        writer = self._writers.get(year)
        if writer is not None:
            return writer
        target = (
            self.output_root
            / "patch"
            / f"year={year}"
            / f"bucket={self.bucket:02d}"
            / "part.parquet"
        )
        target.parent.mkdir(parents=True, exist_ok=True)
        temporary = target.with_name(
            f".{target.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp"
        )
        writer = pq.ParquetWriter(
            temporary,
            _PATCH_ARROW_SCHEMA,
            compression="zstd",
            use_dictionary=True,
        )
        self._writers[year] = writer
        self._temporary_paths[year] = temporary
        self._target_paths[year] = target
        return writer

    def _clear_uncommitted_bucket_files(self) -> None:
        patch_root = (self.output_root / "patch").resolve()
        if not patch_root.exists():
            return
        for path in patch_root.glob(f"year=*/bucket={self.bucket:02d}/part.parquet"):
            if _is_within(path, patch_root):
                _unlink_file_with_retry(path)
        for path in patch_root.glob(f"year=*/bucket={self.bucket:02d}/.part.parquet.*.tmp"):
            if _is_within(path, patch_root):
                _unlink_file_with_retry(path)


def plan_intraday_5m_archive_repairs(
    *,
    workspace_root: str | Path | None = None,
    archive_path: str | Path = DEFAULT_ARCHIVE_PATH,
    output_root: str | Path | None = None,
    runtime_root: str | Path | None = None,
    start_date: str = DEFAULT_START_DATE,
    end_date: str = DEFAULT_END_DATE,
    symbols: Iterable[str] = (),
    eligible_symbols: Iterable[str] | None = None,
    dry_run: bool = True,
    resume: bool = True,
    csv_chunksize: int = 100_000,
    parquet_row_group_rows: int = 100_000,
    base_scan_batch_rows: int = 262_144,
    memory_floor_gib: float = 0.5,
    memory_floor_seconds: float = 5.0,
    source_name: str = SOURCE_NAME,
) -> dict[str, Any]:
    """Plan or materialize isolated, complete-stock-day v2 5m repair bundles.

    The active dataset is a read-only base.  Its complete stock-days always
    win.  Archive gaps are admitted only as exact 48-bar days and are streamed
    directly into natural-year × stable-16-bucket Parquet bundles.  SQLite
    stores only bucket/symbol receipts and counters; it never stores bars or
    the millions of base stock-day keys.
    """

    start = _date_text(start_date)
    end = _date_text(end_date)
    if start > end:
        raise ValueError(f"intraday_repair_invalid_range:{start}>{end}")
    if int(csv_chunksize) <= 0:
        raise ValueError("intraday_repair_csv_chunksize_must_be_positive")
    if int(parquet_row_group_rows) <= 0:
        raise ValueError("intraday_repair_row_group_rows_must_be_positive")
    if int(base_scan_batch_rows) <= 0:
        raise ValueError("intraday_repair_base_scan_batch_rows_must_be_positive")
    normalized_source_name = str(source_name or "").strip()
    if not normalized_source_name:
        raise ValueError("intraday_repair_source_name_required")

    archive = Path(archive_path).expanduser().resolve()
    if not archive.is_file() or archive.suffix.lower() != ".zip":
        raise FileNotFoundError(f"intraday_repair_archive_missing:{archive}")
    snapshot = _resolve_base_snapshot(workspace_root)
    members, archive_fingerprint = _archive_inventory(archive)
    requested_symbols = tuple(
        sorted({_canonical_symbol(item) for item in symbols if str(item).strip()})
    )
    if requested_symbols:
        target_symbols = requested_symbols
        selection_source = "explicit_symbols"
        missing_requested = sorted(set(requested_symbols) - set(members))
        if missing_requested:
            raise IntradayRepairError(
                f"intraday_repair_requested_symbols_missing:{missing_requested[:20]}"
            )
        eligible_missing_archive: tuple[str, ...] = ()
        selected_symbols = requested_symbols
    else:
        if eligible_symbols is None:
            eligible = _default_eligible_symbols(
                workspace_root=workspace_root,
                start_date=start,
                end_date=end,
            )
            selection_source = "tushare_stock_basic_mainboard_overlap"
        else:
            eligible = tuple(
                sorted({_canonical_symbol(item) for item in eligible_symbols if str(item).strip()})
            )
            selection_source = "explicit_eligible_symbols"
        target_symbols = tuple(sorted(set(eligible)))
        eligible_missing_archive = tuple(sorted(set(target_symbols).difference(members)))
        selected_symbols = tuple(sorted(set(target_symbols).intersection(members)))
    if not selected_symbols:
        raise IntradayRepairError("intraday_repair_selected_symbols_empty")
    selected_members = {symbol: members[symbol] for symbol in selected_symbols}
    selection_fingerprint = _stable_hash(
        {
            "repair_plan_version": REPAIR_PLAN_VERSION,
            "archive_fingerprint": archive_fingerprint,
            "base_manifest_sha256": snapshot.manifest_sha256,
            "start_date": start,
            "end_date": end,
            "target_symbols": list(target_symbols),
            "archive_selected_symbols": list(selected_symbols),
            "eligible_missing_archive": list(eligible_missing_archive),
            "selection_source": selection_source,
            "source_name": normalized_source_name,
            "aliases": dict(CURRENT_RESEARCH_CODE_ALIASES),
        }
    )
    job_id = f"intraday_5m_direct_repair__{selection_fingerprint[:24]}"
    resolved_output = (
        Path(output_root).expanduser().resolve()
        if output_root is not None
        else (snapshot.root / "repairs" / "intraday_5m" / job_id).resolve()
    )
    fixed_runtime = (
        qdp_paths(workspace_root).data_dir
        / "qdp_runtime"
        / "direct_repair"
        / job_id
    ).resolve()
    if runtime_root is not None and Path(runtime_root).expanduser().resolve() != fixed_runtime:
        raise IntradayRepairError(
            f"intraday_repair_runtime_root_is_fixed:{fixed_runtime}"
        )
    fixed_runtime.parent.mkdir(parents=True, exist_ok=True)
    guard = _MemoryGuard(memory_floor_gib, memory_floor_seconds)

    temporary: tempfile.TemporaryDirectory[str] | None = None
    if dry_run:
        temporary = tempfile.TemporaryDirectory(
            prefix=f".{job_id}.dry_run.",
            dir=fixed_runtime.parent,
        )
        working_runtime = Path(temporary.name).resolve()
        working_output = working_runtime / "output"
    else:
        fixed_runtime.mkdir(parents=True, exist_ok=True)
        working_runtime = fixed_runtime
        working_output = resolved_output
    state_path = working_runtime / "state.json"
    database_path = working_runtime / "repair.sqlite3"
    if not resume and (state_path.exists() or database_path.exists()):
        raise IntradayRepairError(
            f"intraday_repair_existing_runtime_requires_resume:{working_runtime}"
        )

    job_lock = _acquire_job_process_lock(
        fixed_runtime.parent / ".locks" / f"{job_id}.lock"
    )
    try:
        connection = sqlite3.connect(database_path)
        connection.execute("PRAGMA journal_mode=WAL")
        connection.execute("PRAGMA synchronous=FULL")
        connection.execute("PRAGMA temp_store=FILE")
        _initialize_database(connection)
        _bind_job_metadata(
            connection,
            job_id=job_id,
            selection_fingerprint=selection_fingerprint,
            archive_path=archive,
            archive_fingerprint=archive_fingerprint,
            snapshot=snapshot,
            start_date=start,
            end_date=end,
            selection_source=selection_source,
            source_name=normalized_source_name,
            output_root=resolved_output,
            target_symbols=target_symbols,
            selected_symbols=selected_symbols,
            eligible_missing_archive=eligible_missing_archive,
        )
        if not dry_run:
            resolved_output.mkdir(parents=True, exist_ok=True)
    except BaseException:
        _release_job_process_lock(job_lock)
        if temporary is not None:
            temporary.cleanup()
        raise

    manifest_path: Path | None = None
    bundle_records: list[dict[str, Any]] = []
    any_new_bucket = False
    try:
        _write_runtime_state(
            state_path,
            connection,
            status="indexing_base",
            job_id=job_id,
            archive_path=archive,
            archive_fingerprint=archive_fingerprint,
            snapshot=snapshot,
            selected_symbol_count=len(selected_symbols),
            output_root=resolved_output,
            dry_run=dry_run,
        )
        base_index_manifest = _ensure_base_day_summary_index(
            connection,
            snapshot=snapshot,
            selected_symbols=selected_symbols,
            start_date=start,
            end_date=end,
            index_root=working_runtime / "base_day_index",
            state_path=state_path,
            scan_batch_rows=int(base_scan_batch_rows),
            guard=guard,
        )
        _set_meta(connection, "base_index_manifest", str(base_index_manifest))
        _set_meta(connection, "base_index_manifest_sha256", _sha256_file(base_index_manifest))
        connection.commit()

        selected_by_bucket: dict[int, list[str]] = {}
        for symbol in selected_symbols:
            selected_by_bucket.setdefault(stable_symbol_bucket(symbol), []).append(symbol)
        with zipfile.ZipFile(archive, mode="r", metadata_encoding="gbk") as zip_handle:
            for bucket, bucket_symbols in sorted(selected_by_bucket.items()):
                guard.check(f"bucket:{bucket}")
                _assert_snapshot_unchanged(snapshot)
                reusable, records = _completed_bucket_reusable(
                    connection,
                    bucket=bucket,
                    output_root=working_output,
                    dry_run=dry_run,
                )
                if reusable:
                    bundle_records.extend(records)
                    continue
                any_new_bucket = True
                _clear_bucket_receipts(connection, bucket)
                (
                    base_days_by_symbol,
                    base_incomplete_by_symbol,
                    base_complete_count,
                    base_incomplete_count,
                ) = _load_base_complete_days(
                    base_index_manifest,
                    bucket=bucket,
                    guard=guard,
                )
                sink: _BucketBundleSink | None = None
                receipts: list[SymbolReceipt] = []
                isolated_failures: list[tuple[str, str]] = []
                current_symbol = ""
                try:
                    sink = _BucketBundleSink(
                        output_root=working_output,
                        bucket=bucket,
                        row_group_rows=int(parquet_row_group_rows),
                        dry_run=dry_run,
                        guard=guard,
                    )
                    for current_symbol in sorted(bucket_symbols):
                        guard.check(f"archive_symbol:{current_symbol}")
                        _write_runtime_state(
                            state_path,
                            connection,
                            status="scanning_archive",
                            job_id=job_id,
                            archive_path=archive,
                            archive_fingerprint=archive_fingerprint,
                            snapshot=snapshot,
                            selected_symbol_count=len(selected_symbols),
                            output_root=resolved_output,
                            dry_run=dry_run,
                            current_bucket=bucket,
                            current_symbol=current_symbol,
                        )
                        try:
                            receipt, buffered_days = _process_archive_symbol(
                                zip_handle=zip_handle,
                                symbol=current_symbol,
                                members=selected_members[current_symbol],
                                base_complete_dates=base_days_by_symbol.get(
                                    current_symbol,
                                    set(),
                                ),
                                base_incomplete_dates=base_incomplete_by_symbol.get(
                                    current_symbol,
                                    set(),
                                ),
                                start_date=start,
                                end_date=end,
                                csv_chunksize=int(csv_chunksize),
                                guard=guard,
                                source_name=normalized_source_name,
                            )
                        except (IntradayRepairResourceGuardError, KeyboardInterrupt, SystemExit):
                            raise
                        except Exception as exc:
                            isolated_failures.append(
                                (current_symbol, f"{type(exc).__name__}:{exc}")
                            )
                            continue
                        for frame, content_sha in buffered_days:
                            sink.append_day(frame, content_sha256=content_sha)
                        receipts.append(receipt)
                    _assert_snapshot_unchanged(snapshot)
                    records = sink.close_and_finalize()
                    _record_completed_bucket(
                        connection,
                        bucket=bucket,
                        selected_symbol_count=len(bucket_symbols),
                        base_complete_days=base_complete_count,
                        base_incomplete_days=base_incomplete_count,
                        receipts=receipts,
                        isolated_failures=isolated_failures,
                        bundle_records=records,
                        content_fingerprint=sink.content_fingerprint,
                    )
                    bundle_records.extend(records)
                except (IntradayRepairResourceGuardError, KeyboardInterrupt, SystemExit):
                    if sink is not None:
                        sink.abort()
                    raise
                except Exception as exc:
                    if sink is not None:
                        sink.abort()
                    _record_bucket_failure(
                        connection,
                        bucket=bucket,
                        selected_symbol_count=len(bucket_symbols),
                        current_symbol=current_symbol,
                        error=exc,
                    )

        failed = _failed_symbols(connection)
        if failed:
            status = "completed_with_failures"
        elif dry_run:
            status = "planned"
        else:
            _assert_snapshot_unchanged(snapshot, verify_shards=True)
            manifest_path, bundle_records, manifest_reused = _write_patch_manifest(
                connection,
                output_root=resolved_output,
                job_id=job_id,
                selection_fingerprint=selection_fingerprint,
                archive_path=archive,
                archive_fingerprint=archive_fingerprint,
                snapshot=snapshot,
                base_index_manifest=base_index_manifest,
                selected_symbols=selected_symbols,
                selection_source=selection_source,
                source_name=normalized_source_name,
            )
            status = "reused" if manifest_reused and not any_new_bucket else "completed"
        _write_runtime_state(
            state_path,
            connection,
            status=status,
            job_id=job_id,
            archive_path=archive,
            archive_fingerprint=archive_fingerprint,
            snapshot=snapshot,
            selected_symbol_count=len(selected_symbols),
            output_root=resolved_output,
            dry_run=dry_run,
            manifest_path=manifest_path,
        )
        return _result_payload(
            connection,
            status=status,
            job_id=job_id,
            archive_path=archive,
            archive_fingerprint=archive_fingerprint,
            snapshot=snapshot,
            selected_symbol_count=len(selected_symbols),
            selection_source=selection_source,
            output_root=resolved_output,
            runtime_root=fixed_runtime,
            state_path=None if dry_run else state_path,
            database_path=None if dry_run else database_path,
            manifest_path=manifest_path,
            bundles=bundle_records,
            dry_run=dry_run,
        )
    except IntradayRepairResourceGuardError as exc:
        _write_runtime_state(
            state_path,
            connection,
            status="paused_memory_guard",
            job_id=job_id,
            archive_path=archive,
            archive_fingerprint=archive_fingerprint,
            snapshot=snapshot,
            selected_symbol_count=len(selected_symbols),
            output_root=resolved_output,
            dry_run=dry_run,
            error=str(exc),
        )
        return _result_payload(
            connection,
            status="paused_memory_guard",
            job_id=job_id,
            archive_path=archive,
            archive_fingerprint=archive_fingerprint,
            snapshot=snapshot,
            selected_symbol_count=len(selected_symbols),
            selection_source=selection_source,
            output_root=resolved_output,
            runtime_root=fixed_runtime,
            state_path=None if dry_run else state_path,
            database_path=None if dry_run else database_path,
            manifest_path=None,
            bundles=_bundle_records(connection),
            dry_run=dry_run,
            error=str(exc),
        )
    except (KeyboardInterrupt, SystemExit):
        _write_runtime_state(
            state_path,
            connection,
            status="interrupted_recoverable",
            job_id=job_id,
            archive_path=archive,
            archive_fingerprint=archive_fingerprint,
            snapshot=snapshot,
            selected_symbol_count=len(selected_symbols),
            output_root=resolved_output,
            dry_run=dry_run,
        )
        raise
    except Exception as exc:
        _write_runtime_state(
            state_path,
            connection,
            status="failed_recoverable",
            job_id=job_id,
            archive_path=archive,
            archive_fingerprint=archive_fingerprint,
            snapshot=snapshot,
            selected_symbol_count=len(selected_symbols),
            output_root=resolved_output,
            dry_run=dry_run,
            error=f"{type(exc).__name__}:{exc}",
        )
        raise
    finally:
        connection.close()
        if temporary is not None:
            temporary.cleanup()
        _release_job_process_lock(job_lock)


def bulk_append_intraday_5m_patch(
    *,
    plan_manifest_path: str | Path,
    workspace_root: str | Path | None = None,
    execute: bool = False,
) -> dict[str, Any]:
    """Validate and optionally append one planner-produced missing-day patch.

    ``execute=False`` is a read-only preflight.  ``execute=True`` accepts only
    the current planner contract, proves that every bundle contains complete
    48-bar stock-days absent from the frozen base day index, and delegates one
    append-only manifest CAS to :func:`mutate_active_shards_from_parquet`.
    The active pointer and dataset id are immutable.  Replaying an already
    committed plan is idempotent through the repair mutation id recorded in
    the dataset manifest.
    """

    path = Path(plan_manifest_path).expanduser().resolve()
    workspace = qdp_paths(workspace_root).workspace_root.resolve()
    if not _is_within(path, workspace):
        raise IntradayRepairError(
            f"intraday_repair_patch_manifest_outside_workspace:{path}"
        )
    payload = read_json(path)
    if not payload or payload.get("repair_plan_version") != REPAIR_PLAN_VERSION:
        raise IntradayRepairError(f"intraday_repair_patch_manifest_invalid:{path}")
    _validate_apply_manifest_contract(payload, manifest_path=path)
    archive_provenance = Path(str(payload.get("archive_path", "") or ""))
    if not archive_provenance.is_absolute():
        raise IntradayRepairError("intraday_repair_archive_provenance_not_absolute")
    if (
        workspace.drive
        and archive_provenance.drive
        and workspace.drive.casefold() != archive_provenance.drive.casefold()
    ):
        raise IntradayRepairError(
            "intraday_repair_archive_provenance_not_on_workspace_volume"
        )
    root = qdp_v2_root(workspace_root).resolve()
    if not _is_within(root, workspace):
        raise IntradayRepairError(f"intraday_repair_qdp_root_outside_workspace:{root}")
    snapshot = _resolve_base_snapshot(workspace_root)
    for frozen_path in (snapshot.active_path, snapshot.manifest_path, *snapshot.shard_paths):
        if not _is_within(frozen_path, workspace):
            raise IntradayRepairError(
                f"intraday_repair_active_path_outside_workspace:{frozen_path}"
            )
    if str(payload.get("base_dataset_id", "") or "") != snapshot.dataset_id:
        raise IntradayRepairError("intraday_repair_patch_base_dataset_is_not_active")
    if str(payload.get("base_active_sha256", "") or "") != snapshot.active_sha256:
        raise IntradayRepairError("intraday_repair_patch_base_active_hash_mismatch")
    if Path(str(payload.get("base_active_path", "") or "")).resolve() != snapshot.active_path:
        raise IntradayRepairError("intraday_repair_patch_base_active_path_mismatch")
    if Path(str(payload.get("base_manifest_path", "") or "")).resolve() != snapshot.manifest_path:
        raise IntradayRepairError("intraday_repair_patch_base_manifest_path_mismatch")

    records = list(payload.get("bundles", ()) or ())
    expected_mutation_id = _planned_append_mutation_id(payload, records=records)
    current_manifest = read_dataset_manifest(snapshot.manifest_path)
    already_applied = expected_mutation_id in dict(
        current_manifest.source.get("repair_shard_mutations", {}) or {}
    )
    base_manifest_matches = (
        str(payload.get("base_manifest_sha256", "") or "")
        == snapshot.manifest_sha256
    )
    if not base_manifest_matches and not already_applied:
        raise IntradayRepairError("intraday_repair_patch_base_manifest_hash_mismatch")
    expected_shards = str(payload.get("base_shard_stat_fingerprint", "") or "")
    if (
        base_manifest_matches
        and expected_shards != snapshot.shard_stat_fingerprint
    ):
        raise IntradayRepairError("intraday_repair_patch_base_shards_changed")
    base_index = _validate_apply_base_index(
        payload,
        workspace=workspace,
    )
    if not _validate_bundle_records(
        records,
        output_root=path.parent,
        expected_total=int(payload.get("patch_row_count", 0) or 0),
    ):
        raise IntradayRepairError(f"intraday_repair_patch_bundle_validation_failed:{path}")
    bundle_paths = [
        (path.parent / str(item.get("path", "") or "")).resolve()
        for item in records
    ]
    for bundle_path in bundle_paths:
        if not _is_within(bundle_path, workspace):
            raise IntradayRepairError(
                f"intraday_repair_patch_bundle_outside_workspace:{bundle_path}"
            )
    patch_symbols = _validate_apply_bundle_contents(
        payload,
        records=records,
        output_root=path.parent,
        base_index_manifest=base_index,
        qdp_root=root,
    )
    missing_archive = {
        str(item) for item in list(payload.get("eligible_missing_archive", ()) or ())
    }
    selected_symbols = {
        str(item) for item in list(payload.get("selected_symbols", ()) or ())
    }
    unexpected = sorted(patch_symbols.difference(selected_symbols))
    if unexpected:
        raise IntradayRepairError(
            f"intraday_repair_patch_contains_unselected_symbol:{unexpected[:20]}"
        )
    overlap = sorted(patch_symbols.intersection(missing_archive))
    if overlap:
        raise IntradayRepairError(
            f"intraday_repair_patch_contains_eligible_archive_gap:{overlap[:20]}"
        )
    plan = {
        "status": "planned",
        "execute": bool(execute),
        "qdp_v2_root": str(root),
        "plan_manifest_path": str(path),
        "base_dataset_id": str(payload.get("base_dataset_id", "") or ""),
        "patch_row_count": int(payload.get("patch_row_count", 0) or 0),
        "patch_stock_day_count": int(payload.get("patch_stock_day_count", 0) or 0),
        "bundle_count": len(records),
        "eligible_missing_archive_count": int(
            payload.get("eligible_missing_archive_count", 0) or 0
        ),
        "eligible_missing_archive": sorted(missing_archive),
        "active_mutated": False,
        "active_pointer_mutated": False,
    }
    if not execute:
        return plan
    if not bundle_paths:
        return {
            **plan,
            "status": "reused",
            "execute": True,
            "mutation_id": "",
            "dataset_id_unchanged": True,
        }

    # Keep the validation snapshot live until the mutation API takes its own
    # CAS snapshot.  The latter also validates PK overlap against active data.
    _assert_snapshot_unchanged(snapshot, verify_shards=True)
    active_before = snapshot.active_path.read_bytes()
    mutation = mutate_active_shards_from_parquet(
        DOMAIN,
        appends=bundle_paths,
        reason=(
            "append planner-validated complete 5m missing days:"
            f"{str(payload.get('job_id', '') or path.stem)}"
        ),
        workspace_root=workspace_root,
        primary_keys_prevalidated=True,
        expected_mutation_id=expected_mutation_id,
    )
    if str(mutation.get("mutation_id", "") or "") != expected_mutation_id:
        raise IntradayRepairError("intraday_repair_append_mutation_id_mismatch")
    if snapshot.active_path.read_bytes() != active_before:
        raise IntradayRepairError("intraday_repair_active_pointer_was_mutated")
    active_after = read_active_manifest(root)
    if str(active_dataset_map(active_after).get(DOMAIN, "") or "") != snapshot.dataset_id:
        raise IntradayRepairError("intraday_repair_active_dataset_id_was_mutated")
    installed_paths = [
        Path(str(item)).resolve()
        for item in list(mutation.get("new_shard_paths", ()) or ())
    ]
    if any(not _is_within(item, workspace) for item in installed_paths):
        raise IntradayRepairError("intraday_repair_installed_shard_outside_workspace")
    committed_manifest = read_dataset_manifest(snapshot.manifest_path)
    if expected_mutation_id not in dict(
        committed_manifest.source.get("repair_shard_mutations", {}) or {}
    ):
        raise IntradayRepairError("intraday_repair_append_mutation_not_recorded")
    mutation_status = str(mutation.get("status", "") or "")
    return {
        **plan,
        "status": "applied" if mutation_status == "mutated" else "reused",
        "execute": True,
        "active_mutated": mutation_status == "mutated",
        "active_pointer_mutated": False,
        "dataset_id_unchanged": True,
        "mutation_id": expected_mutation_id,
        "manifest_path": str(snapshot.manifest_path),
        "manifest_sha256_before": str(
            mutation.get("manifest_sha256_before", snapshot.manifest_sha256)
        ),
        "manifest_sha256_after": _sha256_file(snapshot.manifest_path),
        "installed_shard_paths": [str(item) for item in installed_paths],
        "appended_bundle_count": int(mutation.get("appended_count", 0) or 0),
        "reused_bundle_count": int(
            mutation.get("reused_append_count", 0) or 0
        ),
    }


def _validate_apply_manifest_contract(
    payload: Mapping[str, Any],
    *,
    manifest_path: Path,
) -> None:
    exact = {
        "status": "completed",
        "domain": DOMAIN,
        "storage_layout": "natural_year_x_stable_16_symbol_bucket",
        "checkpoint_unit": "bucket",
        "sqlite_detail_policy": "receipts_and_counts_only_no_bar_rows",
        "primary_key": list(PATCH_PRIMARY_KEY),
        "columns": list(PATCH_COLUMNS),
        "expected_5m_bar_ends": list(EXPECTED_5M_BAR_ENDS),
        "auction_0930_policy": (
            "merge_single_0930_call_auction_into_0935_preserving_ohlc_volume_amount"
        ),
        "whole_stock_day_only": True,
        "merge_semantics": "append_only_completely_absent_stock_days",
        "incomplete_base_day_policy": (
            "skip_and_report_never_append_over_existing_keys"
        ),
        "code_aliases": dict(CURRENT_RESEARCH_CODE_ALIASES),
        "active_mutated": False,
    }
    for key, expected in exact.items():
        if payload.get(key) != expected:
            raise IntradayRepairError(
                f"intraday_repair_patch_contract_mismatch:{key}:{manifest_path}"
            )

    count_fields = (
        "target_symbol_count",
        "archive_selected_symbol_count",
        "eligible_missing_archive_count",
        "selected_symbol_count",
        "patch_stock_day_count",
        "incomplete_existing_stock_days_skipped",
        "patch_row_count",
        "bundle_count",
    )
    counts: dict[str, int] = {}
    for key in count_fields:
        value = payload.get(key)
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise IntradayRepairError(
                f"intraday_repair_patch_count_invalid:{key}:{manifest_path}"
            )
        counts[key] = int(value)
    if counts["patch_row_count"] != counts["patch_stock_day_count"] * 48:
        raise IntradayRepairError("intraday_repair_patch_stock_day_row_count_mismatch")
    records = payload.get("bundles")
    if not isinstance(records, list) or counts["bundle_count"] != len(records):
        raise IntradayRepairError("intraday_repair_patch_bundle_count_mismatch")
    if counts["patch_row_count"] == 0 and records:
        raise IntradayRepairError("intraday_repair_empty_patch_has_bundles")
    if counts["patch_row_count"] > 0 and not records:
        raise IntradayRepairError("intraday_repair_nonempty_patch_has_no_bundles")

    missing = payload.get("eligible_missing_archive")
    if not isinstance(missing, list):
        raise IntradayRepairError("intraday_repair_eligible_gap_report_missing")
    normalized_missing: list[str] = []
    for item in missing:
        symbol = str(item or "")
        try:
            canonical = _canonical_symbol(symbol)
        except ValueError as exc:
            raise IntradayRepairError(
                f"intraday_repair_eligible_gap_symbol_invalid:{symbol}"
            ) from exc
        if canonical != symbol or not _is_mainboard_a_symbol(symbol):
            raise IntradayRepairError(
                f"intraday_repair_eligible_gap_symbol_invalid:{symbol}"
            )
        normalized_missing.append(symbol)
    if normalized_missing != sorted(set(normalized_missing)):
        raise IntradayRepairError("intraday_repair_eligible_gap_report_not_canonical")
    if len(normalized_missing) != counts["eligible_missing_archive_count"]:
        raise IntradayRepairError("intraday_repair_eligible_gap_count_mismatch")
    if counts["selected_symbol_count"] != counts["archive_selected_symbol_count"]:
        raise IntradayRepairError("intraday_repair_selected_symbol_count_mismatch")
    if (
        counts["archive_selected_symbol_count"]
        + counts["eligible_missing_archive_count"]
        != counts["target_symbol_count"]
    ):
        raise IntradayRepairError("intraday_repair_target_symbol_count_mismatch")
    selected = payload.get("selected_symbols")
    if not isinstance(selected, list):
        raise IntradayRepairError("intraday_repair_selected_symbol_report_missing")
    normalized_selected: list[str] = []
    for item in selected:
        symbol = str(item or "")
        try:
            canonical = _canonical_symbol(symbol)
        except ValueError as exc:
            raise IntradayRepairError(
                f"intraday_repair_selected_symbol_invalid:{symbol}"
            ) from exc
        if canonical != symbol or not _is_mainboard_a_symbol(symbol):
            raise IntradayRepairError(
                f"intraday_repair_selected_symbol_invalid:{symbol}"
            )
        normalized_selected.append(symbol)
    if normalized_selected != sorted(set(normalized_selected)):
        raise IntradayRepairError("intraday_repair_selected_symbol_report_not_canonical")
    if len(normalized_selected) != counts["selected_symbol_count"]:
        raise IntradayRepairError("intraday_repair_selected_symbol_report_count_mismatch")
    if set(normalized_selected).intersection(normalized_missing):
        raise IntradayRepairError("intraday_repair_selected_and_missing_symbol_overlap")
    target_symbols = sorted({*normalized_selected, *normalized_missing})
    selected_sha = hashlib.sha256(
        "\n".join(normalized_selected).encode("utf-8")
    ).hexdigest()
    target_sha = hashlib.sha256("\n".join(target_symbols).encode("utf-8")).hexdigest()
    if str(payload.get("selected_symbols_sha256", "") or "") != selected_sha:
        raise IntradayRepairError("intraday_repair_selected_symbol_hash_mismatch")
    if str(payload.get("target_symbols_sha256", "") or "") != target_sha:
        raise IntradayRepairError("intraday_repair_target_symbol_hash_mismatch")

    try:
        start = _date_text(payload.get("start_date"))
        end = _date_text(payload.get("end_date"))
    except ValueError as exc:
        raise IntradayRepairError("intraday_repair_patch_date_range_invalid") from exc
    if start != payload.get("start_date") or end != payload.get("end_date") or start > end:
        raise IntradayRepairError("intraday_repair_patch_date_range_invalid")
    if not str(payload.get("selection_source", "") or "").strip():
        raise IntradayRepairError("intraday_repair_patch_selection_source_missing")
    for key in (
        "selection_fingerprint",
        "patch_content_fingerprint",
        "archive_fingerprint",
        "base_active_sha256",
        "base_manifest_sha256",
        "base_shard_stat_fingerprint",
        "base_day_index_manifest_sha256",
        "selected_symbols_sha256",
        "target_symbols_sha256",
    ):
        if not re.fullmatch(r"[0-9a-f]{64}", str(payload.get(key, "") or "")):
            raise IntradayRepairError(f"intraday_repair_patch_hash_invalid:{key}")


def _planned_append_mutation_id(
    payload: Mapping[str, Any],
    *,
    records: Sequence[Mapping[str, Any]],
) -> str:
    hashes = [str(item.get("sha256", "") or "") for item in records]
    if any(not re.fullmatch(r"[0-9a-f]{64}", item) for item in hashes):
        raise IntradayRepairError("intraday_repair_patch_bundle_hash_invalid")
    if len(hashes) != len(set(hashes)):
        raise IntradayRepairError("intraday_repair_patch_duplicate_bundle_content")
    mutation_payload = {
        "version": 1,
        "domain": DOMAIN,
        "dataset_id": str(payload.get("base_dataset_id", "") or ""),
        "replacements": [],
        "removals": [],
        "appends": sorted(hashes),
    }
    encoded = json.dumps(
        mutation_payload,
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return f"shard-mutation-v1:{hashlib.sha256(encoded).hexdigest()}"


def _validate_apply_base_index(
    payload: Mapping[str, Any],
    *,
    workspace: Path,
) -> Path:
    raw_path = Path(str(payload.get("base_day_index_manifest", "") or ""))
    if not raw_path.is_absolute():
        raise IntradayRepairError("intraday_repair_patch_base_day_index_not_absolute")
    manifest_path = raw_path.resolve()
    if not _is_within(manifest_path, workspace):
        raise IntradayRepairError(
            f"intraday_repair_base_day_index_outside_workspace:{manifest_path}"
        )
    expected_hash = str(payload.get("base_day_index_manifest_sha256", "") or "")
    if not manifest_path.is_file() or _sha256_file(manifest_path) != expected_hash:
        raise IntradayRepairError("intraday_repair_patch_base_day_index_invalid")
    index = read_json(manifest_path)
    exact = {
        "version": 2,
        "engine": "duckdb_symbol_date_cross_shard_count_v1",
        "base_dataset_id": str(payload.get("base_dataset_id", "") or ""),
        "base_manifest_sha256": str(payload.get("base_manifest_sha256", "") or ""),
        "base_shard_stat_fingerprint": str(
            payload.get("base_shard_stat_fingerprint", "") or ""
        ),
        "start_date": str(payload.get("start_date", "") or ""),
        "end_date": str(payload.get("end_date", "") or ""),
        "selected_symbol_count": int(payload.get("selected_symbol_count", 0) or 0),
        "completeness_rule": (
            "row_count_equals_48_under_frozen_active_primary_key"
        ),
        "fingerprint": _stable_hash(
            {
                "engine": "duckdb_symbol_date_cross_shard_count_v1",
                "base_manifest_sha256": str(
                    payload.get("base_manifest_sha256", "") or ""
                ),
                "base_shard_stat_fingerprint": str(
                    payload.get("base_shard_stat_fingerprint", "") or ""
                ),
                "selected_symbols": list(payload.get("selected_symbols", ()) or ()),
                "start_date": str(payload.get("start_date", "") or ""),
                "end_date": str(payload.get("end_date", "") or ""),
            }
        ),
    }
    for key, expected in exact.items():
        if index.get(key) != expected:
            raise IntradayRepairError(
                f"intraday_repair_base_day_index_contract_mismatch:{key}"
            )
    files = index.get("files")
    if not isinstance(files, list):
        raise IntradayRepairError("intraday_repair_base_day_index_files_invalid")
    try:
        index_valid = _validate_base_index_manifest(
            index,
            manifest_path=manifest_path,
        )
    except Exception as exc:
        raise IntradayRepairError(
            "intraday_repair_patch_base_day_index_invalid"
        ) from exc
    if not index_valid:
        raise IntradayRepairError("intraday_repair_patch_base_day_index_invalid")
    seen_buckets: set[int] = set()
    for item in files:
        if not isinstance(item, Mapping):
            raise IntradayRepairError("intraday_repair_base_day_index_record_invalid")
        bucket = item.get("bucket")
        if isinstance(bucket, bool) or not isinstance(bucket, int):
            raise IntradayRepairError("intraday_repair_base_day_index_bucket_invalid")
        bucket = int(bucket)
        if bucket in seen_buckets or not 0 <= bucket < BUCKET_COUNT:
            raise IntradayRepairError("intraday_repair_base_day_index_bucket_invalid")
        seen_buckets.add(bucket)
        expected_relative = Path(f"bucket={bucket:02d}.parquet")
        if Path(str(item.get("path", "") or "")) != expected_relative:
            raise IntradayRepairError("intraday_repair_base_day_index_path_invalid")
        file_path = (manifest_path.parent / expected_relative).resolve()
        if not _is_within(file_path, workspace):
            raise IntradayRepairError(
                f"intraday_repair_base_day_index_file_outside_workspace:{file_path}"
            )
        try:
            parquet = pq.ParquetFile(file_path)
            try:
                schema_matches = parquet.schema_arrow.equals(
                    _BASE_DAY_SUMMARY_SCHEMA,
                    check_metadata=False,
                )
            finally:
                parquet.close()
        except Exception as exc:
            raise IntradayRepairError(
                "intraday_repair_base_day_index_schema_invalid"
            ) from exc
        if not schema_matches:
            raise IntradayRepairError("intraday_repair_base_day_index_schema_invalid")
    return manifest_path


def _validate_apply_bundle_contents(
    payload: Mapping[str, Any],
    *,
    records: Sequence[Mapping[str, Any]],
    output_root: Path,
    base_index_manifest: Path,
    qdp_root: Path,
) -> set[str]:
    if not records:
        return set()

    bundle_paths = [
        (output_root / str(item.get("path", "") or "")).resolve()
        for item in records
    ]
    expected_by_path = {
        path: (str(item.get("year", "") or ""), int(item.get("bucket", -1)))
        for path, item in zip(bundle_paths, records, strict=True)
    }
    if len(expected_by_path) != len(bundle_paths):
        raise IntradayRepairError("intraday_repair_patch_bundle_inventory_duplicate")
    records_by_bucket: dict[int, list[tuple[Path, Mapping[str, Any]]]] = {}
    for path, record in zip(bundle_paths, records, strict=True):
        records_by_bucket.setdefault(int(record.get("bucket", -1)), []).append(
            (path, record)
        )
    index_payload = read_json(base_index_manifest)
    base_index_by_bucket = {
        int(item.get("bucket", -1)): (
            base_index_manifest.parent / str(item.get("path", "") or "")
        ).resolve()
        for item in list(index_payload.get("files", ()) or ())
    }
    _MemoryGuard(0.5, 5.0).check("patch_apply_validation_start")
    spill_root = (qdp_root / "tmp" / "intraday_apply_validation").resolve()
    spill_root.mkdir(parents=True, exist_ok=True)
    database = open_guarded_duckdb(
        temp_directory=spill_root,
        threads=max(1, min(os.cpu_count() or 1, 8)),
        error_factory=IntradayRepairResourceGuardError,
    )
    try:
        allowed_bar_times = ",".join(
            "'" + _legacy_bar_time(item).replace("'", "''") + "'"
            for item in EXPECTED_5M_BAR_ENDS
        )
        start = str(payload.get("start_date", "") or "")
        end = str(payload.get("end_date", "") or "")
        source_name = str(payload.get("source_name", "") or SOURCE_NAME)
        expected_rows = int(payload.get("patch_row_count", 0) or 0)
        expected_days = int(payload.get("patch_stock_day_count", 0) or 0)
        seen_paths: set[Path] = set()
        patch_symbols: set[str] = set()
        actual_rows = 0
        actual_days = 0
        for bucket, bucket_records in sorted(records_by_bucket.items()):
            bucket_paths = [item[0] for item in bucket_records]
            expected_bucket_paths = set(bucket_paths)
            database.execute(
                "CREATE OR REPLACE TEMP VIEW patch_rows AS SELECT * FROM read_parquet("
                + _duckdb_path_list(bucket_paths)
                + ", union_by_name=false, hive_partitioning=false, filename=true)"
            )
            rows = database.execute(
                f"""
                SELECT filename, count(*) AS row_count,
                       count(*) FILTER (
                           WHERE symbol IS NULL OR trade_date IS NULL OR bar_time IS NULL
                           OR open IS NULL OR high IS NULL OR low IS NULL OR close IS NULL
                           OR volume IS NULL OR amount IS NULL
                           OR source IS NULL OR adjusted_flag IS NULL
                           OR try_cast(trade_date AS DATE) IS NULL
                           OR CAST(trade_date AS VARCHAR) !=
                              strftime(try_cast(trade_date AS DATE), '%Y-%m-%d')
                           OR CAST(trade_date AS VARCHAR) NOT BETWEEN ? AND ?
                           OR CAST(bar_time AS VARCHAR) NOT IN ({allowed_bar_times})
                           OR NOT isfinite(open) OR NOT isfinite(high)
                           OR NOT isfinite(low) OR NOT isfinite(close)
                           OR NOT isfinite(volume) OR NOT isfinite(amount)
                           OR open <= 0 OR high <= 0 OR low <= 0 OR close <= 0
                           OR high < low OR high < open OR high < close
                           OR low > open OR low > close
                           OR volume < 0 OR amount < 0
                           OR source != ? OR adjusted_flag != 'none'
                       ) AS invalid_rows
                FROM patch_rows
                GROUP BY filename
                """,
                [start, end, source_name],
            ).fetchall()
            observed_bucket_paths: set[Path] = set()
            for filename, row_count, invalid_rows in rows:
                file_path = Path(str(filename)).resolve()
                expected = expected_by_path.get(file_path)
                if expected is None or file_path not in expected_bucket_paths:
                    raise IntradayRepairError(
                        "intraday_repair_patch_unknown_bundle_path"
                    )
                observed_bucket_paths.add(file_path)
                expected_record = next(
                    item for path, item in bucket_records if path == file_path
                )
                if (
                    int(row_count or 0)
                    != int(expected_record.get("row_count", 0) or 0)
                    or int(invalid_rows or 0)
                ):
                    raise IntradayRepairError(
                        "intraday_repair_patch_bundle_content_invalid"
                    )
                actual_rows += int(row_count or 0)
            if observed_bucket_paths != expected_bucket_paths:
                raise IntradayRepairError(
                    "intraday_repair_patch_bundle_inventory_mismatch"
                )

            days = database.execute(
                """
                SELECT count(*), coalesce(sum(row_count), 0),
                       count(*) FILTER (
                           WHERE row_count != 48 OR distinct_bar_times != 48
                       )
                FROM (
                    SELECT symbol, trade_date, count(*) AS row_count,
                           count(DISTINCT bar_time) AS distinct_bar_times
                    FROM patch_rows
                    GROUP BY symbol, trade_date
                )
                """
            ).fetchone()
            expected_bucket_days = sum(
                int(item.get("stock_day_count", 0) or 0)
                for _, item in bucket_records
            )
            expected_bucket_rows = sum(
                int(item.get("row_count", 0) or 0) for _, item in bucket_records
            )
            if (
                days is None
                or int(days[0] or 0) != expected_bucket_days
                or int(days[1] or 0) != expected_bucket_rows
                or int(days[2] or 0)
            ):
                raise IntradayRepairError(
                    "intraday_repair_patch_requires_exact_48_bar_days"
                )
            actual_days += int(days[0] or 0)

            assignments = database.execute(
                """
                SELECT filename, CAST(symbol AS VARCHAR),
                       substr(CAST(trade_date AS VARCHAR), 1, 4)
                FROM patch_rows
                GROUP BY filename, symbol,
                         substr(CAST(trade_date AS VARCHAR), 1, 4)
                """
            ).fetchall()
            for filename, raw_symbol, year in assignments:
                file_path = Path(str(filename)).resolve()
                expected = expected_by_path.get(file_path)
                if expected is None:
                    raise IntradayRepairError(
                        "intraday_repair_patch_unknown_bundle_path"
                    )
                seen_paths.add(file_path)
                symbol = str(raw_symbol or "")
                try:
                    canonical = _canonical_symbol(symbol)
                except ValueError as exc:
                    raise IntradayRepairError(
                        f"intraday_repair_patch_symbol_invalid:{symbol}"
                    ) from exc
                if (
                    canonical != symbol
                    or not _is_mainboard_a_symbol(symbol)
                    or str(year) != expected[0]
                    or stable_symbol_bucket(symbol) != expected[1]
                ):
                    raise IntradayRepairError(
                        f"intraday_repair_patch_partition_assignment_invalid:{symbol}"
                    )
                patch_symbols.add(symbol)

            index_path = base_index_by_bucket.get(bucket)
            # The frozen index intentionally omits buckets with no base days.
            # A present bucket must prove non-overlap; an absent one is already
            # an authenticated empty partition of the validated index manifest.
            if index_path is not None:
                overlap = database.execute(
                    """
                    SELECT p.symbol, p.trade_date
                    FROM (SELECT DISTINCT symbol, trade_date FROM patch_rows) AS p
                    INNER JOIN read_parquet(?) AS b
                        ON p.symbol = b.symbol AND p.trade_date = b.trade_date
                    LIMIT 1
                    """,
                    [str(index_path)],
                ).fetchone()
                if overlap:
                    raise IntradayRepairError(
                        "intraday_repair_patch_overlaps_any_frozen_base_stock_day:"
                        f"{overlap[0]}:{overlap[1]}"
                    )
            _MemoryGuard(0.5, 5.0).check(
                f"patch_apply_validation_bucket:{bucket}"
            )

        if actual_rows != expected_rows:
            raise IntradayRepairError("intraday_repair_patch_bundle_content_invalid")
        if actual_days != expected_days:
            raise IntradayRepairError(
                "intraday_repair_patch_requires_exact_48_bar_days"
            )
        if seen_paths != set(bundle_paths):
            raise IntradayRepairError("intraday_repair_patch_bundle_inventory_mismatch")
        _MemoryGuard(0.5, 5.0).check("patch_apply_validation_complete")
        return patch_symbols
    finally:
        database.close()
        for item in spill_root.glob("*"):
            if item.is_file() and _is_within(item, spill_root):
                _unlink_file_with_retry(item)
        try:
            spill_root.rmdir()
        except OSError:
            pass


def _duckdb_path_list(paths: Sequence[Path]) -> str:
    return "[" + ",".join(
        "'" + str(item).replace("'", "''") + "'" for item in paths
    ) + "]"


def stable_symbol_bucket(symbol: str, *, bucket_count: int = BUCKET_COUNT) -> int:
    normalized = _canonical_symbol(symbol)
    if int(bucket_count) <= 0:
        raise ValueError("intraday_repair_bucket_count_must_be_positive")
    digest = hashlib.sha256(normalized.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], byteorder="big", signed=False) % int(bucket_count)


def _resolve_base_snapshot(workspace_root: str | Path | None) -> BaseSnapshot:
    root = qdp_v2_root(workspace_root).resolve()
    active_path = root / "active" / "active.json"
    if not active_path.is_file():
        raise FileNotFoundError(f"intraday_repair_active_missing:{active_path}")
    active = read_active_manifest(root)
    dataset_id = str(active_dataset_map(active).get(DOMAIN, "") or "")
    if not dataset_id:
        raise IntradayRepairError("intraday_repair_active_5m_dataset_missing")
    manifest_path = dataset_manifest_for_id(root, dataset_id, DOMAIN)
    if manifest_path is None or not manifest_path.is_file():
        raise FileNotFoundError(f"intraday_repair_base_manifest_missing:{dataset_id}")
    manifest = read_dataset_manifest(manifest_path)
    if manifest.domain != DOMAIN or manifest.dataset_id != dataset_id:
        raise IntradayRepairError("intraday_repair_base_manifest_identity_mismatch")
    if tuple(manifest.primary_key) != ("trade_date", "symbol", "bar_time"):
        raise IntradayRepairError(
            f"intraday_repair_base_primary_key_invalid:{manifest.primary_key}"
        )
    if not manifest.shards:
        raise IntradayRepairError("intraday_repair_base_shards_empty")
    shard_paths: list[Path] = []
    shard_stats: list[dict[str, Any]] = []
    seen: set[Path] = set()
    for shard in manifest.shards:
        path = resolve_manifest_path(shard.path, root=root).resolve()
        if path in seen:
            raise IntradayRepairError(f"intraday_repair_duplicate_base_shard:{path}")
        if not path.is_file():
            raise FileNotFoundError(f"intraday_repair_base_shard_missing:{path}")
        stat = path.stat()
        seen.add(path)
        shard_paths.append(path)
        shard_stats.append(
            {
                "path": str(path),
                "size": int(stat.st_size),
                "mtime_ns": int(stat.st_mtime_ns),
            }
        )
    return BaseSnapshot(
        root=root,
        active_path=active_path.resolve(),
        active_sha256=_sha256_file(active_path),
        dataset_id=dataset_id,
        manifest_path=manifest_path.resolve(),
        manifest_sha256=_sha256_file(manifest_path),
        shard_paths=tuple(shard_paths),
        shard_stat_fingerprint=_stable_hash({"shards": shard_stats}),
    )


def _assert_snapshot_unchanged(
    snapshot: BaseSnapshot,
    *,
    verify_shards: bool = False,
) -> None:
    if not snapshot.active_path.is_file() or _sha256_file(snapshot.active_path) != snapshot.active_sha256:
        raise IntradayRepairError("intraday_repair_active_changed_during_plan")
    if not snapshot.manifest_path.is_file() or _sha256_file(snapshot.manifest_path) != snapshot.manifest_sha256:
        raise IntradayRepairError("intraday_repair_base_manifest_changed_during_plan")
    if verify_shards:
        current: list[dict[str, Any]] = []
        for path in snapshot.shard_paths:
            if not path.is_file():
                raise IntradayRepairError(
                    f"intraday_repair_base_shard_changed_during_plan:{path}"
                )
            stat = path.stat()
            current.append(
                {
                    "path": str(path),
                    "size": int(stat.st_size),
                    "mtime_ns": int(stat.st_mtime_ns),
                }
            )
        if _stable_hash({"shards": current}) != snapshot.shard_stat_fingerprint:
            raise IntradayRepairError("intraday_repair_base_shards_changed_during_plan")


def _archive_inventory(archive_path: Path) -> tuple[dict[str, tuple[ArchiveMember, ...]], str]:
    grouped: dict[str, list[ArchiveMember]] = {}
    with zipfile.ZipFile(archive_path, mode="r", metadata_encoding="gbk") as archive:
        for info in archive.infolist():
            if info.is_dir():
                continue
            archive_symbol = _symbol_from_member_name(info.filename)
            if not archive_symbol:
                continue
            canonical = _canonical_symbol(archive_symbol)
            grouped.setdefault(canonical, []).append(
                ArchiveMember(
                    member_name=info.filename,
                    archive_symbol=archive_symbol,
                    canonical_symbol=canonical,
                    crc32=f"{int(info.CRC):08x}",
                    compressed_size=int(info.compress_size),
                    uncompressed_size=int(info.file_size),
                )
            )
    if not grouped:
        raise IntradayRepairError("intraday_repair_archive_has_no_symbol_csv")
    normalized = {
        symbol: tuple(sorted(items, key=lambda item: (item.archive_symbol, item.member_name)))
        for symbol, items in sorted(grouped.items())
    }
    stat = archive_path.stat()
    fingerprint = _stable_hash(
        {
            "path": str(archive_path),
            "size": int(stat.st_size),
            "mtime_ns": int(stat.st_mtime_ns),
            "members": [
                item.fingerprint_record()
                for values in normalized.values()
                for item in values
            ],
        }
    )
    return normalized, fingerprint


def _default_eligible_symbols(
    *,
    workspace_root: str | Path | None,
    start_date: str,
    end_date: str,
) -> tuple[str, ...]:
    """Read the latest small Tushare stock_basic raw partition from this workspace."""

    from quant_data_platform.qdp_v3.constants import RAW_TUSHARE_PROXY_STOCK_BASIC
    from quant_data_platform.qdp_v3.storage import iter_raw_partitions, read_raw_partition

    refs = list(
        iter_raw_partitions(
            RAW_TUSHARE_PROXY_STOCK_BASIC,
            workspace_root=workspace_root,
        )
    )
    if not refs:
        raise FileNotFoundError("intraday_repair_tushare_stock_basic_raw_missing")
    frame = read_raw_partition(refs[-1])
    if "ts_code" not in frame.columns:
        raise IntradayRepairError("intraday_repair_stock_basic_ts_code_missing")
    list_dates = _compact_iso_date_series(
        frame["list_date"] if "list_date" in frame.columns else pd.Series("", index=frame.index)
    )
    delist_dates = _compact_iso_date_series(
        frame["delist_date"] if "delist_date" in frame.columns else pd.Series("", index=frame.index)
    )
    symbols: set[str] = set()
    for raw_symbol, list_date, delist_date in zip(
        frame["ts_code"], list_dates, delist_dates, strict=True
    ):
        try:
            symbol = _canonical_symbol(raw_symbol)
        except ValueError:
            continue
        if not _is_mainboard_a_symbol(symbol):
            continue
        if list_date and list_date > end_date:
            continue
        if delist_date and delist_date < start_date:
            continue
        symbols.add(symbol)
    if not symbols:
        raise IntradayRepairError("intraday_repair_stock_basic_eligible_symbols_empty")
    return tuple(sorted(symbols))


def _initialize_database(connection: sqlite3.Connection) -> None:
    connection.executescript(
        """
        CREATE TABLE IF NOT EXISTS metadata (
            key TEXT PRIMARY KEY,
            value TEXT NOT NULL
        );
        CREATE TABLE IF NOT EXISTS bucket_receipts (
            bucket INTEGER PRIMARY KEY,
            status TEXT NOT NULL,
            selected_symbols INTEGER NOT NULL DEFAULT 0,
            completed_symbols INTEGER NOT NULL DEFAULT 0,
            failed_symbols INTEGER NOT NULL DEFAULT 0,
            base_complete_days INTEGER NOT NULL DEFAULT 0,
            base_incomplete_days INTEGER NOT NULL DEFAULT 0,
            accepted_days INTEGER NOT NULL DEFAULT 0,
            existing_days INTEGER NOT NULL DEFAULT 0,
            incomplete_existing_days INTEGER NOT NULL DEFAULT 0,
            rejected_days INTEGER NOT NULL DEFAULT 0,
            conflict_days INTEGER NOT NULL DEFAULT 0,
            patch_rows INTEGER NOT NULL DEFAULT 0,
            content_fingerprint TEXT NOT NULL DEFAULT '',
            current_symbol TEXT NOT NULL DEFAULT '',
            error TEXT NOT NULL DEFAULT '',
            completed_at TEXT NOT NULL
        );
        CREATE TABLE IF NOT EXISTS symbol_receipts (
            symbol TEXT PRIMARY KEY,
            bucket INTEGER NOT NULL,
            status TEXT NOT NULL,
            accepted_days INTEGER NOT NULL DEFAULT 0,
            existing_days INTEGER NOT NULL DEFAULT 0,
            incomplete_existing_days INTEGER NOT NULL DEFAULT 0,
            rejected_days INTEGER NOT NULL DEFAULT 0,
            conflict_days INTEGER NOT NULL DEFAULT 0,
            patch_rows INTEGER NOT NULL DEFAULT 0,
            reject_sample_json TEXT NOT NULL DEFAULT '[]',
            content_fingerprint TEXT NOT NULL DEFAULT '',
            error TEXT NOT NULL DEFAULT '',
            completed_at TEXT NOT NULL
        );
        CREATE INDEX IF NOT EXISTS symbol_receipts_bucket
            ON symbol_receipts(bucket, status, symbol);
        CREATE TABLE IF NOT EXISTS bundle_receipts (
            year TEXT NOT NULL,
            bucket INTEGER NOT NULL,
            path TEXT NOT NULL,
            row_count INTEGER NOT NULL,
            stock_day_count INTEGER NOT NULL,
            file_size INTEGER NOT NULL,
            sha256 TEXT NOT NULL,
            PRIMARY KEY (year, bucket)
        ) WITHOUT ROWID;
        """
    )
    connection.commit()


def _bind_job_metadata(
    connection: sqlite3.Connection,
    *,
    job_id: str,
    selection_fingerprint: str,
    archive_path: Path,
    archive_fingerprint: str,
    snapshot: BaseSnapshot,
    start_date: str,
    end_date: str,
    selection_source: str,
    source_name: str,
    output_root: Path,
    target_symbols: Sequence[str],
    selected_symbols: Sequence[str],
    eligible_missing_archive: Sequence[str],
) -> None:
    expected = {
        "repair_plan_version": str(REPAIR_PLAN_VERSION),
        "job_id": job_id,
        "selection_fingerprint": selection_fingerprint,
        "archive_path": str(archive_path),
        "archive_fingerprint": archive_fingerprint,
        "base_active_sha256": snapshot.active_sha256,
        "base_dataset_id": snapshot.dataset_id,
        "base_manifest_sha256": snapshot.manifest_sha256,
        "base_shard_stat_fingerprint": snapshot.shard_stat_fingerprint,
        "start_date": start_date,
        "end_date": end_date,
        "selection_source": selection_source,
        "source_name": _get_meta(connection, "source_name") or SOURCE_NAME,
        "source_name": source_name,
        "output_root": str(output_root.resolve()),
        "target_symbol_count": str(len(target_symbols)),
        "archive_selected_symbol_count": str(len(selected_symbols)),
        "eligible_missing_archive_count": str(len(eligible_missing_archive)),
        "eligible_missing_archive_json": json.dumps(
            list(eligible_missing_archive),
            ensure_ascii=False,
            separators=(",", ":"),
        ),
        "target_symbols_sha256": hashlib.sha256(
            "\n".join(target_symbols).encode("utf-8")
        ).hexdigest(),
        "selected_symbols_sha256": hashlib.sha256(
            "\n".join(selected_symbols).encode("utf-8")
        ).hexdigest(),
    }
    for key, value in expected.items():
        current = _get_meta(connection, key)
        if current and current != value:
            raise IntradayRepairError(
                f"intraday_repair_runtime_fingerprint_mismatch:{key}:"
                f"expected={value}:actual={current}"
            )
        _set_meta(connection, key, value)
    connection.commit()



# This definition intentionally replaces the older Arrow batch prototype
# above.  The production path lets DuckDB aggregate only symbol/date across
# every shard, avoiding a 400M-row pandas materialization while preserving
# cross-shard stock-day counts.
def _ensure_base_day_summary_index(
    connection: sqlite3.Connection,
    *,
    snapshot: BaseSnapshot,
    selected_symbols: Sequence[str],
    start_date: str,
    end_date: str,
    index_root: Path,
    state_path: Path,
    scan_batch_rows: int,
    guard: _MemoryGuard,
) -> Path:
    del scan_batch_rows

    index_root.mkdir(parents=True, exist_ok=True)
    manifest_path = index_root / "index_manifest.json"
    fingerprint = _stable_hash(
        {
            "engine": "duckdb_symbol_date_cross_shard_count_v1",
            "base_manifest_sha256": snapshot.manifest_sha256,
            "base_shard_stat_fingerprint": snapshot.shard_stat_fingerprint,
            "selected_symbols": list(selected_symbols),
            "start_date": start_date,
            "end_date": end_date,
        }
    )
    existing = read_json(manifest_path)
    if (
        existing
        and existing.get("fingerprint") == fingerprint
        and _validate_base_index_manifest(existing, manifest_path=manifest_path)
    ):
        _set_meta(connection, "base_shards_completed", str(len(snapshot.shard_paths)))
        connection.commit()
        return manifest_path

    for path in index_root.glob("bucket=*.parquet"):
        if _is_within(path, index_root):
            _unlink_file_with_retry(path)
    for path in index_root.glob(".bucket=*.parquet.*.tmp"):
        if _is_within(path, index_root):
            _unlink_file_with_retry(path)
    _unlink_file_with_retry(manifest_path)
    spill_root = index_root / ".duckdb_spill"
    spill_root.mkdir(parents=True, exist_ok=True)

    guard.check("base_duckdb_start")
    selected = pd.DataFrame(
        {
            "symbol": list(selected_symbols),
            "bucket": [stable_symbol_bucket(symbol) for symbol in selected_symbols],
        }
    )
    database = open_guarded_duckdb(
        temp_directory=spill_root,
        threads=max(1, min(os.cpu_count() or 1, 8)),
        floor_bytes=guard.floor_bytes,
        low_memory_seconds=guard.continuous_seconds,
        error_factory=IntradayRepairResourceGuardError,
    )
    records: list[dict[str, Any]] = []
    created_targets: list[Path] = []
    temporary_paths: list[Path] = []
    try:
        database.register("selected_symbol_frame", selected)
        database.execute(
            "CREATE TEMP TABLE selected_symbols AS "
            "SELECT CAST(symbol AS VARCHAR) AS symbol, CAST(bucket AS INTEGER) AS bucket "
            "FROM selected_symbol_frame"
        )
        database.from_parquet(
            [str(path) for path in snapshot.shard_paths],
            union_by_name=True,
        ).create_view("active_intraday_rows")
        alias_cases = " ".join(
            f"WHEN '{source}' THEN '{target}'"
            for source, target in CURRENT_RESEARCH_CODE_ALIASES.items()
        )
        database.execute(
            f"""
            CREATE TEMP TABLE base_day_counts AS
            WITH normalized AS (
                SELECT
                    CASE upper(trim(CAST(symbol AS VARCHAR)))
                        {alias_cases}
                        ELSE upper(trim(CAST(symbol AS VARCHAR)))
                    END AS symbol,
                    substr(CAST(trade_date AS VARCHAR), 1, 10) AS trade_date
                FROM active_intraday_rows
            )
            SELECT
                selected.symbol,
                selected.bucket,
                normalized.trade_date,
                CAST(count(*) AS INTEGER) AS row_count
            FROM normalized
            INNER JOIN selected_symbols AS selected USING (symbol)
            WHERE normalized.trade_date BETWEEN ? AND ?
            GROUP BY selected.symbol, selected.bucket, normalized.trade_date
            """,
            [start_date, end_date],
        )
        _set_meta(connection, "base_shards_completed", str(len(snapshot.shard_paths)))
        connection.commit()
        state = read_json(state_path)
        state.update(
            {
                "status": "indexing_base",
                "base_shards_completed": len(snapshot.shard_paths),
                "base_shard_count": len(snapshot.shard_paths),
                "base_index_engine": "duckdb_symbol_date_cross_shard_count_v1",
                "heartbeat_at": _utc_now(),
            }
        )
        atomic_write_json(state_path, state)
        for bucket in sorted({stable_symbol_bucket(symbol) for symbol in selected_symbols}):
            guard.check(f"base_duckdb_export:{bucket}")
            row = database.execute(
                "SELECT count(*) FROM base_day_counts WHERE bucket = ?",
                [int(bucket)],
            ).fetchone()
            row_count = int(row[0] or 0) if row else 0
            if row_count == 0:
                continue
            target = index_root / f"bucket={bucket:02d}.parquet"
            temporary = target.with_name(
                f".{target.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp"
            )
            temporary_paths.append(temporary)
            quoted = str(temporary).replace("'", "''")
            database.execute(
                f"""
                COPY (
                    SELECT symbol, trade_date, row_count
                    FROM base_day_counts
                    WHERE bucket = {int(bucket)}
                    ORDER BY symbol, trade_date
                ) TO '{quoted}' (
                    FORMAT PARQUET,
                    COMPRESSION ZSTD,
                    ROW_GROUP_SIZE 100000
                )
                """
            )
            metadata = pq.ParquetFile(temporary)
            try:
                actual_rows = metadata.metadata.num_rows
                schema_matches = metadata.schema_arrow.equals(
                    _BASE_DAY_SUMMARY_SCHEMA,
                    check_metadata=False,
                )
            finally:
                metadata.close()
            if actual_rows != row_count or not schema_matches:
                raise IntradayRepairError(
                    f"intraday_repair_base_duckdb_export_invalid:{bucket}"
                )
            _replace_file_with_retry(temporary, target)
            created_targets.append(target)
            records.append(
                {
                    "bucket": int(bucket),
                    "path": target.relative_to(index_root).as_posix(),
                    "summary_row_count": row_count,
                    "file_size": int(target.stat().st_size),
                    "sha256": _sha256_file(target),
                }
            )
        _assert_snapshot_unchanged(snapshot, verify_shards=True)
        atomic_write_json(
            manifest_path,
            {
                "version": 2,
                "fingerprint": fingerprint,
                "engine": "duckdb_symbol_date_cross_shard_count_v1",
                "base_dataset_id": snapshot.dataset_id,
                "base_manifest_sha256": snapshot.manifest_sha256,
                "base_shard_stat_fingerprint": snapshot.shard_stat_fingerprint,
                "start_date": start_date,
                "end_date": end_date,
                "selected_symbol_count": len(selected_symbols),
                "completeness_rule": "row_count_equals_48_under_frozen_active_primary_key",
                "files": records,
                "created_at": _utc_now(),
            },
        )
        return manifest_path
    except BaseException:
        for path in temporary_paths:
            _unlink_file_with_retry(path)
        for path in created_targets:
            _unlink_file_with_retry(path)
        raise
    finally:
        database.close()
        for path in list(spill_root.glob("*")):
            if path.is_file() and _is_within(path, spill_root):
                _unlink_file_with_retry(path)
        try:
            spill_root.rmdir()
        except OSError:
            pass


def _validate_base_index_manifest(payload: Mapping[str, Any], *, manifest_path: Path) -> bool:
    for item in list(payload.get("files", ()) or ()):
        path = (manifest_path.parent / str(item.get("path", "") or "")).resolve()
        if not path.is_file():
            return False
        if int(item.get("file_size", 0) or 0) != int(path.stat().st_size):
            return False
        if str(item.get("sha256", "") or "") != _sha256_file(path):
            return False
        parquet = pq.ParquetFile(path)
        try:
            actual_rows = parquet.metadata.num_rows
        finally:
            parquet.close()
        if actual_rows != int(item.get("summary_row_count", 0) or 0):
            return False
    return True


def _load_base_complete_days(
    index_manifest_path: Path,
    *,
    bucket: int,
    guard: _MemoryGuard,
) -> tuple[dict[str, set[str]], dict[str, set[str]], int, int]:
    payload = read_json(index_manifest_path)
    record = next(
        (
            item
            for item in list(payload.get("files", ()) or ())
            if int(item.get("bucket", -1)) == int(bucket)
        ),
        None,
    )
    if record is None:
        return {}, {}, 0, 0
    path = (index_manifest_path.parent / str(record.get("path", "") or "")).resolve()
    complete_by_symbol: dict[str, set[str]] = {}
    incomplete_by_symbol: dict[str, set[str]] = {}
    complete = 0
    incomplete = 0
    parquet = pq.ParquetFile(path)
    try:
        for batch in parquet.iter_batches(
            batch_size=262_144,
            columns=["symbol", "trade_date", "row_count"],
        ):
            guard.check(f"load_base_index:{bucket}")
            values = batch.to_pydict()
            for symbol, trade_date, row_count in zip(
                values["symbol"],
                values["trade_date"],
                values["row_count"],
                strict=True,
            ):
                target = complete_by_symbol if int(row_count) == 48 else incomplete_by_symbol
                target.setdefault(str(symbol), set()).add(str(trade_date))
                if int(row_count) == 48:
                    complete += 1
                else:
                    incomplete += 1
    finally:
        parquet.close()
    return complete_by_symbol, incomplete_by_symbol, complete, incomplete


def _process_archive_symbol(
    *,
    zip_handle: zipfile.ZipFile,
    symbol: str,
    members: Sequence[ArchiveMember],
    base_complete_dates: set[str],
    base_incomplete_dates: set[str],
    start_date: str,
    end_date: str,
    csv_chunksize: int,
    guard: _MemoryGuard,
    source_name: str = SOURCE_NAME,
) -> tuple[SymbolReceipt, list[tuple[pd.DataFrame, str]]]:
    accepted = 0
    existing = 0
    incomplete_existing = 0
    rejected = 0
    conflicts = 0
    reject_sample: list[dict[str, str]] = []
    buffered_days: list[tuple[pd.DataFrame, str]] = []
    digest = hashlib.sha256()
    for trade_date, representations in _iter_archive_symbol_days(
        zip_handle,
        members,
        csv_chunksize=csv_chunksize,
        skip_dates=base_complete_dates | base_incomplete_dates,
        start_date=start_date,
        end_date=end_date,
    ):
        if trade_date < start_date or trade_date > end_date:
            continue
        guard.check(f"archive_day:{symbol}:{trade_date}")
        if trade_date in base_complete_dates:
            existing += 1
            continue
        if trade_date in base_incomplete_dates:
            incomplete_existing += 1
            continue
        normalized_days: list[tuple[ArchiveMember, pd.DataFrame, str]] = []
        invalid_reasons: list[str] = []
        for member, raw_day in representations:
            normalized, reason = _normalize_complete_archive_day(
                raw_day,
                canonical_symbol=symbol,
                trade_date=trade_date,
                source_name=source_name,
            )
            if reason:
                invalid_reasons.append(f"{reason}:{member.member_name}")
                continue
            normalized_days.append((member, normalized, _frame_sha256(normalized)))
        reason = ""
        if invalid_reasons:
            reason = "archive_representation_invalid:" + "|".join(sorted(invalid_reasons))
        elif not normalized_days:
            reason = "archive_stock_day_missing_valid_representation"
        elif len({item[2] for item in normalized_days}) != 1:
            reason = "archive_alias_stock_day_conflict"
            conflicts += 1
        if reason:
            rejected += 1
            if len(reject_sample) < 20:
                reject_sample.append({"trade_date": trade_date, "reason": reason})
            continue
        frame = normalized_days[0][1]
        content_sha = normalized_days[0][2]
        buffered_days.append((frame, content_sha))
        digest.update(f"{trade_date}\0{content_sha}\n".encode("utf-8"))
        accepted += 1
    return SymbolReceipt(
        symbol=symbol,
        bucket=stable_symbol_bucket(symbol),
        accepted_days=accepted,
        existing_days=existing,
        incomplete_existing_days=incomplete_existing,
        rejected_days=rejected,
        conflict_days=conflicts,
        patch_rows=accepted * 48,
        reject_sample_json=json.dumps(
            reject_sample,
            ensure_ascii=False,
            sort_keys=True,
        ),
        content_fingerprint=digest.hexdigest(),
    ), buffered_days


def _iter_archive_symbol_days(
    zip_handle: zipfile.ZipFile,
    members: Sequence[ArchiveMember],
    *,
    csv_chunksize: int,
    skip_dates: set[str] | None = None,
    start_date: str = "",
    end_date: str = "",
) -> Iterator[tuple[str, list[tuple[ArchiveMember, pd.DataFrame]]]]:
    skipped = skip_dates or set()
    iterators = [
        iter(
            _iter_archive_member_days(
                zip_handle,
                member,
                csv_chunksize=csv_chunksize,
                skip_dates=skipped,
                start_date=start_date,
                end_date=end_date,
            )
        )
        for member in members
    ]
    heap: list[tuple[str, int, pd.DataFrame | None]] = []
    for index, iterator in enumerate(iterators):
        try:
            trade_date, day = next(iterator)
        except StopIteration:
            continue
        heapq.heappush(heap, (trade_date, index, day))
    while heap:
        trade_date = heap[0][0]
        representations: list[tuple[ArchiveMember, pd.DataFrame]] = []
        while heap and heap[0][0] == trade_date:
            _, index, day = heapq.heappop(heap)
            if day is not None:
                representations.append((members[index], day))
            try:
                next_date, next_day = next(iterators[index])
            except StopIteration:
                continue
            heapq.heappush(heap, (next_date, index, next_day))
        yield trade_date, representations


def _iter_archive_member_days(
    zip_handle: zipfile.ZipFile,
    member: ArchiveMember,
    *,
    csv_chunksize: int,
    skip_dates: set[str] | None = None,
    start_date: str = "",
    end_date: str = "",
) -> Iterator[tuple[str, pd.DataFrame | None]]:
    skipped = skip_dates or set()
    try:
        binary = zip_handle.open(member.member_name, mode="r")
    except (KeyError, zipfile.BadZipFile) as exc:
        raise IntradayRepairError(
            f"intraday_repair_archive_member_unreadable:{member.member_name}"
        ) from exc
    carry = pd.DataFrame()
    last_emitted = ""
    with binary:
        text = io.TextIOWrapper(binary, encoding="utf-8-sig", newline="")
        try:
            reader = pd.read_csv(text, chunksize=int(csv_chunksize))
            for raw_chunk in reader:
                chunk = _normalize_archive_chunk(
                    raw_chunk,
                    start_date=start_date,
                    end_date=end_date,
                )
                if chunk.empty:
                    continue
                if not carry.empty:
                    chunk = pd.concat([carry, chunk], ignore_index=True, sort=False)
                if last_emitted and str(chunk["trade_date"].min()) <= last_emitted:
                    raise IntradayRepairError(
                        f"intraday_repair_archive_member_not_date_sorted:{member.member_name}"
                    )
                if not chunk["trade_date"].is_monotonic_increasing:
                    raise IntradayRepairError(
                        f"intraday_repair_archive_member_not_date_sorted:{member.member_name}"
                    )
                final_date = str(chunk["trade_date"].iloc[-1])
                complete = chunk.loc[chunk["trade_date"].ne(final_date)].copy()
                carry = chunk.loc[chunk["trade_date"].eq(final_date)].copy()
                grouped_days = complete.groupby("trade_date", sort=False)
                for trade_date in grouped_days.indices:
                    value = str(trade_date)
                    if value <= last_emitted:
                        raise IntradayRepairError(
                            f"intraday_repair_archive_member_duplicate_date_segment:"
                            f"{member.member_name}:{value}"
                        )
                    last_emitted = value
                    if value in skipped:
                        yield value, None
                    else:
                        day = grouped_days.get_group(trade_date)
                        yield value, day.reset_index(drop=True)
            if not carry.empty:
                for trade_date in carry["trade_date"].drop_duplicates():
                    value = str(trade_date)
                    if value <= last_emitted:
                        raise IntradayRepairError(
                            f"intraday_repair_archive_member_duplicate_date_segment:"
                            f"{member.member_name}:{value}"
                        )
                    last_emitted = value
                    if value in skipped:
                        yield value, None
                    else:
                        day = carry.loc[carry["trade_date"].eq(trade_date)]
                        yield value, day.reset_index(drop=True)
        finally:
            text.detach()


def _normalize_archive_chunk(
    frame: pd.DataFrame,
    *,
    start_date: str = "",
    end_date: str = "",
) -> pd.DataFrame:
    timestamp_column = _find_column(frame, _COLUMN_ALIASES["timestamp"])
    if not timestamp_column:
        raise IntradayRepairError("intraday_repair_archive_timestamp_column_missing")
    timestamps = pd.to_datetime(frame[timestamp_column], errors="coerce")
    if timestamps.isna().any():
        raise IntradayRepairError(
            f"intraday_repair_archive_timestamp_invalid:{int(timestamps.isna().sum())}"
        )
    normalized = pd.DataFrame(
        {
            "trade_date": timestamps.dt.strftime("%Y-%m-%d").astype("string"),
            "bar_end": timestamps.dt.strftime("%H:%M").astype("string"),
        }
    )
    selected = pd.Series(True, index=normalized.index)
    if start_date:
        selected &= normalized["trade_date"].ge(start_date)
    if end_date:
        selected &= normalized["trade_date"].le(end_date)
    if not bool(selected.all()):
        frame = frame.loc[selected].reset_index(drop=True)
        normalized = normalized.loc[selected].reset_index(drop=True)
    for column in _NUMERIC_COLUMNS:
        source_column = _find_column(frame, _COLUMN_ALIASES[column])
        if not source_column:
            raise IntradayRepairError(f"intraday_repair_archive_column_missing:{column}")
        normalized[column] = pd.to_numeric(frame[source_column], errors="coerce").astype("float64")
    return normalized.reset_index(drop=True)


def _normalize_complete_archive_day(
    day: pd.DataFrame,
    *,
    canonical_symbol: str,
    trade_date: str,
    source_name: str = SOURCE_NAME,
) -> tuple[pd.DataFrame, str]:
    data = day.copy()
    bars = data["bar_end"].astype(str).to_numpy(copy=False)
    if len(set(bars)) != len(bars):
        return _empty_patch_frame(), "duplicate_bar_time"
    is_49_bar_day = len(data) == 49 and set(bars) == {
        "09:30",
        *EXPECTED_5M_BAR_ENDS,
    }
    if not is_49_bar_day and (
        len(data) != 48 or set(bars) != set(EXPECTED_5M_BAR_ENDS)
    ):
        return _empty_patch_frame(), "not_exact_48_bar_time_set"

    numeric = _coerce_numeric_values(data)
    if is_49_bar_day:
        auction_positions = np.flatnonzero(bars == "09:30")
        first_positions = np.flatnonzero(bars == "09:35")
        if len(auction_positions) != 1:
            return _empty_patch_frame(), "invalid_0930_count"
        if len(first_positions) != 1:
            return _empty_patch_frame(), "invalid_0935_count"
        auction_position = int(auction_positions[0])
        first_position = int(first_positions[0])
        pair_reason = _numeric_rejection_reason(
            numeric[[auction_position, first_position]],
            pair=True,
        )
        if pair_reason:
            return _empty_patch_frame(), pair_reason

        auction_values = numeric[auction_position].copy()
        first_values = numeric[first_position].copy()
        keep = bars != "09:30"
        data = data.loc[keep].reset_index(drop=True)
        bars = bars[keep]
        numeric = numeric[keep].copy()
        target_position = int(np.flatnonzero(bars == "09:35")[0])
        numeric[target_position, 0] = auction_values[0]
        numeric[target_position, 1] = max(auction_values[1], first_values[1])
        numeric[target_position, 2] = min(auction_values[2], first_values[2])
        numeric[target_position, 3] = first_values[3]
        numeric[target_position, 4] = auction_values[4] + first_values[4]
        numeric[target_position, 5] = auction_values[5] + first_values[5]

    reason = _numeric_rejection_reason(numeric, pair=False)
    if reason:
        return _empty_patch_frame(), reason
    if float(numeric[:, 4].sum()) == 0.0 and float(numeric[:, 5].sum()) == 0.0:
        return _empty_patch_frame(), "zero_turnover_stock_day"
    data.loc[:, list(_NUMERIC_COLUMNS)] = numeric
    data = data.assign(
        symbol=canonical_symbol,
        trade_date=trade_date,
        bar_time=data["bar_end"].map(_legacy_bar_time),
        source=str(source_name),
        adjusted_flag="none",
    )
    result = data.loc[:, PATCH_COLUMNS].sort_values("bar_time", kind="stable").reset_index(drop=True)
    if result.duplicated(list(PATCH_PRIMARY_KEY)).any():
        return _empty_patch_frame(), "duplicate_primary_key"
    return result, ""


def _coerce_numeric_values(frame: pd.DataFrame) -> np.ndarray:
    selected = frame.loc[:, list(_NUMERIC_COLUMNS)]
    try:
        return selected.to_numpy(dtype="float64", na_value=np.nan, copy=True)
    except (TypeError, ValueError):
        coerced = selected.apply(pd.to_numeric, errors="coerce")
        return coerced.to_numpy(dtype="float64", na_value=np.nan, copy=True)


def _numeric_rejection_reason(values: np.ndarray, *, pair: bool) -> str:
    if not bool(np.isfinite(values).all()):
        return "nonfinite_0930_0935_pair" if pair else "nonfinite_numeric"
    if bool((values[:, :4] <= 0).any()):
        return "nonpositive_0930_0935_price" if pair else "nonpositive_price"
    if bool((values[:, 4:6] < 0).any()):
        return (
            "negative_0930_0935_volume_or_amount"
            if pair
            else "negative_volume_or_amount"
        )
    invalid_ohlc = (values[:, 1] < values[:, [0, 3, 2]].max(axis=1)) | (
        values[:, 2] > values[:, [0, 3, 1]].min(axis=1)
    )
    if bool(invalid_ohlc.any()):
        return "invalid_0930_0935_ohlc_order" if pair else "invalid_ohlc_order"
    return ""


def _record_completed_bucket(
    connection: sqlite3.Connection,
    *,
    bucket: int,
    selected_symbol_count: int,
    base_complete_days: int,
    base_incomplete_days: int,
    receipts: Sequence[SymbolReceipt],
    isolated_failures: Sequence[tuple[str, str]],
    bundle_records: Sequence[Mapping[str, Any]],
    content_fingerprint: str,
) -> None:
    accepted = sum(item.accepted_days for item in receipts)
    existing = sum(item.existing_days for item in receipts)
    incomplete_existing = sum(item.incomplete_existing_days for item in receipts)
    rejected = sum(item.rejected_days for item in receipts)
    conflicts = sum(item.conflict_days for item in receipts)
    patch_rows = sum(item.patch_rows for item in receipts)
    if patch_rows != sum(int(item.get("row_count", 0) or 0) for item in bundle_records):
        raise IntradayRepairError(f"intraday_repair_bucket_receipt_row_mismatch:{bucket}")
    if len(receipts) + len(isolated_failures) != int(selected_symbol_count):
        raise IntradayRepairError(f"intraday_repair_bucket_symbol_receipt_mismatch:{bucket}")
    now = _utc_now()
    with connection:
        connection.execute("DELETE FROM symbol_receipts WHERE bucket = ?", [int(bucket)])
        connection.execute("DELETE FROM bundle_receipts WHERE bucket = ?", [int(bucket)])
        connection.executemany(
            """
            INSERT INTO symbol_receipts(
                symbol, bucket, status, accepted_days, existing_days,
                incomplete_existing_days, rejected_days, conflict_days,
                patch_rows, reject_sample_json,
                content_fingerprint, error, completed_at
            ) VALUES (?, ?, 'completed', ?, ?, ?, ?, ?, ?, ?, ?, '', ?)
            """,
            [
                (
                    item.symbol,
                    int(bucket),
                    item.accepted_days,
                    item.existing_days,
                    item.incomplete_existing_days,
                    item.rejected_days,
                    item.conflict_days,
                    item.patch_rows,
                    item.reject_sample_json,
                    item.content_fingerprint,
                    now,
                )
                for item in receipts
            ],
        )
        connection.executemany(
            """
            INSERT INTO symbol_receipts(
                symbol, bucket, status, reject_sample_json,
                content_fingerprint, error, completed_at
            ) VALUES (?, ?, 'failed', '[]', '', ?, ?)
            """,
            [
                (symbol, int(bucket), error, now)
                for symbol, error in isolated_failures
            ],
        )
        connection.executemany(
            """
            INSERT INTO bundle_receipts(
                year, bucket, path, row_count, stock_day_count,
                file_size, sha256
            ) VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            [
                (
                    str(item.get("year", "") or ""),
                    int(bucket),
                    str(item.get("path", "") or ""),
                    int(item.get("row_count", 0) or 0),
                    int(item.get("stock_day_count", 0) or 0),
                    int(item.get("file_size", 0) or 0),
                    str(item.get("sha256", "") or ""),
                )
                for item in bundle_records
            ],
        )
        connection.execute(
            """
            INSERT OR REPLACE INTO bucket_receipts(
                bucket, status, selected_symbols, completed_symbols,
                failed_symbols, base_complete_days, base_incomplete_days,
                accepted_days, existing_days, incomplete_existing_days,
                rejected_days, conflict_days, patch_rows,
                content_fingerprint, current_symbol, error, completed_at
            ) VALUES (?, 'completed', ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, '', '', ?)
            """,
            [
                int(bucket),
                int(selected_symbol_count),
                len(receipts),
                len(isolated_failures),
                int(base_complete_days),
                int(base_incomplete_days),
                accepted,
                existing,
                incomplete_existing,
                rejected,
                conflicts,
                patch_rows,
                content_fingerprint,
                now,
            ],
        )


def _record_bucket_failure(
    connection: sqlite3.Connection,
    *,
    bucket: int,
    selected_symbol_count: int,
    current_symbol: str,
    error: BaseException,
) -> None:
    message = f"{type(error).__name__}:{error}"
    now = _utc_now()
    with connection:
        connection.execute("DELETE FROM symbol_receipts WHERE bucket = ?", [int(bucket)])
        connection.execute("DELETE FROM bundle_receipts WHERE bucket = ?", [int(bucket)])
        if current_symbol:
            connection.execute(
                """
                INSERT OR REPLACE INTO symbol_receipts(
                    symbol, bucket, status, reject_sample_json,
                    content_fingerprint, error, completed_at
                ) VALUES (?, ?, 'failed', '[]', '', ?, ?)
                """,
                [current_symbol, int(bucket), message, now],
            )
        connection.execute(
            """
            INSERT OR REPLACE INTO bucket_receipts(
                bucket, status, selected_symbols, completed_symbols,
                failed_symbols, current_symbol, error, completed_at
            ) VALUES (?, 'failed', ?, 0, 1, ?, ?, ?)
            """,
            [int(bucket), int(selected_symbol_count), current_symbol, message, now],
        )


def _clear_bucket_receipts(connection: sqlite3.Connection, bucket: int) -> None:
    with connection:
        connection.execute("DELETE FROM symbol_receipts WHERE bucket = ?", [int(bucket)])
        connection.execute("DELETE FROM bundle_receipts WHERE bucket = ?", [int(bucket)])
        connection.execute("DELETE FROM bucket_receipts WHERE bucket = ?", [int(bucket)])


def _completed_bucket_reusable(
    connection: sqlite3.Connection,
    *,
    bucket: int,
    output_root: Path,
    dry_run: bool,
) -> tuple[bool, list[dict[str, Any]]]:
    if dry_run:
        return False, []
    row = connection.execute(
        "SELECT status, patch_rows FROM bucket_receipts WHERE bucket = ?",
        [int(bucket)],
    ).fetchone()
    if not row or str(row[0]) != "completed":
        return False, []
    records = _bundle_records(connection, bucket=bucket)
    if not _validate_bundle_records(records, output_root=output_root, expected_total=int(row[1])):
        return False, []
    return True, records


def _write_patch_manifest(
    connection: sqlite3.Connection,
    *,
    output_root: Path,
    job_id: str,
    selection_fingerprint: str,
    archive_path: Path,
    archive_fingerprint: str,
    snapshot: BaseSnapshot,
    base_index_manifest: Path,
    selected_symbols: Sequence[str],
    selection_source: str,
    source_name: str,
) -> tuple[Path, list[dict[str, Any]], bool]:
    if _scalar(connection, "SELECT count(*) FROM bucket_receipts WHERE status != 'completed'"):
        raise IntradayRepairError("intraday_repair_cannot_manifest_incomplete_buckets")
    records = _bundle_records(connection)
    patch_rows = _sum_receipt_column(connection, "patch_rows")
    if not _validate_bundle_records(records, output_root=output_root, expected_total=patch_rows):
        raise IntradayRepairError("intraday_repair_final_bundle_validation_failed")
    content_fingerprint = _patch_content_fingerprint(connection)
    manifest_path = output_root / "patch_manifest.json"
    existing = read_json(manifest_path)
    if (
        existing
        and existing.get("repair_plan_version") == REPAIR_PLAN_VERSION
        and existing.get("selection_fingerprint") == selection_fingerprint
        and existing.get("patch_content_fingerprint") == content_fingerprint
        and existing.get("selected_symbols") == list(selected_symbols)
        and existing.get("selected_symbols_sha256")
        == hashlib.sha256("\n".join(selected_symbols).encode("utf-8")).hexdigest()
        and existing.get("target_symbols_sha256")
        == _get_meta(connection, "target_symbols_sha256")
        and existing.get("base_manifest_sha256") == snapshot.manifest_sha256
        and existing.get("base_day_index_manifest_sha256")
        == _sha256_file(base_index_manifest)
        and _validate_bundle_records(
            list(existing.get("bundles", ()) or ()),
            output_root=manifest_path.parent,
            expected_total=int(existing.get("patch_row_count", 0) or 0),
        )
    ):
        return manifest_path, list(existing.get("bundles", ()) or ()), True
    payload = {
        "repair_plan_version": REPAIR_PLAN_VERSION,
        "job_id": job_id,
        "status": "completed",
        "domain": DOMAIN,
        "storage_layout": "natural_year_x_stable_16_symbol_bucket",
        "checkpoint_unit": "bucket",
        "sqlite_detail_policy": "receipts_and_counts_only_no_bar_rows",
        "primary_key": list(PATCH_PRIMARY_KEY),
        "columns": list(PATCH_COLUMNS),
        "expected_5m_bar_ends": list(EXPECTED_5M_BAR_ENDS),
        "auction_0930_policy": (
            "merge_single_0930_call_auction_into_0935_preserving_ohlc_volume_amount"
        ),
        "whole_stock_day_only": True,
        "merge_semantics": "append_only_completely_absent_stock_days",
        "incomplete_base_day_policy": "skip_and_report_never_append_over_existing_keys",
        "code_aliases": dict(CURRENT_RESEARCH_CODE_ALIASES),
        "selection_fingerprint": selection_fingerprint,
        "selection_source": selection_source,
        "source_name": source_name,
        "target_symbol_count": int(_get_meta(connection, "target_symbol_count") or 0),
        "archive_selected_symbol_count": int(
            _get_meta(connection, "archive_selected_symbol_count") or 0
        ),
        "eligible_missing_archive_count": int(
            _get_meta(connection, "eligible_missing_archive_count") or 0
        ),
        "eligible_missing_archive": json.loads(
            _get_meta(connection, "eligible_missing_archive_json") or "[]"
        ),
        "selected_symbol_count": len(selected_symbols),
        "selected_symbols": list(selected_symbols),
        "selected_symbols_sha256": hashlib.sha256(
            "\n".join(selected_symbols).encode("utf-8")
        ).hexdigest(),
        "target_symbols_sha256": _get_meta(connection, "target_symbols_sha256"),
        "patch_content_fingerprint": content_fingerprint,
        "archive_path": str(archive_path),
        "archive_fingerprint": archive_fingerprint,
        "base_active_path": str(snapshot.active_path),
        "base_active_sha256": snapshot.active_sha256,
        "base_dataset_id": snapshot.dataset_id,
        "base_manifest_path": str(snapshot.manifest_path),
        "base_manifest_sha256": snapshot.manifest_sha256,
        "base_shard_stat_fingerprint": snapshot.shard_stat_fingerprint,
        "base_day_index_manifest": str(base_index_manifest),
        "base_day_index_manifest_sha256": _sha256_file(base_index_manifest),
        "start_date": _get_meta(connection, "start_date"),
        "end_date": _get_meta(connection, "end_date"),
        "patch_stock_day_count": _sum_receipt_column(connection, "accepted_days"),
        "incomplete_existing_stock_days_skipped": _sum_receipt_column(
            connection,
            "incomplete_existing_days",
        ),
        "patch_row_count": patch_rows,
        "bundle_count": len(records),
        "bundles": records,
        "created_at": _utc_now(),
        "active_mutated": False,
    }
    atomic_write_json(manifest_path, payload)
    return manifest_path, records, False


def _validate_bundle_records(
    records: Sequence[Mapping[str, Any]],
    *,
    output_root: Path,
    expected_total: int,
) -> bool:
    actual_total = 0
    for item in records:
        relative = Path(str(item.get("path", "") or ""))
        if relative.is_absolute():
            return False
        year = str(item.get("year", "") or "")
        bucket = int(item.get("bucket", -1))
        if not re.fullmatch(r"\d{4}", year) or not 0 <= bucket < BUCKET_COUNT:
            return False
        expected_relative = Path("patch") / f"year={year}" / f"bucket={bucket:02d}" / "part.parquet"
        if relative.as_posix() != expected_relative.as_posix():
            return False
        path = (output_root / relative).resolve()
        if not _is_within(path, output_root):
            return False
        if not path.is_file():
            return False
        if int(item.get("file_size", 0) or 0) != int(path.stat().st_size):
            return False
        if str(item.get("sha256", "") or "") != _sha256_file(path):
            return False
        row_count = int(item.get("row_count", 0) or 0)
        parquet = pq.ParquetFile(path)
        try:
            actual_rows = parquet.metadata.num_rows
            schema_matches = parquet.schema_arrow.equals(
                _PATCH_ARROW_SCHEMA,
                check_metadata=False,
            )
        finally:
            parquet.close()
        if (
            row_count % 48
            or int(item.get("stock_day_count", 0) or 0) * 48 != row_count
            or actual_rows != row_count
            or not schema_matches
        ):
            return False
        actual_total += row_count
    return actual_total == int(expected_total)


def _patch_content_fingerprint(connection: sqlite3.Connection) -> str:
    digest = hashlib.sha256()
    for row in connection.execute(
        """
        SELECT bucket, accepted_days, patch_rows, content_fingerprint
        FROM bucket_receipts WHERE status = 'completed' ORDER BY bucket
        """
    ):
        digest.update(
            f"{int(row[0])}\0{int(row[1])}\0{int(row[2])}\0{row[3]}\n".encode("utf-8")
        )
    for record in _bundle_records(connection):
        digest.update(
            f"{record['year']}\0{record['bucket']}\0{record['row_count']}\0"
            f"{record['sha256']}\n".encode("utf-8")
        )
    return digest.hexdigest()


def _bundle_records(
    connection: sqlite3.Connection,
    *,
    bucket: int | None = None,
) -> list[dict[str, Any]]:
    query = (
        "SELECT year, bucket, path, row_count, stock_day_count, file_size, sha256 "
        "FROM bundle_receipts"
    )
    parameters: list[Any] = []
    if bucket is not None:
        query += " WHERE bucket = ?"
        parameters.append(int(bucket))
    query += " ORDER BY year, bucket"
    return [
        {
            "year": str(row[0]),
            "bucket": int(row[1]),
            "path": str(row[2]),
            "row_count": int(row[3]),
            "stock_day_count": int(row[4]),
            "file_size": int(row[5]),
            "sha256": str(row[6]),
        }
        for row in connection.execute(query, parameters)
    ]


def _failed_symbols(connection: sqlite3.Connection) -> list[dict[str, str]]:
    failures = [
        {"symbol": str(row[0]), "error": str(row[1])}
        for row in connection.execute(
            """
            SELECT symbol, error FROM symbol_receipts
            WHERE status = 'failed' ORDER BY symbol
            """
        )
    ]
    if failures:
        return failures
    return [
        {"symbol": str(row[0] or f"bucket={int(row[1]):02d}"), "error": str(row[2])}
        for row in connection.execute(
            """
            SELECT current_symbol, bucket, error FROM bucket_receipts
            WHERE status = 'failed' ORDER BY bucket
            """
        )
    ]


def _write_runtime_state(
    state_path: Path,
    connection: sqlite3.Connection,
    *,
    status: str,
    job_id: str,
    archive_path: Path,
    archive_fingerprint: str,
    snapshot: BaseSnapshot,
    selected_symbol_count: int,
    output_root: Path,
    dry_run: bool,
    current_bucket: int | None = None,
    current_symbol: str = "",
    manifest_path: Path | None = None,
    error: str = "",
) -> None:
    completed = _scalar(
        connection, "SELECT count(*) FROM symbol_receipts WHERE status = 'completed'"
    )
    failed = _scalar(
        connection, "SELECT count(*) FROM symbol_receipts WHERE status = 'failed'"
    )
    payload = {
        "repair_plan_version": REPAIR_PLAN_VERSION,
        "job_id": job_id,
        "status": status,
        "archive_path": str(archive_path),
        "archive_fingerprint": archive_fingerprint,
        "base_active_path": str(snapshot.active_path),
        "base_active_sha256": snapshot.active_sha256,
        "base_dataset_id": snapshot.dataset_id,
        "base_manifest_path": str(snapshot.manifest_path),
        "base_manifest_sha256": snapshot.manifest_sha256,
        "base_shard_count": len(snapshot.shard_paths),
        "base_shards_completed": int(_get_meta(connection, "base_shards_completed") or 0),
        "base_complete_stock_day_count": _sum_receipt_column(
            connection, "base_complete_days"
        ),
        "base_incomplete_stock_day_count": _sum_receipt_column(
            connection, "base_incomplete_days"
        ),
        "target_symbol_count": int(_get_meta(connection, "target_symbol_count") or 0),
        "archive_selected_symbol_count": int(
            _get_meta(connection, "archive_selected_symbol_count") or 0
        ),
        "eligible_missing_archive_count": int(
            _get_meta(connection, "eligible_missing_archive_count") or 0
        ),
        "selected_symbol_count": int(selected_symbol_count),
        "completed_symbols": int(completed),
        "pending_symbols": max(0, int(selected_symbol_count) - int(completed) - int(failed)),
        "failed_symbols": int(failed),
        "completed_buckets": _scalar(
            connection, "SELECT count(*) FROM bucket_receipts WHERE status = 'completed'"
        ),
        "failed_buckets": _scalar(
            connection, "SELECT count(*) FROM bucket_receipts WHERE status = 'failed'"
        ),
        "accepted_stock_days": _sum_receipt_column(connection, "accepted_days"),
        "existing_stock_days_skipped": _sum_receipt_column(connection, "existing_days"),
        "incomplete_existing_stock_days_skipped": _sum_receipt_column(
            connection,
            "incomplete_existing_days",
        ),
        "rejected_stock_days": _sum_receipt_column(connection, "rejected_days"),
        "patch_row_count": _sum_receipt_column(connection, "patch_rows"),
        "current_bucket": "" if current_bucket is None else int(current_bucket),
        "current_symbol": current_symbol,
        "output_root": str(output_root),
        "patch_manifest_path": str(manifest_path) if manifest_path else "",
        "dry_run": bool(dry_run),
        "active_mutated": False,
        "error": error,
        "heartbeat_at": _utc_now(),
    }
    atomic_write_json(state_path, payload)


def _result_payload(
    connection: sqlite3.Connection,
    *,
    status: str,
    job_id: str,
    archive_path: Path,
    archive_fingerprint: str,
    snapshot: BaseSnapshot,
    selected_symbol_count: int,
    selection_source: str,
    output_root: Path,
    runtime_root: Path,
    state_path: Path | None,
    database_path: Path | None,
    manifest_path: Path | None,
    bundles: Sequence[Mapping[str, Any]],
    dry_run: bool,
    error: str = "",
) -> dict[str, Any]:
    completed = _scalar(
        connection, "SELECT count(*) FROM symbol_receipts WHERE status = 'completed'"
    )
    failed = _failed_symbols(connection)
    return {
        "status": status,
        "job_id": job_id,
        "archive_path": str(archive_path),
        "archive_fingerprint": archive_fingerprint,
        "base_active_path": str(snapshot.active_path),
        "base_active_sha256": snapshot.active_sha256,
        "base_dataset_id": snapshot.dataset_id,
        "base_manifest_path": str(snapshot.manifest_path),
        "base_manifest_sha256": snapshot.manifest_sha256,
        "base_shard_count": len(snapshot.shard_paths),
        "base_complete_stock_day_count": _sum_receipt_column(
            connection, "base_complete_days"
        ),
        "base_incomplete_stock_day_count": _sum_receipt_column(
            connection, "base_incomplete_days"
        ),
        "selection_source": selection_source,
        "target_symbol_count": int(_get_meta(connection, "target_symbol_count") or 0),
        "archive_selected_symbol_count": int(
            _get_meta(connection, "archive_selected_symbol_count") or 0
        ),
        "eligible_missing_archive_count": int(
            _get_meta(connection, "eligible_missing_archive_count") or 0
        ),
        "eligible_missing_archive": json.loads(
            _get_meta(connection, "eligible_missing_archive_json") or "[]"
        ),
        "selected_symbol_count": int(selected_symbol_count),
        "completed_symbols": int(completed),
        "pending_symbols": max(0, int(selected_symbol_count) - int(completed) - len(failed)),
        "failed_symbols": failed,
        "accepted_stock_days": _sum_receipt_column(connection, "accepted_days"),
        "existing_stock_days_skipped": _sum_receipt_column(connection, "existing_days"),
        "incomplete_existing_stock_days_skipped": _sum_receipt_column(
            connection,
            "incomplete_existing_days",
        ),
        "rejected_stock_days": _sum_receipt_column(connection, "rejected_days"),
        "conflict_stock_days": _sum_receipt_column(connection, "conflict_days"),
        "patch_stock_day_count": _sum_receipt_column(connection, "accepted_days"),
        "patch_row_count": _sum_receipt_column(connection, "patch_rows"),
        "bundle_count": len(bundles),
        "bundles": [dict(item) for item in bundles],
        "output_root": str(output_root),
        "runtime_root": str(runtime_root),
        "state_path": str(state_path) if state_path else "",
        "database_path": str(database_path) if database_path else "",
        "patch_manifest_path": str(manifest_path) if manifest_path else "",
        "dry_run": bool(dry_run),
        "active_mutated": False,
        "error": error,
    }


def _find_column(frame: pd.DataFrame, aliases: Sequence[str]) -> str:
    by_normalized = {str(column).strip().lower(): str(column) for column in frame.columns}
    for alias in aliases:
        resolved = by_normalized.get(str(alias).strip().lower())
        if resolved:
            return resolved
    return ""


def _symbol_from_member_name(member_name: str) -> str:
    name = Path(str(member_name).replace("\\", "/")).name
    match = _MEMBER_PATTERN.fullmatch(name)
    if not match:
        return ""
    exchange = match.group("exchange").upper()
    return f"{match.group('code')}.{exchange}"


def _canonical_symbol(value: Any) -> str:
    text = str(value or "").strip().upper()
    if not text:
        raise ValueError("intraday_repair_symbol_empty")
    if _NORMAL_SYMBOL_PATTERN.fullmatch(text):
        normalized = text
    else:
        compact = re.sub(r"[^A-Z0-9]", "", text)
        prefix = re.fullmatch(r"(SH|SZ|BJ)(\d{6})", compact)
        suffix = re.fullmatch(r"(\d{6})(SH|SZ|BJ)", compact)
        if prefix:
            normalized = f"{prefix.group(2)}.{prefix.group(1)}"
        elif suffix:
            normalized = f"{suffix.group(1)}.{suffix.group(2)}"
        else:
            raise ValueError(f"intraday_repair_symbol_invalid:{value}")
    return str(CURRENT_RESEARCH_CODE_ALIASES.get(normalized, normalized))


def _is_mainboard_a_symbol(value: str) -> bool:
    symbol = _canonical_symbol(value)
    # A historical mainboard identity may use a current research code whose
    # numeric prefix no longer describes that historical board membership.
    if symbol in set(CURRENT_RESEARCH_CODE_ALIASES.values()):
        return True
    code = symbol.split(".", 1)[0]
    if symbol.endswith(".SH"):
        return code.startswith(("600", "601", "603", "605"))
    if symbol.endswith(".SZ"):
        return code.startswith(("000", "001", "002", "003"))
    return False


def _normalize_bar_end(value: Any) -> str:
    raw = str(value or "").strip()
    match = re.search(r"(?<!\d)(\d{2}):(\d{2})(?::\d{2})?", raw)
    if match:
        return f"{match.group(1)}:{match.group(2)}"
    digits = "".join(character for character in raw.split(".", 1)[0] if character.isdigit())
    if len(digits) < 4:
        return ""
    if len(digits) >= 12 and digits[:4].isdigit() and 1990 <= int(digits[:4]) <= 2100:
        hour, minute = digits[8:10], digits[10:12]
    else:
        hour, minute = digits[:2], digits[2:4]
    if not hour.isdigit() or not minute.isdigit():
        return ""
    if int(hour) > 23 or int(minute) > 59:
        return ""
    return f"{hour}:{minute}"


def _legacy_bar_time(value: Any) -> str:
    normalized = _normalize_bar_end(value)
    if not normalized:
        return ""
    return normalized.replace(":", "") + "00000"


def _date_text(value: Any) -> str:
    parsed = pd.to_datetime(str(value or ""), errors="coerce")
    if pd.isna(parsed):
        raise ValueError(f"intraday_repair_date_invalid:{value}")
    return parsed.strftime("%Y-%m-%d")


def _compact_iso_date_series(values: pd.Series) -> pd.Series:
    text = values.fillna("").astype(str).str.strip().str.replace("-", "", regex=False)
    valid = text.str.fullmatch(r"\d{8}")
    result = pd.Series("", index=values.index, dtype="string")
    result.loc[valid] = (
        text.loc[valid].str.slice(0, 4)
        + "-"
        + text.loc[valid].str.slice(4, 6)
        + "-"
        + text.loc[valid].str.slice(6, 8)
    )
    return result.astype(str)


def _empty_patch_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "symbol": pd.Series(dtype="string"),
            "trade_date": pd.Series(dtype="string"),
            "bar_time": pd.Series(dtype="string"),
            "open": pd.Series(dtype="float64"),
            "high": pd.Series(dtype="float64"),
            "low": pd.Series(dtype="float64"),
            "close": pd.Series(dtype="float64"),
            "volume": pd.Series(dtype="float64"),
            "amount": pd.Series(dtype="float64"),
            "source": pd.Series(dtype="string"),
            "adjusted_flag": pd.Series(dtype="string"),
        }
    )


def _frame_sha256(frame: pd.DataFrame) -> str:
    ordered = frame.loc[:, PATCH_COLUMNS].sort_values(list(PATCH_PRIMARY_KEY), kind="stable")
    row_hashes = pd.util.hash_pandas_object(ordered, index=False, categorize=True)
    digest = hashlib.sha256("\0".join(PATCH_COLUMNS).encode("utf-8"))
    digest.update(row_hashes.to_numpy(dtype="uint64", copy=False).tobytes())
    return digest.hexdigest()


def _sha256_file(path: str | Path, *, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while True:
            chunk = handle.read(int(chunk_size))
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def _stable_hash(payload: Mapping[str, Any]) -> str:
    encoded = json.dumps(
        dict(payload),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _get_meta(connection: sqlite3.Connection, key: str) -> str:
    row = connection.execute("SELECT value FROM metadata WHERE key = ?", [str(key)]).fetchone()
    return str(row[0]) if row else ""


def _set_meta(connection: sqlite3.Connection, key: str, value: str) -> None:
    connection.execute(
        "INSERT OR REPLACE INTO metadata(key, value) VALUES (?, ?)",
        [str(key), str(value)],
    )


def _scalar(connection: sqlite3.Connection, query: str) -> int:
    row = connection.execute(query).fetchone()
    return int(row[0] or 0) if row else 0


def _sum_receipt_column(connection: sqlite3.Connection, column: str) -> int:
    allowed = {
        "base_complete_days",
        "base_incomplete_days",
        "accepted_days",
        "existing_days",
        "incomplete_existing_days",
        "rejected_days",
        "conflict_days",
        "patch_rows",
    }
    if column not in allowed:
        raise ValueError(f"intraday_repair_invalid_receipt_column:{column}")
    return _scalar(
        connection,
        f"SELECT coalesce(sum({column}), 0) FROM bucket_receipts WHERE status = 'completed'",
    )


def _replace_file_with_retry(source: Path, target: Path, *, attempts: int = 8) -> None:
    delay = 0.05
    for attempt in range(max(1, int(attempts))):
        try:
            os.replace(source, target)
            return
        except PermissionError:
            if attempt + 1 >= int(attempts):
                raise
            time.sleep(delay)
            delay = min(0.5, delay * 2)


def _unlink_file_with_retry(path: Path, *, attempts: int = 8) -> None:
    delay = 0.05
    for attempt in range(max(1, int(attempts))):
        try:
            path.unlink(missing_ok=True)
            return
        except PermissionError:
            if attempt + 1 >= int(attempts):
                raise
            time.sleep(delay)
            delay = min(0.5, delay * 2)


def _is_within(path: Path, root: Path) -> bool:
    try:
        path.resolve().relative_to(root.resolve())
        return True
    except ValueError:
        return False


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m quant_data_platform.qdp_v2.intraday_repair",
        description="Plan complete-stock-day 5m gaps from the local archive without touching v2 active.",
    )
    parser.add_argument("--workspace-root", default="")
    parser.add_argument("--archive", default=str(DEFAULT_ARCHIVE_PATH))
    parser.add_argument("--output-root", default="")
    parser.add_argument("--start-date", default=DEFAULT_START_DATE)
    parser.add_argument("--end-date", default=DEFAULT_END_DATE)
    parser.add_argument("--symbols", default="")
    parser.add_argument("--write-patch", action="store_true")
    parser.add_argument("--no-resume", action="store_true")
    parser.add_argument("--csv-chunksize", type=int, default=100_000)
    parser.add_argument("--parquet-row-group-rows", type=int, default=100_000)
    parser.add_argument("--base-scan-batch-rows", type=int, default=262_144)
    parser.add_argument("--memory-floor-gib", type=float, default=0.5)
    parser.add_argument("--memory-floor-seconds", type=float, default=5.0)
    parser.add_argument("--source-name", default=SOURCE_NAME)
    parser.add_argument("--json", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    symbols = [item.strip() for item in str(args.symbols or "").split(",") if item.strip()]
    result = plan_intraday_5m_archive_repairs(
        workspace_root=str(args.workspace_root or "") or None,
        archive_path=args.archive,
        output_root=str(args.output_root or "") or None,
        start_date=args.start_date,
        end_date=args.end_date,
        symbols=symbols,
        dry_run=not bool(args.write_patch),
        resume=not bool(args.no_resume),
        csv_chunksize=int(args.csv_chunksize),
        parquet_row_group_rows=int(args.parquet_row_group_rows),
        base_scan_batch_rows=int(args.base_scan_batch_rows),
        memory_floor_gib=float(args.memory_floor_gib),
        memory_floor_seconds=float(args.memory_floor_seconds),
        source_name=str(args.source_name),
    )
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0 if result.get("status") not in {"completed_with_failures", "paused_memory_guard"} else 1


__all__ = [
    "CURRENT_RESEARCH_CODE_ALIASES",
    "DEFAULT_ARCHIVE_PATH",
    "EXPECTED_5M_BAR_ENDS",
    "IntradayRepairError",
    "IntradayRepairResourceGuardError",
    "PATCH_COLUMNS",
    "PATCH_PRIMARY_KEY",
    "bulk_append_intraday_5m_patch",
    "plan_intraday_5m_archive_repairs",
    "stable_symbol_bucket",
]


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
