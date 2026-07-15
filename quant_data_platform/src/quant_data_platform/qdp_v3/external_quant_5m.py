from __future__ import annotations

import hashlib
import io
import json
import math
import multiprocessing
import os
import re
import shutil
import threading
import time
import zipfile
from concurrent.futures import (
    FIRST_COMPLETED,
    ProcessPoolExecutor,
    ThreadPoolExecutor,
    as_completed,
    wait,
)
from contextlib import ExitStack
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd

from quant_data_platform.core.json_io import read_json
from quant_data_platform.qdp_v3.constants import EXPECTED_5M_BAR_ENDS
from quant_data_platform.qdp_v3.identity import normalize_symbol
from quant_data_platform.qdp_v3.manifest import atomic_write_json, sha256_file, utc_now
from quant_data_platform.qdp_v3.paths import ensure_qdp_v3_layout
from quant_data_platform.qdp_v3.storage import (
    RawPartitionRef,
    get_raw_partition,
    read_raw_partition,
    read_raw_receipt,
    write_raw_partition,
)


DEFAULT_RAW_DOMAIN = "external_quant_intraday_5m_raw"
SOURCE_NAME = "external_quant_archive"
MAX_EXTERNAL_QUANT_WORKERS = 8
PROCESS_WATCHDOG_INTERVAL_SECONDS = 1.0
PROCESS_HEARTBEAT_SECONDS = 30.0
RAW_COLUMNS = [
    "provider_symbol",
    "trade_date",
    "bar_time",
    "open",
    "high",
    "low",
    "close",
    "volume",
    "amount",
    "source",
]
_NUMERIC_COLUMNS = ["open", "high", "low", "close", "volume", "amount"]
_PRICE_COLUMNS = ["open", "high", "low", "close"]
_MEMBER_PATTERN = re.compile(r"^(?P<exchange>sh|sz|bj)(?P<code>\d{6})\.csv$", re.IGNORECASE)
_COLUMN_ALIASES: Mapping[str, tuple[str, ...]] = {
    "timestamp": ("日期", "时间", "datetime", "trade_time", "trade_datetime", "date", "time"),
    "open": ("开盘", "open"),
    "high": ("最高", "high"),
    "low": ("最低", "low"),
    "close": ("收盘", "close"),
    "volume": ("成交量(股)", "成交量", "volume", "vol"),
    "amount": ("成交额(元)", "成交额", "amount"),
}


class ExternalQuant5mError(RuntimeError):
    """Raised when the external archive cannot be admitted without guessing."""


class ExternalQuant5mResourceGuardError(ExternalQuant5mError):
    """Raised after a memory or disk safety boundary remains breached."""


@dataclass(frozen=True)
class ExternalQuant5mSegment:
    provider_symbol: str
    source_path: Path
    source_kind: str
    member_name: str
    relative_name: str
    compressed_size: int
    uncompressed_size: int
    crc32: str
    source_size: int
    source_mtime_ns: int

    def inventory_record(self) -> dict[str, Any]:
        return {
            "provider_symbol": self.provider_symbol,
            "source_path": str(self.source_path),
            "source_kind": self.source_kind,
            "member_name": self.member_name,
            "relative_name": self.relative_name,
            "compressed_size": int(self.compressed_size),
            "uncompressed_size": int(self.uncompressed_size),
            "crc32": self.crc32,
            "source_size": int(self.source_size),
            "source_mtime_ns": int(self.source_mtime_ns),
        }


@dataclass(frozen=True)
class ExternalQuant5mSourceIndex:
    source_paths: tuple[Path, ...]
    segments: tuple[ExternalQuant5mSegment, ...]

    @property
    def by_symbol(self) -> dict[str, tuple[ExternalQuant5mSegment, ...]]:
        grouped: dict[str, list[ExternalQuant5mSegment]] = {}
        for segment in self.segments:
            grouped.setdefault(segment.provider_symbol, []).append(segment)
        return {
            symbol: tuple(sorted(items, key=_segment_sort_key))
            for symbol, items in sorted(grouped.items())
        }

    @property
    def symbols(self) -> tuple[str, ...]:
        return tuple(self.by_symbol)


@dataclass(frozen=True)
class ExternalQuant5mSymbolResult:
    provider_symbol: str
    status: str
    row_count: int = 0
    day_count: int = 0
    created: bool = False
    ref: RawPartitionRef | None = None
    error: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "provider_symbol": self.provider_symbol,
            "status": self.status,
            "row_count": int(self.row_count),
            "day_count": int(self.day_count),
            "created": bool(self.created),
            "content_sha256": self.ref.content_sha256 if self.ref else "",
            "error": self.error,
        }


@dataclass(frozen=True)
class ExternalQuant5mImportResult:
    status: str
    raw_domain: str
    discovered_symbols: int
    selected_symbols: int
    completed_symbols: int
    created_symbols: int
    reused_symbols: int
    empty_symbols: int
    failed_symbols: int
    row_count: int
    day_count: int
    results: tuple[ExternalQuant5mSymbolResult, ...] = field(default_factory=tuple)

    @property
    def refs(self) -> tuple[RawPartitionRef, ...]:
        return tuple(item.ref for item in self.results if item.ref is not None)

    @property
    def errors(self) -> tuple[dict[str, str], ...]:
        return tuple(
            {"provider_symbol": item.provider_symbol, "error": item.error}
            for item in self.results
            if item.error
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "raw_domain": self.raw_domain,
            "discovered_symbols": int(self.discovered_symbols),
            "selected_symbols": int(self.selected_symbols),
            "completed_symbols": int(self.completed_symbols),
            "created_symbols": int(self.created_symbols),
            "reused_symbols": int(self.reused_symbols),
            "empty_symbols": int(self.empty_symbols),
            "failed_symbols": int(self.failed_symbols),
            "row_count": int(self.row_count),
            "day_count": int(self.day_count),
            "errors": list(self.errors),
            "results": [item.to_dict() for item in self.results],
        }


def scan_external_quant_5m_sources(
    source_paths: Iterable[str | Path],
) -> ExternalQuant5mSourceIndex:
    """Index explicitly supplied ZIP archives and unpacked CSV directories.

    The historical archive stores filenames in GBK while CSV payloads are
    UTF-8-SIG.  ``metadata_encoding`` is therefore deliberately set even
    though newer ZIPs may carry the UTF-8 filename flag.
    """

    resolved_paths = _resolve_source_paths(source_paths)
    segments: list[ExternalQuant5mSegment] = []
    for source_path in resolved_paths:
        stat = source_path.stat()
        if source_path.is_file():
            if source_path.suffix.lower() != ".zip":
                raise ValueError(f"external_quant_5m_source_must_be_zip_or_directory:{source_path}")
            with zipfile.ZipFile(source_path, mode="r", metadata_encoding="gbk") as archive:
                for info in archive.infolist():
                    if info.is_dir():
                        continue
                    symbol = _provider_symbol_from_name(info.filename)
                    if not symbol:
                        continue
                    segments.append(
                        ExternalQuant5mSegment(
                            provider_symbol=symbol,
                            source_path=source_path,
                            source_kind="zip",
                            member_name=info.filename,
                            relative_name=info.filename,
                            compressed_size=int(info.compress_size),
                            uncompressed_size=int(info.file_size),
                            crc32=f"{int(info.CRC):08x}",
                            source_size=int(stat.st_size),
                            source_mtime_ns=int(stat.st_mtime_ns),
                        )
                    )
        else:
            for csv_path in sorted(source_path.rglob("*.csv")):
                if not csv_path.is_file():
                    continue
                symbol = _provider_symbol_from_name(csv_path.name)
                if not symbol:
                    continue
                csv_stat = csv_path.stat()
                relative_name = csv_path.relative_to(source_path).as_posix()
                segments.append(
                    ExternalQuant5mSegment(
                        provider_symbol=symbol,
                        source_path=source_path,
                        source_kind="directory",
                        member_name=str(csv_path),
                        relative_name=relative_name,
                        compressed_size=int(csv_stat.st_size),
                        uncompressed_size=int(csv_stat.st_size),
                        crc32="",
                        source_size=int(csv_stat.st_size),
                        source_mtime_ns=int(csv_stat.st_mtime_ns),
                    )
                )
    if not segments:
        raise ExternalQuant5mError("external_quant_5m_no_matching_csv_members")
    duplicate_locations = [
        key
        for key, count in pd.Series(
            [
                f"{item.source_path}|{item.source_kind}|{item.member_name}"
                for item in segments
            ],
            dtype="string",
        ).value_counts().items()
        if int(count) > 1
    ]
    if duplicate_locations:
        raise ExternalQuant5mError(
            f"external_quant_5m_duplicate_source_member:{duplicate_locations[0]}"
        )
    return ExternalQuant5mSourceIndex(
        source_paths=resolved_paths,
        segments=tuple(sorted(segments, key=lambda item: (item.provider_symbol, *_segment_sort_key(item)))),
    )


def normalize_external_quant_5m(
    frame: pd.DataFrame,
    *,
    provider_symbol: str,
    start_date: str = "",
    end_date: str = "",
) -> pd.DataFrame:
    """Normalize one source CSV and enforce the archive's 49-to-48 contract.

    A raw 09:30 call-auction row is merged into the right-closed 09:35 bar.
    Already canonical 48-bar days remain unchanged.  Any other time set,
    including duplicates or a missing 09:35 pair, is preserved without
    guessing, padding, or splicing so the canonical gate can quarantine only
    that stock-day.
    """

    if frame is None or frame.empty:
        return _empty_raw_frame()
    symbol = normalize_symbol(provider_symbol)
    if not re.fullmatch(r"\d{6}\.(SH|SZ|BJ)", symbol):
        raise ValueError(f"external_quant_5m_invalid_provider_symbol:{provider_symbol}")
    data = frame.copy()
    timestamp_column = _find_column(data, _COLUMN_ALIASES["timestamp"])
    if timestamp_column:
        timestamps = pd.to_datetime(data[timestamp_column], errors="coerce")
    elif "trade_date" in data.columns and ("bar_time" in data.columns or "bar_end" in data.columns):
        time_column = "bar_time" if "bar_time" in data.columns else "bar_end"
        timestamps = pd.to_datetime(
            data["trade_date"].astype(str).str.slice(0, 10)
            + " "
            + data[time_column].astype(str),
            errors="coerce",
        )
    else:
        raise ValueError("external_quant_5m_timestamp_column_missing")
    if timestamps.isna().any():
        bad_count = int(timestamps.isna().sum())
        raise ValueError(f"external_quant_5m_timestamp_invalid:count={bad_count}")
    normalized = pd.DataFrame(
        {
            "provider_symbol": pd.Series(symbol, index=data.index, dtype="string"),
            "trade_date": timestamps.dt.strftime("%Y-%m-%d").astype("string"),
            "bar_time": timestamps.dt.strftime("%H:%M").astype("string"),
        },
        index=data.index,
    )
    for canonical in _NUMERIC_COLUMNS:
        source_column = _find_column(data, _COLUMN_ALIASES[canonical])
        if not source_column:
            raise ValueError(f"external_quant_5m_column_missing:{canonical}")
        normalized[canonical] = pd.to_numeric(data[source_column], errors="coerce").astype("float64")
    normalized["source"] = pd.Series(SOURCE_NAME, index=data.index, dtype="string")
    if start_date:
        normalized = normalized.loc[normalized["trade_date"] >= _date_text(start_date)]
    if end_date:
        normalized = normalized.loc[normalized["trade_date"] <= _date_text(end_date)]
    if normalized.empty:
        return _empty_raw_frame()
    days: list[pd.DataFrame] = []
    for trade_date, day in normalized.groupby("trade_date", sort=True):
        days.append(_normalize_day(day, provider_symbol=symbol, trade_date=str(trade_date)))
    result = pd.concat(days, ignore_index=True)
    return _coerce_raw_schema(result)


def _multiprocess_worker_available() -> bool:
    """Whether Windows spawn can import the current Python entry point."""

    try:
        import __main__

        entry = str(getattr(__main__, "__file__", "") or "")
    except Exception:
        return False
    return bool(entry and entry not in {"<stdin>", "-c"})


def _process_symbol_task(
    *,
    provider_symbol: str,
    segments: Sequence[ExternalQuant5mSegment],
    container_hashes: Mapping[Path, str],
    raw_domain: str,
    workspace_root: str | Path | None,
    start_date: str,
    end_date: str,
    min_available_gib: float,
    low_memory_seconds: float,
    min_free_disk_gib: float,
) -> ExternalQuant5mSymbolResult:
    """One spawn-safe CPU-bound archive task with independent ZIP handles."""

    try:
        paths = ensure_qdp_v3_layout(workspace_root)
        _resource_guard(
            paths.root,
            min_available_gib=min_available_gib,
            low_memory_seconds=low_memory_seconds,
            min_free_disk_gib=min_free_disk_gib,
        )
        with ExitStack() as stack:
            zip_handles = {
                source_path: stack.enter_context(
                    zipfile.ZipFile(source_path, mode="r", metadata_encoding="gbk")
                )
                for source_path in sorted(
                    {item.source_path for item in segments if item.source_kind == "zip"}
                )
            }
            return _import_symbol(
                provider_symbol=provider_symbol,
                segments=segments,
                zip_handles=zip_handles,
                container_hashes=container_hashes,
                raw_domain=raw_domain,
                workspace_root=workspace_root,
                start_date=start_date,
                end_date=end_date,
            )
    except ExternalQuant5mResourceGuardError as exc:
        return ExternalQuant5mSymbolResult(
            provider_symbol=provider_symbol,
            status="resource_guard",
            error=f"{type(exc).__name__}:{exc}",
        )
    except Exception as exc:
        return ExternalQuant5mSymbolResult(
            provider_symbol=provider_symbol,
            status="failed",
            error=f"{type(exc).__name__}:{exc}",
        )


def import_external_quant_5m(
    source_paths: Iterable[str | Path],
    *,
    workspace_root: str | Path | None = None,
    raw_domain: str = DEFAULT_RAW_DOMAIN,
    symbols: Iterable[str] = (),
    start_date: str = "2010-01-01",
    end_date: str = "",
    max_symbols: int = 0,
    workers: int = 1,
    strict: bool = True,
    hash_containers: bool = True,
    job_path: str | Path | None = None,
    min_available_gib: float = 0.5,
    low_memory_seconds: float = 5.0,
    min_free_disk_gib: float = 200.0,
) -> ExternalQuant5mImportResult:
    """Import a trusted local 5m archive into immutable v3 raw partitions.

    Multiple source segments are combined per provider symbol before the sole
    partition is written.  Exact overlapping bars are deduplicated; any value
    conflict is a hard error.  Workers own independent ZIP handles so archive
    reads are not serialized by a shared ``ZipFile`` lock.
    """

    start = _date_text(start_date) if start_date else ""
    end = _date_text(end_date) if end_date else ""
    if start and end and start > end:
        raise ValueError(f"external_quant_5m_invalid_date_range:{start}:{end}")
    paths = ensure_qdp_v3_layout(workspace_root)
    _resource_guard(
        paths.root,
        min_available_gib=min_available_gib,
        low_memory_seconds=low_memory_seconds,
        min_free_disk_gib=min_free_disk_gib,
    )
    source_index = scan_external_quant_5m_sources(source_paths)
    by_symbol = source_index.by_symbol
    requested = {
        normalize_symbol(item)
        for item in symbols
        if str(item or "").strip()
    }
    invalid_requested = sorted(
        item for item in requested if not re.fullmatch(r"\d{6}\.(SH|SZ|BJ)", item)
    )
    if invalid_requested:
        raise ValueError(f"external_quant_5m_invalid_requested_symbol:{invalid_requested[0]}")
    selected = sorted(set(by_symbol).intersection(requested) if requested else by_symbol)
    if max_symbols:
        selected = selected[: max(0, int(max_symbols))]
    resolved_job_path = (
        Path(job_path).expanduser().resolve()
        if job_path
        else paths.jobs
        / (
            "external_quant_5m__"
            f"{start or 'begin'}_{end or 'latest'}__"
            f"{_selection_fingerprint(source_index.source_paths, selected)[:12]}.json"
        )
    )
    started_monotonic = time.monotonic()
    progress_lock = threading.RLock()
    progress: dict[str, Any] = {
        "job_version": 1,
        "provider": SOURCE_NAME,
        "raw_domain": str(raw_domain),
        "status": "hashing_sources",
        "requested_start_date": start,
        "requested_end_date": end,
        "total": len(selected),
        "completed": 0,
        "processed": 0,
        "pending": len(selected),
        "created": 0,
        "reused": 0,
        "empty": 0,
        "failed": 0,
        "running": 0,
        "last_symbol": "",
        "started_at": utc_now(),
    }

    def write_progress(*, status: str = "") -> None:
        with progress_lock:
            if status:
                progress["status"] = status
            progress["heartbeat_at"] = utc_now()
            progress["elapsed_seconds"] = round(time.monotonic() - started_monotonic, 3)
            progress["available_memory_gib"] = round(
                _available_memory_bytes() / float(1024**3), 3
            )
            progress["free_disk_gib"] = round(
                shutil.disk_usage(paths.root).free / float(1024**3), 3
            )
            atomic_write_json(resolved_job_path, dict(progress))

    write_progress()
    container_hashes = _container_hashes(
        source_index.source_paths,
        enabled=bool(hash_containers),
        cache_path=paths.jobs / "external_quant_5m_container_hash_cache.json",
    )
    write_progress(status="running")
    requested_workers = int(workers or 1)
    if requested_workers < 1 or requested_workers > MAX_EXTERNAL_QUANT_WORKERS:
        raise ValueError(
            "external_quant_5m_workers_out_of_range:"
            f"requested={requested_workers}:max={MAX_EXTERNAL_QUANT_WORKERS}"
        )
    worker_count = min(requested_workers, max(1, len(selected)))
    chunks = [selected[index::worker_count] for index in range(worker_count)]
    chunks = [chunk for chunk in chunks if chunk]
    stop_event = threading.Event()
    fatal_resource_errors: list[str] = []

    def process_chunk(chunk: Sequence[str]) -> list[ExternalQuant5mSymbolResult]:
        outcomes: list[ExternalQuant5mSymbolResult] = []
        with ExitStack() as stack:
            zip_handles = {
                source_path: stack.enter_context(
                    zipfile.ZipFile(source_path, mode="r", metadata_encoding="gbk")
                )
                for source_path in source_index.source_paths
                if source_path.is_file()
            }
            for symbol in chunk:
                if stop_event.is_set():
                    break
                with progress_lock:
                    progress["running"] = int(progress["running"]) + 1
                    progress["last_symbol"] = symbol
                try:
                    _resource_guard(
                        paths.root,
                        min_available_gib=min_available_gib,
                        low_memory_seconds=low_memory_seconds,
                        min_free_disk_gib=min_free_disk_gib,
                    )
                    outcome = _import_symbol(
                        provider_symbol=symbol,
                        segments=by_symbol[symbol],
                        zip_handles=zip_handles,
                        container_hashes=container_hashes,
                        raw_domain=str(raw_domain),
                        workspace_root=workspace_root,
                        start_date=start,
                        end_date=end,
                    )
                except ExternalQuant5mResourceGuardError as exc:
                    stop_event.set()
                    with progress_lock:
                        fatal_resource_errors.append(str(exc))
                    outcome = ExternalQuant5mSymbolResult(
                        provider_symbol=symbol,
                        status="failed",
                        error=f"{type(exc).__name__}:{exc}",
                    )
                except Exception as exc:  # the caller chooses fail-fast after deterministic collection
                    outcome = ExternalQuant5mSymbolResult(
                        provider_symbol=symbol,
                        status="failed",
                        error=f"{type(exc).__name__}:{exc}",
                    )
                outcomes.append(outcome)
                with progress_lock:
                    progress["running"] = max(0, int(progress["running"]) - 1)
                    progress["last_symbol"] = symbol
                    progress["processed"] = int(progress["processed"]) + 1
                    progress["pending"] = max(0, int(progress["total"]) - int(progress["processed"]))
                    if outcome.status != "failed":
                        progress["completed"] = int(progress["completed"]) + 1
                    progress[outcome.status] = int(progress.get(outcome.status, 0)) + 1
                write_progress()
                if stop_event.is_set():
                    break
        return outcomes

    all_results: list[ExternalQuant5mSymbolResult] = []
    use_process_pool = worker_count > 1 and _multiprocess_worker_available()
    with progress_lock:
        progress["parallel_backend"] = "process" if use_process_pool else "thread"
        progress["worker_count"] = worker_count
    write_progress()
    if use_process_pool:
        context = multiprocessing.get_context("spawn")
        selected_iterator = iter(selected)
        pending_futures: dict[Any, str] = {}
        watchdog = _ParentResourceWatchdog(
            destination=paths.root,
            min_available_gib=float(min_available_gib),
            low_memory_seconds=float(low_memory_seconds),
            min_free_disk_gib=float(min_free_disk_gib),
        )
        last_watchdog_heartbeat = time.monotonic()

        def submit_next(executor: ProcessPoolExecutor) -> bool:
            try:
                symbol = next(selected_iterator)
            except StopIteration:
                return False
            future = executor.submit(
                _process_symbol_task,
                provider_symbol=symbol,
                segments=tuple(by_symbol[symbol]),
                container_hashes=dict(container_hashes),
                raw_domain=str(raw_domain),
                workspace_root=workspace_root,
                start_date=start,
                end_date=end,
                min_available_gib=float(min_available_gib),
                low_memory_seconds=float(low_memory_seconds),
                min_free_disk_gib=float(min_free_disk_gib),
            )
            pending_futures[future] = symbol
            with progress_lock:
                progress["running"] = int(progress["running"]) + 1
                progress["last_symbol"] = symbol
            write_progress()
            return True

        executor = ProcessPoolExecutor(max_workers=worker_count, mp_context=context)
        executor_aborted = False
        try:
            for _ in range(worker_count):
                if not submit_next(executor):
                    break
            while pending_futures:
                completed_futures, _ = wait(
                    tuple(pending_futures),
                    timeout=PROCESS_WATCHDOG_INTERVAL_SECONDS,
                    return_when=FIRST_COMPLETED,
                )
                watchdog.poll()
                now = time.monotonic()
                if now - last_watchdog_heartbeat >= PROCESS_HEARTBEAT_SECONDS:
                    write_progress()
                    last_watchdog_heartbeat = now
                for future in completed_futures:
                    symbol = pending_futures.pop(future)
                    try:
                        outcome = future.result()
                    except Exception as exc:
                        raise RuntimeError(
                            f"external_quant_5m_process_result_failed:{type(exc).__name__}"
                        ) from exc
                    if outcome.status == "resource_guard":
                        raise ExternalQuant5mResourceGuardError(
                            outcome.error or "external_quant_5m_worker_resource_guard"
                        )
                    all_results.append(outcome)
                    with progress_lock:
                        progress["running"] = max(0, int(progress["running"]) - 1)
                        progress["last_symbol"] = symbol
                        progress["processed"] = int(progress["processed"]) + 1
                        progress["pending"] = max(
                            0,
                            int(progress["total"]) - int(progress["processed"]),
                        )
                        if outcome.status != "failed":
                            progress["completed"] = int(progress["completed"]) + 1
                        progress[outcome.status] = int(progress.get(outcome.status, 0)) + 1
                    write_progress()
                    last_watchdog_heartbeat = time.monotonic()
                    submit_next(executor)
        except ExternalQuant5mResourceGuardError as exc:
            executor_aborted = True
            stop_event.set()
            fatal_resource_errors.append(str(exc))
            with progress_lock:
                progress["pause_reason"] = "resource_guard"
                progress["error_type"] = type(exc).__name__
            write_progress(status="stopping_resource_guard")
            unreaped = _abort_process_pool(executor, tuple(pending_futures))
            with progress_lock:
                progress["running"] = 0
                progress["pending"] = max(
                    0,
                    int(progress["total"]) - int(progress["processed"]),
                )
                progress["worker_cleanup_incomplete"] = bool(unreaped)
                progress["unreaped_worker_count"] = len(unreaped)
            write_progress(status="paused_resource_guard")
        except KeyboardInterrupt:
            executor_aborted = True
            stop_event.set()
            with progress_lock:
                progress["pause_reason"] = "keyboard_interrupt"
                progress["error_type"] = "KeyboardInterrupt"
            write_progress(status="stopping_interrupt")
            unreaped = _abort_process_pool(executor, tuple(pending_futures))
            with progress_lock:
                progress["running"] = 0
                progress["pending"] = max(
                    0,
                    int(progress["total"]) - int(progress["processed"]),
                )
                progress["worker_cleanup_incomplete"] = bool(unreaped)
                progress["unreaped_worker_count"] = len(unreaped)
            write_progress(status="interrupted_recoverable")
            raise
        except Exception as exc:
            executor_aborted = True
            stop_event.set()
            with progress_lock:
                progress["pause_reason"] = "process_pool_failure"
                progress["error_type"] = type(exc).__name__
            write_progress(status="stopping_pool_failure")
            unreaped = _abort_process_pool(executor, tuple(pending_futures))
            with progress_lock:
                progress["running"] = 0
                progress["pending"] = max(
                    0,
                    int(progress["total"]) - int(progress["processed"]),
                )
                progress["worker_cleanup_incomplete"] = bool(unreaped)
                progress["unreaped_worker_count"] = len(unreaped)
            write_progress(status="interrupted_recoverable")
            raise ExternalQuant5mError(
                f"external_quant_5m_process_pool_interrupted:{type(exc).__name__}"
            ) from exc
        finally:
            if not executor_aborted:
                executor.shutdown(wait=True, cancel_futures=False)
    elif len(chunks) <= 1:
        all_results.extend(process_chunk(chunks[0]) if chunks else [])
    else:
        with ThreadPoolExecutor(max_workers=len(chunks), thread_name_prefix="qdp-external-5m") as executor:
            futures = [executor.submit(process_chunk, chunk) for chunk in chunks]
            for future in as_completed(futures):
                all_results.extend(future.result())
    all_results.sort(key=lambda item: item.provider_symbol)
    if fatal_resource_errors:
        write_progress(status="paused_resource_guard")
        raise ExternalQuant5mResourceGuardError(fatal_resource_errors[0])
    failures = [item for item in all_results if item.status == "failed"]
    if strict and failures:
        first = failures[0]
        write_progress(status="failed")
        raise ExternalQuant5mError(
            f"external_quant_5m_import_failed:{first.provider_symbol}:{first.error}"
        )
    completed = [item for item in all_results if item.status in {"created", "reused"}]
    empty = [item for item in all_results if item.status == "empty"]
    result = ExternalQuant5mImportResult(
        status="completed" if not failures else "completed_with_errors",
        raw_domain=str(raw_domain),
        discovered_symbols=len(by_symbol),
        selected_symbols=len(selected),
        completed_symbols=len(completed),
        created_symbols=sum(item.status == "created" for item in completed),
        reused_symbols=sum(item.status == "reused" for item in completed),
        empty_symbols=len(empty),
        failed_symbols=len(failures),
        row_count=sum(item.row_count for item in completed),
        day_count=sum(item.day_count for item in completed),
        results=tuple(all_results),
    )
    write_progress(status=result.status)
    return result


def _import_symbol(
    *,
    provider_symbol: str,
    segments: Sequence[ExternalQuant5mSegment],
    zip_handles: Mapping[Path, zipfile.ZipFile],
    container_hashes: Mapping[Path, str],
    raw_domain: str,
    workspace_root: str | Path | None,
    start_date: str,
    end_date: str,
) -> ExternalQuant5mSymbolResult:
    inventory_fingerprint = _inventory_fingerprint(
        segments,
        container_hashes=container_hashes,
        start_date=start_date,
        end_date=end_date,
    )
    previous = get_raw_partition(
        raw_domain,
        partition_field="provider_symbol",
        partition_value=provider_symbol,
        workspace_root=workspace_root,
    )
    previous_receipt: dict[str, Any] = {}
    previous_frame = _empty_raw_frame()
    if previous is not None and previous.row_count > 0 and previous.quality_tier == "strict":
        previous_receipt = read_raw_receipt(previous)
    # ZIP containers have a whole-file digest, so an exact prior receipt can
    # be resumed without inflating and hashing every member again. Directory
    # files are re-read because their per-file digest is the provenance root.
    if previous is not None and all(item.source_kind == "zip" for item in segments):
        receipt = previous_receipt or read_raw_receipt(previous)
        if (
            previous.row_count > 0
            and previous.quality_tier == "strict"
            and str(
                receipt.get(
                    "last_request_source_inventory_fingerprint",
                    receipt.get("source_inventory_fingerprint", ""),
                )
            )
            == inventory_fingerprint
            and str(receipt.get("requested_start_date", "")) == start_date
            and str(receipt.get("requested_end_date", "")) == end_date
        ):
            return ExternalQuant5mSymbolResult(
                provider_symbol=provider_symbol,
                status="reused",
                row_count=int(previous.row_count),
                day_count=int(receipt.get("complete_stock_day_count", 0) or 0),
                created=False,
                ref=previous,
            )

    if previous_receipt:
        previous_frame = _coerce_raw_schema(read_raw_partition(previous))
    frames: list[pd.DataFrame] = [previous_frame] if not previous_frame.empty else []
    segment_receipts: list[dict[str, Any]] = [
        dict(item)
        for item in list(previous_receipt.get("source_segments", ()) or ())
        if isinstance(item, Mapping)
    ]
    for segment in segments:
        payload = _read_segment(segment, zip_handles=zip_handles)
        member_sha = hashlib.sha256(payload).hexdigest()
        try:
            raw = pd.read_csv(io.BytesIO(payload), encoding="utf-8-sig")
        except UnicodeDecodeError as exc:
            raise ExternalQuant5mError(
                f"external_quant_5m_csv_not_utf8_sig:{provider_symbol}:{segment.relative_name}"
            ) from exc
        raw_row_count = int(len(raw))
        normalized = normalize_external_quant_5m(
            raw,
            provider_symbol=provider_symbol,
            start_date=start_date,
            end_date=end_date,
        )
        if not normalized.empty:
            frames.append(normalized)
        segment_receipts.append(
            {
                "source_kind": segment.source_kind,
                "container_path": str(segment.source_path),
                "container_sha256": str(container_hashes.get(segment.source_path, "")),
                "container_size": int(segment.source_path.stat().st_size)
                if segment.source_kind == "zip"
                else 0,
                "member_name": segment.relative_name,
                "member_crc32": segment.crc32,
                "member_sha256": member_sha,
                "compressed_size": int(segment.compressed_size),
                "uncompressed_size": int(segment.uncompressed_size),
                "raw_row_count": raw_row_count,
                "normalized_row_count": int(len(normalized)),
                "min_trade_date": str(normalized["trade_date"].min()) if not normalized.empty else "",
                "max_trade_date": str(normalized["trade_date"].max()) if not normalized.empty else "",
            }
        )
    if not frames:
        return ExternalQuant5mSymbolResult(provider_symbol=provider_symbol, status="empty")
    segment_receipts = _merge_segment_receipts(segment_receipts)
    combined = _combine_segment_frames(
        frames,
        provider_symbol=provider_symbol,
    )
    combined = _coerce_raw_schema(combined.sort_values(["trade_date", "bar_time"]).reset_index(drop=True))
    day_counts = combined.groupby("trade_date", sort=False).size()
    complete_dates = [
        str(trade_date)
        for trade_date, day in combined.groupby("trade_date", sort=True)
        if _is_complete_valid_day(day)
    ]
    all_trade_dates = sorted(set(combined["trade_date"].astype(str)))
    incomplete_dates = sorted(set(all_trade_dates).difference(complete_dates))
    complete_count = len(complete_dates)
    incomplete_count = len(incomplete_dates)
    receipt = {
        "provider": SOURCE_NAME,
        "production_role": "historical_bootstrap_primary",
        "upstream_provenance": "user_supplied_baidu_archive",
        "source_reference_url": "https://pan.baidu.com/s/1o6Fwks5c6Ug5GQM8l-oXYg?pwd=uisf",
        "quality_tier": "strict",
        "provider_symbol": provider_symbol,
        "requested_start_date": start_date,
        "requested_end_date": end_date,
        "coverage_start_date": min(complete_dates) if complete_dates else "",
        "coverage_end_date": max(complete_dates) if complete_dates else "",
        "observed_min_trade_date": str(combined["trade_date"].min()),
        "observed_max_trade_date": str(combined["trade_date"].max()),
        "complete_stock_day_count": int(complete_count),
        "incomplete_stock_day_count": int(incomplete_count),
        "complete_trade_dates": complete_dates,
        "incomplete_trade_dates": incomplete_dates,
        "source_raw_row_count": int(sum(int(item.get("raw_row_count", 0) or 0) for item in segment_receipts)),
        "normalized_row_count": int(len(combined)),
        "source_inventory_fingerprint": _receipt_provenance_fingerprint(segment_receipts),
        "last_request_source_inventory_fingerprint": inventory_fingerprint,
        "archive_filename_encoding": "gbk",
        "csv_payload_encoding": "utf-8-sig",
        "bar_contract": "right_closed_5m_48_0935_1500",
        "auction_policy": "merge_0930_call_auction_into_0935",
        "overlap_policy": "exact_deduplicate_conflict_reject",
        "source_segments": segment_receipts,
    }
    ref, created = write_raw_partition(
        raw_domain=raw_domain,
        partition_field="provider_symbol",
        partition_value=provider_symbol,
        frame=combined,
        receipt=receipt,
        workspace_root=workspace_root,
    )
    return ExternalQuant5mSymbolResult(
        provider_symbol=provider_symbol,
        status="created" if created else "reused",
        row_count=int(len(combined)),
        day_count=int(complete_count),
        created=bool(created),
        ref=ref,
    )


def _normalize_day(day: pd.DataFrame, *, provider_symbol: str, trade_date: str) -> pd.DataFrame:
    expected = set(EXPECTED_5M_BAR_ENDS)
    actual = set(day["bar_time"].astype(str))
    if len(day) == 48 and len(actual) == 48 and actual == expected:
        normalized = day.copy()
    elif len(day) == 49 and len(actual) == 49 and actual == expected | {"09:30"}:
        auction = day.loc[day["bar_time"].eq("09:30")]
        first = day.loc[day["bar_time"].eq("09:35")]
        if len(auction) != 1 or len(first) != 1:
            return day.sort_values("bar_time", kind="stable").reset_index(drop=True)
        normalized = day.loc[~day["bar_time"].eq("09:30")].copy()
        first_index = normalized.index[normalized["bar_time"].eq("09:35")]
        if len(first_index) != 1:
            return day.sort_values("bar_time", kind="stable").reset_index(drop=True)
        auction_row = auction.iloc[0]
        first_row = first.iloc[0]
        target_index = first_index[0]
        normalized.loc[target_index, "open"] = (
            float(auction_row["open"])
            if _positive_finite(auction_row["open"])
            else float(first_row["open"])
        )
        positive_highs = [
            float(value)
            for value in (auction_row["high"], first_row["high"])
            if _positive_finite(value)
        ]
        positive_lows = [
            float(value)
            for value in (auction_row["low"], first_row["low"])
            if _positive_finite(value)
        ]
        if not positive_highs or not positive_lows:
            return day.sort_values("bar_time", kind="stable").reset_index(drop=True)
        normalized.loc[target_index, "high"] = max(positive_highs)
        normalized.loc[target_index, "low"] = min(positive_lows)
        normalized.loc[target_index, "close"] = float(first_row["close"])
        normalized.loc[target_index, "volume"] = float(auction_row["volume"]) + float(first_row["volume"])
        normalized.loc[target_index, "amount"] = float(auction_row["amount"]) + float(first_row["amount"])
    else:
        # Preserve malformed evidence exactly.  The canonical 48-bar gate will
        # quarantine only this stock-day; the rest of a multi-year symbol must
        # remain available.
        return day.sort_values("bar_time", kind="stable").reset_index(drop=True)
    normalized = normalized.sort_values("bar_time", kind="stable").reset_index(drop=True)
    return normalized


def _validate_complete_day(day: pd.DataFrame, *, provider_symbol: str, trade_date: str) -> None:
    if len(day) != 48 or tuple(day["bar_time"].astype(str)) != EXPECTED_5M_BAR_ENDS:
        raise ExternalQuant5mError(
            f"external_quant_5m_day_not_48_right_closed:{provider_symbol}:{trade_date}"
        )
    numeric = day[_NUMERIC_COLUMNS].apply(pd.to_numeric, errors="coerce")
    if not np.isfinite(numeric.to_numpy(dtype="float64")).all():
        raise ExternalQuant5mError(
            f"external_quant_5m_non_finite:{provider_symbol}:{trade_date}"
        )
    if numeric[_PRICE_COLUMNS].le(0).any().any():
        raise ExternalQuant5mError(
            f"external_quant_5m_non_positive_price:{provider_symbol}:{trade_date}"
        )
    if numeric[["volume", "amount"]].lt(0).any().any():
        raise ExternalQuant5mError(
            f"external_quant_5m_negative_volume_or_amount:{provider_symbol}:{trade_date}"
        )
    if (numeric["high"] < numeric[["open", "low", "close"]].max(axis=1)).any():
        raise ExternalQuant5mError(
            f"external_quant_5m_high_relation_invalid:{provider_symbol}:{trade_date}"
        )
    if (numeric["low"] > numeric[["open", "high", "close"]].min(axis=1)).any():
        raise ExternalQuant5mError(
            f"external_quant_5m_low_relation_invalid:{provider_symbol}:{trade_date}"
        )


def _combine_segment_frames(
    frames: Sequence[pd.DataFrame],
    *,
    provider_symbol: str,
) -> pd.DataFrame:
    if not frames:
        return _empty_raw_frame()
    keys = ["provider_symbol", "trade_date", "bar_time"]
    combined = _coerce_raw_schema(frames[0])
    for frame in frames[1:]:
        incoming = _coerce_raw_schema(frame)
        existing_keys = set(map(tuple, combined[keys].itertuples(index=False, name=None)))
        incoming_keys = set(map(tuple, incoming[keys].itertuples(index=False, name=None)))
        overlap = sorted(existing_keys.intersection(incoming_keys))
        for key in overlap:
            left_mask = pd.Series(True, index=combined.index)
            right_mask = pd.Series(True, index=incoming.index)
            for column, value in zip(keys, key):
                left_mask &= combined[column].eq(value)
                right_mask &= incoming[column].eq(value)
            left = (
                combined.loc[left_mask, RAW_COLUMNS]
                .drop_duplicates()
                .sort_values(RAW_COLUMNS, kind="stable")
                .reset_index(drop=True)
            )
            right = (
                incoming.loc[right_mask, RAW_COLUMNS]
                .drop_duplicates()
                .sort_values(RAW_COLUMNS, kind="stable")
                .reset_index(drop=True)
            )
            if not left.equals(right):
                raise ExternalQuant5mError(
                    "external_quant_5m_overlap_conflict:"
                    f"{provider_symbol}:{key[1]}:{key[2]}"
                )
        if overlap:
            overlap_set = set(overlap)
            keep = [
                tuple(row) not in overlap_set
                for row in incoming[keys].itertuples(index=False, name=None)
            ]
            incoming = incoming.loc[keep]
        combined = pd.concat([combined, incoming], ignore_index=True)
    return _coerce_raw_schema(combined)


def _merge_segment_receipts(items: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    merged: dict[tuple[str, str, str, str], dict[str, Any]] = {}
    for source in items:
        item = dict(source)
        key = (
            str(item.get("container_path", "")),
            str(item.get("container_sha256", "")),
            str(item.get("member_name", "")),
            str(item.get("member_sha256", "")),
        )
        previous = merged.get(key)
        if previous is None:
            merged[key] = item
            continue
        previous_score = (
            int(previous.get("normalized_row_count", 0) or 0),
            str(previous.get("max_trade_date", "")),
            str(previous.get("min_trade_date", "")),
        )
        current_score = (
            int(item.get("normalized_row_count", 0) or 0),
            str(item.get("max_trade_date", "")),
            str(item.get("min_trade_date", "")),
        )
        if current_score > previous_score:
            merged[key] = item
    return [merged[key] for key in sorted(merged)]


def _receipt_provenance_fingerprint(items: Sequence[Mapping[str, Any]]) -> str:
    records = [
        {
            "source_kind": str(item.get("source_kind", "")),
            "container_path": str(item.get("container_path", "")),
            "container_sha256": str(item.get("container_sha256", "")),
            "member_name": str(item.get("member_name", "")),
            "member_crc32": str(item.get("member_crc32", "")),
            "member_sha256": str(item.get("member_sha256", "")),
        }
        for item in items
    ]
    return hashlib.sha256(
        json.dumps(records, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def _is_complete_valid_day(day: pd.DataFrame) -> bool:
    try:
        ordered = day.sort_values("bar_time", kind="stable").reset_index(drop=True)
        _validate_complete_day(
            ordered,
            provider_symbol=str(ordered["provider_symbol"].iloc[0]),
            trade_date=str(ordered["trade_date"].iloc[0]),
        )
    except (ExternalQuant5mError, IndexError):
        return False
    return True


def _read_segment(
    segment: ExternalQuant5mSegment,
    *,
    zip_handles: Mapping[Path, zipfile.ZipFile],
) -> bytes:
    if segment.source_kind == "zip":
        archive = zip_handles.get(segment.source_path)
        if archive is None:
            raise ExternalQuant5mError(f"external_quant_5m_zip_handle_missing:{segment.source_path}")
        try:
            payload = archive.read(segment.member_name)
        except zipfile.BadZipFile as exc:
            raise ExternalQuant5mError(
                f"external_quant_5m_zip_crc_or_body_error:{segment.source_path}:{segment.member_name}"
            ) from exc
        actual_crc = f"{zipfile.crc32(payload) & 0xFFFFFFFF:08x}"
        if actual_crc != segment.crc32:
            raise ExternalQuant5mError(
                f"external_quant_5m_member_crc_mismatch:{segment.member_name}:"
                f"expected={segment.crc32}:actual={actual_crc}"
            )
        return payload
    path = Path(segment.member_name)
    payload = path.read_bytes()
    if len(payload) != segment.uncompressed_size:
        raise ExternalQuant5mError(
            f"external_quant_5m_file_size_changed:{path}:"
            f"expected={segment.uncompressed_size}:actual={len(payload)}"
        )
    return payload


def _container_hashes(
    source_paths: Sequence[Path],
    *,
    enabled: bool,
    cache_path: Path,
) -> dict[Path, str]:
    if not enabled:
        return {}
    cached_payload = read_json(cache_path)
    cached_entries = dict(cached_payload.get("entries", {}) or {})
    resolved: dict[Path, str] = {}
    changed = False
    for path in source_paths:
        if not path.is_file():
            continue
        stat = path.stat()
        key = str(path.resolve())
        cached = dict(cached_entries.get(key, {}) or {})
        if (
            int(cached.get("size", -1)) == int(stat.st_size)
            and int(cached.get("mtime_ns", -1)) == int(stat.st_mtime_ns)
            and re.fullmatch(r"[0-9a-f]{64}", str(cached.get("sha256", "")))
        ):
            digest = str(cached["sha256"])
        else:
            digest = sha256_file(path)
            cached_entries[key] = {
                "path": key,
                "size": int(stat.st_size),
                "mtime_ns": int(stat.st_mtime_ns),
                "sha256": digest,
                "verified_at": utc_now(),
            }
            changed = True
        resolved[path] = digest
    if changed or not cache_path.exists():
        atomic_write_json(
            cache_path,
            {
                "cache_version": 1,
                "entries": cached_entries,
                "updated_at": utc_now(),
            },
        )
    return resolved


def _selection_fingerprint(source_paths: Sequence[Path], symbols: Sequence[str]) -> str:
    payload = {
        "source_paths": [str(item.resolve()) for item in source_paths],
        "symbols": list(symbols),
    }
    return hashlib.sha256(
        json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def _inventory_fingerprint(
    segments: Sequence[ExternalQuant5mSegment],
    *,
    container_hashes: Mapping[Path, str],
    start_date: str,
    end_date: str,
) -> str:
    payload = {
        "start_date": start_date,
        "end_date": end_date,
        "segments": [
            {
                **item.inventory_record(),
                "container_sha256": str(container_hashes.get(item.source_path, "")),
            }
            for item in sorted(segments, key=_segment_sort_key)
        ],
    }
    return hashlib.sha256(
        json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def _coerce_raw_schema(frame: pd.DataFrame) -> pd.DataFrame:
    data = frame.copy()
    for column in RAW_COLUMNS:
        if column not in data.columns:
            data[column] = np.nan if column in _NUMERIC_COLUMNS else ""
    for column in ("provider_symbol", "trade_date", "bar_time", "source"):
        data[column] = data[column].fillna("").astype("string")
    for column in _NUMERIC_COLUMNS:
        data[column] = pd.to_numeric(data[column], errors="coerce").astype("float64")
    return data.loc[:, RAW_COLUMNS].reset_index(drop=True)


def _empty_raw_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "provider_symbol": pd.Series(dtype="string"),
            "trade_date": pd.Series(dtype="string"),
            "bar_time": pd.Series(dtype="string"),
            "open": pd.Series(dtype="float64"),
            "high": pd.Series(dtype="float64"),
            "low": pd.Series(dtype="float64"),
            "close": pd.Series(dtype="float64"),
            "volume": pd.Series(dtype="float64"),
            "amount": pd.Series(dtype="float64"),
            "source": pd.Series(dtype="string"),
        }
    )


def _find_column(frame: pd.DataFrame, aliases: Sequence[str]) -> str:
    exact = {str(column).strip(): str(column) for column in frame.columns}
    lower = {str(column).strip().lower(): str(column) for column in frame.columns}
    for alias in aliases:
        if alias in exact:
            return exact[alias]
        if alias.lower() in lower:
            return lower[alias.lower()]
    return ""


def _provider_symbol_from_name(value: str) -> str:
    basename = str(value).replace("\\", "/").rsplit("/", 1)[-1]
    match = _MEMBER_PATTERN.fullmatch(basename)
    if not match:
        return ""
    return f"{match.group('code')}.{match.group('exchange').upper()}"


def _resolve_source_paths(source_paths: Iterable[str | Path]) -> tuple[Path, ...]:
    resolved: list[Path] = []
    for value in source_paths:
        path = Path(value).expanduser().resolve()
        if path in resolved:
            continue
        if not path.exists():
            raise FileNotFoundError(f"external_quant_5m_source_missing:{path}")
        if not path.is_file() and not path.is_dir():
            raise ValueError(f"external_quant_5m_source_not_file_or_directory:{path}")
        resolved.append(path)
    if not resolved:
        raise ValueError("external_quant_5m_explicit_source_paths_required")
    return tuple(resolved)


def _segment_sort_key(segment: ExternalQuant5mSegment) -> tuple[str, str, str]:
    return (str(segment.source_path).lower(), segment.source_kind, segment.relative_name.lower())


def _date_text(value: Any) -> str:
    parsed = pd.to_datetime(str(value or "").strip(), errors="coerce")
    if pd.isna(parsed):
        raise ValueError(f"external_quant_5m_invalid_date:{value}")
    return parsed.strftime("%Y-%m-%d")


def _positive_finite(value: Any) -> bool:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return False
    return math.isfinite(number) and number > 0


def _available_memory_bytes() -> int:
    try:
        import psutil  # type: ignore[import-not-found]

        return int(psutil.virtual_memory().available)
    except (ImportError, OSError):
        if os.name != "nt":
            return 2**63 - 1
        import ctypes

        class MemoryStatus(ctypes.Structure):
            _fields_ = [
                ("dwLength", ctypes.c_ulong),
                ("dwMemoryLoad", ctypes.c_ulong),
                ("ullTotalPhys", ctypes.c_ulonglong),
                ("ullAvailPhys", ctypes.c_ulonglong),
                ("ullTotalPageFile", ctypes.c_ulonglong),
                ("ullAvailPageFile", ctypes.c_ulonglong),
                ("ullTotalVirtual", ctypes.c_ulonglong),
                ("ullAvailVirtual", ctypes.c_ulonglong),
                ("sullAvailExtendedVirtual", ctypes.c_ulonglong),
            ]

        status = MemoryStatus()
        status.dwLength = ctypes.sizeof(MemoryStatus)
        if not ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
            return 2**63 - 1
        return int(status.ullAvailPhys)


@dataclass
class _ParentResourceWatchdog:
    """Non-blocking parent-side guard for work already running in children."""

    destination: Path
    min_available_gib: float
    low_memory_seconds: float
    min_free_disk_gib: float
    clock: Callable[[], float] = time.monotonic
    memory_reader: Callable[[], int] = _available_memory_bytes
    disk_reader: Callable[[Path], int] = lambda path: int(shutil.disk_usage(path).free)
    low_memory_since: float | None = None

    def poll(self) -> None:
        gib = float(1024**3)
        now = float(self.clock())
        minimum_memory = max(0.0, float(self.min_available_gib)) * gib
        if minimum_memory > 0 and int(self.memory_reader()) < minimum_memory:
            if self.low_memory_since is None:
                self.low_memory_since = now
            if now - self.low_memory_since >= max(0.0, float(self.low_memory_seconds)):
                raise ExternalQuant5mResourceGuardError(
                    "external_quant_5m_low_memory_sustained:"
                    f"threshold_gib={float(self.min_available_gib):.3f}"
                )
        else:
            self.low_memory_since = None
        free_disk = int(self.disk_reader(self.destination))
        if free_disk < max(0.0, float(self.min_free_disk_gib)) * gib:
            raise ExternalQuant5mResourceGuardError(
                "external_quant_5m_low_disk:"
                f"free_gib={free_disk / gib:.3f}:"
                f"threshold_gib={float(self.min_free_disk_gib):.3f}"
            )


def _abort_process_pool(
    executor: ProcessPoolExecutor,
    pending_futures: Sequence[Any],
) -> tuple[int, ...]:
    """Cancel queued work and terminate only worker processes owned by this pool."""

    for future in pending_futures:
        try:
            future.cancel()
        except Exception:
            pass
    processes = tuple((getattr(executor, "_processes", None) or {}).values())
    for process in processes:
        try:
            if process.is_alive():
                process.terminate()
        except (OSError, ValueError):
            pass
    deadline = time.monotonic() + 3.0
    for process in processes:
        try:
            process.join(timeout=max(0.0, deadline - time.monotonic()))
        except (OSError, ValueError):
            pass
    for process in processes:
        try:
            if process.is_alive():
                process.kill()
        except (AttributeError, OSError, ValueError):
            pass
    kill_deadline = time.monotonic() + 2.0
    for process in processes:
        try:
            process.join(timeout=max(0.0, kill_deadline - time.monotonic()))
        except (OSError, ValueError):
            pass
    unreaped: list[int] = []
    for process in processes:
        try:
            if process.is_alive():
                unreaped.append(int(process.pid or 0))
        except (OSError, ValueError):
            unreaped.append(int(getattr(process, "pid", 0) or 0))
    executor.shutdown(wait=False, cancel_futures=True)
    return tuple(item for item in unreaped if item > 0)


def _resource_guard(
    destination: Path,
    *,
    min_available_gib: float,
    low_memory_seconds: float,
    min_free_disk_gib: float,
) -> None:
    gib = float(1024**3)
    minimum_memory = max(0.0, float(min_available_gib)) * gib
    if minimum_memory > 0 and _available_memory_bytes() < minimum_memory:
        deadline = time.monotonic() + max(0.0, float(low_memory_seconds))
        while time.monotonic() < deadline:
            time.sleep(min(0.25, max(0.0, deadline - time.monotonic())))
            if _available_memory_bytes() >= minimum_memory:
                break
        else:
            raise ExternalQuant5mResourceGuardError(
                f"external_quant_5m_low_memory_sustained:threshold_gib={float(min_available_gib):.3f}"
            )
    free_disk = int(shutil.disk_usage(destination).free)
    if free_disk < max(0.0, float(min_free_disk_gib)) * gib:
        raise ExternalQuant5mResourceGuardError(
            "external_quant_5m_low_disk:"
            f"free_gib={free_disk / gib:.3f}:threshold_gib={float(min_free_disk_gib):.3f}"
        )


__all__ = [
    "DEFAULT_RAW_DOMAIN",
    "RAW_COLUMNS",
    "SOURCE_NAME",
    "ExternalQuant5mError",
    "ExternalQuant5mImportResult",
    "ExternalQuant5mResourceGuardError",
    "ExternalQuant5mSegment",
    "ExternalQuant5mSourceIndex",
    "ExternalQuant5mSymbolResult",
    "import_external_quant_5m",
    "normalize_external_quant_5m",
    "scan_external_quant_5m_sources",
]
