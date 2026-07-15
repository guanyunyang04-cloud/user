from __future__ import annotations

import ctypes
import os
import re
import shutil
import threading
import time
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence

import duckdb
import numpy as np
import pandas as pd

from quant_data_platform.core.json_io import read_json
from quant_data_platform.qdp_v2.manifest import (
    DatasetManifest,
    ShardManifestEntry,
    dataset_manifest_path,
    qdp_v2_root,
    read_dataset_manifest,
    resolve_manifest_path,
)
from quant_data_platform.qdp_v2.status import active_dataset_map
from quant_data_platform.qdp_v3.constants import (
    EXPECTED_5M_BAR_ENDS,
    QUALITY_STRICT,
    RAW_LEGACY_V2_INTRADAY_5M,
)
from quant_data_platform.qdp_v3.external_quant_5m import RAW_COLUMNS
from quant_data_platform.qdp_v3.identity import normalize_symbol
from quant_data_platform.qdp_v3.manifest import (
    atomic_write_json,
    sha256_file,
    stable_hash,
    utc_now,
)
from quant_data_platform.qdp_v3.paths import ensure_qdp_v3_layout, qdp_v3_paths
from quant_data_platform.qdp_v3.storage import (
    RawPartitionRef,
    get_raw_partition,
    read_raw_receipt,
    write_raw_partition,
)


SOURCE_NAME = "qdp_v2_migration"
DEFAULT_RAW_DOMAIN = RAW_LEGACY_V2_INTRADAY_5M
MAX_WORKERS = 8
_V2_DOMAIN = "market_intraday_5m"
_EXPECTED_PRIMARY_KEY = ("trade_date", "symbol", "bar_time")
_REQUIRED_COLUMNS = (
    "symbol",
    "trade_date",
    "bar_time",
    "open",
    "high",
    "low",
    "close",
    "volume",
    "amount",
    "adjusted_flag",
)
_NUMERIC_COLUMNS = ("open", "high", "low", "close", "volume", "amount")
_PRICE_COLUMNS = ("open", "high", "low", "close")
_SYMBOL_PATTERN = re.compile(r"\d{6}\.(?:SH|SZ|BJ)")


class LegacyV2FiveMinuteError(RuntimeError):
    """Raised when the active v2 5m evidence cannot be migrated safely."""


class LegacyV2FiveMinuteResourceGuardError(LegacyV2FiveMinuteError):
    """Raised after the configured memory or disk safety line is breached."""


@dataclass(frozen=True)
class LegacyV2ShardEvidence:
    manifest_path: str
    resolved_path: Path
    row_count: int
    start_date: str
    end_date: str
    file_size: int
    size: int
    modified_ns: int

    @property
    def stat_fingerprint(self) -> dict[str, Any]:
        return {
            "manifest_path": self.manifest_path,
            "size": int(self.size),
            "modified_ns": int(self.modified_ns),
        }


@dataclass(frozen=True)
class LegacyV2Snapshot:
    root: Path
    active_path: Path
    active_sha256: str
    manifest_path: Path
    manifest_sha256: str
    manifest: DatasetManifest
    shards: tuple[LegacyV2ShardEvidence, ...]
    missing_parent_dataset_ids: tuple[str, ...]

    @property
    def shard_stat_fingerprint(self) -> str:
        return stable_hash(
            {"shards": [item.stat_fingerprint for item in self.shards]}
        )


@dataclass(frozen=True)
class LegacyV2FiveMinuteSymbolResult:
    provider_symbol: str
    status: str
    row_count: int = 0
    day_count: int = 0
    rejected_day_count: int = 0
    created: bool = False
    ref: RawPartitionRef | None = None
    rejected_trade_dates: tuple[str, ...] = ()
    error: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "provider_symbol": self.provider_symbol,
            "status": self.status,
            "row_count": int(self.row_count),
            "day_count": int(self.day_count),
            "rejected_day_count": int(self.rejected_day_count),
            "created": bool(self.created),
            "content_sha256": self.ref.content_sha256 if self.ref else "",
            "rejected_trade_dates": list(self.rejected_trade_dates),
            "error": self.error,
        }


@dataclass(frozen=True)
class LegacyV2FiveMinuteMigrationResult:
    status: str
    raw_domain: str
    v2_dataset_id: str
    discovered_symbols: int
    selected_symbols: int
    completed_symbols: int
    created_symbols: int
    reused_symbols: int
    empty_symbols: int
    failed_symbols: int
    row_count: int
    day_count: int
    rejected_day_count: int
    results: tuple[LegacyV2FiveMinuteSymbolResult, ...] = field(default_factory=tuple)

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
            "v2_dataset_id": self.v2_dataset_id,
            "discovered_symbols": int(self.discovered_symbols),
            "selected_symbols": int(self.selected_symbols),
            "completed_symbols": int(self.completed_symbols),
            "created_symbols": int(self.created_symbols),
            "reused_symbols": int(self.reused_symbols),
            "empty_symbols": int(self.empty_symbols),
            "failed_symbols": int(self.failed_symbols),
            "row_count": int(self.row_count),
            "day_count": int(self.day_count),
            "rejected_day_count": int(self.rejected_day_count),
            "errors": list(self.errors),
            "results": [item.to_dict() for item in self.results],
        }


@dataclass
class _ResourceWatchdog:
    destination: Path
    min_available_gib: float
    low_memory_seconds: float
    min_free_disk_gib: float
    clock: Callable[[], float] = time.monotonic
    memory_reader: Callable[[], int] = lambda: _available_memory_bytes()
    disk_reader: Callable[[Path], int] = lambda path: int(shutil.disk_usage(path).free)
    low_memory_since: float | None = None

    def poll(self) -> None:
        gib = float(1024**3)
        now = float(self.clock())
        threshold = max(0.0, float(self.min_available_gib)) * gib
        available = int(self.memory_reader())
        if threshold > 0 and available < threshold:
            if self.low_memory_since is None:
                self.low_memory_since = now
            if now - self.low_memory_since >= max(0.0, float(self.low_memory_seconds)):
                raise LegacyV2FiveMinuteResourceGuardError(
                    "legacy_v2_5m_low_memory_sustained:"
                    f"available_gib={available / gib:.3f}:"
                    f"threshold_gib={float(self.min_available_gib):.3f}"
                )
        else:
            self.low_memory_since = None
        free = int(self.disk_reader(self.destination))
        if free < max(0.0, float(self.min_free_disk_gib)) * gib:
            raise LegacyV2FiveMinuteResourceGuardError(
                "legacy_v2_5m_low_disk:"
                f"free_gib={free / gib:.3f}:"
                f"threshold_gib={float(self.min_free_disk_gib):.3f}"
            )


def migrate_v2_active_intraday_5m(
    *,
    workspace_root: str | Path | None = None,
    start_date: str = "",
    end_date: str = "",
    symbols: Iterable[str] = (),
    workers: int = MAX_WORKERS,
    strict: bool = True,
    job_path: str | Path | None = None,
    min_available_gib: float = 0.5,
    low_memory_seconds: float = 5.0,
    min_free_disk_gib: float = 200.0,
    min_staging_free_disk_gib: float = 20.0,
) -> LegacyV2FiveMinuteMigrationResult:
    """Migrate only valid stock-days from the v2 active 5m dataset.

    The v2 active pointer and its exact manifest-listed shards are immutable
    inputs. DuckDB performs one fan-out into runtime staging; validation and
    content-addressed writes then happen independently per provider symbol.
    No v2 file is modified or deleted.
    """

    requested_start = _date_text(start_date) if str(start_date or "").strip() else ""
    requested_end = _date_text(end_date) if str(end_date or "").strip() else ""
    if requested_start and requested_end and requested_start > requested_end:
        raise ValueError(
            f"legacy_v2_5m_invalid_date_range:{requested_start}:{requested_end}"
        )
    requested_symbols = tuple(sorted({_validated_symbol(item) for item in symbols}))
    worker_count = int(workers or MAX_WORKERS)
    if worker_count < 1 or worker_count > MAX_WORKERS:
        raise ValueError(
            f"legacy_v2_5m_workers_out_of_range:requested={worker_count}:max={MAX_WORKERS}"
        )

    paths = ensure_qdp_v3_layout(workspace_root)
    snapshot = _resolve_v2_snapshot(workspace_root)
    selection_fingerprint = stable_hash(
        {
            "source": SOURCE_NAME,
            "raw_domain": DEFAULT_RAW_DOMAIN,
            "v2_active_sha256": snapshot.active_sha256,
            "v2_dataset_manifest_sha256": snapshot.manifest_sha256,
            "v2_shard_stat_fingerprint": snapshot.shard_stat_fingerprint,
            "start_date": requested_start,
            "end_date": requested_end,
            "symbols": list(requested_symbols),
            "strict": bool(strict),
        }
    )
    resolved_job_path = (
        Path(job_path).expanduser().resolve()
        if job_path
        else paths.jobs / f"legacy_v2_5m__{selection_fingerprint[:16]}.json"
    )
    existing_job = read_json(resolved_job_path)
    if existing_job and str(existing_job.get("selection_fingerprint", "")) not in {
        "",
        selection_fingerprint,
    }:
        raise LegacyV2FiveMinuteError(
            f"legacy_v2_5m_job_identity_conflict:{resolved_job_path}"
        )
    completed_resume = _completed_job_result(
        existing_job,
        snapshot=snapshot,
        selection_fingerprint=selection_fingerprint,
        raw_domain=DEFAULT_RAW_DOMAIN,
        workspace_root=workspace_root,
    )
    if completed_resume is not None:
        return completed_resume

    started_monotonic = time.monotonic()
    progress_lock = threading.RLock()
    progress: dict[str, Any] = {
        "job_version": 1,
        "provider": SOURCE_NAME,
        "raw_domain": DEFAULT_RAW_DOMAIN,
        "v2_dataset_id": snapshot.manifest.dataset_id,
        "v2_active_sha256": snapshot.active_sha256,
        "v2_dataset_manifest_sha256": snapshot.manifest_sha256,
        "v2_shard_stat_fingerprint": snapshot.shard_stat_fingerprint,
        "selection_fingerprint": selection_fingerprint,
        "requested_start_date": requested_start,
        "requested_end_date": requested_end,
        "strict": bool(strict),
        "worker_count": worker_count,
        "status": "preparing_fanout",
        "total": 0,
        "completed": 0,
        "processed": 0,
        "pending": 0,
        "created": 0,
        "reused": 0,
        "empty": 0,
        "failed": 0,
        "running": 0,
        "row_count": 0,
        "day_count": 0,
        "rejected_day_count": 0,
        "last_symbol": "",
        "started_at": str(existing_job.get("started_at", "") or utc_now()),
        "results": [],
    }

    def write_progress(*, status: str = "") -> None:
        with progress_lock:
            if status:
                progress["status"] = status
            progress["heartbeat_at"] = utc_now()
            progress["elapsed_seconds"] = round(
                time.monotonic() - started_monotonic, 3
            )
            progress["available_memory_gib"] = round(
                _available_memory_bytes() / float(1024**3), 3
            )
            progress["free_disk_gib"] = round(
                shutil.disk_usage(paths.root).free / float(1024**3), 3
            )
            progress["staging_free_disk_gib"] = round(
                shutil.disk_usage(paths.staging).free / float(1024**3), 3
            )
            atomic_write_json(resolved_job_path, dict(progress))

    write_progress()
    try:
        _resource_guard(
            paths.root,
            min_available_gib=min_available_gib,
            low_memory_seconds=low_memory_seconds,
            min_free_disk_gib=min_free_disk_gib,
        )
        _resource_guard(
            paths.staging,
            min_available_gib=min_available_gib,
            low_memory_seconds=low_memory_seconds,
            min_free_disk_gib=min_staging_free_disk_gib,
        )
        staging_root = paths.staging / f"legacy_v2_5m__{selection_fingerprint[:16]}"
        write_progress(status="fanout")
        stage_symbols = _fan_out_active_shards(
            snapshot,
            staging_root=staging_root,
            requested_start=requested_start,
            requested_end=requested_end,
            requested_symbols=requested_symbols,
            workers=worker_count,
            destination=paths.root,
            min_available_gib=min_available_gib,
            low_memory_seconds=low_memory_seconds,
            min_free_disk_gib=min_free_disk_gib,
            min_staging_free_disk_gib=min_staging_free_disk_gib,
        )
        _verify_snapshot_unchanged(snapshot)
        selected_symbols = requested_symbols or stage_symbols
        found_symbols = set(stage_symbols)
        missing_requested = tuple(
            symbol for symbol in selected_symbols if symbol not in found_symbols
        )
        with progress_lock:
            progress["discovered_symbols"] = len(stage_symbols)
            progress["total"] = len(selected_symbols)
            progress["pending"] = len(selected_symbols)
        write_progress(status="running")

        shard_by_key = {
            _path_key(item.resolved_path): item for item in snapshot.shards
        }
        results: list[LegacyV2FiveMinuteSymbolResult] = []
        watchdog = _ResourceWatchdog(
            destination=paths.root,
            min_available_gib=float(min_available_gib),
            low_memory_seconds=float(low_memory_seconds),
            min_free_disk_gib=float(min_free_disk_gib),
        )
        staging_watchdog = _ResourceWatchdog(
            destination=paths.staging,
            min_available_gib=float(min_available_gib),
            low_memory_seconds=float(low_memory_seconds),
            min_free_disk_gib=float(min_staging_free_disk_gib),
        )

        def run_symbol(symbol: str) -> LegacyV2FiveMinuteSymbolResult:
            _resource_guard(
                paths.root,
                min_available_gib=min_available_gib,
                low_memory_seconds=low_memory_seconds,
                min_free_disk_gib=min_free_disk_gib,
            )
            _resource_guard(
                paths.staging,
                min_available_gib=min_available_gib,
                low_memory_seconds=low_memory_seconds,
                min_free_disk_gib=min_staging_free_disk_gib,
            )
            return _migrate_staged_symbol(
                provider_symbol=symbol,
                stage_directory=staging_root / f"provider_symbol={symbol}",
                snapshot=snapshot,
                shard_by_key=shard_by_key,
                raw_domain=DEFAULT_RAW_DOMAIN,
                requested_start=requested_start,
                requested_end=requested_end,
                workspace_root=workspace_root,
            )

        futures: dict[Future[LegacyV2FiveMinuteSymbolResult], str] = {}
        executor = ThreadPoolExecutor(
            max_workers=min(worker_count, max(1, len(stage_symbols))),
            thread_name_prefix="qdp-v2-5m-migrate",
        )
        try:
            for symbol in stage_symbols:
                futures[executor.submit(run_symbol, symbol)] = symbol
            pending_futures = set(futures)
            while pending_futures:
                completed_futures, pending_futures = wait(
                    pending_futures,
                    timeout=1.0,
                    return_when=FIRST_COMPLETED,
                )
                watchdog.poll()
                staging_watchdog.poll()
                for future in completed_futures:
                    symbol = futures[future]
                    try:
                        outcome = future.result()
                    except LegacyV2FiveMinuteResourceGuardError:
                        raise
                    except Exception as exc:
                        outcome = LegacyV2FiveMinuteSymbolResult(
                            provider_symbol=symbol,
                            status="failed",
                            error=f"{type(exc).__name__}:{exc}",
                        )
                    results.append(outcome)
                    _record_progress_outcome(progress, outcome, lock=progress_lock)
                    write_progress()
        except BaseException:
            for future in futures:
                future.cancel()
            executor.shutdown(wait=True, cancel_futures=True)
            raise
        else:
            executor.shutdown(wait=True)

        for symbol in missing_requested:
            outcome = LegacyV2FiveMinuteSymbolResult(
                provider_symbol=symbol,
                status="empty",
            )
            results.append(outcome)
            _record_progress_outcome(progress, outcome, lock=progress_lock)
            write_progress()

        results.sort(key=lambda item: item.provider_symbol)
        _verify_snapshot_unchanged(snapshot)
        failed_count = sum(item.status == "failed" for item in results)
        final_status = "completed_with_errors" if failed_count else "completed"
        result = _migration_result(
            status=final_status,
            raw_domain=DEFAULT_RAW_DOMAIN,
            v2_dataset_id=snapshot.manifest.dataset_id,
            discovered_symbols=len(stage_symbols),
            selected_symbols=len(selected_symbols),
            results=results,
        )
        with progress_lock:
            progress["result"] = result.to_dict()
            progress["results"] = [item.to_dict() for item in results]
        write_progress(status=final_status)
        if failed_count and strict:
            first = next(item for item in results if item.status == "failed")
            with progress_lock:
                progress["first_error"] = first.error
            write_progress(status="failed")
            raise LegacyV2FiveMinuteError(
                f"legacy_v2_5m_migration_failed:{first.provider_symbol}:{first.error}"
            )
        _safe_remove_staging(staging_root, staging_parent=paths.staging)
        return result
    except LegacyV2FiveMinuteResourceGuardError as exc:
        with progress_lock:
            progress["pause_reason"] = "resource_guard"
            progress["error"] = f"{type(exc).__name__}:{exc}"
        write_progress(status="paused_resource_guard")
        raise
    except (KeyboardInterrupt, SystemExit):
        write_progress(status="interrupted_recoverable")
        raise
    except LegacyV2FiveMinuteError:
        if str(progress.get("status", "")) not in {"failed", "paused_resource_guard"}:
            write_progress(status="failed")
        raise
    except Exception as exc:
        with progress_lock:
            progress["error"] = f"{type(exc).__name__}:{exc}"
        write_progress(status="failed")
        raise LegacyV2FiveMinuteError(
            f"legacy_v2_5m_migration_failed:{type(exc).__name__}:{exc}"
        ) from exc


def _resolve_v2_snapshot(
    workspace_root: str | Path | None,
) -> LegacyV2Snapshot:
    root = qdp_v2_root(workspace_root).resolve()
    active_path = root / "active" / "active.json"
    if not active_path.exists():
        raise FileNotFoundError(f"legacy_v2_5m_active_missing:{active_path}")
    active_sha_before = sha256_file(active_path)
    active = read_json(active_path)
    active_sha_after = sha256_file(active_path)
    if active_sha_before != active_sha_after:
        raise LegacyV2FiveMinuteError("legacy_v2_5m_active_changed_while_reading")
    dataset_id = active_dataset_map(active).get(_V2_DOMAIN, "")
    if not dataset_id:
        raise LegacyV2FiveMinuteError("legacy_v2_5m_active_dataset_missing")
    manifest_path = dataset_manifest_path(root, _V2_DOMAIN, dataset_id)
    if not manifest_path.exists():
        raise FileNotFoundError(
            f"legacy_v2_5m_active_dataset_manifest_missing:{manifest_path}"
        )
    manifest_sha_before = sha256_file(manifest_path)
    manifest = read_dataset_manifest(manifest_path)
    manifest_sha_after = sha256_file(manifest_path)
    if manifest_sha_before != manifest_sha_after:
        raise LegacyV2FiveMinuteError("legacy_v2_5m_manifest_changed_while_reading")
    if manifest.dataset_id != dataset_id or manifest.domain != _V2_DOMAIN:
        raise LegacyV2FiveMinuteError(
            "legacy_v2_5m_manifest_identity_mismatch:"
            f"active={dataset_id}:manifest={manifest.dataset_id}:{manifest.domain}"
        )
    if str(manifest.frequency or "").lower() not in {"", "5m"}:
        raise LegacyV2FiveMinuteError(
            f"legacy_v2_5m_frequency_invalid:{manifest.frequency}"
        )
    if tuple(manifest.primary_key) != _EXPECTED_PRIMARY_KEY:
        raise LegacyV2FiveMinuteError(
            "legacy_v2_5m_primary_key_contract_invalid:"
            f"{tuple(manifest.primary_key)}"
        )
    schema_columns = {
        str(item.get("name", "") or "") for item in manifest.schema
    }
    if schema_columns:
        missing_columns = sorted(set(_REQUIRED_COLUMNS).difference(schema_columns))
        if missing_columns:
            raise LegacyV2FiveMinuteError(
                f"legacy_v2_5m_manifest_columns_missing:{missing_columns}"
            )
    if not manifest.shards:
        raise LegacyV2FiveMinuteError("legacy_v2_5m_manifest_shards_empty")

    shards: list[LegacyV2ShardEvidence] = []
    seen_paths: set[str] = set()
    for entry in manifest.shards:
        if entry.status not in {"", "stored"}:
            raise LegacyV2FiveMinuteError(
                f"legacy_v2_5m_shard_status_invalid:{entry.path}:{entry.status}"
            )
        resolved = resolve_manifest_path(entry.path, root=root).resolve()
        key = _path_key(resolved)
        if key in seen_paths:
            raise LegacyV2FiveMinuteError(
                f"legacy_v2_5m_manifest_duplicate_shard:{entry.path}"
            )
        seen_paths.add(key)
        if not resolved.is_file():
            raise FileNotFoundError(f"legacy_v2_5m_shard_missing:{resolved}")
        if resolved.suffix.lower() != ".parquet":
            raise LegacyV2FiveMinuteError(
                f"legacy_v2_5m_shard_not_parquet:{resolved}"
            )
        stat = resolved.stat()
        if int(entry.file_size or 0) and int(entry.file_size) != int(stat.st_size):
            raise LegacyV2FiveMinuteError(
                "legacy_v2_5m_shard_size_mismatch:"
                f"{entry.path}:manifest={entry.file_size}:actual={stat.st_size}"
            )
        shards.append(
            LegacyV2ShardEvidence(
                manifest_path=str(entry.path),
                resolved_path=resolved,
                row_count=int(entry.row_count),
                start_date=str(entry.start_date),
                end_date=str(entry.end_date),
                file_size=int(entry.file_size),
                size=int(stat.st_size),
                modified_ns=int(stat.st_mtime_ns),
            )
        )
    manifest_rows = sum(max(0, item.row_count) for item in shards)
    if manifest.row_count and manifest_rows and manifest_rows != int(manifest.row_count):
        raise LegacyV2FiveMinuteError(
            "legacy_v2_5m_manifest_row_count_sum_mismatch:"
            f"manifest={manifest.row_count}:shards={manifest_rows}"
        )
    parent_ids = _manifest_parent_dataset_ids(manifest)
    missing_parent_ids = tuple(
        sorted(
            parent_id
            for parent_id in parent_ids
            if not any((root / "datasets").glob(f"*/{parent_id}/dataset.json"))
        )
    )
    return LegacyV2Snapshot(
        root=root,
        active_path=active_path,
        active_sha256=active_sha_after,
        manifest_path=manifest_path,
        manifest_sha256=manifest_sha_after,
        manifest=manifest,
        shards=tuple(shards),
        missing_parent_dataset_ids=missing_parent_ids,
    )


def _fan_out_active_shards(
    snapshot: LegacyV2Snapshot,
    *,
    staging_root: Path,
    requested_start: str,
    requested_end: str,
    requested_symbols: Sequence[str],
    workers: int,
    destination: Path,
    min_available_gib: float,
    low_memory_seconds: float,
    min_free_disk_gib: float,
    min_staging_free_disk_gib: float,
) -> tuple[str, ...]:
    marker_path = staging_root / "fanout.json"
    expected_marker = {
        "v2_active_sha256": snapshot.active_sha256,
        "v2_dataset_manifest_sha256": snapshot.manifest_sha256,
        "v2_shard_stat_fingerprint": snapshot.shard_stat_fingerprint,
        "requested_start_date": requested_start,
        "requested_end_date": requested_end,
        "requested_symbols": list(requested_symbols),
    }
    marker = read_json(marker_path)
    if marker and all(marker.get(key) == value for key, value in expected_marker.items()):
        symbols = _staged_symbols(staging_root)
        if tuple(marker.get("symbols", ())) == symbols:
            return symbols
    if staging_root.exists():
        _safe_remove_staging(staging_root, staging_parent=staging_root.parent)
    staging_root.parent.mkdir(parents=True, exist_ok=True)
    fanout_target = staging_root / "fanout"
    # DuckDB creates the target itself. Keeping it absent also makes partial
    # COPY output distinguishable from a marker-backed resumable fan-out.
    shard_sql = ",".join(_sql_literal(str(item.resolved_path)) for item in snapshot.shards)
    where: list[str] = []
    if requested_start:
        where.append(
            f"CAST(trade_date AS VARCHAR) >= {_sql_literal(requested_start)}"
        )
    if requested_end:
        where.append(
            f"CAST(trade_date AS VARCHAR) <= {_sql_literal(requested_end)}"
        )
    if requested_symbols:
        requested_sql = ",".join(_sql_literal(item) for item in requested_symbols)
        where.append(
            "UPPER(TRIM(CAST(symbol AS VARCHAR))) "
            f"IN ({requested_sql})"
        )
    where_sql = " WHERE " + " AND ".join(where) if where else ""
    query = f"""
        COPY (
            SELECT
                UPPER(TRIM(CAST(symbol AS VARCHAR))) AS provider_symbol,
                CAST(trade_date AS VARCHAR) AS trade_date,
                CAST(bar_time AS VARCHAR) AS bar_time,
                CAST(open AS DOUBLE) AS open,
                CAST(high AS DOUBLE) AS high,
                CAST(low AS DOUBLE) AS low,
                CAST(close AS DOUBLE) AS close,
                CAST(volume AS DOUBLE) AS volume,
                CAST(amount AS DOUBLE) AS amount,
                CAST(adjusted_flag AS VARCHAR) AS adjusted_flag,
                CAST(filename AS VARCHAR) AS _v2_shard_path
            FROM read_parquet([{shard_sql}], union_by_name=true, filename=true)
            {where_sql}
        ) TO {_sql_literal(str(fanout_target))}
        (FORMAT PARQUET, COMPRESSION ZSTD, PARTITION_BY (provider_symbol))
    """
    staging_root.mkdir(parents=True, exist_ok=True)
    connection = duckdb.connect(database=":memory:")
    connection.execute(f"PRAGMA threads={max(1, min(int(workers), MAX_WORKERS))}")
    # DuckDB otherwise defaults to roughly 80% of total physical memory, which
    # can violate the user's 0.5 GiB system safety line when other applications
    # are active. Use the currently available budget minus the safety reserve,
    # and spill only inside the explicitly guarded runtime staging tree.
    available_gib = _available_memory_bytes() / float(1024**3)
    reserve_gib = max(0.5, float(min_available_gib)) + 0.25
    duckdb_memory_gib = max(0.25, available_gib - reserve_gib)
    duckdb_temp = staging_root / "duckdb_temp"
    duckdb_temp.mkdir(parents=True, exist_ok=True)
    connection.execute(f"SET memory_limit={_sql_literal(f'{duckdb_memory_gib:.3f}GB')}")
    connection.execute(f"SET temp_directory={_sql_literal(str(duckdb_temp))}")
    failure: list[BaseException] = []

    def execute_copy() -> None:
        try:
            connection.execute(query)
        except BaseException as exc:  # forwarded to the controlling thread
            failure.append(exc)

    thread = threading.Thread(target=execute_copy, name="qdp-v2-5m-fanout")
    raw_watchdog = _ResourceWatchdog(
        destination=destination,
        min_available_gib=float(min_available_gib),
        low_memory_seconds=float(low_memory_seconds),
        min_free_disk_gib=float(min_free_disk_gib),
    )
    staging_watchdog = _ResourceWatchdog(
        destination=staging_root,
        min_available_gib=float(min_available_gib),
        low_memory_seconds=float(low_memory_seconds),
        min_free_disk_gib=float(min_staging_free_disk_gib),
    )
    thread.start()
    guard_error: LegacyV2FiveMinuteResourceGuardError | None = None
    try:
        while thread.is_alive():
            thread.join(timeout=1.0)
            try:
                raw_watchdog.poll()
                staging_watchdog.poll()
            except LegacyV2FiveMinuteResourceGuardError as exc:
                guard_error = exc
                connection.interrupt()
                break
        thread.join()
    finally:
        connection.close()
    if guard_error is not None:
        raise guard_error
    if failure:
        raise LegacyV2FiveMinuteError(
            f"legacy_v2_5m_duckdb_fanout_failed:{type(failure[0]).__name__}:{failure[0]}"
        ) from failure[0]
    _verify_snapshot_unchanged(snapshot)
    symbols = _staged_symbols(staging_root)
    atomic_write_json(
        marker_path,
        {
            "fanout_version": 1,
            **expected_marker,
            "symbols": list(symbols),
            "created_at": utc_now(),
        },
    )
    return symbols


def _staged_symbols(staging_root: Path) -> tuple[str, ...]:
    fanout = staging_root / "fanout"
    if not fanout.exists():
        return ()
    symbols: list[str] = []
    for directory in sorted(fanout.glob("provider_symbol=*")):
        if not directory.is_dir() or not list(directory.glob("*.parquet")):
            continue
        raw_value = directory.name.split("=", 1)[1]
        symbol = _validated_symbol(raw_value)
        if directory.name != f"provider_symbol={symbol}":
            raise LegacyV2FiveMinuteError(
                f"legacy_v2_5m_staging_partition_not_canonical:{directory.name}"
            )
        symbols.append(symbol)
    if len(symbols) != len(set(symbols)):
        raise LegacyV2FiveMinuteError("legacy_v2_5m_staging_symbol_duplicate")
    return tuple(sorted(symbols))


def _migrate_staged_symbol(
    *,
    provider_symbol: str,
    stage_directory: Path,
    snapshot: LegacyV2Snapshot,
    shard_by_key: Mapping[str, LegacyV2ShardEvidence],
    raw_domain: str,
    requested_start: str,
    requested_end: str,
    workspace_root: str | Path | None,
) -> LegacyV2FiveMinuteSymbolResult:
    actual_stage_directory = stage_directory.parent / "fanout" / stage_directory.name
    parquet_files = sorted(actual_stage_directory.glob("*.parquet"))
    if not parquet_files:
        raise LegacyV2FiveMinuteError(
            f"legacy_v2_5m_staging_symbol_missing:{provider_symbol}"
        )
    parts = [pd.read_parquet(path, engine="pyarrow") for path in parquet_files]
    staged = pd.concat(parts, ignore_index=True) if len(parts) > 1 else parts[0]
    # DuckDB omits partition columns from the physical Parquet payload.  The
    # validated hive directory is therefore the authoritative value here.
    if "provider_symbol" not in staged.columns:
        staged["provider_symbol"] = provider_symbol
    required = set(_REQUIRED_COLUMNS).difference({"symbol"}) | {"provider_symbol", "_v2_shard_path"}
    missing = sorted(required.difference(staged.columns))
    if missing:
        raise LegacyV2FiveMinuteError(
            f"legacy_v2_5m_staging_columns_missing:{provider_symbol}:{missing}"
        )
    staged_symbols = {
        _validated_symbol(item)
        for item in staged["provider_symbol"].dropna().astype(str).unique()
    }
    if staged_symbols != {provider_symbol}:
        raise LegacyV2FiveMinuteError(
            f"legacy_v2_5m_staging_symbol_mismatch:{provider_symbol}:{sorted(staged_symbols)}"
        )
    evidence = _relevant_shard_evidence(staged, shard_by_key=shard_by_key)
    complete, complete_dates, rejected = _validated_complete_days(
        staged,
        provider_symbol=provider_symbol,
    )
    rejected_dates = tuple(sorted(rejected))
    if complete.empty:
        return LegacyV2FiveMinuteSymbolResult(
            provider_symbol=provider_symbol,
            status="empty",
            rejected_day_count=len(rejected_dates),
            rejected_trade_dates=rejected_dates,
        )
    migration_fingerprint = stable_hash(
        {
            "provider_symbol": provider_symbol,
            "v2_active_sha256": snapshot.active_sha256,
            "v2_dataset_manifest_sha256": snapshot.manifest_sha256,
            "requested_start_date": requested_start,
            "requested_end_date": requested_end,
            "relevant_shards": evidence,
            "complete_trade_dates": list(complete_dates),
            "rejected_trade_dates": list(rejected_dates),
        }
    )
    previous = get_raw_partition(
        raw_domain,
        partition_field="provider_symbol",
        partition_value=provider_symbol,
        workspace_root=workspace_root,
    )
    if previous is not None and previous.quality_tier == QUALITY_STRICT:
        previous_receipt = read_raw_receipt(previous)
        if (
            str(previous_receipt.get("migration_fingerprint", ""))
            == migration_fingerprint
            and int(previous.row_count) == len(complete)
        ):
            return LegacyV2FiveMinuteSymbolResult(
                provider_symbol=provider_symbol,
                status="reused",
                row_count=int(previous.row_count),
                day_count=len(complete_dates),
                rejected_day_count=len(rejected_dates),
                created=False,
                ref=previous,
                rejected_trade_dates=rejected_dates,
            )
    day_results = {
        trade_date: {"status": "strict", "bar_count": 48}
        for trade_date in complete_dates
    }
    day_results.update(
        {
            trade_date: {"status": "rejected", "reasons": list(reasons)}
            for trade_date, reasons in sorted(rejected.items())
        }
    )
    receipt = {
        "provider": SOURCE_NAME,
        "production_role": "trusted_legacy_migration",
        "upstream_provenance": "qdp_v2_active_immutable_leaf",
        "quality_tier": QUALITY_STRICT,
        "provider_symbol": provider_symbol,
        "requested_start_date": requested_start,
        "requested_end_date": requested_end,
        "coverage_start_date": complete_dates[0],
        "coverage_end_date": complete_dates[-1],
        "complete_stock_day_count": len(complete_dates),
        "rejected_stock_day_count": len(rejected_dates),
        "complete_trade_dates": list(complete_dates),
        "rejected_trade_dates": list(rejected_dates),
        "rejected_day_reasons": {
            key: list(value) for key, value in sorted(rejected.items())
        },
        "day_results": day_results,
        "v2_dataset_id": snapshot.manifest.dataset_id,
        "v2_active_manifest_path": str(snapshot.active_path),
        "v2_active_manifest_sha256": snapshot.active_sha256,
        "v2_dataset_manifest_path": str(snapshot.manifest_path),
        "v2_dataset_manifest_sha256": snapshot.manifest_sha256,
        "v2_relevant_shards": evidence,
        "missing_parent_lineage": bool(snapshot.missing_parent_dataset_ids),
        "missing_parent_dataset_ids": list(snapshot.missing_parent_dataset_ids),
        "migration_fingerprint": migration_fingerprint,
        "adjusted_flag_contract": "none_only",
        "primary_key": ["provider_symbol", "trade_date", "bar_time"],
        "bar_contract": "right_closed_5m_48_0935_1500",
        "rejection_policy": "exclude_invalid_stock_day_keep_v2_source_immutable",
        "normalized_row_count": int(len(complete)),
    }
    ref, created = write_raw_partition(
        raw_domain=raw_domain,
        partition_field="provider_symbol",
        partition_value=provider_symbol,
        frame=complete,
        receipt=receipt,
        workspace_root=workspace_root,
    )
    return LegacyV2FiveMinuteSymbolResult(
        provider_symbol=provider_symbol,
        status="created" if created else "reused",
        row_count=int(len(complete)),
        day_count=len(complete_dates),
        rejected_day_count=len(rejected_dates),
        created=bool(created),
        ref=ref,
        rejected_trade_dates=rejected_dates,
    )


def _validated_complete_days(
    staged: pd.DataFrame,
    *,
    provider_symbol: str,
) -> tuple[pd.DataFrame, tuple[str, ...], dict[str, tuple[str, ...]]]:
    data = staged.copy()
    parsed_dates = pd.to_datetime(data["trade_date"], errors="coerce")
    if parsed_dates.isna().any():
        raise LegacyV2FiveMinuteError(
            f"legacy_v2_5m_invalid_trade_date:{provider_symbol}:"
            f"count={int(parsed_dates.isna().sum())}"
        )
    data["trade_date"] = parsed_dates.dt.strftime("%Y-%m-%d")
    data["bar_time"] = data["bar_time"].map(_normalize_bar_time)
    for column in _NUMERIC_COLUMNS:
        data[column] = pd.to_numeric(data[column], errors="coerce")
    data["adjusted_flag"] = data["adjusted_flag"].astype("string")
    expected = set(EXPECTED_5M_BAR_ENDS)
    numeric = data.loc[:, list(_NUMERIC_COLUMNS)].to_numpy(dtype="float64")
    data["_pk_duplicate"] = data.duplicated(
        ["trade_date", "bar_time"], keep=False
    )
    data["_time_expected"] = data["bar_time"].isin(expected)
    adjusted = data["adjusted_flag"].str.strip().str.lower()
    data["_adjusted_none"] = adjusted.notna() & adjusted.eq("none")
    data["_numeric_finite"] = np.isfinite(numeric).all(axis=1)
    data["_price_positive"] = data.loc[:, list(_PRICE_COLUMNS)].gt(0).all(axis=1)
    data["_volume_amount_nonnegative"] = data[["volume", "amount"]].ge(0).all(axis=1)
    data["_high_valid"] = data["high"].ge(
        data[["open", "low", "close"]].max(axis=1)
    )
    data["_low_valid"] = data["low"].le(
        data[["open", "high", "close"]].min(axis=1)
    )
    # This is the hot path for roughly 4,000 dates per security.  Aggregate
    # every stock-day rule in one groupby instead of constructing thousands
    # of tiny DataFrames in a Python loop.
    day_metrics = data.groupby("trade_date", sort=True, dropna=False).agg(
        row_count=("bar_time", "size"),
        unique_bar_times=("bar_time", "nunique"),
        has_pk_duplicate=("_pk_duplicate", "any"),
        all_times_expected=("_time_expected", "all"),
        all_adjusted_none=("_adjusted_none", "all"),
        all_numeric_finite=("_numeric_finite", "all"),
        all_prices_positive=("_price_positive", "all"),
        all_volume_amount_nonnegative=("_volume_amount_nonnegative", "all"),
        all_high_valid=("_high_valid", "all"),
        all_low_valid=("_low_valid", "all"),
    )
    exact_time_set = (
        day_metrics["row_count"].eq(48)
        & day_metrics["unique_bar_times"].eq(48)
        & day_metrics["all_times_expected"]
    )
    valid_day = (
        ~day_metrics["has_pk_duplicate"]
        & exact_time_set
        & day_metrics["all_adjusted_none"]
        & day_metrics["all_numeric_finite"]
        & day_metrics["all_prices_positive"]
        & day_metrics["all_volume_amount_nonnegative"]
        & day_metrics["all_high_valid"]
        & day_metrics["all_low_valid"]
    )
    complete_dates = tuple(
        str(item) for item in day_metrics.index[valid_day].tolist()
    )
    rejected: dict[str, tuple[str, ...]] = {}
    # Rejections are intentionally rare. Build human-readable reasons only
    # for those dates; complete dates never cross the Python iteration boundary.
    for trade_date, metrics in day_metrics.loc[~valid_day].iterrows():
        reasons: list[str] = []
        if bool(metrics["has_pk_duplicate"]):
            reasons.append("primary_key_duplicate")
        if not bool(exact_time_set.loc[trade_date]):
            reasons.append("bar_time_set_not_exact_48")
        if not bool(metrics["all_adjusted_none"]):
            reasons.append("adjusted_flag_not_none")
        if not bool(metrics["all_numeric_finite"]):
            reasons.append("numeric_non_finite")
        else:
            if not bool(metrics["all_prices_positive"]):
                reasons.append("price_not_positive")
            if not bool(metrics["all_volume_amount_nonnegative"]):
                reasons.append("volume_or_amount_negative")
            if not bool(metrics["all_high_valid"]):
                reasons.append("high_relation_invalid")
            if not bool(metrics["all_low_valid"]):
                reasons.append("low_relation_invalid")
        rejected[str(trade_date)] = tuple(reasons)
    if complete_dates:
        complete = data.loc[data["trade_date"].isin(complete_dates)].copy()
        complete["provider_symbol"] = provider_symbol
        complete["source"] = SOURCE_NAME
        complete = complete.loc[:, RAW_COLUMNS]
        complete = complete.sort_values(
            ["trade_date", "bar_time"], kind="stable"
        ).reset_index(drop=True)
        for column in _NUMERIC_COLUMNS:
            complete[column] = complete[column].astype("float64")
        complete["provider_symbol"] = complete["provider_symbol"].astype("string")
        complete["trade_date"] = complete["trade_date"].astype("string")
        complete["bar_time"] = complete["bar_time"].astype("string")
        complete["source"] = complete["source"].astype("string")
    else:
        complete = _empty_raw_frame()
    return complete, complete_dates, rejected


def _relevant_shard_evidence(
    staged: pd.DataFrame,
    *,
    shard_by_key: Mapping[str, LegacyV2ShardEvidence],
) -> list[dict[str, Any]]:
    evidence: list[dict[str, Any]] = []
    paths = sorted(set(staged["_v2_shard_path"].dropna().astype(str)))
    if not paths:
        raise LegacyV2FiveMinuteError("legacy_v2_5m_staging_source_path_missing")
    for raw_path in paths:
        key = _path_key(Path(raw_path).resolve())
        shard = shard_by_key.get(key)
        if shard is None:
            raise LegacyV2FiveMinuteError(
                f"legacy_v2_5m_unlisted_shard_in_staging:{raw_path}"
            )
        _verify_shard_stat(shard)
        digest = _cached_sha256(
            str(shard.resolved_path), shard.size, shard.modified_ns
        )
        _verify_shard_stat(shard)
        evidence.append(
            {
                "manifest_path": shard.manifest_path,
                "resolved_path": str(shard.resolved_path),
                "sha256": digest,
                "manifest_row_count": int(shard.row_count),
                "manifest_start_date": shard.start_date,
                "manifest_end_date": shard.end_date,
                "file_size": int(shard.size),
            }
        )
    return evidence


def _completed_job_result(
    job: Mapping[str, Any],
    *,
    snapshot: LegacyV2Snapshot,
    selection_fingerprint: str,
    raw_domain: str,
    workspace_root: str | Path | None,
) -> LegacyV2FiveMinuteMigrationResult | None:
    if str(job.get("status", "")) not in {"completed", "completed_with_errors"}:
        return None
    if str(job.get("selection_fingerprint", "")) != selection_fingerprint:
        return None
    if str(job.get("v2_shard_stat_fingerprint", "")) != snapshot.shard_stat_fingerprint:
        return None
    result_payload = dict(job.get("result", {}) or {})
    rows = list(result_payload.get("results", ()) or job.get("results", ()) or ())
    results: list[LegacyV2FiveMinuteSymbolResult] = []
    for payload in rows:
        if not isinstance(payload, Mapping):
            return None
        symbol = _validated_symbol(str(payload.get("provider_symbol", "")))
        content_sha = str(payload.get("content_sha256", "") or "")
        ref: RawPartitionRef | None = None
        if content_sha:
            ref = get_raw_partition(
                raw_domain,
                partition_field="provider_symbol",
                partition_value=symbol,
                workspace_root=workspace_root,
            )
            if ref is None or ref.content_sha256 != content_sha:
                return None
        results.append(
            LegacyV2FiveMinuteSymbolResult(
                provider_symbol=symbol,
                status="reused" if ref is not None else str(payload.get("status", "") or "empty"),
                row_count=int(payload.get("row_count", 0) or 0),
                day_count=int(payload.get("day_count", 0) or 0),
                rejected_day_count=int(payload.get("rejected_day_count", 0) or 0),
                created=False,
                ref=ref,
                rejected_trade_dates=tuple(
                    str(item)
                    for item in list(payload.get("rejected_trade_dates", ()) or ())
                ),
                error=str(payload.get("error", "") or ""),
            )
        )
    _verify_snapshot_unchanged(snapshot)
    return _migration_result(
        status=str(result_payload.get("status", "") or job.get("status", "")),
        raw_domain=raw_domain,
        v2_dataset_id=snapshot.manifest.dataset_id,
        discovered_symbols=int(result_payload.get("discovered_symbols", len(results)) or 0),
        selected_symbols=int(result_payload.get("selected_symbols", len(results)) or 0),
        results=results,
    )


def _migration_result(
    *,
    status: str,
    raw_domain: str,
    v2_dataset_id: str,
    discovered_symbols: int,
    selected_symbols: int,
    results: Sequence[LegacyV2FiveMinuteSymbolResult],
) -> LegacyV2FiveMinuteMigrationResult:
    completed = sum(item.status != "failed" for item in results)
    return LegacyV2FiveMinuteMigrationResult(
        status=status,
        raw_domain=raw_domain,
        v2_dataset_id=v2_dataset_id,
        discovered_symbols=int(discovered_symbols),
        selected_symbols=int(selected_symbols),
        completed_symbols=int(completed),
        created_symbols=sum(item.status == "created" for item in results),
        reused_symbols=sum(item.status == "reused" for item in results),
        empty_symbols=sum(item.status == "empty" for item in results),
        failed_symbols=sum(item.status == "failed" for item in results),
        row_count=sum(int(item.row_count) for item in results),
        day_count=sum(int(item.day_count) for item in results),
        rejected_day_count=sum(int(item.rejected_day_count) for item in results),
        results=tuple(sorted(results, key=lambda item: item.provider_symbol)),
    )


def _record_progress_outcome(
    progress: dict[str, Any],
    outcome: LegacyV2FiveMinuteSymbolResult,
    *,
    lock: threading.RLock,
) -> None:
    with lock:
        progress["last_symbol"] = outcome.provider_symbol
        progress["processed"] = int(progress["processed"]) + 1
        progress["pending"] = max(
            0, int(progress["total"]) - int(progress["processed"])
        )
        if outcome.status != "failed":
            progress["completed"] = int(progress["completed"]) + 1
        progress[outcome.status] = int(progress.get(outcome.status, 0)) + 1
        progress["row_count"] = int(progress["row_count"]) + int(outcome.row_count)
        progress["day_count"] = int(progress["day_count"]) + int(outcome.day_count)
        progress["rejected_day_count"] = int(progress["rejected_day_count"]) + int(
            outcome.rejected_day_count
        )
        progress["results"].append(outcome.to_dict())


def _verify_snapshot_unchanged(snapshot: LegacyV2Snapshot) -> None:
    if sha256_file(snapshot.active_path) != snapshot.active_sha256:
        raise LegacyV2FiveMinuteError("legacy_v2_5m_active_changed_during_migration")
    if sha256_file(snapshot.manifest_path) != snapshot.manifest_sha256:
        raise LegacyV2FiveMinuteError("legacy_v2_5m_manifest_changed_during_migration")
    for shard in snapshot.shards:
        _verify_shard_stat(shard)


def _verify_shard_stat(shard: LegacyV2ShardEvidence) -> None:
    if not shard.resolved_path.is_file():
        raise LegacyV2FiveMinuteError(
            f"legacy_v2_5m_shard_removed_during_migration:{shard.resolved_path}"
        )
    stat = shard.resolved_path.stat()
    if int(stat.st_size) != shard.size or int(stat.st_mtime_ns) != shard.modified_ns:
        raise LegacyV2FiveMinuteError(
            f"legacy_v2_5m_shard_changed_during_migration:{shard.resolved_path}"
        )


@lru_cache(maxsize=16_384)
def _cached_sha256(path: str, size: int, modified_ns: int) -> str:
    del size, modified_ns
    return sha256_file(path)


def _manifest_parent_dataset_ids(manifest: DatasetManifest) -> tuple[str, ...]:
    found: set[str] = set()

    def visit(value: Any, key: str = "") -> None:
        if isinstance(value, Mapping):
            for child_key, child_value in value.items():
                visit(child_value, str(child_key))
            return
        if isinstance(value, (list, tuple)):
            for child in value:
                visit(child, key)
            return
        if "dataset_id" in key.lower() and str(value or "").strip():
            candidate = str(value).strip()
            if candidate != manifest.dataset_id:
                found.add(candidate)

    visit(manifest.source)
    return tuple(sorted(found))


def _resource_guard(
    destination: Path,
    *,
    min_available_gib: float,
    low_memory_seconds: float,
    min_free_disk_gib: float,
) -> None:
    watchdog = _ResourceWatchdog(
        destination=destination,
        min_available_gib=float(min_available_gib),
        low_memory_seconds=float(low_memory_seconds),
        min_free_disk_gib=float(min_free_disk_gib),
    )
    watchdog.poll()
    if watchdog.low_memory_since is None:
        return
    deadline = time.monotonic() + max(0.0, float(low_memory_seconds))
    while time.monotonic() < deadline:
        time.sleep(min(0.25, max(0.0, deadline - time.monotonic())))
        watchdog.poll()
        if watchdog.low_memory_since is None:
            return
    watchdog.poll()


def _available_memory_bytes() -> int:
    try:
        import psutil  # type: ignore[import-not-found]

        return int(psutil.virtual_memory().available)
    except (ImportError, OSError):
        if os.name != "nt":
            return 2**63 - 1

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


def _safe_remove_staging(path: Path, *, staging_parent: Path) -> None:
    resolved = path.resolve()
    parent = staging_parent.resolve()
    if resolved.parent != parent or not resolved.name.startswith("legacy_v2_5m__"):
        raise LegacyV2FiveMinuteError(
            f"legacy_v2_5m_unsafe_staging_cleanup_refused:{resolved}"
        )
    if resolved.exists():
        shutil.rmtree(resolved)


def _normalize_bar_time(value: Any) -> str:
    text = str(value or "").strip()
    if text.endswith(".0") and text[:-2].isdigit():
        text = text[:-2]
    digits = re.sub(r"\D", "", text)
    if text.isdigit() and len(text) <= 9:
        digits = text.zfill(9)
    if len(digits) >= 4:
        hour = int(digits[:2])
        minute = int(digits[2:4])
        if 0 <= hour <= 23 and 0 <= minute <= 59:
            return f"{hour:02d}:{minute:02d}"
    match = re.search(r"(?<!\d)(\d{1,2}):(\d{2})(?::\d{2})?", text)
    if match:
        hour, minute = int(match.group(1)), int(match.group(2))
        if 0 <= hour <= 23 and 0 <= minute <= 59:
            return f"{hour:02d}:{minute:02d}"
    return ""


def _date_text(value: Any) -> str:
    parsed = pd.to_datetime(str(value), errors="coerce")
    if pd.isna(parsed):
        raise ValueError(f"legacy_v2_5m_invalid_date:{value}")
    return parsed.strftime("%Y-%m-%d")


def _validated_symbol(value: Any) -> str:
    symbol = normalize_symbol(str(value or "").strip())
    if not _SYMBOL_PATTERN.fullmatch(symbol):
        raise ValueError(f"legacy_v2_5m_invalid_symbol:{value}")
    return symbol


def _empty_raw_frame() -> pd.DataFrame:
    frame = pd.DataFrame(columns=RAW_COLUMNS)
    for column in ("provider_symbol", "trade_date", "bar_time", "source"):
        frame[column] = frame[column].astype("string")
    for column in _NUMERIC_COLUMNS:
        frame[column] = frame[column].astype("float64")
    return frame.loc[:, RAW_COLUMNS]


def _path_key(path: Path) -> str:
    return os.path.normcase(os.path.abspath(str(path)))


def _sql_literal(value: str) -> str:
    return "'" + str(value).replace("'", "''") + "'"


__all__ = [
    "DEFAULT_RAW_DOMAIN",
    "MAX_WORKERS",
    "SOURCE_NAME",
    "LegacyV2FiveMinuteError",
    "LegacyV2FiveMinuteMigrationResult",
    "LegacyV2FiveMinuteResourceGuardError",
    "LegacyV2FiveMinuteSymbolResult",
    "migrate_v2_active_intraday_5m",
]
