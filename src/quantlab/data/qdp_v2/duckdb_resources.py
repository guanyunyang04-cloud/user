from __future__ import annotations

"""Dynamic DuckDB resources and lightweight query profiling.

DuckDB still needs a finite buffer-manager limit, but a small fixed ceiling
wastes most of the machine. This module puts that internal ceiling at available
RAM minus the safety floor when the connection opens, then continuously
interrupts work when available memory remains below that floor.
"""

import hashlib
import json
import os
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import duckdb

GIB = 1024**3
MIB = 1024**2
DEFAULT_MEMORY_FLOOR_BYTES = int(0.5 * GIB)
DEFAULT_LOW_MEMORY_SECONDS = 2.0
DEFAULT_POLL_SECONDS = 0.25
DEFAULT_MINIMUM_LIMIT_BYTES = 64 * MIB
LOW_MEMORY_REASON = "available_memory_below_0.5_gib_for_2_seconds"


class DuckDbMemoryFloorError(RuntimeError):
    """Raised after DuckDB is interrupted by the sustained-memory guard."""


@dataclass(frozen=True)
class DuckDbResourceSettings:
    available_at_open_bytes: int
    total_physical_memory_bytes: int
    memory_limit_bytes: int
    memory_floor_bytes: int
    low_memory_seconds: float
    poll_seconds: float
    temp_directory: str
    threads: int | None
    profiling_enabled: bool
    profiling_path: str


class DuckDbQueryProfiler:
    """Collect bounded query timings without changing query semantics.

    DuckDB's native profiler is useful for one query at a time, but the minute
    builder executes a large number of short setup statements and COPYs.  A
    small JSONL-style aggregate gives us a stable audit trail for *all* of
    them while keeping the profiler out of the data contract.  The file is
    written atomically when the connection closes.
    """

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path).resolve()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._records: list[dict[str, Any]] = []
        self._lock = threading.Lock()
        self._started = time.time()

    def record(
        self,
        query: Any,
        *,
        elapsed_seconds: float,
        ok: bool,
        error: BaseException | None = None,
    ) -> None:
        text = str(query)
        compact = " ".join(text.split())
        record: dict[str, Any] = {
            "sequence": 0,
            "elapsed_seconds": float(max(0.0, elapsed_seconds)),
            "ok": bool(ok),
            "query_sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
            # Keep the profile useful in a text editor without duplicating
            # very large generated SQL strings.
            "query_preview": compact[:500],
        }
        if error is not None:
            record["error"] = type(error).__name__ + ":" + str(error)[:300]
        with self._lock:
            record["sequence"] = len(self._records) + 1
            self._records.append(record)

    def close(self) -> None:
        with self._lock:
            records = list(self._records)
        payload = {
            "schema": "quantlab.duckdb_query_profile/1",
            "started_unix_seconds": self._started,
            "query_count": len(records),
            "total_elapsed_seconds": sum(
                float(item["elapsed_seconds"]) for item in records
            ),
            "queries": records,
        }
        partial = self.path.with_suffix(self.path.suffix + ".partial")
        partial.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        os.replace(partial, self.path)


def available_memory_bytes() -> int:
    try:
        import psutil  # type: ignore
    except ImportError as exc:  # pragma: no cover - production dependency
        raise DuckDbMemoryFloorError("psutil_required_for_duckdb_memory_guard") from exc
    return int(psutil.virtual_memory().available)


def total_physical_memory_bytes() -> int:
    try:
        import psutil  # type: ignore
    except ImportError as exc:  # pragma: no cover - production dependency
        raise DuckDbMemoryFloorError("psutil_required_for_duckdb_memory_guard") from exc
    return int(psutil.virtual_memory().total)


def auto_thread_count(*, available_bytes: int, total_bytes: int | None = None) -> int:
    """Choose a conservative worker count from the current machine state.

    Explicit thread counts remain available for reproducible comparisons.  In
    ``auto`` mode we avoid starting one worker per logical Hyper-Thread when
    little RAM is free; roughly one worker per GiB available is a practical
    upper bound for the window-heavy minute queries.
    """

    del total_bytes  # reserved for future NUMA-aware policies
    logical = max(1, int(os.cpu_count() or 1))
    available_gib = max(1, int(max(0, int(available_bytes)) / GIB))
    return max(1, min(logical, available_gib))


def dynamic_memory_limit_bytes(
    *,
    available_bytes: int,
    total_bytes: int | None = None,
    floor_bytes: int = DEFAULT_MEMORY_FLOOR_BYTES,
    minimum_bytes: int = DEFAULT_MINIMUM_LIMIT_BYTES,
) -> int:
    """Use currently available RAM while reserving the physical safety floor.

    DuckDB's limit is not a complete process-memory limit, so the two-second
    available-memory watchdog remains the final authority after connection
    startup.
    """

    available = max(0, int(available_bytes))
    total = available if total_bytes is None else max(0, int(total_bytes))
    floor = max(0, int(floor_bytes))
    minimum = max(1, int(minimum_bytes))
    return max(minimum, min(available, total) - floor)


def configure_dynamic_duckdb(
    connection: Any,
    *,
    temp_directory: str | Path | None = None,
    threads: int | str | None = None,
    profiling_path: str | Path | None = None,
    memory_sampler: Callable[[], int] = available_memory_bytes,
    total_memory_sampler: Callable[[], int] = total_physical_memory_bytes,
    floor_bytes: int = DEFAULT_MEMORY_FLOOR_BYTES,
    low_memory_seconds: float = DEFAULT_LOW_MEMORY_SECONDS,
    poll_seconds: float = DEFAULT_POLL_SECONDS,
    minimum_limit_bytes: int = DEFAULT_MINIMUM_LIMIT_BYTES,
) -> DuckDbResourceSettings:
    available = int(memory_sampler())
    total = int(total_memory_sampler())
    limit = dynamic_memory_limit_bytes(
        available_bytes=available,
        total_bytes=total,
        floor_bytes=floor_bytes,
        minimum_bytes=minimum_limit_bytes,
    )
    resolved_temp = ""
    if temp_directory is not None:
        spill = Path(temp_directory).resolve()
        spill.mkdir(parents=True, exist_ok=True)
        resolved_temp = str(spill)
    if threads is None or (isinstance(threads, str) and threads.strip().lower() == "auto"):
        thread_count = auto_thread_count(available_bytes=available, total_bytes=total)
    else:
        thread_count = max(1, int(threads))

    connection.execute(f"SET memory_limit='{limit}B'")
    if thread_count is not None:
        connection.execute(f"SET threads={thread_count}")
    connection.execute("SET preserve_insertion_order=false")
    if resolved_temp:
        escaped = resolved_temp.replace("'", "''")
        connection.execute(f"SET temp_directory='{escaped}'")

    resolved_profile = ""
    if profiling_path is not None:
        profile = Path(profiling_path).resolve()
        profile.parent.mkdir(parents=True, exist_ok=True)
        resolved_profile = str(profile)

    return DuckDbResourceSettings(
        available_at_open_bytes=available,
        total_physical_memory_bytes=total,
        memory_limit_bytes=limit,
        memory_floor_bytes=max(0, int(floor_bytes)),
        low_memory_seconds=max(0.0, float(low_memory_seconds)),
        poll_seconds=max(0.01, float(poll_seconds)),
        temp_directory=resolved_temp,
        threads=thread_count,
        profiling_enabled=bool(resolved_profile),
        profiling_path=resolved_profile,
    )


class DuckDbMemoryWatchdog:
    """Interrupt one connection after a continuously breached memory floor."""

    def __init__(
        self,
        connection: Any,
        *,
        memory_sampler: Callable[[], int] = available_memory_bytes,
        clock: Callable[[], float] = time.monotonic,
        floor_bytes: int = DEFAULT_MEMORY_FLOOR_BYTES,
        low_memory_seconds: float = DEFAULT_LOW_MEMORY_SECONDS,
        poll_seconds: float = DEFAULT_POLL_SECONDS,
    ) -> None:
        self._connection = connection
        self._memory_sampler = memory_sampler
        self._clock = clock
        self._floor_bytes = max(0, int(floor_bytes))
        self._low_memory_seconds = max(0.0, float(low_memory_seconds))
        self._poll_seconds = max(0.01, float(poll_seconds))
        self._stop = threading.Event()
        self._triggered = threading.Event()
        self._thread: threading.Thread | None = None
        self._lock = threading.Lock()
        self._low_since: float | None = None
        self._last_available_bytes: int | None = None
        self._minimum_available_bytes: int | None = None

    @property
    def triggered(self) -> bool:
        return self._triggered.is_set()

    @property
    def last_available_bytes(self) -> int | None:
        with self._lock:
            return self._last_available_bytes

    @property
    def minimum_available_bytes(self) -> int | None:
        with self._lock:
            return self._minimum_available_bytes

    def sample_once(
        self,
        *,
        available_bytes: int | None = None,
        now: float | None = None,
    ) -> bool:
        """Observe one sample; exposed for deterministic safety tests."""

        available = int(
            self._memory_sampler() if available_bytes is None else available_bytes
        )
        observed_at = float(self._clock() if now is None else now)
        should_interrupt = False
        with self._lock:
            self._last_available_bytes = available
            if (
                self._minimum_available_bytes is None
                or available < self._minimum_available_bytes
            ):
                self._minimum_available_bytes = available
            if available >= self._floor_bytes:
                self._low_since = None
            elif self._low_since is None:
                self._low_since = observed_at
            elif (
                observed_at - self._low_since >= self._low_memory_seconds
                and not self._triggered.is_set()
            ):
                self._triggered.set()
                should_interrupt = True
        if should_interrupt:
            self._connection.interrupt()
        return self._triggered.is_set()

    def start(self) -> None:
        if self._thread is not None:
            return
        self._thread = threading.Thread(
            target=self._run,
            name="qdp-duckdb-memory-watchdog",
            daemon=True,
        )
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        thread = self._thread
        if thread is not None and thread is not threading.current_thread():
            thread.join(timeout=max(1.0, self._poll_seconds * 4.0))

    def _run(self) -> None:
        while not self._stop.is_set() and not self._triggered.is_set():
            try:
                self.sample_once()
            except BaseException:
                # A transient sampler failure must not disable future checks.
                pass
            self._stop.wait(self._poll_seconds)


class GuardedDuckDbConnection:
    """Small proxy that owns the watchdog and translates its interruption."""

    def __init__(
        self,
        connection: Any,
        *,
        settings: DuckDbResourceSettings,
        memory_sampler: Callable[[], int] = available_memory_bytes,
        clock: Callable[[], float] = time.monotonic,
        error_factory: Callable[[str], BaseException] = DuckDbMemoryFloorError,
        profiler: DuckDbQueryProfiler | None = None,
    ) -> None:
        self._connection = connection
        self.settings = settings
        self._error_factory = error_factory
        self.profiler = profiler
        self.watchdog = DuckDbMemoryWatchdog(
            connection,
            memory_sampler=memory_sampler,
            clock=clock,
            floor_bytes=settings.memory_floor_bytes,
            low_memory_seconds=settings.low_memory_seconds,
            poll_seconds=settings.poll_seconds,
        )
        self._closed = False
        self.watchdog.start()

    def execute(self, *args: Any, **kwargs: Any) -> Any:
        self._raise_if_triggered()
        started = time.perf_counter()
        try:
            result = self._connection.execute(*args, **kwargs)
        except BaseException as exc:
            if self.profiler is not None:
                self.profiler.record(
                    args[0] if args else "",
                    elapsed_seconds=time.perf_counter() - started,
                    ok=False,
                    error=exc,
                )
            if self.watchdog.triggered:
                raise self._error_factory(LOW_MEMORY_REASON) from exc
            raise
        if self.profiler is not None:
            self.profiler.record(
                args[0] if args else "",
                elapsed_seconds=time.perf_counter() - started,
                ok=True,
            )
        self._raise_if_triggered()
        return result

    def executemany(self, *args: Any, **kwargs: Any) -> Any:
        self._raise_if_triggered()
        started = time.perf_counter()
        try:
            result = self._connection.executemany(*args, **kwargs)
        except BaseException as exc:
            if self.profiler is not None:
                self.profiler.record(
                    args[0] if args else "",
                    elapsed_seconds=time.perf_counter() - started,
                    ok=False,
                    error=exc,
                )
            if self.watchdog.triggered:
                raise self._error_factory(LOW_MEMORY_REASON) from exc
            raise
        if self.profiler is not None:
            self.profiler.record(
                args[0] if args else "",
                elapsed_seconds=time.perf_counter() - started,
                ok=True,
            )
        self._raise_if_triggered()
        return result

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        self.watchdog.stop()
        try:
            self._connection.close()
        finally:
            if self.profiler is not None:
                self.profiler.close()

    def __enter__(self) -> GuardedDuckDbConnection:
        return self

    def __exit__(self, exc_type: Any, exc: BaseException | None, traceback: Any) -> bool:
        triggered = self.watchdog.triggered
        self.close()
        if triggered and not isinstance(exc, DuckDbMemoryFloorError):
            raise self._error_factory(LOW_MEMORY_REASON) from exc
        return False

    def __getattr__(self, name: str) -> Any:
        return getattr(self._connection, name)

    def _raise_if_triggered(self) -> None:
        if self.watchdog.triggered:
            raise self._error_factory(LOW_MEMORY_REASON)


def guard_configured_duckdb(
    connection: Any,
    *,
    settings: DuckDbResourceSettings,
    memory_sampler: Callable[[], int] = available_memory_bytes,
    clock: Callable[[], float] = time.monotonic,
    error_factory: Callable[[str], BaseException] = DuckDbMemoryFloorError,
    profiler: DuckDbQueryProfiler | None = None,
) -> GuardedDuckDbConnection:
    return GuardedDuckDbConnection(
        connection,
        settings=settings,
        memory_sampler=memory_sampler,
        clock=clock,
        error_factory=error_factory,
        profiler=profiler,
    )


def open_guarded_duckdb(
    database: str | Path = ":memory:",
    *,
    temp_directory: str | Path | None = None,
    threads: int | str | None = None,
    profiling_path: str | Path | None = None,
    memory_sampler: Callable[[], int] = available_memory_bytes,
    total_memory_sampler: Callable[[], int] = total_physical_memory_bytes,
    clock: Callable[[], float] = time.monotonic,
    error_factory: Callable[[str], BaseException] = DuckDbMemoryFloorError,
    floor_bytes: int = DEFAULT_MEMORY_FLOOR_BYTES,
    low_memory_seconds: float = DEFAULT_LOW_MEMORY_SECONDS,
    poll_seconds: float = DEFAULT_POLL_SECONDS,
    minimum_limit_bytes: int = DEFAULT_MINIMUM_LIMIT_BYTES,
) -> GuardedDuckDbConnection:
    connection = duckdb.connect(str(database))
    try:
        settings = configure_dynamic_duckdb(
            connection,
            temp_directory=temp_directory,
            threads=threads,
            memory_sampler=memory_sampler,
            total_memory_sampler=total_memory_sampler,
            floor_bytes=floor_bytes,
            low_memory_seconds=low_memory_seconds,
            poll_seconds=poll_seconds,
            minimum_limit_bytes=minimum_limit_bytes,
            profiling_path=profiling_path,
        )
        profiler = DuckDbQueryProfiler(settings.profiling_path) if settings.profiling_enabled else None
        return guard_configured_duckdb(
            connection,
            settings=settings,
            memory_sampler=memory_sampler,
            clock=clock,
            error_factory=error_factory,
            profiler=profiler,
        )
    except BaseException:
        connection.close()
        raise


__all__ = [
    "DEFAULT_LOW_MEMORY_SECONDS",
    "DEFAULT_MEMORY_FLOOR_BYTES",
    "DEFAULT_MINIMUM_LIMIT_BYTES",
    "DEFAULT_POLL_SECONDS",
    "DuckDbMemoryFloorError",
    "DuckDbQueryProfiler",
    "DuckDbMemoryWatchdog",
    "DuckDbResourceSettings",
    "GuardedDuckDbConnection",
    "LOW_MEMORY_REASON",
    "available_memory_bytes",
    "auto_thread_count",
    "configure_dynamic_duckdb",
    "dynamic_memory_limit_bytes",
    "guard_configured_duckdb",
    "open_guarded_duckdb",
    "total_physical_memory_bytes",
]
