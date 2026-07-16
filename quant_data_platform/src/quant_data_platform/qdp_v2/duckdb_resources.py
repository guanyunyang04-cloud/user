from __future__ import annotations

"""Dynamic DuckDB memory control for repository-local QDP work.

DuckDB still needs a finite buffer-manager limit, but a small fixed ceiling
wastes most of the machine.  This module puts that internal ceiling at physical
RAM minus the safety floor, then continuously interrupts work only when actual
available memory remains below the configured free-memory floor.
"""

import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import duckdb


GIB = 1024**3
MIB = 1024**2
DEFAULT_MEMORY_FLOOR_BYTES = int(0.5 * GIB)
DEFAULT_LOW_MEMORY_SECONDS = 5.0
DEFAULT_POLL_SECONDS = 0.25
DEFAULT_MINIMUM_LIMIT_BYTES = 64 * MIB
LOW_MEMORY_REASON = "available_memory_below_0.5_gib_for_5_seconds"


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


def dynamic_memory_limit_bytes(
    *,
    available_bytes: int,
    total_bytes: int | None = None,
    floor_bytes: int = DEFAULT_MEMORY_FLOOR_BYTES,
    minimum_bytes: int = DEFAULT_MINIMUM_LIMIT_BYTES,
) -> int:
    """Keep DuckDB's internal ceiling from binding before the safety floor.

    The ceiling is derived from physical RAM, not from free RAM at connection
    open, because DuckDB's internal accounting can exceed its resident working
    set.  The actual-available-memory watchdog is the final authority and
    interrupts after five continuously low seconds.
    """

    available = max(0, int(available_bytes))
    total = available if total_bytes is None else max(0, int(total_bytes))
    floor = max(0, int(floor_bytes))
    minimum = max(1, int(minimum_bytes))
    return max(minimum, total - floor)


def configure_dynamic_duckdb(
    connection: Any,
    *,
    temp_directory: str | Path | None = None,
    threads: int | None = None,
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
    thread_count = None if threads is None else max(1, int(threads))

    connection.execute(f"SET memory_limit='{limit}B'")
    if thread_count is not None:
        connection.execute(f"SET threads={thread_count}")
    connection.execute("SET preserve_insertion_order=false")
    if resolved_temp:
        escaped = resolved_temp.replace("'", "''")
        connection.execute(f"SET temp_directory='{escaped}'")

    return DuckDbResourceSettings(
        available_at_open_bytes=available,
        total_physical_memory_bytes=total,
        memory_limit_bytes=limit,
        memory_floor_bytes=max(0, int(floor_bytes)),
        low_memory_seconds=max(0.0, float(low_memory_seconds)),
        poll_seconds=max(0.01, float(poll_seconds)),
        temp_directory=resolved_temp,
        threads=thread_count,
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
    ) -> None:
        self._connection = connection
        self.settings = settings
        self._error_factory = error_factory
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
        try:
            result = self._connection.execute(*args, **kwargs)
        except BaseException as exc:
            if self.watchdog.triggered:
                raise self._error_factory(LOW_MEMORY_REASON) from exc
            raise
        self._raise_if_triggered()
        return result

    def executemany(self, *args: Any, **kwargs: Any) -> Any:
        self._raise_if_triggered()
        try:
            result = self._connection.executemany(*args, **kwargs)
        except BaseException as exc:
            if self.watchdog.triggered:
                raise self._error_factory(LOW_MEMORY_REASON) from exc
            raise
        self._raise_if_triggered()
        return result

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        self.watchdog.stop()
        self._connection.close()

    def __enter__(self) -> "GuardedDuckDbConnection":
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
) -> GuardedDuckDbConnection:
    return GuardedDuckDbConnection(
        connection,
        settings=settings,
        memory_sampler=memory_sampler,
        clock=clock,
        error_factory=error_factory,
    )


def open_guarded_duckdb(
    database: str | Path = ":memory:",
    *,
    temp_directory: str | Path | None = None,
    threads: int | None = None,
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
        )
        return guard_configured_duckdb(
            connection,
            settings=settings,
            memory_sampler=memory_sampler,
            clock=clock,
            error_factory=error_factory,
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
    "DuckDbMemoryWatchdog",
    "DuckDbResourceSettings",
    "GuardedDuckDbConnection",
    "LOW_MEMORY_REASON",
    "available_memory_bytes",
    "configure_dynamic_duckdb",
    "dynamic_memory_limit_bytes",
    "guard_configured_duckdb",
    "open_guarded_duckdb",
    "total_physical_memory_bytes",
]
