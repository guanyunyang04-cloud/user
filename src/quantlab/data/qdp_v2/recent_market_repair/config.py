"""Recent Market Repair: config responsibilities."""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Self

import pyarrow as pa

from quantlab.data.qdp_v2.duckdb_resources import (
    DEFAULT_LOW_MEMORY_SECONDS,
    DEFAULT_MEMORY_FLOOR_BYTES,
    DEFAULT_POLL_SECONDS,
    LOW_MEMORY_REASON,
)

DAILY_PRICE_RELATIVE_TOLERANCE = 0.02


DAILY_VOLUME_RELATIVE_TOLERANCE = 0.05


DAILY_AMOUNT_RELATIVE_TOLERANCE = 0.05


VOLUME_UNIT_SCALES = (1.0, 100.0, 0.01)


AMOUNT_UNIT_SCALES = (1.0, 1000.0, 0.001, 100.0, 0.01)


REPAIR_VERSION = 1


BUCKET_COUNT = 16


DEFAULT_WORKERS = 4


DEFAULT_START_DATE = "2026-06-29"


DEFAULT_END_DATE = "2026-07-13"


MEMORY_FLOOR_BYTES = DEFAULT_MEMORY_FLOOR_BYTES


LOW_MEMORY_SECONDS = DEFAULT_LOW_MEMORY_SECONDS


POLL_SECONDS = DEFAULT_POLL_SECONDS


DAILY_DOMAIN = "market_daily_raw"


INTRADAY_DOMAIN = "market_intraday_5m"


DAILY_COLUMNS = (
    "symbol",
    "trade_date",
    "open",
    "high",
    "low",
    "close",
    "volume",
    "amount",
    "source",
    "adjusted_flag",
)


INTRADAY_COLUMNS = (
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


NUMERIC_COLUMNS = ("open", "high", "low", "close", "volume", "amount")


DAILY_SCHEMA = pa.schema(
    [
        pa.field("symbol", pa.string()),
        pa.field("trade_date", pa.string()),
        *[pa.field(item, pa.float64()) for item in NUMERIC_COLUMNS],
        pa.field("source", pa.string()),
        pa.field("adjusted_flag", pa.string()),
    ]
)


INTRADAY_SCHEMA = pa.schema(
    [
        pa.field("symbol", pa.string()),
        pa.field("trade_date", pa.string()),
        pa.field("bar_time", pa.string()),
        *[pa.field(item, pa.float64()) for item in NUMERIC_COLUMNS],
        pa.field("source", pa.string()),
        pa.field("adjusted_flag", pa.string()),
    ]
)


class RecentMarketRepairError(RuntimeError):
    pass


class RecentMarketMemoryError(RecentMarketRepairError):
    pass


class _SystemMemoryGuard:
    """Continuously observe the shared physical-memory safety floor."""

    def __init__(self) -> None:
        self._stop = threading.Event()
        self._triggered = threading.Event()
        self._thread: threading.Thread | None = None
        self._low_since: float | None = None
        self.minimum_available_bytes: int | None = None

    def __enter__(self) -> Self:
        self._thread = threading.Thread(
            target=self._run,
            name="qdp-recent-repair-memory-guard",
            daemon=True,
        )
        self._thread.start()
        return self

    def __exit__(self, *_: object) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)

    def check(self, label: str) -> None:
        if self._triggered.is_set():
            raise RecentMarketMemoryError(f"{LOW_MEMORY_REASON}:{label}")

    def _run(self) -> None:
        try:
            import psutil  # type: ignore
        except ImportError:
            self._triggered.set()
            return
        while not self._stop.wait(POLL_SECONDS):
            available = int(psutil.virtual_memory().available)
            if self.minimum_available_bytes is None or available < self.minimum_available_bytes:
                self.minimum_available_bytes = available
            if available >= MEMORY_FLOOR_BYTES:
                self._low_since = None
                continue
            now = time.monotonic()
            if self._low_since is None:
                self._low_since = now
            elif now - self._low_since >= LOW_MEMORY_SECONDS:
                self._triggered.set()
                return


@dataclass(frozen=True)
class _DomainInput:
    domain: str
    dataset_id: str
    manifest_path: Path
    paths: tuple[Path, ...]


@dataclass(frozen=True)
class _BaostockTask:
    symbol: str
    dates: tuple[str, ...]

    @property
    def start_date(self) -> str:
        return min(self.dates)

    @property
    def end_date(self) -> str:
        return max(self.dates)


