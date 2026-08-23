"""Contracts shared by local minute-archive readers and importers."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import pyarrow as pa

SYMBOL_PATTERN = re.compile(r"(?:^|/)(?P<market>sh|sz|bj)(?P<code>\d{6})\.csv$", re.I)
PERIOD_PATTERN = re.compile(r"(?P<period>\d+)\s*(?:分钟|min)", re.I)
RAW_COLUMNS = (
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
)
CORE_COLUMNS = (
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
OUTPUT_COLUMNS = CORE_COLUMNS
PRICE_MODE = "raw_unadjusted"
SOURCE_NAME = "local_minute_zip"
AUCTION_TIME = "093000000"
CONTINUOUS_TIMES = frozenset(
    [f"{hour:02d}{minute:02d}00000" for hour, minute in [(9, value) for value in range(31, 60)]]
    + [f"{hour:02d}{minute:02d}00000" for hour, minute in [(10, value) for value in range(60)]]
    + [f"{hour:02d}{minute:02d}00000" for hour, minute in [(11, value) for value in range(31)]]
    + [f"{hour:02d}{minute:02d}00000" for hour, minute in [(13, value) for value in range(1, 60)]]
    + [f"{hour:02d}{minute:02d}00000" for hour, minute in [(14, value) for value in range(60)]]
    + ["150000000"]
)
BAR_TIME_BY_HHMM = {int(value[:4]): value for value in CONTINUOUS_TIMES | {AUCTION_TIME}}
BAR_SCHEMA = pa.schema(
    [
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


class MinuteArchiveError(ValueError):
    """Raised when a source member or formal import violates its contract."""


@dataclass(frozen=True)
class ArchiveMember:
    archive: Path
    member: str
    normalized_member: str
    symbol: str
    period: str
    crc: int = 0
    compressed_size: int = 0
    uncompressed_size: int = 0
