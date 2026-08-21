"""Read-only adapter for the local iQuant/QMT K-line cache.

The public ``xtquant.xtdata.get_local_data`` function in the installed client
is an empty compatibility stub.  The platform cache is nevertheless a simple
fixed-record file for the K-line periods used by this project.  This module
keeps the small, read-only decoder in one place and makes its assumptions
explicit so they can be tested before the data is used in research.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Final

import numpy as np
import pandas as pd

HEADER_SIZE: Final = 8
RECORD_SIZE: Final = 64
HEADER_MAGIC: Final = 0x7FFFFFFFFFFFFFFE
PRICE_SCALE: Final = 1_000.0
VOLUME_LOT_SIZE: Final = 100.0
LOCAL_TIMEZONE: Final = "Asia/Shanghai"

# The first ten fields are the values used by the K-line API.  The remaining
# bytes are platform-specific/reserved fields and are intentionally ignored.
KLINE_DTYPE: Final = np.dtype(
    {
        "names": [
            "time",
            "open",
            "high",
            "low",
            "close",
            "unused",
            "volume_lots",
            "reserved",
            "amount",
            "tail",
        ],
        "formats": [
            "<u4",
            "<u4",
            "<u4",
            "<u4",
            "<u4",
            "<u4",
            "<u4",
            "<u4",
            "<u8",
            "V24",
        ],
        "offsets": [0, 4, 8, 12, 16, 20, 24, 28, 32, 40],
        "itemsize": RECORD_SIZE,
    }
)


class IQuantDataError(ValueError):
    """Raised when a local iQuant file violates the decoder contract."""


def _validate_period(period: str) -> str:
    normalized = str(period).strip().lower()
    if normalized not in {"1d", "1m"}:
        raise IQuantDataError(f"unsupported_period:{period}")
    return normalized


def _record_count(path: Path) -> int:
    size = path.stat().st_size
    if size < HEADER_SIZE or (size - HEADER_SIZE) % RECORD_SIZE:
        raise IQuantDataError(
            f"invalid_file_size:{path}:{size}; expected 8-byte header plus 64-byte records"
        )
    return (size - HEADER_SIZE) // RECORD_SIZE


def _read_header(path: Path) -> int:
    with path.open("rb") as stream:
        raw = stream.read(HEADER_SIZE)
    if len(raw) != HEADER_SIZE:
        raise IQuantDataError(f"short_header:{path}")
    value = int.from_bytes(raw, "little", signed=False)
    if value != HEADER_MAGIC:
        raise IQuantDataError(f"unexpected_header_magic:{path}:0x{value:016x}")
    return value


def _mapped_records(path: Path) -> np.memmap:
    count = _record_count(path)
    _read_header(path)
    return np.memmap(path, dtype=KLINE_DTYPE, mode="r", offset=HEADER_SIZE, shape=(count,))


def sha256_file(path: str | Path, *, block_size: int = 8 << 20) -> str:
    """Return a content hash without loading a K-line file into memory."""

    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        while block := stream.read(block_size):
            digest.update(block)
    return digest.hexdigest()


def inspect_file(
    path: str | Path,
    *,
    period: str,
    symbol: str = "",
    include_sha256: bool = False,
) -> dict[str, object]:
    """Inspect a local file and report its decoded date/time coverage."""

    resolved = Path(path).resolve()
    normalized_period = _validate_period(period)
    if not resolved.is_file():
        raise IQuantDataError(f"missing_file:{resolved}")
    before = resolved.stat()
    records = _mapped_records(resolved)
    if len(records) == 0:
        raise IQuantDataError(f"empty_file:{resolved}")
    first = int(records["time"][0])
    last = int(records["time"][-1])
    payload: dict[str, object] = {
        "path": str(resolved),
        "symbol": str(symbol),
        "period": normalized_period,
        "header_magic": f"0x{HEADER_MAGIC:016x}",
        "header_size": HEADER_SIZE,
        "record_size": RECORD_SIZE,
        "row_count": len(records),
        "file_size": before.st_size,
        "mtime_ns": before.st_mtime_ns,
        "stable_during_read": False,
        "decoded_start": pd.Timestamp(first, unit="s", tz=LOCAL_TIMEZONE).isoformat(),
        "decoded_end": pd.Timestamp(last, unit="s", tz=LOCAL_TIMEZONE).isoformat(),
        "price_scale": PRICE_SCALE,
        "volume_unit": "shares_after_lot_size_conversion",
        "volume_lot_size": VOLUME_LOT_SIZE,
        "amount_unit": "currency_units",
        "adjustment_metadata": "not_stored_in_DAT_header; assessed by parity",
    }
    if include_sha256:
        payload["sha256"] = sha256_file(resolved)
    after = resolved.stat()
    payload["stable_during_read"] = before.st_size == after.st_size and before.st_mtime_ns == after.st_mtime_ns
    return payload


def _local_timestamps(values: np.ndarray) -> pd.DatetimeIndex:
    return pd.to_datetime(values.astype(np.int64), unit="s", utc=True).tz_convert(LOCAL_TIMEZONE)


def _bar_time_from_seconds(seconds: np.ndarray) -> np.ndarray:
    hours = seconds // 3600
    minutes = (seconds % 3600) // 60
    return np.asarray(
        [f"{int(hour):02d}{int(minute):02d}00000" for hour, minute in zip(hours, minutes, strict=True)],
        dtype=object,
    )


def read_file(
    path: str | Path,
    *,
    symbol: str,
    period: str,
    start_date: str = "",
    end_date: str = "",
) -> pd.DataFrame:
    """Decode one local iQuant daily or one-minute K-line file.

    Prices are converted from the platform's fixed-point 1/1000 representation;
    volume is converted from lots to shares.  The returned timestamps use the
    Shanghai exchange timezone and the same date/bar-time strings as QDP.
    """

    normalized_period = _validate_period(period)
    resolved = Path(path).resolve()
    records = _mapped_records(resolved)
    timestamps = _local_timestamps(records["time"])
    frame = pd.DataFrame(
        {
            "symbol": str(symbol),
            "open": records["open"].astype(np.float64) / PRICE_SCALE,
            "high": records["high"].astype(np.float64) / PRICE_SCALE,
            "low": records["low"].astype(np.float64) / PRICE_SCALE,
            "close": records["close"].astype(np.float64) / PRICE_SCALE,
            "volume": records["volume_lots"].astype(np.float64) * VOLUME_LOT_SIZE,
            "amount": records["amount"].astype(np.float64),
            "trade_date": timestamps.strftime("%Y-%m-%d"),
        }
    )
    if normalized_period == "1m":
        frame.insert(2, "bar_time", _bar_time_from_seconds(timestamps.hour * 3600 + timestamps.minute * 60 + timestamps.second))
        columns = ["symbol", "trade_date", "bar_time", "open", "high", "low", "close", "volume", "amount"]
    else:
        columns = ["symbol", "trade_date", "open", "high", "low", "close", "volume", "amount"]
    frame = frame.loc[:, columns]
    if start_date:
        frame = frame.loc[frame["trade_date"] >= str(start_date)]
    if end_date:
        frame = frame.loc[frame["trade_date"] <= str(end_date)]
    return frame.reset_index(drop=True)


def aggregate_1m_to_5m(frame: pd.DataFrame) -> pd.DataFrame:
    """Aggregate one-minute rows into QDP's 48 right-edge 5-minute buckets."""

    required = {"symbol", "trade_date", "bar_time", "open", "high", "low", "close", "volume", "amount"}
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise IQuantDataError(f"minute_columns_missing:{','.join(missing)}")
    if frame.empty:
        return pd.DataFrame(columns=[*sorted(required), "minute_count"])
    working = frame.copy()
    clock = working["bar_time"].astype(str).str[:6]
    seconds = (
        pd.to_numeric(clock.str[:2], errors="raise") * 3600
        + pd.to_numeric(clock.str[2:4], errors="raise") * 60
        + pd.to_numeric(clock.str[4:6], errors="raise")
    )
    # A bar ending at 09:35 contains 09:31..09:35.  The same rule naturally
    # skips the lunch break because no 11:31..13:00 source rows exist.
    end_seconds = ((seconds + 299) // 300) * 300
    working["bar_time"] = _bar_time_from_seconds(end_seconds.to_numpy())
    result = (
        working.groupby(["symbol", "trade_date", "bar_time"], sort=True, observed=True)
        .agg(
            open=("open", "first"),
            high=("high", "max"),
            low=("low", "min"),
            close=("close", "last"),
            volume=("volume", "sum"),
            amount=("amount", "sum"),
            minute_count=("bar_time", "size"),
        )
        .reset_index()
    )
    return result.loc[
        :, ["symbol", "trade_date", "bar_time", "open", "high", "low", "close", "volume", "amount", "minute_count"]
    ]


__all__ = [
    "HEADER_MAGIC",
    "HEADER_SIZE",
    "IQuantDataError",
    "KLINE_DTYPE",
    "LOCAL_TIMEZONE",
    "PRICE_SCALE",
    "RECORD_SIZE",
    "VOLUME_LOT_SIZE",
    "aggregate_1m_to_5m",
    "inspect_file",
    "read_file",
    "sha256_file",
]
