from __future__ import annotations

import struct
from pathlib import Path

import pandas as pd
import pytest

from quantlab.data.iquant import (
    HEADER_MAGIC,
    IQuantDataError,
    aggregate_1m_to_5m,
    inspect_file,
    read_file,
)


def _write_dat(path: Path, rows: list[tuple[int, int, int, int, int, int, int]]) -> None:
    with path.open("wb") as stream:
        stream.write(struct.pack("<Q", HEADER_MAGIC))
        for timestamp, opening, high, low, close, volume_lots, amount in rows:
            stream.write(
                struct.pack(
                    "<8I Q 24x",
                    timestamp,
                    opening,
                    high,
                    low,
                    close,
                    0,
                    volume_lots,
                    0,
                    amount,
                )
            )


def test_read_daily_decodes_date_price_and_units(tmp_path: Path) -> None:
    # 2020-01-02 09:31 Asia/Shanghai, represented as UTC epoch seconds by DAT.
    timestamp = int(pd.Timestamp("2020-01-02 01:31:00", tz="UTC").timestamp())
    path = tmp_path / "600000.DAT"
    _write_dat(path, [(timestamp, 12345, 12400, 12200, 12300, 321, 456789)])

    result = read_file(path, symbol="600000.SH", period="1d")

    assert result.to_dict("records") == [
        {
            "symbol": "600000.SH",
            "trade_date": "2020-01-02",
            "open": 12.345,
            "high": 12.4,
            "low": 12.2,
            "close": 12.3,
            "volume": 32100.0,
            "amount": 456789.0,
        }
    ]
    metadata = inspect_file(path, period="1d", symbol="600000.SH")
    assert metadata["row_count"] == 1
    assert metadata["stable_during_read"] is True


def test_read_minute_converts_utc_epoch_to_shanghai_bar_time(tmp_path: Path) -> None:
    timestamp = int(pd.Timestamp("2020-01-02 01:31:00", tz="UTC").timestamp())
    path = tmp_path / "600000.DAT"
    _write_dat(path, [(timestamp, 10000, 10000, 10000, 10000, 1, 1000)])

    result = read_file(path, symbol="600000.SH", period="1m")

    assert result.loc[0, "trade_date"] == "2020-01-02"
    assert result.loc[0, "bar_time"] == "093100000"


def test_minute_aggregation_uses_right_edge_and_keeps_lunch_break() -> None:
    rows = []
    clock_values = [
        "09:31",
        "09:32",
        "09:33",
        "09:34",
        "09:35",
        "09:36",
        "09:37",
        "09:38",
        "09:39",
        "09:40",
        "13:01",
        "13:02",
        "13:03",
        "13:04",
        "13:05",
        "13:06",
        "13:07",
        "13:08",
        "13:09",
        "13:10",
    ]
    for index, clock in enumerate(clock_values):
        rows.append(
            {
                "symbol": "600000.SH",
                "trade_date": "2020-01-02",
                "bar_time": clock.replace(":", "") + "00000",
                "open": float(index),
                "high": float(index) + 0.5,
                "low": float(index) - 0.5,
                "close": float(index) + 0.25,
                "volume": 100.0,
                "amount": 1000.0,
            }
        )
    result = aggregate_1m_to_5m(pd.DataFrame(rows))

    assert result["bar_time"].tolist() == ["093500000", "094000000", "130500000", "131000000"]
    assert result["minute_count"].tolist() == [5, 5, 5, 5]
    assert result.loc[result["bar_time"] == "093500000", "volume"].item() == 500.0


def test_minute_aggregation_can_fold_opening_auction_into_first_bucket() -> None:
    rows = []
    for minute, volume in (("09:30", 20.0), ("09:31", 100.0), ("09:32", 100.0), ("09:33", 100.0), ("09:34", 100.0), ("09:35", 100.0)):
        rows.append(
            {
                "symbol": "600000.SH",
                "trade_date": "2020-01-02",
                "bar_time": minute.replace(":", "") + "00000",
                "open": 10.0,
                "high": 10.0,
                "low": 10.0,
                "close": 10.0,
                "volume": volume,
                "amount": volume * 10.0,
            }
        )
    result = aggregate_1m_to_5m(pd.DataFrame(rows), include_opening_auction=True)
    assert result["bar_time"].tolist() == ["093500000"]
    assert result.loc[0, "volume"] == 520.0


def test_continuous_minute_aggregation_drops_standalone_auction() -> None:
    rows = []
    for minute in ("09:30", "09:31", "09:32", "09:33", "09:34", "09:35"):
        rows.append(
            {
                "symbol": "600000.SH",
                "trade_date": "2020-01-02",
                "bar_time": minute.replace(":", "") + "00000",
                "open": 10.0,
                "high": 10.0,
                "low": 10.0,
                "close": 10.0,
                "volume": 100.0,
                "amount": 1000.0,
            }
        )
    result = aggregate_1m_to_5m(pd.DataFrame(rows))
    assert result["bar_time"].tolist() == ["093500000"]
    assert result.loc[0, "minute_count"] == 5


def test_invalid_header_and_size_are_rejected(tmp_path: Path) -> None:
    bad_header = tmp_path / "bad_header.DAT"
    bad_header.write_bytes(b"\x00" * 72)
    with pytest.raises(IQuantDataError, match="unexpected_header_magic"):
        read_file(bad_header, symbol="600000.SH", period="1d")

    bad_size = tmp_path / "bad_size.DAT"
    bad_size.write_bytes(struct.pack("<Q", HEADER_MAGIC) + b"\x00")
    with pytest.raises(IQuantDataError, match="invalid_file_size"):
        read_file(bad_size, symbol="600000.SH", period="1d")
