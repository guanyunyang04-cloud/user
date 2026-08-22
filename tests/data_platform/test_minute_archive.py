from __future__ import annotations

from pathlib import Path
from zipfile import ZipFile

import pandas as pd

from quantlab.data.minute_archive import extract_to_parquet, list_members, standardize_frame


def _csv_payload() -> bytes:
    rows = []
    for date in ("2025-01-02", "2025-01-03"):
        rows.append(f"{date} 09:30:00,10,10,10,10,0,0,0,0,0,1000,2000")
        for minute in range(31, 36):
            rows.append(f"{date} 09:{minute:02d}:00,10,10.1,9.9,10,100,1000,0,0,1,1000,2000")
    header = "日期时间,开盘,最高,最低,收盘,成交量(股),成交额(元),涨跌额(元),涨跌幅(%),换手率(%),流通股本(股),总股本(股)"
    return (header + "\n" + "\n".join(rows) + "\n").encode("utf-8-sig")


def test_standardize_removes_standalone_opening_row() -> None:
    raw = pd.DataFrame(
        {
            "symbol": ["600000.SH", "600000.SH"],
            "timestamp": pd.to_datetime(["2025-01-02 09:30", "2025-01-02 09:31"]),
            "open": [10.0, 10.0],
            "high": [10.0, 10.1],
            "low": [10.0, 9.9],
            "close": [10.0, 10.0],
            "volume": [0.0, 100.0],
            "amount": [0.0, 1000.0],
            "price_change": [0.0, 0.0],
            "pct_change": [0.0, 0.0],
            "turnover_pct": [0.0, 1.0],
            "float_shares": [1000.0, 1000.0],
            "total_shares": [2000.0, 2000.0],
        }
    )
    result = standardize_frame(raw)
    assert result["bar_time"].tolist() == ["093100000"]
    assert result["trade_date"].tolist() == ["2025-01-02"]


def test_zip_extracts_selected_member_to_parquet(tmp_path: Path) -> None:
    archive = tmp_path / "minute.zip"
    with ZipFile(archive, "w") as zipped:
        zipped.writestr("1分钟/sh600000.csv", _csv_payload())
        zipped.writestr("1分钟/sz000001.csv", _csv_payload())
    members = list_members([archive], symbols=["600000.SH"])
    assert len(members) == 1
    output = tmp_path / "extract.parquet"
    manifest = extract_to_parquet(
        [archive],
        symbols=["600000.SH"],
        output_path=output,
        start_date="2025-01-02",
        end_date="2025-01-03",
    )
    frame = pd.read_parquet(output)
    assert manifest["row_count"] == 10
    assert len(frame) == 10
    assert not (frame["bar_time"] == "093000000").any()
