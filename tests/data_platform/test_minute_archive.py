from __future__ import annotations

from pathlib import Path
from zipfile import ZipFile

import pandas as pd

from quantlab.data.minute_archive import extract_to_parquet, import_year, list_members, standardize_frame
from quantlab.data.minute_archive.reader import read_member_year
from quantlab.data.qdp_v2.manifest import (
    DatasetManifest,
    ShardManifestEntry,
    write_active_manifest,
    write_dataset_manifest,
)


def _csv_payload() -> bytes:
    rows = []
    for date in ("2025-01-02", "2025-01-03"):
        rows.append(f"{date} 09:30:00,10,10,10,10,0,0,0,0,0,1000,2000")
        for minute in range(31, 36):
            rows.append(f"{date} 09:{minute:02d}:00,10,10.1,9.9,10,100,1000,0,0,1,1000,2000")
    header = "日期时间,开盘,最高,最低,收盘,成交量(股),成交额(元),涨跌额(元),涨跌幅(%),换手率(%),流通股本(股),总股本(股)"
    return (header + "\n" + "\n".join(rows) + "\n").encode("utf-8-sig")


def _full_day_csv_payload() -> bytes:
    rows = []
    times = [
        *pd.date_range("2025-01-02 09:31", "2025-01-02 11:30", freq="1min").strftime("%H:%M:%S"),
        *pd.date_range("2025-01-02 13:01", "2025-01-02 15:00", freq="1min").strftime("%H:%M:%S"),
    ]
    for date in ("2025-01-02", "2025-01-03"):
        rows.append(f"{date} 09:30:00,10,10,10,10,0,0,0,0,0,1000,2000")
        rows.extend(
            f"{date} {time},10,10.1,9.9,10,100,1000,0,0,1,1000,2000"
            for time in times
        )
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
    assert not {"turnover_pct", "float_shares", "total_shares"}.intersection(frame.columns)


def test_member_year_reader_stops_at_next_year(tmp_path: Path) -> None:
    archive = tmp_path / "minute.zip"
    header = "日期时间,开盘,最高,最低,收盘,成交量(股),成交额(元),涨跌额(元),涨跌幅(%),换手率(%),流通股本(股),总股本(股)"
    rows = [
        f"{year}-01-02 09:31:00,10,10,10,10,100,1000,0,0,1,1000,2000"
        for year in (2024, 2025, 2026)
    ]
    with ZipFile(archive, "w") as zipped:
        zipped.writestr("1分钟/sh600000.csv", (header + "\n" + "\n".join(rows)).encode("utf-8"))
    with ZipFile(archive) as zipped:
        frame = read_member_year(
            zipped,
            zipped.getinfo("1分钟/sh600000.csv"),
            symbol="600000.SH",
            year=2025,
        )
    assert frame["timestamp"].dt.year.tolist() == [2025]


def test_formal_import_separates_auction_and_daily_share_evidence(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    root = workspace / "data" / "qdp" / "qdp_v2"
    daily_id = "market_daily_raw_test"
    daily_dir = root / "datasets" / "market_daily_raw" / daily_id
    daily_path = daily_dir / "shards" / "part-0000.parquet"
    daily_path.parent.mkdir(parents=True)
    pd.DataFrame(
        {
            "symbol": ["600000.SH", "600000.SH"],
            "trade_date": ["2025-01-02", "2025-01-03"],
            "open": [10.0, 10.0],
            "high": [10.1, 10.1],
            "low": [9.9, 9.9],
            "close": [10.0, 10.0],
            "volume": [24_000.0, 24_000.0],
            "amount": [240_000.0, 240_000.0],
            "source": ["fixture", "fixture"],
            "adjusted_flag": ["raw_unadjusted", "raw_unadjusted"],
        }
    ).to_parquet(daily_path, index=False)
    write_dataset_manifest(
        root,
        DatasetManifest(
            dataset_id=daily_id,
            domain="market_daily_raw",
            layer="raw",
            frequency="1d",
            contract_version="fixture",
            primary_key=["trade_date", "symbol"],
            start_date="2025-01-02",
            end_date="2025-01-03",
            row_count=2,
            shards=[
                ShardManifestEntry(
                    path=daily_path.relative_to(root).as_posix(),
                    row_count=2,
                    start_date="2025-01-02",
                    end_date="2025-01-03",
                    file_size=daily_path.stat().st_size,
                )
            ],
            source={"provider": "fixture"},
            quality={"primary_key_unique": True},
        ),
    )
    write_active_manifest(
        root,
        {"version": 2, "datasets": {"market_daily_raw": daily_id}, "scope": {}, "source": {}},
    )
    archive = tmp_path / "minute.zip"
    with ZipFile(archive, "w") as zipped:
        zipped.writestr("1分钟/sh600000.csv", _full_day_csv_payload())

    result = import_year(archive, year=2025, workspace_root_value=workspace)

    assert result["continuous"]["row_count"] == 480
    assert result["opening_auction"]["row_count"] == 2
    assert result["quality"]["unexplained_incomplete_stock_days"] == 0
    assert result["quality"]["minute_feature_exclusion_rows"] == 0
    continuous = pd.read_parquet(
        root / "datasets" / "market_intraday_1m" / "market_intraday_1m" / "shards"
    )
    auction = pd.read_parquet(
        root / "datasets" / "market_opening_auction" / "market_opening_auction" / "shards"
    )
    shares = pd.read_parquet(
        workspace
        / "data"
        / "qdp"
        / "source_archives"
        / "minute"
        / "quality"
        / "year=2025"
        / "daily_share_candidates.parquet"
    )
    assert len(continuous) == 480
    assert continuous.groupby(["symbol", "trade_date"]).size().eq(240).all()
    assert set(auction["bar_time"]) == {"093000000"}
    assert list(shares[["float_shares", "total_shares"]].iloc[0]) == [1000.0, 2000.0]
    exclusions = pd.read_parquet(
        workspace
        / "data"
        / "qdp"
        / "source_archives"
        / "minute"
        / "quality"
        / "year=2025"
        / "minute_feature_exclusions.parquet"
    )
    assert exclusions.empty
