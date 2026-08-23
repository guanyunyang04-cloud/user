from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from zipfile import ZipFile

import pandas as pd
import pytest

import quantlab.data.minute_archive.importer as minute_importer
from quantlab.data.minute_archive import (
    extract_to_parquet,
    import_year,
    import_years,
    list_members,
    resume_staged_import,
    standardize_frame,
)
from quantlab.data.minute_archive.quality import (
    parity_audit,
    repair_mislabeled_1300_as_1130,
    repair_zero_price_placeholders,
)
from quantlab.data.minute_archive.reader import read_member_year, read_member_years
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


def _full_day_csv_payload(years: tuple[int, ...] = (2025,)) -> bytes:
    rows = []
    for year in years:
        times = [
            *pd.date_range(f"{year}-01-02 09:31", f"{year}-01-02 11:30", freq="1min").strftime("%H:%M:%S"),
            *pd.date_range(f"{year}-01-02 13:01", f"{year}-01-02 15:00", freq="1min").strftime("%H:%M:%S"),
        ]
        for date in (f"{year}-01-02", f"{year}-01-03"):
            rows.append(f"{date} 09:30:00,10,10,10,10,0,0,0,0,0,1000,2000")
            rows.extend(f"{date} {time},10,10.1,9.9,10,100,1000,0,0,1,1000,2000" for time in times)
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


def test_zero_flow_all_zero_price_placeholder_is_repaired_with_ledger() -> None:
    frame = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(["2025-01-02 09:30", "2025-01-02 09:31"]),
            "open": [0.0, 10.0],
            "high": [0.0, 10.1],
            "low": [0.0, 9.9],
            "close": [0.0, 10.0],
            "volume": [0.0, 100.0],
            "amount": [0.0, 1000.0],
        }
    )

    ledger = repair_zero_price_placeholders(frame, symbol="600000.SH")

    assert frame.loc[0, ["open", "high", "low", "close"]].tolist() == [10.0] * 4
    assert ledger[["symbol", "trade_date", "bar_time"]].iloc[0].tolist() == [
        "600000.SH",
        "2025-01-02",
        "093000000",
    ]


def test_1300_is_repaired_only_when_it_replaces_missing_1130() -> None:
    frame = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(
                [
                    "2026-01-02 11:29",
                    "2026-01-02 13:00",
                    "2026-01-02 13:01",
                    "2026-01-05 11:30",
                    "2026-01-05 13:00",
                ]
            )
        }
    )

    ledger = repair_mislabeled_1300_as_1130(frame, symbol="600000.SH")

    assert frame["timestamp"].dt.strftime("%Y-%m-%d %H:%M").tolist() == [
        "2026-01-02 11:29",
        "2026-01-02 11:30",
        "2026-01-02 13:01",
        "2026-01-05 11:30",
        "2026-01-05 13:00",
    ]
    assert ledger[["trade_date", "original_bar_time", "repaired_bar_time"]].iloc[0].tolist() == [
        "2026-01-02",
        "130000000",
        "113000000",
    ]


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
    rows = [f"{year}-01-02 09:31:00,10,10,10,10,100,1000,0,0,1,1000,2000" for year in (2024, 2025, 2026)]
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


def test_member_years_reader_starts_at_first_available_selected_year(tmp_path: Path) -> None:
    archive = tmp_path / "minute.zip"
    header = "日期时间,开盘,最高,最低,收盘,成交量(股),成交额(元),涨跌额(元),涨跌幅(%),换手率(%),流通股本(股),总股本(股)"
    rows = [f"{year}-01-02 09:31:00,10,10,10,10,100,1000,0,0,1,1000,2000" for year in (2025, 2026)]
    with ZipFile(archive, "w") as zipped:
        zipped.writestr("1分钟/sh600000.csv", (header + "\n" + "\n".join(rows)).encode())
    with ZipFile(archive) as zipped:
        frame = read_member_years(
            zipped,
            zipped.getinfo("1分钟/sh600000.csv"),
            symbol="600000.SH",
            years=[2024, 2025],
        )
    assert frame["timestamp"].dt.year.tolist() == [2025]


def test_member_year_reader_accepts_missing_trailing_share_fields(tmp_path: Path) -> None:
    archive = tmp_path / "minute.zip"
    header = "日期时间,开盘,最高,最低,收盘,成交量(股),成交额(元),涨跌额(元),涨跌幅(%),换手率(%)"
    row = "2025-01-02 09:31:00,10,10,10,10,100,1000,0,0,1"
    with ZipFile(archive, "w") as zipped:
        zipped.writestr("1分钟/sh600000.csv", (header + "\n" + row).encode())
    with ZipFile(archive) as zipped:
        frame = read_member_year(
            zipped,
            zipped.getinfo("1分钟/sh600000.csv"),
            symbol="600000.SH",
            year=2025,
        )
    assert len(frame) == 1
    assert frame[["float_shares", "total_shares"]].isna().all().all()


def test_import_memory_policy_preserves_headroom_and_scales_down(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    total = 16 * 1024**3
    monkeypatch.setattr(
        minute_importer.psutil,
        "virtual_memory",
        lambda: SimpleNamespace(total=total, available=8 * 1024**3),
    )
    normal = minute_importer._import_memory_policy()
    monkeypatch.setattr(
        minute_importer.psutil,
        "virtual_memory",
        lambda: SimpleNamespace(total=total, available=3 * 1024**3),
    )
    constrained = minute_importer._import_memory_policy()

    assert normal.reserve_bytes == 4 * 1024**3
    assert normal.continuous_max_buffered_rows == 2_000_000
    assert constrained.writer_budget_bytes == 128 * 1024**2
    assert constrained.continuous_max_buffered_rows < normal.continuous_max_buffered_rows
    assert constrained.auction_max_buffered_rows < normal.auction_max_buffered_rows


def test_memory_pressure_flushes_buffered_months(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    writer = minute_importer.MonthlyWriters(
        tmp_path / "minute",
        flush_rows=10_000,
        max_buffered_rows=20_000,
    )
    frame = pd.DataFrame(
        {
            "symbol": ["600000.SH"],
            "trade_date": ["2025-01-02"],
            "bar_time": ["093100000"],
            "open": [10.0],
            "high": [10.0],
            "low": [10.0],
            "close": [10.0],
            "volume": [100.0],
            "amount": [1000.0],
            "source": ["fixture"],
            "adjusted_flag": ["raw_unadjusted"],
        }
    )
    writer.write(frame)
    policy = minute_importer.ImportMemoryPolicy(
        total_bytes=16 * 1024**3,
        available_at_start_bytes=8 * 1024**3,
        reserve_bytes=4 * 1024**3,
        writer_budget_bytes=1024**3,
        pressure_threshold_bytes=4 * 1024**3,
        continuous_flush_rows=10_000,
        continuous_max_buffered_rows=20_000,
        auction_flush_rows=10_000,
        auction_max_buffered_rows=20_000,
    )
    monkeypatch.setattr(
        minute_importer.psutil,
        "virtual_memory",
        lambda: SimpleNamespace(total=16 * 1024**3, available=2 * 1024**3),
    )

    assert minute_importer._flush_if_memory_pressure(policy, writer)
    assert writer.total_buffered_rows == 0
    writer.close()
    assert next((tmp_path / "minute" / "shards").rglob("*.parquet")).is_file()


def _write_daily_fixture(workspace: Path, years: tuple[int, ...]) -> Path:
    root = workspace / "data" / "qdp" / "qdp_v2"
    daily_id = "market_daily_raw_test"
    daily_dir = root / "datasets" / "market_daily_raw" / daily_id
    daily_path = daily_dir / "shards" / "part-0000.parquet"
    daily_path.parent.mkdir(parents=True)
    dates = [f"{year}-01-{day:02d}" for year in years for day in (2, 3)]
    pd.DataFrame(
        {
            "symbol": ["600000.SH"] * len(dates),
            "trade_date": dates,
            "open": [10.0] * len(dates),
            "high": [10.1] * len(dates),
            "low": [9.9] * len(dates),
            "close": [10.0] * len(dates),
            "volume": [24_000.0] * len(dates),
            "amount": [240_000.0] * len(dates),
            "source": ["fixture"] * len(dates),
            "adjusted_flag": ["raw_unadjusted"] * len(dates),
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
            start_date=min(dates),
            end_date=max(dates),
            row_count=len(dates),
            shards=[
                ShardManifestEntry(
                    path=daily_path.relative_to(root).as_posix(),
                    row_count=len(dates),
                    start_date=min(dates),
                    end_date=max(dates),
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
    return root


def test_formal_import_separates_auction_and_daily_share_evidence(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    root = _write_daily_fixture(workspace, (2025,))
    archive = tmp_path / "minute.zip"
    with ZipFile(archive, "w") as zipped:
        zipped.writestr("1分钟/sh600000.csv", _full_day_csv_payload())

    result = import_year(archive, year=2025, workspace_root_value=workspace)

    assert result["continuous"]["row_count"] == 480
    assert result["opening_auction"]["row_count"] == 2
    assert result["quality"]["unexplained_incomplete_stock_days"] == 0
    assert result["quality"]["minute_feature_exclusion_rows"] == 0
    continuous = pd.read_parquet(root / "datasets" / "market_intraday_1m" / "market_intraday_1m" / "shards")
    auction = pd.read_parquet(root / "datasets" / "market_opening_auction" / "market_opening_auction" / "shards")
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


def test_parity_audit_uses_five_cent_relative_and_field_level_price_policy(tmp_path: Path) -> None:
    dates = [f"2025-01-0{day}" for day in range(2, 6)]
    bars_path = tmp_path / "bars.parquet"
    daily_path = tmp_path / "daily.parquet"
    output_dir = tmp_path / "quality"
    final_quality_dir = tmp_path / "installed-quality"
    output_dir.mkdir()
    pd.DataFrame(
        {
            "symbol": ["600000.SH"] * 4,
            "trade_date": dates,
            "bar_time": ["093100000"] * 4,
            "open": [10.0, 100.0, 10.0, 10.0],
            "high": [10.05, 100.06, 10.06, 10.11],
            "low": [10.0, 100.0, 10.0, 10.0],
            "close": [10.0, 100.0, 10.0, 10.0],
            "volume": [100.0] * 4,
            "amount": [1000.0, 10_000.0, 1000.0, 1000.0],
        }
    ).to_parquet(bars_path, index=False)
    pd.DataFrame(
        {
            "symbol": ["600000.SH"] * 4,
            "trade_date": dates,
            "open": [10.0, 100.0, 10.0, 10.0],
            "high": [10.0, 100.0, 10.0, 10.0],
            "low": [10.0, 100.0, 10.0, 10.0],
            "close": [10.0, 100.0, 10.0, 10.0],
            "volume": [100.0] * 4,
            "amount": [1000.0, 10_000.0, 1000.0, 1000.0],
        }
    ).to_parquet(daily_path, index=False)

    result = parity_audit(
        continuous_paths=[bars_path],
        auction_paths=[],
        daily_paths=[daily_path],
        year=2025,
        output_dir=output_dir,
        final_quality_dir=final_quality_dir,
    )

    assert result["price_over_five_cent_rows"] == 3
    assert result["price_warning_rows"] == 1
    assert result["price_unreliable_rows"] == 2
    assert result["price_severe_rows"] == 1
    assert result["minute_feature_exclusion_rows"] == 2
    mismatches = pd.read_parquet(output_dir / "daily_parity_material_mismatches.parquet")
    assert mismatches["price_quality_class"].tolist() == ["warning", "unreliable", "severe"]
    exclusions = pd.read_parquet(output_dir / "minute_feature_exclusions.parquet")
    assert exclusions["exclude_high"].all()
    assert not exclusions[["exclude_open", "exclude_low", "exclude_close"]].any(axis=None)
    assert exclusions["severity"].tolist() == ["unreliable", "severe"]


def test_formal_import_appends_only_absent_year_partitions(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    root = _write_daily_fixture(workspace, (2024, 2025))
    archive = tmp_path / "minute.zip"
    with ZipFile(archive, "w") as zipped:
        zipped.writestr("1分钟/sh600000.csv", _full_day_csv_payload((2024, 2025)))

    import_year(archive, year=2025, workspace_root_value=workspace)
    existing_path = (
        root
        / "datasets"
        / "market_intraday_1m"
        / "market_intraday_1m"
        / "shards"
        / "year=2025"
        / "month=01"
        / "part-0000.parquet"
    )
    existing_bytes = existing_path.read_bytes()
    result = import_years(
        archive,
        years=[2024, 2025],
        workspace_root_value=workspace,
    )

    assert result["status"] == "ready"
    assert result["imported_years"] == [2024]
    assert result["skipped_years"] == [2025]
    assert existing_path.read_bytes() == existing_bytes
    assert result["continuous"]["row_count"] == 960
    assert result["opening_auction"]["row_count"] == 4
    assert result["continuous"]["quality"]["imported_years"] == [2024, 2025]
    continuous = pd.read_parquet(root / "datasets" / "market_intraday_1m" / "market_intraday_1m" / "shards")
    assert len(continuous) == 960
    unchanged = import_years(
        archive,
        years=[2024, 2025],
        workspace_root_value=workspace,
    )
    assert unchanged["status"] == "unchanged"


def test_formal_import_can_resume_a_stream_checkpoint(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    workspace = tmp_path / "workspace"
    _write_daily_fixture(workspace, (2025,))
    archive = tmp_path / "minute.zip"
    with ZipFile(archive, "w") as zipped:
        zipped.writestr("1分钟/sh600000.csv", _full_day_csv_payload())
    real_parity_audit = minute_importer.parity_audit

    def fail_after_streaming(**_: object) -> dict[str, object]:
        raise RuntimeError("fixture_audit_failure")

    monkeypatch.setattr(minute_importer, "parity_audit", fail_after_streaming)
    with pytest.raises(RuntimeError, match="fixture_audit_failure"):
        import_year(archive, year=2025, workspace_root_value=workspace)
    staging = next((workspace / "data" / "qdp" / "qdp_v2" / "tmp").glob("minute_import_*"))
    assert (staging / "stream_checkpoint.json").is_file()
    monkeypatch.setattr(minute_importer, "parity_audit", real_parity_audit)

    result = resume_staged_import(staging, workspace_root_value=workspace)

    assert result["status"] == "ready"
    assert result["continuous"]["row_count"] == 480
    assert not staging.exists()
