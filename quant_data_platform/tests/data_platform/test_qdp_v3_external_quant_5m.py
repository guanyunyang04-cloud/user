from __future__ import annotations

import json
import zipfile
from pathlib import Path

import pandas as pd
import pytest

from quant_data_platform.core.json_io import read_json
from quant_data_platform.qdp_v3.constants import EXPECTED_5M_BAR_ENDS
from quant_data_platform.qdp_v3.external_quant_5m import (
    RAW_COLUMNS,
    ExternalQuant5mError,
    ExternalQuant5mResourceGuardError,
    import_external_quant_5m,
    normalize_external_quant_5m,
    scan_external_quant_5m_sources,
)
from quant_data_platform.qdp_v3.storage import read_raw_partition, read_raw_receipt


def _workspace(tmp_path: Path) -> Path:
    workspace = tmp_path / "workspace"
    manifest = workspace / "brain" / "brain_manifest.json"
    manifest.parent.mkdir(parents=True, exist_ok=True)
    manifest.write_text(json.dumps({"schema_version": 1, "brain_type": "main"}), encoding="utf-8")
    return workspace


def _source_day(
    trade_date: str,
    *,
    auction: bool = True,
    zero_auction_price: bool = False,
    close_shift: float = 0.0,
) -> pd.DataFrame:
    times = (["09:30"] if auction else []) + list(EXPECTED_5M_BAR_ENDS)
    rows: list[dict[str, object]] = []
    for index, bar_time in enumerate(times):
        base = 10.0 + index / 100.0
        rows.append(
            {
                "日期": f"{trade_date} {bar_time}:00",
                "开盘": base,
                "最高": base + 0.2,
                "最低": base - 0.2,
                "收盘": base + 0.1 + (close_shift if bar_time == "15:00" else 0.0),
                "成交量(股)": float(100 + index),
                "成交额(元)": float(1_000 + index),
            }
        )
    if auction and zero_auction_price:
        for column in ("开盘", "最高", "最低", "收盘"):
            rows[0][column] = 0.0
    return pd.DataFrame(rows)


def _write_csv(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False, encoding="utf-8-sig")


def _write_zip(path: Path, members: dict[str, pd.DataFrame]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for name, frame in members.items():
            archive.writestr(name, frame.to_csv(index=False).encode("utf-8-sig"))


def test_normalize_merges_0930_into_0935_and_forces_fixed_schema() -> None:
    raw = _source_day("2010-01-04")
    auction = raw.iloc[0]
    first = raw.iloc[1]

    result = normalize_external_quant_5m(raw, provider_symbol="sh600000")

    assert list(result.columns) == RAW_COLUMNS
    assert len(result) == 48
    assert tuple(result["bar_time"]) == EXPECTED_5M_BAR_ENDS
    merged = result.iloc[0]
    assert merged["open"] == auction["开盘"]
    assert merged["high"] == max(auction["最高"], first["最高"])
    assert merged["low"] == min(auction["最低"], first["最低"])
    assert merged["close"] == first["收盘"]
    assert merged["volume"] == auction["成交量(股)"] + first["成交量(股)"]
    assert merged["amount"] == auction["成交额(元)"] + first["成交额(元)"]
    assert set(result["provider_symbol"]) == {"600000.SH"}
    assert all(str(result[column].dtype) == "float64" for column in ("open", "high", "low", "close", "volume", "amount"))


def test_zero_price_auction_does_not_overwrite_positive_0935_prices() -> None:
    raw = _source_day("2010-01-04", zero_auction_price=True)
    result = normalize_external_quant_5m(raw, provider_symbol="600000.SH")
    first = raw.iloc[1]

    assert result.iloc[0]["open"] == first["开盘"]
    assert result.iloc[0]["high"] == first["最高"]
    assert result.iloc[0]["low"] == first["最低"]
    assert result.iloc[0]["close"] == first["收盘"]


def test_already_48_is_idempotent_and_malformed_day_is_preserved() -> None:
    canonical = _source_day("2010-01-04", auction=False)
    normalized = normalize_external_quant_5m(canonical, provider_symbol="600000.SH")
    assert len(normalized) == 48
    assert tuple(normalized["bar_time"]) == EXPECTED_5M_BAR_ENDS

    missing_0935 = _source_day("2010-01-05").iloc[[0, *range(2, 49)]].copy()
    malformed = normalize_external_quant_5m(missing_0935, provider_symbol="600000.SH")
    assert len(malformed) == 48
    assert "09:30" in set(malformed["bar_time"])
    assert "09:35" not in set(malformed["bar_time"])

    duplicate = pd.concat([_source_day("2010-01-06"), _source_day("2010-01-06").iloc[[1]]])
    preserved = normalize_external_quant_5m(duplicate, provider_symbol="600000.SH")
    assert len(preserved) == 50
    assert int(preserved["bar_time"].eq("09:35").sum()) == 2


def test_scan_and_import_zip_record_gbk_utf8_hashes_and_resume(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    workspace = _workspace(tmp_path)
    source = tmp_path / "五分钟.zip"
    _write_zip(source, {"5分钟/sh600000.csv": _source_day("2010-01-04")})
    import quant_data_platform.qdp_v3.external_quant_5m as module

    real_zip = zipfile.ZipFile
    calls: list[dict[str, object]] = []

    def recording_zip(*args: object, **kwargs: object) -> zipfile.ZipFile:
        calls.append(dict(kwargs))
        return real_zip(*args, **kwargs)

    monkeypatch.setattr(module.zipfile, "ZipFile", recording_zip)
    indexed = scan_external_quant_5m_sources([source])
    assert indexed.symbols == ("600000.SH",)
    first = import_external_quant_5m(
        [source],
        workspace_root=workspace,
        start_date="2010-01-01",
        end_date="2026-07-13",
        min_available_gib=0,
        min_free_disk_gib=0,
    )
    second = import_external_quant_5m(
        [source],
        workspace_root=workspace,
        start_date="2010-01-01",
        end_date="2026-07-13",
        min_available_gib=0,
        min_free_disk_gib=0,
    )

    assert all(call.get("metadata_encoding") == "gbk" for call in calls)
    assert first.created_symbols == 1
    assert second.reused_symbols == 1
    receipt = read_raw_receipt(first.refs[0])
    assert receipt["quality_tier"] == "strict"
    assert receipt["coverage_start_date"] == "2010-01-04"
    assert receipt["coverage_end_date"] == "2010-01-04"
    assert receipt["requested_end_date"] == "2026-07-13"
    assert receipt["source_reference_url"].startswith("https://pan.baidu.com/")
    segment = receipt["source_segments"][0]
    assert len(segment["container_sha256"]) == 64
    assert len(segment["member_sha256"]) == 64
    assert len(segment["member_crc32"]) == 8
    assert segment["raw_row_count"] == 49
    assert segment["normalized_row_count"] == 48
    cache = read_json(
        workspace / "quant_data_platform" / "data" / "qdp_v3" / "jobs" / "external_quant_5m_container_hash_cache.json"
    )
    assert cache["entries"]
    job_files = list(
        (workspace / "quant_data_platform" / "data" / "qdp_v3" / "jobs").glob("external_quant_5m__*.json")
    )
    assert len(job_files) == 1
    job = read_json(job_files[0])
    assert job["status"] == "completed"
    assert job["completed"] == 1
    assert job["pending"] == 0


def test_directory_segments_combine_and_exact_overlap_deduplicates(tmp_path: Path) -> None:
    workspace = _workspace(tmp_path)
    first_dir = tmp_path / "first"
    second_dir = tmp_path / "second"
    first = pd.concat([_source_day("2025-12-31"), _source_day("2026-01-05")], ignore_index=True)
    second = pd.concat([_source_day("2026-01-05"), _source_day("2026-01-06")], ignore_index=True)
    _write_csv(first_dir / "sh600000.csv", first)
    _write_csv(second_dir / "sh600000.csv", second)

    result = import_external_quant_5m(
        [first_dir, second_dir],
        workspace_root=workspace,
        symbols=["600000.SH"],
        workers=2,
        min_available_gib=0,
        min_free_disk_gib=0,
    )

    assert result.completed_symbols == 1
    assert result.row_count == 3 * 48
    frame = read_raw_partition(result.refs[0])
    assert frame.groupby("trade_date").size().to_dict() == {
        "2025-12-31": 48,
        "2026-01-05": 48,
        "2026-01-06": 48,
    }
    receipt = read_raw_receipt(result.refs[0])
    assert receipt["coverage_start_date"] == "2025-12-31"
    assert receipt["coverage_end_date"] == "2026-01-06"
    assert len(receipt["source_segments"]) == 2


def test_cross_segment_overlap_conflict_is_rejected(tmp_path: Path) -> None:
    workspace = _workspace(tmp_path)
    first_dir = tmp_path / "first"
    second_dir = tmp_path / "second"
    _write_csv(first_dir / "sh600000.csv", _source_day("2026-01-05"))
    _write_csv(second_dir / "sh600000.csv", _source_day("2026-01-05", close_shift=1.0))

    with pytest.raises(ExternalQuant5mError, match="overlap_conflict"):
        import_external_quant_5m(
            [first_dir, second_dir],
            workspace_root=workspace,
            min_available_gib=0,
            min_free_disk_gib=0,
        )


def test_incomplete_day_does_not_fail_symbol_and_actual_coverage_excludes_tail(tmp_path: Path) -> None:
    workspace = _workspace(tmp_path)
    source_dir = tmp_path / "source"
    valid = _source_day("2026-01-05")
    incomplete = _source_day("2026-01-06").iloc[:-1].copy()
    valid_after_hole = _source_day("2026-01-07")
    _write_csv(
        source_dir / "sh600000.csv",
        pd.concat([valid, incomplete, valid_after_hole], ignore_index=True),
    )

    result = import_external_quant_5m(
        [source_dir],
        workspace_root=workspace,
        end_date="2026-07-13",
        min_available_gib=0,
        min_free_disk_gib=0,
    )

    assert result.completed_symbols == 1
    receipt = read_raw_receipt(result.refs[0])
    assert receipt["quality_tier"] == "strict"
    assert receipt["complete_stock_day_count"] == 2
    assert receipt["incomplete_stock_day_count"] == 1
    assert receipt["complete_trade_dates"] == ["2026-01-05", "2026-01-07"]
    assert receipt["incomplete_trade_dates"] == ["2026-01-06"]
    assert receipt["coverage_end_date"] == "2026-01-07"
    assert receipt["observed_max_trade_date"] == "2026-01-07"


def test_symbols_filter_and_max_symbols_are_deterministic(tmp_path: Path) -> None:
    workspace = _workspace(tmp_path)
    source = tmp_path / "source.zip"
    _write_zip(
        source,
        {
            "5分钟/sh600000.csv": _source_day("2010-01-04"),
            "5分钟/sz000001.csv": _source_day("2010-01-04"),
            "5分钟/bj920000.csv": _source_day("2010-01-04"),
        },
    )
    result = import_external_quant_5m(
        [source],
        workspace_root=workspace,
        symbols=["600000.SH", "000001.SZ"],
        max_symbols=1,
        workers=3,
        hash_containers=False,
        min_available_gib=0,
        min_free_disk_gib=0,
    )
    assert result.discovered_symbols == 3
    assert result.selected_symbols == 1
    assert [item.provider_symbol for item in result.results] == ["000001.SZ"]


def test_resource_guard_is_fatal_even_when_strict_is_false(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    workspace = _workspace(tmp_path)
    source = tmp_path / "source.zip"
    _write_zip(
        source,
        {
            "5分钟/sh600000.csv": _source_day("2010-01-04"),
            "5分钟/sz000001.csv": _source_day("2010-01-04"),
        },
    )
    import quant_data_platform.qdp_v3.external_quant_5m as module

    calls = 0

    def fail_worker_guard(*args: object, **kwargs: object) -> None:
        nonlocal calls
        calls += 1
        if calls >= 2:
            raise ExternalQuant5mResourceGuardError("unit_low_memory")

    monkeypatch.setattr(module, "_resource_guard", fail_worker_guard)
    with pytest.raises(ExternalQuant5mResourceGuardError, match="unit_low_memory"):
        import_external_quant_5m(
            [source],
            workspace_root=workspace,
            workers=1,
            strict=False,
            hash_containers=False,
            min_available_gib=0,
            min_free_disk_gib=0,
        )
    assert calls == 2
    jobs = list(
        (workspace / "quant_data_platform" / "data" / "qdp_v3" / "jobs").glob("external_quant_5m__*.json")
    )
    assert len(jobs) == 1
    assert read_json(jobs[0])["status"] == "paused_resource_guard"


def test_subrange_smoke_cannot_replace_existing_wider_latest_partition(tmp_path: Path) -> None:
    workspace = _workspace(tmp_path)
    source_dir = tmp_path / "source"
    full = pd.concat([_source_day("2026-01-05"), _source_day("2026-01-06")], ignore_index=True)
    _write_csv(source_dir / "sh600000.csv", full)

    first = import_external_quant_5m(
        [source_dir],
        workspace_root=workspace,
        start_date="2026-01-05",
        end_date="2026-01-06",
        min_available_gib=0,
        min_free_disk_gib=0,
    )
    second = import_external_quant_5m(
        [source_dir],
        workspace_root=workspace,
        start_date="2026-01-05",
        end_date="2026-01-05",
        min_available_gib=0,
        min_free_disk_gib=0,
    )

    assert first.row_count == 96
    assert second.reused_symbols == 1
    latest = read_raw_partition(second.refs[0])
    assert len(latest) == 96
    assert set(latest["trade_date"]) == {"2026-01-05", "2026-01-06"}
    receipt = read_raw_receipt(second.refs[0])
    assert receipt["coverage_start_date"] == "2026-01-05"
    assert receipt["coverage_end_date"] == "2026-01-06"
