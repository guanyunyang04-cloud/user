from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from pandas.testing import assert_frame_equal

from quant_data_platform.core.json_io import read_json
from quant_data_platform.qdp_v2.manifest import (
    DatasetManifest,
    ShardManifestEntry,
    qdp_v2_root,
    write_active_manifest,
    write_dataset_manifest,
)
from quant_data_platform.qdp_v3.constants import (
    EXPECTED_5M_BAR_ENDS,
    RAW_LEGACY_V2_INTRADAY_5M,
)
from quant_data_platform.qdp_v3.external_quant_5m import RAW_COLUMNS
from quant_data_platform.qdp_v3.legacy_v2_5m import (
    LegacyV2FiveMinuteError,
    LegacyV2FiveMinuteResourceGuardError,
    migrate_v2_active_intraday_5m,
)
from quant_data_platform.qdp_v3.manifest import sha256_file
from quant_data_platform.qdp_v3.paths import qdp_v3_paths
from quant_data_platform.qdp_v3.storage import read_raw_partition, read_raw_receipt


def _workspace(tmp_path: Path) -> Path:
    workspace = tmp_path / "workspace"
    manifest = workspace / "brain" / "brain_manifest.json"
    manifest.parent.mkdir(parents=True, exist_ok=True)
    manifest.write_text(
        json.dumps({"schema_version": 1, "brain_type": "main"}),
        encoding="utf-8",
    )
    return workspace


def _day(
    symbol: str,
    trade_date: str,
    *,
    adjusted_flag: str = "none",
    bars: int = 48,
    duplicate_first: bool = False,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for index, bar_time in enumerate(EXPECTED_5M_BAR_ENDS[:bars]):
        base = 10.0 + index / 100.0
        rows.append(
            {
                "symbol": symbol,
                "trade_date": trade_date,
                "bar_time": bar_time.replace(":", "") + "00000",
                "open": base,
                "high": base + 0.2,
                "low": base - 0.2,
                "close": base + 0.1,
                "volume": float(100 + index),
                "amount": float(1_000 + index),
                "source": "qdp_v2_fixture",
                "adjusted_flag": adjusted_flag,
            }
        )
    if duplicate_first:
        rows.append(dict(rows[0]))
    return pd.DataFrame(rows)


def _install_v2_active(
    workspace: Path,
    frames: list[pd.DataFrame],
    *,
    parent_dataset_id: str = "market_intraday_5m__missing_parent",
) -> tuple[Path, Path, list[Path]]:
    root = qdp_v2_root(workspace)
    dataset_id = "market_intraday_5m__active_fixture"
    shard_entries: list[ShardManifestEntry] = []
    shard_paths: list[Path] = []
    for index, frame in enumerate(frames):
        shard = (
            root
            / "datasets"
            / "market_intraday_5m"
            / dataset_id
            / "shards"
            / f"part_{index:06d}.parquet"
        )
        shard.parent.mkdir(parents=True, exist_ok=True)
        frame.to_parquet(shard, index=False, engine="pyarrow")
        shard_paths.append(shard)
        shard_entries.append(
            ShardManifestEntry(
                path=str(shard.relative_to(root)).replace("\\", "/"),
                row_count=len(frame),
                start_date=str(frame["trade_date"].min()),
                end_date=str(frame["trade_date"].max()),
                file_size=shard.stat().st_size,
            )
        )
    manifest = DatasetManifest(
        dataset_id=dataset_id,
        domain="market_intraday_5m",
        layer="raw",
        frequency="5m",
        contract_version="mootdx_5m_48_v1",
        primary_key=["trade_date", "symbol", "bar_time"],
        start_date=min(item.start_date for item in shard_entries),
        end_date=max(item.end_date for item in shard_entries),
        row_count=sum(item.row_count for item in shard_entries),
        schema_hash="fixture",
        shards=shard_entries,
        source={"provider": "qdp_v2", "source_dataset_id": parent_dataset_id},
        quality={"primary_key_unique": True, "bar_count_contract": "48"},
        schema=[
            {"name": column, "type": str(frames[0][column].dtype)}
            for column in frames[0].columns
        ],
    )
    manifest_path = write_dataset_manifest(root, manifest)
    active_path = write_active_manifest(
        root,
        {
            "version": 2,
            "active_as_of_date": manifest.end_date,
            "datasets": {"market_intraday_5m": dataset_id},
        },
    )
    return active_path, manifest_path, shard_paths


def _reference_validated_complete_days(
    staged: pd.DataFrame, provider_symbol: str
) -> tuple[pd.DataFrame, tuple[str, ...], dict[str, tuple[str, ...]]]:
    """Pre-vectorization implementation retained only as a regression oracle."""

    import quant_data_platform.qdp_v3.legacy_v2_5m as module

    data = staged.copy()
    parsed_dates = pd.to_datetime(data["trade_date"], errors="coerce")
    if parsed_dates.isna().any():
        raise ValueError("invalid fixture date")
    data["trade_date"] = parsed_dates.dt.strftime("%Y-%m-%d")
    data["bar_time"] = data["bar_time"].map(module._normalize_bar_time)
    for column in module._NUMERIC_COLUMNS:
        data[column] = pd.to_numeric(data[column], errors="coerce")
    data["adjusted_flag"] = data["adjusted_flag"].astype("string")
    complete_frames: list[pd.DataFrame] = []
    complete_dates: list[str] = []
    rejected: dict[str, tuple[str, ...]] = {}
    expected = set(EXPECTED_5M_BAR_ENDS)
    for trade_date, day in data.groupby("trade_date", sort=True, dropna=False):
        reasons: list[str] = []
        keys = day[["trade_date", "bar_time"]]
        if keys.duplicated(keep=False).any():
            reasons.append("primary_key_duplicate")
        times = tuple(sorted(day["bar_time"].dropna().astype(str)))
        if len(day) != 48 or len(set(times)) != 48 or set(times) != expected:
            reasons.append("bar_time_set_not_exact_48")
        adjusted = day["adjusted_flag"].str.strip().str.lower()
        if adjusted.isna().any() or not adjusted.eq("none").all():
            reasons.append("adjusted_flag_not_none")
        numeric = day[list(module._NUMERIC_COLUMNS)].to_numpy(dtype="float64")
        if not np.isfinite(numeric).all():
            reasons.append("numeric_non_finite")
        else:
            if day[list(module._PRICE_COLUMNS)].le(0).any().any():
                reasons.append("price_not_positive")
            if day[["volume", "amount"]].lt(0).any().any():
                reasons.append("volume_or_amount_negative")
            if (
                day["high"] < day[["open", "low", "close"]].max(axis=1)
            ).any():
                reasons.append("high_relation_invalid")
            if (
                day["low"] > day[["open", "high", "close"]].min(axis=1)
            ).any():
                reasons.append("low_relation_invalid")
        date_text = str(trade_date)
        if reasons:
            rejected[date_text] = tuple(dict.fromkeys(reasons))
            continue
        normalized = day.copy()
        normalized["provider_symbol"] = provider_symbol
        normalized["source"] = module.SOURCE_NAME
        normalized = normalized.loc[:, RAW_COLUMNS]
        normalized = normalized.sort_values("bar_time", kind="stable")
        complete_frames.append(normalized)
        complete_dates.append(date_text)
    if complete_frames:
        complete = pd.concat(complete_frames, ignore_index=True)
        complete = complete.sort_values(
            ["trade_date", "bar_time"], kind="stable"
        ).reset_index(drop=True)
        for column in module._NUMERIC_COLUMNS:
            complete[column] = complete[column].astype("float64")
        for column in ("provider_symbol", "trade_date", "bar_time", "source"):
            complete[column] = complete[column].astype("string")
    else:
        complete = module._empty_raw_frame()
    return complete, tuple(complete_dates), rejected


def _large_valid_fixture(days: int) -> pd.DataFrame:
    dates = pd.bdate_range("2015-01-05", periods=days).strftime("%Y-%m-%d")
    rows = days * 48
    offsets = np.tile(np.arange(48, dtype="float64"), days) / 100.0
    base = 10.0 + offsets
    return pd.DataFrame(
        {
            "provider_symbol": np.repeat("600000.SH", rows),
            "trade_date": np.repeat(dates.to_numpy(), 48),
            "bar_time": np.tile(
                [item.replace(":", "") + "00000" for item in EXPECTED_5M_BAR_ENDS],
                days,
            ),
            "open": base,
            "high": base + 0.2,
            "low": base - 0.2,
            "close": base + 0.1,
            "volume": np.tile(np.arange(100, 148, dtype="float64"), days),
            "amount": np.tile(np.arange(1_000, 1_048, dtype="float64"), days),
            "adjusted_flag": np.repeat("none", rows),
            "_v2_shard_path": np.repeat("fixture.parquet", rows),
        }
    )


def test_migrates_only_complete_unadjusted_days_and_preserves_v2(tmp_path: Path) -> None:
    workspace = _workspace(tmp_path)
    first = pd.concat(
        [
            _day("600000.SH", "2026-01-05"),
            _day("600000.SH", "2026-01-06", bars=47),
        ],
        ignore_index=True,
    )
    second = _day("000001.SZ", "2026-01-05")
    active_path, manifest_path, shards = _install_v2_active(
        workspace, [first, second]
    )
    # A valid-looking unlisted file beside the active dataset must never be read.
    decoy = manifest_path.parent / "shards" / "unlisted.parquet"
    _day("300001.SZ", "2026-01-05").to_parquet(decoy, index=False)
    immutable_hashes = {
        path: sha256_file(path) for path in (active_path, manifest_path, *shards, decoy)
    }

    result = migrate_v2_active_intraday_5m(
        workspace_root=workspace,
        start_date="2026-01-01",
        end_date="2026-01-31",
        workers=2,
        min_free_disk_gib=0,
    )

    assert result.status == "completed"
    assert result.discovered_symbols == 2
    assert result.selected_symbols == 2
    assert result.created_symbols == 2
    assert result.failed_symbols == 0
    assert result.row_count == 96
    assert result.day_count == 2
    assert result.rejected_day_count == 1
    assert {item.provider_symbol for item in result.results} == {
        "000001.SZ",
        "600000.SH",
    }
    assert "300001.SZ" not in {item.provider_symbol for item in result.results}

    by_symbol = {item.provider_symbol: item for item in result.results}
    migrated = read_raw_partition(by_symbol["600000.SH"].ref)
    assert list(migrated.columns) == RAW_COLUMNS
    assert len(migrated) == 48
    assert set(migrated["provider_symbol"]) == {"600000.SH"}
    assert set(migrated["source"]) == {"qdp_v2_migration"}
    assert tuple(migrated["bar_time"]) == EXPECTED_5M_BAR_ENDS
    receipt = read_raw_receipt(by_symbol["600000.SH"].ref)
    assert receipt["v2_dataset_id"] == "market_intraday_5m__active_fixture"
    assert receipt["v2_active_manifest_sha256"] == immutable_hashes[active_path]
    assert receipt["v2_dataset_manifest_sha256"] == immutable_hashes[manifest_path]
    assert receipt["complete_trade_dates"] == ["2026-01-05"]
    assert receipt["rejected_trade_dates"] == ["2026-01-06"]
    assert receipt["rejected_day_reasons"]["2026-01-06"] == [
        "bar_time_set_not_exact_48"
    ]
    assert receipt["missing_parent_lineage"] is True
    assert receipt["missing_parent_dataset_ids"] == [
        "market_intraday_5m__missing_parent"
    ]
    assert len(receipt["v2_relevant_shards"]) == 1
    assert receipt["v2_relevant_shards"][0]["sha256"] == immutable_hashes[shards[0]]
    assert all(sha256_file(path) == digest for path, digest in immutable_hashes.items())
    assert not list(
        qdp_v3_paths(workspace).staging.glob("legacy_v2_5m__*")
    )

    resumed = migrate_v2_active_intraday_5m(
        workspace_root=workspace,
        start_date="2026-01-01",
        end_date="2026-01-31",
        workers=2,
        min_free_disk_gib=0,
    )
    assert resumed.status == "completed"
    assert resumed.reused_symbols == 2
    assert resumed.created_symbols == 0


def test_vectorized_day_validation_is_contract_equivalent_to_reference() -> None:
    import quant_data_platform.qdp_v3.legacy_v2_5m as module

    valid = _day("600000.SH", "2026-01-05")
    short = _day("600000.SH", "2026-01-06", bars=47)
    adjusted = _day("600000.SH", "2026-01-07", adjusted_flag="front")
    duplicate = _day("600000.SH", "2026-01-08", duplicate_first=True)
    non_finite = _day("600000.SH", "2026-01-09")
    non_finite.loc[0, "open"] = np.nan
    bad_price = _day("600000.SH", "2026-01-12")
    bad_price.loc[0, "open"] = 0.0
    bad_volume = _day("600000.SH", "2026-01-13")
    bad_volume.loc[0, "volume"] = -1.0
    bad_relations = _day("600000.SH", "2026-01-14")
    bad_relations.loc[0, "high"] = bad_relations.loc[0, "close"] - 1.0
    bad_relations.loc[1, "low"] = bad_relations.loc[1, "close"] + 1.0
    staged = pd.concat(
        [
            valid,
            short,
            adjusted,
            duplicate,
            non_finite,
            bad_price,
            bad_volume,
            bad_relations,
        ],
        ignore_index=True,
    ).rename(columns={"symbol": "provider_symbol"})
    staged["_v2_shard_path"] = "fixture.parquet"

    expected_frame, expected_dates, expected_rejected = (
        _reference_validated_complete_days(staged, "600000.SH")
    )
    actual_frame, actual_dates, actual_rejected = module._validated_complete_days(
        staged, provider_symbol="600000.SH"
    )

    assert_frame_equal(actual_frame, expected_frame)
    assert actual_dates == expected_dates
    assert actual_rejected == expected_rejected


def test_vectorized_day_validation_has_large_fixture_speedup() -> None:
    import quant_data_platform.qdp_v3.legacy_v2_5m as module

    staged = _large_valid_fixture(800)
    reference_started = time.perf_counter()
    reference = _reference_validated_complete_days(staged, "600000.SH")
    reference_elapsed = time.perf_counter() - reference_started

    vectorized_started = time.perf_counter()
    vectorized = module._validated_complete_days(
        staged, provider_symbol="600000.SH"
    )
    vectorized_elapsed = time.perf_counter() - vectorized_started

    assert_frame_equal(vectorized[0], reference[0])
    assert vectorized[1:] == reference[1:]
    # A relative ratio is intentionally used instead of a wall-clock ceiling:
    # it remains stable on slower or contended machines while catching a
    # regression back to Python iteration over every complete stock-day.
    assert vectorized_elapsed * 3 < reference_elapsed, (
        f"expected >=3x speedup; vectorized={vectorized_elapsed:.4f}s "
        f"reference={reference_elapsed:.4f}s"
    )


def test_rejects_adjusted_and_duplicate_days_without_writing_invalid_raw(
    tmp_path: Path,
) -> None:
    workspace = _workspace(tmp_path)
    frame = pd.concat(
        [
            _day("600000.SH", "2026-01-05", adjusted_flag="front"),
            _day("600000.SH", "2026-01-06", duplicate_first=True),
        ],
        ignore_index=True,
    )
    _install_v2_active(workspace, [frame])

    result = migrate_v2_active_intraday_5m(
        workspace_root=workspace,
        workers=1,
        min_free_disk_gib=0,
    )

    assert result.status == "completed"
    assert result.empty_symbols == 1
    assert result.row_count == 0
    assert result.rejected_day_count == 2
    assert result.refs == ()
    assert not (
        qdp_v3_paths(workspace).raw / RAW_LEGACY_V2_INTRADAY_5M
    ).exists()


def test_resource_guard_records_resumable_pause(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    workspace = _workspace(tmp_path)
    _install_v2_active(workspace, [_day("600000.SH", "2026-01-05")])
    import quant_data_platform.qdp_v3.legacy_v2_5m as module

    def blocked(*args: object, **kwargs: object) -> None:
        raise LegacyV2FiveMinuteResourceGuardError("unit_low_memory")

    monkeypatch.setattr(module, "_resource_guard", blocked)
    with pytest.raises(
        LegacyV2FiveMinuteResourceGuardError, match="unit_low_memory"
    ):
        migrate_v2_active_intraday_5m(
            workspace_root=workspace,
            workers=1,
            min_free_disk_gib=0,
        )
    jobs = list(qdp_v3_paths(workspace).jobs.glob("legacy_v2_5m__*.json"))
    assert len(jobs) == 1
    job = read_json(jobs[0])
    assert job["status"] == "paused_resource_guard"
    assert job["pause_reason"] == "resource_guard"


def test_staging_volume_has_an_independent_free_space_guard(tmp_path: Path) -> None:
    workspace = _workspace(tmp_path)
    _install_v2_active(workspace, [_day("600000.SH", "2026-01-05")])

    with pytest.raises(
        LegacyV2FiveMinuteResourceGuardError, match="legacy_v2_5m_low_disk"
    ):
        migrate_v2_active_intraday_5m(
            workspace_root=workspace,
            workers=1,
            min_free_disk_gib=0,
            min_staging_free_disk_gib=1_000_000,
        )
    jobs = list(qdp_v3_paths(workspace).jobs.glob("legacy_v2_5m__*.json"))
    assert len(jobs) == 1
    assert read_json(jobs[0])["status"] == "paused_resource_guard"


def test_refuses_manifest_shards_that_change_before_migration(tmp_path: Path) -> None:
    workspace = _workspace(tmp_path)
    _, _, shards = _install_v2_active(
        workspace, [_day("600000.SH", "2026-01-05")]
    )
    # The manifest file-size is part of the strict source snapshot.
    shards[0].write_bytes(shards[0].read_bytes() + b"corruption")

    with pytest.raises(LegacyV2FiveMinuteError, match="shard_size_mismatch"):
        migrate_v2_active_intraday_5m(
            workspace_root=workspace,
            workers=1,
            min_free_disk_gib=0,
        )


def test_cli_routes_explicit_v2_migration_provider(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    workspace = _workspace(tmp_path)
    import quant_data_platform.qdp_v3.cli as cli

    captured: dict[str, object] = {}

    class Result:
        def to_dict(self) -> dict[str, object]:
            return {"status": "completed", "selected_symbols": 1}

    def migrate(**kwargs: object) -> Result:
        captured.update(kwargs)
        return Result()

    monkeypatch.setattr(cli, "migrate_v2_active_intraday_5m", migrate)
    status = cli.dispatch(
        [
            "ingest",
            "--workspace-root",
            str(workspace),
            "--provider",
            "qdp-v2-migration",
            "--mode",
            "historical",
            "--domain",
            "intraday",
            "--start-date",
            "2011-11-22",
            "--end-date",
            "2026-06-26",
            "--symbol",
            "600000.SH",
            "--max-workers",
            "8",
            "--json",
        ]
    )

    assert status == 0
    assert captured["workspace_root"] == str(workspace)
    assert captured["symbols"] == ("600000.SH",)
    assert captured["workers"] == 8
    assert json.loads(capsys.readouterr().out)["status"] == "completed"
