from __future__ import annotations

import hashlib
import json
import sqlite3
import zipfile
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq
import pytest

from quant_data_platform.qdp_v2 import intraday_repair as repair_module
from quant_data_platform.qdp_v2.intraday_repair import (
    EXPECTED_5M_BAR_ENDS,
    IntradayRepairError,
    _normalize_complete_archive_day,
    bulk_append_intraday_5m_patch,
    plan_intraday_5m_archive_repairs,
    stable_symbol_bucket,
)
from quant_data_platform.qdp_v2.manifest import (
    DatasetManifest,
    ShardManifestEntry,
    atomic_write_json,
    qdp_v2_root,
    read_dataset_manifest,
    resolve_manifest_path,
    write_dataset_manifest,
)


DOMAIN = "market_intraday_5m"


def test_job_process_lock_is_exclusive_and_recoverable(tmp_path: Path) -> None:
    lock_path = tmp_path / "job.lock"
    first = repair_module._acquire_job_process_lock(lock_path)
    try:
        with pytest.raises(IntradayRepairError, match="intraday_repair_job_lock_held"):
            repair_module._acquire_job_process_lock(lock_path)
    finally:
        repair_module._release_job_process_lock(first)

    recovered = repair_module._acquire_job_process_lock(lock_path)
    repair_module._release_job_process_lock(recovered)


def _raw_day(
    trade_date: str,
    *,
    include_0930_placeholder: bool = False,
    traded_0930: bool = False,
    missing_bar: str = "",
    price_offset: float = 0.0,
) -> pd.DataFrame:
    bars = list(EXPECTED_5M_BAR_ENDS)
    if missing_bar:
        bars.remove(missing_bar)
    rows: list[dict[str, object]] = []
    if include_0930_placeholder:
        volume = 1.0 if traded_0930 else 0.0
        rows.append(
            {
                "日期": f"{trade_date} 09:30:00",
                "开盘": 10.0,
                "最高": 10.0,
                "最低": 10.0,
                "收盘": 10.0,
                "成交量(股)": volume,
                "成交额(元)": volume,
            }
        )
    for index, bar_end in enumerate(bars):
        base = 10.0 + price_offset + index / 100.0
        rows.append(
            {
                "日期": f"{trade_date} {bar_end}:00",
                "开盘": base,
                "最高": base + 0.2,
                "最低": base - 0.2,
                "收盘": base + 0.1,
                "成交量(股)": 1000.0 + index,
                "成交额(元)": 10000.0 + index,
            }
        )
    return pd.DataFrame(rows)


def _patch_day_for_direct_normalize(
    trade_date: str,
    *,
    traded_0930: bool,
) -> pd.DataFrame:
    raw = _raw_day(
        trade_date,
        include_0930_placeholder=True,
        traded_0930=traded_0930,
    )
    timestamps = pd.to_datetime(raw["日期"])
    return pd.DataFrame(
        {
            "trade_date": timestamps.dt.strftime("%Y-%m-%d"),
            "bar_end": timestamps.dt.strftime("%H:%M"),
            "open": raw["开盘"],
            "high": raw["最高"],
            "low": raw["最低"],
            "close": raw["收盘"],
            "volume": raw["成交量(股)"],
            "amount": raw["成交额(元)"],
        }
    )


def _write_archive(path: Path, members: dict[str, pd.DataFrame]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(path, mode="w", compression=zipfile.ZIP_DEFLATED) as archive:
        for member_name, frame in members.items():
            archive.writestr(
                member_name,
                frame.to_csv(index=False).encode("utf-8-sig"),
            )
    return path


def _write_active_fixture(workspace: Path) -> Path:
    root = workspace / "quant_data_platform" / "data" / "qdp_v2"
    dataset_id = "market_intraday_5m__repair_fixture"
    shard_root = root / "datasets" / DOMAIN / dataset_id / "shards"
    shard_root.mkdir(parents=True, exist_ok=True)
    all_bars = [bar.replace(":", "") + "00000" for bar in EXPECTED_5M_BAR_ENDS]
    # Split one complete stock-day across two shards.  The repair index must
    # aggregate the summaries before deciding that active already owns it.
    shards: list[ShardManifestEntry] = []
    for index, bars in enumerate((all_bars[:24], all_bars[24:])):
        path = shard_root / f"part-{index}.parquet"
        pd.DataFrame(
            {
                "symbol": ["600000.SH"] * len(bars),
                "trade_date": ["2020-01-02"] * len(bars),
                "bar_time": bars,
                "open": [10.0] * len(bars),
                "high": [10.2] * len(bars),
                "low": [9.8] * len(bars),
                "close": [10.1] * len(bars),
                "volume": [1000.0] * len(bars),
                "amount": [10000.0] * len(bars),
                "source": ["unit"] * len(bars),
                "adjusted_flag": ["none"] * len(bars),
            }
        ).to_parquet(path, index=False)
        shards.append(
            ShardManifestEntry(
                path=path.relative_to(root).as_posix(),
                row_count=len(bars),
                start_date="2020-01-02",
                end_date="2020-01-02",
                file_size=path.stat().st_size,
            )
        )
    write_dataset_manifest(
        root,
        DatasetManifest(
            dataset_id=dataset_id,
            domain=DOMAIN,
            layer="bronze_silver",
            frequency="5m",
            contract_version="mootdx_5m_48_v1",
            primary_key=["trade_date", "symbol", "bar_time"],
            start_date="2020-01-02",
            end_date="2020-01-02",
            row_count=48,
            schema_hash="unit",
            shards=shards,
            source={"provider": "unit"},
            quality={"bar_count_contract": "48"},
        ),
    )
    active_path = root / "active" / "active.json"
    atomic_write_json(
        active_path,
        {"version": 2, "datasets": {DOMAIN: dataset_id}},
    )
    return active_path


def _main_archive(path: Path) -> Path:
    sh600000 = pd.concat(
        [
            _raw_day("2020-01-02"),
            _raw_day("2020-01-03", include_0930_placeholder=True),
            _raw_day("2020-01-04", missing_bar="10:00"),
        ],
        ignore_index=True,
    )
    return _write_archive(
        path,
        {
            "sh600000.csv": sh600000,
            "sz000022.csv": _raw_day("2020-01-03"),
            "sz000043.csv": _raw_day("2020-01-03"),
            "sz300114.csv": _raw_day("2020-01-03"),
        },
    )


def test_0930_call_auction_is_merged_into_0935() -> None:
    zero, zero_reason = _normalize_complete_archive_day(
        _patch_day_for_direct_normalize("2020-01-03", traded_0930=False),
        canonical_symbol="600000.SH",
        trade_date="2020-01-03",
    )
    assert zero_reason == ""
    assert len(zero) == 48
    assert "093000000" not in set(zero["bar_time"])

    traded_day = _patch_day_for_direct_normalize("2020-01-03", traded_0930=True)
    auction = traded_day["bar_end"].eq("09:30")
    traded_day.loc[auction, ["open", "high", "low", "close"]] = 9.7
    nonzero, nonzero_reason = _normalize_complete_archive_day(
        traded_day,
        canonical_symbol="600000.SH",
        trade_date="2020-01-03",
    )
    assert nonzero_reason == ""
    assert len(nonzero) == 48
    first = nonzero.loc[nonzero["bar_time"].eq("093500000")].iloc[0]
    assert first["open"] == pytest.approx(9.7)
    assert first["high"] == pytest.approx(10.2)
    assert first["low"] == pytest.approx(9.7)
    assert first["close"] == pytest.approx(10.1)
    assert first["volume"] == pytest.approx(1001.0)
    assert first["amount"] == pytest.approx(10001.0)


def test_zero_turnover_stock_day_is_rejected() -> None:
    day = _patch_day_for_direct_normalize("2020-01-03", traded_0930=False)
    day.loc[:, ["volume", "amount"]] = 0.0
    normalized, reason = _normalize_complete_archive_day(
        day,
        canonical_symbol="600000.SH",
        trade_date="2020-01-03",
    )
    assert normalized.empty
    assert reason == "zero_turnover_stock_day"


def test_complete_day_records_explicit_trusted_source() -> None:
    normalized, reason = _normalize_complete_archive_day(
        _patch_day_for_direct_normalize("2020-01-03", traded_0930=False),
        canonical_symbol="600000.SH",
        trade_date="2020-01-03",
        source_name="tushare_proxy_5m_direct_gap_repair",
    )

    assert reason == ""
    assert normalized["source"].eq("tushare_proxy_5m_direct_gap_repair").all()


def test_planner_streams_year_bucket_bundles_and_reuses_receipts(
    tmp_path: Path,
    monkeypatch,
) -> None:
    workspace = tmp_path / "workspace"
    active_path = _write_active_fixture(workspace)
    active_before = active_path.read_bytes()
    archive = _main_archive(tmp_path / "archive.zip")
    output_root = workspace / "quant_data_platform" / "data" / "qdp_v2" / "repairs" / "unit"
    monkeypatch.setenv("QDP_RUNTIME_ROOT", r"C:\forbidden_qdp_runtime")
    kwargs = {
        "workspace_root": workspace,
        "archive_path": archive,
        "output_root": output_root,
        "start_date": "2020-01-01",
        "end_date": "2020-01-31",
        "symbols": ["600000.SH", "001872.SZ", "001914.SZ"],
        "dry_run": False,
        "csv_chunksize": 20,
        "parquet_row_group_rows": 96,
        "base_scan_batch_rows": 10,
        "memory_floor_gib": 0,
    }

    first = plan_intraday_5m_archive_repairs(**kwargs)
    assert first["status"] == "completed"
    assert first["active_mutated"] is False
    assert first["patch_stock_day_count"] == 3
    assert first["patch_row_count"] == 3 * 48
    assert first["existing_stock_days_skipped"] == 1
    assert first["rejected_stock_days"] == 1
    assert active_path.read_bytes() == active_before
    runtime_root = Path(first["runtime_root"])
    assert runtime_root.is_relative_to(
        workspace / "quant_data_platform" / "data" / "qdp_runtime" / "direct_repair"
    )
    assert not str(runtime_root).startswith(r"C:\forbidden_qdp_runtime")

    bundle_paths = [output_root / item["path"] for item in first["bundles"]]
    assert bundle_paths
    assert all(path.name == "part.parquet" for path in bundle_paths)
    assert all("year=2020" in path.as_posix() for path in bundle_paths)
    assert len(bundle_paths) == len({item["bucket"] for item in first["bundles"]})
    patch = pd.concat([pd.read_parquet(path) for path in bundle_paths], ignore_index=True)
    assert set(patch["symbol"]) == {
        "600000.SH",
        "001872.SZ",
        "001914.SZ",
    }
    assert set(patch.loc[patch["symbol"].eq("600000.SH"), "trade_date"]) == {"2020-01-03"}
    assert patch.groupby(["symbol", "trade_date"]).size().eq(48).all()
    assert not list(output_root.rglob("*600000*.parquet"))

    with sqlite3.connect(first["database_path"]) as connection:
        tables = {
            row[0]
            for row in connection.execute(
                "SELECT name FROM sqlite_master WHERE type = 'table'"
            )
        }
    assert {"metadata", "bucket_receipts", "symbol_receipts", "bundle_receipts"} <= tables
    assert {"patch_rows", "patch_days", "base_complete_days"}.isdisjoint(tables)

    validation = bulk_append_intraday_5m_patch(
        plan_manifest_path=first["patch_manifest_path"],
        workspace_root=workspace,
    )
    assert validation["status"] == "planned"
    assert validation["active_mutated"] is False

    hashes_before = {path: path.read_bytes() for path in bundle_paths}
    second = plan_intraday_5m_archive_repairs(**kwargs)
    assert second["status"] == "reused"
    assert second["patch_row_count"] == first["patch_row_count"]
    assert active_path.read_bytes() == active_before
    assert {path: path.read_bytes() for path in bundle_paths} == hashes_before
    with pytest.raises(IntradayRepairError, match="runtime_fingerprint_mismatch:output_root"):
        plan_intraday_5m_archive_repairs(
            **{**kwargs, "output_root": output_root / "different"}
        )


def test_dry_run_leaves_no_patch_or_persistent_runtime(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    active_path = _write_active_fixture(workspace)
    active_before = active_path.read_bytes()
    archive = _write_archive(
        tmp_path / "archive.zip",
        {"sh600000.csv": _raw_day("2020-01-03")},
    )
    output_root = workspace / "planned-output"
    result = plan_intraday_5m_archive_repairs(
        workspace_root=workspace,
        archive_path=archive,
        output_root=output_root,
        start_date="2020-01-01",
        end_date="2020-01-31",
        symbols=["600000.SH"],
        dry_run=True,
        csv_chunksize=20,
        base_scan_batch_rows=10,
        memory_floor_gib=0,
    )
    assert result["status"] == "planned"
    assert result["patch_row_count"] == 48
    assert result["state_path"] == ""
    assert result["database_path"] == ""
    assert not output_root.exists()
    assert active_path.read_bytes() == active_before


def test_eligible_universe_is_not_silently_reduced_to_archive_members(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    _write_active_fixture(workspace)
    archive = _write_archive(
        tmp_path / "archive.zip",
        {"sh600000.csv": _raw_day("2020-01-03")},
    )
    result = plan_intraday_5m_archive_repairs(
        workspace_root=workspace,
        archive_path=archive,
        output_root=workspace / "planned-output",
        start_date="2020-01-01",
        end_date="2020-01-31",
        eligible_symbols=["600000.SH", "601999.SH"],
        dry_run=True,
        csv_chunksize=20,
        base_scan_batch_rows=10,
        memory_floor_gib=0,
    )
    assert result["target_symbol_count"] == 2
    assert result["archive_selected_symbol_count"] == 1
    assert result["selected_symbol_count"] == 1
    assert result["eligible_missing_archive_count"] == 1
    assert result["eligible_missing_archive"] == ["601999.SH"]


def test_bundle_schema_has_only_canonical_patch_columns(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    _write_active_fixture(workspace)
    archive = _write_archive(
        tmp_path / "archive.zip",
        {"sz000022.csv": _raw_day("2020-01-03")},
    )
    output_root = workspace / "patch"
    result = plan_intraday_5m_archive_repairs(
        workspace_root=workspace,
        archive_path=archive,
        output_root=output_root,
        start_date="2020-01-01",
        end_date="2020-01-31",
        symbols=["001872.SZ"],
        dry_run=False,
        csv_chunksize=20,
        base_scan_batch_rows=10,
        memory_floor_gib=0,
    )
    parquet = pq.ParquetFile(output_root / result["bundles"][0]["path"])
    assert parquet.metadata.num_rows == 48
    assert parquet.schema_arrow.names == [
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
    ]


def test_atomic_replace_retries_a_transient_windows_sharing_violation(
    tmp_path: Path,
    monkeypatch,
) -> None:
    source = tmp_path / "source.tmp"
    target = tmp_path / "target.parquet"
    source.write_bytes(b"complete")
    original_replace = repair_module.os.replace
    attempts = 0

    def flaky_replace(left, right) -> None:
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            raise PermissionError("simulated Windows sharing violation")
        original_replace(left, right)

    monkeypatch.setattr(repair_module.os, "replace", flaky_replace)
    repair_module._replace_file_with_retry(source, target, attempts=2)
    assert attempts == 2
    assert target.read_bytes() == b"complete"


def test_incomplete_active_day_is_reported_and_never_appended_over(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    _write_active_fixture(workspace)
    shard = next(
        (
            workspace
            / "quant_data_platform"
            / "data"
            / "qdp_v2"
            / "datasets"
            / DOMAIN
            / "market_intraday_5m__repair_fixture"
            / "shards"
        ).glob("part-1.parquet")
    )
    pd.read_parquet(shard).iloc[:-1].to_parquet(shard, index=False)
    archive = _write_archive(
        tmp_path / "archive.zip",
        {"sh600000.csv": _raw_day("2020-01-02")},
    )
    result = plan_intraday_5m_archive_repairs(
        workspace_root=workspace,
        archive_path=archive,
        output_root=workspace / "planned-output",
        start_date="2020-01-01",
        end_date="2020-01-31",
        symbols=["600000.SH"],
        dry_run=True,
        csv_chunksize=20,
        memory_floor_gib=0,
    )
    assert result["patch_row_count"] == 0
    assert result["existing_stock_days_skipped"] == 0
    assert result["incomplete_existing_stock_days_skipped"] == 1


def test_bad_member_isolated_without_discarding_other_symbols_in_bucket(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    _write_active_fixture(workspace)
    bucket = stable_symbol_bucket("600000.SH")
    bad_symbol = next(
        f"{code:06d}.SH"
        for code in range(600001, 601000)
        if stable_symbol_bucket(f"{code:06d}.SH") == bucket
    )
    archive = _write_archive(
        tmp_path / "archive.zip",
        {
            "sh600000.csv": _raw_day("2020-01-03"),
            f"sh{bad_symbol[:6]}.csv": pd.DataFrame({"broken": [1, 2, 3]}),
        },
    )
    result = plan_intraday_5m_archive_repairs(
        workspace_root=workspace,
        archive_path=archive,
        output_root=workspace / "patch",
        start_date="2020-01-01",
        end_date="2020-01-31",
        symbols=["600000.SH", bad_symbol],
        dry_run=False,
        csv_chunksize=20,
        memory_floor_gib=0,
    )
    assert result["status"] == "completed_with_failures"
    assert result["patch_row_count"] == 48
    assert result["completed_symbols"] == 1
    assert result["failed_symbols"][0]["symbol"] == bad_symbol
    patch = pd.concat(
        [pd.read_parquet(Path(result["output_root"]) / item["path"]) for item in result["bundles"]],
        ignore_index=True,
    )
    assert set(patch["symbol"]) == {"600000.SH"}


def test_alias_members_with_conflicting_same_day_are_rejected_whole(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    _write_active_fixture(workspace)
    archive = _write_archive(
        tmp_path / "archive.zip",
        {
            "sz000022.csv": _raw_day("2020-01-03"),
            "sz001872.csv": _raw_day("2020-01-03", price_offset=1.0),
        },
    )
    result = plan_intraday_5m_archive_repairs(
        workspace_root=workspace,
        archive_path=archive,
        output_root=workspace / "planned-output",
        start_date="2020-01-01",
        end_date="2020-01-31",
        symbols=["001872.SZ"],
        dry_run=True,
        csv_chunksize=20,
        memory_floor_gib=0,
    )
    assert result["patch_row_count"] == 0
    assert result["rejected_stock_days"] == 1
    assert result["conflict_stock_days"] == 1


def _single_day_plan(workspace: Path, tmp_path: Path) -> dict[str, object]:
    archive = _write_archive(
        tmp_path / "apply-archive.zip",
        {"sh600000.csv": _raw_day("2020-01-03")},
    )
    return plan_intraday_5m_archive_repairs(
        workspace_root=workspace,
        archive_path=archive,
        output_root=(
            workspace
            / "quant_data_platform"
            / "data"
            / "qdp_v2"
            / "repairs"
            / "apply-unit"
        ),
        start_date="2020-01-01",
        end_date="2020-01-31",
        symbols=["600000.SH"],
        dry_run=False,
        csv_chunksize=20,
        parquet_row_group_rows=96,
        base_scan_batch_rows=10,
        memory_floor_gib=0,
    )


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _rewrite_first_bundle(
    plan: dict[str, object],
    transform,
) -> Path:
    manifest_path = Path(str(plan["patch_manifest_path"]))
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    record = payload["bundles"][0]
    bundle = manifest_path.parent / record["path"]
    frame = transform(pd.read_parquet(bundle))
    frame.to_parquet(bundle, index=False)
    record["file_size"] = bundle.stat().st_size
    record["sha256"] = _sha256(bundle)
    atomic_write_json(manifest_path, payload)
    return manifest_path


def test_bulk_append_execute_commits_once_and_replays_idempotently(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    workspace = tmp_path / "workspace"
    active_path = _write_active_fixture(workspace)
    active_before = active_path.read_bytes()
    plan = _single_day_plan(workspace, tmp_path)
    root = qdp_v2_root(workspace)
    dataset_id = "market_intraday_5m__repair_fixture"
    manifest_path = root / "datasets" / DOMAIN / dataset_id / "dataset.json"
    base_manifest = read_dataset_manifest(manifest_path)

    import quant_data_platform.qdp_v2.repair as qdp_repair

    original_commit = qdp_repair._commit_manifest
    commit_calls = 0

    def counted_commit(*args, **kwargs) -> None:
        nonlocal commit_calls
        commit_calls += 1
        assert active_path.read_bytes() == active_before
        original_commit(*args, **kwargs)

    monkeypatch.setattr(qdp_repair, "_commit_manifest", counted_commit)
    first = bulk_append_intraday_5m_patch(
        plan_manifest_path=str(plan["patch_manifest_path"]),
        workspace_root=workspace,
        execute=True,
    )
    assert first["status"] == "applied"
    assert first["active_mutated"] is True
    assert first["active_pointer_mutated"] is False
    assert first["dataset_id_unchanged"] is True
    assert first["appended_bundle_count"] == 1
    assert commit_calls == 1
    assert active_path.read_bytes() == active_before

    updated = read_dataset_manifest(manifest_path)
    assert updated.dataset_id == dataset_id
    assert updated.row_count == base_manifest.row_count + 48
    assert len(updated.shards) == len(base_manifest.shards) + 1
    appended = [
        item
        for item in updated.shards
        if Path(resolve_manifest_path(item.path, root=root)).name.startswith(
            "repair_mutate_append_"
        )
    ]
    assert len(appended) == 1
    appended_frame = pd.read_parquet(resolve_manifest_path(appended[0].path, root=root))
    assert appended_frame.groupby(["symbol", "trade_date"]).size().tolist() == [48]
    assert set(appended_frame["trade_date"]) == {"2020-01-03"}
    assert all(Path(item).is_relative_to(workspace) for item in first["installed_shard_paths"])

    manifest_after_first = manifest_path.read_bytes()
    second = bulk_append_intraday_5m_patch(
        plan_manifest_path=str(plan["patch_manifest_path"]),
        workspace_root=workspace,
        execute=True,
    )
    assert second["status"] == "reused"
    assert second["mutation_id"] == first["mutation_id"]
    assert second["reused_bundle_count"] == 1
    assert commit_calls == 1
    assert manifest_path.read_bytes() == manifest_after_first
    assert active_path.read_bytes() == active_before


@pytest.mark.parametrize("tamper", ["duplicate_bar", "base_stock_day"])
def test_bulk_append_rejects_non_48_or_any_base_stock_day_overlap(
    tmp_path: Path,
    tamper: str,
) -> None:
    workspace = tmp_path / "workspace"
    active_path = _write_active_fixture(workspace)
    active_before = active_path.read_bytes()
    plan = _single_day_plan(workspace, tmp_path)
    root = qdp_v2_root(workspace)
    manifest_path = (
        root
        / "datasets"
        / DOMAIN
        / "market_intraday_5m__repair_fixture"
        / "dataset.json"
    )
    manifest_before = manifest_path.read_bytes()

    def transform(frame: pd.DataFrame) -> pd.DataFrame:
        changed = frame.copy()
        if tamper == "duplicate_bar":
            changed.loc[changed.index[-1], "bar_time"] = changed.loc[
                changed.index[-2], "bar_time"
            ]
        else:
            changed["trade_date"] = "2020-01-02"
        return changed

    patch_manifest = _rewrite_first_bundle(plan, transform)
    expected = (
        "requires_exact_48_bar_days"
        if tamper == "duplicate_bar"
        else "overlaps_any_frozen_base_stock_day"
    )
    with pytest.raises(IntradayRepairError, match=expected):
        bulk_append_intraday_5m_patch(
            plan_manifest_path=patch_manifest,
            workspace_root=workspace,
            execute=True,
        )
    assert active_path.read_bytes() == active_before
    assert manifest_path.read_bytes() == manifest_before
    shard_dir = manifest_path.parent / "shards"
    assert not list(shard_dir.glob("repair_mutate_append_*.parquet"))


def test_bulk_append_rejects_stale_version_and_invalid_eligible_gap_report(
    tmp_path: Path,
) -> None:
    workspace = tmp_path / "workspace"
    _write_active_fixture(workspace)
    plan = _single_day_plan(workspace, tmp_path)
    manifest_path = Path(str(plan["patch_manifest_path"]))
    original = json.loads(manifest_path.read_text(encoding="utf-8"))

    stale = dict(original)
    stale["repair_plan_version"] = int(original["repair_plan_version"]) - 1
    atomic_write_json(manifest_path, stale)
    with pytest.raises(IntradayRepairError, match="patch_manifest_invalid"):
        bulk_append_intraday_5m_patch(
            plan_manifest_path=manifest_path,
            workspace_root=workspace,
        )

    invalid_report = dict(original)
    invalid_report["eligible_missing_archive_count"] = 1
    atomic_write_json(manifest_path, invalid_report)
    with pytest.raises(IntradayRepairError, match="eligible_gap_count_mismatch"):
        bulk_append_intraday_5m_patch(
            plan_manifest_path=manifest_path,
            workspace_root=workspace,
        )


@pytest.mark.parametrize(
    ("artifact", "expected"),
    [
        ("base_index", "base_day_index_invalid"),
        ("bundle", "bundle_validation_failed"),
    ],
)
def test_bulk_append_rejects_index_or_bundle_bit_rot(
    tmp_path: Path,
    artifact: str,
    expected: str,
) -> None:
    workspace = tmp_path / "workspace"
    active_path = _write_active_fixture(workspace)
    active_before = active_path.read_bytes()
    plan = _single_day_plan(workspace, tmp_path)
    patch_manifest = Path(str(plan["patch_manifest_path"]))
    payload = json.loads(patch_manifest.read_text(encoding="utf-8"))
    if artifact == "base_index":
        index_manifest = Path(payload["base_day_index_manifest"])
        index = json.loads(index_manifest.read_text(encoding="utf-8"))
        target = index_manifest.parent / index["files"][0]["path"]
    else:
        target = patch_manifest.parent / payload["bundles"][0]["path"]
    with target.open("ab") as handle:
        handle.write(b"bit-rot")

    with pytest.raises(IntradayRepairError, match=expected):
        bulk_append_intraday_5m_patch(
            plan_manifest_path=patch_manifest,
            workspace_root=workspace,
            execute=True,
        )
    assert active_path.read_bytes() == active_before
