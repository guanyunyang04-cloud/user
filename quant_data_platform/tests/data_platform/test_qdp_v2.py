from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd
from quant_data_platform.core.paths import workspace_root
from quant_data_platform.qdp_v2.audit import audit_active
from quant_data_platform.qdp_v2.check import run_check
from quant_data_platform.qdp_v2.database_audit import (
    EXPECTED_BAR_TIMES,
    REQUIRED_DOMAINS,
    _active_manifest,
    _audit_temp_directory,
    _bar_day_check,
    _daily_intraday_consistency_check,
    _factor_semantic_check,
    _ordered_intraday_primary_key_check,
    _status_daily_partition_check,
    audit_database,
    audit_latest_keys,
)
from quant_data_platform.qdp_v2.duckdb_resources import open_guarded_duckdb
from quant_data_platform.qdp_v2.environment import (
    assert_yolos_environment,
    runtime_environment,
)
from quant_data_platform.qdp_v2.gc import lake_gc
from quant_data_platform.qdp_v2.manifest import (
    DatasetManifest,
    ShardManifestEntry,
    _manifest_schema_from_arrow,
    qdp_v2_root,
    read_dataset_manifest,
    write_active_manifest,
    write_dataset_manifest,
)
from quant_data_platform.qdp_v2.status import status_payload


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")


def _write_parquet(path: Path, frame: pd.DataFrame) -> None:
    import duckdb

    path.parent.mkdir(parents=True, exist_ok=True)
    with duckdb.connect(":memory:") as con:
        con.register("frame_to_write", frame)
        con.execute("copy frame_to_write to ? (format parquet)", [str(path)])


def _workspace(tmp_path: Path) -> Path:
    workspace = tmp_path / "workspace"
    (workspace / "quant_data_platform").mkdir(parents=True)
    (workspace / "AGENTS.md").write_text("workspace\n", encoding="utf-8")
    return workspace


def test_workspace_root_does_not_depend_on_brain(tmp_path: Path, monkeypatch) -> None:
    workspace = _workspace(tmp_path)
    nested = workspace / "daily_research" / "path_policy"
    nested.mkdir(parents=True)
    monkeypatch.chdir(nested)
    monkeypatch.delenv("QDP_WORKSPACE_ROOT", raising=False)

    assert workspace_root() == workspace.resolve()


def _write_domain(
    root: Path,
    domain: str,
    frame: pd.DataFrame,
    *,
    contract: str,
    primary_key: list[str],
    quality: dict | None = None,
    dataset_id: str | None = None,
) -> str:
    resolved_id = dataset_id or f"{domain}__unit"
    shard = root / "datasets" / domain / resolved_id / "shards" / "part.parquet"
    _write_parquet(shard, frame)
    import pyarrow.parquet as pq

    schema = _manifest_schema_from_arrow(pq.read_schema(shard))
    date_column = "trade_date" if "trade_date" in frame else ""
    start = str(frame[date_column].min()) if date_column and len(frame) else ""
    end = str(frame[date_column].max()) if date_column and len(frame) else ""
    write_dataset_manifest(
        root,
        DatasetManifest(
            dataset_id=resolved_id,
            domain=domain,
            layer="canonical" if domain == "security_identity" else "raw",
            frequency="5m" if domain == "market_intraday_5m" else "1d",
            contract_version=contract,
            primary_key=primary_key,
            start_date=start,
            end_date=end,
            row_count=len(frame),
            shards=[
                ShardManifestEntry(
                    path=str(shard.relative_to(root)).replace("\\", "/"),
                    row_count=len(frame),
                    start_date=start,
                    end_date=end,
                )
            ],
            source={"provider": "unit"},
            quality=quality or {},
            schema=schema,
        ),
    )
    return resolved_id


def _write_active(
    root: Path, datasets: dict[str, str], as_of: str = "2026-01-05"
) -> None:
    write_active_manifest(
        root,
        {
            "version": 2,
            "active_as_of_date": as_of,
            "scope": {
                "start_date": as_of,
                "end_date": as_of,
                "universe": "unit",
            },
            "datasets": datasets,
        },
    )


def test_required_domains_match_the_current_general_store() -> None:
    assert REQUIRED_DOMAINS == {
        "trading_calendar",
        "security_identity",
        "symbol_history",
        "market_daily_raw",
        "security_status",
        "universe_snapshot",
        "adjust_factor",
        "market_intraday_5m",
        "industry_concept",
        "share_capital",
        "valuation",
        "name_change",
        "corporate_actions",
        "index_constituents",
    }
    assert "market_intraday_1m" not in REQUIRED_DOMAINS
    assert "limit_intraday_features" not in REQUIRED_DOMAINS


def test_status_reads_active_and_dataset_manifests(tmp_path: Path) -> None:
    workspace = _workspace(tmp_path)
    root = qdp_v2_root(workspace)
    dataset_id = _write_domain(
        root,
        "market_daily_raw",
        pd.DataFrame(
            {
                "symbol": ["000001.SZ"],
                "trade_date": ["2026-01-05"],
                "open": [1.0],
                "high": [1.0],
                "low": [1.0],
                "close": [1.0],
            }
        ),
        contract="qdp_v2_market_daily_raw_v1",
        primary_key=["trade_date", "symbol"],
    )
    _write_active(root, {"market_daily_raw": dataset_id})

    payload = status_payload(workspace_root=workspace, verify_files=True)

    assert payload["status"] == "ok"
    assert payload["datasets"]["market_daily_raw"]["existing_shards"] == 1


def test_composite_manifest_keeps_cross_dataset_shards_reachable(
    tmp_path: Path,
) -> None:
    workspace = _workspace(tmp_path)
    root = qdp_v2_root(workspace)
    old_id = _write_domain(
        root,
        "market_daily_raw",
        pd.DataFrame(
            {
                "symbol": ["000001.SZ"],
                "trade_date": ["2026-01-05"],
                "open": [1.0],
                "high": [1.0],
                "low": [1.0],
                "close": [1.0],
            }
        ),
        contract="qdp_v2_market_daily_raw_v1",
        primary_key=["trade_date", "symbol"],
        dataset_id="market_daily_raw__old",
    )
    old_manifest = read_dataset_manifest(
        root / "datasets" / "market_daily_raw" / old_id / "dataset.json"
    )
    new_id = "market_daily_raw__composite"
    new_shard = (
        root / "datasets" / "market_daily_raw" / new_id / "shards" / "restore.parquet"
    )
    _write_parquet(
        new_shard,
        pd.DataFrame(
            {
                "symbol": ["000002.SZ"],
                "trade_date": ["2026-01-05"],
                "open": [2.0],
                "high": [2.0],
                "low": [2.0],
                "close": [2.0],
            }
        ),
    )
    write_dataset_manifest(
        root,
        DatasetManifest(
            dataset_id=new_id,
            domain="market_daily_raw",
            layer="raw",
            frequency="1d",
            contract_version="qdp_v2_market_daily_raw_v1",
            primary_key=["trade_date", "symbol"],
            start_date="2026-01-05",
            end_date="2026-01-05",
            row_count=2,
            shards=[
                *old_manifest.shards,
                ShardManifestEntry(
                    path=str(new_shard.relative_to(root)).replace("\\", "/"),
                    row_count=1,
                    start_date="2026-01-05",
                    end_date="2026-01-05",
                ),
            ],
            source={"provider": "unit-composite"},
            quality={"ohlcv_non_null": True},
            schema=old_manifest.schema,
        ),
    )
    _write_active(root, {"market_daily_raw": new_id})

    status = status_payload(workspace_root=workspace, verify_files=True)
    audit = audit_active(workspace_root=workspace, write=False)
    gc = lake_gc(workspace_root=workspace)

    assert status["status"] == "ok"
    assert status["datasets"]["market_daily_raw"]["existing_shards"] == 2
    assert not audit["errors"], audit["errors"]
    assert old_id not in {item["dataset_id"] for item in gc["unreferenced"]}


def test_audit_and_gc_use_manifests(tmp_path: Path) -> None:
    workspace = _workspace(tmp_path)
    root = qdp_v2_root(workspace)
    active_id = _write_domain(
        root,
        "trading_calendar",
        pd.DataFrame(
            {
                "trade_date": ["2026-01-05"],
                "is_open": [True],
                "exchange": ["SSE"],
                "source": ["unit"],
            }
        ),
        contract="qdp_v2_trading_calendar_v1",
        primary_key=["trade_date", "exchange"],
    )
    orphan_dir = root / "datasets" / "valuation" / "valuation__orphan"
    orphan_dir.mkdir(parents=True, exist_ok=True)
    _write_json(
        orphan_dir / "dataset.json",
        {
            "dataset_id": "valuation__orphan",
            "domain": "valuation",
            "shards": [],
            "row_count": 0,
        },
    )
    _write_active(root, {"trading_calendar": active_id})

    assert audit_active(workspace_root=workspace, write=False)["status"] == "ok"
    dry = lake_gc(workspace_root=workspace, with_size=True)
    deleted = lake_gc(workspace_root=workspace, delete=True, yes=True)

    assert any(
        item["dataset_id"] == "valuation__orphan" for item in dry["unreferenced"]
    )
    assert any(item["dataset_id"] == "valuation__orphan" for item in deleted["deleted"])


def test_gc_can_explicitly_remove_workspace_local_runtime(tmp_path: Path) -> None:
    workspace = _workspace(tmp_path)
    root = qdp_v2_root(workspace)
    runtime_file = root.parent / "qdp_runtime" / "stale_job" / "payload.tmp"
    runtime_file.parent.mkdir(parents=True, exist_ok=True)
    runtime_file.write_bytes(b"temporary")

    dry = lake_gc(workspace_root=workspace, clean_runtime=True, with_size=True)
    deleted = lake_gc(
        workspace_root=workspace,
        clean_runtime=True,
        delete=True,
        yes=True,
        with_size=True,
    )

    assert dry["runtime"]["item_count"] == 1
    assert dry["runtime"]["bytes"] == len(b"temporary")
    assert deleted["runtime"]["deleted_item_count"] == 1
    assert not runtime_file.parent.exists()


def test_quick_check_reads_schema_without_comparing_footer_row_counts(
    tmp_path: Path,
) -> None:
    workspace = _workspace(tmp_path)
    root = qdp_v2_root(workspace)
    dataset_id = _write_domain(
        root,
        "trading_calendar",
        pd.DataFrame(
            {
                "trade_date": ["2026-01-05"],
                "exchange": ["SSE"],
                "is_open": [True],
                "source": ["unit"],
            }
        ),
        contract="qdp_v2_trading_calendar_v1",
        primary_key=["trade_date", "exchange"],
    )
    manifest_path = root / "datasets" / "trading_calendar" / dataset_id / "dataset.json"
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    payload["row_count"] = 999
    payload["shards"][0]["row_count"] = 999
    _write_json(manifest_path, payload)
    _write_active(root, {"trading_calendar": dataset_id})

    quick = run_check(workspace_root=workspace, full=False)
    footer = audit_active(workspace_root=workspace, write=False, verify_footers=True)

    assert quick["active"]["status"] == "ok"
    assert quick["status"] == "needs_attention"  # other core domains are absent
    assert footer["status"] == "error"
    assert any("row_count_mismatch" in item for item in footer["errors"])


def test_latest_key_check_accepts_aligned_complete_48_bar_day(tmp_path: Path) -> None:
    workspace = _workspace(tmp_path)
    root = qdp_v2_root(workspace)
    date = "2026-01-05"
    symbol = "000001.SZ"
    datasets: dict[str, str] = {}
    datasets["trading_calendar"] = _write_domain(
        root,
        "trading_calendar",
        pd.DataFrame(
            {
                "trade_date": [date],
                "is_open": [True],
                "exchange": ["SSE"],
                "source": ["unit"],
            }
        ),
        contract="qdp_v2_trading_calendar_v1",
        primary_key=["trade_date", "exchange"],
    )
    datasets["security_identity"] = _write_domain(
        root,
        "security_identity",
        pd.DataFrame(
            {
                "security_id": ["QDP-CN-SZSE-000001"],
                "official_org_id": [""],
                "issuer_name": ["unit"],
                "exchange": ["SZSE"],
                "list_date": ["1991-04-03"],
                "current_symbol": [symbol],
                "identity_source": ["unit"],
            }
        ),
        contract="qdp_current_identity_v1",
        primary_key=["security_id"],
    )
    datasets["symbol_history"] = _write_domain(
        root,
        "symbol_history",
        pd.DataFrame(
            {
                "security_id": ["QDP-CN-SZSE-000001"],
                "symbol": [symbol],
                "effective_from": ["1991-04-03"],
                "effective_to": ["9999-12-31"],
                "name_on_date": ["unit"],
                "board_on_date": ["MainBoard"],
                "evidence_source": ["unit"],
                "official_document_hash": [""],
            }
        ),
        contract="qdp_current_symbol_history_v1",
        primary_key=["security_id", "symbol", "effective_from"],
    )
    daily = pd.DataFrame(
        {
            "symbol": [symbol],
            "trade_date": [date],
            "open": [10.0],
            "high": [11.0],
            "low": [9.0],
            "close": [10.5],
            "volume": [100.0],
            "amount": [1000.0],
            "source": ["unit"],
            "adjusted_flag": ["none"],
        }
    )
    datasets["market_daily_raw"] = _write_domain(
        root,
        "market_daily_raw",
        daily,
        contract="qdp_v2_market_daily_raw_v1",
        primary_key=["trade_date", "symbol"],
        quality={"ohlcv_non_null": True, "primary_key_unique": True},
    )
    universe = pd.DataFrame(
        {
            "symbol": [symbol],
            "trade_date": [date],
            "name": ["unit"],
            "exchange": ["SZ"],
            "board": ["main"],
            "list_status": ["L"],
            "list_date": ["1991-04-03"],
            "delist_date": [""],
            "source": ["unit"],
        }
    )
    datasets["universe_snapshot"] = _write_domain(
        root,
        "universe_snapshot",
        universe,
        contract="qdp_v2_universe_snapshot_v1",
        primary_key=["trade_date", "symbol"],
    )
    status = pd.DataFrame(
        {
            "symbol": [symbol],
            "trade_date": [date],
            "is_st": [False],
            "is_suspended": [False],
            "is_delisted": [False],
            "status_reason": [""],
            "source": ["unit"],
        }
    )
    datasets["security_status"] = _write_domain(
        root,
        "security_status",
        status,
        contract="qdp_v2_security_status_v1",
        primary_key=["trade_date", "symbol"],
    )
    factor = pd.DataFrame(
        {
            "symbol": [symbol],
            "trade_date": [date],
            "fore_adjust_factor": [1.0],
            "back_adjust_factor": [1.0],
            "adjust_factor": [1.0],
            "factor_provider": ["unit"],
            "factor_semantics": ["unit"],
            "source": ["unit"],
            "factor_source_date": [date],
            "ffill_days": [0],
        }
    )
    datasets["adjust_factor"] = _write_domain(
        root,
        "adjust_factor",
        factor,
        contract="qdp_v2_adjust_factor_standard_v2",
        primary_key=["trade_date", "symbol"],
    )
    times = [
        f"{hour:02d}{minute:02d}00000"
        for hour, minute in (
            *[(9, minute) for minute in range(35, 60, 5)],
            *[(10, minute) for minute in range(0, 60, 5)],
            *[(11, minute) for minute in range(0, 31, 5)],
            *[(13, minute) for minute in range(5, 60, 5)],
            *[(14, minute) for minute in range(0, 60, 5)],
            (15, 0),
        )
    ]
    bars = pd.DataFrame(
        {
            "symbol": symbol,
            "trade_date": date,
            "bar_time": times,
            "open": 10.0,
            "high": 11.0,
            "low": 9.0,
            "close": 10.5,
            "volume": 1.0,
            "amount": 10.0,
            "source": "unit",
            "adjusted_flag": "none",
        }
    )
    datasets["market_intraday_5m"] = _write_domain(
        root,
        "market_intraday_5m",
        bars,
        contract="qdp_current_intraday_5m_48_v1",
        primary_key=["trade_date", "symbol", "bar_time"],
        quality={"bar_count_contract": "48", "primary_key_unique": True},
    )
    _write_active(root, datasets, as_of=date)

    result = audit_latest_keys(workspace_root=workspace)

    assert result["status"] == "ok"
    assert result["five_minute"]["coverage_ratio"] == 1.0
    assert result["key_checks"]["daily_missing_factor"] == 0


def test_full_audit_reports_duplicate_primary_keys(tmp_path: Path) -> None:
    workspace = _workspace(tmp_path)
    root = qdp_v2_root(workspace)
    frame = pd.DataFrame(
        {
            "symbol": ["000001.SZ", "000001.SZ"],
            "trade_date": ["2026-01-05", "2026-01-05"],
            "open": [10.0, 10.0],
            "high": [11.0, 11.0],
            "low": [9.0, 9.0],
            "close": [10.5, 10.5],
            "volume": [100.0, 100.0],
            "amount": [1000.0, 1000.0],
            "source": ["unit", "unit"],
            "adjusted_flag": ["none", "none"],
        }
    )
    dataset_id = _write_domain(
        root,
        "market_daily_raw",
        frame,
        contract="qdp_v2_market_daily_raw_v1",
        primary_key=["trade_date", "symbol"],
        quality={"ohlcv_non_null": True},
    )
    _write_active(root, {"market_daily_raw": dataset_id})

    payload = audit_database(
        workspace_root=workspace,
        deep=True,
        max_shards=1,
        write=False,
    )

    assert payload["status"] == "needs_attention"
    assert any(
        item["code"] == "primary_key_duplicate_rows" for item in payload["findings"]
    )
    report = next(
        item for item in payload["datasets"] if item["domain"] == "market_daily_raw"
    )
    assert report["checks"]["primary_key"]["duplicate_rows"] == 1


def test_full_audit_reports_symbol_lifecycle_effectivity_as_high(
    tmp_path: Path,
) -> None:
    workspace = _workspace(tmp_path)
    root = qdp_v2_root(workspace)
    security_id = "QDP-SECURITY-1"
    datasets = {
        "security_identity": _write_domain(
            root,
            "security_identity",
            pd.DataFrame(
                {
                    "security_id": [security_id],
                    "exchange": ["SZ"],
                    "list_date": ["2010-01-01"],
                    "current_symbol": ["000002.SZ"],
                }
            ),
            contract="unit_identity_v1",
            primary_key=["security_id"],
        ),
        "symbol_history": _write_domain(
            root,
            "symbol_history",
            pd.DataFrame(
                {
                    "security_id": [security_id, security_id],
                    "symbol": ["000001.SZ", "000002.SZ"],
                    "effective_from": ["2010-01-01", "2020-01-01"],
                    "effective_to": ["2019-12-31", "9999-12-31"],
                    "name_on_date": ["Old Co", "New Co"],
                }
            ),
            contract="unit_symbol_history_v1",
            primary_key=["security_id", "symbol", "effective_from"],
        ),
        "market_daily_raw": _write_domain(
            root,
            "market_daily_raw",
            pd.DataFrame(
                {
                    "symbol": ["000002.SZ"],
                    "trade_date": ["2019-06-03"],
                    "open": [10.0],
                    "high": [10.0],
                    "low": [10.0],
                    "close": [10.0],
                    "volume": [100.0],
                    "amount": [1000.0],
                    "source": ["unit"],
                    "adjusted_flag": ["none"],
                }
            ),
            contract="qdp_v2_market_daily_raw_v1",
            primary_key=["trade_date", "symbol"],
        ),
    }
    _write_active(root, datasets, as_of="2019-06-03")

    payload = audit_database(
        workspace_root=workspace,
        deep=True,
        max_shards=1,
        write=False,
    )

    finding = next(
        item
        for item in payload["findings"]
        if item["code"] == "symbol_lifecycle_effective_interval_violation"
    )
    assert finding["severity"] == "high"
    lifecycle = payload["cross_dataset_checks"]["symbol_lifecycle_effectivity"]
    assert lifecycle["status"] == "needs_repair"
    assert lifecycle["outside_effective_interval_rows"] == 1
    assert (
        lifecycle["domains"]["market_daily_raw"]["outside_effective_interval_rows"] == 1
    )


def test_intraday_overlapping_date_shards_verify_actual_keys(tmp_path: Path) -> None:
    def write(name: str, symbol: str) -> tuple[Path, ShardManifestEntry]:
        path = tmp_path / f"{name}.parquet"
        _write_parquet(
            path,
            pd.DataFrame(
                {
                    "symbol": [symbol],
                    "trade_date": ["2026-01-05"],
                    "bar_time": ["09:35:00"],
                }
            ),
        )
        return path, ShardManifestEntry(
            path=path.as_posix(),
            row_count=1,
            start_date="2026-01-05",
            end_date="2026-01-05",
            file_size=path.stat().st_size,
        )

    left_path, left_entry = write("left", "600000.SH")
    disjoint_path, disjoint_entry = write("disjoint", "000001.SZ")
    duplicate_path, duplicate_entry = write("duplicate", "600000.SH")
    with open_guarded_duckdb(
        temp_directory=tmp_path / "spill",
        threads=1,
        memory_sampler=lambda: 8 * 1024**3,
    ) as con:
        disjoint = _ordered_intraday_primary_key_check(
            con,
            [left_path, disjoint_path],
            [left_entry, disjoint_entry],
            ["trade_date", "symbol", "bar_time"],
            20,
        )
        duplicate = _ordered_intraday_primary_key_check(
            con,
            [left_path, duplicate_path],
            [left_entry, duplicate_entry],
            ["trade_date", "symbol", "bar_time"],
            20,
        )

    assert disjoint["status"] == "ok"
    assert disjoint["range_overlap_count"] == 1
    assert disjoint["cross_shard_duplicate_rows"] == 0
    assert duplicate["status"] == "error"
    assert duplicate["cross_shard_duplicate_rows"] == 1


def test_intraday_physical_order_follows_declared_primary_key(tmp_path: Path) -> None:
    path = tmp_path / "date_first.parquet"
    _write_parquet(
        path,
        pd.DataFrame(
            {
                "trade_date": ["2026-01-05", "2026-01-06"],
                "symbol": ["600000.SH", "000001.SZ"],
                "bar_time": ["09:35:00", "09:35:00"],
            }
        ),
    )
    entry = ShardManifestEntry(
        path=path.as_posix(),
        row_count=2,
        start_date="2026-01-05",
        end_date="2026-01-06",
        file_size=path.stat().st_size,
    )
    legacy_path = tmp_path / "symbol_first.parquet"
    _write_parquet(
        legacy_path,
        pd.DataFrame(
            {
                "trade_date": ["2026-01-06", "2026-01-05"],
                "symbol": ["000001.SZ", "600000.SH"],
                "bar_time": ["09:35:00", "09:35:00"],
            }
        ),
    )
    legacy_entry = ShardManifestEntry(
        path=legacy_path.as_posix(),
        row_count=2,
        start_date="2026-01-05",
        end_date="2026-01-06",
        file_size=legacy_path.stat().st_size,
    )

    with open_guarded_duckdb(
        temp_directory=tmp_path / "date-first-spill",
        threads=1,
        memory_sampler=lambda: 8 * 1024**3,
    ) as con:
        result = _ordered_intraday_primary_key_check(
            con,
            [path],
            [entry],
            ["trade_date", "symbol", "bar_time"],
            5,
        )
        legacy = _ordered_intraday_primary_key_check(
            con,
            [legacy_path],
            [legacy_entry],
            ["trade_date", "symbol", "bar_time"],
            5,
        )

    assert result["status"] == "ok"
    assert result["physical_order_violation_rows"] == 0
    assert result["declared_primary_key_order"] == [
        "trade_date",
        "symbol",
        "bar_time",
    ]
    assert result["observed_physical_key_orders"] == {
        "trade_date>symbol>bar_time": 1
    }
    assert legacy["status"] == "ok"
    assert legacy["observed_physical_key_orders"] == {
        "symbol>trade_date>bar_time": 1
    }


def test_bar_day_check_aggregates_independent_date_shards(tmp_path: Path) -> None:
    paths: list[Path] = []
    for index, trade_date in enumerate(("2025-12-31", "2026-01-05")):
        path = tmp_path / f"part-{index}.parquet"
        _write_parquet(
            path,
            pd.DataFrame(
                {
                    "symbol": ["600000.SH"] * len(EXPECTED_BAR_TIMES),
                    "trade_date": [trade_date] * len(EXPECTED_BAR_TIMES),
                    "bar_time": list(EXPECTED_BAR_TIMES),
                }
            ),
        )
        paths.append(path)

    with open_guarded_duckdb(
        temp_directory=tmp_path / "spill",
        threads=1,
        memory_sampler=lambda: 8 * 1024**3,
        total_memory_sampler=lambda: 16 * 1024**3,
    ) as con:
        result = _bar_day_check(con, paths, sample_limit=5)

    assert result == {"status": "ok", "invalid_day_count": 0, "examples": []}


def test_cross_frequency_check_rejects_100x_intraday_volume(tmp_path: Path) -> None:
    workspace = _workspace(tmp_path)
    root = qdp_v2_root(workspace)
    date = "2026-01-05"
    daily_id = _write_domain(
        root,
        "market_daily_raw",
        pd.DataFrame(
            {
                "symbol": ["600584.SH"],
                "trade_date": [date],
                "open": [10.0],
                "high": [10.2],
                "low": [9.8],
                "close": [10.1],
                "volume": [100.0],
                "amount": [1_000.0],
            }
        ),
        contract="unit",
        primary_key=["trade_date", "symbol"],
    )
    five_id = _write_domain(
        root,
        "market_intraday_5m",
        pd.DataFrame(
            {
                "symbol": ["600584.SH"] * 48,
                "trade_date": [date] * 48,
                "bar_time": EXPECTED_BAR_TIMES,
                "open": [10.0] * 48,
                "high": [10.2] * 48,
                "low": [9.8] * 48,
                "close": [10.1] * 48,
                "volume": [10_000.0 / 48] * 48,
                "amount": [1_000.0 / 48] * 48,
            }
        ),
        contract="unit",
        primary_key=["trade_date", "symbol", "bar_time"],
    )
    daily = _active_manifest(root, daily_id, "market_daily_raw")
    five = _active_manifest(root, five_id, "market_intraday_5m")
    with open_guarded_duckdb(
        temp_directory=workspace / "spill",
        threads=1,
        memory_sampler=lambda: 8 * 1024**3,
        total_memory_sampler=lambda: 16 * 1024**3,
    ) as con:
        result = _daily_intraday_consistency_check(
            con,
            root=root,
            daily=daily,
            intraday=five,
            sample_limit=5,
        )

    assert result["volume_100x_day_count"] == 1
    assert result["invalid_day_count"] == 1


def test_cross_frequency_check_scans_overlapping_symbol_shards_once(
    tmp_path: Path,
) -> None:
    workspace = _workspace(tmp_path)
    root = qdp_v2_root(workspace)
    date = "2026-01-05"
    symbols = ["600000.SH", "000001.SZ"]
    daily_id = _write_domain(
        root,
        "market_daily_raw",
        pd.DataFrame(
            {
                "symbol": symbols,
                "trade_date": [date, date],
                "open": [10.0, 20.0],
                "high": [10.2, 20.2],
                "low": [9.8, 19.8],
                "close": [10.1, 20.1],
                "volume": [4800.0, 4800.0],
                "amount": [48_000.0, 96_000.0],
            }
        ),
        contract="unit",
        primary_key=["trade_date", "symbol"],
    )
    five_id = "market_intraday_5m__overlapping_symbol_shards"
    shard_dir = root / "datasets" / "market_intraday_5m" / five_id / "shards"
    entries: list[ShardManifestEntry] = []
    for number, (symbol, price) in enumerate(zip(symbols, (10.0, 20.0), strict=True)):
        path = shard_dir / f"part-{number}.parquet"
        _write_parquet(
            path,
            pd.DataFrame(
                {
                    "symbol": [symbol] * 48,
                    "trade_date": [date] * 48,
                    "bar_time": EXPECTED_BAR_TIMES,
                    "open": [price] * 48,
                    "high": [price + 0.2] * 48,
                    "low": [price - 0.2] * 48,
                    "close": [price + 0.1] * 48,
                    "volume": [100.0] * 48,
                    "amount": [price * 100.0] * 48,
                }
            ),
        )
        entries.append(
            ShardManifestEntry(
                path=str(path.relative_to(root)).replace("\\", "/"),
                row_count=48,
                start_date=date,
                end_date=date,
            )
        )
    import pyarrow.parquet as pq

    write_dataset_manifest(
        root,
        DatasetManifest(
            dataset_id=five_id,
            domain="market_intraday_5m",
            layer="raw",
            frequency="5m",
            contract_version="unit",
            primary_key=["trade_date", "symbol", "bar_time"],
            start_date=date,
            end_date=date,
            row_count=96,
            shards=entries,
            source={"provider": "unit"},
            quality={},
            schema=_manifest_schema_from_arrow(pq.read_schema(root / entries[0].path)),
        ),
    )

    with open_guarded_duckdb(
        temp_directory=workspace / "overlap-five-spill",
        threads=1,
        memory_sampler=lambda: 8 * 1024**3,
        total_memory_sampler=lambda: 16 * 1024**3,
    ) as con:
        result = _daily_intraday_consistency_check(
            con,
            root=root,
            daily=_active_manifest(root, daily_id, "market_daily_raw"),
            intraday=_active_manifest(root, five_id, "market_intraday_5m"),
            sample_limit=5,
        )

    assert result["positive_daily_count"] == 2
    assert result["complete_positive_daily_count"] == 2
    assert result["missing_positive_daily_count"] == 0
    assert result["invalid_day_count"] == 0


def test_cross_frequency_check_resolves_intraday_ticker_lifecycle(
    tmp_path: Path,
) -> None:
    workspace = _workspace(tmp_path)
    root = qdp_v2_root(workspace)
    dates = ["2019-01-02", "2019-01-03"]
    daily_id = _write_domain(
        root,
        "market_daily_raw",
        pd.DataFrame(
            {
                "symbol": ["000001.SZ", "000001.SZ"],
                "trade_date": dates,
                "open": [10.0, 10.0],
                "high": [10.2, 10.2],
                "low": [9.8, 9.8],
                "close": [10.1, 10.1],
                "volume": [4800.0, 4800.0],
                "amount": [48_000.0, 48_000.0],
            }
        ),
        contract="unit",
        primary_key=["trade_date", "symbol"],
    )
    bars: list[pd.DataFrame] = []
    for date, symbol, price in (
        (dates[0], "000001.SZ", 10.0),
        (dates[0], "000002.SZ", 99.0),
        (dates[1], "000002.SZ", 10.0),
    ):
        bars.append(
            pd.DataFrame(
                {
                    "symbol": [symbol] * 48,
                    "trade_date": [date] * 48,
                    "bar_time": EXPECTED_BAR_TIMES,
                    "open": [price] * 48,
                    "high": [price + 0.2] * 48,
                    "low": [price - 0.2] * 48,
                    "close": [price + 0.1] * 48,
                    "volume": [100.0] * 48,
                    "amount": [1000.0] * 48,
                }
            )
        )
    five_id = _write_domain(
        root,
        "market_intraday_5m",
        pd.concat(bars, ignore_index=True),
        contract="unit",
        primary_key=["trade_date", "symbol", "bar_time"],
    )
    history_id = _write_domain(
        root,
        "symbol_history",
        pd.DataFrame(
            {
                "security_id": ["SEC-1", "SEC-1"],
                "symbol": ["000001.SZ", "000002.SZ"],
                "effective_from": ["2010-01-01", "2020-01-01"],
                "effective_to": ["2019-12-31", "9999-12-31"],
            }
        ),
        contract="unit",
        primary_key=["security_id", "symbol", "effective_from"],
    )
    with open_guarded_duckdb(
        temp_directory=workspace / "lifecycle-five-spill",
        threads=1,
        memory_sampler=lambda: 8 * 1024**3,
        total_memory_sampler=lambda: 16 * 1024**3,
    ) as con:
        result = _daily_intraday_consistency_check(
            con,
            root=root,
            daily=_active_manifest(root, daily_id, "market_daily_raw"),
            intraday=_active_manifest(root, five_id, "market_intraday_5m"),
            history=_active_manifest(root, history_id, "symbol_history"),
            sample_limit=5,
        )

    assert result["complete_positive_daily_count"] == 2
    assert result["missing_daily_for_5m_count"] == 0
    assert result["invalid_day_count"] == 0


def test_status_daily_semantics_detects_both_mismatch_directions(
    tmp_path: Path,
) -> None:
    status_path = tmp_path / "status.parquet"
    daily_path = tmp_path / "daily.parquet"
    date = "2026-01-05"
    _write_parquet(
        status_path,
        pd.DataFrame(
            {
                "symbol": ["000001.SZ", "000002.SZ", "000003.SZ", "000004.SZ"],
                "trade_date": [date] * 4,
                "is_suspended": [False, False, True, True],
            }
        ),
    )
    _write_parquet(
        daily_path,
        pd.DataFrame(
            {
                "symbol": ["000001.SZ", "000003.SZ", "000004.SZ"],
                "trade_date": [date] * 3,
                "volume": [100.0, 100.0, 0.0],
            }
        ),
    )
    with open_guarded_duckdb(
        temp_directory=tmp_path / "status-spill",
        threads=1,
        memory_sampler=lambda: 8 * 1024**3,
        total_memory_sampler=lambda: 16 * 1024**3,
    ) as con:
        result = _status_daily_partition_check(
            con,
            status_paths=[status_path],
            daily_paths=[daily_path],
            start_date=date,
            end_date=date,
            sample_limit=5,
        )

    assert result["semantic_mismatch_count"] == 2
    assert result["unsuspended_without_positive_daily_count"] == 1
    assert result["suspended_with_positive_daily_count"] == 1
    assert result["null_suspension_flag_count"] == 0


def test_factor_semantics_separates_short_pollution_from_long_gap(
    tmp_path: Path,
) -> None:
    workspace = _workspace(tmp_path)
    root = qdp_v2_root(workspace)
    daily = pd.DataFrame(
        {
            "symbol": ["000001.SZ", "000001.SZ", "000002.SZ", "000002.SZ"],
            "trade_date": ["2024-01-02", "2024-01-03", "2020-01-02", "2024-01-02"],
            "open": [10.0, 10.0, 10.0, 20.0],
            "close": [10.0, 10.0, 10.0, 20.0],
        }
    )
    factors = pd.DataFrame(
        {
            "symbol": daily["symbol"],
            "trade_date": daily["trade_date"],
            "adjust_factor": [1.0, 1.5, 1.0, 2.0],
        }
    )
    daily_id = _write_domain(
        root,
        "market_daily_raw",
        daily,
        contract="unit",
        primary_key=["trade_date", "symbol"],
    )
    factor_id = _write_domain(
        root,
        "adjust_factor",
        factors,
        contract="unit",
        primary_key=["trade_date", "symbol"],
    )
    with open_guarded_duckdb(
        temp_directory=workspace / "factor-spill",
        threads=1,
        memory_sampler=lambda: 8 * 1024**3,
        total_memory_sampler=lambda: 16 * 1024**3,
    ) as con:
        result = _factor_semantic_check(
            con,
            root=root,
            daily=_active_manifest(root, daily_id, "market_daily_raw"),
            factor=_active_manifest(root, factor_id, "adjust_factor"),
            sample_limit=5,
        )

    assert result["uncompensated_change_count"] == 1
    assert result["raw_discontinuity_count"] == 1


def test_audit_spill_directory_is_workspace_local_and_ephemeral(tmp_path: Path) -> None:
    workspace = _workspace(tmp_path)
    with _audit_temp_directory(workspace) as temporary:
        path = Path(temporary)
        assert path.is_dir()
        assert workspace.resolve() in path.resolve().parents
        assert path.parent.name == "qdp_runtime"
    assert not path.exists()


def test_environment_guard_requires_yolos(monkeypatch) -> None:
    monkeypatch.delenv("QDP_ALLOW_NON_YOLOS", raising=False)
    monkeypatch.setenv("CONDA_DEFAULT_ENV", "base")
    monkeypatch.setattr(sys, "executable", r"C:\Users\ASUS\miniconda3\python.exe")

    try:
        assert_yolos_environment(command="qdp check")
    except RuntimeError as exc:
        assert "qdp_v2_requires_yolos_environment" in str(exc)
    else:
        raise AssertionError("expected environment guard failure")

    monkeypatch.setenv("CONDA_DEFAULT_ENV", "yolos")
    monkeypatch.setattr(
        sys,
        "executable",
        r"C:\Users\ASUS\miniconda3\envs\yolos\python.exe",
    )
    assert_yolos_environment(command="qdp check")
    assert "python_executable" in runtime_environment()
