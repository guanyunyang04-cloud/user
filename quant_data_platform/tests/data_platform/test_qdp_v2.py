from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

from quant_data_platform.qdp_v2.activate import activate_v2
from quant_data_platform.qdp_v2.audit import audit_active
from quant_data_platform.qdp_v2.check import run_check
from quant_data_platform.qdp_v2.cleaning import derive_5m_from_1m, normalize_valuation, split_daily_market
from quant_data_platform.qdp_v2.database_audit import audit_database
from quant_data_platform.qdp_v2.dataset import validate_dataset
from quant_data_platform.qdp_v2.environment import assert_yolos_environment, runtime_environment
from quant_data_platform.qdp_v2.gc import lake_gc
from quant_data_platform.qdp_v2.manifest import (
    DatasetManifest,
    ShardManifestEntry,
    qdp_v2_root,
    write_active_manifest,
    write_dataset_manifest,
)
from quant_data_platform.qdp_v2.meta_quality import audit_meta_domains, rebuild_adjust_factor_standard, rebuild_industry_concept_complete
from quant_data_platform.qdp_v2.status import status_payload


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")


def _write_parquet(path: Path, frame: pd.DataFrame) -> None:
    import duckdb  # type: ignore

    path.parent.mkdir(parents=True, exist_ok=True)
    with duckdb.connect(":memory:") as con:
        con.register("frame_to_write", frame)
        con.execute("copy frame_to_write to ? (format parquet)", [str(path)])


def _workspace(tmp_path: Path) -> Path:
    workspace = tmp_path / "workspace"
    _write_json(workspace / "brain" / "brain_manifest.json", {"schema_version": 1, "brain_type": "main"})
    return workspace


def _active_domain_specs() -> dict[str, tuple[str, str]]:
    return {
        "market_daily_raw": ("qdp_v2_market_daily_raw_v1", "market_daily_raw__ok"),
        "market_intraday_1m": ("mootdx_1m_240_v1", "market_intraday_1m__ok"),
        "market_intraday_5m": ("mootdx_5m_48_v1", "market_intraday_5m__ok"),
        "trading_calendar": ("qdp_v2_trading_calendar_v1", "trading_calendar__ok"),
        "universe_snapshot": ("qdp_v2_universe_snapshot_v1", "universe_snapshot__ok"),
        "security_status": ("qdp_v2_security_status_v1", "security_status__ok"),
        "valuation": ("qdp_v2_valuation_v1", "valuation__ok"),
        "adjust_factor": ("qdp_v2_adjust_factor_v1", "adjust_factor__ok"),
        "industry_concept": ("qdp_v2_industry_v5", "industry_concept__ok"),
        "index_constituents": ("qdp_v2_index_constituents_v1", "index_constituents__ok"),
        "limit_status": ("qdp_v2_limit_status_events_v1", "limit_status__ok"),
        "corporate_actions": ("qdp_v2_corporate_actions_raw_v1", "corporate_actions__ok"),
        "share_capital": ("qdp_v2_share_capital_raw_v1", "share_capital__ok"),
        "name_change": ("qdp_v2_name_change_raw_v1", "name_change__ok"),
        "intraday_daily_features": ("qdp_v2_intraday_daily_features_v1", "intraday_daily_features__ok"),
        "limit_intraday_features": ("qdp_v2_limit_intraday_features_1m_v1", "limit_intraday_features__ok"),
        "market_daily_panel": ("qdp_v2_market_daily_panel_v1", "market_daily_panel__ok"),
    }


def _write_active(root: Path, datasets: dict[str, str], as_of_date: str = "2026-01-05") -> None:
    write_active_manifest(
        root,
        {
            "version": 2,
            "active_as_of_date": as_of_date,
            "scope": {"start_date": as_of_date, "end_date": as_of_date, "universe": "unit", "symbol_start_overrides": {}},
            "datasets": datasets,
        },
    )


def test_qdp_v2_status_reads_active_and_dataset_manifests_without_catalog(tmp_path: Path) -> None:
    workspace = _workspace(tmp_path)
    root = qdp_v2_root(workspace)
    shard = root / "datasets" / "market_daily_raw" / "market_daily_raw__unit" / "shards" / "part.parquet"
    _write_parquet(shard, pd.DataFrame({"symbol": ["000001.SZ"], "trade_date": ["2026-01-05"], "open": [1.0], "high": [1.0], "low": [1.0], "close": [1.0]}))
    manifest = DatasetManifest(
        dataset_id="market_daily_raw__unit",
        domain="market_daily_raw",
        layer="raw",
        frequency="1d",
        contract_version="qdp_v2_market_daily_raw_v1",
        primary_key=["trade_date", "symbol"],
        start_date="2026-01-05",
        end_date="2026-01-05",
        row_count=1,
        schema_hash="unit",
        shards=[ShardManifestEntry(path="datasets/market_daily_raw/market_daily_raw__unit/shards/part.parquet", row_count=1, start_date="2026-01-05", end_date="2026-01-05")],
        source={"provider": "unit"},
        quality={"path_refs_exist": True, "ohlcv_non_null": True},
    )
    write_dataset_manifest(root, manifest)
    _write_active(root, {"market_daily_raw": manifest.dataset_id})

    payload = status_payload(workspace_root=workspace)

    assert payload["status"] == "ok"
    assert payload["datasets"]["market_daily_raw"]["existing_shards"] == 1


def test_qdp_v2_audit_validate_and_gc_use_manifests(tmp_path: Path) -> None:
    workspace = _workspace(tmp_path)
    root = qdp_v2_root(workspace)
    active_shard = root / "datasets" / "trading_calendar" / "trading_calendar__active" / "shards" / "part.parquet"
    _write_parquet(active_shard, pd.DataFrame({"trade_date": ["2026-01-05"], "is_open": [1]}))
    active = DatasetManifest(
        dataset_id="trading_calendar__active",
        domain="trading_calendar",
        layer="raw",
        frequency="",
        contract_version="qdp_v2_trading_calendar_v1",
        primary_key=["trade_date"],
        start_date="2026-01-05",
        end_date="2026-01-05",
        row_count=1,
        schema_hash="unit",
        shards=[ShardManifestEntry(path="datasets/trading_calendar/trading_calendar__active/shards/part.parquet", row_count=1, start_date="2026-01-05", end_date="2026-01-05")],
        source={"provider": "unit"},
        quality={"path_refs_exist": True, "ohlcv_non_null": True},
    )
    write_dataset_manifest(root, active)
    orphan_dir = root / "datasets" / "valuation" / "valuation__orphan"
    orphan_dir.mkdir(parents=True, exist_ok=True)
    _write_json(orphan_dir / "dataset.json", {"dataset_id": "valuation__orphan", "domain": "valuation", "shards": [], "row_count": 0})
    _write_active(root, {"trading_calendar": active.dataset_id})

    assert validate_dataset(active.dataset_id, workspace_root=workspace)["status"] == "ok"
    assert audit_active(workspace_root=workspace, write=False)["status"] == "ok"
    dry = lake_gc(workspace_root=workspace, with_size=True)
    deleted = lake_gc(workspace_root=workspace, delete=True, yes=True)

    assert any(item["dataset_id"] == "valuation__orphan" for item in dry["unreferenced"])
    assert any(item["dataset_id"] == "valuation__orphan" for item in deleted["deleted"])
    assert active_shard.exists() is True


def test_qdp_v2_quick_check_does_not_scan_all_parquet_footers(tmp_path: Path) -> None:
    workspace = _workspace(tmp_path)
    root = qdp_v2_root(workspace)
    shard = root / "datasets" / "trading_calendar" / "trading_calendar__quick" / "shards" / "part.parquet"
    _write_parquet(shard, pd.DataFrame({"trade_date": ["2026-01-05"], "exchange": ["SSE"], "is_open": [True]}))
    manifest = DatasetManifest(
        dataset_id="trading_calendar__quick",
        domain="trading_calendar",
        layer="raw",
        frequency="",
        contract_version="qdp_v2_trading_calendar_v1",
        primary_key=["trade_date"],
        start_date="2026-01-05",
        end_date="2026-01-05",
        row_count=999,
        schema_hash="unit",
        shards=[ShardManifestEntry(path="datasets/trading_calendar/trading_calendar__quick/shards/part.parquet", row_count=999, start_date="2026-01-05", end_date="2026-01-05")],
        source={"provider": "unit"},
        quality={"path_refs_exist": True},
    )
    write_dataset_manifest(root, manifest)
    _write_active(root, {"trading_calendar": manifest.dataset_id})

    quick = run_check(workspace_root=workspace, full=False, no_write=True)
    active = audit_active(workspace_root=workspace, write=False, verify_footers=True)

    assert quick["active"]["status"] == "ok"
    assert active["status"] == "error"
    assert any("row_count_mismatch" in item for item in active["errors"])


def test_qdp_v2_database_audit_reports_duplicate_primary_keys(tmp_path: Path) -> None:
    workspace = _workspace(tmp_path)
    root = qdp_v2_root(workspace)
    shard = root / "datasets" / "market_daily_raw" / "market_daily_raw__dup" / "shards" / "part.parquet"
    _write_parquet(
        shard,
        pd.DataFrame(
            {
                "symbol": ["000001.SZ", "000001.SZ", "000002.SZ"],
                "trade_date": ["2026-01-05", "2026-01-05", "2026-01-05"],
                "open": [10.0, 10.0, 20.0],
                "high": [11.0, 11.0, 21.0],
                "low": [9.0, 9.0, 19.0],
                "close": [10.5, 10.5, 20.5],
                "volume": [100.0, 100.0, 200.0],
                "amount": [1000.0, 1000.0, 4000.0],
            }
        ),
    )
    manifest = DatasetManifest(
        dataset_id="market_daily_raw__dup",
        domain="market_daily_raw",
        layer="raw",
        frequency="1d",
        contract_version="qdp_v2_market_daily_raw_v1",
        primary_key=["trade_date", "symbol"],
        start_date="2026-01-05",
        end_date="2026-01-05",
        row_count=3,
        schema_hash="unit",
        shards=[ShardManifestEntry(path="datasets/market_daily_raw/market_daily_raw__dup/shards/part.parquet", row_count=3, start_date="2026-01-05", end_date="2026-01-05")],
        source={"provider": "unit"},
        quality={"path_refs_exist": True, "ohlcv_non_null": True},
    )
    write_dataset_manifest(root, manifest)
    _write_active(root, {"market_daily_raw": manifest.dataset_id})

    payload = audit_database(workspace_root=workspace, deep=True, max_shards=1, write=False)

    assert payload["status"] == "needs_attention"
    assert any(item["code"] == "primary_key_duplicate_rows" for item in payload["findings"])
    raw_report = payload["datasets"][0]
    assert raw_report["checks"]["primary_key"]["duplicate_rows"] == 1


def test_qdp_v2_database_audit_checks_intraday_cross_frequency_consistency(tmp_path: Path) -> None:
    workspace = _workspace(tmp_path)
    root = qdp_v2_root(workspace)
    one_rows = []
    for minute in range(31, 41):
        one_rows.append(
            {
                "symbol": "000001.SZ",
                "trade_date": "2026-01-05",
                "bar_time": f"09{minute:02d}00000",
                "open": float(minute),
                "high": float(minute) + 0.1,
                "low": float(minute) - 0.1,
                "close": float(minute) + 0.05,
                "volume": 1.0,
                "amount": 10.0,
            }
        )
    five_rows = [
        {
            "symbol": "000001.SZ",
            "trade_date": "2026-01-05",
            "bar_time": "093500000",
            "open": 31.0,
            "high": 35.1,
            "low": 30.9,
            "close": 35.05,
            "volume": 5.0,
            "amount": 50.0,
        },
        {
            "symbol": "000001.SZ",
            "trade_date": "2026-01-05",
            "bar_time": "094000000",
            "open": 36.0,
            "high": 40.1,
            "low": 35.9,
            "close": 40.05,
            "volume": 5.0,
            "amount": 50.0,
        },
    ]
    daily_rows = [
        {
            "symbol": "000001.SZ",
            "trade_date": "2026-01-05",
            "open": 31.0,
            "high": 40.1,
            "low": 30.9,
            "close": 40.05,
            "volume": 10.0,
            "amount": 100.0,
        }
    ]
    panel_rows = [{**daily_rows[0], "has_bar": True}]
    domain_frames = {
        "market_intraday_1m": ("market_intraday_1m__unit", pd.DataFrame(one_rows), "mootdx_1m_240_v1", ["trade_date", "symbol", "bar_time"]),
        "market_intraday_5m": ("market_intraday_5m__unit", pd.DataFrame(five_rows), "mootdx_5m_48_v1", ["trade_date", "symbol", "bar_time"]),
        "market_daily_raw": ("market_daily_raw__unit", pd.DataFrame(daily_rows), "qdp_v2_market_daily_raw_v1", ["trade_date", "symbol"]),
        "market_daily_panel": ("market_daily_panel__unit", pd.DataFrame(panel_rows), "qdp_v2_market_daily_panel_v1", ["trade_date", "symbol"]),
        "trading_calendar": ("trading_calendar__unit", pd.DataFrame({"trade_date": ["2026-01-05"], "exchange": ["SSE"], "is_open": [True]}), "qdp_v2_trading_calendar_v1", ["trade_date", "exchange"]),
        "universe_snapshot": ("universe_snapshot__unit", pd.DataFrame({"trade_date": ["2026-01-05"], "symbol": ["000001.SZ"]}), "qdp_v2_universe_snapshot_v1", ["trade_date", "symbol"]),
        "security_status": (
            "security_status__unit",
            pd.DataFrame({"trade_date": ["2026-01-05"], "symbol": ["000001.SZ"], "is_st": [False], "is_delisted": [False], "is_suspended": [False]}),
            "qdp_v2_security_status_v1",
            ["trade_date", "symbol"],
        ),
    }
    for domain, (dataset_id, frame, contract, pk) in domain_frames.items():
        shard = root / "datasets" / domain / dataset_id / "shards" / "part_000000_market_intraday_1m_mainboard.parquet"
        if domain == "market_intraday_5m":
            shard = root / "datasets" / domain / dataset_id / "shards" / "part_000000_market_intraday_5m_mainboard.parquet"
        elif not domain.startswith("market_intraday"):
            shard = root / "datasets" / domain / dataset_id / "shards" / "part.parquet"
        _write_parquet(shard, frame)
        write_dataset_manifest(
            root,
            DatasetManifest(
                dataset_id=dataset_id,
                domain=domain,
                layer="research_panel" if domain == "market_daily_panel" else "raw",
                frequency="",
                contract_version=contract,
                primary_key=pk,
                start_date="2026-01-05",
                end_date="2026-01-05",
                row_count=len(frame),
                schema_hash="unit",
                shards=[ShardManifestEntry(path=str(shard.relative_to(root)).replace("\\", "/"), row_count=len(frame), start_date="2026-01-05", end_date="2026-01-05")],
                source={"provider": "unit"},
                quality={"path_refs_exist": True, "ohlcv_non_null": True, "has_bar_contract": True},
            ),
        )
    _write_active(
        root,
        {
            "market_daily_raw": "market_daily_raw__unit",
            "market_intraday_1m": "market_intraday_1m__unit",
            "market_intraday_5m": "market_intraday_5m__unit",
            "trading_calendar": "trading_calendar__unit",
            "universe_snapshot": "universe_snapshot__unit",
            "security_status": "security_status__unit",
            "market_daily_panel": "market_daily_panel__unit",
        },
    )

    payload = audit_database(workspace_root=workspace, deep=True, max_shards=1, write=False)

    assert payload["cross_dataset_checks"]["intraday_5m_from_1m"]["status"] == "ok"
    assert payload["cross_dataset_checks"]["intraday_vs_daily"]["status"] == "ok"


def test_qdp_v2_meta_quality_rebuilds_factor_and_industry_then_audits(tmp_path: Path) -> None:
    workspace = _workspace(tmp_path)
    root = qdp_v2_root(workspace)

    def write_domain(domain: str, dataset_id: str, frame: pd.DataFrame, contract: str, pk: list[str]) -> None:
        shard = root / "datasets" / domain / dataset_id / "shards" / "part.parquet"
        _write_parquet(shard, frame)
        write_dataset_manifest(
            root,
            DatasetManifest(
                dataset_id=dataset_id,
                domain=domain,
                layer="raw",
                frequency="1d",
                contract_version=contract,
                primary_key=pk,
                start_date=str(frame["trade_date"].min()) if "trade_date" in frame.columns else "",
                end_date=str(frame["trade_date"].max()) if "trade_date" in frame.columns else "",
                row_count=len(frame),
                schema_hash="unit",
                shards=[ShardManifestEntry(path=str(shard.relative_to(root)).replace("\\", "/"), row_count=len(frame), start_date="2026-01-05", end_date="2026-01-06")],
                source={"provider": "unit"},
                quality={"path_refs_exist": True},
            ),
        )

    daily = pd.DataFrame(
        {
            "trade_date": ["2026-01-05", "2026-01-06"],
            "symbol": ["000001.SZ", "000001.SZ"],
            "open": [10.0, 10.5],
            "high": [11.0, 11.5],
            "low": [9.5, 10.0],
            "close": [10.8, 11.0],
            "volume": [100.0, 120.0],
            "amount": [1000.0, 1320.0],
        }
    )
    factor_source = pd.DataFrame(
        {
            "trade_date": ["2026-01-05"],
            "symbol": ["000001.SZ"],
            "factor_provider": ["tonghuashun"],
            "fore_adjust_factor": [-0.5],
            "back_adjust_factor": [1.25],
        }
    )
    calendar = pd.DataFrame({"trade_date": ["2026-01-05", "2026-01-06"], "is_open": [True, True]})
    universe = pd.DataFrame({"trade_date": ["2026-01-05", "2026-01-06"], "symbol": ["000001.SZ", "000001.SZ"]})
    status = pd.DataFrame(
        {
            "trade_date": ["2026-01-05", "2026-01-06"],
            "symbol": ["000001.SZ", "000001.SZ"],
            "is_st": [False, False],
            "is_delisted": [False, False],
            "is_suspended": [False, False],
        }
    )
    industry = pd.DataFrame({"trade_date": ["2026-01-05"], "symbol": ["000001.SZ"], "industry": [""], "source": ["unit"]})
    index = pd.DataFrame({"trade_date": ["2026-01-05"], "index_symbol": ["000300.SH"], "symbol": ["000001.SZ"]})

    write_domain("market_daily_raw", "market_daily_raw__unit", daily, "qdp_v2_market_daily_raw_v1", ["trade_date", "symbol"])
    write_domain("adjust_factor", "adjust_factor__raw", factor_source, "raw_factor_pool", ["trade_date", "symbol"])
    write_domain("trading_calendar", "trading_calendar__unit", calendar, "qdp_v2_trading_calendar_v1", ["trade_date"])
    write_domain("universe_snapshot", "universe_snapshot__unit", universe, "qdp_v2_universe_snapshot_v1", ["trade_date", "symbol"])
    write_domain("security_status", "security_status__unit", status, "qdp_v2_security_status_v1", ["trade_date", "symbol"])
    write_domain("industry_concept", "industry_concept__raw", industry, "raw_industry", ["trade_date", "symbol"])
    write_domain("index_constituents", "index_constituents__unit", index, "qdp_v2_index_constituents_v1", ["trade_date", "index_symbol", "symbol"])
    _write_active(
        root,
        {
            "market_daily_raw": "market_daily_raw__unit",
            "adjust_factor": "adjust_factor__raw",
            "trading_calendar": "trading_calendar__unit",
            "universe_snapshot": "universe_snapshot__unit",
            "security_status": "security_status__unit",
            "industry_concept": "industry_concept__raw",
            "index_constituents": "index_constituents__unit",
        },
        as_of_date="2026-01-06",
    )

    factor = rebuild_adjust_factor_standard(workspace_root=workspace, runtime="safe", duckdb_memory_limit="1GB", threads=1, activate=True)
    industry_result = rebuild_industry_concept_complete(workspace_root=workspace, runtime="safe", duckdb_memory_limit="1GB", threads=1, activate=True)
    audit = audit_meta_domains(workspace_root=workspace, runtime="safe", duckdb_memory_limit="1GB", threads=1, writeback=True)

    assert factor["status"] == "ok"
    assert factor["stats"]["row_count"] == 2
    assert factor["stats"]["non_positive_fore_rows"] == 2
    assert industry_result["status"] == "ok"
    assert industry_result["stats"]["unknown_industry_rows"] == 2
    assert audit["status"] == "ok"
    assert audit["finding_count"] == 0


def test_qdp_v2_cleaning_derives_48_contract_5m_from_1m_shard(tmp_path: Path) -> None:
    workspace = _workspace(tmp_path)
    root = qdp_v2_root(workspace)
    shard = root / "datasets" / "market_intraday_1m" / "market_intraday_1m__unit" / "shards" / "part.parquet"
    rows = []
    for minute in range(31, 41):
        rows.append(
            {
                "symbol": "000001.SZ",
                "trade_date": "2026-01-05",
                "bar_time": f"09{minute:02d}00000",
                "open": float(minute),
                "high": float(minute + 1),
                "low": float(minute - 1),
                "close": float(minute) + 0.5,
                "volume": 10,
                "amount": 100,
                "source": "unit",
                "adjusted_flag": "",
            }
        )
    _write_parquet(shard, pd.DataFrame(rows))
    source = DatasetManifest(
        dataset_id="market_intraday_1m__unit",
        domain="market_intraday_1m",
        layer="raw",
        frequency="1m",
        contract_version="mootdx_1m_240_v1",
        primary_key=["trade_date", "symbol", "bar_time"],
        start_date="2026-01-05",
        end_date="2026-01-05",
        row_count=len(rows),
        schema_hash="unit",
        shards=[ShardManifestEntry(path="datasets/market_intraday_1m/market_intraday_1m__unit/shards/part.parquet", row_count=len(rows), start_date="2026-01-05", end_date="2026-01-05")],
        source={"provider": "unit"},
        quality={},
    )
    write_dataset_manifest(root, source)

    result = derive_5m_from_1m(workspace_root=workspace, source_dataset_id=source.dataset_id, max_shards=1, runtime="safe")

    assert result["status"] == "ok"
    assert result["row_count"] == 2
    assert validate_dataset(result["target_dataset_id"], workspace_root=workspace, domain="market_intraday_5m")["status"] == "ok"


def test_qdp_v2_cleaning_normalizes_valuation_schema(tmp_path: Path) -> None:
    workspace = _workspace(tmp_path)
    root = qdp_v2_root(workspace)
    shard = root / "datasets" / "valuation" / "valuation__legacy" / "shards" / "part.parquet"
    _write_parquet(
        shard,
        pd.DataFrame(
            {
                "symbol": ["000001.SZ"],
                "trade_date": ["2026-01-05"],
                "total_mv": [100.0],
                "circ_mv": [80.0],
                "peTTM": [12.0],
                "pbMRQ": [1.5],
                "turn": [0.02],
            }
        ),
    )
    source = DatasetManifest(
        dataset_id="valuation__legacy",
        domain="valuation",
        layer="raw",
        frequency="1d",
        contract_version="legacy",
        primary_key=["trade_date", "symbol"],
        start_date="2026-01-05",
        end_date="2026-01-05",
        row_count=1,
        schema_hash="unit",
        shards=[ShardManifestEntry(path="datasets/valuation/valuation__legacy/shards/part.parquet", row_count=1, start_date="2026-01-05", end_date="2026-01-05")],
        source={"provider": "unit"},
        quality={},
    )
    write_dataset_manifest(root, source)

    result = normalize_valuation(workspace_root=workspace, source_dataset_id=source.dataset_id, max_shards=1, runtime="safe")
    described = validate_dataset(result["target_dataset_id"], workspace_root=workspace, domain="valuation")

    assert result["status"] == "ok"
    assert described["status"] == "ok"


def test_qdp_v2_cleaning_splits_daily_market_raw_and_panel(tmp_path: Path) -> None:
    workspace = _workspace(tmp_path)
    root = qdp_v2_root(workspace)
    shard = root / "datasets" / "market_daily_panel" / "market_daily_panel__legacy" / "shards" / "part.parquet"
    _write_parquet(
        shard,
        pd.DataFrame(
            {
                "symbol": ["000001.SZ", "000002.SZ"],
                "trade_date": ["2026-01-05", "2026-01-05"],
                "open": [10.0, None],
                "high": [11.0, None],
                "low": [9.0, None],
                "close": [10.5, None],
                "volume": [1000.0, None],
                "amount": [10000.0, None],
                "source": ["unit", "unit"],
                "ingest_batch_id": ["b1", "b1"],
            }
        ),
    )
    source = DatasetManifest(
        dataset_id="market_daily_panel__legacy",
        domain="market_daily_panel",
        layer="research_panel",
        frequency="1d",
        contract_version="legacy_policy_bundle_research_panel_v1",
        primary_key=["trade_date", "symbol"],
        start_date="2026-01-05",
        end_date="2026-01-05",
        row_count=2,
        schema_hash="unit",
        shards=[ShardManifestEntry(path="datasets/market_daily_panel/market_daily_panel__legacy/shards/part.parquet", row_count=2, start_date="2026-01-05", end_date="2026-01-05")],
        source={"provider": "unit"},
        quality={},
    )
    write_dataset_manifest(root, source)

    result = split_daily_market(workspace_root=workspace, source_dataset_id=source.dataset_id, max_shards=1, runtime="safe")

    assert result["status"] == "ok"
    assert result["raw_row_count"] == 1
    assert result["panel_row_count"] == 2
    assert validate_dataset(result["raw_dataset_id"], workspace_root=workspace, domain="market_daily_raw")["status"] == "ok"
    assert validate_dataset(result["panel_dataset_id"], workspace_root=workspace, domain="market_daily_panel")["status"] == "ok"


def test_qdp_v2_activate_selects_clean_contracts_and_writes_active(tmp_path: Path) -> None:
    workspace = _workspace(tmp_path)
    root = qdp_v2_root(workspace)
    specs = _active_domain_specs()
    for domain, (contract, dataset_id) in specs.items():
        write_dataset_manifest(
            root,
            DatasetManifest(
                dataset_id=dataset_id,
                domain=domain,
                layer="raw",
                frequency="1d",
                contract_version=contract,
                primary_key=[],
                start_date="2026-01-05",
                end_date="2026-01-05",
                row_count=0,
                schema_hash="unit",
                shards=[],
                source={"provider": "unit"},
                quality={},
            ),
        )
    write_dataset_manifest(
        root,
        DatasetManifest(
            dataset_id="market_intraday_5m__pending",
            domain="market_intraday_5m",
            layer="raw",
            frequency="5m",
            contract_version="legacy_5m_pending_rebuild_to_mootdx_5m_48_v1",
            primary_key=[],
            start_date="2026-01-05",
            end_date="2026-01-06",
            row_count=0,
            schema_hash="unit",
            shards=[],
            source={"provider": "unit"},
            quality={},
        ),
    )

    dry = activate_v2(workspace_root=workspace, yes=False)
    written = activate_v2(workspace_root=workspace, yes=True)

    assert dry["status"] == "dry_run"
    assert written["status"] == "activated"
    assert written["active"]["datasets"]["market_intraday_5m"] == "market_intraday_5m__ok"
    assert (root / "active" / "active.json").exists()


def test_qdp_v2_activate_rejects_legacy_market_daily_panel_contract(tmp_path: Path) -> None:
    workspace = _workspace(tmp_path)
    root = qdp_v2_root(workspace)
    required = _active_domain_specs()
    required["market_daily_panel"] = ("legacy_policy_bundle_research_panel_v1", "market_daily_panel__legacy")
    for domain, (contract, dataset_id) in required.items():
        write_dataset_manifest(
            root,
            DatasetManifest(
                dataset_id=dataset_id,
                domain=domain,
                layer="raw",
                frequency="1d",
                contract_version=contract,
                primary_key=[],
                start_date="2026-01-05",
                end_date="2026-01-05",
                row_count=0,
                schema_hash="unit",
                shards=[],
                source={"provider": "unit"},
                quality={},
            ),
        )

    result = activate_v2(workspace_root=workspace, yes=True)

    assert result["status"] == "blocked"
    assert any("required_domain_wrong_contract:market_daily_panel" in item for item in result["errors"])


def test_qdp_v2_environment_guard_requires_yolos(monkeypatch) -> None:
    monkeypatch.delenv("QDP_ALLOW_NON_YOLOS", raising=False)
    monkeypatch.setenv("CONDA_DEFAULT_ENV", "base")
    monkeypatch.setattr(sys, "executable", r"C:\Users\ASUS\miniconda3\python.exe")

    try:
        assert_yolos_environment(command="provider benchmark")
    except RuntimeError as exc:
        assert "qdp_v2_requires_yolos_environment" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("expected environment guard failure")

    monkeypatch.setenv("CONDA_DEFAULT_ENV", "yolos")
    monkeypatch.setattr(sys, "executable", r"C:\Users\ASUS\miniconda3\envs\yolos\python.exe")
    assert_yolos_environment(command="provider benchmark")
    assert "python_executable" in runtime_environment()
