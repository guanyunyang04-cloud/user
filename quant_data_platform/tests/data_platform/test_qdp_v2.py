from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from quant_data_platform.core.paths import qdp_paths
from quant_data_platform.domains.contracts import DataDomain
from quant_data_platform.lake.catalog import ResearchDataLake
from quant_data_platform.qdp_v2.activate import activate_v2
from quant_data_platform.qdp_v2.audit import audit_active
from quant_data_platform.qdp_v2.cleaning import derive_5m_from_1m, normalize_valuation
from quant_data_platform.qdp_v2.dataset import validate_dataset
from quant_data_platform.qdp_v2.gc import lake_gc
from quant_data_platform.qdp_v2.index import rebuild_index
from quant_data_platform.qdp_v2.manifest import (
    DatasetManifest,
    ShardManifestEntry,
    qdp_v2_root,
    write_active_manifest,
    write_dataset_manifest,
)
from quant_data_platform.qdp_v2.migration import build_migration_plan, execute_migration_plan
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
        quality={"path_refs_exist": True},
    )
    write_dataset_manifest(root, manifest)
    write_active_manifest(root, {"active_as_of_date": "2026-01-05", "raw": {"market_daily_raw": manifest.dataset_id}, "derived": {}, "research_panels": {}, "memmap": {"status": "not_part_of_data_base"}})

    payload = status_payload(workspace_root=workspace)

    assert payload["status"] == "ok"
    assert payload["duckdb_catalog_required"] is False
    assert payload["datasets"]["raw.market_daily_raw"]["existing_shards"] == 1


def test_qdp_v2_migrate_copy_verify_activate_from_legacy_sharded_dataset(tmp_path: Path) -> None:
    workspace = _workspace(tmp_path)
    paths = qdp_paths(workspace)
    lake = ResearchDataLake(paths.lake_root)
    source_file = paths.lake_root / "source" / "intraday.parquet"
    source_file.parent.mkdir(parents=True, exist_ok=True)
    _write_parquet(
        source_file,
        pd.DataFrame(
            {
                "symbol": ["000001.SZ"],
                "trade_date": ["2026-01-05"],
                "bar_time": ["09:31:00"],
                "open": [1.0],
                "high": [1.1],
                "low": [0.9],
                "close": [1.0],
            }
        ),
    )
    record = lake.save_sharded_domain_dataset(
        domain=DataDomain.MARKET_INTRADAY_1M,
        spec={"start_date": "2026-01-05", "end_date": "2026-01-05"},
        shard_records=[
            {
                "path": str(source_file.resolve()),
                "row_count": 1,
                "start_date": "2026-01-05",
                "end_date": "2026-01-05",
                "status": "stored",
            }
        ],
        source="unit",
        reuse=False,
    )
    _write_json(
        paths.root_manifest,
        {
            "schema_version": 1,
            "canonical_dataset_id": "policy_input_bundle__unit",
            "canonical_dataset_end_date": "2026-01-05",
            "canonical_component_dataset_ids": {DataDomain.MARKET_INTRADAY_1M: record.dataset_id},
        },
    )

    plan = build_migration_plan(workspace_root=workspace)
    result = execute_migration_plan(plan, verify=True, activate=True)
    root = qdp_v2_root(workspace)
    status = status_payload(workspace_root=workspace)

    assert result["status"] == "migrated"
    assert (root / "active" / "active.json").exists()
    assert status["status"] == "ok"
    assert status["datasets"]["raw.market_intraday_1m"]["row_count"] == 1
    assert source_file.exists() is True


def test_qdp_v2_audit_validate_index_and_gc_use_manifests(tmp_path: Path) -> None:
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
        quality={"path_refs_exist": True},
    )
    write_dataset_manifest(root, active)
    orphan_dir = root / "datasets" / "valuation" / "valuation__orphan"
    orphan_dir.mkdir(parents=True, exist_ok=True)
    _write_json(orphan_dir / "dataset.json", {"dataset_id": "valuation__orphan", "domain": "valuation", "shards": [], "row_count": 0})
    write_active_manifest(root, {"active_as_of_date": "2026-01-05", "raw": {"trading_calendar": active.dataset_id}, "derived": {}, "research_panels": {}, "memmap": {"status": "not_part_of_data_base"}})

    assert validate_dataset(active.dataset_id, workspace_root=workspace)["status"] == "ok"
    assert audit_active(workspace_root=workspace, write=False)["status"] == "ok"
    assert rebuild_index(workspace_root=workspace)["status"] == "ok"
    dry = lake_gc(workspace_root=workspace, with_size=True)
    deleted = lake_gc(workspace_root=workspace, delete=True, yes=True)

    assert any(item["dataset_id"] == "valuation__orphan" for item in dry["unreferenced"])
    assert any(item["dataset_id"] == "valuation__orphan" for item in deleted["deleted"])
    assert active_shard.exists() is True


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


def test_qdp_v2_activate_selects_clean_contracts_and_writes_active(tmp_path: Path) -> None:
    workspace = _workspace(tmp_path)
    root = qdp_v2_root(workspace)
    specs = {
        "market_daily_raw": ("qdp_v2_market_daily_raw_v1", "market_daily_raw__ok"),
        "market_intraday_1m": ("mootdx_1m_240_v1", "market_intraday_1m__ok"),
        "market_intraday_5m": ("mootdx_5m_48_v1", "market_intraday_5m__ok"),
        "trading_calendar": ("qdp_v2_trading_calendar_v1", "trading_calendar__ok"),
        "universe_snapshot": ("qdp_v2_universe_snapshot_v1", "universe_snapshot__ok"),
        "security_status": ("qdp_v2_security_status_v1", "security_status__ok"),
        "valuation": ("qdp_v2_valuation_v1", "valuation__ok"),
        "adjust_factor": ("qdp_v2_adjust_factor_v1", "adjust_factor__ok"),
        "intraday_daily_features": ("qdp_v2_intraday_daily_features_v1", "intraday_daily_features__ok"),
        "market_daily_panel": ("legacy_policy_bundle_research_panel_v1", "market_daily_panel__ok"),
    }
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
    assert written["active"]["raw"]["market_intraday_5m"] == "market_intraday_5m__ok"
    assert (root / "active" / "active.json").exists()
