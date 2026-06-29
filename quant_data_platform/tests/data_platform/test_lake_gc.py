from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from quant_data_platform.core.paths import qdp_paths
from quant_data_platform.domains.contracts import DataDomain
from quant_data_platform.lake.catalog import ResearchDataLake
from quant_data_platform.lake.gc import LakeGcConfig, lake_gc


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")


def test_lake_gc_dry_run_reports_unreferenced_dataset_dirs(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    _write_json(workspace / "brain" / "brain_manifest.json", {"schema_version": 1, "brain_type": "main"})
    paths = qdp_paths(workspace)
    lake_root = workspace / "quant_data_platform" / "data" / "lake"
    lake = ResearchDataLake(lake_root)

    active = lake.save_sharded_domain_dataset(
        domain=DataDomain.MARKET_INTRADAY_1M,
        spec={
            "domain": DataDomain.MARKET_INTRADAY_1M,
            "start_date": "2026-01-05",
            "end_date": "2026-01-05",
            "sharded": True,
            "one_minute_policy": "mootdx_240_0930_merged_into_0931",
        },
        shard_records=[],
        source="unit",
        reuse=False,
    )
    orphan = lake.save_sharded_domain_dataset(
        domain=DataDomain.MARKET_INTRADAY_5M,
        spec={
            "domain": DataDomain.MARKET_INTRADAY_5M,
            "start_date": "2026-01-05",
            "end_date": "2026-01-05",
            "sharded": True,
        },
        shard_records=[],
        source="unit",
        reuse=False,
    )
    orphan_dir = lake.parquet_root / "bronze_silver" / "data_platform_market_intraday_5m" / orphan.fingerprint
    (orphan_dir / "payload.parquet").parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"x": [1]}).to_parquet(orphan_dir / "payload.parquet", index=False)
    active_dir = lake.parquet_root / "bronze_silver" / "data_platform_market_intraday_1m" / active.fingerprint
    pd.DataFrame({"x": [1]}).to_parquet(active_dir / "payload.parquet", index=False)

    _write_json(
        paths.root_manifest,
        {
            "schema_version": 1,
            "canonical_dataset_id": "policy_input_bundle__active",
            "canonical_component_dataset_ids": {
                DataDomain.MARKET_INTRADAY_1M: active.dataset_id,
            },
        },
    )
    _write_json(paths.memmap_registry, {"schema_version": 1})

    result = lake_gc(LakeGcConfig(lake_root=lake_root, workspace_root=workspace, dry_run=True, with_size=True))

    assert result.status == "dry_run"
    assert result.destructive_actions_performed is False
    assert any(item["dataset_id"] == orphan.dataset_id for item in result.unreferenced)
    assert all(item["dataset_id"] != active.dataset_id for item in result.unreferenced)
    assert result.unreferenced_bytes > 0


def test_lake_gc_delete_removes_only_unreferenced_dirs_and_catalog_rows(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    _write_json(workspace / "brain" / "brain_manifest.json", {"schema_version": 1, "brain_type": "main"})
    paths = qdp_paths(workspace)
    lake_root = workspace / "quant_data_platform" / "data" / "lake"
    lake = ResearchDataLake(lake_root)

    active = lake.save_sharded_domain_dataset(
        domain=DataDomain.MARKET_INTRADAY_1M,
        spec={
            "domain": DataDomain.MARKET_INTRADAY_1M,
            "start_date": "2026-01-05",
            "end_date": "2026-01-05",
            "sharded": True,
            "one_minute_policy": "mootdx_240_0930_merged_into_0931",
        },
        shard_records=[],
        source="unit",
        reuse=False,
    )
    orphan = lake.save_sharded_domain_dataset(
        domain=DataDomain.MARKET_INTRADAY_5M,
        spec={
            "domain": DataDomain.MARKET_INTRADAY_5M,
            "start_date": "2026-01-05",
            "end_date": "2026-01-05",
            "sharded": True,
        },
        shard_records=[],
        source="unit",
        reuse=False,
    )
    active_dir = lake.parquet_root / "bronze_silver" / "data_platform_market_intraday_1m" / active.fingerprint
    orphan_dir = lake.parquet_root / "bronze_silver" / "data_platform_market_intraday_5m" / orphan.fingerprint
    pd.DataFrame({"x": [1]}).to_parquet(active_dir / "payload.parquet", index=False)
    pd.DataFrame({"x": [1]}).to_parquet(orphan_dir / "payload.parquet", index=False)

    _write_json(
        paths.root_manifest,
        {
            "schema_version": 1,
            "canonical_component_dataset_ids": {
                DataDomain.MARKET_INTRADAY_1M: active.dataset_id,
            },
        },
    )
    _write_json(paths.memmap_registry, {"schema_version": 1})

    result = lake_gc(
        LakeGcConfig(
            lake_root=lake_root,
            workspace_root=workspace,
            delete=True,
            yes=True,
            max_items=0,
        )
    )

    assert result.status == "deleted"
    assert result.destructive_actions_performed is True
    assert orphan_dir.exists() is False
    assert active_dir.exists() is True
    assert orphan.dataset_id in {item["dataset_id"] for item in result.deleted}
    assert lake.query("select * from datasets where dataset_id = ?", [orphan.dataset_id]).empty
    assert not lake.query("select * from datasets where dataset_id = ?", [active.dataset_id]).empty
