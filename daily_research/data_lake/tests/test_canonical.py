from __future__ import annotations

import json
import zipfile

import pandas as pd

from daily_research.data_lake.canonical import (
    DEFAULT_CANONICAL_ALIAS,
    build_lake_inventory,
    load_canonical_manifest,
    resolve_canonical_dataset_id,
    write_canonical_manifest,
)
from daily_research.data_lake.catalog import ResearchDataLake
from daily_research.data_lake.policy_input_loader import (
    LEGACY_DEFAULT_POLICY_INPUT_LAKE_DATASET_ID,
    resolve_policy_input_dataset_id,
)


def test_canonical_manifest_round_trip(tmp_path) -> None:
    lake = ResearchDataLake(tmp_path / "lake")
    path = write_canonical_manifest(
        lake,
        dataset_id="policy_input_bundle__unit",
        sidecar_dataset_ids={"valuation": "data_platform_valuation__unit"},
        coverage_report={"status": "ok"},
    )

    assert path.exists()
    manifest = load_canonical_manifest(lake)
    assert manifest["alias"] == DEFAULT_CANONICAL_ALIAS
    assert manifest["canonical_dataset_id"] == "policy_input_bundle__unit"
    assert resolve_canonical_dataset_id(lake) == "policy_input_bundle__unit"
    assert manifest["start_date"] == "2010-01-01"
    assert resolve_policy_input_dataset_id(lake, "") == "policy_input_bundle__unit"


def test_policy_input_default_falls_back_without_canonical_manifest(tmp_path) -> None:
    lake = ResearchDataLake(tmp_path / "lake")

    assert resolve_policy_input_dataset_id(lake, "") == LEGACY_DEFAULT_POLICY_INPUT_LAKE_DATASET_ID


def test_build_lake_inventory_reports_external_zips_and_memmaps(tmp_path) -> None:
    lake = ResearchDataLake(tmp_path / "lake")
    lake.save_domain_dataset(
        domain="valuation",
        frame=pd.DataFrame({"symbol": ["000001.SZ"], "trade_date": ["2010-01-04"], "pe": [10.0]}),
        spec={"start_date": "2010-01-01", "end_date": "2010-01-04"},
        source="unit",
    )
    external_root = tmp_path / "量化数据"
    zip_dir = external_root / "1分钟"
    zip_dir.mkdir(parents=True)
    with zipfile.ZipFile(zip_dir / "2010.zip", "w") as archive:
        archive.writestr("sh600000.csv", "日期,开盘\n2010-01-04 09:30:00,10\n")
    path_policy_root = tmp_path / "path_policy"
    study = path_policy_root / "study"
    study.mkdir(parents=True)
    (study / "forecast_dataset_manifest.json").write_text(
        json.dumps({"dataset_mode": "memmap", "sample_count": 3}, ensure_ascii=False),
        encoding="utf-8",
    )
    (study / "forecast_feature_store.dat").write_bytes(b"1234")

    inventory = build_lake_inventory(
        lake,
        external_data_root=external_root,
        path_policy_root=path_policy_root,
        start_date="2010-01-01",
    )

    assert inventory["dataset_count"] == 1
    assert inventory["external_zips"]["domains"]["market_intraday_1m"]["zip_count"] == 1
    assert inventory["memmaps"]["manifest_count"] == 1
    assert inventory["memmaps"]["forecast_dat_bytes"] == 4
    assert inventory["cleanup_dry_run"]["destructive_actions_performed"] is False
