from __future__ import annotations

import json
import zipfile

import pandas as pd

from quant_data_platform.lake.canonical import (
    DEFAULT_CANONICAL_ALIAS,
    build_lake_inventory,
    load_canonical_manifest,
    resolve_canonical_dataset_id,
    write_canonical_manifest,
)
from quant_data_platform.lake.build_canonical_policy_bundle import (
    BuildCanonicalPolicyBundleConfig,
    build_canonical_policy_bundle,
)
from quant_data_platform.lake.catalog import ResearchDataLake
from quant_data_platform.lake.policy_input_loader import (
    LEGACY_DEFAULT_POLICY_INPUT_LAKE_DATASET_ID,
    _shard_symbol_overlaps,
    load_policy_inputs_from_lake,
    resolve_policy_input_dataset_id,
)
from quant_data_platform.domains.contracts import DataDomain


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


def test_sidecar_shard_symbol_filter_uses_complete_source_member_sample() -> None:
    row = {
        "source_member_count": 2,
        "source_members_sample": [
            "复权因子(同花顺)/sh600000.csv",
            "复权因子(同花顺)/sz000001.csv",
        ],
    }

    assert _shard_symbol_overlaps(row, symbols=["600000.SH"])
    assert _shard_symbol_overlaps(row, symbols=["000001.SZ"])
    assert not _shard_symbol_overlaps(row, symbols=["000002.SZ"])
    assert _shard_symbol_overlaps({**row, "source_member_count": 3}, symbols=["000002.SZ"])


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


def test_build_canonical_policy_bundle_references_sharded_market_manifest(tmp_path) -> None:
    lake_root = tmp_path / "lake"
    lake = ResearchDataLake(lake_root)
    shard_path = tmp_path / "market_daily_shard.parquet"
    market = pd.DataFrame(
        {
            "trade_date": [
                "2010-01-04",
                "2010-01-04",
                "2010-01-04",
                "2010-01-05",
                "2010-01-05",
                "2010-01-05",
            ],
            "symbol": ["000001.SZ", "600000.SH", "000300.SH", "000001.SZ", "600000.SH", "000300.SH"],
            "open": [10.0, 20.0, 3000.0, 10.2, 20.2, 3010.0],
            "high": [10.5, 20.5, 3020.0, 10.8, 20.8, 3030.0],
            "low": [9.8, 19.8, 2990.0, 10.0, 20.0, 3005.0],
            "close": [10.3, 20.3, 3015.0, 10.6, 20.6, 3025.0],
            "volume": [1000.0, 2000.0, 0.0, 1100.0, 2100.0, 0.0],
            "amount": [10300.0, 40600.0, 0.0, 11660.0, 43260.0, 0.0],
        }
    )
    market.to_parquet(shard_path, index=False)
    market_record = lake.save_sharded_domain_dataset(
        domain="market_daily",
        spec={"dataset": "data_platform_market_daily", "start_date": "2010-01-01", "end_date": "2010-01-05"},
        shard_records=[
            {
                "status": "stored",
                "path": str(shard_path),
                "row_count": int(len(market)),
                "start_date": "2010-01-04",
                "end_date": "2010-01-05",
                "symbol_count": 3,
            }
        ],
        source="unit",
    )
    valuation_record = lake.save_domain_dataset(
        domain="valuation",
        frame=pd.DataFrame(
            {
                "trade_date": ["2010-01-04", "2010-01-05"],
                "symbol": ["000001.SZ", "000001.SZ"],
                "turn": [1.0, 1.1],
            }
        ),
        spec={"dataset": "data_platform_valuation", "start_date": "2010-01-01", "end_date": "2010-01-05"},
        source="unit",
    )

    result = build_canonical_policy_bundle(
        BuildCanonicalPolicyBundleConfig(
            lake_root=lake_root,
            market_daily_dataset_id=market_record.dataset_id,
            sidecar_dataset_ids={"valuation": valuation_record.dataset_id},
            start_date="2010-01-01",
            end_date="2010-01-05",
        )
    )

    manifest = load_canonical_manifest(lake)
    metadata = lake.describe_dataset(result.dataset_id)
    prepared = load_policy_inputs_from_lake(
        lake=lake,
        dataset_id=result.dataset_id,
        start_date="2010-01-04",
        end_date="2010-01-05",
        universe=["000001.SZ", "600000.SH"],
        min_trading_days=2,
    )

    assert manifest["canonical_dataset_id"] == result.dataset_id
    assert metadata["parameters"]["market_daily_dataset_id"] == market_record.dataset_id
    assert metadata["parameters"]["sidecar_dataset_ids"]["valuation"] == valuation_record.dataset_id
    assert metadata["content_paths"]["bronze_market_shard_manifest"].endswith("shard_manifest.json")
    assert prepared.universe == ("000001.SZ", "600000.SH")
    assert float(prepared.close.loc[pd.Timestamp("2010-01-05"), "000001.SZ"]) == 10.6


def test_policy_loader_reads_intraday_and_adjust_sidecars_without_slow_domains(tmp_path) -> None:
    lake_root = tmp_path / "lake"
    lake = ResearchDataLake(lake_root)
    dates = pd.bdate_range("2024-01-02", periods=8)
    rows = []
    for dt in dates:
        for symbol, base in (("000001.SZ", 10.0), ("600000.SH", 20.0), ("000300.SH", 3000.0)):
            rows.append(
                {
                    "trade_date": dt.strftime("%Y-%m-%d"),
                    "symbol": symbol,
                    "open": base,
                    "high": base * 1.01,
                    "low": base * 0.99,
                    "close": base * (1.0 + 0.001 * len(rows)),
                    "volume": 1000.0,
                    "amount": 10000.0,
                }
            )
    market_record = lake.save_domain_dataset(
        domain=DataDomain.MARKET_DAILY,
        frame=pd.DataFrame(rows),
        spec={"dataset": "data_platform_market_daily", "start_date": "2024-01-02", "end_date": "2024-01-11"},
        source="unit",
    )
    intraday_record = lake.save_domain_dataset(
        domain=DataDomain.INTRADAY_DAILY_FEATURES,
        frame=pd.DataFrame(
            {
                "trade_date": [dates[1].strftime("%Y-%m-%d"), dates[1].strftime("%Y-%m-%d")],
                "symbol": ["000001.SZ", "600000.SH"],
                "first_5m_ret": [0.01, -0.02],
                "last_30m_ret": [0.03, -0.01],
                "high_time_frac": [0.25, 0.75],
            }
        ),
        spec={"dataset": "data_platform_intraday_daily_features", "start_date": "2024-01-02", "end_date": "2024-01-11"},
        source="unit",
    )
    adjust_record = lake.save_domain_dataset(
        domain=DataDomain.ADJUST_FACTOR,
        frame=pd.DataFrame(
            {
                "trade_date": [dates[0].strftime("%Y-%m-%d"), dates[1].strftime("%Y-%m-%d")],
                "symbol": ["000001.SZ", "000001.SZ"],
                "fore_adjust_factor": [1.0, 0.9],
            }
        ),
        spec={"dataset": "data_platform_adjust_factor", "start_date": "2024-01-02", "end_date": "2024-01-11"},
        source="unit",
    )
    result = build_canonical_policy_bundle(
        BuildCanonicalPolicyBundleConfig(
            lake_root=lake_root,
            market_daily_dataset_id=market_record.dataset_id,
            sidecar_dataset_ids={
                DataDomain.INTRADAY_DAILY_FEATURES: intraday_record.dataset_id,
                DataDomain.ADJUST_FACTOR: adjust_record.dataset_id,
            },
            start_date="2024-01-02",
            end_date="2024-01-11",
        )
    )

    prepared = load_policy_inputs_from_lake(
        lake=lake,
        dataset_id=result.dataset_id,
        start_date="2024-01-02",
        end_date="2024-01-11",
        universe=["000001.SZ", "600000.SH"],
        min_trading_days=2,
    )

    assert prepared.metadata_summary["intraday_daily_features_sidecar"]["available"] is True
    assert prepared.metadata_summary["adjust_factor_sidecar"]["available"] is True
    assert prepared.metadata_summary["valuation_sidecar"]["available"] is False
    assert prepared.metadata_summary["industry_sidecar"]["available"] is False
    assert abs(float(prepared.derived_frames["intraday_daily_features_first_5m_ret"].loc[dates[1], "000001.SZ"]) - 0.01) < 1.0e-6
    assert abs(float(prepared.derived_frames["intraday_daily_features_high_time_frac"].loc[dates[1], "600000.SH"]) - 0.75) < 1.0e-6
    assert str(prepared.derived_frames["intraday_daily_features_high_time_frac"].dtypes["000001.SZ"]) == "float32"
    assert abs(float(prepared.derived_frames["adjust_factor_fore_adjust_factor"].loc[dates[1], "000001.SZ"]) - 0.9) < 1.0e-6
    assert str(prepared.derived_frames["adjust_factor_fore_adjust_factor"].dtypes["000001.SZ"]) == "float32"
