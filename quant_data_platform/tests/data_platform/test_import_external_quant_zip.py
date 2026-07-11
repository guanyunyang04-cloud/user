from __future__ import annotations

import json
import zipfile
from pathlib import Path

import pandas as pd
import pytest

from quant_data_platform.lake.catalog import ResearchDataLake
from quant_data_platform.domains.contracts import DataDomain
from quant_data_platform.ingest.build_intraday_daily_features import (
    BuildIntradayDailyFeaturesConfig,
    INTRADAY_DAILY_FEATURE_CONTRACT_VERSION,
    LAST_5M_RET_POLICY,
    _build_intraday_daily_feature_frame_fast,
    _build_auction_1m_shard_index,
    _feature_dataset_spec,
    _matching_auction_1m_shards,
    _shard_intersects_years,
    build_intraday_daily_features,
)
from quant_data_platform.ingest.combine_domain_datasets import CombineDomainDatasetsConfig, combine_domain_datasets
from quant_data_platform.ingest.combine_sharded_domain_datasets import CombineShardedDomainConfig, combine_sharded_domain_datasets
from quant_data_platform.ingest.import_external_quant_zip import ImportConfig, normalize_intraday_1m_to_mootdx_240_frame, run_import
from quant_data_platform.ingest.normalize_intraday_1m_contract import NormalizeIntraday1mContractConfig, normalize_intraday_1m_contract
from quant_data_platform.ingest.recover_external_quant_zip_import import RecoverExternalImportConfig, recover_import


def test_fast_intraday_daily_features_last_5m_ret_uses_previous_close_for_closing_point_bar() -> None:
    frame = pd.DataFrame(
        {
            "symbol": ["000001.SZ"] * 3,
            "trade_date": ["2026-01-05"] * 3,
            "bar_time": ["14:50:00", "14:55:00", "15:00:00"],
            "open": [10.0, 10.1, 10.3],
            "high": [10.2, 10.2, 10.3],
            "low": [9.9, 10.0, 10.3],
            "close": [10.1, 10.2, 10.3],
            "volume": [100, 200, 300],
            "amount": [1010, 2040, 3090],
        }
    )

    features = _build_intraday_daily_feature_frame_fast(frame, source="external_5m", adjusted_flag="none")

    assert float(features["last_5m_ret"].iloc[0]) == pytest.approx(10.3 / 10.2 - 1.0)
    assert float(features["opening_auction_amount"].iloc[0]) == pytest.approx(1010.0)
    assert float(features["closing_auction_amount"].iloc[0]) == pytest.approx(3090.0)
    assert float(features["closing_auction_ret"].iloc[0]) == pytest.approx(10.3 / 10.2 - 1.0)


def test_build_intraday_daily_features_seeds_open_gap_across_year_shards(tmp_path) -> None:
    lake_root = tmp_path / "lake"
    lake = ResearchDataLake(lake_root)
    shard_dir = tmp_path / "raw_5m"
    shard_dir.mkdir()

    def write_day(path: Path, *, trade_date: str, prices: list[float]) -> None:
        opens = prices[:-1]
        closes = prices[1:]
        pd.DataFrame(
            {
                "symbol": ["000001.SZ"] * len(opens),
                "trade_date": [trade_date] * len(opens),
                "bar_time": ["09:30:00", "09:35:00", "09:40:00", "09:45:00", "09:50:00", "09:55:00"],
                "open": opens,
                "high": [max(open_, close) + 0.1 for open_, close in zip(opens, closes)],
                "low": [min(open_, close) - 0.1 for open_, close in zip(opens, closes)],
                "close": closes,
                "volume": [100.0] * len(opens),
                "amount": [close * 100.0 for close in closes],
                "source": ["unit"] * len(opens),
                "adjusted_flag": ["none"] * len(opens),
            }
        ).to_parquet(path, index=False)

    prior_path = shard_dir / "2025.parquet"
    target_path = shard_dir / "2026.parquet"
    write_day(prior_path, trade_date="2025-12-31", prices=[10.0, 10.1, 10.2, 10.3, 10.4, 10.45, 10.5])
    write_day(target_path, trade_date="2026-01-02", prices=[11.0, 11.1, 11.2, 11.3, 11.4, 11.45, 11.5])
    source = lake.save_sharded_domain_dataset(
        domain=DataDomain.MARKET_INTRADAY_5M,
        spec={
            "domain": DataDomain.MARKET_INTRADAY_5M,
            "dataset": "unit_cross_year_5m",
            "start_date": "2025-12-31",
            "end_date": "2026-01-02",
            "sharded": True,
        },
        shard_records=[
            {
                "domain": DataDomain.MARKET_INTRADAY_5M,
                "status": "stored",
                "path": str(prior_path.resolve()),
                "row_count": 6,
                "start_date": "2025-12-31",
                "end_date": "2025-12-31",
            },
            {
                "domain": DataDomain.MARKET_INTRADAY_5M,
                "status": "stored",
                "path": str(target_path.resolve()),
                "row_count": 6,
                "start_date": "2026-01-02",
                "end_date": "2026-01-02",
            },
        ],
        source="unit",
        reuse=False,
    )

    result = build_intraday_daily_features(
        BuildIntradayDailyFeaturesConfig(
            lake_root=lake_root,
            source_dataset_id=source.dataset_id,
            start_date="2026-01-01",
            end_date="2026-12-31",
            years=(2026,),
            workers=2,
            resume=False,
            reuse=False,
        )
    )

    assert result.status == "completed"
    assert result.shard_count == 1
    metadata = lake.describe_dataset(result.dataset_id)
    manifest_path = Path(metadata["content_paths"]["shard_manifest"])
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    features = pd.read_parquet(manifest["shards"][0]["path"])
    assert features["trade_date"].tolist() == ["2026-01-02"]
    assert float(features["open_gap"].iloc[0]) == pytest.approx(11.0 / 10.5 - 1.0)
    assert float(features["open_gap_first_30m_follow_through"].iloc[0]) == pytest.approx(11.5 / 11.0 - 1.0)
    assert float(features["open_gap_first_30m_reversal"].iloc[0]) == pytest.approx(-(11.5 / 11.0 - 1.0))


def test_intraday_daily_feature_dataset_spec_records_last_5m_contract(tmp_path) -> None:
    lake = ResearchDataLake(tmp_path / "lake")
    source_record = lake.save_sharded_domain_dataset(
        domain=DataDomain.MARKET_INTRADAY_5M,
        spec={
            "domain": DataDomain.MARKET_INTRADAY_5M,
            "dataset": "unit_5m",
            "start_date": "2026-01-05",
            "end_date": "2026-01-05",
            "sharded": True,
        },
        shard_records=[],
        source="unit",
        reuse=False,
    )

    cfg = BuildIntradayDailyFeaturesConfig(
        lake_root=tmp_path / "lake",
        source_dataset_id=source_record.dataset_id,
        start_date="2026-01-05",
        end_date="2026-01-05",
        dry_run=True,
    )
    result = build_intraday_daily_features(cfg)
    spec = _feature_dataset_spec(cfg.normalized(), source_metadata=lake.describe_dataset(source_record.dataset_id))

    identity = lake.build_domain_dataset_identity(
        domain=DataDomain.INTRADAY_DAILY_FEATURES,
        spec={**spec, "domain": DataDomain.INTRADAY_DAILY_FEATURES, "sharded": True},
    )

    assert result.status == "dry_run"
    assert identity["dataset_id"].startswith(f"data_platform_{DataDomain.INTRADAY_DAILY_FEATURES}__")
    assert spec["feature_contract_version"] == INTRADAY_DAILY_FEATURE_CONTRACT_VERSION
    assert spec["last_5m_ret_policy"] == LAST_5M_RET_POLICY


def test_build_intraday_daily_features_can_overlay_open_close_auction_from_1m(tmp_path) -> None:
    lake = ResearchDataLake(tmp_path / "lake")
    five_path = tmp_path / "five.parquet"
    one_path = tmp_path / "one.parquet"
    pd.DataFrame(
        {
            "symbol": ["000001.SZ", "000001.SZ"],
            "trade_date": ["2026-01-05", "2026-01-05"],
            "bar_time": ["093000000", "150000000"],
            "open": [10.0, 10.8],
            "high": [10.5, 10.9],
            "low": [9.9, 10.7],
            "close": [10.4, 10.85],
            "volume": [1000.0, 2000.0],
            "amount": [10400.0, 21700.0],
            "source": ["unit_5m", "unit_5m"],
            "adjusted_flag": ["none", "none"],
        }
    ).to_parquet(five_path, index=False)
    pd.DataFrame(
        {
            "symbol": ["000001.SZ", "000001.SZ", "000001.SZ"],
            "trade_date": ["2026-01-05", "2026-01-05", "2026-01-05"],
            "bar_time": ["093100000", "093200000", "150000000"],
            "open": [10.0, 10.2, 10.8],
            "high": [10.2, 10.3, 10.9],
            "low": [9.9, 10.1, 10.7],
            "close": [10.1, 10.25, 10.88],
            "volume": [100.0, 200.0, 300.0],
            "amount": [1010.0, 2050.0, 3264.0],
            "turnover_rate": [0.1, 0.2, 0.3],
            "float_share": [1.0, 1.0, 1.0],
            "total_share": [2.0, 2.0, 2.0],
            "source": ["unit_1m", "unit_1m", "unit_1m"],
            "adjusted_flag": ["none", "none", "none"],
        }
    ).to_parquet(one_path, index=False)
    five = lake.save_sharded_domain_dataset(
        domain=DataDomain.MARKET_INTRADAY_5M,
        spec={"domain": DataDomain.MARKET_INTRADAY_5M, "start_date": "2026-01-05", "end_date": "2026-01-05", "sharded": True},
        shard_records=[{"status": "stored", "path": str(five_path), "row_count": 2, "start_date": "2026-01-05", "end_date": "2026-01-05", "symbol_count": 1}],
        source="unit",
        reuse=False,
    )
    one = lake.save_sharded_domain_dataset(
        domain=DataDomain.MARKET_INTRADAY_1M,
        spec={"domain": DataDomain.MARKET_INTRADAY_1M, "start_date": "2026-01-05", "end_date": "2026-01-05", "sharded": True, "one_minute_policy": "mootdx_240_0930_merged_into_0931"},
        shard_records=[{"status": "stored", "path": str(one_path), "row_count": 3, "start_date": "2026-01-05", "end_date": "2026-01-05", "symbol_count": 1}],
        source="unit",
        reuse=False,
    )

    result = build_intraday_daily_features(
        BuildIntradayDailyFeaturesConfig(
            lake_root=tmp_path / "lake",
            source_dataset_id=five.dataset_id,
            auction_1m_dataset_id=one.dataset_id,
            start_date="2026-01-05",
            end_date="2026-01-05",
        )
    )

    feature_path = next((tmp_path / "lake").glob("parquet/bronze_silver/data_platform_intraday_daily_features/*/shards/*.parquet"))
    features = pd.read_parquet(feature_path)
    assert result.status == "completed"
    assert float(features["first_5m_ret"].iloc[0]) == pytest.approx(10.4 / 10.0 - 1.0)
    assert float(features["opening_auction_amount"].iloc[0]) == pytest.approx(1010.0)
    assert float(features["closing_auction_amount"].iloc[0]) == pytest.approx(3264.0)
    assert float(features["opening_auction_amount_share"].iloc[0]) == pytest.approx(1010.0 / (1010.0 + 2050.0 + 3264.0))
    metadata = lake.describe_dataset(result.dataset_id)
    assert metadata["parameters"]["auction_1m_dataset_id"] == one.dataset_id
    assert metadata["parameters"]["auction_feature_policy"] == "override_opening_closing_auction_fields_from_mootdx_240_1m_0931_1500"


def test_build_intraday_daily_features_can_overlay_existing_feature_sidecar_from_1m(tmp_path) -> None:
    lake = ResearchDataLake(tmp_path / "lake")
    feature_path = tmp_path / "feature.parquet"
    one_path = tmp_path / "one.parquet"
    pd.DataFrame(
        {
            "symbol": ["000001.SZ"],
            "trade_date": ["2026-01-05"],
            "first_5m_ret": [0.05],
            "opening_auction_ret": [0.05],
            "opening_auction_amount": [5000.0],
            "opening_auction_volume": [500.0],
            "opening_auction_amount_share": [0.5],
            "opening_auction_range": [0.06],
            "opening_auction_vwap": [10.0],
            "opening_auction_pressure": [0.025],
            "last_5m_ret": [0.01],
            "closing_auction_ret": [0.01],
            "closing_auction_amount": [8000.0],
            "closing_auction_volume": [800.0],
            "closing_auction_amount_share": [0.8],
            "closing_auction_range": [0.02],
            "closing_auction_vwap": [10.0],
            "closing_auction_pressure": [0.008],
            "source": ["legacy_feature"],
            "adjusted_flag": ["none"],
        }
    ).to_parquet(feature_path, index=False)
    pd.DataFrame(
        {
            "symbol": ["000001.SZ", "000001.SZ", "000001.SZ"],
            "trade_date": ["2026-01-05", "2026-01-05", "2026-01-05"],
            "bar_time": ["093100000", "093200000", "150000000"],
            "open": [10.0, 10.2, 10.8],
            "high": [10.2, 10.3, 10.9],
            "low": [9.9, 10.1, 10.7],
            "close": [10.1, 10.25, 10.88],
            "volume": [100.0, 200.0, 300.0],
            "amount": [1010.0, 2050.0, 3264.0],
            "turnover_rate": [0.1, 0.2, 0.3],
            "float_share": [1.0, 1.0, 1.0],
            "total_share": [2.0, 2.0, 2.0],
            "source": ["unit_1m", "unit_1m", "unit_1m"],
            "adjusted_flag": ["none", "none", "none"],
        }
    ).to_parquet(one_path, index=False)
    feature = lake.save_sharded_domain_dataset(
        domain=DataDomain.INTRADAY_DAILY_FEATURES,
        spec={"domain": DataDomain.INTRADAY_DAILY_FEATURES, "start_date": "2026-01-05", "end_date": "2026-01-05", "sharded": True},
        shard_records=[
            {
                "status": "stored",
                "path": str(feature_path),
                "row_count": 1,
                "start_date": "2026-01-05",
                "end_date": "2026-01-05",
                "symbol_count": 1,
                "source_member": "2026_market_intraday_5m_batch_00001",
            }
        ],
        source="unit",
        reuse=False,
    )
    one = lake.save_sharded_domain_dataset(
        domain=DataDomain.MARKET_INTRADAY_1M,
        spec={"domain": DataDomain.MARKET_INTRADAY_1M, "start_date": "2026-01-05", "end_date": "2026-01-05", "sharded": True},
        shard_records=[
            {
                "status": "stored",
                "path": str(one_path),
                "row_count": 3,
                "start_date": "2026-01-05",
                "end_date": "2026-01-05",
                "symbol_count": 1,
                "source_member": "2026_market_intraday_1m_batch_00001",
            }
        ],
        source="unit",
        reuse=False,
    )

    result = build_intraday_daily_features(
        BuildIntradayDailyFeaturesConfig(
            lake_root=tmp_path / "lake",
            existing_feature_dataset_id=feature.dataset_id,
            auction_1m_dataset_id=one.dataset_id,
            start_date="2026-01-05",
            end_date="2026-01-05",
        )
    )

    out_path = next((tmp_path / "lake").glob("parquet/bronze_silver/data_platform_intraday_daily_features/*/shards/*auction_1m_overlay.parquet"))
    out = pd.read_parquet(out_path)
    assert result.status == "completed"
    assert float(out["first_5m_ret"].iloc[0]) == pytest.approx(0.05)
    assert float(out["opening_auction_amount"].iloc[0]) == pytest.approx(1010.0)
    assert float(out["closing_auction_amount"].iloc[0]) == pytest.approx(3264.0)
    assert float(out["opening_auction_amount_share"].iloc[0]) == pytest.approx(1010.0 / (1010.0 + 2050.0 + 3264.0))
    metadata = lake.describe_dataset(result.dataset_id)
    assert metadata["parameters"]["existing_feature_dataset_id"] == feature.dataset_id
    assert metadata["parameters"]["auction_1m_dataset_id"] == one.dataset_id


def test_auction_1m_overlay_matches_source_member_before_date_fallback() -> None:
    shards = [
        {
            "path": "H:/lake/2016_2016_market_intraday_1m_batch_00001.parquet",
            "source_member": "2016_market_intraday_1m_batch_00001",
            "start_date": "2016-01-01",
            "end_date": "2016-12-31",
        },
        {
            "path": "H:/lake/2016_2016_market_intraday_1m_batch_00002.parquet",
            "source_member": "2016_market_intraday_1m_batch_00002",
            "start_date": "2016-01-01",
            "end_date": "2016-12-31",
        },
    ]
    index = _build_auction_1m_shard_index(shards)

    matches = _matching_auction_1m_shards(
        source_record={
            "path": "H:/lake/2016_2016_market_intraday_5m_batch_00002_derived_5m.parquet",
            "source_member": "2016_market_intraday_5m_batch_00002",
            "start_date": "2016-01-01",
            "end_date": "2016-12-31",
        },
        auction_1m_index=index,
        start_date="2016-01-04",
        end_date="2016-12-30",
    )

    assert len(matches) == 1
    assert matches[0]["source_member"] == "2016_market_intraday_1m_batch_00002"


def test_auction_1m_overlay_matches_mootdx_symbol_block_start() -> None:
    index = _build_auction_1m_shard_index(
        [
            {
                "path": "H:/lake/market_intraday_1m__2026-03-30_2026-06-26__s000001_0050_a71f206dc0.parquet",
                "chunk_id": "market_intraday_1m__2026-03-30_2026-06-26__s000001_0050_a71f206dc0",
                "start_date": "2026-03-30",
                "end_date": "2026-06-26",
            },
            {
                "path": "H:/lake/market_intraday_1m__2026-03-30_2026-06-26__s000104_0050_430d2c0790.parquet",
                "chunk_id": "market_intraday_1m__2026-03-30_2026-06-26__s000104_0050_430d2c0790",
                "start_date": "2026-03-30",
                "end_date": "2026-06-26",
            },
            {
                "path": "H:/lake/market_intraday_1m__2026-03-30_2026-06-26__s000105_0050_51fcd4d797.parquet",
                "chunk_id": "market_intraday_1m__2026-03-30_2026-06-26__s000105_0050_51fcd4d797",
                "start_date": "2026-03-30",
                "end_date": "2026-06-26",
            },
        ]
    )

    matches = _matching_auction_1m_shards(
        source_record={
            "path": "H:/lake/market_intraday_5m__2026-06-11_2026-06-26__s000053_0008_430d2c0790.parquet",
            "chunk_id": "market_intraday_5m__2026-06-11_2026-06-26__s000053_0008_430d2c0790",
            "start_date": "2026-06-11",
            "end_date": "2026-06-26",
        },
        auction_1m_index=index,
        start_date="2026-06-11",
        end_date="2026-06-26",
    )

    assert len(matches) == 1
    assert "s000105" in matches[0]["path"]


def test_auction_1m_overlay_prefers_source_raw_chunk_over_feature_chunk() -> None:
    index = _build_auction_1m_shard_index(
        [
            {
                "path": "H:/lake/market_intraday_1m__2026-03-30_2026-06-26__s000009_0050_bad.parquet",
                "chunk_id": "market_intraday_1m__2026-03-30_2026-06-26__s000009_0050_bad",
                "start_date": "2026-03-30",
                "end_date": "2026-06-26",
            },
            {
                "path": "H:/lake/market_intraday_1m__2026-03-30_2026-06-26__s000105_0008_good.parquet",
                "chunk_id": "market_intraday_1m__2026-03-30_2026-06-26__s000105_0008_good",
                "start_date": "2026-03-30",
                "end_date": "2026-06-26",
            },
        ]
    )

    matches = _matching_auction_1m_shards(
        source_record={
            "chunk_id": "intraday_daily_features__2026-06-11_2026-06-26__s000053_0008_bad",
            "source_raw_chunk_id": "market_intraday_5m__2026-06-11_2026-06-26__s000053_0008_good",
            "start_date": "2026-06-11",
            "end_date": "2026-06-26",
        },
        auction_1m_index=index,
        start_date="2026-06-11",
        end_date="2026-06-26",
    )

    assert len(matches) == 1
    assert matches[0]["chunk_id"] == "market_intraday_1m__2026-03-30_2026-06-26__s000105_0008_good"


def test_auction_1m_overlay_does_not_return_partial_incomplete_symbol_sample() -> None:
    index = _build_auction_1m_shard_index(
        [
            {
                "path": "H:/lake/partial_sample.parquet",
                "source_members_sample": ["sz000021.csv"],
                "source_member_count": 64,
                "start_date": "2026-01-05",
                "end_date": "2026-03-27",
            },
            {
                "path": "H:/lake/year_candidate.parquet",
                "start_date": "2026-01-05",
                "end_date": "2026-03-27",
            },
        ]
    )

    matches = _matching_auction_1m_shards(
        source_record={"start_date": "2026-01-05", "end_date": "2026-03-27"},
        auction_1m_index=index,
        start_date="2026-01-05",
        end_date="2026-03-27",
        feature_symbols={"000001.SZ", "000021.SZ"},
    )

    assert {item["path"] for item in matches} == {"H:/lake/partial_sample.parquet", "H:/lake/year_candidate.parquet"}


def test_intraday_daily_feature_year_filter_uses_shard_date_bounds() -> None:
    assert _shard_intersects_years({"start_date": "2024-12-20", "end_date": "2025-01-10"}, {2025}) is True
    assert _shard_intersects_years({"start_date": "2024-01-01", "end_date": "2024-12-31"}, {2025}) is False
    assert _shard_intersects_years({"start_date": "", "end_date": ""}, {2025}) is True


def test_import_external_quant_zip_streams_1m_and_derives_5m(tmp_path) -> None:
    source_root = tmp_path / "量化数据"
    zip_dir = source_root / "1分钟"
    zip_dir.mkdir(parents=True)
    csv_text = "\n".join(
        [
            "日期,开盘,最高,最低,收盘,成交量(股),成交额(元)",
            "2010-01-04 09:30:00,10.0,10.2,9.9,10.1,100,1010",
            "2010-01-04 09:31:00,10.1,10.3,10.0,10.2,200,2040",
            "2010-01-04 09:32:00,10.2,10.4,10.1,10.3,300,3090",
            "2010-01-04 09:33:00,10.3,10.5,10.2,10.4,400,4160",
            "2010-01-04 09:34:00,10.4,10.6,10.3,10.5,500,5250",
        ]
    )
    with zipfile.ZipFile(zip_dir / "2010.zip", "w") as archive:
        archive.writestr("sz000001.csv", csv_text)

    result = run_import(
        ImportConfig(
            lake_root=tmp_path / "lake",
            source_root=source_root,
            domains=(DataDomain.MARKET_INTRADAY_1M,),
            start_date="2010-01-01",
            years=(2010,),
            derive_5m_from_1m=True,
            hash_zips=False,
        )
    )

    assert result.status == "completed"
    assert result.row_counts[DataDomain.MARKET_INTRADAY_1M] == 4
    assert result.row_counts[DataDomain.MARKET_INTRADAY_5M] == 1
    one_minute = pd.read_parquet(
        next((tmp_path / "lake").glob("parquet/bronze_silver/data_platform_market_intraday_1m/*/shards/*.parquet"))
    )
    five_minute = pd.read_parquet(
        next((tmp_path / "lake").glob("parquet/bronze_silver/data_platform_market_intraday_5m/*/shards/*.parquet"))
    )
    assert one_minute["symbol"].iloc[0] == "000001.SZ"
    assert "093000000" not in set(one_minute["bar_time"])
    first_bar = one_minute.loc[one_minute["bar_time"].eq("093100000")].iloc[0]
    assert float(first_bar["open"]) == pytest.approx(10.0)
    assert float(first_bar["high"]) == pytest.approx(10.3)
    assert float(first_bar["low"]) == pytest.approx(9.9)
    assert float(first_bar["close"]) == pytest.approx(10.2)
    assert float(first_bar["volume"]) == pytest.approx(300.0)
    assert float(first_bar["amount"]) == pytest.approx(3050.0)
    assert five_minute["bar_time"].iloc[0] == "093000000"
    assert float(five_minute["volume"].iloc[0]) == 1500.0


def test_normalize_intraday_1m_to_mootdx_240_ignores_zero_opening_auction_prices() -> None:
    frame = pd.DataFrame(
        {
            "symbol": ["603277.SH", "603277.SH"],
            "trade_date": ["2026-01-06", "2026-01-06"],
            "bar_time": ["093000000", "093100000"],
            "open": [0.0, 16.81],
            "high": [0.0, 16.81],
            "low": [0.0, 16.76],
            "close": [0.0, 16.77],
            "volume": [0.0, 8700.0],
            "amount": [0.0, 145947.0],
            "turnover_rate": [0.0, 0.001427],
            "float_share": [609746015.0, 609746015.0],
            "total_share": [613699296.0, 613699296.0],
            "source": ["external_quant_csv", "external_quant_csv"],
            "adjusted_flag": ["none", "none"],
        }
    )

    normalized = normalize_intraday_1m_to_mootdx_240_frame(frame)

    assert normalized["bar_time"].tolist() == ["093100000"]
    row = normalized.iloc[0]
    assert float(row["open"]) == pytest.approx(16.81)
    assert float(row["high"]) == pytest.approx(16.81)
    assert float(row["low"]) == pytest.approx(16.76)
    assert float(row["close"]) == pytest.approx(16.77)
    assert float(row["volume"]) == pytest.approx(8700.0)
    assert float(row["amount"]) == pytest.approx(145947.0)


def test_import_external_quant_zip_does_not_derive_5m_from_1m_by_default(tmp_path) -> None:
    source_root = tmp_path / "量化数据"
    zip_dir = source_root / "1分钟"
    zip_dir.mkdir(parents=True)
    csv_text = "\n".join(
        [
            "日期,开盘,最高,最低,收盘,成交量(股),成交额(元)",
            "2010-01-04 09:30:00,10.0,10.2,9.9,10.1,100,1010",
            "2010-01-04 09:31:00,10.1,10.3,10.0,10.2,200,2040",
            "2010-01-04 09:32:00,10.2,10.4,10.1,10.3,300,3090",
            "2010-01-04 09:33:00,10.3,10.5,10.2,10.4,400,4160",
            "2010-01-04 09:34:00,10.4,10.6,10.3,10.5,500,5250",
        ]
    )
    with zipfile.ZipFile(zip_dir / "2010.zip", "w") as archive:
        archive.writestr("sz000001.csv", csv_text)

    result = run_import(
        ImportConfig(
            lake_root=tmp_path / "lake",
            source_root=source_root,
            domains=(DataDomain.MARKET_INTRADAY_1M,),
            start_date="2010-01-01",
            years=(2010,),
            hash_zips=False,
        )
    )

    assert result.status == "completed"
    assert result.row_counts[DataDomain.MARKET_INTRADAY_1M] == 4
    assert DataDomain.MARKET_INTRADAY_5M not in result.row_counts
    assert not list((tmp_path / "lake").glob("parquet/bronze_silver/data_platform_market_intraday_5m/*/shards/*.parquet"))


def test_import_external_quant_zip_discovers_unpacked_csv_only_when_enabled(tmp_path) -> None:
    source_root = tmp_path / "量化数据"
    csv_dir = source_root / "2026" / "1分钟"
    csv_dir.mkdir(parents=True)
    (csv_dir / "sz000001.csv").write_text(
        "\n".join(
            [
                "日期,开盘,最高,最低,收盘,成交量(股),成交额(元)",
                "2026-01-05 09:30:00,10.0,10.2,9.9,10.1,100,1010",
            ]
        ),
        encoding="utf-8",
    )

    without_unpacked = run_import(
        ImportConfig(
            lake_root=tmp_path / "lake_without",
            source_root=source_root,
            domains=(DataDomain.MARKET_INTRADAY_1M,),
            start_date="2026-01-01",
            years=(2026,),
            dry_run=True,
        )
    )
    with_unpacked = run_import(
        ImportConfig(
            lake_root=tmp_path / "lake_with",
            source_root=source_root,
            domains=(DataDomain.MARKET_INTRADAY_1M,),
            start_date="2026-01-01",
            years=(2026,),
            dry_run=True,
            include_unpacked_csv=True,
        )
    )

    without_plan = json.loads(Path(without_unpacked.plan_path).read_text(encoding="utf-8"))
    with_plan = json.loads(Path(with_unpacked.plan_path).read_text(encoding="utf-8"))
    assert without_plan["source_paths_by_domain"][DataDomain.MARKET_INTRADAY_1M] == []
    assert with_plan["source_paths_by_domain"][DataDomain.MARKET_INTRADAY_1M] == [
        {"path": str(csv_dir), "kind": "directory"}
    ]


def test_import_external_quant_zip_imports_unpacked_1m_csv_normalizes_to_mootdx_240(tmp_path) -> None:
    source_root = tmp_path / "量化数据"
    csv_dir = source_root / "2026" / "1分钟"
    csv_dir.mkdir(parents=True)
    (csv_dir / "sz000001.csv").write_text(
        "\n".join(
            [
                "日期,开盘,最高,最低,收盘,成交量(股),成交额(元)",
                "2026-01-05 09:30:00,10.0,10.2,9.9,10.1,100,1010",
                "2026-01-05 09:31:00,10.1,10.3,10.0,10.2,200,2040",
            ]
        ),
        encoding="utf-8",
    )

    lake_root = tmp_path / "lake"
    result = run_import(
        ImportConfig(
            lake_root=lake_root,
            source_root=source_root,
            domains=(DataDomain.MARKET_INTRADAY_1M,),
            start_date="2026-01-01",
            end_date="2026-01-05",
            years=(2026,),
            include_unpacked_csv=True,
            shard_batch_members=8,
            hash_zips=False,
        )
    )

    assert result.status == "completed"
    assert result.row_counts[DataDomain.MARKET_INTRADAY_1M] == 1
    lake = ResearchDataLake(lake_root)
    metadata = lake.describe_dataset(result.dataset_ids[DataDomain.MARKET_INTRADAY_1M])
    assert metadata["parameters"]["one_minute_policy"] == "mootdx_240_0930_merged_into_0931"
    frame = pd.read_parquet(
        next(lake_root.glob("parquet/bronze_silver/data_platform_market_intraday_1m/*/shards/*.parquet"))
    )
    assert frame["symbol"].tolist() == ["000001.SZ"]
    assert frame["bar_time"].tolist() == ["093100000"]
    assert float(frame["open"].iloc[0]) == pytest.approx(10.0)
    assert float(frame["volume"].iloc[0]) == pytest.approx(300.0)


def test_import_external_quant_zip_can_preserve_source_1m_bars(tmp_path) -> None:
    source_root = tmp_path / "量化数据"
    csv_dir = source_root / "2026" / "1分钟"
    csv_dir.mkdir(parents=True)
    (csv_dir / "sz000001.csv").write_text(
        "\n".join(
            [
                "日期,开盘,最高,最低,收盘,成交量(股),成交额(元)",
                "2026-01-05 09:30:00,10.0,10.2,9.9,10.1,100,1010",
                "2026-01-05 09:31:00,10.1,10.3,10.0,10.2,200,2040",
            ]
        ),
        encoding="utf-8",
    )

    lake_root = tmp_path / "lake"
    result = run_import(
        ImportConfig(
            lake_root=lake_root,
            source_root=source_root,
            domains=(DataDomain.MARKET_INTRADAY_1M,),
            start_date="2026-01-01",
            end_date="2026-01-05",
            years=(2026,),
            include_unpacked_csv=True,
            normalize_intraday_1m_to_mootdx_240=False,
            hash_zips=False,
        )
    )

    assert result.status == "completed"
    assert result.row_counts[DataDomain.MARKET_INTRADAY_1M] == 2
    frame = pd.read_parquet(
        next(lake_root.glob("parquet/bronze_silver/data_platform_market_intraday_1m/*/shards/*.parquet"))
    )
    assert frame["bar_time"].tolist() == ["093000000", "093100000"]


def test_normalize_intraday_1m_contract_rewrites_existing_241_shard(tmp_path) -> None:
    lake = ResearchDataLake(tmp_path / "lake")
    shard_dir = tmp_path / "source_shards"
    shard_dir.mkdir()
    raw_path = shard_dir / "raw_1m.parquet"
    raw = pd.DataFrame(
        {
            "symbol": ["000001.SZ", "000001.SZ", "000001.SZ"],
            "trade_date": ["2026-01-05", "2026-01-05", "2026-01-05"],
            "bar_time": ["093000000", "093100000", "093200000"],
            "open": [10.0, 10.1, 10.2],
            "high": [10.2, 10.3, 10.4],
            "low": [9.9, 10.0, 10.1],
            "close": [10.1, 10.2, 10.3],
            "volume": [100, 200, 300],
            "amount": [1010, 2040, 3090],
            "turnover_rate": [0.1, 0.2, 0.3],
            "float_share": [1.0, 1.0, 1.0],
            "total_share": [2.0, 2.0, 2.0],
            "source": ["external_quant_zip"] * 3,
            "adjusted_flag": ["none"] * 3,
        }
    )
    raw.to_parquet(raw_path, index=False)
    record = lake.save_sharded_domain_dataset(
        domain=DataDomain.MARKET_INTRADAY_1M,
        spec={
            "domain": DataDomain.MARKET_INTRADAY_1M,
            "start_date": "2026-01-05",
            "end_date": "2026-01-05",
            "sharded": True,
            "one_minute_policy": "store_raw_1m_bars_preserving_source_bar_time_contract",
        },
        shard_records=[
            {
                "domain": DataDomain.MARKET_INTRADAY_1M,
                "status": "stored",
                "path": str(raw_path),
                "row_count": 3,
                "start_date": "2026-01-05",
                "end_date": "2026-01-05",
                "symbol_count": 1,
            }
        ],
        source="unit",
        reuse=False,
    )

    result = normalize_intraday_1m_contract(
        NormalizeIntraday1mContractConfig(
            lake_root=tmp_path / "lake",
            source_dataset_id=record.dataset_id,
        )
    )

    assert result.status == "completed"
    assert result.row_count == 2
    assert result.removed_0930_rows == 1
    metadata = lake.describe_dataset(result.dataset_id)
    assert metadata["parameters"]["one_minute_policy"] == "mootdx_240_0930_merged_into_0931"
    normalized = pd.read_parquet(
        next((tmp_path / "lake").glob("parquet/bronze_silver/data_platform_market_intraday_1m/*/shards/*mootdx_240.parquet"))
    )
    assert normalized["bar_time"].tolist() == ["093100000", "093200000"]
    assert float(normalized["open"].iloc[0]) == pytest.approx(10.0)
    assert float(normalized["volume"].iloc[0]) == pytest.approx(300.0)


def test_normalize_intraday_1m_contract_has_stable_target_identity(tmp_path) -> None:
    lake = ResearchDataLake(tmp_path / "lake")
    shard_dir = tmp_path / "source_shards"
    shard_dir.mkdir()
    raw_path = shard_dir / "raw_1m.parquet"
    pd.DataFrame(
        {
            "symbol": ["000001.SZ", "000001.SZ"],
            "trade_date": ["2026-01-05", "2026-01-05"],
            "bar_time": ["09:30:00", "09:31:00"],
            "open": [10.0, 10.1],
            "high": [10.2, 10.3],
            "low": [9.9, 10.0],
            "close": [10.1, 10.2],
            "volume": [100, 200],
            "amount": [1010, 2040],
            "turnover_rate": [0.1, 0.2],
            "float_share": [1.0, 1.0],
            "total_share": [2.0, 2.0],
            "source": ["external_quant_zip", "external_quant_zip"],
            "adjusted_flag": ["none", "none"],
        }
    ).to_parquet(raw_path, index=False)
    record = lake.save_sharded_domain_dataset(
        domain=DataDomain.MARKET_INTRADAY_1M,
        spec={
            "domain": DataDomain.MARKET_INTRADAY_1M,
            "start_date": "2026-01-05",
            "end_date": "2026-01-05",
            "sharded": True,
            "one_minute_policy": "store_raw_1m_bars_preserving_source_bar_time_contract",
        },
        shard_records=[
            {
                "domain": DataDomain.MARKET_INTRADAY_1M,
                "status": "stored",
                "path": str(raw_path),
                "row_count": 2,
                "start_date": "2026-01-05",
                "end_date": "2026-01-05",
                "symbol_count": 1,
            }
        ],
        source="unit",
        reuse=False,
    )
    cfg = NormalizeIntraday1mContractConfig(lake_root=tmp_path / "lake", source_dataset_id=record.dataset_id)

    first = normalize_intraday_1m_contract(cfg)
    second = normalize_intraday_1m_contract(cfg)

    assert first.dataset_id == second.dataset_id
    assert second.row_count == 1


def test_import_external_quant_zip_reuses_existing_same_spec_dataset(tmp_path) -> None:
    source_root = tmp_path / "量化数据"
    csv_dir = source_root / "2026" / "1分钟"
    csv_dir.mkdir(parents=True)
    (csv_dir / "sz000001.csv").write_text(
        "\n".join(
            [
                "日期,开盘,最高,最低,收盘,成交量(股),成交额(元)",
                "2026-01-05 09:30:00,10.0,10.2,9.9,10.1,100,1010",
            ]
        ),
        encoding="utf-8",
    )
    cfg = ImportConfig(
        lake_root=tmp_path / "lake",
        source_root=source_root,
        domains=(DataDomain.MARKET_INTRADAY_1M,),
        start_date="2026-01-01",
        end_date="2026-01-05",
        years=(2026,),
        include_unpacked_csv=True,
        hash_zips=False,
    )

    first = run_import(cfg)
    second = run_import(cfg)

    assert first.status == "completed"
    assert second.status == "completed_reused"
    assert second.dataset_ids == first.dataset_ids
    assert second.row_counts[DataDomain.MARKET_INTRADAY_1M] == 1


def test_recover_external_quant_zip_import_links_completed_year_shards(tmp_path) -> None:
    source_root = tmp_path / "量化数据"
    zip_dir = source_root / "1分钟"
    zip_dir.mkdir(parents=True)
    csv_text = "\n".join(
        [
            "日期,开盘,最高,最低,收盘,成交量(股),成交额(元)",
            "2010-01-04 09:30:00,10.0,10.2,9.9,10.1,100,1010",
            "2010-01-04 09:31:00,10.1,10.3,10.0,10.2,200,2040",
            "2010-01-04 09:32:00,10.2,10.4,10.1,10.3,300,3090",
            "2010-01-04 09:33:00,10.3,10.5,10.2,10.4,400,4160",
            "2010-01-04 09:34:00,10.4,10.6,10.3,10.5,500,5250",
        ]
    )
    with zipfile.ZipFile(zip_dir / "2010.zip", "w") as archive:
        archive.writestr("sz000001.csv", csv_text)

    progress_path = tmp_path / "progress.jsonl"
    lake_root = tmp_path / "lake"
    run_import(
        ImportConfig(
            lake_root=lake_root,
            source_root=source_root,
            domains=(DataDomain.MARKET_INTRADAY_1M,),
            start_date="2010-01-01",
            years=(),
            derive_5m_from_1m=True,
            hash_zips=False,
            progress_path=progress_path,
        )
    )

    result = recover_import(
        RecoverExternalImportConfig(
            lake_root=lake_root,
            source_root=source_root,
            progress_path=progress_path,
            domains=(DataDomain.MARKET_INTRADAY_1M, DataDomain.MARKET_INTRADAY_5M),
            start_date="2010-01-01",
            years=(2010,),
            derive_5m_from_1m=True,
        )
    )

    assert result.status == "completed"
    assert result.completed_years == [2010]
    assert result.row_counts[DataDomain.MARKET_INTRADAY_1M] == 4
    assert result.row_counts[DataDomain.MARKET_INTRADAY_5M] == 1
    assert result.dataset_ids[DataDomain.MARKET_INTRADAY_1M]
    assert result.dataset_ids[DataDomain.MARKET_INTRADAY_5M]
    lake = ResearchDataLake(lake_root)
    metadata = lake.describe_dataset(result.dataset_ids[DataDomain.MARKET_INTRADAY_1M])
    manifest = json.loads(Path(metadata["content_paths"]["shard_manifest"]).read_text(encoding="utf-8"))
    assert Path(manifest["shards"][0]["path"]).exists()
    assert manifest["shards"][0]["recovery_materialization"] == "manifest_reference"


def test_combine_sharded_domain_datasets_links_source_shards(tmp_path) -> None:
    source_root = tmp_path / "量化数据"
    zip_dir = source_root / "1分钟"
    zip_dir.mkdir(parents=True)
    csv_template = "\n".join(
        [
            "日期,开盘,最高,最低,收盘,成交量(股),成交额(元)",
            "{year}-01-04 09:30:00,10.0,10.2,9.9,10.1,100,1010",
        ]
    )
    for year in (2010, 2011):
        with zipfile.ZipFile(zip_dir / f"{year}.zip", "w") as archive:
            archive.writestr("sz000001.csv", csv_template.format(year=year))

    lake_root = tmp_path / "lake"
    first = run_import(
        ImportConfig(
            lake_root=lake_root,
            source_root=source_root,
            domains=(DataDomain.MARKET_INTRADAY_1M,),
            start_date="2010-01-01",
            years=(2010,),
            derive_5m_from_1m=True,
            hash_zips=False,
        )
    )
    second = run_import(
        ImportConfig(
            lake_root=lake_root,
            source_root=source_root,
            domains=(DataDomain.MARKET_INTRADAY_1M,),
            start_date="2010-01-01",
            years=(2011,),
            derive_5m_from_1m=True,
            hash_zips=False,
        )
    )

    result = combine_sharded_domain_datasets(
        CombineShardedDomainConfig(
            lake_root=lake_root,
            source_root=source_root,
            domain=DataDomain.MARKET_INTRADAY_1M,
            source_dataset_ids=(
                first.dataset_ids[DataDomain.MARKET_INTRADAY_1M],
                second.dataset_ids[DataDomain.MARKET_INTRADAY_1M],
            ),
            start_date="2010-01-01",
            years=(2010, 2011),
        )
    )

    assert result.status == "completed"
    assert result.shard_count == 2
    assert result.row_count == 2
    assert result.dataset_id
    lake = ResearchDataLake(lake_root)
    metadata = lake.describe_dataset(result.dataset_id)
    manifest = json.loads(Path(metadata["content_paths"]["shard_manifest"]).read_text(encoding="utf-8"))
    assert {item["combine_materialization"] for item in manifest["shards"]} == {"manifest_reference"}


def test_combine_sharded_domain_datasets_keeps_chunk_id_only_shards(tmp_path) -> None:
    lake_root = tmp_path / "lake"
    lake = ResearchDataLake(lake_root)
    shard_dir = tmp_path / "shards"
    shard_dir.mkdir()
    records = []
    for idx in (1, 2):
        path = shard_dir / f"market_intraday_1m__2026-03-30_2026-06-26__s{idx:06d}_0050_unit.parquet"
        pd.DataFrame(
            {
                "symbol": [f"{idx:06d}.SZ"],
                "trade_date": ["2026-03-30"],
                "bar_time": ["093100000"],
                "open": [10.0],
                "high": [10.1],
                "low": [9.9],
                "close": [10.0],
                "volume": [100.0],
                "amount": [1000.0],
            }
        ).to_parquet(path, index=False)
        records.append(
            {
                "domain": DataDomain.MARKET_INTRADAY_1M,
                "status": "stored",
                "path": str(path),
                "row_count": 1,
                "start_date": "2026-03-30",
                "end_date": "2026-03-30",
                "symbol_count": 1,
                "chunk_id": path.stem,
            }
        )
    source = lake.save_sharded_domain_dataset(
        domain=DataDomain.MARKET_INTRADAY_1M,
        spec={
            "domain": DataDomain.MARKET_INTRADAY_1M,
            "start_date": "2026-03-30",
            "end_date": "2026-03-30",
            "sharded": True,
            "expected_1m_bars_per_day": 240,
        },
        shard_records=records,
        source="unit",
        reuse=False,
    )

    result = combine_sharded_domain_datasets(
        CombineShardedDomainConfig(
            lake_root=lake_root,
            domain=DataDomain.MARKET_INTRADAY_1M,
            source_dataset_ids=(source.dataset_id,),
            start_date="2026-03-30",
            end_date="2026-03-30",
            years=(2026,),
            reuse=False,
        )
    )

    assert result.status == "completed"
    assert result.shard_count == 2
    assert result.row_count == 2


def test_build_intraday_daily_features_from_imported_5m_dataset(tmp_path) -> None:
    source_root = tmp_path / "量化数据"
    zip_dir = source_root / "1分钟"
    zip_dir.mkdir(parents=True)
    csv_text = "\n".join(
        [
            "日期,开盘,最高,最低,收盘,成交量(股),成交额(元)",
            "2010-01-04 09:30:00,10.0,10.2,9.9,10.1,100,1010",
            "2010-01-04 09:31:00,10.1,10.3,10.0,10.2,200,2040",
            "2010-01-04 09:32:00,10.2,10.4,10.1,10.3,300,3090",
            "2010-01-04 09:33:00,10.3,10.5,10.2,10.4,400,4160",
            "2010-01-04 09:34:00,10.4,10.6,10.3,10.5,500,5250",
        ]
    )
    with zipfile.ZipFile(zip_dir / "2010.zip", "w") as archive:
        archive.writestr("sz000001.csv", csv_text)

    lake_root = tmp_path / "lake"
    imported = run_import(
        ImportConfig(
            lake_root=lake_root,
            source_root=source_root,
            domains=(DataDomain.MARKET_INTRADAY_1M,),
            start_date="2010-01-01",
            years=(2010,),
            derive_5m_from_1m=True,
            hash_zips=False,
        )
    )

    result = build_intraday_daily_features(
        BuildIntradayDailyFeaturesConfig(
            lake_root=lake_root,
            source_dataset_id=imported.dataset_ids[DataDomain.MARKET_INTRADAY_5M],
            start_date="2010-01-01",
            years=(2010,),
        )
    )

    assert result.status == "completed"
    assert result.row_count == 1
    feature_path = next(lake_root.glob("parquet/bronze_silver/data_platform_intraday_daily_features/*/shards/*.parquet"))
    features = pd.read_parquet(feature_path)
    assert features["symbol"].iloc[0] == "000001.SZ"
    assert "first_5m_ret" in features.columns
    assert "opening_auction_pressure" in features.columns
    assert "closing_auction_pressure" in features.columns


def test_combine_domain_datasets_references_sidecar_and_policy_bundle_paths(tmp_path) -> None:
    lake = ResearchDataLake(tmp_path / "lake")
    valuation = pd.DataFrame(
        {
            "symbol": ["000001.SZ"],
            "trade_date": ["2016-01-04"],
            "total_mv": [1.0],
            "circ_mv": [1.0],
            "pe": [10.0],
            "pb": [1.0],
            "turnover_rate": [0.5],
            "source": ["unit"],
        }
    )
    valuation_record = lake.save_domain_dataset(
        domain=DataDomain.VALUATION,
        frame=valuation,
        spec={"dataset": "data_platform_valuation", "domain": DataDomain.VALUATION, "start_date": "2016-01-01", "end_date": "2016-01-31"},
        source="unit",
    )
    valuation_combined = combine_domain_datasets(
        CombineDomainDatasetsConfig(
            lake_root=tmp_path / "lake",
            domain=DataDomain.VALUATION,
            source_dataset_ids=(valuation_record.dataset_id,),
            start_date="2016-01-01",
            end_date="2016-01-31",
        )
    )
    assert valuation_combined.status == "completed"
    assert valuation_combined.row_count == 1

    date_index = pd.to_datetime(["2016-01-04"])
    market_record = lake.save_market_data_bundle(
        spec={"dataset": "policy_input_bundle", "start_date": "2016-01-01", "end_date": "2016-01-31", "benchmark": "000300.SH"},
        market_frames={
            "Open": pd.DataFrame({"000001.SZ": [10.0]}, index=date_index),
            "High": pd.DataFrame({"000001.SZ": [11.0]}, index=date_index),
            "Low": pd.DataFrame({"000001.SZ": [9.0]}, index=date_index),
            "Close": pd.DataFrame({"000001.SZ": [10.5]}, index=date_index),
            "Volume": pd.DataFrame({"000001.SZ": [1000.0]}, index=date_index),
            "Amount": pd.DataFrame({"000001.SZ": [10500.0]}, index=date_index),
        },
        benchmark_close=pd.Series([4000.0], index=date_index, name="000300.SH"),
        benchmark_open=pd.Series([3990.0], index=date_index, name="000300.SH"),
        membership_frame=pd.DataFrame({"000001.SZ": [True]}, index=date_index),
        feature_frames={},
        source="unit",
    )
    market_combined = combine_domain_datasets(
        CombineDomainDatasetsConfig(
            lake_root=tmp_path / "lake",
            domain=DataDomain.MARKET_DAILY,
            source_dataset_ids=(market_record.dataset_id,),
            start_date="2016-01-01",
            end_date="2016-01-31",
        )
    )
    assert market_combined.status == "completed"
    assert market_combined.row_count == 1
