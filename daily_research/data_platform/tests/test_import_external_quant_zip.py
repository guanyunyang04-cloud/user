from __future__ import annotations

import json
import zipfile
from pathlib import Path

import pandas as pd

from daily_research.data_lake.catalog import ResearchDataLake
from daily_research.data_platform.contracts import DataDomain
from daily_research.data_platform.build_intraday_daily_features import BuildIntradayDailyFeaturesConfig, build_intraday_daily_features
from daily_research.data_platform.combine_domain_datasets import CombineDomainDatasetsConfig, combine_domain_datasets
from daily_research.data_platform.combine_sharded_domain_datasets import CombineShardedDomainConfig, combine_sharded_domain_datasets
from daily_research.data_platform.import_external_quant_zip import ImportConfig, run_import
from daily_research.data_platform.recover_external_quant_zip_import import RecoverExternalImportConfig, recover_import


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
    assert result.row_counts[DataDomain.MARKET_INTRADAY_1M] == 5
    assert result.row_counts[DataDomain.MARKET_INTRADAY_5M] == 1
    one_minute = pd.read_parquet(
        next((tmp_path / "lake").glob("parquet/bronze_silver/data_platform_market_intraday_1m/*/shards/*.parquet"))
    )
    five_minute = pd.read_parquet(
        next((tmp_path / "lake").glob("parquet/bronze_silver/data_platform_market_intraday_5m/*/shards/*.parquet"))
    )
    assert one_minute["symbol"].iloc[0] == "000001.SZ"
    assert five_minute["bar_time"].iloc[0] == "093000000"
    assert float(five_minute["volume"].iloc[0]) == 1500.0


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
    assert result.row_counts[DataDomain.MARKET_INTRADAY_1M] == 5
    assert DataDomain.MARKET_INTRADAY_5M not in result.row_counts
    assert not list((tmp_path / "lake").glob("parquet/bronze_silver/data_platform_market_intraday_5m/*/shards/*.parquet"))


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
    assert result.row_counts[DataDomain.MARKET_INTRADAY_1M] == 5
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
