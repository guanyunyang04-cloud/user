from __future__ import annotations

import zipfile

import pandas as pd

from daily_research.data_platform.contracts import DataDomain
from daily_research.data_platform.import_external_quant_zip import ImportConfig, run_import


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
