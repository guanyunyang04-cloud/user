from __future__ import annotations

from pathlib import Path

import pandas as pd

from daily_research.data_lake.catalog import ResearchDataLake
from daily_research.data_platform.contracts import DataDomain
from daily_research.data_platform.import_tdx_daily import (
    ImportTdxDailyConfig,
    _normalize_symbols,
    _tdx_market_data_to_frame,
    run_import_tdx_daily,
)


class FakeTqClient:
    def __init__(self) -> None:
        self.calls: list[dict[str, object]] = []
        self.closed = False

    def get_stock_list(self) -> list[str]:
        return ["000001.SZ", "600000.SH"]

    def get_market_data(self, **kwargs: object) -> dict[str, pd.DataFrame]:
        self.calls.append(dict(kwargs))
        stock_list = [str(item) for item in kwargs["stock_list"]]  # type: ignore[index]
        index = pd.to_datetime(["2020-01-02", "2020-01-03"])
        return {
            "Open": pd.DataFrame({symbol: [10.0, 10.5] for symbol in stock_list}, index=index),
            "High": pd.DataFrame({symbol: [11.0, 11.5] for symbol in stock_list}, index=index),
            "Low": pd.DataFrame({symbol: [9.5, 10.0] for symbol in stock_list}, index=index),
            "Close": pd.DataFrame({symbol: [10.8, 11.0] for symbol in stock_list}, index=index),
            "Volume": pd.DataFrame({symbol: [1000.0, 1200.0] for symbol in stock_list}, index=index),
            "Amount": pd.DataFrame({symbol: [10800.0, 13200.0] for symbol in stock_list}, index=index),
        }

    def close(self) -> None:
        self.closed = True


def test_tdx_market_data_to_frame_normalizes_wide_payload_and_drops_empty_rows() -> None:
    index = pd.to_datetime(["2020-01-02", "2020-01-03"])
    payload = {
        "Open": pd.DataFrame({"000001.SZ": [10.0, None], "600000.SH": [20.0, None]}, index=index),
        "High": pd.DataFrame({"000001.SZ": [11.0, None], "600000.SH": [21.0, None]}, index=index),
        "Low": pd.DataFrame({"000001.SZ": [9.0, None], "600000.SH": [19.0, None]}, index=index),
        "Close": pd.DataFrame({"000001.SZ": [10.5, None], "600000.SH": [20.5, None]}, index=index),
        "Volume": pd.DataFrame({"000001.SZ": [1000, None], "600000.SH": [2000, None]}, index=index),
        "Amount": pd.DataFrame({"000001.SZ": [10500, None], "600000.SH": [41000, None]}, index=index),
    }
    frame = _tdx_market_data_to_frame(payload, source="tdx_tqcenter", adjusted_flag="none")

    assert list(frame["symbol"]) == ["000001.SZ", "600000.SH"]
    assert set(frame["trade_date"]) == {"2020-01-02"}
    assert set(frame["source"]) == {"tdx_tqcenter"}
    assert set(frame["adjusted_flag"]) == {"none"}
    assert frame.loc[frame["symbol"].eq("000001.SZ"), "amount"].iloc[0] == 10500 * 10000


def test_normalize_symbols_accepts_dict_rows_and_plain_codes() -> None:
    assert _normalize_symbols([{"code": "sh.600000"}, "000001", {"证券代码": "bj.430047"}]) == (
        "600000.SH",
        "000001.SZ",
        "430047.BJ",
    )


def test_import_tdx_daily_with_fake_client_registers_sharded_dataset(tmp_path: Path) -> None:
    fake = FakeTqClient()
    result = run_import_tdx_daily(
        ImportTdxDailyConfig(
            lake_root=tmp_path / "lake",
            symbols=("000001.SZ", "600000.SH"),
            start_date="2020-01-01",
            end_date="2020-01-31",
            chunk_size=1,
            retry_count=0,
        ),
        tq_client=fake,
    )

    assert result.status == "completed"
    assert result.shard_count == 2
    assert result.row_count == 4
    assert result.error_count == 0
    assert fake.calls
    assert fake.calls[0]["count"] == -1
    assert fake.calls[0]["fill_data"] is False
    assert fake.calls[0]["dividend_type"] == "none"

    metadata = ResearchDataLake(tmp_path / "lake").describe_dataset(result.dataset_id)
    assert metadata["dataset_kind"] == f"data_platform_{DataDomain.MARKET_DAILY}"
    assert metadata["source"] == "tdx_tqcenter"
    assert metadata["row_counts"]["silver_domain_data"] == 4
    assert metadata["parameters"]["amount_unit_factor"] == 10000.0
    assert metadata["parameters"]["standard_amount_unit"] == "yuan"
    assert Path(metadata["content_paths"]["shard_manifest"]).exists()
