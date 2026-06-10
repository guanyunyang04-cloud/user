from __future__ import annotations

from pathlib import Path

import pandas as pd

from daily_research.data_lake.catalog import ResearchDataLake
from daily_research.data_platform.contracts import DataDomain
from daily_research.data_platform.import_tdx_5m import ImportTdx5mConfig, _tdx_5m_market_data_to_frame, run_import_tdx_5m


class FakeTdx5mClient:
    def __init__(self) -> None:
        self.calls: list[dict[str, object]] = []

    def get_stock_list(self) -> list[str]:
        return ["000001.SZ", "600000.SH"]

    def get_market_data(self, **kwargs: object) -> dict[str, pd.DataFrame]:
        self.calls.append(dict(kwargs))
        stock_list = [str(item) for item in kwargs["stock_list"]]  # type: ignore[index]
        index = pd.to_datetime(["2026-06-01 09:35:00", "2026-06-01 09:40:00"])
        return {
            "Open": pd.DataFrame({symbol: [10.0, 10.5] for symbol in stock_list}, index=index),
            "High": pd.DataFrame({symbol: [11.0, 11.5] for symbol in stock_list}, index=index),
            "Low": pd.DataFrame({symbol: [9.5, 10.0] for symbol in stock_list}, index=index),
            "Close": pd.DataFrame({symbol: [10.8, 11.0] for symbol in stock_list}, index=index),
            "Volume": pd.DataFrame({symbol: [1000.0, 1200.0] for symbol in stock_list}, index=index),
            "Amount": pd.DataFrame({symbol: [1.08, 1.32] for symbol in stock_list}, index=index),
        }

    def close(self) -> None:
        pass


def test_tdx_5m_market_data_to_frame_normalizes_bar_time_and_amount_unit() -> None:
    fake = FakeTdx5mClient()
    payload = fake.get_market_data(stock_list=["000001.SZ"])
    frame = _tdx_5m_market_data_to_frame(payload, source="tdx_tqcenter_5m", adjusted_flag="none")

    assert list(frame["bar_time"]) == ["093500000", "094000000"]
    assert set(frame["trade_date"]) == {"2026-06-01"}
    assert set(frame["source"]) == {"tdx_tqcenter_5m"}
    assert frame.loc[frame["bar_time"].eq("093500000"), "amount"].iloc[0] == 1.08 * 10000


def test_import_tdx_5m_with_fake_client_registers_intraday_dataset(tmp_path: Path) -> None:
    fake = FakeTdx5mClient()
    result = run_import_tdx_5m(
        ImportTdx5mConfig(
            lake_root=tmp_path / "lake",
            symbols=("000001.SZ", "600000.SH"),
            start_date="2026-06-01",
            end_date="2026-06-01",
            chunk_size=1,
            retry_count=0,
        ),
        tq_client=fake,
    )

    assert result.status == "completed"
    assert result.shard_count == 2
    assert result.row_count == 4
    assert result.error_count == 0
    assert fake.calls[0]["period"] == "5m"
    assert fake.calls[0]["count"] == -1
    assert fake.calls[0]["fill_data"] is False

    metadata = ResearchDataLake(tmp_path / "lake").describe_dataset(result.dataset_id)
    assert metadata["dataset_kind"] == f"data_platform_{DataDomain.MARKET_INTRADAY_5M}"
    assert metadata["source"] == "tdx_tqcenter_5m"
    assert metadata["row_counts"]["silver_domain_data"] == 4
    assert metadata["parameters"]["intraday_bar_policy"] == "tdx_native_5m"
    assert metadata["parameters"]["amount_unit_factor"] == 10000.0
    assert Path(metadata["content_paths"]["shard_manifest"]).exists()
