from __future__ import annotations

from pathlib import Path

import pandas as pd

from quant_data_platform.lake.policy_input_loader import _load_index_constituents_sidecar_frames
from quant_data_platform.domains.contracts import DataDomain


class FakeLake:
    def __init__(self, path: Path) -> None:
        self.path = path

    def describe_dataset(self, dataset_id: str) -> dict:
        assert dataset_id == "index_ds"
        return {
            "dataset_id": dataset_id,
            "dataset_kind": "data_platform_index_constituents",
            "parameters": {},
            "content_paths": {"silver_domain_data": str(self.path)},
        }


def test_index_sidecar_loader_builds_lag_ready_membership_panels(tmp_path: Path) -> None:
    sidecar_path = tmp_path / "index.parquet"
    pd.DataFrame(
        {
            "index_symbol": ["000300.SH", "000300.SH", "000905.SH"],
            "index_name": ["沪深300", "沪深300", "中证500"],
            "symbol": ["000001.SZ", "000002.SZ", "000002.SZ"],
            "trade_date": ["2020-01-01", "2020-01-01", "2020-01-02"],
        }
    ).to_parquet(sidecar_path, index=False)

    frames, summary = _load_index_constituents_sidecar_frames(
        lake=FakeLake(sidecar_path),
        metadata={"parameters": {"sidecar_dataset_ids": {DataDomain.INDEX_CONSTITUENTS: "index_ds"}}},
        universe=["000001.SZ", "000002.SZ", "000003.SZ"],
        dates=pd.DatetimeIndex(["2020-01-01", "2020-01-02", "2020-01-03"]),
        start_date="2020-01-01",
        end_date="2020-01-03",
    )

    assert summary["available"] is True
    assert sorted(frames) == ["index_constituents_000300_SH_member", "index_constituents_000905_SH_member"]
    csi300 = frames["index_constituents_000300_SH_member"]
    assert float(csi300.loc[pd.Timestamp("2020-01-01"), "000001.SZ"]) == 1.0
    assert float(csi300.loc[pd.Timestamp("2020-01-03"), "000002.SZ"]) == 1.0
    assert float(csi300.loc[pd.Timestamp("2020-01-03"), "000003.SZ"]) == 0.0
