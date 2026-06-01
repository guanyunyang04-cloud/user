from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from traditional_quant_research.dataset_v2 import load_daily_universe, load_pit_daily_bars, load_pit_snapshot, load_tradeable_panel


def test_v2_loaders_filter_dates_symbols_and_tradeable(tmp_path: Path) -> None:
    root = tmp_path / "snapshot"
    root.mkdir()
    (root / "manifest.json").write_text(json.dumps({"schema_version": 2, "snapshot_id": "fixture"}), encoding="utf-8")
    pd.DataFrame([{"code": "600000.SH", "name": "浦发银行"}]).to_parquet(root / "security_master.parquet", index=False)
    pd.DataFrame(
        [
            {"date": pd.Timestamp("2026-01-02"), "code": "600000.SH", "is_tradeable": True, "reject_reason": ""},
            {"date": pd.Timestamp("2026-01-03"), "code": "000001.SZ", "is_tradeable": False, "reject_reason": "suspended_on_date"},
        ]
    ).to_parquet(root / "daily_universe.parquet", index=False)
    pd.DataFrame(
        [
            {"date": pd.Timestamp("2026-01-02"), "code": "600000.SH", "open": 1.0, "high": 1.1, "low": 0.9, "close": 1.0, "volume": 100.0, "amount": 1000.0, "tradestatus": "1", "isST": "0", "source": "fixture"},
            {"date": pd.Timestamp("2026-01-03"), "code": "000001.SZ", "open": 2.0, "high": 2.1, "low": 1.9, "close": 2.0, "volume": 0.0, "amount": 0.0, "tradestatus": "0", "isST": "0", "source": "fixture"},
        ]
    ).to_parquet(root / "daily_bars.parquet", index=False)
    pd.DataFrame(
        [
            {"date": pd.Timestamp("2026-01-02"), "code": "600000.SH", "has_bar": True, "is_suspended_like": False, "is_tradeable": True},
        ]
    ).to_parquet(root / "daily_status.parquet", index=False)

    snapshot = load_pit_snapshot(root)
    universe = load_daily_universe(root, start_date="2026-01-02", end_date="2026-01-02", tradeable_only=True)
    bars = load_pit_daily_bars(root, symbols=["600000.SH"])
    panel = load_tradeable_panel(root)

    assert snapshot.manifest["snapshot_id"] == "fixture"
    assert universe["code"].tolist() == ["600000.SH"]
    assert bars["code"].tolist() == ["600000.SH"]
    assert panel["code"].tolist() == ["600000.SH"]
