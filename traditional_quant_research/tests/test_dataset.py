from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from traditional_quant_research.dataset import load_daily_snapshot


def test_load_daily_snapshot_reads_parquet_fixture(tmp_path: Path) -> None:
    root = tmp_path / "snapshot"
    root.mkdir()
    manifest = {"schema_version": 1, "snapshot_id": "fixture"}
    (root / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    pd.DataFrame(
        [
            {"code": "600000.SH", "name": "浦发银行", "selected": True},
            {"code": "000001.SZ", "name": "平安银行", "selected": True},
        ]
    ).to_parquet(root / "universe.parquet", index=False)
    pd.DataFrame(
        [
            {"date": pd.Timestamp("2026-01-02"), "code": "600000.SH", "open": 1.0, "high": 1.1, "low": 0.9, "close": 1.0, "volume": 100.0, "amount": 1000.0, "forward_factor": 1.0, "source": "fixture"},
        ]
    ).to_parquet(root / "daily_bars.parquet", index=False)
    pd.DataFrame(
        [
            {"date": pd.Timestamp("2026-01-02"), "code": "600000.SH", "has_bar": True, "is_suspended_like": False, "is_tradeable": True},
        ]
    ).to_parquet(root / "daily_status.parquet", index=False)

    snapshot = load_daily_snapshot(root)

    assert snapshot.manifest["snapshot_id"] == "fixture"
    assert set(snapshot.universe["code"]) == {"600000.SH", "000001.SZ"}
    assert list(snapshot.daily_bars.columns) == [
        "date",
        "code",
        "open",
        "high",
        "low",
        "close",
        "volume",
        "amount",
        "forward_factor",
        "source",
    ]
    assert snapshot.daily_status["is_tradeable"].tolist() == [True]
