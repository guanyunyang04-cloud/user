from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from traditional_quant_research.dataset_v2 import (
    load_daily_universe,
    load_pit_daily_bars,
    load_pit_daily_metrics,
    load_pit_daily_size,
    load_pit_snapshot,
    load_pit_stock_industry,
    load_quality_report,
    load_tradeable_panel,
)


def test_v2_loaders_filter_dates_symbols_and_tradeable(tmp_path: Path) -> None:
    root = tmp_path / "snapshot"
    root.mkdir()
    (root / "manifest.json").write_text(json.dumps({"schema_version": 2, "snapshot_id": "fixture"}), encoding="utf-8")
    (root / "quality_report.json").write_text(json.dumps({"yearly_summary": [{"year": 2026}]}), encoding="utf-8")
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
    pd.DataFrame(
        [
            {
                "date": pd.Timestamp("2026-01-02"),
                "code": "600000.SH",
                "turn": 0.85,
                "pctChg": 1.2,
                "peTTM": 6.5,
                "pbMRQ": 0.7,
                "psTTM": 2.1,
                "pcfNcfTTM": 4.2,
                "source": "fixture",
            }
        ]
    ).to_parquet(root / "daily_metrics.parquet", index=False)
    pd.DataFrame(
        [
            {
                "date": pd.Timestamp("2026-01-02"),
                "code": "600000.SH",
                "name_on_date": "浦发银行",
                "industry": "J66货币金融服务",
                "industry_classification": "证监会行业分类",
                "industry_update_date": "2026-01-02",
                "source": "fixture",
            }
        ]
    ).to_parquet(root / "stock_industry.parquet", index=False)
    pd.DataFrame(
        [
            {
                "date": pd.Timestamp("2026-01-02"),
                "code": "600000.SH",
                "total_market_cap": 1000000.0,
                "float_market_cap": 800000.0,
                "total_share": 10000.0,
                "float_share": 8000.0,
                "free_share": 7000.0,
                "market_cap_unit": "10k CNY",
                "share_unit": "10k shares",
                "source": "fixture_size",
                "source_trade_date": "20260102",
            }
        ]
    ).to_parquet(root / "daily_size.parquet", index=False)

    snapshot = load_pit_snapshot(root)
    universe = load_daily_universe(root, start_date="2026-01-02", end_date="2026-01-02", tradeable_only=True)
    bars = load_pit_daily_bars(root, symbols=["600000.SH"])
    metrics = load_pit_daily_metrics(root, symbols=["600000.SH"])
    size = load_pit_daily_size(root, symbols=["600000.SH"])
    industry = load_pit_stock_industry(root)
    panel = load_tradeable_panel(root, include_industry=True, include_metrics=True, include_size=True)
    quality = load_quality_report(root)

    assert snapshot.manifest["snapshot_id"] == "fixture"
    assert snapshot.daily_metrics["turn"].tolist() == [0.85]
    assert snapshot.stock_industry["industry"].tolist() == ["J66货币金融服务"]
    assert snapshot.daily_size["total_market_cap"].tolist() == [1000000.0]
    assert universe["code"].tolist() == ["600000.SH"]
    assert bars["code"].tolist() == ["600000.SH"]
    assert metrics["turn"].tolist() == [0.85]
    assert size["float_market_cap"].tolist() == [800000.0]
    assert industry["industry_classification"].tolist() == ["证监会行业分类"]
    assert panel["code"].tolist() == ["600000.SH"]
    assert panel["industry"].tolist() == ["J66货币金融服务"]
    assert panel["industry_source"].tolist() == ["fixture"]
    assert panel["turn"].tolist() == [0.85]
    assert panel["metrics_source"].tolist() == ["fixture"]
    assert panel["total_market_cap"].tolist() == [1000000.0]
    assert panel["size_source"].tolist() == ["fixture_size"]
    assert "source_x" not in panel.columns
    assert "source_y" not in panel.columns
    assert quality["yearly_summary"] == [{"year": 2026}]


def test_daily_size_loader_is_optional_for_older_snapshots(tmp_path: Path) -> None:
    root = tmp_path / "snapshot"
    root.mkdir()
    (root / "manifest.json").write_text(json.dumps({"schema_version": 2, "snapshot_id": "fixture"}), encoding="utf-8")
    pd.DataFrame([{"code": "600000.SH", "name": "浦发银行"}]).to_parquet(root / "security_master.parquet", index=False)
    pd.DataFrame(
        [
            {"date": pd.Timestamp("2026-01-02"), "code": "600000.SH", "is_tradeable": True, "reject_reason": ""},
        ]
    ).to_parquet(root / "daily_universe.parquet", index=False)
    pd.DataFrame(
        [
            {
                "date": pd.Timestamp("2026-01-02"),
                "code": "600000.SH",
                "open": 1.0,
                "high": 1.1,
                "low": 0.9,
                "close": 1.0,
                "volume": 100.0,
                "amount": 1000.0,
                "tradestatus": "1",
                "isST": "0",
                "source": "fixture",
            },
        ]
    ).to_parquet(root / "daily_bars.parquet", index=False)
    pd.DataFrame(
        [
            {"date": pd.Timestamp("2026-01-02"), "code": "600000.SH", "has_bar": True, "is_suspended_like": False, "is_tradeable": True},
        ]
    ).to_parquet(root / "daily_status.parquet", index=False)

    size = load_pit_daily_size(root)
    panel = load_tradeable_panel(root, include_size=True)

    assert size.empty
    assert list(size.columns) == [
        "date",
        "code",
        "total_market_cap",
        "float_market_cap",
        "total_share",
        "float_share",
        "free_share",
        "market_cap_unit",
        "share_unit",
        "source",
        "source_trade_date",
    ]
    assert panel["code"].tolist() == ["600000.SH"]
    assert "total_market_cap" not in panel.columns
