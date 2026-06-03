from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from traditional_quant_research.experiments.v2_daily_size_audit import (
    build_daily_size_audit,
    run_v2_daily_size_audit,
    summarize_daily_size_audit,
)


def test_daily_size_audit_reports_tradeable_coverage_and_units() -> None:
    universe = pd.DataFrame(
        [
            {"date": pd.Timestamp("2026-01-02"), "code": "600000.SH", "is_tradeable": True},
            {"date": pd.Timestamp("2026-01-02"), "code": "000001.SZ", "is_tradeable": True},
            {"date": pd.Timestamp("2026-01-02"), "code": "600001.SH", "is_tradeable": False},
        ]
    )
    size = pd.DataFrame(
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
                "source": "fixture",
            },
            {
                "date": pd.Timestamp("2026-01-02"),
                "code": "600001.SH",
                "total_market_cap": 1200000.0,
                "float_market_cap": None,
                "total_share": 11000.0,
                "float_share": None,
                "free_share": None,
                "market_cap_unit": "10k CNY",
                "share_unit": "10k shares",
                "source": "fixture",
            },
        ]
    )

    audit = build_daily_size_audit(universe, size)
    summary = summarize_daily_size_audit(
        manifest={"snapshot_id": "fixture"},
        universe=universe,
        size=size,
        audit=audit,
        run_id="fixture-run",
        start_date=None,
        end_date=None,
    )

    field_summary = audit["field_summary"].set_index(["scope", "field"])
    unit_summary = audit["source_unit_summary"]
    assert field_summary.loc[("all", "total_market_cap"), "coverage_rate"] == 2 / 3
    assert field_summary.loc[("tradeable", "total_market_cap"), "coverage_rate"] == 0.5
    assert field_summary.loc[("tradeable", "float_market_cap"), "coverage_rate"] == 0.5
    assert unit_summary["market_cap_unit"].tolist() == ["10k CNY"]
    assert summary["min_tradeable_coverage"] == 0.5
    assert summary["required_units_present"] is True
    assert summary["daily_size_ready_for_research"] is False
    assert summary["candidate_count"] == 0


def test_run_daily_size_audit_writes_absent_table_artifacts(tmp_path: Path) -> None:
    root = tmp_path / "snapshot"
    root.mkdir()
    (root / "manifest.json").write_text(json.dumps({"snapshot_id": "fixture"}), encoding="utf-8")
    pd.DataFrame(
        [
            {"date": pd.Timestamp("2026-01-02"), "code": "600000.SH", "is_tradeable": True},
        ]
    ).to_parquet(root / "daily_universe.parquet", index=False)

    result = run_v2_daily_size_audit(root=root, output_dir=tmp_path / "output")

    run_dir = Path(result["run_dir"])
    assert result["snapshot_id"] == "fixture"
    assert result["status"] == "daily_size_absent"
    assert result["daily_size_ready_for_research"] is False
    assert result["size_rows"] == 0
    assert (run_dir / "field_summary.csv").exists()
    assert (run_dir / "yearly_summary.csv").exists()
    assert (run_dir / "daily_summary.csv").exists()
    assert (run_dir / "source_unit_summary.csv").exists()
    assert (run_dir / "summary.md").exists()


def test_run_daily_size_audit_marks_full_fixture_ready(tmp_path: Path) -> None:
    root = tmp_path / "snapshot"
    root.mkdir()
    (root / "manifest.json").write_text(json.dumps({"snapshot_id": "fixture"}), encoding="utf-8")
    pd.DataFrame(
        [
            {"date": pd.Timestamp("2026-01-02"), "code": "600000.SH", "is_tradeable": True},
        ]
    ).to_parquet(root / "daily_universe.parquet", index=False)
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
                "source": "fixture",
                "source_trade_date": "20260102",
            },
        ]
    ).to_parquet(root / "daily_size.parquet", index=False)

    result = run_v2_daily_size_audit(
        root=root,
        output_dir=tmp_path / "output",
        write_research_log=True,
        research_log_path=tmp_path / "size_audit.md",
    )

    assert result["status"] == "daily_size_missingness_audit"
    assert result["daily_size_ready_for_research"] is True
    assert result["min_tradeable_coverage"] == 1.0
    assert (tmp_path / "size_audit.md").exists()
