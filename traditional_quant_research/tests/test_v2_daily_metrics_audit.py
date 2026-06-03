from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from traditional_quant_research.experiments.v2_daily_metrics_audit import (
    build_daily_metrics_audit,
    run_v2_daily_metrics_audit,
    summarize_daily_metrics_audit,
)


def test_daily_metrics_audit_reports_tradeable_coverage() -> None:
    universe = pd.DataFrame(
        [
            {"date": pd.Timestamp("2026-01-02"), "code": "600000.SH", "is_tradeable": True},
            {"date": pd.Timestamp("2026-01-02"), "code": "000001.SZ", "is_tradeable": True},
            {"date": pd.Timestamp("2026-01-02"), "code": "600001.SH", "is_tradeable": False},
        ]
    )
    metrics = pd.DataFrame(
        [
            {
                "date": pd.Timestamp("2026-01-02"),
                "code": "600000.SH",
                "turn": 0.8,
                "pctChg": 1.0,
                "peTTM": 6.0,
                "pbMRQ": 0.7,
                "psTTM": 2.0,
                "pcfNcfTTM": 4.0,
            },
            {
                "date": pd.Timestamp("2026-01-02"),
                "code": "600001.SH",
                "turn": 0.0,
                "pctChg": 0.0,
                "peTTM": None,
                "pbMRQ": None,
                "psTTM": None,
                "pcfNcfTTM": None,
            },
        ]
    )

    audit = build_daily_metrics_audit(universe, metrics)
    summary = summarize_daily_metrics_audit(
        manifest={"snapshot_id": "fixture"},
        universe=universe,
        metrics=metrics,
        audit=audit,
        run_id="fixture-run",
        start_date=None,
        end_date=None,
    )

    field_summary = audit["field_summary"].set_index(["scope", "field"])
    assert field_summary.loc[("all", "turn"), "coverage_rate"] == 2 / 3
    assert field_summary.loc[("tradeable", "turn"), "coverage_rate"] == 0.5
    assert summary["min_tradeable_coverage"] == 0.5
    assert summary["full_metrics_ready_for_research"] is False
    assert summary["candidate_count"] == 0


def test_run_daily_metrics_audit_writes_artifacts(tmp_path: Path) -> None:
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
                "turn": 0.8,
                "pctChg": 1.0,
                "peTTM": 6.0,
                "pbMRQ": 0.7,
                "psTTM": 2.0,
                "pcfNcfTTM": 4.0,
                "source": "fixture",
            },
        ]
    ).to_parquet(root / "daily_metrics.parquet", index=False)

    result = run_v2_daily_metrics_audit(root=root, output_dir=tmp_path / "output")

    run_dir = Path(result["run_dir"])
    assert result["snapshot_id"] == "fixture"
    assert result["full_metrics_ready_for_research"] is True
    assert (run_dir / "field_summary.csv").exists()
    assert (run_dir / "yearly_summary.csv").exists()
    assert (run_dir / "daily_summary.csv").exists()
    assert (run_dir / "summary.md").exists()
