from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from traditional_quant_research.experiments.v2_metrics_semantics_audit import (
    build_v2_metrics_semantics_audit,
    run_v2_metrics_semantics_audit,
    summarize_v2_metrics_semantics_audit,
)


def _fixture_panel() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "date": pd.Timestamp("2026-01-02"),
                "code": "600000.SH",
                "close": 100.0,
                "pctChg": 0.0,
                "peTTM": 10.0,
                "pbMRQ": 1.0,
                "psTTM": 2.0,
                "pcfNcfTTM": 3.0,
            },
            {
                "date": pd.Timestamp("2026-01-05"),
                "code": "600000.SH",
                "close": 110.0,
                "pctChg": 10.0,
                "peTTM": 11.0,
                "pbMRQ": 1.1,
                "psTTM": 2.2,
                "pcfNcfTTM": 3.3,
            },
            {
                "date": pd.Timestamp("2026-01-06"),
                "code": "600000.SH",
                "close": 99.0,
                "pctChg": -10.0,
                "peTTM": 9.9,
                "pbMRQ": 0.99,
                "psTTM": 1.98,
                "pcfNcfTTM": 2.97,
            },
        ]
    )


def test_metrics_semantics_audit_compares_pctchg_to_close_returns() -> None:
    panel = _fixture_panel()

    audit = build_v2_metrics_semantics_audit(panel)
    summary = summarize_v2_metrics_semantics_audit(
        manifest={"snapshot_id": "fixture"},
        panel=panel,
        audit=audit,
        run_id="fixture-run",
        start_date=None,
        end_date=None,
    )

    pct_summary = audit["pctchg_summary"].iloc[0]
    valuation_summary = audit["valuation_summary"].set_index("field")
    assert pct_summary["compare_rows"] == 2
    assert pct_summary["abs_diff_max"] < 1e-10
    assert summary["pctchg_close_return_aligned"] is True
    assert summary["valuation_missingness_ready"] is True
    assert summary["valuation_pit_timing_ready"] is False
    assert valuation_summary.loc["peTTM", "non_null_rate"] == 1.0
    assert summary["candidate_count"] == 0


def test_run_metrics_semantics_audit_writes_artifacts(tmp_path: Path) -> None:
    root = tmp_path / "snapshot"
    root.mkdir()
    (root / "manifest.json").write_text(json.dumps({"snapshot_id": "fixture"}), encoding="utf-8")
    panel = _fixture_panel()
    universe = panel[["date", "code"]].copy()
    universe["is_tradeable"] = True
    bars = panel[["date", "code", "close"]].copy()
    bars["open"] = bars["close"]
    bars["high"] = bars["close"]
    bars["low"] = bars["close"]
    bars["volume"] = 100
    bars["amount"] = 1000
    bars["tradestatus"] = "1"
    bars["isST"] = "0"
    metrics = panel.drop(columns=["close"]).copy()
    metrics["turn"] = 1.0
    metrics["source"] = "fixture"
    universe.to_parquet(root / "daily_universe.parquet", index=False)
    bars.to_parquet(root / "daily_bars.parquet", index=False)
    metrics.to_parquet(root / "daily_metrics.parquet", index=False)

    result = run_v2_metrics_semantics_audit(root=root, output_dir=tmp_path / "output")

    run_dir = Path(result["run_dir"])
    assert result["snapshot_id"] == "fixture"
    assert result["pctchg_close_return_aligned"] is True
    assert (run_dir / "pctchg_summary.csv").exists()
    assert (run_dir / "valuation_summary.csv").exists()
    assert (run_dir / "summary.md").exists()
