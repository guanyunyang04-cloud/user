from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from traditional_quant_research.experiments import low_corr_candidate_signal_comparison
from traditional_quant_research.experiments.low_corr_exposure_grid import LOW_CORR_SIGNAL


def _signal_panel() -> pd.DataFrame:
    rows = []
    dates = pd.date_range("2026-01-02", periods=6, freq="B")
    codes = ["600000.SH", "000001.SZ", "000002.SZ"]
    for code_index, code in enumerate(codes):
        for date_index, date in enumerate(dates):
            base = 10.0 + code_index * 2.0 + date_index * 0.1
            rows.append(
                {
                    "date": date,
                    "code": code,
                    "open": base,
                    "close": base + 0.1 + code_index * 0.03,
                    "amount": 100_000_000.0 + code_index * 10_000_000.0,
                    "volume": 1_000_000.0,
                    "is_tradeable": True,
                    "factor_a_z": 0.5 - code_index * 0.2,
                    low_corr_candidate_signal_comparison.EQUAL_SIGNAL: 3.0 - code_index + date_index * 0.01,
                    LOW_CORR_SIGNAL: 2.0 - code_index + date_index * 0.02,
                }
            )
    return pd.DataFrame(rows)


def test_summarize_signal_comparison_adds_low_corr_delta_and_fee_drag() -> None:
    summary = pd.DataFrame(
        [
            {"eval_year": 2025, "signal": LOW_CORR_SIGNAL, "fee_bps": 0.0, "annualized_return": 0.20, "sharpe": 1.0, "max_drawdown": -0.04, "mean_turnover": 0.8, "periods": 4},
            {"eval_year": 2025, "signal": LOW_CORR_SIGNAL, "fee_bps": 30.0, "annualized_return": 0.15, "sharpe": 0.8, "max_drawdown": -0.05, "mean_turnover": 0.8, "periods": 4},
            {"eval_year": 2025, "signal": "other_score", "fee_bps": 0.0, "annualized_return": 0.30, "sharpe": 1.2, "max_drawdown": -0.03, "mean_turnover": 0.6, "periods": 4},
            {"eval_year": 2025, "signal": "other_score", "fee_bps": 30.0, "annualized_return": 0.28, "sharpe": 1.1, "max_drawdown": -0.04, "mean_turnover": 0.6, "periods": 4},
            {"eval_year": 2026, "signal": LOW_CORR_SIGNAL, "fee_bps": 0.0, "annualized_return": 0.10, "sharpe": 0.7, "max_drawdown": -0.06, "mean_turnover": 0.9, "periods": 2},
            {"eval_year": 2026, "signal": LOW_CORR_SIGNAL, "fee_bps": 30.0, "annualized_return": 0.08, "sharpe": 0.5, "max_drawdown": -0.07, "mean_turnover": 0.9, "periods": 2},
            {"eval_year": 2026, "signal": "other_score", "fee_bps": 0.0, "annualized_return": 0.05, "sharpe": 0.2, "max_drawdown": -0.08, "mean_turnover": 0.7, "periods": 2},
            {"eval_year": 2026, "signal": "other_score", "fee_bps": 30.0, "annualized_return": 0.02, "sharpe": 0.1, "max_drawdown": -0.09, "mean_turnover": 0.7, "periods": 2},
        ]
    )

    aggregate = low_corr_candidate_signal_comparison.summarize_signal_comparison(summary)
    other30 = aggregate.loc[(aggregate["signal"] == "other_score") & (aggregate["fee_bps"] == 30.0)].iloc[0]
    low30 = aggregate.loc[(aggregate["signal"] == LOW_CORR_SIGNAL) & (aggregate["fee_bps"] == 30.0)].iloc[0]

    assert other30["mean_annualized_return"] == pytest.approx(0.15)
    assert other30["mean_cost_drag_vs_0bps"] == pytest.approx(-0.025)
    assert other30["mean_delta_annualized_return_vs_low_corr"] == pytest.approx(0.035)
    assert other30["positive_delta_year_rate_vs_low_corr"] == pytest.approx(0.5)
    assert low30["mean_delta_annualized_return_vs_low_corr"] == pytest.approx(0.0)
    assert low30["positive_delta_year_rate_vs_low_corr"] == pytest.approx(0.0)


def test_run_low_corr_candidate_signal_comparison_writes_outputs(tmp_path: Path, monkeypatch) -> None:
    panel = _signal_panel()

    def fake_build_candidate_protocol_signal_panel(**kwargs):
        return {
            "manifest": {"snapshot_id": "fixture-snapshot"},
            "quality": {"failure_count": 0, "missing_bar_rows": 0, "st_rows": 0, "suspended_like_rows": 0},
            "evaluation_panel": panel,
            "available_signals": [
                low_corr_candidate_signal_comparison.EQUAL_SIGNAL,
                LOW_CORR_SIGNAL,
            ],
            "signal_columns": ["factor_a_z"],
            "low_corr_factor_columns": ["factor_a_z"],
            "signal_coverage": pd.DataFrame(
                [
                    {"factor": low_corr_candidate_signal_comparison.EQUAL_SIGNAL, "rows": len(panel), "available_rows": len(panel), "missing_rows": 0, "coverage_rate": 1.0},
                    {"factor": LOW_CORR_SIGNAL, "rows": len(panel), "available_rows": len(panel), "missing_rows": 0, "coverage_rate": 1.0},
                ]
            ),
            "rolling_weight_audit": pd.DataFrame([{"period": "2026-01", "period_type": "monthly", "factor": "factor_a_z", "fallback_rate": 0.0}]),
            "rolling_fallback_rate": 0.0,
        }

    monkeypatch.setattr(
        low_corr_candidate_signal_comparison,
        "build_candidate_protocol_signal_panel",
        fake_build_candidate_protocol_signal_panel,
    )

    result = low_corr_candidate_signal_comparison.run_low_corr_candidate_signal_comparison(
        years=(2026,),
        final_end_date="2026-06-01",
        horizon=1,
        signals=(low_corr_candidate_signal_comparison.EQUAL_SIGNAL, LOW_CORR_SIGNAL),
        top_n=2,
        rebalance_frequency="daily",
        buffer_multiplier=1.0,
        fee_bps_values=(0.0, 30.0),
        execution_constraints=False,
        output_dir=tmp_path,
        write_research_log=True,
        research_log_path=tmp_path / "research_log.md",
    )

    run_dir = tmp_path / result["run_id"]
    expected_files = [
        "signal_protocol_summary.csv",
        "signal_protocol_aggregate.csv",
        "signal_protocol_trades.csv",
        "signal_protocol_yearly_summary.csv",
        "signal_protocol_period_summary.csv",
        "signal_basket_exposure.csv",
        "signal_coverage.csv",
        "rolling_ic_weight_audit.csv",
        "signal_comparison_meta.csv",
        "summary.json",
        "summary.md",
    ]
    for name in expected_files:
        assert (run_dir / name).exists()
    assert (tmp_path / "research_log.md").exists()
    assert result["snapshot_id"] == "fixture-snapshot"
    assert result["candidate_count"] == 0

    aggregate = pd.read_csv(run_dir / "signal_protocol_aggregate.csv")
    assert set(aggregate["signal"]) == {low_corr_candidate_signal_comparison.EQUAL_SIGNAL, LOW_CORR_SIGNAL}
    assert "mean_delta_annualized_return_vs_low_corr" in aggregate.columns
    trades = pd.read_csv(run_dir / "signal_protocol_trades.csv")
    assert set(trades["signal"]) == {low_corr_candidate_signal_comparison.EQUAL_SIGNAL, LOW_CORR_SIGNAL}
    metadata = pd.read_csv(run_dir / "signal_comparison_meta.csv")
    assert metadata.iloc[0]["rolling_fallback_rate"] == pytest.approx(0.0)


def test_run_low_corr_candidate_signal_comparison_rejects_unavailable_signal(tmp_path: Path, monkeypatch) -> None:
    def fake_build_candidate_protocol_signal_panel(**kwargs):
        return {
            "manifest": {},
            "quality": {},
            "evaluation_panel": _signal_panel(),
            "available_signals": [LOW_CORR_SIGNAL],
            "signal_columns": ["factor_a_z"],
            "low_corr_factor_columns": ["factor_a_z"],
            "signal_coverage": pd.DataFrame(),
            "rolling_weight_audit": pd.DataFrame(),
            "rolling_fallback_rate": 0.0,
        }

    monkeypatch.setattr(
        low_corr_candidate_signal_comparison,
        "build_candidate_protocol_signal_panel",
        fake_build_candidate_protocol_signal_panel,
    )

    with pytest.raises(ValueError, match="signals not available"):
        low_corr_candidate_signal_comparison.run_low_corr_candidate_signal_comparison(
            years=(2026,),
            horizon=1,
            signals=("missing_score",),
            output_dir=tmp_path,
        )
