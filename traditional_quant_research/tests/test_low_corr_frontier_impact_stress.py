from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from traditional_quant_research.experiments import low_corr_frontier_impact_stress
from traditional_quant_research.experiments.low_corr_exposure_grid import LOW_CORR_SIGNAL


def _frontier_panel() -> pd.DataFrame:
    rows = []
    dates = pd.date_range("2026-01-02", periods=6, freq="B")
    codes = ["600000.SH", "000001.SZ", "000002.SZ"]
    signals = (
        low_corr_frontier_impact_stress.ROLLING_IC_SIGNAL,
        low_corr_frontier_impact_stress.IC_WEIGHTED_SIGNAL,
        LOW_CORR_SIGNAL,
    )
    for code_index, code in enumerate(codes):
        for date_index, date in enumerate(dates):
            base = 10.0 + code_index * 2.0 + date_index * 0.1
            row = {
                "date": date,
                "code": code,
                "open": base,
                "close": base + 0.1 + code_index * 0.03,
                "amount": 100_000_000.0 + code_index * 20_000_000.0,
                "volume": 1_000_000.0 + code_index * 100_000.0,
                "is_tradeable": True,
                "log_amount_mean_20d_z": -0.2 + code_index * 0.3,
                "momentum_20d_z": 0.1 + code_index * 0.2,
                "reversal_5d_z": -0.1 + code_index * 0.1,
                "neg_volatility_20d_z": 0.2,
                "neg_amplitude_20d_z": 0.3,
            }
            for signal_offset, signal in enumerate(signals):
                row[signal] = 5.0 - code_index + date_index * 0.01 + signal_offset * 0.1
            rows.append(row)
    return pd.DataFrame(rows)


def test_apply_fee_and_participation_impact_combines_fixed_fee_and_impact() -> None:
    trades = pd.DataFrame(
        [
            {"trade_id": 0, "gross_return": 0.05, "turnover": 1.0},
            {"trade_id": 1, "gross_return": 0.02, "turnover": 0.5},
        ]
    )
    liquidity = pd.DataFrame(
        [
            {"trade_id": 0, "participation_p95_100m": 0.02},
            {"trade_id": 1, "participation_p95_100m": 0.01},
        ]
    )

    output = low_corr_frontier_impact_stress.apply_fee_and_participation_impact(
        trades,
        liquidity,
        fee_bps=30.0,
        capital_amount=100_000_000.0,
        impact_bps_per_1pct=5.0,
    )

    assert output.loc[0, "fee_cost"] == pytest.approx(0.003)
    assert output.loc[0, "impact_rate"] == pytest.approx(0.001)
    assert output.loc[0, "impact_cost"] == pytest.approx(0.001)
    assert output.loc[0, "cost"] == pytest.approx(0.004)
    assert output.loc[0, "net_return"] == pytest.approx(0.046)
    assert output.loc[1, "fee_cost"] == pytest.approx(0.0015)
    assert output.loc[1, "impact_cost"] == pytest.approx(0.00025)
    assert output.loc[1, "net_return"] == pytest.approx(0.01825)


def test_apply_fee_and_participation_impact_requires_matching_participation_column() -> None:
    trades = pd.DataFrame([{"trade_id": 0, "gross_return": 0.05, "turnover": 1.0}])
    liquidity = pd.DataFrame([{"trade_id": 0, "participation_p95_10m": 0.02}])

    with pytest.raises(ValueError, match="participation_p95_100m"):
        low_corr_frontier_impact_stress.apply_fee_and_participation_impact(
            trades,
            liquidity,
            fee_bps=30.0,
            capital_amount=100_000_000.0,
            impact_bps_per_1pct=5.0,
        )


def test_summarize_impact_stress_adds_impact_drag_and_low_corr_delta() -> None:
    other = low_corr_frontier_impact_stress.ROLLING_IC_SIGNAL
    summary = pd.DataFrame(
        [
            {"eval_year": 2025, "signal": LOW_CORR_SIGNAL, "fee_bps": 30.0, "capital_amount": 100_000_000.0, "impact_bps_per_1pct": 0.0, "annualized_return": 0.20, "sharpe": 1.0, "max_drawdown": -0.04, "mean_turnover": 0.8, "mean_impact_cost": 0.0, "mean_total_cost": 0.003, "periods": 4},
            {"eval_year": 2025, "signal": LOW_CORR_SIGNAL, "fee_bps": 30.0, "capital_amount": 100_000_000.0, "impact_bps_per_1pct": 5.0, "annualized_return": 0.18, "sharpe": 0.9, "max_drawdown": -0.05, "mean_turnover": 0.8, "mean_impact_cost": 0.001, "mean_total_cost": 0.004, "periods": 4},
            {"eval_year": 2025, "signal": other, "fee_bps": 30.0, "capital_amount": 100_000_000.0, "impact_bps_per_1pct": 0.0, "annualized_return": 0.30, "sharpe": 1.2, "max_drawdown": -0.03, "mean_turnover": 0.7, "mean_impact_cost": 0.0, "mean_total_cost": 0.003, "periods": 4},
            {"eval_year": 2025, "signal": other, "fee_bps": 30.0, "capital_amount": 100_000_000.0, "impact_bps_per_1pct": 5.0, "annualized_return": 0.25, "sharpe": 1.1, "max_drawdown": -0.04, "mean_turnover": 0.7, "mean_impact_cost": 0.002, "mean_total_cost": 0.005, "periods": 4},
            {"eval_year": 2026, "signal": LOW_CORR_SIGNAL, "fee_bps": 30.0, "capital_amount": 100_000_000.0, "impact_bps_per_1pct": 0.0, "annualized_return": 0.10, "sharpe": 0.7, "max_drawdown": -0.06, "mean_turnover": 0.9, "mean_impact_cost": 0.0, "mean_total_cost": 0.003, "periods": 2},
            {"eval_year": 2026, "signal": LOW_CORR_SIGNAL, "fee_bps": 30.0, "capital_amount": 100_000_000.0, "impact_bps_per_1pct": 5.0, "annualized_return": 0.08, "sharpe": 0.6, "max_drawdown": -0.07, "mean_turnover": 0.9, "mean_impact_cost": 0.001, "mean_total_cost": 0.004, "periods": 2},
            {"eval_year": 2026, "signal": other, "fee_bps": 30.0, "capital_amount": 100_000_000.0, "impact_bps_per_1pct": 0.0, "annualized_return": 0.09, "sharpe": 0.4, "max_drawdown": -0.08, "mean_turnover": 0.7, "mean_impact_cost": 0.0, "mean_total_cost": 0.003, "periods": 2},
            {"eval_year": 2026, "signal": other, "fee_bps": 30.0, "capital_amount": 100_000_000.0, "impact_bps_per_1pct": 5.0, "annualized_return": 0.07, "sharpe": 0.3, "max_drawdown": -0.09, "mean_turnover": 0.7, "mean_impact_cost": 0.002, "mean_total_cost": 0.005, "periods": 2},
        ]
    )

    aggregate = low_corr_frontier_impact_stress.summarize_impact_stress(summary)
    other5 = aggregate.loc[(aggregate["signal"] == other) & (aggregate["impact_bps_per_1pct"] == 5.0)].iloc[0]
    low5 = aggregate.loc[(aggregate["signal"] == LOW_CORR_SIGNAL) & (aggregate["impact_bps_per_1pct"] == 5.0)].iloc[0]

    assert other5["mean_annualized_return"] == pytest.approx(0.16)
    assert other5["mean_impact_drag_vs_no_impact"] == pytest.approx(-0.035)
    assert other5["mean_delta_annualized_return_vs_low_corr"] == pytest.approx(0.03)
    assert other5["positive_delta_year_rate_vs_low_corr"] == pytest.approx(0.5)
    assert low5["mean_delta_annualized_return_vs_low_corr"] == pytest.approx(0.0)
    assert low5["positive_delta_year_rate_vs_low_corr"] == pytest.approx(0.0)


def test_run_low_corr_frontier_impact_stress_writes_outputs(tmp_path: Path, monkeypatch) -> None:
    panel = _frontier_panel()
    signals = (
        low_corr_frontier_impact_stress.ROLLING_IC_SIGNAL,
        low_corr_frontier_impact_stress.IC_WEIGHTED_SIGNAL,
        LOW_CORR_SIGNAL,
    )

    def fake_build_candidate_protocol_signal_panel(**kwargs):
        return {
            "manifest": {"snapshot_id": "fixture-snapshot"},
            "quality": {"failure_count": 0, "missing_bar_rows": 0, "st_rows": 0, "suspended_like_rows": 0},
            "evaluation_panel": panel,
            "available_signals": list(signals),
            "rolling_fallback_rate": 0.0,
        }

    monkeypatch.setattr(
        low_corr_frontier_impact_stress,
        "build_candidate_protocol_signal_panel",
        fake_build_candidate_protocol_signal_panel,
    )

    result = low_corr_frontier_impact_stress.run_low_corr_frontier_impact_stress(
        years=(2026,),
        final_end_date="2026-06-01",
        horizon=1,
        signals=signals,
        top_n=2,
        rebalance_frequency="daily",
        buffer_multiplier=1.0,
        fee_bps_values=(30.0,),
        capital_amounts=(10_000_000.0,),
        impact_bps_per_1pct_values=(0.0, 5.0),
        execution_constraints=False,
        output_dir=tmp_path,
        write_research_log=True,
        research_log_path=tmp_path / "research_log.md",
    )

    run_dir = tmp_path / result["run_id"]
    expected_files = [
        "impact_stress_summary.csv",
        "impact_stress_aggregate.csv",
        "impact_adjusted_trades.csv",
        "impact_trade_liquidity.csv",
        "impact_liquidity_summary.csv",
        "impact_stress_meta.csv",
        "summary.json",
        "summary.md",
    ]
    for name in expected_files:
        assert (run_dir / name).exists()
    assert (tmp_path / "research_log.md").exists()
    assert result["snapshot_id"] == "fixture-snapshot"
    assert result["candidate_count"] == 0

    summary = pd.read_csv(run_dir / "impact_stress_summary.csv")
    assert set(summary["signal"]) == set(signals)
    assert set(summary["impact_bps_per_1pct"]) == {0.0, 5.0}
    assert "mean_impact_cost" in summary.columns
    aggregate = pd.read_csv(run_dir / "impact_stress_aggregate.csv")
    assert "mean_impact_drag_vs_no_impact" in aggregate.columns
    liquidity = pd.read_csv(run_dir / "impact_trade_liquidity.csv")
    assert "participation_p95_10m" in liquidity.columns
    metadata = pd.read_csv(run_dir / "impact_stress_meta.csv")
    assert metadata.iloc[0]["rolling_fallback_rate"] == pytest.approx(0.0)
