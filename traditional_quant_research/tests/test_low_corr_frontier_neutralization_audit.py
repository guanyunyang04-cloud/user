from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from traditional_quant_research.experiments import low_corr_frontier_neutralization_audit


def _frontier_panel() -> pd.DataFrame:
    rows = []
    dates = pd.date_range("2026-01-02", periods=6, freq="B")
    codes = ["600000.SH", "000001.SZ", "000002.SZ", "000003.SZ"]
    signals = (
        low_corr_frontier_neutralization_audit.ROLLING_IC_SIGNAL,
        low_corr_frontier_neutralization_audit.IC_WEIGHTED_SIGNAL,
        low_corr_frontier_neutralization_audit.LOW_CORR_SIGNAL,
    )
    for code_index, code in enumerate(codes):
        for date_index, date in enumerate(dates):
            base = 10.0 + code_index * 2.0 + date_index * 0.1
            row = {
                "date": date,
                "code": code,
                "open": base,
                "close": base + 0.1 + code_index * 0.02,
                "amount": 100_000_000.0 + code_index * 20_000_000.0,
                "volume": 1_000_000.0 + code_index * 100_000.0,
                "is_tradeable": True,
                "log_amount_mean_20d_z": -0.4 + code_index * 0.25,
                "momentum_20d_z": 0.1 + code_index * 0.2,
                "reversal_5d_z": -0.1 + code_index * 0.05,
                "neg_volatility_20d_z": 0.2,
                "neg_amplitude_20d_z": 0.3,
            }
            for signal_offset, signal in enumerate(signals):
                row[signal] = 5.0 - code_index + date_index * 0.01 + signal_offset * 0.1
            rows.append(row)
    return pd.DataFrame(rows)


def test_add_proxy_neutralized_signals_residualizes_by_date() -> None:
    frame = pd.DataFrame(
        [
            {"date": "2026-01-02", "code": "A", "score": 1.0, "size": 0.0},
            {"date": "2026-01-02", "code": "B", "score": 3.0, "size": 1.0},
            {"date": "2026-01-02", "code": "C", "score": 5.0, "size": 2.0},
            {"date": "2026-01-05", "code": "A", "score": 2.0, "size": 0.0},
            {"date": "2026-01-05", "code": "B", "score": 4.0, "size": 1.0},
            {"date": "2026-01-05", "code": "C", "score": 6.0, "size": 2.0},
        ]
    )

    output, mapping = low_corr_frontier_neutralization_audit.add_proxy_neutralized_signals(
        frame,
        ("score",),
        ("size",),
    )

    neutral_col = mapping["score"]
    assert neutral_col == "score_proxy_neutral"
    assert output[neutral_col].abs().max() < 1e-12

    corr = low_corr_frontier_neutralization_audit.signal_neutralizer_correlation(
        output,
        ("score", neutral_col),
        ("size",),
    )
    original = corr.loc[corr["signal"] == "score"].iloc[0]
    neutralized = corr.loc[corr["signal"] == neutral_col].iloc[0]
    assert original["mean_abs_daily_corr"] == pytest.approx(1.0)
    assert neutralized["daily_count"] == 0
    assert pd.isna(neutralized["mean_abs_daily_corr"])


def test_summarize_neutralization_audit_adds_delta_vs_original() -> None:
    base = low_corr_frontier_neutralization_audit.ROLLING_IC_SIGNAL
    neutral = f"{base}{low_corr_frontier_neutralization_audit.NEUTRAL_SUFFIX}"
    summary = pd.DataFrame(
        [
            {"eval_year": 2025, "base_signal": base, "variant_signal": base, "neutralized": False, "neutralizer_spec": "none", "fee_bps": 30.0, "annualized_return": 0.20, "sharpe": 1.0, "max_drawdown": -0.04, "mean_turnover": 0.8, "periods": 4},
            {"eval_year": 2025, "base_signal": base, "variant_signal": neutral, "neutralized": True, "neutralizer_spec": "size", "fee_bps": 30.0, "annualized_return": 0.12, "sharpe": 0.7, "max_drawdown": -0.06, "mean_turnover": 0.9, "periods": 4},
            {"eval_year": 2026, "base_signal": base, "variant_signal": base, "neutralized": False, "neutralizer_spec": "none", "fee_bps": 30.0, "annualized_return": 0.10, "sharpe": 0.5, "max_drawdown": -0.08, "mean_turnover": 0.7, "periods": 2},
            {"eval_year": 2026, "base_signal": base, "variant_signal": neutral, "neutralized": True, "neutralizer_spec": "size", "fee_bps": 30.0, "annualized_return": 0.14, "sharpe": 0.8, "max_drawdown": -0.03, "mean_turnover": 0.6, "periods": 2},
        ]
    )

    aggregate = low_corr_frontier_neutralization_audit.summarize_neutralization_audit(summary)
    neutral_row = aggregate.loc[aggregate["neutralized"]].iloc[0]
    original_row = aggregate.loc[~aggregate["neutralized"]].iloc[0]

    assert neutral_row["mean_annualized_return"] == pytest.approx(0.13)
    assert neutral_row["mean_delta_annualized_return_vs_original"] == pytest.approx(-0.02)
    assert neutral_row["positive_delta_year_rate_vs_original"] == pytest.approx(0.5)
    assert original_row["mean_delta_annualized_return_vs_original"] == pytest.approx(0.0)


def test_run_low_corr_frontier_neutralization_audit_writes_outputs(tmp_path: Path, monkeypatch) -> None:
    panel = _frontier_panel()
    signals = (
        low_corr_frontier_neutralization_audit.ROLLING_IC_SIGNAL,
        low_corr_frontier_neutralization_audit.IC_WEIGHTED_SIGNAL,
        low_corr_frontier_neutralization_audit.LOW_CORR_SIGNAL,
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
        low_corr_frontier_neutralization_audit,
        "build_candidate_protocol_signal_panel",
        fake_build_candidate_protocol_signal_panel,
    )

    result = low_corr_frontier_neutralization_audit.run_low_corr_frontier_neutralization_audit(
        years=(2026,),
        final_end_date="2026-06-01",
        horizon=1,
        signals=signals,
        neutralize_by=("log_amount_mean_20d_z",),
        top_n=2,
        rebalance_frequency="daily",
        buffer_multiplier=1.0,
        fee_bps_values=(30.0,),
        execution_constraints=False,
        output_dir=tmp_path,
        write_research_log=True,
        research_log_path=tmp_path / "research_log.md",
    )

    run_dir = tmp_path / result["run_id"]
    expected_files = [
        "neutralization_summary.csv",
        "neutralization_aggregate.csv",
        "neutralization_trades.csv",
        "neutralization_basket_exposure.csv",
        "neutralization_signal_coverage.csv",
        "neutralization_signal_neutralizer_correlation.csv",
        "neutralization_meta.csv",
        "summary.json",
        "summary.md",
    ]
    for name in expected_files:
        assert (run_dir / name).exists()
    assert (tmp_path / "research_log.md").exists()
    assert result["snapshot_id"] == "fixture-snapshot"
    assert result["candidate_count"] == 0

    aggregate = pd.read_csv(run_dir / "neutralization_aggregate.csv")
    assert set(aggregate["base_signal"]) == set(signals)
    assert aggregate["neutralized"].isin([True, False]).all()
    assert "mean_delta_annualized_return_vs_original" in aggregate.columns
    trades = pd.read_csv(run_dir / "neutralization_trades.csv")
    assert set(trades["base_signal"]) == set(signals)
    coverage = pd.read_csv(run_dir / "neutralization_signal_coverage.csv")
    assert coverage["factor"].str.endswith(low_corr_frontier_neutralization_audit.NEUTRAL_SUFFIX).any()
    corr = pd.read_csv(run_dir / "neutralization_signal_neutralizer_correlation.csv")
    assert set(corr["neutralizer"]) == {"log_amount_mean_20d_z"}
