from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from traditional_quant_research.experiments import low_corr_frontier_exposure_penalty_grid


def _frontier_panel() -> pd.DataFrame:
    rows = []
    dates = pd.date_range("2026-01-02", periods=6, freq="B")
    codes = ["600000.SH", "000001.SZ", "000002.SZ", "000003.SZ"]
    signals = (
        low_corr_frontier_exposure_penalty_grid.ROLLING_IC_SIGNAL,
        low_corr_frontier_exposure_penalty_grid.IC_WEIGHTED_SIGNAL,
        low_corr_frontier_exposure_penalty_grid.LOW_CORR_SIGNAL,
    )
    for code_index, code in enumerate(codes):
        for date_index, date in enumerate(dates):
            base = 10.0 + code_index * 2.0 + date_index * 0.1
            row = {
                "date": date,
                "code": code,
                "open": base,
                "close": base + 0.1 + code_index * 0.02,
                "is_tradeable": True,
                "log_amount_mean_20d_z": -1.0 + code_index * 0.5,
                "neg_volatility_20d_z": 1.0 - code_index * 0.25,
                "momentum_20d_z": 0.2 + code_index * 0.2,
                "turn_xsec_z": -0.8 + code_index * 0.4,
                "pctChg_xsec_z": 0.1 * code_index,
            }
            for signal_offset, signal in enumerate(signals):
                row[signal] = 5.0 - code_index + date_index * 0.01 + signal_offset * 0.1
            rows.append(row)
    return pd.DataFrame(rows)


def test_summarize_exposure_penalty_grid_adds_delta_vs_strength0() -> None:
    summary = pd.DataFrame(
        [
            {"eval_year": 2025, "signal": "score", "exposure_penalty_cols": "style", "exposure_penalty_strength": 0.0, "fee_bps": 30.0, "annualized_return": 0.10, "sharpe": 1.0, "max_drawdown": -0.03, "mean_turnover": 0.8, "periods": 4},
            {"eval_year": 2025, "signal": "score", "exposure_penalty_cols": "style", "exposure_penalty_strength": 0.5, "fee_bps": 30.0, "annualized_return": 0.12, "sharpe": 1.1, "max_drawdown": -0.02, "mean_turnover": 0.7, "periods": 4},
            {"eval_year": 2026, "signal": "score", "exposure_penalty_cols": "style", "exposure_penalty_strength": 0.0, "fee_bps": 30.0, "annualized_return": 0.06, "sharpe": 0.7, "max_drawdown": -0.04, "mean_turnover": 0.9, "periods": 2},
            {"eval_year": 2026, "signal": "score", "exposure_penalty_cols": "style", "exposure_penalty_strength": 0.5, "fee_bps": 30.0, "annualized_return": 0.03, "sharpe": 0.4, "max_drawdown": -0.05, "mean_turnover": 0.6, "periods": 2},
        ]
    )

    aggregate = low_corr_frontier_exposure_penalty_grid.summarize_exposure_penalty_grid(summary)
    penalized = aggregate.loc[aggregate["exposure_penalty_strength"].eq(0.5)].iloc[0]

    assert penalized["mean_annualized_return"] == pytest.approx(0.075)
    assert penalized["mean_delta_annualized_return_vs_strength0"] == pytest.approx(-0.005)
    assert penalized["positive_delta_year_rate_vs_strength0"] == pytest.approx(0.5)


def test_run_low_corr_frontier_exposure_penalty_grid_writes_outputs(tmp_path: Path, monkeypatch) -> None:
    panel = _frontier_panel()
    signal = low_corr_frontier_exposure_penalty_grid.ROLLING_IC_SIGNAL
    captured: dict[str, object] = {}

    def fake_build_candidate_protocol_signal_panel(**kwargs):
        captured.update(kwargs)
        return {
            "manifest": {"snapshot_id": "fixture-snapshot"},
            "quality": {"failure_count": 0, "missing_bar_rows": 0, "st_rows": 0, "suspended_like_rows": 0},
            "evaluation_panel": panel,
            "available_signals": [signal],
            "rolling_fallback_rate": 0.0,
        }

    monkeypatch.setattr(
        low_corr_frontier_exposure_penalty_grid,
        "build_candidate_protocol_signal_panel",
        fake_build_candidate_protocol_signal_panel,
    )

    result = low_corr_frontier_exposure_penalty_grid.run_low_corr_frontier_exposure_penalty_grid(
        years=(2026,),
        final_end_date="2026-06-01",
        horizon=1,
        signals=(signal,),
        top_n=2,
        rebalance_frequency="daily",
        buffer_multiplier=1.0,
        fee_bps_values=(30.0,),
        exposure_penalty_cols=("log_amount_mean_20d_z", "turn_xsec_z"),
        exposure_penalty_strengths=(0.0, 0.5),
        exposure_columns=("log_amount_mean_20d_z", "turn_xsec_z"),
        execution_constraints=False,
        include_metrics=True,
        output_dir=tmp_path,
        write_research_log=True,
        research_log_path=tmp_path / "research_log.md",
    )

    run_dir = tmp_path / result["run_id"]
    expected_files = [
        "exposure_penalty_summary.csv",
        "exposure_penalty_aggregate.csv",
        "exposure_penalty_trades.csv",
        "exposure_penalty_basket_exposure.csv",
        "exposure_penalty_basket_exposure_summary.csv",
        "exposure_penalty_meta.csv",
        "summary.json",
        "summary.md",
    ]
    for name in expected_files:
        assert (run_dir / name).exists()
    assert (tmp_path / "research_log.md").exists()
    assert result["snapshot_id"] == "fixture-snapshot"
    assert result["candidate_count"] == 0
    assert result["include_metrics"] is True
    assert result["exposure_penalty_cols"] == ["log_amount_mean_20d_z", "turn_xsec_z"]
    assert result["exposure_penalty_strengths"] == [0.0, 0.5]
    assert captured["include_metrics"] is True

    aggregate = pd.read_csv(run_dir / "exposure_penalty_aggregate.csv")
    assert set(aggregate["exposure_penalty_strength"]) == {0.0, 0.5}
    assert "mean_delta_annualized_return_vs_strength0" in aggregate.columns
    summary = pd.read_csv(run_dir / "exposure_penalty_summary.csv")
    assert set(summary["signal"]) == {signal}
    exposure_summary = pd.read_csv(run_dir / "exposure_penalty_basket_exposure_summary.csv")
    assert set(exposure_summary["factor"]) == {"log_amount_mean_20d_z", "turn_xsec_z"}
    meta = pd.read_csv(run_dir / "exposure_penalty_meta.csv")
    assert set(meta["exposure_penalty_cols"]) == {'["log_amount_mean_20d_z", "turn_xsec_z"]'}
