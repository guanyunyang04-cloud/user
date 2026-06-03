from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from traditional_quant_research.experiments import low_corr_candidate_frontier_audit


def _audit_panel() -> pd.DataFrame:
    rows = []
    dates = pd.date_range("2026-01-02", periods=5, freq="B")
    codes = ["600000.SH", "000001.SZ", "000002.SZ"]
    for code_index, code in enumerate(codes):
        for date_index, date in enumerate(dates):
            base = 10.0 + code_index * 5.0 + date_index * 0.2
            rows.append(
                {
                    "date": date,
                    "code": code,
                    "open": base,
                    "close": base + 0.1 + code_index * 0.02,
                    "amount": 80_000_000.0 + code_index * 20_000_000.0 + date_index * 1_000_000.0,
                    "volume": 1_000_000.0 + code_index * 100_000.0,
                    "is_tradeable": True,
                    low_corr_candidate_frontier_audit.LOW_CORR_SIGNAL: 10.0 - code_index + date_index * 0.01,
                    "factor_a_z": 0.5 - code_index * 0.1,
                    "factor_b_z": -0.2 + code_index * 0.2,
                    "log_amount_mean_20d": 8.0 + code_index,
                    "log_amount_mean_20d_z": -0.2 + code_index * 0.3,
                    "momentum_20d_z": 0.1 + code_index * 0.2,
                    "reversal_5d_z": -0.1,
                    "neg_volatility_20d_z": 0.2,
                    "neg_amplitude_20d_z": 0.3,
                }
            )
    return pd.DataFrame(rows)


def test_expand_selected_holdings_joins_signal_date_columns() -> None:
    frame = pd.DataFrame(
        [
            {"date": "2026-01-02", "code": "A", "amount": 100.0, "score": 0.8},
            {"date": "2026-01-02", "code": "B", "amount": 200.0, "score": 0.4},
            {"date": "2026-01-03", "code": "A", "amount": 120.0, "score": 0.7},
        ]
    )
    trades = pd.DataFrame(
        [
            {
                "signal_date": "2026-01-02",
                "entry_date": "2026-01-03",
                "exit_date": "2026-01-04",
                "codes": "A,B",
            }
        ]
    )

    output = low_corr_candidate_frontier_audit.expand_selected_holdings(
        frame,
        trades,
        extra_columns=("amount", "score"),
    )

    assert output["trade_id"].tolist() == [0, 0]
    assert output["code"].tolist() == ["A", "B"]
    assert output["amount"].tolist() == [100.0, 200.0]
    assert output["score"].tolist() == [0.8, 0.4]


def test_summarize_trade_liquidity_computes_capacity_proxies() -> None:
    selected = pd.DataFrame(
        [
            {
                "trade_id": 0,
                "signal_date": "2026-01-02",
                "entry_date": "2026-01-03",
                "exit_date": "2026-01-31",
                "code": "A",
                "amount": 100_000_000.0,
                "volume": 1_000_000.0,
                "log_amount_mean_20d_z": 0.2,
                "momentum_20d_z": 0.5,
                low_corr_candidate_frontier_audit.LOW_CORR_SIGNAL: 0.9,
            },
            {
                "trade_id": 0,
                "signal_date": "2026-01-02",
                "entry_date": "2026-01-03",
                "exit_date": "2026-01-31",
                "code": "B",
                "amount": 50_000_000.0,
                "volume": 500_000.0,
                "log_amount_mean_20d_z": -0.2,
                "momentum_20d_z": 0.1,
                low_corr_candidate_frontier_audit.LOW_CORR_SIGNAL: 0.7,
            },
        ]
    )

    per_trade, yearly = low_corr_candidate_frontier_audit.summarize_trade_liquidity(
        selected,
        capital_amounts=(10_000_000.0,),
    )
    row = per_trade.iloc[0]

    assert row["selected_count"] == 2
    assert row["amount_mean"] == pytest.approx(75_000_000.0)
    assert row["amount_p10"] == pytest.approx(55_000_000.0)
    assert row["log_amount_mean_20d_z_mean"] == pytest.approx(0.0)
    assert row["momentum_20d_z_mean"] == pytest.approx(0.3)
    assert row["participation_mean_10m"] == pytest.approx(0.075)
    assert row["participation_p95_10m"] == pytest.approx(0.0975)
    assert yearly.iloc[0]["year"] == 2026
    assert yearly.iloc[0]["worst_participation_max_10m"] == pytest.approx(0.1)


def test_summarize_candidate_audit_adds_fee_stress_metrics() -> None:
    summary = pd.DataFrame(
        [
            {"eval_year": 2025, "fee_bps": 0.0, "annualized_return": 0.20, "sharpe": 1.0, "max_drawdown": -0.04, "mean_turnover": 0.7, "periods": 4},
            {"eval_year": 2025, "fee_bps": 30.0, "annualized_return": 0.15, "sharpe": 0.8, "max_drawdown": -0.05, "mean_turnover": 0.7, "periods": 4},
            {"eval_year": 2026, "fee_bps": 0.0, "annualized_return": 0.10, "sharpe": 0.6, "max_drawdown": -0.06, "mean_turnover": 0.9, "periods": 2},
            {"eval_year": 2026, "fee_bps": 30.0, "annualized_return": -0.02, "sharpe": -0.1, "max_drawdown": -0.08, "mean_turnover": 0.9, "periods": 2},
        ]
    )

    aggregate = low_corr_candidate_frontier_audit.summarize_candidate_audit(summary)
    fee30 = aggregate.loc[aggregate["fee_bps"] == 30.0].iloc[0]

    assert fee30["eval_year_count"] == 2
    assert fee30["mean_annualized_return"] == pytest.approx(0.065)
    assert fee30["min_annualized_return"] == pytest.approx(-0.02)
    assert fee30["positive_year_rate"] == pytest.approx(0.5)
    assert fee30["mean_cost_drag_vs_0bps"] == pytest.approx(-0.085)
    assert fee30["total_periods"] == 6


def test_run_low_corr_candidate_frontier_audit_writes_outputs(tmp_path: Path, monkeypatch) -> None:
    panel = _audit_panel()
    captured: list[dict[str, object]] = []

    def fake_build_low_corr_signal_panel(**kwargs):
        captured.append(kwargs)
        return {
            "manifest": {"snapshot_id": "fixture-snapshot"},
            "quality": {"failure_count": 0, "missing_bar_rows": 0, "st_rows": 0, "suspended_like_rows": 0},
            "factor_panel": panel,
            "fit_panel": panel,
            "evaluation_panel": panel,
            "raw_factor_columns": ("factor_a", "factor_b"),
            "signal_columns": ("factor_a_z", "factor_b_z"),
            "label": f"fwd_ret_{kwargs['horizon']}d",
            "single_factor_ic": pd.DataFrame(),
            "factor_correlation": pd.DataFrame(),
            "factor_directions": {"factor_a_z": 1, "factor_b_z": -1},
            "low_corr_factor_columns": ["factor_a_z"],
            "coverage": pd.DataFrame([{"factor": "factor_a_z", "coverage": 1.0}]),
        }

    monkeypatch.setattr(
        low_corr_candidate_frontier_audit,
        "build_low_corr_signal_panel",
        fake_build_low_corr_signal_panel,
    )

    result = low_corr_candidate_frontier_audit.run_low_corr_candidate_frontier_audit(
        years=(2026,),
        final_end_date="2026-06-01",
        horizon=1,
        factor_set="expanded",
        top_n=2,
        rebalance_frequency="daily",
        buffer_multiplier=1.0,
        fee_bps_values=(0.0, 30.0),
        execution_constraints=False,
        capital_amounts=(10_000_000.0,),
        output_dir=tmp_path,
        write_research_log=True,
        research_log_path=tmp_path / "research_log.md",
    )

    run_dir = tmp_path / result["run_id"]
    expected_files = [
        "candidate_protocol_summary.csv",
        "candidate_protocol_aggregate.csv",
        "candidate_protocol_trades.csv",
        "candidate_basket_exposure.csv",
        "candidate_selected_holdings.csv",
        "candidate_trade_liquidity.csv",
        "candidate_liquidity_summary.csv",
        "candidate_factor_coverage.csv",
        "candidate_low_corr_meta.csv",
        "summary.json",
        "summary.md",
    ]
    for name in expected_files:
        assert (run_dir / name).exists()
    assert (tmp_path / "research_log.md").exists()

    assert result["snapshot_id"] == "fixture-snapshot"
    assert result["factor_set"] == "expanded"
    assert result["candidate_count"] == 0
    assert captured[0]["factor_set"] == "expanded"
    aggregate = pd.read_csv(run_dir / "candidate_protocol_aggregate.csv")
    assert sorted(aggregate["fee_bps"].tolist()) == [0.0, 30.0]
    assert "mean_cost_drag_vs_0bps" in aggregate.columns
    holdings = pd.read_csv(run_dir / "candidate_selected_holdings.csv")
    assert set(holdings["code"]).issubset({"600000.SH", "000001.SZ", "000002.SZ"})
    liquidity = pd.read_csv(run_dir / "candidate_trade_liquidity.csv")
    assert "participation_mean_10m" in liquidity.columns
    meta = pd.read_csv(run_dir / "candidate_low_corr_meta.csv")
    assert set(meta["factor_set"]) == {"expanded"}
