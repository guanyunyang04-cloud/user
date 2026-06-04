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


def test_build_candidate_protocol_signal_panel_can_include_metric_exposures(monkeypatch) -> None:
    factor_panel = pd.DataFrame(
        [
            {"date": "2026-01-02", "code": "A", "factor_a_z": 0.1, "fwd_ret_1d": 0.01, "turn": 1.0, LOW_CORR_SIGNAL: 0.2},
            {"date": "2026-01-02", "code": "B", "factor_a_z": 0.2, "fwd_ret_1d": 0.02, "turn": 3.0, LOW_CORR_SIGNAL: 0.3},
            {"date": "2026-01-05", "code": "A", "factor_a_z": 0.3, "fwd_ret_1d": 0.03, "turn": 2.0, LOW_CORR_SIGNAL: 0.4},
            {"date": "2026-01-05", "code": "B", "factor_a_z": 0.4, "fwd_ret_1d": 0.04, "turn": 4.0, LOW_CORR_SIGNAL: 0.5},
        ]
    )
    captured: dict[str, object] = {}

    def fake_build_low_corr_signal_panel(**kwargs):
        captured.update(kwargs)
        return {
            "manifest": {"snapshot_id": "fixture"},
            "quality": {},
            "factor_panel": factor_panel,
            "signal_columns": ["factor_a_z"],
            "factor_directions": {"factor_a_z": 1},
            "label": "fwd_ret_1d",
            "single_factor_ic": pd.DataFrame(),
            "low_corr_factor_columns": ["factor_a_z"],
        }

    def fake_add_baseline_score(frame, **kwargs):
        output = frame.copy()
        output[low_corr_candidate_signal_comparison.BASELINE_SIGNAL] = output["factor_a_z"]
        return output

    def fake_add_score(frame, *args, score_col: str, **kwargs):
        output = frame.copy()
        output[score_col] = output["factor_a_z"] + 0.1
        return output

    def fake_rolling_weights(frame, factor_cols, label_col, **kwargs):
        rows = []
        for date in pd.to_datetime(frame["date"]).unique():
            rows.append(
                {
                    "date": pd.Timestamp(date),
                    "factor": factor_cols[0],
                    "weight": 1.0,
                    "direction": 1,
                    "mean_rank_ic": 0.1,
                    "history_days": 10,
                    "is_fallback": False,
                }
            )
        return pd.DataFrame(rows)

    monkeypatch.setattr(low_corr_candidate_signal_comparison, "build_low_corr_signal_panel", fake_build_low_corr_signal_panel)
    monkeypatch.setattr(low_corr_candidate_signal_comparison, "add_baseline_score", fake_add_baseline_score)
    monkeypatch.setattr(low_corr_candidate_signal_comparison, "add_equal_rank_score", fake_add_score)
    monkeypatch.setattr(low_corr_candidate_signal_comparison, "add_ic_weighted_rank_score", fake_add_score)
    monkeypatch.setattr(low_corr_candidate_signal_comparison, "add_rank_score_from_weight_table", fake_add_score)
    monkeypatch.setattr(low_corr_candidate_signal_comparison, "rolling_ic_weights_by_date", fake_rolling_weights)

    built = low_corr_candidate_signal_comparison.build_candidate_protocol_signal_panel(
        root=None,
        history_start_date="2026-01-01",
        fit_start_date="2026-01-01",
        fit_end_date="2026-01-31",
        start_date="2026-01-01",
        end_date="2026-01-31",
        horizon=1,
        label_mode="raw",
        factor_set="expanded",
        max_factor_corr=0.75,
        rolling_window=10,
        rolling_min_periods=2,
        include_metrics=True,
    )

    panel = built["evaluation_panel"].set_index(["date", "code"])
    assert captured["factor_set"] == "expanded"
    assert captured["include_metrics"] is True
    assert built["factor_set"] == "expanded"
    assert "turn_xsec_z" in built["metric_exposure_columns"]
    assert panel.loc[(pd.Timestamp("2026-01-02"), "A"), "turn_xsec_z"] == pytest.approx(-1.0)
    assert panel.loc[(pd.Timestamp("2026-01-02"), "B"), "turn_xsec_z"] == pytest.approx(1.0)


def test_build_candidate_protocol_signal_panel_can_include_factor_pruned_signal(tmp_path: Path, monkeypatch) -> None:
    factor_panel = pd.DataFrame(
        [
            {"date": "2026-01-02", "code": "A", "value_z": 1.0, "risk_z": 3.0, "fwd_ret_1d": 0.01, LOW_CORR_SIGNAL: 0.1},
            {"date": "2026-01-02", "code": "B", "value_z": 2.0, "risk_z": 2.0, "fwd_ret_1d": 0.02, LOW_CORR_SIGNAL: 0.2},
            {"date": "2026-01-02", "code": "C", "value_z": 3.0, "risk_z": 1.0, "fwd_ret_1d": 0.03, LOW_CORR_SIGNAL: 0.3},
        ]
    )
    pd.DataFrame(
        [
            {
                "eval_year": 2026,
                "horizon": 1,
                "candidate_signal_name": "factor_pruned_rank_score_h1_prior_fit",
                "selected_factors": "value_z,risk_z",
                "selected_factor_directions": '{"value_z": 1, "risk_z": -1}',
                "fit_uses_eval_year": False,
            }
        ]
    ).to_csv(tmp_path / "factor_pruning_plan.csv", index=False)

    def fake_build_low_corr_signal_panel(**kwargs):
        return {
            "manifest": {"snapshot_id": "fixture"},
            "quality": {},
            "factor_panel": factor_panel,
            "signal_columns": ["value_z", "risk_z"],
            "factor_directions": {"value_z": 1, "risk_z": -1},
            "label": "fwd_ret_1d",
            "single_factor_ic": pd.DataFrame(
                [
                    {"signal": "value_z", "mean_rank_ic": 0.1},
                    {"signal": "risk_z", "mean_rank_ic": -0.1},
                ]
            ),
            "low_corr_factor_columns": ["value_z"],
        }

    monkeypatch.setattr(low_corr_candidate_signal_comparison, "build_low_corr_signal_panel", fake_build_low_corr_signal_panel)

    built = low_corr_candidate_signal_comparison.build_candidate_protocol_signal_panel(
        root=None,
        history_start_date="2026-01-01",
        fit_start_date="2026-01-01",
        fit_end_date="2026-01-31",
        start_date="2026-01-01",
        end_date="2026-01-31",
        horizon=1,
        label_mode="raw",
        factor_set="core",
        max_factor_corr=0.75,
        rolling_window=2,
        rolling_min_periods=1,
        factor_pruning_run_dir=tmp_path,
    )

    signal = "factor_pruned_rank_score_h1_prior_fit"
    panel = built["evaluation_panel"].sort_values("code")
    assert built["factor_pruning_signal"] == signal
    assert built["factor_pruning_plan_rows"] == 1
    assert signal in built["available_signals"]
    assert panel[signal].tolist() == pytest.approx([1 / 3, 2 / 3, 1.0])


def test_build_candidate_protocol_signal_panel_can_include_ml_signal(tmp_path: Path, monkeypatch) -> None:
    factor_panel = pd.DataFrame(
        [
            {"date": "2026-01-02", "code": "A", "value_z": 1.0, "fwd_ret_1d": 0.01, LOW_CORR_SIGNAL: 0.1},
            {"date": "2026-01-02", "code": "B", "value_z": 2.0, "fwd_ret_1d": 0.02, LOW_CORR_SIGNAL: 0.2},
            {"date": "2026-01-05", "code": "A", "value_z": 3.0, "fwd_ret_1d": 0.03, LOW_CORR_SIGNAL: 0.3},
        ]
    )
    ml_signal = "ml_lgbm_xsec_excess_score_h1_prior_fit"
    pd.DataFrame(
        [
            {"eval_year": 2026, "date": "2026-01-02", "code": "A", "horizon": 1, "ml_signal_name": ml_signal, "score": 0.7, "fit_uses_eval_year": False},
            {"eval_year": 2026, "date": "2026-01-02", "code": "B", "horizon": 1, "ml_signal_name": ml_signal, "score": 0.4, "fit_uses_eval_year": False},
            {"eval_year": 2026, "date": "2026-01-05", "code": "A", "horizon": 1, "ml_signal_name": ml_signal, "score": 0.2, "fit_uses_eval_year": False},
        ]
    ).to_csv(tmp_path / "ml_signal_predictions.csv", index=False)

    def fake_build_low_corr_signal_panel(**kwargs):
        return {
            "manifest": {"snapshot_id": "fixture"},
            "quality": {},
            "factor_panel": factor_panel,
            "signal_columns": ["value_z"],
            "factor_directions": {"value_z": 1},
            "label": "fwd_ret_1d",
            "single_factor_ic": pd.DataFrame([{"signal": "value_z", "mean_rank_ic": 0.1}]),
            "low_corr_factor_columns": ["value_z"],
        }

    monkeypatch.setattr(low_corr_candidate_signal_comparison, "build_low_corr_signal_panel", fake_build_low_corr_signal_panel)

    built = low_corr_candidate_signal_comparison.build_candidate_protocol_signal_panel(
        root=None,
        history_start_date="2026-01-01",
        fit_start_date="2026-01-01",
        fit_end_date="2026-01-31",
        start_date="2026-01-01",
        end_date="2026-01-31",
        horizon=1,
        label_mode="raw",
        factor_set="core",
        max_factor_corr=0.75,
        rolling_window=2,
        rolling_min_periods=1,
        ml_signal_run_dir=tmp_path,
    )

    panel = built["evaluation_panel"].set_index(["date", "code"])
    assert built["ml_signal"] == ml_signal
    assert built["ml_prediction_rows"] == 3
    assert ml_signal in built["available_signals"]
    assert panel.loc[(pd.Timestamp("2026-01-02"), "A"), ml_signal] == pytest.approx(0.7)


def test_read_ml_signal_predictions_rejects_eval_year_leakage(tmp_path: Path) -> None:
    ml_signal = "ml_lgbm_xsec_excess_score_h1_prior_fit"
    pd.DataFrame(
        [
            {"eval_year": 2026, "date": "2026-01-02", "code": "A", "horizon": 1, "ml_signal_name": ml_signal, "score": 0.7, "fit_uses_eval_year": True},
        ]
    ).to_csv(tmp_path / "ml_signal_predictions.csv", index=False)

    with pytest.raises(ValueError, match="prior-fit"):
        low_corr_candidate_signal_comparison.read_ml_signal_predictions(tmp_path, horizon=1)


def test_run_low_corr_candidate_signal_comparison_writes_outputs(tmp_path: Path, monkeypatch) -> None:
    panel = _signal_panel()
    captured: list[dict[str, object]] = []

    def fake_build_candidate_protocol_signal_panel(**kwargs):
        captured.append(kwargs)
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
        factor_set="expanded",
        signals=(low_corr_candidate_signal_comparison.EQUAL_SIGNAL, LOW_CORR_SIGNAL),
        top_n=2,
        rebalance_frequency="daily",
        buffer_multiplier=1.0,
        fee_bps_values=(0.0, 30.0),
        execution_constraints=False,
        ml_signal_run_dir=tmp_path / "ml_signal",
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
    assert result["factor_set"] == "expanded"
    assert result["candidate_count"] == 0
    assert result["ml_signal_run_dir"] == str(tmp_path / "ml_signal")
    assert captured[0]["factor_set"] == "expanded"
    assert captured[0]["ml_signal_run_dir"] == tmp_path / "ml_signal"

    aggregate = pd.read_csv(run_dir / "signal_protocol_aggregate.csv")
    assert set(aggregate["signal"]) == {low_corr_candidate_signal_comparison.EQUAL_SIGNAL, LOW_CORR_SIGNAL}
    assert "mean_delta_annualized_return_vs_low_corr" in aggregate.columns
    trades = pd.read_csv(run_dir / "signal_protocol_trades.csv")
    assert set(trades["signal"]) == {low_corr_candidate_signal_comparison.EQUAL_SIGNAL, LOW_CORR_SIGNAL}
    metadata = pd.read_csv(run_dir / "signal_comparison_meta.csv")
    assert set(metadata["factor_set"]) == {"expanded"}
    assert set(metadata["ml_signal_run_dir"]) == {str(tmp_path / "ml_signal")}
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
