from __future__ import annotations

import pandas as pd
import pytest

from traditional_quant_research.experiments import multifactor_baseline


def _fixture_panel() -> pd.DataFrame:
    rows = []
    dates = pd.date_range("2026-01-02", periods=10, freq="B")
    closes = {
        "600000.SH": [10.0, 10.2, 10.4, 10.5, 10.7, 10.6, 10.8, 10.9, 11.0, 11.2],
        "000001.SZ": [20.0, 19.9, 19.8, 20.1, 20.3, 20.5, 20.4, 20.7, 20.8, 21.0],
        "000002.SZ": [8.0, 8.2, 8.1, 8.3, 8.6, 8.5, 8.7, 8.9, 8.8, 9.0],
        "600004.SH": [15.0, 14.9, 15.1, 15.4, 15.2, 15.5, 15.7, 15.6, 15.9, 16.0],
        "600005.SH": [6.0, 6.1, 6.0, 6.2, 6.3, 6.4, 6.35, 6.5, 6.6, 6.7],
    }
    for code, code_closes in closes.items():
        for date, close in zip(dates, code_closes):
            rows.append(
                {
                    "date": date,
                    "code": code,
                    "open": close - 0.05,
                    "high": close + 0.20,
                    "low": close - 0.20,
                    "close": close,
                    "volume": 1000.0,
                    "amount": close * 1000.0,
                }
            )
    return pd.DataFrame(rows)


def test_run_multifactor_baseline_writes_reproducible_outputs(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(multifactor_baseline, "load_pit_manifest", lambda root=None: {"snapshot_id": "fixture", "dataset": {"date_min": "2026-01-02", "date_max": "2026-01-15"}})
    monkeypatch.setattr(multifactor_baseline, "load_quality_report", lambda root=None: {})
    load_calls = []

    def fake_load_tradeable_panel(root=None, start_date=None, end_date=None):
        load_calls.append({"start_date": start_date, "end_date": end_date})
        return _fixture_panel()

    monkeypatch.setattr(multifactor_baseline, "load_tradeable_panel", fake_load_tradeable_panel)

    result = multifactor_baseline.run_multifactor_baseline(
        history_start_date="2026-01-01",
        start_date="2026-01-02",
        end_date="2026-01-15",
        horizon=1,
        label_mode="xsec-excess",
        rolling_window=3,
        rolling_min_periods=2,
        max_factor_corr=0.75,
        neutralize_by=("log_amount_mean_20d_z",),
        selected_signals=("multifactor_rolling_ic_weighted_score", "multifactor_low_corr_rank_score"),
        selection_filters=multifactor_baseline.parse_selection_filters("open>=0"),
        top_n_values=(2,),
        fee_bps_values=(0.0,),
        rebalance_frequencies=("daily",),
        include_horizon_backtest=True,
        horizon_buffer_multipliers=(1.0, 1.5),
        execution_constraints=True,
        limit_threshold=0.095,
        output_dir=tmp_path,
    )

    run_dir = tmp_path / result["run_id"]
    assert result["snapshot_id"] == "fixture"
    assert load_calls == [{"start_date": "2026-01-01", "end_date": "2026-01-15"}]
    assert result["history_start_date"] == "2026-01-01"
    assert result["history_panel"]["date_min"] == "2026-01-02"
    assert result["label"] == "xsec_excess_ret_1d"
    assert result["label_mode"] == "xsec-excess"
    assert "alpha-style ranking diagnostics" in result["label_mode_note"]
    assert "1-day forward return" in result["backtest_return_note"]
    assert result["include_horizon_backtest"] is True
    assert result["execution_constraints"] is True
    assert result["limit_threshold"] == 0.095
    assert "Execution constraints are enabled" in result["horizon_backtest_note"]
    assert result["horizon_buffer_multipliers"] == [1.0, 1.5]
    assert result["selected_signals"] == ["multifactor_rolling_ic_weighted_score", "multifactor_low_corr_rank_score"]
    assert result["multifactor_signals"] == ["multifactor_rolling_ic_weighted_score", "multifactor_low_corr_rank_score"]
    assert result["selection_filter_report"]["enabled"] is True
    assert result["selection_filter_report"]["rows_after"] <= result["selection_filter_report"]["rows_before"]
    assert result["portfolio_panel"]["rows"] == result["selection_filter_report"]["rows_after"]
    assert result["low_corr_factor_columns"]
    assert result["neutralized_factor_columns"]
    assert (run_dir / "factor_coverage.csv").exists()
    assert (run_dir / "multifactor_ic.csv").exists()
    assert (run_dir / "quantile_spread.csv").exists()
    assert (run_dir / "rolling_ic_weights.csv").exists()
    assert (run_dir / "rolling_ic_weight_audit.csv").exists()
    assert (run_dir / "low_corr_factors.csv").exists()
    assert (run_dir / "neutralized_factors.csv").exists()
    assert (run_dir / "backtest_summary.csv").exists()
    assert (run_dir / "horizon_backtest_summary.csv").exists()
    assert (run_dir / "horizon_backtest_yearly_summary.csv").exists()
    assert (run_dir / "horizon_backtest_period_summary.csv").exists()
    assert (run_dir / "basket_factor_exposure.csv").exists()
    assert (run_dir / "summary.md").exists()
    assert result["best_quantile_spread"]
    assert result["best_horizon_backtests"]
    assert result["horizon_backtest_years"]
    assert result["horizon_backtest_period_types"] == ["monthly", "quarterly"]
    assert result["horizon_backtest_period_count"] > 0
    assert result["basket_exposure_period_types"] == ["monthly", "quarterly"]
    assert 0 < result["basket_exposure_factor_count"] <= len(multifactor_baseline.default_factor_columns())
    assert result["rolling_weight_audit_period_types"] == ["monthly", "quarterly"]


def test_rolling_ic_weight_audit_summarizes_monthly_and_quarterly_weights() -> None:
    weights = pd.DataFrame(
        [
            {"date": "2026-01-02", "factor": "a", "weight": 0.2, "mean_rank_ic": 0.01, "direction": 1, "history_days": 60, "is_fallback": False},
            {"date": "2026-01-03", "factor": "a", "weight": 0.4, "mean_rank_ic": -0.02, "direction": -1, "history_days": 61, "is_fallback": False},
            {"date": "2026-01-02", "factor": "b", "weight": 0.8, "mean_rank_ic": 0.03, "direction": 1, "history_days": 60, "is_fallback": True},
        ]
    )

    audit = multifactor_baseline.rolling_ic_weight_audit(weights)
    monthly_a = audit.loc[(audit["period_type"] == "monthly") & (audit["period"] == "2026-01") & (audit["factor"] == "a")].iloc[0]
    quarterly_b = audit.loc[(audit["period_type"] == "quarterly") & (audit["period"] == "2026Q1") & (audit["factor"] == "b")].iloc[0]

    assert monthly_a["mean_weight"] == pytest.approx(0.3)
    assert monthly_a["positive_direction_rate"] == pytest.approx(0.5)
    assert monthly_a["fallback_rate"] == pytest.approx(0.0)
    assert monthly_a["date_count"] == 2
    assert quarterly_b["mean_weight"] == pytest.approx(0.8)
    assert quarterly_b["fallback_rate"] == pytest.approx(1.0)


def test_selected_basket_factor_exposure_compares_selected_to_universe() -> None:
    frame = pd.DataFrame(
        [
            {"date": "2026-01-02", "code": "A", "factor_z": 1.0},
            {"date": "2026-01-02", "code": "B", "factor_z": 0.0},
            {"date": "2026-01-02", "code": "C", "factor_z": -1.0},
            {"date": "2026-02-02", "code": "A", "factor_z": 2.0},
            {"date": "2026-02-02", "code": "B", "factor_z": 1.0},
            {"date": "2026-02-02", "code": "C", "factor_z": 0.0},
        ]
    )
    trades = pd.DataFrame(
        [
            {"signal_date": "2026-01-02", "exit_date": "2026-01-09", "codes": "A,B"},
            {"signal_date": "2026-02-02", "exit_date": "2026-02-09", "codes": "A"},
        ]
    )

    exposure = multifactor_baseline.selected_basket_factor_exposure(
        frame,
        trades,
        ["factor_z"],
        signal_col="score",
        horizon=5,
        rebalance_frequency="weekly",
        top_n=2,
        buffer_multiplier=1.5,
    )
    monthly_jan = exposure.loc[(exposure["period_type"] == "monthly") & (exposure["period"] == "2026-01")].iloc[0]
    quarterly = exposure.loc[(exposure["period_type"] == "quarterly") & (exposure["period"] == "2026Q1")].iloc[0]

    assert monthly_jan["selected_mean"] == pytest.approx(0.5)
    assert monthly_jan["universe_mean"] == pytest.approx(0.0)
    assert monthly_jan["active_exposure"] == pytest.approx(0.5)
    assert monthly_jan["selected_count"] == 2
    assert quarterly["trade_count"] == 2
    assert quarterly["selected_mean"] == pytest.approx(1.25)
    assert quarterly["universe_mean"] == pytest.approx(0.5)
    assert quarterly["active_exposure"] == pytest.approx(0.75)


def test_apply_horizon_fee_reuses_trade_path_without_changing_gross_return() -> None:
    trades = pd.DataFrame(
        [
            {"gross_return": 0.02, "turnover": 1.0, "net_return": 0.02, "cost": 0.0},
            {"gross_return": -0.01, "turnover": 1.5, "net_return": -0.01, "cost": 0.0},
        ]
    )

    charged = multifactor_baseline.apply_horizon_fee(trades, fee_bps=30)

    assert charged["gross_return"].tolist() == trades["gross_return"].tolist()
    assert charged["cost"].tolist() == pytest.approx([0.003, 0.0045])
    assert charged["net_return"].tolist() == pytest.approx([0.017, -0.0145])
    assert trades["net_return"].tolist() == [0.02, -0.01]


def test_selection_filter_parser_and_application() -> None:
    rules = multifactor_baseline.parse_selection_filters("liquidity_z>=-0.8,momentum_z<1.5")
    frame = pd.DataFrame(
        [
            {"date": "2026-01-02", "liquidity_z": -0.7, "momentum_z": 1.0},
            {"date": "2026-01-02", "liquidity_z": -1.2, "momentum_z": 1.0},
            {"date": "2026-01-03", "liquidity_z": 0.1, "momentum_z": 2.0},
        ]
    )

    filtered, report = multifactor_baseline.apply_selection_filters(frame, rules)

    assert rules == [
        {"column": "liquidity_z", "operator": ">=", "threshold": -0.8},
        {"column": "momentum_z", "operator": "<", "threshold": 1.5},
    ]
    assert len(filtered) == 1
    assert report["enabled"] is True
    assert report["rows_before"] == 3
    assert report["rows_after"] == 1
    assert report["kept_rate"] == pytest.approx(1 / 3)


def test_selection_filter_to_signals_keeps_price_path_rows() -> None:
    frame = pd.DataFrame(
        [
            {"date": "2026-01-02", "score": 1.0, "liquidity_z": -0.7, "open": 10.0},
            {"date": "2026-01-03", "score": 2.0, "liquidity_z": -1.2, "open": 11.0},
        ]
    )
    rules = multifactor_baseline.parse_selection_filters("liquidity_z>=-0.8")

    output = multifactor_baseline.apply_selection_filters_to_signals(frame, ["score"], rules)

    assert len(output) == 2
    assert output["open"].tolist() == [10.0, 11.0]
    assert output["score"].tolist()[0] == 1.0
    assert pd.isna(output["score"].tolist()[1])


def test_selection_filter_rejects_bad_rule_and_missing_column() -> None:
    with pytest.raises(ValueError, match="invalid selection filter"):
        multifactor_baseline.parse_selection_filters("bad-rule")

    frame = pd.DataFrame([{"a": 1.0}])
    with pytest.raises(ValueError, match="selection filter column not found"):
        multifactor_baseline.apply_selection_filters(frame, [{"column": "missing", "operator": ">=", "threshold": 0}])


def test_backtest_return_note_flags_overlapping_horizon() -> None:
    note = multifactor_baseline.backtest_return_note(5)

    assert "5-day forward return" in note
    assert "overlapping rebalance schedules" in note
    assert "horizon-aligned portfolio simulator" in note
