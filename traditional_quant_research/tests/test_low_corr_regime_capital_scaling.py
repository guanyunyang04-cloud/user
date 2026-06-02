from __future__ import annotations

import pandas as pd
import pytest

from traditional_quant_research.experiments import low_corr_regime_capital_scaling


def _fixture_panel() -> pd.DataFrame:
    rows = []
    dates = pd.date_range("2025-12-01", periods=35, freq="B")
    base_closes = {
        "600000.SH": 10.0,
        "000001.SZ": 20.0,
        "000002.SZ": 8.0,
        "600004.SH": 15.0,
        "600005.SH": 6.0,
    }
    for code_index, (code, base_close) in enumerate(base_closes.items()):
        for index, date in enumerate(dates):
            trend = 0.03 * index * (1 + code_index * 0.1)
            cycle = ((index + code_index) % 5 - 2) * 0.02
            close = base_close + trend + cycle
            rows.append(
                {
                    "date": date,
                    "code": code,
                    "open": close - 0.03,
                    "high": close + 0.15 + code_index * 0.01,
                    "low": close - 0.12,
                    "close": close,
                    "volume": 1000.0 + index * 10 + code_index * 100,
                    "amount": close * (1000.0 + index * 10 + code_index * 100),
                    "is_tradeable": True,
                }
            )
    return pd.DataFrame(rows)


def test_parse_capital_plan_specs_includes_full_capital_baseline() -> None:
    plans = low_corr_regime_capital_scaling.parse_capital_plan_specs(
        "soft=breadth_20d_positive_rate>=0.5@1.0|default@0.3",
        include_baseline=True,
    )

    assert [plan["config_id"] for plan in plans] == ["baseline_full_capital", "soft"]
    assert plans[0]["levels"] == [{"rules": [], "scale": 1.0, "label": "default"}]
    assert plans[1]["levels"][0]["rules"] == [
        {"column": "breadth_20d_positive_rate", "operator": ">=", "threshold": 0.5}
    ]
    assert plans[1]["levels"][1]["scale"] == pytest.approx(0.3)


def test_parse_capital_plan_specs_rejects_out_of_range_scale() -> None:
    with pytest.raises(ValueError, match="capital scale must be between 0 and 1"):
        low_corr_regime_capital_scaling.parse_capital_plan_specs("bad=default@1.1", include_baseline=False)


def test_capital_scale_by_date_applies_ordered_levels_and_default() -> None:
    regime = pd.DataFrame(
        [
            {"date": "2026-01-02", "breadth_20d_positive_rate": 0.55},
            {"date": "2026-01-05", "breadth_20d_positive_rate": 0.47},
            {"date": "2026-01-06", "breadth_20d_positive_rate": 0.30},
        ]
    )
    plan = low_corr_regime_capital_scaling.parse_capital_plan_specs(
        "soft=breadth_20d_positive_rate>=0.50@1.0|breadth_20d_positive_rate>=0.45@0.6|default@0.3",
        include_baseline=False,
    )[0]

    output = low_corr_regime_capital_scaling.capital_scale_by_date(regime, plan["levels"])

    assert output["capital_scale"].tolist() == pytest.approx([1.0, 0.6, 0.3])
    assert output["capital_rule"].tolist() == [
        "breadth_20d_positive_rate>=0.50",
        "breadth_20d_positive_rate>=0.45",
        "default",
    ]


def test_capital_scale_by_date_requires_full_assignment() -> None:
    regime = pd.DataFrame([{"date": "2026-01-02", "breadth_20d_positive_rate": 0.40}])
    plan = low_corr_regime_capital_scaling.parse_capital_plan_specs(
        "soft=breadth_20d_positive_rate>=0.50@1.0",
        include_baseline=False,
    )[0]

    with pytest.raises(ValueError, match="did not assign all dates"):
        low_corr_regime_capital_scaling.capital_scale_by_date(regime, plan["levels"])


def test_attach_capital_scale_requires_every_date() -> None:
    frame = pd.DataFrame(
        [
            {"date": "2026-01-02", "code": "A", "score": 1.0},
            {"date": "2026-01-05", "code": "A", "score": 2.0},
        ]
    )
    capital = pd.DataFrame([{"date": "2026-01-02", "capital_scale": 0.5, "capital_rule": "default"}])

    with pytest.raises(ValueError, match="capital scale missing"):
        low_corr_regime_capital_scaling.attach_capital_scale(frame, capital)


def test_capital_plan_report_uses_scheduled_rebalance_dates() -> None:
    frame = pd.DataFrame(
        [
            {"date": "2026-01-05", "code": "A", "capital_scale": 1.0},
            {"date": "2026-01-06", "code": "A", "capital_scale": 0.6},
            {"date": "2026-01-09", "code": "A", "capital_scale": 0.3},
            {"date": "2026-01-12", "code": "A", "capital_scale": 0.8},
        ]
    )

    report = low_corr_regime_capital_scaling.capital_plan_report(
        frame,
        config_id="soft",
        plan_spec="demo",
        rebalance_frequency="weekly",
    )

    assert report["scheduled_rebalance_count"] == 2
    assert report["mean_scheduled_capital_scale"] == pytest.approx(0.55)
    assert report["reduced_capital_date_count"] == 2


def test_run_low_corr_regime_capital_scaling_writes_outputs_and_passes_capital_col(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(
        low_corr_regime_capital_scaling,
        "load_pit_manifest",
        lambda root=None: {"snapshot_id": "fixture", "dataset": {"date_min": "2025-12-01", "date_max": "2026-01-16"}},
    )
    monkeypatch.setattr(low_corr_regime_capital_scaling, "load_quality_report", lambda root=None: {})
    monkeypatch.setattr(low_corr_regime_capital_scaling, "load_tradeable_panel", lambda root=None, start_date=None, end_date=None: _fixture_panel())
    observed_frames: list[pd.DataFrame] = []
    observed_capital_cols: list[str | None] = []

    def fake_build_market_regime_frame(frame):
        dates = pd.to_datetime(pd.Series(frame["date"].unique())).sort_values().reset_index(drop=True)
        return pd.DataFrame(
            {
                "date": dates,
                "breadth_20d_positive_rate": [0.60 if index % 2 == 0 else 0.40 for index in range(len(dates))],
            }
        )

    def fake_run_horizon_backtest_tables(frame, signal_columns, **kwargs):
        observed_frames.append(frame.copy())
        observed_capital_cols.append(kwargs.get("capital_col"))
        config_index = len(observed_frames)
        summary = pd.DataFrame(
            [
                {
                    "signal": signal_columns[0],
                    "horizon": kwargs["horizon"],
                    "top_n": kwargs["top_n_values"][0],
                    "fee_bps": kwargs["fee_bps_values"][0],
                    "non_overlapping": True,
                    "rebalance_frequency": kwargs["rebalance_frequencies"][0],
                    "buffer_multiplier": kwargs["buffer_multipliers"][0],
                    "execution_constraints": kwargs["execution_constraints"],
                    "limit_threshold": kwargs["limit_threshold"],
                    "annualized_return": 0.10 - config_index * 0.01,
                    "max_drawdown": -0.02 - config_index * 0.01,
                    "mean_turnover": 1.0 + config_index * 0.1,
                    "mean_capital_scale": frame["capital_scale"].mean(),
                    "sharpe": 1.0 - config_index * 0.1,
                }
            ]
        )
        yearly = pd.DataFrame([{"year": 2026, "annualized_return": 0.05, "mean_capital_scale": frame["capital_scale"].mean()}])
        period = pd.DataFrame([{"period": "2026-01", "period_type": "monthly", "annualized_return": 0.05}])
        exposure = pd.DataFrame([{"period": "2026-01", "period_type": "monthly", "factor": "momentum_20d_z", "active_exposure": 0.1}])
        return summary, yearly, period, exposure

    monkeypatch.setattr(low_corr_regime_capital_scaling, "run_horizon_backtest_tables", fake_run_horizon_backtest_tables)
    monkeypatch.setattr(low_corr_regime_capital_scaling, "build_market_regime_frame", fake_build_market_regime_frame)
    plans = low_corr_regime_capital_scaling.parse_capital_plan_specs(
        "soft=breadth_20d_positive_rate>=0.50@1.0|default@0.3",
        include_baseline=True,
    )

    result = low_corr_regime_capital_scaling.run_low_corr_regime_capital_scaling(
        history_start_date="2025-12-01",
        fit_start_date="2025-12-01",
        fit_end_date="2025-12-31",
        start_date="2026-01-01",
        end_date="2026-01-16",
        horizon=1,
        capital_plan_configs=plans,
        top_n_values=(2,),
        fee_bps_values=(0.0,),
        rebalance_frequencies=("weekly",),
        buffer_multipliers=(2.0,),
        output_dir=tmp_path,
    )

    run_dir = tmp_path / result["run_id"]
    assert result["snapshot_id"] == "fixture"
    assert result["capital_plan_count"] == 2
    assert result["candidate_count"] == 0
    assert result["low_corr_factor_columns"]
    assert observed_capital_cols == ["capital_scale", "capital_scale"]
    assert all("capital_scale" in frame.columns for frame in observed_frames)
    assert observed_frames[0]["capital_scale"].eq(1.0).all()
    assert observed_frames[1]["capital_scale"].lt(1.0).any()
    assert (run_dir / "capital_scaling_summary.csv").exists()
    assert (run_dir / "capital_plan_reports.csv").exists()
    assert (run_dir / "capital_scaling_period_summary.csv").exists()
    assert (run_dir / "capital_scaling_basket_exposure.csv").exists()
    assert (run_dir / "summary.json").exists()
    assert (run_dir / "summary.md").exists()

    summary = pd.read_csv(run_dir / "capital_scaling_summary.csv")
    assert "delta_annualized_return_vs_baseline" in summary.columns
