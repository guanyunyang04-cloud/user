from __future__ import annotations

import pandas as pd
import pytest

from traditional_quant_research.experiments import low_corr_regime_filter


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


def test_build_market_regime_frame_summarizes_same_day_state() -> None:
    frame = pd.DataFrame(
        [
            {"date": "2026-01-02", "code": "A", "ret_1d": 0.01, "ret_5d": 0.05, "ret_20d": 0.10, "volatility_20d": 0.2, "amplitude_20d": 0.03},
            {"date": "2026-01-02", "code": "B", "ret_1d": -0.02, "ret_5d": -0.01, "ret_20d": 0.00, "volatility_20d": 0.4, "amplitude_20d": 0.05},
            {"date": "2026-01-05", "code": "A", "ret_1d": 0.03, "ret_5d": 0.02, "ret_20d": -0.02, "volatility_20d": 0.3, "amplitude_20d": 0.04},
        ]
    )

    regime = low_corr_regime_filter.build_market_regime_frame(frame)
    first = regime.loc[regime["date"] == pd.Timestamp("2026-01-02")].iloc[0]

    assert first["security_count"] == 2
    assert first["market_ret_1d_mean"] == pytest.approx(-0.005)
    assert first["market_ret_20d_mean"] == pytest.approx(0.05)
    assert first["breadth_1d_positive_rate"] == pytest.approx(0.5)
    assert first["market_volatility_20d_mean"] == pytest.approx(0.3)


def test_fixed_rebalance_regime_skips_bad_week_without_shifting_earlier() -> None:
    dates = pd.to_datetime(["2026-01-05", "2026-01-06", "2026-01-07", "2026-01-08", "2026-01-09", "2026-01-12"])
    frame = pd.DataFrame(
        [{"date": date, "code": "A", "score": float(index + 1), "open": 10.0, "close": 10.0} for index, date in enumerate(dates)]
    )
    regime = pd.DataFrame(
        [
            {"date": "2026-01-08", "market_ret_20d_mean": 0.10},
            {"date": "2026-01-09", "market_ret_20d_mean": -0.10},
            {"date": "2026-01-12", "market_ret_20d_mean": 0.10},
        ]
    )
    rules = low_corr_regime_filter.parse_regime_filter_specs("market_ret_20d_mean>=0", include_baseline=False)[0]["regime_filters"]

    output, report = low_corr_regime_filter.apply_fixed_rebalance_regime_to_signals(
        frame,
        regime,
        ["score"],
        rules,
        rebalance_frequency="weekly",
    )

    by_date = output.set_index("date")["score"]
    assert pd.isna(by_date.loc[pd.Timestamp("2026-01-08")])
    assert pd.isna(by_date.loc[pd.Timestamp("2026-01-09")])
    assert by_date.loc[pd.Timestamp("2026-01-12")] == pytest.approx(6.0)
    assert report["scheduled_rebalance_count"] == 2
    assert report["allowed_rebalance_count"] == 1
    assert report["blocked_dates"] == ["2026-01-09"]


def test_parse_regime_filter_specs_can_omit_baseline() -> None:
    configs = low_corr_regime_filter.parse_regime_filter_specs(
        "market_ret_20d_mean>=0;breadth_20d_positive_rate>=0.5",
        include_baseline=False,
    )

    assert [config["config_id"] for config in configs] == ["regime_001", "regime_002"]
    assert configs[0]["regime_filters"] == [{"column": "market_ret_20d_mean", "operator": ">=", "threshold": 0.0}]


def test_run_low_corr_regime_filter_writes_outputs_and_reports_fixed_dates(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(low_corr_regime_filter, "load_pit_manifest", lambda root=None: {"snapshot_id": "fixture", "dataset": {"date_min": "2025-12-01", "date_max": "2026-01-16"}})
    monkeypatch.setattr(low_corr_regime_filter, "load_quality_report", lambda root=None: {})
    monkeypatch.setattr(low_corr_regime_filter, "load_tradeable_panel", lambda root=None, start_date=None, end_date=None: _fixture_panel())
    observed_frames: list[pd.DataFrame] = []

    def fake_run_horizon_backtest_tables(frame, signal_columns, **kwargs):
        observed_frames.append(frame.copy())
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
                    "sharpe": 1.0 - config_index * 0.1,
                }
            ]
        )
        yearly = pd.DataFrame([{"year": 2026, "annualized_return": 0.05}])
        period = pd.DataFrame([{"period": "2026-01", "period_type": "monthly", "annualized_return": 0.05}])
        exposure = pd.DataFrame([{"period": "2026-01", "period_type": "monthly", "factor": "momentum_20d_z", "active_exposure": 0.1}])
        return summary, yearly, period, exposure

    monkeypatch.setattr(low_corr_regime_filter, "run_horizon_backtest_tables", fake_run_horizon_backtest_tables)
    regime_configs = low_corr_regime_filter.parse_regime_filter_specs("market_ret_20d_mean>=999", include_baseline=True)

    result = low_corr_regime_filter.run_low_corr_regime_filter(
        history_start_date="2025-12-01",
        fit_start_date="2025-12-01",
        fit_end_date="2025-12-31",
        start_date="2026-01-01",
        end_date="2026-01-16",
        horizon=1,
        regime_filter_configs=regime_configs,
        top_n_values=(2,),
        fee_bps_values=(0.0,),
        rebalance_frequencies=("weekly",),
        buffer_multipliers=(2.0,),
        output_dir=tmp_path,
    )

    run_dir = tmp_path / result["run_id"]
    assert result["snapshot_id"] == "fixture"
    assert result["config_count"] == 2
    assert result["candidate_count"] == 0
    assert result["low_corr_factor_columns"]
    assert (run_dir / "market_regime.csv").exists()
    assert (run_dir / "regime_summary.csv").exists()
    assert (run_dir / "regime_filter_reports.csv").exists()
    assert (run_dir / "regime_period_summary.csv").exists()
    assert (run_dir / "regime_basket_exposure.csv").exists()
    assert (run_dir / "summary.json").exists()
    assert (run_dir / "summary.md").exists()

    assert len(observed_frames) == 2
    assert len(observed_frames[0]) == len(observed_frames[1])
    assert observed_frames[1][low_corr_regime_filter.LOW_CORR_SIGNAL].isna().all()

    reports = pd.read_csv(run_dir / "regime_filter_reports.csv")
    custom = reports.loc[reports["config_id"] == "regime_001"].iloc[0]
    assert custom["allowed_rebalance_count"] == 0
    assert custom["blocked_rebalance_count"] > 0

    summary = pd.read_csv(run_dir / "regime_summary.csv")
    assert "delta_annualized_return_vs_baseline" in summary.columns
