from __future__ import annotations

import pandas as pd
import pytest

from traditional_quant_research.experiments import low_corr_exposure_grid


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


def test_build_exposure_filter_grid_uses_paired_defensive_caps() -> None:
    configs = low_corr_exposure_grid.build_exposure_filter_grid(
        liquidity_thresholds=(-1.0, -0.8),
        momentum_thresholds=(-0.6,),
        volatility_caps=(0.8, 1.0),
        amplitude_caps=(1.0, 1.2),
        defensive_grid_mode="paired",
        include_baseline=True,
    )

    assert len(configs) == 5
    assert configs[0]["config_id"] == "baseline_no_filter"
    specs = [config["filter_spec"] for config in configs[1:]]
    assert "log_amount_mean_20d_z>=-1,momentum_20d_z>=-0.6,neg_volatility_20d_z<=0.8,neg_amplitude_20d_z<=1" in specs
    assert "log_amount_mean_20d_z>=-0.8,momentum_20d_z>=-0.6,neg_volatility_20d_z<=1,neg_amplitude_20d_z<=1.2" in specs


def test_parse_filter_specs_can_omit_baseline() -> None:
    configs = low_corr_exposure_grid.parse_filter_specs(
        "log_amount_mean_20d_z>=-0.8;momentum_20d_z>=-1.0",
        include_baseline=False,
    )

    assert [config["config_id"] for config in configs] == ["custom_001", "custom_002"]
    assert configs[0]["selection_filters"] == [{"column": "log_amount_mean_20d_z", "operator": ">=", "threshold": -0.8}]


def test_add_baseline_delta_columns_compares_same_protocol_rows() -> None:
    summary = pd.DataFrame(
        [
            {
                "config_id": "baseline_no_filter",
                "signal": low_corr_exposure_grid.LOW_CORR_SIGNAL,
                "horizon": 5,
                "rebalance_frequency": "weekly",
                "top_n": 100,
                "fee_bps": 30.0,
                "non_overlapping": True,
                "buffer_multiplier": 2.0,
                "execution_constraints": True,
                "limit_threshold": 0.095,
                "annualized_return": -0.30,
                "sharpe": -2.0,
                "max_drawdown": -0.10,
                "mean_turnover": 1.20,
            },
            {
                "config_id": "custom_001",
                "signal": low_corr_exposure_grid.LOW_CORR_SIGNAL,
                "horizon": 5,
                "rebalance_frequency": "weekly",
                "top_n": 100,
                "fee_bps": 30.0,
                "non_overlapping": True,
                "buffer_multiplier": 2.0,
                "execution_constraints": True,
                "limit_threshold": 0.095,
                "annualized_return": -0.20,
                "sharpe": -1.5,
                "max_drawdown": -0.08,
                "mean_turnover": 1.35,
            },
        ]
    )

    output = low_corr_exposure_grid.add_baseline_delta_columns(summary)
    custom = output.loc[output["config_id"] == "custom_001"].iloc[0]

    assert custom["baseline_annualized_return"] == pytest.approx(-0.30)
    assert custom["delta_annualized_return_vs_baseline"] == pytest.approx(0.10)
    assert custom["delta_sharpe_vs_baseline"] == pytest.approx(0.50)
    assert custom["delta_max_drawdown_vs_baseline"] == pytest.approx(0.02)
    assert custom["delta_mean_turnover_vs_baseline"] == pytest.approx(0.15)


def test_run_low_corr_exposure_grid_writes_outputs_and_keeps_price_path(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(low_corr_exposure_grid, "load_pit_manifest", lambda root=None: {"snapshot_id": "fixture", "dataset": {"date_min": "2025-12-01", "date_max": "2026-01-16"}})
    monkeypatch.setattr(low_corr_exposure_grid, "load_quality_report", lambda root=None: {})
    monkeypatch.setattr(low_corr_exposure_grid, "load_tradeable_panel", lambda root=None, start_date=None, end_date=None: _fixture_panel())
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

    monkeypatch.setattr(low_corr_exposure_grid, "run_horizon_backtest_tables", fake_run_horizon_backtest_tables)
    filter_configs = low_corr_exposure_grid.parse_filter_specs("log_amount_mean_20d_z>=999", include_baseline=True)

    result = low_corr_exposure_grid.run_low_corr_exposure_grid(
        history_start_date="2025-12-01",
        fit_start_date="2025-12-01",
        fit_end_date="2025-12-31",
        start_date="2026-01-01",
        end_date="2026-01-16",
        horizon=1,
        filter_configs=filter_configs,
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
    assert (run_dir / "grid_summary.csv").exists()
    assert (run_dir / "grid_selection_reports.csv").exists()
    assert (run_dir / "grid_period_summary.csv").exists()
    assert (run_dir / "grid_basket_exposure.csv").exists()
    assert (run_dir / "summary.json").exists()
    assert (run_dir / "summary.md").exists()

    assert len(observed_frames) == 2
    assert len(observed_frames[0]) == len(observed_frames[1])
    assert observed_frames[1]["open"].notna().any()
    assert observed_frames[1][low_corr_exposure_grid.LOW_CORR_SIGNAL].isna().all()

    reports = pd.read_csv(run_dir / "grid_selection_reports.csv")
    custom = reports.loc[reports["config_id"] == "custom_001"].iloc[0]
    assert custom["rows_after"] == 0
    assert custom["kept_rate"] == pytest.approx(0.0)

    grid_summary = pd.read_csv(run_dir / "grid_summary.csv")
    assert "delta_annualized_return_vs_baseline" in grid_summary.columns
