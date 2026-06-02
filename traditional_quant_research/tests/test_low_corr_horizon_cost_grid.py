from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from traditional_quant_research.experiments import low_corr_horizon_cost_grid


def test_summarize_horizon_cost_grid_attaches_zero_fee_cost_drag() -> None:
    summary = pd.DataFrame(
        [
            {
                "config_id": "custom_001",
                "filter_spec": "log_amount_mean_20d_z>=-0.8",
                "signal": "score",
                "horizon": 10,
                "rebalance_frequency": "monthly",
                "top_n": 100,
                "fee_bps": 0.0,
                "non_overlapping": True,
                "buffer_multiplier": 3.0,
                "execution_constraints": True,
                "limit_threshold": 0.095,
                "annualized_return": 0.12,
                "sharpe": 1.2,
                "max_drawdown": -0.04,
                "mean_turnover": 0.40,
            },
            {
                "config_id": "custom_001",
                "filter_spec": "log_amount_mean_20d_z>=-0.8",
                "signal": "score",
                "horizon": 10,
                "rebalance_frequency": "monthly",
                "top_n": 100,
                "fee_bps": 30.0,
                "non_overlapping": True,
                "buffer_multiplier": 3.0,
                "execution_constraints": True,
                "limit_threshold": 0.095,
                "annualized_return": 0.08,
                "sharpe": 0.8,
                "max_drawdown": -0.05,
                "mean_turnover": 0.40,
            },
        ]
    )

    output = low_corr_horizon_cost_grid.summarize_horizon_cost_grid(summary)
    fee30 = output.loc[output["fee_bps"] == 30.0].iloc[0]

    assert fee30["annualized_return_0bps"] == pytest.approx(0.12)
    assert fee30["cost_drag_vs_0bps"] == pytest.approx(-0.04)
    assert fee30["turnover_adjusted_return"] == pytest.approx(0.20)


def test_run_low_corr_horizon_cost_grid_collects_child_outputs(tmp_path, monkeypatch) -> None:
    calls = []

    def fake_run_low_corr_exposure_grid(**kwargs):
        horizon = int(kwargs["horizon"])
        calls.append(kwargs)
        run_id = f"child_h{horizon}"
        child_dir = Path(kwargs["output_dir"]) / run_id
        child_dir.mkdir(parents=True, exist_ok=True)
        summary = pd.DataFrame(
            [
                {
                    "config_id": "baseline_no_filter",
                    "filter_spec": "",
                    "signal": "score",
                    "horizon": horizon,
                    "rebalance_frequency": kwargs["rebalance_frequencies"][0],
                    "top_n": kwargs["top_n_values"][0],
                    "fee_bps": 0.0,
                    "non_overlapping": True,
                    "buffer_multiplier": kwargs["buffer_multipliers"][0],
                    "execution_constraints": kwargs["execution_constraints"],
                    "limit_threshold": kwargs["limit_threshold"],
                    "annualized_return": 0.02,
                    "sharpe": 0.2,
                    "max_drawdown": -0.08,
                    "mean_turnover": 0.7,
                },
                {
                    "config_id": "custom_001",
                    "filter_spec": "log_amount_mean_20d_z>=-0.8,momentum_20d_z>=-0.8",
                    "signal": "score",
                    "horizon": horizon,
                    "rebalance_frequency": kwargs["rebalance_frequencies"][0],
                    "top_n": kwargs["top_n_values"][0],
                    "fee_bps": 30.0,
                    "non_overlapping": True,
                    "buffer_multiplier": kwargs["buffer_multipliers"][0],
                    "execution_constraints": kwargs["execution_constraints"],
                    "limit_threshold": kwargs["limit_threshold"],
                    "annualized_return": 0.04 + horizon / 1000.0,
                    "sharpe": 0.4,
                    "max_drawdown": -0.05,
                    "mean_turnover": 0.5,
                },
            ]
        )
        summary.to_csv(child_dir / "grid_summary.csv", index=False)
        pd.DataFrame([{"year": 2026, "horizon": horizon, "annualized_return": 0.04}]).to_csv(child_dir / "grid_yearly_summary.csv", index=False)
        pd.DataFrame([{"period": "2026-01", "period_type": "monthly", "horizon": horizon}]).to_csv(child_dir / "grid_period_summary.csv", index=False)
        pd.DataFrame([{"config_id": "custom_001", "horizon": horizon, "kept_rate": 0.7}]).to_csv(child_dir / "grid_selection_reports.csv", index=False)
        return {
            "run_id": run_id,
            "output_dir": str(child_dir),
            "start_date": kwargs["start_date"],
            "end_date": kwargs["end_date"],
            "fit_start_date": kwargs["fit_start_date"],
            "fit_end_date": kwargs["fit_end_date"],
            "low_corr_factor_columns": ["momentum_20d_z"],
        }

    monkeypatch.setattr(low_corr_horizon_cost_grid, "run_low_corr_exposure_grid", fake_run_low_corr_exposure_grid)

    result = low_corr_horizon_cost_grid.run_low_corr_horizon_cost_grid(
        history_start_date="2025-01-01",
        fit_start_date="2025-01-01",
        fit_end_date="2025-12-31",
        start_date="2026-01-01",
        end_date="2026-06-01",
        horizons=(5, 10),
        filter_specs="log_amount_mean_20d_z>=-0.8,momentum_20d_z>=-0.8",
        top_n_values=(100,),
        fee_bps_values=(0.0, 30.0),
        rebalance_frequencies=("monthly",),
        buffer_multipliers=(3.0,),
        output_dir=tmp_path,
    )

    run_dir = tmp_path / result["run_id"]
    assert [call["horizon"] for call in calls] == [5, 10]
    assert result["horizons"] == [5, 10]
    assert result["config_count"] == 2
    assert (run_dir / "horizon_cost_raw_summary.csv").exists()
    assert (run_dir / "horizon_cost_summary.csv").exists()
    assert (run_dir / "horizon_cost_yearly_summary.csv").exists()
    assert (run_dir / "horizon_cost_period_summary.csv").exists()
    assert (run_dir / "horizon_cost_selection_reports.csv").exists()
    assert (run_dir / "summary.json").exists()
    assert (run_dir / "summary.md").exists()

    cost_summary = pd.read_csv(run_dir / "horizon_cost_summary.csv")
    assert sorted(cost_summary["grid_horizon"].unique().tolist()) == [5, 10]
    assert result["best_30bps_rows"][0]["horizon"] == 10
