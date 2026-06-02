from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from traditional_quant_research.experiments import low_corr_horizon_cost_yearly_validation


def test_summarize_horizon_cost_yearly_validation_aggregates_by_protocol() -> None:
    rows = []
    for eval_year, annualized_return in [(2025, 0.10), (2026, -0.02)]:
        rows.append(
            {
                "eval_year": eval_year,
                "config_id": "baseline_no_filter",
                "filter_spec": "",
                "signal": "score",
                "horizon": 20,
                "rebalance_frequency": "monthly",
                "top_n": 200,
                "fee_bps": 30.0,
                "non_overlapping": True,
                "buffer_multiplier": 3.0,
                "execution_constraints": True,
                "limit_threshold": 0.095,
                "annualized_return": annualized_return,
                "sharpe": 0.5,
                "max_drawdown": -0.06,
                "mean_turnover": 0.8,
                "cost_drag_vs_0bps": -0.03,
            }
        )
    aggregate = low_corr_horizon_cost_yearly_validation.summarize_horizon_cost_yearly_validation(pd.DataFrame(rows))
    row = aggregate.iloc[0]

    assert row["eval_year_count"] == 2
    assert row["mean_annualized_return"] == pytest.approx(0.04)
    assert row["min_annualized_return"] == pytest.approx(-0.02)
    assert row["positive_year_rate"] == pytest.approx(0.5)
    assert row["mean_cost_drag_vs_0bps"] == pytest.approx(-0.03)


def test_run_horizon_cost_yearly_validation_collects_child_outputs(tmp_path, monkeypatch) -> None:
    calls = []

    def fake_run_low_corr_horizon_cost_grid(**kwargs):
        eval_year = int(str(kwargs["start_date"])[:4])
        calls.append(kwargs)
        run_id = f"child_{eval_year}"
        child_dir = Path(kwargs["output_dir"]) / run_id
        child_dir.mkdir(parents=True, exist_ok=True)
        summary = pd.DataFrame(
            [
                {
                    "config_id": "baseline_no_filter",
                    "filter_spec": "",
                    "signal": "score",
                    "horizon": kwargs["horizons"][0],
                    "rebalance_frequency": kwargs["rebalance_frequencies"][0],
                    "top_n": kwargs["top_n_values"][0],
                    "fee_bps": 30.0,
                    "non_overlapping": True,
                    "buffer_multiplier": kwargs["buffer_multipliers"][0],
                    "execution_constraints": kwargs["execution_constraints"],
                    "limit_threshold": kwargs["limit_threshold"],
                    "annualized_return": 0.04,
                    "sharpe": 0.5,
                    "max_drawdown": -0.05,
                    "mean_turnover": 0.8,
                    "cost_drag_vs_0bps": -0.03,
                }
            ]
        )
        summary.to_csv(child_dir / "horizon_cost_summary.csv", index=False)
        pd.DataFrame([{"year": eval_year, "annualized_return": 0.04}]).to_csv(child_dir / "horizon_cost_yearly_summary.csv", index=False)
        pd.DataFrame([{"period": f"{eval_year}Q1", "period_type": "quarterly"}]).to_csv(child_dir / "horizon_cost_period_summary.csv", index=False)
        return {
            "run_id": run_id,
            "output_dir": str(child_dir),
            "horizons": list(kwargs["horizons"]),
            "child_runs": [
                {
                    "start_date": kwargs["start_date"],
                    "end_date": kwargs["end_date"],
                }
            ],
        }

    monkeypatch.setattr(
        low_corr_horizon_cost_yearly_validation,
        "run_low_corr_horizon_cost_grid",
        fake_run_low_corr_horizon_cost_grid,
    )

    result = low_corr_horizon_cost_yearly_validation.run_low_corr_horizon_cost_yearly_validation(
        years=(2025, 2026),
        final_end_date="2026-06-01",
        horizons=(20,),
        top_n_values=(200,),
        fee_bps_values=(30.0,),
        rebalance_frequencies=("monthly",),
        buffer_multipliers=(3.0,),
        output_dir=tmp_path,
    )

    run_dir = tmp_path / result["run_id"]
    assert [call["fit_start_date"] for call in calls] == ["2024-01-01", "2025-01-01"]
    assert calls[-1]["end_date"] == "2026-06-01"
    assert result["years"] == [2025, 2026]
    assert (run_dir / "horizon_cost_yearly_validation_summary.csv").exists()
    assert (run_dir / "horizon_cost_yearly_validation_child_yearly.csv").exists()
    assert (run_dir / "horizon_cost_yearly_validation_period_summary.csv").exists()
    assert (run_dir / "horizon_cost_yearly_validation_aggregate.csv").exists()
    assert (run_dir / "summary.json").exists()
    assert (run_dir / "summary.md").exists()

    aggregate = pd.read_csv(run_dir / "horizon_cost_yearly_validation_aggregate.csv")
    assert aggregate.iloc[0]["eval_year_count"] == 2
    assert result["best_30bps_rows"][0]["mean_annualized_return"] == pytest.approx(0.04)
