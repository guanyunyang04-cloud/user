from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from traditional_quant_research.experiments import low_corr_regime_capital_yearly_validation


def test_summarize_capital_yearly_validation_aggregates_by_plan_and_fee() -> None:
    rows = []
    for year, annualized_return, delta, capital_scale in [(2025, 0.10, 0.03, 0.8), (2026, -0.05, 0.02, 0.6)]:
        rows.append(
            {
                "eval_year": year,
                "config_id": "breadth_soft",
                "capital_plan_spec": "breadth_20d_positive_rate>=0.50@1.0|default@0.3",
                "signal": "score",
                "horizon": 5,
                "rebalance_frequency": "weekly",
                "top_n": 100,
                "fee_bps": 30.0,
                "non_overlapping": True,
                "buffer_multiplier": 2.0,
                "execution_constraints": True,
                "limit_threshold": 0.095,
                "capital_col": "capital_scale",
                "annualized_return": annualized_return,
                "sharpe": 1.0,
                "max_drawdown": -0.10,
                "mean_capital_scale": capital_scale,
                "delta_annualized_return_vs_baseline": delta,
            }
        )
    aggregate = low_corr_regime_capital_yearly_validation.summarize_capital_yearly_validation(pd.DataFrame(rows))
    row = aggregate.iloc[0]

    assert row["eval_year_count"] == 2
    assert row["mean_annualized_return"] == pytest.approx(0.025)
    assert row["min_annualized_return"] == pytest.approx(-0.05)
    assert row["positive_year_rate"] == pytest.approx(0.5)
    assert row["positive_delta_year_rate"] == pytest.approx(1.0)
    assert row["mean_capital_scale"] == pytest.approx(0.7)


def test_run_capital_yearly_validation_collects_child_outputs(tmp_path, monkeypatch) -> None:
    calls = []

    def fake_run_low_corr_regime_capital_scaling(**kwargs):
        eval_year = int(str(kwargs["start_date"])[:4])
        calls.append(kwargs)
        run_id = f"child_{eval_year}"
        child_dir = Path(kwargs["output_dir"]) / run_id
        child_dir.mkdir(parents=True, exist_ok=True)
        summary = pd.DataFrame(
            [
                {
                    "config_id": "baseline_full_capital",
                    "capital_plan_spec": "default@1.0",
                    "signal": "score",
                    "horizon": kwargs["horizon"],
                    "rebalance_frequency": kwargs["rebalance_frequencies"][0],
                    "top_n": kwargs["top_n_values"][0],
                    "fee_bps": kwargs["fee_bps_values"][0],
                    "non_overlapping": True,
                    "buffer_multiplier": kwargs["buffer_multipliers"][0],
                    "execution_constraints": kwargs["execution_constraints"],
                    "limit_threshold": kwargs["limit_threshold"],
                    "capital_col": "capital_scale",
                    "annualized_return": -0.10,
                    "sharpe": -1.0,
                    "max_drawdown": -0.10,
                    "mean_capital_scale": 1.0,
                    "delta_annualized_return_vs_baseline": 0.0,
                },
                {
                    "config_id": "breadth_soft",
                    "capital_plan_spec": "breadth_20d_positive_rate>=0.50@1.0|default@0.3",
                    "signal": "score",
                    "horizon": kwargs["horizon"],
                    "rebalance_frequency": kwargs["rebalance_frequencies"][0],
                    "top_n": kwargs["top_n_values"][0],
                    "fee_bps": kwargs["fee_bps_values"][0],
                    "non_overlapping": True,
                    "buffer_multiplier": kwargs["buffer_multipliers"][0],
                    "execution_constraints": kwargs["execution_constraints"],
                    "limit_threshold": kwargs["limit_threshold"],
                    "capital_col": "capital_scale",
                    "annualized_return": 0.05,
                    "sharpe": 0.5,
                    "max_drawdown": -0.05,
                    "mean_capital_scale": 0.6,
                    "delta_annualized_return_vs_baseline": 0.15,
                },
            ]
        )
        summary.to_csv(child_dir / "capital_scaling_summary.csv", index=False)
        pd.DataFrame([{"config_id": "breadth_soft", "scheduled_rebalance_count": 5}]).to_csv(child_dir / "capital_plan_reports.csv", index=False)
        pd.DataFrame([{"config_id": "breadth_soft", "period": f"{eval_year}Q1", "period_type": "quarterly"}]).to_csv(child_dir / "capital_scaling_period_summary.csv", index=False)
        return {
            "run_id": run_id,
            "output_dir": str(child_dir),
            "start_date": kwargs["start_date"],
            "end_date": kwargs["end_date"],
            "fit_start_date": kwargs["fit_start_date"],
            "fit_end_date": kwargs["fit_end_date"],
        }

    monkeypatch.setattr(
        low_corr_regime_capital_yearly_validation,
        "run_low_corr_regime_capital_scaling",
        fake_run_low_corr_regime_capital_scaling,
    )

    result = low_corr_regime_capital_yearly_validation.run_low_corr_regime_capital_yearly_validation(
        years=(2025, 2026),
        final_end_date="2026-06-01",
        capital_plan_specs="breadth_soft=breadth_20d_positive_rate>=0.50@1.0|default@0.3",
        top_n_values=(100,),
        fee_bps_values=(30.0,),
        rebalance_frequencies=("weekly",),
        buffer_multipliers=(2.0,),
        output_dir=tmp_path,
    )

    run_dir = tmp_path / result["run_id"]
    assert [call["fit_start_date"] for call in calls] == ["2024-01-01", "2025-01-01"]
    assert calls[-1]["end_date"] == "2026-06-01"
    assert result["years"] == [2025, 2026]
    assert result["config_count"] == 2
    assert (run_dir / "capital_yearly_validation_summary.csv").exists()
    assert (run_dir / "capital_yearly_validation_reports.csv").exists()
    assert (run_dir / "capital_yearly_validation_period_summary.csv").exists()
    assert (run_dir / "capital_yearly_validation_aggregate.csv").exists()
    assert (run_dir / "summary.json").exists()
    assert (run_dir / "summary.md").exists()

    aggregate = pd.read_csv(run_dir / "capital_yearly_validation_aggregate.csv")
    custom = aggregate.loc[aggregate["config_id"] == "breadth_soft"].iloc[0]
    assert custom["eval_year_count"] == 2
    assert custom["mean_annualized_return"] == pytest.approx(0.05)
    assert custom["positive_delta_year_rate"] == pytest.approx(1.0)
