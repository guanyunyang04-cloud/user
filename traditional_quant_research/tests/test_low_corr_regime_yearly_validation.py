from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from traditional_quant_research.experiments import low_corr_regime_yearly_validation


def test_build_regime_threshold_specs_crosses_thresholds() -> None:
    specs = low_corr_regime_yearly_validation.build_regime_threshold_specs(
        market_ret_thresholds=(-0.02, -0.01),
        breadth_thresholds=(0.45, 0.50),
    )

    assert specs.split(";") == [
        "market_ret_20d_mean>=-0.02,breadth_20d_positive_rate>=0.45",
        "market_ret_20d_mean>=-0.02,breadth_20d_positive_rate>=0.5",
        "market_ret_20d_mean>=-0.01,breadth_20d_positive_rate>=0.45",
        "market_ret_20d_mean>=-0.01,breadth_20d_positive_rate>=0.5",
    ]


def test_year_windows_use_prior_year_fit_and_final_end_date() -> None:
    assert low_corr_regime_yearly_validation.year_windows(2025, final_end_date="2026-06-01") == {
        "history_start_date": "2024-01-01",
        "fit_start_date": "2024-01-01",
        "fit_end_date": "2024-12-31",
        "start_date": "2025-01-01",
        "end_date": "2025-12-31",
    }
    assert low_corr_regime_yearly_validation.year_windows(2026, final_end_date="2026-06-01")["end_date"] == "2026-06-01"
    with pytest.raises(ValueError, match="after final_end_date"):
        low_corr_regime_yearly_validation.year_windows(2027, final_end_date="2026-06-01")


def test_summarize_yearly_validation_aggregates_by_config_and_fee() -> None:
    rows = []
    for year, annualized_return, delta in [(2025, 0.10, 0.03), (2026, -0.05, 0.02)]:
        rows.append(
            {
                "eval_year": year,
                "config_id": "regime_001",
                "regime_filter_spec": "market_ret_20d_mean>=-0.02,breadth_20d_positive_rate>=0.45",
                "signal": "score",
                "horizon": 5,
                "rebalance_frequency": "weekly",
                "top_n": 100,
                "fee_bps": 30.0,
                "buffer_multiplier": 2.0,
                "execution_constraints": True,
                "limit_threshold": 0.095,
                "annualized_return": annualized_return,
                "sharpe": 1.0,
                "max_drawdown": -0.10,
                "delta_annualized_return_vs_baseline": delta,
            }
        )
    aggregate = low_corr_regime_yearly_validation.summarize_yearly_validation(pd.DataFrame(rows))
    row = aggregate.iloc[0]

    assert row["eval_year_count"] == 2
    assert row["mean_annualized_return"] == pytest.approx(0.025)
    assert row["min_annualized_return"] == pytest.approx(-0.05)
    assert row["positive_year_rate"] == pytest.approx(0.5)
    assert row["positive_delta_year_rate"] == pytest.approx(1.0)


def test_run_yearly_validation_collects_child_outputs(tmp_path, monkeypatch) -> None:
    calls = []

    def fake_run_low_corr_regime_filter(**kwargs):
        eval_year = int(str(kwargs["start_date"])[:4])
        calls.append(kwargs)
        run_id = f"child_{eval_year}"
        child_dir = Path(kwargs["output_dir"]) / run_id
        child_dir.mkdir(parents=True, exist_ok=True)
        summary = pd.DataFrame(
            [
                {
                    "config_id": "baseline_no_regime",
                    "regime_filter_spec": "",
                    "signal": "score",
                    "horizon": kwargs["horizon"],
                    "rebalance_frequency": kwargs["rebalance_frequencies"][0],
                    "top_n": kwargs["top_n_values"][0],
                    "fee_bps": kwargs["fee_bps_values"][0],
                    "buffer_multiplier": kwargs["buffer_multipliers"][0],
                    "execution_constraints": kwargs["execution_constraints"],
                    "limit_threshold": kwargs["limit_threshold"],
                    "annualized_return": -0.10,
                    "sharpe": -1.0,
                    "max_drawdown": -0.10,
                    "delta_annualized_return_vs_baseline": 0.0,
                },
                {
                    "config_id": "regime_001",
                    "regime_filter_spec": "market_ret_20d_mean>=-0.02,breadth_20d_positive_rate>=0.45",
                    "signal": "score",
                    "horizon": kwargs["horizon"],
                    "rebalance_frequency": kwargs["rebalance_frequencies"][0],
                    "top_n": kwargs["top_n_values"][0],
                    "fee_bps": kwargs["fee_bps_values"][0],
                    "buffer_multiplier": kwargs["buffer_multipliers"][0],
                    "execution_constraints": kwargs["execution_constraints"],
                    "limit_threshold": kwargs["limit_threshold"],
                    "annualized_return": 0.05,
                    "sharpe": 0.5,
                    "max_drawdown": -0.05,
                    "delta_annualized_return_vs_baseline": 0.15,
                },
            ]
        )
        summary.to_csv(child_dir / "regime_summary.csv", index=False)
        pd.DataFrame([{"config_id": "regime_001", "allowed_rebalance_count": 5}]).to_csv(child_dir / "regime_filter_reports.csv", index=False)
        pd.DataFrame([{"config_id": "regime_001", "period": f"{eval_year}Q1", "period_type": "quarterly"}]).to_csv(child_dir / "regime_period_summary.csv", index=False)
        return {
            "run_id": run_id,
            "output_dir": str(child_dir),
            "start_date": kwargs["start_date"],
            "end_date": kwargs["end_date"],
            "fit_start_date": kwargs["fit_start_date"],
            "fit_end_date": kwargs["fit_end_date"],
        }

    monkeypatch.setattr(low_corr_regime_yearly_validation, "run_low_corr_regime_filter", fake_run_low_corr_regime_filter)

    result = low_corr_regime_yearly_validation.run_low_corr_regime_yearly_validation(
        years=(2025, 2026),
        final_end_date="2026-06-01",
        regime_filter_specs="market_ret_20d_mean>=-0.02,breadth_20d_positive_rate>=0.45",
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
    assert (run_dir / "yearly_validation_summary.csv").exists()
    assert (run_dir / "yearly_validation_regime_reports.csv").exists()
    assert (run_dir / "yearly_validation_period_summary.csv").exists()
    assert (run_dir / "yearly_validation_aggregate.csv").exists()
    assert (run_dir / "summary.json").exists()
    assert (run_dir / "summary.md").exists()

    aggregate = pd.read_csv(run_dir / "yearly_validation_aggregate.csv")
    regime = aggregate.loc[aggregate["config_id"] == "regime_001"].iloc[0]
    assert regime["eval_year_count"] == 2
    assert regime["mean_annualized_return"] == pytest.approx(0.05)
    assert regime["positive_delta_year_rate"] == pytest.approx(1.0)
