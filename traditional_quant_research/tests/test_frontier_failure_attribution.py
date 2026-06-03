from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from traditional_quant_research.experiments import frontier_failure_attribution as attr


def _summary() -> pd.DataFrame:
    rows = []
    for impact, ret_a_2024, ret_a_2025 in [(0.0, 0.12, -0.03), (10.0, 0.10, -0.08)]:
        rows.extend(
            [
                {
                    "eval_year": 2024,
                    "signal": "signal_a",
                    "exposure_penalty_strength": 0.25,
                    "fee_bps": 30.0,
                    "capital_amount": 100_000_000.0,
                    "impact_bps_per_1pct": impact,
                    "annualized_return": ret_a_2024,
                    "max_drawdown": -0.05,
                    "periods": 4,
                    "mean_turnover": 0.8,
                    "mean_gross_return": 0.02,
                    "mean_net_return": 0.018,
                    "mean_fee_cost": 0.002,
                    "mean_impact_cost": 0.001 if impact else 0.0,
                    "mean_total_cost": 0.003 if impact else 0.002,
                    "blocked_entry_count": 1,
                    "exit_delayed_count": 2,
                },
                {
                    "eval_year": 2025,
                    "signal": "signal_a",
                    "exposure_penalty_strength": 0.25,
                    "fee_bps": 30.0,
                    "capital_amount": 100_000_000.0,
                    "impact_bps_per_1pct": impact,
                    "annualized_return": ret_a_2025,
                    "max_drawdown": -0.12,
                    "periods": 3,
                    "mean_turnover": 0.9,
                    "mean_gross_return": -0.01,
                    "mean_net_return": -0.013,
                    "mean_fee_cost": 0.002,
                    "mean_impact_cost": 0.001 if impact else 0.0,
                    "mean_total_cost": 0.003 if impact else 0.002,
                    "blocked_entry_count": 3,
                    "exit_delayed_count": 4,
                },
                {
                    "eval_year": 2024,
                    "signal": "signal_b",
                    "exposure_penalty_strength": 1.0,
                    "fee_bps": 30.0,
                    "capital_amount": 100_000_000.0,
                    "impact_bps_per_1pct": impact,
                    "annualized_return": 0.04,
                    "max_drawdown": -0.03,
                    "periods": 4,
                    "mean_turnover": 0.7,
                    "mean_gross_return": 0.01,
                    "mean_net_return": 0.008,
                    "mean_fee_cost": 0.002,
                    "mean_impact_cost": 0.001 if impact else 0.0,
                    "mean_total_cost": 0.003 if impact else 0.002,
                    "blocked_entry_count": 0,
                    "exit_delayed_count": 0,
                },
            ]
        )
    return pd.DataFrame(rows)


def _exposure() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"eval_year": 2024, "signal_config": "signal_a", "exposure_penalty_strength": 0.25, "period_type": "monthly", "factor": "log_amount_mean_20d_z", "active_exposure": -0.4},
            {"eval_year": 2024, "signal_config": "signal_a", "exposure_penalty_strength": 0.25, "period_type": "monthly", "factor": "neg_volatility_20d_z", "active_exposure": 0.2},
            {"eval_year": 2025, "signal_config": "signal_a", "exposure_penalty_strength": 0.25, "period_type": "monthly", "factor": "log_amount_mean_20d_z", "active_exposure": -0.9},
            {"eval_year": 2025, "signal_config": "signal_a", "exposure_penalty_strength": 0.25, "period_type": "monthly", "factor": "neg_volatility_20d_z", "active_exposure": 0.6},
            {"eval_year": 2024, "signal_config": "signal_b", "exposure_penalty_strength": 1.0, "period_type": "monthly", "factor": "log_amount_mean_20d_z", "active_exposure": -0.2},
        ]
    )


def _liquidity() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "eval_year": 2024,
                "signal": "signal_a",
                "exposure_penalty_strength": 0.25,
                "trade_count": 4,
                "mean_amount_mean": 10_000_000.0,
                "mean_log_amount_z": -0.4,
                "mean_momentum_z": -0.1,
                "mean_participation_p95_100m": 0.05,
                "worst_participation_max_100m": 0.20,
            },
            {
                "eval_year": 2025,
                "signal": "signal_a",
                "exposure_penalty_strength": 0.25,
                "trade_count": 3,
                "mean_amount_mean": 8_000_000.0,
                "mean_log_amount_z": -0.9,
                "mean_momentum_z": -0.3,
                "mean_participation_p95_100m": 0.08,
                "worst_participation_max_100m": 0.30,
            },
        ]
    )


def test_build_yearly_failure_attribution_joins_impact_exposure_and_liquidity() -> None:
    yearly = attr.build_yearly_failure_attribution(
        _summary(),
        _exposure(),
        _liquidity(),
        required_impact_bps=10.0,
        required_fee_bps=30.0,
        required_capital_amount=100_000_000.0,
        exposure_factors=("log_amount_mean_20d_z", "neg_volatility_20d_z"),
        period_type="monthly",
        min_year_return=0.0,
        severe_year_return=-0.10,
    )

    a2025 = yearly.loc[(yearly["signal"] == "signal_a") & (yearly["eval_year"] == 2025)].iloc[0]
    assert bool(a2025["weak_year"]) is True
    assert bool(a2025["severe_weak_year"]) is False
    assert a2025["impact_drag_vs_no_impact"] == pytest.approx(-0.05)
    assert a2025["active_log_amount_mean_20d_z"] == pytest.approx(-0.9)
    assert a2025["abs_active_neg_volatility_20d_z"] == pytest.approx(0.6)
    assert a2025["mean_participation_p95_100m"] == pytest.approx(0.08)
    assert a2025["return_rank_within_year"] == pytest.approx(1.0)


def test_signal_and_exposure_summaries_describe_weak_years() -> None:
    yearly = attr.build_yearly_failure_attribution(
        _summary(),
        _exposure(),
        _liquidity(),
        required_impact_bps=10.0,
        required_fee_bps=30.0,
        required_capital_amount=100_000_000.0,
        exposure_factors=("log_amount_mean_20d_z", "neg_volatility_20d_z"),
        period_type="monthly",
        min_year_return=0.0,
        severe_year_return=-0.10,
    )

    signal_summary = attr.summarize_signal_failure_profile(
        yearly,
        exposure_factors=("log_amount_mean_20d_z", "neg_volatility_20d_z"),
    )
    exposure_summary = attr.summarize_exposure_failure_profile(
        yearly,
        exposure_factors=("log_amount_mean_20d_z", "neg_volatility_20d_z"),
    )
    signal_a = signal_summary.loc[signal_summary["signal"] == "signal_a"].iloc[0]
    log_amount = exposure_summary.loc[
        (exposure_summary["signal"] == "signal_a") & (exposure_summary["factor"] == "log_amount_mean_20d_z")
    ].iloc[0]

    assert signal_a["weak_year_count"] == 1
    assert signal_a["weak_years"] == "2025"
    assert signal_a["positive_year_rate"] == pytest.approx(0.5)
    assert signal_a["diagnosis"] == "weak_years_and_style_exposure"
    assert log_amount["abs_delta_weak_minus_positive"] == pytest.approx(0.5)


def test_run_frontier_failure_attribution_writes_outputs(tmp_path: Path) -> None:
    combined = tmp_path / "combined"
    combined.mkdir()
    _summary().to_csv(combined / "combined_constraint_summary.csv", index=False)
    _exposure().to_csv(combined / "combined_constraint_basket_exposure.csv", index=False)
    _liquidity().to_csv(combined / "combined_constraint_liquidity_summary.csv", index=False)

    result = attr.run_frontier_failure_attribution(
        combined_run_dir=combined,
        output_dir=tmp_path / "output",
        exposure_factors=("log_amount_mean_20d_z", "neg_volatility_20d_z"),
        write_research_log=True,
        research_log_path=tmp_path / "research_log.md",
    )

    run_dir = Path(result["run_dir"])
    assert result["candidate_count"] == 0
    assert result["best_signal_by_mean_return"] == "signal_b"
    assert result["total_weak_signal_years"] == 1
    assert (run_dir / "yearly_failure_attribution.csv").exists()
    assert (run_dir / "signal_failure_summary.csv").exists()
    assert (run_dir / "exposure_failure_summary.csv").exists()
    assert (run_dir / "annualized_return_pivot.csv").exists()
    assert (run_dir / "summary.json").exists()
    assert (run_dir / "summary.md").exists()
    assert (tmp_path / "research_log.md").exists()
