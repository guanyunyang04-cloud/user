from __future__ import annotations

from pathlib import Path

import pandas as pd

from traditional_quant_research.experiments import frontier_weak_year_rebuild as rebuild


def _yearly_failure() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"eval_year": 2017, "signal": "signal_a", "exposure_penalty_strength": 0.25, "annualized_return": 0.08, "weak_year": False},
            {"eval_year": 2018, "signal": "signal_a", "exposure_penalty_strength": 0.25, "annualized_return": -0.04, "weak_year": True},
            {"eval_year": 2022, "signal": "signal_a", "exposure_penalty_strength": 0.25, "annualized_return": 0.03, "weak_year": False},
            {"eval_year": 2023, "signal": "signal_a", "exposure_penalty_strength": 0.25, "annualized_return": -0.02, "weak_year": True},
        ]
    )


def _yearly_regime() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"eval_year": 2017, "breadth_20d_positive_rate": 0.60, "market_ret_20d_mean": 0.05, "breadth_5d_positive_rate": 0.55},
            {"eval_year": 2018, "breadth_20d_positive_rate": 0.35, "market_ret_20d_mean": -0.03, "breadth_5d_positive_rate": 0.40},
            {"eval_year": 2022, "breadth_20d_positive_rate": 0.50, "market_ret_20d_mean": 0.01, "breadth_5d_positive_rate": 0.45},
            {"eval_year": 2023, "breadth_20d_positive_rate": 0.30, "market_ret_20d_mean": -0.04, "breadth_5d_positive_rate": 0.35},
        ]
    )


def test_fit_eval_regime_candidates_use_prior_years_only() -> None:
    signal_year_regime = rebuild.build_signal_year_regime(_yearly_failure(), _yearly_regime())
    fit_eval = rebuild.build_fit_eval_regime_candidates(signal_year_regime)

    assert not fit_eval.empty
    assert fit_eval["fit_uses_eval_year"].eq(False).all()
    row_2022 = fit_eval.loc[fit_eval["eval_year"] == 2022].iloc[0]
    row_2023 = fit_eval.loc[fit_eval["eval_year"] == 2023].iloc[0]
    assert row_2022["fit_years"] == "2017,2018"
    assert row_2023["fit_years"] == "2017,2018,2022"
    assert "2023" not in row_2023["fit_years"]
    assert fit_eval["evidence_grade"].unique().tolist() == ["diagnostic_not_backtest"]


def test_run_frontier_weak_year_rebuild_writes_diagnostic_outputs(tmp_path: Path) -> None:
    failure = tmp_path / "failure"
    regime = tmp_path / "regime"
    failure.mkdir()
    regime.mkdir()
    _yearly_failure().to_csv(failure / "yearly_failure_attribution.csv", index=False)
    _yearly_regime().to_csv(regime / "yearly_market_regime.csv", index=False)

    result = rebuild.run_frontier_weak_year_rebuild(
        failure_run_dir=failure,
        regime_run_dir=regime,
        output_dir=tmp_path / "output",
        write_research_log=True,
        research_log_path=tmp_path / "rebuild.md",
    )

    run_dir = Path(result["run_dir"])
    assert result["decision"] == "diagnostic_rebuild_rules_ready_for_backtest"
    assert result["candidate_count"] == 0
    assert result["fit_uses_eval_year_count"] == 0
    assert (run_dir / "signal_year_regime.csv").exists()
    assert (run_dir / "fit_eval_regime_candidates.csv").exists()
    assert (run_dir / "rebuild_backlog.csv").exists()
    assert (run_dir / "summary.json").exists()
    assert (tmp_path / "rebuild.md").exists()
