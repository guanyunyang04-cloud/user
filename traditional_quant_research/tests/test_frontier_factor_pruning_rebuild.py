from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from traditional_quant_research.experiments.frontier_factor_pruning_rebuild import (
    build_prior_fit_factor_pruning_plan,
    run_frontier_factor_pruning_rebuild,
)


def _yearly_ic() -> pd.DataFrame:
    rows = []
    values = {
        2016: {
            "factor_a_z": 0.020,
            "factor_b_z": -0.018,
            "factor_c_z": 0.001,
            "factor_d_z": 0.012,
            "baseline_score": 0.030,
        },
        2017: {
            "factor_a_z": 0.018,
            "factor_b_z": -0.020,
            "factor_c_z": -0.001,
            "factor_d_z": -0.012,
            "baseline_score": 0.010,
        },
        2018: {
            "factor_a_z": 0.022,
            "factor_b_z": -0.017,
            "factor_c_z": 0.001,
            "factor_d_z": 0.002,
            "baseline_score": -0.010,
        },
        2019: {
            "factor_a_z": 0.015,
            "factor_b_z": -0.010,
            "factor_c_z": 0.004,
            "factor_d_z": -0.003,
            "baseline_score": 0.000,
        },
    }
    for year, factor_values in values.items():
        for signal, mean_rank_ic in factor_values.items():
            rows.append(
                {
                    "horizon": 20,
                    "year": year,
                    "signal": signal,
                    "label": "fwd_ret_20d",
                    "dates": 100,
                    "observations": 1000,
                    "mean_rank_ic": mean_rank_ic,
                    "rank_icir": mean_rank_ic * 100.0,
                }
            )
    return pd.DataFrame(rows)


def test_factor_pruning_uses_prior_years_only_and_sets_direction() -> None:
    detail, plan = build_prior_fit_factor_pruning_plan(
        _yearly_ic(),
        horizon=20,
        max_prior_years=2,
        min_prior_years=1,
        min_abs_mean_rank_ic=0.005,
        min_sign_consistency=0.67,
        min_keep_factors=3,
        max_keep_factors=4,
    )

    row = plan.loc[plan["eval_year"].eq(2018)].iloc[0]
    directions = json.loads(row["selected_factor_directions"])
    assert row["fit_years"] == "2016,2017"
    assert bool(row["fit_uses_eval_year"]) is False
    assert "baseline_score" not in row["selected_factors"]
    assert "factor_a_z" in row["selected_factors"]
    assert directions["factor_b_z"] == -1
    assert bool(detail.loc[detail["eval_year"].eq(2018) & detail["signal"].eq("factor_b_z"), "fit_uses_eval_year"].iloc[0]) is False


def test_factor_pruning_fallback_preserves_min_keep_count() -> None:
    _, plan = build_prior_fit_factor_pruning_plan(
        _yearly_ic(),
        horizon=20,
        max_prior_years=1,
        min_prior_years=1,
        min_abs_mean_rank_ic=0.05,
        min_sign_consistency=1.0,
        min_keep_factors=3,
        max_keep_factors=3,
    )

    row = plan.loc[plan["eval_year"].eq(2017)].iloc[0]
    assert row["selected_factor_count"] == 3
    assert row["rule_kept_factor_count"] == 0
    assert row["fallback_selected_count"] == 3
    assert row["evidence_grade"] == "diagnostic_not_backtest"


def test_factor_pruning_empty_when_horizon_missing() -> None:
    detail, plan = build_prior_fit_factor_pruning_plan(_yearly_ic(), horizon=5)

    assert detail.empty
    assert plan.empty


def test_run_factor_pruning_writes_artifacts(tmp_path: Path) -> None:
    diagnostics = tmp_path / "factor_diagnostics"
    diagnostics.mkdir()
    _yearly_ic().to_csv(diagnostics / "yearly_ic_summary.csv", index=False)

    result = run_frontier_factor_pruning_rebuild(
        factor_diagnostics_run_dir=diagnostics,
        output_dir=tmp_path / "out",
        horizon=20,
        max_prior_years=2,
        min_keep_factors=3,
        max_keep_factors=4,
        write_research_log=True,
        research_log_path=tmp_path / "factor_pruning.md",
    )

    run_dir = Path(result["run_dir"])
    assert result["decision"] == "diagnostic_factor_pruning_plan_ready"
    assert result["candidate_count"] == 0
    assert result["fit_uses_eval_year_count"] == 0
    assert (run_dir / "factor_pruning_detail.csv").exists()
    assert (run_dir / "factor_pruning_plan.csv").exists()
    assert (run_dir / "summary.json").exists()
    assert (tmp_path / "factor_pruning.md").exists()
