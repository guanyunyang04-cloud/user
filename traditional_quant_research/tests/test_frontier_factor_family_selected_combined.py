from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from traditional_quant_research.experiments import frontier_factor_family_selected_combined as selected


SIGNAL = "multifactor_rolling_ic_weighted_score"


def _rule_rows(*, leak: bool = False) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "eval_year": 2018,
                "signal": SIGNAL,
                "top_n": 200,
                "constraint_variant": "baseline",
                "exposure_penalty_strength": 0.25,
                "fee_bps": 30.0,
                "capital_amount": 100_000_000.0,
                "impact_bps_per_1pct": 10.0,
                "fit_years": "2017",
                "fit_row_count": 1,
                "candidate_factor_families": "core,expanded",
                "selected_factor_family": "core",
                "eval_selected_family_annualized_return": 0.05,
                "eval_best_available_family": "expanded",
                "eval_best_available_annualized_return": 0.50,
                "fit_uses_eval_year": leak,
                "evidence_grade": "diagnostic_not_backtest",
            }
        ]
    )


def _family_evidence() -> pd.DataFrame:
    rows = []
    for family, value_2017, value_2018 in [("core", 0.01, 0.05), ("expanded", 0.02, 0.50)]:
        for year, value in [(2017, value_2017), (2018, value_2018)]:
            rows.append(
                {
                    "factor_family": family,
                    "eval_year": year,
                    "signal": SIGNAL,
                    "top_n": 200,
                    "constraint_variant": "baseline",
                    "exposure_penalty_strength": 0.25,
                    "fee_bps": 30.0,
                    "capital_amount": 100_000_000.0,
                    "impact_bps_per_1pct": 10.0,
                    "annualized_return": value,
                    "max_drawdown": -0.10,
                    "periods": 4,
                }
            )
    return pd.DataFrame(rows)


def test_selected_factor_family_plan_uses_selected_family_not_oracle() -> None:
    plan = selected.build_selected_factor_family_plan(
        _rule_rows(),
        factor_family_evidence=_family_evidence(),
        cold_start_family="core",
    )

    row_2017 = plan.loc[plan["eval_year"].eq(2017)].iloc[0]
    row_2018 = plan.loc[plan["eval_year"].eq(2018)].iloc[0]
    assert row_2017["selected_factor_family"] == "core"
    assert row_2017["factor_family_selection_reason"] == "cold_start_family"
    assert row_2018["selected_factor_family"] == "core"
    assert row_2018["eval_best_available_family"] == "expanded"
    assert row_2018["factor_family_selection_reason"] == "prior_fit_selected"


def test_selected_factor_family_plan_honors_output_constraint_variant() -> None:
    plan = selected.build_selected_factor_family_plan(
        _rule_rows(),
        factor_family_evidence=_family_evidence(),
        cold_start_family="core",
        output_constraint_variant="custom_prior_fit",
    )

    assert set(plan["source_constraint_variant"]) == {"baseline"}
    assert set(plan["constraint_variant"]) == {"custom_prior_fit"}


def test_selected_factor_family_plan_rejects_eval_year_leakage() -> None:
    with pytest.raises(ValueError, match="fit_uses_eval_year"):
        selected.build_selected_factor_family_plan(
            _rule_rows(leak=True),
            factor_family_evidence=_family_evidence(),
            cold_start_family="core",
        )


def _summary_row(year: int, family: str, annualized_return: float) -> dict[str, object]:
    return {
        "eval_year": year,
        "top_n": 200,
        "signal": SIGNAL,
        "constraint_variant": "baseline",
        "evidence_grade": "backtest_only",
        "exposure_penalty_cols": "log_amount_mean_20d_z",
        "exposure_penalty_strength": 0.25,
        "group_col": "industry",
        "max_group_weight": 0.1,
        "fee_bps": 30.0,
        "capital_amount": 100_000_000.0,
        "impact_bps_per_1pct": 10.0,
        "annualized_return": annualized_return,
        "sharpe": annualized_return * 10,
        "max_drawdown": -0.10,
        "mean_turnover": 0.5,
        "mean_impact_cost": 0.001,
        "mean_total_cost": 0.004,
        "periods": 4,
        "constraint_fallback_count": 0,
        "constraint_fallback_rate": 0.0,
        "family_marker": family,
    }


def _write_combined_run(path: Path, *, family: str, returns: dict[int, float]) -> None:
    path.mkdir(parents=True)
    summary = pd.DataFrame([_summary_row(year, family, value) for year, value in returns.items()])
    summary.to_csv(path / "combined_constraint_summary.csv", index=False)
    pd.DataFrame(
        [
            {
                "eval_year": year,
                "signal": SIGNAL,
                "top_n": 200,
                "constraint_variant": "baseline",
                "exposure_penalty_strength": 0.25,
                "fee_bps": 30.0,
                "capital_amount": 100_000_000.0,
                "impact_bps_per_1pct": 10.0,
                "trade_id": 0,
                "net_return": value / 4,
            }
            for year, value in returns.items()
        ]
    ).to_csv(path / "combined_constraint_trades.csv", index=False)
    pd.DataFrame(
        [
            {
                "eval_year": year,
                "signal": SIGNAL,
                "signal_config": SIGNAL,
                "top_n": 200,
                "constraint_variant": "baseline",
                "exposure_penalty_strength": 0.25,
                "period_type": "monthly",
                "factor": "log_amount_mean_20d_z",
                "active_exposure": 0.20 if family == "core" else 0.80,
            }
            for year in returns
        ]
    ).to_csv(path / "combined_constraint_basket_exposure.csv", index=False)
    pd.DataFrame(
        [
            {
                "eval_year": year,
                "signal": SIGNAL,
                "signal_config": SIGNAL,
                "top_n": 200,
                "constraint_variant": "baseline",
                "exposure_penalty_strength": 0.25,
                "period_type": "monthly",
                "abs_active_weight": 0.10,
            }
            for year in returns
        ]
    ).to_csv(path / "combined_constraint_industry_exposure.csv", index=False)
    pd.DataFrame(
        [
            {
                "eval_year": year,
                "fit_end_date": f"{year - 1}-12-31",
                "start_date": f"{year}-01-01",
                "factor_set": family,
            }
            for year in returns
        ]
    ).to_csv(path / "combined_constraint_meta.csv", index=False)
    (path / "summary.json").write_text(
        json.dumps(
            {
                "run_id": path.name,
                "snapshot_id": "baostock_v2_fixture",
                "execution_constraints": True,
                "horizon": 20,
                "rebalance_frequency": "monthly",
                "top_n": 200,
                "top_n_values": [200],
                "factor_set": family,
                "buffer_multiplier": 3.0,
            }
        ),
        encoding="utf-8",
    )


def test_run_factor_family_selected_combined_writes_gate_compatible_outputs(tmp_path: Path) -> None:
    weak = tmp_path / "weak_rebuild"
    weak.mkdir()
    _rule_rows().to_csv(weak / "fit_eval_factor_family_candidates.csv", index=False)
    _family_evidence().to_csv(weak / "factor_family_evidence.csv", index=False)
    core = tmp_path / "core"
    expanded = tmp_path / "expanded"
    _write_combined_run(core, family="core", returns={2017: 0.01, 2018: 0.05})
    _write_combined_run(expanded, family="expanded", returns={2017: 0.02, 2018: 0.50})

    result = selected.run_frontier_factor_family_selected_combined(
        weak_year_rebuild_run_dir=weak,
        factor_family_run_dirs={"core": core, "expanded": expanded},
        output_dir=tmp_path / "out",
        top_n_values=(200,),
        impact_bps_per_1pct_values=(10.0,),
        cold_start_family="core",
        output_constraint_variant="custom_prior_fit",
    )

    run_dir = Path(result["output_dir"])
    assert result["candidate_count"] == 0
    assert result["selected_summary_rows"] == 2
    assert (run_dir / "combined_constraint_summary.csv").exists()
    assert (run_dir / "combined_constraint_aggregate.csv").exists()
    assert (run_dir / "combined_constraint_basket_exposure_summary.csv").exists()
    assert (run_dir / "combined_constraint_meta.csv").exists()
    assert (run_dir / "factor_family_selection_plan.csv").exists()

    summary = pd.read_csv(run_dir / "combined_constraint_summary.csv")
    assert set(summary["selected_factor_family"]) == {"core"}
    assert set(summary["source_constraint_variant"]) == {"baseline"}
    assert set(summary["constraint_variant"]) == {"custom_prior_fit"}
    plan = pd.read_csv(run_dir / "factor_family_selection_plan.csv")
    assert set(plan["constraint_variant"]) == {"custom_prior_fit"}
    row_2018 = summary.loc[summary["eval_year"].eq(2018)].iloc[0]
    assert row_2018["annualized_return"] == pytest.approx(0.05)
    assert row_2018["eval_best_available_family"] == "expanded"
    assert row_2018["family_marker"] == "core"
    aggregate = pd.read_csv(run_dir / "combined_constraint_aggregate.csv")
    assert aggregate.iloc[0]["eval_year_count"] == 2
    meta = pd.read_csv(run_dir / "combined_constraint_meta.csv")
    assert set(meta["eval_year"]) == {2017, 2018}
