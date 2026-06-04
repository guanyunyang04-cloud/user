from __future__ import annotations

from pathlib import Path

import json
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


def test_fit_eval_regime_candidates_can_emit_generic_market_rules() -> None:
    yearly_failure = pd.concat(
        [
            _yearly_failure(),
            _yearly_failure().assign(signal="signal_b", annualized_return=[0.04, -0.02, 0.06, -0.01]),
        ],
        ignore_index=True,
    )
    signal_year_regime = rebuild.build_signal_year_regime(yearly_failure, _yearly_regime())

    fit_eval = rebuild.build_fit_eval_regime_candidates(signal_year_regime, include_generic_regime=True)

    generic = fit_eval.loc[fit_eval["signal"].eq(rebuild.GENERIC_REGIME_SIGNAL)].copy()
    assert not generic.empty
    assert set(generic["rule_scope"]) == {rebuild.GENERIC_REGIME_RULE_SCOPE}
    assert generic["fit_uses_eval_year"].eq(False).all()
    row_2022 = generic.loc[generic["eval_year"].eq(2022)].iloc[0]
    assert row_2022["fit_years"] == "2017,2018"
    assert row_2022["source_signal_count"] == 2
    assert "2022" not in row_2022["fit_years"]


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


def _factor_family_evidence() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"factor_family": "core", "eval_year": 2017, "signal": "signal_a", "top_n": 200, "constraint_variant": "baseline", "exposure_penalty_strength": 0.25, "fee_bps": 30.0, "capital_amount": 100_000_000.0, "impact_bps_per_1pct": 10.0, "annualized_return": 0.08, "max_drawdown": -0.05, "periods": 4},
            {"factor_family": "core", "eval_year": 2018, "signal": "signal_a", "top_n": 200, "constraint_variant": "baseline", "exposure_penalty_strength": 0.25, "fee_bps": 30.0, "capital_amount": 100_000_000.0, "impact_bps_per_1pct": 10.0, "annualized_return": -0.10, "max_drawdown": -0.08, "periods": 3},
            {"factor_family": "core", "eval_year": 2019, "signal": "signal_a", "top_n": 200, "constraint_variant": "baseline", "exposure_penalty_strength": 0.25, "fee_bps": 30.0, "capital_amount": 100_000_000.0, "impact_bps_per_1pct": 10.0, "annualized_return": 0.02, "max_drawdown": -0.07, "periods": 5},
            {"factor_family": "core", "eval_year": 2020, "signal": "signal_a", "top_n": 200, "constraint_variant": "baseline", "exposure_penalty_strength": 0.25, "fee_bps": 30.0, "capital_amount": 100_000_000.0, "impact_bps_per_1pct": 10.0, "annualized_return": 0.50, "max_drawdown": -0.03, "periods": 6},
            {"factor_family": "expanded", "eval_year": 2017, "signal": "signal_a", "top_n": 200, "constraint_variant": "baseline", "exposure_penalty_strength": 0.25, "fee_bps": 30.0, "capital_amount": 100_000_000.0, "impact_bps_per_1pct": 10.0, "annualized_return": -0.02, "max_drawdown": -0.06, "periods": 4},
            {"factor_family": "expanded", "eval_year": 2018, "signal": "signal_a", "top_n": 200, "constraint_variant": "baseline", "exposure_penalty_strength": 0.25, "fee_bps": 30.0, "capital_amount": 100_000_000.0, "impact_bps_per_1pct": 10.0, "annualized_return": 0.14, "max_drawdown": -0.05, "periods": 3},
            {"factor_family": "expanded", "eval_year": 2019, "signal": "signal_a", "top_n": 200, "constraint_variant": "baseline", "exposure_penalty_strength": 0.25, "fee_bps": 30.0, "capital_amount": 100_000_000.0, "impact_bps_per_1pct": 10.0, "annualized_return": 0.22, "max_drawdown": -0.04, "periods": 5},
            {"factor_family": "expanded", "eval_year": 2020, "signal": "signal_a", "top_n": 200, "constraint_variant": "baseline", "exposure_penalty_strength": 0.25, "fee_bps": 30.0, "capital_amount": 100_000_000.0, "impact_bps_per_1pct": 10.0, "annualized_return": -0.30, "max_drawdown": -0.12, "periods": 6},
        ]
    )


def test_fit_eval_factor_family_candidates_use_prior_years_only() -> None:
    fit_eval = rebuild.build_fit_eval_factor_family_candidates(_factor_family_evidence())

    assert not fit_eval.empty
    assert fit_eval["fit_uses_eval_year"].eq(False).all()
    row_2020 = fit_eval.loc[fit_eval["eval_year"] == 2020].iloc[0]
    assert row_2020["fit_years"] == "2017,2018,2019"
    assert "2020" not in row_2020["fit_years"]
    assert row_2020["selected_factor_family"] == "expanded"
    assert row_2020["eval_selected_family_annualized_return"] == -0.30
    assert row_2020["evidence_grade"] == "diagnostic_not_backtest"


def test_factor_family_tie_break_prefers_core() -> None:
    evidence = pd.DataFrame(
        [
            {"factor_family": "core", "eval_year": 2017, "signal": "signal_a", "top_n": 200, "constraint_variant": "baseline", "exposure_penalty_strength": 0.25, "fee_bps": 30.0, "capital_amount": 100_000_000.0, "impact_bps_per_1pct": 10.0, "annualized_return": 0.10, "max_drawdown": -0.05, "periods": 4},
            {"factor_family": "expanded", "eval_year": 2017, "signal": "signal_a", "top_n": 200, "constraint_variant": "baseline", "exposure_penalty_strength": 0.25, "fee_bps": 30.0, "capital_amount": 100_000_000.0, "impact_bps_per_1pct": 10.0, "annualized_return": 0.10, "max_drawdown": -0.05, "periods": 4},
            {"factor_family": "core", "eval_year": 2018, "signal": "signal_a", "top_n": 200, "constraint_variant": "baseline", "exposure_penalty_strength": 0.25, "fee_bps": 30.0, "capital_amount": 100_000_000.0, "impact_bps_per_1pct": 10.0, "annualized_return": 0.05, "max_drawdown": -0.06, "periods": 4},
            {"factor_family": "expanded", "eval_year": 2018, "signal": "signal_a", "top_n": 200, "constraint_variant": "baseline", "exposure_penalty_strength": 0.25, "fee_bps": 30.0, "capital_amount": 100_000_000.0, "impact_bps_per_1pct": 10.0, "annualized_return": 0.50, "max_drawdown": -0.02, "periods": 4},
        ]
    )

    fit_eval = rebuild.build_fit_eval_factor_family_candidates(evidence)

    row_2018 = fit_eval.loc[fit_eval["eval_year"] == 2018].iloc[0]
    assert row_2018["fit_years"] == "2017"
    assert row_2018["selected_factor_family"] == "core"
    assert row_2018["eval_selected_family_annualized_return"] == 0.05


def _write_factor_family_run(path: Path, *, factor_set: str, returns: dict[int, float]) -> None:
    path.mkdir(parents=True)
    rows = [
        {
            "eval_year": year,
            "signal": "signal_a",
            "top_n": 200,
            "constraint_variant": "baseline",
            "exposure_penalty_strength": 0.25,
            "fee_bps": 30.0,
            "capital_amount": 100_000_000.0,
            "impact_bps_per_1pct": 10.0,
            "annualized_return": value,
            "max_drawdown": -0.05,
            "periods": 4,
        }
        for year, value in returns.items()
    ]
    pd.DataFrame(rows).to_csv(path / "combined_constraint_summary.csv", index=False)
    (path / "summary.json").write_text(json.dumps({"factor_set": factor_set, "run_id": path.name}), encoding="utf-8")


def test_run_frontier_weak_year_rebuild_writes_factor_family_diagnostics(tmp_path: Path) -> None:
    failure = tmp_path / "failure"
    regime = tmp_path / "regime"
    core = tmp_path / "core_combined"
    expanded = tmp_path / "expanded_combined"
    failure.mkdir()
    regime.mkdir()
    _yearly_failure().to_csv(failure / "yearly_failure_attribution.csv", index=False)
    _yearly_regime().to_csv(regime / "yearly_market_regime.csv", index=False)
    _write_factor_family_run(core, factor_set="core", returns={2017: 0.08, 2018: -0.10, 2019: 0.02, 2020: 0.50})
    _write_factor_family_run(expanded, factor_set="expanded", returns={2017: -0.02, 2018: 0.14, 2019: 0.22, 2020: -0.30})

    result = rebuild.run_frontier_weak_year_rebuild(
        failure_run_dir=failure,
        regime_run_dir=regime,
        factor_family_run_dirs={"core": core, "expanded": expanded},
        output_dir=tmp_path / "output",
    )

    run_dir = Path(result["run_dir"])
    assert result["fit_eval_factor_family_candidate_rows"] > 0
    assert result["factor_family_fit_uses_eval_year_count"] == 0
    assert (run_dir / "factor_family_evidence.csv").exists()
    assert (run_dir / "fit_eval_factor_family_candidates.csv").exists()
    candidates = pd.read_csv(run_dir / "fit_eval_factor_family_candidates.csv")
    row_2018 = candidates.loc[candidates["eval_year"] == 2018].iloc[0]
    row_2020 = candidates.loc[candidates["eval_year"] == 2020].iloc[0]
    assert row_2018["fit_years"] == "2017"
    assert row_2018["selected_factor_family"] == "core"
    assert row_2020["fit_years"] == "2017,2018,2019"
    assert row_2020["selected_factor_family"] == "expanded"
    assert row_2020["eval_selected_family_annualized_return"] == -0.30


def test_read_factor_family_evidence_keeps_multiple_unnamed_paths(tmp_path: Path) -> None:
    family_a = tmp_path / "family_a"
    family_b = tmp_path / "family_b"
    _write_factor_family_run(family_a, factor_set="ignored_a", returns={2017: 0.01})
    _write_factor_family_run(family_b, factor_set="ignored_b", returns={2017: 0.02})

    evidence = rebuild.read_factor_family_evidence([str(family_a), str(family_b)])

    assert set(evidence["factor_family"]) == {"family_a", "family_b"}
