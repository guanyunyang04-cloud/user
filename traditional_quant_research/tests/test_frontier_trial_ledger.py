from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from traditional_quant_research.experiments import frontier_trial_ledger as ledger_exp


def _aggregate() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "constraint_variant": "baseline",
                "signal": "signal_a",
                "fee_bps": 30.0,
                "impact_bps_per_1pct": 10.0,
                "capital_amount": 100_000_000.0,
                "exposure_penalty_strength": 0.25,
                "constraint_fallback_count": 0,
                "constraint_fallback_rate": 0.0,
                "mean_annualized_return": 0.04,
                "min_annualized_return": -0.02,
                "positive_year_rate": 0.8,
                "worst_max_drawdown": -0.18,
                "total_periods": 36,
                "evidence_grade": "backtest_only",
            },
            {
                "constraint_variant": "regime_gated",
                "signal": "signal_b",
                "fee_bps": 30.0,
                "impact_bps_per_1pct": 10.0,
                "capital_amount": 100_000_000.0,
                "exposure_penalty_strength": 0.50,
                "constraint_fallback_count": 0,
                "constraint_fallback_rate": 0.0,
                "mean_annualized_return": 0.16,
                "min_annualized_return": 0.01,
                "positive_year_rate": 1.0,
                "worst_max_drawdown": -0.10,
                "total_periods": 36,
                "evidence_grade": "out_of_sample_supported",
            },
        ]
    )


def _promotion() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "constraint_variant": "regime_gated",
                "signal": "signal_b",
                "exposure_penalty_strength": 0.50,
                "promotion_level": "strategy_candidate",
                "failed_gates": "",
                "size_gate": "True",
                "promoted": "True",
            }
        ]
    )


def test_trial_ledger_marks_best_trial_and_selection_bias() -> None:
    ledger = ledger_exp.build_trial_ledger(_aggregate(), _promotion())
    selection = ledger_exp.build_selection_bias_report(
        ledger,
        pd.DataFrame([{"signal": "signal_b", "weak_year_count": 1}]),
    )
    maturity = ledger_exp.build_research_maturity_report(ledger, selection, _promotion())

    selected = ledger.loc[ledger["used_for_selection"]].iloc[0]
    assert selected["signal"] == "signal_b"
    assert selected["constraint_variant"] == "regime_gated"
    assert selected["promotion_level"] == "strategy_candidate"
    assert selected["evidence_grade"] == "strategy_candidate"
    assert selection.iloc[0]["trial_count"] == 2
    assert selection.iloc[0]["best_minus_median_return"] == pytest.approx(0.06)
    assert selection.iloc[0]["weak_year_count"] == 1
    readiness = maturity.loc[maturity["dimension"] == "strategy_candidate_readiness"].iloc[0]
    assert readiness["status"] == "passed"


def test_trial_ledger_merges_promotion_by_constraint_variant() -> None:
    aggregate = pd.DataFrame(
        [
            {
                "constraint_variant": "baseline",
                "signal": "signal_a",
                "fee_bps": 30.0,
                "impact_bps_per_1pct": 10.0,
                "capital_amount": 100_000_000.0,
                "exposure_penalty_strength": 0.25,
                "mean_annualized_return": 0.04,
                "min_annualized_return": 0.01,
                "positive_year_rate": 1.0,
                "worst_max_drawdown": -0.10,
                "total_periods": 36,
                "evidence_grade": "backtest_only",
            },
            {
                "constraint_variant": "regime_gated",
                "signal": "signal_a",
                "fee_bps": 30.0,
                "impact_bps_per_1pct": 10.0,
                "capital_amount": 100_000_000.0,
                "exposure_penalty_strength": 0.25,
                "mean_annualized_return": 0.08,
                "min_annualized_return": 0.02,
                "positive_year_rate": 1.0,
                "worst_max_drawdown": -0.08,
                "total_periods": 36,
                "evidence_grade": "out_of_sample_supported",
            },
        ]
    )
    promotion = pd.DataFrame(
        [
            {
                "constraint_variant": "regime_gated",
                "signal": "signal_a",
                "exposure_penalty_strength": 0.25,
                "promotion_level": "strategy_candidate",
                "failed_gates": "",
            }
        ]
    )

    ledger = ledger_exp.build_trial_ledger(aggregate, promotion)

    baseline = ledger.loc[ledger["constraint_variant"].eq("baseline")].iloc[0]
    regime = ledger.loc[ledger["constraint_variant"].eq("regime_gated")].iloc[0]
    assert baseline["promotion_level"] == "candidate-frontier/backtest_only"
    assert regime["promotion_level"] == "strategy_candidate"


def test_trial_ledger_separates_baostock_only_research_from_strategy_candidate() -> None:
    promotion = pd.DataFrame(
        [
            {
                "research_mode": "baostock_only",
                "constraint_variant": "regime_gated",
                "signal": "signal_b",
                "exposure_penalty_strength": 0.50,
                "promotion_level": "candidate-frontier/baostock_only",
                "failed_gates": "",
                "size_gate": "True",
                "true_size_gate": "False",
                "promoted": "True",
            }
        ]
    )

    ledger = ledger_exp.build_trial_ledger(_aggregate(), promotion)
    selection = ledger_exp.build_selection_bias_report(ledger)
    maturity = ledger_exp.build_research_maturity_report(ledger, selection, promotion)
    summary = ledger_exp.summarize_trial_ledger(
        ledger,
        selection,
        maturity,
        run_id="ledger",
        combined_run_dir=Path("combined"),
        promotion_run_dir=Path("promotion"),
        failure_run_dir=None,
    )

    row = ledger.loc[ledger["signal"].eq("signal_b")].iloc[0]
    assert row["research_mode"] == "baostock_only"
    assert row["promotion_level"] == "candidate-frontier/baostock_only"
    assert row["evidence_grade"] == "baostock_only_research_candidate"
    assert maturity.loc[maturity["dimension"].eq("baostock_only_research_readiness"), "status"].iloc[0] == "passed"
    assert maturity.loc[maturity["dimension"].eq("strategy_candidate_readiness"), "status"].iloc[0] == "blocked"
    assert summary["decision"] == "baostock_only_research_review_ready"
    assert summary["candidate_count"] == 1
    assert summary["baostock_only_candidate_count"] == 1
    assert summary["strategy_candidate_count"] == 0


def test_trial_ledger_recognizes_personal_backtest_candidate() -> None:
    promotion = pd.DataFrame(
        [
            {
                "research_mode": "baostock_only",
                "constraint_variant": "baseline",
                "signal": "signal_a",
                "exposure_penalty_strength": 0.25,
                "promotion_level": "personal_backtest_candidate",
                "failed_gates": "",
            }
        ]
    )

    ledger = ledger_exp.build_trial_ledger(_aggregate(), promotion)
    selection = ledger_exp.build_selection_bias_report(ledger)
    maturity = ledger_exp.build_research_maturity_report(ledger, selection, promotion)
    summary = ledger_exp.summarize_trial_ledger(
        ledger,
        selection,
        maturity,
        run_id="ledger",
        combined_run_dir=Path("combined"),
        promotion_run_dir=Path("promotion"),
        failure_run_dir=None,
    )

    row = ledger.loc[ledger["signal"].eq("signal_a")].iloc[0]
    assert row["promotion_level"] == "personal_backtest_candidate"
    assert row["evidence_grade"] == "personal_backtest_candidate"
    assert maturity.loc[maturity["dimension"].eq("personal_backtest_candidate_readiness"), "status"].iloc[0] == "passed"
    assert maturity.loc[maturity["dimension"].eq("strategy_candidate_readiness"), "status"].iloc[0] == "blocked"
    assert summary["decision"] == "personal_strategy_candidates_selected"
    assert summary["personal_backtest_candidate_count"] == 1
    assert summary["strategy_candidate_count"] == 0


def test_run_frontier_trial_ledger_writes_governance_artifacts(tmp_path: Path) -> None:
    combined = tmp_path / "combined"
    promotion = tmp_path / "promotion"
    failure = tmp_path / "failure"
    combined.mkdir()
    promotion.mkdir()
    failure.mkdir()
    _aggregate().to_csv(combined / "combined_constraint_aggregate.csv", index=False)
    _promotion().to_csv(promotion / "promotion_gate_summary.csv", index=False)
    pd.DataFrame([{"signal": "signal_b", "weak_year_count": 1}]).to_csv(
        failure / "signal_failure_summary.csv",
        index=False,
    )

    result = ledger_exp.run_frontier_trial_ledger(
        combined_run_dir=combined,
        promotion_run_dir=promotion,
        failure_run_dir=failure,
        output_dir=tmp_path / "output",
        write_research_log=True,
        research_log_path=tmp_path / "ledger.md",
    )

    run_dir = Path(result["run_dir"])
    assert result["decision"] == "promotion_review_ready"
    assert result["candidate_count"] == 1
    assert (run_dir / "trial_ledger.csv").exists()
    assert (run_dir / "selection_bias_report.csv").exists()
    assert (run_dir / "research_maturity_report.csv").exists()
    assert (run_dir / "summary.json").exists()
    assert (tmp_path / "ledger.md").exists()
