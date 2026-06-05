from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from traditional_quant_research.experiments.frontier_personal_candidate_gate import (
    DIAGNOSTIC_PERSONAL_GATE_SCOPE,
    FORMAL_PERSONAL_GATE_SCOPE,
    PERSONAL_BACKTEST_ONLY_LEVEL,
    PERSONAL_BACKTEST_PROMOTION_LEVEL,
    POST_SELECTION_RECOMMENDATION,
    evaluate_personal_candidate_gates,
    run_frontier_personal_candidate_gate,
)


def _aggregate() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "signal": "multifactor_rolling_ic_weighted_score",
                "constraint_variant": "baseline",
                "top_n": 50,
                "fee_bps": 30.0,
                "capital_amount": 100_000_000.0,
                "impact_bps_per_1pct": 10.0,
                "exposure_penalty_strength": 0.25,
                "constraint_fallback_count": 0,
                "constraint_fallback_rate": 0.0,
                "eval_year_count": 2,
                "mean_annualized_return": 0.12,
                "min_annualized_return": -0.20,
                "positive_year_rate": 0.50,
                "worst_max_drawdown": -0.10,
                "total_periods": 24,
                "exposure_penalty_cols": "log_amount_mean_20d_z,turn_xsec_z",
                "group_col": "industry",
            },
            {
                "signal": "multifactor_low_corr_rank_score",
                "constraint_variant": "baseline",
                "top_n": 50,
                "fee_bps": 30.0,
                "capital_amount": 100_000_000.0,
                "impact_bps_per_1pct": 10.0,
                "exposure_penalty_strength": 1.0,
                "constraint_fallback_count": 0,
                "constraint_fallback_rate": 0.0,
                "eval_year_count": 2,
                "mean_annualized_return": 0.01,
                "min_annualized_return": -0.40,
                "positive_year_rate": 0.50,
                "worst_max_drawdown": -0.20,
                "total_periods": 24,
                "exposure_penalty_cols": "log_amount_mean_20d_z,turn_xsec_z",
                "group_col": "industry",
            },
        ]
    )


def _formal_aggregate() -> pd.DataFrame:
    frame = _aggregate()
    frame.loc[0, "eval_year_count"] = 10
    frame.loc[0, "total_periods"] = 60
    frame.loc[0, "positive_year_rate"] = 0.60
    frame.loc[1, "eval_year_count"] = 10
    frame.loc[1, "total_periods"] = 60
    return frame


def _exposure(value: float = 0.80) -> pd.DataFrame:
    rows = []
    for signal, strength in [
        ("multifactor_rolling_ic_weighted_score", 0.25),
        ("multifactor_low_corr_rank_score", 1.0),
    ]:
        for factor in ["log_amount_mean_20d_z", "neg_volatility_20d_z", "momentum_20d_z", "turn_xsec_z"]:
            rows.append(
                {
                    "signal": signal,
                    "constraint_variant": "baseline",
                    "top_n": 50,
                    "exposure_penalty_strength": strength,
                    "factor": factor,
                    "period_type": "monthly",
                    "mean_abs_active_exposure": value,
                }
            )
    return pd.DataFrame(rows)


def _meta() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"eval_year": 2017, "fit_end_date": "2016-12-31", "start_date": "2017-01-01"},
            {"eval_year": 2018, "fit_end_date": "2017-12-31", "start_date": "2018-01-01"},
        ]
    )


def _formal_meta() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "eval_year": year,
                "fit_end_date": f"{year - 1}-12-31",
                "start_date": f"{year}-01-01",
            }
            for year in range(2017, 2027)
        ]
    )


def _summary() -> dict[str, object]:
    return {
        "snapshot_id": "baostock_v2_fixture",
        "execution_constraints": True,
    }


def test_personal_gate_promotes_pragmatic_baostock_backtest_candidate() -> None:
    gate = evaluate_personal_candidate_gates(
        _formal_aggregate(),
        _exposure(),
        _formal_meta(),
        combined_summary=_summary(),
        required_fee_bps=30.0,
        required_impact_bps_per_1pct=10.0,
        personal_capital_amount=1_000_000.0,
        min_eval_year_count=10,
        required_start_year=2017,
        required_end_year=2026,
        min_total_periods=50,
        min_mean_annualized_return=0.05,
        min_positive_year_rate=0.60,
        min_weakest_year_annualized_return=-0.35,
        max_worst_drawdown=-0.25,
        max_proxy_mean_abs_active_exposure=1.25,
        exposure_fields=["log_amount_mean_20d_z", "neg_volatility_20d_z", "momentum_20d_z", "turn_xsec_z"],
    )

    promoted = gate.loc[gate["signal"].eq("multifactor_rolling_ic_weighted_score")].iloc[0]
    rejected = gate.loc[gate["signal"].eq("multifactor_low_corr_rank_score")].iloc[0]
    assert promoted["promotion_level"] == PERSONAL_BACKTEST_PROMOTION_LEVEL
    assert promoted["evidence_scope"] == FORMAL_PERSONAL_GATE_SCOPE
    assert bool(promoted["formal_profile_gate"]) is True
    assert promoted["top_n"] == 50
    assert promoted["paper_tracking_recommendation"] == POST_SELECTION_RECOMMENDATION
    assert bool(promoted["walk_forward_gate"]) is True
    assert bool(promoted["capital_stress_gate"]) is True
    assert rejected["promotion_level"] == PERSONAL_BACKTEST_ONLY_LEVEL
    assert "return_gate" in rejected["failed_gates"]
    assert "weak_year_damage_gate" in rejected["failed_gates"]


def test_personal_gate_keeps_relaxed_short_sample_diagnostic_only() -> None:
    gate = evaluate_personal_candidate_gates(
        _aggregate(),
        _exposure(),
        _meta(),
        combined_summary=_summary(),
        required_fee_bps=30.0,
        required_impact_bps_per_1pct=10.0,
        personal_capital_amount=1_000_000.0,
        min_eval_year_count=2,
        required_start_year=2017,
        required_end_year=2018,
        min_total_periods=20,
        min_mean_annualized_return=0.05,
        min_positive_year_rate=0.50,
        min_weakest_year_annualized_return=-0.30,
        max_worst_drawdown=-0.25,
        max_proxy_mean_abs_active_exposure=1.25,
        exposure_fields=["log_amount_mean_20d_z", "neg_volatility_20d_z", "momentum_20d_z", "turn_xsec_z"],
    )

    row = gate.loc[gate["signal"].eq("multifactor_rolling_ic_weighted_score")].iloc[0]
    assert row["promotion_level"] == PERSONAL_BACKTEST_ONLY_LEVEL
    assert row["evidence_scope"] == DIAGNOSTIC_PERSONAL_GATE_SCOPE
    assert bool(row["formal_profile_gate"]) is False
    assert "formal_profile_gate" in row["failed_gates"]
    assert row["paper_tracking_recommendation"] == "none"


def test_personal_gate_marks_short_evidence_diagnostic_even_with_formal_thresholds() -> None:
    gate = evaluate_personal_candidate_gates(
        _aggregate().head(1),
        _exposure(),
        _meta(),
        combined_summary=_summary(),
        required_fee_bps=30.0,
        required_impact_bps_per_1pct=10.0,
        personal_capital_amount=1_000_000.0,
        min_eval_year_count=10,
        required_start_year=2017,
        required_end_year=2026,
        min_total_periods=50,
        min_mean_annualized_return=0.05,
        min_positive_year_rate=0.60,
        min_weakest_year_annualized_return=-0.35,
        max_worst_drawdown=-0.25,
        max_proxy_mean_abs_active_exposure=1.25,
        exposure_fields=["log_amount_mean_20d_z", "neg_volatility_20d_z", "momentum_20d_z", "turn_xsec_z"],
    )

    row = gate.iloc[0]
    assert row["promotion_level"] == PERSONAL_BACKTEST_ONLY_LEVEL
    assert row["evidence_scope"] == DIAGNOSTIC_PERSONAL_GATE_SCOPE
    assert bool(row["formal_profile_gate"]) is False
    assert "formal_profile_gate" in row["failed_gates"]
    assert "walk_forward_gate" in row["failed_gates"]
    assert "sample_gate" in row["failed_gates"]


def test_personal_gate_blocks_non_prior_fit_meta() -> None:
    bad_meta = pd.DataFrame([{"eval_year": 2017, "fit_end_date": "2017-01-01", "start_date": "2017-01-01"}])
    gate = evaluate_personal_candidate_gates(
        _aggregate().head(1),
        _exposure(),
        bad_meta,
        combined_summary=_summary(),
        required_fee_bps=30.0,
        required_impact_bps_per_1pct=10.0,
        personal_capital_amount=1_000_000.0,
        min_eval_year_count=1,
        required_start_year=2017,
        required_end_year=2017,
        min_total_periods=20,
        min_mean_annualized_return=0.05,
        min_positive_year_rate=0.50,
        min_weakest_year_annualized_return=-0.30,
        max_worst_drawdown=-0.25,
        max_proxy_mean_abs_active_exposure=1.25,
        exposure_fields=["log_amount_mean_20d_z", "neg_volatility_20d_z", "momentum_20d_z", "turn_xsec_z"],
    )

    row = gate.iloc[0]
    assert bool(row["walk_forward_gate"]) is False
    assert "walk_forward_gate" in row["failed_gates"]
    assert row["walk_forward_detail"] == "fit_window_not_prior_to_eval"


def test_personal_gate_blocks_optimizer_fallback_even_when_returns_pass() -> None:
    aggregate = _formal_aggregate().head(1).copy()
    aggregate["constraint_variant"] = "explicit_exposure_constraint"
    aggregate["constraint_fallback_count"] = 1
    aggregate["constraint_fallback_rate"] = 0.10
    exposure = _exposure().loc[lambda frame: frame["signal"].eq("multifactor_rolling_ic_weighted_score")].copy()
    exposure["constraint_variant"] = "explicit_exposure_constraint"

    gate = evaluate_personal_candidate_gates(
        aggregate,
        exposure,
        _formal_meta(),
        combined_summary=_summary(),
        required_fee_bps=30.0,
        required_impact_bps_per_1pct=10.0,
        personal_capital_amount=1_000_000.0,
        min_eval_year_count=10,
        required_start_year=2017,
        required_end_year=2026,
        min_total_periods=50,
        min_mean_annualized_return=0.05,
        min_positive_year_rate=0.60,
        min_weakest_year_annualized_return=-0.35,
        max_worst_drawdown=-0.25,
        max_proxy_mean_abs_active_exposure=1.25,
        exposure_fields=["log_amount_mean_20d_z", "neg_volatility_20d_z", "momentum_20d_z", "turn_xsec_z"],
    )

    row = gate.iloc[0]
    assert row["promotion_level"] == PERSONAL_BACKTEST_ONLY_LEVEL
    assert bool(row["optimizer_fallback_gate"]) is False
    assert "optimizer_fallback_gate" in row["failed_gates"]
    assert row["paper_tracking_recommendation"] == "none"


def test_run_frontier_personal_candidate_gate_writes_artifacts(tmp_path: Path) -> None:
    combined = tmp_path / "combined"
    combined.mkdir()
    _formal_aggregate().to_csv(combined / "combined_constraint_aggregate.csv", index=False)
    _exposure().to_csv(combined / "combined_constraint_basket_exposure_summary.csv", index=False)
    _formal_meta().to_csv(combined / "combined_constraint_meta.csv", index=False)
    (combined / "summary.json").write_text(
        '{"snapshot_id":"baostock_v2_fixture","execution_constraints":true}',
        encoding="utf-8",
    )

    result = run_frontier_personal_candidate_gate(
        combined_run_dir=combined,
        output_dir=tmp_path / "out",
        write_research_log=True,
        research_log_path=tmp_path / "personal_gate.md",
    )

    run_dir = Path(result["run_dir"])
    assert result["decision"] == "personal_strategy_candidates_selected"
    assert result["post_selection_boundary"] == "agent_selects_models_and_strategies_only; user_handles_risk_recording_and_live_decisions"
    assert result["evidence_scope"] == FORMAL_PERSONAL_GATE_SCOPE
    assert result["formal_gate_profile"] is True
    assert result["personal_backtest_candidate_count"] == 1
    assert result["strategy_candidate_count"] == 0
    assert (run_dir / "personal_candidate_gate_summary.csv").exists()
    assert (run_dir / "summary.json").exists()
    assert (tmp_path / "personal_gate.md").exists()
    saved = pd.read_csv(run_dir / "personal_candidate_gate_summary.csv")
    assert PERSONAL_BACKTEST_PROMOTION_LEVEL in set(saved["promotion_level"])
    assert FORMAL_PERSONAL_GATE_SCOPE in set(saved["evidence_scope"])
