from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from traditional_quant_research.experiments.frontier_personal_candidate_gate import (
    PERSONAL_BACKTEST_ONLY_LEVEL,
    PERSONAL_BACKTEST_PROMOTION_LEVEL,
    evaluate_personal_candidate_gates,
    run_frontier_personal_candidate_gate,
)


def _aggregate() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "signal": "multifactor_rolling_ic_weighted_score",
                "constraint_variant": "baseline",
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


def _summary() -> dict[str, object]:
    return {
        "snapshot_id": "baostock_v2_fixture",
        "execution_constraints": True,
    }


def test_personal_gate_promotes_pragmatic_baostock_backtest_candidate() -> None:
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

    promoted = gate.loc[gate["signal"].eq("multifactor_rolling_ic_weighted_score")].iloc[0]
    rejected = gate.loc[gate["signal"].eq("multifactor_low_corr_rank_score")].iloc[0]
    assert promoted["promotion_level"] == PERSONAL_BACKTEST_PROMOTION_LEVEL
    assert promoted["paper_tracking_recommendation"] == "start_paper_tracking"
    assert bool(promoted["walk_forward_gate"]) is True
    assert bool(promoted["capital_stress_gate"]) is True
    assert rejected["promotion_level"] == PERSONAL_BACKTEST_ONLY_LEVEL
    assert "return_gate" in rejected["failed_gates"]
    assert "weak_year_damage_gate" in rejected["failed_gates"]


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


def test_run_frontier_personal_candidate_gate_writes_artifacts(tmp_path: Path) -> None:
    combined = tmp_path / "combined"
    combined.mkdir()
    _aggregate().to_csv(combined / "combined_constraint_aggregate.csv", index=False)
    _exposure().to_csv(combined / "combined_constraint_basket_exposure_summary.csv", index=False)
    _meta().to_csv(combined / "combined_constraint_meta.csv", index=False)
    (combined / "summary.json").write_text(
        '{"snapshot_id":"baostock_v2_fixture","execution_constraints":true}',
        encoding="utf-8",
    )

    result = run_frontier_personal_candidate_gate(
        combined_run_dir=combined,
        output_dir=tmp_path / "out",
        min_eval_year_count=2,
        required_start_year=2017,
        required_end_year=2018,
        min_total_periods=20,
        min_positive_year_rate=0.50,
        write_research_log=True,
        research_log_path=tmp_path / "personal_gate.md",
    )

    run_dir = Path(result["run_dir"])
    assert result["decision"] == "personal_paper_tracking_ready"
    assert result["personal_backtest_candidate_count"] == 1
    assert result["strategy_candidate_count"] == 0
    assert (run_dir / "personal_candidate_gate_summary.csv").exists()
    assert (run_dir / "summary.json").exists()
    assert (tmp_path / "personal_gate.md").exists()
    saved = pd.read_csv(run_dir / "personal_candidate_gate_summary.csv")
    assert PERSONAL_BACKTEST_PROMOTION_LEVEL in set(saved["promotion_level"])
