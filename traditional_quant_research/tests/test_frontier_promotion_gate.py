from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from traditional_quant_research.experiments.frontier_promotion_gate import (
    evaluate_promotion_gates,
    run_frontier_promotion_gate,
)


def _aggregate(*, total_periods: int = 30, constraint_variant: str = "baseline", fallback_count: int = 0) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "constraint_variant": constraint_variant,
                "signal": "signal_a",
                "exposure_penalty_strength": 0.25,
                "fee_bps": 30.0,
                "impact_bps_per_1pct": 10.0,
                "eval_year_count": 3,
                "mean_annualized_return": 0.20,
                "min_annualized_return": 0.05,
                "positive_year_rate": 1.0,
                "worst_max_drawdown": -0.10,
                "total_periods": total_periods,
                "constraint_fallback_count": fallback_count,
                "constraint_fallback_rate": 1.0 if fallback_count else 0.0,
            }
        ]
    )


def _exposure(value: float = 0.20, constraint_variant: str = "baseline") -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "constraint_variant": constraint_variant,
                "signal": "signal_a",
                "exposure_penalty_strength": 0.25,
                "factor": factor,
                "period_type": "monthly",
                "mean_abs_active_exposure": value,
            }
            for factor in ["log_amount_mean_20d_z", "neg_volatility_20d_z", "momentum_20d_z", "turn_xsec_z"]
        ]
    )


def test_promotion_gate_promotes_only_when_all_gates_pass() -> None:
    gate = evaluate_promotion_gates(
        _aggregate(),
        _exposure(),
        size_summary={"daily_size_ready_for_research": True},
        required_impact_bps=10.0,
        required_fee_bps=30.0,
        min_eval_year_count=3,
        min_total_periods=24,
        max_monthly_mean_abs_active_exposure=0.5,
        exposure_fields=["log_amount_mean_20d_z", "neg_volatility_20d_z", "momentum_20d_z", "turn_xsec_z"],
    )

    row = gate.iloc[0]
    assert bool(row["promoted"]) is True
    assert row["promotion_level"] == "strategy_candidate"
    assert row["failed_gates"] == ""


def test_promotion_gate_blocks_missing_size_short_sample_and_exposure() -> None:
    gate = evaluate_promotion_gates(
        _aggregate(total_periods=18),
        _exposure(value=0.9),
        size_summary={"daily_size_ready_for_research": False},
        required_impact_bps=10.0,
        required_fee_bps=30.0,
        min_eval_year_count=3,
        min_total_periods=24,
        max_monthly_mean_abs_active_exposure=0.5,
        exposure_fields=["log_amount_mean_20d_z", "neg_volatility_20d_z", "momentum_20d_z", "turn_xsec_z"],
    )

    row = gate.iloc[0]
    assert bool(row["promoted"]) is False
    assert "size_gate" in row["failed_gates"]
    assert "sample_gate" in row["failed_gates"]
    assert "style_exposure_gate" in row["failed_gates"]
    assert "log_amount_mean_20d_z" in row["exposure_failures"]
    assert row["promotion_level"] == "candidate-frontier/backtest_only"


def test_promotion_gate_blocks_optimizer_fallback_even_when_exposure_is_small() -> None:
    gate = evaluate_promotion_gates(
        _aggregate(fallback_count=1),
        _exposure(value=0.1),
        size_summary={"daily_size_ready_for_research": True},
        required_impact_bps=10.0,
        required_fee_bps=30.0,
        min_eval_year_count=3,
        min_total_periods=24,
        max_monthly_mean_abs_active_exposure=0.5,
        exposure_fields=["log_amount_mean_20d_z", "neg_volatility_20d_z", "momentum_20d_z", "turn_xsec_z"],
    )

    row = gate.iloc[0]
    assert bool(row["promoted"]) is False
    assert row["style_exposure_gate"] is False
    assert "style_exposure_gate" in row["failed_gates"]
    assert "optimizer_fallback" in row["exposure_failures"]
    assert row["constraint_fallback_count"] == 1


def test_promotion_gate_matches_exposure_by_constraint_variant() -> None:
    aggregate = pd.concat(
        [
            _aggregate(constraint_variant="baseline"),
            _aggregate(constraint_variant="regime_gated"),
        ],
        ignore_index=True,
    )
    exposure = pd.concat(
        [
            _exposure(value=0.9, constraint_variant="baseline"),
            _exposure(value=0.1, constraint_variant="regime_gated"),
        ],
        ignore_index=True,
    )

    gate = evaluate_promotion_gates(
        aggregate,
        exposure,
        size_summary={"daily_size_ready_for_research": True},
        required_impact_bps=10.0,
        required_fee_bps=30.0,
        min_eval_year_count=3,
        min_total_periods=24,
        max_monthly_mean_abs_active_exposure=0.5,
        exposure_fields=["log_amount_mean_20d_z", "neg_volatility_20d_z", "momentum_20d_z", "turn_xsec_z"],
    )

    baseline = gate.loc[gate["constraint_variant"].eq("baseline")].iloc[0]
    regime = gate.loc[gate["constraint_variant"].eq("regime_gated")].iloc[0]
    assert baseline["style_exposure_gate"] is False
    assert bool(regime["promoted"]) is True


def test_run_frontier_promotion_gate_writes_artifacts(tmp_path: Path) -> None:
    combined = tmp_path / "combined"
    size = tmp_path / "size"
    combined.mkdir()
    size.mkdir()
    _aggregate(total_periods=18).to_csv(combined / "combined_constraint_aggregate.csv", index=False)
    _exposure(value=0.9).to_csv(combined / "combined_constraint_basket_exposure_summary.csv", index=False)
    (size / "summary.json").write_text(
        json.dumps({"status": "daily_size_absent", "daily_size_ready_for_research": False}),
        encoding="utf-8",
    )

    result = run_frontier_promotion_gate(
        combined_run_dir=combined,
        size_audit_run_dir=size,
        output_dir=tmp_path / "output",
        write_research_log=True,
        research_log_path=tmp_path / "gate.md",
    )

    run_dir = Path(result["run_dir"])
    assert result["candidate_count"] == 0
    assert result["decision"] == "keep_candidate_frontier_backtest_only"
    assert result["top_failed_gates"]["size_gate"] == 1
    assert (run_dir / "promotion_gate_summary.csv").exists()
    assert (run_dir / "summary.json").exists()
    assert (tmp_path / "gate.md").exists()
