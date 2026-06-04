from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from traditional_quant_research.experiments.frontier_personal_paper_tracking_plan import (
    FUTURE_RECORD_STATUS,
    PLAN_STATUS_READY,
    build_historical_context,
    build_paper_tracking_live_log_starter,
    build_paper_tracking_plan_calendar,
    run_frontier_personal_paper_tracking_plan,
)
from traditional_quant_research.experiments.frontier_personal_paper_tracking_review import evaluate_paper_tracking_log


def _candidates() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "candidate_id": "candidate_a",
                "tracking_status": "paper_tracking_bootstrapped",
                "current_level": "personal_backtest_candidate",
                "target_next_level": "personal_paper_candidate",
                "signal": "multifactor_rolling_ic_weighted_score",
                "constraint_variant": "capital_scaled",
                "exposure_penalty_strength": 0.25,
                "fee_bps": 30.0,
                "impact_bps_per_1pct": 10.0,
                "personal_capital_amount": 1_000_000.0,
                "horizon": 20,
                "rebalance_frequency": "monthly",
                "top_n": 200,
                "buffer_multiplier": 3.0,
            }
        ]
    )


def _protocol() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "candidate_id": "candidate_a",
                "rebalance_frequency": "monthly",
                "horizon": 20,
                "top_n": 200,
                "buffer_multiplier": 3.0,
                "fee_bps": 30.0,
                "impact_bps_per_1pct": 10.0,
                "personal_capital_amount": 1_000_000.0,
                "min_tracking_periods": 6,
                "min_tracking_days": 120,
                "max_paper_drawdown": -0.20,
                "max_single_period_loss": -0.12,
                "min_paper_excess_return": -0.02,
            }
        ]
    )


def _bootstrap_summary(combined_dir: Path) -> dict[str, object]:
    return {
        "run_id": "bootstrap_run",
        "combined_run_dir": str(combined_dir),
        "snapshot_id": "baostock_v2_fixture",
    }


def _combined_summary() -> dict[str, object]:
    return {
        "run_id": "combined_run",
        "final_end_date": "2026-06-01",
        "rebalance_frequency": "monthly",
        "horizon": 20,
        "top_n": 200,
        "buffer_multiplier": 3.0,
    }


def _write_combined_trades(combined_dir: Path) -> None:
    combined_dir.mkdir(parents=True)
    pd.DataFrame(
        [
            {
                "eval_year": 2026,
                "signal": "multifactor_rolling_ic_weighted_score",
                "constraint_variant": "capital_scaled",
                "top_n": 200,
                "exposure_penalty_strength": 0.25,
                "signal_date": "2026-05-29",
                "entry_date": "2026-06-01",
                "exit_date": "2026-06-01",
                "holdings": 198,
                "requested_holdings": 200,
                "blocked_entry_count": 2,
                "entry_limit_up_count": 2,
                "exit_delayed_count": 0,
                "exit_limit_down_count": 0,
                "turnover": 0.75,
                "net_return": 0.01,
                "capital_scale": 0.7,
            }
        ]
    ).to_csv(combined_dir / "combined_constraint_trades.csv", index=False)
    (combined_dir / "summary.json").write_text(json.dumps(_combined_summary()), encoding="utf-8")


def test_plan_calendar_generates_future_monthly_rows_without_promotion(tmp_path: Path) -> None:
    combined_dir = tmp_path / "combined"
    _write_combined_trades(combined_dir)
    historical = build_historical_context(_candidates(), _bootstrap_summary(combined_dir))

    plan = build_paper_tracking_plan_calendar(
        _candidates(),
        _protocol(),
        _bootstrap_summary(combined_dir),
        _combined_summary(),
        historical,
        plan_periods=3,
    )

    assert len(plan) == 3
    assert plan["expected_signal_period_end"].tolist() == ["2026-06-30", "2026-07-31", "2026-08-31"]
    assert set(plan["tracking_plan_status"]) == {PLAN_STATUS_READY}
    assert set(plan["record_status"]) == {FUTURE_RECORD_STATUS}
    assert bool(plan["personal_paper_candidate"].any()) is False
    assert bool(plan["strategy_candidate"].any()) is False
    assert plan.loc[0, "last_historical_selected_count"] == 198


def test_live_log_starter_remains_incomplete_under_review(tmp_path: Path) -> None:
    combined_dir = tmp_path / "combined"
    _write_combined_trades(combined_dir)
    historical = build_historical_context(_candidates(), _bootstrap_summary(combined_dir))
    plan = build_paper_tracking_plan_calendar(
        _candidates(),
        _protocol(),
        _bootstrap_summary(combined_dir),
        _combined_summary(),
        historical,
        plan_periods=2,
    )
    live_log = build_paper_tracking_live_log_starter(plan)

    review = evaluate_paper_tracking_log(_candidates(), _protocol(), live_log)

    row = review.iloc[0]
    assert row["promotion_level"] == "personal_backtest_candidate"
    assert row["recommendation"] == "continue_paper_tracking"
    assert row["completed_periods"] == 0
    assert bool(row["strategy_candidate"]) is False


def test_run_plan_writes_operational_artifacts(tmp_path: Path) -> None:
    bootstrap_dir = tmp_path / "bootstrap"
    combined_dir = tmp_path / "combined"
    bootstrap_dir.mkdir()
    _write_combined_trades(combined_dir)
    _candidates().to_csv(bootstrap_dir / "paper_tracking_candidates.csv", index=False)
    _protocol().to_csv(bootstrap_dir / "paper_tracking_protocol.csv", index=False)
    (bootstrap_dir / "summary.json").write_text(json.dumps(_bootstrap_summary(combined_dir)), encoding="utf-8")

    result = run_frontier_personal_paper_tracking_plan(
        bootstrap_run_dir=bootstrap_dir,
        output_dir=tmp_path / "out",
        plan_periods=2,
        write_research_log=True,
        research_log_path=tmp_path / "plan.md",
    )

    run_dir = Path(result["run_dir"])
    assert result["decision"] == PLAN_STATUS_READY
    assert result["candidate_count"] == 1
    assert result["plan_calendar_rows"] == 2
    assert result["live_log_starter_rows"] == 2
    assert result["personal_paper_candidate_count"] == 0
    assert result["strategy_candidate_count"] == 0
    assert (run_dir / "paper_tracking_historical_context.csv").exists()
    assert (run_dir / "paper_tracking_plan_calendar.csv").exists()
    assert (run_dir / "paper_tracking_live_log_starter.csv").exists()
    assert (tmp_path / "plan.md").exists()
