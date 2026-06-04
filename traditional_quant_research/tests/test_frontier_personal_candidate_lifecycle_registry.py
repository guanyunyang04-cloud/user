from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from traditional_quant_research.experiments.frontier_personal_candidate_lifecycle_registry import (
    build_lifecycle_registry,
    build_lifecycle_stage_counts,
    run_frontier_personal_candidate_lifecycle_registry,
)


def _candidates() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "candidate_id": "candidate_a",
                "tracking_status": "paper_tracking_bootstrapped",
                "current_level": "personal_backtest_candidate",
                "target_next_level": "personal_paper_candidate",
                "signal": "signal_a",
                "constraint_variant": "capital_scaled",
                "exposure_penalty_strength": 0.25,
                "fee_bps": 30.0,
                "impact_bps_per_1pct": 10.0,
                "personal_capital_amount": 1_000_000.0,
                "horizon": 20,
                "rebalance_frequency": "monthly",
                "top_n": 200,
                "mean_annualized_return": 0.10,
                "min_annualized_return": -0.18,
                "positive_year_rate": 0.60,
                "worst_max_drawdown": -0.14,
                "total_periods": 60,
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
                "fee_bps": 30.0,
                "impact_bps_per_1pct": 10.0,
                "personal_capital_amount": 1_000_000.0,
            }
        ]
    )


def _plan_calendar() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "candidate_id": "candidate_a",
                "observation_number": 1,
                "record_status": "future_observation_incomplete",
                "expected_signal_period_end": "2026-06-30",
                "actual_signal_date": "",
            },
            {
                "candidate_id": "candidate_a",
                "observation_number": 2,
                "record_status": "future_observation_incomplete",
                "expected_signal_period_end": "2026-07-31",
                "actual_signal_date": "",
            },
        ]
    )


def _review(*, recommendation: str, promotion_level: str) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "candidate_id": "candidate_a",
                "promotion_level": promotion_level,
                "recommendation": recommendation,
                "completed_periods": 0 if recommendation == "continue_paper_tracking" else 6,
                "tracking_days": 0 if recommendation == "continue_paper_tracking" else 151,
                "cumulative_paper_return": 0.05,
                "cumulative_excess_return": 0.02,
                "worst_drawdown": -0.04,
                "failed_gates": "",
                "strategy_candidate": False,
            }
        ]
    )


def _bootstrap_summary() -> dict[str, object]:
    return {"run_id": "bootstrap_run", "snapshot_id": "baostock_v2_fixture"}


def _plan_summary() -> dict[str, object]:
    return {"run_id": "plan_run", "bootstrap_run_id": "bootstrap_run"}


def _review_summary() -> dict[str, object]:
    return {"run_id": "review_run", "bootstrap_run_id": "bootstrap_run"}


def test_registry_bootstrap_only_requests_plan_creation() -> None:
    registry = build_lifecycle_registry(
        _candidates(),
        _protocol(),
        pd.DataFrame(),
        pd.DataFrame(),
        bootstrap_summary=_bootstrap_summary(),
    )

    row = registry.iloc[0]
    assert row["lifecycle_level"] == "personal_backtest_candidate"
    assert row["lifecycle_status"] == "tracking_bootstrapped"
    assert row["next_action"] == "create_paper_tracking_plan"
    assert bool(row["personal_paper_candidate"]) is False
    assert bool(row["strategy_candidate"]) is False


def test_registry_plan_ready_requests_observation_fill() -> None:
    plan_calendar = _plan_calendar()
    plan_calendar["actual_signal_date"] = pd.NA
    registry = build_lifecycle_registry(
        _candidates(),
        _protocol(),
        plan_calendar,
        pd.DataFrame(),
        bootstrap_summary=_bootstrap_summary(),
        plan_summary=_plan_summary(),
        plan_context_status="plan_matched",
    )

    row = registry.iloc[0]
    assert row["lifecycle_status"] == "tracking_plan_ready"
    assert row["next_action"] == "fill_next_paper_observation"
    assert row["planned_observation_count"] == 2
    assert row["future_observation_pending_count"] == 2
    assert row["next_expected_signal_period_end"] == "2026-06-30"


def test_registry_review_continue_overrides_plan_ready() -> None:
    registry = build_lifecycle_registry(
        _candidates(),
        _protocol(),
        _plan_calendar(),
        _review(recommendation="continue_paper_tracking", promotion_level="personal_backtest_candidate"),
        bootstrap_summary=_bootstrap_summary(),
        plan_summary=_plan_summary(),
        review_summary=_review_summary(),
        plan_context_status="plan_matched",
        review_context_status="review_matched",
    )

    row = registry.iloc[0]
    assert row["lifecycle_level"] == "personal_backtest_candidate"
    assert row["lifecycle_status"] == "paper_tracking_incomplete"
    assert row["next_action"] == "continue_paper_tracking"


def test_registry_review_can_promote_personal_paper_but_not_strategy() -> None:
    registry = build_lifecycle_registry(
        _candidates(),
        _protocol(),
        _plan_calendar(),
        _review(recommendation="promote_personal_paper_candidate", promotion_level="personal_paper_candidate"),
        bootstrap_summary=_bootstrap_summary(),
        review_summary=_review_summary(),
        plan_context_status="plan_matched",
        review_context_status="review_matched",
    )
    stage_counts = build_lifecycle_stage_counts(registry)

    row = registry.iloc[0]
    assert row["lifecycle_level"] == "personal_paper_candidate"
    assert row["lifecycle_status"] == "paper_evidence_passed"
    assert row["next_action"] == "prepare_personal_trading_review"
    assert bool(row["personal_paper_candidate"]) is True
    assert bool(row["strategy_candidate"]) is False
    assert stage_counts.iloc[0]["candidate_count"] == 1


def test_registry_review_downgrades_failed_paper_candidate() -> None:
    registry = build_lifecycle_registry(
        _candidates(),
        _protocol(),
        _plan_calendar(),
        _review(recommendation="downgrade_to_personal_research", promotion_level="personal_research/backtest_only"),
        bootstrap_summary=_bootstrap_summary(),
        review_summary=_review_summary(),
        plan_context_status="plan_matched",
        review_context_status="review_matched",
    )

    row = registry.iloc[0]
    assert row["lifecycle_level"] == "personal_research/backtest_only"
    assert row["lifecycle_status"] == "paper_evidence_failed"
    assert row["next_action"] == "rebuild_or_retire_candidate"


def test_run_lifecycle_registry_writes_artifacts(tmp_path: Path) -> None:
    bootstrap_dir = tmp_path / "bootstrap"
    plan_dir = tmp_path / "plan"
    review_dir = tmp_path / "review"
    bootstrap_dir.mkdir()
    plan_dir.mkdir()
    review_dir.mkdir()

    _candidates().to_csv(bootstrap_dir / "paper_tracking_candidates.csv", index=False)
    _protocol().to_csv(bootstrap_dir / "paper_tracking_protocol.csv", index=False)
    (bootstrap_dir / "summary.json").write_text(json.dumps(_bootstrap_summary()), encoding="utf-8")
    _plan_calendar().to_csv(plan_dir / "paper_tracking_plan_calendar.csv", index=False)
    (plan_dir / "summary.json").write_text(json.dumps(_plan_summary()), encoding="utf-8")
    _review(recommendation="continue_paper_tracking", promotion_level="personal_backtest_candidate").to_csv(
        review_dir / "paper_tracking_review_summary.csv",
        index=False,
    )
    (review_dir / "summary.json").write_text(json.dumps(_review_summary()), encoding="utf-8")

    result = run_frontier_personal_candidate_lifecycle_registry(
        bootstrap_run_dir=bootstrap_dir,
        plan_run_dir=plan_dir,
        review_run_dir=review_dir,
        output_dir=tmp_path / "out",
        write_research_log=True,
        research_log_path=tmp_path / "registry.md",
    )

    run_dir = Path(result["run_dir"])
    assert result["decision"] == "continue_paper_tracking"
    assert result["candidate_count"] == 1
    assert result["personal_backtest_candidate_count"] == 1
    assert result["personal_paper_candidate_count"] == 0
    assert result["strategy_candidate_count"] == 0
    assert (run_dir / "personal_candidate_lifecycle_registry.csv").exists()
    assert (run_dir / "personal_candidate_lifecycle_stage_counts.csv").exists()
    assert (run_dir / "summary.json").exists()
    assert (tmp_path / "registry.md").exists()
