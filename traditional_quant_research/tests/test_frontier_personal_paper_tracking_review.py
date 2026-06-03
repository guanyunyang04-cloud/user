from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from traditional_quant_research.experiments.frontier_personal_paper_tracking_review import (
    evaluate_paper_tracking_log,
    run_frontier_personal_paper_tracking_review,
)


def _candidates() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "candidate_id": "candidate_a",
                "current_level": "personal_backtest_candidate",
                "target_next_level": "personal_paper_candidate",
                "tracking_status": "paper_tracking_bootstrapped",
                "signal": "multifactor_rolling_ic_weighted_score",
                "constraint_variant": "baseline",
            }
        ]
    )


def _protocol() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "candidate_id": "candidate_a",
                "min_tracking_periods": 3,
                "min_tracking_days": 60,
                "max_paper_drawdown": -0.20,
                "max_single_period_loss": -0.12,
                "min_paper_excess_return": -0.02,
            }
        ]
    )


def _tracking_log(
    *,
    periods: int = 3,
    start_date: str = "2026-01-01",
    paper_returns: list[float] | None = None,
    excess_returns: list[float] | None = None,
    max_drawdowns: list[float] | None = None,
    missing_execution: bool = False,
) -> pd.DataFrame:
    dates = pd.date_range(start_date, periods=periods, freq="30D")
    paper = paper_returns or [0.02, 0.01, 0.015][:periods]
    excess = excess_returns or [0.01, -0.005, 0.008][:periods]
    drawdowns = max_drawdowns or [-0.02, -0.03, -0.025][:periods]
    rows = []
    for idx, date in enumerate(dates):
        rows.append(
            {
                "record_date": date.date().isoformat(),
                "candidate_id": "candidate_a",
                "signal": "multifactor_rolling_ic_weighted_score",
                "constraint_variant": "baseline",
                "selected_count": 200,
                "blocked_entry_count": 1,
                "entry_limit_up_count": 1,
                "exit_delayed_count": 0,
                "exit_limit_down_count": 0,
                "gross_return": paper[idx] + 0.003,
                "net_return": paper[idx],
                "paper_account_return": paper[idx],
                "benchmark_return": paper[idx] - excess[idx],
                "excess_return": excess[idx],
                "realized_turnover": 0.8,
                "realized_fee_bps": 30.0,
                "max_drawdown_to_date": drawdowns[idx],
            }
        )
    frame = pd.DataFrame(rows)
    if missing_execution:
        frame.loc[0, "selected_count"] = None
    return frame


def test_review_continues_when_paper_sample_is_incomplete() -> None:
    review = evaluate_paper_tracking_log(_candidates(), _protocol(), _tracking_log(periods=2))

    row = review.iloc[0]
    assert row["promotion_level"] == "personal_backtest_candidate"
    assert row["recommendation"] == "continue_paper_tracking"
    assert bool(row["sample_gate"]) is False
    assert "sample_gate" in row["failed_gates"]


def test_review_promotes_personal_paper_candidate_when_rules_pass() -> None:
    review = evaluate_paper_tracking_log(_candidates(), _protocol(), _tracking_log())

    row = review.iloc[0]
    assert row["promotion_level"] == "personal_paper_candidate"
    assert row["recommendation"] == "promote_personal_paper_candidate"
    assert bool(row["sample_gate"]) is True
    assert bool(row["execution_integrity_gate"]) is True
    assert bool(row["paper_return_gate"]) is True
    assert bool(row["drawdown_gate"]) is True
    assert bool(row["single_period_gate"]) is True
    assert bool(row["strategy_candidate"]) is False


def test_review_downgrades_on_drawdown_breach() -> None:
    review = evaluate_paper_tracking_log(
        _candidates(),
        _protocol(),
        _tracking_log(max_drawdowns=[-0.05, -0.22, -0.18]),
    )

    row = review.iloc[0]
    assert row["promotion_level"] == "personal_research/backtest_only"
    assert row["recommendation"] == "downgrade_to_personal_research"
    assert bool(row["drawdown_gate"]) is False
    assert "drawdown_gate" in row["failed_gates"]


def test_review_downgrades_when_complete_sample_has_missing_execution_fields() -> None:
    review = evaluate_paper_tracking_log(_candidates(), _protocol(), _tracking_log(missing_execution=True))

    row = review.iloc[0]
    assert row["promotion_level"] == "personal_research/backtest_only"
    assert row["recommendation"] == "downgrade_to_personal_research"
    assert bool(row["execution_integrity_gate"]) is False
    assert "selected_count" in row["missing_required_fields"]


def test_run_review_writes_artifacts(tmp_path: Path) -> None:
    bootstrap = tmp_path / "bootstrap"
    bootstrap.mkdir()
    _candidates().to_csv(bootstrap / "paper_tracking_candidates.csv", index=False)
    _protocol().to_csv(bootstrap / "paper_tracking_protocol.csv", index=False)
    (bootstrap / "summary.json").write_text(json.dumps({"run_id": "bootstrap_run"}), encoding="utf-8")
    tracking_log = tmp_path / "tracking_log.csv"
    _tracking_log().to_csv(tracking_log, index=False)

    result = run_frontier_personal_paper_tracking_review(
        bootstrap_run_dir=bootstrap,
        tracking_log_path=tracking_log,
        output_dir=tmp_path / "out",
        write_research_log=True,
        research_log_path=tmp_path / "review.md",
    )

    run_dir = Path(result["run_dir"])
    assert result["decision"] == "personal_paper_candidate_ready"
    assert result["personal_paper_candidate_count"] == 1
    assert result["strategy_candidate_count"] == 0
    assert (run_dir / "paper_tracking_review_summary.csv").exists()
    assert (run_dir / "summary.json").exists()
    assert (tmp_path / "review.md").exists()
