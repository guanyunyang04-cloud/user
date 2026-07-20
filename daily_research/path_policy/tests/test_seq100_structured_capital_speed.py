from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from daily_research.path_policy import seq100_finite_capital_backtest as finite
from daily_research.path_policy import seq100_structured_capital_speed as speed


def test_job_grid_covers_top1_top3_all_fixed_days_and_own_exits() -> None:
    jobs = speed._job_specs(("model_a", "model_b"))
    assert len(jobs) == 2_684
    assert {job.fixed_day for job in jobs if job.policy_kind == "fixed"} == set(
        range(2, 61)
    )
    assert {job.policy_kind for job in jobs} == {"fixed", "model_plan", "rolling"}
    assert {job.top_k for job in jobs} == {1, 3}


def test_top_k_book_changes_only_daily_selection() -> None:
    book = finite.ForecastBook("source")
    symbols = np.asarray([0, 1, 2, 3], dtype=np.int32)
    book.add_day(
        date_idx=10,
        symbol_idx=symbols,
        score=np.asarray([0.2, 0.9, 0.5, 0.1]),
        planned_day=np.asarray([10, 20, 30, 40], dtype=np.int16),
    )
    top1 = speed._top_k_book(book, top_k=1, profile_name="top1")
    top3 = speed._top_k_book(book, top_k=3, profile_name="top3")
    assert top1.days[10].top3_symbol_idx == (1,)
    assert top3.days[10].top3_symbol_idx == (1, 2, 0)
    assert top1.days[10].score is book.days[10].score
    assert top3.days[10].planned_day is book.days[10].planned_day


def _metric_row(
    *, model_id: str, policy_kind: str, policy_name: str, growth: float, cost: str
) -> dict[str, object]:
    return {
        "model_id": model_id,
        "top_k": 1,
        "policy_name": policy_name,
        "policy_kind": policy_kind,
        "fixed_day": 10 if policy_kind == "fixed" else None,
        "slot_count": 1,
        "cost_scenario": cost,
        "annualized_log_growth": growth,
        "signal_period_cagr_trading_days": float(np.expm1(growth)),
        "signal_period_total_return": 1.0,
        "liquidated_ending_equity_cny": 2_000_000.0,
        "signal_period_maximum_drawdown": -0.2,
        "worst_calendar_year_return": 0.1,
        "mean_occupied_sessions": 10.0,
        "mean_signal_capital_utilization": 0.8,
        "skipped_no_slot_signal_count": 10,
        "closed_trade_count": 100,
        "net_pnl_per_deployed_capital_session": 0.002,
        "signal_session_count": 727,
    }


def test_selection_uses_all_three_years_and_fixed_fallback() -> None:
    rows = [
        _metric_row(
            model_id="model_a",
            policy_kind="fixed",
            policy_name="fixed_d10",
            growth=0.50,
            cost="base",
        ),
        _metric_row(
            model_id="model_a",
            policy_kind="rolling",
            policy_name="rolling_path",
            growth=0.40,
            cost="base",
        ),
        _metric_row(
            model_id="model_b",
            policy_kind="fixed",
            policy_name="fixed_d10",
            growth=0.30,
            cost="base",
        ),
        _metric_row(
            model_id="model_b",
            policy_kind="rolling",
            policy_name="rolling_path",
            growth=0.60,
            cost="base",
        ),
    ]
    for row in list(rows):
        rows.append(
            {
                **row,
                "cost_scenario": "double_slippage",
                "annualized_log_growth": float(row["annualized_log_growth"]) - 0.05,
            }
        )
    descriptors = {
        "model_a": {"variant": {"variant_id": "model_a"}},
        "model_b": {"variant": {"variant_id": "model_b"}},
    }
    result = speed._selection_summary(pd.DataFrame(rows), descriptors=descriptors)
    assert result["selection_years"] == [2023, 2024, 2025]
    assert result["selection_year_role"] == "symmetric_and_simultaneous"
    assert result["global_growth_winner"]["model_id"] == "model_b"
    exit_rows = {row["model_id"]: row for row in result["own_exit_value"]}
    assert exit_rows["model_a"]["selected_execution"] == "fixed_exit_fallback"
    assert exit_rows["model_b"]["selected_execution"] == "own_model_exit"
    assert result["cross_model_hybrids_selection_allowed"] is False

