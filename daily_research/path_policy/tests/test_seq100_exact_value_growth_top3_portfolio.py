from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from daily_research.path_policy import seq100_exact_value_growth_top3_portfolio as top3
from daily_research.path_policy import seq100_finite_capital_backtest as finite


def test_load_study_freezes_literal_and_overlap_variants() -> None:
    study, path = top3.load_study()
    assert path == Path(top3.DEFAULT_STUDY_PATH).resolve()
    assert study["selection"]["selected_ranks"] == [1, 2, 3]
    literal = [
        row for row in study["strategy_family"] if row["literal_maximum_three_names"]
    ]
    assert {row["name"] for row in literal} == {
        "monthly_rebalance_max3",
        "fixed_d20_max3",
        "fixed_d60_max3",
        "fixed_d120_max3",
    }


def test_cohort_monthly_uses_complete_fixed_denominators() -> None:
    study, _ = top3.load_study()
    rows = []
    for rank in range(1, 11):
        rows.append(
            {
                "trade_date": "2012-01-31",
                "evaluation_year": 2012,
                "selection_rank": rank,
                "legal_net_return_d20": rank / 100.0,
                "legal_net_return_d60": rank / 100.0,
                "legal_net_return_d120": rank / 100.0,
            }
        )
    incomplete = dict(rows[0])
    incomplete["trade_date"] = "2012-02-29"
    paths = pd.DataFrame([*rows, incomplete])
    monthly = top3._cohort_monthly(paths, study)
    top3_month = monthly.loc[
        monthly["trade_date"].eq("2012-01-31")
        & monthly["horizon"].eq(20)
        & monthly["rank_group"].eq("top3")
    ].iloc[0]
    assert bool(top3_month["complete_month"])
    assert np.isclose(top3_month["cash_denominator_net_return"], 0.02)
    incomplete_month = monthly.loc[
        monthly["trade_date"].eq("2012-02-29")
        & monthly["horizon"].eq(20)
        & monthly["rank_group"].eq("top3")
    ].iloc[0]
    assert not bool(incomplete_month["complete_month"])
    assert np.isnan(incomplete_month["cash_denominator_net_return"])


def test_strict_top3_book_removes_lower_ranks_and_uses_next_signal_plan() -> None:
    source = finite.ForecastBook("source", top_k=10, candidate_scan_k=10)
    for date_idx in (100, 121):
        source.add_day(
            date_idx=date_idx,
            symbol_idx=np.arange(10, dtype=np.int64),
            score=np.arange(10, 0, -1, dtype=np.float64),
            planned_day=np.full(10, 60, dtype=np.int16),
        )
    result, audit = top3._strict_top3_book(
        source,
        maximum_date_idx=100,
        monthly_plan=True,
        profile="monthly_top3",
    )
    assert result.top_k == 3
    assert result.candidate_scan_k == 3
    assert result.days[100].ranked_symbol_idx == (0, 1, 2)
    assert tuple(result.days[100].planned_day) == (21, 21, 21)
    assert audit["rank4_or_lower_present"] is False


def test_overlap_requirement_and_decision_rule() -> None:
    assert top3._unconstrained_cohort_slots([0, 19, 38, 57], horizon=60) == 12
    strategies = [
        {"name": "a", "literal_maximum_three_names": True},
        {"name": "b", "literal_maximum_three_names": True},
        {"name": "c", "literal_maximum_three_names": False},
    ]
    metrics = {
        "a": {"annualized_log_growth": 0.08, "signal_period_maximum_drawdown": -0.2},
        "b": {"annualized_log_growth": 0.10, "signal_period_maximum_drawdown": -0.3},
        "c": {"annualized_log_growth": 0.12, "signal_period_maximum_drawdown": -0.1},
    }
    gates = {"a": {"passed": True}, "b": {"passed": False}, "c": {"passed": True}}
    decision = top3._decision(metrics, gates, strategies)
    assert decision["primary_literal_max3_choice"] == "a"
    assert decision["highest_return_literal_max3_variant"] == "b"
    assert decision["lowest_drawdown_literal_max3_variant"] == "a"
    assert decision["best_overlapping_monthly_top3_diagnostic"] == "c"
