from __future__ import annotations

import numpy as np
import pandas as pd

from daily_research.path_policy import seq100_value_staggered_account as account


def test_contract_marks_candidate_posthoc_and_disables_target_exit() -> None:
    study, _ = account.load_study()

    assert study["epistemic_contract"][
        "candidate_was_selected_after_retrospective_factor_comparison"
    ]
    assert study["account"]["valuation_target_exit_used"] is False
    assert study["account"]["sleeve_count"] == 5


def test_cohort_table_assigns_months_to_non_overlapping_sleeves() -> None:
    records = []
    for month, date_idx in enumerate((10, 30, 50)):
        for rank in (1, 2):
            records.append(
                {
                    "selection_rank": rank,
                    "date_idx": date_idx,
                    "trade_date": f"2020-0{month + 1}-28",
                    "calendar_evaluable": True,
                    "outcome_valid": True,
                    "simple_return": 0.02,
                    "exit_offset": 60,
                    "applied_cost": 0.006,
                }
            )
    chosen, cohorts = account._cohort_table(
        pd.DataFrame(records),
        breadth=2,
        sleeve_count=5,
        cost=0.006,
    )

    np.testing.assert_array_equal(cohorts["sleeve"], [0, 1, 2])
    np.testing.assert_allclose(cohorts["stress_net_return"], 0.014)
    assert len(chosen) == 6


def test_account_metrics_report_drawdown_and_calendar_years() -> None:
    dates = np.array(["2020-01-02", "2020-12-31", "2021-12-31"])
    metrics = account._account_metrics(
        dates=dates,
        wealth=np.array([100.0, 120.0, 90.0]),
        invested=np.array([0.0, 60.0, 0.0]),
        position_count=np.array([0, 2, 0]),
        start_idx=0,
        initial_cash=100.0,
    )

    assert metrics["terminal_multiple"] == 0.9
    assert metrics["maximum_drawdown"] == -0.25
    assert metrics["positive_year_count"] == 1
    assert metrics["year_count"] == 2
