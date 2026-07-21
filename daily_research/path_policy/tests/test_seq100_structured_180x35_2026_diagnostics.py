from __future__ import annotations

import math

import numpy as np
import pandas as pd

from daily_research.path_policy import seq100_structured_180x35_2026 as study
from daily_research.path_policy import seq100_structured_180x35_2026_diagnostics as diagnostics


def test_diagnostic_scope_does_not_expand_the_preselected_study() -> None:
    assert study.MODEL_ORDER == (study.MODEL_2026_180,)
    assert diagnostics.MODEL_ORDER == (diagnostics.MODEL_2025, diagnostics.MODEL_2026)


def test_execution_scan_has_d2_d60_and_two_model_exit_policies() -> None:
    assert len(diagnostics.SCAN_POLICIES) == 61
    assert [policy.fixed_day for policy in diagnostics.SCAN_POLICIES[:3]] == [2, 3, 4]
    assert diagnostics.SCAN_POLICIES[58].fixed_day == 60
    assert [policy.kind for policy in diagnostics.SCAN_POLICIES[-2:]] == [
        "model_plan",
        "rolling",
    ]


def test_log_growth_uses_the_common_signal_session_denominator() -> None:
    actual = diagnostics._log_growth(0.10, 48)
    expected = math.log1p(0.10) * 252.0 / 48.0
    assert actual == expected


def test_forecast_book_keeps_all_scores_but_selects_requested_top_k() -> None:
    frame = pd.DataFrame(
        {
            "date_idx": [10, 10, 10],
            "symbol_idx": [2, 1, 3],
            "score": [0.5, 0.5, 0.4],
            "predicted_exit_day": [7, 8, 9],
        }
    )
    book = diagnostics._forecast_book(frame, top_k=1)
    day = book.days[10]
    assert len(day.symbol_idx) == 3
    assert day.top3_symbol_idx == (1,)
    np.testing.assert_array_equal(day.planned_day, np.asarray([8, 7, 9]))


def test_best_scan_rows_keeps_primary_and_liquidated_objectives_separate() -> None:
    frame = pd.DataFrame(
        [
            {
                "top_k": 1,
                "slot_count": 1,
                "cost_scenario": "double_slippage",
                "policy_name": "fixed_d7",
                "policy_kind": "fixed",
                "fixed_day": 7,
                "signal_period_annualized_log_growth": 0.4,
                "liquidated_annualized_log_growth": 0.1,
                "signal_period_total_return": 0.08,
                "liquidated_total_return": 0.03,
                "signal_period_maximum_drawdown": -0.1,
                "full_path_maximum_drawdown": -0.2,
                "closed_trade_count": 4,
                "winning_trade_rate": 0.5,
                "mean_occupied_sessions": 7.0,
            },
            {
                "top_k": 1,
                "slot_count": 1,
                "cost_scenario": "double_slippage",
                "policy_name": "fixed_d20",
                "policy_kind": "fixed",
                "fixed_day": 20,
                "signal_period_annualized_log_growth": 0.2,
                "liquidated_annualized_log_growth": 0.3,
                "signal_period_total_return": 0.04,
                "liquidated_total_return": 0.09,
                "signal_period_maximum_drawdown": -0.1,
                "full_path_maximum_drawdown": -0.2,
                "closed_trade_count": 2,
                "winning_trade_rate": 0.5,
                "mean_occupied_sessions": 20.0,
            },
        ]
    )
    best = diagnostics._best_scan_rows(frame)
    assert [row["policy_name"] for row in best] == ["fixed_d7", "fixed_d20"]

