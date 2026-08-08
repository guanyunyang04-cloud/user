from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pandas as pd

from daily_research.path_policy import seq100_causal_value_policy as policy


def test_contract_forbids_low_price_reward_and_freezes_coverage_control() -> None:
    study, _ = policy.load_study()

    assert study["epistemic_contract"][
        "52_week_low_or_distance_from_low_reward_is_forbidden"
    ]
    assert "value_analyst_covered_control" in study["signals"]["policies"]
    assert study["decision_boundary"]["account_replay_performed"] is False


def test_month_end_rows_use_last_pool_date_and_never_2026() -> None:
    panel = SimpleNamespace(
        row_index=pd.DataFrame(
            {
                "date_idx": [1, 1, 2, 2, 3, 3],
                "trade_date": [
                    "2025-01-30",
                    "2025-01-30",
                    "2025-01-31",
                    "2025-01-31",
                    "2025-02-28",
                    "2025-02-28",
                ],
            }
        ),
        date_idx=np.array([1, 1, 2, 2, 3, 3]),
        trade_date=np.array(
            [
                "2025-01-30",
                "2025-01-30",
                "2025-01-31",
                "2025-01-31",
                "2025-02-28",
                "2025-02-28",
            ]
        ),
    )

    np.testing.assert_array_equal(policy._month_end_rows(panel), [2, 3, 4, 5])


def test_industry_capped_selection_is_deterministic() -> None:
    selected = policy._select_industry_capped(
        score=np.array([0.9, 0.8, 0.7, 0.6, 0.5]),
        eligible=np.ones(5, dtype=bool),
        candidate_id=np.array([50, 10, 40, 20, 30]),
        industry_code=np.array([1, 1, 1, 2, 3]),
        top_k=3,
        industry_cap=1,
    )

    np.testing.assert_array_equal(selected, [0, 3, 4])


def test_peer_targets_only_rerate_up_to_industry_reference() -> None:
    quarter, median, edge = policy._peer_target_ratios(
        np.array([5.0, 10.0, 20.0, 30.0]),
        np.array([0.5, 1.0, 2.0, 3.0]),
        np.ones(4, dtype=int),
        minimum_group_size=2,
    )

    assert np.all(quarter >= 1.0)
    assert np.all(median >= quarter)
    assert np.all(edge >= median)
    assert median[-1] == 1.0


def test_price_confirmation_is_a_veto_not_a_low_price_score() -> None:
    frame = pd.DataFrame(
        {
            "value_score": [0.8, 0.8],
            "actual_improvement_score": [0.7, 0.7],
            "revision_score": [0.6, 0.6],
            "quality_score": [0.5, 0.5],
            "analyst_covered": [True, True],
            "price_confirmed": [False, True],
        }
    )

    score, eligible = policy._policy_score_and_eligibility(
        frame, "value_actual_revision_price_confirmed"
    )

    np.testing.assert_allclose(score, [0.7, 0.7])
    np.testing.assert_array_equal(eligible, [False, True])
