from __future__ import annotations

import numpy as np

from daily_research.path_policy import seq100_margin_value_overlay as overlay


def _arguments() -> dict[str, np.ndarray | float | int]:
    return {
        "residual_score": np.array([0.9, 0.8, 0.7, 0.6]),
        "bad_tail_score": np.zeros(4),
        "value_score": np.array([0.2, 0.8, 0.7, 0.6]),
        "candidate_id": np.arange(4),
        "industry_code": np.arange(4),
        "margin_observed": np.ones(4, dtype=bool),
        "top_k": 2,
        "veto_fraction": 0.0,
        "industry_cap": 1,
    }


def test_contract_reuses_frozen_forecasts_and_forbids_low_price_reward() -> None:
    study, _ = overlay.load_study()

    assert study["epistemic_contract"][
        "the_frozen_stock_forecasts_are_reused_without_refit"
    ]
    assert study["epistemic_contract"][
        "52_week_low_or_distance_from_low_reward_is_forbidden"
    ]
    assert study["decision_boundary"]["account_replay_performed"] is False


def test_value_gate_skips_expensive_high_residual_candidate() -> None:
    selected = overlay._variant_selection(
        policy="residual_with_value_median_gate", **_arguments()
    )

    np.testing.assert_array_equal(selected, [1, 2])


def test_frozen_baseline_does_not_change_selection() -> None:
    selected = overlay._variant_selection(
        policy="frozen_residual_baseline", **_arguments()
    )

    np.testing.assert_array_equal(selected, [0, 1])
