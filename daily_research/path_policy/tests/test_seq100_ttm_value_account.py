from __future__ import annotations

import numpy as np

from daily_research.path_policy import seq100_ttm_value_account as ttm


def test_contract_rejects_latest_period_fcf_as_ttm_evidence() -> None:
    study, _ = ttm.load_study()

    assert study["epistemic_contract"][
        "the_previous_latest_period_fcf_result_is_not_accepted_as_ttm_evidence"
    ]
    assert study["epistemic_contract"][
        "52_week_low_or_distance_from_low_reward_is_forbidden"
    ]
    assert tuple(study["variants"]) == ttm.VARIANTS


def test_ttm_formula_uses_annual_or_ytd_bridge() -> None:
    result = ttm._ttm_formula(
        np.array([100.0, 30.0, 70.0, 90.0]),
        np.array([4.0, 1.0, 2.0, 3.0]),
        np.array([80.0, 80.0, 80.0, 80.0]),
        np.array([np.nan, 20.0, 50.0, 60.0]),
    )

    np.testing.assert_allclose(result, [100.0, 90.0, 100.0, 110.0])


def test_ttm_formula_requires_all_interim_components() -> None:
    result = ttm._ttm_formula(
        np.array([30.0]),
        np.array([1.0]),
        np.array([np.nan]),
        np.array([20.0]),
    )

    assert np.isnan(result[0])
