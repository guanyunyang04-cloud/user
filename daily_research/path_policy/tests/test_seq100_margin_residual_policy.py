from __future__ import annotations

import numpy as np

from daily_research.path_policy import seq100_margin_residual_policy as policy


def test_study_freezes_expanded_development_before_confirmation() -> None:
    study, _ = policy.load_study()

    assert study["policy"]["primary"] == policy.PRIMARY_POLICY
    assert tuple(study["evaluation"]["new_untouched_development_years"]) == (
        2014,
        2015,
        2016,
    )
    assert study["epistemic_contract"]["margin_balance_growth_is_not_assumed_bullish"]
    assert study["decision_boundary"]["account_replay_performed"] is False


def test_selection_applies_full_pool_risk_veto_then_margin_and_industry_cap() -> None:
    selected = policy._select_industry_capped(
        residual_score=np.arange(10.0, 0.0, -1.0),
        bad_tail_score=np.array([10.0] + [0.0] * 9),
        candidate_id=np.arange(10),
        industry_code=np.array([1, 1, 1, 1, 2, 2, 3, 3, 4, 4]),
        margin_observed=np.array(
            [True, False, True, True, True, True, True, True, True, True]
        ),
        top_k=5,
        veto_fraction=0.1,
        industry_cap=2,
    )

    # Candidate 0 is risk-vetoed, 1 lacks observable margin detail, and the
    # third remaining member of industry 1 is skipped by the cap.
    np.testing.assert_array_equal(selected, [2, 3, 4, 5, 6])


def test_selection_ties_break_on_candidate_identity() -> None:
    selected = policy._select_industry_capped(
        residual_score=np.ones(5),
        bad_tail_score=np.zeros(5),
        candidate_id=np.array([50, 10, 40, 20, 30]),
        industry_code=np.arange(5),
        margin_observed=np.ones(5, dtype=bool),
        top_k=3,
        veto_fraction=0.0,
        industry_cap=4,
    )

    np.testing.assert_array_equal(selected, [1, 3, 4])


def test_gate_requires_absolute_excess_coverage_and_year_stability() -> None:
    summary = {
        "annual": [
            {"gross_net_stress": value}
            for value in (0.01, 0.01, 0.01, 0.01, 0.01, 0.01, -0.01, -0.01, -0.01)
        ],
        "gross_net_stress": {"lcb_95": 0.001, "block": {"lcb_95": 0.001}},
        "selected_excess_gross": {"lcb_95": 0.001},
        "mean_selected_count": 48.0,
        "observed_outcome_fraction": 0.99,
    }
    contract = {
        "minimum_positive_years": 6,
        "minimum_mean_selected_count": 40,
        "minimum_observed_outcome_fraction": 0.98,
        "selected_excess_hac_lower_bound_strictly_positive": True,
    }

    assert policy._gate(summary, contract)["passed"] is True
    summary["selected_excess_gross"]["lcb_95"] = -0.001
    assert policy._gate(summary, contract)["passed"] is False
