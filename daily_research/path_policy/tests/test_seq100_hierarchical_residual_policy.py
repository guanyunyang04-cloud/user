from __future__ import annotations

import numpy as np

from daily_research.path_policy import seq100_hierarchical_residual_policy as policy


def test_study_freezes_cash_gate_and_reserved_confirmation() -> None:
    study, _ = policy.load_study()

    assert study["policy"]["primary"] == policy.PRIMARY_POLICY
    assert study["evaluation"]["development_years"] == list(policy.DEVELOPMENT_YEARS)
    assert study["evaluation"]["retrospective_confirmation_years"] == list(
        policy.CONFIRMATION_YEARS
    )
    assert study["epistemic_contract"]["cash_is_an_admissible_action"] is True
    assert study["decision_boundary"]["account_optimization_performed"] is False


def test_selection_vetoes_highest_loss_risk_before_residual_ranking() -> None:
    result = policy._select_day(
        residual_score=np.arange(10.0, 0.0, -1.0),
        bad_tail_score=np.array([10.0, 9.0] + [0.0] * 8),
        hierarchical_score=np.full(10, 0.01),
        identity=np.arange(10),
        top_k=3,
        veto_fraction=0.2,
        stress_cost=0.006,
    )

    np.testing.assert_array_equal(result["residual_top48"], [0, 1, 2])
    np.testing.assert_array_equal(result["residual_veto10_top48"], [2, 3, 4])
    np.testing.assert_array_equal(result[policy.PRIMARY_POLICY], [2, 3, 4])


def test_primary_policy_holds_cash_when_basket_forecast_misses_cost() -> None:
    result = policy._select_day(
        residual_score=np.arange(10.0, 0.0, -1.0),
        bad_tail_score=np.arange(10.0),
        hierarchical_score=np.full(10, 0.005),
        identity=np.arange(10),
        top_k=3,
        veto_fraction=0.1,
        stress_cost=0.006,
    )

    assert len(result[policy.PRIMARY_POLICY]) == 0


def test_selection_tie_break_is_identity_stable() -> None:
    result = policy._select_day(
        residual_score=np.ones(5),
        bad_tail_score=np.zeros(5),
        hierarchical_score=np.full(5, 0.01),
        identity=np.array([50, 10, 40, 20, 30]),
        top_k=2,
        veto_fraction=0.0,
        stress_cost=0.006,
    )

    np.testing.assert_array_equal(result["residual_top48"], [1, 3])


def test_development_gate_requires_every_predeclared_check() -> None:
    summary = {
        "annual": [
            {"gross_net_stress": value}
            for value in (0.01, 0.01, 0.01, 0.01, -0.01, -0.01)
        ],
        "gross_net_stress": {"lcb_95": 0.001, "block": {"lcb_95": 0.001}},
        "hedged_residual_net_stress": {"lcb_95": 0.001},
        "mean_selected_count_when_invested": 48.0,
        "observed_outcome_fraction": 0.99,
    }
    gate = {
        "minimum_positive_years": 4,
        "minimum_mean_selected_count_when_invested": 40,
        "minimum_observed_outcome_fraction": 0.98,
        "hedged_residual_stress_net_hac_lower_bound_strictly_positive": True,
    }

    passed = policy._evaluate_gate(summary, gate=gate)
    assert passed["passed"] is True

    summary["observed_outcome_fraction"] = 0.97
    failed = policy._evaluate_gate(summary, gate=gate)
    assert failed["passed"] is False
    assert failed["checks"]["minimum_observed_outcome_fraction"] is False
