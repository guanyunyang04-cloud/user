from __future__ import annotations

import numpy as np
import pandas as pd

from daily_research.path_policy import seq100_qver_confirmation_effect as effect


def test_contract_freezes_user_weights_and_retrospective_boundary() -> None:
    study, _ = effect.load_study()

    assert study["score_contract"]["family_weights"] == effect.FRAMEWORK_WEIGHTS
    assert study["selection"]["top_k"] == 10
    assert study["selection"]["maximum_names_per_pit_industry"] == 2
    assert study["decision_boundary"]["score_or_threshold_optimization_performed"] is False
    assert study["epistemic_contract"]["2026_inputs_and_outcomes_are_forbidden"]


def test_combine_score_uses_exact_seven_family_weights() -> None:
    frame = pd.DataFrame(
        {
            "normalized_valuation_score": [1.0],
            "actual_earnings_improvement_score": [0.9],
            "earnings_revision_framework_score": [0.8],
            "company_quality_score": [0.7],
            "earnings_quality_cashflow_score": [0.6],
            "price_confirmation_framework_score": [0.5],
            "governance_risk_score": [0.4],
        }
    )

    result = effect._combine_score(frame)

    expected = (
        0.25 * 1.0
        + 0.20 * 0.9
        + 0.15 * 0.8
        + 0.20 * 0.7
        + 0.10 * 0.6
        + 0.05 * 0.5
        + 0.05 * 0.4
    )
    assert np.isclose(result[0], expected)


def test_rating_boundaries_match_supplied_framework() -> None:
    assert effect._rating(0.85) == "A+"
    assert effect._rating(0.80) == "A"
    assert effect._rating(0.75) == "A-"
    assert effect._rating(0.68) == "B+"
    assert effect._rating(0.60) == "B"
    assert effect._rating(0.5999) == "C"


def test_selection_respects_top_k_and_industry_cap() -> None:
    rows = 30
    frame = pd.DataFrame(
        {
            "framework_score": np.linspace(1.0, 0.0, rows),
            "eligible__framework": np.ones(rows, dtype=bool),
            "candidate_id": np.arange(rows, dtype=np.int64),
            "industry_code": np.repeat(np.arange(10), 3),
        }
    )

    chosen = effect._select_indices(frame, top_k=10, industry_cap=2)

    selected = frame.iloc[chosen]
    assert len(selected) == 10
    assert selected.groupby("industry_code").size().max() <= 2


def test_valuation_quadrants_separate_eps_and_multiple_outcomes() -> None:
    assert effect._valuation_quadrant(0.10, 0.10) == "eps_and_multiple_double_hit"
    assert (
        effect._valuation_quadrant(0.10, -0.10)
        == "earnings_delivered_multiple_compressed"
    )
    assert (
        effect._valuation_quadrant(-0.10, 0.10)
        == "earnings_weakened_multiple_supported"
    )
    assert (
        effect._valuation_quadrant(-0.10, -0.10)
        == "earnings_and_multiple_double_miss"
    )
    assert effect._valuation_quadrant(0.01, 0.01) == "mixed_or_stable"
