from __future__ import annotations

import numpy as np
import pandas as pd

from daily_research.path_policy import seq100_stock_bad_tail as bad_tail


def test_study_freezes_primary_target_and_separates_decision_layer() -> None:
    study = bad_tail.load_study()

    assert study["targets"]["primary_horizon"] == 20
    assert study["targets"]["diagnostic_horizon"] == 10
    assert tuple(study["targets"]["quantile_probabilities"]) == (
        0.05,
        0.10,
        0.25,
        0.50,
        0.75,
        0.90,
    )
    assert study["decision_boundary"]["portfolio_selection_performed"] is False
    assert study["decision_boundary"]["account_optimization_performed"] is False
    assert "PIT concatenation is excluded" in study["features"]["pit_policy"]


def test_mid_percentile_rank_handles_ties_and_missing_neutrally() -> None:
    result = bad_tail._mid_percentile_rank(
        np.array([1.0, 2.0, 2.0, 4.0, np.nan])
    )

    np.testing.assert_allclose(result[:4], [0.125, 0.5, 0.5, 0.875])
    assert result[4] == 0.5


def test_monotone_rearrangement_removes_crossing_without_row_mixing() -> None:
    raw = np.array([[3.0, 1.0, 2.0], [-2.0, -1.0, 0.0]])

    assert bad_tail._quantile_crossing_rate(raw) == 0.5
    rearranged = bad_tail._monotone_rearrange(raw)

    assert bad_tail._quantile_crossing_rate(rearranged) == 0.0
    np.testing.assert_array_equal(rearranged[0], [1.0, 2.0, 3.0])
    np.testing.assert_array_equal(rearranged[1], raw[1])


def test_quantile_crps_approximation_is_nonnegative_and_center_sensitive() -> None:
    probabilities = bad_tail.QUANTILE_PROBABILITIES
    centered = np.zeros((1, len(probabilities)))
    displaced = np.ones((1, len(probabilities)))

    centered_score = bad_tail._quantile_crps_approximation(
        np.array([0.0]), centered
    )[0]
    displaced_score = bad_tail._quantile_crps_approximation(
        np.array([0.0]), displaced
    )[0]

    assert centered_score == 0.0
    assert displaced_score > centered_score


def test_discrete_hazard_expansion_distinguishes_event_and_censoring() -> None:
    subject, delay, event = bad_tail._expand_discrete_hazard(
        np.array([0, 2, 21]), retry_days=20
    )

    assert len(subject) == 1 + 3 + 21
    np.testing.assert_array_equal(delay[subject == 0], [0])
    np.testing.assert_array_equal(event[subject == 0], [1])
    np.testing.assert_array_equal(delay[subject == 1], [0, 1, 2])
    np.testing.assert_array_equal(event[subject == 1], [0, 0, 1])
    assert event[subject == 2].sum() == 0


def test_hazard_mass_and_censoring_probability_sum_to_one() -> None:
    hazard = np.array([[0.5, 0.25], [0.1, 0.2]])
    mass, censored = bad_tail._hazard_mass(hazard)

    np.testing.assert_allclose(mass.sum(axis=1) + censored, 1.0)
    np.testing.assert_allclose(mass[0], [0.5, 0.125])
    assert censored[0] == pytest_approx(0.375)


def test_top_k_uses_only_score_and_deterministic_identity_tiebreak() -> None:
    scores = np.array([0.5, 0.9, 0.9, 0.1])
    candidate_ids = np.array([40, 30, 20, 10])

    selected = bad_tail._top_k_indices(scores, candidate_ids, top_k=2)

    np.testing.assert_array_equal(selected, [2, 1])


def test_tail_decile_diagnostic_ranks_full_candidates_before_observation_mask() -> None:
    frame = pd.DataFrame(
        {
            "date_idx": [0] * 10,
            "gross_log_return_observed": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, np.nan],
            "gross_return_observed": [True] * 9 + [False],
            "loss_probability": np.arange(0.1, 1.1, 0.1),
        }
    )

    result = bad_tail._tail_decile_daily(
        frame,
        probability_column="loss_probability",
        model="test",
        horizon=20,
        evaluation_year=2022,
    )

    high = result.loc[result["risk_decile"] == "high_risk_decile"].iloc[0]
    low = result.loc[result["risk_decile"] == "low_risk_decile"].iloc[0]
    assert high["candidate_count"] == 2
    assert high["observed_count"] == 1
    assert high["loss_rate"] == 0.0
    assert low["loss_rate"] == 0.0


def test_development_gate_does_not_promote_residual_distribution_gain() -> None:
    comparisons = pd.DataFrame(
        [
            {
                "family": "distribution",
                "metric": "mean_pinball",
                "horizon": 20,
                "target": "residual_log_return_20",
                "period": "pooled",
                "model": "lightgbm_direct_quantiles",
                "hac_lcb_95": 0.1,
                "block_lcb_95": 0.1,
                "mean_improvement": 0.1,
            },
            {
                "family": "hurdle",
                "metric": "log_score",
                "horizon": 20,
                "target": "gross_return_le_zero",
                "period": "pooled",
                "model": "lightgbm_binary",
                "hac_lcb_95": 0.1,
                "block_lcb_95": 0.1,
                "mean_improvement": 0.1,
            },
            {
                "family": "ranking",
                "metric": "top_k_cost_proxy_return_0060bps",
                "horizon": 20,
                "target": "gross",
                "period": "pooled",
                "model": "lightgbm_mean_gross",
                "hac_lcb_95": 0.1,
                "block_lcb_95": 0.1,
                "mean_improvement": 0.1,
            },
        ]
    )
    reality = pd.DataFrame(
        columns=["family", "horizon", "target", "metric", "p_value"]
    )

    gate = bad_tail._development_gate(
        evaluation_years=bad_tail.DEVELOPMENT_YEARS,
        comparisons=comparisons,
        reality=reality,
        ranking_daily=pd.DataFrame(),
        cost_proxies=(0.003, 0.006),
    )

    assert gate["distribution_pinball_supported"] is False
    assert gate["development_forecast_gate_passed"] is False


def pytest_approx(value: float) -> object:
    import pytest

    return pytest.approx(value)
