from __future__ import annotations

import math

import numpy as np
import pandas as pd

from daily_research.path_policy import seq100_dynamic_oracle_observable_audit as audit


def test_study_keeps_continuous_no_fixed_horizon_contract() -> None:
    study = audit.load_study()
    assert study["target"]["primary"] == "buy_advantage_vs_cash"
    assert study["target"]["fixed_holding_horizon_used"] is False
    assert study["target"]["binary_good_stock_label_used"] is False
    assert study["boundaries"]["causal_policy_evaluated"] is False


def test_midrank_percentile_handles_ties_and_missing_values() -> None:
    values = np.asarray(
        [
            [1.0, 2.0, np.nan],
            [2.0, 2.0, 4.0],
            [3.0, 5.0, 6.0],
            [4.0, np.nan, 8.0],
        ]
    )
    result = audit._midrank_percentile_at_row(values, 1)
    assert np.allclose(result[:2], [0.375, 1.0 / 3.0])
    assert math.isclose(result[2], 1.0 / 6.0)


def test_standardized_correlations_recover_direction() -> None:
    target = np.asarray([-2.0, -1.0, 1.0, 2.0])
    values = np.column_stack([target, -target, np.ones(4)])
    standardized, correlation = audit._standardize_and_correlate(
        values, target, clip=5.0
    )
    assert standardized.shape == values.shape
    assert math.isclose(correlation[0], 1.0)
    assert math.isclose(correlation[1], -1.0)
    assert np.isnan(correlation[2])


def test_family_balanced_neighbor_finds_nearest_failure() -> None:
    standardized = np.asarray(
        [
            [0.0, 0.0, 0.0],
            [0.1, 0.1, 0.1],
            [2.0, 2.0, 2.0],
        ]
    )
    result = audit._nearest_neighbors(
        standardized=standardized,
        target=np.asarray([0.4, -0.1, 0.2]),
        symbol_idx=np.asarray([10, 11, 12]),
        selected_row=0,
        feature_weights=np.asarray([0.25, 0.25, 0.5]),
        neighbor_count=2,
        minimum_coverage=0.5,
    )
    assert result["neighbor_available"] is True
    assert result["nearest_symbol_idx"] == 11
    assert result["failure_neighbor_available"] is True
    assert result["nearest_failure_symbol_idx"] == 11
    assert math.isclose(result["nearest_distance"], 0.01)


def test_hac_and_fdr_helpers_return_finite_ordered_results() -> None:
    values = np.asarray(
        [
            [1.0, 0.0],
            [2.0, 0.0],
            [3.0, 0.0],
            [4.0, 0.0],
        ]
    )
    means, standard_errors = audit._hac_mean(values, lag=1)
    assert np.allclose(means, [2.5, 0.0])
    assert np.isfinite(standard_errors).all()
    adjusted = audit._benjamini_hochberg(np.asarray([0.01, 0.04, 0.03]))
    assert np.allclose(adjusted, [0.03, 0.04, 0.04])
    means = audit._nanmean_columns(
        np.asarray([[1.0, np.nan], [3.0, np.nan]])
    )
    assert math.isclose(means[0], 2.0)
    assert np.isnan(means[1])


def test_feature_weights_give_each_family_equal_total_weight() -> None:
    metadata = pd.DataFrame(
        {
            "feature_name": ["a", "b", "c"],
            "analytic_family": ["one", "one", "two"],
        }
    )
    weights = audit._feature_weights(metadata)
    assert math.isclose(float(weights[:2].sum()), 0.5)
    assert math.isclose(float(weights[2]), 0.5)
