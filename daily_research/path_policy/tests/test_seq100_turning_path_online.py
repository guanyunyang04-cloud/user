from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from daily_research.path_policy import seq100_turning_path_online as online


def test_contract_uses_quality_pool_and_chronological_outcomes() -> None:
    study = online.load_study()
    assert study["source"]["quality_pool_name"] == "quality_liquidity_pit"
    assert study["period"]["rolling_evaluation_years"] == list(range(2019, 2026))
    assert study["outcome_semantics"]["same_day_resolutions"].startswith("known state")
    assert study["boundaries"]["profit_claim_allowed"] is False
    assert len(online._scale_pairs(study)) == 8


def test_episode_query_excludes_known_and_censored_states() -> None:
    query = online._episode_query(
        Path("panel.parquet"),
        episode_type="down_rebound",
        probe_multiplier=2,
        major_multiplier=8,
        maximum_signal_year=2018,
        maximum_resolution_date_idx_exclusive=2200,
    )
    assert online.FUTURE_RESOLUTION_FILTER in query
    assert "signal_year <= 2018" in query
    assert "resolution_date_idx < 2200" in query
    assert "outcome = 'major_bottom_confirmed'" in query


def _fixture_frame() -> pd.DataFrame:
    frame = pd.DataFrame(
        {
            "onset_reversal_log_return": [-0.012, -0.018],
            "probe_log_return": [0.012, 0.012],
            "signed_leg_move_to_anchor": [0.048, 0.096],
            "major_threshold_log_return": [0.048, 0.048],
            "anchor_age_market_days": [1, 20],
            "days_since_major_confirmation": [2, 40],
            "size_bin_10": [1, np.nan],
            "all_rank_features_complete": [True, False],
        }
    )
    for index, feature in enumerate(online.RANK_FEATURES):
        frame[feature] = [0.25 + index * 0.001, np.nan]
    return frame


def test_geometry_is_dimensionless_and_missing_state_is_explicit() -> None:
    frame = _fixture_frame()
    geometry, geometry_names = online._model_matrix(frame, "geometry")
    enhanced, enhanced_names = online._model_matrix(frame, "activity_price")
    assert geometry_names == online.GEOMETRY_FEATURES
    assert geometry[0, 0] == 0.0
    assert geometry[1, 0] == pytest.approx(np.log1p(0.5))
    assert enhanced.shape[1] == len(enhanced_names)
    assert enhanced[1, -1] == 1.0
    assert np.isfinite(enhanced).all()


def test_fold_training_requires_resolution_before_first_test_onset() -> None:
    frame = pd.DataFrame(
        {
            "signal_year": [2018, 2018, 2019, 2019],
            "onset_date_idx": [90, 95, 100, 101],
            "resolution_date_idx": [96, 100, 101, 102],
        }
    )
    train, test, cutoff = online._fold_masks(frame, 2019)
    assert cutoff == 100
    assert train.tolist() == [True, False, False, False]
    assert test.tolist() == [False, False, True, True]


def test_one_standard_error_rule_prefers_stronger_regularization() -> None:
    rows = []
    for c_value, losses in ((0.001, [0.61, 0.63]), (0.1, [0.55, 0.65])):
        for date_idx, loss in enumerate(losses, start=1):
            rows.append(
                {
                    "feature_set": "geometry",
                    "regularization_c": c_value,
                    "evaluation_year": 2018,
                    "onset_date_idx": date_idx,
                    "daily_log_loss": loss,
                }
            )
    diagnostics, selected = online._selection_diagnostics(
        pd.DataFrame(rows), online.load_study()
    )
    assert selected["geometry"] == 0.001
    assert diagnostics[diagnostics["selected"]]["regularization_c"].item() == 0.001


def test_bh_q_values_preserve_input_order_and_dominate_p_values() -> None:
    p_values = np.array([0.04, 0.001, 0.02])
    q_values = online._bh_q_values(p_values)
    assert q_values.shape == p_values.shape
    assert np.all(q_values >= p_values)
    assert q_values[1] == 0.003
