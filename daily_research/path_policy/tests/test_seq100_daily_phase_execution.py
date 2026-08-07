from __future__ import annotations

import numpy as np
import pandas as pd

from daily_research.path_policy import seq100_daily_phase_execution as phase


def test_phase_study_contract_uses_posterior_boundary_without_fixed_target() -> None:
    study = phase.load_study()
    assert study["source"]["quality_pool_name"] == "quality_liquidity_pit"
    assert study["belief"]["turning_scale"] == 16
    assert study["belief"]["phase_boundary_probability"] == 0.5
    assert study["belief"]["fixed_holding_target_used"] is False
    assert study["execution"]["fixed_exit_controls"] == [5, 10, 20]


def test_belief_query_is_scale16_causal_and_does_not_read_realized_outcomes() -> None:
    query = phase._belief_query(
        phase.WORKSPACE_ROOT / "prediction.parquet",
        phase.WORKSPACE_ROOT / "panel.parquet",
    )
    assert "turning_scale = 16" in query
    assert "geometry_long_probability" in query
    assert "entry_setup" in query
    assert "entry_confirmation_log_return" not in query
    assert "prediction.parquet" in query


def test_belief_audit_rejects_duplicates_and_accepts_finite_phase_posteriors() -> None:
    frame = pd.DataFrame(
        {
            "symbol": ["A", "B"],
            "date_idx": [10, 10],
            "trade_date": ["2019-01-02", "2019-01-02"],
            "geometry_long_probability": [0.6, 0.4],
            "blend01_long_probability": [0.59, 0.41],
            "entry_setup": ["uptrend_pullback", "other"],
            "maximum_training_resolution_date_idx": [9, 9],
        }
    )
    audit = phase._audit_belief_frame(
        frame, year=2019, expected_rows=2, cutoff="2025-12-31"
    )
    assert audit["duplicate_rows"] == 0
    assert audit["nonfinite_or_boundary_probability_rows"] == 0
    duplicate = pd.concat([frame, frame.iloc[[0]]], ignore_index=True)
    duplicate_audit = phase._audit_belief_frame(
        duplicate, year=2019, expected_rows=3, cutoff="2025-12-31"
    )
    assert duplicate_audit["duplicate_rows"] == 1


def test_profile_probability_is_transformed_to_long_score() -> None:
    frame = pd.DataFrame(
        {
            "geometry_long_probability": np.asarray([0.51, 0.49]),
            "entry_setup": ["uptrend_pullback", "downtrend_rebound"],
        }
    )
    score = frame["geometry_long_probability"] - 0.5
    assert score.iloc[0] > 0.0
    assert score.iloc[1] < 0.0
