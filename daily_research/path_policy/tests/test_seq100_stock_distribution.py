from __future__ import annotations

import numpy as np
import pandas as pd

from daily_research.path_policy import seq100_stock_distribution as stock


def test_study_contract_is_frozen_without_training() -> None:
    study = stock.load_study()
    assert tuple(study["target"]["horizons"]) == stock.HORIZONS
    assert study["evaluation"]["training_performed"] is False
    assert study["evaluation"]["portfolio_selection_performed"] is False


def test_leave_one_out_factors_exclude_own_return() -> None:
    values = np.array([1.0, 3.0, 5.0, 7.0])
    valid = np.ones(4, dtype=bool)
    dates = np.zeros(4, dtype=np.int64)
    industries = np.array([0, 0, 1, -1], dtype=np.int32)
    residual, market, industry = stock._leave_one_out_factors(
        values, valid, dates, industries
    )

    assert market[0] == 5.0
    assert industry[0] == -2.0
    assert residual[0] == -2.0
    assert market[1] == pytest_approx(13.0 / 3.0)
    assert residual[1] == 2.0
    # A singleton industry and an unknown industry both fall back to market.
    assert industry[2] == 0.0
    assert industry[3] == 0.0


def test_date_stratified_sample_keeps_each_date_bounded() -> None:
    rows = np.arange(15, dtype=np.int64)
    dates = np.repeat([10, 11, 12], [3, 8, 4])
    sampled = stock._sample_rows_by_date(rows, dates, maximum_per_date=3)
    sampled_dates = dates[sampled]
    assert len(sampled) == 9
    assert all(np.sum(sampled_dates == date) <= 3 for date in (10, 11, 12))
    assert set(sampled_dates) == {10, 11, 12}


def test_student_t_crps_is_nonnegative_at_the_center() -> None:
    values = np.array([0.0, 1.0])
    location = np.array([0.0, 1.0])
    scale = np.array([1.0, 1.0])
    score = stock._crps_from_quantiles(values, location, scale, 8.0)
    assert np.all(score >= 0.0)
    assert score[0] == pytest_approx(score[1])


def test_hac_interval_is_finite_for_constant_gain() -> None:
    summary = stock._hac_mean(np.full(80, 0.01), lag=20)
    assert summary["mean"] == pytest_approx(0.01)
    assert summary["lcb_95"] == pytest_approx(0.01)


def test_location_scale_audit_preserves_factorial_gain_identity() -> None:
    model_scores = {
        "zero_mean_student_t": 1.00,
        "zero_mean_feature_scale_student_t": 1.08,
        "ridge_student_t": 1.02,
        "ridge_hetero_student_t": 1.11,
    }
    rows = [
        {
            "evaluation_year": 2022,
            "horizon": 10,
            "target": "residual_log_return_10",
            "feature_block": "core",
            "date_idx": date_idx,
            "model": model,
            "log_score": score,
        }
        for date_idx in range(80)
        for model, score in model_scores.items()
    ]

    audit = stock._location_scale_audit(pd.DataFrame(rows)).iloc[0]

    assert audit["total_log_score_gain"] == pytest_approx(0.11)
    assert audit["location_gain_constant_scale"] == pytest_approx(0.02)
    assert audit["location_gain_feature_scale"] == pytest_approx(0.03)
    assert audit["scale_gain_zero_mean"] == pytest_approx(0.08)
    assert audit["scale_gain_ridge_mean"] == pytest_approx(0.09)
    assert (
        audit["location_shapley_gain"] + audit["scale_shapley_gain"]
    ) == pytest_approx(audit["total_log_score_gain"])
    assert audit["conditional_location_hac_lcb_95"] == pytest_approx(0.03)
    assert audit["conditional_location_block_lcb_95"] == pytest_approx(0.03)


def pytest_approx(value: float) -> object:
    import pytest

    return pytest.approx(value)
