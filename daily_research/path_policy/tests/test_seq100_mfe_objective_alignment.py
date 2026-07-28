from __future__ import annotations

import math

import numpy as np
import pytest

from daily_research.path_policy import seq100_mfe_objective_alignment as audit


def test_study_builds_exactly_twelve_new_objective_tasks() -> None:
    study = audit.load_study()
    tasks = audit._task_plan()

    assert len(tasks) == 12
    assert {task["objective_variant"] for task in tasks} == {
        "tail_weighted_huber",
        "daily_lambdarank",
    }
    assert {(task["horizon"], task["fold_year"]) for task in tasks} == {
        (horizon, year)
        for horizon in study["labels"]["horizons"]
        for year in study["folds"]["fold_years"]
    }


def test_within_date_percentiles_average_ties() -> None:
    dates = np.array([1, 1, 1, 1, 2, 2], dtype=np.int32)
    values = np.array([1.0, 2.0, 2.0, 4.0, 9.0, 3.0])

    result = audit.within_date_percentile(dates, values)

    np.testing.assert_allclose(result, [0.125, 0.5, 0.5, 0.875, 0.75, 0.25])


def test_tail_weights_are_monotone_and_keep_equal_date_totals() -> None:
    dates = np.repeat(np.array([1, 2], dtype=np.int32), [4, 2])
    values = np.array([0.0, 1.0, 2.0, 3.0, -1.0, 4.0])

    weights = audit.tail_training_weights(
        date_idx=dates,
        values=values,
        maximum_multiplier=4.0,
        power=4.0,
    )

    assert np.all(np.diff(weights[:4]) > 0.0)
    assert weights[5] > weights[4]
    assert weights[:4].sum() == pytest.approx(weights[4:].sum())


def test_lambdarank_labels_and_groups_follow_dates() -> None:
    dates = np.repeat(np.array([1, 2], dtype=np.int32), [4, 3])
    values = np.array([0.0, 1.0, 2.0, 3.0, 5.0, 5.0, 1.0])

    labels = audit.daily_relevance_labels(
        date_idx=dates, values=values, levels=100
    )

    np.testing.assert_array_equal(audit.date_group_sizes(dates), [4, 3])
    np.testing.assert_array_equal(labels[:4], [12, 37, 62, 87])
    np.testing.assert_array_equal(labels[4:], [66, 66, 16])


def test_legal_peak_day_excludes_entry_day_and_uses_first_legal_tie() -> None:
    close = np.array(
        [
            [0.50, 0.10, 0.30, 0.30],
            [0.10, 0.20, 0.40, 0.30],
            [0.10, 0.20, 0.30, 0.40],
        ],
        dtype=np.float32,
    )
    sellable = np.array(
        [
            [True, True, True, True],
            [True, True, False, True],
            [True, False, False, False],
        ]
    )

    peak_day, mfe = audit._legal_peak_day_and_mfe(close, sellable)

    np.testing.assert_allclose(peak_day[:2], [3.0, 4.0])
    np.testing.assert_allclose(mfe[:2], [0.30, 0.30])
    assert math.isnan(float(peak_day[2]))
    assert math.isnan(float(mfe[2]))


def test_path_quality_separates_tail_speed_risk_and_endpoint() -> None:
    count = 100
    dates = np.zeros(count, dtype=np.int32)
    score = np.arange(count, dtype=np.float64)
    target = np.arange(count, dtype=np.float64) / 100.0
    early = target * 0.5
    early[-1] = np.nan
    peak_day = np.full(count, 8.0)
    peak_day[-5:] = 3.0
    adverse = np.full(count, -0.08)
    adverse[-5:] = -0.02
    endpoint = np.full(count, -0.01)
    endpoint[-5:] = 0.06
    state = np.zeros(count, dtype=np.int8)
    state[-5:] = 2

    daily, metrics = audit.path_quality_metrics(
        date_idx=dates,
        score=score,
        target_mfe=target,
        early_mfe=early,
        peak_day=peak_day,
        pre_peak_mae=adverse,
        endpoint_return=endpoint,
        state=state,
        horizon=10,
    )

    assert len(daily) == 1
    assert metrics["top5_daily_tail_rate"] == pytest.approx(1.0)
    assert metrics["top5_daily_tail_lift"] == pytest.approx(0.8)
    assert metrics["top5_peak_day_mean"] == pytest.approx(3.0)
    assert metrics["top5_pre_peak_mae_mean"] == pytest.approx(-0.02)
    assert metrics["top5_endpoint_return_mean"] == pytest.approx(0.06)
    assert metrics["top5_high_state_rate"] == pytest.approx(1.0)
    assert 0.39 < metrics["top5_early_opportunity_share"] < 0.41
