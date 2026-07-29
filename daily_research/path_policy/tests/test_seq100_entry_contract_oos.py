from __future__ import annotations

import numpy as np

from daily_research.path_policy import seq100_entry_contract_oos as contract


def test_strict_contract_task_counts_and_no_2026() -> None:
    study = contract.load_study()
    tasks = contract._training_tasks(study)

    assert len(tasks) == 48
    assert sum(task["family"] == "mfe" for task in tasks) == 12
    assert sum(task["family"] != "mfe" for task in tasks) == 36
    assert len(contract._inference_tasks()) == 6
    assert study["folds"]["forbidden_outcome_year"] == 2026


def test_date_rank_is_date_equal_and_ties_are_stable() -> None:
    dates = np.asarray([1, 1, 1, 2, 2, 2], dtype=np.int32)
    values = np.asarray([3.0, 1.0, 2.0, 4.0, 4.0, np.nan])

    ranked = contract._rank_by_date(values, dates)

    np.testing.assert_allclose(ranked[:3], [1.0, 0.0, 0.5])
    np.testing.assert_allclose(ranked[3:5], [0.5, 0.5])
    assert np.isnan(ranked[5])


def test_state_assignment_preserves_endpoint_ordered_semantics() -> None:
    centers = np.asarray([[-0.20, -0.10], [0.20, 0.30], [0.00, 0.02]], dtype=np.float32)
    model = {
        "clip_low": np.full(2, -1.0, dtype=np.float32),
        "clip_high": np.full(2, 1.0, dtype=np.float32),
        "center": np.zeros(2, dtype=np.float32),
        "scale": np.ones(2, dtype=np.float32),
        "pca_mean": np.zeros(2, dtype=np.float32),
        "pca_components": np.eye(2, dtype=np.float32),
        "fixed_k3_centers": centers,
        "raw_to_state": np.asarray([0, 2, 1], dtype=np.int8),
    }

    assigned = contract._assign_state(centers, model)

    np.testing.assert_array_equal(assigned, [0, 2, 1])


def test_daily_spearman_does_not_pool_dates() -> None:
    dates = np.repeat(np.asarray([1, 2], dtype=np.int32), 20)
    left = np.tile(np.arange(20, dtype=np.float64), 2)
    right = np.concatenate([left[:20], -left[20:]])

    value, count = contract._daily_spearman(left, right, dates)

    assert count == 2
    assert abs(value) < 1.0e-12
