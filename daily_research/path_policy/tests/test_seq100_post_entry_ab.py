from __future__ import annotations

import numpy as np

from daily_research.path_policy import seq100_post_entry_ab as audit


def test_formal_task_counts_feature_dimensions_and_no_2026() -> None:
    study = audit.load_study()
    tasks = audit._tasks()

    assert len(tasks) == 63
    assert sum(task["target"] == "risk" for task in tasks) == 27
    assert sum(task["target"] == "state" for task in tasks) == 36
    assert len(audit._feature_names(study, 1, "A")) == 6
    assert len(audit._feature_names(study, 1, "B")) == 16
    assert len(audit._feature_names(study, 3, "B")) == 24
    assert len(audit._feature_names(study, 5, "B")) == 24
    assert not (
        set(study["features"]["D1_forbidden_duplicates"])
        & set(audit._feature_names(study, 1, "B"))
    )
    assert study["folds"]["forbidden_outcome_year"] == 2026
    assert study["folds"]["maximum_outcome_date"] == "2025-12-31"


def test_split_purges_ten_days_and_is_variant_independent() -> None:
    study = audit.load_study()
    date_values = np.asarray(
        [f"{year}-01-{day:02d}" for year in range(2019, 2026) for day in range(1, 21)],
        dtype=str,
    )
    count = len(date_values)
    feature_names = audit._full_feature_names(study)
    data = audit.LandmarkData(
        age=3,
        feature_names=feature_names,
        features=np.ones((count, len(feature_names)), dtype=np.float32),
        entry_rows=np.arange(count, dtype=np.int64),
        current_rows=np.arange(count, dtype=np.int64),
        decision_date_idx=np.arange(count, dtype=np.int32),
        entry_either_top5=np.ones(count, dtype=bool),
        target_risk=np.zeros(count, dtype=np.float32),
        target_state=np.zeros(count, dtype=np.int8),
        tradable_day_fraction=np.ones(count, dtype=np.float32),
        manifest={},
    )

    train, evaluation = audit._split_rows(
        data=data,
        study=study,
        target="risk",
        test_year=2023,
        stage="outer",
        date_values=date_values,
    )

    evaluation_start = audit._year_start(date_values, 2023)
    assert int(data.decision_date_idx[train].max()) + 10 < evaluation_start
    assert int(data.decision_date_idx[evaluation].min()) == evaluation_start
    assert int(data.decision_date_idx[evaluation].max()) < audit._year_start(
        date_values, 2024
    )


def test_lightgbm_sequence_exposes_double_precision_batches() -> None:
    matrix = np.arange(30, dtype=np.float32).reshape(6, 5)
    sequence = audit._MatrixSequence(
        matrix=matrix,
        rows=np.asarray([5, 2, 1], dtype=np.int64),
        columns=np.asarray([4, 0], dtype=np.int64),
        batch_size=2,
    )

    batch = sequence[0:2]

    assert batch.dtype == np.float64
    np.testing.assert_array_equal(batch, matrix[[5, 2]][:, [4, 0]])


def test_temperature_is_fit_on_supplied_validation_rows_only() -> None:
    probability = np.asarray(
        [[0.98, 0.01, 0.01], [0.01, 0.98, 0.01], [0.01, 0.01, 0.98]] * 50,
        dtype=np.float32,
    )
    actual = np.tile(np.asarray([0, 1, 1], dtype=np.int64), 50)
    dates = np.repeat(np.arange(50, dtype=np.int32), 3)

    fitted = audit.fit_temperature(
        probability=probability,
        actual=actual,
        date_idx=dates,
    )
    calibrated = audit.apply_temperature(probability, float(fitted["temperature"]))

    assert fitted["validation_row_count"] == len(actual)
    assert fitted["fit_year_start_date_idx"] == 0
    assert fitted["fit_year_end_date_idx"] == 49
    assert fitted["weighted_logloss_after"] < fitted["weighted_logloss_before"]
    np.testing.assert_allclose(calibrated.sum(axis=1), 1.0, atol=1.0e-6)


def test_risk_metrics_identify_deep_adverse_tail() -> None:
    dates = np.repeat(np.asarray([1, 2], dtype=np.int32), 100)
    actual = np.tile(np.linspace(-0.20, 0.0, 100), 2)

    daily = audit.risk_daily_metrics(
        date_idx=dates,
        actual=actual,
        prediction=actual,
        entry_either_top5=np.ones(len(actual), dtype=bool),
    )

    np.testing.assert_allclose(daily["rank_ic"], 1.0)
    np.testing.assert_allclose(daily["mae"], 0.0)
    np.testing.assert_allclose(daily["deep_adverse_pr_auc"], 1.0)
    np.testing.assert_allclose(daily["entry_either_top5_rank_ic"], 1.0)


def test_risk_is_primary_and_state_cannot_rescue_failure() -> None:
    risk = {
        1: {"passed": True, "annual_rank_ic_delta": [0.004, 0.004, 0.004]},
        3: {"passed": False, "annual_rank_ic_delta": [0.001, 0.001, 0.001]},
        5: {"passed": False, "annual_rank_ic_delta": [0.001, 0.001, 0.001]},
    }
    state = {
        age: {"passed": True, "annual_raw_ordinal_ic_delta": [0.01] * 3}
        for age in audit.AGES
    }

    decision = audit._general_decision(
        risk_by_age=risk,
        state_by_age=state,
        rules=audit.load_study()["decision"]["general_model"],
    )

    assert decision["status"] == "age_specific_information_only"
    assert decision["passing_state_ages"] == [1, 3, 5]
    assert decision["state_cannot_rescue_failed_risk"]
