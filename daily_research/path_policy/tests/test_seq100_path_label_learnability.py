from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from daily_research.path_policy import seq100_path_label_learnability as learnability
from daily_research.path_policy import seq100_signal_quality as signal_quality


ARCHIVED_STUDY_PATH = (
    learnability.WORKSPACE_ROOT
    / "daily_research/research_records/seq100/seq100_path_label_learnability_v1/config.json"
)


def _load_archived_study() -> dict[str, object]:
    return learnability.load_study(ARCHIVED_STUDY_PATH)


def test_load_study_keeps_scientific_configuration() -> None:
    study = _load_archived_study()

    assert study["study_id"] == learnability.STUDY_ID
    assert study["folds"]["maximum_outcome_date"] == "2025-12-31"
    assert study["folds"]["horizons"] == [5, 10, 20, 40, 60]


def test_horizon_purge_uses_actual_dependency_and_date_weights_are_equal() -> None:
    calendar = np.asarray(
        [
            "2022-12-26",
            "2022-12-27",
            "2022-12-28",
            "2022-12-29",
            "2022-12-30",
            "2023-01-03",
            "2023-01-04",
            "2023-01-05",
        ]
    )
    candidate_dates = np.repeat(np.arange(len(calendar), dtype=np.int32), 2)
    values = np.arange(len(candidate_dates), dtype=np.float32)

    fold = learnability.build_fold_rows(
        candidate_date_idx=candidate_dates,
        date_values=calendar,
        values=values,
        year=2023,
        dependency_days=2,
        valid=np.isfinite,
    )

    assert fold.maximum_train_signal_date_idx == 2
    assert int(candidate_dates[fold.train_rows[-1]]) + 2 < fold.oos_start_date_idx
    assert np.array_equal(
        np.unique(candidate_dates[fold.evaluation_rows]), np.asarray([5, 6, 7])
    )

    dates = np.asarray([1, 1, 2, 2, 2, 3], dtype=np.int32)
    weights = learnability.date_equal_weights(dates)
    totals = [float(weights[dates == value].sum()) for value in (1, 2, 3)]
    assert totals == pytest.approx([2.0, 2.0, 2.0])


def test_outcome_boundary_rejects_any_later_outcome() -> None:
    calendar = np.asarray(["2025-12-31", "2026-01-05"])
    dates = np.asarray([0, 1], dtype=np.int32)
    labels = np.asarray([[0.1], [np.nan]], dtype=np.float32)
    states = np.asarray([[2], [-1]], dtype=np.int8)
    entry = np.asarray([1, -1], dtype=np.int8)

    assert learnability.validate_outcome_boundary(
        candidate_date_idx=dates,
        date_values=calendar,
        labels=labels,
        states=states,
        entry_fill=entry,
        maximum_outcome_date="2025-12-31",
    ) == {"post_boundary_candidate_count": 1}

    labels[1, 0] = 0.0
    with pytest.raises(ValueError, match="post-boundary candidate"):
        learnability.validate_outcome_boundary(
            candidate_date_idx=dates,
            date_values=calendar,
            labels=labels,
            states=states,
            entry_fill=entry,
            maximum_outcome_date="2025-12-31",
        )


def test_invalid_shared_dataset_rows_receive_zero_weight() -> None:
    values = np.asarray([1.0, np.nan, 3.0, 4.0], dtype=np.float32)
    dates = np.asarray([1, 1, 2, 2], dtype=np.int32)
    target, weight = learnability.aligned_target_weights(
        values=values,
        date_idx=dates,
        valid=np.isfinite(values),
    )

    assert target.tolist() == [1.0, 0.0, 3.0, 4.0]
    assert weight[1] == 0.0
    assert float(weight[dates == 1].sum()) == pytest.approx(1.5)
    assert float(weight[dates == 2].sum()) == pytest.approx(1.5)


def test_daily_ranking_and_state_calibration_metrics() -> None:
    dates = np.repeat(np.asarray([0, 1], dtype=np.int32), 100)
    truth = np.tile(np.arange(100, dtype=np.float64), 2)
    calendar = np.asarray(["2023-01-03", "2023-01-04"])

    _daily, ranking = learnability.daily_score_metrics(
        date_idx=dates,
        actual=truth,
        score=truth,
        date_values=calendar,
        horizon=5,
    )

    assert ranking["rank_ic_mean"] == pytest.approx(1.0)
    assert ranking["decile_spearman"] == pytest.approx(1.0)
    assert ranking["top_1pct_lift"] > ranking["top_5pct_lift"] > 0.0

    state = np.tile(np.repeat(np.asarray([0, 1, 2], dtype=np.int8), [34, 33, 33]), 2)
    probability = np.full((len(state), 3), 0.005, dtype=np.float64)
    probability[np.arange(len(state)), state] = 0.99
    _state_daily, state_summary = learnability.state_metrics(
        date_idx=dates,
        actual=state,
        probability=probability,
        date_values=calendar,
        horizon=5,
    )

    assert state_summary["multiclass_brier_skill"] > 0.99
    assert state_summary["high_state_probability"]["brier_skill"] > 0.99
    assert state_summary["ordinal_ranking"]["rank_ic_mean"] > 0.99


def _decision_result(
    target: str,
    horizon: int,
    year: int,
    *,
    passing: bool,
) -> dict[str, object]:
    rank_ic = 0.03 if passing else -0.01
    decile = 0.9 if passing else -0.2
    lift = 0.02 if passing else -0.01
    p_value = 0.01 if passing else 0.5
    ranking = {
        "rank_ic_mean": rank_ic,
        "decile_spearman": decile,
        "top_1pct_lift": lift,
        "top_5pct_lift": lift,
        "top_5pct_positive_rate_lift": lift,
        "rank_ic_hac": {"p_value_two_sided": p_value},
    }
    if target == "state":
        metrics = {
            "ordinal_ranking": ranking,
            "high_state_probability": {
                "ranking": ranking,
                "brier_skill": 0.2 if passing else -0.1,
            },
            "multiclass_brier_skill": 0.2 if passing else -0.1,
        }
    else:
        metrics = {"ranking": ranking}
    return {
        "target": target,
        "horizon": horizon,
        "fold_year": year,
        "metrics": metrics,
    }


def test_mechanical_decision_selects_only_predeclared_qualified_family() -> None:
    results = [
        _decision_result(target, horizon, year, passing=(target == "mfe" and horizon in {5, 10}))
        for target in learnability.PATH_TARGETS
        for horizon in learnability.HORIZONS
        for year in learnability.FOLD_YEARS
    ]
    decision_config = _load_archived_study()["decision"]

    decision = learnability.decide_from_results(results, decision_config)

    assert decision["verdict"] == "learn_upside_opportunity_then_manage_realized_path"
    assert decision["primary_target"] == "mfe"
    assert decision["retained_horizons"] == [5, 10]
    assert decision["pre_peak_mae_auxiliary_horizons"] == []


def test_lgb_sequence_streams_the_float64_blocks_required_by_lightgbm() -> None:
    continuous = np.arange(12, dtype=np.float32).reshape(4, 3)
    categorical = np.asarray([[10], [20], [10], [30]], dtype=np.int64)
    rows = np.arange(4, dtype=np.int64)
    vocabulary = [np.asarray([10, 20], dtype=np.int64)]

    sequence = signal_quality._make_lgb_sequence(
        continuous=continuous,
        categorical=categorical,
        row_ids=rows,
        continuous_columns=np.asarray([0, 2], dtype=np.int32),
        categorical_columns=np.asarray([0], dtype=np.int32),
        category_vocabularies=vocabulary,
    )

    assert sequence[:].dtype == np.float64
    assert sequence[:][:, -1].tolist() == [1.0, 2.0, 1.0, 0.0]


def test_lgb_bins_can_be_reused_for_regression_and_multiclass() -> None:
    import lightgbm as lgb

    random = np.random.default_rng(7)
    row_count = 240
    fake_inputs = SimpleNamespace(
        continuous=random.normal(size=(row_count, 4)).astype(np.float32),
        categorical=random.integers(1, 4, size=(row_count, 1), dtype=np.int64),
        continuous_columns=np.arange(4, dtype=np.int32),
        categorical_columns=np.arange(1, dtype=np.int32),
        feature_names=["x0", "x1", "x2", "x3", "category"],
    )
    train_rows = np.arange(200, dtype=np.int64)
    evaluation_rows = np.arange(200, row_count, dtype=np.int64)
    regression = fake_inputs.continuous[:, 0]
    weights = np.ones(row_count, dtype=np.float32)
    study = _load_archived_study()
    datasets = learnability.build_lgb_datasets(
        inputs=fake_inputs,
        train_rows=train_rows,
        evaluation_rows=evaluation_rows,
        initial_train_label=regression[train_rows],
        initial_evaluation_label=regression[evaluation_rows],
        initial_train_weight=weights[train_rows],
        initial_evaluation_weight=weights[evaluation_rows],
        study=study,
    )

    regression_parameters, _rounds, _patience = learnability._model_parameters(
        study, target="g"
    )
    regression_model = lgb.train(
        regression_parameters,
        datasets.train_set,
        num_boost_round=2,
        valid_sets=[datasets.evaluation_set],
    )
    state = np.digitize(regression, [-0.25, 0.25]).astype(np.int8)
    datasets.train_set.set_label(state[train_rows])
    datasets.evaluation_set.set_label(state[evaluation_rows])
    state_parameters, _rounds, _patience = learnability._model_parameters(
        study, target="state"
    )
    state_model = lgb.train(
        state_parameters,
        datasets.train_set,
        num_boost_round=2,
        valid_sets=[datasets.evaluation_set],
    )

    assert regression_model.current_iteration() == 1
    assert state_model.predict(datasets.evaluation_sequence).shape == (40, 3)
