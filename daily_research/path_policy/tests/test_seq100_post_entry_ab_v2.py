from __future__ import annotations

import numpy as np

from daily_research.path_policy import seq100_post_entry_ab_v2 as post
from daily_research.path_policy import seq100_post_entry_capacity_audit as capacity


def _synthetic_landmark() -> tuple[dict, post.LandmarkData, np.ndarray]:
    study = post.load_study()
    names = post._full_feature_names(study)
    row_count = 120
    features = np.ones((row_count, len(names)), dtype=np.float32)
    dates = np.arange(row_count, dtype=np.int32)
    targets = {
        "mfe_10": np.full(row_count, 0.1, dtype=np.float32),
        "mfe_20": np.full(row_count, 0.2, dtype=np.float32),
        "risk_10": np.full(row_count, -0.02, dtype=np.float32),
        "risk_20": np.full(row_count, -0.03, dtype=np.float32),
        "state_10": np.tile(np.asarray([0, 1, 2], dtype=np.int8), 40),
    }
    data = post.LandmarkData(
        age=3,
        feature_names=names,
        features=features,
        entry_rows=np.arange(row_count, dtype=np.int64),
        current_rows=np.arange(row_count, dtype=np.int64),
        decision_date_idx=dates,
        entry_either_top5=np.zeros(row_count, dtype=bool),
        targets=targets,
        endpoints={
            10: np.full(row_count, 0.03, dtype=np.float32),
            20: np.full(row_count, 0.05, dtype=np.float32),
        },
        tradable_day_fraction=np.ones(row_count, dtype=np.float32),
        manifest={},
    )
    date_values = np.asarray(
        [
            *[f"2020-01-{index + 1:02d}" for index in range(30)],
            *[f"2021-01-{index + 1:02d}" for index in range(30)],
            *[f"2022-01-{index + 1:02d}" for index in range(30)],
            *[f"2023-01-{index + 1:02d}" for index in range(30)],
        ],
        dtype=str,
    )
    return study, data, date_values


def test_d1_keeps_only_nonduplicated_realized_path_features() -> None:
    study = post.load_study()
    d1 = set(post._feature_names(study, 1, "B"))
    d3 = set(post._feature_names(study, 3, "B"))

    assert len(d1) == 16
    assert len(d3) == 24
    assert not d1.intersection(study["features"]["D1_forbidden_duplicates"])


def test_target_specific_split_keeps_d10_rows_when_d20_is_missing() -> None:
    study, data, date_values = _synthetic_landmark()
    data.targets["mfe_20"][35] = np.nan

    _, d10 = post._split_rows(
        data=data,
        study=study,
        target="mfe_10",
        evaluation_year=2021,
        date_values=date_values,
        selection_outcome_cutoff=True,
    )
    _, d20 = post._split_rows(
        data=data,
        study=study,
        target="mfe_20",
        evaluation_year=2021,
        date_values=date_values,
        selection_outcome_cutoff=True,
    )

    assert len(d10) == 20
    assert len(d20) == 9
    assert 35 in d10
    assert 35 not in d20


def test_capacity_selection_purges_training_and_caps_outcomes_at_year_end() -> None:
    study, data, date_values = _synthetic_landmark()
    train, evaluation = post._split_rows(
        data=data,
        study=study,
        target="mfe_10",
        evaluation_year=2021,
        date_values=date_values,
        selection_outcome_cutoff=True,
    )

    assert train[-1] == 19
    assert evaluation[0] == 30
    assert evaluation[-1] == 49
    assert data.decision_date_idx[train[-1]] + 10 < 30
    assert data.decision_date_idx[evaluation[-1]] + 10 < 60


def test_capacity_tasks_use_A_only_and_uniform_target_candidates() -> None:
    study = capacity.load_study()
    tasks = capacity._tasks(study)

    assert len(tasks) == 30
    assert {task["variant"] for task in tasks} == {"A"}
    for target in post.TARGETS:
        assert {
            tuple(task["capacities"]) for task in tasks if task["target"] == target
        } == {tuple(study["capacities"][target])}


def test_formal_tasks_share_one_frozen_capacity_per_target() -> None:
    study = post.load_study()
    decision = {
        "selected_rounds": {
            "mfe_10": 512,
            "mfe_20": 256,
            "risk_10": 64,
            "risk_20": 128,
            "state_10": 32,
        }
    }
    tasks = post._tasks(study, decision)

    assert len(tasks) == 90
    for target, rounds in decision["selected_rounds"].items():
        target_tasks = [task for task in tasks if task["target"] == target]
        assert {task["rounds"] for task in target_tasks} == {rounds}
        assert {task["variant"] for task in target_tasks} == {"A", "B"}


def test_remaining_mfe_contract_uses_D2_sell_window() -> None:
    study = post.load_study()

    assert study["targets"]["mfe_10"]["source"] == "matched_mfe_10"
    assert study["targets"]["mfe_20"]["source"] == "matched_mfe_20"
    assert "D2-D10" in study["targets"]["mfe_10"]["semantics"]
    assert "D2-D20" in study["targets"]["mfe_20"]["semantics"]


def test_state_capacity_rounds_expand_to_three_internal_trees() -> None:
    study = post.load_study()
    parameters = post._model_parameters(study, "state_10")

    assert parameters["num_class"] == 3
    assert 256 * parameters["num_class"] == 768
