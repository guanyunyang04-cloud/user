from __future__ import annotations

import numpy as np

from daily_research.path_policy import seq100_entry_fixed_capacity_audit as audit


def test_fixed_capacity_task_contract_and_final_heads() -> None:
    study = audit.load_study()
    tasks = audit._tasks(study)
    mfe = [task for task in tasks if task["kind"] == "mfe"]
    safety = [task for task in tasks if task["kind"] == "safety"]

    assert len(mfe) == 6
    assert len(safety) == 18
    assert {task["rounds"] for task in mfe} == {512}
    assert {task["rounds"] for task in safety} == {256}
    assert audit._head(study, 10)["families"] == ("turnover_cost_proxy",)
    assert audit._head(study, 20)["families"] == ("breakout_retest_levels",)


def test_state_rounds_are_not_confused_with_internal_tree_count() -> None:
    study = audit.load_study()
    task = next(
        task
        for task in audit._tasks(study)
        if task["kind"] == "safety" and task["name"] == "state_10"
    )
    assert task["rounds"] == 256
    assert (
        task["rounds"] * study["state_and_risk"]["state_internal_tree_multiplier"]
        == 768
    )


def test_prefix_equivalence_requires_values_and_daily_order() -> None:
    old = np.asarray([0.1, 0.2, 0.3, 0.4, 0.5, 0.6], dtype=np.float32)
    dates = np.asarray([1, 1, 1, 2, 2, 2], dtype=np.int32)
    exact = audit._prefix_equivalence(
        old=old,
        current=old.copy(),
        dates=dates,
        absolute_tolerance=1.0e-7,
        relative_tolerance=1.0e-6,
        minimum_daily_spearman=0.999999,
    )
    changed = audit._prefix_equivalence(
        old=old,
        current=old[::-1],
        dates=dates,
        absolute_tolerance=1.0e-7,
        relative_tolerance=1.0e-6,
        minimum_daily_spearman=0.999999,
    )

    assert exact["passed"]
    assert not changed["passed"]


def test_state_and_risk_safety_gates_are_head_specific() -> None:
    study = audit.load_study()
    state = [
        {
            "year": year,
            "ordinal_ic_delta": 0.001,
            "high_state_top5_lift_delta": 0.001,
            "relative_brier_harm": 0.0,
            "relative_logloss_harm": 0.0,
            "collapsed": False,
        }
        for year in audit.CONTRACT_YEARS
    ]
    risk = [
        {
            "year": year,
            "rank_ic_delta": 0.001,
            "relative_mae_harm": 0.0,
            "deep_adverse_pr_auc_delta": 0.001,
            "absolute_bias_harm": -0.001 if year == 2025 else 0.0,
        }
        for year in audit.CONTRACT_YEARS
    ]

    assert audit._state_gate(annual=state, rules=study["safety_gate"]["state"])[
        "passed"
    ]
    assert audit._risk_gate(annual=risk, rules=study["safety_gate"]["risk"])["passed"]

    state[-1]["relative_brier_harm"] = 0.02
    assert not audit._state_gate(annual=state, rules=study["safety_gate"]["state"])[
        "passed"
    ]
    assert audit._risk_gate(annual=risk, rules=study["safety_gate"]["risk"])["passed"]


def test_all_contract_date_ranks_are_regenerated_with_ties() -> None:
    raw = np.asarray(
        [
            [1.0, 3.0],
            [1.0, 2.0],
            [2.0, 1.0],
            [4.0, 7.0],
            [3.0, 7.0],
        ],
        dtype=np.float32,
    )
    dates = np.asarray([1, 1, 1, 2, 2], dtype=np.int32)
    ranks = audit._rank_matrix(raw, dates)

    np.testing.assert_allclose(ranks[:3, 0], [0.25, 0.25, 1.0])
    np.testing.assert_allclose(ranks[3:, 1], [0.5, 0.5])
    assert np.all((ranks >= 0.0) & (ranks <= 1.0))


def test_config_forbids_2026_and_keeps_rank_first_semantics() -> None:
    study = audit.load_study()

    assert study["folds"]["maximum_outcome_date"] == "2025-12-31"
    assert study["folds"]["forbidden_outcome_year"] == 2026
    assert study["mfe"]["rank_first"] is True
    assert study["mfe"]["literal_expected_return"] is False
