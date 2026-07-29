from __future__ import annotations

import json

import pytest

from daily_research.path_policy import seq100_mfe_capacity_stability_audit as audit


def test_study_freezes_heads_years_and_task_counts() -> None:
    study = audit.load_study()

    assert study["folds"]["capacity_selection_years"] == [2019, 2020, 2021, 2022]
    assert study["folds"]["decision_years"] == [2023, 2024, 2025]
    assert study["folds"]["forbidden_outcome_year"] == 2026
    assert study["capacity_selection"]["uses_decision_year_for_selection"] is False
    assert study["heads"]["10"] == {
        "name": "turnover_cost_proxy",
        "families": ["turnover_cost_proxy"],
    }
    assert study["heads"]["20"] == {
        "name": "breakout_retest_levels",
        "families": ["breakout_retest_levels"],
    }
    assert len(audit._selection_tasks(study)) == 8
    assert len(audit._outer_tasks(study)) == 6


def test_one_standard_error_selects_smallest_eligible_capacity() -> None:
    result = audit._select_from_losses(
        {
            2019: {4: 1.10, 8: 1.018, 16: 1.00, 32: 1.01},
            2020: {4: 1.08, 8: 1.00, 16: 1.01, 32: 1.03},
            2021: {4: 1.12, 8: 1.03, 16: 1.00, 32: 1.01},
            2022: {4: 1.09, 8: 1.00, 16: 1.02, 32: 1.04},
        },
        (4, 8, 16, 32),
    )

    assert result["aggregate_minimizer"] in (8, 16)
    assert result["selected_iteration"] == 8
    assert result["selected_iteration"] <= result["aggregate_minimizer"]
    assert len(result["grid"]) == 4


def test_capacity_gate_and_contract_change_are_independent() -> None:
    rules = audit.load_study()["decision"]
    gate = audit._decision_gate(
        relative_mae_harm=(0.001, -0.001, 0.002),
        rank_ic_delta=(0.0, 0.001, -0.001),
        top5_mfe_delta=(0.0, -0.0005, 0.001),
        path_guardrail={"rejected": False},
        rules=rules,
    )
    assert gate["passed"] is True

    changed = audit._contract_rebuild_required(
        adopted=True,
        prediction_spearman=(0.99, 0.94, 0.98),
        top5_jaccard=(0.90, 0.85, 0.79),
        rules=rules,
    )
    assert changed["required"] is True
    assert changed["materially_changed_years"] == [2024, 2025]

    rejected_policy = audit._contract_rebuild_required(
        adopted=False,
        prediction_spearman=(0.50, 0.50, 0.50),
        top5_jaccard=(0.10, 0.10, 0.10),
        rules=rules,
    )
    assert rejected_policy["required"] is False


def test_study_rejects_decision_year_capacity_selection(tmp_path) -> None:
    payload = json.loads(audit.DEFAULT_STUDY_PATH.read_text(encoding="utf-8"))
    payload["capacity_selection"]["uses_decision_year_for_selection"] = True
    path = tmp_path / "invalid.json"
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="cannot select capacity"):
        audit.load_study(path)
