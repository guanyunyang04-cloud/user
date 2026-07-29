from __future__ import annotations

import json

import numpy as np
import pytest

from daily_research.path_policy import seq100_mfe_feature_union_audit as audit


def test_study_has_preregistered_stage_counts_and_no_2026() -> None:
    study = audit.load_study()

    assert len(audit._tasks(study, stage="stage1")) == 18
    assert len(audit._tasks(study, stage="stage2")) == 6
    assert study["folds"]["forbidden_outcome_year"] == 2026


def test_union_extra_concatenates_feature_major_memmaps_by_requested_rows() -> None:
    first = np.arange(15, dtype=np.float32).reshape(3, 5)
    second = np.arange(10, dtype=np.float32).reshape(2, 5) + 100.0
    rows = np.asarray([4, 1, 3], dtype=np.int64)
    union = audit.UnionExtra(
        arrays=(first, second),  # type: ignore[arg-type]
        feature_names=("a", "b", "c", "d", "e"),
    )

    np.testing.assert_array_equal(
        union[:, rows],
        np.concatenate([first[:, rows], second[:, rows]], axis=0),
    )
    with pytest.raises(IndexError, match="complete feature axis"):
        _ = union[0, rows]


def test_role_gate_requires_three_positive_top5_years() -> None:
    rules = audit.load_study()["decision"]
    annual = []
    relationships = []
    for index, _year in enumerate(audit.FOLD_YEARS):
        annual.append(
            {
                "metrics": {
                    "top_5pct_lift": {"mean_delta": -0.001 if index == 1 else 0.001},
                    "daily_tail_top5_lift": {"mean_delta": 0.01},
                    "rank_ic": {"mean_delta": 0.001},
                }
            }
        )
        relationships.append(
            {
                "summary": {
                    "metrics": {
                        "top5_right_only_minus_left_only_mfe_mean": {"mean": 0.002}
                    }
                }
            }
        )

    result = audit._role_gate(
        annual=annual,
        relationships=relationships,
        rules=rules,
    )

    assert not result["passed"]
    assert result["checks"]["positive_top5_mfe_years"] == 2


def test_changed_union_contract_is_rejected(tmp_path) -> None:
    study = audit.load_study()
    changed = json.loads(json.dumps(study))
    changed["heads"]["10"]["stage1"][
        "turnover_cost_proxy__completed_week_month_context"
    ] = ["completed_week_month_context", "turnover_cost_proxy"]
    path = tmp_path / "changed.json"
    path.write_text(json.dumps(changed), encoding="utf-8")

    with pytest.raises(ValueError, match="does not extend its incumbent"):
        audit.load_study(path)
