from __future__ import annotations

import json

import numpy as np

from daily_research.path_policy import seq100_entry_contract_oos as contract
from daily_research.path_policy import (
    seq100_mfe_final_head_capacity_audit as audit,
)


def _passing_pairs(top5: tuple[float, float, float]) -> list[dict[str, object]]:
    return [
        {
            "metrics": {
                "top_5pct_lift": {"mean_delta": value},
                "rank_ic": {"mean_delta": 0.001},
                "daily_tail_top5_lift": {"mean_delta": 0.001},
            }
        }
        for value in top5
    ]


def test_study_freezes_heads_policies_and_task_counts() -> None:
    study = audit.load_study()

    assert study["folds"]["model_years"] == [2023, 2024, 2025]
    assert study["folds"]["new_tuning_model_years"] == [2024, 2025]
    assert study["folds"]["forbidden_outcome_year"] == 2026
    assert study["heads"]["10"]["families"] == ["turnover_cost_proxy"]
    assert study["heads"]["20"]["families"] == ["breakout_retest_levels"]
    assert tuple(study["policies"]["formal_candidates"]) == audit.FORMAL_POLICIES
    assert len(audit._tuning_tasks(study)) == 4
    assert len(audit._outer_tasks(study)) == 6


def test_tuning_resolution_requires_full_patience_after_best() -> None:
    assert audit._tuning_resolved(recorded_rounds=300, best_iteration=200, patience=100)
    assert not audit._tuning_resolved(
        recorded_rounds=299, best_iteration=200, patience=100
    )


def test_prefix_equivalence_checks_values_and_daily_ranks() -> None:
    old = np.arange(40, dtype=np.float64) / 40.0
    dates = np.repeat(np.asarray([1, 2], dtype=np.int32), 20)
    result = audit._prefix_equivalence(
        old=old,
        current_prefix=old.copy(),
        dates=dates,
        absolute_tolerance=1.0e-7,
        relative_tolerance=1.0e-6,
        minimum_daily_spearman=0.999999,
    )

    assert result["passed"] is True
    assert result["maximum_absolute_difference"] == 0.0


def test_candidate_gate_applies_all_formal_guards() -> None:
    rules = audit.load_study()["decision"]
    annual = [{"relative_mae_harm": 0.0} for _ in audit.MODEL_YEARS]
    passed = audit._candidate_gate(
        paired=_passing_pairs((0.001, 0.002, -0.0005)),
        annual=annual,
        path_guardrail={"rejected": False},
        q_value=0.05,
        rules=rules,
        unresolved=False,
    )
    assert passed["passed"] is True

    unresolved = audit._candidate_gate(
        paired=_passing_pairs((0.001, 0.002, -0.0005)),
        annual=annual,
        path_guardrail={"rejected": False},
        q_value=0.05,
        rules=rules,
        unresolved=True,
    )
    assert unresolved["passed"] is False

    multiplicity_failure = audit._candidate_gate(
        paired=_passing_pairs((0.001, 0.002, -0.0005)),
        annual=annual,
        path_guardrail={"rejected": False},
        q_value=0.11,
        rules=rules,
        unresolved=False,
    )
    assert multiplicity_failure["passed"] is False


def test_winner_order_prefers_top5_then_smaller_capacity() -> None:
    def evidence(policy: str, top5: float, iterations: int) -> dict[str, object]:
        return {
            "policy": policy,
            "iterations_2023_2024_2025": [iterations] * 3,
            "gate": {
                "top5_mfe_delta_2023_2024_2025": [top5] * 3,
                "rank_ic_delta_2023_2024_2025": [0.001] * 3,
                "relative_mae_harm_2023_2024_2025": [0.0] * 3,
            },
        }

    higher_effect = evidence("fixed_512", 0.002, 512)
    lower_effect = evidence("fixed_256", 0.001, 256)
    assert audit._winner_sort_key(higher_effect) > audit._winner_sort_key(lower_effect)

    tied_large = evidence("fixed_512", 0.001, 512)
    tied_small = evidence("fixed_256", 0.001, 256)
    assert audit._winner_sort_key(tied_small) > audit._winner_sort_key(tied_large)


def test_contract_replacement_preserves_unselected_columns() -> None:
    old_raw = np.arange(35, dtype=np.float32).reshape(5, 7)
    old_rank = old_raw / 35.0
    replacement = np.asarray([5.0, 4.0, 3.0, 2.0, 1.0], dtype=np.float32)
    dates = np.asarray([1, 1, 2, 2, 2], dtype=np.int32)

    raw, rank, unchanged = audit._replace_contract_columns(
        old_raw=old_raw,
        old_rank=old_rank,
        replacements={0: replacement},
        date_idx=dates,
    )

    assert unchanged == [1, 2, 3, 4, 5, 6]
    np.testing.assert_array_equal(raw[:, 0], replacement)
    np.testing.assert_array_equal(raw[:, 1:], old_raw[:, 1:])
    np.testing.assert_array_equal(rank[:, 1:], old_rank[:, 1:])


def test_existing_v3_is_reconnected_only_when_it_matches_decision(tmp_path) -> None:
    study = audit.load_study()
    study["contract_v3"]["output_root"] = str(tmp_path)
    decision = {
        "final_capacity_policies": {
            "10": {"policy": "fixed_256"},
            "20": {"policy": audit.BASELINE_POLICY},
        },
        "v3_rebuild_horizons": [10],
        "v3_material_trigger_horizons": [10],
    }
    contract_dir = tmp_path / "contract"
    contract_dir.mkdir()
    files = {}
    for name in ("candidate_rows", "raw_predictions", "date_rank_predictions"):
        path = contract_dir / f"{name}.npy"
        np.save(path, np.asarray([1], dtype=np.int64), allow_pickle=False)
        files[name] = {
            "path": str(path),
            "size": path.stat().st_size,
            "sha256": audit._sha256(path),
        }
    manifest = {
        "schema": audit.CONTRACT_V3_SCHEMA,
        "status": "completed_candidate_aligned_strict_oos_contract",
        "study_id": "seq100_entry_contract_oos_v3",
        "capacity_decision": decision["final_capacity_policies"],
        "rebuild_horizons": [10],
        "material_trigger_horizons": [10],
        "candidate_years": list(audit.CONTRACT_YEARS),
        "physical_columns": list(contract.PHYSICAL_COLUMNS),
        "unchanged_state_and_risk_verified": True,
        "files": files,
    }
    (tmp_path / "contract_manifest.json").write_text(
        json.dumps(manifest), encoding="utf-8"
    )

    loaded = audit._load_existing_contract_v3(study=study, decision=decision)

    assert loaded == manifest
