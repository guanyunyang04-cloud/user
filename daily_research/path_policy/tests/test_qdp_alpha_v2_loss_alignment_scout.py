from __future__ import annotations

import json
from pathlib import Path

import pytest

from daily_research.path_policy import qdp_alpha_v2_loss_alignment_scout as scout


def _write_manifest(path: Path, overrides: dict[str, object] | None = None) -> None:
    payload = {
        "artifact_type": "qdp_training_pack_v1",
        "feature_profile": scout.FEATURE_PROFILE,
        "feature_count": scout.FEATURE_COUNT,
        "label_schema_name": scout.LABEL_SCHEMA,
        "label_schema_version": scout.LABEL_SCHEMA_VERSION,
        "canonical_dataset_id": scout.DATASET_ID,
        "source_pool_view_id": scout.POOL_VIEW_ID,
        "cumulative_horizons": [1, 3, 5, 10, 20],
        "role_years": {
            "train_start_year": 2012,
            "train_end_year": 2023,
            "validation_year": 2024,
            "test_year": 2025,
        },
        "sample_count": 7421714,
    }
    if overrides:
        payload.update(overrides)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_build_training_tasks_fixed_alpha_v2_hybrid_loss_axis() -> None:
    tasks = scout.build_training_tasks(
        output_root=Path("anchor"),
        seeds=(7,),
        loss_profiles=("topn_excess_rank_v1", "decision_score_topk_alignment_v1"),
    )

    assert len(tasks) == 2
    assert {task["loss_profile"] for task in tasks} == {"topn_excess_rank_v1", "decision_score_topk_alignment_v1"}
    assert {task["model_family"] for task in tasks} == {scout.MODEL_FAMILY}
    assert {task["hidden_dim"] for task in tasks} == {256}
    assert {task["batch_size"] for task in tasks} == {512}
    assert {task["training_pack_manifest"] for task in tasks} == {str(scout.TRAINING_PACK_MANIFEST)}
    for task in tasks:
        assert task["decision_cost_bps"] == pytest.approx(20.0)
        assert task["decision_hit_threshold_bps"] == pytest.approx(20.0)
        assert task["decision_drawdown_penalty"] == pytest.approx(0.25)
        assert task["study_family"] == scout.STUDY_FAMILY
        assert task["anchor_run_tag"] == scout.ANCHOR_RUN_TAG
        assert task["output_profile"] == scout.OUTPUT_PROFILE
        assert task["selection_profile"] == scout.SELECTION_PROFILE
        assert task["promotion_allowed"] is False
        assert task["active_execution_strategy_expected_diff"] == "none"
        assert task["personal_topk_required_after_completion"] is True
        command = task["command"]
        assert command[command.index("--forecast-memmap-manifest") + 1] == str(scout.TRAINING_PACK_MANIFEST)
        assert command[command.index("--forecast-loss-profile") + 1] == task["loss_profile"]
        assert command[command.index("--forecast-hidden-dim") + 1] == "256"
        assert command[command.index("--forecast-transformer-layers") + 1] == "4"
        assert command[command.index("--forecast-transformer-heads") + 1] == "8"
        assert command[command.index("--forecast-decision-cost-bps") + 1] == "20"
        assert command[command.index("--forecast-decision-hit-threshold-bps") + 1] == "20"
        assert command[command.index("--forecast-decision-drawdown-penalty") + 1] == "0.25"


def test_sampled_smoke_tasks_do_not_reuse_full_study_tag() -> None:
    full = scout.build_training_tasks(
        output_root=Path("anchor"),
        seeds=(7,),
        loss_profiles=("decision_score_topk_alignment_v1",),
    )[0]
    smoke = scout.build_training_tasks(
        output_root=Path("anchor"),
        seeds=(7,),
        loss_profiles=("decision_score_topk_alignment_v1",),
        max_samples_per_role=64,
        max_samples_per_date_per_role=16,
    )[0]

    assert full["tag"] != smoke["tag"]
    assert "_smoke_" not in full["tag"]
    assert "_smoke_" in smoke["tag"]
    assert full["evidence_grade"] == "single_seed_scout"
    assert smoke["evidence_grade"] == "smoke_only"


def test_date_capped_smoke_task_is_marked_smoke_only() -> None:
    smoke = scout.build_training_tasks(
        output_root=Path("anchor"),
        seeds=(7,),
        loss_profiles=("decision_score_topk_alignment_v1",),
        max_samples_per_date_per_role=16,
    )[0]

    assert "_smoke_" in smoke["tag"]
    assert smoke["evidence_grade"] == "smoke_only"


def test_decision_score_topk_alignment_contract_is_personal_topk_proxy() -> None:
    contract = scout.forecast_loss_profile_contract(
        "decision_score_topk_alignment_v1",
        cumulative_horizons=scout.HORIZON_GRID,
        forecast_horizon=20,
    )

    assert contract["required_output_profile"] == "decision_utility_v1"
    proxy = contract["high_return_proxy_objective"]
    weights = contract["loss_component_weights"]
    assert proxy["enabled"] is True
    assert proxy["method"] == "batch_small_topk_decision_score_alignment_surrogate"
    assert "personal top1/top3/top5" in proxy["description"]
    assert proxy["not_a_backtest"] is True
    assert weights["decision_score_topk_alignment"] > 0.0
    assert weights["decision_rank_aux"] >= 0.50


def test_write_task_list_records_training_pack_validation(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    manifest = tmp_path / "pack" / "qdp_training_pack_manifest.json"
    _write_manifest(manifest)
    monkeypatch.setattr(scout, "TRAINING_PACK_MANIFEST", manifest)

    path = scout.write_task_list(
        output_root=tmp_path / "out",
        seeds=(7,),
        loss_profiles=("decision_score_topk_alignment_v1",),
        enforce_active_artifact_clean=False,
    )
    payload = json.loads(path.read_text(encoding="utf-8"))

    assert payload["stage"] == "alpha_v2_loss_alignment_scout"
    assert payload["training_pack_validation"]["status"] == "ok"
    assert payload["training_pack_validation"]["label_schema"] == scout.LABEL_SCHEMA
    assert payload["training_pack_validation"]["pool_view_id"] == scout.POOL_VIEW_ID
    assert payload["loss_profiles"] == ["decision_score_topk_alignment_v1"]
    assert payload["evaluation_requirements"]["multi_seed_deferred"] is True
    assert payload["evaluation_requirements"]["score_backtest_bridge_deferred"] is True
    assert payload["boundary"]["promotion_allowed"] is False


def test_validate_training_pack_manifest_blocks_schema_mismatch(tmp_path: Path) -> None:
    manifest = tmp_path / "bad_manifest.json"
    _write_manifest(manifest, {"feature_profile": "wrong"})

    result = scout.validate_training_pack_manifest(manifest)

    assert result["status"] == "blocked"
    assert "feature_profile_mismatch" in result["blockers"]


def test_run_training_tasks_blocks_when_manifest_invalid(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    missing = tmp_path / "missing_manifest.json"
    monkeypatch.setattr(scout, "TRAINING_PACK_MANIFEST", missing)
    monkeypatch.setattr(scout, "_active_artifact_has_diff", lambda: False)

    payload = scout.run_training_tasks(
        output_root=tmp_path / "out",
        seeds=(7,),
        loss_profiles=("decision_score_topk_alignment_v1",),
        max_tasks=1,
    )

    assert payload["status"] == "blocked"
    assert "invalid_or_missing_qdp_alpha_v2_training_pack_manifest" in payload["blockers"]
    assert (tmp_path / "out" / "qdp_alpha_v2_loss_alignment_run_summary.json").exists()


def test_main_returns_nonzero_when_training_run_blocks(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    missing = tmp_path / "missing_manifest.json"
    monkeypatch.setattr(scout, "TRAINING_PACK_MANIFEST", missing)
    monkeypatch.setattr(scout, "_active_artifact_has_diff", lambda: False)

    code = scout.main(["--run-training", "--output-root", str(tmp_path / "out"), "--json"])

    assert code == 1
