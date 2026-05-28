from __future__ import annotations

import json
from pathlib import Path

from daily_research.path_policy.stage30_arch_input_scout import (
    BASELINE_STAGE28_SEED7_TAG,
    CONFIRMATION_SEED,
    FINAL_SEED,
    RESEARCH_PROGRAM,
    RUN_TAG,
    SCOUT_SEED,
    STAGE30_SCOUT_SPECS,
    STUDY_FAMILY,
    build_stage30_scout_tasks,
    build_stage31_confirmation_tasks,
    run_stage30_comparison,
    run_stage31_comparison,
    run_stage32_comparison,
    write_stage30_task_list,
)
from daily_research.path_policy.stage_universe_scope import FULL_ROLLING_LIQUID500_SCOPE


def _summary(
    *,
    feature_profile: str,
    universe_size: int = 2430,
    train_rows: int = 452034,
    sector_context_feature_count: int = 0,
) -> dict:
    return {
        "prepared_summary": {"universe_size": universe_size},
        "dataset_manifest": {
            "feature_profile": feature_profile,
            "source_pool_view_kind": "rolling_liquidity",
            "source_pool_view_name": "rolling_liquid500",
            "sample_count_by_role": {"train": train_rows, "validation": 103958, "test": 104972},
            "feature_store_shape": [1699, universe_size, 156],
            "feature_group_counts": {"sector_context": sector_context_feature_count},
            "sector_context_feature_count": sector_context_feature_count,
        },
    }


def _aggregate_row(
    *,
    model_family: str,
    feature_profile: str,
    role: str,
    rank: float,
    spread: float,
    hit: float,
    monthly: float = 0.90,
    negative: int = 1,
    thirty: float = 0.70,
    gap: float = 14.0,
) -> dict:
    return {
        "loss_profile": "target_norm_head_constraint_v1",
        "output_profile": "decision_utility_v1",
        "model_family": model_family,
        "feature_profile": feature_profile,
        "horizon_grid_key": "1,2,3,5,8,10,15,20,30",
        "horizon_grid": "1,2,3,5,8,10,15,20,30",
        "role": role,
        "score_name": "pred_decision_score",
        "seed_count": 1,
        "rank_ic_mean": rank,
        "rank_ic_min": rank,
        "spread_mean": spread,
        "spread_min": spread,
        "hit_lift_mean": hit,
        "hit_lift_min": hit,
        "monthly_positive_rate_mean": monthly,
        "monthly_positive_rate_min": monthly,
        "negative_month_count_max": negative,
        "long_horizon_share_mean": 0.70,
        "thirty_d_concentration_mean": thirty,
        "future_long_horizon_share_mean": 0.43,
        "pred_future_horizon_gap_mean": gap,
    }


def test_stage30_scout_tasks_are_single_seed_full_pool_and_use_expected_manifests() -> None:
    tasks = build_stage30_scout_tasks(output_root=Path("stage30_root"))

    assert RUN_TAG == "mh_stage30_arch_input_scout_20260528_01"
    assert STUDY_FAMILY == "stage30_arch_input_scout"
    assert SCOUT_SEED == 7
    assert CONFIRMATION_SEED == 11
    assert FINAL_SEED == 19
    assert len(STAGE30_SCOUT_SPECS) == 6
    assert len(tasks) == 6
    assert {task["seed"] for task in tasks} == {7}
    assert {task["scope"] for task in tasks} == {"research_shadow_only"}

    raw_tasks = [task for task in tasks if task["feature_profile"] == "raw_kline_context_no_alpha_prior_v1"]
    sector_tasks = [task for task in tasks if task["feature_profile"] == "raw_kline_context_sector_v1"]
    assert len(raw_tasks) == 3
    assert len(sector_tasks) == 3

    sector_anchor = next(task for task in sector_tasks if task["model_family"] == "gru_sequence_static_context")
    sector_anchor_manifest = str(Path(sector_anchor["study_dir"]) / "forecast_dataset_manifest.json")
    for task in tasks:
        command = task["command"]
        assert task["tag"].startswith("mh30_scout_")
        assert not task["tag"].startswith(("mh25_", "mh26_", "mh27_", "mh28_"))
        assert command[command.index("--forecast-model-families") + 1] == task["model_family"]
        assert command[command.index("--forecast-feature-profile") + 1] == task["feature_profile"]
        assert command[command.index("--forecast-loss-profile") + 1] == "target_norm_head_constraint_v1"
        assert command[command.index("--forecast-output-profile") + 1] == "decision_utility_v1"
        assert command[command.index("--forecast-seeds") + 1] == "7"
        assert command[command.index("--forecast-epochs") + 1] == "16"
        assert command[command.index("--forecast-min-epochs") + 1] == "6"
        assert command[command.index("--forecast-early-stop-patience") + 1] == "4"
        assert command[command.index("--pool-view-kind") + 1] == "rolling_liquidity"
        assert command[command.index("--pool-view-name") + 1] == "rolling_liquid500"
        assert command[command.index("--max-universe-size") + 1] == "0"
        assert task["universe_scope"] == FULL_ROLLING_LIQUID500_SCOPE
        assert task["may_touch_active_manifest"] is False
        if task in raw_tasks:
            assert command[command.index("--forecast-memmap-manifest") + 1].endswith("forecast_dataset_manifest.json")
            assert BASELINE_STAGE28_SEED7_TAG not in command[command.index("--forecast-memmap-manifest") + 1]
        elif task is sector_anchor:
            assert "--forecast-memmap-manifest" not in command
        else:
            assert command[command.index("--forecast-memmap-manifest") + 1] == sector_anchor_manifest
        if task["model_family"] == "sector_slot_mixer_sequence":
            assert "--forecast-slot-diagnostics" in command


def test_write_stage30_task_list_marks_scout_not_evidence_grade(tmp_path: Path) -> None:
    path = write_stage30_task_list(tmp_path)

    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["run_tag"] == RUN_TAG
    assert payload["research_program"] == RESEARCH_PROGRAM
    assert payload["study_family"] == STUDY_FAMILY
    assert payload["training_task_count"] == 6
    assert payload["baseline_stage28_seed7_tag"] == BASELINE_STAGE28_SEED7_TAG
    assert payload["single_seed_scout_only"] is True
    assert payload["evidence_grade_architecture_pass"] is False
    assert payload["stage3b_max_candidates"] == 2
    assert payload["boundary"] == "research-only / shadow-only; no active manifest, live/default, production root, paper, or broker integration"


def test_stage30_comparison_blocks_wrong_scope_or_missing_sector_context(tmp_path: Path, monkeypatch) -> None:
    bad_cap80 = tmp_path / "cap80"
    bad_sector = tmp_path / "sector_no_context"
    bad_cap80.mkdir()
    bad_sector.mkdir()
    (bad_cap80 / "study_summary.json").write_text(
        json.dumps(_summary(feature_profile="raw_kline_context_no_alpha_prior_v1", universe_size=80, train_rows=74640)),
        encoding="utf-8",
    )
    (bad_sector / "study_summary.json").write_text(
        json.dumps(_summary(feature_profile="raw_kline_context_sector_v1", sector_context_feature_count=0)),
        encoding="utf-8",
    )
    tasks = [
        {
            "tag": "mh30_scout_raw_patch_transformer_static_context_seed7_20260528_01",
            "study_dir": str(bad_cap80),
            "feature_profile": "raw_kline_context_no_alpha_prior_v1",
            "model_family": "patch_transformer_static_context",
            "universe_scope": FULL_ROLLING_LIQUID500_SCOPE,
            "expected_min_universe_size": 500,
            "expected_min_train_rows": 400000,
        },
        {
            "tag": "mh30_scout_sector_stock_mixer_sequence_seed7_20260528_01",
            "study_dir": str(bad_sector),
            "feature_profile": "raw_kline_context_sector_v1",
            "model_family": "stock_mixer_sequence",
            "universe_scope": FULL_ROLLING_LIQUID500_SCOPE,
            "expected_min_universe_size": 500,
            "expected_min_train_rows": 400000,
        },
    ]
    monkeypatch.setattr(
        "daily_research.path_policy.stage30_arch_input_scout.build_stage30_scout_tasks",
        lambda output_root=None: tasks,
    )

    payload = run_stage30_comparison(tmp_path)

    assert payload["status"] == "blocked_scope_mismatch"
    assert payload["stage3b_candidates"] == []
    reasons = {reason for item in payload["scope_mismatches"] for reason in item["reasons"]}
    assert "universe_scope_mismatch" in reasons
    assert "missing_sector_context_features" in reasons


def test_stage30_comparison_scores_raw_tasks_when_sector_branch_lacks_context(tmp_path: Path, monkeypatch) -> None:
    raw_dir = tmp_path / "raw_candidate"
    sector_anchor_dir = tmp_path / "sector_anchor"
    sector_dependent_dir = tmp_path / "sector_dependent_missing"
    raw_dir.mkdir()
    sector_anchor_dir.mkdir()
    (raw_dir / "study_summary.json").write_text(
        json.dumps(_summary(feature_profile="raw_kline_context_no_alpha_prior_v1")),
        encoding="utf-8",
    )
    (sector_anchor_dir / "study_summary.json").write_text(
        json.dumps(_summary(feature_profile="raw_kline_context_sector_v1", sector_context_feature_count=0)),
        encoding="utf-8",
    )
    tasks = [
        {
            "tag": "mh30_scout_raw_stock_mixer_sequence_seed7_20260528_01",
            "study_dir": str(raw_dir),
            "candidate_id": "raw_stock_mixer_sequence",
            "feature_profile": "raw_kline_context_no_alpha_prior_v1",
            "model_family": "stock_mixer_sequence",
            "universe_scope": FULL_ROLLING_LIQUID500_SCOPE,
            "expected_min_universe_size": 500,
            "expected_min_train_rows": 400000,
        },
        {
            "tag": "mh30_scout_sector_gru_sequence_static_context_seed7_20260528_01",
            "study_dir": str(sector_anchor_dir),
            "candidate_id": "sector_gru_sequence_static_context",
            "feature_profile": "raw_kline_context_sector_v1",
            "model_family": "gru_sequence_static_context",
            "universe_scope": FULL_ROLLING_LIQUID500_SCOPE,
            "expected_min_universe_size": 500,
            "expected_min_train_rows": 400000,
        },
        {
            "tag": "mh30_scout_sector_stock_mixer_sequence_seed7_20260528_01",
            "study_dir": str(sector_dependent_dir),
            "candidate_id": "sector_stock_mixer_sequence",
            "feature_profile": "raw_kline_context_sector_v1",
            "model_family": "stock_mixer_sequence",
            "universe_scope": FULL_ROLLING_LIQUID500_SCOPE,
            "expected_min_universe_size": 500,
            "expected_min_train_rows": 400000,
        },
    ]
    baseline_validation = _aggregate_row(
        model_family="gru_sequence_static_context",
        feature_profile="raw_kline_context_no_alpha_prior_v1",
        role="validation",
        rank=0.10,
        spread=0.03,
        hit=0.01,
        thirty=0.80,
        gap=16.0,
    )
    candidate_validation = _aggregate_row(
        model_family="stock_mixer_sequence",
        feature_profile="raw_kline_context_no_alpha_prior_v1",
        role="validation",
        rank=0.13,
        spread=0.04,
        hit=0.014,
        thirty=0.70,
        gap=14.0,
    )
    candidate_test = _aggregate_row(
        model_family="stock_mixer_sequence",
        feature_profile="raw_kline_context_no_alpha_prior_v1",
        role="test",
        rank=0.08,
        spread=0.02,
        hit=0.004,
    )
    monkeypatch.setattr(
        "daily_research.path_policy.stage30_arch_input_scout.build_stage30_scout_tasks",
        lambda output_root=None: tasks,
    )
    monkeypatch.setattr(
        "daily_research.path_policy.stage30_arch_input_scout.build_output_aux_profile_comparison",
        lambda study_dirs, *, run_tag: {"status": "completed", "run_tag": run_tag, "study_dirs": study_dirs},
    )
    monkeypatch.setattr(
        "daily_research.path_policy.stage30_arch_input_scout.write_output_aux_profile_comparison",
        lambda report_payload, output_dir: {"profile_aggregate_csv": Path(output_dir) / "profile_aggregate.csv"},
    )
    monkeypatch.setattr(
        "daily_research.path_policy.stage30_arch_input_scout.profile_aggregate_rows",
        lambda report_payload: [baseline_validation, candidate_validation, candidate_test],
    )

    payload = run_stage30_comparison(tmp_path)

    assert payload["status"] == "completed_with_branch_blockers"
    assert payload["scope_mismatches"] == []
    assert payload["branch_blockers"][0]["branch"] == "sector_input"
    assert payload["branch_blockers"][0]["reason"] == "missing_sector_context_features"
    assert all(row["feature_profile"] == "raw_kline_context_no_alpha_prior_v1" for row in payload["scout_leaderboard"])
    assert [item["model_family"] for item in payload["stage3b_candidates"]] == ["stock_mixer_sequence"]
    assert (tmp_path / "stage31_candidate_task_list.json").exists()


def test_stage30_scout_gate_uses_validation_ranking_and_test_veto(tmp_path: Path, monkeypatch) -> None:
    tasks = [
        {
            "tag": "mh30_scout_raw_stock_mixer_sequence_seed7_20260528_01",
            "study_dir": str(tmp_path / "candidate"),
            "feature_profile": "raw_kline_context_no_alpha_prior_v1",
            "model_family": "stock_mixer_sequence",
        },
        {
            "tag": "mh30_scout_raw_patch_transformer_static_context_seed7_20260528_01",
            "study_dir": str(tmp_path / "veto"),
            "feature_profile": "raw_kline_context_no_alpha_prior_v1",
            "model_family": "patch_transformer_static_context",
        },
    ]
    baseline_validation = _aggregate_row(
        model_family="gru_sequence_static_context",
        feature_profile="raw_kline_context_no_alpha_prior_v1",
        role="validation",
        rank=0.10,
        spread=0.03,
        hit=0.01,
        thirty=0.80,
        gap=16.0,
    )
    candidate_validation = _aggregate_row(
        model_family="stock_mixer_sequence",
        feature_profile="raw_kline_context_no_alpha_prior_v1",
        role="validation",
        rank=0.13,
        spread=0.04,
        hit=0.014,
        thirty=0.70,
        gap=14.0,
    )
    candidate_test = _aggregate_row(
        model_family="stock_mixer_sequence",
        feature_profile="raw_kline_context_no_alpha_prior_v1",
        role="test",
        rank=0.08,
        spread=0.02,
        hit=0.004,
    )
    veto_validation = _aggregate_row(
        model_family="patch_transformer_static_context",
        feature_profile="raw_kline_context_no_alpha_prior_v1",
        role="validation",
        rank=0.14,
        spread=0.05,
        hit=0.02,
        thirty=0.60,
        gap=13.0,
    )
    veto_test = _aggregate_row(
        model_family="patch_transformer_static_context",
        feature_profile="raw_kline_context_no_alpha_prior_v1",
        role="test",
        rank=0.08,
        spread=-0.01,
        hit=0.004,
    )
    monkeypatch.setattr(
        "daily_research.path_policy.stage30_arch_input_scout.build_stage30_scout_tasks",
        lambda output_root=None: tasks,
    )
    monkeypatch.setattr("daily_research.path_policy.stage30_arch_input_scout._stage30_scope_mismatches", lambda tasks: [])
    monkeypatch.setattr(
        "daily_research.path_policy.stage30_arch_input_scout.build_output_aux_profile_comparison",
        lambda study_dirs, *, run_tag: {"status": "completed", "run_tag": run_tag, "studies": []},
    )
    monkeypatch.setattr(
        "daily_research.path_policy.stage30_arch_input_scout.write_output_aux_profile_comparison",
        lambda report_payload, output_dir: {"profile_aggregate_csv": Path(output_dir) / "profile_aggregate.csv"},
    )
    monkeypatch.setattr(
        "daily_research.path_policy.stage30_arch_input_scout.profile_aggregate_rows",
        lambda report_payload: [baseline_validation, candidate_validation, candidate_test, veto_validation, veto_test],
    )

    payload = run_stage30_comparison(tmp_path)

    assert payload["status"] == "completed"
    assert payload["evidence_grade_architecture_pass"] is False
    assert [item["model_family"] for item in payload["stage3b_candidates"]] == ["stock_mixer_sequence"]
    assert payload["scout_leaderboard"][0]["model_family"] == "patch_transformer_static_context"
    assert payload["scout_leaderboard"][0]["test_veto"] is True
    assert (tmp_path / "stage31_candidate_task_list.json").exists()


def test_stage30_gate_treats_zero_negative_months_as_valid_value(tmp_path: Path, monkeypatch) -> None:
    tasks = [
        {
            "tag": "mh30_scout_raw_patch_transformer_static_context_seed7_20260528_01",
            "study_dir": str(tmp_path / "candidate"),
            "feature_profile": "raw_kline_context_no_alpha_prior_v1",
            "model_family": "patch_transformer_static_context",
        },
    ]
    baseline_validation = _aggregate_row(
        model_family="gru_sequence_static_context",
        feature_profile="raw_kline_context_no_alpha_prior_v1",
        role="validation",
        rank=0.10,
        spread=0.03,
        hit=0.01,
        thirty=0.80,
        gap=16.0,
    )
    candidate_validation = _aggregate_row(
        model_family="patch_transformer_static_context",
        feature_profile="raw_kline_context_no_alpha_prior_v1",
        role="validation",
        rank=0.13,
        spread=0.04,
        hit=0.014,
        monthly=1.0,
        negative=0,
        thirty=0.70,
        gap=14.0,
    )
    candidate_test = _aggregate_row(
        model_family="patch_transformer_static_context",
        feature_profile="raw_kline_context_no_alpha_prior_v1",
        role="test",
        rank=0.08,
        spread=0.02,
        hit=0.004,
    )
    monkeypatch.setattr(
        "daily_research.path_policy.stage30_arch_input_scout.build_stage30_scout_tasks",
        lambda output_root=None: tasks,
    )
    monkeypatch.setattr("daily_research.path_policy.stage30_arch_input_scout._stage30_scope_mismatches", lambda tasks: [])
    monkeypatch.setattr(
        "daily_research.path_policy.stage30_arch_input_scout.build_output_aux_profile_comparison",
        lambda study_dirs, *, run_tag: {"status": "completed", "run_tag": run_tag, "studies": []},
    )
    monkeypatch.setattr(
        "daily_research.path_policy.stage30_arch_input_scout.write_output_aux_profile_comparison",
        lambda report_payload, output_dir: {"profile_aggregate_csv": Path(output_dir) / "profile_aggregate.csv"},
    )
    monkeypatch.setattr(
        "daily_research.path_policy.stage30_arch_input_scout.profile_aggregate_rows",
        lambda report_payload: [baseline_validation, candidate_validation, candidate_test],
    )

    payload = run_stage30_comparison(tmp_path)

    row = payload["scout_leaderboard"][0]
    assert row["validation_negative_month_count_max"] == 0
    assert row["stage3a_hard_gate_pass"] is True
    assert [item["model_family"] for item in payload["stage3b_candidates"]] == ["patch_transformer_static_context"]


def test_stage31_gate_limits_final_confirmation_to_one_candidate(tmp_path: Path, monkeypatch) -> None:
    candidates = [
        {
            "model_family": "stock_mixer_sequence",
            "feature_profile": "raw_kline_context_no_alpha_prior_v1",
            "stage3a_composite": 1.20,
        },
        {
            "model_family": "sector_slot_mixer_sequence",
            "feature_profile": "raw_kline_context_sector_v1",
            "stage3a_composite": 1.10,
        },
    ]
    tasks = build_stage31_confirmation_tasks(candidates, output_root=tmp_path)
    assert len(tasks) == 2
    assert {task["seed"] for task in tasks} == {11}
    assert all(task["command"][task["command"].index("--forecast-epochs") + 1] == "24" for task in tasks)

    baseline_validation = _aggregate_row(
        model_family="gru_sequence_static_context",
        feature_profile="raw_kline_context_no_alpha_prior_v1",
        role="validation",
        rank=0.10,
        spread=0.03,
        hit=0.01,
        monthly=0.90,
        negative=1,
        thirty=0.80,
        gap=16.0,
    )
    baseline_test = _aggregate_row(
        model_family="gru_sequence_static_context",
        feature_profile="raw_kline_context_no_alpha_prior_v1",
        role="test",
        rank=0.08,
        spread=0.02,
        hit=0.004,
        monthly=0.80,
        negative=2,
        thirty=0.78,
        gap=15.0,
    )
    winner_validation = _aggregate_row(
        model_family="stock_mixer_sequence",
        feature_profile="raw_kline_context_no_alpha_prior_v1",
        role="validation",
        rank=0.13,
        spread=0.04,
        hit=0.014,
        monthly=0.90,
        negative=1,
        thirty=0.70,
        gap=14.0,
    )
    winner_test = _aggregate_row(
        model_family="stock_mixer_sequence",
        feature_profile="raw_kline_context_no_alpha_prior_v1",
        role="test",
        rank=0.09,
        spread=0.03,
        hit=0.006,
        monthly=0.80,
        negative=2,
        thirty=0.70,
        gap=14.0,
    )
    runner_validation = _aggregate_row(
        model_family="sector_slot_mixer_sequence",
        feature_profile="raw_kline_context_sector_v1",
        role="validation",
        rank=0.11,
        spread=0.031,
        hit=0.011,
        monthly=0.90,
        negative=1,
        thirty=0.77,
        gap=15.6,
    )
    runner_test = _aggregate_row(
        model_family="sector_slot_mixer_sequence",
        feature_profile="raw_kline_context_sector_v1",
        role="test",
        rank=0.08,
        spread=0.02,
        hit=0.004,
        monthly=0.80,
        negative=2,
        thirty=0.77,
        gap=15.2,
    )
    monkeypatch.setattr("daily_research.path_policy.stage30_arch_input_scout._stage30_scope_mismatches", lambda tasks: [])
    monkeypatch.setattr(
        "daily_research.path_policy.stage30_arch_input_scout.build_stage31_confirmation_tasks",
        lambda candidates, output_root=None: tasks,
    )
    monkeypatch.setattr(
        "daily_research.path_policy.stage30_arch_input_scout._load_stage31_candidates",
        lambda output_root=None: candidates,
    )
    monkeypatch.setattr(
        "daily_research.path_policy.stage30_arch_input_scout.build_output_aux_profile_comparison",
        lambda study_dirs, *, run_tag: {"status": "completed", "run_tag": run_tag, "studies": []},
    )
    monkeypatch.setattr(
        "daily_research.path_policy.stage30_arch_input_scout.write_output_aux_profile_comparison",
        lambda report_payload, output_dir: {"profile_aggregate_csv": Path(output_dir) / "profile_aggregate.csv"},
    )
    monkeypatch.setattr(
        "daily_research.path_policy.stage30_arch_input_scout.profile_aggregate_rows",
        lambda report_payload: [
            baseline_validation,
            baseline_test,
            winner_validation,
            winner_test,
            runner_validation,
            runner_test,
        ],
    )

    payload = run_stage31_comparison(tmp_path)

    assert payload["status"] == "completed"
    assert len(payload["stage3c_candidates"]) == 1
    assert payload["stage3c_candidates"][0]["model_family"] == "stock_mixer_sequence"
    assert (tmp_path / "stage32_candidate_task_list.json").exists()


def test_stage32_final_comparison_requires_full_gate_and_advantage(tmp_path: Path, monkeypatch) -> None:
    candidates = [
        {
            "model_family": "patch_transformer_static_context",
            "feature_profile": "raw_kline_context_no_alpha_prior_v1",
            "tag": "mh30_scout_raw_patch_transformer_static_context_seed7_20260528_01",
            "study_dir": str(tmp_path / "seed7"),
        }
    ]
    baseline_validation = _aggregate_row(
        model_family="gru_sequence_static_context",
        feature_profile="raw_kline_context_no_alpha_prior_v1",
        role="validation",
        rank=0.10,
        spread=0.03,
        hit=0.01,
        monthly=0.90,
        negative=1,
        thirty=0.80,
        gap=16.0,
    )
    baseline_test = _aggregate_row(
        model_family="gru_sequence_static_context",
        feature_profile="raw_kline_context_no_alpha_prior_v1",
        role="test",
        rank=0.10,
        spread=0.03,
        hit=0.01,
        monthly=0.85,
        negative=2,
        thirty=0.78,
        gap=15.0,
    )
    baseline_test["long_horizon_share_mean"] = 0.80
    candidate_validation = _aggregate_row(
        model_family="patch_transformer_static_context",
        feature_profile="raw_kline_context_no_alpha_prior_v1",
        role="validation",
        rank=0.12,
        spread=0.04,
        hit=0.012,
        monthly=0.90,
        negative=1,
        thirty=0.70,
        gap=14.0,
    )
    candidate_test = _aggregate_row(
        model_family="patch_transformer_static_context",
        feature_profile="raw_kline_context_no_alpha_prior_v1",
        role="test",
        rank=0.11,
        spread=0.034,
        hit=0.012,
        monthly=0.85,
        negative=2,
        thirty=0.72,
        gap=14.0,
    )
    candidate_test["long_horizon_share_mean"] = 0.70
    monkeypatch.setattr(
        "daily_research.path_policy.stage30_arch_input_scout._load_stage32_candidates",
        lambda output_root=None: candidates,
    )
    monkeypatch.setattr("daily_research.path_policy.stage30_arch_input_scout._stage30_scope_mismatches", lambda tasks: [])
    monkeypatch.setattr(
        "daily_research.path_policy.stage30_arch_input_scout.build_output_aux_profile_comparison",
        lambda study_dirs, *, run_tag: {"status": "completed", "run_tag": run_tag, "studies": []},
    )
    monkeypatch.setattr(
        "daily_research.path_policy.stage30_arch_input_scout.write_output_aux_profile_comparison",
        lambda report_payload, output_dir: {"profile_aggregate_csv": Path(output_dir) / "profile_aggregate.csv"},
    )
    monkeypatch.setattr(
        "daily_research.path_policy.stage30_arch_input_scout.profile_aggregate_rows",
        lambda report_payload: [baseline_validation, baseline_test, candidate_validation, candidate_test],
    )

    payload = run_stage32_comparison(tmp_path)

    assert payload["status"] == "completed"
    assert payload["architecture_upgrade_allowed"] is True
    assert payload["final_candidate_leaderboard"][0]["stage32_full_gate_pass"] is True
    assert payload["final_candidate_leaderboard"][0]["clear_advantage"] is True
    assert (tmp_path / "stage32_final_arch_confirmation_summary.json").exists()
