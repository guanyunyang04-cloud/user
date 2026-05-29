from __future__ import annotations

import json
from pathlib import Path

from daily_research.path_policy.stage33_sector_input_repair_scout import (
    BASELINE_STAGE28_SEED7_TAG,
    CONFIRMATION_SEED,
    FINAL_SEED,
    RESEARCH_PROGRAM,
    RUN_TAG,
    SCOUT_SEED,
    STAGE33_SCOUT_SPECS,
    STAGE34_RUN_TAG,
    STAGE35_RUN_TAG,
    STUDY_FAMILY,
    build_stage33_scout_tasks,
    build_stage34_confirmation_tasks,
    run_stage33_comparison,
    run_stage34_comparison,
    run_stage35_comparison,
    write_stage33_task_list,
)
from daily_research.path_policy.stage_universe_scope import FULL_ROLLING_LIQUID500_SCOPE


def _summary(
    *,
    feature_profile: str = "raw_kline_context_sector_v1",
    universe_size: int = 2430,
    train_rows: int = 452034,
    sector_context_feature_count: int = 4,
    sector_view_id: str = "policy_sector_board_view__unit",
) -> dict:
    return {
        "prepared_summary": {"universe_size": universe_size},
        "dataset_manifest": {
            "feature_profile": feature_profile,
            "source_pool_view_kind": "rolling_liquidity",
            "source_pool_view_name": "rolling_liquid500",
            "source_sector_board_view_id": sector_view_id,
            "source_sector_board_view_kind": "latest_static_snapshot" if sector_view_id else "",
            "sample_count_by_role": {"train": train_rows, "validation": 103958, "test": 104972},
            "feature_store_shape": [1699, universe_size, 192],
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


def test_stage33_scout_tasks_are_sector_only_full_pool_and_request_sector_board_view() -> None:
    tasks = build_stage33_scout_tasks(output_root=Path("stage33_root"))

    assert RUN_TAG == "mh_stage33_sector_input_repair_scout_20260529_01"
    assert STUDY_FAMILY == "stage33_sector_input_repair_scout"
    assert SCOUT_SEED == 7
    assert CONFIRMATION_SEED == 11
    assert FINAL_SEED == 19
    assert len(STAGE33_SCOUT_SPECS) == 3
    assert len(tasks) == 3
    assert {task["seed"] for task in tasks} == {7}
    assert {task["feature_profile"] for task in tasks} == {"raw_kline_context_sector_v1"}
    assert {task["scope"] for task in tasks} == {"research_shadow_only"}

    anchor = next(task for task in tasks if task["model_family"] == "gru_sequence_static_context")
    anchor_manifest = str(Path(anchor["study_dir"]) / "forecast_dataset_manifest.json")
    for task in tasks:
        command = task["command"]
        assert task["tag"].startswith("mh33_scout_sector_")
        assert not task["tag"].startswith(("mh28_", "mh30_", "mh31_", "mh32_"))
        assert command[command.index("--forecast-model-families") + 1] == task["model_family"]
        assert command[command.index("--forecast-feature-profile") + 1] == "raw_kline_context_sector_v1"
        assert command[command.index("--forecast-loss-profile") + 1] == "target_norm_head_constraint_v1"
        assert command[command.index("--forecast-output-profile") + 1] == "decision_utility_v1"
        assert command[command.index("--forecast-seeds") + 1] == "7"
        assert command[command.index("--forecast-epochs") + 1] == "16"
        assert command[command.index("--pool-view-kind") + 1] == "rolling_liquidity"
        assert command[command.index("--pool-view-name") + 1] == "rolling_liquid500"
        assert command[command.index("--max-universe-size") + 1] == "0"
        assert command[command.index("--sector-board-view-id") + 1].startswith("policy_sector_board_view__")
        assert command[command.index("--sector-board-view-kind") + 1] == "latest_static_snapshot"
        assert task["universe_scope"] == FULL_ROLLING_LIQUID500_SCOPE
        assert task["may_touch_active_manifest"] is False
        if task is anchor:
            assert "--forecast-memmap-manifest" not in command
            assert task["reuses_forecast_memmap_manifest"] is False
        else:
            assert command[command.index("--forecast-memmap-manifest") + 1] == anchor_manifest
            assert task["reuses_forecast_memmap_manifest"] is True
        if task["model_family"] == "sector_slot_mixer_sequence":
            assert "--forecast-slot-diagnostics" in command


def test_write_stage33_task_list_marks_single_seed_scout_only(tmp_path: Path) -> None:
    path = write_stage33_task_list(tmp_path)

    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["run_tag"] == RUN_TAG
    assert payload["research_program"] == RESEARCH_PROGRAM
    assert payload["study_family"] == STUDY_FAMILY
    assert payload["training_task_count"] == 3
    assert payload["baseline_stage28_seed7_tag"] == BASELINE_STAGE28_SEED7_TAG
    assert payload["single_seed_scout_only"] is True
    assert payload["evidence_grade_input_or_architecture_pass"] is False
    assert payload["stage3d_dataset_gate"]["sector_context_feature_count_min"] == 1


def test_stage33_comparison_blocks_zero_sector_context_or_missing_sector_view(tmp_path: Path, monkeypatch) -> None:
    bad_sector = tmp_path / "sector_no_context"
    missing_view = tmp_path / "sector_missing_view"
    bad_sector.mkdir()
    missing_view.mkdir()
    (bad_sector / "study_summary.json").write_text(
        json.dumps(_summary(sector_context_feature_count=0)),
        encoding="utf-8",
    )
    (missing_view / "study_summary.json").write_text(
        json.dumps(_summary(sector_context_feature_count=4, sector_view_id="")),
        encoding="utf-8",
    )
    tasks = [
        {
            "tag": "mh33_scout_sector_gru_sequence_static_context_seed7_20260529_01",
            "study_dir": str(bad_sector),
            "candidate_id": "sector_gru_sequence_static_context",
            "feature_profile": "raw_kline_context_sector_v1",
            "model_family": "gru_sequence_static_context",
            "universe_scope": FULL_ROLLING_LIQUID500_SCOPE,
            "expected_min_universe_size": 500,
            "expected_min_train_rows": 400000,
        },
        {
            "tag": "mh33_scout_sector_stock_mixer_sequence_seed7_20260529_01",
            "study_dir": str(missing_view),
            "candidate_id": "sector_stock_mixer_sequence",
            "feature_profile": "raw_kline_context_sector_v1",
            "model_family": "stock_mixer_sequence",
            "universe_scope": FULL_ROLLING_LIQUID500_SCOPE,
            "expected_min_universe_size": 500,
            "expected_min_train_rows": 400000,
        },
    ]
    monkeypatch.setattr(
        "daily_research.path_policy.stage33_sector_input_repair_scout.build_stage33_scout_tasks",
        lambda output_root=None: tasks,
    )

    payload = run_stage33_comparison(tmp_path)

    assert payload["status"] == "blocked_scope_mismatch"
    assert payload["stage3e_candidates"] == []
    reasons = {reason for item in payload["scope_mismatches"] for reason in item["reasons"]}
    assert "missing_sector_context_features" in reasons
    assert "missing_sector_board_view_id" in reasons


def test_stage33_scout_gate_uses_sector_validation_composite_and_test_veto(tmp_path: Path, monkeypatch) -> None:
    tasks = [
        {
            "tag": "mh33_scout_sector_stock_mixer_sequence_seed7_20260529_01",
            "study_dir": str(tmp_path / "candidate"),
            "candidate_id": "sector_stock_mixer_sequence",
            "feature_profile": "raw_kline_context_sector_v1",
            "model_family": "stock_mixer_sequence",
        },
        {
            "tag": "mh33_scout_sector_sector_slot_mixer_sequence_seed7_20260529_01",
            "study_dir": str(tmp_path / "veto"),
            "candidate_id": "sector_sector_slot_mixer_sequence",
            "feature_profile": "raw_kline_context_sector_v1",
            "model_family": "sector_slot_mixer_sequence",
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
        feature_profile="raw_kline_context_sector_v1",
        role="validation",
        rank=0.13,
        spread=0.04,
        hit=0.014,
        thirty=0.70,
        gap=14.0,
    )
    candidate_test = _aggregate_row(
        model_family="stock_mixer_sequence",
        feature_profile="raw_kline_context_sector_v1",
        role="test",
        rank=0.08,
        spread=0.02,
        hit=0.004,
    )
    veto_validation = _aggregate_row(
        model_family="sector_slot_mixer_sequence",
        feature_profile="raw_kline_context_sector_v1",
        role="validation",
        rank=0.14,
        spread=0.05,
        hit=0.02,
        thirty=0.60,
        gap=13.0,
    )
    veto_test = _aggregate_row(
        model_family="sector_slot_mixer_sequence",
        feature_profile="raw_kline_context_sector_v1",
        role="test",
        rank=0.08,
        spread=-0.01,
        hit=0.004,
    )
    monkeypatch.setattr(
        "daily_research.path_policy.stage33_sector_input_repair_scout.build_stage33_scout_tasks",
        lambda output_root=None: tasks,
    )
    monkeypatch.setattr("daily_research.path_policy.stage33_sector_input_repair_scout._stage33_scope_mismatches", lambda tasks: [])
    monkeypatch.setattr(
        "daily_research.path_policy.stage33_sector_input_repair_scout.build_output_aux_profile_comparison",
        lambda study_dirs, *, run_tag: {"status": "completed", "run_tag": run_tag, "studies": []},
    )
    monkeypatch.setattr(
        "daily_research.path_policy.stage33_sector_input_repair_scout.write_output_aux_profile_comparison",
        lambda report_payload, output_dir: {"profile_aggregate_csv": Path(output_dir) / "profile_aggregate.csv"},
    )
    monkeypatch.setattr(
        "daily_research.path_policy.stage33_sector_input_repair_scout.profile_aggregate_rows",
        lambda report_payload: [baseline_validation, candidate_validation, candidate_test, veto_validation, veto_test],
    )

    payload = run_stage33_comparison(tmp_path)

    assert payload["status"] == "completed"
    assert payload["evidence_grade_input_or_architecture_pass"] is False
    assert [item["model_family"] for item in payload["stage3e_candidates"]] == ["stock_mixer_sequence"]
    assert payload["sector_scout_leaderboard"][0]["model_family"] == "sector_slot_mixer_sequence"
    assert payload["sector_scout_leaderboard"][0]["test_veto"] is True
    assert (tmp_path / "stage34_confirmation_task_list.json").exists()


def test_stage34_confirmation_limits_final_to_one_candidate(tmp_path: Path, monkeypatch) -> None:
    candidates = [
        {
            "model_family": "stock_mixer_sequence",
            "feature_profile": "raw_kline_context_sector_v1",
            "stage3d_composite": 1.20,
        },
        {
            "model_family": "sector_slot_mixer_sequence",
            "feature_profile": "raw_kline_context_sector_v1",
            "stage3d_composite": 1.10,
        },
    ]
    tasks = build_stage34_confirmation_tasks(candidates, output_root=tmp_path)
    assert STAGE34_RUN_TAG == "mh_stage34_sector_input_confirmation_20260529_01"
    assert len(tasks) == 2
    assert {task["seed"] for task in tasks} == {11}
    assert all(task["tag"].startswith("mh34_confirm_sector_") for task in tasks)

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
        feature_profile="raw_kline_context_sector_v1",
        role="validation",
        rank=0.13,
        spread=0.04,
        hit=0.014,
    )
    winner_test = _aggregate_row(
        model_family="stock_mixer_sequence",
        feature_profile="raw_kline_context_sector_v1",
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
    monkeypatch.setattr("daily_research.path_policy.stage33_sector_input_repair_scout._stage33_scope_mismatches", lambda tasks: [])
    monkeypatch.setattr(
        "daily_research.path_policy.stage33_sector_input_repair_scout.build_stage34_confirmation_tasks",
        lambda candidates, output_root=None: tasks,
    )
    monkeypatch.setattr(
        "daily_research.path_policy.stage33_sector_input_repair_scout._load_stage34_candidates",
        lambda output_root=None: candidates,
    )
    monkeypatch.setattr(
        "daily_research.path_policy.stage33_sector_input_repair_scout.build_output_aux_profile_comparison",
        lambda study_dirs, *, run_tag: {"status": "completed", "run_tag": run_tag, "studies": []},
    )
    monkeypatch.setattr(
        "daily_research.path_policy.stage33_sector_input_repair_scout.write_output_aux_profile_comparison",
        lambda report_payload, output_dir: {"profile_aggregate_csv": Path(output_dir) / "profile_aggregate.csv"},
    )
    monkeypatch.setattr(
        "daily_research.path_policy.stage33_sector_input_repair_scout.profile_aggregate_rows",
        lambda report_payload: [baseline_test, winner_validation, winner_test, runner_validation, runner_test],
    )

    payload = run_stage34_comparison(tmp_path)

    assert payload["status"] == "completed"
    assert len(payload["stage3f_candidates"]) == 1
    assert payload["stage3f_candidates"][0]["model_family"] == "stock_mixer_sequence"
    assert (tmp_path / "stage35_final_task_list.json").exists()


def test_stage35_final_comparison_requires_full_gate_and_advantage(tmp_path: Path, monkeypatch) -> None:
    candidates = [
        {
            "model_family": "stock_mixer_sequence",
            "feature_profile": "raw_kline_context_sector_v1",
            "tag": "mh33_scout_sector_stock_mixer_sequence_seed7_20260529_01",
            "study_dir": str(tmp_path / "seed7"),
        }
    ]
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
        model_family="stock_mixer_sequence",
        feature_profile="raw_kline_context_sector_v1",
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
        model_family="stock_mixer_sequence",
        feature_profile="raw_kline_context_sector_v1",
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
        "daily_research.path_policy.stage33_sector_input_repair_scout._load_stage35_candidates",
        lambda output_root=None: candidates,
    )
    monkeypatch.setattr("daily_research.path_policy.stage33_sector_input_repair_scout._stage33_scope_mismatches", lambda tasks: [])
    monkeypatch.setattr(
        "daily_research.path_policy.stage33_sector_input_repair_scout.build_output_aux_profile_comparison",
        lambda study_dirs, *, run_tag: {"status": "completed", "run_tag": run_tag, "studies": []},
    )
    monkeypatch.setattr(
        "daily_research.path_policy.stage33_sector_input_repair_scout.write_output_aux_profile_comparison",
        lambda report_payload, output_dir: {"profile_aggregate_csv": Path(output_dir) / "profile_aggregate.csv"},
    )
    monkeypatch.setattr(
        "daily_research.path_policy.stage33_sector_input_repair_scout.profile_aggregate_rows",
        lambda report_payload: [baseline_test, candidate_validation, candidate_test],
    )

    payload = run_stage35_comparison(tmp_path)

    assert STAGE35_RUN_TAG == "mh_stage35_sector_input_final_confirmation_20260529_01"
    assert payload["status"] == "completed"
    assert payload["input_or_architecture_upgrade_allowed"] is True
    assert payload["final_candidate_leaderboard"][0]["stage35_full_gate_pass"] is True
    assert payload["final_candidate_leaderboard"][0]["clear_advantage"] is True
    assert payload["baseline_test_aggregate"]["model_family"] == "gru_sequence_static_context"
    assert payload["baseline_test_aggregate"]["seed_count"] == 1
    assert payload["candidate_test_aggregate"]["model_family"] == "stock_mixer_sequence"
    assert payload["candidate_test_aggregate"]["rank_ic_mean"] == 0.11
    assert payload["candidate_test_aggregate"]["hit_lift_min"] == 0.012
    assert payload["test_metric_delta_candidate_minus_baseline"]["rank_ic_mean_delta"] > 0.0
    assert payload["test_metric_delta_candidate_minus_baseline"]["thirty_d_concentration_mean_delta"] < 0.0
    assert payload["final_candidate_leaderboard"][0]["test_rank_ic_mean"] == 0.11
    assert payload["final_candidate_leaderboard"][0]["test_thirty_d_concentration_mean"] == 0.72
    assert (tmp_path / "stage35_final_sector_confirmation_summary.json").exists()
