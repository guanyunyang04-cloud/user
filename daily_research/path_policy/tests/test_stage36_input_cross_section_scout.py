from __future__ import annotations

import json
from pathlib import Path

from daily_research.path_policy.stage36_input_cross_section_scout import (
    BASELINE_STAGE28_SEED7_TAG,
    CONFIRMATION_SEED,
    FINAL_SEED,
    RESEARCH_PROGRAM,
    RUN_TAG,
    SCOUT_SEED,
    STAGE36_SCOUT_SPECS,
    STAGE37_RUN_TAG,
    STAGE38_RUN_TAG,
    STAGE39_RUN_TAG,
    STUDY_FAMILY,
    _training_progress_payload,
    build_stage36_scout_tasks,
    build_stage37_arch_retest_tasks,
    build_stage38_confirmation_tasks,
    run_stage36_comparison,
    run_stage39_comparison,
    write_stage36_task_list,
)
from daily_research.path_policy.stage_universe_scope import FULL_ROLLING_LIQUID500_SCOPE


def _summary(
    *,
    feature_profile: str,
    universe_size: int = 2430,
    train_rows: int = 452034,
    sector_relative_context_feature_count: int = 10,
    regime_context_feature_count: int = 8,
    sector_view_id: str = "policy_sector_board_view__unit",
) -> dict:
    return {
        "prepared_summary": {"universe_size": universe_size},
        "dataset_manifest": {
            "feature_profile": feature_profile,
            "source_pool_view_kind": "rolling_liquidity",
            "source_pool_view_name": "rolling_liquid500",
            "source_sector_board_view_id": sector_view_id,
            "sample_count_by_role": {"train": train_rows, "validation": 103958, "test": 104972},
            "feature_store_shape": [1699, universe_size, 192],
            "feature_group_counts": {
                "sector_relative_context": sector_relative_context_feature_count,
                "regime_context": regime_context_feature_count,
            },
            "sector_relative_context_feature_count": sector_relative_context_feature_count,
            "regime_context_feature_count": regime_context_feature_count,
            "feature_profile_audit": {
                "feature_profile": feature_profile,
                "group_stats": {
                    "sector_relative_context": {"feature_count": sector_relative_context_feature_count, "finite_ratio": 1.0},
                    "regime_context": {"feature_count": regime_context_feature_count, "finite_ratio": 1.0},
                },
                "retained_groups": {
                    "raw_kline": True,
                    "sector_relative_context": sector_relative_context_feature_count > 0,
                    "regime_context": regime_context_feature_count > 0,
                },
                "future_leakage_smoke": {"passed": True},
            },
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


def test_stage36_scout_tasks_are_gru_only_full_pool_and_use_expected_profiles() -> None:
    tasks = build_stage36_scout_tasks(output_root=Path("stage36_root"))

    assert RUN_TAG == "mh_stage36_input_cross_section_scout_20260529_01"
    assert STUDY_FAMILY == "stage36_input_cross_section_scout"
    assert SCOUT_SEED == 7
    assert CONFIRMATION_SEED == 11
    assert FINAL_SEED == 19
    assert len(STAGE36_SCOUT_SPECS) == 3
    assert len(tasks) == 3
    assert {task["seed"] for task in tasks} == {7}
    assert {task["model_family"] for task in tasks} == {"gru_sequence_static_context"}
    assert {task["feature_profile"] for task in tasks} == {
        "raw_kline_context_sector_relative_v1",
        "raw_kline_context_regime_v1",
        "raw_kline_context_sector_relative_regime_v1",
    }

    for task in tasks:
        command = task["command"]
        assert task["tag"].startswith("mh36_scout_")
        assert not task["tag"].startswith(("mh28_", "mh30_", "mh33_", "mh35_"))
        assert command[command.index("--forecast-model-families") + 1] == "gru_sequence_static_context"
        assert command[command.index("--forecast-loss-profile") + 1] == "target_norm_head_constraint_v1"
        assert command[command.index("--forecast-output-profile") + 1] == "decision_utility_v1"
        assert command[command.index("--forecast-seeds") + 1] == "7"
        assert command[command.index("--forecast-epochs") + 1] == "16"
        assert command[command.index("--pool-view-kind") + 1] == "rolling_liquidity"
        assert command[command.index("--pool-view-name") + 1] == "rolling_liquid500"
        assert command[command.index("--max-universe-size") + 1] == "0"
        assert command[command.index("--sector-board-view-id") + 1].startswith("policy_sector_board_view__")
        assert task["universe_scope"] == FULL_ROLLING_LIQUID500_SCOPE


def test_write_stage36_task_list_marks_single_seed_scout_only(tmp_path: Path) -> None:
    path = write_stage36_task_list(tmp_path)

    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["run_tag"] == RUN_TAG
    assert payload["research_program"] == RESEARCH_PROGRAM
    assert payload["study_family"] == STUDY_FAMILY
    assert payload["training_task_count"] == 3
    assert payload["baseline_stage28_seed7_tag"] == BASELINE_STAGE28_SEED7_TAG
    assert payload["single_seed_scout_only"] is True
    assert payload["evidence_grade_input_or_architecture_pass"] is False


def test_stage36_progress_payload_supports_short_poll_recovery() -> None:
    payload = _training_progress_payload(
        status="running",
        completed=[],
        failed=[],
        run_tag=RUN_TAG,
        study_family=STUDY_FAMILY,
        current_tag="mh36_scout_raw_kline_context_regime_v1_seed7_20260529_01",
        current_step=0,
        total_steps=3,
    )

    assert payload["current_tag"] == "mh36_scout_raw_kline_context_regime_v1_seed7_20260529_01"
    assert payload["completed_steps"] == 0
    assert payload["total_steps"] == 3
    assert payload["poll_window_seconds"] == 7200
    assert payload["next_decision"] == "continue_short_polling"


def test_stage36_comparison_blocks_missing_required_feature_groups(tmp_path: Path, monkeypatch) -> None:
    bad_sector = tmp_path / "sector_relative_missing"
    bad_regime = tmp_path / "regime_missing"
    bad_sector.mkdir()
    bad_regime.mkdir()
    (bad_sector / "study_summary.json").write_text(
        json.dumps(
            _summary(
                feature_profile="raw_kline_context_sector_relative_v1",
                sector_relative_context_feature_count=0,
                sector_view_id="",
            )
        ),
        encoding="utf-8",
    )
    (bad_regime / "study_summary.json").write_text(
        json.dumps(
            _summary(
                feature_profile="raw_kline_context_regime_v1",
                regime_context_feature_count=0,
            )
        ),
        encoding="utf-8",
    )
    tasks = [
        {
            "tag": "mh36_scout_raw_kline_context_sector_relative_v1_seed7_20260529_01",
            "study_dir": str(bad_sector),
            "candidate_id": "raw_kline_context_sector_relative_v1",
            "feature_profile": "raw_kline_context_sector_relative_v1",
            "model_family": "gru_sequence_static_context",
            "universe_scope": FULL_ROLLING_LIQUID500_SCOPE,
            "expected_min_universe_size": 500,
            "expected_min_train_rows": 400000,
        },
        {
            "tag": "mh36_scout_raw_kline_context_regime_v1_seed7_20260529_01",
            "study_dir": str(bad_regime),
            "candidate_id": "raw_kline_context_regime_v1",
            "feature_profile": "raw_kline_context_regime_v1",
            "model_family": "gru_sequence_static_context",
            "universe_scope": FULL_ROLLING_LIQUID500_SCOPE,
            "expected_min_universe_size": 500,
            "expected_min_train_rows": 400000,
        },
    ]
    monkeypatch.setattr(
        "daily_research.path_policy.stage36_input_cross_section_scout.build_stage36_scout_tasks",
        lambda output_root=None: tasks,
    )

    payload = run_stage36_comparison(tmp_path)

    assert payload["status"] == "blocked_scope_or_feature_mismatch"
    assert payload["stage38_candidates"] == []
    reasons = {reason for item in payload["scope_mismatches"] for reason in item["reasons"]}
    assert "missing_sector_relative_context_features" in reasons
    assert "missing_sector_board_view_id" in reasons
    assert "missing_regime_context_features" in reasons


def test_stage36_comparison_selects_confirmation_and_arch_retest_candidates(tmp_path: Path, monkeypatch) -> None:
    tasks = [
        {
            "tag": "mh36_scout_raw_kline_context_sector_relative_regime_v1_seed7_20260529_01",
            "study_dir": str(tmp_path / "combined"),
            "candidate_id": "raw_kline_context_sector_relative_regime_v1",
            "feature_profile": "raw_kline_context_sector_relative_regime_v1",
            "model_family": "gru_sequence_static_context",
        },
        {
            "tag": "mh36_scout_raw_kline_context_regime_v1_seed7_20260529_01",
            "study_dir": str(tmp_path / "regime"),
            "candidate_id": "raw_kline_context_regime_v1",
            "feature_profile": "raw_kline_context_regime_v1",
            "model_family": "gru_sequence_static_context",
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
    baseline_test = _aggregate_row(
        model_family="gru_sequence_static_context",
        feature_profile="raw_kline_context_no_alpha_prior_v1",
        role="test",
        rank=0.09,
        spread=0.03,
        hit=0.01,
        thirty=0.78,
        gap=15.0,
    )
    combined_validation = _aggregate_row(
        model_family="gru_sequence_static_context",
        feature_profile="raw_kline_context_sector_relative_regime_v1",
        role="validation",
        rank=0.13,
        spread=0.041,
        hit=0.011,
        thirty=0.68,
        gap=13.8,
    )
    combined_test = _aggregate_row(
        model_family="gru_sequence_static_context",
        feature_profile="raw_kline_context_sector_relative_regime_v1",
        role="test",
        rank=0.085,
        spread=0.028,
        hit=0.0085,
        thirty=0.66,
        gap=13.5,
    )
    regime_validation = _aggregate_row(
        model_family="gru_sequence_static_context",
        feature_profile="raw_kline_context_regime_v1",
        role="validation",
        rank=0.12,
        spread=0.04,
        hit=0.0095,
        thirty=0.60,
        gap=13.0,
    )
    regime_test = _aggregate_row(
        model_family="gru_sequence_static_context",
        feature_profile="raw_kline_context_regime_v1",
        role="test",
        rank=0.082,
        spread=0.027,
        hit=0.007,
        thirty=0.62,
        gap=13.2,
    )
    monkeypatch.setattr(
        "daily_research.path_policy.stage36_input_cross_section_scout.build_stage36_scout_tasks",
        lambda output_root=None: tasks,
    )
    monkeypatch.setattr("daily_research.path_policy.stage36_input_cross_section_scout._stage36_scope_mismatches", lambda tasks: [])
    monkeypatch.setattr(
        "daily_research.path_policy.stage36_input_cross_section_scout._write_stage36_feature_profile_audit",
        lambda output_root=None, tasks=None: tmp_path / "stage36_feature_profile_audit.json",
    )
    monkeypatch.setattr(
        "daily_research.path_policy.stage36_input_cross_section_scout.build_output_aux_profile_comparison",
        lambda study_dirs, *, run_tag: {"status": "completed", "run_tag": run_tag, "studies": []},
    )
    monkeypatch.setattr(
        "daily_research.path_policy.stage36_input_cross_section_scout.write_output_aux_profile_comparison",
        lambda report_payload, output_dir: {"profile_aggregate_csv": Path(output_dir) / "profile_aggregate.csv"},
    )
    monkeypatch.setattr(
        "daily_research.path_policy.stage36_input_cross_section_scout.profile_aggregate_rows",
        lambda report_payload: [
            baseline_validation,
            baseline_test,
            combined_validation,
            combined_test,
            regime_validation,
            regime_test,
        ],
    )

    payload = run_stage36_comparison(tmp_path)

    assert payload["status"] == "completed"
    assert [item["feature_profile"] for item in payload["stage38_candidates"]] == ["raw_kline_context_sector_relative_regime_v1"]
    assert payload["stage37_input_candidate"]["feature_profile"] == "raw_kline_context_sector_relative_regime_v1"
    assert (tmp_path / "stage37_arch_retest_task_list.json").exists()
    assert (tmp_path / "stage38_confirmation_task_list.json").exists()


def test_stage37_arch_retest_tasks_use_best_stage36_input_manifest(tmp_path: Path) -> None:
    candidates = [
        {
            "tag": "mh36_scout_raw_kline_context_sector_relative_regime_v1_seed7_20260529_01",
            "study_dir": str(tmp_path / "best_input"),
            "candidate_id": "raw_kline_context_sector_relative_regime_v1",
            "feature_profile": "raw_kline_context_sector_relative_regime_v1",
            "model_family": "gru_sequence_static_context",
            "stage36_composite": 1.12,
        }
    ]
    tasks = build_stage37_arch_retest_tasks(candidates, output_root=tmp_path)

    assert STAGE37_RUN_TAG == "mh_stage37_cross_section_arch_retest_20260529_01"
    assert len(tasks) == 2
    assert {task["seed"] for task in tasks} == {7}
    assert {task["model_family"] for task in tasks} == {"stock_mixer_sequence", "sector_slot_mixer_sequence"}
    assert {
        task["command"][task["command"].index("--forecast-feature-profile") + 1]
        for task in tasks
    } == {"raw_kline_context_sector_relative_regime_v1"}
    for task in tasks:
        assert task["command"][task["command"].index("--forecast-memmap-manifest") + 1].endswith("forecast_dataset_manifest.json")


def test_stage38_confirmation_tasks_limit_to_two_and_preserve_manifest_sources(tmp_path: Path) -> None:
    candidates = [
        {
            "tag": "mh36_scout_raw_kline_context_sector_relative_regime_v1_seed7_20260529_01",
            "study_dir": str(tmp_path / "input_a"),
            "candidate_id": "raw_kline_context_sector_relative_regime_v1",
            "feature_profile": "raw_kline_context_sector_relative_regime_v1",
            "model_family": "gru_sequence_static_context",
        },
        {
            "tag": "mh37_retest_stock_mixer_sequence_raw_kline_context_sector_relative_regime_v1_seed7_20260529_01",
            "study_dir": str(tmp_path / "arch_b"),
            "candidate_id": "stock_mixer_sequence_raw_kline_context_sector_relative_regime_v1",
            "feature_profile": "raw_kline_context_sector_relative_regime_v1",
            "model_family": "stock_mixer_sequence",
            "manifest_source_study_dir": str(tmp_path / "input_a"),
        },
        {
            "tag": "mh36_scout_raw_kline_context_regime_v1_seed7_20260529_01",
            "study_dir": str(tmp_path / "input_c"),
            "candidate_id": "raw_kline_context_regime_v1",
            "feature_profile": "raw_kline_context_regime_v1",
            "model_family": "gru_sequence_static_context",
        },
    ]

    tasks = build_stage38_confirmation_tasks(candidates, output_root=tmp_path)

    assert STAGE38_RUN_TAG == "mh_stage38_input_arch_confirmation_20260529_01"
    assert len(tasks) == 2
    assert {task["seed"] for task in tasks} == {11}
    assert not any(task["tag"].startswith("mh39_") for task in tasks)


def test_stage39_final_comparison_requires_core_metric_gain_not_only_concentration(tmp_path: Path, monkeypatch) -> None:
    candidates = [
        {
            "model_family": "gru_sequence_static_context",
            "feature_profile": "raw_kline_context_sector_relative_regime_v1",
            "tag": "mh36_scout_raw_kline_context_sector_relative_regime_v1_seed7_20260529_01",
            "study_dir": str(tmp_path / "seed7"),
        }
    ]
    baseline_validation = _aggregate_row(
        model_family="gru_sequence_static_context",
        feature_profile="raw_kline_context_no_alpha_prior_v1",
        role="validation",
        rank=0.12,
        spread=0.038,
        hit=0.017,
        monthly=0.91,
        negative=1,
        thirty=0.68,
        gap=14.5,
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
        model_family="gru_sequence_static_context",
        feature_profile="raw_kline_context_sector_relative_regime_v1",
        role="validation",
        rank=0.105,
        spread=0.031,
        hit=0.0105,
        monthly=0.85,
        negative=2,
        thirty=0.60,
        gap=13.0,
    )
    candidate_test = _aggregate_row(
        model_family="gru_sequence_static_context",
        feature_profile="raw_kline_context_sector_relative_regime_v1",
        role="test",
        rank=0.095,
        spread=0.029,
        hit=0.009,
        monthly=0.83,
        negative=2,
        thirty=0.58,
        gap=12.8,
    )
    candidate_test["long_horizon_share_mean"] = 0.60
    monkeypatch.setattr(
        "daily_research.path_policy.stage36_input_cross_section_scout._load_stage39_candidates",
        lambda output_root=None: candidates,
    )
    monkeypatch.setattr("daily_research.path_policy.stage36_input_cross_section_scout._stage36_scope_mismatches", lambda tasks: [])
    monkeypatch.setattr(
        "daily_research.path_policy.stage36_input_cross_section_scout.build_output_aux_profile_comparison",
        lambda study_dirs, *, run_tag: {"status": "completed", "run_tag": run_tag, "studies": []},
    )
    monkeypatch.setattr(
        "daily_research.path_policy.stage36_input_cross_section_scout.write_output_aux_profile_comparison",
        lambda report_payload, output_dir: {"profile_aggregate_csv": Path(output_dir) / "profile_aggregate.csv"},
    )
    monkeypatch.setattr(
        "daily_research.path_policy.stage36_input_cross_section_scout.profile_aggregate_rows",
        lambda report_payload: [baseline_validation, baseline_test, candidate_validation, candidate_test],
    )

    payload = run_stage39_comparison(tmp_path)

    assert STAGE39_RUN_TAG == "mh_stage39_final_input_arch_confirmation_20260529_01"
    assert payload["status"] == "completed"
    assert payload["input_or_architecture_upgrade_allowed"] is False
    assert payload["final_candidate_leaderboard"][0]["stage39_full_gate_pass"] is False
    assert payload["test_metric_delta_candidate_minus_baseline"]["thirty_d_concentration_mean_delta"] < 0.0
    assert payload["test_metric_delta_candidate_minus_baseline"]["rank_ic_mean_delta"] < 0.0
