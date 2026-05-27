import json
from pathlib import Path

from daily_research.path_policy.stage27_target_head_stability import (
    BASELINE_LONG_HORIZON_SHARE,
    RESEARCH_PROGRAM,
    RUN_TAG,
    STAGE27_CANDIDATES,
    STUDY_FAMILY,
    build_stage27_training_tasks,
    run_stage27_comparison,
    write_stage27_task_list,
)


def test_stage27_training_tasks_are_fixed_gru_evidence_grade_and_shadow_only() -> None:
    tasks = build_stage27_training_tasks(output_root=Path("stage27_root"))

    assert RUN_TAG == "mh_stage27_target_head_stability_20260527_01"
    assert STAGE27_CANDIDATES == (
        "horizon_target_normalized_v1",
        "horizon_head_soft_constraint_v1",
        "target_norm_head_constraint_v1",
    )
    assert len(tasks) == 9
    assert {task["candidate"] for task in tasks} == set(STAGE27_CANDIDATES)
    assert {task["seed"] for task in tasks} == {7, 11, 19}
    for task in tasks:
        command = task["command"]
        assert task["horizons"] == "1,2,3,5,8,10,15,20,30"
        assert task["max_horizon"] == 30
        assert command[command.index("--forecast-model-families") + 1] == "gru_sequence_static_context"
        assert command[command.index("--forecast-feature-profile") + 1] == "raw_kline_context_no_alpha_prior_v1"
        assert command[command.index("--forecast-output-profile") + 1] == "decision_utility_v1"
        assert command[command.index("--forecast-loss-profile") + 1] == task["candidate"]
        assert command[command.index("--forecast-epochs") + 1] == "24"
        assert command[command.index("--forecast-min-epochs") + 1] == "8"
        assert command[command.index("--forecast-early-stop-patience") + 1] == "6"
        assert task["scope"] == "research_shadow_only"
        assert task["may_touch_active_manifest"] is False
        assert task["research_program"] == RESEARCH_PROGRAM
        assert task["study_family"] == STUDY_FAMILY


def test_write_stage27_task_list_records_baseline_and_boundary(tmp_path: Path) -> None:
    path = write_stage27_task_list(tmp_path)

    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["run_tag"] == RUN_TAG
    assert payload["research_program"] == RESEARCH_PROGRAM
    assert payload["study_family"] == STUDY_FAMILY
    assert payload["training_task_count"] == 9
    assert payload["baseline_stage26_profiles"] == [
        "score_monthly_robust_v1",
        "horizon_entropy_regularized_v1",
        "risk_drawdown_reweighted_v1",
    ]
    assert payload["baseline_long_horizon_share"] == BASELINE_LONG_HORIZON_SHARE
    assert payload["boundary"] == "research-only / shadow-only; no active manifest, live/default, production root, paper, or broker integration"


def test_stage27_comparison_gate_rejects_high_concentration_and_seed_negative(
    tmp_path: Path,
    monkeypatch,
) -> None:
    report = {"status": "completed", "run_tag": RUN_TAG, "studies": []}
    rows = [
        {
            "loss_profile": "horizon_target_normalized_v1",
            "role": "test",
            "score_name": "pred_decision_score",
            "seed_count": 3,
            "rank_ic_mean": 0.08,
            "rank_ic_min": 0.01,
            "spread_mean": 0.02,
            "spread_min": 0.01,
            "hit_lift_mean": 0.02,
            "hit_lift_min": 0.01,
            "monthly_positive_rate_mean": 0.80,
            "negative_month_count_max": 1,
            "long_horizon_share_mean": BASELINE_LONG_HORIZON_SHARE + 0.01,
            "thirty_d_concentration_mean": 0.70,
        },
        {
            "loss_profile": "horizon_head_soft_constraint_v1",
            "role": "test",
            "score_name": "pred_decision_score",
            "seed_count": 3,
            "rank_ic_mean": 0.08,
            "rank_ic_min": 0.01,
            "spread_mean": 0.02,
            "spread_min": -0.001,
            "hit_lift_mean": 0.02,
            "hit_lift_min": 0.01,
            "monthly_positive_rate_mean": 0.80,
            "negative_month_count_max": 1,
            "long_horizon_share_mean": BASELINE_LONG_HORIZON_SHARE - 0.10,
            "thirty_d_concentration_mean": 0.70,
        },
    ]

    monkeypatch.setattr(
        "daily_research.path_policy.stage27_target_head_stability.build_output_aux_profile_comparison",
        lambda study_dirs, *, run_tag: report,
    )
    monkeypatch.setattr(
        "daily_research.path_policy.stage27_target_head_stability.write_output_aux_profile_comparison",
        lambda report_payload, output_dir: {"profile_aggregate_csv": Path(output_dir) / "profile_aggregate.csv"},
    )
    monkeypatch.setattr(
        "daily_research.path_policy.stage27_target_head_stability.profile_aggregate_rows",
        lambda report_payload: rows,
    )

    payload = run_stage27_comparison(tmp_path)

    assert payload["stage3_architecture_allowed"] is False
    assert payload["stage3_candidates"] == []
    verdict = Path(payload["stage27_research_verdict_md"]).read_text(encoding="utf-8")
    assert "gate: `fail`" in verdict


def test_stage27_comparison_gate_accepts_stable_lower_concentration_candidate(
    tmp_path: Path,
    monkeypatch,
) -> None:
    passing = {
        "loss_profile": "target_norm_head_constraint_v1",
        "role": "test",
        "score_name": "pred_decision_score",
        "seed_count": 3,
        "rank_ic_mean": 0.08,
        "rank_ic_min": 0.01,
        "spread_mean": 0.02,
        "spread_min": 0.01,
        "hit_lift_mean": 0.02,
        "hit_lift_min": 0.01,
        "monthly_positive_rate_mean": 0.80,
        "negative_month_count_max": 1,
        "long_horizon_share_mean": BASELINE_LONG_HORIZON_SHARE - 0.10,
        "thirty_d_concentration_mean": 0.70,
    }
    monkeypatch.setattr(
        "daily_research.path_policy.stage27_target_head_stability.build_output_aux_profile_comparison",
        lambda study_dirs, *, run_tag: {"status": "completed", "run_tag": RUN_TAG, "studies": []},
    )
    monkeypatch.setattr(
        "daily_research.path_policy.stage27_target_head_stability.write_output_aux_profile_comparison",
        lambda report_payload, output_dir: {"profile_aggregate_csv": Path(output_dir) / "profile_aggregate.csv"},
    )
    monkeypatch.setattr(
        "daily_research.path_policy.stage27_target_head_stability.profile_aggregate_rows",
        lambda report_payload: [passing],
    )

    payload = run_stage27_comparison(tmp_path)

    assert payload["stage3_architecture_allowed"] is True
    assert payload["stage3_candidates"] == [passing]
