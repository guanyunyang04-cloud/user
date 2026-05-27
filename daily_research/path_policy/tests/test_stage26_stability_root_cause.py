from __future__ import annotations

import json
from pathlib import Path

from daily_research.path_policy.stage26_stability_root_cause import (
    RESEARCH_PROGRAM,
    RUN_TAG,
    STUDY_FAMILY,
    build_stage26_training_tasks,
    run_training_tasks,
    run_stage26_comparison,
    selected_stage25_audit_tasks,
    write_stage26_task_list,
)


def test_stage26_training_tasks_are_fullgrid_evidence_grade_and_shadow_only() -> None:
    tasks = build_stage26_training_tasks(output_root=Path("stage26_root"))

    assert RUN_TAG == "mh_stage26_stability_root_cause_20260527_01"
    assert len(tasks) == 9
    assert {task["candidate"] for task in tasks} == {
        "score_monthly_robust_v1",
        "horizon_entropy_regularized_v1",
        "risk_drawdown_reweighted_v1",
    }
    assert {task["seed"] for task in tasks} == {7, 11, 19}
    for task in tasks:
        command = task["command"]
        assert task["horizons"] == "1,2,3,5,8,10,15,20,30"
        assert task["max_horizon"] == 30
        assert "--forecast-model-families" in command
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


def test_selected_stage25_audit_tasks_only_include_fullgrid_and_daily_candidates() -> None:
    tasks = selected_stage25_audit_tasks(output_root=Path("stage26_root"))

    assert len(tasks) == 6
    assert {task["candidate"] for task in tasks} == {"fullgrid_rebudget", "daily1_45_multiseed"}
    assert {task["seed"] for task in tasks} == {7, 11, 19}
    assert all("long_family_simplified" not in task["source_tag"] for task in tasks)
    assert all(task["audit_output_dir"].endswith(task["source_tag"]) for task in tasks)


def test_write_stage26_task_list_outputs_audit_and_training_tasks(tmp_path: Path) -> None:
    path = write_stage26_task_list(tmp_path)

    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["run_tag"] == RUN_TAG
    assert payload["research_program"] == RESEARCH_PROGRAM
    assert payload["study_family"] == STUDY_FAMILY
    assert payload["audit_task_count"] == 6
    assert payload["training_task_count"] == 9
    assert payload["training_tasks"][0]["tag"].startswith("mh26_")
    assert payload["boundary"] == "research-only / shadow-only; no active manifest, live/default, production root, paper, or broker integration"


def test_run_stage26_comparison_writes_stage26_verdict_and_summary(tmp_path: Path, monkeypatch) -> None:
    report = {
        "status": "completed",
        "run_tag": RUN_TAG,
        "studies": [],
        "shadow_only": True,
        "promotion_allowed": False,
    }
    aggregate_row = {
        "loss_profile": "score_monthly_robust_v1",
        "role": "test",
        "score_name": "pred_decision_score",
        "seed_count": 3,
        "rank_ic_mean": 0.02,
        "spread_mean": 0.01,
        "hit_lift_mean": 0.01,
        "monthly_positive_rate_mean": 0.80,
        "negative_month_count_max": 1,
        "long_horizon_share_mean": 0.50,
        "stage3_weak_gate_pass": True,
    }

    def fake_comparison(study_dirs, *, run_tag):
        assert run_tag == RUN_TAG
        assert len(study_dirs) == 9
        return report

    def fake_write(report_payload, output_dir):
        path = Path(output_dir) / "profile_aggregate.csv"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("loss_profile\nscore_monthly_robust_v1\n", encoding="utf-8")
        return {"profile_aggregate_csv": path}

    monkeypatch.setattr(
        "daily_research.path_policy.stage26_stability_root_cause.build_output_aux_profile_comparison",
        fake_comparison,
    )
    monkeypatch.setattr(
        "daily_research.path_policy.stage26_stability_root_cause.write_output_aux_profile_comparison",
        fake_write,
    )
    monkeypatch.setattr(
        "daily_research.path_policy.stage26_stability_root_cause.profile_aggregate_rows",
        lambda _: [aggregate_row],
    )

    payload = run_stage26_comparison(tmp_path)

    assert payload["stage3_architecture_allowed"] is True
    assert payload["research_program"] == RESEARCH_PROGRAM
    assert payload["study_family"] == STUDY_FAMILY
    assert payload["stage3_candidates"] == [aggregate_row]
    assert Path(payload["stage26_research_verdict_md"]).read_text(encoding="utf-8").startswith(
        "# Alpha Multi-Horizon Stage 2.6"
    )
    saved = json.loads((tmp_path / "stage26_comparison_summary.json").read_text(encoding="utf-8"))
    assert saved["comparison_paths"]["profile_aggregate_csv"].endswith("profile_aggregate.csv")


def test_run_training_tasks_marks_progress_completed_when_all_existing_summaries_are_skipped(
    tmp_path: Path,
    monkeypatch,
) -> None:
    tasks = []
    for seed in (7, 11, 19):
        study_dir = tmp_path / f"study_seed{seed}"
        study_dir.mkdir()
        (study_dir / "study_summary.json").write_text("{}", encoding="utf-8")
        tasks.append({"tag": f"unit_seed{seed}", "study_dir": str(study_dir)})
    monkeypatch.setattr(
        "daily_research.path_policy.stage26_stability_root_cause.build_stage26_training_tasks",
        lambda output_root=None: tasks,
    )

    payload = run_training_tasks(output_root=tmp_path, skip_existing=True)
    progress = json.loads((tmp_path / "stage26_progress.json").read_text(encoding="utf-8"))

    assert payload["status"] == "completed"
    assert progress["status"] == "completed"
    assert progress["completed_tags"] == ["unit_seed7", "unit_seed11", "unit_seed19"]
    assert progress["research_program"] == RESEARCH_PROGRAM
    assert progress["study_family"] == STUDY_FAMILY
