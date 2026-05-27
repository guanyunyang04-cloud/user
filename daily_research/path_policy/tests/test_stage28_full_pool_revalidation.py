from __future__ import annotations

import json
from pathlib import Path

from daily_research.path_policy.stage28_full_pool_revalidation import (
    BASELINE_LONG_HORIZON_SHARE,
    RESEARCH_PROGRAM,
    RUN_TAG,
    STAGE28_CANDIDATES,
    STUDY_FAMILY,
    build_stage28_training_tasks,
    run_stage28_comparison,
    write_stage28_task_list,
)
from daily_research.path_policy.stage_universe_scope import FULL_ROLLING_LIQUID500_SCOPE


def test_stage28_training_tasks_are_full_pool_core_decision_chain() -> None:
    tasks = build_stage28_training_tasks(output_root=Path("stage28_root"))

    assert RUN_TAG == "mh_stage28_full_pool_revalidation_20260527_01"
    assert STAGE28_CANDIDATES == (
        "decision_utility_v1",
        "horizon_target_normalized_v1",
        "target_norm_head_constraint_v1",
    )
    assert len(tasks) == 9
    assert {task["candidate"] for task in tasks} == set(STAGE28_CANDIDATES)
    assert {task["seed"] for task in tasks} == {7, 11, 19}
    for task in tasks:
        command = task["command"]
        assert task["tag"].startswith("mh28_fullpool_")
        assert not task["tag"].startswith(("mh25_", "mh26_", "mh27_"))
        assert task["horizons"] == "1,2,3,5,8,10,15,20,30"
        assert task["max_horizon"] == 30
        assert command[command.index("--forecast-model-families") + 1] == "gru_sequence_static_context"
        assert command[command.index("--forecast-feature-profile") + 1] == "raw_kline_context_no_alpha_prior_v1"
        assert command[command.index("--forecast-output-profile") + 1] == "decision_utility_v1"
        assert command[command.index("--forecast-loss-profile") + 1] == task["candidate"]
        assert command[command.index("--pool-view-kind") + 1] == "rolling_liquidity"
        assert command[command.index("--pool-view-name") + 1] == "rolling_liquid500"
        assert command[command.index("--max-universe-size") + 1] == "0"
        assert command[command.index("--forecast-memmap-manifest") + 1].endswith("forecast_dataset_manifest.json")
        assert command[command.index("--forecast-epochs") + 1] == "24"
        assert command[command.index("--forecast-min-epochs") + 1] == "8"
        assert command[command.index("--forecast-early-stop-patience") + 1] == "6"
        assert task["scope"] == "research_shadow_only"
        assert task["universe_scope"] == FULL_ROLLING_LIQUID500_SCOPE
        assert task["expected_min_universe_size"] >= 500
        assert task["expected_min_train_rows"] >= 400_000
        assert task["may_touch_active_manifest"] is False
        assert task["research_program"] == RESEARCH_PROGRAM
        assert task["study_family"] == STUDY_FAMILY


def test_write_stage28_task_list_records_full_pool_boundary(tmp_path: Path) -> None:
    path = write_stage28_task_list(tmp_path)

    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["run_tag"] == RUN_TAG
    assert payload["research_program"] == RESEARCH_PROGRAM
    assert payload["study_family"] == STUDY_FAMILY
    assert payload["training_task_count"] == 9
    assert payload["universe_scope"] == FULL_ROLLING_LIQUID500_SCOPE
    assert payload["reused_forecast_memmap_manifest"].endswith("forecast_dataset_manifest.json")
    assert payload["baseline_long_horizon_share"] == BASELINE_LONG_HORIZON_SHARE
    assert payload["stage3_architecture_allowed_by_default"] is False
    assert payload["boundary"] == "research-only / shadow-only; no active manifest, live/default, production root, paper, or broker integration"


def test_stage28_comparison_blocks_cap80_scope_before_profile_gate(tmp_path: Path, monkeypatch) -> None:
    tasks = [
        {
            "tag": "mh28_fullpool_decision_utility_v1_fullgrid_seed7_20260527_01",
            "study_dir": str(tmp_path / "cap80"),
            "universe_scope": FULL_ROLLING_LIQUID500_SCOPE,
        }
    ]
    monkeypatch.setattr(
        "daily_research.path_policy.stage28_full_pool_revalidation.build_stage28_training_tasks",
        lambda output_root=None: tasks,
    )
    monkeypatch.setattr(
        "daily_research.path_policy.stage28_full_pool_revalidation.study_scope_from_summary_path",
        lambda _: {"universe_scope": "cap80_diagnostic", "universe_size": 80, "train_rows": 74640},
    )

    payload = run_stage28_comparison(tmp_path)

    assert payload["status"] == "blocked_scope_mismatch"
    assert payload["stage3_architecture_allowed"] is False
    assert payload["scope_mismatches"][0]["observed_scope"]["universe_scope"] == "cap80_diagnostic"


def test_stage28_comparison_gate_rejects_monthly_instability_and_high_concentration(
    tmp_path: Path,
    monkeypatch,
) -> None:
    rows = [
        {
            "loss_profile": "target_norm_head_constraint_v1",
            "role": "test",
            "score_name": "pred_decision_score",
            "seed_count": 3,
            "rank_ic_min": 0.01,
            "spread_min": 0.01,
            "hit_lift_min": 0.01,
            "monthly_positive_rate_mean": 0.80,
            "negative_month_count_max": 4,
            "long_horizon_share_mean": 0.70,
            "thirty_d_concentration_mean": 0.50,
        },
        {
            "loss_profile": "horizon_target_normalized_v1",
            "role": "test",
            "score_name": "pred_decision_score",
            "seed_count": 3,
            "rank_ic_min": 0.01,
            "spread_min": 0.01,
            "hit_lift_min": 0.01,
            "monthly_positive_rate_mean": 0.80,
            "negative_month_count_max": 1,
            "long_horizon_share_mean": 0.70,
            "thirty_d_concentration_mean": 0.95,
        },
    ]
    monkeypatch.setattr(
        "daily_research.path_policy.stage28_full_pool_revalidation._stage28_scope_mismatches",
        lambda tasks: [],
    )
    monkeypatch.setattr(
        "daily_research.path_policy.stage28_full_pool_revalidation.build_output_aux_profile_comparison",
        lambda study_dirs, *, run_tag: {"status": "completed", "run_tag": RUN_TAG, "studies": []},
    )
    monkeypatch.setattr(
        "daily_research.path_policy.stage28_full_pool_revalidation.write_output_aux_profile_comparison",
        lambda report_payload, output_dir: {"profile_aggregate_csv": Path(output_dir) / "profile_aggregate.csv"},
    )
    monkeypatch.setattr(
        "daily_research.path_policy.stage28_full_pool_revalidation.profile_aggregate_rows",
        lambda report_payload: rows,
    )

    payload = run_stage28_comparison(tmp_path)

    assert payload["status"] == "completed"
    assert payload["stage3_architecture_allowed"] is False
    assert payload["stage3_candidates"] == []


def test_stage28_comparison_gate_accepts_stable_target_norm_candidate(
    tmp_path: Path,
    monkeypatch,
) -> None:
    passing = {
        "loss_profile": "target_norm_head_constraint_v1",
        "role": "test",
        "score_name": "pred_decision_score",
        "seed_count": 3,
        "rank_ic_min": 0.01,
        "spread_min": 0.01,
        "hit_lift_min": 0.01,
        "monthly_positive_rate_mean": 0.80,
        "negative_month_count_max": 1,
        "long_horizon_share_mean": 0.70,
        "thirty_d_concentration_mean": 0.50,
    }
    monkeypatch.setattr(
        "daily_research.path_policy.stage28_full_pool_revalidation._stage28_scope_mismatches",
        lambda tasks: [],
    )
    monkeypatch.setattr(
        "daily_research.path_policy.stage28_full_pool_revalidation.build_output_aux_profile_comparison",
        lambda study_dirs, *, run_tag: {"status": "completed", "run_tag": RUN_TAG, "studies": []},
    )
    monkeypatch.setattr(
        "daily_research.path_policy.stage28_full_pool_revalidation.write_output_aux_profile_comparison",
        lambda report_payload, output_dir: {"profile_aggregate_csv": Path(output_dir) / "profile_aggregate.csv"},
    )
    monkeypatch.setattr(
        "daily_research.path_policy.stage28_full_pool_revalidation.profile_aggregate_rows",
        lambda report_payload: [passing],
    )

    payload = run_stage28_comparison(tmp_path)

    assert payload["stage3_architecture_allowed"] is True
    assert payload["stage3_candidates"] == [passing]
