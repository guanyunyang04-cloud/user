from __future__ import annotations

import json
from pathlib import Path

import pytest

from daily_research.path_policy import v2_high_return_model_discovery as hrd


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_task_list_defaults_to_augmented_single_seed_multi_split_research_only(tmp_path: Path, monkeypatch) -> None:
    studies = tmp_path / "studies"
    monkeypatch.setattr(hrd, "STUDIES_ROOT", studies)

    path = hrd.write_task_list(
        output_root=tmp_path / "out",
        allow_single_seed_scout=True,
        enforce_active_artifact_clean=False,
    )
    payload = json.loads(path.read_text(encoding="utf-8"))
    tasks = payload["training_tasks"]

    assert payload["stage"] == "high_return_model_discovery"
    assert payload["single_seed_scout_only"] is True
    assert payload["evidence_grade_input_or_model_quality_pass"] is False
    assert payload["boundary"]["promotion_allowed"] is False
    assert payload["boundary"]["active_execution_strategy_expected_diff"] == "none"
    assert payload["high_return_scout_policy"]["research_gate_id"] == hrd.SMALL_CAPITAL_GATE_ID
    assert payload["high_return_scout_policy"]["quick_bridge_default"]["rebalance_freq"] == "10d"
    assert payload["high_return_scout_policy"]["quick_bridge_default"]["costs_bps"] == {"transaction": 10, "slippage": 5, "sell_tax": 10}
    assert payload["high_return_scout_policy"]["candidate_matrix_grid"]["holding_counts"] == [10, 20, 30]
    assert payload["high_return_scout_policy"]["candidate_matrix_grid"]["max_weights"] == [0.08, 0.12, 0.16]
    assert payload["high_return_scout_policy"]["candidate_matrix_grid"]["rebalance_freqs"] == ["5d", "10d", "20d"]
    assert payload["training_task_count"] == 15
    assert payload["loss_profile"] == hrd.LOSS_PROFILE
    assert payload["loss_profiles"] == [hrd.LOSS_PROFILE]
    assert {task["seed"] for task in tasks} == {7}
    assert {task["dataset_id"] for task in tasks} == {hrd.TRADITIONAL_BAOSTOCK_V2_1_DATASET_ID}
    assert {task["pool_view_id"] for task in tasks} == {hrd.SAME_PERIOD_POOL_VIEW_ID, hrd.LONG_HISTORY_POOL_VIEW_ID}
    assert {task["feature_profile"] for task in tasks} == {hrd.FEATURE_PROFILE}
    assert {task["loss_profile"] for task in tasks} == {hrd.LOSS_PROFILE}
    assert {task["split_key"] for task in tasks} == {"same", "a", "b", "c", "long"}
    assert {task["model_family"] for task in tasks} == set(hrd.DEFAULT_MODEL_FAMILIES)

    same_gru = next(task for task in tasks if task["dataset_key"] == "same_period_augmented" and task["model_family"] == "gru_sequence_static_context")
    command = same_gru["command"]
    assert command[command.index("--lake-dataset-id") + 1] == hrd.TRADITIONAL_BAOSTOCK_V2_1_DATASET_ID
    assert command[command.index("--pool-view-id") + 1] == hrd.SAME_PERIOD_POOL_VIEW_ID
    assert command[command.index("--forecast-feature-profile") + 1] == hrd.FEATURE_PROFILE
    assert command[command.index("--forecast-loss-profile") + 1] == hrd.LOSS_PROFILE
    assert command[command.index("--forecast-output-profile") + 1] == hrd.OUTPUT_PROFILE
    assert command[command.index("--forecast-train-start-year") + 1] == "2019"
    assert command[command.index("--forecast-test-year") + 1] == "2024"
    assert same_gru["builds_forecast_memmap"] is True

    same_hybrid = next(
        task for task in tasks if task["dataset_key"] == "same_period_augmented" and task["model_family"] == "hybrid_expert_fusion_static_context"
    )
    assert same_hybrid["reuses_forecast_memmap_manifest"] is True
    assert same_hybrid["depends_on_manifest_task_tag"] == same_gru["tag"]
    assert "--forecast-memmap-manifest" in same_hybrid["command"]

    regime = next(task for task in tasks if task["model_family"] == "regime_routed_multi_expert_horizon_v1")
    assert regime["command"][regime["command"].index("--forecast-hidden-dim") + 1] == "256"
    assert regime["command"][regime["command"].index("--forecast-transformer-heads") + 1] == "8"
    assert regime["command"][regime["command"].index("--forecast-patch-sizes") + 1] == "4,10,20"


def test_single_seed_requires_explicit_scout_acknowledgement() -> None:
    with pytest.raises(ValueError, match="single_seed_scout_requires_explicit"):
        hrd.build_forecast_tasks(seeds=(7,), allow_single_seed_scout=False)


def test_forecast_task_tag_suffix_follows_outer_run_tag(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(hrd, "STUDIES_ROOT", tmp_path / "studies")
    task = hrd.build_forecast_tasks(
        output_root=tmp_path / "out",
        run_tag="v2_hrd_topn_long_mainboard_wide_gru_seed7_scout_20260605_01",
        datasets=("long_history_augmented",),
        splits=("a",),
        model_families=("gru_sequence_static_context",),
        loss_profiles=("topn_excess_rank_v1",),
        seeds=(7,),
        allow_single_seed_scout=True,
    )[0]

    assert task["tag"] == "mh_v2_hrd_topn_long_a_gru_seed7_20260605_01"
    assert task["command"][task["command"].index("--tag") + 1] == task["tag"]


def test_smoke32_tasks_reuse_prebuilt_memmap_and_stay_smoke_only(tmp_path: Path, monkeypatch) -> None:
    studies = tmp_path / "studies"
    monkeypatch.setattr(hrd, "STUDIES_ROOT", studies)
    smoke_manifest = tmp_path / "smoke" / "forecast_dataset_manifest.json"
    monkeypatch.setattr(hrd, "SMOKE32_MEMMAP_MANIFEST", smoke_manifest)

    path = hrd.write_task_list(
        output_root=tmp_path / "out",
        datasets=("smoke32_augmented",),
        model_families=("gru_sequence_static_context",),
        seeds=(7,),
        allow_single_seed_scout=True,
        max_samples_per_role=8,
        max_samples_per_date_per_role=2,
        enforce_active_artifact_clean=False,
    )
    payload = json.loads(path.read_text(encoding="utf-8"))
    task = payload["training_tasks"][0]
    command = task["command"]

    assert payload["training_task_count"] == 1
    assert payload["smoke_only_task_count"] == 1
    assert payload["max_samples_per_role"] == 8
    assert task["tag"] == "mh_v2_hrd_smoke32_same_gru_seed7_20260604_01"
    assert task["pool_view_id"] == hrd.SMOKE32_POOL_VIEW_ID
    assert task["evidence_grade"] == "smoke_only"
    assert task["smoke_only"] is True
    assert task["max_samples_per_date_per_role"] == 2
    assert task["builds_forecast_memmap"] is False
    assert task["reuses_forecast_memmap_manifest"] is True
    assert task["depends_on_manifest_task_tag"] == "prebuilt_smoke32_augmented_memmap"
    assert command[command.index("--forecast-memmap-manifest") + 1] == str(smoke_manifest)
    assert command[command.index("--forecast-max-samples-per-role") + 1] == "8"
    assert command[command.index("--forecast-max-samples-per-date-per-role") + 1] == "2"


def test_full_pool_pilot_tasks_are_excluded_from_formal_shortlist(tmp_path: Path, monkeypatch) -> None:
    studies = tmp_path / "studies"
    monkeypatch.setattr(hrd, "STUDIES_ROOT", studies)
    path = hrd.write_task_list(
        output_root=tmp_path / "out",
        datasets=("same_period_augmented_pilot",),
        model_families=("gru_sequence_static_context",),
        seeds=(7,),
        allow_single_seed_scout=True,
        enforce_active_artifact_clean=False,
    )
    payload = json.loads(path.read_text(encoding="utf-8"))
    task = payload["training_tasks"][0]
    command = task["command"]

    assert payload["training_task_count"] == 1
    assert payload["pilot_only_task_count"] == 1
    assert task["tag"] == "mh_v2_hrd_same_pilot_pilot_gru_seed7_20260604_01"
    assert task["dataset_key"] == "same_period_augmented_pilot"
    assert task["evidence_grade"] == "pilot_only"
    assert task["pilot_only"] is True
    assert task["smoke_only"] is False
    assert task["builds_forecast_memmap"] is True
    assert command[command.index("--start-date") + 1] == "20220101"
    assert command[command.index("--forecast-train-start-year") + 1] == "2022"
    assert command[command.index("--forecast-validation-year") + 1] == "2023"
    assert command[command.index("--forecast-test-year") + 1] == "2024"


def test_bridge_and_matrix_commands_are_aggressive_research_scouts(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(hrd, "STUDIES_ROOT", tmp_path / "studies")
    task = hrd.build_forecast_tasks(
        output_root=tmp_path / "out",
        datasets=("same_period_augmented",),
        model_families=("gru_sequence_static_context",),
        seeds=(7,),
        allow_single_seed_scout=True,
    )[0]

    bridge = task["quick_bridge_command"]
    assert bridge[bridge.index("-m") + 1] == "daily_research.path_policy.v2_score_backtest_bridge"
    assert "--allow-partial-seeds" in bridge
    assert "--run-backtest" in bridge
    assert bridge[bridge.index("--source-anchor-run-tag") + 1] == task["tag"]
    assert bridge[bridge.index("--seed-study-tags") + 1] == task["tag"]
    assert bridge[bridge.index("--holding-count") + 1] == "20"
    assert bridge[bridge.index("--max-weight") + 1] == "0.12"
    assert bridge[bridge.index("--rebalance-freq") + 1] == "10d"
    assert bridge[bridge.index("--transaction-cost-bps") + 1] == "10"
    assert bridge[bridge.index("--slippage-bps") + 1] == "5"

    matrix = task["candidate_matrix_command"]
    assert matrix[matrix.index("-m") + 1] == "daily_research.path_policy.v2_candidate_review_matrix"
    assert "--run-backtests" in matrix
    assert matrix[matrix.index("--holding-counts") + 1] == "10,20,30"
    assert matrix[matrix.index("--max-weights") + 1] == "0.08,0.12,0.16"
    assert matrix[matrix.index("--rebalance-freqs") + 1] == "5d,10d,20d"
    assert matrix[matrix.index("--transaction-cost-bps-values") + 1] == "10"
    assert matrix[matrix.index("--slippage-bps-values") + 1] == "5"


def test_loss_profiles_generate_distinct_tasks_and_report_groups(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(hrd, "STUDIES_ROOT", tmp_path / "studies")
    out = tmp_path / "out"
    loss_profiles = ("horizon_30d_soft_penalty_v1", "topn_excess_rank_v1")
    path = hrd.write_task_list(
        output_root=out,
        datasets=("same_period_augmented",),
        splits=("same",),
        model_families=("gru_sequence_static_context",),
        loss_profiles=loss_profiles,
        seeds=(7,),
        allow_single_seed_scout=True,
        max_samples_per_date_per_role=2,
        enforce_active_artifact_clean=False,
    )
    payload = json.loads(path.read_text(encoding="utf-8"))
    tasks = payload["training_tasks"]

    assert payload["training_task_count"] == 2
    assert payload["loss_profile"] == ""
    assert payload["loss_profiles"] == list(loss_profiles)
    assert {task["loss_profile"] for task in tasks} == set(loss_profiles)
    assert len({task["tag"] for task in tasks}) == 2
    assert len({task["candidate_id"] for task in tasks}) == 2
    assert all(task["output_profile"] == hrd.OUTPUT_PROFILE for task in tasks)
    assert all(task["selection_profile"] == hrd.SELECTION_PROFILE for task in tasks)
    assert all(task["active_execution_strategy_expected_diff"] == "none" for task in tasks)

    first, second = tasks
    assert first["builds_forecast_memmap"] is True
    assert second["reuses_forecast_memmap_manifest"] is True
    assert second["depends_on_manifest_task_tag"] == first["tag"]
    for task in tasks:
        command = task["command"]
        assert command[command.index("--forecast-loss-profile") + 1] == task["loss_profile"]
        assert command[command.index("--forecast-max-samples-per-date-per-role") + 1] == "2"
        assert hrd._loss_profile_alias(task["loss_profile"]) in task["tag"]
        assert hrd._loss_profile_alias(task["loss_profile"]) in task["candidate_id"]

    for task, rank, excess in ((first, 0.02, 0.10), (second, 0.08, 0.25)):
        _write_json(
            Path(task["study_dir"]) / "study_summary.json",
            {
                "status": "completed",
                "evidence_verdict": "forecast_test_confirmed",
                "training_summary": {
                    "test_metrics": {
                        "decision_score_rank_ic": rank,
                        "decision_score_top_bottom_spread": rank / 2.0,
                        "decision_hit_lift_top20_mean": rank,
                    }
                },
            },
        )
        _write_json(
            out / "bridges" / hrd._bridge_tag(str(task["tag"])) / "v2_score_backtest_bridge_report.json",
            {
                "status": "completed",
                "shared_backtest": {
                    "status": "completed",
                    "artifacts": {
                        "core_metrics": {
                            "annual_return": excess + 0.20,
                            "excess_annual_return": excess,
                            "excess_sharpe": 1.10 if task is first else 1.40,
                            "max_drawdown": -0.20,
                        },
                        "monthly_backtest_diagnostics": {
                            "positive_month_ratio": 0.62,
                            "negative_month_count": 2,
                            "worst_monthly_excess_return": -0.06,
                        },
                    },
                },
            },
        )

    report = hrd.collect_high_return_report(output_root=out, task_list_path=path)

    assert report["loss_profiles"] == list(loss_profiles)
    assert {row["loss_profile"] for row in report["leaderboard"]} == set(loss_profiles)
    assert report["leaderboard"][0]["loss_profile"] == "topn_excess_rank_v1"
    assert len(report["split_consistency"]) == 2
    assert {row["loss_profile"] for row in report["split_consistency"]} == set(loss_profiles)
    assert all(row["completed_split_count"] == 1 for row in report["split_consistency"])
    assert {row["loss_profile"] for row in report["shortlist"]} == set(loss_profiles)


def test_collect_report_ranks_completed_high_return_candidate(tmp_path: Path, monkeypatch) -> None:
    studies = tmp_path / "studies"
    monkeypatch.setattr(hrd, "STUDIES_ROOT", studies)
    out = tmp_path / "out"
    path = hrd.write_task_list(
        output_root=out,
        datasets=("same_period_augmented",),
        splits=("same",),
        model_families=("gru_sequence_static_context", "hybrid_expert_fusion_static_context"),
        seeds=(7,),
        allow_single_seed_scout=True,
        enforce_active_artifact_clean=False,
    )
    tasks = json.loads(path.read_text(encoding="utf-8"))["training_tasks"]
    strong, weak = tasks

    for task, rank, spread, hit in ((strong, 0.08, 0.035, 0.04), (weak, 0.01, 0.005, 0.0)):
        _write_json(
            Path(task["study_dir"]) / "study_summary.json",
            {
                "status": "completed",
                "evidence_verdict": "forecast_test_confirmed",
                "training_summary": {
                    "test_metrics": {
                        "decision_score_rank_ic": rank,
                        "decision_score_top_bottom_spread": spread,
                        "decision_hit_lift_top20_mean": hit,
                    }
                },
            },
        )

    _write_json(
        out / "bridges" / hrd._bridge_tag(str(strong["tag"])) / "v2_score_backtest_bridge_report.json",
        {
            "status": "completed",
            "shared_backtest": {
                "status": "completed",
                "artifacts": {
                    "core_metrics": {
                        "annual_return": 0.75,
                        "excess_annual_return": 0.45,
                        "excess_sharpe": 1.35,
                        "max_drawdown": -0.28,
                    },
                    "monthly_backtest_diagnostics": {
                        "positive_month_ratio": 0.70,
                        "negative_month_count": 3,
                        "worst_monthly_return": -0.08,
                    },
                },
            },
        },
    )

    report = hrd.collect_high_return_report(output_root=out, task_list_path=path)

    assert report["status"] == "completed"
    assert report["leaderboard"][0]["tag"] == strong["tag"]
    assert report["leaderboard"][0]["high_return_score"] > report["leaderboard"][1]["high_return_score"]
    assert report["leaderboard"][0]["positive_transfer"] is True
    assert report["leaderboard"][0]["matrix_entry_candidate"] is True
    assert report["shortlist"][0]["tag"] == strong["tag"]
    assert report["small_capital_gate_summary"]["positive_transfer_count"] == 1
    assert report["small_capital_gate_summary"]["matrix_entry_candidate_count"] == 1
    assert report["split_consistency"][0]["model_family"] == strong["model_family"]
    assert report["split_consistency"][0]["completed_split_count"] == 1
    assert report["split_consistency"][0]["positive_transfer_split_count"] == 1
    assert report["split_consistency"][0]["matrix_entry_split_count"] == 1
    assert report["bad_month_preview"][0]["tag"] == strong["tag"]
    assert report["bad_month_preview"][0]["issue_hint"] == "ordinary_bad_month_preview"
    assert report["scoring_policy"]["three_seed_required_before_model_quality_evidence"] is True


def test_collect_report_excludes_smoke_only_rows_from_formal_shortlist(tmp_path: Path, monkeypatch) -> None:
    studies = tmp_path / "studies"
    monkeypatch.setattr(hrd, "STUDIES_ROOT", studies)
    out = tmp_path / "out"
    path = hrd.write_task_list(
        output_root=out,
        datasets=("smoke32_augmented",),
        model_families=("gru_sequence_static_context",),
        seeds=(7,),
        allow_single_seed_scout=True,
        enforce_active_artifact_clean=False,
    )
    task = json.loads(path.read_text(encoding="utf-8"))["training_tasks"][0]
    _write_json(
        Path(task["study_dir"]) / "study_summary.json",
        {
            "status": "completed",
            "evidence_verdict": "forecast_test_confirmed",
            "training_summary": {
                "test_metrics": {
                    "decision_score_rank_ic": 0.20,
                    "decision_score_top_bottom_spread": 0.10,
                    "decision_hit_lift_top20_mean": 0.08,
                }
            },
        },
    )
    _write_json(
        out / "bridges" / hrd._bridge_tag(str(task["tag"])) / "v2_score_backtest_bridge_report.json",
        {
            "status": "completed",
            "shared_backtest": {
                "status": "completed",
                "artifacts": {
                    "core_metrics": {
                        "annual_return": 1.20,
                        "excess_annual_return": 0.90,
                        "excess_sharpe": 2.0,
                        "max_drawdown": -0.20,
                    },
                    "monthly_backtest_diagnostics": {
                        "positive_month_ratio": 0.90,
                        "negative_month_count": 1,
                        "worst_monthly_return": -0.03,
                    },
                },
            },
        },
    )

    report = hrd.collect_high_return_report(output_root=out, task_list_path=path)

    assert report["status"] == "completed"
    assert report["completed_forecast_count"] == 1
    assert report["smoke_completed_forecast_count"] == 1
    assert report["leaderboard"][0]["smoke_only"] is True
    assert report["leaderboard"][0]["high_return_score"] > 0.0
    assert report["shortlist"] == []
    assert report["shortlist_count"] == 0
    assert Path(report["outputs"]["report_md"]).exists()


def test_collect_report_excludes_pilot_only_rows_from_formal_shortlist(tmp_path: Path, monkeypatch) -> None:
    studies = tmp_path / "studies"
    monkeypatch.setattr(hrd, "STUDIES_ROOT", studies)
    out = tmp_path / "out"
    path = hrd.write_task_list(
        output_root=out,
        datasets=("same_period_augmented_pilot",),
        model_families=("gru_sequence_static_context",),
        seeds=(7,),
        allow_single_seed_scout=True,
        enforce_active_artifact_clean=False,
    )
    task = json.loads(path.read_text(encoding="utf-8"))["training_tasks"][0]
    _write_json(
        Path(task["study_dir"]) / "study_summary.json",
        {
            "status": "completed",
            "evidence_verdict": "forecast_test_confirmed",
            "training_summary": {
                "test_metrics": {
                    "decision_score_rank_ic": 0.20,
                    "decision_score_top_bottom_spread": 0.10,
                    "decision_hit_lift_top20_mean": 0.08,
                }
            },
        },
    )
    _write_json(
        out / "bridges" / hrd._bridge_tag(str(task["tag"])) / "v2_score_backtest_bridge_report.json",
        {
            "status": "completed",
            "shared_backtest": {
                "status": "completed",
                "artifacts": {
                    "core_metrics": {
                        "annual_return": 1.20,
                        "excess_annual_return": 0.90,
                        "excess_sharpe": 2.0,
                        "max_drawdown": -0.20,
                    },
                    "monthly_backtest_diagnostics": {
                        "positive_month_ratio": 0.90,
                        "negative_month_count": 1,
                        "worst_monthly_return": -0.03,
                    },
                },
            },
        },
    )

    report = hrd.collect_high_return_report(output_root=out, task_list_path=path)

    assert report["status"] == "completed"
    assert report["completed_forecast_count"] == 1
    assert report["pilot_completed_forecast_count"] == 1
    assert report["leaderboard"][0]["pilot_only"] is True
    assert report["leaderboard"][0]["high_return_score"] > 0.0
    assert report["shortlist"] == []
    assert report["shortlist_count"] == 0


def test_run_quick_bridge_tasks_executes_only_completed_forecasts(tmp_path: Path, monkeypatch) -> None:
    studies = tmp_path / "studies"
    monkeypatch.setattr(hrd, "STUDIES_ROOT", studies)
    monkeypatch.setattr(hrd, "_active_artifact_has_diff", lambda: False)
    out = tmp_path / "out"
    path = hrd.write_task_list(
        output_root=out,
        datasets=("same_period_augmented",),
        splits=("same",),
        model_families=("gru_sequence_static_context", "hybrid_expert_fusion_static_context"),
        seeds=(7,),
        allow_single_seed_scout=True,
        enforce_active_artifact_clean=False,
    )
    tasks = json.loads(path.read_text(encoding="utf-8"))["training_tasks"]
    completed, pending = tasks
    _write_json(Path(completed["study_dir"]) / "study_summary.json", {"status": "completed"})

    calls: list[list[str]] = []

    def fake_run(command, *, stdout_path, stderr_path):
        calls.append(list(command))
        return {"returncode": 0, "stdout": str(stdout_path), "stderr": str(stderr_path)}

    monkeypatch.setattr(hrd, "_run_command", fake_run)

    summary = hrd.run_quick_bridge_tasks(output_root=out, task_list_path=path)

    assert summary["status"] == "completed"
    assert summary["completed_tags"] == [completed["tag"]]
    assert summary["skipped"] == [{"tag": pending["tag"], "reason": "awaiting_completed_forecast"}]
    assert len(calls) == 1
    assert calls[0][calls[0].index("-m") + 1] == "daily_research.path_policy.v2_score_backtest_bridge"
    assert (out / "v2_high_return_model_discovery_quick_bridge_run_summary.json").exists()


def test_run_candidate_matrix_tasks_executes_all_matrix_entry_candidates(tmp_path: Path, monkeypatch) -> None:
    studies = tmp_path / "studies"
    monkeypatch.setattr(hrd, "STUDIES_ROOT", studies)
    monkeypatch.setattr(hrd, "_active_artifact_has_diff", lambda: False)
    out = tmp_path / "out"
    loss_profiles = (
        "horizon_30d_soft_penalty_v1",
        "score_monthly_robust_v1",
        "risk_drawdown_reweighted_v1",
        "horizon_entropy_regularized_v1",
    )
    path = hrd.write_task_list(
        output_root=out,
        datasets=("same_period_augmented",),
        splits=("same",),
        model_families=("gru_sequence_static_context",),
        loss_profiles=loss_profiles,
        seeds=(7,),
        allow_single_seed_scout=True,
        enforce_active_artifact_clean=False,
    )
    tasks = json.loads(path.read_text(encoding="utf-8"))["training_tasks"]
    for idx, task in enumerate(tasks):
        _write_json(
            Path(task["study_dir"]) / "study_summary.json",
            {
                "status": "completed",
                "evidence_verdict": "forecast_test_confirmed",
                "training_summary": {
                    "test_metrics": {
                        "decision_score_rank_ic": 0.02 + idx * 0.01,
                        "decision_score_top_bottom_spread": 0.01,
                        "decision_hit_lift_top20_mean": 0.02,
                    }
                },
            },
        )
        _write_json(
            out / "bridges" / hrd._bridge_tag(str(task["tag"])) / "v2_score_backtest_bridge_report.json",
            {
                "status": "completed",
                "shared_backtest": {
                    "status": "completed",
                    "artifacts": {
                        "core_metrics": {
                            "annual_return": 0.30 + idx * 0.01,
                            "excess_annual_return": 0.18 + idx * 0.01,
                            "excess_sharpe": 1.10 + idx * 0.05,
                            "max_drawdown": -0.12,
                        },
                        "monthly_backtest_diagnostics": {
                            "positive_month_ratio": 0.70,
                            "negative_month_count": 2,
                            "worst_monthly_return": -0.05,
                        },
                    },
                },
            },
        )

    report = hrd.collect_high_return_report(output_root=out, task_list_path=path)
    assert report["shortlist_count"] == 3
    assert report["small_capital_gate_summary"]["matrix_entry_candidate_count"] == 4

    calls: list[list[str]] = []

    def fake_run(command, *, stdout_path, stderr_path):
        calls.append(list(command))
        return {"returncode": 0, "stdout": str(stdout_path), "stderr": str(stderr_path)}

    monkeypatch.setattr(hrd, "_run_command", fake_run)

    summary = hrd.run_candidate_matrix_tasks(output_root=out, task_list_path=path)

    assert summary["status"] == "completed"
    assert summary["matrix_entry_only"] is True
    assert set(summary["completed_tags"]) == {task["tag"] for task in tasks}
    assert summary["skipped"] == []
    assert len(calls) == 4
    assert all(call[call.index("-m") + 1] == "daily_research.path_policy.v2_candidate_review_matrix" for call in calls)


def test_run_forecast_tasks_supports_max_tasks_for_reboot_safe_batches(tmp_path: Path, monkeypatch) -> None:
    studies = tmp_path / "studies"
    monkeypatch.setattr(hrd, "STUDIES_ROOT", studies)
    monkeypatch.setattr(hrd, "_active_artifact_has_diff", lambda: False)
    calls: list[list[str]] = []

    def fake_run(command, *, stdout_path, stderr_path):
        calls.append(list(command))
        tag = command[command.index("--tag") + 1]
        _write_json(studies / tag / "study_summary.json", {"status": "completed"})
        return {"returncode": 0, "stdout": str(stdout_path), "stderr": str(stderr_path)}

    monkeypatch.setattr(hrd, "_run_command", fake_run)

    summary = hrd.run_forecast_tasks(
        output_root=tmp_path / "out",
        datasets=("same_period_augmented",),
        splits=("same",),
        model_families=("gru_sequence_static_context", "hybrid_expert_fusion_static_context"),
        seeds=(7,),
        allow_single_seed_scout=True,
        max_tasks=1,
    )

    assert summary["status"] == "completed"
    assert summary["launched_task_count"] == 1
    assert Path(summary["task_list_path"]).exists()
    assert len(calls) == 1
    assert any(result.get("status") == "deferred_by_max_tasks" for result in summary["results"])


def test_cli_write_task_list_routes_to_high_return_discovery(tmp_path: Path, monkeypatch, capsys) -> None:
    monkeypatch.setattr(hrd, "STUDIES_ROOT", tmp_path / "studies")
    monkeypatch.setattr(hrd, "_active_artifact_has_diff", lambda: False)

    rc = hrd.main(
        [
            "--write-task-list",
            "--allow-single-seed-scout",
            "--datasets",
            "same_period_augmented",
            "--model-families",
            "gru_sequence_static_context",
            "--output-root",
            str(tmp_path / "out"),
            "--json",
        ]
    )

    captured = capsys.readouterr()
    payload = json.loads(captured.out)
    task_list = Path(payload["actions"]["task_list"])
    assert rc == 0
    assert payload["stage"] == "high_return_model_discovery"
    assert task_list.exists()
    assert json.loads(task_list.read_text(encoding="utf-8"))["training_task_count"] == 1
