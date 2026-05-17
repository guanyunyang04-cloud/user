from __future__ import annotations

import pytest

from daily_research.path_policy.tests.fixtures import make_prepared_policy_inputs
from daily_research.path_policy.run_alpha_path20_protocol import (
    _aggregate_projection_parity_summaries,
    _baseline_targets_for_episode,
    _run_forecast_walkforward_study,
    _run_baseline_suite,
    _run_v5_dt_validation_study,
    _matrix_evidence_diagnostics,
    _multiyear_aggregate,
    _run_v4_validation_repair_study,
    _run_walkforward_matrix,
    _run_walkforward_study,
    _select_v4_checkpoint,
    _v4_multi_seed_summary,
    _v4_verdict,
    _walkforward_evidence_diagnostics,
    _walkforward_evidence_verdict,
    _validate_protocol_args,
    build_arg_parser,
)


def test_rl_protocol_parser_accepts_sequence_stage_and_rejects_latest_in_main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args(
        [
            "--stage",
            "rl-dataset-smoke",
            "--tag",
            "unit",
            "--data-source",
            "lake",
            "--lake-dataset-id",
            "policy_input_bundle__fixed",
            "--sequence-length",
            "5",
            "--model-family",
            "sequence_gru",
        ]
    )
    _validate_protocol_args(parser, args)

    assert args.stage == "rl-dataset-smoke"
    assert args.no_oracle_input is True
    assert args.policy_version == "alpha_path20_sequence_policy_v1"


def test_protocol_parser_defaults_to_current_neural_mainline_dataset_stage() -> None:
    parser = build_arg_parser()
    args = parser.parse_args(
        [
            "--tag",
            "unit",
            "--data-source",
            "lake",
            "--lake-dataset-id",
            "policy_input_bundle__fixed",
        ]
    )
    _validate_protocol_args(parser, args)

    assert args.stage == "dataset-smoke"
    assert args.policy_version == "alpha_path20_neural_policy_v1"


def test_protocol_parser_accepts_forecast_walkforward_stage_with_neural_policy_defaults() -> None:
    parser = build_arg_parser()
    args = parser.parse_args(
        [
            "--stage",
            "forecast-walkforward-study",
            "--tag",
            "unit_forecast",
            "--data-source",
            "lake",
            "--lake-dataset-id",
            "policy_input_bundle__fixed",
        ]
    )
    _validate_protocol_args(parser, args)

    assert args.stage == "forecast-walkforward-study"
    assert args.policy_version == "alpha_path20_neural_policy_v1"
    assert args.forecast_lookback_days == 252
    assert args.forecast_train_start_year == 2019
    assert args.forecast_train_end_year == 2022
    assert args.forecast_validation_year == 2023
    assert args.forecast_test_year == 2024
    assert args.forecast_hidden_dim == 192
    assert args.forecast_dropout == pytest.approx(0.15)
    assert args.forecast_gru_layers == 2
    assert args.forecast_transformer_layers == 4
    assert args.forecast_transformer_heads == 6
    assert args.forecast_patch_sizes == "4,20"
    assert args.forecast_device == "auto"
    assert args.forecast_amp is True
    assert args.forecast_min_epochs == 20
    assert args.forecast_early_stop_patience == 12
    assert args.forecast_seeds == "7"


def test_protocol_parser_accepts_walkforward_matrix_stage() -> None:
    args = build_arg_parser().parse_args(
        [
            "--stage",
            "rl-walkforward-matrix",
            "--tag",
            "unit",
            "--data-source",
            "lake",
            "--lake-dataset-id",
            "policy_input_bundle__fixed",
            "--model-family-matrix",
            "sequence_gru,decision_transformer",
            "--rollout-grad-mode",
            "truncated",
            "--rollout-chunk-days",
            "5",
        ]
    )

    assert args.stage == "rl-walkforward-matrix"
    assert args.model_family_matrix == "sequence_gru,decision_transformer"
    assert args.rollout_grad_mode == "truncated"


def test_protocol_parser_accepts_v4_validation_repair_stage() -> None:
    args = build_arg_parser().parse_args(
        [
            "--stage",
            "rl-v4-validation-repair-study",
            "--tag",
            "unit_v4",
            "--data-source",
            "lake",
            "--lake-dataset-id",
            "policy_input_bundle__fixed",
            "--rl-seed-matrix",
            "7,11",
            "--projection-penalty-grid",
            "0.05,0.20",
            "--tail-mass-penalty-weight",
            "0.1",
        ]
    )

    assert args.stage == "rl-v4-validation-repair-study"
    assert args.rl_seed_matrix == "7,11"
    assert args.projection_penalty_grid == "0.05,0.20"
    assert args.tail_mass_penalty_weight == pytest.approx(0.1)


def test_protocol_parser_accepts_v5_dt_validation_stage() -> None:
    parser = build_arg_parser()
    args = parser.parse_args(
        [
            "--stage",
            "rl-v5-dt-validation-study",
            "--tag",
            "unit_v5",
            "--data-source",
            "lake",
            "--lake-dataset-id",
            "policy_input_bundle__fixed",
            "--dt-context-grid",
            "20,60",
        ]
    )

    assert args.stage == "rl-v5-dt-validation-study"
    assert args.dt_context_grid == "20,60"
    assert args.rl_hidden_dim is None
    assert args.rl_dropout is None
    assert args.smoke_lr is None
    _validate_protocol_args(parser, args)
    assert args.rl_hidden_dim == 64
    assert args.rl_dropout == pytest.approx(0.1)
    assert args.smoke_lr == pytest.approx(3.0e-4)


def test_forecast_walkforward_study_contract_fixture(tmp_path) -> None:
    prepared = make_prepared_policy_inputs(days=820, stocks=("AAA", "BBB", "CCC", "DDD"), start_date="2018-01-02")
    parser = build_arg_parser()
    args = parser.parse_args(
        [
            "--stage",
            "forecast-walkforward-study",
            "--tag",
            "unit_forecast_fixture",
            "--data-source",
            "lake",
            "--lake-dataset-id",
            "policy_input_bundle__fixed",
            "--forecast-lookback-days",
            "5",
            "--forecast-train-start-year",
            "2018",
            "--forecast-train-end-year",
            "2019",
            "--forecast-validation-year",
            "2020",
            "--forecast-test-year",
            "2021",
            "--forecast-model-families",
            "linear_last_day,mlp_last_day",
            "--forecast-epochs",
            "3",
            "--forecast-min-epochs",
            "1",
            "--forecast-early-stop-patience",
            "1",
            "--forecast-batch-size",
            "4",
            "--forecast-hidden-dim",
            "24",
            "--forecast-transformer-heads",
            "3",
            "--forecast-transformer-layers",
            "1",
            "--forecast-patch-sizes",
            "2,3",
            "--forecast-device",
            "cpu",
            "--no-forecast-amp",
            "--forecast-seeds",
            "7,11",
            "--forecast-max-samples-per-role",
            "8",
        ]
    )
    _validate_protocol_args(parser, args)

    summary = _run_forecast_walkforward_study(
        prepared=prepared,
        study_root=tmp_path,
        tag="unit_forecast_fixture",
        args=args,
    )

    assert summary["status"] == "completed"
    assert summary["stage"] == "forecast_walkforward_study"
    assert summary["policy_version"] == "alpha_path20_neural_policy_v1"
    assert summary["shadow_only"] is True
    assert summary["promotion_allowed"] is False
    assert summary["active_execution_strategy_expected_diff"] == "none"
    assert summary["dataset_manifest"]["normalization"]["fit_role"] == "train_only"
    assert "linear_last_day" in summary["training_summary"]["models"]
    assert "mlp_last_day" in summary["training_summary"]["models"]
    assert summary["training_summary"]["selected_seed"] in {7, 11}
    assert "forecast_learning_curve_csv" in summary["training_summary"]


def test_current_neural_mainline_stage_no_longer_requires_legacy_allow_flag() -> None:
    parser = build_arg_parser()
    args = parser.parse_args(
        [
            "--stage",
            "tiny-smoke",
            "--tag",
            "unit",
            "--data-source",
            "lake",
            "--lake-dataset-id",
            "policy_input_bundle__fixed",
        ]
    )

    _validate_protocol_args(parser, args)
    assert args.stage == "tiny-smoke"
    assert args.policy_version == "alpha_path20_neural_policy_v1"


def test_rl_stage_rejects_loose_latest_dataset_id() -> None:
    parser = build_arg_parser()
    for loose_id in ("latest", "default", "latest_snapshot"):
        args = parser.parse_args(
            [
                "--stage",
                "rl-dataset-smoke",
                "--tag",
                "unit",
                "--data-source",
                "lake",
                "--lake-dataset-id",
                loose_id,
            ]
        )

        with pytest.raises(SystemExit):
            _validate_protocol_args(parser, args)


def test_protocol_parser_rejects_unknown_rl_model_family() -> None:
    with pytest.raises(SystemExit):
        build_arg_parser().parse_args(
            [
                "--stage",
                "rl-train-smoke",
                "--tag",
                "unit",
                "--lake-dataset-id",
                "policy_input_bundle__fixed",
                "--model-family",
                "unknown",
            ]
        )


def test_multiyear_aggregate_excludes_incomplete_years() -> None:
    aggregate = _multiyear_aggregate(
        {
            "2019": {
                "dataset_summary": {"status": "completed", "trading_day_count": 244},
                "replay_summary": {
                    "status": "completed",
                    "metrics": {
                        "total_return": 0.10,
                        "annual_return": 0.11,
                        "sharpe": 0.8,
                        "max_drawdown": -0.05,
                        "monthly_win_rate": 0.6,
                        "avg_turnover": 0.2,
                        "avg_projected_gross_exposure": 0.7,
                        "avg_projection_l1_distance": 0.03,
                    },
                },
            },
            "2020": {
                "dataset_summary": {
                    "status": "incomplete",
                    "incomplete_reason": "trading_day_count_below_180",
                    "trading_day_count": 90,
                },
                "replay_summary": {"status": "skipped", "reason": "incomplete_dataset"},
            },
        }
    )

    assert aggregate["completed_year_count"] == 1
    assert aggregate["completed_years"] == ["2019"]
    assert aggregate["incomplete_year_count"] == 1
    assert aggregate["incomplete_years"]["2020"]["reason"] == "trading_day_count_below_180; incomplete_dataset"
    assert aggregate["mean_total_return"] == pytest.approx(0.10)
    assert aggregate["worst_max_drawdown"] == pytest.approx(-0.05)


def test_walkforward_summary_separates_train_validation_test(tmp_path, monkeypatch) -> None:
    import daily_research.path_policy.run_alpha_path20_protocol as protocol

    args = build_arg_parser().parse_args(
        [
            "--stage",
            "rl-walkforward-study",
            "--tag",
            "unit_walk",
            "--data-source",
            "lake",
            "--lake-dataset-id",
            "policy_input_bundle__fixed",
            "--sequence-length",
            "3",
            "--smoke-epochs",
            "1",
            "--rl-hidden-dim",
            "16",
            "--max-position-weight",
            "0.20",
            "--max-gross-exposure",
            "0.50",
            "--max-positions",
            "2",
            "--turnover-budget",
            "0.40",
            "--min-year-trading-days",
            "2",
        ]
    )
    args.lake_dataset_id = "policy_input_bundle__fixed"
    monkeypatch.setattr(protocol, "PATH_POLICY_EPISODE_DATASETS_ROOT", tmp_path / "episode_datasets")
    monkeypatch.setattr(
        protocol,
        "_prepare_for_window",
        lambda args, start_date, end_date, tag: make_prepared_policy_inputs(
            days=12,
            stocks=("AAA", "BBB", "CCC"),
            start_date=f"{start_date[:4]}-01-02",
        ),
    )

    summary = _run_walkforward_study(study_root=tmp_path / "study", tag="unit_walk", args=args)

    assert summary["stage"] == "rl_walkforward_study"
    assert summary["train_years"] == [2019, 2020]
    assert summary["validation_years"] == [2022]
    assert summary["test_years"] == [2024]
    assert set(summary["aggregate"]) == {"train", "validation", "test"}
    assert summary["yearly"]["2024"]["role"] == "test"
    assert "validation_surrogate_metrics" in summary["train_summary"]
    assert "validation_replay_metrics" not in summary["train_summary"]
    assert "validation_exact_replay_metrics" in summary
    assert "surrogate_exact_gap" in summary
    assert "projection_parity_summary" in summary
    assert "evidence_diagnostics" in summary
    assert "attribution_by_role" in summary["evidence_diagnostics"]
    assert "attribution_csv" in summary["yearly"]["2024"]["replay_summary"]["artifacts"]
    assert "raw_intent_diagnostics_summary" in summary["train_summary"]
    assert "projected_learning_health" in summary["train_summary"]
    assert summary["evidence_verdict"] in {
        "contract_passed",
        "diagnosed_failure",
        "validation_promising",
        "test_promising",
        "insufficient_or_incomplete",
    }
    assert summary["oracle_used"] is False


def test_episode_artifact_reuse_and_hash_mismatch_failure(tmp_path, monkeypatch) -> None:
    import json
    import daily_research.path_policy.run_alpha_path20_protocol as protocol

    args = build_arg_parser().parse_args(
        [
            "--stage",
            "rl-episode-dataset",
            "--tag",
            "unit_reuse",
            "--data-source",
            "lake",
            "--lake-dataset-id",
            "policy_input_bundle__fixed",
            "--sequence-length",
            "3",
            "--min-year-trading-days",
            "2",
        ]
    )
    args.lake_dataset_id = "policy_input_bundle__fixed"
    monkeypatch.setattr(protocol, "PATH_POLICY_EPISODE_DATASETS_ROOT", tmp_path / "episode_datasets")
    prepared = make_prepared_policy_inputs(days=12, stocks=("AAA", "BBB", "CCC"), start_date="2019-01-02")

    first = protocol._build_episode_dataset_artifact(
        prepared=prepared,
        study_root=tmp_path / "study1",
        tag="unit_reuse",
        args=args,
        year=2019,
        start_date="20190101",
        end_date="20191231",
    )
    second = protocol._build_episode_dataset_artifact(
        prepared=prepared,
        study_root=tmp_path / "study2",
        tag="unit_reuse_other_tag",
        args=args,
        year=2019,
        start_date="20190101",
        end_date="20191231",
    )

    assert second["manifest"]["artifact_reused"] is True
    manifest_path = first["manifest"]["manifest_json"]
    manifest = json.loads(open(manifest_path, encoding="utf-8").read())
    manifest["manifest_hash"] = "broken"
    with open(manifest_path, "w", encoding="utf-8") as handle:
        json.dump(manifest, handle)

    with pytest.raises(ValueError, match="hash mismatch"):
        protocol._build_episode_dataset_artifact(
            prepared=prepared,
            study_root=tmp_path / "study3",
            tag="unit_reuse",
            args=args,
            year=2019,
            start_date="20190101",
            end_date="20191231",
        )


def test_walkforward_verdict_downgrades_on_projection_mismatch() -> None:
    verdict = _walkforward_evidence_verdict(
        aggregate={
            "validation": {"completed_year_count": 1, "mean_total_return": 0.10, "worst_max_drawdown": -0.01},
            "test": {"completed_year_count": 1, "mean_total_return": 0.10, "worst_max_drawdown": -0.01},
        },
        projection_parity_summary={"projection_mismatch_warning": True},
        validation_exact_replay_metrics={"status": "completed"},
        test_exact_replay_metrics={"status": "completed"},
    )

    assert verdict == "insufficient_or_incomplete"


def test_walkforward_verdict_requires_complete_validation_before_test_success() -> None:
    verdict = _walkforward_evidence_verdict(
        aggregate={
            "validation": {"completed_year_count": 0, "mean_total_return": 0.10, "worst_max_drawdown": -0.01},
            "test": {"completed_year_count": 1, "mean_total_return": 0.10, "worst_max_drawdown": -0.01},
        },
        projection_parity_summary={"projection_mismatch_warning": False},
        validation_exact_replay_metrics={"status": "not_available"},
        test_exact_replay_metrics={"status": "completed"},
    )

    assert verdict == "insufficient_or_incomplete"


def test_walkforward_verdict_negative_validation_can_be_diagnosed_failure() -> None:
    diagnostics = {
        "warnings": {
            "validation_generalization_failure": True,
            "under_exposure_warning": False,
            "projection_overcorrection_warning": False,
            "surrogate_exact_gap_warning": False,
        }
    }

    verdict = _walkforward_evidence_verdict(
        aggregate={
            "validation": {"completed_year_count": 1, "mean_total_return": -0.01, "mean_sharpe": -0.2, "worst_max_drawdown": -0.01},
            "test": {"completed_year_count": 1, "mean_total_return": 0.10, "mean_sharpe": 0.8, "worst_max_drawdown": -0.01},
        },
        projection_parity_summary={"projection_mismatch_warning": False},
        validation_exact_replay_metrics={"status": "completed"},
        test_exact_replay_metrics={"status": "completed"},
        evidence_diagnostics=diagnostics,
    )

    assert verdict == "diagnosed_failure"


def test_walkforward_evidence_diagnostics_flags_projection_and_exposure() -> None:
    yearly = {
        "2022": {
            "role": "validation",
            "replay_summary": {
                "status": "completed",
                "attribution_summary": {
                    "status": "completed",
                    "gross_return_sum": -0.01,
                    "estimated_cost_sum": 0.001,
                    "net_return_sum": -0.011,
                    "benchmark_return_sum": 0.0,
                    "excess_return_sum": -0.011,
                    "avg_cash_weight": 0.96,
                    "avg_projected_gross_exposure": 0.04,
                    "avg_projected_turnover": 0.01,
                    "avg_projection_l1_distance": 0.10,
                    "avg_target_count": 1.0,
                },
            },
        }
    }
    diagnostics = _walkforward_evidence_diagnostics(
        yearly=yearly,
        train_summary={
            "raw_intent_diagnostics_summary": {"avg_raw_gross_exposure": 0.20},
            "projection_diagnostics_summary": {"avg_projection_l1_distance": 0.10},
        },
        aggregate={
            "train": {"mean_total_return": 0.05},
            "validation": {
                "mean_total_return": -0.02,
                "mean_sharpe": -0.5,
                "mean_avg_projected_gross_exposure": 0.04,
                "mean_avg_projection_l1_distance": 0.10,
                "mean_avg_turnover": 0.01,
            },
            "test": {"mean_total_return": 0.01},
        },
        projection_parity_summary={"projection_mismatch_warning": False},
        surrogate_exact_gap={"status": "completed", "projection_l1_gap": 0.03},
    )

    assert diagnostics["status"] == "diagnosed"
    assert diagnostics["warnings"]["under_exposure_warning"] is True
    assert diagnostics["warnings"]["projection_overcorrection_warning"] is True
    assert diagnostics["warnings"]["surrogate_exact_gap_warning"] is True
    assert diagnostics["warnings"]["validation_generalization_failure"] is True


def test_walkforward_matrix_outputs_two_model_families(tmp_path, monkeypatch) -> None:
    import daily_research.path_policy.run_alpha_path20_protocol as protocol

    args = build_arg_parser().parse_args(
        [
            "--stage",
            "rl-walkforward-matrix",
            "--tag",
            "unit_matrix",
            "--data-source",
            "lake",
            "--lake-dataset-id",
            "policy_input_bundle__fixed",
            "--sequence-length",
            "3",
            "--smoke-epochs",
            "1",
            "--rl-hidden-dim",
            "8",
            "--max-position-weight",
            "0.20",
            "--max-gross-exposure",
            "0.50",
            "--max-positions",
            "2",
            "--turnover-budget",
            "0.40",
            "--model-family-matrix",
            "sequence_gru,decision_transformer",
            "--min-year-trading-days",
            "2",
        ]
    )
    args.lake_dataset_id = "policy_input_bundle__fixed"
    monkeypatch.setattr(protocol, "PATH_POLICY_EPISODE_DATASETS_ROOT", tmp_path / "episode_datasets")
    monkeypatch.setattr(
        protocol,
        "_prepare_for_window",
        lambda args, start_date, end_date, tag: make_prepared_policy_inputs(
            days=12,
            stocks=("AAA", "BBB", "CCC"),
            start_date=f"{start_date[:4]}-01-02",
        ),
    )

    summary = _run_walkforward_matrix(study_root=tmp_path / "matrix", tag="unit_matrix", args=args)

    assert summary["stage"] == "rl_walkforward_matrix"
    assert set(summary["models"]) == {"sequence_gru", "decision_transformer"}
    assert summary["models"]["sequence_gru"]["walkforward_summary"]["stage"] == "rl_walkforward_study"
    assert summary["models"]["decision_transformer"]["walkforward_summary"]["train_summary"]["model_family"] == "decision_transformer"
    assert "evidence_diagnostics" in summary
    assert "evidence_diagnostics" in summary["models"]["sequence_gru"]
    assert summary["evidence_verdict"] in {
        "contract_passed",
        "diagnosed_failure",
        "validation_promising",
        "test_promising",
        "insufficient_or_incomplete",
    }


def test_matrix_evidence_diagnostics_selects_best_validation_family() -> None:
    diagnostics = _matrix_evidence_diagnostics(
        {
            "a": {
                "walkforward_summary": {"aggregate": {"validation": {"mean_total_return": -0.10}}},
                "evidence_diagnostics": {"warnings": {"validation_generalization_failure": True}},
            },
            "b": {
                "walkforward_summary": {"aggregate": {"validation": {"mean_total_return": 0.02}}},
                "evidence_diagnostics": {"warnings": {}},
            },
        }
    )

    assert diagnostics["status"] == "diagnosed"
    assert diagnostics["best_validation_family"] == "b"


def test_v4_checkpoint_selection_ignores_test_metrics() -> None:
    selected = _select_v4_checkpoint(
        [
            {
                "epoch": 1,
                "validation_metrics": {
                    "total_return": -0.01,
                    "sharpe": -0.1,
                    "avg_projection_l1_distance": 0.01,
                    "avg_turnover": 0.01,
                },
                "test_metrics": {"total_return": 10.0},
            },
            {
                "epoch": 2,
                "validation_metrics": {
                    "total_return": 0.01,
                    "sharpe": 0.2,
                    "avg_projection_l1_distance": 0.10,
                    "avg_turnover": 0.20,
                },
                "test_metrics": {"total_return": -10.0},
            },
        ]
    )

    assert selected["epoch"] == 2
    assert selected["test_metrics_used_for_selection"] is False


def test_v4_verdict_requires_positive_validation_and_baseline_win() -> None:
    verdict = _v4_verdict(
        selected_checkpoint={
            "status": "selected",
            "validation_metrics": {"total_return": 0.01, "sharpe": 0.4},
        },
        baseline_comparison={
            "score_blend_top30": {
                "validation": {"aggregate": {"mean_total_return": 0.02}},
            }
        },
        projection_parity_summary={"projection_mismatch_warning": False},
    )

    assert verdict == "diagnosed_failure"

    promising = _v4_verdict(
        selected_checkpoint={
            "status": "selected",
            "validation_metrics": {"total_return": 0.03, "sharpe": 0.4},
        },
        baseline_comparison={
            "score_blend_top30": {
                "validation": {"aggregate": {"mean_total_return": 0.02}},
            }
        },
        projection_parity_summary={"projection_mismatch_warning": False},
    )
    assert promising == "validation_promising"


def test_v4_projection_mismatch_downgrades_verdict() -> None:
    parity = _aggregate_projection_parity_summaries(
        [
            {"status": "passed", "max_target_l1_gap": 0.0, "projection_mismatch_warning": False},
            {"status": "warning", "max_target_l1_gap": 0.05, "projection_mismatch_warning": True},
        ]
    )
    verdict = _v4_verdict(
        selected_checkpoint={
            "status": "selected",
            "validation_metrics": {"total_return": 0.03, "sharpe": 0.4},
        },
        baseline_comparison={
            "score_blend_top30": {
                "validation": {"aggregate": {"mean_total_return": 0.02}},
            }
        },
        projection_parity_summary=parity,
    )

    assert parity["projection_mismatch_warning"] is True
    assert verdict == "insufficient_or_incomplete"


def test_v4_multiseed_summary_calculates_distribution() -> None:
    summary = _v4_multi_seed_summary(
        [
            {"selected_checkpoint": {"validation_metrics": {"total_return": -0.01, "sharpe": -0.1}}},
            {"selected_checkpoint": {"validation_metrics": {"total_return": 0.03, "sharpe": 0.4}}},
            {"selected_checkpoint": {"validation_metrics": {"total_return": 0.01, "sharpe": 0.2}}},
        ]
    )

    assert summary["seed_count"] == 3
    assert summary["validation_return_median"] == pytest.approx(0.01)
    assert summary["positive_validation_seed_count"] == 2


def test_v4_baseline_targets_do_not_use_oracle_or_future_columns() -> None:
    from daily_research.path_policy.rl_episode import build_path20_market_episode

    prepared = make_prepared_policy_inputs(days=12, stocks=("AAA", "BBB", "CCC"), start_date="2022-01-02")
    episode = build_path20_market_episode(
        prepared,
        start_date="20220101",
        end_date="20221231",
        lake_dataset_id="policy_input_bundle__fixture",
        year=2022,
        sequence_length=3,
        min_trading_days=2,
    )

    targets = _baseline_targets_for_episode(episode, "score_blend_top30", max_positions=2)

    assert targets
    assert not any("oracle" in str(column).lower() or "future" in str(column).lower() for column in episode.feature_columns)
    assert all(float(target.sum()) <= 1.0 for target in targets.values())


def test_v4_v3_final_checkpoint_baseline_skips_without_explicit_model(tmp_path) -> None:
    from daily_research.path_policy.rl_episode import build_path20_market_episode

    prepared = make_prepared_policy_inputs(days=12, stocks=("AAA", "BBB", "CCC"), start_date="2022-01-02")
    episode = build_path20_market_episode(
        prepared,
        start_date="20220101",
        end_date="20221231",
        lake_dataset_id="policy_input_bundle__fixture",
        year=2022,
        sequence_length=3,
        min_trading_days=2,
    )
    args = build_arg_parser().parse_args(
        [
            "--stage",
            "rl-v4-validation-repair-study",
            "--tag",
            "unit_v4",
            "--data-source",
            "lake",
            "--lake-dataset-id",
            "policy_input_bundle__fixed",
            "--sequence-length",
            "3",
            "--max-positions",
            "2",
            "--min-year-trading-days",
            "2",
        ]
    )
    results = _run_baseline_suite(
        prepared_by_year={2022: prepared, 2024: prepared},
        episode_by_year={2022: episode, 2024: episode},
        study_root=tmp_path,
        args=args,
    )

    assert results["v3_final_checkpoint"]["status"] == "skipped"
    assert results["v3_final_checkpoint"]["reason"] == "missing_explicit_v3_checkpoint_model_pt"


def test_v4_v3_final_checkpoint_baseline_loads_explicit_checkpoint(tmp_path) -> None:
    import torch

    import daily_research.path_policy.run_alpha_path20_protocol as protocol
    from daily_research.path_policy.rl_episode import (
        EPISODE_DYNAMIC_STOCK_FEATURE_COLUMNS,
        build_path20_market_episode,
        fit_episode_normalization,
    )
    from daily_research.path_policy.rl_models import SequencePolicyConfig, SequenceTargetWeightPolicy

    prepared = make_prepared_policy_inputs(days=12, stocks=("AAA", "BBB", "CCC"), start_date="2022-01-02")
    episode = build_path20_market_episode(
        prepared,
        start_date="20220101",
        end_date="20221231",
        lake_dataset_id="policy_input_bundle__fixture",
        year=2022,
        sequence_length=3,
        min_trading_days=2,
    )
    config = SequencePolicyConfig(
        stock_feature_dim=len(episode.feature_columns) + len(EPISODE_DYNAMIC_STOCK_FEATURE_COLUMNS),
        portfolio_feature_dim=8,
        hidden_dim=8,
        dropout=0.0,
        max_position_weight=0.20,
    )
    checkpoint_path = tmp_path / "sequence_policy_episode.pt"
    torch.save(
        {
            "model_family": "sequence_gru",
            "state_dict": SequenceTargetWeightPolicy(config).state_dict(),
            "feature_columns": episode.feature_columns,
            "dynamic_stock_feature_columns": list(EPISODE_DYNAMIC_STOCK_FEATURE_COLUMNS),
            "sequence_length": 3,
            "normalization": fit_episode_normalization([episode]),
            "policy_version": "alpha_path20_sequence_policy_v1",
            "shadow_only": True,
        },
        checkpoint_path,
    )
    args = build_arg_parser().parse_args(
        [
            "--stage",
            "rl-v4-validation-repair-study",
            "--tag",
            "unit_v4",
            "--data-source",
            "lake",
            "--lake-dataset-id",
            "policy_input_bundle__fixed",
            "--sequence-length",
            "3",
            "--max-position-weight",
            "0.20",
            "--max-gross-exposure",
            "0.50",
            "--max-positions",
            "2",
            "--turnover-budget",
            "0.40",
            "--min-year-trading-days",
            "2",
            "--v3-checkpoint-model-pt",
            str(checkpoint_path),
        ]
    )
    called = {"value": False}
    original_predict = protocol.predict_episode_targets

    def wrapped_predict(*items, **kwargs):
        called["value"] = True
        return original_predict(*items, **kwargs)

    protocol.predict_episode_targets = wrapped_predict
    try:
        results = _run_baseline_suite(
            prepared_by_year={2022: prepared, 2024: prepared},
            episode_by_year={2022: episode, 2024: episode},
            study_root=tmp_path / "baselines",
            args=args,
        )
    finally:
        protocol.predict_episode_targets = original_predict

    baseline = results["v3_final_checkpoint"]
    assert called["value"] is True
    assert baseline["status"] == "completed"
    assert baseline["checkpoint_model_pt"] == str(checkpoint_path.resolve())
    assert baseline["checkpoint_model_family"] == "sequence_gru"


def test_v4_validation_repair_study_contract_fixture(tmp_path, monkeypatch) -> None:
    import daily_research.path_policy.run_alpha_path20_protocol as protocol

    args = build_arg_parser().parse_args(
        [
            "--stage",
            "rl-v4-validation-repair-study",
            "--tag",
            "unit_v4",
            "--data-source",
            "lake",
            "--lake-dataset-id",
            "policy_input_bundle__fixed",
            "--sequence-length",
            "3",
            "--smoke-epochs",
            "1",
            "--rl-hidden-dim",
            "8",
            "--max-position-weight",
            "0.20",
            "--max-gross-exposure",
            "0.50",
            "--max-positions",
            "2",
            "--turnover-budget",
            "0.40",
            "--min-year-trading-days",
            "2",
            "--rl-seed-matrix",
            "7",
            "--projection-penalty-grid",
            "0.05",
        ]
    )
    args.lake_dataset_id = "policy_input_bundle__fixed"
    monkeypatch.setattr(protocol, "PATH_POLICY_EPISODE_DATASETS_ROOT", tmp_path / "episode_datasets")
    monkeypatch.setattr(
        protocol,
        "_prepare_for_window",
        lambda args, start_date, end_date, tag: make_prepared_policy_inputs(
            days=12,
            stocks=("AAA", "BBB", "CCC"),
            start_date=f"{start_date[:4]}-01-02",
        ),
    )

    summary = _run_v4_validation_repair_study(study_root=tmp_path / "v4", tag="unit_v4", args=args)

    assert summary["stage"] == "rl_v4_validation_repair_study"
    assert summary["seeds"] == [7]
    assert summary["projection_penalty_grid"] == [0.05]
    assert "checkpoint_exact_replay" in summary
    assert summary["selected_checkpoint"]["test_metrics_used_for_selection"] is False
    assert "baseline_comparison" in summary
    assert "multi_seed_summary" in summary
    assert "projection_tail_mass_summary" in summary
    assert summary["test_interpretable"] is summary["validation_passed"]
    assert summary["promotion_allowed"] is False


def test_v5_dt_validation_study_contract_fixture(tmp_path, monkeypatch) -> None:
    import daily_research.path_policy.run_alpha_path20_protocol as protocol

    args = build_arg_parser().parse_args(
        [
            "--stage",
            "rl-v5-dt-validation-study",
            "--tag",
            "unit_v5",
            "--data-source",
            "lake",
            "--lake-dataset-id",
            "policy_input_bundle__fixed",
            "--dt-context-grid",
            "3,5",
            "--smoke-epochs",
            "1",
            "--rl-hidden-dim",
            "8",
            "--dt-num-heads",
            "1",
            "--max-position-weight",
            "0.20",
            "--max-gross-exposure",
            "0.50",
            "--max-positions",
            "2",
            "--turnover-budget",
            "0.40",
            "--min-year-trading-days",
            "2",
            "--rl-seed-matrix",
            "7",
        ]
    )
    args.lake_dataset_id = "policy_input_bundle__fixed"
    monkeypatch.setattr(protocol, "PATH_POLICY_EPISODE_DATASETS_ROOT", tmp_path / "episode_datasets")
    monkeypatch.setattr(
        protocol,
        "_prepare_for_window",
        lambda args, start_date, end_date, tag: make_prepared_policy_inputs(
            days=12,
            stocks=("AAA", "BBB", "CCC"),
            start_date=f"{start_date[:4]}-01-02",
        ),
    )

    summary = _run_v5_dt_validation_study(study_root=tmp_path / "v5", tag="unit_v5", args=args)

    assert summary["stage"] == "rl_v5_dt_validation_study"
    assert summary["model_family"] == "decision_transformer_v2"
    assert summary["architecture_contract"]["action_imitation"] is False
    assert "context_length_comparison" in summary
    assert "baseline_comparison" in summary
    assert "temporal_context_diagnostics" in summary
    assert summary["selected_checkpoint"]["test_metrics_used_for_selection"] is False
    assert summary["promotion_allowed"] is False


def test_v5_long_context_incomplete_downgrades_verdict(tmp_path, monkeypatch) -> None:
    import daily_research.path_policy.run_alpha_path20_protocol as protocol

    args = build_arg_parser().parse_args(
        [
            "--stage",
            "rl-v5-dt-validation-study",
            "--tag",
            "unit_v5_incomplete",
            "--data-source",
            "lake",
            "--lake-dataset-id",
            "policy_input_bundle__fixed",
            "--dt-context-grid",
            "60",
            "--smoke-epochs",
            "1",
            "--rl-hidden-dim",
            "8",
            "--dt-num-heads",
            "1",
            "--min-year-trading-days",
            "2",
            "--rl-seed-matrix",
            "7",
        ]
    )
    args.lake_dataset_id = "policy_input_bundle__fixed"
    monkeypatch.setattr(protocol, "PATH_POLICY_EPISODE_DATASETS_ROOT", tmp_path / "episode_datasets")
    monkeypatch.setattr(
        protocol,
        "_prepare_for_window",
        lambda args, start_date, end_date, tag: make_prepared_policy_inputs(
            days=12,
            stocks=("AAA", "BBB", "CCC"),
            start_date=f"{start_date[:4]}-01-02",
        ),
    )

    summary = _run_v5_dt_validation_study(study_root=tmp_path / "v5_incomplete", tag="unit_v5_incomplete", args=args)

    assert summary["long_context_incomplete_warning"] is True
    assert summary["evidence_verdict"] == "insufficient_or_incomplete"
