from __future__ import annotations

import pytest

from daily_research.path_policy.tests.fixtures import make_prepared_policy_inputs
from daily_research.path_policy.run_alpha_path20_protocol import (
    _matrix_evidence_diagnostics,
    _multiyear_aggregate,
    _run_walkforward_matrix,
    _run_walkforward_study,
    _walkforward_evidence_diagnostics,
    _walkforward_evidence_verdict,
    _validate_protocol_args,
    build_arg_parser,
)


def test_rl_protocol_parser_accepts_sequence_stage_and_rejects_latest_in_main() -> None:
    args = build_arg_parser().parse_args(
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

    assert args.stage == "rl-dataset-smoke"
    assert args.no_oracle_input is True
    assert args.policy_version == "alpha_path20_sequence_policy_v1"


def test_protocol_parser_accepts_episode_mainline_default_stage() -> None:
    args = build_arg_parser().parse_args(
        [
            "--tag",
            "unit",
            "--data-source",
            "lake",
            "--lake-dataset-id",
            "policy_input_bundle__fixed",
        ]
    )

    assert args.stage == "rl-episode-dataset"


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


def test_legacy_neural_stage_requires_explicit_allow_flag() -> None:
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

    with pytest.raises(SystemExit):
        _validate_protocol_args(parser, args)

    allowed = parser.parse_args(
        [
            "--stage",
            "tiny-smoke",
            "--tag",
            "unit",
            "--data-source",
            "lake",
            "--lake-dataset-id",
            "policy_input_bundle__fixed",
            "--allow-legacy-neural-policy",
        ]
    )
    _validate_protocol_args(parser, allowed)


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
