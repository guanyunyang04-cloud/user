from __future__ import annotations

import pytest

from daily_research.path_policy.run_alpha_path20_protocol import (
    _multiyear_aggregate,
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
