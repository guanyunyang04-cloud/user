from __future__ import annotations

import pytest

from daily_research.path_policy.run_alpha_path20_protocol import build_arg_parser


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
