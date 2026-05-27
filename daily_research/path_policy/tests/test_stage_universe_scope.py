from __future__ import annotations

import json

from daily_research.path_policy.stage_universe_scope import (
    CAP80_DIAGNOSTIC_SCOPE,
    FULL_ROLLING_LIQUID500_SCOPE,
    full_rolling_liquid500_command_args,
    study_scope_from_summary,
    study_scope_from_summary_path,
)


def test_study_scope_classifies_full_rolling_liquid500() -> None:
    summary = {
        "prepared_summary": {"universe_size": 2430},
        "dataset_manifest": {
            "source_pool_view_kind": "rolling_liquidity",
            "source_pool_view_name": "rolling_liquid500",
            "source_pool_view_id": "policy_pool_view__c11400fa72ad263f3d1eecfa",
            "feature_store_shape": [1699, 2430, 156],
            "sample_count_by_role": {"train": 452034, "validation": 103958, "test": 104972},
        },
    }

    scope = study_scope_from_summary(summary)

    assert scope["universe_scope"] == FULL_ROLLING_LIQUID500_SCOPE
    assert scope["universe_size"] == 2430
    assert scope["train_rows"] == 452034


def test_study_scope_classifies_default_cap80_as_diagnostic(tmp_path) -> None:
    summary_path = tmp_path / "study_summary.json"
    summary_path.write_text(
        json.dumps(
            {
                "prepared_summary": {"universe_size": 80},
                "dataset_manifest": {
                    "source_pool_view_kind": "",
                    "source_pool_view_name": "",
                    "feature_store_shape": [1699, 80, 156],
                },
                "training_summary": {
                    "models": {
                        "gru_sequence_static_context": {
                            "train_rows": 74640,
                        }
                    }
                },
            }
        ),
        encoding="utf-8",
    )

    scope = study_scope_from_summary_path(summary_path)

    assert scope["universe_scope"] == CAP80_DIAGNOSTIC_SCOPE
    assert scope["universe_size"] == 80
    assert scope["train_rows"] == 74640


def test_full_pool_command_args_bind_pool_view_and_manifest() -> None:
    command_args = full_rolling_liquid500_command_args("studies_root")

    assert command_args[command_args.index("--pool-view-kind") + 1] == "rolling_liquidity"
    assert command_args[command_args.index("--pool-view-name") + 1] == "rolling_liquid500"
    assert command_args[command_args.index("--max-universe-size") + 1] == "0"
    assert command_args[command_args.index("--forecast-memmap-manifest") + 1].endswith(
        "forecast_dataset_manifest.json"
    )
