from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from daily_research.path_policy import v2_local_state_input_scout as scout
from daily_research.path_policy.tests.fixtures import make_prepared_policy_inputs


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_build_training_tasks_use_strict_pool_and_local_state_profile() -> None:
    tasks = scout.build_training_tasks(output_root=Path("anchor"), seeds=(7, 11, 19))

    assert [task["seed"] for task in tasks] == [7, 11, 19]
    assert tasks[0]["source_pool_view_id"] == scout.V2_STRICT_POOL_VIEW_ID
    assert tasks[0]["feature_profile"] == scout.FEATURE_PROFILE
    assert tasks[0]["baseline_feature_profile"] == scout.BASELINE_FEATURE_PROFILE
    assert tasks[0]["command"][tasks[0]["command"].index("--forecast-feature-profile") + 1] == scout.FEATURE_PROFILE
    assert "--forecast-memmap-manifest" not in tasks[0]["command"]
    assert "--forecast-memmap-manifest" in tasks[1]["command"]
    assert tasks[1]["command"][tasks[1]["command"].index("--forecast-memmap-manifest") + 1].endswith(
        "forecast_dataset_manifest.json"
    )
    assert tasks[0]["promotion_allowed"] is False
    assert tasks[0]["active_execution_strategy_expected_diff"] == "none"


def test_validate_local_state_manifest_requires_local_group_and_blocks_alpha(tmp_path: Path) -> None:
    manifest = {
        "source_market_dataset_id": scout.DATASET_ID,
        "source_pool_view_id": scout.V2_STRICT_POOL_VIEW_ID,
        "feature_profile": scout.FEATURE_PROFILE,
        "feature_store_shape": [10, 3, 4],
        "feature_columns": ["raw_open_gap_1d", "local_vol_20d", "local_high_volatility_x_reversal", "score_rank_pct"],
        "feature_group_counts": {"raw_kline": 1, "local_state_context": 0},
        "feature_profile_audit": {"amount_unit": {"amount_unit_policy": "as_is"}},
    }
    path = tmp_path / "forecast_dataset_manifest.json"
    _write_json(path, manifest)

    payload = scout.validate_local_state_manifest(path)

    assert payload["status"] == "blocked"
    assert "missing_local_state_context_features" in payload["blockers"]
    assert "alpha_prior_or_score_columns_present" in payload["blockers"]
    assert payload["alpha_like_feature_count"] == 1


def test_validate_local_state_manifest_accepts_valid_manifest(tmp_path: Path) -> None:
    manifest = {
        "source_market_dataset_id": scout.DATASET_ID,
        "source_pool_view_id": scout.V2_STRICT_POOL_VIEW_ID,
        "feature_profile": scout.FEATURE_PROFILE,
        "feature_store_shape": [10, 3, 3],
        "feature_columns": ["raw_open_gap_1d", "local_vol_20d", "local_high_volatility_x_reversal"],
        "feature_group_counts": {"raw_kline": 1, "local_state_context": 2},
        "feature_profile_audit": {"amount_unit": {"amount_unit_policy": "as_is"}},
    }
    path = tmp_path / "forecast_dataset_manifest.json"
    _write_json(path, manifest)

    payload = scout.validate_local_state_manifest(path)

    assert payload["status"] == "ok"
    assert payload["local_state_context_feature_count"] == 2


def test_run_profile_smoke_writes_real_contract_payload(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    prepared = make_prepared_policy_inputs(days=90, stocks=("AAA", "BBB", "CCC", "DDD"), start_date="2024-01-02")
    monkeypatch.setattr(scout, "ResearchDataLake", lambda *_args, **_kwargs: object())
    monkeypatch.setattr(scout, "load_policy_inputs_from_lake", lambda **_kwargs: prepared)

    payload = scout.run_profile_smoke(output_root=tmp_path / "out", data_lake_root=tmp_path / "lake")

    assert payload["status"] == "ok"
    assert payload["feature_profile"] == scout.FEATURE_PROFILE
    assert payload["local_state_feature_count"] > 0
    assert payload["local_finite_ratio"] > 0.0
    assert payload["has_alpha_like"] is False
    assert (tmp_path / "out" / "v2_local_state_profile_smoke.json").exists()


def test_write_task_list_is_research_only(tmp_path: Path) -> None:
    path = scout.write_task_list(tmp_path / "out", seeds=(7,))
    payload = json.loads(path.read_text(encoding="utf-8"))

    assert payload["research_program"] == scout.RESEARCH_PROGRAM
    assert payload["study_family"] == scout.STUDY_FAMILY
    assert payload["training_task_count"] == 1
    assert payload["training_tasks"][0]["promotion_allowed"] is False
    assert payload["boundary"]["active_execution_strategy_expected_diff"] == "none"


def test_local_state_input_scout_active_artifact_guard_blocks(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(scout, "_active_artifact_has_diff", lambda: True)

    with pytest.raises(ValueError, match="active_artifact_diff_blocker"):
        scout.run_training_tasks(output_root=tmp_path / "out", seeds=(7,))
