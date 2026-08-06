from __future__ import annotations

import hashlib
import json

import pandas as pd
import pytest

from daily_research.path_policy import seq100_quality_liquidity_data_prep as base
from daily_research.path_policy import seq100_quality_liquidity_research_scope as scope


def _config() -> dict[str, object]:
    return {
        "study_id": scope.STUDY_ID,
        "source_data_prep": {"study_id": scope.SOURCE_STUDY_ID},
        "period": {
            "data_history_start": scope.DATA_HISTORY_START,
            "research_start_date": scope.RESEARCH_START,
            "end_date": scope.END_DATE,
            "forbidden_year": 2026,
            "future_oos_prediction_years": list(scope.OOS_YEARS),
        },
        "burn_in": {
            "years": list(scope.BURN_IN_YEARS),
            "available_for_feature_history": True,
            "eligible_for_training": False,
            "eligible_for_evaluation": False,
            "eligible_for_atlas_statistics": False,
            "eligible_for_labels": False,
        },
        "training": {"performed": False},
    }


def test_config_keeps_2010_2011_as_burn_in_only(tmp_path) -> None:
    path = tmp_path / "study.json"
    path.write_text(json.dumps(_config()), encoding="utf-8")

    config = scope._load_config(path)

    assert config["period"]["research_start_date"] == "2012-01-01"
    assert config["burn_in"]["available_for_feature_history"] is True
    assert config["burn_in"]["eligible_for_training"] is False


def test_config_rejects_burn_in_as_training_year(tmp_path) -> None:
    config = _config()
    config["burn_in"]["eligible_for_training"] = True
    path = tmp_path / "study.json"
    path.write_text(json.dumps(config), encoding="utf-8")

    with pytest.raises(scope.ResearchScopeError, match="burn_in_semantics"):
        scope._load_config(path)


@pytest.mark.parametrize(
    "field", ["eligible_for_atlas_statistics", "eligible_for_labels"]
)
def test_config_rejects_burn_in_in_formal_derivations(tmp_path, field) -> None:
    config = _config()
    config["burn_in"][field] = True
    path = tmp_path / "study.json"
    path.write_text(json.dumps(config), encoding="utf-8")

    with pytest.raises(scope.ResearchScopeError, match="burn_in_semantics"):
        scope._load_config(path)


def test_common_support_hash_can_exclude_burn_in_partition(tmp_path) -> None:
    records = {}
    for year, candidate_id in ((2010, 10), (2011, 11)):
        path = tmp_path / f"{year}.parquet"
        pd.DataFrame(
            {
                "candidate_id": [candidate_id],
                "trade_date": [f"{year}-01-04"],
                "symbol": ["000001.SZ"],
            }
        ).to_parquet(path, index=False)
        records[str(year)] = {"support_path": str(path)}
    expected = hashlib.sha256(b"11|2011-01-04|000001.SZ\n").hexdigest()

    assert base._common_support_hash(records, years=(2011,)) == expected
    assert base._common_support_hash(records, years=(2010, 2011)) != expected


def test_feature_block_hash_excludes_burn_in_partition() -> None:
    def record(value: str) -> dict[str, str]:
        return {
            "sha256": value,
            "support_sha256": f"support-{value}",
            "input_fingerprint": f"input-{value}",
        }

    state = {
        "feature_blocks": {
            "2010": {family: record("old") for family in scope.FEATURE_FAMILIES},
            "2011": {family: record("new") for family in scope.FEATURE_FAMILIES},
        }
    }
    without_burn_in = base._feature_blocks_hash(state, years=(2011,))
    state["feature_blocks"]["2010"]["minute"]["sha256"] = "changed"

    assert base._feature_blocks_hash(state, years=(2011,)) == without_burn_in
    assert base._feature_blocks_hash(state, years=(2010, 2011)) != without_burn_in


def test_rolling_folds_start_in_2012_and_expand() -> None:
    membership = {
        str(year): {
            "start_date": f"{year}-01-01",
            "end_date": f"{year}-12-31",
            "common_support_row_count": year,
        }
        for year in scope.RESEARCH_YEARS
    }

    folds = scope._rolling_oos_folds(membership)

    assert [fold["evaluation_year"] for fold in folds] == [2023, 2024, 2025]
    assert folds[0]["training_years"] == list(range(2012, 2023))
    assert folds[1]["training_years"] == list(range(2012, 2024))
    assert folds[2]["training_years"] == list(range(2012, 2025))
    assert all(not {2010, 2011}.intersection(fold["training_years"]) for fold in folds)
    assert folds[0]["training_candidate_row_count_before_target_purge"] == sum(
        range(2012, 2023)
    )


def test_only_matching_completed_atlas_is_reused() -> None:
    previous = {
        "study_id": scope.STUDY_ID,
        "source_study_id": scope.SOURCE_STUDY_ID,
        "input_fingerprint": "same",
        "atlas": {"status": "completed", "manifest_sha256": "abc"},
    }

    assert (
        scope._reusable_atlas_state(previous, input_fingerprint="same")
        == previous["atlas"]
    )
    assert scope._reusable_atlas_state(previous, input_fingerprint="changed") == {}
    previous["atlas"]["status"] = "running"
    assert scope._reusable_atlas_state(previous, input_fingerprint="same") == {}


def test_self_test_forbids_2026_and_training() -> None:
    result = scope.self_test()

    assert result["status"] == "ok"
    assert result["checks"]["research_start_date"] == "2012-01-01"
    assert result["checks"]["training_performed"] is False
