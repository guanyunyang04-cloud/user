from __future__ import annotations

import json
from pathlib import Path
from typing import Any


FULL_ROLLING_LIQUID500_SCOPE = "full_rolling_liquid500"
CAP80_DIAGNOSTIC_SCOPE = "cap80_diagnostic"
PARTIAL_OR_UNKNOWN_SCOPE = "partial_or_unknown"

FULL_ROLLING_LIQUID500_REFERENCE_TAG = (
    "path20_horizon_discovery_no_alpha_gru_liquid500_"
    "h1_2_3_5_8_10_15_20_30_du_cost20_hit10_dd010_20260523_01"
)
FULL_ROLLING_LIQUID500_POOL_VIEW_ID = "policy_pool_view__c11400fa72ad263f3d1eecfa"
FULL_ROLLING_LIQUID500_POOL_VIEW_KIND = "rolling_liquidity"
FULL_ROLLING_LIQUID500_POOL_VIEW_NAME = "rolling_liquid500"
FULL_ROLLING_LIQUID500_MIN_UNIVERSE_SIZE = 500
FULL_ROLLING_LIQUID500_MIN_TRAIN_ROWS = 400_000
CAP80_MAX_UNIVERSE_SIZE = 80
CAP80_MAX_TRAIN_ROWS = 100_000


def full_rolling_liquid500_memmap_manifest(studies_root: str | Path) -> Path:
    return Path(studies_root) / FULL_ROLLING_LIQUID500_REFERENCE_TAG / "forecast_dataset_manifest.json"


def full_rolling_liquid500_command_args(studies_root: str | Path) -> list[str]:
    return [
        "--pool-name",
        FULL_ROLLING_LIQUID500_POOL_VIEW_NAME,
        "--pool-view-kind",
        FULL_ROLLING_LIQUID500_POOL_VIEW_KIND,
        "--pool-view-name",
        FULL_ROLLING_LIQUID500_POOL_VIEW_NAME,
        "--max-universe-size",
        "0",
        "--forecast-memmap-manifest",
        str(full_rolling_liquid500_memmap_manifest(studies_root)),
    ]


def full_pool_task_metadata(studies_root: str | Path) -> dict[str, Any]:
    return {
        "universe_scope": FULL_ROLLING_LIQUID500_SCOPE,
        "pool_view_kind": FULL_ROLLING_LIQUID500_POOL_VIEW_KIND,
        "pool_view_name": FULL_ROLLING_LIQUID500_POOL_VIEW_NAME,
        "pool_view_id": FULL_ROLLING_LIQUID500_POOL_VIEW_ID,
        "max_universe_size": 0,
        "forecast_memmap_manifest": str(full_rolling_liquid500_memmap_manifest(studies_root)),
        "expected_min_universe_size": FULL_ROLLING_LIQUID500_MIN_UNIVERSE_SIZE,
        "expected_min_train_rows": FULL_ROLLING_LIQUID500_MIN_TRAIN_ROWS,
    }


def _nested_int(payload: dict[str, Any], *path: str) -> int:
    value: Any = payload
    for key in path:
        if not isinstance(value, dict):
            return 0
        value = value.get(key)
    try:
        return int(value or 0)
    except (TypeError, ValueError):
        return 0


def _first_model_train_rows(payload: dict[str, Any]) -> int:
    models = payload.get("models")
    if not isinstance(models, dict):
        models = _nested_dict(payload, "training_summary", "models")
    if not isinstance(models, dict):
        return 0
    for model_payload in models.values():
        if isinstance(model_payload, dict):
            train_rows = _nested_int(model_payload, "train_rows")
            if train_rows:
                return train_rows
    return 0


def _nested_dict(payload: dict[str, Any], *path: str) -> dict[str, Any]:
    value: Any = payload
    for key in path:
        if not isinstance(value, dict):
            return {}
        value = value.get(key)
    return value if isinstance(value, dict) else {}


def _feature_shape(payload: dict[str, Any]) -> list[int]:
    manifest = _nested_dict(payload, "dataset_manifest")
    shape = manifest.get("feature_store_shape")
    if not shape:
        shape = _nested_dict(payload, "feature_manifest").get("feature_store_shape")
    if not isinstance(shape, list):
        return []
    result: list[int] = []
    for item in shape:
        try:
            result.append(int(item))
        except (TypeError, ValueError):
            return []
    return result


def study_scope_from_summary(payload: dict[str, Any]) -> dict[str, Any]:
    manifest = _nested_dict(payload, "dataset_manifest") or _nested_dict(payload, "feature_manifest")
    prepared = _nested_dict(payload, "prepared_summary")
    sample_count_by_role = manifest.get("sample_count_by_role") if isinstance(manifest, dict) else {}
    sample_count_by_role = sample_count_by_role if isinstance(sample_count_by_role, dict) else {}
    feature_shape = _feature_shape(payload)
    universe_size = _nested_int(prepared, "universe_size")
    if not universe_size and len(feature_shape) >= 2:
        universe_size = int(feature_shape[1])
    train_rows = _nested_int(sample_count_by_role, "train") or _first_model_train_rows(payload)
    pool_view_kind = str(manifest.get("source_pool_view_kind", "") or "")
    pool_view_name = str(manifest.get("source_pool_view_name", "") or "")
    pool_view_id = str(manifest.get("source_pool_view_id", "") or "")

    is_full_pool = (
        pool_view_kind == FULL_ROLLING_LIQUID500_POOL_VIEW_KIND
        and pool_view_name == FULL_ROLLING_LIQUID500_POOL_VIEW_NAME
        and universe_size >= FULL_ROLLING_LIQUID500_MIN_UNIVERSE_SIZE
        and train_rows >= FULL_ROLLING_LIQUID500_MIN_TRAIN_ROWS
    )
    is_cap80 = (
        0 < universe_size <= CAP80_MAX_UNIVERSE_SIZE
        or (len(feature_shape) >= 2 and 0 < int(feature_shape[1]) <= CAP80_MAX_UNIVERSE_SIZE)
        or (0 < train_rows <= CAP80_MAX_TRAIN_ROWS)
    )
    scope = FULL_ROLLING_LIQUID500_SCOPE if is_full_pool else CAP80_DIAGNOSTIC_SCOPE if is_cap80 else PARTIAL_OR_UNKNOWN_SCOPE
    return {
        "universe_scope": scope,
        "universe_size": universe_size,
        "train_rows": train_rows,
        "feature_store_shape": feature_shape,
        "pool_view_id": pool_view_id,
        "pool_view_kind": pool_view_kind,
        "pool_view_name": pool_view_name,
    }


def study_scope_from_summary_path(summary_path: str | Path) -> dict[str, Any]:
    path = Path(summary_path)
    if not path.exists():
        return {"universe_scope": "missing", "summary_path": str(path)}
    payload = json.loads(path.read_text(encoding="utf-8"))
    scope = study_scope_from_summary(payload if isinstance(payload, dict) else {})
    scope["summary_path"] = str(path)
    return scope

