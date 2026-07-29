from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import os
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime
from itertools import pairwise
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy import optimize, special, stats
from sklearn.metrics import average_precision_score

from daily_research.path_policy import seq100_entry_contract_oos as entry_contract
from daily_research.path_policy import seq100_mfe_feature_family_audit as feature
from daily_research.path_policy import seq100_path_label_learnability as base
from daily_research.path_policy.seq100_post_entry_incremental_information import (
    HoldingPathReader,
)

WORKSPACE_ROOT = Path(__file__).resolve().parents[2]
STUDY_ID = "seq100_post_entry_ab_v1"
DEFAULT_STUDY_PATH = (
    WORKSPACE_ROOT / "daily_research/studies/seq100_post_entry_ab_v1.json"
)
DEFAULT_OUTPUT_ROOT = (
    WORKSPACE_ROOT / "daily_research/output/path_policy/studies/seq100_post_entry_ab_v1"
)
DEFAULT_RECORD_ROOT = (
    WORKSPACE_ROOT / "daily_research/research_records/seq100/seq100_post_entry_ab_v1"
)
AGES = (1, 3, 5)
TEST_YEARS = (2023, 2024, 2025)
TASK_SCHEMA = "seq100_post_entry_ab_task/v1"
LANDMARK_SCHEMA = "seq100_post_entry_landmark/v1"
SUMMARY_SCHEMA = "seq100_post_entry_ab_summary/v1"


def _now() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def _resolve(value: str | Path) -> Path:
    path = Path(value)
    return (
        (WORKSPACE_ROOT / path).resolve() if not path.is_absolute() else path.resolve()
    )


def _json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise TypeError(f"unsupported JSON value: {type(value).__name__}")


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, default=_json_default) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _save_npy(path: Path, values: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as handle:
        np.save(handle, np.asarray(values), allow_pickle=False)
    os.replace(temporary, path)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _file_record(path: Path, **extra: Any) -> dict[str, Any]:
    resolved = path.resolve()
    try:
        display = resolved.relative_to(WORKSPACE_ROOT).as_posix()
    except ValueError:
        display = str(resolved)
    return {
        "path": display,
        "size": int(resolved.stat().st_size),
        "sha256": _sha256(resolved),
        **extra,
    }


def _emit(event: str, **payload: Any) -> None:
    print(
        json.dumps(
            {"event": event, "at": _now(), **payload},
            ensure_ascii=False,
            default=_json_default,
        ),
        flush=True,
    )


def _config_hash(path: Path) -> str:
    return _sha256(path.resolve())


def load_study(path: Path = DEFAULT_STUDY_PATH) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("study_id") != STUDY_ID:
        raise ValueError(f"study_id must be {STUDY_ID}")
    folds = dict(payload["folds"])
    if tuple(int(value) for value in folds["test_years"]) != TEST_YEARS:
        raise ValueError("post-entry test years changed")
    if folds["maximum_outcome_date"] != "2025-12-31":
        raise ValueError("post-entry study may not read outcomes after 2025")
    if int(folds["forbidden_outcome_year"]) != 2026:
        raise ValueError("2026 must remain forbidden")
    if tuple(int(value) for value in payload["landmarks"]["ages"]) != AGES:
        raise ValueError("landmark ages changed")
    dimensions = dict(payload["features"]["dimension_count"])
    if dimensions != {"A": 6, "B_D1": 16, "B_D3": 24, "B_D5": 24}:
        raise ValueError("A/B dimensions changed")
    forbidden = set(payload["features"]["D1_forbidden_duplicates"])
    if forbidden & set(payload["features"]["D1_path"]):
        raise ValueError("D1 includes a structurally duplicate feature")
    counts = dict(payload["model"]["formal_booster_count"])
    if counts != {"risk": 27, "state": 36, "total": 63}:
        raise ValueError("formal booster count changed")
    return payload


@dataclass(frozen=True)
class FrozenEntryContract:
    rows: np.ndarray
    raw: np.ndarray
    rank: np.ndarray
    manifest: dict[str, Any]


def _verify_record(record: Mapping[str, Any]) -> Path:
    path = _resolve(record["path"])
    if not path.is_file() or path.stat().st_size != int(record["size"]):
        raise ValueError(f"contract file is absent or changed size: {path}")
    if _sha256(path) != record["sha256"]:
        raise ValueError(f"contract file hash changed: {path}")
    return path


def load_entry_contract(study: Mapping[str, Any]) -> FrozenEntryContract:
    manifest_path = _resolve(study["sources"]["entry_contract_manifest"])
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("schema") != entry_contract.MANIFEST_SCHEMA:
        raise ValueError("strict entry-contract schema changed")
    if manifest.get("status") != "completed_candidate_aligned_strict_oos_contract":
        raise ValueError("strict entry contract is incomplete")
    if manifest["scope"]["maximum_consumed_outcome_date"] != "2025-12-31":
        raise ValueError("strict entry contract crossed the outcome boundary")
    if int(manifest["scope"]["forbidden_2026_row_count"]) != 0:
        raise ValueError("strict entry contract contains a 2026 row")
    if tuple(manifest["physical_columns"]) != entry_contract.PHYSICAL_COLUMNS:
        raise ValueError("strict entry-contract columns changed")
    files = dict(manifest["files"])
    rows = np.load(_verify_record(files["candidate_rows"]), mmap_mode="r")
    raw = np.load(_verify_record(files["raw_predictions"]), mmap_mode="r")
    rank = np.load(_verify_record(files["date_rank_predictions"]), mmap_mode="r")
    if raw.shape != rank.shape or raw.shape != (
        len(rows),
        len(entry_contract.PHYSICAL_COLUMNS),
    ):
        raise ValueError("strict entry-contract matrix shape changed")
    return FrozenEntryContract(
        rows=np.asarray(rows),
        raw=np.asarray(raw),
        rank=np.asarray(rank),
        manifest=manifest,
    )


def _candidate_lookup(
    candidate_keys: np.ndarray, requested_keys: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    source = np.asarray(candidate_keys, dtype=np.int64)
    requested = np.asarray(requested_keys, dtype=np.int64)
    if source.ndim != 1 or requested.ndim != 1 or not source.size:
        raise ValueError("candidate lookup requires non-empty one-dimensional keys")
    if bool(np.any(source[1:] <= source[:-1])):
        raise ValueError("candidate keys must be unique and ordered")
    positions = np.searchsorted(source, requested)
    safe = np.minimum(positions, len(source) - 1)
    found = (positions < len(source)) & (source[safe] == requested)
    return np.where(found, safe, -1).astype(np.int64), found


def _feature_names(study: Mapping[str, Any], age: int, variant: str) -> tuple[str, ...]:
    features = dict(study["features"])
    baseline = tuple(str(value) for value in features["A"])
    if variant == "A":
        return baseline
    if variant != "B":
        raise ValueError(f"unknown variant: {variant}")
    path = features["D1_path"] if int(age) == 1 else features["D3_D5_path"]
    result = (
        *baseline,
        *(str(value) for value in features["rank_change"]),
        *(str(value) for value in path),
    )
    expected = int(features["dimension_count"][f"B_D{int(age)}"])
    if len(result) != expected or len(result) != len(set(result)):
        raise ValueError(f"B-D{age} feature contract is not unique or has wrong size")
    return tuple(result)


def _full_feature_names(study: Mapping[str, Any]) -> tuple[str, ...]:
    return _feature_names(study, 3, "B")


def _landmark_root(output_root: Path, age: int) -> Path:
    return output_root / "landmarks" / f"age_{int(age):02d}"


def _open_npy_memmap(
    path: Path, *, dtype: Any, shape: tuple[int, ...], mode: str
) -> np.memmap:
    return np.lib.format.open_memmap(path, mode=mode, dtype=dtype, shape=shape)


def _landmark_complete(
    *, path: Path, study_hash: str, contract_manifest: Mapping[str, Any], age: int
) -> bool:
    if not path.is_file():
        return False
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        if (
            payload.get("schema") != LANDMARK_SCHEMA
            or payload.get("status") != "completed"
            or payload.get("study_config_sha256") != study_hash
            or int(payload.get("age", -1)) != int(age)
            or payload.get("entry_contract_sha256")
            != contract_manifest["files"]["raw_predictions"]["sha256"]
            or int(payload.get("state_atlas_cutoff", -1)) != 2022
        ):
            return False
        _verify_record(payload["state_atlas_model"])
        for record in payload["files"].values():
            _verify_record(record)
        return True
    except (
        KeyError,
        OSError,
        TypeError,
        ValueError,
        json.JSONDecodeError,
    ):
        return False


def _initialize_landmark_files(
    *,
    root: Path,
    count: int,
    feature_count: int,
    progress: Mapping[str, Any] | None,
) -> tuple[dict[str, np.memmap], dict[str, Path]]:
    root.mkdir(parents=True, exist_ok=True)
    paths = {
        "features": root / "features.npy",
        "entry_rows": root / "entry_rows.npy",
        "current_rows": root / "current_rows.npy",
        "decision_date_idx": root / "decision_date_idx.npy",
        "entry_either_top5": root / "entry_either_top5.npy",
        "target_risk": root / "target_risk.npy",
        "target_state": root / "target_state.npy",
        "tradable_day_fraction": root / "tradable_day_fraction.npy",
    }
    mode = "r+" if progress is not None else "w+"
    arrays = {
        "features": _open_npy_memmap(
            paths["features"],
            dtype=np.float32,
            shape=(count, feature_count),
            mode=mode,
        ),
        "entry_rows": _open_npy_memmap(
            paths["entry_rows"], dtype=np.int64, shape=(count,), mode=mode
        ),
        "current_rows": _open_npy_memmap(
            paths["current_rows"], dtype=np.int64, shape=(count,), mode=mode
        ),
        "decision_date_idx": _open_npy_memmap(
            paths["decision_date_idx"], dtype=np.int32, shape=(count,), mode=mode
        ),
        "entry_either_top5": _open_npy_memmap(
            paths["entry_either_top5"], dtype=bool, shape=(count,), mode=mode
        ),
        "target_risk": _open_npy_memmap(
            paths["target_risk"], dtype=np.float32, shape=(count,), mode=mode
        ),
        "target_state": _open_npy_memmap(
            paths["target_state"], dtype=np.int8, shape=(count,), mode=mode
        ),
        "tradable_day_fraction": _open_npy_memmap(
            paths["tradable_day_fraction"],
            dtype=np.float32,
            shape=(count,),
            mode=mode,
        ),
    }
    if mode == "w+":
        arrays["features"][:] = np.nan
        arrays["entry_rows"][:] = -1
        arrays["current_rows"][:] = -1
        arrays["decision_date_idx"][:] = -1
        arrays["entry_either_top5"][:] = False
        arrays["target_risk"][:] = np.nan
        arrays["target_state"][:] = -1
        arrays["tradable_day_fraction"][:] = np.nan
        for array in arrays.values():
            array.flush()
    return arrays, paths


def build_landmark(
    *,
    age: int,
    study: Mapping[str, Any],
    study_hash: str,
    output_root: Path,
    inputs: base.LearnabilityInputs,
    contract: FrozenEntryContract,
    reader: HoldingPathReader,
) -> dict[str, Any]:
    root = _landmark_root(output_root, age)
    complete_path = root / "complete.json"
    if _landmark_complete(
        path=complete_path,
        study_hash=study_hash,
        contract_manifest=contract.manifest,
        age=age,
    ):
        return json.loads(complete_path.read_text(encoding="utf-8"))

    contract_rows = np.asarray(contract.rows, dtype=np.int64)
    original_mask = np.asarray(inputs.entry_fill[contract_rows]) == 1
    original_dates_all = np.asarray(
        inputs.candidate_date_idx[contract_rows], dtype=np.int32
    )
    decision_dates_all = original_dates_all + int(age)
    original_mask &= decision_dates_all <= reader.cutoff_idx
    entry_positions = np.flatnonzero(original_mask).astype(np.int64, copy=False)
    count = len(entry_positions)
    if not count:
        raise ValueError(f"D{age} landmark cohort is empty")
    entry_rows = contract_rows[entry_positions]
    original_dates = original_dates_all[entry_positions]
    symbols = np.asarray(inputs.candidate_symbol_idx[entry_rows], dtype=np.int32)
    decision_dates = original_dates + int(age)
    if bool(np.any(decision_dates[1:] < decision_dates[:-1])):
        raise AssertionError("landmark decision dates are not ordered")
    if bool(np.any(np.asarray(inputs.entry_fill[entry_rows]) != 1)):
        raise AssertionError("landmark cohort contains an unfilled original entry")

    contract_keys = np.asarray(
        inputs.candidate_date_idx[contract_rows], dtype=np.int64
    ) * reader.symbol_count + np.asarray(
        inputs.candidate_symbol_idx[contract_rows], dtype=np.int64
    )
    if bool(np.any(contract_keys[1:] <= contract_keys[:-1])):
        raise ValueError("strict contract date-symbol keys are not unique and ordered")

    feature_names = _full_feature_names(study)
    progress_path = root / "progress.json"
    progress: dict[str, Any] | None = None
    if progress_path.is_file():
        progress = json.loads(progress_path.read_text(encoding="utf-8"))
        if (
            progress.get("study_config_sha256") != study_hash
            or progress.get("entry_contract_sha256")
            != contract.manifest["files"]["raw_predictions"]["sha256"]
            or int(progress.get("age", -1)) != int(age)
            or int(progress.get("row_count", -1)) != count
            or tuple(progress.get("feature_names", ())) != feature_names
        ):
            raise ValueError("landmark recovery metadata differs from current contract")
    arrays, paths = _initialize_landmark_files(
        root=root,
        count=count,
        feature_count=len(feature_names),
        progress=progress,
    )
    next_offset = int(progress.get("next_offset", 0)) if progress else 0
    if next_offset < 0 or next_offset > count:
        raise ValueError("landmark recovery offset is invalid")

    physical = {
        name: index for index, name in enumerate(entry_contract.PHYSICAL_COLUMNS)
    }
    a_sources = {
        "current_mfe10_rank": "mfe_10",
        "current_mfe20_rank": "mfe_20",
        "current_state_low_rank": "state_low_10",
        "current_state_high_rank": "state_high_10",
        "current_risk10_rank": "pre_peak_mae_10",
        "current_risk20_rank": "pre_peak_mae_20",
    }
    change_sources = {
        "mfe10_rank_change": "mfe_10",
        "mfe20_rank_change": "mfe_20",
        "state_low_rank_change": "state_low_10",
        "state_high_rank_change": "state_high_10",
        "risk10_rank_change": "pre_peak_mae_10",
        "risk20_rank_change": "pre_peak_mae_20",
    }
    feature_position = {name: index for index, name in enumerate(feature_names)}
    started = time.perf_counter()
    batch_size = 100_000
    for left in range(next_offset, count, batch_size):
        right = min(left + batch_size, count)
        current_entry_positions = entry_positions[left:right]
        current_entry_rows = entry_rows[left:right]
        current_original_dates = original_dates[left:right]
        current_symbols = symbols[left:right]
        current_decision_dates = decision_dates[left:right]
        requested_keys = current_decision_dates.astype(
            np.int64
        ) * reader.symbol_count + current_symbols.astype(np.int64)
        current_positions, current_found = _candidate_lookup(
            contract_keys, requested_keys
        )
        safe_positions = np.where(current_found, current_positions, 0)
        current_rows = np.where(
            current_found, contract_rows[safe_positions], -1
        ).astype(np.int64)
        current_rank = np.asarray(contract.rank[safe_positions], dtype=np.float32)
        current_rank[~current_found] = np.nan
        entry_rank = np.asarray(
            contract.rank[current_entry_positions], dtype=np.float32
        )
        realized, quality = reader.realized_features(
            current_original_dates, current_symbols, age
        )
        targets, _target_flags = reader.holding_targets(
            current_decision_dates, current_symbols
        )
        block = np.full((right - left, len(feature_names)), np.nan, dtype=np.float32)
        for name, source in a_sources.items():
            block[:, feature_position[name]] = current_rank[:, physical[source]]
        for name, source in change_sources.items():
            column = physical[source]
            block[:, feature_position[name]] = (
                current_rank[:, column] - entry_rank[:, column]
            )
        for name in study["features"]["D3_D5_path"]:
            block[:, feature_position[str(name)]] = np.asarray(
                realized[str(name)], dtype=np.float32
            )
        arrays["features"][left:right] = block
        arrays["entry_rows"][left:right] = current_entry_rows
        arrays["current_rows"][left:right] = current_rows
        arrays["decision_date_idx"][left:right] = current_decision_dates
        arrays["entry_either_top5"][left:right] = (
            entry_rank[:, physical["mfe_10"]] >= 0.95
        ) | (entry_rank[:, physical["mfe_20"]] >= 0.95)
        arrays["target_risk"][left:right] = np.asarray(
            targets["pre_peak_mae_10"], dtype=np.float32
        )
        state = np.asarray(targets["state_10"], dtype=np.float32)
        arrays["target_state"][left:right] = np.where(
            np.isin(state, (0.0, 1.0, 2.0)), state, -1
        ).astype(np.int8)
        arrays["tradable_day_fraction"][left:right] = np.asarray(
            realized["tradable_day_fraction"], dtype=np.float32
        )
        if not np.array_equal(
            np.asarray(quality["realized_path_valid"], dtype=bool),
            np.isfinite(realized["entry_to_close_return"]),
        ):
            raise AssertionError("realized-path validity differs from feature validity")
        for array in arrays.values():
            array.flush()
        progress = {
            "status": "running",
            "study_config_sha256": study_hash,
            "entry_contract_sha256": contract.manifest["files"]["raw_predictions"][
                "sha256"
            ],
            "age": int(age),
            "row_count": count,
            "feature_names": list(feature_names),
            "next_offset": right,
            "elapsed_seconds_this_run": float(time.perf_counter() - started),
            "updated_at": _now(),
        }
        _write_json(progress_path, progress)
        _emit(
            "landmark_progress",
            age=age,
            completed_rows=right,
            total_rows=count,
        )

    for array in arrays.values():
        array.flush()
    selected_b_names = _feature_names(study, age, "B")
    selected_columns = np.asarray(
        [feature_position[name] for name in selected_b_names], dtype=np.int64
    )
    feature_matrix = np.asarray(arrays["features"])
    selected_complete = np.isfinite(feature_matrix[:, selected_columns]).all(axis=1)
    risk_valid = np.isfinite(np.asarray(arrays["target_risk"]))
    state_valid = np.isin(np.asarray(arrays["target_state"]), (0, 1, 2))
    current_found = np.asarray(arrays["current_rows"]) >= 0
    decision_years = np.asarray(
        [
            int(str(inputs.date_values[int(date_idx)])[:4])
            for date_idx in np.asarray(arrays["decision_date_idx"])
        ],
        dtype=np.int16,
    )
    if bool(np.any(decision_years == 2026)):
        raise AssertionError("a 2026 landmark row reached the dataset")
    entry_top5 = np.asarray(arrays["entry_either_top5"], dtype=bool)
    quality = {
        "cohort_count": count,
        "current_contract_found_count": int(current_found.sum()),
        "selected_B_feature_complete_count": int(selected_complete.sum()),
        "risk_target_valid_count": int(risk_valid.sum()),
        "state_target_valid_count": int(state_valid.sum()),
        "risk_common_support_count": int(
            (selected_complete & risk_valid & current_found).sum()
        ),
        "state_common_support_count": int(
            (selected_complete & state_valid & current_found).sum()
        ),
        "entry_either_top5_count": int(entry_top5.sum()),
        "tradability_alert_count": int(
            np.sum(
                np.isfinite(arrays["tradable_day_fraction"])
                & (arrays["tradable_day_fraction"] < 1.0)
            )
        ),
        "decision_year_counts": {
            str(year): int(np.sum(decision_years == year)) for year in range(2020, 2026)
        },
        "maximum_observation_date": str(
            inputs.date_values[int(np.max(arrays["decision_date_idx"]))]
        ),
        "maximum_outcome_date_read": "2025-12-31",
        "forbidden_2026_row_count": 0,
    }
    files = {
        name: _file_record(
            path,
            shape=list(arrays[name].shape),
            dtype=str(arrays[name].dtype),
        )
        for name, path in paths.items()
    }
    state_atlas_cutoff = int(np.asarray(reader.state10_model["window_year"])[0])
    if state_atlas_cutoff != 2022:
        raise AssertionError(
            "post-entry state target does not use the frozen 2022 atlas"
        )
    state_atlas_source = dict(inputs.label_manifest["state_models"]["10"])
    state_atlas_model = _file_record(_resolve(state_atlas_source["path"]))
    completed = {
        "schema": LANDMARK_SCHEMA,
        "status": "completed",
        "completed_at": _now(),
        "study_id": STUDY_ID,
        "study_config_sha256": study_hash,
        "entry_contract_sha256": contract.manifest["files"]["raw_predictions"][
            "sha256"
        ],
        "age": int(age),
        "row_count": count,
        "feature_names": list(feature_names),
        "A_feature_names": list(_feature_names(study, age, "A")),
        "B_feature_names": list(selected_b_names),
        "state_atlas_cutoff": state_atlas_cutoff,
        "state_atlas_model": state_atlas_model,
        "target_anchor": study["landmarks"]["target_anchor"],
        "quality": quality,
        "files": files,
    }
    _write_json(complete_path, completed)
    _write_json(
        progress_path,
        {
            **(progress or {}),
            "status": "completed",
            "next_offset": count,
            "updated_at": _now(),
        },
    )
    del feature_matrix
    gc.collect()
    _emit(
        "landmark_completed",
        age=age,
        cohort_count=count,
        risk_common_support=quality["risk_common_support_count"],
        state_common_support=quality["state_common_support_count"],
    )
    return completed


@dataclass(frozen=True)
class LandmarkData:
    age: int
    feature_names: tuple[str, ...]
    features: np.ndarray
    entry_rows: np.ndarray
    current_rows: np.ndarray
    decision_date_idx: np.ndarray
    entry_either_top5: np.ndarray
    target_risk: np.ndarray
    target_state: np.ndarray
    tradable_day_fraction: np.ndarray
    manifest: dict[str, Any]


def load_landmark(
    *,
    age: int,
    study: Mapping[str, Any],
    study_hash: str,
    output_root: Path,
    contract: FrozenEntryContract,
) -> LandmarkData:
    complete_path = _landmark_root(output_root, age) / "complete.json"
    if not _landmark_complete(
        path=complete_path,
        study_hash=study_hash,
        contract_manifest=contract.manifest,
        age=age,
    ):
        raise ValueError(f"D{age} landmark dataset is incomplete")
    manifest = json.loads(complete_path.read_text(encoding="utf-8"))
    arrays = {
        name: np.load(_verify_record(record), mmap_mode="r")
        for name, record in manifest["files"].items()
    }
    if tuple(manifest["feature_names"]) != _full_feature_names(study):
        raise ValueError("landmark feature order changed")
    return LandmarkData(
        age=age,
        feature_names=tuple(manifest["feature_names"]),
        features=arrays["features"],
        entry_rows=arrays["entry_rows"],
        current_rows=arrays["current_rows"],
        decision_date_idx=arrays["decision_date_idx"],
        entry_either_top5=arrays["entry_either_top5"],
        target_risk=arrays["target_risk"],
        target_state=arrays["target_state"],
        tradable_day_fraction=arrays["tradable_day_fraction"],
        manifest=manifest,
    )


def _model_parameters(
    study: Mapping[str, Any], target: str
) -> tuple[dict[str, Any], int, int]:
    model = dict(study["model"])
    parameters: dict[str, Any] = {
        "boosting_type": "gbdt",
        "device_type": "cpu",
        "learning_rate": float(model["learning_rate"]),
        "num_leaves": int(model["num_leaves"]),
        "max_depth": int(model["max_depth"]),
        "min_data_in_leaf": int(model["min_data_in_leaf"]),
        "feature_fraction": float(model["feature_fraction"]),
        "bagging_fraction": float(model["bagging_fraction"]),
        "bagging_freq": int(model["bagging_freq"]),
        "lambda_l1": float(model["lambda_l1"]),
        "lambda_l2": float(model["lambda_l2"]),
        "max_bin": int(model["max_bin"]),
        "deterministic": bool(model["deterministic"]),
        "force_col_wise": bool(model["force_col_wise"]),
        "num_threads": int(model["num_threads"]),
        "histogram_pool_size": int(model["histogram_pool_size_mb"]),
        "seed": int(model["seed"]),
        "feature_fraction_seed": int(model["seed"]),
        "bagging_seed": int(model["seed"]),
        "data_random_seed": int(model["seed"]),
        "drop_seed": int(model["seed"]),
        "extra_seed": int(model["seed"]),
        "feature_pre_filter": False,
        "verbosity": -1,
    }
    if target == "risk":
        parameters.update(
            {
                "objective": "huber",
                "metric": "huber",
                "alpha": float(model["huber_alpha"]),
            }
        )
    elif target == "state":
        parameters.update(
            {
                "objective": "multiclass",
                "metric": "multi_logloss",
                "num_class": 3,
            }
        )
    else:
        raise ValueError(f"unknown post-entry target: {target}")
    return (
        parameters,
        int(model["num_boost_round"]),
        int(model["early_stopping_rounds"]),
    )


def _tasks() -> list[dict[str, Any]]:
    tasks: list[dict[str, Any]] = []
    for age in AGES:
        for year in TEST_YEARS:
            tasks.extend(
                [
                    {
                        "task_id": f"risk_D{age}_{year}_inner_A",
                        "target": "risk",
                        "age": age,
                        "year": year,
                        "stage": "inner",
                        "variant": "A",
                        "selects_iterations": True,
                    },
                    {
                        "task_id": f"risk_D{age}_{year}_outer_A",
                        "target": "risk",
                        "age": age,
                        "year": year,
                        "stage": "outer",
                        "variant": "A",
                        "selects_iterations": False,
                    },
                    {
                        "task_id": f"risk_D{age}_{year}_outer_B",
                        "target": "risk",
                        "age": age,
                        "year": year,
                        "stage": "outer",
                        "variant": "B",
                        "selects_iterations": False,
                    },
                    {
                        "task_id": f"state_D{age}_{year}_inner_A",
                        "target": "state",
                        "age": age,
                        "year": year,
                        "stage": "inner",
                        "variant": "A",
                        "selects_iterations": True,
                    },
                    {
                        "task_id": f"state_D{age}_{year}_inner_B",
                        "target": "state",
                        "age": age,
                        "year": year,
                        "stage": "inner",
                        "variant": "B",
                        "selects_iterations": False,
                    },
                    {
                        "task_id": f"state_D{age}_{year}_outer_A",
                        "target": "state",
                        "age": age,
                        "year": year,
                        "stage": "outer",
                        "variant": "A",
                        "selects_iterations": False,
                    },
                    {
                        "task_id": f"state_D{age}_{year}_outer_B",
                        "target": "state",
                        "age": age,
                        "year": year,
                        "stage": "outer",
                        "variant": "B",
                        "selects_iterations": False,
                    },
                ]
            )
    return tasks


def _task_dir(output_root: Path, task: Mapping[str, Any]) -> Path:
    return output_root / "tasks" / str(task["task_id"])


def _task_result_path(output_root: Path, task: Mapping[str, Any]) -> Path:
    return _task_dir(output_root, task) / "task_result.json"


def _inner_a_task_id(target: str, age: int, year: int) -> str:
    return f"{target}_D{int(age)}_{int(year)}_inner_A"


def _task_by_id(task_id: str) -> dict[str, Any]:
    for task in _tasks():
        if task["task_id"] == task_id:
            return task
    raise KeyError(task_id)


def _year_start(date_values: np.ndarray, year: int) -> int:
    rows = np.flatnonzero(
        np.char.startswith(np.asarray(date_values, dtype=str), f"{int(year)}-")
    )
    if not rows.size:
        raise ValueError(f"calendar lacks year {year}")
    return int(rows[0])


def _split_rows(
    *,
    data: LandmarkData,
    study: Mapping[str, Any],
    target: str,
    test_year: int,
    stage: str,
    date_values: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    b_names = _feature_names(study, data.age, "B")
    position = {name: index for index, name in enumerate(data.feature_names)}
    columns = np.asarray([position[name] for name in b_names], dtype=np.int64)
    complete = np.isfinite(np.asarray(data.features[:, columns])).all(axis=1)
    if target == "risk":
        complete &= np.isfinite(np.asarray(data.target_risk))
    elif target == "state":
        complete &= np.isin(np.asarray(data.target_state), (0, 1, 2))
    else:
        raise ValueError(target)
    dates = np.asarray(data.decision_date_idx, dtype=np.int32)
    evaluation_year = int(test_year) - 1 if stage == "inner" else int(test_year)
    evaluation_start = _year_start(date_values, evaluation_year)
    evaluation_stop = (
        _year_start(date_values, evaluation_year + 1)
        if evaluation_year < 2025
        else len(date_values)
    )
    dependency_days = int(study["targets"]["dependency_days"])
    train = complete & (dates + dependency_days < evaluation_start)
    evaluation = complete & (dates >= evaluation_start) & (dates < evaluation_stop)
    train_rows = np.flatnonzero(train).astype(np.int64, copy=False)
    evaluation_rows = np.flatnonzero(evaluation).astype(np.int64, copy=False)
    if not train_rows.size or not evaluation_rows.size:
        raise ValueError(f"empty {stage} split for {target} D{data.age} {test_year}")
    if int(dates[train_rows[-1]]) + dependency_days >= evaluation_start:
        raise AssertionError("post-entry target purge failed")
    return train_rows, evaluation_rows


class _MatrixSequence:
    def __new__(
        cls,
        *,
        matrix: np.ndarray,
        rows: np.ndarray,
        columns: np.ndarray,
        batch_size: int,
    ) -> Any:
        import lightgbm as lgb

        source = matrix
        row_ids = np.asarray(rows, dtype=np.int64)
        column_ids = np.asarray(columns, dtype=np.int64)

        class MatrixSequence(lgb.Sequence):
            def __init__(self) -> None:
                self.batch_size = int(batch_size)

            def __len__(self) -> int:
                return len(row_ids)

            def _block(self, local: np.ndarray) -> np.ndarray:
                return np.asarray(
                    source[np.ix_(row_ids[local], column_ids)], dtype=np.float64
                )

            def __getitem__(self, index: Any) -> np.ndarray:
                if isinstance(index, slice):
                    local = np.arange(
                        0 if index.start is None else int(index.start),
                        len(row_ids) if index.stop is None else int(index.stop),
                        1 if index.step is None else int(index.step),
                        dtype=np.int64,
                    )
                    return self._block(local)
                if isinstance(index, (list, tuple, np.ndarray)):
                    return self._block(np.asarray(index, dtype=np.int64))
                if isinstance(index, (int, np.integer)):
                    return self._block(np.asarray([int(index)], dtype=np.int64))[0]
                raise TypeError(type(index).__name__)

        return MatrixSequence()


@dataclass
class _Datasets:
    train_rows: np.ndarray
    evaluation_rows: np.ndarray
    train_sequence: Any
    evaluation_sequence: Any
    train_set: Any
    evaluation_set: Any
    feature_names: tuple[str, ...]


def _build_datasets(
    *,
    data: LandmarkData,
    study: Mapping[str, Any],
    target: str,
    variant: str,
    test_year: int,
    stage: str,
    date_values: np.ndarray,
) -> _Datasets:
    import lightgbm as lgb

    train_rows, evaluation_rows = _split_rows(
        data=data,
        study=study,
        target=target,
        test_year=test_year,
        stage=stage,
        date_values=date_values,
    )
    names = _feature_names(study, data.age, variant)
    name_to_position = {name: index for index, name in enumerate(data.feature_names)}
    columns = np.asarray([name_to_position[name] for name in names], dtype=np.int64)
    batch_size = int(study["model"]["sequence_batch_size"])
    train_sequence = _MatrixSequence(
        matrix=data.features,
        rows=train_rows,
        columns=columns,
        batch_size=batch_size,
    )
    evaluation_sequence = _MatrixSequence(
        matrix=data.features,
        rows=evaluation_rows,
        columns=columns,
        batch_size=batch_size,
    )
    values = data.target_risk if target == "risk" else data.target_state
    train_label = np.asarray(values[train_rows])
    evaluation_label = np.asarray(values[evaluation_rows])
    train_weight = base.date_equal_weights(data.decision_date_idx[train_rows])
    evaluation_weight = base.date_equal_weights(data.decision_date_idx[evaluation_rows])
    construction = {
        "max_bin": int(study["model"]["max_bin"]),
        "data_random_seed": int(study["model"]["seed"]),
        "feature_pre_filter": False,
        "verbosity": -1,
    }
    train_set = lgb.Dataset(
        train_sequence,
        label=train_label,
        weight=train_weight,
        feature_name=list(names),
        free_raw_data=True,
        params=construction,
    )
    evaluation_set = lgb.Dataset(
        evaluation_sequence,
        label=evaluation_label,
        weight=evaluation_weight,
        feature_name=list(names),
        reference=train_set,
        free_raw_data=True,
        params=construction,
    )
    train_set.construct()
    evaluation_set.construct()
    return _Datasets(
        train_rows=train_rows,
        evaluation_rows=evaluation_rows,
        train_sequence=train_sequence,
        evaluation_sequence=evaluation_sequence,
        train_set=train_set,
        evaluation_set=evaluation_set,
        feature_names=names,
    )


def _release_datasets(datasets: _Datasets) -> None:
    datasets.train_set = None
    datasets.evaluation_set = None
    datasets.train_sequence = None
    datasets.evaluation_sequence = None
    gc.collect()
    try:
        from daily_research.path_policy import seq100_signal_quality
    except ImportError:
        seq100_signal_quality = None
    if seq100_signal_quality is not None:
        seq100_signal_quality.sequence_training._trim_working_set()


def _predict(model: Any, sequence: Any, iterations: int) -> np.ndarray:
    first = np.asarray(model.predict(sequence[0:1], num_iteration=iterations))
    shape = (len(sequence),) if first.ndim == 1 else (len(sequence), first.shape[1])
    output = np.empty(shape, dtype=np.float32)
    for left in range(0, len(sequence), 250_000):
        right = min(left + 250_000, len(sequence))
        output[left:right] = np.asarray(
            model.predict(sequence[left:right], num_iteration=iterations),
            dtype=np.float32,
        )
    return output


def apply_temperature(probability: np.ndarray, temperature: float) -> np.ndarray:
    values = np.asarray(probability, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] != 3:
        raise ValueError("temperature calibration requires three-class probabilities")
    if not math.isfinite(float(temperature)) or float(temperature) <= 0.0:
        raise ValueError("temperature must be finite and positive")
    logits = np.log(np.clip(values, 1.0e-12, 1.0)) / float(temperature)
    calibrated = special.softmax(logits, axis=1)
    return calibrated.astype(np.float32)


def fit_temperature(
    *,
    probability: np.ndarray,
    actual: np.ndarray,
    date_idx: np.ndarray,
) -> dict[str, Any]:
    values = np.asarray(probability, dtype=np.float64)
    target = np.asarray(actual, dtype=np.int64)
    dates = np.asarray(date_idx, dtype=np.int32)
    valid = (
        np.isfinite(values).all(axis=1)
        & np.isin(target, (0, 1, 2))
        & np.isfinite(values.sum(axis=1))
    )
    if int(valid.sum()) < 100:
        raise ValueError("temperature calibration has too few validation rows")
    values = values[valid]
    target = target[valid]
    dates = dates[valid]
    weights = base.date_equal_weights(dates).astype(np.float64)
    weights /= weights.sum()

    def objective(log_temperature: float) -> float:
        temperature = math.exp(float(log_temperature))
        calibrated = apply_temperature(values, temperature).astype(np.float64)
        loss = -np.log(
            np.clip(calibrated[np.arange(len(target)), target], 1.0e-12, 1.0)
        )
        return float(np.dot(weights, loss))

    result = optimize.minimize_scalar(
        objective,
        method="bounded",
        bounds=(math.log(0.05), math.log(20.0)),
        options={"xatol": 1.0e-5, "maxiter": 200},
    )
    if not result.success:
        raise RuntimeError(f"temperature optimization failed: {result.message}")
    temperature = math.exp(float(result.x))
    return {
        "temperature": temperature,
        "validation_row_count": int(valid.sum()),
        "weighted_logloss_before": objective(0.0),
        "weighted_logloss_after": objective(float(result.x)),
        "fit_year_start_date_idx": int(dates.min()),
        "fit_year_end_date_idx": int(dates.max()),
        "optimization_success": True,
    }


def _safe_spearman(left: np.ndarray, right: np.ndarray) -> float:
    x = np.asarray(left, dtype=np.float64)
    y = np.asarray(right, dtype=np.float64)
    valid = np.isfinite(x) & np.isfinite(y)
    if (
        int(valid.sum()) < 3
        or np.std(x[valid]) <= 1.0e-12
        or np.std(y[valid]) <= 1.0e-12
    ):
        return math.nan
    return float(stats.spearmanr(x[valid], y[valid]).statistic)


def _ece(probability: np.ndarray, actual: np.ndarray, bins: int = 10) -> float:
    values = np.asarray(probability, dtype=np.float64)
    target = np.asarray(actual, dtype=np.int64)
    confidence = values.max(axis=1)
    predicted = values.argmax(axis=1)
    correct = predicted == target
    boundaries = np.linspace(0.0, 1.0, int(bins) + 1)
    total = len(values)
    error = 0.0
    for index in range(int(bins)):
        left = boundaries[index]
        right = boundaries[index + 1]
        selected = (
            (confidence >= left) & (confidence < right)
            if index < int(bins) - 1
            else (confidence >= left) & (confidence <= right)
        )
        if bool(selected.any()):
            error += (
                float(selected.sum())
                / total
                * abs(
                    float(correct[selected].mean()) - float(confidence[selected].mean())
                )
            )
    return float(error)


def risk_daily_metrics(
    *,
    date_idx: np.ndarray,
    actual: np.ndarray,
    prediction: np.ndarray,
    entry_either_top5: np.ndarray,
) -> pd.DataFrame:
    dates = np.asarray(date_idx, dtype=np.int32)
    target = np.asarray(actual, dtype=np.float64)
    score = np.asarray(prediction, dtype=np.float64)
    entry_top5 = np.asarray(entry_either_top5, dtype=bool)
    rows: list[dict[str, Any]] = []
    boundaries = np.flatnonzero(np.r_[True, dates[1:] != dates[:-1], True])
    for left, right in pairwise(boundaries):
        valid = np.isfinite(target[left:right]) & np.isfinite(score[left:right])
        if int(valid.sum()) < 20:
            continue
        current_target = target[left:right][valid]
        current_score = score[left:right][valid]
        current_top5 = entry_top5[left:right][valid]
        order = np.argsort(current_target, kind="mergesort")
        event_count = max(1, math.ceil(0.20 * len(order)))
        event = np.zeros(len(order), dtype=np.int8)
        event[order[:event_count]] = 1
        top5_ic = (
            _safe_spearman(current_score[current_top5], current_target[current_top5])
            if int(current_top5.sum()) >= 20
            else math.nan
        )
        rows.append(
            {
                "date_idx": int(dates[left]),
                "row_count": int(valid.sum()),
                "rank_ic": _safe_spearman(current_score, current_target),
                "mae": float(np.mean(np.abs(current_score - current_target))),
                "deep_adverse_pr_auc": float(
                    average_precision_score(event, -current_score)
                ),
                "entry_either_top5_row_count": int(current_top5.sum()),
                "entry_either_top5_rank_ic": top5_ic,
            }
        )
    frame = pd.DataFrame.from_records(rows)
    if frame.empty:
        raise ValueError("risk evaluation produced no valid dates")
    return frame


def state_daily_metrics(
    *,
    date_idx: np.ndarray,
    actual: np.ndarray,
    raw_probability: np.ndarray,
    calibrated_probability: np.ndarray,
) -> pd.DataFrame:
    dates = np.asarray(date_idx, dtype=np.int32)
    target = np.asarray(actual, dtype=np.int64)
    raw = np.asarray(raw_probability, dtype=np.float64)
    calibrated = np.asarray(calibrated_probability, dtype=np.float64)
    rows: list[dict[str, Any]] = []
    boundaries = np.flatnonzero(np.r_[True, dates[1:] != dates[:-1], True])
    for left, right in pairwise(boundaries):
        valid = (
            np.isin(target[left:right], (0, 1, 2))
            & np.isfinite(raw[left:right]).all(axis=1)
            & np.isfinite(calibrated[left:right]).all(axis=1)
        )
        if int(valid.sum()) < 20:
            continue
        y = target[left:right][valid]
        raw_p = raw[left:right][valid]
        calibrated_p = calibrated[left:right][valid]
        one_hot = np.eye(3, dtype=np.float64)[y]
        raw_expected = raw_p[:, 1] + 2.0 * raw_p[:, 2]
        calibrated_expected = calibrated_p[:, 1] + 2.0 * calibrated_p[:, 2]
        index = np.arange(len(y))
        rows.append(
            {
                "date_idx": int(dates[left]),
                "row_count": int(valid.sum()),
                "raw_ordinal_ic": _safe_spearman(raw_expected, y),
                "calibrated_ordinal_ic": _safe_spearman(calibrated_expected, y),
                "raw_brier": float(np.mean(np.square(raw_p - one_hot).sum(axis=1))),
                "calibrated_brier": float(
                    np.mean(np.square(calibrated_p - one_hot).sum(axis=1))
                ),
                "raw_logloss": float(
                    np.mean(-np.log(np.clip(raw_p[index, y], 1.0e-12, 1.0)))
                ),
                "calibrated_logloss": float(
                    np.mean(-np.log(np.clip(calibrated_p[index, y], 1.0e-12, 1.0)))
                ),
                "raw_ece": _ece(raw_p, y),
                "calibrated_ece": _ece(calibrated_p, y),
            }
        )
    frame = pd.DataFrame.from_records(rows)
    if frame.empty:
        raise ValueError("state evaluation produced no valid dates")
    return frame


def _task_complete(
    path: Path,
    *,
    task: Mapping[str, Any],
    study: Mapping[str, Any],
    study_hash: str,
) -> bool:
    if not path.is_file():
        return False
    try:
        import lightgbm as lgb

        payload = json.loads(path.read_text(encoding="utf-8"))
        expected = {
            "schema": TASK_SCHEMA,
            "status": "completed",
            "study_id": STUDY_ID,
            "study_config_sha256": study_hash,
            "task_id": task["task_id"],
            "target": task["target"],
            "age": int(task["age"]),
            "test_year": int(task["year"]),
            "stage": task["stage"],
            "variant": task["variant"],
            "parameters": _model_parameters(study, str(task["target"]))[0],
            "feature_names": list(
                _feature_names(study, int(task["age"]), str(task["variant"]))
            ),
        }
        if any(payload.get(key) != value for key, value in expected.items()):
            return False
        files = dict(payload.get("files", {}) or {})
        for record in files.values():
            _verify_record(record)
        lgb.Booster(model_file=str(_resolve(files["model"]["path"])))
        prediction = np.load(_resolve(files["prediction"]["path"]), allow_pickle=False)
        rows = np.load(_resolve(files["evaluation_rows"]["path"]), allow_pickle=False)
        if list(prediction.shape) != list(files["prediction"]["shape"]):
            return False
        if list(rows.shape) != list(files["evaluation_rows"]["shape"]):
            return False
        pd.read_parquet(_resolve(files["daily_metrics"]["path"]))
        if task["target"] == "state":
            calibration = dict(payload["calibration"])
            if not math.isfinite(float(calibration["temperature"])):
                return False
            expected_fit_year = int(task["year"]) - 1
            if int(calibration["fit_validation_year"]) != expected_fit_year:
                return False
        return True
    except (
        EOFError,
        ImportError,
        KeyError,
        OSError,
        RuntimeError,
        TypeError,
        ValueError,
        json.JSONDecodeError,
    ):
        return False


def _daily_summary(frame: pd.DataFrame) -> dict[str, Any]:
    return {
        column: float(np.nanmean(frame[column].to_numpy(dtype=np.float64)))
        for column in frame.columns
        if column != "date_idx"
    }


def _run_task(
    *,
    task: Mapping[str, Any],
    study: Mapping[str, Any],
    study_hash: str,
    output_root: Path,
    data: LandmarkData,
    date_values: np.ndarray,
) -> dict[str, Any]:
    import lightgbm as lgb

    result_path = _task_result_path(output_root, task)
    if _task_complete(result_path, task=task, study=study, study_hash=study_hash):
        return json.loads(result_path.read_text(encoding="utf-8"))
    datasets = _build_datasets(
        data=data,
        study=study,
        target=str(task["target"]),
        variant=str(task["variant"]),
        test_year=int(task["year"]),
        stage=str(task["stage"]),
        date_values=date_values,
    )
    parameters, maximum_rounds, patience = _model_parameters(study, str(task["target"]))
    if bool(task["selects_iterations"]):
        iterations = maximum_rounds
        callbacks = [
            lgb.early_stopping(
                stopping_rounds=patience,
                first_metric_only=True,
                verbose=False,
            )
        ]
    else:
        source_task = _task_by_id(
            _inner_a_task_id(str(task["target"]), int(task["age"]), int(task["year"]))
        )
        source_path = _task_result_path(output_root, source_task)
        source = json.loads(source_path.read_text(encoding="utf-8"))
        iterations = int(source["best_iteration"])
        callbacks = []
    _emit(
        "task_training_started",
        task_id=task["task_id"],
        train_rows=len(datasets.train_rows),
        evaluation_rows=len(datasets.evaluation_rows),
        feature_count=len(datasets.feature_names),
        requested_iterations=iterations,
    )
    started = time.perf_counter()
    model = lgb.train(
        parameters,
        datasets.train_set,
        num_boost_round=iterations,
        valid_sets=[datasets.evaluation_set],
        valid_names=["evaluation"],
        callbacks=callbacks,
    )
    elapsed = float(time.perf_counter() - started)
    best_iteration = (
        int(model.best_iteration)
        if bool(task["selects_iterations"])
        else int(iterations)
    )
    prediction = _predict(model, datasets.evaluation_sequence, best_iteration)
    actual = (
        np.asarray(data.target_risk[datasets.evaluation_rows], dtype=np.float32)
        if task["target"] == "risk"
        else np.asarray(data.target_state[datasets.evaluation_rows], dtype=np.int8)
    )
    decision_dates = np.asarray(
        data.decision_date_idx[datasets.evaluation_rows], dtype=np.int32
    )
    calibration: dict[str, Any] | None = None
    if task["target"] == "state":
        fit_year = int(task["year"]) - 1
        if task["stage"] == "inner":
            calibration = {
                **fit_temperature(
                    probability=prediction,
                    actual=actual,
                    date_idx=decision_dates,
                ),
                "fit_validation_year": fit_year,
                "source_task_id": task["task_id"],
            }
        else:
            source_task = _task_by_id(
                f"state_D{int(task['age'])}_{int(task['year'])}_inner_{task['variant']}"
            )
            source = json.loads(
                _task_result_path(output_root, source_task).read_text(encoding="utf-8")
            )
            calibration = {
                **dict(source["calibration"]),
                "source_task_id": source_task["task_id"],
            }
        calibrated = apply_temperature(prediction, float(calibration["temperature"]))
        daily = state_daily_metrics(
            date_idx=decision_dates,
            actual=actual,
            raw_probability=prediction,
            calibrated_probability=calibrated,
        )
    else:
        daily = risk_daily_metrics(
            date_idx=decision_dates,
            actual=actual,
            prediction=prediction,
            entry_either_top5=np.asarray(
                data.entry_either_top5[datasets.evaluation_rows], dtype=bool
            ),
        )
    output_dir = result_path.parent
    output_dir.mkdir(parents=True, exist_ok=True)
    model_path = output_dir / "model.txt"
    prediction_path = output_dir / "prediction.npy"
    rows_path = output_dir / "evaluation_rows.npy"
    daily_path = output_dir / "daily_metrics.parquet"
    model.save_model(str(model_path), num_iteration=best_iteration)
    _save_npy(prediction_path, prediction)
    _save_npy(rows_path, datasets.evaluation_rows)
    daily.to_parquet(daily_path, index=False, compression="zstd")
    evaluation_year = (
        int(task["year"]) - 1 if task["stage"] == "inner" else int(task["year"])
    )
    maximum_train_date_idx = int(np.max(data.decision_date_idx[datasets.train_rows]))
    if maximum_train_date_idx + int(study["targets"]["dependency_days"]) >= _year_start(
        date_values, evaluation_year
    ):
        raise AssertionError("saved post-entry task violates target purge")
    result = {
        "schema": TASK_SCHEMA,
        "status": "completed",
        "completed_at": _now(),
        "study_id": STUDY_ID,
        "study_config_sha256": study_hash,
        "task_id": task["task_id"],
        "target": task["target"],
        "age": int(task["age"]),
        "test_year": int(task["year"]),
        "evaluation_year": evaluation_year,
        "stage": task["stage"],
        "variant": task["variant"],
        "selects_iterations": bool(task["selects_iterations"]),
        "best_iteration": best_iteration,
        "iteration_source_task_id": (
            task["task_id"]
            if bool(task["selects_iterations"])
            else _inner_a_task_id(
                str(task["target"]), int(task["age"]), int(task["year"])
            )
        ),
        "train_row_count": len(datasets.train_rows),
        "evaluation_row_count": len(datasets.evaluation_rows),
        "maximum_train_decision_date_idx": maximum_train_date_idx,
        "evaluation_start_date_idx": _year_start(date_values, evaluation_year),
        "target_purge_days": int(study["targets"]["dependency_days"]),
        "parameters": parameters,
        "feature_names": list(datasets.feature_names),
        "feature_count": len(datasets.feature_names),
        "common_support_rule": "B-complete rows are used identically by A and B",
        "training_seconds": elapsed,
        "calibration": calibration,
        "metrics": _daily_summary(daily),
        "files": {
            "model": _file_record(model_path),
            "prediction": _file_record(
                prediction_path,
                shape=list(prediction.shape),
                dtype=str(prediction.dtype),
            ),
            "evaluation_rows": _file_record(
                rows_path,
                shape=list(datasets.evaluation_rows.shape),
                dtype=str(datasets.evaluation_rows.dtype),
            ),
            "daily_metrics": _file_record(daily_path),
        },
    }
    _write_json(result_path, result)
    _release_datasets(datasets)
    del model, prediction, actual, daily, datasets
    gc.collect()
    _emit(
        "task_training_completed",
        task_id=task["task_id"],
        best_iteration=best_iteration,
        seconds=elapsed,
    )
    return result


def _load_completed_task(
    *,
    output_root: Path,
    task: Mapping[str, Any],
    study: Mapping[str, Any],
    study_hash: str,
) -> tuple[dict[str, Any], pd.DataFrame]:
    result_path = _task_result_path(output_root, task)
    if not _task_complete(
        result_path,
        task=task,
        study=study,
        study_hash=study_hash,
    ):
        raise ValueError(f"post-entry task is incomplete: {task['task_id']}")
    result = json.loads(result_path.read_text(encoding="utf-8"))
    daily = pd.read_parquet(_verify_record(result["files"]["daily_metrics"]))
    return result, daily


def _paired_daily(
    *,
    target: str,
    age: int,
    year: int,
    output_root: Path,
    study: Mapping[str, Any],
    study_hash: str,
) -> tuple[dict[str, Any], pd.DataFrame]:
    task_a = _task_by_id(f"{target}_D{age}_{year}_outer_A")
    task_b = _task_by_id(f"{target}_D{age}_{year}_outer_B")
    result_a, daily_a = _load_completed_task(
        output_root=output_root,
        task=task_a,
        study=study,
        study_hash=study_hash,
    )
    result_b, daily_b = _load_completed_task(
        output_root=output_root,
        task=task_b,
        study=study,
        study_hash=study_hash,
    )
    rows_a = np.load(
        _verify_record(result_a["files"]["evaluation_rows"]),
        allow_pickle=False,
    )
    rows_b = np.load(
        _verify_record(result_b["files"]["evaluation_rows"]),
        allow_pickle=False,
    )
    if not np.array_equal(rows_a, rows_b):
        raise AssertionError(f"A/B common support changed for {target} D{age} {year}")
    paired = daily_a.merge(
        daily_b,
        on="date_idx",
        suffixes=("_A", "_B"),
        how="inner",
        validate="one_to_one",
    )
    if len(paired) != len(daily_a) or len(paired) != len(daily_b):
        raise AssertionError(f"A/B daily support changed for {target} D{age} {year}")
    for column in ("row_count",):
        if not np.array_equal(
            paired[f"{column}_A"].to_numpy(),
            paired[f"{column}_B"].to_numpy(),
        ):
            raise AssertionError(
                f"A/B daily {column} changed for {target} D{age} {year}"
            )
    if target == "risk" and not np.array_equal(
        paired["entry_either_top5_row_count_A"].to_numpy(),
        paired["entry_either_top5_row_count_B"].to_numpy(),
    ):
        raise AssertionError(f"A/B entry-top5 support changed for risk D{age} {year}")
    paired_path = (
        output_root
        / "evaluation"
        / target
        / f"age_{age:02d}"
        / f"fold_{year}_paired_daily.parquet"
    )
    paired_path.parent.mkdir(parents=True, exist_ok=True)
    paired.to_parquet(paired_path, index=False, compression="zstd")
    provenance = {
        "target": target,
        "age": age,
        "year": year,
        "A_task_id": task_a["task_id"],
        "B_task_id": task_b["task_id"],
        "common_evaluation_row_count": len(rows_a),
        "common_daily_count": len(paired),
        "paired_daily": _file_record(paired_path),
    }
    return provenance, paired


def _finite_mean(values: pd.Series | np.ndarray) -> float:
    array = np.asarray(values, dtype=np.float64)
    finite = array[np.isfinite(array)]
    return float(finite.mean()) if finite.size else math.nan


def _relative_harm(
    *, challenger: float, baseline: float, lower_is_better: bool = True
) -> float:
    if not math.isfinite(challenger) or not math.isfinite(baseline):
        return math.nan
    denominator = max(abs(float(baseline)), 1.0e-12)
    difference = (
        float(challenger) - float(baseline)
        if lower_is_better
        else float(baseline) - float(challenger)
    )
    return float(max(0.0, difference) / denominator)


def _risk_annual_record(*, age: int, year: int, paired: pd.DataFrame) -> dict[str, Any]:
    rank_delta = paired["rank_ic_B"].to_numpy(dtype=np.float64) - paired[
        "rank_ic_A"
    ].to_numpy(dtype=np.float64)
    entry_delta = paired["entry_either_top5_rank_ic_B"].to_numpy(
        dtype=np.float64
    ) - paired["entry_either_top5_rank_ic_A"].to_numpy(dtype=np.float64)
    mae_a = _finite_mean(paired["mae_A"])
    mae_b = _finite_mean(paired["mae_B"])
    pr_a = _finite_mean(paired["deep_adverse_pr_auc_A"])
    pr_b = _finite_mean(paired["deep_adverse_pr_auc_B"])
    return {
        "target": "risk",
        "age": age,
        "year": year,
        "rank_ic_A": _finite_mean(paired["rank_ic_A"]),
        "rank_ic_B": _finite_mean(paired["rank_ic_B"]),
        "rank_ic_delta": _finite_mean(rank_delta),
        "rank_ic_delta_hac": base._hac_mean_test(rank_delta, maximum_lag=9),
        "mae_A": mae_a,
        "mae_B": mae_b,
        "mae_improvement_A_minus_B": mae_a - mae_b,
        "relative_mae_harm": _relative_harm(challenger=mae_b, baseline=mae_a),
        "deep_adverse_pr_auc_A": pr_a,
        "deep_adverse_pr_auc_B": pr_b,
        "deep_adverse_pr_auc_delta": pr_b - pr_a,
        "entry_either_top5_rank_ic_A": _finite_mean(
            paired["entry_either_top5_rank_ic_A"]
        ),
        "entry_either_top5_rank_ic_B": _finite_mean(
            paired["entry_either_top5_rank_ic_B"]
        ),
        "entry_either_top5_rank_ic_delta": _finite_mean(entry_delta),
        "daily_count": len(paired),
    }


def _state_annual_record(
    *, age: int, year: int, paired: pd.DataFrame
) -> dict[str, Any]:
    ordinal_delta = paired["raw_ordinal_ic_B"].to_numpy(dtype=np.float64) - paired[
        "raw_ordinal_ic_A"
    ].to_numpy(dtype=np.float64)
    record: dict[str, Any] = {
        "target": "state",
        "age": age,
        "year": year,
        "raw_ordinal_ic_A": _finite_mean(paired["raw_ordinal_ic_A"]),
        "raw_ordinal_ic_B": _finite_mean(paired["raw_ordinal_ic_B"]),
        "raw_ordinal_ic_delta": _finite_mean(ordinal_delta),
        "raw_ordinal_ic_delta_hac": base._hac_mean_test(ordinal_delta, maximum_lag=9),
        "daily_count": len(paired),
    }
    for metric in ("brier", "logloss", "ece"):
        for calibration in ("raw", "calibrated"):
            column_a = f"{calibration}_{metric}_A"
            column_b = f"{calibration}_{metric}_B"
            value_a = _finite_mean(paired[column_a])
            value_b = _finite_mean(paired[column_b])
            prefix = f"{calibration}_{metric}"
            record[f"{prefix}_A"] = value_a
            record[f"{prefix}_B"] = value_b
            record[f"{prefix}_improvement_A_minus_B"] = value_a - value_b
            record[f"{prefix}_relative_harm"] = _relative_harm(
                challenger=value_b, baseline=value_a
            )
    return record


def _risk_age_decision(
    annual: Sequence[Mapping[str, Any]], rules: Mapping[str, Any]
) -> dict[str, Any]:
    ordered = sorted(annual, key=lambda row: int(row["year"]))
    if len(ordered) != len(TEST_YEARS):
        raise ValueError("risk age decision requires all three test years")
    rank = [float(row["rank_ic_delta"]) for row in ordered]
    supported = [
        bool(
            float(row["rank_ic_delta_hac"]["p_value_two_sided"])
            <= float(rules["maximum_hac_p_value"])
            and float(row["rank_ic_bh_q_value"])
            <= float(rules["maximum_bh_q_value_across_9_age_year_tests"])
        )
        for row in ordered
    ]
    mae = [float(row["mae_improvement_A_minus_B"]) for row in ordered]
    mae_harm = [float(row["relative_mae_harm"]) for row in ordered]
    pr = [float(row["deep_adverse_pr_auc_delta"]) for row in ordered]
    top5 = [float(row["entry_either_top5_rank_ic_delta"]) for row in ordered]
    checks = {
        "positive_rank_ic_years": sum(value > 0.0 for value in rank),
        "worst_rank_ic_delta": min(rank),
        "hac_and_bh_supported_years": sum(supported),
        "mae_improvement_years": sum(value > 0.0 for value in mae),
        "worst_relative_mae_harm": max(mae_harm),
        "pr_auc_improvement_years": sum(value > 0.0 for value in pr),
        "worst_pr_auc_delta": min(pr),
        "entry_either_top5_positive_rank_ic_years": sum(value > 0.0 for value in top5),
    }
    passed = bool(
        checks["positive_rank_ic_years"] >= int(rules["minimum_positive_rank_ic_years"])
        and checks["worst_rank_ic_delta"]
        >= float(rules["minimum_worst_year_rank_ic_delta"])
        and checks["hac_and_bh_supported_years"]
        >= int(rules["minimum_hac_p_supported_years"])
        and checks["mae_improvement_years"]
        >= int(rules["minimum_mae_improvement_years"])
        and checks["worst_relative_mae_harm"]
        <= float(rules["maximum_worst_year_relative_mae_harm"])
        and checks["pr_auc_improvement_years"]
        >= int(rules["minimum_pr_auc_improvement_years"])
        and checks["worst_pr_auc_delta"]
        >= -float(rules["maximum_worst_year_pr_auc_decline"])
        and (
            checks["entry_either_top5_positive_rank_ic_years"] == len(TEST_YEARS)
            if bool(rules["entry_either_top5_requires_three_positive_rank_ic_years"])
            else True
        )
    )
    return {
        "passed": passed,
        "checks": checks,
        "annual_rank_ic_delta": rank,
        "annual_mae_improvement": mae,
        "annual_pr_auc_delta": pr,
        "annual_entry_either_top5_rank_ic_delta": top5,
    }


def _state_age_decision(
    annual: Sequence[Mapping[str, Any]], rules: Mapping[str, Any]
) -> dict[str, Any]:
    ordered = sorted(annual, key=lambda row: int(row["year"]))
    if len(ordered) != len(TEST_YEARS):
        raise ValueError("state age decision requires all three test years")
    ordinal = [float(row["raw_ordinal_ic_delta"]) for row in ordered]
    brier = [float(row["calibrated_brier_improvement_A_minus_B"]) for row in ordered]
    logloss = [
        float(row["calibrated_logloss_improvement_A_minus_B"]) for row in ordered
    ]
    brier_harm = [float(row["calibrated_brier_relative_harm"]) for row in ordered]
    logloss_harm = [float(row["calibrated_logloss_relative_harm"]) for row in ordered]
    checks = {
        "positive_ordinal_ic_years": sum(value > 0.0 for value in ordinal),
        "worst_ordinal_ic_delta": min(ordinal),
        "calibrated_brier_improvement_years": sum(value > 0.0 for value in brier),
        "calibrated_logloss_improvement_years": sum(value > 0.0 for value in logloss),
        "worst_calibrated_brier_relative_harm": max(brier_harm),
        "worst_calibrated_logloss_relative_harm": max(logloss_harm),
    }
    passed = bool(
        checks["positive_ordinal_ic_years"]
        >= int(rules["minimum_positive_ordinal_ic_years"])
        and checks["worst_ordinal_ic_delta"]
        >= float(rules["minimum_worst_year_ordinal_ic_delta"])
        and checks["calibrated_brier_improvement_years"]
        >= int(rules["minimum_calibrated_brier_improvement_years"])
        and checks["calibrated_logloss_improvement_years"]
        >= int(rules["minimum_calibrated_logloss_improvement_years"])
        and checks["worst_calibrated_brier_relative_harm"]
        <= float(rules["maximum_worst_year_relative_proper_score_harm"])
        and checks["worst_calibrated_logloss_relative_harm"]
        <= float(rules["maximum_worst_year_relative_proper_score_harm"])
    )
    return {
        "passed": passed,
        "checks": checks,
        "annual_raw_ordinal_ic_delta": ordinal,
        "annual_calibrated_brier_improvement": brier,
        "annual_calibrated_logloss_improvement": logloss,
    }


def _calibration_decisions(annual: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    decisions: list[dict[str, Any]] = []
    for age in AGES:
        for variant in ("A", "B"):
            rows = sorted(
                (
                    row
                    for row in annual
                    if int(row["age"]) == age and row["target"] == "state"
                ),
                key=lambda row: int(row["year"]),
            )
            if len(rows) != len(TEST_YEARS):
                raise ValueError("calibration decision requires all state folds")
            brier = [
                float(row[f"raw_brier_{variant}"])
                - float(row[f"calibrated_brier_{variant}"])
                for row in rows
            ]
            logloss = [
                float(row[f"raw_logloss_{variant}"])
                - float(row[f"calibrated_logloss_{variant}"])
                for row in rows
            ]
            supported = bool(
                sum(value > 0.0 for value in brier) >= 2
                and sum(value > 0.0 for value in logloss) >= 2
            )
            decisions.append(
                {
                    "age": age,
                    "variant": variant,
                    "brier_improvement_years": sum(value > 0.0 for value in brier),
                    "logloss_improvement_years": sum(value > 0.0 for value in logloss),
                    "annual_brier_calibration_improvement": brier,
                    "annual_logloss_calibration_improvement": logloss,
                    "literal_probability_supported": supported,
                    "output_semantics": (
                        "calibrated_probability"
                        if supported
                        else "relative_state_score"
                    ),
                }
            )
    return decisions


def _general_decision(
    *,
    risk_by_age: Mapping[int, Mapping[str, Any]],
    state_by_age: Mapping[int, Mapping[str, Any]],
    rules: Mapping[str, Any],
) -> dict[str, Any]:
    passing = sorted(
        age for age, record in risk_by_age.items() if bool(record["passed"])
    )
    nonpassing = [age for age in AGES if age not in passing]
    harm_limit = float(rules["nonpassing_age_material_rank_harm"])
    nonpassing_without_material_harm = all(
        min(float(value) for value in risk_by_age[age]["annual_rank_ic_delta"])
        >= harm_limit
        for age in nonpassing
    )
    if (
        len(passing) >= int(rules["minimum_passing_risk_ages"])
        and nonpassing_without_material_harm
    ):
        status = "dedicated_post_entry_model_supported"
        next_step = "compare_hold_and_switch_value_net_of_cost"
    elif len(passing) == 1:
        status = "age_specific_information_only"
        next_step = "retain_daily_entry_contract_recompute_baseline"
    elif not passing:
        status = "dedicated_post_entry_model_rejected"
        next_step = "use_daily_recomputed_five_coordinate_entry_contract"
    else:
        status = "mixed_risk_evidence_with_material_age_harm"
        next_step = "retain_daily_entry_contract_recompute_baseline"
    return {
        "status": status,
        "passing_risk_ages": passing,
        "passing_state_ages": sorted(
            age for age, record in state_by_age.items() if bool(record["passed"])
        ),
        "nonpassing_risk_ages": nonpassing,
        "nonpassing_ages_without_material_rank_harm": bool(
            nonpassing_without_material_harm
        ),
        "state_cannot_rescue_failed_risk": bool(
            rules["state_cannot_rescue_failed_risk"]
        ),
        "next_step": next_step,
    }


def _task_inventory(
    *,
    output_root: Path,
    study: Mapping[str, Any],
    study_hash: str,
) -> list[dict[str, Any]]:
    inventory: list[dict[str, Any]] = []
    for task in _tasks():
        result, _daily = _load_completed_task(
            output_root=output_root,
            task=task,
            study=study,
            study_hash=study_hash,
        )
        result_path = _task_result_path(output_root, task)
        inventory.append(
            {
                "task_id": result["task_id"],
                "target": result["target"],
                "age": result["age"],
                "test_year": result["test_year"],
                "evaluation_year": result["evaluation_year"],
                "stage": result["stage"],
                "variant": result["variant"],
                "best_iteration": result["best_iteration"],
                "iteration_source_task_id": result["iteration_source_task_id"],
                "train_row_count": result["train_row_count"],
                "evaluation_row_count": result["evaluation_row_count"],
                "maximum_train_decision_date_idx": result[
                    "maximum_train_decision_date_idx"
                ],
                "evaluation_start_date_idx": result["evaluation_start_date_idx"],
                "target_purge_days": result["target_purge_days"],
                "calibration": result["calibration"],
                "metrics": result["metrics"],
                "files": result["files"],
                "task_result": _file_record(result_path),
            }
        )
    return inventory


def evaluate(
    *,
    study_path: Path = DEFAULT_STUDY_PATH,
    output_root: Path = DEFAULT_OUTPUT_ROOT,
) -> dict[str, Any]:
    study = load_study(study_path)
    study_hash = _config_hash(study_path)
    task_inventory = _task_inventory(
        output_root=output_root,
        study=study,
        study_hash=study_hash,
    )
    annual: list[dict[str, Any]] = []
    paired_records: list[dict[str, Any]] = []
    for target in ("risk", "state"):
        for age in AGES:
            for year in TEST_YEARS:
                provenance, paired = _paired_daily(
                    target=target,
                    age=age,
                    year=year,
                    output_root=output_root,
                    study=study,
                    study_hash=study_hash,
                )
                record = (
                    _risk_annual_record(age=age, year=year, paired=paired)
                    if target == "risk"
                    else _state_annual_record(age=age, year=year, paired=paired)
                )
                annual.append(record)
                paired_records.append(provenance)

    risk_records = [row for row in annual if row["target"] == "risk"]
    p_values = {
        f"D{int(row['age'])}:{int(row['year'])}": float(
            row["rank_ic_delta_hac"]["p_value_two_sided"]
        )
        for row in risk_records
    }
    q_values = feature._benjamini_hochberg(p_values)
    for row in risk_records:
        key = f"D{int(row['age'])}:{int(row['year'])}"
        row["rank_ic_bh_q_value"] = float(q_values[key])

    risk_rules = dict(study["decision"]["risk"])
    state_rules = dict(study["decision"]["state"])
    risk_by_age = {
        age: _risk_age_decision(
            [row for row in risk_records if int(row["age"]) == age],
            risk_rules,
        )
        for age in AGES
    }
    state_records = [row for row in annual if row["target"] == "state"]
    state_by_age = {
        age: _state_age_decision(
            [row for row in state_records if int(row["age"]) == age],
            state_rules,
        )
        for age in AGES
    }
    calibration = _calibration_decisions(annual)
    decision = {
        **_general_decision(
            risk_by_age=risk_by_age,
            state_by_age=state_by_age,
            rules=dict(study["decision"]["general_model"]),
        ),
        "risk_by_age": {str(age): risk_by_age[age] for age in AGES},
        "state_by_age": {str(age): state_by_age[age] for age in AGES},
        "state_probability_semantics": calibration,
        "does_not_select": list(study["non_selections"]),
    }
    landmarks = {
        str(age): json.loads(
            (_landmark_root(output_root, age) / "complete.json").read_text(
                encoding="utf-8"
            )
        )
        for age in AGES
    }
    summary = {
        "schema": SUMMARY_SCHEMA,
        "status": "completed",
        "completed_at": _now(),
        "study_id": STUDY_ID,
        "study_config_sha256": study_hash,
        "scope": {
            "landmark_ages": list(AGES),
            "test_years": list(TEST_YEARS),
            "formal_booster_count": len(_tasks()),
            "completed_booster_count": len(_tasks()),
            "maximum_consumed_outcome_date": "2025-12-31",
            "forbidden_2026_row_count": 0,
            "top5_is_not_a_slot_rule": True,
            "HAC_maximum_lag": 9,
            "risk_B_is_primary": True,
            "state_B_is_secondary": True,
        },
        "landmarks": {
            age: {
                "row_count": manifest["row_count"],
                "state_atlas_cutoff": manifest["state_atlas_cutoff"],
                "state_atlas_model": manifest["state_atlas_model"],
                "target_anchor": manifest["target_anchor"],
                "quality": manifest["quality"],
                "files": manifest["files"],
            }
            for age, manifest in landmarks.items()
        },
        "annual": annual,
        "paired": paired_records,
        "task_inventory": task_inventory,
        "calibration": calibration,
        "decision": decision,
        "disclosure": {
            "fold_reuse": study["folds"]["fold_role"],
            "population": study["landmarks"]["population"],
            "common_support": "A and B use identical B-complete rows.",
            "capacity": study["model"]["capacity_rule"],
            "top5": "Entry top 5% is a transfer diagnostic, not a slot rule.",
            "2026": "No 2026 row, outcome, prediction, or metric was read.",
            "policy": "No score fusion, exit, switching threshold, holding period, slot, leverage, stop, or account policy was selected.",
        },
    }
    _write_json(output_root / "summary.json", summary)
    _write_json(output_root / "decision.json", decision)
    record_root = DEFAULT_RECORD_ROOT
    _write_json(record_root / "config.json", study)
    _write_json(record_root / "result.json", summary)
    return summary


def _load_runtime(
    study: Mapping[str, Any],
) -> tuple[
    base.LearnabilityInputs,
    Mapping[str, Any],
    FrozenEntryContract,
    HoldingPathReader,
]:
    feature_study = feature.load_study(
        _resolve(study["sources"]["feature_study_config"])
    )
    inputs, pack = feature._load_validated_inputs(feature_study)
    if inputs.maximum_outcome_date != "2025-12-31":
        raise ValueError("post-entry inputs crossed the 2025 outcome boundary")
    contract = load_entry_contract(study)
    reader = HoldingPathReader(pack, inputs)
    return inputs, pack, contract, reader


def prepare_landmarks(
    *,
    study_path: Path = DEFAULT_STUDY_PATH,
    output_root: Path = DEFAULT_OUTPUT_ROOT,
) -> dict[str, Any]:
    study = load_study(study_path)
    study_hash = _config_hash(study_path)
    inputs, _pack, contract, reader = _load_runtime(study)
    manifests = {
        str(age): build_landmark(
            age=age,
            study=study,
            study_hash=study_hash,
            output_root=output_root,
            inputs=inputs,
            contract=contract,
            reader=reader,
        )
        for age in AGES
    }
    return {
        "status": "completed",
        "landmarks": {
            age: {
                "row_count": manifest["row_count"],
                "quality": manifest["quality"],
            }
            for age, manifest in manifests.items()
        },
    }


def status(
    *,
    study_path: Path = DEFAULT_STUDY_PATH,
    output_root: Path = DEFAULT_OUTPUT_ROOT,
) -> dict[str, Any]:
    study = load_study(study_path)
    study_hash = _config_hash(study_path)
    groups: dict[str, list[str]] = {"completed": [], "pending": []}
    for task in _tasks():
        complete = _task_complete(
            _task_result_path(output_root, task),
            task=task,
            study=study,
            study_hash=study_hash,
        )
        groups["completed" if complete else "pending"].append(str(task["task_id"]))
    landmarks = {
        str(age): _landmark_complete(
            path=_landmark_root(output_root, age) / "complete.json",
            study_hash=study_hash,
            contract_manifest=json.loads(
                _resolve(study["sources"]["entry_contract_manifest"]).read_text(
                    encoding="utf-8"
                )
            ),
            age=age,
        )
        for age in AGES
    }
    return {
        "study_id": STUDY_ID,
        "formal_task_count": len(_tasks()),
        "risk_task_count": sum(task["target"] == "risk" for task in _tasks()),
        "state_task_count": sum(task["target"] == "state" for task in _tasks()),
        "landmarks": landmarks,
        "summary_complete": (output_root / "summary.json").is_file(),
        **groups,
    }


def run_pending(
    *,
    study_path: Path = DEFAULT_STUDY_PATH,
    output_root: Path = DEFAULT_OUTPUT_ROOT,
) -> dict[str, Any]:
    study = load_study(study_path)
    study_hash = _config_hash(study_path)
    inputs, _pack, contract, reader = _load_runtime(study)
    for age in AGES:
        build_landmark(
            age=age,
            study=study,
            study_hash=study_hash,
            output_root=output_root,
            inputs=inputs,
            contract=contract,
            reader=reader,
        )
    current_age: int | None = None
    data: LandmarkData | None = None
    for task in _tasks():
        age = int(task["age"])
        if age != current_age:
            data = load_landmark(
                age=age,
                study=study,
                study_hash=study_hash,
                output_root=output_root,
                contract=contract,
            )
            current_age = age
        if data is None:
            raise AssertionError("post-entry landmark data was not loaded")
        _run_task(
            task=task,
            study=study,
            study_hash=study_hash,
            output_root=output_root,
            data=data,
            date_values=inputs.date_values,
        )
    return evaluate(study_path=study_path, output_root=output_root)


def self_test() -> dict[str, Any]:
    study = load_study()
    tasks = _tasks()
    if len(tasks) != 63:
        raise AssertionError("post-entry formal booster count changed")
    if sum(task["target"] == "risk" for task in tasks) != 27:
        raise AssertionError("post-entry risk booster count changed")
    if sum(task["target"] == "state" for task in tasks) != 36:
        raise AssertionError("post-entry state booster count changed")
    if len(_feature_names(study, 1, "A")) != 6:
        raise AssertionError("A capacity changed")
    if len(_feature_names(study, 1, "B")) != 16:
        raise AssertionError("B-D1 capacity changed")
    if len(_feature_names(study, 3, "B")) != 24:
        raise AssertionError("B-D3 capacity changed")
    if set(study["features"]["D1_forbidden_duplicates"]) & set(
        _feature_names(study, 1, "B")
    ):
        raise AssertionError("D1 contains a duplicate realized-path feature")

    probability = np.asarray(
        [[0.98, 0.01, 0.01], [0.01, 0.98, 0.01], [0.01, 0.01, 0.98]] * 40,
        dtype=np.float32,
    )
    actual = np.tile(np.asarray([0, 1, 1], dtype=np.int64), 40)
    dates = np.repeat(np.arange(40, dtype=np.int32), 3)
    fitted = fit_temperature(
        probability=probability,
        actual=actual,
        date_idx=dates,
    )
    if int(fitted["validation_row_count"]) != len(actual):
        raise AssertionError("temperature fit lost valid rows")
    calibrated = apply_temperature(probability, float(fitted["temperature"]))
    np.testing.assert_allclose(calibrated.sum(axis=1), 1.0, atol=1.0e-6)

    risk_rules = dict(study["decision"]["risk"])
    synthetic = []
    for year in TEST_YEARS:
        synthetic.append(
            {
                "year": year,
                "rank_ic_delta": 0.004,
                "rank_ic_delta_hac": {"p_value_two_sided": 0.01},
                "rank_ic_bh_q_value": 0.05,
                "mae_improvement_A_minus_B": 0.001,
                "relative_mae_harm": 0.0,
                "deep_adverse_pr_auc_delta": 0.003,
                "entry_either_top5_rank_ic_delta": 0.002,
            }
        )
    if not _risk_age_decision(synthetic, risk_rules)["passed"]:
        raise AssertionError("synthetic passing risk evidence was rejected")
    return {"status": "passed", "formal_task_count": len(tasks)}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the matched-capacity Seq100 D1/D3/D5 post-entry A/B study."
    )
    parser.add_argument("--study-path", type=Path, default=DEFAULT_STUDY_PATH)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    action = parser.add_mutually_exclusive_group()
    action.add_argument("--prepare-landmarks", action="store_true")
    action.add_argument("--run-pending", action="store_true")
    action.add_argument("--evaluate", action="store_true")
    action.add_argument("--self-test", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    study_path = _resolve(args.study_path)
    output_root = _resolve(args.output_root)
    if args.prepare_landmarks:
        payload = prepare_landmarks(study_path=study_path, output_root=output_root)
    elif args.run_pending:
        summary = run_pending(study_path=study_path, output_root=output_root)
        payload = {
            "status": summary["status"],
            "scope": summary["scope"],
            "decision": summary["decision"],
        }
    elif args.evaluate:
        payload = evaluate(study_path=study_path, output_root=output_root)
    elif args.self_test:
        payload = self_test()
    else:
        payload = status(study_path=study_path, output_root=output_root)
    print(json.dumps(payload, ensure_ascii=False, indent=2, default=_json_default))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
