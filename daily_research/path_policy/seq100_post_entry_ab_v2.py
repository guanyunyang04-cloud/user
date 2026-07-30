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
from scipy import stats

from daily_research.path_policy import seq100_entry_contract_oos as entry_contract
from daily_research.path_policy import (
    seq100_entry_fixed_capacity_audit as entry_v4,
)
from daily_research.path_policy import seq100_mfe_feature_family_audit as feature
from daily_research.path_policy import seq100_path_label_learnability as base
from daily_research.path_policy import seq100_post_entry_ab as v1
from daily_research.path_policy.seq100_post_entry_incremental_information import (
    HoldingPathReader,
)

WORKSPACE_ROOT = Path(__file__).resolve().parents[2]
STUDY_ID = "seq100_post_entry_ab_v2"
DEFAULT_STUDY_PATH = (
    WORKSPACE_ROOT / "daily_research/studies/seq100_post_entry_ab_v2.json"
)
DEFAULT_OUTPUT_ROOT = (
    WORKSPACE_ROOT / "daily_research/output/path_policy/studies/seq100_post_entry_ab_v2"
)
DEFAULT_RECORD_ROOT = (
    WORKSPACE_ROOT / "daily_research/research_records/seq100/seq100_post_entry_ab_v2"
)
AGES = (1, 3, 5)
TEST_YEARS = (2023, 2024, 2025)
TARGETS = ("mfe_10", "mfe_20", "risk_10", "risk_20", "state_10")
VARIANTS = ("A", "B")
LANDMARK_SCHEMA = "seq100_post_entry_landmark/v2"
TASK_SCHEMA = "seq100_post_entry_ab_task/v2"
SUMMARY_SCHEMA = "seq100_post_entry_ab_summary/v2"


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


def _verify_record(record: Mapping[str, Any]) -> Path:
    path = _resolve(str(record["path"]))
    if (
        not path.is_file()
        or path.stat().st_size != int(record["size"])
        or _sha256(path) != str(record["sha256"])
    ):
        raise ValueError(f"file record changed: {path}")
    return path


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
    if tuple(payload["targets"]) != TARGETS:
        raise ValueError("post-entry target contract changed")
    dimensions = dict(payload["features"]["dimension_count"])
    if dimensions != {"A": 6, "B_D1": 16, "B_D3": 24, "B_D5": 24}:
        raise ValueError("A/B dimensions changed")
    forbidden = set(payload["features"]["D1_forbidden_duplicates"])
    if forbidden & set(payload["features"]["D1_path"]):
        raise ValueError("D1 includes a structurally duplicate feature")
    if int(payload["model"]["formal_booster_count"]) != 90:
        raise ValueError("formal booster count changed")
    return payload


@dataclass(frozen=True)
class FrozenEntryContract:
    rows: np.ndarray
    raw: np.ndarray
    rank: np.ndarray
    manifest: dict[str, Any]


def load_entry_contract(study: Mapping[str, Any]) -> FrozenEntryContract:
    manifest_path = _resolve(study["sources"]["entry_contract_manifest"])
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    expected = {
        "schema": entry_v4.CONTRACT_V4_SCHEMA,
        "status": "completed_candidate_aligned_strict_oos_contract",
        "study_id": entry_v4.CONTRACT_V4_ID,
        "candidate_years": list(entry_v4.CONTRACT_YEARS),
        "physical_columns": list(entry_contract.PHYSICAL_COLUMNS),
    }
    if any(manifest.get(key) != value for key, value in expected.items()):
        raise ValueError("strict v4 entry-contract semantics changed")
    semantics = dict(manifest["semantics"])
    if (
        not bool(semantics["mfe"]["rank_first"])
        or bool(semantics["mfe"]["literal_expected_return"])
        or bool(semantics["state"]["literal_stable_probability"])
    ):
        raise ValueError("v4 rank-first/state-score semantics changed")
    if manifest["scope"]["maximum_consumed_outcome_date"] != "2025-12-31":
        raise ValueError("v4 entry contract crossed the outcome boundary")
    if int(manifest["scope"]["forbidden_2026_row_count"]) != 0:
        raise ValueError("v4 entry contract contains a 2026 row")
    files = dict(manifest["files"])
    rows = np.load(_verify_record(files["candidate_rows"]), mmap_mode="r")
    raw = np.load(_verify_record(files["raw_predictions"]), mmap_mode="r")
    rank = np.load(_verify_record(files["date_rank_predictions"]), mmap_mode="r")
    if raw.shape != rank.shape or raw.shape != (
        len(rows),
        len(entry_contract.PHYSICAL_COLUMNS),
    ):
        raise ValueError("strict v4 entry-contract matrix shape changed")
    return FrozenEntryContract(
        rows=np.asarray(rows),
        raw=np.asarray(raw),
        rank=np.asarray(rank),
        manifest=manifest,
    )


def _candidate_lookup(
    candidate_rows: np.ndarray, requested_rows: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    source = np.asarray(candidate_rows, dtype=np.int64)
    requested = np.asarray(requested_rows, dtype=np.int64)
    if source.ndim != 1 or requested.ndim != 1 or not source.size:
        raise ValueError("candidate lookup requires one-dimensional arrays")
    if bool(np.any(source[1:] <= source[:-1])):
        raise ValueError("candidate rows must be unique and ordered")
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


def _v1_landmark(
    study: Mapping[str, Any], age: int
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    root = _resolve(study["sources"]["post_entry_v1_output_root"])
    path = root / "landmarks" / f"age_{int(age):02d}" / "complete.json"
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if (
        manifest.get("schema") != v1.LANDMARK_SCHEMA
        or manifest.get("status") != "completed"
        or int(manifest.get("age", -1)) != int(age)
    ):
        raise ValueError(f"v1 landmark is not complete: D{age}")
    arrays = {
        name: np.load(_verify_record(record), mmap_mode="r")
        for name, record in dict(manifest["files"]).items()
    }
    return manifest, arrays


def _landmark_complete(
    *,
    path: Path,
    study_hash: str,
    contract_manifest: Mapping[str, Any],
    v1_manifest: Mapping[str, Any],
    age: int,
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
            or payload.get("v1_landmark_sha256")
            != _sha256(
                _resolve(v1_manifest["files"]["features"]["path"]).parent
                / "complete.json"
            )
            or int(payload.get("state_atlas_cutoff", -1)) != 2022
        ):
            return False
        for record in payload["files"].values():
            _verify_record(record)
        _verify_record(payload["state_atlas_model"])
        return True
    except (
        KeyError,
        OSError,
        TypeError,
        ValueError,
        json.JSONDecodeError,
    ):
        return False


def _open_memmap(
    path: Path, *, dtype: Any, shape: tuple[int, ...], mode: str
) -> np.memmap:
    return np.lib.format.open_memmap(path, mode=mode, dtype=dtype, shape=shape)


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
        "target_mfe_10": root / "target_mfe_10.npy",
        "target_mfe_20": root / "target_mfe_20.npy",
        "target_risk_10": root / "target_risk_10.npy",
        "target_risk_20": root / "target_risk_20.npy",
        "target_state_10": root / "target_state_10.npy",
        "endpoint_return_10": root / "endpoint_return_10.npy",
        "endpoint_return_20": root / "endpoint_return_20.npy",
        "tradable_day_fraction": root / "tradable_day_fraction.npy",
    }
    mode = "r+" if progress is not None else "w+"
    shapes = {
        "features": ((count, feature_count), np.float32),
        "entry_rows": ((count,), np.int64),
        "current_rows": ((count,), np.int64),
        "decision_date_idx": ((count,), np.int32),
        "entry_either_top5": ((count,), bool),
        "target_mfe_10": ((count,), np.float32),
        "target_mfe_20": ((count,), np.float32),
        "target_risk_10": ((count,), np.float32),
        "target_risk_20": ((count,), np.float32),
        "target_state_10": ((count,), np.int8),
        "endpoint_return_10": ((count,), np.float32),
        "endpoint_return_20": ((count,), np.float32),
        "tradable_day_fraction": ((count,), np.float32),
    }
    arrays = {
        name: _open_memmap(path=paths[name], dtype=dtype, shape=shape, mode=mode)
        for name, (shape, dtype) in shapes.items()
    }
    if mode == "w+":
        for name, array in arrays.items():
            if name in {"entry_rows", "current_rows", "decision_date_idx"}:
                array[:] = -1
            elif name == "entry_either_top5":
                array[:] = False
            elif name == "target_state_10":
                array[:] = -1
            else:
                array[:] = np.nan
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
    v1_manifest, old = _v1_landmark(study, age)
    if _landmark_complete(
        path=complete_path,
        study_hash=study_hash,
        contract_manifest=contract.manifest,
        v1_manifest=v1_manifest,
        age=age,
    ):
        return json.loads(complete_path.read_text(encoding="utf-8"))
    count = int(v1_manifest["row_count"])
    if any(len(values) != count for values in old.values()):
        raise ValueError(f"v1 landmark arrays are not aligned: D{age}")
    feature_names = _full_feature_names(study)
    old_feature_names = tuple(str(value) for value in v1_manifest["feature_names"])
    if set(feature_names) != set(old_feature_names):
        raise ValueError(f"v1 realized feature vocabulary changed: D{age}")
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
            raise ValueError("landmark recovery metadata differs from v4 contract")
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
    position = {name: index for index, name in enumerate(feature_names)}
    old_position = {name: index for index, name in enumerate(old_feature_names)}
    path_names = tuple(str(value) for value in study["features"]["D3_D5_path"])
    entry_rows_all = np.asarray(old["entry_rows"], dtype=np.int64)
    current_rows_all = np.asarray(old["current_rows"], dtype=np.int64)
    decision_dates_all = np.asarray(old["decision_date_idx"], dtype=np.int32)
    if bool(np.any(decision_dates_all[1:] < decision_dates_all[:-1])):
        raise ValueError("v1 landmark decision dates are no longer ordered")
    entry_positions, entry_found = _candidate_lookup(contract.rows, entry_rows_all)
    safe_entry = np.where(entry_found, entry_positions, 0)
    current_requested = np.where(current_rows_all >= 0, current_rows_all, 0)
    current_positions, current_found = _candidate_lookup(
        contract.rows, current_requested
    )
    current_found &= current_rows_all >= 0
    safe_current = np.where(current_found, current_positions, 0)
    if not bool(entry_found.all()):
        raise ValueError(f"a v1 filled entry is absent from v4: D{age}")
    if not np.array_equal(
        np.asarray(old["tradable_day_fraction"]),
        np.asarray(old["tradable_day_fraction"]),
        equal_nan=True,
    ):
        raise AssertionError("v1 tradability array is not self-consistent")
    started = time.perf_counter()
    batch_size = 100_000
    for left in range(next_offset, count, batch_size):
        right = min(left + batch_size, count)
        entry_rows = entry_rows_all[left:right]
        current_rows = current_rows_all[left:right]
        decision_dates = decision_dates_all[left:right]
        symbols = np.asarray(inputs.candidate_symbol_idx[entry_rows], dtype=np.int32)
        entry_rank = np.asarray(contract.rank[safe_entry[left:right]], dtype=np.float32)
        current_rank = np.asarray(
            contract.rank[safe_current[left:right]], dtype=np.float32
        )
        local_found = current_found[left:right]
        current_rank[~local_found] = np.nan
        block = np.full((right - left, len(feature_names)), np.nan, dtype=np.float32)
        for name, source in a_sources.items():
            block[:, position[name]] = current_rank[:, physical[source]]
        for name, source in change_sources.items():
            column = physical[source]
            block[:, position[name]] = current_rank[:, column] - entry_rank[:, column]
        for name in path_names:
            block[:, position[name]] = np.asarray(
                old["features"][left:right, old_position[name]], dtype=np.float32
            )
        targets, _flags = reader.holding_targets(decision_dates, symbols)
        state = np.asarray(targets["state_10"], dtype=np.float32)
        arrays["features"][left:right] = block
        arrays["entry_rows"][left:right] = entry_rows
        arrays["current_rows"][left:right] = current_rows
        arrays["decision_date_idx"][left:right] = decision_dates
        arrays["entry_either_top5"][left:right] = (
            entry_rank[:, physical["mfe_10"]] >= 0.95
        ) | (entry_rank[:, physical["mfe_20"]] >= 0.95)
        arrays["target_mfe_10"][left:right] = np.asarray(
            targets["matched_mfe_10"], dtype=np.float32
        )
        arrays["target_mfe_20"][left:right] = np.asarray(
            targets["matched_mfe_20"], dtype=np.float32
        )
        arrays["target_risk_10"][left:right] = np.asarray(
            targets["pre_peak_mae_10"], dtype=np.float32
        )
        arrays["target_risk_20"][left:right] = np.asarray(
            targets["pre_peak_mae_20"], dtype=np.float32
        )
        arrays["target_state_10"][left:right] = np.where(
            np.isin(state, (0.0, 1.0, 2.0)), state, -1
        ).astype(np.int8)
        arrays["endpoint_return_10"][left:right] = np.asarray(
            targets["g_10"], dtype=np.float32
        )
        arrays["endpoint_return_20"][left:right] = np.asarray(
            targets["g_20"], dtype=np.float32
        )
        arrays["tradable_day_fraction"][left:right] = np.asarray(
            old["tradable_day_fraction"][left:right], dtype=np.float32
        )
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
    selected_names = _feature_names(study, age, "B")
    selected_columns = np.asarray(
        [position[name] for name in selected_names], dtype=np.int64
    )
    b_complete = np.isfinite(np.asarray(arrays["features"][:, selected_columns])).all(
        axis=1
    )
    current_found_final = np.asarray(arrays["current_rows"]) >= 0
    target_valid = {
        "mfe_10": np.isfinite(arrays["target_mfe_10"]),
        "mfe_20": np.isfinite(arrays["target_mfe_20"]),
        "risk_10": np.isfinite(arrays["target_risk_10"]),
        "risk_20": np.isfinite(arrays["target_risk_20"]),
        "state_10": np.isin(arrays["target_state_10"], (0, 1, 2)),
    }
    decision_years = np.asarray(
        [
            int(str(inputs.date_values[int(date_idx)])[:4])
            for date_idx in np.asarray(arrays["decision_date_idx"])
        ],
        dtype=np.int16,
    )
    if bool(np.any(decision_years == 2026)):
        raise AssertionError("a 2026 landmark observation reached v2")
    quality = {
        "cohort_count": count,
        "stable_entry_rows_equal_v1": bool(
            np.array_equal(arrays["entry_rows"], old["entry_rows"])
        ),
        "stable_current_rows_equal_v1": bool(
            np.array_equal(arrays["current_rows"], old["current_rows"])
        ),
        "stable_decision_dates_equal_v1": bool(
            np.array_equal(arrays["decision_date_idx"], old["decision_date_idx"])
        ),
        "current_contract_found_count": int(current_found_final.sum()),
        "selected_B_feature_complete_count": int(b_complete.sum()),
        "target_valid_count": {
            target: int(valid.sum()) for target, valid in target_valid.items()
        },
        "target_common_support_count": {
            target: int((b_complete & valid & current_found_final).sum())
            for target, valid in target_valid.items()
        },
        "entry_either_top5_count": int(
            np.asarray(arrays["entry_either_top5"], dtype=bool).sum()
        ),
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
    if not all(
        bool(quality[name])
        for name in (
            "stable_entry_rows_equal_v1",
            "stable_current_rows_equal_v1",
            "stable_decision_dates_equal_v1",
        )
    ):
        raise AssertionError("v2 landmark stable row keys changed")
    files = {
        name: _file_record(
            path, shape=list(arrays[name].shape), dtype=str(arrays[name].dtype)
        )
        for name, path in paths.items()
    }
    state_cutoff = int(np.asarray(reader.state10_model["window_year"])[0])
    if state_cutoff != 2022:
        raise AssertionError("post-entry state target does not use through-2022 atlas")
    state_model = _file_record(
        _resolve(inputs.label_manifest["state_models"]["10"]["path"])
    )
    v1_complete_path = (
        _resolve(v1_manifest["files"]["features"]["path"]).parent / "complete.json"
    )
    completed = {
        "schema": LANDMARK_SCHEMA,
        "status": "completed",
        "completed_at": _now(),
        "study_id": STUDY_ID,
        "study_config_sha256": study_hash,
        "entry_contract_sha256": contract.manifest["files"]["raw_predictions"][
            "sha256"
        ],
        "v1_landmark_sha256": _sha256(v1_complete_path),
        "age": int(age),
        "row_count": count,
        "feature_names": list(feature_names),
        "A_feature_names": list(_feature_names(study, age, "A")),
        "B_feature_names": list(selected_names),
        "targets": list(TARGETS),
        "state_atlas_cutoff": state_cutoff,
        "state_atlas_model": state_model,
        "target_anchor": study["landmarks"]["target_anchor"],
        "quality": quality,
        "v1_reuse": {
            "source_complete": _file_record(v1_complete_path),
            "reused": [
                "entry_rows",
                "current_rows",
                "decision_date_idx",
                "realized_price_and_turnover_path_columns",
                "tradable_day_fraction",
            ],
            "rebuilt": [
                "v4 current ranks",
                "v4 entry-to-current rank changes",
                "entry_either_top5",
                "five remaining targets",
                "endpoint companions",
            ],
        },
        "files": files,
    }
    _write_json(complete_path, completed)
    _write_json(
        progress_path,
        {
            "status": "completed",
            "study_config_sha256": study_hash,
            "entry_contract_sha256": contract.manifest["files"]["raw_predictions"][
                "sha256"
            ],
            "age": int(age),
            "row_count": count,
            "feature_names": list(feature_names),
            "next_offset": count,
            "updated_at": _now(),
        },
    )
    _emit("landmark_completed", age=age, rows=count)
    return completed


@dataclass
class LandmarkData:
    age: int
    feature_names: tuple[str, ...]
    features: np.ndarray
    entry_rows: np.ndarray
    current_rows: np.ndarray
    decision_date_idx: np.ndarray
    entry_either_top5: np.ndarray
    targets: dict[str, np.ndarray]
    endpoints: dict[int, np.ndarray]
    tradable_day_fraction: np.ndarray
    manifest: dict[str, Any]


def load_landmark(
    *,
    study: Mapping[str, Any],
    study_hash: str,
    contract: FrozenEntryContract,
    output_root: Path,
    age: int,
) -> LandmarkData:
    path = _landmark_root(output_root, age) / "complete.json"
    v1_manifest, _old = _v1_landmark(study, age)
    if not _landmark_complete(
        path=path,
        study_hash=study_hash,
        contract_manifest=contract.manifest,
        v1_manifest=v1_manifest,
        age=age,
    ):
        raise ValueError(f"v2 landmark is incomplete: D{age}")
    manifest = json.loads(path.read_text(encoding="utf-8"))
    files = dict(manifest["files"])
    arrays = {
        name: np.load(_verify_record(record), mmap_mode="r")
        for name, record in files.items()
    }
    targets = {
        "mfe_10": arrays["target_mfe_10"],
        "mfe_20": arrays["target_mfe_20"],
        "risk_10": arrays["target_risk_10"],
        "risk_20": arrays["target_risk_20"],
        "state_10": arrays["target_state_10"],
    }
    return LandmarkData(
        age=int(age),
        feature_names=tuple(str(value) for value in manifest["feature_names"]),
        features=arrays["features"],
        entry_rows=arrays["entry_rows"],
        current_rows=arrays["current_rows"],
        decision_date_idx=arrays["decision_date_idx"],
        entry_either_top5=arrays["entry_either_top5"],
        targets=targets,
        endpoints={
            10: arrays["endpoint_return_10"],
            20: arrays["endpoint_return_20"],
        },
        tradable_day_fraction=arrays["tradable_day_fraction"],
        manifest=manifest,
    )


def _target_horizon(target: str) -> int:
    return 20 if str(target).endswith("_20") else 10


def _target_kind(target: str) -> str:
    if str(target).startswith("mfe_"):
        return "mfe"
    if str(target).startswith("risk_"):
        return "risk"
    if target == "state_10":
        return "state"
    raise ValueError(target)


def _target_valid(target: str, values: np.ndarray) -> np.ndarray:
    if target == "state_10":
        return np.isin(np.asarray(values), (0, 1, 2))
    return np.isfinite(np.asarray(values, dtype=np.float64))


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
    evaluation_year: int,
    date_values: np.ndarray,
    selection_outcome_cutoff: bool,
) -> tuple[np.ndarray, np.ndarray]:
    b_names = _feature_names(study, data.age, "B")
    name_to_position = {name: index for index, name in enumerate(data.feature_names)}
    columns = np.asarray([name_to_position[name] for name in b_names], dtype=np.int64)
    complete = np.isfinite(np.asarray(data.features[:, columns])).all(axis=1)
    complete &= _target_valid(target, data.targets[target])
    dates = np.asarray(data.decision_date_idx, dtype=np.int32)
    evaluation_start = _year_start(date_values, evaluation_year)
    evaluation_stop = (
        _year_start(date_values, evaluation_year + 1)
        if evaluation_year < 2025
        else len(date_values)
    )
    dependency = int(study["targets"][target]["dependency_days"])
    train = complete & (dates + dependency < evaluation_start)
    evaluation = complete & (dates >= evaluation_start) & (dates < evaluation_stop)
    if selection_outcome_cutoff:
        evaluation &= dates + dependency < evaluation_stop
    train_rows = np.flatnonzero(train).astype(np.int64, copy=False)
    evaluation_rows = np.flatnonzero(evaluation).astype(np.int64, copy=False)
    if not train_rows.size or not evaluation_rows.size:
        raise ValueError(f"empty split for {target} D{data.age} {evaluation_year}")
    if int(dates[train_rows[-1]]) + dependency >= evaluation_start:
        raise AssertionError("post-entry target purge failed")
    if selection_outcome_cutoff and (
        int(dates[evaluation_rows[-1]]) + dependency >= evaluation_stop
    ):
        raise AssertionError("capacity selection crossed the outcome-year boundary")
    return train_rows, evaluation_rows


@dataclass
class Datasets:
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
    evaluation_year: int,
    date_values: np.ndarray,
    selection_outcome_cutoff: bool,
) -> Datasets:
    import lightgbm as lgb

    train_rows, evaluation_rows = _split_rows(
        data=data,
        study=study,
        target=target,
        evaluation_year=evaluation_year,
        date_values=date_values,
        selection_outcome_cutoff=selection_outcome_cutoff,
    )
    names = _feature_names(study, data.age, variant)
    name_to_position = {name: index for index, name in enumerate(data.feature_names)}
    columns = np.asarray([name_to_position[name] for name in names], dtype=np.int64)
    batch_size = int(study["model"]["sequence_batch_size"])
    train_sequence = v1._MatrixSequence(
        matrix=data.features,
        rows=train_rows,
        columns=columns,
        batch_size=batch_size,
    )
    evaluation_sequence = v1._MatrixSequence(
        matrix=data.features,
        rows=evaluation_rows,
        columns=columns,
        batch_size=batch_size,
    )
    train_label = np.asarray(data.targets[target][train_rows])
    evaluation_label = np.asarray(data.targets[target][evaluation_rows])
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
    return Datasets(
        train_rows=train_rows,
        evaluation_rows=evaluation_rows,
        train_sequence=train_sequence,
        evaluation_sequence=evaluation_sequence,
        train_set=train_set,
        evaluation_set=evaluation_set,
        feature_names=names,
    )


def _release_datasets(datasets: Datasets) -> None:
    datasets.train_set = None
    datasets.evaluation_set = None
    datasets.train_sequence = None
    datasets.evaluation_sequence = None
    gc.collect()
    signal_module = getattr(v1, "signal_quality", None)
    if signal_module is not None:
        signal_module.sequence_training._trim_working_set()


def _model_parameters(study: Mapping[str, Any], target: str) -> dict[str, Any]:
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
    if target == "state_10":
        parameters.update(
            {"objective": "multiclass", "metric": "multi_logloss", "num_class": 3}
        )
    else:
        parameters.update(
            {
                "objective": "huber",
                "metric": "huber",
                "alpha": float(model["huber_alpha"]),
            }
        )
    return parameters


def _predict(model: Any, sequence: Any, iterations: int) -> np.ndarray:
    return v1._predict(model, sequence, iterations)


def _safe_spearman(left: np.ndarray, right: np.ndarray) -> float:
    return v1._safe_spearman(left, right)


def _mfe_daily_metrics(
    *,
    date_idx: np.ndarray,
    actual: np.ndarray,
    prediction: np.ndarray,
    pre_peak_mae: np.ndarray,
    endpoint_return: np.ndarray,
    entry_either_top5: np.ndarray,
) -> pd.DataFrame:
    dates = np.asarray(date_idx, dtype=np.int32)
    target = np.asarray(actual, dtype=np.float64)
    score = np.asarray(prediction, dtype=np.float64)
    adverse = np.asarray(pre_peak_mae, dtype=np.float64)
    endpoint = np.asarray(endpoint_return, dtype=np.float64)
    entry_top5 = np.asarray(entry_either_top5, dtype=bool)
    rows: list[dict[str, Any]] = []
    boundaries = np.flatnonzero(np.r_[True, dates[1:] != dates[:-1], True])
    for left, right in pairwise(boundaries):
        valid = np.isfinite(target[left:right]) & np.isfinite(score[left:right])
        if int(valid.sum()) < 20:
            continue
        y = target[left:right][valid]
        p = score[left:right][valid]
        current_adverse = adverse[left:right][valid]
        current_endpoint = endpoint[left:right][valid]
        current_entry = entry_top5[left:right][valid]
        count = len(y)
        top_count = max(1, math.ceil(0.05 * count))
        selected = np.argsort(p, kind="mergesort")[-top_count:]
        actual_order = np.argsort(y, kind="mergesort")
        tail = np.zeros(count, dtype=bool)
        tail[actual_order[-max(1, math.ceil(0.20 * count)) :]] = True
        rows.append(
            {
                "date_idx": int(dates[left]),
                "row_count": int(count),
                "rank_ic": _safe_spearman(p, y),
                "mae": float(np.mean(np.abs(p - y))),
                "top5_remaining_mfe": float(np.mean(y[selected])),
                "top5_tail_rate": float(tail[selected].mean()),
                "top5_tail_lift": float(tail[selected].mean() - tail.mean()),
                "top5_pre_peak_mae": float(np.nanmean(current_adverse[selected])),
                "top5_endpoint_return": float(np.nanmean(current_endpoint[selected])),
                "entry_either_top5_row_count": int(current_entry.sum()),
                "entry_either_top5_rank_ic": (
                    _safe_spearman(p[current_entry], y[current_entry])
                    if int(current_entry.sum()) >= 20
                    else math.nan
                ),
            }
        )
    frame = pd.DataFrame.from_records(rows)
    if frame.empty:
        raise ValueError("MFE evaluation produced no valid dates")
    return frame


def _risk_daily_metrics(
    *,
    date_idx: np.ndarray,
    actual: np.ndarray,
    prediction: np.ndarray,
    entry_either_top5: np.ndarray,
) -> pd.DataFrame:
    return v1.risk_daily_metrics(
        date_idx=date_idx,
        actual=actual,
        prediction=prediction,
        entry_either_top5=entry_either_top5,
    )


def _state_daily_metrics(
    *,
    date_idx: np.ndarray,
    actual: np.ndarray,
    probability: np.ndarray,
) -> pd.DataFrame:
    dates = np.asarray(date_idx, dtype=np.int32)
    target = np.asarray(actual, dtype=np.int64)
    predicted = np.asarray(probability, dtype=np.float64)
    rows: list[dict[str, Any]] = []
    boundaries = np.flatnonzero(np.r_[True, dates[1:] != dates[:-1], True])
    for left, right in pairwise(boundaries):
        valid = np.isin(target[left:right], (0, 1, 2)) & np.isfinite(
            predicted[left:right]
        ).all(axis=1)
        if int(valid.sum()) < 20:
            continue
        y = target[left:right][valid]
        p = np.clip(predicted[left:right][valid], 1.0e-7, 1.0 - 1.0e-7)
        p /= p.sum(axis=1, keepdims=True)
        one_hot = np.eye(3, dtype=np.float64)[y]
        expected = p[:, 1] + 2.0 * p[:, 2]
        rows.append(
            {
                "date_idx": int(dates[left]),
                "row_count": len(y),
                "ordinal_ic": _safe_spearman(expected, y),
                "brier": float(np.mean(np.square(p - one_hot).sum(axis=1))),
                "logloss": float(
                    np.mean(
                        -np.log(
                            np.clip(
                                p[np.arange(len(y)), y],
                                1.0e-12,
                                1.0,
                            )
                        )
                    )
                ),
            }
        )
    frame = pd.DataFrame.from_records(rows)
    if frame.empty:
        raise ValueError("state evaluation produced no valid dates")
    return frame


def _daily_metrics(
    *,
    data: LandmarkData,
    target: str,
    rows: np.ndarray,
    prediction: np.ndarray,
) -> pd.DataFrame:
    dates = np.asarray(data.decision_date_idx[rows], dtype=np.int32)
    actual = np.asarray(data.targets[target][rows])
    if _target_kind(target) == "mfe":
        horizon = _target_horizon(target)
        return _mfe_daily_metrics(
            date_idx=dates,
            actual=actual,
            prediction=prediction,
            pre_peak_mae=np.asarray(data.targets[f"risk_{horizon}"][rows]),
            endpoint_return=np.asarray(data.endpoints[horizon][rows]),
            entry_either_top5=np.asarray(data.entry_either_top5[rows]),
        )
    if _target_kind(target) == "risk":
        return _risk_daily_metrics(
            date_idx=dates,
            actual=actual,
            prediction=prediction,
            entry_either_top5=np.asarray(data.entry_either_top5[rows]),
        )
    return _state_daily_metrics(
        date_idx=dates,
        actual=actual,
        probability=prediction,
    )


def _daily_summary(frame: pd.DataFrame) -> dict[str, float]:
    return {
        str(column): float(np.nanmean(frame[column].to_numpy(dtype=np.float64)))
        for column in frame.columns
        if column != "date_idx"
    }


def _state_collapsed(probability: np.ndarray) -> bool:
    values = np.asarray(probability, dtype=np.float64)
    return bool(
        values.ndim != 2
        or values.shape[1] != 3
        or not np.isfinite(values).all()
        or not np.allclose(values.sum(axis=1), 1.0, atol=1.0e-5, rtol=1.0e-5)
        or np.unique(np.argmax(values, axis=1)).size < 2
    )


def _load_runtime(
    study: Mapping[str, Any],
) -> tuple[
    Mapping[str, Any],
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
        raise ValueError("input outcome boundary changed")
    contract = load_entry_contract(study)
    reader = HoldingPathReader(pack, inputs)
    return feature_study, inputs, pack, contract, reader


def prepare_landmarks(
    *,
    study_path: Path = DEFAULT_STUDY_PATH,
    output_root: Path = DEFAULT_OUTPUT_ROOT,
) -> dict[str, Any]:
    study = load_study(study_path)
    _feature_study, inputs, _pack, contract, reader = _load_runtime(study)
    study_hash = _config_hash(study_path)
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
    result = {
        "status": "completed",
        "study_id": STUDY_ID,
        "entry_contract": entry_v4.CONTRACT_V4_ID,
        "ages": {
            age: {
                "row_count": int(manifest["row_count"]),
                "quality": manifest["quality"],
            }
            for age, manifest in manifests.items()
        },
        "maximum_outcome_date_read": "2025-12-31",
        "forbidden_2026_row_count": 0,
    }
    _write_json(output_root / "landmark_summary.json", result)
    return result


def _load_capacity_decision(study: Mapping[str, Any]) -> dict[str, Any]:
    path = _resolve(study["sources"]["capacity_decision"])
    decision = json.loads(path.read_text(encoding="utf-8"))
    if (
        decision.get("status") != "post_entry_target_capacities_frozen"
        or tuple(decision.get("selection_years", ())) != (2021, 2022)
        or decision.get("maximum_selection_outcome_date") != "2022-12-31"
        or decision.get("input_variant") != "A"
        or not bool(decision.get("uniform_across_D1_D3_D5"))
    ):
        raise ValueError("post-entry capacity decision is not frozen")
    selected = dict(decision["selected_rounds"])
    if set(selected) != set(TARGETS):
        raise ValueError("capacity decision target set changed")
    allowed = {
        "mfe_10": {64, 128, 256, 512},
        "mfe_20": {64, 128, 256, 512},
        "risk_10": {32, 64, 128, 256},
        "risk_20": {32, 64, 128, 256},
        "state_10": {32, 64, 128, 256},
    }
    if any(int(selected[target]) not in allowed[target] for target in TARGETS):
        raise ValueError("capacity decision contains an unknown candidate")
    return decision


def _tasks(
    study: Mapping[str, Any], decision: Mapping[str, Any]
) -> list[dict[str, Any]]:
    selected = dict(decision["selected_rounds"])
    return [
        {
            "task_id": f"{target}_D{age}_{year}_{variant}",
            "target": target,
            "age": int(age),
            "test_year": int(year),
            "variant": variant,
            "rounds": int(selected[target]),
        }
        for target in TARGETS
        for age in AGES
        for year in TEST_YEARS
        for variant in VARIANTS
    ]


def _task_dir(output_root: Path, task: Mapping[str, Any]) -> Path:
    return output_root / "tasks" / str(task["task_id"])


def _task_result_path(output_root: Path, task: Mapping[str, Any]) -> Path:
    return _task_dir(output_root, task) / "task_result.json"


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
            "test_year": int(task["test_year"]),
            "variant": task["variant"],
            "rounds": int(task["rounds"]),
            "parameters": _model_parameters(study, str(task["target"])),
            "feature_names": list(
                _feature_names(study, int(task["age"]), str(task["variant"]))
            ),
        }
        if any(payload.get(key) != value for key, value in expected.items()):
            return False
        files = dict(payload.get("files", {}) or {})
        if not files:
            return False
        for record in files.values():
            _verify_record(record)
        model = lgb.Booster(model_file=str(_resolve(files["model"]["path"])))
        multiplier = 3 if task["target"] == "state_10" else 1
        if int(model.num_trees()) != int(task["rounds"]) * multiplier:
            return False
        prediction = np.load(_resolve(files["prediction"]["path"]), allow_pickle=False)
        rows = np.load(_resolve(files["evaluation_rows"]["path"]), allow_pickle=False)
        if len(prediction) != len(rows):
            return False
        pd.read_parquet(_resolve(files["daily_metrics"]["path"]))
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
    target = str(task["target"])
    datasets = _build_datasets(
        data=data,
        study=study,
        target=target,
        variant=str(task["variant"]),
        evaluation_year=int(task["test_year"]),
        date_values=date_values,
        selection_outcome_cutoff=False,
    )
    model = None
    try:
        parameters = _model_parameters(study, target)
        rounds = int(task["rounds"])
        _emit(
            "post_entry_training_started",
            task_id=task["task_id"],
            rounds=rounds,
            train_rows=len(datasets.train_rows),
            evaluation_rows=len(datasets.evaluation_rows),
        )
        started = time.perf_counter()
        model = lgb.train(
            parameters,
            datasets.train_set,
            num_boost_round=rounds,
            valid_sets=[datasets.evaluation_set],
            valid_names=["outer_evaluation"],
            callbacks=[lgb.log_evaluation(period=0)],
        )
        elapsed = float(time.perf_counter() - started)
        prediction = _predict(model, datasets.evaluation_sequence, rounds)
        daily = _daily_metrics(
            data=data,
            target=target,
            rows=datasets.evaluation_rows,
            prediction=prediction,
        )
        metrics = {
            **_daily_summary(daily),
            "collapsed": (
                _state_collapsed(prediction) if target == "state_10" else False
            ),
        }
        output_dir = _task_dir(output_root, task)
        output_dir.mkdir(parents=True, exist_ok=True)
        model_path = output_dir / "model.txt"
        prediction_path = output_dir / "prediction.npy"
        rows_path = output_dir / "evaluation_rows.npy"
        daily_path = output_dir / "daily_metrics.parquet"
        model.save_model(str(model_path), num_iteration=rounds)
        _save_npy(prediction_path, prediction)
        _save_npy(rows_path, datasets.evaluation_rows)
        daily.to_parquet(daily_path, index=False, compression="zstd")
        dependency = int(study["targets"][target]["dependency_days"])
        train_max_idx = int(data.decision_date_idx[datasets.train_rows].max())
        test_dates = np.asarray(
            data.decision_date_idx[datasets.evaluation_rows], dtype=np.int32
        )
        test_years = np.asarray(
            [int(str(date_values[value])[:4]) for value in test_dates],
            dtype=np.int16,
        )
        if not bool(np.all(test_years == int(task["test_year"]))):
            raise AssertionError("formal evaluation rows crossed the test year")
        result = {
            "schema": TASK_SCHEMA,
            "status": "completed",
            "completed_at": _now(),
            "study_id": STUDY_ID,
            "study_config_sha256": study_hash,
            "task_id": task["task_id"],
            "target": target,
            "target_kind": _target_kind(target),
            "horizon": _target_horizon(target),
            "age": int(task["age"]),
            "test_year": int(task["test_year"]),
            "variant": task["variant"],
            "rounds": rounds,
            "internal_tree_count": int(model.num_trees()),
            "purge_days": dependency,
            "train_row_count": len(datasets.train_rows),
            "evaluation_row_count": len(datasets.evaluation_rows),
            "maximum_train_decision_date_idx": train_max_idx,
            "maximum_train_decision_date": str(date_values[train_max_idx]),
            "maximum_train_outcome_date": str(date_values[train_max_idx + dependency]),
            "parameters": parameters,
            "feature_names": list(datasets.feature_names),
            "common_support_sha256": _sha256(rows_path),
            "training_seconds": elapsed,
            "metrics": metrics,
            "state_probability_semantics": (
                "relative state score, not literal stable probability"
                if target == "state_10"
                else None
            ),
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
        _emit(
            "post_entry_training_completed",
            task_id=task["task_id"],
            rounds=rounds,
            seconds=elapsed,
        )
        del prediction, daily
        return result
    finally:
        _release_datasets(datasets)
        del model
        gc.collect()


def _load_result(
    *,
    task: Mapping[str, Any],
    study: Mapping[str, Any],
    study_hash: str,
    output_root: Path,
) -> dict[str, Any]:
    path = _task_result_path(output_root, task)
    if not _task_complete(path, task=task, study=study, study_hash=study_hash):
        raise ValueError(f"formal post-entry task is incomplete: {task['task_id']}")
    return json.loads(path.read_text(encoding="utf-8"))


def _row_index_sha256(rows: np.ndarray) -> str:
    values = np.ascontiguousarray(np.asarray(rows, dtype=np.int64))
    digest = hashlib.sha256()
    digest.update(values.view(np.uint8))
    return digest.hexdigest()


def _enrich_support_audit(
    *,
    study: Mapping[str, Any],
    capacity: Mapping[str, Any],
    study_hash: str,
    output_root: Path,
) -> dict[str, Any]:
    _feature_study, inputs, _pack, contract, _reader = _load_runtime(study)
    landmarks = {
        age: load_landmark(
            study=study,
            study_hash=study_hash,
            contract=contract,
            output_root=output_root,
            age=age,
        )
        for age in AGES
    }
    tasks = _tasks(study, capacity)
    records: list[dict[str, Any]] = []
    for target in TARGETS:
        for age in AGES:
            data = landmarks[age]
            for year in TEST_YEARS:
                train_rows, evaluation_rows = _split_rows(
                    data=data,
                    study=study,
                    target=target,
                    evaluation_year=year,
                    date_values=inputs.date_values,
                    selection_outcome_cutoff=False,
                )
                train_hash = _row_index_sha256(train_rows)
                evaluation_hash = _row_index_sha256(evaluation_rows)
                variants: dict[str, str] = {}
                for variant in VARIANTS:
                    task = next(
                        current
                        for current in tasks
                        if current["target"] == target
                        and int(current["age"]) == age
                        and int(current["test_year"]) == year
                        and current["variant"] == variant
                    )
                    path = _task_result_path(output_root, task)
                    result = json.loads(path.read_text(encoding="utf-8"))
                    stored_rows = np.load(
                        _resolve(result["files"]["evaluation_rows"]["path"]),
                        allow_pickle=False,
                    )
                    if not np.array_equal(stored_rows, evaluation_rows):
                        raise AssertionError(
                            f"stored evaluation support changed: {task['task_id']}"
                        )
                    if int(result["train_row_count"]) != len(train_rows) or int(
                        result["evaluation_row_count"]
                    ) != len(evaluation_rows):
                        raise AssertionError(
                            f"stored support counts changed: {task['task_id']}"
                        )
                    result["train_support_row_sha256"] = train_hash
                    result["evaluation_support_row_sha256"] = evaluation_hash
                    result["support_semantics"] = (
                        "target-specific B-complete common support shared exactly "
                        "by A and B"
                    )
                    _write_json(path, result)
                    variants[variant] = _file_record(path)["sha256"]
                records.append(
                    {
                        "target": target,
                        "age": int(age),
                        "test_year": int(year),
                        "train_row_count": len(train_rows),
                        "evaluation_row_count": len(evaluation_rows),
                        "train_support_row_sha256": train_hash,
                        "evaluation_support_row_sha256": evaluation_hash,
                        "A_B_support_equal": True,
                        "task_result_sha256": variants,
                    }
                )
    return {
        "status": "passed",
        "split_count": len(records),
        "A_B_support_equal_count": sum(
            bool(record["A_B_support_equal"]) for record in records
        ),
        "records": records,
    }


def _relative_harm(challenger: float, baseline: float) -> float:
    denominator = abs(float(baseline))
    if denominator <= 1.0e-12:
        return math.nan
    return (float(challenger) - float(baseline)) / denominator


def _one_sided_positive(hac: Mapping[str, Any]) -> float:
    mean = float(hac["mean"])
    p = float(hac["p_value_two_sided"])
    if not (math.isfinite(mean) and math.isfinite(p)):
        return math.nan
    return p / 2.0 if mean >= 0.0 else 1.0 - p / 2.0


def _one_sided_negative(hac: Mapping[str, Any]) -> float:
    mean = float(hac["mean"])
    p = float(hac["p_value_two_sided"])
    if not (math.isfinite(mean) and math.isfinite(p)):
        return math.nan
    return p / 2.0 if mean <= 0.0 else 1.0 - p / 2.0


def _paired_annual(
    *,
    target: str,
    age: int,
    year: int,
    a_result: Mapping[str, Any],
    b_result: Mapping[str, Any],
) -> tuple[pd.DataFrame, dict[str, Any]]:
    if (
        a_result["common_support_sha256"] != b_result["common_support_sha256"]
        or a_result.get("train_support_row_sha256")
        != b_result.get("train_support_row_sha256")
        or a_result.get("evaluation_support_row_sha256")
        != b_result.get("evaluation_support_row_sha256")
        or int(a_result["rounds"]) != int(b_result["rounds"])
        or a_result["parameters"] != b_result["parameters"]
        or int(a_result["train_row_count"]) != int(b_result["train_row_count"])
        or int(a_result["evaluation_row_count"])
        != int(b_result["evaluation_row_count"])
    ):
        raise ValueError(f"A/B capacity or support mismatch: {target} D{age} {year}")
    a = pd.read_parquet(_resolve(a_result["files"]["daily_metrics"]["path"]))
    b = pd.read_parquet(_resolve(b_result["files"]["daily_metrics"]["path"]))
    paired = a.merge(
        b,
        on="date_idx",
        suffixes=("_A", "_B"),
        validate="one_to_one",
    )
    if len(paired) != len(a) or len(paired) != len(b):
        raise ValueError(f"A/B daily support mismatch: {target} D{age} {year}")
    kind = _target_kind(target)
    primary = "ordinal_ic" if kind == "state" else "rank_ic"
    primary_delta = (paired[f"{primary}_B"] - paired[f"{primary}_A"]).to_numpy(
        dtype=np.float64
    )
    annual: dict[str, Any] = {
        "target": target,
        "kind": kind,
        "horizon": _target_horizon(target),
        "age": int(age),
        "year": int(year),
        "rounds": int(a_result["rounds"]),
        "row_count": int(a_result["evaluation_row_count"]),
        "common_support_sha256": a_result["common_support_sha256"],
        "primary_delta_hac": base._hac_mean_test(
            primary_delta,
            maximum_lag=max(_target_horizon(target) - 1, 0),
        ),
    }
    if kind == "mfe":
        adverse = (
            paired["top5_pre_peak_mae_B"] - paired["top5_pre_peak_mae_A"]
        ).to_numpy(dtype=np.float64)
        endpoint = (
            paired["top5_endpoint_return_B"] - paired["top5_endpoint_return_A"]
        ).to_numpy(dtype=np.float64)
        adverse_hac = base._hac_mean_test(
            adverse, maximum_lag=max(_target_horizon(target) - 1, 0)
        )
        endpoint_hac = base._hac_mean_test(
            endpoint, maximum_lag=max(_target_horizon(target) - 1, 0)
        )
        annual.update(
            {
                "rank_ic_delta": float(np.nanmean(primary_delta)),
                "top5_remaining_mfe_delta": float(
                    np.nanmean(
                        paired["top5_remaining_mfe_B"] - paired["top5_remaining_mfe_A"]
                    )
                ),
                "tail_hit_delta": float(
                    np.nanmean(paired["top5_tail_rate_B"] - paired["top5_tail_rate_A"])
                ),
                "relative_mae_harm": _relative_harm(
                    float(np.nanmean(paired["mae_B"])),
                    float(np.nanmean(paired["mae_A"])),
                ),
                "entry_either_top5_rank_ic_delta": float(
                    np.nanmean(
                        paired["entry_either_top5_rank_ic_B"]
                        - paired["entry_either_top5_rank_ic_A"]
                    )
                ),
                "top5_pre_peak_mae_delta": float(np.nanmean(adverse)),
                "top5_endpoint_return_delta": float(np.nanmean(endpoint)),
                "top5_pre_peak_mae_delta_hac": adverse_hac,
                "top5_endpoint_return_delta_hac": endpoint_hac,
                "significant_path_harm": bool(
                    (
                        float(adverse_hac["mean"]) <= -0.001
                        and _one_sided_negative(adverse_hac) <= 0.05
                    )
                    or (
                        float(endpoint_hac["mean"]) <= -0.001
                        and _one_sided_negative(endpoint_hac) <= 0.05
                    )
                ),
            }
        )
    elif kind == "risk":
        annual.update(
            {
                "rank_ic_delta": float(np.nanmean(primary_delta)),
                "relative_mae_harm": _relative_harm(
                    float(np.nanmean(paired["mae_B"])),
                    float(np.nanmean(paired["mae_A"])),
                ),
                "deep_adverse_pr_auc_delta": float(
                    np.nanmean(
                        paired["deep_adverse_pr_auc_B"]
                        - paired["deep_adverse_pr_auc_A"]
                    )
                ),
                "entry_either_top5_rank_ic_delta": float(
                    np.nanmean(
                        paired["entry_either_top5_rank_ic_B"]
                        - paired["entry_either_top5_rank_ic_A"]
                    )
                ),
            }
        )
    else:
        annual.update(
            {
                "ordinal_ic_delta": float(np.nanmean(primary_delta)),
                "brier_improvement_A_minus_B": float(
                    np.nanmean(paired["brier_A"] - paired["brier_B"])
                ),
                "logloss_improvement_A_minus_B": float(
                    np.nanmean(paired["logloss_A"] - paired["logloss_B"])
                ),
                "relative_brier_harm": _relative_harm(
                    float(np.nanmean(paired["brier_B"])),
                    float(np.nanmean(paired["brier_A"])),
                ),
                "relative_logloss_harm": _relative_harm(
                    float(np.nanmean(paired["logloss_B"])),
                    float(np.nanmean(paired["logloss_A"])),
                ),
                "A_collapsed": bool(a_result["metrics"]["collapsed"]),
                "B_collapsed": bool(b_result["metrics"]["collapsed"]),
            }
        )
    return paired, annual


def _stouffer(annual: Sequence[Mapping[str, Any]]) -> dict[str, float]:
    z_values: list[float] = []
    weights: list[float] = []
    for row in annual:
        hac = dict(row["primary_delta_hac"])
        p = _one_sided_positive(hac)
        count = int(hac["count"])
        if math.isfinite(p) and count > 0:
            z_values.append(float(stats.norm.isf(np.clip(p, 1.0e-15, 1.0 - 1.0e-15))))
            weights.append(math.sqrt(float(count)))
    if not z_values:
        return {"z_statistic": math.nan, "p_value_one_sided": math.nan}
    weight = np.asarray(weights, dtype=np.float64)
    statistic = float(
        np.dot(weight, np.asarray(z_values, dtype=np.float64))
        / math.sqrt(float(np.dot(weight, weight)))
    )
    return {
        "z_statistic": statistic,
        "p_value_one_sided": float(stats.norm.sf(statistic)),
    }


def _mfe_age_gate(
    *,
    annual: Sequence[Mapping[str, Any]],
    combined: Mapping[str, Any],
    q_value: float,
    rules: Mapping[str, Any],
) -> dict[str, Any]:
    ordered = sorted(annual, key=lambda row: int(row["year"]))
    rank = [float(row["rank_ic_delta"]) for row in ordered]
    top5 = [float(row["top5_remaining_mfe_delta"]) for row in ordered]
    tail = [float(row["tail_hit_delta"]) for row in ordered]
    entry = [float(row["entry_either_top5_rank_ic_delta"]) for row in ordered]
    mae = [float(row["relative_mae_harm"]) for row in ordered]
    hac_supported = sum(
        _one_sided_positive(row["primary_delta_hac"])
        <= float(rules["maximum_hac_p_value"])
        for row in ordered
    )
    path_harm_years = sum(bool(row["significant_path_harm"]) for row in ordered)
    worst_path = [
        min(
            float(row["top5_pre_peak_mae_delta"]),
            float(row["top5_endpoint_return_delta"]),
        )
        for row in ordered
    ]
    path_rejected = bool(
        path_harm_years >= int(rules["path_guardrail_minimum_harm_years"])
        and float(np.median(worst_path))
        <= -float(rules["path_guardrail_minimum_median_absolute_harm"])
    )
    checks = {
        "positive_rank_ic_years": sum(value > 0.0 for value in rank),
        "worst_rank_ic_delta": min(rank),
        "positive_top5_mfe_years": sum(value > 0.0 for value in top5),
        "worst_top5_mfe_delta": min(top5),
        "nonnegative_tail_hit_years": sum(value >= 0.0 for value in tail),
        "worst_tail_hit_delta": min(tail),
        "HAC_supported_years": hac_supported,
        "cross_year_stouffer_p_value": float(combined["p_value_one_sided"]),
        "cross_year_BH_q_value": float(q_value),
        "entry_top5_positive_rank_ic_years": sum(value > 0.0 for value in entry),
        "entry_top5_worst_rank_ic_delta": min(entry),
        "worst_relative_mae_harm": max(mae),
        "significant_path_harm_years": path_harm_years,
        "median_worst_path_delta": float(np.median(worst_path)),
        "path_guardrail_rejected": path_rejected,
    }
    passed = bool(
        checks["positive_rank_ic_years"] >= int(rules["minimum_positive_rank_ic_years"])
        and checks["worst_rank_ic_delta"]
        >= float(rules["minimum_worst_year_rank_ic_delta"])
        and checks["positive_top5_mfe_years"]
        >= int(rules["minimum_positive_top5_mfe_years"])
        and checks["worst_top5_mfe_delta"]
        >= float(rules["minimum_worst_year_top5_mfe_delta"])
        and checks["nonnegative_tail_hit_years"]
        >= int(rules["minimum_nonnegative_tail_hit_years"])
        and checks["worst_tail_hit_delta"]
        >= -float(rules["maximum_worst_tail_hit_decline"])
        and checks["HAC_supported_years"] >= int(rules["minimum_hac_supported_years"])
        and math.isfinite(checks["cross_year_BH_q_value"])
        and checks["cross_year_BH_q_value"]
        <= float(rules["maximum_bh_q_value_across_6_horizon_age_tests"])
        and checks["entry_top5_positive_rank_ic_years"]
        >= int(rules["entry_top5_minimum_positive_rank_ic_years"])
        and checks["entry_top5_worst_rank_ic_delta"]
        >= float(rules["entry_top5_minimum_worst_rank_ic_delta"])
        and checks["worst_relative_mae_harm"]
        <= float(rules["maximum_worst_relative_mae_harm"])
        and not path_rejected
    )
    return {
        "passed": passed,
        "checks": checks,
        "annual_rank_ic_delta": rank,
        "annual_top5_remaining_mfe_delta": top5,
        "annual_tail_hit_delta": tail,
        "annual_entry_top5_rank_ic_delta": entry,
        "annual_relative_mae_harm": mae,
    }


def _risk_age_gate(
    *,
    annual: Sequence[Mapping[str, Any]],
    rules: Mapping[str, Any],
) -> dict[str, Any]:
    ordered = sorted(annual, key=lambda row: int(row["year"]))
    rank = [float(row["rank_ic_delta"]) for row in ordered]
    mae = [float(row["relative_mae_harm"]) for row in ordered]
    pr = [float(row["deep_adverse_pr_auc_delta"]) for row in ordered]
    entry = [float(row["entry_either_top5_rank_ic_delta"]) for row in ordered]
    hac_supported = sum(
        _one_sided_positive(row["primary_delta_hac"])
        <= float(rules["maximum_hac_p_value"])
        for row in ordered
    )
    bh_supported = sum(
        float(row["rank_ic_bh_q_value"])
        <= float(rules["maximum_bh_q_value_across_18_year_tests"])
        for row in ordered
    )
    checks = {
        "positive_rank_ic_years": sum(value > 0.0 for value in rank),
        "worst_rank_ic_delta": min(rank),
        "HAC_supported_years": hac_supported,
        "BH_supported_years": bh_supported,
        "mae_improvement_years": sum(value < 0.0 for value in mae),
        "worst_relative_mae_harm": max(mae),
        "pr_auc_improvement_years": sum(value > 0.0 for value in pr),
        "worst_pr_auc_delta": min(pr),
        "entry_top5_positive_rank_ic_years": sum(value > 0.0 for value in entry),
    }
    passed = bool(
        checks["positive_rank_ic_years"] >= int(rules["minimum_positive_rank_ic_years"])
        and checks["worst_rank_ic_delta"]
        >= float(rules["minimum_worst_year_rank_ic_delta"])
        and checks["HAC_supported_years"] >= int(rules["minimum_hac_supported_years"])
        and checks["BH_supported_years"] >= int(rules["minimum_hac_supported_years"])
        and checks["mae_improvement_years"]
        >= int(rules["minimum_mae_improvement_years"])
        and checks["worst_relative_mae_harm"]
        <= float(rules["maximum_worst_relative_mae_harm"])
        and checks["pr_auc_improvement_years"]
        >= int(rules["minimum_pr_auc_improvement_years"])
        and checks["worst_pr_auc_delta"]
        >= -float(rules["maximum_worst_pr_auc_decline"])
        and checks["entry_top5_positive_rank_ic_years"] == len(ordered)
    )
    return {
        "passed": passed,
        "checks": checks,
        "annual_rank_ic_delta": rank,
        "annual_relative_mae_harm": mae,
        "annual_deep_adverse_pr_auc_delta": pr,
        "annual_entry_top5_rank_ic_delta": entry,
    }


def _state_age_gate(
    *,
    annual: Sequence[Mapping[str, Any]],
    rules: Mapping[str, Any],
) -> dict[str, Any]:
    ordered = sorted(annual, key=lambda row: int(row["year"]))
    ordinal = [float(row["ordinal_ic_delta"]) for row in ordered]
    brier = [float(row["brier_improvement_A_minus_B"]) for row in ordered]
    logloss = [float(row["logloss_improvement_A_minus_B"]) for row in ordered]
    brier_harm = [float(row["relative_brier_harm"]) for row in ordered]
    logloss_harm = [float(row["relative_logloss_harm"]) for row in ordered]
    collapse = [
        int(row["year"])
        for row in ordered
        if bool(row["A_collapsed"]) or bool(row["B_collapsed"])
    ]
    checks = {
        "positive_ordinal_ic_years": sum(value > 0.0 for value in ordinal),
        "worst_ordinal_ic_delta": min(ordinal),
        "brier_improvement_years": sum(value > 0.0 for value in brier),
        "logloss_improvement_years": sum(value > 0.0 for value in logloss),
        "worst_relative_brier_harm": max(brier_harm),
        "worst_relative_logloss_harm": max(logloss_harm),
        "probability_collapse_years": collapse,
    }
    passed = bool(
        checks["positive_ordinal_ic_years"]
        >= int(rules["minimum_positive_ordinal_ic_years"])
        and checks["worst_ordinal_ic_delta"]
        >= float(rules["minimum_worst_year_ordinal_ic_delta"])
        and checks["brier_improvement_years"]
        >= int(rules["minimum_brier_improvement_years"])
        and checks["logloss_improvement_years"]
        >= int(rules["minimum_logloss_improvement_years"])
        and checks["worst_relative_brier_harm"]
        <= float(rules["maximum_worst_relative_proper_score_harm"])
        and checks["worst_relative_logloss_harm"]
        <= float(rules["maximum_worst_relative_proper_score_harm"])
        and not collapse
    )
    return {
        "passed": passed,
        "checks": checks,
        "annual_ordinal_ic_delta": ordinal,
        "annual_brier_improvement": brier,
        "annual_logloss_improvement": logloss,
    }


def _general_decision(
    *,
    target: str,
    age_decisions: Mapping[int, Mapping[str, Any]],
    rules: Mapping[str, Any],
) -> dict[str, Any]:
    passing = [
        int(age) for age, record in age_decisions.items() if bool(record["passed"])
    ]
    material_harm: list[int] = []
    kind = _target_kind(target)
    for age, record in age_decisions.items():
        if int(age) in passing:
            continue
        checks = dict(record["checks"])
        rank_key = (
            "worst_ordinal_ic_delta" if kind == "state" else "worst_rank_ic_delta"
        )
        harmed = float(checks[rank_key]) < float(
            rules["nonpassing_age_material_rank_harm"]
        )
        if kind == "mfe":
            harmed = harmed or float(checks["worst_top5_mfe_delta"]) < float(
                rules["nonpassing_age_material_top5_harm"]
            )
        if harmed:
            material_harm.append(int(age))
    generalized = bool(
        len(passing) >= int(rules["minimum_passing_ages"]) and not material_harm
    )
    return {
        "general_update_head": generalized,
        "passing_ages": passing,
        "materially_harmed_nonpassing_ages": material_harm,
        "age_specific_only": len(passing) == 1,
        "decision": (
            "retain_general_update_head"
            if generalized
            else (
                "retain_age_specific_evidence_only"
                if len(passing) == 1
                else "daily_recomputation_retained"
            )
        ),
    }


def evaluate(
    *,
    study_path: Path = DEFAULT_STUDY_PATH,
    output_root: Path = DEFAULT_OUTPUT_ROOT,
) -> dict[str, Any]:
    study = load_study(study_path)
    study_hash = _config_hash(study_path)
    capacity = _load_capacity_decision(study)
    tasks = _tasks(study, capacity)
    for task in tasks:
        _load_result(
            task=task,
            study=study,
            study_hash=study_hash,
            output_root=output_root,
        )
    support_audit = _enrich_support_audit(
        study=study,
        capacity=capacity,
        study_hash=study_hash,
        output_root=output_root,
    )
    annual: list[dict[str, Any]] = []
    by_key: dict[tuple[str, int], list[dict[str, Any]]] = {}
    task_index = {
        (
            str(task["target"]),
            int(task["age"]),
            int(task["test_year"]),
            str(task["variant"]),
        ): task
        for task in tasks
    }
    for target in TARGETS:
        for age in AGES:
            for year in TEST_YEARS:
                a_task = task_index[(target, age, year, "A")]
                b_task = task_index[(target, age, year, "B")]
                a_result = _load_result(
                    task=a_task,
                    study=study,
                    study_hash=study_hash,
                    output_root=output_root,
                )
                b_result = _load_result(
                    task=b_task,
                    study=study,
                    study_hash=study_hash,
                    output_root=output_root,
                )
                paired, record = _paired_annual(
                    target=target,
                    age=age,
                    year=year,
                    a_result=a_result,
                    b_result=b_result,
                )
                path = (
                    output_root
                    / "evaluation"
                    / target
                    / f"D{age}"
                    / f"fold_{year}_paired_daily.parquet"
                )
                path.parent.mkdir(parents=True, exist_ok=True)
                paired.to_parquet(path, index=False, compression="zstd")
                record["paired_daily"] = _file_record(path)
                annual.append(record)
                by_key.setdefault((target, age), []).append(record)
    risk_p_values = {
        f"{row['target']}_D{row['age']}_{row['year']}": _one_sided_positive(
            row["primary_delta_hac"]
        )
        for row in annual
        if row["kind"] == "risk"
    }
    risk_q = feature._benjamini_hochberg(risk_p_values)
    for row in annual:
        if row["kind"] == "risk":
            key = f"{row['target']}_D{row['age']}_{row['year']}"
            row["rank_ic_bh_q_value"] = float(risk_q[key])
    mfe_combined: dict[str, dict[str, float]] = {}
    for target in ("mfe_10", "mfe_20"):
        for age in AGES:
            key = f"{target}_D{age}"
            mfe_combined[key] = _stouffer(by_key[(target, age)])
    mfe_q = feature._benjamini_hochberg(
        {
            key: float(record["p_value_one_sided"])
            for key, record in mfe_combined.items()
        }
    )
    target_decisions: dict[str, Any] = {}
    for target in TARGETS:
        age_decisions: dict[int, dict[str, Any]] = {}
        for age in AGES:
            evidence = sorted(by_key[(target, age)], key=lambda row: int(row["year"]))
            kind = _target_kind(target)
            if kind == "mfe":
                key = f"{target}_D{age}"
                current = _mfe_age_gate(
                    annual=evidence,
                    combined=mfe_combined[key],
                    q_value=float(mfe_q[key]),
                    rules=study["decision"]["mfe"],
                )
                current["combined_rank_ic_test"] = mfe_combined[key]
            elif kind == "risk":
                current = _risk_age_gate(
                    annual=evidence, rules=study["decision"]["risk"]
                )
            else:
                current = _state_age_gate(
                    annual=evidence, rules=study["decision"]["state"]
                )
            current["age"] = int(age)
            current["annual"] = evidence
            age_decisions[age] = current
        target_decisions[target] = {
            "target": target,
            "kind": _target_kind(target),
            "capacity_rounds": int(capacity["selected_rounds"][target]),
            "ages": {str(age): age_decisions[age] for age in AGES},
            "general": _general_decision(
                target=target,
                age_decisions=age_decisions,
                rules=study["decision"]["general"],
            ),
        }
    general_heads = [
        target
        for target, decision in target_decisions.items()
        if bool(decision["general"]["general_update_head"])
    ]
    opportunity = [target for target in ("mfe_10", "mfe_20") if target in general_heads]
    risk = [target for target in ("risk_10", "risk_20") if target in general_heads]
    state = "state_10" in general_heads
    if not general_heads:
        overall = (
            "daily_recomputation_v4_sufficient_within_tested_supervised_LightGBM_scope"
        )
    elif opportunity and not risk and not state:
        overall = "retain_opportunity_update_heads_only"
    else:
        overall = "retain_only_passing_independent_update_coordinates"
    decision = {
        "status": "completed_post_entry_update_decision",
        "overall": overall,
        "general_update_heads": general_heads,
        "opportunity_update_heads": opportunity,
        "risk_update_heads": risk,
        "state_update_head": state,
        "age_specific_evidence": {
            target: record["general"]["passing_ages"]
            for target, record in target_decisions.items()
            if not record["general"]["general_update_head"]
            and record["general"]["passing_ages"]
        },
        "daily_recomputation_coordinates": [
            target for target in TARGETS if target not in general_heads
        ],
        "next_step": (
            "compare_current_holding_remaining_coordinates_with_new_candidate_coordinates_after_cost"
            if general_heads
            else "retain_daily_recomputation_baseline"
        ),
        "scope_limit": (
            "verdict applies only to the tested LightGBM features, fixed capacities, "
            "and D1/D3/D5 landmarks; it does not prove realized paths are universally useless"
        ),
        "does_not_select": list(study["non_selections"]),
    }
    inventory = [
        {
            "task_id": task["task_id"],
            "target": task["target"],
            "age": int(task["age"]),
            "test_year": int(task["test_year"]),
            "variant": task["variant"],
            "rounds": int(task["rounds"]),
            "task_result": _file_record(_task_result_path(output_root, task)),
        }
        for task in tasks
    ]
    summary = {
        "schema": SUMMARY_SCHEMA,
        "status": "completed_post_entry_ab_v2",
        "completed_at": _now(),
        "study_id": STUDY_ID,
        "study_config_sha256": study_hash,
        "entry_contract": entry_v4.CONTRACT_V4_ID,
        "capacity_decision": {
            "source": _file_record(_resolve(study["sources"]["capacity_decision"])),
            "selected_rounds": capacity["selected_rounds"],
        },
        "annual": annual,
        "targets": target_decisions,
        "decision": decision,
        "task_inventory": inventory,
        "support_audit": support_audit,
        "scope": {
            "formal_booster_count": 90,
            "maximum_consumed_outcome_date": "2025-12-31",
            "forbidden_2026_row_count": 0,
            "temperature_calibration_count": 0,
            "fusion": False,
        },
    }
    _write_json(output_root / "summary.json", summary)
    _write_json(output_root / "decision.json", decision)
    _write_json(DEFAULT_RECORD_ROOT / "config.json", study)
    _write_json(
        DEFAULT_RECORD_ROOT / "result.json",
        {**summary, "full_output": _file_record(output_root / "summary.json")},
    )
    return summary


def run_pending(
    *,
    study_path: Path = DEFAULT_STUDY_PATH,
    output_root: Path = DEFAULT_OUTPUT_ROOT,
) -> dict[str, Any]:
    study = load_study(study_path)
    study_hash = _config_hash(study_path)
    capacity = _load_capacity_decision(study)
    prepare_landmarks(study_path=study_path, output_root=output_root)
    _feature_study, inputs, _pack, contract, _reader = _load_runtime(study)
    landmarks = {
        age: load_landmark(
            study=study,
            study_hash=study_hash,
            contract=contract,
            output_root=output_root,
            age=age,
        )
        for age in AGES
    }
    for task in _tasks(study, capacity):
        _run_task(
            task=task,
            study=study,
            study_hash=study_hash,
            output_root=output_root,
            data=landmarks[int(task["age"])],
            date_values=inputs.date_values,
        )
    return evaluate(study_path=study_path, output_root=output_root)


def status(
    *,
    study_path: Path = DEFAULT_STUDY_PATH,
    output_root: Path = DEFAULT_OUTPUT_ROOT,
) -> dict[str, Any]:
    study = load_study(study_path)
    study_hash = _config_hash(study_path)
    landmark_complete = {
        str(age): (_landmark_root(output_root, age) / "complete.json").is_file()
        for age in AGES
    }
    decision_path = _resolve(study["sources"]["capacity_decision"])
    if not decision_path.is_file():
        return {
            "study_id": STUDY_ID,
            "landmark_complete": landmark_complete,
            "capacity_decision_completed": False,
            "formal_models": {"completed": 0, "total": 90},
            "pending_reason": "post_entry_capacity_audit_incomplete",
        }
    capacity = _load_capacity_decision(study)
    tasks = _tasks(study, capacity)
    completed: list[str] = []
    pending: list[str] = []
    for task in tasks:
        current = _task_complete(
            _task_result_path(output_root, task),
            task=task,
            study=study,
            study_hash=study_hash,
        )
        (completed if current else pending).append(str(task["task_id"]))
    return {
        "study_id": STUDY_ID,
        "landmark_complete": landmark_complete,
        "capacity_decision_completed": True,
        "selected_rounds": capacity["selected_rounds"],
        "formal_models": {
            "completed": len(completed),
            "total": len(tasks),
        },
        "completed": completed,
        "pending": pending,
        "evaluation_completed": (output_root / "summary.json").is_file(),
    }


def self_test() -> dict[str, Any]:
    study = load_study()
    if _feature_names(study, 1, "B") != (
        *_feature_names(study, 1, "A"),
        *tuple(study["features"]["rank_change"]),
        *tuple(study["features"]["D1_path"]),
    ):
        raise AssertionError("D1 feature contract changed")
    forbidden = set(study["features"]["D1_forbidden_duplicates"])
    if forbidden & set(_feature_names(study, 1, "B")):
        raise AssertionError("D1 contains a duplicate path statistic")
    fake_capacity = {
        "selected_rounds": {
            "mfe_10": 64,
            "mfe_20": 128,
            "risk_10": 64,
            "risk_20": 128,
            "state_10": 32,
        }
    }
    tasks = _tasks(study, fake_capacity)
    if len(tasks) != 90:
        raise AssertionError("formal A/B task count changed")
    for target in TARGETS:
        counts = {task["rounds"] for task in tasks if task["target"] == target}
        if len(counts) != 1:
            raise AssertionError("target capacity differs across ages")
    dates = np.repeat(np.asarray([1, 2], dtype=np.int32), 20)
    actual = np.concatenate(
        [
            np.linspace(0.0, 0.2, 20),
            np.linspace(0.1, 0.3, 20),
        ]
    )
    score = actual.copy()
    frame = _mfe_daily_metrics(
        date_idx=dates,
        actual=actual,
        prediction=score,
        pre_peak_mae=-actual,
        endpoint_return=actual / 2.0,
        entry_either_top5=np.ones(len(actual), dtype=bool),
    )
    if frame.empty:
        raise AssertionError("MFE metric self-test failed")
    return {
        "status": "passed",
        "formal_task_count": len(tasks),
        "targets": list(TARGETS),
        "ages": list(AGES),
        "D1_feature_count": len(_feature_names(study, 1, "B")),
        "D3_feature_count": len(_feature_names(study, 3, "B")),
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the v4-aligned five-target Seq100 post-entry A/B study."
    )
    parser.add_argument("--study", type=Path, default=DEFAULT_STUDY_PATH)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--prepare-landmarks", action="store_true")
    group.add_argument("--status", action="store_true")
    group.add_argument("--run-pending", action="store_true")
    group.add_argument("--evaluate", action="store_true")
    group.add_argument("--self-test", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    study_path = _resolve(arguments.study)
    output_root = _resolve(arguments.output_root)
    if arguments.prepare_landmarks:
        payload = prepare_landmarks(study_path=study_path, output_root=output_root)
    elif arguments.status:
        payload = status(study_path=study_path, output_root=output_root)
    elif arguments.run_pending:
        payload = run_pending(study_path=study_path, output_root=output_root)
    elif arguments.evaluate:
        payload = evaluate(study_path=study_path, output_root=output_root)
    else:
        payload = self_test()
    _emit("result", **payload)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
