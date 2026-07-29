from __future__ import annotations

import argparse
import gc
import hashlib
import importlib.util
import json
import math
import os
import sys
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

from daily_research.path_policy import seq100_mfe_feature_family_audit as feature
from daily_research.path_policy import seq100_mfe_feature_union_audit as union
from daily_research.path_policy import seq100_path_label_learnability as base
from daily_research.path_policy import seq100_signal_quality as signal_quality

WORKSPACE_ROOT = Path(__file__).resolve().parents[2]
STUDY_ID = "seq100_entry_contract_oos_v2"
DEFAULT_STUDY_PATH = (
    WORKSPACE_ROOT / "daily_research/studies/seq100_entry_contract_oos_v2.json"
)
DEFAULT_OUTPUT_ROOT = (
    WORKSPACE_ROOT
    / "daily_research/output/path_policy/studies/seq100_entry_contract_oos_v2"
)
DEFAULT_RECORD_ROOT = (
    WORKSPACE_ROOT
    / "daily_research/research_records/seq100/seq100_entry_contract_oos_v2"
)
CONTRACT_YEARS = (2020, 2021, 2022, 2023, 2024, 2025)
DECISION_YEARS = (2023, 2024, 2025)
MFE_HORIZONS = (10, 20)
PHYSICAL_COLUMNS = (
    "mfe_10",
    "mfe_20",
    "state_low_10",
    "state_mid_10",
    "state_high_10",
    "pre_peak_mae_10",
    "pre_peak_mae_20",
)
TASK_SCHEMA = "seq100_entry_contract_oos_task/v2"
MANIFEST_SCHEMA = "seq100_entry_contract_oos_manifest/v2"


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


def load_study(path: Path = DEFAULT_STUDY_PATH) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("study_id") != STUDY_ID:
        raise ValueError(f"study_id must be {STUDY_ID}")
    folds = dict(payload["folds"])
    if tuple(int(value) for value in folds["contract_years"]) != CONTRACT_YEARS:
        raise ValueError("contract years changed")
    if tuple(int(value) for value in folds["decision_years"]) != DECISION_YEARS:
        raise ValueError("decision years changed")
    if folds["maximum_outcome_date"] != "2025-12-31":
        raise ValueError("entry contract may not consume outcomes after 2025")
    if int(folds["forbidden_outcome_year"]) != 2026:
        raise ValueError("2026 must remain forbidden")
    if tuple(payload["entry_contract"]["physical_columns"]) != PHYSICAL_COLUMNS:
        raise ValueError("physical entry-contract columns changed")
    mapping = {
        int(year): int(cutoff)
        for year, cutoff in payload["state_and_risk"][
            "atlas_cutoff_by_test_year"
        ].items()
    }
    expected = {2020: 2019, 2021: 2020, 2022: 2021, 2023: 2022, 2024: 2022, 2025: 2022}
    if mapping != expected:
        raise ValueError("state atlas cutoff mapping changed")
    return payload


def _load_module(path: Path, name: str) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _year_rows(inputs: base.LearnabilityInputs, year: int) -> np.ndarray:
    calendar_rows = np.flatnonzero(
        np.char.startswith(np.asarray(inputs.date_values, dtype=str), f"{int(year)}-")
    )
    if not calendar_rows.size:
        raise ValueError(f"calendar lacks year {year}")
    left = int(
        np.searchsorted(inputs.candidate_date_idx, int(calendar_rows[0]), side="left")
    )
    right = int(
        np.searchsorted(inputs.candidate_date_idx, int(calendar_rows[-1]), side="right")
    )
    rows = np.arange(left, right, dtype=np.int64)
    if not rows.size:
        raise ValueError(f"candidate universe lacks year {year}")
    dates = np.asarray(inputs.candidate_date_idx[rows], dtype=np.int32)
    if bool(np.any(dates[1:] < dates[:-1])):
        raise AssertionError("candidate year rows are not date ordered")
    return rows


def _rank_by_date(values: np.ndarray, date_idx: np.ndarray) -> np.ndarray:
    current = np.asarray(values, dtype=np.float64)
    dates = np.asarray(date_idx, dtype=np.int32)
    output = np.full(len(current), np.nan, dtype=np.float32)
    boundaries = np.flatnonzero(np.r_[True, dates[1:] != dates[:-1], True])
    for start, stop in pairwise(boundaries):
        finite = np.isfinite(current[start:stop])
        if not bool(finite.any()):
            continue
        ranked = stats.rankdata(current[start:stop][finite], method="average")
        denominator = max(len(ranked) - 1, 1)
        output[start:stop][finite] = ((ranked - 1.0) / denominator).astype(np.float32)
    return output


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
    if target == "state":
        parameters.update(
            {"objective": "multiclass", "metric": "multi_logloss", "num_class": 3}
        )
    elif target in {"mfe", "pre_peak_mae"}:
        parameters.update(
            {
                "objective": "huber",
                "metric": "huber",
                "alpha": float(model["huber_alpha"]),
            }
        )
    else:
        raise ValueError(f"unsupported target: {target}")
    return (
        parameters,
        int(model["num_boost_round"]),
        int(model["early_stopping_rounds"]),
    )


def _load_atlas_model(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as stored:
        model = {name: np.asarray(stored[name]) for name in stored.files}
    if int(model["horizon"].reshape(-1)[0]) != 10:
        raise ValueError("only the D10 state atlas is valid for the entry contract")
    if int(model["fixed_k"].reshape(-1)[0]) != 3:
        raise ValueError("state atlas is not fixed K3")
    raw_centers = np.asarray(model["fixed_k3_centers"], dtype=np.float32) @ np.asarray(
        model["pca_components"], dtype=np.float32
    ) + np.asarray(model["pca_mean"], dtype=np.float32)
    raw_centers = raw_centers * model["scale"] + model["center"]
    order = np.argsort(raw_centers[:, -1], kind="mergesort")
    raw_to_state = np.empty(3, dtype=np.int8)
    raw_to_state[order] = np.arange(3, dtype=np.int8)
    model["raw_to_state"] = raw_to_state
    model["raw_center_endpoint"] = raw_centers[:, -1].astype(np.float32)
    return model


def _assign_state(path: np.ndarray, model: Mapping[str, np.ndarray]) -> np.ndarray:
    values = np.asarray(path, dtype=np.float32)
    scaled = (
        np.clip(values, model["clip_low"], model["clip_high"]) - model["center"]
    ) / model["scale"]
    transformed = (
        scaled - np.asarray(model["pca_mean"], dtype=np.float32)
    ) @ np.asarray(model["pca_components"], dtype=np.float32).T
    centers = np.asarray(model["fixed_k3_centers"], dtype=np.float32)
    squared = (
        np.square(transformed).sum(axis=1, keepdims=True)
        - 2.0 * transformed @ centers.T
        + np.square(centers).sum(axis=1)[None, :]
    )
    raw = np.argmin(squared, axis=1)
    return np.asarray(model["raw_to_state"], dtype=np.int8)[raw]


def prepare_atlas(
    *,
    study: Mapping[str, Any],
    output_root: Path,
    inputs: base.LearnabilityInputs,
    pack: Mapping[str, Any],
) -> dict[str, Any]:
    sources = dict(study["sources"])
    atlas_script = _resolve(sources["atlas_script"])
    atlas_root = _resolve(sources["atlas_output_root"])
    atlas = _load_module(atlas_script, "seq100_contract_atlas_source")
    # The historical atlas builder predates the workspace API simplification.
    # Supply its old read-only parser name locally without restoring a project-wide
    # compatibility layer.
    from daily_research.path_policy import seq100_candidate_execution as execution

    if not hasattr(execution, "parse_execution_cost_contract"):
        execution.parse_execution_cost_contract = execution.parse_execution_costs
    source_module = atlas._load_source_module()
    atlas_source = source_module.PathAtlasSource()
    paths, status, _macro = atlas._open_arrays(source_module, atlas_source)
    atlas._fit_window_horizon(
        module=source_module,
        source=atlas_source,
        paths=paths,
        status=status,
        horizon=10,
        window_year=2019,
        output_root=atlas_root,
    )
    model_paths = {
        cutoff: atlas_root / "models" / "h10" / f"through_{cutoff}" / "model.npz"
        for cutoff in (2019, 2020, 2021, 2022)
    }
    for path in model_paths.values():
        if not path.is_file():
            raise FileNotFoundError(path)

    reference_model = _load_atlas_model(model_paths[2022])
    reference_indices = np.asarray(reference_model["sample_indices"], dtype=np.int64)
    if len(reference_indices) > 100_000:
        rng = np.random.default_rng(20260729)
        reference_indices = np.sort(
            rng.choice(reference_indices, size=100_000, replace=False)
        )
    reference_path = np.asarray(paths[reference_indices, :10], dtype=np.float32)
    models = {year: _load_atlas_model(path) for year, path in model_paths.items()}
    from sklearn.metrics import adjusted_rand_score

    assignments = {
        year: _assign_state(reference_path, model) for year, model in models.items()
    }
    raw_centers: dict[int, np.ndarray] = {}
    for year, model in models.items():
        raw = np.asarray(model["fixed_k3_centers"], dtype=np.float32) @ np.asarray(
            model["pca_components"], dtype=np.float32
        ) + np.asarray(model["pca_mean"], dtype=np.float32)
        raw_centers[year] = raw * model["scale"] + model["center"]
    stability: list[dict[str, Any]] = []
    for left, right in ((2019, 2020), (2020, 2021), (2021, 2022)):
        matched, rmse, correlation = atlas._matched_centroid_metrics(
            raw_centers[left], raw_centers[right]
        )
        stability.append(
            {
                "left_atlas_cutoff": left,
                "right_atlas_cutoff": right,
                "assignment_ari": float(
                    adjusted_rand_score(assignments[left], assignments[right])
                ),
                "matched_centroid_count": matched,
                "matched_centroid_rmse": rmse,
                "matched_centroid_correlation": correlation,
                "common_reference_path_count": len(reference_indices),
            }
        )
    rules = dict(study["state_and_risk"]["atlas_stability"])
    state_formal = all(
        row["assignment_ari"] >= float(rules["minimum_adjacent_assignment_ari"])
        and row["matched_centroid_correlation"]
        >= float(rules["minimum_adjacent_centroid_correlation"])
        for row in stability
    )

    from daily_research.path_policy.seq100_post_entry_incremental_information import (
        HoldingPathReader,
    )

    reader = HoldingPathReader(pack, inputs)
    labels_root = output_root / "state_labels"
    label_records: dict[str, Any] = {}
    existing_valid = np.asarray(inputs.label_values("state", 10)) >= 0
    for cutoff in (2019, 2020, 2021):
        label_path = labels_root / f"state10_through_{cutoff}.int8.dat"
        complete_path = labels_root / f"state10_through_{cutoff}.json"
        model = models[cutoff]
        model_hash = _sha256(model_paths[cutoff])
        if complete_path.is_file() and label_path.is_file():
            completed = json.loads(complete_path.read_text(encoding="utf-8"))
            expected_size = inputs.candidate_count * np.dtype(np.int8).itemsize
            if (
                completed.get("status") == "completed"
                and completed.get("model_sha256") == model_hash
                and label_path.stat().st_size == expected_size
            ):
                label_records[str(cutoff)] = completed
                continue
        labels_root.mkdir(parents=True, exist_ok=True)
        temporary = label_path.with_suffix(label_path.suffix + ".tmp")
        labels = np.memmap(
            temporary, dtype=np.int8, mode="w+", shape=(inputs.candidate_count,)
        )
        labels[:] = -1
        assigned = 0
        dates = np.asarray(inputs.candidate_date_idx, dtype=np.int32)
        boundaries = np.flatnonzero(np.r_[True, dates[1:] != dates[:-1], True])
        for position, (left, right) in enumerate(pairwise(boundaries)):
            valid = existing_valid[left:right]
            if bool(valid.any()):
                date_idx = int(dates[left])
                symbols = np.asarray(
                    inputs.candidate_symbol_idx[left:right][valid], dtype=np.int64
                )
                future = reader._future(date_idx, symbols, 10)
                anchor = 1.0 + np.asarray(future[:, 0, 0], dtype=np.float64)
                close = (
                    np.divide(
                        1.0 + np.asarray(future[:, :10, 3], dtype=np.float64),
                        anchor[:, None],
                        out=np.full((len(symbols), 10), np.nan, dtype=np.float64),
                        where=np.isfinite(anchor[:, None]) & (anchor[:, None] > 1.0e-8),
                    )
                    - 1.0
                )
                if not bool(np.isfinite(close).all()):
                    raise AssertionError(
                        "existing valid state lacks a complete close path"
                    )
                local = np.flatnonzero(valid) + left
                labels[local] = _assign_state(close, model)
                assigned += len(local)
            if position and position % 250 == 0:
                labels.flush()
        labels.flush()
        del labels
        os.replace(temporary, label_path)
        completed = {
            "status": "completed",
            "atlas_cutoff": cutoff,
            "model_sha256": model_hash,
            "candidate_count": inputs.candidate_count,
            "assigned_count": assigned,
            "file": _file_record(
                label_path, shape=[inputs.candidate_count], dtype="int8"
            ),
            "maximum_outcome_date_read": "2025-12-31",
            "forbidden_2026_row_count": 0,
            "completed_at": _now(),
        }
        _write_json(complete_path, completed)
        label_records[str(cutoff)] = completed
        _emit("state_label_atlas_completed", atlas_cutoff=cutoff, assigned=assigned)
    label_records["2022"] = {
        "status": "reused_existing_candidate_labels",
        "atlas_cutoff": 2022,
        "model_sha256": _sha256(model_paths[2022]),
        "assigned_count": int(existing_valid.sum()),
        "file": inputs.label_manifest["files"]["state_labels"],
    }
    result = {
        "status": "completed",
        "state_challenge_formal": state_formal,
        "stability": stability,
        "models": {str(year): _file_record(path) for year, path in model_paths.items()},
        "label_sources": label_records,
        "maximum_outcome_date_read": "2025-12-31",
        "forbidden_2026_row_count": 0,
        "completed_at": _now(),
    }
    _write_json(output_root / "atlas_manifest.json", result)
    return result


def _final_mfe_heads(study: Mapping[str, Any]) -> dict[int, dict[str, Any]]:
    sources = dict(study["sources"])
    decision_path = _resolve(sources["union_output_root"]) / "decision.json"
    if not decision_path.is_file():
        raise RuntimeError(
            "feature-union decision must complete before freezing MFE heads"
        )
    decision = json.loads(decision_path.read_text(encoding="utf-8"))
    if decision.get("status") != "completed_feature_union_audit":
        raise RuntimeError(
            f"feature-union decision is not valid: {decision.get('status')}"
        )
    union_study = union.load_study(_resolve(sources["union_study_config"]))
    definitions = union._variant_definitions(union_study)
    result: dict[int, dict[str, Any]] = {}
    for horizon in MFE_HORIZONS:
        variant = str(decision["final_mfe_heads"][str(horizon)])
        incumbent = str(union_study["heads"][str(horizon)]["incumbent"])
        if variant == incumbent:
            families = (incumbent,)
            source_kind = "feature_family"
        elif variant in definitions[horizon]:
            families = tuple(definitions[horizon][variant]["families"])
            source_kind = "feature_union"
        else:
            raise ValueError(f"unknown frozen MFE head: D{horizon} {variant}")
        if "deterministic_noise" in families:
            raise ValueError(
                "a negative-control family cannot become a frozen MFE head"
            )
        result[horizon] = {
            "variant": variant,
            "families": families,
            "source_kind": source_kind,
        }
    return result


def _task_dir(output_root: Path, task_id: str) -> Path:
    return output_root / "tasks" / task_id


def _task_result_path(output_root: Path, task_id: str) -> Path:
    return _task_dir(output_root, task_id) / "task_result.json"


def _training_tasks(study: Mapping[str, Any]) -> list[dict[str, Any]]:
    atlas_mapping = {
        int(year): int(cutoff)
        for year, cutoff in study["state_and_risk"]["atlas_cutoff_by_test_year"].items()
    }
    tasks: list[dict[str, Any]] = []
    for horizon in MFE_HORIZONS:
        for year in (2020, 2021, 2022):
            tasks.extend(
                [
                    {
                        "task_id": f"mfe{horizon}_{year}_tuning",
                        "family": "mfe",
                        "target": "mfe",
                        "horizon": horizon,
                        "year": year,
                        "stage": "tuning",
                        "atlas_cutoff": None,
                    },
                    {
                        "task_id": f"mfe{horizon}_{year}_outer",
                        "family": "mfe",
                        "target": "mfe",
                        "horizon": horizon,
                        "year": year,
                        "stage": "outer",
                        "atlas_cutoff": None,
                    },
                ]
            )
    for target, horizon in (("state", 10), ("pre_peak_mae", 10), ("pre_peak_mae", 20)):
        target_name = "state10" if target == "state" else f"risk{horizon}"
        for year in CONTRACT_YEARS:
            tasks.extend(
                [
                    {
                        "task_id": f"{target_name}_{year}_tuning",
                        "family": target_name,
                        "target": target,
                        "horizon": horizon,
                        "year": year,
                        "stage": "tuning",
                        "atlas_cutoff": atlas_mapping[year]
                        if target == "state"
                        else None,
                    },
                    {
                        "task_id": f"{target_name}_{year}_outer",
                        "family": target_name,
                        "target": target,
                        "horizon": horizon,
                        "year": year,
                        "stage": "outer",
                        "atlas_cutoff": atlas_mapping[year]
                        if target == "state"
                        else None,
                    },
                ]
            )
    return tasks


def _inference_tasks() -> list[dict[str, Any]]:
    return [
        {
            "task_id": f"mfe{horizon}_{year}_reuse_inference",
            "family": "mfe",
            "target": "mfe",
            "horizon": horizon,
            "year": year,
            "stage": "reuse_inference",
            "atlas_cutoff": None,
        }
        for horizon in MFE_HORIZONS
        for year in DECISION_YEARS
    ]


def _state_values(
    *, inputs: base.LearnabilityInputs, output_root: Path, atlas_cutoff: int
) -> np.ndarray:
    if int(atlas_cutoff) == 2022:
        return np.asarray(inputs.label_values("state", 10))
    path = (
        output_root / "state_labels" / f"state10_through_{int(atlas_cutoff)}.int8.dat"
    )
    if not path.is_file():
        raise FileNotFoundError(path)
    return np.memmap(path, dtype=np.int8, mode="r", shape=(inputs.candidate_count,))


def _target_values(
    *, task: Mapping[str, Any], inputs: base.LearnabilityInputs, output_root: Path
) -> np.ndarray:
    if task["target"] == "state":
        return _state_values(
            inputs=inputs,
            output_root=output_root,
            atlas_cutoff=int(task["atlas_cutoff"]),
        )
    return inputs.label_values(str(task["target"]), int(task["horizon"]))


def _valid_target(values: np.ndarray, target: str) -> np.ndarray:
    if target == "state":
        return np.isin(np.asarray(values), (0, 1, 2))
    return np.isfinite(np.asarray(values, dtype=np.float64))


def _task_parameters(
    *,
    task: Mapping[str, Any],
    study: Mapping[str, Any],
    feature_study: Mapping[str, Any],
) -> dict[str, Any]:
    if task["family"] == "mfe":
        return feature._model_parameters(feature_study)[0]
    return _model_parameters(study, str(task["target"]))[0]


def _task_complete(
    path: Path,
    *,
    task: Mapping[str, Any],
    study: Mapping[str, Any],
    feature_study: Mapping[str, Any],
) -> bool:
    if not path.is_file():
        return False
    try:
        import lightgbm as lgb

        result = json.loads(path.read_text(encoding="utf-8"))
        expected = {
            "schema": TASK_SCHEMA,
            "status": "completed",
            "study_id": STUDY_ID,
            "task_id": task["task_id"],
            "family": task["family"],
            "target": task["target"],
            "horizon": int(task["horizon"]),
            "model_year": int(task["year"]),
            "stage": task["stage"],
            "atlas_cutoff": task["atlas_cutoff"],
            "parameters": _task_parameters(
                task=task, study=study, feature_study=feature_study
            ),
        }
        if any(result.get(key) != value for key, value in expected.items()):
            return False
        files = dict(result.get("files", {}) or {})
        for record in files.values():
            current = _resolve(record["path"])
            if (
                not current.is_file()
                or current.stat().st_size != int(record["size"])
                or _sha256(current) != record["sha256"]
            ):
                return False
        if "model" in files:
            lgb.Booster(model_file=str(_resolve(files["model"]["path"])))
        if "prediction" in files:
            prediction = np.load(
                _resolve(files["prediction"]["path"]), allow_pickle=False
            )
            if list(prediction.shape) != list(files["prediction"]["shape"]):
                return False
        if "candidate_rows" in files:
            rows = np.load(
                _resolve(files["candidate_rows"]["path"]), allow_pickle=False
            )
            if list(rows.shape) != list(files["candidate_rows"]["shape"]):
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


def _predict(model: Any, sequence: Any, iterations: int) -> np.ndarray:
    first = np.asarray(model.predict(sequence[0:1], num_iteration=int(iterations)))
    shape = (len(sequence),) if first.ndim == 1 else (len(sequence), first.shape[1])
    output = np.empty(shape, dtype=np.float32)
    for left in range(0, len(sequence), 250_000):
        right = min(left + 250_000, len(sequence))
        output[left:right] = np.asarray(
            model.predict(sequence[left:right], num_iteration=int(iterations)),
            dtype=np.float32,
        )
    return output


def _base_sequence(
    *,
    inputs: base.LearnabilityInputs,
    rows: np.ndarray,
    vocabularies: Sequence[np.ndarray],
    study: Mapping[str, Any],
) -> Any:
    sequence = signal_quality._make_lgb_sequence(
        continuous=inputs.continuous,
        categorical=inputs.categorical,
        row_ids=np.asarray(rows, dtype=np.int64),
        continuous_columns=inputs.continuous_columns,
        categorical_columns=inputs.categorical_columns,
        category_vocabularies=vocabularies,
        batch_size=int(study["model"]["sequence_batch_size"]),
    )
    return base._memory_trimmed_sequence(sequence, study)


def _save_vocabularies(path: Path, values: Sequence[np.ndarray]) -> dict[str, Any]:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as handle:
        np.savez(
            handle, **{f"category_{index}": value for index, value in enumerate(values)}
        )
    os.replace(temporary, path)
    return _file_record(path, category_count=len(values))


def _metrics(
    *,
    task: Mapping[str, Any],
    inputs: base.LearnabilityInputs,
    rows: np.ndarray,
    actual: np.ndarray,
    prediction: np.ndarray,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    if task["target"] == "state":
        return base.state_metrics(
            date_idx=inputs.candidate_date_idx[rows],
            actual=np.asarray(actual),
            probability=np.asarray(prediction),
            date_values=inputs.date_values,
            horizon=int(task["horizon"]),
        )
    return base.regression_metrics(
        date_idx=inputs.candidate_date_idx[rows],
        actual=np.asarray(actual),
        prediction=np.asarray(prediction),
        date_values=inputs.date_values,
        horizon=int(task["horizon"]),
    )


@dataclass
class _BuiltBaseDataset:
    datasets: base.LightGBMDatasets
    evaluation_raw: np.ndarray
    fold: base.FoldRows


def _build_base_target_dataset(
    *,
    task: Mapping[str, Any],
    study: Mapping[str, Any],
    inputs: base.LearnabilityInputs,
    output_root: Path,
    split_year: int,
) -> _BuiltBaseDataset:
    horizon = int(task["horizon"])
    fold = inputs.common_path_rows(split_year, horizon)
    values = _target_values(task=task, inputs=inputs, output_root=output_root)
    train_raw = np.asarray(values[fold.train_rows])
    evaluation_raw = np.asarray(values[fold.evaluation_rows])
    train_label, train_weight = base.aligned_target_weights(
        values=train_raw,
        date_idx=inputs.candidate_date_idx[fold.train_rows],
        valid=_valid_target(train_raw, str(task["target"])),
    )
    evaluation_label, evaluation_weight = base.aligned_target_weights(
        values=evaluation_raw,
        date_idx=inputs.candidate_date_idx[fold.evaluation_rows],
        valid=_valid_target(evaluation_raw, str(task["target"])),
    )
    datasets = base.build_lgb_datasets(
        inputs=inputs,
        train_rows=fold.train_rows,
        evaluation_rows=fold.evaluation_rows,
        initial_train_label=train_label,
        initial_evaluation_label=evaluation_label,
        initial_train_weight=train_weight,
        initial_evaluation_weight=evaluation_weight,
        study=study,
    )
    return _BuiltBaseDataset(
        datasets=datasets, evaluation_raw=evaluation_raw, fold=fold
    )


def _run_mfe_tuning(
    *,
    task: Mapping[str, Any],
    study: Mapping[str, Any],
    feature_study: Mapping[str, Any],
    inputs: base.LearnabilityInputs,
    output_root: Path,
) -> dict[str, Any]:
    import lightgbm as lgb

    result_path = _task_result_path(output_root, str(task["task_id"]))
    if _task_complete(result_path, task=task, study=study, feature_study=feature_study):
        return json.loads(result_path.read_text(encoding="utf-8"))
    validation_year = int(task["year"]) - 1
    horizon = int(task["horizon"])
    fold = inputs.common_path_rows(validation_year, horizon)
    datasets, _weight, evaluation_raw = feature._build_datasets(
        study=feature_study,
        inputs=inputs,
        train_rows=fold.train_rows,
        evaluation_rows=fold.evaluation_rows,
        horizon=horizon,
        extra=None,
        extra_names=(),
    )
    parameters, maximum_rounds, patience = feature._model_parameters(feature_study)
    _emit("task_training_started", task_id=task["task_id"], stage="tuning")
    started = time.perf_counter()
    model = lgb.train(
        parameters,
        datasets.train_set,
        num_boost_round=maximum_rounds,
        valid_sets=[datasets.evaluation_set],
        valid_names=["inner_validation"],
        callbacks=[
            lgb.early_stopping(
                stopping_rounds=patience, first_metric_only=True, verbose=False
            )
        ],
    )
    elapsed = float(time.perf_counter() - started)
    iterations = int(model.best_iteration)
    output_dir = result_path.parent
    output_dir.mkdir(parents=True, exist_ok=True)
    model_path = output_dir / "model.txt"
    model.save_model(str(model_path), num_iteration=iterations)
    prediction = feature._predict(model, datasets.evaluation_sequence, iterations)
    prediction_path = output_dir / "prediction.npy"
    rows_path = output_dir / "candidate_rows.npy"
    _save_npy(prediction_path, prediction)
    _save_npy(rows_path, fold.evaluation_rows)
    daily, metrics = base.regression_metrics(
        date_idx=inputs.candidate_date_idx[fold.evaluation_rows],
        actual=evaluation_raw,
        prediction=prediction,
        date_values=inputs.date_values,
        horizon=horizon,
    )
    daily_path = output_dir / "daily_metrics.parquet"
    daily.to_parquet(daily_path, index=False, compression="zstd")
    result = {
        "schema": TASK_SCHEMA,
        "status": "completed",
        "completed_at": _now(),
        "study_id": STUDY_ID,
        "task_id": task["task_id"],
        "family": task["family"],
        "target": task["target"],
        "horizon": horizon,
        "model_year": int(task["year"]),
        "stage": task["stage"],
        "atlas_cutoff": None,
        "inner_validation_year": validation_year,
        "train_row_count": len(fold.train_rows),
        "evaluation_row_count": len(fold.evaluation_rows),
        "maximum_train_signal_date_idx": fold.maximum_train_signal_date_idx,
        "best_iteration": iterations,
        "training_seconds": elapsed,
        "parameters": parameters,
        "metrics": metrics,
        "files": {
            "model": _file_record(model_path),
            "prediction": _file_record(
                prediction_path,
                shape=list(prediction.shape),
                dtype=str(prediction.dtype),
            ),
            "candidate_rows": _file_record(
                rows_path,
                shape=list(fold.evaluation_rows.shape),
                dtype=str(fold.evaluation_rows.dtype),
            ),
            "daily_metrics": _file_record(daily_path),
        },
    }
    _write_json(result_path, result)
    feature._release_datasets(datasets)
    del model, prediction, fold
    _emit(
        "task_training_completed",
        task_id=task["task_id"],
        best_iteration=iterations,
        seconds=elapsed,
    )
    return result


def _run_mfe_outer(
    *,
    task: Mapping[str, Any],
    study: Mapping[str, Any],
    feature_study: Mapping[str, Any],
    inputs: base.LearnabilityInputs,
    output_root: Path,
    final_heads: Mapping[int, Mapping[str, Any]],
) -> dict[str, Any]:
    import lightgbm as lgb

    result_path = _task_result_path(output_root, str(task["task_id"]))
    if _task_complete(result_path, task=task, study=study, feature_study=feature_study):
        return json.loads(result_path.read_text(encoding="utf-8"))
    horizon = int(task["horizon"])
    year = int(task["year"])
    tuning_id = f"mfe{horizon}_{year}_tuning"
    tuning_path = _task_result_path(output_root, tuning_id)
    tuning = json.loads(tuning_path.read_text(encoding="utf-8"))
    iterations = int(tuning["best_iteration"])
    head = dict(final_heads[horizon])
    feature_root = _resolve(study["sources"]["feature_output_root"])
    extra, manifests = union._open_union(
        feature_output_root=feature_root,
        families=head["families"],
        candidate_count=inputs.candidate_count,
    )
    fold = inputs.common_path_rows(year, horizon)
    datasets, _weight, evaluation_raw = feature._build_datasets(
        study=feature_study,
        inputs=inputs,
        train_rows=fold.train_rows,
        evaluation_rows=fold.evaluation_rows,
        horizon=horizon,
        extra=extra,
        extra_names=extra.feature_names,
    )
    parameters = feature._model_parameters(feature_study)[0]
    _emit(
        "task_training_started",
        task_id=task["task_id"],
        stage="outer",
        iterations=iterations,
    )
    started = time.perf_counter()
    model = lgb.train(
        parameters,
        datasets.train_set,
        num_boost_round=iterations,
        valid_sets=[datasets.evaluation_set],
        valid_names=["outer_evaluation"],
    )
    elapsed = float(time.perf_counter() - started)
    evaluation_prediction = feature._predict(
        model, datasets.evaluation_sequence, iterations
    )
    daily, metrics = base.regression_metrics(
        date_idx=inputs.candidate_date_idx[fold.evaluation_rows],
        actual=evaluation_raw,
        prediction=evaluation_prediction,
        date_values=inputs.date_values,
        horizon=horizon,
    )
    vocabularies = signal_quality._fit_category_vocabularies(
        inputs.categorical, fold.train_rows, inputs.categorical_columns
    )
    candidate_rows = _year_rows(inputs, year)
    sequence = feature._combined_sequence(
        inputs=inputs,
        row_ids=candidate_rows,
        category_vocabularies=vocabularies,
        batch_size=int(feature_study["model"]["sequence_batch_size"]),
        extra=extra,
    )
    sequence = base._memory_trimmed_sequence(sequence, feature_study)
    prediction = _predict(model, sequence, iterations)
    output_dir = result_path.parent
    output_dir.mkdir(parents=True, exist_ok=True)
    model_path = output_dir / "model.txt"
    prediction_path = output_dir / "prediction.npy"
    rows_path = output_dir / "candidate_rows.npy"
    daily_path = output_dir / "daily_metrics.parquet"
    vocabulary_path = output_dir / "category_vocabularies.npz"
    model.save_model(str(model_path), num_iteration=iterations)
    _save_npy(prediction_path, prediction)
    _save_npy(rows_path, candidate_rows)
    daily.to_parquet(daily_path, index=False, compression="zstd")
    vocabulary_record = _save_vocabularies(vocabulary_path, vocabularies)
    result = {
        "schema": TASK_SCHEMA,
        "status": "completed",
        "completed_at": _now(),
        "study_id": STUDY_ID,
        "task_id": task["task_id"],
        "family": task["family"],
        "target": task["target"],
        "horizon": horizon,
        "model_year": year,
        "stage": task["stage"],
        "atlas_cutoff": None,
        "head_variant": head["variant"],
        "head_families": list(head["families"]),
        "family_manifests": manifests,
        "inner_validation_year": year - 1,
        "train_row_count": len(fold.train_rows),
        "evaluation_row_count": len(fold.evaluation_rows),
        "candidate_prediction_row_count": len(candidate_rows),
        "maximum_train_signal_date_idx": fold.maximum_train_signal_date_idx,
        "purge_days": horizon,
        "best_iteration": iterations,
        "training_seconds": elapsed,
        "parameters": parameters,
        "metrics": metrics,
        "files": {
            "model": _file_record(model_path),
            "prediction": _file_record(
                prediction_path,
                shape=list(prediction.shape),
                dtype=str(prediction.dtype),
            ),
            "candidate_rows": _file_record(
                rows_path,
                shape=list(candidate_rows.shape),
                dtype=str(candidate_rows.dtype),
            ),
            "category_vocabularies": vocabulary_record,
            "daily_metrics": _file_record(daily_path),
        },
    }
    _write_json(result_path, result)
    feature._release_datasets(datasets)
    del model, prediction, evaluation_prediction, sequence, extra, fold
    gc.collect()
    _emit("task_training_completed", task_id=task["task_id"], seconds=elapsed)
    return result


def _run_base_tuning(
    *,
    task: Mapping[str, Any],
    study: Mapping[str, Any],
    feature_study: Mapping[str, Any],
    inputs: base.LearnabilityInputs,
    output_root: Path,
) -> dict[str, Any]:
    import lightgbm as lgb

    result_path = _task_result_path(output_root, str(task["task_id"]))
    if _task_complete(result_path, task=task, study=study, feature_study=feature_study):
        return json.loads(result_path.read_text(encoding="utf-8"))
    validation_year = int(task["year"]) - 1
    built = _build_base_target_dataset(
        task=task,
        study=study,
        inputs=inputs,
        output_root=output_root,
        split_year=validation_year,
    )
    parameters, maximum_rounds, patience = _model_parameters(study, str(task["target"]))
    _emit("task_training_started", task_id=task["task_id"], stage="tuning")
    started = time.perf_counter()
    model = lgb.train(
        parameters,
        built.datasets.train_set,
        num_boost_round=maximum_rounds,
        valid_sets=[built.datasets.evaluation_set],
        valid_names=["inner_validation"],
        callbacks=[
            lgb.early_stopping(
                stopping_rounds=patience, first_metric_only=True, verbose=False
            )
        ],
    )
    elapsed = float(time.perf_counter() - started)
    iterations = int(model.best_iteration)
    prediction = _predict(model, built.datasets.evaluation_sequence, iterations)
    daily, metrics = _metrics(
        task=task,
        inputs=inputs,
        rows=built.fold.evaluation_rows,
        actual=built.evaluation_raw,
        prediction=prediction,
    )
    output_dir = result_path.parent
    output_dir.mkdir(parents=True, exist_ok=True)
    model_path = output_dir / "model.txt"
    prediction_path = output_dir / "prediction.npy"
    rows_path = output_dir / "candidate_rows.npy"
    daily_path = output_dir / "daily_metrics.parquet"
    vocabulary_path = output_dir / "category_vocabularies.npz"
    model.save_model(str(model_path), num_iteration=iterations)
    _save_npy(prediction_path, prediction)
    _save_npy(rows_path, built.fold.evaluation_rows)
    daily.to_parquet(daily_path, index=False, compression="zstd")
    vocabulary_record = _save_vocabularies(
        vocabulary_path, built.datasets.category_vocabularies
    )
    result = {
        "schema": TASK_SCHEMA,
        "status": "completed",
        "completed_at": _now(),
        "study_id": STUDY_ID,
        "task_id": task["task_id"],
        "family": task["family"],
        "target": task["target"],
        "horizon": int(task["horizon"]),
        "model_year": int(task["year"]),
        "stage": task["stage"],
        "atlas_cutoff": task["atlas_cutoff"],
        "inner_validation_year": validation_year,
        "train_row_count": len(built.fold.train_rows),
        "evaluation_row_count": len(built.fold.evaluation_rows),
        "maximum_train_signal_date_idx": built.fold.maximum_train_signal_date_idx,
        "purge_days": int(task["horizon"]),
        "best_iteration": iterations,
        "training_seconds": elapsed,
        "parameters": parameters,
        "metrics": metrics,
        "files": {
            "model": _file_record(model_path),
            "prediction": _file_record(
                prediction_path,
                shape=list(prediction.shape),
                dtype=str(prediction.dtype),
            ),
            "candidate_rows": _file_record(
                rows_path,
                shape=list(built.fold.evaluation_rows.shape),
                dtype=str(built.fold.evaluation_rows.dtype),
            ),
            "category_vocabularies": vocabulary_record,
            "daily_metrics": _file_record(daily_path),
        },
    }
    _write_json(result_path, result)
    base._release_training_memory(built.datasets)
    del model, prediction, built
    gc.collect()
    _emit(
        "task_training_completed",
        task_id=task["task_id"],
        best_iteration=iterations,
        seconds=elapsed,
    )
    return result


def _run_base_outer(
    *,
    task: Mapping[str, Any],
    study: Mapping[str, Any],
    feature_study: Mapping[str, Any],
    inputs: base.LearnabilityInputs,
    output_root: Path,
) -> dict[str, Any]:
    import lightgbm as lgb

    result_path = _task_result_path(output_root, str(task["task_id"]))
    if _task_complete(result_path, task=task, study=study, feature_study=feature_study):
        return json.loads(result_path.read_text(encoding="utf-8"))
    year = int(task["year"])
    tuning_id = f"{task['family']}_{year}_tuning"
    tuning = json.loads(
        _task_result_path(output_root, tuning_id).read_text(encoding="utf-8")
    )
    iterations = int(tuning["best_iteration"])
    built = _build_base_target_dataset(
        task=task,
        study=study,
        inputs=inputs,
        output_root=output_root,
        split_year=year,
    )
    parameters = _model_parameters(study, str(task["target"]))[0]
    _emit(
        "task_training_started",
        task_id=task["task_id"],
        stage="outer",
        iterations=iterations,
    )
    started = time.perf_counter()
    model = lgb.train(
        parameters,
        built.datasets.train_set,
        num_boost_round=iterations,
        valid_sets=[built.datasets.evaluation_set],
        valid_names=["outer_evaluation"],
    )
    elapsed = float(time.perf_counter() - started)
    evaluation_prediction = _predict(
        model, built.datasets.evaluation_sequence, iterations
    )
    daily, metrics = _metrics(
        task=task,
        inputs=inputs,
        rows=built.fold.evaluation_rows,
        actual=built.evaluation_raw,
        prediction=evaluation_prediction,
    )
    candidate_rows = _year_rows(inputs, year)
    sequence = _base_sequence(
        inputs=inputs,
        rows=candidate_rows,
        vocabularies=built.datasets.category_vocabularies,
        study=study,
    )
    prediction = _predict(model, sequence, iterations)
    output_dir = result_path.parent
    output_dir.mkdir(parents=True, exist_ok=True)
    model_path = output_dir / "model.txt"
    prediction_path = output_dir / "prediction.npy"
    rows_path = output_dir / "candidate_rows.npy"
    daily_path = output_dir / "daily_metrics.parquet"
    vocabulary_path = output_dir / "category_vocabularies.npz"
    model.save_model(str(model_path), num_iteration=iterations)
    _save_npy(prediction_path, prediction)
    _save_npy(rows_path, candidate_rows)
    daily.to_parquet(daily_path, index=False, compression="zstd")
    vocabulary_record = _save_vocabularies(
        vocabulary_path, built.datasets.category_vocabularies
    )
    result = {
        "schema": TASK_SCHEMA,
        "status": "completed",
        "completed_at": _now(),
        "study_id": STUDY_ID,
        "task_id": task["task_id"],
        "family": task["family"],
        "target": task["target"],
        "horizon": int(task["horizon"]),
        "model_year": year,
        "stage": task["stage"],
        "atlas_cutoff": task["atlas_cutoff"],
        "inner_validation_year": year - 1,
        "train_row_count": len(built.fold.train_rows),
        "evaluation_row_count": len(built.fold.evaluation_rows),
        "candidate_prediction_row_count": len(candidate_rows),
        "maximum_train_signal_date_idx": built.fold.maximum_train_signal_date_idx,
        "purge_days": int(task["horizon"]),
        "best_iteration": iterations,
        "training_seconds": elapsed,
        "parameters": parameters,
        "metrics": metrics,
        "files": {
            "model": _file_record(model_path),
            "prediction": _file_record(
                prediction_path,
                shape=list(prediction.shape),
                dtype=str(prediction.dtype),
            ),
            "candidate_rows": _file_record(
                rows_path,
                shape=list(candidate_rows.shape),
                dtype=str(candidate_rows.dtype),
            ),
            "category_vocabularies": vocabulary_record,
            "daily_metrics": _file_record(daily_path),
        },
    }
    _write_json(result_path, result)
    base._release_training_memory(built.datasets)
    del model, prediction, evaluation_prediction, sequence, built
    gc.collect()
    _emit("task_training_completed", task_id=task["task_id"], seconds=elapsed)
    return result


def _run_mfe_reuse_inference(
    *,
    task: Mapping[str, Any],
    study: Mapping[str, Any],
    feature_study: Mapping[str, Any],
    inputs: base.LearnabilityInputs,
    output_root: Path,
    final_heads: Mapping[int, Mapping[str, Any]],
) -> dict[str, Any]:
    import lightgbm as lgb

    result_path = _task_result_path(output_root, str(task["task_id"]))
    if _task_complete(result_path, task=task, study=study, feature_study=feature_study):
        return json.loads(result_path.read_text(encoding="utf-8"))
    horizon = int(task["horizon"])
    year = int(task["year"])
    head = dict(final_heads[horizon])
    feature_root = _resolve(study["sources"]["feature_output_root"])
    union_root = _resolve(study["sources"]["union_output_root"])
    if head["source_kind"] == "feature_family":
        source_result, _source_prediction, source_rows = feature._load_outer_result(
            feature_root,
            year=year,
            horizon=horizon,
            variant=str(head["variant"]),
            study=feature_study,
        )
    else:
        source_task = next(
            current
            for current in union._tasks(
                union.load_study(_resolve(study["sources"]["union_study_config"]))
            )
            if int(current["year"]) == year
            and int(current["horizon"]) == horizon
            and current["variant"] == head["variant"]
        )
        source_result, _source_prediction, source_rows = union._load_union_result(
            union_root, task=source_task
        )
    source_model_path = _resolve(source_result["files"]["model"]["path"])
    model = lgb.Booster(model_file=str(source_model_path))
    iterations = int(
        source_result.get(
            "best_iteration",
            source_result["fixed_iteration_source"]["best_iteration"],
        )
    )
    fold = inputs.common_path_rows(year, horizon)
    vocabularies = signal_quality._fit_category_vocabularies(
        inputs.categorical, fold.train_rows, inputs.categorical_columns
    )
    extra, manifests = union._open_union(
        feature_output_root=feature_root,
        families=head["families"],
        candidate_count=inputs.candidate_count,
    )
    candidate_rows = _year_rows(inputs, year)
    sequence = feature._combined_sequence(
        inputs=inputs,
        row_ids=candidate_rows,
        category_vocabularies=vocabularies,
        batch_size=int(feature_study["model"]["sequence_batch_size"]),
        extra=extra,
    )
    sequence = base._memory_trimmed_sequence(sequence, feature_study)
    prediction = _predict(model, sequence, iterations)
    positions = np.searchsorted(candidate_rows, np.asarray(source_rows, dtype=np.int64))
    if bool(np.any(positions >= len(candidate_rows))) or not np.array_equal(
        candidate_rows[positions], source_rows
    ):
        raise AssertionError(
            "reused MFE evaluation rows are outside candidate-year inference"
        )
    evaluation_prediction = prediction[positions]
    actual = np.asarray(inputs.label_values("mfe", horizon)[source_rows])
    daily, metrics = base.regression_metrics(
        date_idx=inputs.candidate_date_idx[source_rows],
        actual=actual,
        prediction=evaluation_prediction,
        date_values=inputs.date_values,
        horizon=horizon,
    )
    output_dir = result_path.parent
    output_dir.mkdir(parents=True, exist_ok=True)
    prediction_path = output_dir / "prediction.npy"
    rows_path = output_dir / "candidate_rows.npy"
    daily_path = output_dir / "daily_metrics.parquet"
    vocabulary_path = output_dir / "category_vocabularies.npz"
    _save_npy(prediction_path, prediction)
    _save_npy(rows_path, candidate_rows)
    daily.to_parquet(daily_path, index=False, compression="zstd")
    vocabulary_record = _save_vocabularies(vocabulary_path, vocabularies)
    result = {
        "schema": TASK_SCHEMA,
        "status": "completed",
        "completed_at": _now(),
        "study_id": STUDY_ID,
        "task_id": task["task_id"],
        "family": task["family"],
        "target": task["target"],
        "horizon": horizon,
        "model_year": year,
        "stage": task["stage"],
        "atlas_cutoff": None,
        "head_variant": head["variant"],
        "head_families": list(head["families"]),
        "family_manifests": manifests,
        "source_task_id": source_result["task_id"],
        "source_model": _file_record(source_model_path),
        "inner_validation_year": int(
            source_result["fixed_iteration_source"]["inner_validation_year"]
        ),
        "train_row_count": int(source_result["train_row_count"]),
        "evaluation_row_count": len(source_rows),
        "candidate_prediction_row_count": len(candidate_rows),
        "maximum_train_signal_date_idx": fold.maximum_train_signal_date_idx,
        "purge_days": horizon,
        "best_iteration": iterations,
        "training_seconds": 0.0,
        "parameters": feature._model_parameters(feature_study)[0],
        "metrics": metrics,
        "files": {
            "model": _file_record(source_model_path),
            "prediction": _file_record(
                prediction_path,
                shape=list(prediction.shape),
                dtype=str(prediction.dtype),
            ),
            "candidate_rows": _file_record(
                rows_path,
                shape=list(candidate_rows.shape),
                dtype=str(candidate_rows.dtype),
            ),
            "category_vocabularies": vocabulary_record,
            "daily_metrics": _file_record(daily_path),
        },
    }
    _write_json(result_path, result)
    del model, prediction, evaluation_prediction, sequence, extra, fold
    gc.collect()
    _emit("reuse_inference_completed", task_id=task["task_id"])
    return result


def _load_task_arrays(output_root: Path, task_id: str) -> tuple[np.ndarray, np.ndarray]:
    result = json.loads(
        _task_result_path(output_root, task_id).read_text(encoding="utf-8")
    )
    rows = np.load(
        _resolve(result["files"]["candidate_rows"]["path"]), allow_pickle=False
    )
    prediction = np.load(
        _resolve(result["files"]["prediction"]["path"]), allow_pickle=False
    )
    return np.asarray(rows, dtype=np.int64), np.asarray(prediction, dtype=np.float32)


def _daily_spearman(
    left: np.ndarray, right: np.ndarray, date_idx: np.ndarray
) -> tuple[float, int]:
    x = np.asarray(left, dtype=np.float64)
    y = np.asarray(right, dtype=np.float64)
    dates = np.asarray(date_idx, dtype=np.int32)
    values: list[float] = []
    boundaries = np.flatnonzero(np.r_[True, dates[1:] != dates[:-1], True])
    for start, stop in pairwise(boundaries):
        valid = np.isfinite(x[start:stop]) & np.isfinite(y[start:stop])
        if int(valid.sum()) < 20:
            continue
        left_values = x[start:stop][valid]
        right_values = y[start:stop][valid]
        if np.std(left_values) <= 1.0e-12 or np.std(right_values) <= 1.0e-12:
            continue
        values.append(float(stats.spearmanr(left_values, right_values).statistic))
    return (float(np.mean(values)) if values else math.nan, len(values))


def _old_prediction_comparison(
    *,
    study: Mapping[str, Any],
    inputs: base.LearnabilityInputs,
    contract_rows: np.ndarray,
    contract_raw: np.ndarray,
) -> list[dict[str, Any]]:
    path_root = base.DEFAULT_OUTPUT_ROOT
    path_study = base.load_study(_resolve(study["sources"]["path_study_config"]))
    comparisons: list[dict[str, Any]] = []
    column_by_target = {
        "state": (2, 3, 4),
        "pre_peak_mae_10": (5,),
        "pre_peak_mae_20": (6,),
    }
    for year in DECISION_YEARS:
        for target_name, columns in column_by_target.items():
            if target_name == "state":
                target = "state"
                horizon = 10
            else:
                target = "pre_peak_mae"
                horizon = int(target_name.rsplit("_", 1)[1])
            result_path = (
                path_root
                / "folds"
                / f"fold_{year}"
                / f"h{horizon:02d}"
                / target
                / "task_result.json"
            )
            expected = base._task_semantics(
                path_study, year=year, horizon=horizon, target=target
            )
            if not base._task_result_complete(result_path, expected=expected):
                raise ValueError(f"old path prediction is incomplete: {result_path}")
            result = json.loads(result_path.read_text(encoding="utf-8"))
            group = json.loads(
                (result_path.parent.parent / "group_manifest.json").read_text(
                    encoding="utf-8"
                )
            )
            old_rows = np.load(
                base._verify_file_record(group["files"]["evaluation_rows"]),
                allow_pickle=False,
            )
            old_prediction = np.load(
                base._verify_file_record(result["files"]["prediction"]),
                allow_pickle=False,
            )
            positions = np.searchsorted(contract_rows, old_rows)
            if bool(np.any(positions >= len(contract_rows))) or not np.array_equal(
                contract_rows[positions], old_rows
            ):
                raise AssertionError("strict contract does not cover an old OOS row")
            if target == "state":
                new_prediction = (
                    contract_raw[positions, columns[1]]
                    + 2.0 * contract_raw[positions, columns[2]]
                )
                old_score = old_prediction[:, 1] + 2.0 * old_prediction[:, 2]
                actual = np.asarray(inputs.label_values("state", 10)[old_rows])
            else:
                new_prediction = contract_raw[positions, columns[0]]
                old_score = old_prediction
                actual = np.asarray(inputs.label_values(target, horizon)[old_rows])
            dates = np.asarray(inputs.candidate_date_idx[old_rows], dtype=np.int32)
            correlation, date_count = _daily_spearman(old_score, new_prediction, dates)
            old_ic, old_ic_dates = _daily_spearman(old_score, actual, dates)
            new_ic, new_ic_dates = _daily_spearman(new_prediction, actual, dates)
            threshold = float(
                study["validation"]["minimum_daily_spearman_for_equivalence"]
            )
            comparisons.append(
                {
                    "year": year,
                    "target": target_name,
                    "common_row_count": len(old_rows),
                    "daily_prediction_spearman": correlation,
                    "daily_prediction_spearman_date_count": date_count,
                    "old_rank_ic": old_ic,
                    "new_rank_ic": new_ic,
                    "rank_ic_date_count": min(old_ic_dates, new_ic_dates),
                    "role_direction_flipped": bool(
                        math.isfinite(old_ic)
                        and math.isfinite(new_ic)
                        and np.sign(old_ic) != np.sign(new_ic)
                    ),
                    "equivalent_at_threshold": bool(
                        math.isfinite(correlation) and correlation >= threshold
                    ),
                }
            )
    return comparisons


def materialize_contract(
    *,
    study: Mapping[str, Any],
    feature_study: Mapping[str, Any],
    inputs: base.LearnabilityInputs,
    output_root: Path,
    final_heads: Mapping[int, Mapping[str, Any]],
) -> dict[str, Any]:
    manifest_path = output_root / "contract_manifest.json"
    task_inventory: list[dict[str, Any]] = []
    for task in [*_training_tasks(study), *_inference_tasks()]:
        result_path = _task_result_path(output_root, str(task["task_id"]))
        if not _task_complete(
            result_path,
            task=task,
            study=study,
            feature_study=feature_study,
        ):
            raise ValueError(f"strict contract task is incomplete: {task['task_id']}")
        result = json.loads(result_path.read_text(encoding="utf-8"))
        task_inventory.append(
            {
                "task_id": result["task_id"],
                "family": result["family"],
                "target": result["target"],
                "horizon": result["horizon"],
                "model_year": result["model_year"],
                "stage": result["stage"],
                "atlas_cutoff": result["atlas_cutoff"],
                "inner_validation_year": result["inner_validation_year"],
                "train_row_count": result["train_row_count"],
                "evaluation_row_count": result["evaluation_row_count"],
                "candidate_prediction_row_count": result.get(
                    "candidate_prediction_row_count"
                ),
                "maximum_train_signal_date_idx": result[
                    "maximum_train_signal_date_idx"
                ],
                "purge_days": result.get("purge_days"),
                "best_iteration": result["best_iteration"],
                "files": result["files"],
                "task_result": _file_record(result_path),
            }
        )
    row_parts: list[np.ndarray] = []
    raw_parts: list[np.ndarray] = []
    sources_by_year: list[dict[str, Any]] = []
    for year in CONTRACT_YEARS:
        mfe_suffix = "outer" if year <= 2022 else "reuse_inference"
        task_ids = {
            "mfe_10": f"mfe10_{year}_{mfe_suffix}",
            "mfe_20": f"mfe20_{year}_{mfe_suffix}",
            "state_10": f"state10_{year}_outer",
            "pre_peak_mae_10": f"risk10_{year}_outer",
            "pre_peak_mae_20": f"risk20_{year}_outer",
        }
        loaded = {
            name: _load_task_arrays(output_root, task_id)
            for name, task_id in task_ids.items()
        }
        rows = loaded["mfe_10"][0]
        for name, (current_rows, _prediction) in loaded.items():
            if not np.array_equal(rows, current_rows):
                raise AssertionError(
                    f"candidate row alignment changed in {year}: {name}"
                )
        state_probability = loaded["state_10"][1]
        if state_probability.shape != (len(rows), 3):
            raise ValueError("strict state prediction must have three columns")
        if not np.allclose(
            state_probability.sum(axis=1), 1.0, rtol=1.0e-5, atol=1.0e-5
        ):
            raise ValueError("strict state probabilities do not sum to one")
        raw = np.column_stack(
            [
                loaded["mfe_10"][1],
                loaded["mfe_20"][1],
                state_probability,
                loaded["pre_peak_mae_10"][1],
                loaded["pre_peak_mae_20"][1],
            ]
        ).astype(np.float32, copy=False)
        if raw.shape != (len(rows), len(PHYSICAL_COLUMNS)):
            raise AssertionError("physical contract matrix shape changed")
        row_parts.append(rows)
        raw_parts.append(raw)
        sources_by_year.append(
            {
                "year": year,
                "candidate_row_count": len(rows),
                "tasks": task_ids,
                "atlas_cutoff": int(
                    study["state_and_risk"]["atlas_cutoff_by_test_year"][str(year)]
                ),
            }
        )
    contract_rows = np.concatenate(row_parts)
    contract_raw = np.concatenate(raw_parts)
    if bool(np.any(contract_rows[1:] <= contract_rows[:-1])):
        raise AssertionError("contract candidate rows are not unique and ordered")
    dates = np.asarray(inputs.candidate_date_idx[contract_rows], dtype=np.int32)
    if any(
        str(inputs.date_values[int(value)])[:4] == "2026" for value in np.unique(dates)
    ):
        raise AssertionError("a 2026 candidate reached the entry contract")
    contract_rank = np.column_stack(
        [
            _rank_by_date(contract_raw[:, column], dates)
            for column in range(contract_raw.shape[1])
        ]
    ).astype(np.float32, copy=False)
    contract_dir = output_root / "contract"
    rows_path = contract_dir / "candidate_rows.npy"
    raw_path = contract_dir / "raw_predictions.npy"
    rank_path = contract_dir / "date_rank_predictions.npy"
    _save_npy(rows_path, contract_rows)
    _save_npy(raw_path, contract_raw)
    _save_npy(rank_path, contract_rank)
    atlas_manifest = json.loads(
        (output_root / "atlas_manifest.json").read_text(encoding="utf-8")
    )
    comparison = _old_prediction_comparison(
        study=study,
        inputs=inputs,
        contract_rows=contract_rows,
        contract_raw=contract_raw,
    )
    rerun_required = any(
        not row["equivalent_at_threshold"] or row["role_direction_flipped"]
        for row in comparison
    )
    manifest = {
        "schema": MANIFEST_SCHEMA,
        "status": "completed_candidate_aligned_strict_oos_contract",
        "completed_at": _now(),
        "study_id": STUDY_ID,
        "candidate_count": len(contract_rows),
        "candidate_years": list(CONTRACT_YEARS),
        "physical_columns": list(PHYSICAL_COLUMNS),
        "logical_coordinates": list(study["entry_contract"]["coordinates"]),
        "mfe_heads": {
            str(horizon): {
                **dict(head),
                "families": list(head["families"]),
            }
            for horizon, head in final_heads.items()
        },
        "state_challenge_formal": bool(atlas_manifest["state_challenge_formal"]),
        "atlas_stability": atlas_manifest["stability"],
        "sources_by_year": sources_by_year,
        "task_inventory": task_inventory,
        "old_prediction_comparison": comparison,
        "untrained_diagnostic_rerun_required": rerun_required,
        "files": {
            "candidate_rows": _file_record(
                rows_path,
                shape=list(contract_rows.shape),
                dtype=str(contract_rows.dtype),
            ),
            "raw_predictions": _file_record(
                raw_path, shape=list(contract_raw.shape), dtype=str(contract_raw.dtype)
            ),
            "date_rank_predictions": _file_record(
                rank_path,
                shape=list(contract_rank.shape),
                dtype=str(contract_rank.dtype),
            ),
            "atlas_manifest": _file_record(output_root / "atlas_manifest.json"),
        },
        "scope": {
            "maximum_consumed_outcome_date": inputs.maximum_outcome_date,
            "forbidden_2026_row_count": 0,
            "new_mfe_booster_count": 12,
            "new_state_and_risk_booster_count": 36,
            "reused_mfe_booster_count": 6,
            "fusion": False,
        },
        "does_not_select": list(study["non_selections"]),
    }
    _write_json(manifest_path, manifest)
    _write_json(
        output_root / "decision.json",
        {
            "status": "entry_contract_frozen",
            "mfe_heads": manifest["mfe_heads"],
            "state_challenge_formal": manifest["state_challenge_formal"],
            "next_step": "run_D1_D3_D5_matched_capacity_AB",
            "does_not_select": list(study["non_selections"]),
        },
    )
    _write_json(DEFAULT_RECORD_ROOT / "config.json", study)
    _write_json(
        DEFAULT_RECORD_ROOT / "result.json",
        {
            **manifest,
            "full_output": _file_record(manifest_path),
        },
    )
    return manifest


def status(
    *, study_path: Path = DEFAULT_STUDY_PATH, output_root: Path = DEFAULT_OUTPUT_ROOT
) -> dict[str, Any]:
    study = load_study(study_path)
    feature_study = feature.load_study(
        _resolve(study["sources"]["feature_study_config"])
    )
    groups: dict[str, list[str]] = {"completed": [], "pending": []}
    for task in [*_training_tasks(study), *_inference_tasks()]:
        complete = _task_complete(
            _task_result_path(output_root, str(task["task_id"])),
            task=task,
            study=study,
            feature_study=feature_study,
        )
        groups["completed" if complete else "pending"].append(str(task["task_id"]))
    return {
        "study_id": STUDY_ID,
        "training_task_count": len(_training_tasks(study)),
        "inference_task_count": len(_inference_tasks()),
        "manifest_complete": (output_root / "contract_manifest.json").is_file(),
        **groups,
    }


def run_pending(
    *, study_path: Path = DEFAULT_STUDY_PATH, output_root: Path = DEFAULT_OUTPUT_ROOT
) -> dict[str, Any]:
    study = load_study(study_path)
    feature_study = feature.load_study(
        _resolve(study["sources"]["feature_study_config"])
    )
    inputs, pack = feature._load_validated_inputs(feature_study)
    if inputs.maximum_outcome_date != "2025-12-31":
        raise ValueError("input outcome boundary changed")
    final_heads = _final_mfe_heads(study)
    prepare_atlas(study=study, output_root=output_root, inputs=inputs, pack=pack)
    for task in _training_tasks(study):
        if task["family"] == "mfe":
            if task["stage"] == "tuning":
                _run_mfe_tuning(
                    task=task,
                    study=study,
                    feature_study=feature_study,
                    inputs=inputs,
                    output_root=output_root,
                )
            else:
                _run_mfe_outer(
                    task=task,
                    study=study,
                    feature_study=feature_study,
                    inputs=inputs,
                    output_root=output_root,
                    final_heads=final_heads,
                )
        elif task["stage"] == "tuning":
            _run_base_tuning(
                task=task,
                study=study,
                feature_study=feature_study,
                inputs=inputs,
                output_root=output_root,
            )
        else:
            _run_base_outer(
                task=task,
                study=study,
                feature_study=feature_study,
                inputs=inputs,
                output_root=output_root,
            )
    for task in _inference_tasks():
        _run_mfe_reuse_inference(
            task=task,
            study=study,
            feature_study=feature_study,
            inputs=inputs,
            output_root=output_root,
            final_heads=final_heads,
        )
    return materialize_contract(
        study=study,
        feature_study=feature_study,
        inputs=inputs,
        output_root=output_root,
        final_heads=final_heads,
    )


def self_test() -> dict[str, Any]:
    dates = np.asarray([1, 1, 1, 2, 2], dtype=np.int32)
    values = np.asarray([3.0, 1.0, 2.0, 5.0, 5.0])
    ranks = _rank_by_date(values, dates)
    np.testing.assert_allclose(ranks[:3], [1.0, 0.0, 0.5])
    np.testing.assert_allclose(ranks[3:], [0.5, 0.5])
    study = load_study()
    tasks = _training_tasks(study)
    if len(tasks) != 48:
        raise AssertionError("strict OOS booster count changed")
    if sum(task["family"] == "mfe" for task in tasks) != 12:
        raise AssertionError("MFE supplement booster count changed")
    if sum(task["family"] != "mfe" for task in tasks) != 36:
        raise AssertionError("state/risk booster count changed")
    path = np.asarray([[-0.1, -0.2], [0.1, 0.2], [0.0, 0.0]], dtype=np.float32)
    model = {
        "clip_low": np.asarray([-1.0, -1.0], dtype=np.float32),
        "clip_high": np.asarray([1.0, 1.0], dtype=np.float32),
        "center": np.zeros(2, dtype=np.float32),
        "scale": np.ones(2, dtype=np.float32),
        "pca_mean": np.zeros(2, dtype=np.float32),
        "pca_components": np.eye(2, dtype=np.float32),
        "fixed_k3_centers": path.copy(),
        "raw_to_state": np.asarray([0, 2, 1], dtype=np.int8),
    }
    np.testing.assert_array_equal(_assign_state(path, model), [0, 2, 1])
    return {"status": "passed", "training_task_count": len(tasks)}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build the strict 2020-2025 Seq100 entry contract."
    )
    parser.add_argument(
        "command", choices=("self-test", "status", "prepare-atlas", "run-pending")
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_STUDY_PATH)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    study_path = _resolve(args.config)
    output_root = _resolve(args.output_root)
    if args.command == "self-test":
        _emit("self_test_completed", **self_test())
    elif args.command == "status":
        _emit("status", **status(study_path=study_path, output_root=output_root))
    elif args.command == "prepare-atlas":
        study = load_study(study_path)
        feature_study = feature.load_study(
            _resolve(study["sources"]["feature_study_config"])
        )
        inputs, pack = feature._load_validated_inputs(feature_study)
        _emit(
            "atlas_preparation_completed",
            **prepare_atlas(
                study=study, output_root=output_root, inputs=inputs, pack=pack
            ),
        )
    else:
        manifest = run_pending(study_path=study_path, output_root=output_root)
        _emit("entry_contract_completed", candidate_count=manifest["candidate_count"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
