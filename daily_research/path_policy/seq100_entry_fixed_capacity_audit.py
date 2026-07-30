from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import os
import time
from collections.abc import Mapping, Sequence
from datetime import datetime
from itertools import pairwise
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import average_precision_score

from daily_research.path_policy import seq100_entry_contract_oos as contract_v2
from daily_research.path_policy import seq100_entry_role_synthesis as role
from daily_research.path_policy import seq100_mfe_feature_family_audit as feature
from daily_research.path_policy import seq100_mfe_feature_union_audit as union
from daily_research.path_policy import (
    seq100_mfe_final_head_capacity_audit as final_capacity,
)
from daily_research.path_policy import seq100_mfe_objective_alignment as objective
from daily_research.path_policy import seq100_path_label_learnability as base
from daily_research.path_policy import seq100_signal_quality as signal_quality

WORKSPACE_ROOT = Path(__file__).resolve().parents[2]
STUDY_ID = "seq100_entry_fixed_capacity_audit_v1"
CONTRACT_V4_ID = "seq100_entry_contract_oos_v4"
DEFAULT_STUDY_PATH = (
    WORKSPACE_ROOT / "daily_research/studies/seq100_entry_fixed_capacity_audit_v1.json"
)
DEFAULT_OUTPUT_ROOT = (
    WORKSPACE_ROOT
    / "daily_research/output/path_policy/studies/seq100_entry_fixed_capacity_audit_v1"
)
DEFAULT_RECORD_ROOT = (
    WORKSPACE_ROOT
    / "daily_research/research_records/seq100/seq100_entry_fixed_capacity_audit_v1"
)
CONTRACT_YEARS = (2020, 2021, 2022, 2023, 2024, 2025)
DECISION_YEARS = (2023, 2024, 2025)
HISTORICAL_YEARS = (2020, 2021, 2022)
MFE_HORIZONS = (10, 20)
HEAD_NAMES = ("state_10", "risk_10", "risk_20")
MFE_TASK_SCHEMA = "seq100_entry_fixed_mfe_task/v1"
SAFETY_TASK_SCHEMA = "seq100_entry_fixed_safety_task/v1"
SUMMARY_SCHEMA = "seq100_entry_fixed_capacity_summary/v1"
CONTRACT_V4_SCHEMA = "seq100_entry_contract_oos_manifest/v4"
PREFIX_ORDER = ("adaptive", "fixed_256")


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


def _record_exists(record: Mapping[str, Any]) -> bool:
    try:
        path = _resolve(str(record["path"]))
        return (
            path.is_file()
            and path.stat().st_size == int(record["size"])
            and _sha256(path) == str(record["sha256"])
        )
    except (KeyError, OSError, TypeError, ValueError):
        return False


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
    if tuple(int(value) for value in folds["contract_years"]) != CONTRACT_YEARS:
        raise ValueError("contract years changed")
    if tuple(int(value) for value in folds["decision_years"]) != DECISION_YEARS:
        raise ValueError("decision years changed")
    if (
        tuple(int(value) for value in folds["historical_guard_years"])
        != HISTORICAL_YEARS
    ):
        raise ValueError("historical guard years changed")
    if folds["maximum_outcome_date"] != "2025-12-31":
        raise ValueError("entry capacity audit may not read outcomes after 2025")
    if int(folds["forbidden_outcome_year"]) != 2026:
        raise ValueError("2026 must remain forbidden")
    if int(payload["mfe"]["fixed_rounds"]) != 512:
        raise ValueError("MFE capacity must remain fixed at 512 boosting rounds")
    if int(payload["state_and_risk"]["fixed_rounds"]) != 256:
        raise ValueError("state/risk capacity must remain fixed at 256 boosting rounds")
    if int(payload["state_and_risk"]["state_internal_tree_multiplier"]) != 3:
        raise ValueError("K3 state must use three internal trees per boosting round")
    if int(payload["mfe"]["booster_count"]) != 6:
        raise ValueError("MFE booster count changed")
    if int(payload["state_and_risk"]["booster_count"]) != 18:
        raise ValueError("state/risk booster count changed")
    return payload


def _load_v3_manifest(study: Mapping[str, Any]) -> dict[str, Any]:
    path = _resolve(study["sources"]["entry_contract_v3_manifest"])
    manifest = json.loads(path.read_text(encoding="utf-8"))
    expected = {
        "schema": final_capacity.CONTRACT_V3_SCHEMA,
        "status": "completed_candidate_aligned_strict_oos_contract",
        "study_id": "seq100_entry_contract_oos_v3",
        "candidate_count": int(study["contract_v4"]["candidate_count"]),
        "candidate_years": list(CONTRACT_YEARS),
        "physical_columns": list(contract_v2.PHYSICAL_COLUMNS),
    }
    if any(manifest.get(key) != value for key, value in expected.items()):
        raise ValueError("entry contract v3 semantics changed")
    for record in dict(manifest["files"]).values():
        if not _record_exists(record):
            raise ValueError(f"entry contract v3 file changed: {record.get('path')}")
    return manifest


def _head(study: Mapping[str, Any], horizon: int) -> dict[str, Any]:
    current = dict(study["mfe"]["heads"][str(int(horizon))])
    return {
        "horizon": int(horizon),
        "name": str(current["name"]),
        "families": tuple(str(value) for value in current["families"]),
    }


def _safety_head(study: Mapping[str, Any], name: str) -> dict[str, Any]:
    current = dict(study["state_and_risk"]["heads"][str(name)])
    return {
        "name": str(name),
        "target": str(current["target"]),
        "horizon": int(current["horizon"]),
        "columns": tuple(int(value) for value in current["columns"]),
    }


def _tasks(study: Mapping[str, Any]) -> list[dict[str, Any]]:
    tasks: list[dict[str, Any]] = []
    for horizon in MFE_HORIZONS:
        for year in HISTORICAL_YEARS:
            tasks.append(
                {
                    **_head(study, horizon),
                    "task_id": f"mfe{horizon}_{year}_fixed512_outer",
                    "kind": "mfe",
                    "year": int(year),
                    "rounds": 512,
                }
            )
    atlas = {
        int(year): int(cutoff)
        for year, cutoff in study["state_and_risk"]["atlas_cutoff_by_test_year"].items()
    }
    for name in HEAD_NAMES:
        head = _safety_head(study, name)
        for year in CONTRACT_YEARS:
            tasks.append(
                {
                    **head,
                    "task_id": f"{name}_{year}_adaptive_vs_fixed256_outer",
                    "kind": "safety",
                    "year": int(year),
                    "rounds": 256,
                    "atlas_cutoff": atlas[year] if name == "state_10" else None,
                }
            )
    return tasks


def _task_dir(output_root: Path, task: Mapping[str, Any]) -> Path:
    return output_root / "tasks" / str(task["task_id"])


def _task_result_path(output_root: Path, task: Mapping[str, Any]) -> Path:
    return _task_dir(output_root, task) / "task_result.json"


def _task_by_id(study: Mapping[str, Any], task_id: str) -> dict[str, Any]:
    for task in _tasks(study):
        if task["task_id"] == task_id:
            return task
    raise KeyError(task_id)


def _v2_outer_result_path(study: Mapping[str, Any], *, head: str, year: int) -> Path:
    root = _resolve(study["sources"]["entry_contract_v2_output_root"])
    return (
        root
        / "tasks"
        / f"{head.replace('_', '')}_{int(year)}_outer"
        / "task_result.json"
    )


def _adaptive_source(
    study: Mapping[str, Any], *, head: str, year: int
) -> dict[str, Any]:
    task_name = {
        "state_10": "state10",
        "risk_10": "risk10",
        "risk_20": "risk20",
    }[head]
    root = _resolve(study["sources"]["entry_contract_v2_output_root"])
    path = root / "tasks" / f"{task_name}_{int(year)}_outer" / "task_result.json"
    result = json.loads(path.read_text(encoding="utf-8"))
    if (
        result.get("schema") != contract_v2.TASK_SCHEMA
        or result.get("status") != "completed"
        or int(result.get("model_year", -1)) != int(year)
    ):
        raise ValueError(f"adaptive source is invalid: {path}")
    expected_target = "state" if head == "state_10" else "pre_peak_mae"
    expected_horizon = 20 if head == "risk_20" else 10
    if (
        result.get("target") != expected_target
        or int(result.get("horizon", -1)) != expected_horizon
    ):
        raise ValueError(f"adaptive source target changed: {path}")
    return {**result, "_result_path": path}


def _parameters(
    *,
    task: Mapping[str, Any],
    feature_study: Mapping[str, Any],
    v2_study: Mapping[str, Any],
) -> dict[str, Any]:
    if task["kind"] == "mfe":
        return feature._model_parameters(feature_study)[0]
    return contract_v2._model_parameters(v2_study, str(task["target"]))[0]


def _task_complete(
    path: Path,
    *,
    task: Mapping[str, Any],
    study_path: Path,
    feature_study: Mapping[str, Any],
    v2_study: Mapping[str, Any],
) -> bool:
    if not path.is_file():
        return False
    try:
        import lightgbm as lgb

        result = json.loads(path.read_text(encoding="utf-8"))
        expected = {
            "schema": MFE_TASK_SCHEMA if task["kind"] == "mfe" else SAFETY_TASK_SCHEMA,
            "status": "completed",
            "study_id": STUDY_ID,
            "task_id": task["task_id"],
            "kind": task["kind"],
            "model_year": int(task["year"]),
            "horizon": int(task["horizon"]),
            "trained_rounds": int(task["rounds"]),
            "parameters": _parameters(
                task=task, feature_study=feature_study, v2_study=v2_study
            ),
            "config_sha256": _config_hash(study_path),
        }
        if any(result.get(key) != value for key, value in expected.items()):
            return False
        files = dict(result.get("files", {}) or {})
        if not files or not all(_record_exists(record) for record in files.values()):
            return False
        model = lgb.Booster(model_file=str(_resolve(files["model"]["path"])))
        expected_trees = int(task["rounds"])
        if task.get("target") == "state":
            expected_trees *= 3
        if int(model.num_trees()) != expected_trees:
            return False
        candidate_rows = np.load(
            _resolve(files["candidate_rows"]["path"]), allow_pickle=False
        )
        if task["kind"] == "mfe":
            candidate = np.load(
                _resolve(files["candidate_prediction"]["path"]), allow_pickle=False
            )
            evaluation = np.load(
                _resolve(files["evaluation_prediction"]["path"]), allow_pickle=False
            )
            evaluation_rows = np.load(
                _resolve(files["evaluation_rows"]["path"]), allow_pickle=False
            )
            return (
                candidate.shape == candidate_rows.shape
                and evaluation.shape == evaluation_rows.shape
            )
        for prefix in PREFIX_ORDER:
            candidate = np.load(
                _resolve(files[f"{prefix}_candidate_prediction"]["path"]),
                allow_pickle=False,
            )
            evaluation = np.load(
                _resolve(files[f"{prefix}_evaluation_prediction"]["path"]),
                allow_pickle=False,
            )
            evaluation_rows = np.load(
                _resolve(files["evaluation_rows"]["path"]), allow_pickle=False
            )
            if len(candidate) != len(candidate_rows) or len(evaluation) != len(
                evaluation_rows
            ):
                return False
        return bool(result.get("prefix_equivalence", {}).get("passed"))
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


def _daily_spearman(left: np.ndarray, right: np.ndarray, date_idx: np.ndarray) -> float:
    x = np.asarray(left, dtype=np.float64)
    y = np.asarray(right, dtype=np.float64)
    dates = np.asarray(date_idx, dtype=np.int32)
    if x.ndim != 1 or y.ndim != 1 or x.shape != y.shape or x.shape != dates.shape:
        raise ValueError("daily Spearman arrays must be aligned and one-dimensional")
    values: list[float] = []
    boundaries = np.flatnonzero(np.r_[True, dates[1:] != dates[:-1], True])
    for left_pos, right_pos in pairwise(boundaries):
        valid = np.isfinite(x[left_pos:right_pos]) & np.isfinite(y[left_pos:right_pos])
        if int(valid.sum()) < 3:
            continue
        xv = x[left_pos:right_pos][valid]
        yv = y[left_pos:right_pos][valid]
        if np.std(xv) <= 1.0e-12 or np.std(yv) <= 1.0e-12:
            continue
        value = float(stats.spearmanr(xv, yv).statistic)
        if math.isfinite(value):
            values.append(value)
    return float(np.mean(values)) if values else math.nan


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


def _prefix_equivalence(
    *,
    old: np.ndarray,
    current: np.ndarray,
    dates: np.ndarray,
    absolute_tolerance: float,
    relative_tolerance: float,
    minimum_daily_spearman: float,
) -> dict[str, Any]:
    previous = np.asarray(old, dtype=np.float64)
    reproduced = np.asarray(current, dtype=np.float64)
    if previous.shape != reproduced.shape:
        return {
            "passed": False,
            "reason": "shape_mismatch",
            "old_shape": list(previous.shape),
            "current_shape": list(reproduced.shape),
        }
    finite = np.isfinite(previous) & np.isfinite(reproduced)
    finite_equal = np.array_equal(np.isfinite(previous), np.isfinite(reproduced))
    difference = np.abs(previous - reproduced)
    maximum_error = float(np.nanmax(difference)) if difference.size else 0.0
    allclose = bool(
        np.allclose(
            previous,
            reproduced,
            atol=float(absolute_tolerance),
            rtol=float(relative_tolerance),
            equal_nan=True,
        )
    )
    if previous.ndim == 1:
        spearman = _daily_spearman(previous, reproduced, dates)
    else:
        spearman = float(
            np.nanmin(
                [
                    _daily_spearman(previous[:, column], reproduced[:, column], dates)
                    for column in range(previous.shape[1])
                ]
            )
        )
    return {
        "passed": bool(
            finite_equal
            and allclose
            and math.isfinite(spearman)
            and spearman >= float(minimum_daily_spearman)
        ),
        "finite_pattern_equal": finite_equal,
        "allclose": allclose,
        "maximum_absolute_error": maximum_error,
        "daily_spearman": spearman,
        "minimum_daily_spearman": float(minimum_daily_spearman),
        "finite_value_count": int(finite.sum()),
    }


def _state_collapsed(probability: np.ndarray) -> bool:
    values = np.asarray(probability, dtype=np.float64)
    if (
        values.ndim != 2
        or values.shape[1] != 3
        or not bool(np.isfinite(values).all())
        or not np.allclose(values.sum(axis=1), 1.0, atol=1.0e-5, rtol=1.0e-5)
    ):
        return True
    predicted = np.argmax(values, axis=1)
    return bool(np.unique(predicted).size < 2)


def _save_vocabularies(path: Path, values: Sequence[np.ndarray]) -> dict[str, Any]:
    return contract_v2._save_vocabularies(path, values)


def _risk_daily_metrics(
    *,
    date_idx: np.ndarray,
    actual: np.ndarray,
    prediction: np.ndarray,
) -> pd.DataFrame:
    dates = np.asarray(date_idx, dtype=np.int32)
    target = np.asarray(actual, dtype=np.float64)
    score = np.asarray(prediction, dtype=np.float64)
    records: list[dict[str, Any]] = []
    boundaries = np.flatnonzero(np.r_[True, dates[1:] != dates[:-1], True])
    for left, right in pairwise(boundaries):
        valid = np.isfinite(target[left:right]) & np.isfinite(score[left:right])
        if int(valid.sum()) < 20:
            continue
        y = target[left:right][valid]
        p = score[left:right][valid]
        order = np.argsort(y, kind="mergesort")
        event_count = max(1, math.ceil(0.20 * len(order)))
        event = np.zeros(len(order), dtype=np.int8)
        event[order[:event_count]] = 1
        records.append(
            {
                "date_idx": int(dates[left]),
                "rank_ic": _safe_spearman(p, y),
                "mae": float(np.mean(np.abs(p - y))),
                "deep_adverse_pr_auc": float(average_precision_score(event, -p)),
                "target_mean": float(np.mean(y)),
                "prediction_mean": float(np.mean(p)),
                "absolute_bias": abs(float(np.mean(p) - np.mean(y))),
                "row_count": int(valid.sum()),
            }
        )
    frame = pd.DataFrame.from_records(records)
    if frame.empty:
        raise ValueError("risk safety evaluation produced no valid dates")
    return frame


def _state_daily_metrics(
    *,
    date_idx: np.ndarray,
    actual: np.ndarray,
    probability: np.ndarray,
) -> pd.DataFrame:
    dates = np.asarray(date_idx, dtype=np.int32)
    target = np.asarray(actual, dtype=np.int8)
    predicted = np.asarray(probability, dtype=np.float64)
    records: list[dict[str, Any]] = []
    boundaries = np.flatnonzero(np.r_[True, dates[1:] != dates[:-1], True])
    for left, right in pairwise(boundaries):
        valid = np.isin(target[left:right], (0, 1, 2)) & np.isfinite(
            predicted[left:right]
        ).all(axis=1)
        if int(valid.sum()) < 20:
            continue
        y = target[left:right][valid]
        p = predicted[left:right][valid]
        p = np.clip(p, 1.0e-7, 1.0 - 1.0e-7)
        p /= p.sum(axis=1, keepdims=True)
        expected = p[:, 1] + 2.0 * p[:, 2]
        one_hot = np.eye(3, dtype=np.float64)[y]
        count = len(y)
        top_count = max(1, math.ceil(0.05 * count))
        selected = np.argsort(p[:, 2], kind="mergesort")[-top_count:]
        high = y == 2
        records.append(
            {
                "date_idx": int(dates[left]),
                "ordinal_ic": _safe_spearman(expected, y),
                "high_state_top5_rate": float(high[selected].mean()),
                "high_state_top5_lift": float(high[selected].mean() - high.mean()),
                "brier": float(np.mean(np.square(p - one_hot).sum(axis=1))),
                "logloss": float(
                    np.mean(-np.log(np.clip(p[np.arange(count), y], 1.0e-12, 1.0)))
                ),
                "row_count": int(count),
            }
        )
    frame = pd.DataFrame.from_records(records)
    if frame.empty:
        raise ValueError("state safety evaluation produced no valid dates")
    return frame


def _daily_summary(frame: pd.DataFrame) -> dict[str, float]:
    return {
        str(column): float(np.nanmean(frame[column].to_numpy(dtype=np.float64)))
        for column in frame.columns
        if column != "date_idx"
    }


def _load_v3_arrays(
    manifest: Mapping[str, Any],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    files = dict(manifest["files"])
    rows = np.load(_resolve(files["candidate_rows"]["path"]), mmap_mode="r")
    raw = np.load(_resolve(files["raw_predictions"]["path"]), mmap_mode="r")
    rank = np.load(_resolve(files["date_rank_predictions"]["path"]), mmap_mode="r")
    if raw.shape != rank.shape or raw.shape != (
        len(rows),
        len(contract_v2.PHYSICAL_COLUMNS),
    ):
        raise ValueError("v3 arrays are not aligned")
    return np.asarray(rows), np.asarray(raw), np.asarray(rank)


def _v3_year_values(
    *,
    inputs: base.LearnabilityInputs,
    v3_rows: np.ndarray,
    v3_raw: np.ndarray,
    year: int,
    columns: Sequence[int],
) -> tuple[np.ndarray, np.ndarray]:
    expected_rows = contract_v2._year_rows(inputs, year)
    positions = np.searchsorted(v3_rows, expected_rows)
    if bool(np.any(positions >= len(v3_rows))) or not np.array_equal(
        v3_rows[positions], expected_rows
    ):
        raise ValueError(f"v3 candidate rows changed in {year}")
    values = np.asarray(v3_raw[np.ix_(positions, np.asarray(columns, dtype=np.int64))])
    if len(columns) == 1:
        values = values[:, 0]
    return expected_rows, values


def _run_mfe_task(
    *,
    task: Mapping[str, Any],
    study: Mapping[str, Any],
    study_path: Path,
    feature_study: Mapping[str, Any],
    v2_study: Mapping[str, Any],
    inputs: base.LearnabilityInputs,
    output_root: Path,
) -> dict[str, Any]:
    import lightgbm as lgb

    result_path = _task_result_path(output_root, task)
    if _task_complete(
        result_path,
        task=task,
        study_path=study_path,
        feature_study=feature_study,
        v2_study=v2_study,
    ):
        return json.loads(result_path.read_text(encoding="utf-8"))
    horizon = int(task["horizon"])
    year = int(task["year"])
    rounds = int(task["rounds"])
    feature_root = _resolve(study["sources"]["feature_output_root"])
    extra, manifests = union._open_union(
        feature_output_root=feature_root,
        families=task["families"],
        candidate_count=inputs.candidate_count,
    )
    fold = inputs.common_path_rows(year, horizon)
    datasets = None
    model = None
    try:
        datasets, _weight, evaluation_raw = feature._build_datasets(
            study=feature_study,
            inputs=inputs,
            train_rows=fold.train_rows,
            evaluation_rows=fold.evaluation_rows,
            horizon=horizon,
            extra=extra,
            extra_names=extra.feature_names,
        )
        parameters = _parameters(
            task=task, feature_study=feature_study, v2_study=v2_study
        )
        _emit(
            "entry_mfe_training_started",
            task_id=task["task_id"],
            rounds=rounds,
            train_rows=len(fold.train_rows),
            evaluation_rows=len(fold.evaluation_rows),
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
        evaluation_prediction = feature._predict(
            model, datasets.evaluation_sequence, rounds
        )
        daily, metrics = base.regression_metrics(
            date_idx=inputs.candidate_date_idx[datasets.evaluation_rows],
            actual=evaluation_raw,
            prediction=evaluation_prediction,
            date_values=inputs.date_values,
            horizon=horizon,
        )
        vocabularies = signal_quality._fit_category_vocabularies(
            inputs.categorical, fold.train_rows, inputs.categorical_columns
        )
        candidate_rows = contract_v2._year_rows(inputs, year)
        sequence = feature._combined_sequence(
            inputs=inputs,
            row_ids=candidate_rows,
            category_vocabularies=vocabularies,
            batch_size=int(feature_study["model"]["sequence_batch_size"]),
            extra=extra,
        )
        sequence = base._memory_trimmed_sequence(sequence, feature_study)
        candidate_prediction = feature._predict(model, sequence, rounds)
        output_dir = _task_dir(output_root, task)
        output_dir.mkdir(parents=True, exist_ok=True)
        paths = {
            "model": output_dir / "model.txt",
            "evaluation_prediction": output_dir / "evaluation_prediction.npy",
            "evaluation_rows": output_dir / "evaluation_rows.npy",
            "candidate_prediction": output_dir / "candidate_prediction.npy",
            "candidate_rows": output_dir / "candidate_rows.npy",
            "category_vocabularies": output_dir / "category_vocabularies.npz",
            "daily_metrics": output_dir / "daily_metrics.parquet",
        }
        model.save_model(str(paths["model"]), num_iteration=rounds)
        _save_npy(paths["evaluation_prediction"], evaluation_prediction)
        _save_npy(paths["evaluation_rows"], datasets.evaluation_rows)
        _save_npy(paths["candidate_prediction"], candidate_prediction)
        _save_npy(paths["candidate_rows"], candidate_rows)
        vocabulary_record = _save_vocabularies(
            paths["category_vocabularies"], vocabularies
        )
        daily.to_parquet(paths["daily_metrics"], index=False, compression="zstd")
        result = {
            "schema": MFE_TASK_SCHEMA,
            "status": "completed",
            "completed_at": _now(),
            "study_id": STUDY_ID,
            "task_id": task["task_id"],
            "kind": "mfe",
            "head": task["name"],
            "families": list(task["families"]),
            "target": "mfe",
            "horizon": horizon,
            "model_year": year,
            "purge_days": horizon,
            "trained_rounds": rounds,
            "internal_tree_count": int(model.num_trees()),
            "train_row_count": len(fold.train_rows),
            "evaluation_row_count": len(fold.evaluation_rows),
            "candidate_prediction_row_count": len(candidate_rows),
            "maximum_train_signal_date_idx": fold.maximum_train_signal_date_idx,
            "maximum_train_signal_date": str(
                inputs.date_values[int(fold.maximum_train_signal_date_idx)]
            ),
            "training_seconds": elapsed,
            "parameters": parameters,
            "feature_count": len(datasets.feature_names),
            "added_feature_count": len(extra.feature_names),
            "family_manifests": manifests,
            "metrics": metrics,
            "config_sha256": _config_hash(study_path),
            "files": {
                "model": _file_record(paths["model"]),
                "evaluation_prediction": _file_record(
                    paths["evaluation_prediction"],
                    shape=list(evaluation_prediction.shape),
                    dtype=str(evaluation_prediction.dtype),
                ),
                "evaluation_rows": _file_record(
                    paths["evaluation_rows"],
                    shape=list(datasets.evaluation_rows.shape),
                    dtype=str(datasets.evaluation_rows.dtype),
                ),
                "candidate_prediction": _file_record(
                    paths["candidate_prediction"],
                    shape=list(candidate_prediction.shape),
                    dtype=str(candidate_prediction.dtype),
                ),
                "candidate_rows": _file_record(
                    paths["candidate_rows"],
                    shape=list(candidate_rows.shape),
                    dtype=str(candidate_rows.dtype),
                ),
                "category_vocabularies": vocabulary_record,
                "daily_metrics": _file_record(paths["daily_metrics"]),
            },
        }
        _write_json(result_path, result)
        _emit(
            "entry_mfe_training_completed",
            task_id=task["task_id"],
            rounds=rounds,
            seconds=elapsed,
        )
        del sequence, candidate_prediction, evaluation_prediction
        return result
    finally:
        if datasets is not None:
            feature._release_datasets(datasets)
        del model, fold, extra
        gc.collect()


def _run_safety_task(
    *,
    task: Mapping[str, Any],
    study: Mapping[str, Any],
    study_path: Path,
    feature_study: Mapping[str, Any],
    v2_study: Mapping[str, Any],
    inputs: base.LearnabilityInputs,
    v3_rows: np.ndarray,
    v3_raw: np.ndarray,
    output_root: Path,
) -> dict[str, Any]:
    import lightgbm as lgb

    result_path = _task_result_path(output_root, task)
    if _task_complete(
        result_path,
        task=task,
        study_path=study_path,
        feature_study=feature_study,
        v2_study=v2_study,
    ):
        return json.loads(result_path.read_text(encoding="utf-8"))
    year = int(task["year"])
    horizon = int(task["horizon"])
    rounds = int(task["rounds"])
    source = _adaptive_source(study, head=str(task["name"]), year=year)
    adaptive_rounds = int(source["best_iteration"])
    if adaptive_rounds > rounds:
        raise ValueError(
            f"adaptive prefix exceeds fixed model: {task['task_id']} "
            f"{adaptive_rounds}>{rounds}"
        )
    source_task = {
        "target": task["target"],
        "horizon": horizon,
        "atlas_cutoff": task.get("atlas_cutoff"),
    }
    built = contract_v2._build_base_target_dataset(
        task=source_task,
        study=v2_study,
        inputs=inputs,
        output_root=_resolve(study["sources"]["entry_contract_v2_output_root"]),
        split_year=year,
    )
    model = None
    try:
        parameters = _parameters(
            task=task, feature_study=feature_study, v2_study=v2_study
        )
        _emit(
            "entry_safety_training_started",
            task_id=task["task_id"],
            adaptive_rounds=adaptive_rounds,
            fixed_rounds=rounds,
            train_rows=len(built.fold.train_rows),
            evaluation_rows=len(built.fold.evaluation_rows),
        )
        started = time.perf_counter()
        model = lgb.train(
            parameters,
            built.datasets.train_set,
            num_boost_round=rounds,
            valid_sets=[built.datasets.evaluation_set],
            valid_names=["outer_evaluation"],
            callbacks=[lgb.log_evaluation(period=0)],
        )
        elapsed = float(time.perf_counter() - started)
        evaluation_predictions = {
            "adaptive": contract_v2._predict(
                model, built.datasets.evaluation_sequence, adaptive_rounds
            ),
            "fixed_256": contract_v2._predict(
                model, built.datasets.evaluation_sequence, rounds
            ),
        }
        candidate_rows = contract_v2._year_rows(inputs, year)
        sequence = contract_v2._base_sequence(
            inputs=inputs,
            rows=candidate_rows,
            vocabularies=built.datasets.category_vocabularies,
            study=v2_study,
        )
        candidate_predictions = {
            "adaptive": contract_v2._predict(model, sequence, adaptive_rounds),
            "fixed_256": contract_v2._predict(model, sequence, rounds),
        }
        expected_rows, old = _v3_year_values(
            inputs=inputs,
            v3_rows=v3_rows,
            v3_raw=v3_raw,
            year=year,
            columns=task["columns"],
        )
        if not np.array_equal(candidate_rows, expected_rows):
            raise ValueError(f"candidate rows changed: {task['task_id']}")
        equivalence = _prefix_equivalence(
            old=old,
            current=candidate_predictions["adaptive"],
            dates=np.asarray(inputs.candidate_date_idx[candidate_rows], dtype=np.int32),
            absolute_tolerance=float(study["prefix_equivalence"]["absolute_tolerance"]),
            relative_tolerance=float(study["prefix_equivalence"]["relative_tolerance"]),
            minimum_daily_spearman=float(
                study["prefix_equivalence"]["minimum_daily_spearman"]
            ),
        )
        output_dir = _task_dir(output_root, task)
        output_dir.mkdir(parents=True, exist_ok=True)
        if not equivalence["passed"]:
            _write_json(
                output_dir / "prefix_failure.json",
                {
                    "status": "invalid_prefix_equivalence",
                    "task_id": task["task_id"],
                    "adaptive_source": _file_record(source["_result_path"]),
                    "prefix_equivalence": equivalence,
                },
            )
            raise RuntimeError(f"prefix equivalence failed: {task['task_id']}")
        daily: dict[str, pd.DataFrame] = {}
        metrics: dict[str, dict[str, Any]] = {}
        actual = np.asarray(built.evaluation_raw)
        dates = np.asarray(
            inputs.candidate_date_idx[built.fold.evaluation_rows], dtype=np.int32
        )
        for prefix in PREFIX_ORDER:
            prediction = evaluation_predictions[prefix]
            if task["target"] == "state":
                current_daily = _state_daily_metrics(
                    date_idx=dates, actual=actual, probability=prediction
                )
                summary = _daily_summary(current_daily)
                summary["class_prediction_counts"] = np.bincount(
                    np.argmax(prediction, axis=1), minlength=3
                ).tolist()
                summary["collapsed"] = _state_collapsed(prediction)
            else:
                current_daily = _risk_daily_metrics(
                    date_idx=dates, actual=actual, prediction=prediction
                )
                summary = _daily_summary(current_daily)
            daily[prefix] = current_daily
            metrics[prefix] = summary
        daily_frame = pd.concat(
            [frame.assign(prefix=prefix) for prefix, frame in daily.items()],
            ignore_index=True,
        )
        paths = {
            "model": output_dir / "model.txt",
            "evaluation_rows": output_dir / "evaluation_rows.npy",
            "candidate_rows": output_dir / "candidate_rows.npy",
            "category_vocabularies": output_dir / "category_vocabularies.npz",
            "daily_metrics": output_dir / "daily_metrics.parquet",
        }
        for prefix in PREFIX_ORDER:
            paths[f"{prefix}_evaluation_prediction"] = (
                output_dir / f"{prefix}_evaluation_prediction.npy"
            )
            paths[f"{prefix}_candidate_prediction"] = (
                output_dir / f"{prefix}_candidate_prediction.npy"
            )
        model.save_model(str(paths["model"]), num_iteration=rounds)
        _save_npy(paths["evaluation_rows"], built.fold.evaluation_rows)
        _save_npy(paths["candidate_rows"], candidate_rows)
        for prefix in PREFIX_ORDER:
            _save_npy(
                paths[f"{prefix}_evaluation_prediction"],
                evaluation_predictions[prefix],
            )
            _save_npy(
                paths[f"{prefix}_candidate_prediction"],
                candidate_predictions[prefix],
            )
        vocabulary_record = _save_vocabularies(
            paths["category_vocabularies"],
            built.datasets.category_vocabularies,
        )
        daily_frame.to_parquet(paths["daily_metrics"], index=False, compression="zstd")
        files = {
            "model": _file_record(paths["model"]),
            "evaluation_rows": _file_record(
                paths["evaluation_rows"],
                shape=list(built.fold.evaluation_rows.shape),
                dtype=str(built.fold.evaluation_rows.dtype),
            ),
            "candidate_rows": _file_record(
                paths["candidate_rows"],
                shape=list(candidate_rows.shape),
                dtype=str(candidate_rows.dtype),
            ),
            "category_vocabularies": vocabulary_record,
            "daily_metrics": _file_record(paths["daily_metrics"]),
        }
        for prefix in PREFIX_ORDER:
            evaluation_prediction = evaluation_predictions[prefix]
            candidate_prediction = candidate_predictions[prefix]
            files[f"{prefix}_evaluation_prediction"] = _file_record(
                paths[f"{prefix}_evaluation_prediction"],
                shape=list(evaluation_prediction.shape),
                dtype=str(evaluation_prediction.dtype),
            )
            files[f"{prefix}_candidate_prediction"] = _file_record(
                paths[f"{prefix}_candidate_prediction"],
                shape=list(candidate_prediction.shape),
                dtype=str(candidate_prediction.dtype),
            )
        result = {
            "schema": SAFETY_TASK_SCHEMA,
            "status": "completed",
            "completed_at": _now(),
            "study_id": STUDY_ID,
            "task_id": task["task_id"],
            "kind": "safety",
            "head": task["name"],
            "target": task["target"],
            "horizon": horizon,
            "model_year": year,
            "atlas_cutoff": task.get("atlas_cutoff"),
            "purge_days": horizon,
            "adaptive_rounds": adaptive_rounds,
            "trained_rounds": rounds,
            "internal_tree_count": int(model.num_trees()),
            "train_row_count": len(built.fold.train_rows),
            "evaluation_row_count": len(built.fold.evaluation_rows),
            "candidate_prediction_row_count": len(candidate_rows),
            "maximum_train_signal_date_idx": built.fold.maximum_train_signal_date_idx,
            "maximum_train_signal_date": str(
                inputs.date_values[int(built.fold.maximum_train_signal_date_idx)]
            ),
            "training_seconds": elapsed,
            "parameters": parameters,
            "metrics": metrics,
            "prefix_equivalence": equivalence,
            "adaptive_source": _file_record(source["_result_path"]),
            "config_sha256": _config_hash(study_path),
            "files": files,
        }
        _write_json(result_path, result)
        _emit(
            "entry_safety_training_completed",
            task_id=task["task_id"],
            adaptive_rounds=adaptive_rounds,
            fixed_rounds=rounds,
            seconds=elapsed,
        )
        del sequence, candidate_predictions, evaluation_predictions
        return result
    finally:
        contract_v2.base._release_training_memory(built.datasets)
        del model, built
        gc.collect()


def _load_task_result(
    *,
    study: Mapping[str, Any],
    study_path: Path,
    feature_study: Mapping[str, Any],
    v2_study: Mapping[str, Any],
    output_root: Path,
    task: Mapping[str, Any],
) -> dict[str, Any]:
    path = _task_result_path(output_root, task)
    if not _task_complete(
        path,
        task=task,
        study_path=study_path,
        feature_study=feature_study,
        v2_study=v2_study,
    ):
        raise ValueError(f"entry fixed-capacity task is incomplete: {task['task_id']}")
    return json.loads(path.read_text(encoding="utf-8"))


def _mfe_source(
    *,
    study: Mapping[str, Any],
    study_path: Path,
    feature_study: Mapping[str, Any],
    v2_study: Mapping[str, Any],
    output_root: Path,
    horizon: int,
    year: int,
) -> tuple[dict[str, Any], np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if year in HISTORICAL_YEARS:
        task = next(
            current
            for current in _tasks(study)
            if current["kind"] == "mfe"
            and int(current["horizon"]) == int(horizon)
            and int(current["year"]) == int(year)
        )
        result = _load_task_result(
            study=study,
            study_path=study_path,
            feature_study=feature_study,
            v2_study=v2_study,
            output_root=output_root,
            task=task,
        )
        files = dict(result["files"])
        return (
            result,
            np.load(
                _resolve(files["evaluation_prediction"]["path"]),
                allow_pickle=False,
            ),
            np.load(_resolve(files["evaluation_rows"]["path"]), allow_pickle=False),
            np.load(
                _resolve(files["candidate_prediction"]["path"]),
                allow_pickle=False,
            ),
            np.load(_resolve(files["candidate_rows"]["path"]), allow_pickle=False),
        )
    capacity_study = final_capacity.load_study(
        _resolve(study["sources"]["final_capacity_config"])
    )
    capacity_root = _resolve(study["sources"]["final_capacity_output_root"])
    task = next(
        current
        for current in final_capacity._outer_tasks(capacity_study)
        if int(current["horizon"]) == int(horizon) and int(current["year"]) == int(year)
    )
    result, evaluation, candidate, evaluation_rows, candidate_rows = (
        final_capacity._load_outer_arrays(capacity_root, task=task)
    )
    if "fixed_512" not in result["policy_order"]:
        raise ValueError(f"fixed-512 prefix is absent: {task['task_id']}")
    position = list(result["policy_order"]).index("fixed_512")
    if int(result["policy_iterations"]["fixed_512"]) != 512:
        raise ValueError(f"fixed-512 prefix changed: {task['task_id']}")
    return (
        result,
        np.asarray(evaluation[:, position], dtype=np.float32),
        np.asarray(evaluation_rows, dtype=np.int64),
        np.asarray(candidate[:, position], dtype=np.float32),
        np.asarray(candidate_rows, dtype=np.int64),
    )


def _mfe_diagnostic(
    *,
    study: Mapping[str, Any],
    inputs: base.LearnabilityInputs,
    reader: objective.FuturePathReader,
    horizon: int,
    year: int,
    prediction: np.ndarray,
    rows: np.ndarray,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    target = np.asarray(inputs.label_values("mfe", horizon)[rows], dtype=np.float32)
    ranking_daily, regression = base.regression_metrics(
        date_idx=inputs.candidate_date_idx[rows],
        actual=target,
        prediction=prediction,
        date_values=inputs.date_values,
        horizon=horizon,
    )
    early_horizon = int(study["mfe"]["early_horizon_by_target"][str(horizon)])
    actual = role._path_actual(
        inputs=inputs,
        reader=reader,
        rows=rows,
        horizon=horizon,
        early_horizon=early_horizon,
    )
    path_daily, path_metrics = objective.path_quality_metrics(
        date_idx=actual.date_idx,
        score=prediction,
        target_mfe=actual.target_mfe,
        early_mfe=actual.early_mfe,
        peak_day=actual.peak_day,
        pre_peak_mae=actual.pre_peak_mae,
        endpoint_return=actual.endpoint_return,
        state=actual.state,
        horizon=horizon,
    )
    daily = ranking_daily.merge(path_daily, on="date_idx", how="inner")
    if daily.empty:
        raise ValueError(f"MFE diagnostic daily merge is empty: D{horizon} {year}")
    bias = float(regression["date_equal_prediction_mean"]) - float(
        regression["date_equal_target_mean"]
    )
    metrics = {
        "year": int(year),
        "horizon": int(horizon),
        "fixed_rounds": 512,
        "row_count": int(regression["row_count"]),
        "rank_ic": float(regression["ranking"]["rank_ic_mean"]),
        "decile_spearman": float(regression["ranking"]["decile_spearman"]),
        "top1_mfe": float(regression["ranking"]["top_1pct_mean"]),
        "top5_mfe": float(regression["ranking"]["top_5pct_mean"]),
        "top1_mfe_lift": float(regression["ranking"]["top_1pct_lift"]),
        "top5_mfe_lift": float(regression["ranking"]["top_5pct_lift"]),
        "date_equal_mae": float(regression["date_equal_mae"]),
        "target_mean": float(regression["date_equal_target_mean"]),
        "prediction_mean": float(regression["date_equal_prediction_mean"]),
        "prediction_bias": bias,
        "top1_tail_rate": float(path_metrics["top1_daily_tail_rate"]),
        "top5_tail_rate": float(path_metrics["top5_daily_tail_rate"]),
        "top1_early_opportunity_share": float(
            path_metrics["top1_early_opportunity_share"]
        ),
        "top5_early_opportunity_share": float(
            path_metrics["top5_early_opportunity_share"]
        ),
        "top1_peak_day_mean": float(path_metrics["top1_peak_day_mean"]),
        "top5_peak_day_mean": float(path_metrics["top5_peak_day_mean"]),
        "top1_pre_peak_mae_mean": float(path_metrics["top1_pre_peak_mae_mean"]),
        "top5_pre_peak_mae_mean": float(path_metrics["top5_pre_peak_mae_mean"]),
        "top1_endpoint_return_mean": float(path_metrics["top1_endpoint_return_mean"]),
        "top5_endpoint_return_mean": float(path_metrics["top5_endpoint_return_mean"]),
        "top1_high_state_rate": float(path_metrics["top1_high_state_rate"]),
        "top5_high_state_rate": float(path_metrics["top5_high_state_rate"]),
    }
    return daily, metrics


def _relative_harm(challenger: float, baseline: float) -> float:
    denominator = abs(float(baseline))
    if denominator <= 1.0e-12:
        return math.nan
    return (float(challenger) - float(baseline)) / denominator


def _harm_stouffer(records: Sequence[Mapping[str, Any]]) -> dict[str, float]:
    z_values: list[float] = []
    weights: list[float] = []
    for record in records:
        hac = dict(record["primary_delta_hac"])
        mean = float(hac["mean"])
        two_sided = float(hac["p_value_two_sided"])
        count = int(hac["count"])
        if not (math.isfinite(mean) and math.isfinite(two_sided) and count > 0):
            continue
        harm_p = two_sided / 2.0 if mean <= 0.0 else 1.0 - two_sided / 2.0
        z_values.append(float(stats.norm.isf(np.clip(harm_p, 1.0e-15, 1.0 - 1.0e-15))))
        weights.append(math.sqrt(float(count)))
    if not z_values:
        return {"z_statistic": math.nan, "p_value_one_sided_harm": math.nan}
    weight = np.asarray(weights, dtype=np.float64)
    statistic = float(
        np.dot(weight, np.asarray(z_values, dtype=np.float64))
        / math.sqrt(float(np.dot(weight, weight)))
    )
    return {
        "z_statistic": statistic,
        "p_value_one_sided_harm": float(stats.norm.sf(statistic)),
    }


def _safety_annual(
    *,
    result: Mapping[str, Any],
    task: Mapping[str, Any],
) -> tuple[pd.DataFrame, dict[str, Any]]:
    daily = pd.read_parquet(_resolve(result["files"]["daily_metrics"]["path"]))
    adaptive = daily.loc[daily["prefix"] == "adaptive"].sort_values("date_idx")
    fixed = daily.loc[daily["prefix"] == "fixed_256"].sort_values("date_idx")
    if not np.array_equal(
        adaptive["date_idx"].to_numpy(), fixed["date_idx"].to_numpy()
    ):
        raise ValueError(f"safety daily dates changed: {task['task_id']}")
    merged = adaptive.merge(
        fixed,
        on="date_idx",
        suffixes=("_adaptive", "_fixed_256"),
        validate="one_to_one",
    )
    if task["target"] == "state":
        primary = "ordinal_ic"
        deltas = {
            "ordinal_ic_delta": float(
                np.nanmean(
                    merged["ordinal_ic_fixed_256"] - merged["ordinal_ic_adaptive"]
                )
            ),
            "high_state_top5_lift_delta": float(
                np.nanmean(
                    merged["high_state_top5_lift_fixed_256"]
                    - merged["high_state_top5_lift_adaptive"]
                )
            ),
            "relative_brier_harm": _relative_harm(
                float(np.nanmean(merged["brier_fixed_256"])),
                float(np.nanmean(merged["brier_adaptive"])),
            ),
            "relative_logloss_harm": _relative_harm(
                float(np.nanmean(merged["logloss_fixed_256"])),
                float(np.nanmean(merged["logloss_adaptive"])),
            ),
            "collapsed": bool(result["metrics"]["fixed_256"]["collapsed"]),
        }
    else:
        primary = "rank_ic"
        deltas = {
            "rank_ic_delta": float(
                np.nanmean(merged["rank_ic_fixed_256"] - merged["rank_ic_adaptive"])
            ),
            "relative_mae_harm": _relative_harm(
                float(np.nanmean(merged["mae_fixed_256"])),
                float(np.nanmean(merged["mae_adaptive"])),
            ),
            "deep_adverse_pr_auc_delta": float(
                np.nanmean(
                    merged["deep_adverse_pr_auc_fixed_256"]
                    - merged["deep_adverse_pr_auc_adaptive"]
                )
            ),
            "absolute_bias_harm": float(
                np.nanmean(
                    merged["absolute_bias_fixed_256"] - merged["absolute_bias_adaptive"]
                )
            ),
        }
    primary_delta = (
        merged[f"{primary}_fixed_256"] - merged[f"{primary}_adaptive"]
    ).to_numpy(dtype=np.float64)
    horizon = int(task["horizon"])
    annual = {
        "head": task["name"],
        "target": task["target"],
        "horizon": horizon,
        "year": int(task["year"]),
        "adaptive_rounds": int(result["adaptive_rounds"]),
        "fixed_rounds": int(result["trained_rounds"]),
        "primary_delta_hac": base._hac_mean_test(
            primary_delta, maximum_lag=max(horizon - 1, 0)
        ),
        "prefix_equivalence": result["prefix_equivalence"],
        **deltas,
    }
    return merged, annual


def _state_gate(
    *,
    annual: Sequence[Mapping[str, Any]],
    rules: Mapping[str, Any],
) -> dict[str, Any]:
    recent = sorted(
        (row for row in annual if int(row["year"]) in DECISION_YEARS),
        key=lambda row: int(row["year"]),
    )
    history = sorted(
        (row for row in annual if int(row["year"]) in HISTORICAL_YEARS),
        key=lambda row: int(row["year"]),
    )
    ordinal = [float(row["ordinal_ic_delta"]) for row in recent]
    high = [float(row["high_state_top5_lift_delta"]) for row in recent]
    brier = [float(row["relative_brier_harm"]) for row in recent]
    logloss = [float(row["relative_logloss_harm"]) for row in recent]
    historical_bad = [
        int(row["year"])
        for row in history
        if (
            float(row["ordinal_ic_delta"])
            < float(rules["historical_bad_ordinal_ic_delta"])
            or float(row["relative_brier_harm"])
            > float(rules["historical_bad_relative_proper_score_harm"])
            or float(row["relative_logloss_harm"])
            > float(rules["historical_bad_relative_proper_score_harm"])
        )
    ]
    checks = {
        "worst_decision_year_ordinal_ic_delta": min(ordinal),
        "median_decision_year_ordinal_ic_delta": float(np.median(ordinal)),
        "nonnegative_high_state_top5_years": sum(value >= 0.0 for value in high),
        "worst_high_state_top5_delta": min(high),
        "median_relative_brier_harm": float(np.median(brier)),
        "worst_relative_brier_harm": max(brier),
        "median_relative_logloss_harm": float(np.median(logloss)),
        "worst_relative_logloss_harm": max(logloss),
        "collapsed_years": [
            int(row["year"]) for row in annual if bool(row["collapsed"])
        ],
        "historical_bad_years": historical_bad,
    }
    passed = bool(
        checks["worst_decision_year_ordinal_ic_delta"]
        >= float(rules["minimum_worst_decision_year_ordinal_ic_delta"])
        and checks["median_decision_year_ordinal_ic_delta"]
        >= float(rules["minimum_median_decision_year_ordinal_ic_delta"])
        and checks["nonnegative_high_state_top5_years"]
        >= int(rules["minimum_nonnegative_high_state_top5_years"])
        and checks["worst_high_state_top5_delta"]
        >= -float(rules["maximum_worst_high_state_top5_decline"])
        and checks["median_relative_brier_harm"]
        <= float(rules["maximum_median_relative_brier_harm"])
        and checks["worst_relative_brier_harm"]
        <= float(rules["maximum_worst_relative_brier_harm"])
        and checks["median_relative_logloss_harm"]
        <= float(rules["maximum_median_relative_logloss_harm"])
        and checks["worst_relative_logloss_harm"]
        <= float(rules["maximum_worst_relative_logloss_harm"])
        and not checks["collapsed_years"]
        and len(historical_bad) < int(rules["historical_minimum_bad_years"])
    )
    return {
        "passed": passed,
        "policy": "fixed_256" if passed else "v3_time_consistent_adaptive",
        "checks": checks,
        "annual_ordinal_ic_delta": ordinal,
        "annual_high_state_top5_lift_delta": high,
        "annual_relative_brier_harm": brier,
        "annual_relative_logloss_harm": logloss,
    }


def _risk_gate(
    *,
    annual: Sequence[Mapping[str, Any]],
    rules: Mapping[str, Any],
) -> dict[str, Any]:
    recent = sorted(
        (row for row in annual if int(row["year"]) in DECISION_YEARS),
        key=lambda row: int(row["year"]),
    )
    history = sorted(
        (row for row in annual if int(row["year"]) in HISTORICAL_YEARS),
        key=lambda row: int(row["year"]),
    )
    rank_ic = [float(row["rank_ic_delta"]) for row in recent]
    mae = [float(row["relative_mae_harm"]) for row in recent]
    pr_auc = [float(row["deep_adverse_pr_auc_delta"]) for row in recent]
    bias = [float(row["absolute_bias_harm"]) for row in recent]
    historical_bad = [
        int(row["year"])
        for row in history
        if (
            float(row["rank_ic_delta"]) < float(rules["historical_bad_rank_ic_delta"])
            or float(row["relative_mae_harm"])
            > float(rules["historical_bad_relative_mae_harm"])
            or float(row["deep_adverse_pr_auc_delta"])
            < -float(rules["historical_bad_pr_auc_decline"])
        )
    ]
    checks = {
        "worst_decision_year_rank_ic_delta": min(rank_ic),
        "median_decision_year_rank_ic_delta": float(np.median(rank_ic)),
        "median_relative_mae_harm": float(np.median(mae)),
        "worst_relative_mae_harm": max(mae),
        "nonnegative_deep_adverse_pr_auc_years": sum(value >= 0.0 for value in pr_auc),
        "worst_deep_adverse_pr_auc_delta": min(pr_auc),
        "absolute_bias_worsened_all_decision_years": all(value > 0.0 for value in bias),
        "historical_bad_years": historical_bad,
    }
    passed = bool(
        checks["worst_decision_year_rank_ic_delta"]
        >= float(rules["minimum_worst_decision_year_rank_ic_delta"])
        and checks["median_decision_year_rank_ic_delta"]
        >= float(rules["minimum_median_decision_year_rank_ic_delta"])
        and checks["median_relative_mae_harm"]
        <= float(rules["maximum_median_relative_mae_harm"])
        and checks["worst_relative_mae_harm"]
        <= float(rules["maximum_worst_relative_mae_harm"])
        and checks["nonnegative_deep_adverse_pr_auc_years"]
        >= int(rules["minimum_nonnegative_deep_adverse_pr_auc_years"])
        and checks["worst_deep_adverse_pr_auc_delta"]
        >= -float(rules["maximum_worst_deep_adverse_pr_auc_decline"])
        and not checks["absolute_bias_worsened_all_decision_years"]
        and len(historical_bad) < int(rules["historical_minimum_bad_years"])
    )
    return {
        "passed": passed,
        "policy": "fixed_256" if passed else "v3_time_consistent_adaptive",
        "checks": checks,
        "annual_rank_ic_delta": rank_ic,
        "annual_relative_mae_harm": mae,
        "annual_deep_adverse_pr_auc_delta": pr_auc,
        "annual_absolute_bias_harm": bias,
    }


def evaluate(
    *,
    study_path: Path = DEFAULT_STUDY_PATH,
    output_root: Path = DEFAULT_OUTPUT_ROOT,
) -> dict[str, Any]:
    study = load_study(study_path)
    feature_study = feature.load_study(
        _resolve(study["sources"]["feature_study_config"])
    )
    v2_study = contract_v2.load_study(
        _resolve(study["sources"]["entry_contract_v2_config"])
    )
    inputs, pack = feature._load_validated_inputs(feature_study)
    if inputs.maximum_outcome_date != "2025-12-31":
        raise ValueError("input outcome boundary changed")
    for task in _tasks(study):
        _load_task_result(
            study=study,
            study_path=study_path,
            feature_study=feature_study,
            v2_study=v2_study,
            output_root=output_root,
            task=task,
        )
    reader = objective.FuturePathReader(pack)
    mfe_annual: list[dict[str, Any]] = []
    for horizon in MFE_HORIZONS:
        for year in CONTRACT_YEARS:
            source, prediction, rows, _candidate, _candidate_rows = _mfe_source(
                study=study,
                study_path=study_path,
                feature_study=feature_study,
                v2_study=v2_study,
                output_root=output_root,
                horizon=horizon,
                year=year,
            )
            daily, metrics = _mfe_diagnostic(
                study=study,
                inputs=inputs,
                reader=reader,
                horizon=horizon,
                year=year,
                prediction=prediction,
                rows=rows,
            )
            path = (
                output_root
                / "evaluation"
                / "mfe"
                / f"h{horizon:02d}"
                / f"fold_{year}_daily.parquet"
            )
            path.parent.mkdir(parents=True, exist_ok=True)
            daily.to_parquet(path, index=False, compression="zstd")
            mfe_annual.append(
                {
                    **metrics,
                    "source_task_id": source["task_id"],
                    "source_task_result": _file_record(
                        _task_result_path(
                            output_root,
                            next(
                                task
                                for task in _tasks(study)
                                if task["kind"] == "mfe"
                                and int(task["horizon"]) == horizon
                                and int(task["year"]) == year
                            ),
                        )
                    )
                    if year in HISTORICAL_YEARS
                    else source["files"]["model"],
                    "daily_metrics": _file_record(path),
                }
            )
    safety_annual: list[dict[str, Any]] = []
    by_head: dict[str, list[dict[str, Any]]] = {name: [] for name in HEAD_NAMES}
    for task in _tasks(study):
        if task["kind"] != "safety":
            continue
        result = _load_task_result(
            study=study,
            study_path=study_path,
            feature_study=feature_study,
            v2_study=v2_study,
            output_root=output_root,
            task=task,
        )
        merged, annual = _safety_annual(result=result, task=task)
        path = (
            output_root
            / "evaluation"
            / "safety"
            / str(task["name"])
            / f"fold_{int(task['year'])}_paired_daily.parquet"
        )
        path.parent.mkdir(parents=True, exist_ok=True)
        merged.to_parquet(path, index=False, compression="zstd")
        annual["paired_daily"] = _file_record(path)
        safety_annual.append(annual)
        by_head[str(task["name"])].append(annual)
    gates: dict[str, dict[str, Any]] = {}
    harm_combined: dict[str, dict[str, float]] = {}
    for name in HEAD_NAMES:
        ordered = sorted(by_head[name], key=lambda row: int(row["year"]))
        if len(ordered) != len(CONTRACT_YEARS):
            raise ValueError(f"safety annual evidence is incomplete: {name}")
        if name == "state_10":
            gate = _state_gate(
                annual=ordered,
                rules=study["safety_gate"]["state"],
            )
        else:
            gate = _risk_gate(
                annual=ordered,
                rules=study["safety_gate"]["risk"],
            )
        harm = _harm_stouffer(
            [row for row in ordered if int(row["year"]) in DECISION_YEARS]
        )
        harm_combined[name] = harm
        gates[name] = {
            **gate,
            "head": name,
            "fixed_rounds": 256,
            "annual": ordered,
            "combined_primary_harm_test": harm,
        }
    p_values = {
        name: float(record["p_value_one_sided_harm"])
        for name, record in harm_combined.items()
    }
    q_values = feature._benjamini_hochberg(p_values)
    for name in HEAD_NAMES:
        gates[name]["combined_primary_harm_bh_q_value"] = float(q_values[name])
    decision = {
        "status": "entry_fixed_capacity_policy_frozen",
        "mfe": {
            "mfe_10": {
                "head": _head(study, 10)["name"],
                "policy": "fixed_512",
                "rounds": 512,
                "rank_first": True,
                "literal_expected_return": False,
                "owner_utility_choice": True,
            },
            "mfe_20": {
                "head": _head(study, 20)["name"],
                "policy": "fixed_512",
                "rounds": 512,
                "rank_first": True,
                "literal_expected_return": False,
                "owner_utility_choice": True,
            },
        },
        "state_and_risk": {
            name: {
                "policy": gates[name]["policy"],
                "rounds": 256 if gates[name]["passed"] else None,
                "passed_safety_gate": bool(gates[name]["passed"]),
            }
            for name in HEAD_NAMES
        },
        "next_step": "materialize_seq100_entry_contract_oos_v4",
        "does_not_select": list(study["non_selections"]),
    }
    summary = {
        "schema": SUMMARY_SCHEMA,
        "status": "completed_entry_fixed_capacity_audit",
        "completed_at": _now(),
        "study_id": STUDY_ID,
        "config_sha256": _config_hash(study_path),
        "mfe_fixed_512_diagnostics": mfe_annual,
        "state_and_risk_safety_gates": gates,
        "decision": decision,
        "scope": {
            "maximum_consumed_outcome_date": inputs.maximum_outcome_date,
            "forbidden_2026_row_count": 0,
            "new_mfe_booster_count": 6,
            "new_state_and_risk_booster_count": 18,
            "total_new_booster_count": 24,
        },
    }
    _write_json(output_root / "summary.json", summary)
    _write_json(output_root / "decision.json", decision)
    return summary


def _selected_safety_values(
    *,
    study: Mapping[str, Any],
    study_path: Path,
    feature_study: Mapping[str, Any],
    v2_study: Mapping[str, Any],
    output_root: Path,
    name: str,
    year: int,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    task = next(
        current
        for current in _tasks(study)
        if current["kind"] == "safety"
        and current["name"] == name
        and int(current["year"]) == int(year)
    )
    result = _load_task_result(
        study=study,
        study_path=study_path,
        feature_study=feature_study,
        v2_study=v2_study,
        output_root=output_root,
        task=task,
    )
    files = dict(result["files"])
    rows = np.load(_resolve(files["candidate_rows"]["path"]), allow_pickle=False)
    values = np.load(
        _resolve(files["fixed_256_candidate_prediction"]["path"]),
        allow_pickle=False,
    )
    return (
        np.asarray(rows, dtype=np.int64),
        np.asarray(values, dtype=np.float32),
        {
            "task_id": result["task_id"],
            "model_year": int(year),
            "policy": "fixed_256",
            "rounds": 256,
            "internal_tree_count": int(result["internal_tree_count"]),
            "atlas_cutoff": result.get("atlas_cutoff"),
            "task_result": _file_record(_task_result_path(output_root, task)),
        },
    )


def _rank_matrix(raw: np.ndarray, dates: np.ndarray) -> np.ndarray:
    values = np.asarray(raw, dtype=np.float32)
    date_idx = np.asarray(dates, dtype=np.int32)
    if values.ndim != 2 or len(values) != len(date_idx):
        raise ValueError("contract rank inputs are not aligned")
    ranked = np.column_stack(
        [
            contract_v2._rank_by_date(values[:, column], date_idx)
            for column in range(values.shape[1])
        ]
    ).astype(np.float32, copy=False)
    finite = np.isfinite(ranked)
    if bool(np.any(ranked[finite] < 0.0)) or bool(np.any(ranked[finite] > 1.0)):
        raise AssertionError("date ranks left the unit interval")
    return ranked


def materialize_contract_v4(
    *,
    study: Mapping[str, Any],
    study_path: Path,
    output_root: Path,
    summary: Mapping[str, Any],
    feature_study: Mapping[str, Any],
    v2_study: Mapping[str, Any],
    inputs: base.LearnabilityInputs,
    v3_manifest: Mapping[str, Any],
    v3_rows: np.ndarray,
    v3_raw: np.ndarray,
) -> dict[str, Any]:
    decision = dict(summary["decision"])
    if decision.get("status") != "entry_fixed_capacity_policy_frozen":
        raise ValueError("fixed-capacity decision is not frozen")
    rows = np.asarray(v3_rows, dtype=np.int64)
    raw = np.asarray(v3_raw, dtype=np.float32).copy()
    if len(rows) != int(study["contract_v4"]["candidate_count"]):
        raise ValueError("v4 candidate count changed")
    sources: dict[str, Any] = {}
    for horizon, column in ((10, 0), (20, 1)):
        row_parts: list[np.ndarray] = []
        value_parts: list[np.ndarray] = []
        year_sources: list[dict[str, Any]] = []
        for year in CONTRACT_YEARS:
            source, _evaluation, _evaluation_rows, values, current_rows = _mfe_source(
                study=study,
                study_path=study_path,
                feature_study=feature_study,
                v2_study=v2_study,
                output_root=output_root,
                horizon=horizon,
                year=year,
            )
            expected_rows = contract_v2._year_rows(inputs, year)
            if not np.array_equal(current_rows, expected_rows):
                raise ValueError(f"MFE v4 candidate rows changed: D{horizon} {year}")
            row_parts.append(current_rows)
            value_parts.append(values)
            year_sources.append(
                {
                    "year": int(year),
                    "task_id": source["task_id"],
                    "rounds": 512,
                    "source": (
                        "new_2020_2022_fixed_outer"
                        if year in HISTORICAL_YEARS
                        else "reused_final_capacity_fixed_512_prefix"
                    ),
                    "task_result": (
                        _file_record(
                            _task_result_path(
                                output_root,
                                next(
                                    task
                                    for task in _tasks(study)
                                    if task["kind"] == "mfe"
                                    and int(task["horizon"]) == horizon
                                    and int(task["year"]) == year
                                ),
                            )
                        )
                        if year in HISTORICAL_YEARS
                        else source["files"]["model"]
                    ),
                }
            )
        combined_rows = np.concatenate(row_parts)
        if not np.array_equal(combined_rows, rows):
            raise ValueError(f"MFE v4 combined rows changed: D{horizon}")
        combined = np.concatenate(value_parts).astype(np.float32, copy=False)
        if not bool(np.isfinite(combined).all()):
            raise ValueError(f"MFE v4 predictions are non-finite: D{horizon}")
        raw[:, column] = combined
        sources[f"mfe_{horizon}"] = {
            "head": _head(study, horizon)["name"],
            "families": list(_head(study, horizon)["families"]),
            "policy": "fixed_512",
            "rank_first": True,
            "literal_expected_return": False,
            "years": year_sources,
        }
    replaced_heads: list[str] = []
    retained_heads: list[str] = []
    for name in HEAD_NAMES:
        current = decision["state_and_risk"][name]
        columns = _safety_head(study, name)["columns"]
        if not bool(current["passed_safety_gate"]):
            retained_heads.append(name)
            if not np.array_equal(
                raw[:, list(columns)],
                np.asarray(v3_raw)[:, list(columns)],
                equal_nan=True,
            ):
                raise AssertionError(f"retained v3 columns changed: {name}")
            sources[name] = {
                "policy": "v3_time_consistent_adaptive",
                "reused_v3_exact": True,
                "v3_columns": list(columns),
            }
            continue
        row_parts = []
        value_parts = []
        year_sources = []
        for year in CONTRACT_YEARS:
            current_rows, values, source_record = _selected_safety_values(
                study=study,
                study_path=study_path,
                feature_study=feature_study,
                v2_study=v2_study,
                output_root=output_root,
                name=name,
                year=year,
            )
            expected_rows = contract_v2._year_rows(inputs, year)
            if not np.array_equal(current_rows, expected_rows):
                raise ValueError(f"safety v4 candidate rows changed: {name} {year}")
            row_parts.append(current_rows)
            value_parts.append(values)
            year_sources.append(source_record)
        if not np.array_equal(np.concatenate(row_parts), rows):
            raise ValueError(f"safety v4 combined rows changed: {name}")
        values = np.concatenate(value_parts)
        if name == "state_10":
            if values.shape != (len(rows), 3) or _state_collapsed(values):
                raise ValueError("fixed state v4 probability is invalid")
            raw[:, list(columns)] = values
        else:
            values = np.asarray(values, dtype=np.float32)
            if values.shape != (len(rows),) or not bool(np.isfinite(values).all()):
                raise ValueError(f"fixed risk v4 prediction is invalid: {name}")
            raw[:, columns[0]] = values
        replaced_heads.append(name)
        sources[name] = {
            "policy": "fixed_256",
            "rounds": 256,
            "columns": list(columns),
            "years": year_sources,
        }
    if not np.array_equal(rows, np.asarray(v3_rows)):
        raise AssertionError("v4 candidate rows differ from v3")
    dates = np.asarray(inputs.candidate_date_idx[rows], dtype=np.int32)
    years = np.asarray([int(str(inputs.date_values[value])[:4]) for value in dates])
    if bool(np.any(years == 2026)):
        raise AssertionError("a 2026 candidate reached v4")
    rank = _rank_matrix(raw, dates)
    v4_root = _resolve(study["contract_v4"]["output_root"])
    contract_dir = v4_root / "contract"
    rows_path = contract_dir / "candidate_rows.npy"
    raw_path = contract_dir / "raw_predictions.npy"
    rank_path = contract_dir / "date_rank_predictions.npy"
    _save_npy(rows_path, rows)
    _save_npy(raw_path, raw)
    _save_npy(rank_path, rank)
    failed_columns = [
        column
        for name in retained_heads
        for column in _safety_head(study, name)["columns"]
    ]
    if failed_columns and not np.array_equal(
        raw[:, failed_columns],
        np.asarray(v3_raw)[:, failed_columns],
        equal_nan=True,
    ):
        raise AssertionError("a safety-gate failure did not preserve v3 raw values")
    manifest = {
        "schema": CONTRACT_V4_SCHEMA,
        "status": "completed_candidate_aligned_strict_oos_contract",
        "completed_at": _now(),
        "study_id": CONTRACT_V4_ID,
        "candidate_count": len(rows),
        "candidate_years": list(CONTRACT_YEARS),
        "physical_columns": list(contract_v2.PHYSICAL_COLUMNS),
        "logical_coordinates": list(v3_manifest["logical_coordinates"]),
        "semantics": {
            "mfe": {
                "rank_first": True,
                "literal_expected_return": False,
                "raw_preserved": True,
            },
            "state": {
                "raw_probability_preserved": True,
                "literal_stable_probability": False,
                "reader_name": "relative state score",
            },
            "fusion": False,
        },
        "capacity_policy": decision,
        "sources": sources,
        "replaced_heads": ["mfe_10", "mfe_20", *replaced_heads],
        "retained_v3_heads": retained_heads,
        "candidate_rows_equal_v3": True,
        "failed_head_raw_columns_equal_v3": True,
        "all_date_ranks_regenerated": True,
        "v3_source": {
            "manifest": _file_record(
                _resolve(study["sources"]["entry_contract_v3_manifest"])
            ),
            "files": v3_manifest["files"],
        },
        "post_entry_ab_alignment": {
            "seq100_post_entry_ab_v1": "v2_only_historical_evidence",
            "v4_revalidation_required": True,
            "retrained_in_this_stage": False,
        },
        "files": {
            "candidate_rows": _file_record(
                rows_path, shape=list(rows.shape), dtype=str(rows.dtype)
            ),
            "raw_predictions": _file_record(
                raw_path, shape=list(raw.shape), dtype=str(raw.dtype)
            ),
            "date_rank_predictions": _file_record(
                rank_path, shape=list(rank.shape), dtype=str(rank.dtype)
            ),
        },
        "scope": {
            "maximum_consumed_outcome_date": inputs.maximum_outcome_date,
            "forbidden_2026_row_count": 0,
            "new_mfe_booster_count": 6,
            "new_state_and_risk_booster_count": 18,
            "fusion": False,
        },
        "does_not_select": list(study["non_selections"]),
    }
    manifest_path = v4_root / "contract_manifest.json"
    _write_json(manifest_path, manifest)
    _write_json(
        v4_root / "decision.json",
        {
            "status": "entry_contract_v4_frozen",
            "active_contract": CONTRACT_V4_ID,
            "capacity_policy": decision,
            "next_step": "prepare_seq100_post_entry_ab_v2_landmarks",
            "does_not_select": list(study["non_selections"]),
        },
    )
    record_root = _resolve(study["contract_v4"]["research_record_root"])
    _write_json(
        record_root / "config.json",
        {
            "study_id": CONTRACT_V4_ID,
            "source_capacity_study": STUDY_ID,
            "source_capacity_config": _file_record(study_path),
            "source_capacity_decision": _file_record(output_root / "decision.json"),
            "source_entry_contract_v3": _file_record(
                _resolve(study["sources"]["entry_contract_v3_manifest"])
            ),
            "rank_first": True,
            "literal_expected_return": False,
            "maximum_outcome_date": "2025-12-31",
            "forbidden_outcome_year": 2026,
        },
    )
    _write_json(
        record_root / "result.json",
        {**manifest, "full_output": _file_record(manifest_path)},
    )
    return manifest


def _write_audit_record(
    *,
    study: Mapping[str, Any],
    study_path: Path,
    output_root: Path,
    summary: Mapping[str, Any],
    manifest: Mapping[str, Any],
) -> None:
    final_summary = {
        **summary,
        "entry_contract_v4": {
            "status": "completed",
            "manifest": _file_record(
                _resolve(study["contract_v4"]["output_root"]) / "contract_manifest.json"
            ),
            "candidate_count": int(manifest["candidate_count"]),
            "replaced_heads": list(manifest["replaced_heads"]),
            "retained_v3_heads": list(manifest["retained_v3_heads"]),
        },
    }
    _write_json(output_root / "summary.json", final_summary)
    _write_json(DEFAULT_RECORD_ROOT / "config.json", study)
    _write_json(
        DEFAULT_RECORD_ROOT / "result.json",
        {
            **final_summary,
            "source_config": _file_record(study_path),
            "full_output": _file_record(output_root / "summary.json"),
        },
    )


def run_pending(
    *,
    study_path: Path = DEFAULT_STUDY_PATH,
    output_root: Path = DEFAULT_OUTPUT_ROOT,
) -> dict[str, Any]:
    study = load_study(study_path)
    feature_study = feature.load_study(
        _resolve(study["sources"]["feature_study_config"])
    )
    v2_study = contract_v2.load_study(
        _resolve(study["sources"]["entry_contract_v2_config"])
    )
    inputs, _pack = feature._load_validated_inputs(feature_study)
    if inputs.maximum_outcome_date != "2025-12-31":
        raise ValueError("input outcome boundary changed")
    v3_manifest = _load_v3_manifest(study)
    v3_rows, v3_raw, _v3_rank = _load_v3_arrays(v3_manifest)
    for task in _tasks(study):
        if task["kind"] == "mfe":
            _run_mfe_task(
                task=task,
                study=study,
                study_path=study_path,
                feature_study=feature_study,
                v2_study=v2_study,
                inputs=inputs,
                output_root=output_root,
            )
        else:
            _run_safety_task(
                task=task,
                study=study,
                study_path=study_path,
                feature_study=feature_study,
                v2_study=v2_study,
                inputs=inputs,
                v3_rows=v3_rows,
                v3_raw=v3_raw,
                output_root=output_root,
            )
    summary = evaluate(study_path=study_path, output_root=output_root)
    manifest = materialize_contract_v4(
        study=study,
        study_path=study_path,
        output_root=output_root,
        summary=summary,
        feature_study=feature_study,
        v2_study=v2_study,
        inputs=inputs,
        v3_manifest=v3_manifest,
        v3_rows=v3_rows,
        v3_raw=v3_raw,
    )
    _write_audit_record(
        study=study,
        study_path=study_path,
        output_root=output_root,
        summary=summary,
        manifest=manifest,
    )
    return json.loads((output_root / "summary.json").read_text(encoding="utf-8"))


def status(
    *,
    study_path: Path = DEFAULT_STUDY_PATH,
    output_root: Path = DEFAULT_OUTPUT_ROOT,
) -> dict[str, Any]:
    study = load_study(study_path)
    feature_study = feature.load_study(
        _resolve(study["sources"]["feature_study_config"])
    )
    v2_study = contract_v2.load_study(
        _resolve(study["sources"]["entry_contract_v2_config"])
    )
    completed: list[str] = []
    pending: list[str] = []
    for task in _tasks(study):
        current = _task_complete(
            _task_result_path(output_root, task),
            task=task,
            study_path=study_path,
            feature_study=feature_study,
            v2_study=v2_study,
        )
        (completed if current else pending).append(str(task["task_id"]))
    v4_manifest = (
        _resolve(study["contract_v4"]["output_root"]) / "contract_manifest.json"
    )
    return {
        "study_id": STUDY_ID,
        "task_count": len(completed) + len(pending),
        "mfe_model_count": sum(task["kind"] == "mfe" for task in _tasks(study)),
        "state_and_risk_model_count": sum(
            task["kind"] == "safety" for task in _tasks(study)
        ),
        "completed": completed,
        "pending": pending,
        "evaluation_completed": (output_root / "summary.json").is_file(),
        "entry_contract_v4_completed": v4_manifest.is_file(),
    }


def self_test() -> dict[str, Any]:
    study = load_study()
    tasks = _tasks(study)
    if len(tasks) != 24:
        raise AssertionError("entry fixed-capacity task count changed")
    if sum(task["kind"] == "mfe" for task in tasks) != 6:
        raise AssertionError("MFE task count changed")
    if sum(task["kind"] == "safety" for task in tasks) != 18:
        raise AssertionError("state/risk task count changed")
    state_task = next(task for task in tasks if task["name"] == "state_10")
    if int(state_task["rounds"]) * 3 != 768:
        raise AssertionError("state internal tree interpretation changed")
    old = np.asarray([0.1, 0.2, 0.3, 0.4, 0.5, 0.6], dtype=np.float32)
    equivalence = _prefix_equivalence(
        old=old,
        current=old.copy(),
        dates=np.asarray([1, 1, 1, 2, 2, 2], dtype=np.int32),
        absolute_tolerance=1.0e-7,
        relative_tolerance=1.0e-6,
        minimum_daily_spearman=0.999999,
    )
    if not equivalence["passed"]:
        raise AssertionError("exact prefix did not pass equivalence")
    state_annual = [
        {
            "year": year,
            "ordinal_ic_delta": 0.001,
            "high_state_top5_lift_delta": 0.001,
            "relative_brier_harm": 0.0,
            "relative_logloss_harm": 0.0,
            "collapsed": False,
        }
        for year in CONTRACT_YEARS
    ]
    if not _state_gate(annual=state_annual, rules=study["safety_gate"]["state"])[
        "passed"
    ]:
        raise AssertionError("valid state gate example failed")
    risk_annual = [
        {
            "year": year,
            "rank_ic_delta": 0.001,
            "relative_mae_harm": 0.0,
            "deep_adverse_pr_auc_delta": 0.001,
            "absolute_bias_harm": -0.001 if year == 2025 else 0.0,
        }
        for year in CONTRACT_YEARS
    ]
    if not _risk_gate(annual=risk_annual, rules=study["safety_gate"]["risk"])["passed"]:
        raise AssertionError("valid risk gate example failed")
    raw = np.asarray(
        [[1.0, 2.0], [2.0, 1.0], [3.0, 4.0], [4.0, 3.0]],
        dtype=np.float32,
    )
    rank = _rank_matrix(raw, np.asarray([1, 1, 2, 2], dtype=np.int32))
    if not bool(((rank >= 0.0) & (rank <= 1.0)).all()):
        raise AssertionError("rank matrix left the unit interval")
    return {
        "status": "passed",
        "task_count": len(tasks),
        "mfe_rounds": 512,
        "state_and_risk_rounds": 256,
        "state_internal_tree_count": 768,
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Freeze Seq100 entry fixed capacities, safety-gate state/risk, "
            "and build the strict v4 contract."
        )
    )
    parser.add_argument("--study", type=Path, default=DEFAULT_STUDY_PATH)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--status", action="store_true")
    group.add_argument("--run-pending", action="store_true")
    group.add_argument("--evaluate", action="store_true")
    group.add_argument("--self-test", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    study_path = _resolve(arguments.study)
    output_root = _resolve(arguments.output_root)
    if arguments.status:
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
