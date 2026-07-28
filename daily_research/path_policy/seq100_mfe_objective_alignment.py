from __future__ import annotations

import argparse
import gc
import json
import math
import os
import subprocess
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

from daily_research.path_policy import seq100_mfe_feature_family_audit as source
from daily_research.path_policy import seq100_path_label_learnability as base
from daily_research.path_policy import seq100_short_horizon_target_reaudit as short

WORKSPACE_ROOT = source.WORKSPACE_ROOT
STUDY_ID = "seq100_mfe_objective_alignment_v1"
DEFAULT_STUDY_PATH = (
    WORKSPACE_ROOT / "daily_research/studies/seq100_mfe_objective_alignment_v1.json"
)
DEFAULT_OUTPUT_ROOT = (
    WORKSPACE_ROOT
    / "daily_research/output/path_policy/studies/seq100_mfe_objective_alignment_v1"
)
TARGET_HORIZONS = (10, 20)
FOLD_YEARS = (2023, 2024, 2025)
SOURCE_OBJECTIVE = "source_huber"
TRAINED_OBJECTIVES = ("tail_weighted_huber", "daily_lambdarank")
ALL_OBJECTIVES = (SOURCE_OBJECTIVE, *TRAINED_OBJECTIVES)
TASK_RESULT_SCHEMA = "seq100_mfe_objective_alignment_task_result/v1"
SUMMARY_SCHEMA = "seq100_mfe_objective_alignment_summary/v1"


def _now() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _resolve(path: str | Path) -> Path:
    value = Path(path)
    if not value.is_absolute():
        value = WORKSPACE_ROOT / value
    return value.resolve()


def _file_record(path: Path) -> dict[str, Any]:
    return {
        "path": path.relative_to(WORKSPACE_ROOT).as_posix(),
        "size": int(path.stat().st_size),
    }


def load_study(path: Path = DEFAULT_STUDY_PATH) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("study_id") != STUDY_ID:
        raise ValueError(f"study_id must be {STUDY_ID}")
    if tuple(int(value) for value in payload["labels"]["horizons"]) != TARGET_HORIZONS:
        raise ValueError("objective-alignment target horizons changed")
    folds = dict(payload["folds"])
    if tuple(int(value) for value in folds["fold_years"]) != FOLD_YEARS:
        raise ValueError("objective-alignment fold years changed")
    objectives = dict(payload["objectives"])
    if tuple(objectives) != ALL_OBJECTIVES:
        raise ValueError("objective-alignment objective order changed")
    tail = dict(objectives["tail_weighted_huber"])
    if float(tail["maximum_multiplier"]) <= 1.0 or float(tail["power"]) <= 0.0:
        raise ValueError("tail weighting must be positive and emphasize the upper tail")
    ranking = dict(objectives["daily_lambdarank"])
    if int(ranking["relevance_levels"]) < 10:
        raise ValueError("LambdaRank requires at least ten relevance levels")
    if str(ranking["label_gain"]) != "linear":
        raise ValueError("only linear LambdaRank gain is supported")
    source_study = source.load_study(_resolve(payload["source"]["study_config"]))
    if str(payload["model"]["library"]) != str(source_study["model"]["library"]):
        raise ValueError("objective-alignment LightGBM version differs from its source")
    return payload


def _source_study(study: Mapping[str, Any]) -> dict[str, Any]:
    return source.load_study(_resolve(study["source"]["study_config"]))


def _source_output_root(study: Mapping[str, Any]) -> Path:
    return _resolve(study["source"]["output_root"])


def _task_plan() -> list[dict[str, Any]]:
    return [
        {
            "task_id": f"mfe{horizon}_{year}_{objective}_outer",
            "stage": "outer_evaluation",
            "objective_variant": objective,
            "fold_year": year,
            "horizon": horizon,
        }
        for horizon in TARGET_HORIZONS
        for year in FOLD_YEARS
        for objective in TRAINED_OBJECTIVES
    ]


def _task_by_id(task_id: str) -> dict[str, Any]:
    for task in _task_plan():
        if task["task_id"] == str(task_id):
            return task
    raise ValueError(f"unknown objective-alignment task: {task_id}")


def _task_dir(output_root: Path, task: Mapping[str, Any]) -> Path:
    return (
        output_root
        / "outer"
        / f"fold_{int(task['fold_year'])}"
        / f"h{int(task['horizon']):02d}"
        / str(task["objective_variant"])
    )


def _task_result_path(output_root: Path, task: Mapping[str, Any]) -> Path:
    return _task_dir(output_root, task) / "task_result.json"


def _source_baseline_result(
    *,
    study: Mapping[str, Any],
    source_study: Mapping[str, Any],
    year: int,
    horizon: int,
) -> dict[str, Any]:
    output_root = _source_output_root(study)
    task = source._task_by_id(f"mfe{int(horizon)}_{int(year)}_baseline_outer")
    path = source._task_result_path(output_root, task)
    if not source._task_complete(
        path,
        task=task,
        study=source_study,
        output_root=output_root,
    ):
        raise ValueError(f"source Huber result cannot be loaded: D{horizon} {year}")
    return json.loads(path.read_text(encoding="utf-8"))


def _objective_parameters(
    *,
    study: Mapping[str, Any],
    source_study: Mapping[str, Any],
    objective_variant: str,
) -> dict[str, Any]:
    parameters, _maximum_rounds, _patience = source._model_parameters(source_study)
    if objective_variant == "tail_weighted_huber":
        return parameters
    if objective_variant != "daily_lambdarank":
        raise ValueError(f"unsupported trained objective: {objective_variant}")
    ranking = dict(study["objectives"][objective_variant])
    levels = int(ranking["relevance_levels"])
    parameters.pop("alpha", None)
    parameters.update(
        {
            "objective": "lambdarank",
            "metric": "ndcg",
            "label_gain": list(range(levels)),
            "lambdarank_truncation_level": int(
                ranking["lambdarank_truncation_level"]
            ),
            "lambdarank_norm": bool(ranking["lambdarank_norm"]),
        }
    )
    return parameters


def _date_boundaries(date_idx: np.ndarray) -> np.ndarray:
    dates = np.asarray(date_idx, dtype=np.int32)
    if dates.ndim != 1 or not dates.size:
        raise ValueError("date-grouped arrays must be non-empty and one-dimensional")
    if bool(np.any(dates[1:] < dates[:-1])):
        raise ValueError("date-grouped arrays must be ordered")
    return np.flatnonzero(np.r_[True, dates[1:] != dates[:-1], True])


def within_date_percentile(date_idx: np.ndarray, values: np.ndarray) -> np.ndarray:
    dates = np.asarray(date_idx, dtype=np.int32)
    current_values = np.asarray(values, dtype=np.float64)
    if dates.shape != current_values.shape or not bool(np.isfinite(current_values).all()):
        raise ValueError("percentile inputs must be aligned and finite")
    result = np.empty(len(current_values), dtype=np.float64)
    boundaries = _date_boundaries(dates)
    for start, stop in pairwise(boundaries):
        ranks = stats.rankdata(current_values[start:stop], method="average")
        result[start:stop] = (ranks - 0.5) / float(stop - start)
    return result


def tail_training_weights(
    *,
    date_idx: np.ndarray,
    values: np.ndarray,
    maximum_multiplier: float,
    power: float,
) -> np.ndarray:
    dates = np.asarray(date_idx, dtype=np.int32)
    percentile = within_date_percentile(dates, values)
    multiplier = 1.0 + (float(maximum_multiplier) - 1.0) * np.power(
        percentile, float(power)
    )
    weights = base.date_equal_weights(dates).astype(np.float64)
    boundaries = _date_boundaries(dates)
    for start, stop in pairwise(boundaries):
        current = multiplier[start:stop]
        weights[start:stop] *= current / float(current.mean())
    return weights.astype(np.float32)


def daily_relevance_labels(
    *, date_idx: np.ndarray, values: np.ndarray, levels: int
) -> np.ndarray:
    if int(levels) < 2:
        raise ValueError("relevance levels must be at least two")
    percentile = within_date_percentile(date_idx, values)
    labels = np.floor(percentile * int(levels)).astype(np.int32)
    return np.clip(labels, 0, int(levels) - 1)


def date_group_sizes(date_idx: np.ndarray) -> np.ndarray:
    return np.diff(_date_boundaries(date_idx)).astype(np.int32)


@dataclass(frozen=True)
class TrainingTarget:
    rows: np.ndarray
    label: np.ndarray
    weight: np.ndarray
    group: np.ndarray | None
    semantics: dict[str, Any]


def _training_target(
    *,
    study: Mapping[str, Any],
    inputs: base.LearnabilityInputs,
    train_rows: np.ndarray,
    horizon: int,
    objective_variant: str,
) -> TrainingTarget:
    rows = np.asarray(train_rows, dtype=np.int64)
    raw = np.asarray(inputs.label_values("mfe", horizon)[rows], dtype=np.float32)
    valid = np.isfinite(raw)
    rows = rows[valid]
    raw = raw[valid]
    dates = np.asarray(inputs.candidate_date_idx[rows], dtype=np.int32)
    if objective_variant == "tail_weighted_huber":
        config = dict(study["objectives"][objective_variant])
        weight = tail_training_weights(
            date_idx=dates,
            values=raw,
            maximum_multiplier=float(config["maximum_multiplier"]),
            power=float(config["power"]),
        )
        return TrainingTarget(
            rows=rows,
            label=raw,
            weight=weight,
            group=None,
            semantics={
                "label": "raw_mfe",
                "weight": "smooth_within_date_upper_tail_then_date_equal",
                "maximum_multiplier": float(config["maximum_multiplier"]),
                "power": float(config["power"]),
            },
        )
    if objective_variant != "daily_lambdarank":
        raise ValueError(f"unsupported objective target: {objective_variant}")
    config = dict(study["objectives"][objective_variant])
    levels = int(config["relevance_levels"])
    return TrainingTarget(
        rows=rows,
        label=daily_relevance_labels(date_idx=dates, values=raw, levels=levels),
        weight=base.date_equal_weights(dates),
        group=date_group_sizes(dates),
        semantics={
            "label": "within_date_mfe_percentile_relevance",
            "relevance_levels": levels,
            "label_gain": "linear",
            "weight": "date_equal",
        },
    )


@dataclass
class TrainingData:
    train_set: Any
    train_sequence: Any
    evaluation_sequence: Any
    evaluation_rows: np.ndarray
    feature_names: list[str]


def _build_training_data(
    *,
    source_study: Mapping[str, Any],
    inputs: base.LearnabilityInputs,
    target: TrainingTarget,
    evaluation_rows: np.ndarray,
) -> TrainingData:
    import lightgbm as lgb

    source.signal_quality.sequence_training._trim_working_set()
    category_vocabularies = source.signal_quality._fit_category_vocabularies(
        inputs.categorical,
        target.rows,
        inputs.categorical_columns,
    )
    batch_size = int(source_study["model"]["sequence_batch_size"])
    train_sequence = source._combined_sequence(
        inputs=inputs,
        row_ids=target.rows,
        category_vocabularies=category_vocabularies,
        batch_size=batch_size,
        extra=None,
    )
    evaluation_rows = np.asarray(evaluation_rows, dtype=np.int64)
    evaluation_sequence = source._combined_sequence(
        inputs=inputs,
        row_ids=evaluation_rows,
        category_vocabularies=category_vocabularies,
        batch_size=batch_size,
        extra=None,
    )
    train_sequence = base._memory_trimmed_sequence(train_sequence, source_study)
    evaluation_sequence = base._memory_trimmed_sequence(
        evaluation_sequence, source_study
    )
    feature_names = [
        str(item["name"]) for item in inputs.continuous_catalog
    ] + [str(item["name"]) for item in inputs.categorical_catalog]
    categorical_count = len(inputs.categorical_columns)
    categorical_positions = list(
        range(len(feature_names) - categorical_count, len(feature_names))
    )
    construction = {
        "max_bin": int(source_study["model"]["max_bin"]),
        "data_random_seed": int(source_study["model"]["seed"]),
        "feature_pre_filter": False,
        "verbosity": -1,
    }
    train_set = lgb.Dataset(
        train_sequence,
        label=target.label,
        weight=target.weight,
        group=target.group,
        feature_name=feature_names,
        categorical_feature=categorical_positions,
        free_raw_data=True,
        params=construction,
    )
    train_set.construct()
    source.signal_quality.sequence_training._trim_working_set()
    return TrainingData(
        train_set=train_set,
        train_sequence=train_sequence,
        evaluation_sequence=evaluation_sequence,
        evaluation_rows=evaluation_rows,
        feature_names=feature_names,
    )


def _release_training_data(data: TrainingData) -> None:
    data.train_set = None
    data.train_sequence = None
    data.evaluation_sequence = None
    gc.collect()
    source.signal_quality.sequence_training._trim_working_set()


def _task_complete(
    path: Path,
    *,
    task: Mapping[str, Any],
    study: Mapping[str, Any],
    source_study: Mapping[str, Any],
) -> bool:
    if not path.is_file():
        return False
    try:
        import lightgbm as lgb

        result = json.loads(path.read_text(encoding="utf-8"))
        if result.get("schema") != TASK_RESULT_SCHEMA:
            return False
        if result.get("study_id") != STUDY_ID or result.get("status") != "completed":
            return False
        for key in ("task_id", "stage", "objective_variant", "fold_year", "horizon"):
            if result.get(key) != task.get(key):
                return False
        expected = _objective_parameters(
            study=study,
            source_study=source_study,
            objective_variant=str(task["objective_variant"]),
        )
        if dict(result.get("parameters", {})) != expected:
            return False
        baseline = _source_baseline_result(
            study=study,
            source_study=source_study,
            year=int(task["fold_year"]),
            horizon=int(task["horizon"]),
        )
        if int(result["fixed_iteration_source"]["best_iteration"]) != int(
            baseline["fixed_iteration_source"]["best_iteration"]
        ):
            return False
        files = dict(result.get("files", {}))
        for record in files.values():
            current = _resolve(record["path"])
            if not current.is_file() or current.stat().st_size != int(record["size"]):
                return False
        lgb.Booster(model_file=str(_resolve(files["model"]["path"])))
        prediction = np.load(_resolve(files["prediction"]["path"]), allow_pickle=False)
        rows = np.load(_resolve(files["evaluation_rows"]["path"]), allow_pickle=False)
        if tuple(prediction.shape) != tuple(files["prediction"]["shape"]):
            return False
        if tuple(rows.shape) != tuple(files["evaluation_rows"]["shape"]):
            return False
        if prediction.shape != rows.shape:
            return False
        pd.read_parquet(_resolve(files["daily_metrics"]["path"]))
        return True
    except Exception:  # noqa: BLE001 - an unreadable artifact is simply incomplete
        return False


def run_task(
    *,
    task_id: str,
    study_path: Path = DEFAULT_STUDY_PATH,
    output_root: Path = DEFAULT_OUTPUT_ROOT,
) -> dict[str, Any]:
    import lightgbm as lgb

    study = load_study(study_path)
    source_study = _source_study(study)
    task = _task_by_id(task_id)
    result_path = _task_result_path(output_root, task)
    if _task_complete(
        result_path,
        task=task,
        study=study,
        source_study=source_study,
    ):
        return {"task_id": task_id, "status": "completed", "skipped": True}
    try:
        inputs, _pack = source._load_validated_inputs(source_study)
        year = int(task["fold_year"])
        horizon = int(task["horizon"])
        objective_variant = str(task["objective_variant"])
        fold = inputs.common_path_rows(year, horizon)
        baseline = _source_baseline_result(
            study=study,
            source_study=source_study,
            year=year,
            horizon=horizon,
        )
        baseline_rows = np.load(
            _resolve(baseline["files"]["evaluation_rows"]["path"]),
            allow_pickle=False,
        )
        if not np.array_equal(baseline_rows, fold.evaluation_rows):
            raise ValueError("source Huber evaluation rows differ from the configured fold")
        iterations = int(baseline["fixed_iteration_source"]["best_iteration"])
        target = _training_target(
            study=study,
            inputs=inputs,
            train_rows=fold.train_rows,
            horizon=horizon,
            objective_variant=objective_variant,
        )
        data = _build_training_data(
            source_study=source_study,
            inputs=inputs,
            target=target,
            evaluation_rows=fold.evaluation_rows,
        )
        parameters = _objective_parameters(
            study=study,
            source_study=source_study,
            objective_variant=objective_variant,
        )
        started = time.perf_counter()
        model = lgb.train(parameters, data.train_set, num_boost_round=iterations)
        elapsed = float(time.perf_counter() - started)
        output_dir = _task_dir(output_root, task)
        output_dir.mkdir(parents=True, exist_ok=True)
        model_path = output_dir / "model.txt"
        model.save_model(str(model_path), num_iteration=iterations)
        prediction = source._predict(model, data.evaluation_sequence, iterations)
        prediction_path = output_dir / "prediction.npy"
        source._save_npy(prediction_path, prediction)
        evaluation_rows_path = output_dir / "evaluation_rows.npy"
        source._save_npy(evaluation_rows_path, data.evaluation_rows)
        evaluation_raw = np.asarray(
            inputs.label_values("mfe", horizon)[data.evaluation_rows],
            dtype=np.float32,
        )
        daily, metrics = base.regression_metrics(
            date_idx=inputs.candidate_date_idx[data.evaluation_rows],
            actual=evaluation_raw,
            prediction=prediction,
            date_values=inputs.date_values,
            horizon=horizon,
        )
        daily_path = output_dir / "daily_metrics.parquet"
        daily.to_parquet(daily_path, index=False, compression="zstd")
        result = {
            "schema": TASK_RESULT_SCHEMA,
            "status": "completed",
            "completed_at": _now(),
            "study_id": STUDY_ID,
            **task,
            "target": "mfe",
            "purge_days": horizon,
            "train_row_count": len(target.rows),
            "evaluation_row_count": len(data.evaluation_rows),
            "fixed_iteration_source": {
                "study_id": source.STUDY_ID,
                "task_id": baseline["task_id"],
                "inner_validation_year": int(
                    baseline["fixed_iteration_source"]["inner_validation_year"]
                ),
                "best_iteration": iterations,
            },
            "training_seconds": elapsed,
            "training_target": target.semantics,
            "parameters": parameters,
            "feature_count": len(data.feature_names),
            "metrics": metrics,
            "files": {
                "model": _file_record(model_path),
                "prediction": {
                    **_file_record(prediction_path),
                    "shape": list(prediction.shape),
                    "dtype": str(prediction.dtype),
                },
                "evaluation_rows": {
                    **_file_record(evaluation_rows_path),
                    "shape": list(data.evaluation_rows.shape),
                    "dtype": str(data.evaluation_rows.dtype),
                },
                "daily_metrics": _file_record(daily_path),
            },
        }
        _write_json(result_path, result)
        _release_training_data(data)
        del model, prediction, fold, target, baseline_rows
        return {"task_id": task_id, "status": "completed", "skipped": False}
    except Exception as exc:
        failed = {
            "schema": TASK_RESULT_SCHEMA,
            "study_id": STUDY_ID,
            **task,
            "status": "failed",
            "failed_at": _now(),
            "error": f"{type(exc).__name__}: {exc}",
        }
        _write_json(result_path, failed)
        raise


def task_status(
    *,
    study_path: Path = DEFAULT_STUDY_PATH,
    output_root: Path = DEFAULT_OUTPUT_ROOT,
) -> dict[str, Any]:
    study = load_study(study_path)
    source_study = _source_study(study)
    groups: dict[str, list[str]] = {"completed": [], "pending": [], "failed": []}
    for task in _task_plan():
        path = _task_result_path(output_root, task)
        if _task_complete(
            path,
            task=task,
            study=study,
            source_study=source_study,
        ):
            state = "completed"
        elif path.is_file():
            state = "failed"
        else:
            state = "pending"
        groups[state].append(str(task["task_id"]))
    return {
        "study_id": STUDY_ID,
        "total_task_count": len(_task_plan()),
        "completed_count": len(groups["completed"]),
        "pending_count": len(groups["pending"]),
        "failed_count": len(groups["failed"]),
        **groups,
    }


def run_pending(
    *,
    study_path: Path = DEFAULT_STUDY_PATH,
    output_root: Path = DEFAULT_OUTPUT_ROOT,
) -> dict[str, Any]:
    study_path = study_path.resolve()
    output_root = output_root.resolve()
    initial = task_status(study_path=study_path, output_root=output_root)
    runnable = set(initial["pending"]) | set(initial["failed"])
    completed_now: list[str] = []
    for task in _task_plan():
        task_id = str(task["task_id"])
        if task_id not in runnable:
            continue
        print(json.dumps({"event": "task_started", "task_id": task_id}), flush=True)
        command = [
            sys.executable,
            "-m",
            "daily_research.path_policy.seq100_mfe_objective_alignment",
            "--config",
            str(study_path),
            "--output-root",
            str(output_root),
            "--task-id",
            task_id,
            "run-task",
        ]
        child = subprocess.run(command, cwd=WORKSPACE_ROOT, check=False)
        if child.returncode != 0:
            raise RuntimeError(
                f"task {task_id} exited with code {child.returncode}; rerun continues here"
            )
        completed_now.append(task_id)
        print(json.dumps({"event": "task_completed", "task_id": task_id}), flush=True)
    final = task_status(study_path=study_path, output_root=output_root)
    return {**final, "completed_now": completed_now}


def _legal_peak_day_and_mfe(
    close: np.ndarray, sellable: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    values = np.asarray(close, dtype=np.float64)
    legal = np.asarray(sellable, dtype=bool)
    if values.ndim != 2 or values.shape != legal.shape or values.shape[1] < 2:
        raise ValueError("peak inputs must share shape [candidate, horizon>=2]")
    legal_exit = legal[:, 1:] & np.isfinite(values[:, 1:])
    has_peak = legal_exit.any(axis=1)
    masked = np.where(legal_exit, values[:, 1:], -np.inf)
    local = np.argmax(masked, axis=1) + 1
    rows = np.arange(len(values), dtype=np.int64)
    peak_day = np.full(len(values), np.nan, dtype=np.float32)
    mfe = np.full(len(values), np.nan, dtype=np.float32)
    peak_day[has_peak] = (local[has_peak] + 1).astype(np.float32)
    mfe[has_peak] = values[rows[has_peak], local[has_peak]].astype(np.float32)
    return peak_day, mfe


class FuturePathReader:
    def __init__(self, pack: Mapping[str, Any]) -> None:
        path = dict(pack["label_arrays"]["future_ohlcva_path"])
        self.shards = tuple(dict(item) for item in path["shards"])
        date_count, symbol_count = (int(value) for value in pack["masks"]["exit_sellable"]["shape"])
        self.exit_sellable = np.memmap(
            _resolve(pack["masks"]["exit_sellable"]["path"]),
            dtype=bool,
            mode="r",
            shape=(date_count, symbol_count),
        )
        self.cached_key: tuple[int, int] | None = None
        self.cached: np.memmap | None = None

    def _shard(self, date_idx: int) -> tuple[np.memmap, int]:
        for record in self.shards:
            start = int(record["date_start_idx"])
            end = int(record["date_end_idx"])
            if start <= int(date_idx) <= end:
                key = (start, end)
                if self.cached_key != key:
                    self.cached = np.memmap(
                        _resolve(record["path"]),
                        dtype=np.float32,
                        mode="r",
                        shape=tuple(int(value) for value in record["shape"]),
                    )
                    self.cached_key = key
                if self.cached is None:
                    raise AssertionError("future path shard cache is empty")
                return self.cached, int(date_idx) - start
        raise ValueError(f"no future path shard covers date {date_idx}")

    def peak_day(
        self,
        *,
        inputs: base.LearnabilityInputs,
        rows: np.ndarray,
        horizon: int,
        expected_mfe: np.ndarray,
    ) -> np.ndarray:
        candidate_rows = np.asarray(rows, dtype=np.int64)
        dates = np.asarray(inputs.candidate_date_idx[candidate_rows], dtype=np.int32)
        symbols = np.asarray(inputs.candidate_symbol_idx[candidate_rows], dtype=np.int32)
        result = np.full(len(candidate_rows), np.nan, dtype=np.float32)
        reproduced = np.full(len(candidate_rows), np.nan, dtype=np.float32)
        boundaries = _date_boundaries(dates)
        for start, stop in pairwise(boundaries):
            date_idx = int(dates[start])
            current_symbols = symbols[start:stop]
            shard, local_idx = self._shard(date_idx)
            future = np.asarray(
                shard[local_idx, current_symbols, : int(horizon), :4],
                dtype=np.float64,
            )
            anchor = 1.0 + future[:, 0, 0]
            close = np.divide(
                1.0 + future[:, :, 3],
                anchor[:, None],
                out=np.full((stop - start, int(horizon)), np.nan, dtype=np.float64),
                where=np.isfinite(anchor[:, None]) & (anchor[:, None] > 1.0e-8),
            ) - 1.0
            sellable = np.asarray(
                self.exit_sellable[
                    date_idx + 1 : date_idx + 1 + int(horizon), current_symbols
                ],
                dtype=bool,
            ).T
            current_day, current_mfe = _legal_peak_day_and_mfe(close, sellable)
            result[start:stop] = current_day
            reproduced[start:stop] = current_mfe
        expected = np.asarray(expected_mfe, dtype=np.float32)
        comparable = np.isfinite(expected) & np.isfinite(reproduced)
        if bool(comparable.any()) and not np.allclose(
            expected[comparable], reproduced[comparable], rtol=2.0e-5, atol=2.0e-5
        ):
            raise AssertionError("peak-day path reconstruction disagrees with MFE labels")
        return result


def _finite_mean(values: np.ndarray) -> float:
    current = np.asarray(values, dtype=np.float64)
    current = current[np.isfinite(current)]
    return float(current.mean()) if current.size else math.nan


def _positive_share(early: np.ndarray, target: np.ndarray) -> float:
    early_values = np.asarray(early, dtype=np.float64)
    target_values = np.asarray(target, dtype=np.float64)
    numerator = float(
        np.where(np.isfinite(early_values), np.maximum(early_values, 0.0), 0.0).sum()
    )
    denominator = float(
        np.where(
            np.isfinite(target_values), np.maximum(target_values, 0.0), 0.0
        ).sum()
    )
    return numerator / denominator if denominator > 1.0e-12 else math.nan


def path_quality_metrics(
    *,
    date_idx: np.ndarray,
    score: np.ndarray,
    target_mfe: np.ndarray,
    early_mfe: np.ndarray,
    peak_day: np.ndarray,
    pre_peak_mae: np.ndarray,
    endpoint_return: np.ndarray,
    state: np.ndarray,
    horizon: int,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    dates = np.asarray(date_idx, dtype=np.int32)
    predicted = np.asarray(score, dtype=np.float64)
    target = np.asarray(target_mfe, dtype=np.float64)
    early = np.asarray(early_mfe, dtype=np.float64)
    peak = np.asarray(peak_day, dtype=np.float64)
    adverse = np.asarray(pre_peak_mae, dtype=np.float64)
    endpoint = np.asarray(endpoint_return, dtype=np.float64)
    path_state = np.asarray(state)
    if not all(array.shape == dates.shape for array in (predicted, target, early, peak, adverse, endpoint, path_state)):
        raise ValueError("path-quality arrays must be aligned")
    records: list[dict[str, Any]] = []
    boundaries = _date_boundaries(dates)
    for start, stop in pairwise(boundaries):
        valid = np.isfinite(predicted[start:stop]) & np.isfinite(target[start:stop])
        if int(valid.sum()) < 20:
            continue
        current_score = predicted[start:stop][valid]
        current_target = target[start:stop][valid]
        current_early = early[start:stop][valid]
        current_peak = peak[start:stop][valid]
        current_adverse = adverse[start:stop][valid]
        current_endpoint = endpoint[start:stop][valid]
        current_state = path_state[start:stop][valid]
        count = len(current_score)
        order = np.argsort(current_score, kind="mergesort")
        actual_order = np.argsort(current_target, kind="mergesort")
        daily_tail = np.zeros(count, dtype=bool)
        daily_tail[actual_order[-max(1, math.ceil(0.20 * count)) :]] = True
        row: dict[str, Any] = {"date_idx": int(dates[start]), "candidate_count": count}
        for name, fraction in (("top1", 0.01), ("top5", 0.05)):
            selected = order[-max(1, math.ceil(fraction * count)) :]
            row.update(
                {
                    f"{name}_daily_tail_rate": float(daily_tail[selected].mean()),
                    f"{name}_daily_tail_lift": float(
                        daily_tail[selected].mean() - daily_tail.mean()
                    ),
                    f"{name}_early_opportunity_share": _positive_share(
                        current_early[selected], current_target[selected]
                    ),
                    f"{name}_peak_day_mean": _finite_mean(current_peak[selected]),
                    f"{name}_peak_day_fraction": _finite_mean(
                        current_peak[selected] / float(horizon)
                    ),
                    f"{name}_pre_peak_mae_mean": _finite_mean(
                        current_adverse[selected]
                    ),
                    f"{name}_endpoint_return_mean": _finite_mean(
                        current_endpoint[selected]
                    ),
                    f"{name}_high_state_rate": _finite_mean(
                        (current_state[selected] == 2).astype(np.float64)
                    ),
                }
            )
        records.append(row)
    frame = pd.DataFrame(records)
    if frame.empty:
        raise ValueError("path-quality evaluation has no valid dates")
    metrics = {
        column: float(frame[column].mean())
        for column in frame.columns
        if column not in {"date_idx", "candidate_count"}
    }
    for column in ("top1_daily_tail_lift", "top5_daily_tail_lift"):
        metrics[f"{column}_hac"] = base._hac_mean_test(
            frame[column].to_numpy(), max(int(horizon) - 1, 0)
        )
    return frame, metrics


@dataclass(frozen=True)
class PredictionBundle:
    variant: str
    year: int
    horizon: int
    rows: np.ndarray
    prediction: np.ndarray
    result: dict[str, Any]


def _prediction_bundle(
    *,
    study: Mapping[str, Any],
    source_study: Mapping[str, Any],
    output_root: Path,
    variant: str,
    year: int,
    horizon: int,
) -> PredictionBundle:
    if variant == SOURCE_OBJECTIVE:
        result = _source_baseline_result(
            study=study,
            source_study=source_study,
            year=year,
            horizon=horizon,
        )
    else:
        task = _task_by_id(f"mfe{horizon}_{year}_{variant}_outer")
        path = _task_result_path(output_root, task)
        if not _task_complete(
            path,
            task=task,
            study=study,
            source_study=source_study,
        ):
            raise ValueError(f"objective output cannot be loaded: {variant} D{horizon} {year}")
        result = json.loads(path.read_text(encoding="utf-8"))
    files = dict(result["files"])
    rows = np.load(_resolve(files["evaluation_rows"]["path"]), allow_pickle=False)
    prediction = np.load(_resolve(files["prediction"]["path"]), allow_pickle=False)
    if rows.shape != prediction.shape:
        raise ValueError("prediction rows and values differ in shape")
    return PredictionBundle(
        variant=variant,
        year=year,
        horizon=horizon,
        rows=np.asarray(rows, dtype=np.int64),
        prediction=np.asarray(prediction, dtype=np.float32),
        result=result,
    )


def _daily_evaluation(
    *,
    inputs: base.LearnabilityInputs,
    bundle: PredictionBundle,
    peak_day: np.ndarray,
    early_horizon: int,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    rows = bundle.rows
    dates = np.asarray(inputs.candidate_date_idx[rows], dtype=np.int32)
    target = np.asarray(inputs.label_values("mfe", bundle.horizon)[rows], dtype=np.float32)
    early = np.asarray(inputs.label_values("mfe", early_horizon)[rows], dtype=np.float32)
    adverse = np.asarray(
        inputs.label_values("pre_peak_mae", bundle.horizon)[rows], dtype=np.float32
    )
    endpoint = np.asarray(inputs.label_values("g", bundle.horizon)[rows], dtype=np.float32)
    state = np.asarray(inputs.label_values("state", bundle.horizon)[rows], dtype=np.int8)
    ranking_daily, regression = base.regression_metrics(
        date_idx=dates,
        actual=target,
        prediction=bundle.prediction,
        date_values=inputs.date_values,
        horizon=bundle.horizon,
    )
    tail_daily, raw_tail = short._daily_event_enrichment(
        date_idx=dates,
        actual=target,
        score=bundle.prediction,
        absolute_threshold=math.inf,
        horizon=bundle.horizon,
    )
    tail_daily = tail_daily[
        [
            "date_idx",
            "candidate_count",
            "daily_tail_base_rate",
            "daily_tail_top1_rate",
            "daily_tail_top5_rate",
            "daily_tail_top1_lift",
            "daily_tail_top5_lift",
        ]
    ]
    tail = {
        key: value
        for key, value in raw_tail.items()
        if key == "date_count" or key.startswith("daily_tail_")
    }
    path_daily, path = path_quality_metrics(
        date_idx=dates,
        score=bundle.prediction,
        target_mfe=target,
        early_mfe=early,
        peak_day=peak_day,
        pre_peak_mae=adverse,
        endpoint_return=endpoint,
        state=state,
        horizon=bundle.horizon,
    )
    daily = ranking_daily.merge(tail_daily, on=["date_idx", "candidate_count"])
    daily = daily.merge(path_daily, on=["date_idx", "candidate_count"])
    if bundle.variant == "daily_lambdarank":
        magnitude_error: dict[str, Any] = {
            "status": "not_applicable",
            "reason": "LambdaRank output is an ordinal score, not an MFE magnitude estimate",
        }
    else:
        magnitude_error = {
            "status": "reported",
            **{key: value for key, value in regression.items() if key != "ranking"},
        }
    return daily, {
        "broad_ranking": regression["ranking"],
        "magnitude_error": magnitude_error,
        "upper_tail": tail,
        "path_quality": path,
    }


def _paired_delta(
    *, challenger: pd.DataFrame, baseline: pd.DataFrame, horizon: int
) -> dict[str, Any]:
    columns = (
        "rank_ic",
        "top_1pct_lift",
        "top_5pct_lift",
        "daily_tail_top1_lift",
        "daily_tail_top5_lift",
        "top1_early_opportunity_share",
        "top5_early_opportunity_share",
        "top1_peak_day_mean",
        "top5_peak_day_mean",
        "top1_pre_peak_mae_mean",
        "top5_pre_peak_mae_mean",
        "top1_endpoint_return_mean",
        "top5_endpoint_return_mean",
        "top1_high_state_rate",
        "top5_high_state_rate",
    )
    paired = baseline[["date_idx", *columns]].merge(
        challenger[["date_idx", *columns]],
        on="date_idx",
        suffixes=("_baseline", "_challenger"),
        validate="one_to_one",
    )
    metrics: dict[str, Any] = {"date_count": len(paired)}
    for column in columns:
        delta = (
            paired[f"{column}_challenger"] - paired[f"{column}_baseline"]
        ).to_numpy(dtype=np.float64)
        metrics[column] = {
            "mean_delta": float(np.nanmean(delta)),
            "hac": base._hac_mean_test(delta, max(int(horizon) - 1, 0)),
        }
    return metrics


def _metric_leaders(annual: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    metrics = {
        "broad_rank_ic": lambda item: item["metrics"]["broad_ranking"]["rank_ic_mean"],
        "decile_spearman": lambda item: item["metrics"]["broad_ranking"]["decile_spearman"],
        "top1_mfe_lift": lambda item: item["metrics"]["broad_ranking"]["top_1pct_lift"],
        "top5_mfe_lift": lambda item: item["metrics"]["broad_ranking"]["top_5pct_lift"],
        "top5_tail_lift": lambda item: item["metrics"]["upper_tail"]["daily_tail_top5_lift"],
        "top5_early_share": lambda item: item["metrics"]["path_quality"]["top5_early_opportunity_share"],
        "top5_peak_speed": lambda item: -item["metrics"]["path_quality"]["top5_peak_day_mean"],
        "top5_endpoint_return": lambda item: item["metrics"]["path_quality"]["top5_endpoint_return_mean"],
        "top5_pre_peak_mae": lambda item: item["metrics"]["path_quality"]["top5_pre_peak_mae_mean"],
    }
    leaders: dict[str, Any] = {}
    for horizon in TARGET_HORIZONS:
        current = [item for item in annual if int(item["horizon"]) == horizon]
        leaders[str(horizon)] = {}
        for name, getter in metrics.items():
            medians = {
                variant: float(
                    np.nanmedian(
                        [getter(item) for item in current if item["variant"] == variant]
                    )
                )
                for variant in ALL_OBJECTIVES
            }
            leader = max(medians, key=medians.get)
            leaders[str(horizon)][name] = {
                "leader": leader,
                "median_2023_2025": medians,
                "year_winner_count": {
                    variant: sum(
                        item["variant"] == variant
                        and getter(item)
                        == max(
                            getter(other)
                            for other in current
                            if int(other["year"]) == int(item["year"])
                        )
                        for item in current
                    )
                    for variant in ALL_OBJECTIVES
                },
            }
    return leaders


def evaluate_results(
    *,
    study_path: Path = DEFAULT_STUDY_PATH,
    output_root: Path = DEFAULT_OUTPUT_ROOT,
) -> dict[str, Any]:
    study = load_study(study_path)
    source_study = _source_study(study)
    status = task_status(study_path=study_path, output_root=output_root)
    if status["completed_count"] != status["total_task_count"]:
        raise RuntimeError("all objective-alignment tasks must complete before evaluation")
    inputs, pack = source._load_validated_inputs(source_study)
    reader = FuturePathReader(pack)
    annual: list[dict[str, Any]] = []
    paired: list[dict[str, Any]] = []
    evaluation_root = output_root / "evaluation"
    for horizon in TARGET_HORIZONS:
        early_horizon = int(study["evaluation"]["early_horizon_by_target"][str(horizon)])
        for year in FOLD_YEARS:
            bundles = {
                variant: _prediction_bundle(
                    study=study,
                    source_study=source_study,
                    output_root=output_root,
                    variant=variant,
                    year=year,
                    horizon=horizon,
                )
                for variant in ALL_OBJECTIVES
            }
            baseline_rows = bundles[SOURCE_OBJECTIVE].rows
            if any(
                not np.array_equal(bundle.rows, baseline_rows)
                for bundle in bundles.values()
            ):
                raise AssertionError("objective variants do not share evaluation rows")
            expected_mfe = np.asarray(
                inputs.label_values("mfe", horizon)[baseline_rows], dtype=np.float32
            )
            peak_day = reader.peak_day(
                inputs=inputs,
                rows=baseline_rows,
                horizon=horizon,
                expected_mfe=expected_mfe,
            )
            daily_by_variant: dict[str, pd.DataFrame] = {}
            for variant, bundle in bundles.items():
                daily, metrics = _daily_evaluation(
                    inputs=inputs,
                    bundle=bundle,
                    peak_day=peak_day,
                    early_horizon=early_horizon,
                )
                daily_path = (
                    evaluation_root
                    / variant
                    / f"h{horizon:02d}"
                    / f"fold_{year}_daily.parquet"
                )
                daily_path.parent.mkdir(parents=True, exist_ok=True)
                daily.to_parquet(daily_path, index=False, compression="zstd")
                daily_by_variant[variant] = daily
                annual.append(
                    {
                        "year": year,
                        "horizon": horizon,
                        "early_horizon": early_horizon,
                        "variant": variant,
                        "metrics": metrics,
                        "daily_metrics": _file_record(daily_path),
                    }
                )
            baseline_daily = daily_by_variant[SOURCE_OBJECTIVE]
            for variant in TRAINED_OBJECTIVES:
                paired.append(
                    {
                        "year": year,
                        "horizon": horizon,
                        "challenger": variant,
                        "baseline": SOURCE_OBJECTIVE,
                        "metrics": _paired_delta(
                            challenger=daily_by_variant[variant],
                            baseline=baseline_daily,
                            horizon=horizon,
                        ),
                    }
                )
    summary = {
        "schema": SUMMARY_SCHEMA,
        "status": "completed",
        "completed_at": _now(),
        "study_id": STUDY_ID,
        "scope": {
            "candidate_count": inputs.candidate_count,
            "fold_years": list(FOLD_YEARS),
            "horizons": list(TARGET_HORIZONS),
            "objectives": list(ALL_OBJECTIVES),
            "new_booster_count": len(_task_plan()),
            "source_huber_booster_count_reused": len(TARGET_HORIZONS) * len(FOLD_YEARS),
            "iteration_selection": study["folds"]["iteration_source"],
            "fold_role": study["folds"]["fold_role"],
        },
        "annual": annual,
        "paired_against_source_huber": paired,
        "leaders_by_metric": _metric_leaders(annual),
        "interpretation_limit": "role leaders are descriptive; broad ranking and strongest-candidate quality are not collapsed into one gate",
        "non_selections": list(study["non_selections"]),
    }
    _write_json(output_root / "summary.json", summary)
    _write_json(
        output_root / "decision.json",
        {
            "status": "completed_objective_comparison",
            "primary_mfe_objective": SOURCE_OBJECTIVE,
            "tail_weighted_huber_role": "fast_upside_diagnostic_not_default_head",
            "daily_lambdarank_role": "path_quality_evidence_not_strong_mfe_selector",
            "leaders_by_metric": summary["leaders_by_metric"],
            "next_step": "reuse_existing_source_huber_feature_predictions_for_role_synthesis_without_retraining",
            "does_not_select": list(study["non_selections"]),
        },
    )
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Seq100 MFE objective alignment")
    parser.add_argument("--config", type=Path, default=DEFAULT_STUDY_PATH)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--task-id")
    parser.add_argument(
        "command", choices=("status", "run", "run-task", "evaluate")
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "status":
        result = task_status(study_path=args.config, output_root=args.output_root)
    elif args.command == "run":
        result = run_pending(study_path=args.config, output_root=args.output_root)
    elif args.command == "run-task":
        if not args.task_id:
            raise ValueError("--task-id is required for run-task")
        result = run_task(
            task_id=args.task_id,
            study_path=args.config,
            output_root=args.output_root,
        )
    else:
        result = evaluate_results(study_path=args.config, output_root=args.output_root)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
