from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import time
from collections.abc import Mapping, Sequence
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from daily_research.path_policy import seq100_entry_role_synthesis as entry
from daily_research.path_policy import seq100_mfe_feature_family_audit as source
from daily_research.path_policy import seq100_mfe_feature_union_audit as union
from daily_research.path_policy import seq100_mfe_objective_alignment as objective
from daily_research.path_policy import seq100_path_label_learnability as base

WORKSPACE_ROOT = source.WORKSPACE_ROOT
STUDY_ID = "seq100_mfe_capacity_stability_audit_v1"
DEFAULT_STUDY_PATH = (
    WORKSPACE_ROOT
    / "daily_research/studies/seq100_mfe_capacity_stability_audit_v1.json"
)
DEFAULT_OUTPUT_ROOT = (
    WORKSPACE_ROOT
    / "daily_research/output/path_policy/studies"
    / "seq100_mfe_capacity_stability_audit_v1"
)
DEFAULT_RECORD_ROOT = (
    WORKSPACE_ROOT
    / "daily_research/research_records/seq100"
    / "seq100_mfe_capacity_stability_audit_v1"
)
CURVE_TASK_SCHEMA = "seq100_mfe_capacity_curve_task/v1"
OUTER_TASK_SCHEMA = "seq100_mfe_fixed_capacity_outer_task/v1"
SELECTION_SCHEMA = "seq100_mfe_capacity_selection/v1"
SUMMARY_SCHEMA = "seq100_mfe_capacity_stability_summary/v1"
SELECTION_YEARS = (2019, 2020, 2021, 2022)
DECISION_YEARS = (2023, 2024, 2025)
HORIZONS = (10, 20)


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
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _file_record(path: Path) -> dict[str, Any]:
    return {
        "path": path.relative_to(WORKSPACE_ROOT).as_posix(),
        "size": int(path.stat().st_size),
        "sha256": _sha256(path),
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
        raise ValueError("capacity-stability study id changed")
    folds = dict(payload.get("folds", {}) or {})
    if (
        tuple(int(value) for value in folds.get("capacity_selection_years", []))
        != SELECTION_YEARS
    ):
        raise ValueError("capacity-selection years changed")
    if tuple(int(value) for value in folds.get("decision_years", [])) != DECISION_YEARS:
        raise ValueError("capacity decision years changed")
    if folds.get("maximum_outcome_date") != "2025-12-31":
        raise ValueError("capacity outcome boundary changed")
    if int(folds.get("forbidden_outcome_year", -1)) != 2026:
        raise ValueError("2026 must remain forbidden")
    heads = dict(payload.get("heads", {}) or {})
    if tuple(sorted(int(value) for value in heads)) != HORIZONS:
        raise ValueError("capacity horizons changed")
    expected = {
        10: ("turnover_cost_proxy", ("turnover_cost_proxy",)),
        20: ("breakout_retest_levels", ("breakout_retest_levels",)),
    }
    catalog = source.feature_catalog()
    for horizon, (name, families) in expected.items():
        head = dict(heads[str(horizon)])
        current_families = tuple(str(value) for value in head.get("families", []))
        if str(head.get("name")) != name or current_families != families:
            raise ValueError(f"frozen D{horizon} head changed")
        if any(family not in catalog for family in current_families):
            raise ValueError(f"D{horizon} contains an unknown feature family")
    capacity = dict(payload.get("capacity_selection", {}) or {})
    grid = tuple(int(value) for value in capacity.get("iteration_grid", []))
    maximum = int(capacity.get("maximum_iteration", -1))
    if (
        not grid
        or tuple(sorted(set(grid))) != grid
        or grid[0] <= 0
        or grid[-1] != maximum
        or maximum > 512
    ):
        raise ValueError("capacity grid is invalid or no longer bounded")
    if bool(capacity.get("uses_decision_year_for_selection", True)):
        raise ValueError("decision years cannot select capacity")
    model = dict(payload.get("model", {}) or {})
    if not bool(model.get("selection_models_have_no_early_stopping", False)):
        raise ValueError("capacity curves must not use early stopping")
    return payload


def _config_hash(path: Path) -> str:
    return _sha256(path)


def _head(study: Mapping[str, Any], horizon: int) -> dict[str, Any]:
    value = dict(study["heads"][str(int(horizon))])
    return {
        "horizon": int(horizon),
        "name": str(value["name"]),
        "families": tuple(str(item) for item in value["families"]),
    }


def _selection_tasks(study: Mapping[str, Any]) -> list[dict[str, Any]]:
    tasks: list[dict[str, Any]] = []
    for horizon in HORIZONS:
        head = _head(study, horizon)
        for year in SELECTION_YEARS:
            tasks.append(
                {
                    **head,
                    "year": int(year),
                    "stage": "capacity_curve",
                    "task_id": f"mfe{horizon}_{year}_capacity_curve",
                }
            )
    return tasks


def _outer_tasks(study: Mapping[str, Any]) -> list[dict[str, Any]]:
    tasks: list[dict[str, Any]] = []
    for horizon in HORIZONS:
        head = _head(study, horizon)
        for year in DECISION_YEARS:
            tasks.append(
                {
                    **head,
                    "year": int(year),
                    "stage": "fixed_capacity_outer",
                    "task_id": f"mfe{horizon}_{year}_fixed_capacity_outer",
                }
            )
    return tasks


def _task_dir(output_root: Path, task: Mapping[str, Any]) -> Path:
    horizon = int(task["horizon"])
    year = int(task["year"])
    if task["stage"] == "capacity_curve":
        return (
            output_root
            / "capacity_curves"
            / f"fold_{year}"
            / f"h{horizon:02d}"
            / str(task["name"])
        )
    return (
        output_root / "outer" / f"fold_{year}" / f"h{horizon:02d}" / str(task["name"])
    )


def _task_result_path(output_root: Path, task: Mapping[str, Any]) -> Path:
    return _task_dir(output_root, task) / "task_result.json"


def _record_exists(record: Mapping[str, Any]) -> bool:
    path = _resolve(record["path"])
    return path.is_file() and path.stat().st_size == int(record["size"])


def _curve_task_complete(
    path: Path,
    *,
    task: Mapping[str, Any],
    study: Mapping[str, Any],
    study_path: Path,
    feature_study: Mapping[str, Any],
) -> bool:
    if not path.is_file():
        return False
    try:
        import lightgbm as lgb

        result = json.loads(path.read_text(encoding="utf-8"))
        maximum = int(study["capacity_selection"]["maximum_iteration"])
        expected = {
            "schema": CURVE_TASK_SCHEMA,
            "status": "completed",
            "study_id": STUDY_ID,
            "task_id": task["task_id"],
            "stage": task["stage"],
            "head": task["name"],
            "fold_year": int(task["year"]),
            "horizon": int(task["horizon"]),
            "families": list(task["families"]),
            "trained_iterations": maximum,
            "config_sha256": _config_hash(study_path),
        }
        if any(result.get(key) != value for key, value in expected.items()):
            return False
        parameters, _rounds, _patience = source._model_parameters(feature_study)
        if dict(result.get("parameters", {}) or {}) != parameters:
            return False
        files = dict(result.get("files", {}) or {})
        if not files or not all(_record_exists(record) for record in files.values()):
            return False
        model = lgb.Booster(model_file=str(_resolve(files["model"]["path"])))
        if int(model.num_trees()) != maximum:
            return False
        curve = pd.read_parquet(_resolve(files["loss_curve"]["path"]))
        if (
            list(curve.columns) != ["iteration", "huber"]
            or len(curve) != maximum
            or not np.array_equal(
                curve["iteration"].to_numpy(), np.arange(1, maximum + 1)
            )
        ):
            return False
        rows = np.load(_resolve(files["evaluation_rows"]["path"]), allow_pickle=False)
        return tuple(rows.shape) == tuple(files["evaluation_rows"]["shape"])
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


def _outer_task_complete(
    path: Path,
    *,
    task: Mapping[str, Any],
    study: Mapping[str, Any],
    study_path: Path,
    feature_study: Mapping[str, Any],
    selected_iteration: int,
) -> bool:
    if not path.is_file():
        return False
    try:
        import lightgbm as lgb

        result = json.loads(path.read_text(encoding="utf-8"))
        expected = {
            "schema": OUTER_TASK_SCHEMA,
            "status": "completed",
            "study_id": STUDY_ID,
            "task_id": task["task_id"],
            "stage": task["stage"],
            "head": task["name"],
            "fold_year": int(task["year"]),
            "horizon": int(task["horizon"]),
            "families": list(task["families"]),
            "fixed_iteration": int(selected_iteration),
            "config_sha256": _config_hash(study_path),
        }
        if any(result.get(key) != value for key, value in expected.items()):
            return False
        parameters, _rounds, _patience = source._model_parameters(feature_study)
        if dict(result.get("parameters", {}) or {}) != parameters:
            return False
        files = dict(result.get("files", {}) or {})
        if not files or not all(_record_exists(record) for record in files.values()):
            return False
        model = lgb.Booster(model_file=str(_resolve(files["model"]["path"])))
        if int(model.num_trees()) != int(selected_iteration):
            return False
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


def _run_curve_task(
    *,
    study: Mapping[str, Any],
    study_path: Path,
    feature_study: Mapping[str, Any],
    feature_output_root: Path,
    output_root: Path,
    inputs: base.LearnabilityInputs,
    task: Mapping[str, Any],
) -> dict[str, Any]:
    import lightgbm as lgb

    result_path = _task_result_path(output_root, task)
    if _curve_task_complete(
        result_path,
        task=task,
        study=study,
        study_path=study_path,
        feature_study=feature_study,
    ):
        return json.loads(result_path.read_text(encoding="utf-8"))
    horizon = int(task["horizon"])
    year = int(task["year"])
    maximum = int(study["capacity_selection"]["maximum_iteration"])
    extra, manifests = union._open_union(
        feature_output_root=feature_output_root,
        families=task["families"],
        candidate_count=inputs.candidate_count,
    )
    fold = inputs.common_path_rows(year, horizon)
    datasets = None
    model = None
    try:
        datasets, _evaluation_weight, _evaluation_raw = source._build_datasets(
            study=feature_study,
            inputs=inputs,
            train_rows=fold.train_rows,
            evaluation_rows=fold.evaluation_rows,
            horizon=horizon,
            extra=extra,
            extra_names=extra.feature_names,
        )
        parameters, _rounds, _patience = source._model_parameters(feature_study)
        evaluations: dict[str, dict[str, list[float]]] = {}
        _emit(
            "capacity_curve_training_started",
            task_id=task["task_id"],
            iterations=maximum,
            train_rows=len(fold.train_rows),
            evaluation_rows=len(fold.evaluation_rows),
            feature_count=len(datasets.feature_names),
        )
        started = time.perf_counter()
        model = lgb.train(
            parameters,
            datasets.train_set,
            num_boost_round=maximum,
            valid_sets=[datasets.evaluation_set],
            valid_names=["capacity_validation"],
            callbacks=[
                lgb.record_evaluation(evaluations),
                lgb.log_evaluation(period=0),
            ],
        )
        elapsed = float(time.perf_counter() - started)
        values = evaluations.get("capacity_validation", {}).get("huber", [])
        if len(values) != maximum:
            raise RuntimeError(
                f"{task['task_id']} recorded {len(values)} of {maximum} losses"
            )
        curve = pd.DataFrame(
            {
                "iteration": np.arange(1, maximum + 1, dtype=np.int32),
                "huber": np.asarray(values, dtype=np.float64),
            }
        )
        grid = [int(value) for value in study["capacity_selection"]["iteration_grid"]]
        grid_losses = {
            str(iteration): float(curve.iloc[iteration - 1]["huber"])
            for iteration in grid
        }
        output_dir = _task_dir(output_root, task)
        output_dir.mkdir(parents=True, exist_ok=True)
        model_path = output_dir / "model.txt"
        curve_path = output_dir / "loss_curve.parquet"
        rows_path = output_dir / "evaluation_rows.npy"
        model.save_model(str(model_path), num_iteration=maximum)
        curve.to_parquet(curve_path, index=False, compression="zstd")
        _save_npy(rows_path, datasets.evaluation_rows)
        minimum_index = int(np.argmin(curve["huber"].to_numpy()))
        result = {
            "schema": CURVE_TASK_SCHEMA,
            "status": "completed",
            "completed_at": _now(),
            "study_id": STUDY_ID,
            "task_id": task["task_id"],
            "stage": task["stage"],
            "head": task["name"],
            "families": list(task["families"]),
            "fold_year": year,
            "horizon": horizon,
            "target": "mfe",
            "purge_days": horizon,
            "train_row_count": len(fold.train_rows),
            "evaluation_row_count": len(fold.evaluation_rows),
            "trained_iterations": maximum,
            "early_stopping_used": False,
            "minimum_full_curve_iteration": minimum_index + 1,
            "minimum_full_curve_huber": float(curve.iloc[minimum_index]["huber"]),
            "grid_losses": grid_losses,
            "training_seconds": elapsed,
            "parameters": parameters,
            "feature_count": len(datasets.feature_names),
            "added_feature_count": len(extra.feature_names),
            "family_manifests": manifests,
            "config_sha256": _config_hash(study_path),
            "files": {
                "model": _file_record(model_path),
                "loss_curve": _file_record(curve_path),
                "evaluation_rows": {
                    **_file_record(rows_path),
                    "shape": list(datasets.evaluation_rows.shape),
                    "dtype": str(datasets.evaluation_rows.dtype),
                },
            },
        }
        _write_json(result_path, result)
        _emit(
            "capacity_curve_training_completed",
            task_id=task["task_id"],
            seconds=elapsed,
            full_curve_minimum=minimum_index + 1,
        )
        return result
    finally:
        if datasets is not None:
            source._release_datasets(datasets)
        del model, fold, extra


def _select_from_losses(
    losses_by_year: Mapping[int, Mapping[int, float]],
    grid: Sequence[int],
) -> dict[str, Any]:
    years = tuple(sorted(int(value) for value in losses_by_year))
    counts = tuple(int(value) for value in grid)
    if not years or not counts:
        raise ValueError("capacity selection requires years and counts")
    if tuple(sorted(set(counts))) != counts:
        raise ValueError("capacity grid must be unique and increasing")
    normalized: dict[int, dict[int, float]] = {}
    for year in years:
        losses = {int(key): float(value) for key, value in losses_by_year[year].items()}
        if tuple(sorted(losses)) != counts:
            raise ValueError(f"capacity losses are incomplete for {year}")
        values = np.asarray([losses[count] for count in counts], dtype=np.float64)
        if not bool(np.isfinite(values).all()) or bool(np.any(values <= 0.0)):
            raise ValueError(f"capacity losses are invalid for {year}")
        minimum = float(values.min())
        normalized[year] = {
            count: float(losses[count] / minimum - 1.0) for count in counts
        }
    rows: list[dict[str, Any]] = []
    for count in counts:
        regrets = np.asarray(
            [normalized[year][count] for year in years], dtype=np.float64
        )
        standard_deviation = float(np.std(regrets, ddof=1)) if len(regrets) > 1 else 0.0
        rows.append(
            {
                "iteration": count,
                "mean_normalized_regret": float(np.mean(regrets)),
                "standard_deviation": standard_deviation,
                "standard_error": standard_deviation / math.sqrt(len(regrets)),
                **{
                    f"normalized_regret_{year}": normalized[year][count]
                    for year in years
                },
                **{
                    f"huber_{year}": float(losses_by_year[year][count])
                    for year in years
                },
            }
        )
    minimum_row = min(
        rows, key=lambda row: (float(row["mean_normalized_regret"]), row["iteration"])
    )
    threshold = float(minimum_row["mean_normalized_regret"]) + float(
        minimum_row["standard_error"]
    )
    eligible = [
        row
        for row in rows
        if float(row["mean_normalized_regret"]) <= threshold + 1.0e-15
    ]
    selected = min(int(row["iteration"]) for row in eligible)
    return {
        "selection_years": list(years),
        "iteration_grid": list(counts),
        "aggregate_minimizer": int(minimum_row["iteration"]),
        "minimum_mean_normalized_regret": float(minimum_row["mean_normalized_regret"]),
        "standard_error_at_minimizer": float(minimum_row["standard_error"]),
        "one_standard_error_threshold": threshold,
        "selected_iteration": selected,
        "grid": rows,
    }


def select_capacities(
    *,
    study: Mapping[str, Any],
    study_path: Path,
    feature_study: Mapping[str, Any],
    output_root: Path,
) -> dict[str, Any]:
    selections: dict[str, Any] = {}
    grid = tuple(int(value) for value in study["capacity_selection"]["iteration_grid"])
    for horizon in HORIZONS:
        tasks = [
            task for task in _selection_tasks(study) if int(task["horizon"]) == horizon
        ]
        losses: dict[int, dict[int, float]] = {}
        sources: list[dict[str, Any]] = []
        for task in tasks:
            path = _task_result_path(output_root, task)
            if not _curve_task_complete(
                path,
                task=task,
                study=study,
                study_path=study_path,
                feature_study=feature_study,
            ):
                raise RuntimeError(f"capacity curve is incomplete: {task['task_id']}")
            result = json.loads(path.read_text(encoding="utf-8"))
            losses[int(task["year"])] = {
                int(key): float(value)
                for key, value in dict(result["grid_losses"]).items()
            }
            sources.append(
                {
                    "task_id": task["task_id"],
                    "fold_year": int(task["year"]),
                    "result": _file_record(path),
                    "full_curve_minimum": int(result["minimum_full_curve_iteration"]),
                }
            )
        selected = _select_from_losses(losses, grid)
        grid_frame = pd.DataFrame(selected.pop("grid"))
        grid_path = (
            output_root / "capacity_selection" / f"h{horizon:02d}_grid_summary.parquet"
        )
        grid_path.parent.mkdir(parents=True, exist_ok=True)
        grid_frame.to_parquet(grid_path, index=False, compression="zstd")
        selections[str(horizon)] = {
            **_head(study, horizon),
            **selected,
            "curve_sources": sources,
            "grid_summary": _file_record(grid_path),
        }
    payload = {
        "schema": SELECTION_SCHEMA,
        "status": "completed",
        "completed_at": _now(),
        "study_id": STUDY_ID,
        "config_sha256": _config_hash(study_path),
        "uses_decision_year_for_selection": False,
        "selection_years": list(SELECTION_YEARS),
        "decision_years_withheld_from_selection": list(DECISION_YEARS),
        "heads": selections,
    }
    _write_json(output_root / "capacity_selection" / "selection.json", payload)
    _emit(
        "capacity_selection_completed",
        selected_iterations={
            horizon: int(record["selected_iteration"])
            for horizon, record in selections.items()
        },
    )
    return payload


def _load_selection(
    *,
    output_root: Path,
    study_path: Path,
) -> dict[str, Any]:
    path = output_root / "capacity_selection" / "selection.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    if (
        payload.get("schema") != SELECTION_SCHEMA
        or payload.get("status") != "completed"
        or payload.get("study_id") != STUDY_ID
        or payload.get("config_sha256") != _config_hash(study_path)
        or payload.get("selection_years") != list(SELECTION_YEARS)
        or payload.get("decision_years_withheld_from_selection") != list(DECISION_YEARS)
        or bool(payload.get("uses_decision_year_for_selection", True))
    ):
        raise ValueError("capacity selection is stale or semantically invalid")
    return payload


def _run_outer_task(
    *,
    study: Mapping[str, Any],
    study_path: Path,
    feature_study: Mapping[str, Any],
    feature_output_root: Path,
    output_root: Path,
    inputs: base.LearnabilityInputs,
    selection: Mapping[str, Any],
    task: Mapping[str, Any],
) -> dict[str, Any]:
    import lightgbm as lgb

    horizon = int(task["horizon"])
    year = int(task["year"])
    iterations = int(selection["heads"][str(horizon)]["selected_iteration"])
    result_path = _task_result_path(output_root, task)
    if _outer_task_complete(
        result_path,
        task=task,
        study=study,
        study_path=study_path,
        feature_study=feature_study,
        selected_iteration=iterations,
    ):
        return json.loads(result_path.read_text(encoding="utf-8"))
    extra, manifests = union._open_union(
        feature_output_root=feature_output_root,
        families=task["families"],
        candidate_count=inputs.candidate_count,
    )
    fold = inputs.common_path_rows(year, horizon)
    datasets = None
    model = None
    try:
        datasets, _evaluation_weight, evaluation_raw = source._build_datasets(
            study=feature_study,
            inputs=inputs,
            train_rows=fold.train_rows,
            evaluation_rows=fold.evaluation_rows,
            horizon=horizon,
            extra=extra,
            extra_names=extra.feature_names,
        )
        parameters, _rounds, _patience = source._model_parameters(feature_study)
        _emit(
            "fixed_capacity_training_started",
            task_id=task["task_id"],
            iterations=iterations,
            train_rows=len(fold.train_rows),
            evaluation_rows=len(fold.evaluation_rows),
            feature_count=len(datasets.feature_names),
        )
        started = time.perf_counter()
        model = lgb.train(
            parameters,
            datasets.train_set,
            num_boost_round=iterations,
            valid_sets=[datasets.evaluation_set],
            valid_names=["outer_evaluation"],
            callbacks=[lgb.log_evaluation(period=0)],
        )
        elapsed = float(time.perf_counter() - started)
        prediction = source._predict(model, datasets.evaluation_sequence, iterations)
        daily, metrics = base.regression_metrics(
            date_idx=inputs.candidate_date_idx[datasets.evaluation_rows],
            actual=evaluation_raw,
            prediction=prediction,
            date_values=inputs.date_values,
            horizon=horizon,
        )
        output_dir = _task_dir(output_root, task)
        output_dir.mkdir(parents=True, exist_ok=True)
        model_path = output_dir / "model.txt"
        prediction_path = output_dir / "prediction.npy"
        rows_path = output_dir / "evaluation_rows.npy"
        daily_path = output_dir / "daily_metrics.parquet"
        model.save_model(str(model_path), num_iteration=iterations)
        _save_npy(prediction_path, prediction)
        _save_npy(rows_path, datasets.evaluation_rows)
        daily.to_parquet(daily_path, index=False, compression="zstd")
        result = {
            "schema": OUTER_TASK_SCHEMA,
            "status": "completed",
            "completed_at": _now(),
            "study_id": STUDY_ID,
            "task_id": task["task_id"],
            "stage": task["stage"],
            "head": task["name"],
            "families": list(task["families"]),
            "fold_year": year,
            "horizon": horizon,
            "target": "mfe",
            "purge_days": horizon,
            "train_row_count": len(fold.train_rows),
            "evaluation_row_count": len(fold.evaluation_rows),
            "fixed_iteration": iterations,
            "capacity_selection": {
                "selection_years": list(SELECTION_YEARS),
                "aggregate_minimizer": int(
                    selection["heads"][str(horizon)]["aggregate_minimizer"]
                ),
                "selected_iteration": iterations,
                "uses_decision_year_for_selection": False,
            },
            "training_seconds": elapsed,
            "parameters": parameters,
            "feature_count": len(datasets.feature_names),
            "added_feature_count": len(extra.feature_names),
            "family_manifests": manifests,
            "metrics": metrics,
            "config_sha256": _config_hash(study_path),
            "files": {
                "model": _file_record(model_path),
                "prediction": {
                    **_file_record(prediction_path),
                    "shape": list(prediction.shape),
                    "dtype": str(prediction.dtype),
                },
                "evaluation_rows": {
                    **_file_record(rows_path),
                    "shape": list(datasets.evaluation_rows.shape),
                    "dtype": str(datasets.evaluation_rows.dtype),
                },
                "daily_metrics": _file_record(daily_path),
            },
        }
        _write_json(result_path, result)
        _emit(
            "fixed_capacity_training_completed",
            task_id=task["task_id"],
            seconds=elapsed,
        )
        return result
    finally:
        if datasets is not None:
            source._release_datasets(datasets)
        del model, fold, extra


def _load_outer_result(
    output_root: Path,
    *,
    task: Mapping[str, Any],
) -> tuple[dict[str, Any], np.ndarray, np.ndarray]:
    result = json.loads(
        _task_result_path(output_root, task).read_text(encoding="utf-8")
    )
    files = dict(result["files"])
    prediction = np.load(_resolve(files["prediction"]["path"]), allow_pickle=False)
    rows = np.load(_resolve(files["evaluation_rows"]["path"]), allow_pickle=False)
    return result, prediction, rows


def _decision_gate(
    *,
    relative_mae_harm: Sequence[float],
    rank_ic_delta: Sequence[float],
    top5_mfe_delta: Sequence[float],
    path_guardrail: Mapping[str, Any],
    rules: Mapping[str, Any],
) -> dict[str, Any]:
    mae = np.asarray(relative_mae_harm, dtype=np.float64)
    rank_ic = np.asarray(rank_ic_delta, dtype=np.float64)
    top5 = np.asarray(top5_mfe_delta, dtype=np.float64)
    if mae.size != 3 or rank_ic.size != 3 or top5.size != 3:
        raise ValueError("capacity decision requires exactly three decision years")
    if not bool(np.isfinite(np.concatenate([mae, rank_ic, top5])).all()):
        raise ValueError("capacity decision inputs must be finite")
    checks = {
        "median_relative_mae_harm": float(np.median(mae)),
        "worst_relative_mae_harm": float(np.max(mae)),
        "worst_rank_ic_delta": float(np.min(rank_ic)),
        "median_top5_mfe_delta": float(np.median(top5)),
        "nonmaterial_top5_years": int(
            np.sum(top5 >= -float(rules["maximum_nonmaterial_top5_year_harm"]))
        ),
        "path_guardrail_rejected": bool(path_guardrail["rejected"]),
    }
    passed = (
        checks["median_relative_mae_harm"]
        <= float(rules["maximum_median_relative_mae_harm"])
        and checks["worst_relative_mae_harm"]
        <= float(rules["maximum_worst_relative_mae_harm"])
        and checks["worst_rank_ic_delta"] >= float(rules["minimum_worst_rank_ic_delta"])
        and checks["median_top5_mfe_delta"]
        >= float(rules["minimum_median_top5_mfe_delta"])
        and checks["nonmaterial_top5_years"]
        >= int(rules["minimum_nonmaterial_top5_years"])
        and not checks["path_guardrail_rejected"]
    )
    return {
        "passed": bool(passed),
        "checks": checks,
        "relative_mae_harm_2023_2024_2025": mae.tolist(),
        "rank_ic_delta_2023_2024_2025": rank_ic.tolist(),
        "top5_mfe_delta_2023_2024_2025": top5.tolist(),
    }


def _contract_rebuild_required(
    *,
    adopted: bool,
    prediction_spearman: Sequence[float],
    top5_jaccard: Sequence[float],
    rules: Mapping[str, Any],
) -> dict[str, Any]:
    spearman = np.asarray(prediction_spearman, dtype=np.float64)
    jaccard = np.asarray(top5_jaccard, dtype=np.float64)
    if spearman.size != 3 or jaccard.size != 3:
        raise ValueError("contract comparison requires exactly three decision years")
    materially_changed_years = [
        int(year)
        for year, correlation, overlap in zip(
            DECISION_YEARS, spearman, jaccard, strict=True
        )
        if (
            not math.isfinite(float(correlation))
            or correlation
            < float(rules["contract_rebuild_minimum_prediction_spearman"])
            or not math.isfinite(float(overlap))
            or overlap < float(rules["contract_rebuild_minimum_top5_jaccard"])
        )
    ]
    return {
        "required": bool(adopted and materially_changed_years),
        "materially_changed_years": materially_changed_years,
        "prediction_spearman_2023_2024_2025": spearman.tolist(),
        "top5_jaccard_2023_2024_2025": jaccard.tolist(),
    }


def evaluate(
    *,
    study_path: Path = DEFAULT_STUDY_PATH,
    output_root: Path = DEFAULT_OUTPUT_ROOT,
) -> dict[str, Any]:
    study = load_study(study_path)
    feature_study = source.load_study(
        _resolve(study["sources"]["feature_study_config"])
    )
    feature_output_root = _resolve(study["sources"]["feature_output_root"])
    selection = _load_selection(output_root=output_root, study_path=study_path)
    inputs, pack = source._load_validated_inputs(feature_study)
    if inputs.maximum_outcome_date != "2025-12-31":
        raise ValueError("capacity evaluation boundary changed")
    reader = objective.FuturePathReader(pack)
    annual: list[dict[str, Any]] = []
    paired: list[dict[str, Any]] = []
    relationships: list[dict[str, Any]] = []
    rules = dict(study["decision"])
    for task in _outer_tasks(study):
        horizon = int(task["horizon"])
        year = int(task["year"])
        iterations = int(selection["heads"][str(horizon)]["selected_iteration"])
        result_path = _task_result_path(output_root, task)
        if not _outer_task_complete(
            result_path,
            task=task,
            study=study,
            study_path=study_path,
            feature_study=feature_study,
            selected_iteration=iterations,
        ):
            raise RuntimeError(
                f"fixed-capacity output is incomplete: {task['task_id']}"
            )
        fixed_result, fixed_prediction, rows = _load_outer_result(
            output_root, task=task
        )
        adaptive_result, adaptive_prediction, adaptive_rows = source._load_outer_result(
            feature_output_root,
            year=year,
            horizon=horizon,
            variant=str(task["name"]),
            study=feature_study,
        )
        if not np.array_equal(rows, adaptive_rows):
            raise ValueError(f"row alignment changed for {task['task_id']}")
        actual = entry._path_actual(
            inputs=inputs,
            reader=reader,
            rows=rows,
            horizon=horizon,
            early_horizon=5 if horizon == 10 else 10,
        )
        adaptive_bundle = objective.PredictionBundle(
            variant=f"{task['name']}_adaptive",
            year=year,
            horizon=horizon,
            rows=rows,
            prediction=adaptive_prediction,
            result=adaptive_result,
        )
        fixed_bundle = objective.PredictionBundle(
            variant=f"{task['name']}_fixed",
            year=year,
            horizon=horizon,
            rows=rows,
            prediction=fixed_prediction,
            result=fixed_result,
        )
        adaptive_daily, adaptive_metrics = objective._daily_evaluation(
            inputs=inputs,
            bundle=adaptive_bundle,
            peak_day=actual.peak_day,
            early_horizon=5 if horizon == 10 else 10,
        )
        fixed_daily, fixed_metrics = objective._daily_evaluation(
            inputs=inputs,
            bundle=fixed_bundle,
            peak_day=actual.peak_day,
            early_horizon=5 if horizon == 10 else 10,
        )
        pair_metrics = objective._paired_delta(
            challenger=fixed_daily,
            baseline=adaptive_daily,
            horizon=horizon,
        )
        pair_frame = adaptive_daily.merge(
            fixed_daily,
            on="date_idx",
            suffixes=("_adaptive", "_fixed"),
            validate="one_to_one",
        )
        pair_path = (
            output_root
            / "evaluation"
            / f"h{horizon:02d}"
            / f"fold_{year}_paired_daily.parquet"
        )
        pair_path.parent.mkdir(parents=True, exist_ok=True)
        pair_frame.to_parquet(pair_path, index=False, compression="zstd")
        relation_frame = entry._daily_model_relationship(
            actual=actual,
            left_name="adaptive_capacity",
            left_score=adaptive_prediction,
            right_name="fixed_capacity",
            right_score=fixed_prediction,
        )
        relation_path = (
            output_root
            / "relationships"
            / f"h{horizon:02d}"
            / f"fold_{year}_fixed_vs_adaptive.parquet"
        )
        relation_path.parent.mkdir(parents=True, exist_ok=True)
        relation_frame.to_parquet(relation_path, index=False, compression="zstd")
        relationship_summary = entry._summarize_daily_frame(
            relation_frame, horizon=horizon
        )
        adaptive_mae = float(adaptive_metrics["magnitude_error"]["date_equal_mae"])
        fixed_mae = float(fixed_metrics["magnitude_error"]["date_equal_mae"])
        adaptive_count = (
            int(adaptive_result["best_iteration"])
            if "best_iteration" in adaptive_result
            else int(adaptive_result["fixed_iteration_source"]["best_iteration"])
        )
        annual.append(
            {
                "year": year,
                "horizon": horizon,
                "head": task["name"],
                "adaptive_iteration": adaptive_count,
                "fixed_iteration": iterations,
                "adaptive_metrics": adaptive_metrics,
                "fixed_metrics": fixed_metrics,
                "relative_mae_harm": fixed_mae / adaptive_mae - 1.0,
            }
        )
        paired.append(
            {
                "year": year,
                "horizon": horizon,
                "head": task["name"],
                "metrics": pair_metrics,
                "paired_daily": _file_record(pair_path),
            }
        )
        relationships.append(
            {
                "year": year,
                "horizon": horizon,
                "head": task["name"],
                "summary": relationship_summary,
                "daily": _file_record(relation_path),
            }
        )
    evidence: dict[str, Any] = {}
    final_heads: dict[str, Any] = {}
    any_rebuild = False
    for horizon in HORIZONS:
        current_annual = [row for row in annual if int(row["horizon"]) == horizon]
        current_paired = [row for row in paired if int(row["horizon"]) == horizon]
        current_relationships = [
            row for row in relationships if int(row["horizon"]) == horizon
        ]
        current_annual.sort(key=lambda row: int(row["year"]))
        current_paired.sort(key=lambda row: int(row["year"]))
        current_relationships.sort(key=lambda row: int(row["year"]))
        path_guardrail = union._path_guardrail(current_paired, rules)
        gate = _decision_gate(
            relative_mae_harm=[
                float(row["relative_mae_harm"]) for row in current_annual
            ],
            rank_ic_delta=[
                float(row["metrics"]["rank_ic"]["mean_delta"]) for row in current_paired
            ],
            top5_mfe_delta=[
                float(row["metrics"]["top_5pct_lift"]["mean_delta"])
                for row in current_paired
            ],
            path_guardrail=path_guardrail,
            rules=rules,
        )
        prediction_spearman = [
            float(row["summary"]["metrics"]["prediction_spearman"]["mean"])
            for row in current_relationships
        ]
        top5_jaccard = [
            float(row["summary"]["metrics"]["top5_jaccard"]["mean"])
            for row in current_relationships
        ]
        rebuild = _contract_rebuild_required(
            adopted=bool(gate["passed"]),
            prediction_spearman=prediction_spearman,
            top5_jaccard=top5_jaccard,
            rules=rules,
        )
        any_rebuild = any_rebuild or bool(rebuild["required"])
        head = _head(study, horizon)
        evidence[str(horizon)] = {
            **head,
            "capacity_selection": selection["heads"][str(horizon)],
            "adaptive_iterations_2023_2024_2025": [
                int(row["adaptive_iteration"]) for row in current_annual
            ],
            "fixed_capacity_gate": gate,
            "path_guardrail": path_guardrail,
            "contract_change": rebuild,
        }
        final_heads[str(horizon)] = {
            "name": head["name"],
            "capacity_policy": (
                "pre_2023_multi_origin_fixed"
                if gate["passed"]
                else "existing_prior_year_adaptive"
            ),
            "fixed_iteration": (
                int(selection["heads"][str(horizon)]["selected_iteration"])
                if gate["passed"]
                else None
            ),
            "requires_oos_contract_rebuild": bool(rebuild["required"]),
        }
    decision = {
        "status": "completed_capacity_stability_audit",
        "final_mfe_heads": final_heads,
        "any_fixed_capacity_adopted": any(
            value["capacity_policy"] == "pre_2023_multi_origin_fixed"
            for value in final_heads.values()
        ),
        "requires_oos_mfe_contract_rebuild": bool(any_rebuild),
        "rebuild_horizons": [
            int(horizon)
            for horizon, value in final_heads.items()
            if value["requires_oos_contract_rebuild"]
        ],
        "next_step": (
            "rebuild_only_materially_changed_adopted_mfe_heads"
            if any_rebuild
            else "retain_current_oos_mfe_contract"
        ),
        "does_not_select": list(study["non_selections"]),
    }
    summary = {
        "schema": SUMMARY_SCHEMA,
        "status": decision["status"],
        "completed_at": _now(),
        "study_id": STUDY_ID,
        "scope": {
            "capacity_selection_years": list(SELECTION_YEARS),
            "decision_years": list(DECISION_YEARS),
            "horizons": list(HORIZONS),
            "maximum_consumed_outcome_date": inputs.maximum_outcome_date,
            "selection_task_count": len(_selection_tasks(study)),
            "outer_task_count": len(_outer_tasks(study)),
            "forbidden_2026_row_count": 0,
        },
        "selection": selection,
        "annual": annual,
        "paired": paired,
        "relationships": relationships,
        "evidence": evidence,
        "decision": decision,
        "disclosure": {
            "capacity_selection": (
                "Only 2019-2022 rolling origins selected the fixed count; "
                "2023-2025 outcomes were not used."
            ),
            "decision_fold_reuse": study["folds"]["decision_fold_role"],
            "top5": "Top 5% is a strong-candidate diagnostic, not a slot count.",
            "2026": "No 2026 row, outcome, prediction, or metric was read.",
        },
    }
    output_root.mkdir(parents=True, exist_ok=True)
    summary_path = output_root / "summary.json"
    _write_json(summary_path, summary)
    _write_json(output_root / "decision.json", decision)
    _write_json(DEFAULT_RECORD_ROOT / "config.json", study)
    _write_json(
        DEFAULT_RECORD_ROOT / "result.json",
        {
            "schema": SUMMARY_SCHEMA,
            "status": summary["status"],
            "completed_at": summary["completed_at"],
            "study_id": STUDY_ID,
            "scope": summary["scope"],
            "evidence": evidence,
            "decision": decision,
            "disclosure": summary["disclosure"],
            "full_output": _file_record(summary_path),
        },
    )
    _emit(
        "capacity_stability_evaluation_completed",
        final_capacity_policy={
            horizon: value["capacity_policy"] for horizon, value in final_heads.items()
        },
        rebuild_horizons=decision["rebuild_horizons"],
    )
    return summary


def run_pending(
    *,
    study_path: Path = DEFAULT_STUDY_PATH,
    output_root: Path = DEFAULT_OUTPUT_ROOT,
) -> dict[str, Any]:
    study = load_study(study_path)
    feature_study = source.load_study(
        _resolve(study["sources"]["feature_study_config"])
    )
    feature_output_root = _resolve(study["sources"]["feature_output_root"])
    inputs, _pack = source._load_validated_inputs(feature_study)
    for task in _selection_tasks(study):
        _run_curve_task(
            study=study,
            study_path=study_path,
            feature_study=feature_study,
            feature_output_root=feature_output_root,
            output_root=output_root,
            inputs=inputs,
            task=task,
        )
    selection = select_capacities(
        study=study,
        study_path=study_path,
        feature_study=feature_study,
        output_root=output_root,
    )
    for task in _outer_tasks(study):
        _run_outer_task(
            study=study,
            study_path=study_path,
            feature_study=feature_study,
            feature_output_root=feature_output_root,
            output_root=output_root,
            inputs=inputs,
            selection=selection,
            task=task,
        )
    return evaluate(study_path=study_path, output_root=output_root)


def status(
    *,
    study_path: Path = DEFAULT_STUDY_PATH,
    output_root: Path = DEFAULT_OUTPUT_ROOT,
) -> dict[str, Any]:
    study = load_study(study_path)
    feature_study = source.load_study(
        _resolve(study["sources"]["feature_study_config"])
    )
    curve_tasks = _selection_tasks(study)
    completed_curves = sum(
        _curve_task_complete(
            _task_result_path(output_root, task),
            task=task,
            study=study,
            study_path=study_path,
            feature_study=feature_study,
        )
        for task in curve_tasks
    )
    selection_path = output_root / "capacity_selection" / "selection.json"
    selection: dict[str, Any] | None = None
    if selection_path.is_file():
        try:
            selection = _load_selection(output_root=output_root, study_path=study_path)
        except (KeyError, OSError, TypeError, ValueError, json.JSONDecodeError):
            selection = None
    outer_tasks = _outer_tasks(study)
    completed_outer = 0
    if selection is not None:
        completed_outer = sum(
            _outer_task_complete(
                _task_result_path(output_root, task),
                task=task,
                study=study,
                study_path=study_path,
                feature_study=feature_study,
                selected_iteration=int(
                    selection["heads"][str(task["horizon"])]["selected_iteration"]
                ),
            )
            for task in outer_tasks
        )
    return {
        "study_id": STUDY_ID,
        "capacity_curves": {
            "completed": int(completed_curves),
            "total": len(curve_tasks),
        },
        "capacity_selection_completed": selection is not None,
        "outer_models": {
            "completed": int(completed_outer),
            "total": len(outer_tasks),
        },
        "summary_completed": (output_root / "summary.json").is_file(),
    }


def self_test() -> dict[str, Any]:
    selected = _select_from_losses(
        {
            2019: {4: 1.03, 8: 1.01, 16: 1.00, 32: 1.005},
            2020: {4: 1.02, 8: 1.00, 16: 1.002, 32: 1.01},
            2021: {4: 1.03, 8: 1.01, 16: 1.00, 32: 1.004},
            2022: {4: 1.01, 8: 1.00, 16: 1.001, 32: 1.006},
        },
        (4, 8, 16, 32),
    )
    if selected["selected_iteration"] not in (8, 16):
        raise AssertionError("one-standard-error selection is invalid")
    rules = load_study()["decision"]
    gate = _decision_gate(
        relative_mae_harm=(0.0, 0.001, -0.001),
        rank_ic_delta=(0.001, 0.0, -0.001),
        top5_mfe_delta=(0.0, 0.001, -0.0005),
        path_guardrail={"rejected": False},
        rules=rules,
    )
    if not gate["passed"]:
        raise AssertionError("capacity decision gate rejected a valid example")
    rebuild = _contract_rebuild_required(
        adopted=True,
        prediction_spearman=(0.99, 0.94, 0.99),
        top5_jaccard=(0.90, 0.90, 0.90),
        rules=rules,
    )
    if not rebuild["required"] or rebuild["materially_changed_years"] != [2024]:
        raise AssertionError("contract rebuild boundary is invalid")
    return {
        "status": "passed",
        "selected_iteration": selected["selected_iteration"],
        "decision_gate": gate["passed"],
        "rebuild_years": rebuild["materially_changed_years"],
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Audit fixed multi-origin capacity for the frozen Seq100 MFE heads."
    )
    parser.add_argument("--study", type=Path, default=DEFAULT_STUDY_PATH)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--status", action="store_true")
    group.add_argument("--run-pending", action="store_true")
    group.add_argument("--evaluate", action="store_true")
    group.add_argument("--self-test", action="store_true")
    return parser


def main() -> None:
    arguments = _parser().parse_args()
    study_path = _resolve(arguments.study)
    output_root = _resolve(arguments.output_root)
    if arguments.status:
        result = status(study_path=study_path, output_root=output_root)
    elif arguments.run_pending:
        result = run_pending(study_path=study_path, output_root=output_root)
    elif arguments.evaluate:
        result = evaluate(study_path=study_path, output_root=output_root)
    else:
        result = self_test()
    print(json.dumps(result, ensure_ascii=False, indent=2, default=_json_default))


if __name__ == "__main__":
    main()
