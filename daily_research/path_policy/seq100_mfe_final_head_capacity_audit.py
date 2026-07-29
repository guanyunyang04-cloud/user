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
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from daily_research.path_policy import seq100_entry_contract_oos as contract
from daily_research.path_policy import seq100_entry_role_synthesis as entry
from daily_research.path_policy import seq100_mfe_capacity_stability_audit as prior
from daily_research.path_policy import seq100_mfe_feature_family_audit as feature
from daily_research.path_policy import seq100_mfe_feature_union_audit as union
from daily_research.path_policy import seq100_mfe_objective_alignment as objective
from daily_research.path_policy import seq100_path_label_learnability as base
from daily_research.path_policy import seq100_signal_quality as signal_quality

WORKSPACE_ROOT = feature.WORKSPACE_ROOT
STUDY_ID = "seq100_mfe_final_head_capacity_audit_v1"
DEFAULT_STUDY_PATH = (
    WORKSPACE_ROOT
    / "daily_research/studies/seq100_mfe_final_head_capacity_audit_v1.json"
)
DEFAULT_OUTPUT_ROOT = (
    WORKSPACE_ROOT
    / "daily_research/output/path_policy/studies"
    / "seq100_mfe_final_head_capacity_audit_v1"
)
DEFAULT_RECORD_ROOT = (
    WORKSPACE_ROOT
    / "daily_research/research_records/seq100"
    / "seq100_mfe_final_head_capacity_audit_v1"
)
TUNING_TASK_SCHEMA = "seq100_mfe_final_head_tuning_task/v1"
OUTER_TASK_SCHEMA = "seq100_mfe_final_head_long_outer_task/v1"
CONTRACT_TASK_SCHEMA = "seq100_mfe_capacity_contract_task/v1"
SELECTION_SCHEMA = "seq100_mfe_final_head_capacity_selection/v1"
SUMMARY_SCHEMA = "seq100_mfe_final_head_capacity_summary/v1"
CONTRACT_V3_SCHEMA = "seq100_entry_contract_oos_manifest/v3"
MODEL_YEARS = (2023, 2024, 2025)
NEW_TUNING_YEARS = (2024, 2025)
CONTRACT_REBUILD_YEARS = (2020, 2021, 2022)
CONTRACT_YEARS = (2020, 2021, 2022, 2023, 2024, 2025)
HORIZONS = (10, 20)
BASELINE_POLICY = "current_base_head_prior_year"
FORMAL_POLICIES = ("final_head_prior_year", "fixed_256", "fixed_512")
ALL_POLICIES = (BASELINE_POLICY, *FORMAL_POLICIES)


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


def _file_record(path: Path, **metadata: Any) -> dict[str, Any]:
    return {
        "path": path.relative_to(WORKSPACE_ROOT).as_posix(),
        "size": int(path.stat().st_size),
        "sha256": _sha256(path),
        **metadata,
    }


def _record_exists(record: Mapping[str, Any]) -> bool:
    path = _resolve(record["path"])
    return path.is_file() and path.stat().st_size == int(record["size"])


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
        raise ValueError("final-head capacity study id changed")
    folds = dict(payload.get("folds", {}) or {})
    if tuple(int(value) for value in folds.get("model_years", [])) != MODEL_YEARS:
        raise ValueError("final-head model years changed")
    if (
        tuple(int(value) for value in folds.get("new_tuning_model_years", []))
        != NEW_TUNING_YEARS
    ):
        raise ValueError("new final-head tuning years changed")
    if (
        tuple(int(value) for value in folds.get("contract_rebuild_years", []))
        != CONTRACT_REBUILD_YEARS
    ):
        raise ValueError("contract rebuild years changed")
    if folds.get("maximum_outcome_date") != "2025-12-31":
        raise ValueError("final-head outcome boundary changed")
    if int(folds.get("forbidden_outcome_year", -1)) != 2026:
        raise ValueError("2026 must remain forbidden")
    heads = dict(payload.get("heads", {}) or {})
    expected_heads = {
        10: ("turnover_cost_proxy", ("turnover_cost_proxy",)),
        20: ("breakout_retest_levels", ("breakout_retest_levels",)),
    }
    if tuple(sorted(int(value) for value in heads)) != HORIZONS:
        raise ValueError("final-head horizons changed")
    catalog = feature.feature_catalog()
    for horizon, (name, families) in expected_heads.items():
        head = dict(heads[str(horizon)])
        current_families = tuple(str(value) for value in head.get("families", []))
        if str(head.get("name")) != name or current_families != families:
            raise ValueError(f"frozen D{horizon} head changed")
        if any(family not in catalog for family in current_families):
            raise ValueError(f"D{horizon} contains an unknown feature family")
    policies = dict(payload.get("policies", {}) or {})
    if str(policies.get("baseline")) != BASELINE_POLICY:
        raise ValueError("capacity baseline policy changed")
    if tuple(str(value) for value in policies.get("formal_candidates", [])) != (
        FORMAL_POLICIES
    ):
        raise ValueError("formal capacity policies changed")
    fixed = {
        str(key): int(value) for key, value in policies["fixed_iterations"].items()
    }
    if fixed != {"fixed_256": 256, "fixed_512": 512}:
        raise ValueError("fixed capacity checkpoints changed")
    tuning = dict(payload.get("tuning", {}) or {})
    if (
        int(tuning.get("initial_maximum_iteration", -1)) != 1500
        or int(tuning.get("extension_maximum_iteration", -1)) != 2500
        or int(tuning.get("early_stopping_rounds", -1)) != 100
        or int(tuning.get("reuse_model_year", -1)) != 2023
        or int(tuning.get("reuse_validation_year", -1)) != 2022
    ):
        raise ValueError("final-head tuning protocol changed")
    expected_reuse = {"10": 86, "20": 41}
    if {
        str(key): int(value)
        for key, value in tuning.get("reuse_expected_iterations", {}).items()
    } != expected_reuse:
        raise ValueError("reused final-head counts changed")
    model = dict(payload.get("model", {}) or {})
    if (
        bool(model.get("change_learning_rate", True))
        or bool(model.get("change_tree_shape", True))
        or bool(model.get("hyperparameter_search", True))
        or int(model.get("long_outer_minimum_iteration", -1)) != 512
    ):
        raise ValueError("model parameters must remain frozen")
    v3_root = _resolve(payload["contract_v3"]["output_root"])
    v2_manifest = _resolve(payload["sources"]["entry_contract_v2_manifest"])
    if v3_root in v2_manifest.parents:
        raise ValueError("v3 cannot overwrite the v2 contract")
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


def _tuning_tasks(study: Mapping[str, Any]) -> list[dict[str, Any]]:
    return [
        {
            **_head(study, horizon),
            "year": int(year),
            "validation_year": int(year - 1),
            "stage": "final_head_tuning",
            "task_id": f"mfe{horizon}_{year}_final_head_tuning",
        }
        for horizon in HORIZONS
        for year in NEW_TUNING_YEARS
    ]


def _outer_tasks(study: Mapping[str, Any]) -> list[dict[str, Any]]:
    return [
        {
            **_head(study, horizon),
            "year": int(year),
            "stage": "long_outer",
            "task_id": f"mfe{horizon}_{year}_long_outer",
        }
        for horizon in HORIZONS
        for year in MODEL_YEARS
    ]


def _task_dir(output_root: Path, task: Mapping[str, Any]) -> Path:
    horizon = int(task["horizon"])
    year = int(task["year"])
    if task["stage"] == "final_head_tuning":
        prefix = "tuning"
    elif task["stage"] == "long_outer":
        prefix = "outer"
    else:
        prefix = "contract_rebuild"
    return output_root / prefix / f"fold_{year}" / f"h{horizon:02d}" / str(task["name"])


def _task_result_path(output_root: Path, task: Mapping[str, Any]) -> Path:
    return _task_dir(output_root, task) / "task_result.json"


def _parameters(feature_study: Mapping[str, Any]) -> dict[str, Any]:
    return feature._model_parameters(feature_study)[0]


def _tuning_resolved(
    *, recorded_rounds: int, best_iteration: int, patience: int
) -> bool:
    if recorded_rounds <= 0 or best_iteration <= 0 or best_iteration > recorded_rounds:
        raise ValueError("invalid tuning history")
    return int(recorded_rounds) - int(best_iteration) >= int(patience)


def _tuning_task_complete(
    path: Path,
    *,
    task: Mapping[str, Any],
    study_path: Path,
    feature_study: Mapping[str, Any],
) -> bool:
    if not path.is_file():
        return False
    try:
        import lightgbm as lgb

        result = json.loads(path.read_text(encoding="utf-8"))
        expected = {
            "schema": TUNING_TASK_SCHEMA,
            "status": "completed",
            "study_id": STUDY_ID,
            "task_id": task["task_id"],
            "stage": task["stage"],
            "head": task["name"],
            "families": list(task["families"]),
            "model_year": int(task["year"]),
            "validation_year": int(task["validation_year"]),
            "horizon": int(task["horizon"]),
            "config_sha256": _config_hash(study_path),
        }
        if any(result.get(key) != value for key, value in expected.items()):
            return False
        if dict(result.get("parameters", {}) or {}) != _parameters(feature_study):
            return False
        files = dict(result.get("files", {}) or {})
        if not files or not all(_record_exists(record) for record in files.values()):
            return False
        model = lgb.Booster(model_file=str(_resolve(files["model"]["path"])))
        if int(model.num_trees()) != int(result["best_iteration"]):
            return False
        curve = pd.read_parquet(_resolve(files["loss_curve"]["path"]))
        if (
            list(curve.columns) != ["iteration", "huber"]
            or len(curve) != int(result["recorded_rounds"])
            or not np.array_equal(
                curve["iteration"].to_numpy(), np.arange(1, len(curve) + 1)
            )
        ):
            return False
        rows = np.load(_resolve(files["evaluation_rows"]["path"]), allow_pickle=False)
        prediction = np.load(_resolve(files["prediction"]["path"]), allow_pickle=False)
        if rows.shape != prediction.shape:
            return False
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


def _train_tuning_search(
    *,
    lgb: Any,
    parameters: Mapping[str, Any],
    datasets: Any,
    maximum_rounds: int,
    patience: int,
) -> tuple[Any, list[float]]:
    evaluations: dict[str, dict[str, list[float]]] = {}
    model = lgb.train(
        dict(parameters),
        datasets.train_set,
        num_boost_round=int(maximum_rounds),
        valid_sets=[datasets.evaluation_set],
        valid_names=["inner_validation"],
        callbacks=[
            lgb.early_stopping(
                stopping_rounds=int(patience),
                first_metric_only=True,
                verbose=False,
            ),
            lgb.record_evaluation(evaluations),
            lgb.log_evaluation(period=0),
        ],
    )
    losses = evaluations.get("inner_validation", {}).get("huber", [])
    if not losses:
        raise RuntimeError("final-head tuning did not record Huber loss")
    return model, [float(value) for value in losses]


def _run_tuning_task(
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
    if _tuning_task_complete(
        result_path,
        task=task,
        study_path=study_path,
        feature_study=feature_study,
    ):
        return json.loads(result_path.read_text(encoding="utf-8"))
    horizon = int(task["horizon"])
    validation_year = int(task["validation_year"])
    tuning = dict(study["tuning"])
    initial_maximum = int(tuning["initial_maximum_iteration"])
    extension_maximum = int(tuning["extension_maximum_iteration"])
    patience = int(tuning["early_stopping_rounds"])
    extra, manifests = union._open_union(
        feature_output_root=feature_output_root,
        families=task["families"],
        candidate_count=inputs.candidate_count,
    )
    fold = inputs.common_path_rows(validation_year, horizon)
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
        parameters = _parameters(feature_study)
        _emit(
            "final_head_tuning_started",
            task_id=task["task_id"],
            maximum_rounds=initial_maximum,
            train_rows=len(fold.train_rows),
            evaluation_rows=len(fold.evaluation_rows),
        )
        started = time.perf_counter()
        model, losses = _train_tuning_search(
            lgb=lgb,
            parameters=parameters,
            datasets=datasets,
            maximum_rounds=initial_maximum,
            patience=patience,
        )
        best_iteration = int(np.argmin(np.asarray(losses, dtype=np.float64))) + 1
        extended = False
        if not _tuning_resolved(
            recorded_rounds=len(losses),
            best_iteration=best_iteration,
            patience=patience,
        ):
            extended = True
            del model
            gc.collect()
            _emit(
                "final_head_tuning_extended",
                task_id=task["task_id"],
                maximum_rounds=extension_maximum,
            )
            model, losses = _train_tuning_search(
                lgb=lgb,
                parameters=parameters,
                datasets=datasets,
                maximum_rounds=extension_maximum,
                patience=patience,
            )
            best_iteration = int(np.argmin(np.asarray(losses, dtype=np.float64))) + 1
        resolved = _tuning_resolved(
            recorded_rounds=len(losses),
            best_iteration=best_iteration,
            patience=patience,
        )
        elapsed = float(time.perf_counter() - started)
        prediction = feature._predict(
            model, datasets.evaluation_sequence, best_iteration
        )
        daily, metrics = base.regression_metrics(
            date_idx=inputs.candidate_date_idx[datasets.evaluation_rows],
            actual=evaluation_raw,
            prediction=prediction,
            date_values=inputs.date_values,
            horizon=horizon,
        )
        curve = pd.DataFrame(
            {
                "iteration": np.arange(1, len(losses) + 1, dtype=np.int32),
                "huber": np.asarray(losses, dtype=np.float64),
            }
        )
        output_dir = _task_dir(output_root, task)
        output_dir.mkdir(parents=True, exist_ok=True)
        model_path = output_dir / "model.txt"
        curve_path = output_dir / "loss_curve.parquet"
        prediction_path = output_dir / "prediction.npy"
        rows_path = output_dir / "evaluation_rows.npy"
        daily_path = output_dir / "daily_metrics.parquet"
        model.save_model(str(model_path), num_iteration=best_iteration)
        curve.to_parquet(curve_path, index=False, compression="zstd")
        _save_npy(prediction_path, prediction)
        _save_npy(rows_path, datasets.evaluation_rows)
        daily.to_parquet(daily_path, index=False, compression="zstd")
        result = {
            "schema": TUNING_TASK_SCHEMA,
            "status": "completed",
            "completed_at": _now(),
            "study_id": STUDY_ID,
            "task_id": task["task_id"],
            "stage": task["stage"],
            "head": task["name"],
            "families": list(task["families"]),
            "model_year": int(task["year"]),
            "validation_year": validation_year,
            "horizon": horizon,
            "target": "mfe",
            "purge_days": horizon,
            "train_row_count": len(fold.train_rows),
            "evaluation_row_count": len(fold.evaluation_rows),
            "best_iteration": best_iteration,
            "recorded_rounds": len(losses),
            "patience": patience,
            "extended_search": bool(extended),
            "capacity_resolved": bool(resolved),
            "training_seconds": elapsed,
            "parameters": parameters,
            "feature_count": len(datasets.feature_names),
            "added_feature_count": len(extra.feature_names),
            "family_manifests": manifests,
            "metrics": metrics,
            "config_sha256": _config_hash(study_path),
            "files": {
                "model": _file_record(model_path),
                "loss_curve": _file_record(curve_path),
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
            "final_head_tuning_completed",
            task_id=task["task_id"],
            best_iteration=best_iteration,
            recorded_rounds=len(losses),
            resolved=resolved,
            seconds=elapsed,
        )
        return result
    finally:
        if datasets is not None:
            feature._release_datasets(datasets)
        del model, fold, extra


def _prior_curve_task(
    *,
    prior_study: Mapping[str, Any],
    horizon: int,
    validation_year: int,
) -> dict[str, Any]:
    return next(
        task
        for task in prior._selection_tasks(prior_study)
        if int(task["horizon"]) == int(horizon)
        and int(task["year"]) == int(validation_year)
    )


def _load_prior_curve_count(
    *,
    study: Mapping[str, Any],
    feature_study: Mapping[str, Any],
    horizon: int,
    validation_year: int,
) -> dict[str, Any]:
    prior_study_path = _resolve(study["sources"]["prior_capacity_study_config"])
    prior_output_root = _resolve(study["sources"]["prior_capacity_output_root"])
    prior_study = prior.load_study(prior_study_path)
    task = _prior_curve_task(
        prior_study=prior_study,
        horizon=horizon,
        validation_year=validation_year,
    )
    path = prior._task_result_path(prior_output_root, task)
    if not prior._curve_task_complete(
        path,
        task=task,
        study=prior_study,
        study_path=prior_study_path,
        feature_study=feature_study,
    ):
        raise RuntimeError(
            f"reused final-head curve is incomplete: D{horizon} {validation_year}"
        )
    result = json.loads(path.read_text(encoding="utf-8"))
    head = _head(study, horizon)
    if (
        result.get("head") != head["name"]
        or tuple(result.get("families", [])) != head["families"]
    ):
        raise ValueError("reused curve does not belong to the frozen final head")
    best_iteration = int(result["minimum_full_curve_iteration"])
    return {
        "best_iteration": best_iteration,
        "capacity_resolved": (
            int(result["trained_iterations"]) - best_iteration
            >= int(study["tuning"]["early_stopping_rounds"])
        ),
        "validation_year": int(validation_year),
        "source": "reused_complete_final_head_curve",
        "source_task_id": result["task_id"],
        "source_result": _file_record(path),
    }


def select_capacities(
    *,
    study: Mapping[str, Any],
    study_path: Path,
    feature_study: Mapping[str, Any],
    feature_output_root: Path,
    output_root: Path,
) -> dict[str, Any]:
    heads: dict[str, Any] = {}
    for horizon in HORIZONS:
        years: dict[str, Any] = {}
        for year in MODEL_YEARS:
            if year == int(study["tuning"]["reuse_model_year"]):
                final_record = _load_prior_curve_count(
                    study=study,
                    feature_study=feature_study,
                    horizon=horizon,
                    validation_year=year - 1,
                )
                expected = int(
                    study["tuning"]["reuse_expected_iterations"][str(horizon)]
                )
                if int(final_record["best_iteration"]) != expected:
                    raise ValueError(
                        f"reused D{horizon} count changed: "
                        f"{final_record['best_iteration']} != {expected}"
                    )
            else:
                task = next(
                    task
                    for task in _tuning_tasks(study)
                    if int(task["horizon"]) == horizon and int(task["year"]) == year
                )
                path = _task_result_path(output_root, task)
                if not _tuning_task_complete(
                    path,
                    task=task,
                    study_path=study_path,
                    feature_study=feature_study,
                ):
                    raise RuntimeError(f"tuning task is incomplete: {task['task_id']}")
                result = json.loads(path.read_text(encoding="utf-8"))
                final_record = {
                    "best_iteration": int(result["best_iteration"]),
                    "capacity_resolved": bool(result["capacity_resolved"]),
                    "validation_year": int(result["validation_year"]),
                    "source": "new_final_head_prior_year_tuning",
                    "source_task_id": result["task_id"],
                    "source_result": _file_record(path),
                }
            current_result, _prediction, _rows = feature._load_outer_result(
                feature_output_root,
                year=year,
                horizon=horizon,
                variant=_head(study, horizon)["name"],
                study=feature_study,
            )
            current_iteration = (
                int(current_result["best_iteration"])
                if "best_iteration" in current_result
                else int(current_result["fixed_iteration_source"]["best_iteration"])
            )
            years[str(year)] = {
                "current_iteration": current_iteration,
                "final_head_iteration": int(final_record["best_iteration"]),
                "final_head_capacity_resolved": bool(final_record["capacity_resolved"]),
                "validation_year": int(final_record["validation_year"]),
                "final_head_source": final_record,
                "current_source_task_id": current_result["task_id"],
            }
        heads[str(horizon)] = {**_head(study, horizon), "years": years}
    payload = {
        "schema": SELECTION_SCHEMA,
        "status": "completed",
        "completed_at": _now(),
        "study_id": STUDY_ID,
        "config_sha256": _config_hash(study_path),
        "model_years": list(MODEL_YEARS),
        "heads": heads,
        "formal_candidates": list(FORMAL_POLICIES),
        "fixed_iterations": {
            key: int(value)
            for key, value in study["policies"]["fixed_iterations"].items()
        },
    }
    _write_json(output_root / "capacity_selection.json", payload)
    _emit(
        "final_head_capacity_selection_completed",
        iterations={
            horizon: {
                year: {
                    "current": record["current_iteration"],
                    "final_head": record["final_head_iteration"],
                    "resolved": record["final_head_capacity_resolved"],
                }
                for year, record in head["years"].items()
            }
            for horizon, head in heads.items()
        },
    )
    return payload


def _load_selection(
    *,
    output_root: Path,
    study_path: Path,
) -> dict[str, Any]:
    path = output_root / "capacity_selection.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    if (
        payload.get("schema") != SELECTION_SCHEMA
        or payload.get("status") != "completed"
        or payload.get("study_id") != STUDY_ID
        or payload.get("config_sha256") != _config_hash(study_path)
        or payload.get("model_years") != list(MODEL_YEARS)
        or payload.get("formal_candidates") != list(FORMAL_POLICIES)
    ):
        raise ValueError("capacity selection is stale or semantically invalid")
    return payload


def _policy_iterations(
    *,
    study: Mapping[str, Any],
    selection: Mapping[str, Any],
    horizon: int,
    year: int,
) -> dict[str, int | None]:
    record = selection["heads"][str(horizon)]["years"][str(year)]
    return {
        BASELINE_POLICY: int(record["current_iteration"]),
        "final_head_prior_year": (
            int(record["final_head_iteration"])
            if bool(record["final_head_capacity_resolved"])
            else None
        ),
        "fixed_256": int(study["policies"]["fixed_iterations"]["fixed_256"]),
        "fixed_512": int(study["policies"]["fixed_iterations"]["fixed_512"]),
    }


def _prefix_equivalence(
    *,
    old: np.ndarray,
    current_prefix: np.ndarray,
    dates: np.ndarray,
    absolute_tolerance: float,
    relative_tolerance: float,
    minimum_daily_spearman: float,
) -> dict[str, Any]:
    left = np.asarray(old, dtype=np.float64)
    right = np.asarray(current_prefix, dtype=np.float64)
    if left.shape != right.shape:
        raise ValueError("prefix predictions have different shapes")
    valid = np.isfinite(left) & np.isfinite(right)
    if not bool(valid.all()):
        raise ValueError("prefix equivalence requires finite predictions")
    absolute = np.abs(right - left)
    correlation, date_count = contract._daily_spearman(left, right, dates)
    allclose = bool(
        np.allclose(
            left,
            right,
            atol=float(absolute_tolerance),
            rtol=float(relative_tolerance),
        )
    )
    passed = (
        allclose
        and math.isfinite(correlation)
        and correlation >= float(minimum_daily_spearman)
    )
    return {
        "passed": bool(passed),
        "allclose": allclose,
        "maximum_absolute_difference": float(np.max(absolute)),
        "mean_absolute_difference": float(np.mean(absolute)),
        "daily_spearman": float(correlation),
        "daily_spearman_date_count": int(date_count),
        "absolute_tolerance": float(absolute_tolerance),
        "relative_tolerance": float(relative_tolerance),
        "minimum_daily_spearman": float(minimum_daily_spearman),
    }


def _outer_task_complete(
    path: Path,
    *,
    task: Mapping[str, Any],
    study_path: Path,
    feature_study: Mapping[str, Any],
    policy_iterations: Mapping[str, int | None],
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
            "families": list(task["families"]),
            "model_year": int(task["year"]),
            "horizon": int(task["horizon"]),
            "policy_order": list(ALL_POLICIES),
            "policy_iterations": dict(policy_iterations),
            "config_sha256": _config_hash(study_path),
        }
        if any(result.get(key) != value for key, value in expected.items()):
            return False
        if dict(result.get("parameters", {}) or {}) != _parameters(feature_study):
            return False
        files = dict(result.get("files", {}) or {})
        if not files or not all(_record_exists(record) for record in files.values()):
            return False
        model = lgb.Booster(model_file=str(_resolve(files["model"]["path"])))
        if int(model.num_trees()) != int(result["trained_iterations"]):
            return False
        evaluation = np.load(
            _resolve(files["evaluation_predictions"]["path"]), allow_pickle=False
        )
        candidate = np.load(
            _resolve(files["candidate_predictions"]["path"]), allow_pickle=False
        )
        evaluation_rows = np.load(
            _resolve(files["evaluation_rows"]["path"]), allow_pickle=False
        )
        candidate_rows = np.load(
            _resolve(files["candidate_rows"]["path"]), allow_pickle=False
        )
        if evaluation.shape != (len(evaluation_rows), len(ALL_POLICIES)):
            return False
        if candidate.shape != (len(candidate_rows), len(ALL_POLICIES)):
            return False
        if tuple(evaluation.shape) != tuple(files["evaluation_predictions"]["shape"]):
            return False
        if tuple(candidate.shape) != tuple(files["candidate_predictions"]["shape"]):
            return False
        return bool(result["prefix_equivalence"]["passed"])
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


def _prediction_matrix(
    *,
    model: Any,
    sequence: Any,
    policy_iterations: Mapping[str, int | None],
) -> np.ndarray:
    cache: dict[int, np.ndarray] = {}
    columns: list[np.ndarray] = []
    for policy in ALL_POLICIES:
        iterations = policy_iterations[policy]
        if iterations is None:
            columns.append(np.full(len(sequence), np.nan, dtype=np.float32))
            continue
        count = int(iterations)
        if count not in cache:
            cache[count] = feature._predict(model, sequence, count)
        columns.append(cache[count])
    return np.column_stack(columns).astype(np.float32, copy=False)


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
    policy_iterations = _policy_iterations(
        study=study,
        selection=selection,
        horizon=horizon,
        year=year,
    )
    result_path = _task_result_path(output_root, task)
    if _outer_task_complete(
        result_path,
        task=task,
        study_path=study_path,
        feature_study=feature_study,
        policy_iterations=policy_iterations,
    ):
        return json.loads(result_path.read_text(encoding="utf-8"))
    counts = [int(value) for value in policy_iterations.values() if value is not None]
    trained_iterations = max(
        int(study["model"]["long_outer_minimum_iteration"]), *counts
    )
    extra, manifests = union._open_union(
        feature_output_root=feature_output_root,
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
        parameters = _parameters(feature_study)
        _emit(
            "long_outer_training_started",
            task_id=task["task_id"],
            iterations=trained_iterations,
            policy_iterations=policy_iterations,
            train_rows=len(fold.train_rows),
            evaluation_rows=len(fold.evaluation_rows),
        )
        started = time.perf_counter()
        model = lgb.train(
            parameters,
            datasets.train_set,
            num_boost_round=trained_iterations,
            valid_sets=[datasets.evaluation_set],
            valid_names=["outer_evaluation"],
            callbacks=[lgb.log_evaluation(period=0)],
        )
        elapsed = float(time.perf_counter() - started)
        evaluation_predictions = _prediction_matrix(
            model=model,
            sequence=datasets.evaluation_sequence,
            policy_iterations=policy_iterations,
        )
        old_result, old_prediction, old_rows = feature._load_outer_result(
            feature_output_root,
            year=year,
            horizon=horizon,
            variant=str(task["name"]),
            study=feature_study,
        )
        if not np.array_equal(datasets.evaluation_rows, old_rows):
            raise ValueError(f"current rows changed for {task['task_id']}")
        baseline_position = ALL_POLICIES.index(BASELINE_POLICY)
        equivalence = _prefix_equivalence(
            old=old_prediction,
            current_prefix=evaluation_predictions[:, baseline_position],
            dates=inputs.candidate_date_idx[datasets.evaluation_rows],
            absolute_tolerance=float(study["prefix_equivalence"]["absolute_tolerance"]),
            relative_tolerance=float(study["prefix_equivalence"]["relative_tolerance"]),
            minimum_daily_spearman=float(
                study["prefix_equivalence"]["minimum_daily_spearman"]
            ),
        )
        output_dir = _task_dir(output_root, task)
        output_dir.mkdir(parents=True, exist_ok=True)
        if not equivalence["passed"]:
            diagnostic = {
                "status": "invalid_prefix_equivalence",
                "task_id": task["task_id"],
                "old_source_task_id": old_result["task_id"],
                "prefix_equivalence": equivalence,
            }
            _write_json(output_dir / "prefix_failure.json", diagnostic)
            raise RuntimeError(f"prefix equivalence failed: {task['task_id']}")
        vocabularies = signal_quality._fit_category_vocabularies(
            inputs.categorical, fold.train_rows, inputs.categorical_columns
        )
        candidate_rows = contract._year_rows(inputs, year)
        candidate_sequence = feature._combined_sequence(
            inputs=inputs,
            row_ids=candidate_rows,
            category_vocabularies=vocabularies,
            batch_size=int(feature_study["model"]["sequence_batch_size"]),
            extra=extra,
        )
        candidate_sequence = base._memory_trimmed_sequence(
            candidate_sequence, feature_study
        )
        candidate_predictions = _prediction_matrix(
            model=model,
            sequence=candidate_sequence,
            policy_iterations=policy_iterations,
        )
        daily_frames: list[pd.DataFrame] = []
        metrics: dict[str, Any] = {}
        for position, policy in enumerate(ALL_POLICIES):
            values = evaluation_predictions[:, position]
            if not bool(np.isfinite(values).all()):
                metrics[policy] = {
                    "status": "unresolved_final_head_capacity",
                }
                continue
            daily, current_metrics = base.regression_metrics(
                date_idx=inputs.candidate_date_idx[datasets.evaluation_rows],
                actual=evaluation_raw,
                prediction=values,
                date_values=inputs.date_values,
                horizon=horizon,
            )
            daily.insert(0, "policy", policy)
            daily_frames.append(daily)
            metrics[policy] = {"status": "reported", **current_metrics}
        daily_frame = pd.concat(daily_frames, ignore_index=True)
        model_path = output_dir / "model.txt"
        evaluation_path = output_dir / "evaluation_predictions.npy"
        candidate_path = output_dir / "candidate_predictions.npy"
        evaluation_rows_path = output_dir / "evaluation_rows.npy"
        candidate_rows_path = output_dir / "candidate_rows.npy"
        vocabulary_path = output_dir / "category_vocabularies.npz"
        daily_path = output_dir / "daily_metrics.parquet"
        model.save_model(str(model_path), num_iteration=trained_iterations)
        _save_npy(evaluation_path, evaluation_predictions)
        _save_npy(candidate_path, candidate_predictions)
        _save_npy(evaluation_rows_path, datasets.evaluation_rows)
        _save_npy(candidate_rows_path, candidate_rows)
        vocabulary_record = contract._save_vocabularies(vocabulary_path, vocabularies)
        daily_frame.to_parquet(daily_path, index=False, compression="zstd")
        result = {
            "schema": OUTER_TASK_SCHEMA,
            "status": "completed",
            "completed_at": _now(),
            "study_id": STUDY_ID,
            "task_id": task["task_id"],
            "stage": task["stage"],
            "head": task["name"],
            "families": list(task["families"]),
            "model_year": year,
            "horizon": horizon,
            "target": "mfe",
            "purge_days": horizon,
            "train_row_count": len(fold.train_rows),
            "evaluation_row_count": len(fold.evaluation_rows),
            "candidate_prediction_row_count": len(candidate_rows),
            "maximum_train_signal_date_idx": fold.maximum_train_signal_date_idx,
            "trained_iterations": trained_iterations,
            "policy_order": list(ALL_POLICIES),
            "policy_iterations": policy_iterations,
            "prefix_equivalence": equivalence,
            "current_source_task_id": old_result["task_id"],
            "training_seconds": elapsed,
            "parameters": parameters,
            "feature_count": len(datasets.feature_names),
            "added_feature_count": len(extra.feature_names),
            "family_manifests": manifests,
            "metrics": metrics,
            "config_sha256": _config_hash(study_path),
            "files": {
                "model": _file_record(model_path),
                "evaluation_predictions": _file_record(
                    evaluation_path,
                    shape=list(evaluation_predictions.shape),
                    dtype=str(evaluation_predictions.dtype),
                ),
                "candidate_predictions": _file_record(
                    candidate_path,
                    shape=list(candidate_predictions.shape),
                    dtype=str(candidate_predictions.dtype),
                ),
                "evaluation_rows": _file_record(
                    evaluation_rows_path,
                    shape=list(datasets.evaluation_rows.shape),
                    dtype=str(datasets.evaluation_rows.dtype),
                ),
                "candidate_rows": _file_record(
                    candidate_rows_path,
                    shape=list(candidate_rows.shape),
                    dtype=str(candidate_rows.dtype),
                ),
                "category_vocabularies": vocabulary_record,
                "daily_metrics": _file_record(daily_path),
            },
        }
        _write_json(result_path, result)
        _emit(
            "long_outer_training_completed",
            task_id=task["task_id"],
            iterations=trained_iterations,
            seconds=elapsed,
        )
        del candidate_sequence, candidate_predictions, evaluation_predictions
        return result
    finally:
        if datasets is not None:
            feature._release_datasets(datasets)
        del model, fold, extra
        gc.collect()


def _load_outer_arrays(
    output_root: Path,
    *,
    task: Mapping[str, Any],
) -> tuple[dict[str, Any], np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    result = json.loads(
        _task_result_path(output_root, task).read_text(encoding="utf-8")
    )
    files = dict(result["files"])
    return (
        result,
        np.load(_resolve(files["evaluation_predictions"]["path"]), allow_pickle=False),
        np.load(_resolve(files["candidate_predictions"]["path"]), allow_pickle=False),
        np.load(_resolve(files["evaluation_rows"]["path"]), allow_pickle=False),
        np.load(_resolve(files["candidate_rows"]["path"]), allow_pickle=False),
    )


def _candidate_gate(
    *,
    paired: Sequence[Mapping[str, Any]],
    annual: Sequence[Mapping[str, Any]],
    path_guardrail: Mapping[str, Any],
    q_value: float,
    rules: Mapping[str, Any],
    unresolved: bool,
) -> dict[str, Any]:
    top5 = [float(row["metrics"]["top_5pct_lift"]["mean_delta"]) for row in paired]
    rank_ic = [float(row["metrics"]["rank_ic"]["mean_delta"]) for row in paired]
    tail = [
        float(row["metrics"]["daily_tail_top5_lift"]["mean_delta"]) for row in paired
    ]
    mae_harm = [float(row["relative_mae_harm"]) for row in annual]
    if not (
        len(top5) == len(rank_ic) == len(tail) == len(mae_harm) == len(MODEL_YEARS)
    ):
        raise ValueError("candidate gate requires three aligned decision years")
    values = np.asarray([*top5, *rank_ic, *tail, *mae_harm], dtype=np.float64)
    if not bool(np.isfinite(values).all()):
        raise ValueError("candidate gate inputs must be finite")
    checks = {
        "positive_top5_years": int(sum(value > 0.0 for value in top5)),
        "median_top5_mfe_delta": float(np.median(top5)),
        "worst_top5_mfe_delta": float(np.min(top5)),
        "top5_fdr_q_value": float(q_value),
        "worst_rank_ic_delta": float(np.min(rank_ic)),
        "nonnegative_tail_hit_years": int(sum(value >= 0.0 for value in tail)),
        "median_relative_mae_harm": float(np.median(mae_harm)),
        "worst_relative_mae_harm": float(np.max(mae_harm)),
        "path_guardrail_rejected": bool(path_guardrail["rejected"]),
        "capacity_unresolved": bool(unresolved),
    }
    passed = (
        checks["positive_top5_years"] >= int(rules["minimum_positive_top5_years"])
        and checks["median_top5_mfe_delta"]
        > float(rules["minimum_median_top5_mfe_delta"])
        and checks["worst_top5_mfe_delta"]
        >= float(rules["minimum_worst_top5_mfe_delta"])
        and math.isfinite(checks["top5_fdr_q_value"])
        and checks["top5_fdr_q_value"] <= float(rules["maximum_top5_fdr_q_value"])
        and checks["worst_rank_ic_delta"] >= float(rules["minimum_worst_rank_ic_delta"])
        and checks["nonnegative_tail_hit_years"]
        >= int(rules["minimum_nonnegative_tail_hit_years"])
        and checks["median_relative_mae_harm"]
        <= float(rules["maximum_median_relative_mae_harm"])
        and checks["worst_relative_mae_harm"]
        <= float(rules["maximum_worst_relative_mae_harm"])
        and not checks["path_guardrail_rejected"]
        and not checks["capacity_unresolved"]
    )
    return {
        "passed": bool(passed),
        "checks": checks,
        "top5_mfe_delta_2023_2024_2025": top5,
        "rank_ic_delta_2023_2024_2025": rank_ic,
        "tail_hit_delta_2023_2024_2025": tail,
        "relative_mae_harm_2023_2024_2025": mae_harm,
    }


def _winner_sort_key(evidence: Mapping[str, Any]) -> tuple[float, ...]:
    gate = dict(evidence["gate"])
    top5 = [float(value) for value in gate["top5_mfe_delta_2023_2024_2025"]]
    rank_ic = [float(value) for value in gate["rank_ic_delta_2023_2024_2025"]]
    mae_harm = [float(value) for value in gate["relative_mae_harm_2023_2024_2025"]]
    iterations = [int(value) for value in evidence["iterations_2023_2024_2025"]]
    return (
        float(np.median(top5)),
        float(np.min(top5)),
        float(np.median(rank_ic)),
        -float(np.median(mae_harm)),
        -float(np.median(iterations)),
    )


def _contract_change(
    *,
    relationships: Sequence[Mapping[str, Any]],
    rules: Mapping[str, Any],
) -> dict[str, Any]:
    ordered = sorted(relationships, key=lambda row: int(row["year"]))
    spearman = [
        float(row["summary"]["metrics"]["prediction_spearman"]["mean"])
        for row in ordered
    ]
    jaccard = [
        float(row["summary"]["metrics"]["top5_jaccard"]["mean"]) for row in ordered
    ]
    materially_changed_years = [
        int(row["year"])
        for row, correlation, overlap in zip(ordered, spearman, jaccard, strict=True)
        if (
            not math.isfinite(correlation)
            or correlation
            < float(rules["contract_rebuild_minimum_prediction_spearman"])
            or not math.isfinite(overlap)
            or overlap < float(rules["contract_rebuild_minimum_top5_jaccard"])
        )
    ]
    return {
        "material": bool(materially_changed_years),
        "materially_changed_years": materially_changed_years,
        "prediction_spearman_2023_2024_2025": spearman,
        "top5_jaccard_2023_2024_2025": jaccard,
    }


def _write_audit_outputs(
    *,
    study: Mapping[str, Any],
    output_root: Path,
    summary: Mapping[str, Any],
) -> None:
    summary_path = output_root / "summary.json"
    _write_json(summary_path, summary)
    _write_json(output_root / "decision.json", summary["decision"])
    _write_json(DEFAULT_RECORD_ROOT / "config.json", study)
    _write_json(
        DEFAULT_RECORD_ROOT / "result.json",
        {
            "schema": SUMMARY_SCHEMA,
            "status": summary["status"],
            "completed_at": summary["completed_at"],
            "study_id": STUDY_ID,
            "scope": summary["scope"],
            "selection": summary["selection"],
            "evidence": summary["evidence"],
            "decision": summary["decision"],
            "disclosure": summary["disclosure"],
            "full_output": _file_record(summary_path),
        },
    )


def evaluate(
    *,
    study_path: Path = DEFAULT_STUDY_PATH,
    output_root: Path = DEFAULT_OUTPUT_ROOT,
) -> dict[str, Any]:
    study = load_study(study_path)
    feature_study = feature.load_study(
        _resolve(study["sources"]["feature_study_config"])
    )
    selection = _load_selection(output_root=output_root, study_path=study_path)
    inputs, pack = feature._load_validated_inputs(feature_study)
    if inputs.maximum_outcome_date != "2025-12-31":
        raise ValueError("final-head evaluation boundary changed")
    reader = objective.FuturePathReader(pack)
    annual: list[dict[str, Any]] = []
    paired: list[dict[str, Any]] = []
    relationships: list[dict[str, Any]] = []
    unresolved_by_head = {
        str(horizon): any(
            not bool(
                selection["heads"][str(horizon)]["years"][str(year)][
                    "final_head_capacity_resolved"
                ]
            )
            for year in MODEL_YEARS
        )
        for horizon in HORIZONS
    }
    for task in _outer_tasks(study):
        horizon = int(task["horizon"])
        year = int(task["year"])
        policy_iterations = _policy_iterations(
            study=study,
            selection=selection,
            horizon=horizon,
            year=year,
        )
        path = _task_result_path(output_root, task)
        if not _outer_task_complete(
            path,
            task=task,
            study_path=study_path,
            feature_study=feature_study,
            policy_iterations=policy_iterations,
        ):
            raise RuntimeError(f"long outer task is incomplete: {task['task_id']}")
        result, evaluation_predictions, _candidate, rows, _candidate_rows = (
            _load_outer_arrays(output_root, task=task)
        )
        actual = entry._path_actual(
            inputs=inputs,
            reader=reader,
            rows=rows,
            horizon=horizon,
            early_horizon=5 if horizon == 10 else 10,
        )
        baseline_position = ALL_POLICIES.index(BASELINE_POLICY)
        baseline_bundle = objective.PredictionBundle(
            variant=BASELINE_POLICY,
            year=year,
            horizon=horizon,
            rows=rows,
            prediction=evaluation_predictions[:, baseline_position],
            result=result,
        )
        baseline_daily, baseline_metrics = objective._daily_evaluation(
            inputs=inputs,
            bundle=baseline_bundle,
            peak_day=actual.peak_day,
            early_horizon=5 if horizon == 10 else 10,
        )
        baseline_mae = float(baseline_metrics["magnitude_error"]["date_equal_mae"])
        for policy in FORMAL_POLICIES:
            position = ALL_POLICIES.index(policy)
            prediction = evaluation_predictions[:, position]
            if not bool(np.isfinite(prediction).all()):
                continue
            bundle = objective.PredictionBundle(
                variant=policy,
                year=year,
                horizon=horizon,
                rows=rows,
                prediction=prediction,
                result=result,
            )
            daily, metrics = objective._daily_evaluation(
                inputs=inputs,
                bundle=bundle,
                peak_day=actual.peak_day,
                early_horizon=5 if horizon == 10 else 10,
            )
            pair_metrics = objective._paired_delta(
                challenger=daily,
                baseline=baseline_daily,
                horizon=horizon,
            )
            pair_frame = baseline_daily.merge(
                daily,
                on="date_idx",
                suffixes=("_baseline", "_candidate"),
                validate="one_to_one",
            )
            pair_path = (
                output_root
                / "evaluation"
                / policy
                / f"h{horizon:02d}"
                / f"fold_{year}_paired_daily.parquet"
            )
            pair_path.parent.mkdir(parents=True, exist_ok=True)
            pair_frame.to_parquet(pair_path, index=False, compression="zstd")
            relation_frame = entry._daily_model_relationship(
                actual=actual,
                left_name=BASELINE_POLICY,
                left_score=evaluation_predictions[:, baseline_position],
                right_name=policy,
                right_score=prediction,
            )
            relation_path = (
                output_root
                / "relationships"
                / policy
                / f"h{horizon:02d}"
                / f"fold_{year}.parquet"
            )
            relation_path.parent.mkdir(parents=True, exist_ok=True)
            relation_frame.to_parquet(relation_path, index=False, compression="zstd")
            candidate_mae = float(metrics["magnitude_error"]["date_equal_mae"])
            annual.append(
                {
                    "year": year,
                    "horizon": horizon,
                    "head": task["name"],
                    "policy": policy,
                    "iterations": int(policy_iterations[policy]),
                    "baseline_iterations": int(policy_iterations[BASELINE_POLICY]),
                    "metrics": metrics,
                    "baseline_metrics": baseline_metrics,
                    "relative_mae_harm": candidate_mae / baseline_mae - 1.0,
                }
            )
            paired.append(
                {
                    "year": year,
                    "horizon": horizon,
                    "policy": policy,
                    "metrics": pair_metrics,
                    "paired_daily": _file_record(pair_path),
                }
            )
            relationships.append(
                {
                    "year": year,
                    "horizon": horizon,
                    "policy": policy,
                    "summary": entry._summarize_daily_frame(
                        relation_frame, horizon=horizon
                    ),
                    "daily": _file_record(relation_path),
                }
            )
    evidence: dict[str, Any] = {}
    combined_p_values: dict[str, float] = {}
    for horizon in HORIZONS:
        for policy in FORMAL_POLICIES:
            current_paired = [
                row
                for row in paired
                if int(row["horizon"]) == horizon and row["policy"] == policy
            ]
            key = f"h{horizon}:{policy}"
            if len(current_paired) == len(MODEL_YEARS):
                combined = union._stouffer(current_paired)
                combined_p_values[key] = float(combined["p_value_one_sided"])
            else:
                combined = {
                    "z_statistic": math.nan,
                    "p_value_one_sided": math.nan,
                }
            evidence[key] = {
                **_head(study, horizon),
                "policy": policy,
                "combined_top5_test": combined,
            }
    q_values = feature._benjamini_hochberg(combined_p_values)
    rules = dict(study["decision"])
    for horizon in HORIZONS:
        for policy in FORMAL_POLICIES:
            key = f"h{horizon}:{policy}"
            current_paired = sorted(
                [
                    row
                    for row in paired
                    if int(row["horizon"]) == horizon and row["policy"] == policy
                ],
                key=lambda row: int(row["year"]),
            )
            current_annual = sorted(
                [
                    row
                    for row in annual
                    if int(row["horizon"]) == horizon and row["policy"] == policy
                ],
                key=lambda row: int(row["year"]),
            )
            current_relationships = sorted(
                [
                    row
                    for row in relationships
                    if int(row["horizon"]) == horizon and row["policy"] == policy
                ],
                key=lambda row: int(row["year"]),
            )
            unresolved = (
                policy == "final_head_prior_year" and unresolved_by_head[str(horizon)]
            )
            if len(current_paired) != len(MODEL_YEARS):
                evidence[key].update(
                    {
                        "iterations_2023_2024_2025": [],
                        "gate": {
                            "passed": False,
                            "checks": {"capacity_unresolved": bool(unresolved)},
                            "reason": "incomplete_candidate_predictions",
                        },
                        "path_guardrail": {
                            "rejected": False,
                            "status": "not_applicable",
                        },
                        "relationship": {
                            "material": True,
                            "status": "not_applicable",
                        },
                    }
                )
                continue
            guardrail = union._path_guardrail(current_paired, rules)
            gate = _candidate_gate(
                paired=current_paired,
                annual=current_annual,
                path_guardrail=guardrail,
                q_value=float(q_values.get(key, math.nan)),
                rules=rules,
                unresolved=unresolved,
            )
            evidence[key].update(
                {
                    "iterations_2023_2024_2025": [
                        int(row["iterations"]) for row in current_annual
                    ],
                    "baseline_iterations_2023_2024_2025": [
                        int(row["baseline_iterations"]) for row in current_annual
                    ],
                    "gate": gate,
                    "path_guardrail": guardrail,
                    "relationship": _contract_change(
                        relationships=current_relationships,
                        rules=rules,
                    ),
                }
            )
    final_policies: dict[str, Any] = {}
    material_horizons: list[int] = []
    for horizon in HORIZONS:
        passing = [
            record
            for key, record in evidence.items()
            if key.startswith(f"h{horizon}:") and record["gate"]["passed"]
        ]
        passing.sort(key=_winner_sort_key, reverse=True)
        if passing:
            winner = passing[0]
            policy = str(winner["policy"])
            change = dict(winner["relationship"])
            if change["material"]:
                material_horizons.append(horizon)
        else:
            policy = BASELINE_POLICY
            change = {
                "material": False,
                "materially_changed_years": [],
                "status": "baseline_retained",
            }
        final_policies[str(horizon)] = {
            "head": _head(study, horizon)["name"],
            "policy": policy,
            "adopted_challenger": policy != BASELINE_POLICY,
            "contract_change": change,
        }
    adopted_horizons = [
        int(horizon)
        for horizon, record in final_policies.items()
        if bool(record["adopted_challenger"])
    ]
    decision = {
        "status": "completed_final_head_capacity_audit",
        "final_capacity_policies": final_policies,
        "lightgbm_capacity_research_closed": True,
        "requires_entry_contract_v3": bool(material_horizons),
        "adopted_horizons": adopted_horizons,
        "v3_material_trigger_horizons": material_horizons,
        "v3_rebuild_horizons": (adopted_horizons if material_horizons else []),
        "entry_contract_status": (
            "v3_rebuild_required"
            if material_horizons
            else "retain_entry_contract_oos_v2"
        ),
        "post_entry_ab_status": (
            "v2_only_revalidation_required_before_future_holding_research"
            if material_horizons
            else "existing_v2_result_remains_aligned"
        ),
        "does_not_select": list(study["non_selections"]),
    }
    summary = {
        "schema": SUMMARY_SCHEMA,
        "status": decision["status"],
        "completed_at": _now(),
        "study_id": STUDY_ID,
        "scope": {
            "model_years": list(MODEL_YEARS),
            "horizons": list(HORIZONS),
            "formal_policy_count": len(FORMAL_POLICIES),
            "new_tuning_task_count": len(_tuning_tasks(study)),
            "long_outer_task_count": len(_outer_tasks(study)),
            "maximum_consumed_outcome_date": inputs.maximum_outcome_date,
            "forbidden_2026_row_count": 0,
        },
        "selection": selection,
        "annual": annual,
        "paired": paired,
        "relationships": relationships,
        "evidence": evidence,
        "decision": decision,
        "disclosure": {
            "fold_reuse": study["folds"]["decision_fold_role"],
            "multiple_testing": (
                "BH correction covers all six horizon-by-formal-policy tests."
            ),
            "top5": "Top 5% is a strong-candidate diagnostic, not a slot count.",
            "model_change": (
                "Only the capacity policy changed; features, objective, learning "
                "rate, tree shape, regularization, and folds remained fixed."
            ),
            "2026": "No 2026 row, outcome, prediction, or metric was read.",
        },
    }
    output_root.mkdir(parents=True, exist_ok=True)
    _write_audit_outputs(study=study, output_root=output_root, summary=summary)
    _emit(
        "final_head_capacity_evaluation_completed",
        final_policies={
            horizon: record["policy"] for horizon, record in final_policies.items()
        },
        v3_rebuild_horizons=material_horizons,
    )
    return summary


def _contract_tasks(
    study: Mapping[str, Any],
    decision: Mapping[str, Any],
) -> list[dict[str, Any]]:
    tasks: list[dict[str, Any]] = []
    for horizon in decision.get("v3_rebuild_horizons", []):
        policy = str(decision["final_capacity_policies"][str(horizon)]["policy"])
        if policy == BASELINE_POLICY:
            raise ValueError("v3 cannot rebuild a retained baseline head")
        for year in CONTRACT_REBUILD_YEARS:
            tasks.append(
                {
                    **_head(study, int(horizon)),
                    "year": int(year),
                    "policy": policy,
                    "stage": "contract_rebuild_outer",
                    "task_id": (f"mfe{int(horizon)}_{year}_{policy}_contract_outer"),
                }
            )
    return tasks


def _contract_iteration(
    *,
    study: Mapping[str, Any],
    feature_study: Mapping[str, Any],
    policy: str,
    horizon: int,
    year: int,
) -> tuple[int, dict[str, Any]]:
    if policy in ("fixed_256", "fixed_512"):
        value = int(study["policies"]["fixed_iterations"][policy])
        return value, {
            "source": "formal_fixed_capacity_policy",
            "policy": policy,
            "best_iteration": value,
        }
    if policy != "final_head_prior_year":
        raise ValueError(f"unknown adopted capacity policy: {policy}")
    record = _load_prior_curve_count(
        study=study,
        feature_study=feature_study,
        horizon=horizon,
        validation_year=year - 1,
    )
    if not bool(record["capacity_resolved"]):
        raise RuntimeError(
            f"contract final-head capacity is unresolved: D{horizon} {year}"
        )
    return int(record["best_iteration"]), record


def _contract_task_complete(
    path: Path,
    *,
    task: Mapping[str, Any],
    study_path: Path,
    feature_study: Mapping[str, Any],
    expected_iteration: int,
) -> bool:
    if not path.is_file():
        return False
    try:
        import lightgbm as lgb

        result = json.loads(path.read_text(encoding="utf-8"))
        expected = {
            "schema": CONTRACT_TASK_SCHEMA,
            "status": "completed",
            "study_id": STUDY_ID,
            "task_id": task["task_id"],
            "stage": task["stage"],
            "head": task["name"],
            "families": list(task["families"]),
            "model_year": int(task["year"]),
            "horizon": int(task["horizon"]),
            "policy": task["policy"],
            "best_iteration": int(expected_iteration),
            "config_sha256": _config_hash(study_path),
        }
        if any(result.get(key) != value for key, value in expected.items()):
            return False
        if dict(result.get("parameters", {}) or {}) != _parameters(feature_study):
            return False
        files = dict(result.get("files", {}) or {})
        if not files or not all(_record_exists(record) for record in files.values()):
            return False
        model = lgb.Booster(model_file=str(_resolve(files["model"]["path"])))
        if int(model.num_trees()) != int(expected_iteration):
            return False
        prediction = np.load(
            _resolve(files["candidate_prediction"]["path"]), allow_pickle=False
        )
        rows = np.load(_resolve(files["candidate_rows"]["path"]), allow_pickle=False)
        return (
            prediction.shape == rows.shape
            and tuple(prediction.shape) == tuple(files["candidate_prediction"]["shape"])
            and tuple(rows.shape) == tuple(files["candidate_rows"]["shape"])
        )
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


def _run_contract_task(
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

    horizon = int(task["horizon"])
    year = int(task["year"])
    iterations, capacity_source = _contract_iteration(
        study=study,
        feature_study=feature_study,
        policy=str(task["policy"]),
        horizon=horizon,
        year=year,
    )
    result_path = _task_result_path(output_root, task)
    if _contract_task_complete(
        result_path,
        task=task,
        study_path=study_path,
        feature_study=feature_study,
        expected_iteration=iterations,
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
        datasets, _weight, evaluation_raw = feature._build_datasets(
            study=feature_study,
            inputs=inputs,
            train_rows=fold.train_rows,
            evaluation_rows=fold.evaluation_rows,
            horizon=horizon,
            extra=extra,
            extra_names=extra.feature_names,
        )
        parameters = _parameters(feature_study)
        _emit(
            "contract_mfe_training_started",
            task_id=task["task_id"],
            iterations=iterations,
            train_rows=len(fold.train_rows),
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
        evaluation_prediction = feature._predict(
            model, datasets.evaluation_sequence, iterations
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
        candidate_rows = contract._year_rows(inputs, year)
        candidate_sequence = feature._combined_sequence(
            inputs=inputs,
            row_ids=candidate_rows,
            category_vocabularies=vocabularies,
            batch_size=int(feature_study["model"]["sequence_batch_size"]),
            extra=extra,
        )
        candidate_sequence = base._memory_trimmed_sequence(
            candidate_sequence, feature_study
        )
        candidate_prediction = feature._predict(model, candidate_sequence, iterations)
        output_dir = _task_dir(output_root, task)
        output_dir.mkdir(parents=True, exist_ok=True)
        model_path = output_dir / "model.txt"
        prediction_path = output_dir / "candidate_prediction.npy"
        rows_path = output_dir / "candidate_rows.npy"
        vocabulary_path = output_dir / "category_vocabularies.npz"
        daily_path = output_dir / "daily_metrics.parquet"
        model.save_model(str(model_path), num_iteration=iterations)
        _save_npy(prediction_path, candidate_prediction)
        _save_npy(rows_path, candidate_rows)
        vocabulary_record = contract._save_vocabularies(vocabulary_path, vocabularies)
        daily.to_parquet(daily_path, index=False, compression="zstd")
        result = {
            "schema": CONTRACT_TASK_SCHEMA,
            "status": "completed",
            "completed_at": _now(),
            "study_id": STUDY_ID,
            "task_id": task["task_id"],
            "stage": task["stage"],
            "head": task["name"],
            "families": list(task["families"]),
            "model_year": year,
            "horizon": horizon,
            "policy": task["policy"],
            "target": "mfe",
            "purge_days": horizon,
            "train_row_count": len(fold.train_rows),
            "evaluation_row_count": len(fold.evaluation_rows),
            "candidate_prediction_row_count": len(candidate_rows),
            "maximum_train_signal_date_idx": fold.maximum_train_signal_date_idx,
            "best_iteration": iterations,
            "capacity_source": capacity_source,
            "training_seconds": elapsed,
            "parameters": parameters,
            "feature_count": len(datasets.feature_names),
            "added_feature_count": len(extra.feature_names),
            "family_manifests": manifests,
            "metrics": metrics,
            "config_sha256": _config_hash(study_path),
            "files": {
                "model": _file_record(model_path),
                "candidate_prediction": _file_record(
                    prediction_path,
                    shape=list(candidate_prediction.shape),
                    dtype=str(candidate_prediction.dtype),
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
        _emit(
            "contract_mfe_training_completed",
            task_id=task["task_id"],
            iterations=iterations,
            seconds=elapsed,
        )
        del candidate_sequence, candidate_prediction, evaluation_prediction
        return result
    finally:
        if datasets is not None:
            feature._release_datasets(datasets)
        del model, fold, extra
        gc.collect()


def _verify_file_hash(record: Mapping[str, Any]) -> Path:
    path = _resolve(record["path"])
    if (
        not path.is_file()
        or path.stat().st_size != int(record["size"])
        or _sha256(path) != str(record["sha256"])
    ):
        raise ValueError(f"file record changed: {path}")
    return path


def _replace_contract_columns(
    *,
    old_raw: np.ndarray,
    old_rank: np.ndarray,
    replacements: Mapping[int, np.ndarray],
    date_idx: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, list[int]]:
    raw = np.asarray(old_raw, dtype=np.float32).copy()
    rank = np.asarray(old_rank, dtype=np.float32).copy()
    if raw.shape != rank.shape or raw.ndim != 2:
        raise ValueError("contract prediction matrices are not aligned")
    unchanged = [column for column in range(raw.shape[1]) if column not in replacements]
    for column, values in replacements.items():
        current = np.asarray(values, dtype=np.float32)
        if current.shape != (len(raw),):
            raise ValueError(f"replacement column {column} has the wrong shape")
        if not bool(np.isfinite(current).all()):
            raise ValueError(f"replacement column {column} is not finite")
        raw[:, int(column)] = current
        rank[:, int(column)] = contract._rank_by_date(current, date_idx)
    if not np.array_equal(
        raw[:, unchanged],
        np.asarray(old_raw)[:, unchanged],
        equal_nan=True,
    ):
        raise AssertionError("unchanged raw contract columns changed")
    if not np.array_equal(
        rank[:, unchanged],
        np.asarray(old_rank)[:, unchanged],
        equal_nan=True,
    ):
        raise AssertionError("unchanged rank contract columns changed")
    return raw, rank, unchanged


def _selected_candidate_year_values(
    *,
    study: Mapping[str, Any],
    output_root: Path,
    decision: Mapping[str, Any],
    inputs: base.LearnabilityInputs,
    horizon: int,
    year: int,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    policy = str(decision["final_capacity_policies"][str(horizon)]["policy"])
    if policy == BASELINE_POLICY:
        raise ValueError("baseline values should remain sourced from v2")
    if year in CONTRACT_REBUILD_YEARS:
        task = next(
            task
            for task in _contract_tasks(study, decision)
            if int(task["horizon"]) == horizon and int(task["year"]) == year
        )
        result = json.loads(
            _task_result_path(output_root, task).read_text(encoding="utf-8")
        )
        rows = np.load(
            _resolve(result["files"]["candidate_rows"]["path"]), allow_pickle=False
        )
        values = np.load(
            _resolve(result["files"]["candidate_prediction"]["path"]),
            allow_pickle=False,
        )
        return (
            np.asarray(rows, dtype=np.int64),
            np.asarray(values, dtype=np.float32),
            {
                "task_id": result["task_id"],
                "model_year": year,
                "policy": policy,
                "best_iteration": int(result["best_iteration"]),
                "result": _file_record(_task_result_path(output_root, task)),
            },
        )
    task = next(
        task
        for task in _outer_tasks(study)
        if int(task["horizon"]) == horizon and int(task["year"]) == year
    )
    result, _evaluation, candidate, _evaluation_rows, rows = _load_outer_arrays(
        output_root, task=task
    )
    position = ALL_POLICIES.index(policy)
    values = candidate[:, position]
    if not bool(np.isfinite(values).all()):
        raise ValueError(
            f"selected candidate predictions are missing: {task['task_id']}"
        )
    return (
        np.asarray(rows, dtype=np.int64),
        np.asarray(values, dtype=np.float32),
        {
            "task_id": result["task_id"],
            "model_year": year,
            "policy": policy,
            "best_iteration": int(result["policy_iterations"][policy]),
            "result": _file_record(_task_result_path(output_root, task)),
        },
    )


def materialize_contract_v3(
    *,
    study: Mapping[str, Any],
    output_root: Path,
    summary: dict[str, Any],
    inputs: base.LearnabilityInputs,
) -> dict[str, Any]:
    decision = dict(summary["decision"])
    if not bool(decision["requires_entry_contract_v3"]):
        raise ValueError("v3 materialization was not requested")
    v2_manifest_path = _resolve(study["sources"]["entry_contract_v2_manifest"])
    v2_manifest = json.loads(v2_manifest_path.read_text(encoding="utf-8"))
    if (
        v2_manifest.get("schema") != contract.MANIFEST_SCHEMA
        or v2_manifest.get("status")
        != "completed_candidate_aligned_strict_oos_contract"
    ):
        raise ValueError("entry contract v2 is not frozen")
    v2_files = dict(v2_manifest["files"])
    rows = np.load(_verify_file_hash(v2_files["candidate_rows"]), allow_pickle=False)
    old_raw = np.load(
        _verify_file_hash(v2_files["raw_predictions"]), allow_pickle=False
    )
    old_rank = np.load(
        _verify_file_hash(v2_files["date_rank_predictions"]), allow_pickle=False
    )
    rows = np.asarray(rows, dtype=np.int64)
    if tuple(v2_manifest["candidate_years"]) != CONTRACT_YEARS:
        raise ValueError("v2 contract years changed")
    if list(v2_manifest["physical_columns"]) != list(contract.PHYSICAL_COLUMNS):
        raise ValueError("v2 contract columns changed")
    replacements: dict[int, np.ndarray] = {}
    sources: dict[str, Any] = {}
    for horizon in decision["v3_rebuild_horizons"]:
        pieces: list[np.ndarray] = []
        row_pieces: list[np.ndarray] = []
        year_sources: list[dict[str, Any]] = []
        for year in CONTRACT_YEARS:
            current_rows, values, source_record = _selected_candidate_year_values(
                study=study,
                output_root=output_root,
                decision=decision,
                inputs=inputs,
                horizon=int(horizon),
                year=year,
            )
            expected_rows = contract._year_rows(inputs, year)
            if not np.array_equal(current_rows, expected_rows):
                raise ValueError(f"candidate rows changed for D{horizon} {year}")
            row_pieces.append(current_rows)
            pieces.append(values)
            year_sources.append(source_record)
        combined_rows = np.concatenate(row_pieces)
        if not np.array_equal(combined_rows, rows):
            raise ValueError(f"v3 rows differ from v2 for D{horizon}")
        column = 0 if int(horizon) == 10 else 1
        replacements[column] = np.concatenate(pieces).astype(np.float32, copy=False)
        sources[str(horizon)] = {
            "head": _head(study, int(horizon))["name"],
            "policy": decision["final_capacity_policies"][str(horizon)]["policy"],
            "years": year_sources,
        }
    dates = np.asarray(inputs.candidate_date_idx[rows], dtype=np.int32)
    raw, rank, unchanged = _replace_contract_columns(
        old_raw=old_raw,
        old_rank=old_rank,
        replacements=replacements,
        date_idx=dates,
    )
    v3_root = _resolve(study["contract_v3"]["output_root"])
    contract_dir = v3_root / "contract"
    rows_path = contract_dir / "candidate_rows.npy"
    raw_path = contract_dir / "raw_predictions.npy"
    rank_path = contract_dir / "date_rank_predictions.npy"
    _save_npy(rows_path, rows)
    _save_npy(raw_path, raw)
    _save_npy(rank_path, rank)
    manifest = {
        "schema": CONTRACT_V3_SCHEMA,
        "status": "completed_candidate_aligned_strict_oos_contract",
        "completed_at": _now(),
        "study_id": "seq100_entry_contract_oos_v3",
        "candidate_count": len(rows),
        "candidate_years": list(CONTRACT_YEARS),
        "physical_columns": list(contract.PHYSICAL_COLUMNS),
        "logical_coordinates": list(v2_manifest["logical_coordinates"]),
        "capacity_decision": decision["final_capacity_policies"],
        "rebuild_horizons": list(decision["v3_rebuild_horizons"]),
        "material_trigger_horizons": list(decision["v3_material_trigger_horizons"]),
        "mfe_sources": sources,
        "v2_source": {
            "manifest": _file_record(v2_manifest_path),
            "candidate_rows": v2_files["candidate_rows"],
            "raw_predictions": v2_files["raw_predictions"],
            "date_rank_predictions": v2_files["date_rank_predictions"],
        },
        "unchanged_columns": [
            contract.PHYSICAL_COLUMNS[column] for column in unchanged
        ],
        "unchanged_state_and_risk_verified": all(
            column in unchanged for column in (2, 3, 4, 5, 6)
        ),
        "post_entry_ab_alignment": {
            "status": "v2_only_revalidation_required_before_future_holding_research",
            "source_result": study["sources"]["post_entry_v1_result"],
            "retrained_in_this_study": False,
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
            "state_and_risk_booster_count": 0,
            "fusion": False,
        },
        "does_not_select": list(study["non_selections"]),
    }
    manifest_path = v3_root / "contract_manifest.json"
    _write_json(manifest_path, manifest)
    _write_json(
        v3_root / "decision.json",
        {
            "status": "entry_contract_v3_frozen",
            "capacity_policies": decision["final_capacity_policies"],
            "post_entry_ab_status": manifest["post_entry_ab_alignment"]["status"],
            "does_not_select": list(study["non_selections"]),
        },
    )
    record_root = _resolve(study["contract_v3"]["research_record_root"])
    _write_json(
        record_root / "config.json",
        {
            "study_id": "seq100_entry_contract_oos_v3",
            "source_capacity_study": STUDY_ID,
            "source_capacity_config": _file_record(DEFAULT_STUDY_PATH),
            "source_capacity_decision": _file_record(output_root / "decision.json"),
            "source_entry_contract_v2": _file_record(v2_manifest_path),
            "replace_only_adopted_mfe_columns": True,
            "reuse_v2_state_and_risk": True,
            "maximum_outcome_date": "2025-12-31",
            "forbidden_outcome_year": 2026,
        },
    )
    _write_json(
        record_root / "result.json",
        {**manifest, "full_output": _file_record(manifest_path)},
    )
    return manifest


def _finalize_contract_status(
    *,
    study: Mapping[str, Any],
    output_root: Path,
    summary: dict[str, Any],
    manifest: Mapping[str, Any] | None,
) -> dict[str, Any]:
    decision = dict(summary["decision"])
    if manifest is None:
        decision["entry_contract_v3_status"] = "not_required"
        decision["entry_contract_active"] = "seq100_entry_contract_oos_v2"
    else:
        v3_root = _resolve(study["contract_v3"]["output_root"])
        decision["entry_contract_v3_status"] = "completed"
        decision["entry_contract_active"] = "seq100_entry_contract_oos_v3"
        decision["entry_contract_v3_manifest"] = _file_record(
            v3_root / "contract_manifest.json"
        )
    summary["decision"] = decision
    _write_audit_outputs(study=study, output_root=output_root, summary=summary)
    return summary


def _load_existing_contract_v3(
    *,
    study: Mapping[str, Any],
    decision: Mapping[str, Any],
) -> dict[str, Any] | None:
    manifest_path = (
        _resolve(study["contract_v3"]["output_root"]) / "contract_manifest.json"
    )
    if not manifest_path.is_file():
        return None
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    expected = {
        "schema": CONTRACT_V3_SCHEMA,
        "status": "completed_candidate_aligned_strict_oos_contract",
        "study_id": "seq100_entry_contract_oos_v3",
        "capacity_decision": decision["final_capacity_policies"],
        "rebuild_horizons": decision["v3_rebuild_horizons"],
        "material_trigger_horizons": decision["v3_material_trigger_horizons"],
        "candidate_years": list(CONTRACT_YEARS),
        "physical_columns": list(contract.PHYSICAL_COLUMNS),
    }
    if any(manifest.get(key) != value for key, value in expected.items()):
        raise ValueError("existing entry contract v3 does not match this decision")
    files = dict(manifest.get("files", {}) or {})
    if not files:
        raise ValueError("existing entry contract v3 files are incomplete")
    for record in files.values():
        _verify_file_hash(record)
    if not bool(manifest.get("unchanged_state_and_risk_verified")):
        raise ValueError("existing entry contract v3 did not preserve state and risk")
    return manifest


def run_pending(
    *,
    study_path: Path = DEFAULT_STUDY_PATH,
    output_root: Path = DEFAULT_OUTPUT_ROOT,
) -> dict[str, Any]:
    study = load_study(study_path)
    feature_study = feature.load_study(
        _resolve(study["sources"]["feature_study_config"])
    )
    feature_output_root = _resolve(study["sources"]["feature_output_root"])
    inputs, _pack = feature._load_validated_inputs(feature_study)
    if inputs.maximum_outcome_date != "2025-12-31":
        raise ValueError("input outcome boundary changed")
    for task in _tuning_tasks(study):
        _run_tuning_task(
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
        feature_output_root=feature_output_root,
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
    summary = evaluate(study_path=study_path, output_root=output_root)
    if not bool(summary["decision"]["requires_entry_contract_v3"]):
        return _finalize_contract_status(
            study=study,
            output_root=output_root,
            summary=summary,
            manifest=None,
        )
    for task in _contract_tasks(study, summary["decision"]):
        _run_contract_task(
            study=study,
            study_path=study_path,
            feature_study=feature_study,
            feature_output_root=feature_output_root,
            output_root=output_root,
            inputs=inputs,
            task=task,
        )
    manifest = materialize_contract_v3(
        study=study,
        output_root=output_root,
        summary=summary,
        inputs=inputs,
    )
    return _finalize_contract_status(
        study=study,
        output_root=output_root,
        summary=summary,
        manifest=manifest,
    )


def status(
    *,
    study_path: Path = DEFAULT_STUDY_PATH,
    output_root: Path = DEFAULT_OUTPUT_ROOT,
) -> dict[str, Any]:
    study = load_study(study_path)
    feature_study = feature.load_study(
        _resolve(study["sources"]["feature_study_config"])
    )
    tuning_tasks = _tuning_tasks(study)
    tuning_completed = sum(
        _tuning_task_complete(
            _task_result_path(output_root, task),
            task=task,
            study_path=study_path,
            feature_study=feature_study,
        )
        for task in tuning_tasks
    )
    selection: dict[str, Any] | None = None
    selection_path = output_root / "capacity_selection.json"
    if selection_path.is_file():
        try:
            selection = _load_selection(output_root=output_root, study_path=study_path)
        except (KeyError, OSError, TypeError, ValueError, json.JSONDecodeError):
            selection = None
    outer_tasks = _outer_tasks(study)
    outer_completed = 0
    if selection is not None:
        outer_completed = sum(
            _outer_task_complete(
                _task_result_path(output_root, task),
                task=task,
                study_path=study_path,
                feature_study=feature_study,
                policy_iterations=_policy_iterations(
                    study=study,
                    selection=selection,
                    horizon=int(task["horizon"]),
                    year=int(task["year"]),
                ),
            )
            for task in outer_tasks
        )
    decision: dict[str, Any] | None = None
    decision_path = output_root / "decision.json"
    if decision_path.is_file():
        try:
            decision = json.loads(decision_path.read_text(encoding="utf-8"))
        except (OSError, ValueError, json.JSONDecodeError):
            decision = None
    contract_tasks: list[dict[str, Any]] = []
    contract_completed = 0
    if decision is not None and bool(decision.get("requires_entry_contract_v3")):
        contract_tasks = _contract_tasks(study, decision)
        for task in contract_tasks:
            iterations, _source = _contract_iteration(
                study=study,
                feature_study=feature_study,
                policy=str(task["policy"]),
                horizon=int(task["horizon"]),
                year=int(task["year"]),
            )
            contract_completed += int(
                _contract_task_complete(
                    _task_result_path(output_root, task),
                    task=task,
                    study_path=study_path,
                    feature_study=feature_study,
                    expected_iteration=iterations,
                )
            )
    v3_manifest = (
        _resolve(study["contract_v3"]["output_root"]) / "contract_manifest.json"
    )
    return {
        "study_id": STUDY_ID,
        "tuning_models": {
            "completed": int(tuning_completed),
            "total": len(tuning_tasks),
        },
        "capacity_selection_completed": selection is not None,
        "long_outer_models": {
            "completed": int(outer_completed),
            "total": len(outer_tasks),
        },
        "evaluation_completed": (output_root / "summary.json").is_file(),
        "contract_rebuild_models": {
            "completed": int(contract_completed),
            "total": len(contract_tasks),
        },
        "entry_contract_v3_completed": v3_manifest.is_file(),
    }


def self_test() -> dict[str, Any]:
    study = load_study()
    if not _tuning_resolved(recorded_rounds=300, best_iteration=200, patience=100):
        raise AssertionError("resolved tuning boundary failed")
    if _tuning_resolved(recorded_rounds=299, best_iteration=200, patience=100):
        raise AssertionError("unresolved tuning boundary failed")
    old = np.arange(28, dtype=np.float32).reshape(4, 7)
    rank = old / 28.0
    replacement = np.asarray([4.0, 3.0, 2.0, 1.0], dtype=np.float32)
    new_raw, new_rank, unchanged = _replace_contract_columns(
        old_raw=old,
        old_rank=rank,
        replacements={0: replacement},
        date_idx=np.asarray([1, 1, 2, 2], dtype=np.int32),
    )
    if not np.array_equal(new_raw[:, 0], replacement):
        raise AssertionError("contract replacement failed")
    if unchanged != [1, 2, 3, 4, 5, 6]:
        raise AssertionError("unchanged contract columns failed")
    if not np.array_equal(new_rank[:, 1:], rank[:, 1:]):
        raise AssertionError("unchanged rank columns failed")
    rules = study["decision"]
    paired = [
        {
            "metrics": {
                "top_5pct_lift": {"mean_delta": 0.001},
                "rank_ic": {"mean_delta": 0.001},
                "daily_tail_top5_lift": {"mean_delta": 0.001},
            }
        }
        for _year in MODEL_YEARS
    ]
    annual = [{"relative_mae_harm": 0.0} for _year in MODEL_YEARS]
    gate = _candidate_gate(
        paired=paired,
        annual=annual,
        path_guardrail={"rejected": False},
        q_value=0.05,
        rules=rules,
        unresolved=False,
    )
    if not gate["passed"]:
        raise AssertionError("candidate gate rejected a valid example")
    return {
        "status": "passed",
        "tuning_task_count": len(_tuning_tasks(study)),
        "outer_task_count": len(_outer_tasks(study)),
        "formal_policy_count": len(FORMAL_POLICIES),
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Close Seq100 MFE capacity research with final-head prior-year, "
            "fixed-256, and fixed-512 formal challengers."
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
        if bool(result["decision"]["requires_entry_contract_v3"]):
            study = load_study(study_path)
            manifest = _load_existing_contract_v3(
                study=study,
                decision=result["decision"],
            )
            if manifest is not None:
                result = _finalize_contract_status(
                    study=study,
                    output_root=output_root,
                    summary=result,
                    manifest=manifest,
                )
    else:
        result = self_test()
    print(json.dumps(result, ensure_ascii=False, indent=2, default=_json_default))


if __name__ == "__main__":
    main()
