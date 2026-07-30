from __future__ import annotations

import argparse
import gc
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
from scipy import stats

from daily_research.path_policy import seq100_path_label_learnability as base
from daily_research.path_policy import seq100_post_entry_ab_v2 as post

WORKSPACE_ROOT = Path(__file__).resolve().parents[2]
STUDY_ID = "seq100_post_entry_capacity_audit_v1"
DEFAULT_STUDY_PATH = (
    WORKSPACE_ROOT / "daily_research/studies/seq100_post_entry_capacity_audit_v1.json"
)
DEFAULT_OUTPUT_ROOT = (
    WORKSPACE_ROOT
    / "daily_research/output/path_policy/studies/seq100_post_entry_capacity_audit_v1"
)
DEFAULT_RECORD_ROOT = (
    WORKSPACE_ROOT
    / "daily_research/research_records/seq100/seq100_post_entry_capacity_audit_v1"
)
SELECTION_YEARS = (2021, 2022)
AGES = (1, 3, 5)
TARGETS = post.TARGETS
TASK_SCHEMA = "seq100_post_entry_capacity_task/v1"
SUMMARY_SCHEMA = "seq100_post_entry_capacity_summary/v1"


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
    post._save_npy(path, values)


def _file_record(path: Path, **extra: Any) -> dict[str, Any]:
    return post._file_record(path, **extra)


def _verify_record(record: Mapping[str, Any]) -> Path:
    return post._verify_record(record)


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
    return post._config_hash(path)


def load_study(path: Path = DEFAULT_STUDY_PATH) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("study_id") != STUDY_ID:
        raise ValueError(f"study_id must be {STUDY_ID}")
    folds = dict(payload["folds"])
    if tuple(int(value) for value in folds["selection_years"]) != SELECTION_YEARS:
        raise ValueError("capacity selection years changed")
    if folds["maximum_selection_outcome_date"] != "2022-12-31":
        raise ValueError("capacity audit may not read outcomes after 2022")
    if tuple(int(value) for value in payload["ages"]) != AGES:
        raise ValueError("capacity audit ages changed")
    if tuple(str(value) for value in payload["targets"]) != TARGETS:
        raise ValueError("capacity audit targets changed")
    if int(payload["reference_rounds"]) != 64:
        raise ValueError("capacity reference changed")
    if int(payload["model"]["long_booster_count"]) != 30:
        raise ValueError("capacity booster count changed")
    if str(payload["model"]["input_variant"]) != "A":
        raise ValueError("capacity audit must use A features only")
    expected = {
        "mfe_10": (64, 128, 256, 512),
        "mfe_20": (64, 128, 256, 512),
        "risk_10": (32, 64, 128, 256),
        "risk_20": (32, 64, 128, 256),
        "state_10": (32, 64, 128, 256),
    }
    current = {
        target: tuple(int(value) for value in payload["capacities"][target])
        for target in TARGETS
    }
    if current != expected:
        raise ValueError("candidate capacities changed")
    return payload


def _tasks(study: Mapping[str, Any]) -> list[dict[str, Any]]:
    return [
        {
            "task_id": f"{target}_D{age}_{year}_long_A",
            "target": target,
            "age": int(age),
            "selection_year": int(year),
            "variant": "A",
            "capacities": tuple(int(value) for value in study["capacities"][target]),
            "trained_rounds": max(int(value) for value in study["capacities"][target]),
        }
        for target in TARGETS
        for age in AGES
        for year in SELECTION_YEARS
    ]


def _task_dir(output_root: Path, task: Mapping[str, Any]) -> Path:
    return output_root / "tasks" / str(task["task_id"])


def _task_result_path(output_root: Path, task: Mapping[str, Any]) -> Path:
    return _task_dir(output_root, task) / "task_result.json"


def _task_complete(
    path: Path,
    *,
    task: Mapping[str, Any],
    study_path: Path,
    post_study: Mapping[str, Any],
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
            "target": task["target"],
            "age": int(task["age"]),
            "selection_year": int(task["selection_year"]),
            "variant": "A",
            "capacities": list(task["capacities"]),
            "trained_rounds": int(task["trained_rounds"]),
            "parameters": post._model_parameters(post_study, str(task["target"])),
            "config_sha256": _config_hash(study_path),
        }
        if any(result.get(key) != value for key, value in expected.items()):
            return False
        files = dict(result.get("files", {}) or {})
        if not files:
            return False
        for record in files.values():
            _verify_record(record)
        model = lgb.Booster(model_file=str(_resolve(files["model"]["path"])))
        multiplier = 3 if task["target"] == "state_10" else 1
        if int(model.num_trees()) != int(task["trained_rounds"]) * multiplier:
            return False
        prediction = np.load(_resolve(files["predictions"]["path"]), allow_pickle=False)
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


def _prediction_matrix(
    *,
    model: Any,
    sequence: Any,
    capacities: Sequence[int],
    state: bool,
) -> np.ndarray:
    predictions = [post._predict(model, sequence, int(rounds)) for rounds in capacities]
    if state:
        return np.stack(predictions, axis=1).astype(np.float32, copy=False)
    return np.column_stack(predictions).astype(np.float32, copy=False)


def _run_task(
    *,
    task: Mapping[str, Any],
    study: Mapping[str, Any],
    study_path: Path,
    post_study: Mapping[str, Any],
    output_root: Path,
    data: post.LandmarkData,
    date_values: np.ndarray,
) -> dict[str, Any]:
    import lightgbm as lgb

    result_path = _task_result_path(output_root, task)
    if _task_complete(
        result_path,
        task=task,
        study_path=study_path,
        post_study=post_study,
    ):
        return json.loads(result_path.read_text(encoding="utf-8"))
    target = str(task["target"])
    year = int(task["selection_year"])
    datasets = post._build_datasets(
        data=data,
        study=post_study,
        target=target,
        variant="A",
        evaluation_year=year,
        date_values=date_values,
        selection_outcome_cutoff=True,
    )
    model = None
    try:
        parameters = post._model_parameters(post_study, target)
        rounds = int(task["trained_rounds"])
        _emit(
            "capacity_training_started",
            task_id=task["task_id"],
            trained_rounds=rounds,
            train_rows=len(datasets.train_rows),
            evaluation_rows=len(datasets.evaluation_rows),
        )
        started = time.perf_counter()
        model = lgb.train(
            parameters,
            datasets.train_set,
            num_boost_round=rounds,
            valid_sets=[datasets.evaluation_set],
            valid_names=["capacity_evaluation"],
            callbacks=[lgb.log_evaluation(period=0)],
        )
        elapsed = float(time.perf_counter() - started)
        capacities = tuple(int(value) for value in task["capacities"])
        predictions = _prediction_matrix(
            model=model,
            sequence=datasets.evaluation_sequence,
            capacities=capacities,
            state=target == "state_10",
        )
        daily_frames: list[pd.DataFrame] = []
        metrics: dict[str, dict[str, Any]] = {}
        for position, capacity in enumerate(capacities):
            prediction = (
                predictions[:, position, :]
                if target == "state_10"
                else predictions[:, position]
            )
            daily = post._daily_metrics(
                data=data,
                target=target,
                rows=datasets.evaluation_rows,
                prediction=prediction,
            )
            daily.insert(0, "rounds", int(capacity))
            daily_frames.append(daily)
            metrics[str(capacity)] = {
                **post._daily_summary(daily.drop(columns=["rounds"])),
                "collapsed": (
                    post._state_collapsed(prediction) if target == "state_10" else False
                ),
            }
        daily_frame = pd.concat(daily_frames, ignore_index=True)
        output_dir = _task_dir(output_root, task)
        output_dir.mkdir(parents=True, exist_ok=True)
        model_path = output_dir / "model.txt"
        prediction_path = output_dir / "predictions.npy"
        rows_path = output_dir / "evaluation_rows.npy"
        daily_path = output_dir / "daily_metrics.parquet"
        model.save_model(str(model_path), num_iteration=rounds)
        _save_npy(prediction_path, predictions)
        _save_npy(rows_path, datasets.evaluation_rows)
        daily_frame.to_parquet(daily_path, index=False, compression="zstd")
        dependency = int(post_study["targets"][target]["dependency_days"])
        evaluation_dates = np.asarray(
            data.decision_date_idx[datasets.evaluation_rows], dtype=np.int32
        )
        maximum_outcome_idx = int(evaluation_dates.max()) + dependency
        maximum_outcome_date = str(date_values[maximum_outcome_idx])
        if maximum_outcome_date > f"{year}-12-31":
            raise AssertionError(
                "capacity task read an outcome after its selection year"
            )
        result = {
            "schema": TASK_SCHEMA,
            "status": "completed",
            "completed_at": _now(),
            "study_id": STUDY_ID,
            "task_id": task["task_id"],
            "target": target,
            "target_kind": post._target_kind(target),
            "horizon": post._target_horizon(target),
            "age": int(task["age"]),
            "selection_year": year,
            "variant": "A",
            "purge_days": dependency,
            "train_row_count": len(datasets.train_rows),
            "evaluation_row_count": len(datasets.evaluation_rows),
            "maximum_train_decision_date_idx": int(
                data.decision_date_idx[datasets.train_rows].max()
            ),
            "maximum_train_decision_date": str(
                date_values[int(data.decision_date_idx[datasets.train_rows].max())]
            ),
            "maximum_evaluation_outcome_date": maximum_outcome_date,
            "capacities": list(capacities),
            "trained_rounds": rounds,
            "internal_tree_count": int(model.num_trees()),
            "training_seconds": elapsed,
            "parameters": parameters,
            "feature_names": list(datasets.feature_names),
            "metrics": metrics,
            "common_support_sha256": post._sha256(rows_path),
            "config_sha256": _config_hash(study_path),
            "post_entry_config_sha256": post._config_hash(
                _resolve(study["sources"]["post_entry_config"])
            ),
            "files": {
                "model": _file_record(model_path),
                "predictions": _file_record(
                    prediction_path,
                    shape=list(predictions.shape),
                    dtype=str(predictions.dtype),
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
            "capacity_training_completed",
            task_id=task["task_id"],
            trained_rounds=rounds,
            seconds=elapsed,
        )
        del predictions, daily_frame
        return result
    finally:
        post._release_datasets(datasets)
        del model
        gc.collect()


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


def _stouffer(records: Sequence[Mapping[str, Any]]) -> dict[str, float]:
    z_values: list[float] = []
    weights: list[float] = []
    for record in records:
        hac = dict(record["primary_delta_hac"])
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


def _load_result(
    *,
    task: Mapping[str, Any],
    study_path: Path,
    post_study: Mapping[str, Any],
    output_root: Path,
) -> dict[str, Any]:
    path = _task_result_path(output_root, task)
    if not _task_complete(
        path,
        task=task,
        study_path=study_path,
        post_study=post_study,
    ):
        raise ValueError(f"capacity task is incomplete: {task['task_id']}")
    return json.loads(path.read_text(encoding="utf-8"))


def _unit_record(
    *,
    task: Mapping[str, Any],
    result: Mapping[str, Any],
    rounds: int,
    reference: int,
) -> dict[str, Any]:
    daily = pd.read_parquet(_resolve(result["files"]["daily_metrics"]["path"]))
    candidate = daily.loc[daily["rounds"] == int(rounds)].sort_values("date_idx")
    baseline = daily.loc[daily["rounds"] == int(reference)].sort_values("date_idx")
    if not np.array_equal(
        candidate["date_idx"].to_numpy(), baseline["date_idx"].to_numpy()
    ):
        raise ValueError(f"capacity common support changed: {task['task_id']}")
    paired = baseline.merge(
        candidate,
        on="date_idx",
        suffixes=("_reference", "_candidate"),
        validate="one_to_one",
    )
    target = str(task["target"])
    kind = post._target_kind(target)
    primary = "ordinal_ic" if kind == "state" else "rank_ic"
    primary_delta = (
        paired[f"{primary}_candidate"] - paired[f"{primary}_reference"]
    ).to_numpy(dtype=np.float64)
    record: dict[str, Any] = {
        "target": target,
        "age": int(task["age"]),
        "selection_year": int(task["selection_year"]),
        "rounds": int(rounds),
        "reference_rounds": int(reference),
        "metrics": dict(result["metrics"][str(rounds)]),
        "reference_metrics": dict(result["metrics"][str(reference)]),
        "primary_delta_hac": base._hac_mean_test(
            primary_delta,
            maximum_lag=max(post._target_horizon(target) - 1, 0),
        ),
    }
    if kind == "mfe":
        adverse_delta = (
            paired["top5_pre_peak_mae_candidate"]
            - paired["top5_pre_peak_mae_reference"]
        ).to_numpy(dtype=np.float64)
        endpoint_delta = (
            paired["top5_endpoint_return_candidate"]
            - paired["top5_endpoint_return_reference"]
        ).to_numpy(dtype=np.float64)
        adverse_hac = base._hac_mean_test(
            adverse_delta, maximum_lag=max(post._target_horizon(target) - 1, 0)
        )
        endpoint_hac = base._hac_mean_test(
            endpoint_delta, maximum_lag=max(post._target_horizon(target) - 1, 0)
        )
        record.update(
            {
                "rank_ic_delta": float(np.nanmean(primary_delta)),
                "relative_mae_harm": _relative_harm(
                    float(np.nanmean(paired["mae_candidate"])),
                    float(np.nanmean(paired["mae_reference"])),
                ),
                "top5_remaining_mfe_delta": float(
                    np.nanmean(
                        paired["top5_remaining_mfe_candidate"]
                        - paired["top5_remaining_mfe_reference"]
                    )
                ),
                "top5_pre_peak_mae_delta": float(np.nanmean(adverse_delta)),
                "top5_endpoint_return_delta": float(np.nanmean(endpoint_delta)),
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
        record.update(
            {
                "rank_ic_delta": float(np.nanmean(primary_delta)),
                "relative_mae_harm": _relative_harm(
                    float(np.nanmean(paired["mae_candidate"])),
                    float(np.nanmean(paired["mae_reference"])),
                ),
                "deep_adverse_pr_auc_delta": float(
                    np.nanmean(
                        paired["deep_adverse_pr_auc_candidate"]
                        - paired["deep_adverse_pr_auc_reference"]
                    )
                ),
            }
        )
    else:
        record.update(
            {
                "ordinal_ic_delta": float(np.nanmean(primary_delta)),
                "relative_brier_harm": _relative_harm(
                    float(np.nanmean(paired["brier_candidate"])),
                    float(np.nanmean(paired["brier_reference"])),
                ),
                "relative_logloss_harm": _relative_harm(
                    float(np.nanmean(paired["logloss_candidate"])),
                    float(np.nanmean(paired["logloss_reference"])),
                ),
                "collapsed": bool(result["metrics"][str(rounds)]["collapsed"]),
            }
        )
    return record


def _candidate_gate(
    *,
    target: str,
    rounds: int,
    units: Sequence[Mapping[str, Any]],
    study: Mapping[str, Any],
) -> dict[str, Any]:
    kind = post._target_kind(target)
    if int(rounds) == int(study["reference_rounds"]):
        return {
            "passed": True,
            "reference_fallback": True,
            "checks": {"reference_64_always_legal": True},
        }
    if len(units) != 6:
        raise ValueError(f"capacity gate requires six units: {target} {rounds}")
    rules = study["gate"][kind]
    if kind == "mfe":
        rank = [float(row["rank_ic_delta"]) for row in units]
        mae = [float(row["relative_mae_harm"]) for row in units]
        harm_units = [row for row in units if bool(row["significant_path_harm"])]
        worst_path = [
            min(
                float(row["top5_pre_peak_mae_delta"]),
                float(row["top5_endpoint_return_delta"]),
            )
            for row in units
        ]
        path_rejected = bool(
            len(harm_units) >= int(rules["path_guardrail_minimum_harm_units"])
            and float(np.median(worst_path))
            <= -float(rules["path_guardrail_minimum_median_absolute_harm"])
        )
        checks = {
            "worst_rank_ic_delta": min(rank),
            "median_relative_mae_harm": float(np.median(mae)),
            "worst_relative_mae_harm": max(mae),
            "significant_path_harm_units": len(harm_units),
            "median_worst_path_delta": float(np.median(worst_path)),
            "path_guardrail_rejected": path_rejected,
        }
        passed = bool(
            checks["worst_rank_ic_delta"]
            >= float(rules["minimum_worst_rank_ic_delta_vs_64"])
            and checks["median_relative_mae_harm"]
            <= float(rules["maximum_median_relative_mae_harm"])
            and checks["worst_relative_mae_harm"]
            <= float(rules["maximum_worst_relative_mae_harm"])
            and not path_rejected
        )
    elif kind == "risk":
        rank = [float(row["rank_ic_delta"]) for row in units]
        pr = [float(row["deep_adverse_pr_auc_delta"]) for row in units]
        mae = [float(row["relative_mae_harm"]) for row in units]
        checks = {
            "worst_rank_ic_delta": min(rank),
            "worst_deep_adverse_pr_auc_delta": min(pr),
            "median_relative_mae_harm": float(np.median(mae)),
            "worst_relative_mae_harm": max(mae),
        }
        passed = bool(
            checks["worst_rank_ic_delta"]
            >= float(rules["minimum_worst_rank_ic_delta_vs_64"])
            and checks["worst_deep_adverse_pr_auc_delta"]
            >= float(rules["minimum_worst_deep_adverse_pr_auc_delta_vs_64"])
            and checks["median_relative_mae_harm"]
            <= float(rules["maximum_median_relative_mae_harm"])
            and checks["worst_relative_mae_harm"]
            <= float(rules["maximum_worst_relative_mae_harm"])
        )
    else:
        ordinal = [float(row["ordinal_ic_delta"]) for row in units]
        brier = [float(row["relative_brier_harm"]) for row in units]
        logloss = [float(row["relative_logloss_harm"]) for row in units]
        collapsed = [row for row in units if bool(row["collapsed"])]
        checks = {
            "worst_ordinal_ic_delta": min(ordinal),
            "median_relative_brier_harm": float(np.median(brier)),
            "worst_relative_brier_harm": max(brier),
            "median_relative_logloss_harm": float(np.median(logloss)),
            "worst_relative_logloss_harm": max(logloss),
            "collapsed_unit_count": len(collapsed),
        }
        passed = bool(
            checks["worst_ordinal_ic_delta"]
            >= float(rules["minimum_worst_ordinal_ic_delta_vs_64"])
            and checks["median_relative_brier_harm"]
            <= float(rules["maximum_median_relative_brier_harm"])
            and checks["worst_relative_brier_harm"]
            <= float(rules["maximum_worst_relative_brier_harm"])
            and checks["median_relative_logloss_harm"]
            <= float(rules["maximum_median_relative_logloss_harm"])
            and checks["worst_relative_logloss_harm"]
            <= float(rules["maximum_worst_relative_logloss_harm"])
            and not collapsed
        )
    return {"passed": passed, "reference_fallback": False, "checks": checks}


def _selection_vector(
    *,
    target: str,
    rounds: int,
    units: Sequence[Mapping[str, Any]],
) -> tuple[float, ...]:
    kind = post._target_kind(target)
    metrics = [dict(row["metrics"]) for row in units]
    if kind == "mfe":
        top5 = [float(row["top5_remaining_mfe"]) for row in metrics]
        rank = [float(row["rank_ic"]) for row in metrics]
        return (
            float(np.median(top5)),
            min(top5),
            float(np.median(rank)),
            min(rank),
            -float(rounds),
        )
    if kind == "risk":
        rank = [float(row["rank_ic"]) for row in metrics]
        pr = [float(row["deep_adverse_pr_auc"]) for row in metrics]
        mae = [float(row["mae"]) for row in metrics]
        return (
            float(np.median(rank)),
            min(rank),
            float(np.median(pr)),
            -float(np.median(mae)),
            -float(rounds),
        )
    ordinal = [float(row["ordinal_ic"]) for row in metrics]
    brier = [float(row["brier"]) for row in metrics]
    logloss = [float(row["logloss"]) for row in metrics]
    return (
        float(np.median(ordinal)),
        min(ordinal),
        -float(np.median(brier)),
        -float(np.median(logloss)),
        -float(rounds),
    )


def _is_better(left: Sequence[float], right: Sequence[float], tolerance: float) -> bool:
    for current, incumbent in zip(left, right, strict=True):
        if float(current) > float(incumbent) + float(tolerance):
            return True
        if float(current) < float(incumbent) - float(tolerance):
            return False
    return False


def evaluate(
    *,
    study_path: Path = DEFAULT_STUDY_PATH,
    output_root: Path = DEFAULT_OUTPUT_ROOT,
) -> dict[str, Any]:
    study = load_study(study_path)
    post_study = post.load_study(_resolve(study["sources"]["post_entry_config"]))
    for task in _tasks(study):
        _load_result(
            task=task,
            study_path=study_path,
            post_study=post_study,
            output_root=output_root,
        )
    target_evidence: dict[str, Any] = {}
    selected: dict[str, int] = {}
    reference = int(study["reference_rounds"])
    tolerance = float(study["model"]["numerical_tie_tolerance"])
    for target in TARGETS:
        candidates: dict[str, Any] = {}
        for rounds in study["capacities"][target]:
            units: list[dict[str, Any]] = []
            for task in _tasks(study):
                if task["target"] != target:
                    continue
                result = _load_result(
                    task=task,
                    study_path=study_path,
                    post_study=post_study,
                    output_root=output_root,
                )
                units.append(
                    _unit_record(
                        task=task,
                        result=result,
                        rounds=int(rounds),
                        reference=reference,
                    )
                )
            units = sorted(
                units,
                key=lambda row: (int(row["age"]), int(row["selection_year"])),
            )
            gate = _candidate_gate(
                target=target,
                rounds=int(rounds),
                units=units,
                study=study,
            )
            candidates[str(rounds)] = {
                "rounds": int(rounds),
                "gate": gate,
                "units": units,
                "selection_vector": list(
                    _selection_vector(target=target, rounds=int(rounds), units=units)
                ),
                "uncertainty_vs_64": _stouffer(units),
            }
        legal = [
            record for record in candidates.values() if bool(record["gate"]["passed"])
        ]
        if not legal:
            raise AssertionError(f"64-round fallback was not legal: {target}")
        winner = legal[0]
        for candidate in legal[1:]:
            if _is_better(
                candidate["selection_vector"],
                winner["selection_vector"],
                tolerance,
            ):
                winner = candidate
        selected[target] = int(winner["rounds"])
        target_evidence[target] = {
            "selected_rounds": int(winner["rounds"]),
            "selection_vector": list(winner["selection_vector"]),
            "candidate_evidence": candidates,
            "winner_order": list(study["winner_order"][post._target_kind(target)]),
            "capacity_uniform_across_ages": True,
        }
    decision = {
        "status": "post_entry_target_capacities_frozen",
        "study_id": STUDY_ID,
        "selected_rounds": selected,
        "selection_years": list(SELECTION_YEARS),
        "maximum_selection_outcome_date": "2022-12-31",
        "input_variant": "A",
        "uniform_across_D1_D3_D5": True,
        "next_step": "run_seq100_post_entry_ab_v2",
        "does_not_select": list(study["non_selections"]),
    }
    summary = {
        "schema": SUMMARY_SCHEMA,
        "status": "completed_post_entry_capacity_audit",
        "completed_at": _now(),
        "study_id": STUDY_ID,
        "config_sha256": _config_hash(study_path),
        "targets": target_evidence,
        "decision": decision,
        "scope": {
            "long_booster_count": 30,
            "maximum_selection_outcome_date": "2022-12-31",
            "forbidden_2023_2026_selection_row_count": 0,
            "B_feature_model_count": 0,
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
    post_path = _resolve(study["sources"]["post_entry_config"])
    post_study = post.load_study(post_path)
    post_root = _resolve(study["sources"]["post_entry_output_root"])
    post.prepare_landmarks(study_path=post_path, output_root=post_root)
    _feature_study, inputs, _pack, contract, _reader = post._load_runtime(post_study)
    study_hash = post._config_hash(post_path)
    landmarks = {
        age: post.load_landmark(
            study=post_study,
            study_hash=study_hash,
            contract=contract,
            output_root=post_root,
            age=age,
        )
        for age in AGES
    }
    for task in _tasks(study):
        _run_task(
            task=task,
            study=study,
            study_path=study_path,
            post_study=post_study,
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
    post_study = post.load_study(_resolve(study["sources"]["post_entry_config"]))
    completed: list[str] = []
    pending: list[str] = []
    for task in _tasks(study):
        current = _task_complete(
            _task_result_path(output_root, task),
            task=task,
            study_path=study_path,
            post_study=post_study,
        )
        (completed if current else pending).append(str(task["task_id"]))
    return {
        "study_id": STUDY_ID,
        "task_count": len(completed) + len(pending),
        "completed": completed,
        "pending": pending,
        "evaluation_completed": (output_root / "summary.json").is_file(),
        "decision_completed": (output_root / "decision.json").is_file(),
    }


def self_test() -> dict[str, Any]:
    study = load_study()
    tasks = _tasks(study)
    if len(tasks) != 30:
        raise AssertionError("capacity task count changed")
    if any(task["variant"] != "A" for task in tasks):
        raise AssertionError("capacity audit includes B")
    if any(int(task["selection_year"]) not in SELECTION_YEARS for task in tasks):
        raise AssertionError("capacity audit includes a forbidden year")
    if not _is_better((1.0, 0.0), (0.9, 100.0), 1.0e-6):
        raise AssertionError("lexicographic role priority changed")
    if _is_better((1.0, 0.0), (1.0 + 0.5e-6, 1.0), 1.0e-6):
        raise AssertionError("numerical tie tolerance changed")
    return {
        "status": "passed",
        "task_count": len(tasks),
        "selection_years": list(SELECTION_YEARS),
        "uses_B": False,
        "maximum_selection_outcome_date": "2022-12-31",
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Select uniform target-specific Seq100 post-entry capacities."
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
