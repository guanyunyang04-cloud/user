from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats

from daily_research.path_policy import seq100_entry_role_synthesis as entry
from daily_research.path_policy import seq100_mfe_feature_family_audit as source
from daily_research.path_policy import seq100_mfe_objective_alignment as objective
from daily_research.path_policy import seq100_path_label_learnability as base

WORKSPACE_ROOT = source.WORKSPACE_ROOT
STUDY_ID = "seq100_mfe_feature_union_audit_v1"
DEFAULT_STUDY_PATH = (
    WORKSPACE_ROOT / "daily_research/studies/seq100_mfe_feature_union_audit_v1.json"
)
DEFAULT_OUTPUT_ROOT = (
    WORKSPACE_ROOT
    / "daily_research/output/path_policy/studies/seq100_mfe_feature_union_audit_v1"
)
DEFAULT_RECORD_ROOT = (
    WORKSPACE_ROOT
    / "daily_research/research_records/seq100/seq100_mfe_feature_union_audit_v1"
)
TASK_SCHEMA = "seq100_mfe_feature_union_task/v1"
SUMMARY_SCHEMA = "seq100_mfe_feature_union_summary/v1"
FOLD_YEARS = (2023, 2024, 2025)
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
        raise ValueError("feature-union study id changed")
    folds = dict(payload.get("folds", {}) or {})
    if tuple(int(value) for value in folds.get("fold_years", [])) != FOLD_YEARS:
        raise ValueError("feature-union folds changed")
    if folds.get("maximum_outcome_date") != "2025-12-31":
        raise ValueError("feature-union outcome boundary changed")
    if int(folds.get("forbidden_outcome_year", -1)) != 2026:
        raise ValueError("2026 must remain forbidden")
    heads = dict(payload.get("heads", {}) or {})
    if tuple(sorted(int(value) for value in heads)) != HORIZONS:
        raise ValueError("feature-union horizons changed")
    catalog = source.feature_catalog()
    for horizon in HORIZONS:
        head = dict(heads[str(horizon)])
        incumbent = str(head["incumbent"])
        for stage in ("stage1", "stage2"):
            variants = dict(head[stage])
            for variant, families in variants.items():
                names = tuple(str(value) for value in families)
                if not names or names[0] != incumbent:
                    raise ValueError(f"{variant} does not extend its incumbent")
                if len(names) != len(set(names)):
                    raise ValueError(f"{variant} repeats a feature family")
                if any(name not in catalog for name in names):
                    raise ValueError(f"{variant} contains an unknown feature family")
    return payload


def _variant_definitions(study: Mapping[str, Any]) -> dict[int, dict[str, Any]]:
    definitions: dict[int, dict[str, Any]] = {}
    for horizon in HORIZONS:
        head = dict(study["heads"][str(horizon)])
        current: dict[str, Any] = {}
        for stage in ("stage1", "stage2"):
            for variant, families in dict(head[stage]).items():
                current[str(variant)] = {
                    "variant": str(variant),
                    "horizon": int(horizon),
                    "stage": stage,
                    "families": tuple(str(value) for value in families),
                    "control": "deterministic_noise" in families,
                }
        definitions[horizon] = current
    return definitions


def _tasks(
    study: Mapping[str, Any], *, stage: str | None = None
) -> list[dict[str, Any]]:
    tasks: list[dict[str, Any]] = []
    for horizon, variants in _variant_definitions(study).items():
        for variant, definition in variants.items():
            if stage is not None and definition["stage"] != stage:
                continue
            for year in FOLD_YEARS:
                tasks.append(
                    {
                        **definition,
                        "year": int(year),
                        "task_id": f"mfe{horizon}_{year}_{variant}_outer",
                    }
                )
    return tasks


@dataclass
class UnionExtra:
    arrays: tuple[np.memmap, ...]
    feature_names: tuple[str, ...]

    def __getitem__(self, key: Any) -> np.ndarray:
        if not isinstance(key, tuple) or len(key) != 2:
            raise TypeError("union extra requires feature and row selectors")
        feature_selector, row_selector = key
        if feature_selector != slice(None):
            raise IndexError("union extra only supports the complete feature axis")
        parts = [
            np.asarray(values[:, row_selector], dtype=np.float32)
            for values in self.arrays
        ]
        return np.concatenate(parts, axis=0)


def _open_union(
    *,
    feature_output_root: Path,
    families: Sequence[str],
    candidate_count: int,
) -> tuple[UnionExtra, list[dict[str, Any]]]:
    arrays: list[np.memmap] = []
    names: list[str] = []
    manifests: list[dict[str, Any]] = []
    for family in families:
        prepared = source.load_prepared_family(
            feature_output_root,
            str(family),
            candidate_count=int(candidate_count),
        )
        arrays.append(prepared.open())
        names.extend(prepared.feature_names)
        manifest_path = source._family_manifest_path(feature_output_root, str(family))
        manifests.append(
            {
                "family": str(family),
                "manifest": _file_record(manifest_path),
                "feature_count": len(prepared.feature_names),
            }
        )
    if len(names) != len(set(names)):
        raise ValueError("union feature names are not unique")
    return UnionExtra(tuple(arrays), tuple(names)), manifests


def _task_dir(output_root: Path, task: Mapping[str, Any]) -> Path:
    return (
        output_root
        / "outer"
        / f"fold_{int(task['year'])}"
        / f"h{int(task['horizon']):02d}"
        / str(task["variant"])
    )


def _task_result_path(output_root: Path, task: Mapping[str, Any]) -> Path:
    return _task_dir(output_root, task) / "task_result.json"


def _task_complete(
    path: Path,
    *,
    task: Mapping[str, Any],
    study: Mapping[str, Any],
    feature_study: Mapping[str, Any],
    feature_output_root: Path,
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
            "variant": task["variant"],
            "fold_year": int(task["year"]),
            "horizon": int(task["horizon"]),
            "stage": task["stage"],
            "families": list(task["families"]),
        }
        if any(result.get(key) != value for key, value in expected.items()):
            return False
        parameters, _rounds, _patience = source._model_parameters(feature_study)
        if dict(result.get("parameters", {}) or {}) != parameters:
            return False
        tuning_task = source._task_by_id(
            f"mfe{int(task['horizon'])}_{int(task['year'])}_baseline_tuning"
        )
        tuning_path = source._task_result_path(feature_output_root, tuning_task)
        if not source._task_complete(
            tuning_path,
            task=tuning_task,
            study=feature_study,
            output_root=feature_output_root,
        ):
            return False
        tuning = json.loads(tuning_path.read_text(encoding="utf-8"))
        if int(result.get("best_iteration", -1)) != int(tuning["best_iteration"]):
            return False
        files = dict(result.get("files", {}) or {})
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
    study: Mapping[str, Any],
    feature_study: Mapping[str, Any],
    feature_output_root: Path,
    output_root: Path,
    inputs: base.LearnabilityInputs,
    task: Mapping[str, Any],
) -> dict[str, Any]:
    import lightgbm as lgb

    result_path = _task_result_path(output_root, task)
    if _task_complete(
        result_path,
        task=task,
        study=study,
        feature_study=feature_study,
        feature_output_root=feature_output_root,
    ):
        return json.loads(result_path.read_text(encoding="utf-8"))
    horizon = int(task["horizon"])
    year = int(task["year"])
    tuning_task = source._task_by_id(f"mfe{horizon}_{year}_baseline_tuning")
    tuning_path = source._task_result_path(feature_output_root, tuning_task)
    if not source._task_complete(
        tuning_path,
        task=tuning_task,
        study=feature_study,
        output_root=feature_output_root,
    ):
        raise RuntimeError(f"incumbent iteration source is incomplete: {tuning_path}")
    tuning = json.loads(tuning_path.read_text(encoding="utf-8"))
    iterations = int(tuning["best_iteration"])
    extra, manifests = _open_union(
        feature_output_root=feature_output_root,
        families=task["families"],
        candidate_count=inputs.candidate_count,
    )
    fold = inputs.common_path_rows(year, horizon)
    datasets, _evaluation_weight, evaluation_raw = source._build_datasets(
        study=feature_study,
        inputs=inputs,
        train_rows=fold.train_rows,
        evaluation_rows=fold.evaluation_rows,
        horizon=horizon,
        extra=extra,
        extra_names=extra.feature_names,
    )
    parameters, _maximum_rounds, _patience = source._model_parameters(feature_study)
    _emit(
        "task_training_started",
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
    )
    elapsed = float(time.perf_counter() - started)
    output_dir = _task_dir(output_root, task)
    output_dir.mkdir(parents=True, exist_ok=True)
    model_path = output_dir / "model.txt"
    model.save_model(str(model_path), num_iteration=iterations)
    prediction = source._predict(model, datasets.evaluation_sequence, iterations)
    prediction_path = output_dir / "prediction.npy"
    rows_path = output_dir / "evaluation_rows.npy"
    _save_npy(prediction_path, prediction)
    _save_npy(rows_path, datasets.evaluation_rows)
    daily, metrics = base.regression_metrics(
        date_idx=inputs.candidate_date_idx[datasets.evaluation_rows],
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
        "stage": task["stage"],
        "variant": task["variant"],
        "families": list(task["families"]),
        "control": bool(task["control"]),
        "fold_year": year,
        "horizon": horizon,
        "target": "mfe",
        "purge_days": horizon,
        "train_row_count": len(fold.train_rows),
        "evaluation_row_count": len(fold.evaluation_rows),
        "best_iteration": iterations,
        "fixed_iteration_source": {
            "task_id": tuning_task["task_id"],
            "inner_validation_year": int(tuning["inner_validation_year"]),
            "best_iteration": iterations,
        },
        "training_seconds": elapsed,
        "parameters": parameters,
        "feature_count": len(datasets.feature_names),
        "added_feature_count": len(extra.feature_names),
        "family_manifests": manifests,
        "metrics": metrics,
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
    source._release_datasets(datasets)
    del model, prediction, fold, extra
    _emit("task_training_completed", task_id=task["task_id"], seconds=elapsed)
    return result


def _load_union_result(
    output_root: Path,
    *,
    task: Mapping[str, Any],
) -> tuple[dict[str, Any], np.ndarray, np.ndarray]:
    result_path = _task_result_path(output_root, task)
    result = json.loads(result_path.read_text(encoding="utf-8"))
    files = dict(result["files"])
    return (
        result,
        np.load(_resolve(files["prediction"]["path"]), allow_pickle=False),
        np.load(_resolve(files["evaluation_rows"]["path"]), allow_pickle=False),
    )


def _one_sided_from_hac(record: Mapping[str, Any]) -> float:
    mean = float(record["mean"])
    two_sided = float(record["p_value_two_sided"])
    if not math.isfinite(mean) or not math.isfinite(two_sided):
        return math.nan
    return float(two_sided / 2.0 if mean >= 0.0 else 1.0 - two_sided / 2.0)


def _stouffer(records: Sequence[Mapping[str, Any]]) -> dict[str, float]:
    z_values: list[float] = []
    weights: list[float] = []
    for record in records:
        hac = dict(record["metrics"]["top_5pct_lift"]["hac"])
        p_value = _one_sided_from_hac(hac)
        count = int(hac["count"])
        if math.isfinite(p_value) and count > 0:
            z_values.append(
                float(stats.norm.isf(np.clip(p_value, 1.0e-15, 1.0 - 1.0e-15)))
            )
            weights.append(math.sqrt(float(count)))
    if not z_values:
        return {"z_statistic": math.nan, "p_value_one_sided": math.nan}
    weight = np.asarray(weights, dtype=np.float64)
    statistic = float(
        np.dot(weight, np.asarray(z_values, dtype=np.float64))
        / np.sqrt(np.dot(weight, weight))
    )
    return {
        "z_statistic": statistic,
        "p_value_one_sided": float(stats.norm.sf(statistic)),
    }


def _role_gate(
    *,
    annual: Sequence[Mapping[str, Any]],
    relationships: Sequence[Mapping[str, Any]],
    rules: Mapping[str, Any],
) -> dict[str, Any]:
    top5 = [float(row["metrics"]["top_5pct_lift"]["mean_delta"]) for row in annual]
    tail = [
        float(row["metrics"]["daily_tail_top5_lift"]["mean_delta"]) for row in annual
    ]
    rank_ic = [float(row["metrics"]["rank_ic"]["mean_delta"]) for row in annual]
    exclusive = [
        float(
            row["summary"]["metrics"]["top5_right_only_minus_left_only_mfe_mean"][
                "mean"
            ]
        )
        for row in relationships
    ]
    checks = {
        "positive_top5_mfe_years": sum(value > 0.0 for value in top5),
        "positive_tail_hit_years": sum(value > 0.0 for value in tail),
        "worst_year_rank_ic_delta": min(rank_ic),
        "positive_exclusive_mfe_years": sum(value > 0.0 for value in exclusive),
    }
    passed = (
        checks["positive_top5_mfe_years"]
        >= int(rules["minimum_positive_top5_mfe_years"])
        and checks["positive_tail_hit_years"]
        >= int(rules["minimum_positive_tail_hit_years"])
        and checks["worst_year_rank_ic_delta"]
        >= float(rules["minimum_worst_year_rank_ic_delta"])
        and checks["positive_exclusive_mfe_years"]
        >= int(rules["minimum_positive_exclusive_mfe_years"])
    )
    return {
        "passed": bool(passed),
        "checks": checks,
        "top5_mfe_delta_2023_2024_2025": top5,
        "tail_hit_delta_2023_2024_2025": tail,
        "rank_ic_delta_2023_2024_2025": rank_ic,
        "exclusive_mfe_delta_2023_2024_2025": exclusive,
    }


def _path_guardrail(
    annual: Sequence[Mapping[str, Any]], rules: Mapping[str, Any]
) -> dict[str, Any]:
    required_years = int(rules["path_guardrail_minimum_harm_years"])
    minimum_harm = float(rules["path_guardrail_minimum_median_absolute_harm"])
    findings: dict[str, Any] = {}
    rejected = False
    for name in ("top5_pre_peak_mae_mean", "top5_endpoint_return_mean"):
        rows = [dict(row["metrics"][name]) for row in annual]
        deltas = [float(row["mean_delta"]) for row in rows]
        significant_harm_years = sum(
            value < 0.0 and float(row["hac"]["p_value_two_sided"]) <= 0.10
            for value, row in zip(deltas, rows, strict=True)
        )
        median_harm = max(0.0, -float(np.median(deltas)))
        current_rejected = (
            significant_harm_years >= required_years and median_harm >= minimum_harm
        )
        rejected = rejected or current_rejected
        findings[name] = {
            "delta_2023_2024_2025": deltas,
            "significant_harm_years": significant_harm_years,
            "median_absolute_harm": median_harm,
            "rejected": bool(current_rejected),
        }
    return {"rejected": bool(rejected), "metrics": findings}


def evaluate(
    *,
    study_path: Path = DEFAULT_STUDY_PATH,
    output_root: Path = DEFAULT_OUTPUT_ROOT,
    include_stage2: bool = True,
) -> dict[str, Any]:
    study = load_study(study_path)
    feature_study = source.load_study(
        _resolve(study["sources"]["feature_study_config"])
    )
    feature_output_root = _resolve(study["sources"]["feature_output_root"])
    inputs, pack = source._load_validated_inputs(feature_study)
    if inputs.maximum_outcome_date != "2025-12-31":
        raise ValueError("feature-union evaluation boundary changed")
    reader = objective.FuturePathReader(pack)
    definitions = _variant_definitions(study)
    stages = {"stage1", "stage2"} if include_stage2 else {"stage1"}
    annual: list[dict[str, Any]] = []
    paired: list[dict[str, Any]] = []
    relationships: list[dict[str, Any]] = []

    for horizon in HORIZONS:
        incumbent = str(study["heads"][str(horizon)]["incumbent"])
        for variant, definition in definitions[horizon].items():
            if definition["stage"] not in stages:
                continue
            tasks = [
                task
                for task in _tasks(study, stage=definition["stage"])
                if task["variant"] == variant
            ]
            if not all(
                _task_result_path(output_root, task).is_file() for task in tasks
            ):
                continue
            for task in tasks:
                year = int(task["year"])
                incumbent_result, incumbent_prediction, incumbent_rows = (
                    source._load_outer_result(
                        feature_output_root,
                        year=year,
                        horizon=horizon,
                        variant=incumbent,
                        study=feature_study,
                    )
                )
                result, prediction, rows = _load_union_result(output_root, task=task)
                if not np.array_equal(rows, incumbent_rows):
                    raise ValueError(f"row alignment changed for {task['task_id']}")
                actual = entry._path_actual(
                    inputs=inputs,
                    reader=reader,
                    rows=rows,
                    horizon=horizon,
                    early_horizon=int(
                        study["evaluation"]["early_horizon_by_target"][str(horizon)]
                    ),
                )
                incumbent_bundle = objective.PredictionBundle(
                    variant=incumbent,
                    year=year,
                    horizon=horizon,
                    rows=rows,
                    prediction=incumbent_prediction,
                    result=incumbent_result,
                )
                challenger_bundle = objective.PredictionBundle(
                    variant=variant,
                    year=year,
                    horizon=horizon,
                    rows=rows,
                    prediction=prediction,
                    result=result,
                )
                incumbent_daily, _incumbent_metrics = objective._daily_evaluation(
                    inputs=inputs,
                    bundle=incumbent_bundle,
                    peak_day=actual.peak_day,
                    early_horizon=int(
                        study["evaluation"]["early_horizon_by_target"][str(horizon)]
                    ),
                )
                challenger_daily, challenger_metrics = objective._daily_evaluation(
                    inputs=inputs,
                    bundle=challenger_bundle,
                    peak_day=actual.peak_day,
                    early_horizon=int(
                        study["evaluation"]["early_horizon_by_target"][str(horizon)]
                    ),
                )
                pair_metrics = objective._paired_delta(
                    challenger=challenger_daily,
                    baseline=incumbent_daily,
                    horizon=horizon,
                )
                pair_frame = incumbent_daily.merge(
                    challenger_daily,
                    on="date_idx",
                    suffixes=("_incumbent", "_challenger"),
                    validate="one_to_one",
                )
                pair_path = (
                    output_root
                    / "evaluation"
                    / variant
                    / f"h{horizon:02d}"
                    / f"fold_{year}_paired_daily.parquet"
                )
                pair_path.parent.mkdir(parents=True, exist_ok=True)
                pair_frame.to_parquet(pair_path, index=False, compression="zstd")
                relation_frame = entry._daily_model_relationship(
                    actual=actual,
                    left_name=incumbent,
                    left_score=incumbent_prediction,
                    right_name=variant,
                    right_score=prediction,
                )
                relation_path = (
                    output_root
                    / "relationships"
                    / f"h{horizon:02d}"
                    / f"fold_{year}_{variant}.parquet"
                )
                relation_path.parent.mkdir(parents=True, exist_ok=True)
                relation_frame.to_parquet(
                    relation_path, index=False, compression="zstd"
                )
                annual.append(
                    {
                        "year": year,
                        "horizon": horizon,
                        "variant": variant,
                        "stage": definition["stage"],
                        "control": bool(definition["control"]),
                        "metrics": challenger_metrics,
                    }
                )
                paired.append(
                    {
                        "year": year,
                        "horizon": horizon,
                        "incumbent": incumbent,
                        "challenger": variant,
                        "stage": definition["stage"],
                        "control": bool(definition["control"]),
                        "metrics": pair_metrics,
                        "paired_daily": _file_record(pair_path),
                    }
                )
                relationships.append(
                    {
                        "year": year,
                        "horizon": horizon,
                        "incumbent": incumbent,
                        "challenger": variant,
                        "stage": definition["stage"],
                        "control": bool(definition["control"]),
                        "summary": entry._summarize_daily_frame(
                            relation_frame, horizon=horizon
                        ),
                        "daily": _file_record(relation_path),
                    }
                )

    rules = dict(study["decision"])
    evidence: dict[str, dict[str, Any]] = {}
    p_values: dict[str, float] = {}
    for horizon in HORIZONS:
        for variant, definition in definitions[horizon].items():
            current_paired = [
                row
                for row in paired
                if int(row["horizon"]) == horizon and row["challenger"] == variant
            ]
            if len(current_paired) != len(FOLD_YEARS):
                continue
            current_relationships = [
                row
                for row in relationships
                if int(row["horizon"]) == horizon and row["challenger"] == variant
            ]
            role = _role_gate(
                annual=current_paired,
                relationships=current_relationships,
                rules=rules,
            )
            combined = _stouffer(current_paired)
            guardrail = _path_guardrail(current_paired, rules)
            key = f"h{horizon}:{variant}"
            evidence[key] = {
                **definition,
                "role_gate": role,
                "combined_top5_test": combined,
                "path_guardrail": guardrail,
                "added_feature_count": sum(
                    len(source.feature_catalog()[family])
                    for family in definition["families"]
                ),
            }
            if not definition["control"]:
                p_values[key] = float(combined["p_value_one_sided"])
    q_values = source._benjamini_hochberg(p_values)
    for key, record in evidence.items():
        q_value = q_values.get(key, math.nan)
        record["top5_fdr_q_value"] = q_value
        record["formal_pass"] = bool(
            record["role_gate"]["passed"]
            and math.isfinite(q_value)
            and q_value <= float(rules["maximum_top5_fdr_q_value"])
            and not record["path_guardrail"]["rejected"]
        )

    stage2_eligible: dict[str, bool] = {}
    for horizon in HORIZONS:
        stage1_non_control = [
            record
            for record in evidence.values()
            if int(record["horizon"]) == horizon
            and record["stage"] == "stage1"
            and not record["control"]
        ]
        stage2_eligible[str(horizon)] = bool(
            len(stage1_non_control) == 2
            and all(record["role_gate"]["passed"] for record in stage1_non_control)
        )

    negative_control_invalid = any(
        record["control"]
        and (
            record["role_gate"]["passed"]
            or float(record["combined_top5_test"]["p_value_one_sided"]) <= 0.10
        )
        for record in evidence.values()
    )
    final_heads: dict[str, str] = {}
    for horizon in HORIZONS:
        incumbent = str(study["heads"][str(horizon)]["incumbent"])
        passing = [
            record
            for record in evidence.values()
            if int(record["horizon"]) == horizon
            and not record["control"]
            and record["formal_pass"]
        ]
        passing.sort(
            key=lambda record: (
                float(np.median(record["role_gate"]["top5_mfe_delta_2023_2024_2025"])),
                min(record["role_gate"]["top5_mfe_delta_2023_2024_2025"]),
                -int(record["added_feature_count"]),
            ),
            reverse=True,
        )
        final_heads[str(horizon)] = str(passing[0]["variant"]) if passing else incumbent
    decision = {
        "status": (
            "invalid_negative_control"
            if negative_control_invalid
            else "completed_feature_union_audit"
        ),
        "stage2_eligible": stage2_eligible,
        "negative_control_invalid": bool(negative_control_invalid),
        "final_mfe_heads": final_heads,
        "entry_coordinate_count_changed": False,
        "next_step": (
            "stop_and_debug_negative_control"
            if negative_control_invalid
            else "build_strict_oos_entry_contract"
        ),
        "does_not_select": list(study["non_selections"]),
    }
    summary = {
        "schema": SUMMARY_SCHEMA,
        "status": decision["status"],
        "completed_at": _now(),
        "study_id": STUDY_ID,
        "scope": {
            "fold_years": list(FOLD_YEARS),
            "horizons": list(HORIZONS),
            "maximum_consumed_outcome_date": inputs.maximum_outcome_date,
            "trained_task_count": len(
                [
                    task
                    for task in _tasks(study)
                    if _task_result_path(output_root, task).is_file()
                ]
            ),
            "forbidden_2026_row_count": 0,
        },
        "evidence": evidence,
        "annual": annual,
        "paired": paired,
        "relationships": relationships,
        "decision": decision,
        "disclosure": {
            "fold_reuse": study["folds"]["fold_role"],
            "top5": "Top 5% is a strong-candidate diagnostic, not a slot count.",
            "2026": "No 2026 row, outcome, prediction, or metric was read.",
        },
    }
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
            "evidence": summary["evidence"],
            "decision": decision,
            "disclosure": summary["disclosure"],
            "full_output": _file_record(summary_path),
        },
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
    for task in _tasks(study, stage="stage1"):
        _run_task(
            study=study,
            feature_study=feature_study,
            feature_output_root=feature_output_root,
            output_root=output_root,
            inputs=inputs,
            task=task,
        )
    interim = evaluate(
        study_path=study_path,
        output_root=output_root,
        include_stage2=False,
    )
    eligibility = dict(interim["decision"]["stage2_eligible"])
    for task in _tasks(study, stage="stage2"):
        if not bool(eligibility[str(task["horizon"])]):
            continue
        _run_task(
            study=study,
            feature_study=feature_study,
            feature_output_root=feature_output_root,
            output_root=output_root,
            inputs=inputs,
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
    feature_output_root = _resolve(study["sources"]["feature_output_root"])
    groups: dict[str, list[str]] = {"completed": [], "pending": []}
    for task in _tasks(study):
        complete = _task_complete(
            _task_result_path(output_root, task),
            task=task,
            study=study,
            feature_study=feature_study,
            feature_output_root=feature_output_root,
        )
        groups["completed" if complete else "pending"].append(str(task["task_id"]))
    return {
        "study_id": STUDY_ID,
        "stage1_task_count": len(_tasks(study, stage="stage1")),
        "stage2_maximum_task_count": len(_tasks(study, stage="stage2")),
        **groups,
    }


def self_test() -> dict[str, Any]:
    left = np.arange(15, dtype=np.float32).reshape(3, 5)
    right = np.arange(10, dtype=np.float32).reshape(2, 5) + 100.0
    union = UnionExtra((left, right), ("a", "b", "c", "d", "e"))  # type: ignore[arg-type]
    rows = np.asarray([4, 1, 3], dtype=np.int64)
    np.testing.assert_array_equal(
        union[:, rows], np.concatenate([left[:, rows], right[:, rows]], axis=0)
    )
    rules = load_study()["decision"]
    annual = []
    relationships = []
    for year in FOLD_YEARS:
        annual.append(
            {
                "metrics": {
                    "top_5pct_lift": {"mean_delta": 0.001},
                    "daily_tail_top5_lift": {"mean_delta": 0.01},
                    "rank_ic": {"mean_delta": 0.001},
                }
            }
        )
        relationships.append(
            {
                "summary": {
                    "metrics": {
                        "top5_right_only_minus_left_only_mfe_mean": {"mean": 0.002}
                    }
                }
            }
        )
    assert _role_gate(annual=annual, relationships=relationships, rules=rules)["passed"]
    return {"status": "passed"}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the Seq100 MFE feature-union audit."
    )
    parser.add_argument("--study-path", type=Path, default=DEFAULT_STUDY_PATH)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    action = parser.add_mutually_exclusive_group()
    action.add_argument("--run-pending", action="store_true")
    action.add_argument("--evaluate", action="store_true")
    action.add_argument("--self-test", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.self_test:
        payload = self_test()
    elif args.run_pending:
        summary = run_pending(study_path=args.study_path, output_root=args.output_root)
        payload = {
            "status": summary["status"],
            "scope": summary["scope"],
            "decision": summary["decision"],
        }
    elif args.evaluate:
        payload = evaluate(study_path=args.study_path, output_root=args.output_root)
    else:
        payload = status(study_path=args.study_path, output_root=args.output_root)
    print(json.dumps(payload, ensure_ascii=False, indent=2, default=_json_default))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
