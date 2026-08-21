from __future__ import annotations

import gc
import json
import math
import os
import time
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import lightgbm as lgb
import numpy as np
import pandas as pd
import psutil

from quantlab.core.io import stable_hash

from .data import (
    DEFAULT_VALIDATION_START_DATE,
    OUTPUT_ROOT,
    ResearchData,
    ResearchDataError,
    build_forward_folds,
    cutoff_audit_fields,
    cutoff_violation_count,
    daily_rank_metrics,
    date_group_sizes,
    date_relevance_labels,
    equal_date_weights,
    fold_rows,
    load_data,
    open_array,
    write_json,
)
from .portfolio import (
    account_specs,
    add_account_diagnostics,
    newey_west_interval,
    parse_execution_costs,
    simulate_account,
)

TARGET_NAME = "exact_net_return_d10_base"
SEED = 20260809
TREE_PARAMETERS: dict[str, Any] = {
    "boosting_type": "gbdt",
    "device_type": "cpu",
    "objective": "lambdarank",
    "metric": "ndcg",
    "label_gain": list(range(10)),
    "lambdarank_truncation_level": 50,
    "ndcg_eval_at": [10, 30, 100],
    "learning_rate": 0.03,
    "num_leaves": 127,
    "max_depth": 8,
    "min_data_in_leaf": 2000,
    "lambda_l1": 0.1,
    "lambda_l2": 10.0,
    "feature_fraction": 0.9,
    "bagging_fraction": 1.0,
    "bagging_freq": 0,
    "max_bin": 127,
    "feature_pre_filter": False,
    "deterministic": True,
    "force_col_wise": True,
    "seed": SEED,
    "feature_fraction_seed": SEED,
    "bagging_seed": SEED,
    "data_random_seed": SEED,
    "verbosity": -1,
}
MAX_BOOST_ROUNDS = 1000
EARLY_STOPPING_ROUNDS = 75


class MatrixPrefixSequence(lgb.Sequence):
    """Expose a contiguous row range and matrix prefix without materializing it."""

    def __init__(
        self,
        matrix: np.ndarray,
        rows: np.ndarray,
        feature_count: int,
        *,
        batch_size: int = 32_768,
    ) -> None:
        self.matrix = matrix
        self.rows = np.asarray(rows, dtype=np.int64)
        self.feature_count = int(feature_count)
        self.batch_size = int(batch_size)

    def __len__(self) -> int:
        return len(self.rows)

    def _block(self, local: np.ndarray) -> np.ndarray:
        selected = self.rows[np.asarray(local, dtype=np.int64)]
        if len(selected) and np.all(np.diff(selected) == 1):
            block = self.matrix[int(selected[0]) : int(selected[-1]) + 1, : self.feature_count]
        else:
            block = self.matrix[selected, : self.feature_count]
        # LightGBM's Sequence sampler requires float64 even though the source
        # matrix and the final bins are float32/uint8-sized.  Conversion stays
        # batch-local, so no second 158/183 matrix is materialized.
        return np.asarray(block, dtype=np.float64)

    def __getitem__(self, index: Any) -> np.ndarray:
        if isinstance(index, slice):
            start = 0 if index.start is None else int(index.start)
            stop = len(self.rows) if index.stop is None else int(index.stop)
            step = 1 if index.step is None else int(index.step)
            return self._block(np.arange(start, stop, step, dtype=np.int64))
        if isinstance(index, (list, tuple, np.ndarray)):
            return self._block(np.asarray(index, dtype=np.int64))
        if isinstance(index, (int, np.integer)):
            return self._block(np.asarray([int(index)], dtype=np.int64))[0]
        raise TypeError(type(index).__name__)


def _folds(data: Any) -> list[dict[str, Any]]:
    return build_forward_folds(
        date_idx=data.dates,
        trade_date=data.row_index["trade_date"].astype(str).to_numpy(),
        validation_start_date=DEFAULT_VALIDATION_START_DATE,
        validation_end_date=data.maximum_outcome_date,
        fold_count=5,
        purge_days=30,
    )


def _task_root(feature_count: int, fold_number: int) -> Path:
    return OUTPUT_ROOT / f"tree_{int(feature_count)}" / f"fold_{int(fold_number):02d}"


def _task_fingerprint(data: Any, feature_count: int, fold: Mapping[str, Any]) -> str:
    feature_names = data.feature_names[: int(feature_count)]
    return stable_hash(
        {
            "pipeline": "technical_tree",
            "input_fingerprint": data.fingerprints["input"],
            "view_fingerprint": data.fingerprints["features"],
            "target_fingerprint": data.fingerprints["exact_targets"],
            "feature_names": feature_names,
            "fold": fold,
            "target": TARGET_NAME,
            "parameters": TREE_PARAMETERS,
            "maximum_rounds": MAX_BOOST_ROUNDS,
            "early_stopping_rounds": EARLY_STOPPING_ROUNDS,
        }
    )


def _safe_json(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _safe_json(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_safe_json(item) for item in value]
    if isinstance(value, np.generic):
        return _safe_json(value.item())
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def _task_complete(path: Path, fingerprint: str) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    try:
        result = dict(json.loads(path.read_text(encoding="utf-8")))
        files = result["files"]
        if (
            result.get("status") == "completed"
            and result.get("fingerprint") == fingerprint
            and cutoff_violation_count(result) == 0
            and all(Path(record["path"]).is_file() for record in files.values())
        ):
            return result
    except (KeyError, OSError, TypeError, ValueError, json.JSONDecodeError):
        return None
    return None


def _predict(
    booster: lgb.Booster,
    matrix: np.ndarray,
    rows: np.ndarray,
    feature_count: int,
    *,
    batch_size: int = 32_768,
) -> np.ndarray:
    output = np.empty(len(rows), dtype=np.float32)
    for left in range(0, len(rows), int(batch_size)):
        right = min(left + int(batch_size), len(rows))
        block = np.asarray(matrix[rows[left:right], : int(feature_count)], dtype=np.float32)
        output[left:right] = np.asarray(
            booster.predict(block, num_iteration=booster.best_iteration),
            dtype=np.float32,
        )
    return output


@dataclass(frozen=True)
class _TreeFoldInputs:
    data: ResearchData
    fold: dict[str, Any]
    feature_names: list[str]
    fingerprint: str
    root: Path
    train_rows: np.ndarray
    validation_rows: np.ndarray
    train_raw: np.ndarray
    validation_raw: np.ndarray
    train_valid: np.ndarray
    validation_valid: np.ndarray
    train_dates: np.ndarray
    validation_dates: np.ndarray
    train_label: np.ndarray
    validation_label: np.ndarray
    train_weight: np.ndarray
    validation_weight: np.ndarray


def _tree_fold_inputs(feature_count: int, fold_number: int) -> _TreeFoldInputs:
    data = load_data()
    data.feature_positions(int(feature_count))
    folds = _folds(data)
    if not 1 <= int(fold_number) <= len(folds):
        raise ResearchDataError("fold number is outside 1..5")
    fold = folds[int(fold_number) - 1]
    feature_names = data.feature_names[: int(feature_count)]
    train_rows = fold_rows(data.dates, fold, "train")
    validation_rows = fold_rows(data.dates, fold, "validation")
    values, valid, column = data.target(TARGET_NAME)
    train_raw = np.asarray(values[train_rows, column], dtype=np.float32)
    validation_raw = np.asarray(values[validation_rows, column], dtype=np.float32)
    train_valid = np.asarray(valid[train_rows, column], dtype=bool) & np.isfinite(train_raw)
    validation_valid = np.asarray(valid[validation_rows, column], dtype=bool) & np.isfinite(validation_raw)
    train_dates = data.dates[train_rows]
    validation_dates = data.dates[validation_rows]
    train_label = date_relevance_labels(train_dates, train_raw, train_valid)
    validation_label = date_relevance_labels(validation_dates, validation_raw, validation_valid)
    train_label[~train_valid] = 0.0
    validation_label[~validation_valid] = 0.0
    train_weight = equal_date_weights(train_dates, train_valid)
    validation_weight = equal_date_weights(validation_dates, validation_valid)
    return _TreeFoldInputs(
        data=data,
        fold=dict(fold),
        feature_names=list(feature_names),
        fingerprint=_task_fingerprint(data, feature_count, fold),
        root=_task_root(feature_count, fold_number),
        train_rows=train_rows,
        validation_rows=validation_rows,
        train_raw=train_raw,
        validation_raw=validation_raw,
        train_valid=train_valid,
        validation_valid=validation_valid,
        train_dates=train_dates,
        validation_dates=validation_dates,
        train_label=train_label,
        validation_label=validation_label,
        train_weight=train_weight,
        validation_weight=validation_weight,
    )


def _tree_parameters() -> tuple[dict[str, Any], dict[str, Any], int]:
    available_mb = int(psutil.virtual_memory().available / (1 << 20))
    histogram_pool_mb = max(256, min(1024, (available_mb - 2048) // 4))
    parameters = {
        **TREE_PARAMETERS,
        "num_threads": min(16, os.cpu_count() or 1),
        "histogram_pool_size": histogram_pool_mb,
    }
    binary = {
        "max_bin": parameters["max_bin"],
        "data_random_seed": SEED,
        "feature_pre_filter": False,
        "num_threads": parameters["num_threads"],
        "verbosity": -1,
    }
    return parameters, binary, histogram_pool_mb


def _build_tree_datasets(
    inputs: _TreeFoldInputs,
    *,
    feature_count: int,
    binary_params: Mapping[str, Any],
) -> tuple[lgb.Dataset, lgb.Dataset]:
    train_source = MatrixPrefixSequence(inputs.data.matrix, inputs.train_rows, feature_count, batch_size=32_768)
    validation_source = MatrixPrefixSequence(
        inputs.data.matrix, inputs.validation_rows, feature_count, batch_size=32_768
    )
    train_set = lgb.Dataset(
        train_source,
        label=inputs.train_label,
        weight=inputs.train_weight,
        group=date_group_sizes(inputs.train_dates),
        feature_name=inputs.feature_names,
        params=dict(binary_params),
        free_raw_data=True,
    ).construct()
    validation_set = lgb.Dataset(
        validation_source,
        label=inputs.validation_label,
        weight=inputs.validation_weight,
        group=date_group_sizes(inputs.validation_dates),
        feature_name=inputs.feature_names,
        reference=train_set,
        params=dict(binary_params),
        free_raw_data=True,
    ).construct()
    del train_source, validation_source
    gc.collect()
    return train_set, validation_set


def _write_tree_outputs(
    *,
    inputs: _TreeFoldInputs,
    booster: lgb.Booster,
    prediction: np.ndarray,
    daily: pd.DataFrame,
    fold_number: int,
) -> tuple[dict[str, dict[str, str]], pd.DataFrame]:
    inputs.root.mkdir(parents=True, exist_ok=True)
    model_path = inputs.root / "model.txt"
    partial_model = model_path.with_suffix(".txt.partial")
    booster.save_model(str(partial_model), num_iteration=booster.best_iteration)
    partial_model.replace(model_path)
    prediction_frame = inputs.data.row_index.iloc[inputs.validation_rows][
        ["candidate_id", "date_idx", "trade_date", "symbol", "symbol_idx"]
    ].copy()
    prediction_frame.insert(0, "row_position", inputs.validation_rows)
    prediction_frame["fold"] = int(fold_number)
    prediction_frame["score"] = prediction
    prediction_frame["actual"] = inputs.validation_raw
    prediction_frame["target_valid"] = inputs.validation_valid
    predictions_path = inputs.root / "predictions.parquet"
    prediction_frame.to_parquet(predictions_path, index=False)
    daily_path = inputs.root / "daily_metrics.parquet"
    daily.to_parquet(daily_path, index=False)
    importance = pd.DataFrame(
        {
            "feature_name": inputs.feature_names,
            "gain": booster.feature_importance(importance_type="gain"),
            "split": booster.feature_importance(importance_type="split"),
        }
    ).sort_values(["gain", "split"], ascending=False, kind="stable")
    importance_path = inputs.root / "feature_importance.parquet"
    importance.to_parquet(importance_path, index=False)
    return {
        "model": {"path": str(model_path)},
        "predictions": {"path": str(predictions_path)},
        "daily_metrics": {"path": str(daily_path)},
        "feature_importance": {"path": str(importance_path)},
    }, prediction_frame


def train_fold(feature_count: int, fold_number: int) -> dict[str, Any]:
    inputs = _tree_fold_inputs(feature_count, fold_number)
    result_path = inputs.root / "result.json"
    cached = _task_complete(result_path, inputs.fingerprint)
    if cached is not None:
        return cached

    parameters, binary_params, histogram_pool_mb = _tree_parameters()
    train_set, validation_set = _build_tree_datasets(
        inputs,
        feature_count=feature_count,
        binary_params=binary_params,
    )

    print(
        json.dumps(
            {
                "event": "technical_tree_training_started",
                "feature_count": feature_count,
                "fold": fold_number,
                "train_rows": len(inputs.train_rows),
                "validation_rows": len(inputs.validation_rows),
                "histogram_pool_mb": histogram_pool_mb,
            },
            ensure_ascii=False,
        ),
        flush=True,
    )
    started = time.perf_counter()
    booster = lgb.train(
        parameters,
        train_set,
        num_boost_round=MAX_BOOST_ROUNDS,
        valid_sets=[validation_set],
        valid_names=["forward_validation"],
        callbacks=[
            lgb.early_stopping(EARLY_STOPPING_ROUNDS, verbose=True),
            lgb.log_evaluation(period=25),
        ],
    )
    prediction = _predict(
        booster,
        inputs.data.matrix,
        inputs.validation_rows,
        feature_count,
    )
    elapsed = time.perf_counter() - started
    daily, metrics = daily_rank_metrics(
        dates=inputs.validation_dates[inputs.validation_valid],
        actual=inputs.validation_raw[inputs.validation_valid],
        prediction=prediction[inputs.validation_valid],
    )
    files, prediction_frame = _write_tree_outputs(
        inputs=inputs,
        booster=booster,
        prediction=prediction,
        daily=daily,
        fold_number=fold_number,
    )
    result = {
        "status": "completed",
        "completed_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "fingerprint": inputs.fingerprint,
        "feature_count": int(feature_count),
        "feature_families": sorted(set(inputs.data.feature_families[: int(feature_count)])),
        "target": TARGET_NAME,
        "fold": inputs.fold,
        "best_iteration": int(booster.best_iteration),
        "metrics": metrics,
        "parameters": parameters,
        "elapsed_seconds": elapsed,
        "train_valid_count": int(inputs.train_valid.sum()),
        "validation_valid_count": int(inputs.validation_valid.sum()),
        **cutoff_audit_fields(),
        "files": files,
    }
    write_json(result_path, _safe_json(result))
    print(
        json.dumps(
            {
                "event": "technical_tree_training_completed",
                "feature_count": feature_count,
                "fold": fold_number,
                "best_iteration": booster.best_iteration,
                "rank_ic": metrics["daily_rank_ic_mean"],
                "elapsed_seconds": elapsed,
            },
            ensure_ascii=False,
        ),
        flush=True,
    )
    del booster, train_set, validation_set, prediction_frame
    gc.collect()
    return result


def train_all(feature_count: int) -> list[dict[str, Any]]:
    return [train_fold(feature_count, fold) for fold in range(1, 6)]


def load_fold_results(feature_count: int) -> tuple[list[dict[str, Any]], pd.DataFrame]:
    data = load_data()
    data.feature_positions(int(feature_count))
    folds = _folds(data)
    results: list[dict[str, Any]] = []
    frames: list[pd.DataFrame] = []
    for fold in range(1, 6):
        path = _task_root(feature_count, fold) / "result.json"
        if not path.is_file():
            raise ResearchDataError(f"tree fold is missing: {path}")
        result = dict(json.loads(path.read_text(encoding="utf-8")))
        expected_fingerprint = _task_fingerprint(data, feature_count, folds[fold - 1])
        if (
            result.get("status") != "completed"
            or result.get("fingerprint") != expected_fingerprint
            or cutoff_violation_count(result) != 0
            or not all(Path(record["path"]).is_file() for record in result.get("files", {}).values())
        ):
            raise ResearchDataError(f"tree fold is stale or invalid: {path}")
        results.append(result)
        frames.append(pd.read_parquet(Path(result["files"]["predictions"]["path"])))
    oof = pd.concat(frames, ignore_index=True).sort_values(["date_idx", "symbol_idx"], kind="stable")
    if oof.duplicated("row_position").any():
        raise ResearchDataError("OOF row positions overlap")
    return results, oof


def _top10_selections(data: ResearchData, oof: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    ranked = oof[np.isfinite(oof["score"])].sort_values(
        ["date_idx", "score", "symbol_idx"],
        ascending=[True, False, True],
        kind="stable",
    )
    ranked["selection_rank"] = ranked.groupby("date_idx", sort=False).cumcount() + 1
    selections = ranked.loc[ranked["selection_rank"] <= 10].copy()
    path_values, path_valid, fill_days, path_column = data.legal_exit(10)
    positions = selections["row_position"].to_numpy(dtype=np.int64)
    gross = np.asarray(path_values[positions, path_column], dtype=np.float32)
    gross_valid = np.asarray(path_valid[positions, path_column], dtype=bool)
    exits = np.asarray(fill_days[positions, path_column], dtype=np.int16)
    usable = gross_valid & np.isfinite(gross) & (exits >= 10)
    dropped = int((~usable).sum())
    selections = selections.loc[usable].copy()
    selections["legal_gross_return"] = gross[usable]
    selections["fill_day"] = exits[usable]
    return selections, dropped


def _simulate_tree_accounts(
    *,
    data: ResearchData,
    selections: pd.DataFrame,
    evaluation_root: Path,
) -> list[dict[str, Any]]:
    execution = data.execution
    daily_raw = open_array(execution["daily_raw"])
    raw_open = open_array(execution["entry_open"])
    entry_filled = open_array(execution["entry_filled"])
    costs = parse_execution_costs(execution["costs"])
    account_results: list[dict[str, Any]] = []
    for spec in account_specs(10):
        result, equity, trades = simulate_account(
            spec=spec,
            selections=selections,
            date_values=data.date_values,
            symbol_values=data.symbol_values,
            cutoff_idx=data.cutoff_idx,
            daily_raw=daily_raw,
            raw_open=raw_open,
            entry_filled=entry_filled,
            costs=costs,
        )
        result = add_account_diagnostics(result=result, equity=equity, trades=trades, selections=selections)
        task_root = evaluation_root / result["task_id"]
        task_root.mkdir(parents=True, exist_ok=True)
        equity.to_parquet(task_root / "equity.parquet", index=False)
        trades.to_parquet(task_root / "trades.parquet", index=False)
        write_json(task_root / "result.json", _safe_json(result))
        account_results.append(result)
    return account_results


def _write_tree_evaluation_files(
    *,
    evaluation_root: Path,
    oof: pd.DataFrame,
    selections: pd.DataFrame,
    daily: pd.DataFrame,
    account_results: list[dict[str, Any]],
) -> dict[str, Path]:
    oof_path = evaluation_root / "oof_predictions.parquet"
    selections_path = evaluation_root / "top10_selections.parquet"
    daily_path = evaluation_root / "daily_metrics.parquet"
    summary_path = evaluation_root / "account_summary.parquet"
    oof.to_parquet(oof_path, index=False)
    selections.to_parquet(selections_path, index=False)
    daily.to_parquet(daily_path, index=False)
    summary = pd.DataFrame(
        [
            {
                "task_id": item["task_id"],
                "top_k": item["spec"]["top_k"],
                "cost_scenario": item["spec"]["cost_scenario"],
                "maximum_credited_gross_return": item["spec"].get("maximum_credited_gross_return"),
                "allow_overlapping_same_symbol": item["spec"].get("allow_overlapping_same_symbol", True),
                "total_net_return": item["total_net_return"],
                "maximum_drawdown": item["maximum_drawdown"],
                "positive_year_count": item["positive_year_count"],
                "trade_count": item["trade_count"],
                "hac20_lower": item["daily_hac20_net_return"]["lower"],
                "hac20_upper": item["daily_hac20_net_return"]["upper"],
            }
            for item in account_results
        ]
    )
    summary.to_parquet(summary_path, index=False)
    return {
        "oof_predictions": oof_path,
        "top10_selections": selections_path,
        "daily_metrics": daily_path,
        "account_summary": summary_path,
    }


def evaluate_predictions(
    *,
    run_name: str,
    fold_results: list[dict[str, Any]],
    oof: pd.DataFrame,
    metadata: Mapping[str, Any],
) -> dict[str, Any]:
    data = load_data()
    valid = oof["target_valid"].astype(bool) & np.isfinite(oof["actual"]) & np.isfinite(oof["score"])
    daily, metrics = daily_rank_metrics(
        dates=oof.loc[valid, "date_idx"].to_numpy(dtype=np.int32),
        actual=oof.loc[valid, "actual"].to_numpy(dtype=np.float32),
        prediction=oof.loc[valid, "score"].to_numpy(dtype=np.float32),
    )
    selections, dropped = _top10_selections(data, oof)
    evaluation_root = OUTPUT_ROOT / str(run_name) / "evaluation"
    evaluation_root.mkdir(parents=True, exist_ok=True)
    account_results = _simulate_tree_accounts(
        data=data,
        selections=selections,
        evaluation_root=evaluation_root,
    )
    paths = _write_tree_evaluation_files(
        evaluation_root=evaluation_root,
        oof=oof,
        selections=selections,
        daily=daily,
        account_results=account_results,
    )
    result = {
        "status": "completed",
        "completed_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        **dict(metadata),
        "input_fingerprint": data.fingerprints["input"],
        "view_fingerprint": data.fingerprints["features"],
        "target_fingerprint": data.fingerprints["exact_targets"],
        "fold_fingerprints": [item.get("fingerprint") for item in fold_results],
        "fold_metrics": [item["metrics"] for item in fold_results],
        "oof_metrics": metrics,
        "selected_row_count": len(selections),
        "right_censored_top10_selection_count": dropped,
        "accounts": account_results,
        **cutoff_audit_fields(),
        "files": {
            key: {"path": str(path)} for key, path in paths.items()
        },
    }
    write_json(evaluation_root / "result.json", _safe_json(result))
    return result


def evaluate(feature_count: int) -> dict[str, Any]:
    fold_results, oof = load_fold_results(feature_count)
    return evaluate_predictions(
        run_name=f"tree_{int(feature_count)}",
        fold_results=fold_results,
        oof=oof,
        metadata={
            "model_family": "lightgbm_lambdarank",
            "feature_count": int(feature_count),
        },
    )


def _find_account(
    result: Mapping[str, Any],
    *,
    cap: float | None,
    no_overlap: bool,
) -> Mapping[str, Any]:
    matches = [
        item
        for item in result["accounts"]
        if int(item["spec"]["top_k"]) == 10
        and item["spec"]["cost_scenario"] == "stress"
        and item["spec"].get("maximum_credited_gross_return") == cap
        and (not bool(item["spec"].get("allow_overlapping_same_symbol", True))) == bool(no_overlap)
    ]
    if len(matches) != 1:
        raise ResearchDataError(f"account result is not unique: {cap}, {no_overlap}")
    return matches[0]


@dataclass(frozen=True)
class _TreeEvaluation:
    result: dict[str, Any]
    daily: pd.DataFrame
    oof: pd.DataFrame
    selections: pd.DataFrame


def _load_tree_evaluation(feature_count: int) -> _TreeEvaluation:
    result_path = OUTPUT_ROOT / f"tree_{feature_count}/evaluation/result.json"
    if not result_path.is_file():
        raise ResearchDataError(f"evaluation is missing: {result_path}")
    result = dict(json.loads(result_path.read_text(encoding="utf-8")))
    files = result["files"]
    return _TreeEvaluation(
        result=result,
        daily=pd.read_parquet(Path(files["daily_metrics"]["path"])),
        oof=pd.read_parquet(
            Path(files["oof_predictions"]["path"]),
            columns=["row_position", "date_idx", "score"],
        ),
        selections=pd.read_parquet(Path(files["top10_selections"]["path"])),
    )


def _tree_comparison_row(feature_count: int, result: Mapping[str, Any]) -> dict[str, Any]:
    raw = _find_account(result, cap=None, no_overlap=False)
    capped = _find_account(result, cap=0.10, no_overlap=False)
    no_overlap = _find_account(result, cap=None, no_overlap=True)
    robust = _find_account(result, cap=0.10, no_overlap=True)
    return {
        "feature_count": feature_count,
        "rank_ic": result["oof_metrics"]["daily_rank_ic_mean"],
        "rank_ic_positive_fraction": result["oof_metrics"]["daily_rank_ic_positive_fraction"],
        "top10_stress_return": raw["total_net_return"],
        "top10_stress_drawdown": raw["maximum_drawdown"],
        "top10_stress_cap10_return": capped["total_net_return"],
        "top10_no_overlap_return": no_overlap["total_net_return"],
        "top10_no_overlap_drawdown": no_overlap["maximum_drawdown"],
        "top10_no_overlap_cap10_return": robust["total_net_return"],
        "positive_years": no_overlap["positive_year_count"],
        "no_overlap_annual": no_overlap["annual"],
    }


def _paired_rank_ic_increment(
    evaluations: Mapping[int, _TreeEvaluation],
) -> tuple[dict[str, Any], Path]:
    left = evaluations[158].daily[["date_idx", "rank_ic"]].rename(columns={"rank_ic": "rank_ic_158"})
    right = evaluations[183].daily[["date_idx", "rank_ic"]].rename(columns={"rank_ic": "rank_ic_183"})
    paired = left.merge(right, on="date_idx", validate="one_to_one")
    paired["minute_rank_ic_increment"] = paired["rank_ic_183"] - paired["rank_ic_158"]
    interval = newey_west_interval(paired["minute_rank_ic_increment"].to_numpy(dtype=np.float64), lag=20)
    path = OUTPUT_ROOT / "tree_158_vs_183_daily.parquet"
    paired.to_parquet(path, index=False)
    return interval, path


def _mean_daily_score_rank_correlation(evaluations: Mapping[int, _TreeEvaluation]) -> float:
    score_pairs = evaluations[158].oof.merge(
        evaluations[183].oof,
        on=["row_position", "date_idx"],
        suffixes=("_158", "_183"),
        validate="one_to_one",
    )
    score_pairs["rank_158"] = score_pairs.groupby("date_idx", sort=False)["score_158"].rank(pct=True)
    score_pairs["rank_183"] = score_pairs.groupby("date_idx", sort=False)["score_183"].rank(pct=True)
    correlations = score_pairs.groupby("date_idx")[["rank_158", "rank_183"]].corr()
    return float(np.nanmean(correlations.iloc[0::2, -1].to_numpy(dtype=np.float64)))


def _topk_return_increments(evaluations: Mapping[int, _TreeEvaluation]) -> dict[str, Any]:
    increments: dict[str, Any] = {}
    for top_k in (1, 3, 5, 10):
        daily = {
            feature_count: evaluation.selections.loc[
                evaluation.selections["selection_rank"] <= top_k
            ].groupby("date_idx")["actual"].mean()
            for feature_count, evaluation in evaluations.items()
        }
        paired = pd.concat(
            [daily[158].rename("net_158"), daily[183].rename("net_183")],
            axis=1,
            join="inner",
        ).dropna()
        difference = paired["net_183"].to_numpy(dtype=np.float64) - paired["net_158"].to_numpy(
            dtype=np.float64
        )
        increments[str(top_k)] = newey_west_interval(difference, lag=20)
    return increments


def _mean_top10_overlap(evaluations: Mapping[int, _TreeEvaluation]) -> float:
    keys = ["date_idx", "candidate_id"]
    overlap = (
        evaluations[158].selections[keys]
        .merge(evaluations[183].selections[keys], on=keys, how="inner")
        .groupby("date_idx")
        .size()
    )
    return float(overlap.mean() / 10.0)


def _minute_feature_importance(data: ResearchData) -> list[dict[str, Any]]:
    minute_names = set(data.feature_names[158:])
    rows: list[dict[str, Any]] = []
    for fold in range(1, 6):
        importance = pd.read_parquet(_task_root(183, fold) / "feature_importance.parquet")
        minute = importance["feature_name"].isin(minute_names)
        total_gain = float(importance["gain"].sum())
        total_split = float(importance["split"].sum())
        rows.append(
            {
                "fold": fold,
                "minute_gain_share": (
                    float(importance.loc[minute, "gain"].sum() / total_gain) if total_gain > 0.0 else 0.0
                ),
                "minute_split_share": (
                    float(importance.loc[minute, "split"].sum() / total_split) if total_split > 0.0 else 0.0
                ),
                "minute_features_with_gain": int((importance.loc[minute, "gain"] > 0.0).sum()),
            }
        )
    return rows


def compare() -> dict[str, Any]:
    data = load_data()
    evaluations = {feature_count: _load_tree_evaluation(feature_count) for feature_count in (158, 183)}
    comparison = pd.DataFrame(
        [
            _tree_comparison_row(feature_count, evaluation.result)
            for feature_count, evaluation in evaluations.items()
        ]
    ).sort_values("feature_count")
    summary_path = OUTPUT_ROOT / "tree_158_vs_183.parquet"
    comparison.to_parquet(summary_path, index=False)
    increment, paired_path = _paired_rank_ic_increment(evaluations)
    output = {
        "status": "completed",
        "completed_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "question": "Do same-day 5-minute features add value beyond daily price-volume and market state?",
        "minute_feature_increment_daily_rank_ic": increment,
        "minute_feature_increment_topk_exact_net_return": _topk_return_increments(evaluations),
        "mean_daily_score_rank_correlation": _mean_daily_score_rank_correlation(evaluations),
        "mean_daily_top10_name_overlap_fraction": _mean_top10_overlap(evaluations),
        "minute_feature_importance": _minute_feature_importance(data),
        "metrics": comparison.to_dict("records"),
        "files": {"summary": str(summary_path), "paired_daily": str(paired_path)},
        **cutoff_audit_fields(),
    }
    write_json(OUTPUT_ROOT / "tree_158_vs_183.json", _safe_json(output))
    return output
