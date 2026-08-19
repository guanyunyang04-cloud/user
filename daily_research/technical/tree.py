from __future__ import annotations

import gc
import json
import math
import os
import time
from collections.abc import Mapping
from datetime import datetime
from pathlib import Path
from typing import Any

import lightgbm as lgb
import numpy as np
import pandas as pd
import psutil

from .data import (
    OUTPUT_ROOT,
    TechnicalDataError,
    build_forward_folds,
    daily_rank_metrics,
    date_group_sizes,
    date_relevance_labels,
    equal_date_weights,
    fold_rows,
    load_data,
    open_array,
    stable_hash,
    write_json,
)
from .portfolio import (
    account_specs,
    add_account_diagnostics,
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
            block = self.matrix[
                int(selected[0]) : int(selected[-1]) + 1, : self.feature_count
            ]
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
        validation_start_date="2020-01-01",
        validation_end_date="2025-12-31",
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
            "input_fingerprint": data.model_manifest["input_fingerprint"],
            "view_fingerprint": data.view_manifest["fingerprint"],
            "target_fingerprint": data.target_manifest["fingerprint"],
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
            and result.get("forbidden_2026_read_count") == 0
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
        block = np.asarray(
            matrix[rows[left:right], : int(feature_count)], dtype=np.float32
        )
        output[left:right] = np.asarray(
            booster.predict(block, num_iteration=booster.best_iteration),
            dtype=np.float32,
        )
    return output


def train_fold(feature_count: int, fold_number: int) -> dict[str, Any]:
    data = load_data()
    data.feature_positions(int(feature_count))
    folds = _folds(data)
    if not 1 <= int(fold_number) <= len(folds):
        raise TechnicalDataError("fold number is outside 1..5")
    fold = folds[int(fold_number) - 1]
    feature_names = data.feature_names[: int(feature_count)]
    fingerprint = _task_fingerprint(data, feature_count, fold)
    root = _task_root(feature_count, fold_number)
    result_path = root / "result.json"
    cached = _task_complete(result_path, fingerprint)
    if cached is not None:
        return cached

    train_rows = fold_rows(data.dates, fold, "train")
    validation_rows = fold_rows(data.dates, fold, "validation")
    values, valid, column = data.target(TARGET_NAME)
    train_raw = np.asarray(values[train_rows, column], dtype=np.float32)
    validation_raw = np.asarray(values[validation_rows, column], dtype=np.float32)
    train_valid = np.asarray(valid[train_rows, column], dtype=bool) & np.isfinite(
        train_raw
    )
    validation_valid = np.asarray(
        valid[validation_rows, column], dtype=bool
    ) & np.isfinite(validation_raw)
    train_dates = data.dates[train_rows]
    validation_dates = data.dates[validation_rows]
    train_label = date_relevance_labels(train_dates, train_raw, train_valid)
    validation_label = date_relevance_labels(
        validation_dates, validation_raw, validation_valid
    )
    train_label[~train_valid] = 0.0
    validation_label[~validation_valid] = 0.0
    train_weight = equal_date_weights(train_dates, train_valid)
    validation_weight = equal_date_weights(validation_dates, validation_valid)

    available_mb = int(psutil.virtual_memory().available / (1 << 20))
    histogram_pool_mb = max(256, min(1024, (available_mb - 2048) // 4))
    parameters = {
        **TREE_PARAMETERS,
        "num_threads": min(16, os.cpu_count() or 1),
        "histogram_pool_size": histogram_pool_mb,
    }
    binary_params = {
        "max_bin": parameters["max_bin"],
        "data_random_seed": SEED,
        "feature_pre_filter": False,
        "num_threads": parameters["num_threads"],
        "verbosity": -1,
    }
    train_source = MatrixPrefixSequence(
        data.matrix, train_rows, feature_count, batch_size=32_768
    )
    validation_source = MatrixPrefixSequence(
        data.matrix, validation_rows, feature_count, batch_size=32_768
    )
    train_set = lgb.Dataset(
        train_source,
        label=train_label,
        weight=train_weight,
        group=date_group_sizes(train_dates),
        feature_name=list(feature_names),
        params=binary_params,
        free_raw_data=True,
    ).construct()
    validation_set = lgb.Dataset(
        validation_source,
        label=validation_label,
        weight=validation_weight,
        group=date_group_sizes(validation_dates),
        feature_name=list(feature_names),
        reference=train_set,
        params=binary_params,
        free_raw_data=True,
    ).construct()
    del train_source, validation_source
    gc.collect()

    print(
        json.dumps(
            {
                "event": "technical_tree_training_started",
                "feature_count": feature_count,
                "fold": fold_number,
                "train_rows": len(train_rows),
                "validation_rows": len(validation_rows),
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
    prediction = _predict(booster, data.matrix, validation_rows, feature_count)
    elapsed = time.perf_counter() - started
    daily, metrics = daily_rank_metrics(
        dates=validation_dates[validation_valid],
        actual=validation_raw[validation_valid],
        prediction=prediction[validation_valid],
    )

    root.mkdir(parents=True, exist_ok=True)
    model_path = root / "model.txt"
    partial_model = model_path.with_suffix(".txt.partial")
    booster.save_model(str(partial_model), num_iteration=booster.best_iteration)
    partial_model.replace(model_path)
    prediction_frame = data.row_index.iloc[validation_rows][
        ["candidate_id", "date_idx", "trade_date", "symbol", "symbol_idx"]
    ].copy()
    prediction_frame.insert(0, "row_position", validation_rows)
    prediction_frame["fold"] = int(fold_number)
    prediction_frame["score"] = prediction
    prediction_frame["actual"] = validation_raw
    prediction_frame["target_valid"] = validation_valid
    predictions_path = root / "predictions.parquet"
    prediction_frame.to_parquet(predictions_path, index=False)
    daily_path = root / "daily_metrics.parquet"
    daily.to_parquet(daily_path, index=False)
    importance = pd.DataFrame(
        {
            "feature_name": feature_names,
            "gain": booster.feature_importance(importance_type="gain"),
            "split": booster.feature_importance(importance_type="split"),
        }
    ).sort_values(["gain", "split"], ascending=False, kind="stable")
    importance_path = root / "feature_importance.parquet"
    importance.to_parquet(importance_path, index=False)
    result = {
        "status": "completed",
        "completed_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "fingerprint": fingerprint,
        "feature_count": int(feature_count),
        "feature_families": sorted(set(data.feature_families[: int(feature_count)])),
        "target": TARGET_NAME,
        "fold": fold,
        "best_iteration": int(booster.best_iteration),
        "metrics": metrics,
        "parameters": parameters,
        "elapsed_seconds": elapsed,
        "train_valid_count": int(train_valid.sum()),
        "validation_valid_count": int(validation_valid.sum()),
        "forbidden_2026_read_count": 0,
        "files": {
            "model": {"path": str(model_path)},
            "predictions": {"path": str(predictions_path)},
            "daily_metrics": {"path": str(daily_path)},
            "feature_importance": {"path": str(importance_path)},
        },
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
            raise TechnicalDataError(f"tree fold is missing: {path}")
        result = dict(json.loads(path.read_text(encoding="utf-8")))
        expected_fingerprint = _task_fingerprint(data, feature_count, folds[fold - 1])
        if (
            result.get("status") != "completed"
            or result.get("fingerprint") != expected_fingerprint
            or result.get("forbidden_2026_read_count") != 0
            or not all(
                Path(record["path"]).is_file()
                for record in result.get("files", {}).values()
            )
        ):
            raise TechnicalDataError(f"tree fold is stale or invalid: {path}")
        results.append(result)
        frames.append(pd.read_parquet(Path(result["files"]["predictions"]["path"])))
    oof = pd.concat(frames, ignore_index=True).sort_values(
        ["date_idx", "symbol_idx"], kind="stable"
    )
    if oof.duplicated("row_position").any():
        raise TechnicalDataError("OOF row positions overlap")
    return results, oof


def evaluate_predictions(
    *,
    run_name: str,
    fold_results: list[dict[str, Any]],
    oof: pd.DataFrame,
    metadata: Mapping[str, Any],
) -> dict[str, Any]:
    data = load_data()
    valid = (
        oof["target_valid"].astype(bool)
        & np.isfinite(oof["actual"])
        & np.isfinite(oof["score"])
    )
    daily, metrics = daily_rank_metrics(
        dates=oof.loc[valid, "date_idx"].to_numpy(dtype=np.int32),
        actual=oof.loc[valid, "actual"].to_numpy(dtype=np.float32),
        prediction=oof.loc[valid, "score"].to_numpy(dtype=np.float32),
    )
    ranked = oof[np.isfinite(oof["score"])].sort_values(
        ["date_idx", "score", "symbol_idx"],
        ascending=[True, False, True],
        kind="stable",
    )
    ranked["selection_rank"] = ranked.groupby("date_idx", sort=False).cumcount() + 1
    selections = ranked.loc[ranked["selection_rank"] <= 10].copy()

    path_manifest = data.path_target_manifest
    path_values = open_array(path_manifest["files"]["values"], np.float32)
    path_valid = open_array(path_manifest["files"]["valid"], np.uint8)
    fill_days = open_array(path_manifest["files"]["legal_fill_days"], np.int16)
    gross_column = list(path_manifest["target_columns"]).index("legal_exit_return_d10")
    fill_column = list(path_manifest["files"]["legal_fill_days"]["horizons"]).index(10)
    positions = selections["row_position"].to_numpy(dtype=np.int64)
    gross = np.asarray(path_values[positions, gross_column], dtype=np.float32)
    gross_valid = np.asarray(path_valid[positions, gross_column], dtype=bool)
    exits = np.asarray(fill_days[positions, fill_column], dtype=np.int16)
    usable = gross_valid & np.isfinite(gross) & (exits >= 10)
    dropped = int((~usable).sum())
    selections = selections.loc[usable].copy()
    selections["legal_gross_return"] = gross[usable]
    selections["fill_day"] = exits[usable]

    daily_raw = open_array(data.pack["feature_channels"]["daily_raw"], np.float32)
    raw_open = open_array(data.pack["execution_arrays"]["entry_open_raw"], np.float32)
    entry_filled = open_array(data.pack["masks"]["entry_filled"], np.bool_)
    costs = parse_execution_costs(data.pack)
    evaluation_root = OUTPUT_ROOT / str(run_name) / "evaluation"
    evaluation_root.mkdir(parents=True, exist_ok=True)
    account_results: list[dict[str, Any]] = []
    for spec in account_specs(10):
        result, equity, trades = simulate_account(
            spec=spec,
            selections=selections,
            date_values=data.date_values,
            symbol_values=data.pack["symbol_values"],
            cutoff_idx=data.cutoff_idx,
            daily_raw=daily_raw,
            raw_open=raw_open,
            entry_filled=entry_filled,
            costs=costs,
        )
        result = add_account_diagnostics(
            result=result, equity=equity, trades=trades, selections=selections
        )
        task_root = evaluation_root / result["task_id"]
        task_root.mkdir(parents=True, exist_ok=True)
        equity.to_parquet(task_root / "equity.parquet", index=False)
        trades.to_parquet(task_root / "trades.parquet", index=False)
        write_json(task_root / "result.json", _safe_json(result))
        account_results.append(result)

    oof_path = evaluation_root / "oof_predictions.parquet"
    selections_path = evaluation_root / "top10_selections.parquet"
    daily_path = evaluation_root / "daily_metrics.parquet"
    oof.to_parquet(oof_path, index=False)
    selections.to_parquet(selections_path, index=False)
    daily.to_parquet(daily_path, index=False)
    summary = pd.DataFrame(
        [
            {
                "task_id": item["task_id"],
                "top_k": item["spec"]["top_k"],
                "cost_scenario": item["spec"]["cost_scenario"],
                "maximum_credited_gross_return": item["spec"].get(
                    "maximum_credited_gross_return"
                ),
                "allow_overlapping_same_symbol": item["spec"].get(
                    "allow_overlapping_same_symbol", True
                ),
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
    summary_path = evaluation_root / "account_summary.parquet"
    summary.to_parquet(summary_path, index=False)
    result = {
        "status": "completed",
        "completed_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        **dict(metadata),
        "input_fingerprint": data.model_manifest["input_fingerprint"],
        "view_fingerprint": data.view_manifest["fingerprint"],
        "target_fingerprint": data.target_manifest["fingerprint"],
        "fold_fingerprints": [item.get("fingerprint") for item in fold_results],
        "fold_metrics": [item["metrics"] for item in fold_results],
        "oof_metrics": metrics,
        "selected_row_count": len(selections),
        "right_censored_top10_selection_count": dropped,
        "accounts": account_results,
        "forbidden_2026_read_count": 0,
        "files": {
            "oof_predictions": {"path": str(oof_path)},
            "top10_selections": {"path": str(selections_path)},
            "daily_metrics": {"path": str(daily_path)},
            "account_summary": {"path": str(summary_path)},
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
        and (not bool(item["spec"].get("allow_overlapping_same_symbol", True)))
        == bool(no_overlap)
    ]
    if len(matches) != 1:
        raise TechnicalDataError(f"account result is not unique: {cap}, {no_overlap}")
    return matches[0]


def compare() -> dict[str, Any]:
    data = load_data()
    results: dict[int, dict[str, Any]] = {}
    rows: list[dict[str, Any]] = []
    daily_frames: dict[int, pd.DataFrame] = {}
    oof_frames: dict[int, pd.DataFrame] = {}
    selection_frames: dict[int, pd.DataFrame] = {}
    for count in (158, 183):
        path = OUTPUT_ROOT / f"tree_{count}/evaluation/result.json"
        if not path.is_file():
            raise TechnicalDataError(f"evaluation is missing: {path}")
        result = dict(json.loads(path.read_text(encoding="utf-8")))
        results[count] = result
        daily_frames[count] = pd.read_parquet(
            Path(result["files"]["daily_metrics"]["path"])
        )
        oof_frames[count] = pd.read_parquet(
            Path(result["files"]["oof_predictions"]["path"]),
            columns=["row_position", "date_idx", "score"],
        )
        selection_frames[count] = pd.read_parquet(
            Path(result["files"]["top10_selections"]["path"])
        )
        raw = _find_account(result, cap=None, no_overlap=False)
        capped = _find_account(result, cap=0.10, no_overlap=False)
        no_overlap = _find_account(result, cap=None, no_overlap=True)
        robust = _find_account(result, cap=0.10, no_overlap=True)
        rows.append(
            {
                "feature_count": count,
                "rank_ic": result["oof_metrics"]["daily_rank_ic_mean"],
                "rank_ic_positive_fraction": result["oof_metrics"][
                    "daily_rank_ic_positive_fraction"
                ],
                "top10_stress_return": raw["total_net_return"],
                "top10_stress_drawdown": raw["maximum_drawdown"],
                "top10_stress_cap10_return": capped["total_net_return"],
                "top10_no_overlap_return": no_overlap["total_net_return"],
                "top10_no_overlap_drawdown": no_overlap["maximum_drawdown"],
                "top10_no_overlap_cap10_return": robust["total_net_return"],
                "positive_years": no_overlap["positive_year_count"],
                "no_overlap_annual": no_overlap["annual"],
            }
        )
    comparison = pd.DataFrame(rows).sort_values("feature_count")
    path = OUTPUT_ROOT / "tree_158_vs_183.parquet"
    comparison.to_parquet(path, index=False)
    left = daily_frames[158][["date_idx", "rank_ic"]].rename(
        columns={"rank_ic": "rank_ic_158"}
    )
    right = daily_frames[183][["date_idx", "rank_ic"]].rename(
        columns={"rank_ic": "rank_ic_183"}
    )
    paired = left.merge(right, on="date_idx", validate="one_to_one")
    paired["minute_rank_ic_increment"] = paired["rank_ic_183"] - paired["rank_ic_158"]
    from .portfolio import newey_west_interval

    increment = newey_west_interval(
        paired["minute_rank_ic_increment"].to_numpy(dtype=np.float64), lag=20
    )
    paired_path = OUTPUT_ROOT / "tree_158_vs_183_daily.parquet"
    paired.to_parquet(paired_path, index=False)

    score_pairs = oof_frames[158].merge(
        oof_frames[183],
        on=["row_position", "date_idx"],
        suffixes=("_158", "_183"),
        validate="one_to_one",
    )
    score_pairs["rank_158"] = score_pairs.groupby("date_idx", sort=False)[
        "score_158"
    ].rank(pct=True)
    score_pairs["rank_183"] = score_pairs.groupby("date_idx", sort=False)[
        "score_183"
    ].rank(pct=True)
    rank_correlation = score_pairs.groupby("date_idx")[["rank_158", "rank_183"]].corr()
    rank_correlation = rank_correlation.iloc[0::2, -1].to_numpy(dtype=np.float64)

    topk_increments: dict[str, Any] = {}
    for top_k in (1, 3, 5, 10):
        daily_topk: dict[int, pd.Series] = {}
        for count in (158, 183):
            current = selection_frames[count]
            current = current.loc[current["selection_rank"] <= top_k]
            daily_topk[count] = current.groupby("date_idx")["actual"].mean()
        joined = pd.concat(
            [daily_topk[158].rename("net_158"), daily_topk[183].rename("net_183")],
            axis=1,
            join="inner",
        ).dropna()
        difference = joined["net_183"].to_numpy(dtype=np.float64) - joined[
            "net_158"
        ].to_numpy(dtype=np.float64)
        topk_increments[str(top_k)] = newey_west_interval(difference, lag=20)

    top10_left = selection_frames[158][["date_idx", "candidate_id"]]
    top10_right = selection_frames[183][["date_idx", "candidate_id"]]
    overlap = (
        top10_left.merge(
            top10_right,
            on=["date_idx", "candidate_id"],
            how="inner",
        )
        .groupby("date_idx")
        .size()
    )

    minute_names = set(data.feature_names[158:])
    minute_importance_rows: list[dict[str, Any]] = []
    for fold in range(1, 6):
        importance = pd.read_parquet(
            _task_root(183, fold) / "feature_importance.parquet"
        )
        minute = importance["feature_name"].isin(minute_names)
        total_gain = float(importance["gain"].sum())
        total_split = float(importance["split"].sum())
        minute_importance_rows.append(
            {
                "fold": fold,
                "minute_gain_share": (
                    float(importance.loc[minute, "gain"].sum() / total_gain)
                    if total_gain > 0.0
                    else 0.0
                ),
                "minute_split_share": (
                    float(importance.loc[minute, "split"].sum() / total_split)
                    if total_split > 0.0
                    else 0.0
                ),
                "minute_features_with_gain": int(
                    (importance.loc[minute, "gain"] > 0.0).sum()
                ),
            }
        )
    output = {
        "status": "completed",
        "completed_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "question": "Do same-day 5-minute features add value beyond daily price-volume and market state?",
        "minute_feature_increment_daily_rank_ic": increment,
        "minute_feature_increment_topk_exact_net_return": topk_increments,
        "mean_daily_score_rank_correlation": float(np.nanmean(rank_correlation)),
        "mean_daily_top10_name_overlap_fraction": float(overlap.mean() / 10.0),
        "minute_feature_importance": minute_importance_rows,
        "metrics": comparison.to_dict("records"),
        "files": {"summary": str(path), "paired_daily": str(paired_path)},
        "forbidden_2026_read_count": 0,
    }
    write_json(OUTPUT_ROOT / "tree_158_vs_183.json", _safe_json(output))
    return output
