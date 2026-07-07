from __future__ import annotations

import argparse
import gc
import json
import math
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd

from daily_research.path_policy.qdp_v2_sequence_path_pack import _json_default, _write_json
from daily_research.path_policy.qdp_v2_sequence_path_training import (
    DEFAULT_OUTPUT_ROOT,
    DEFAULT_SEED,
    DEFAULT_TOP_K,
    SequencePathPackDataset,
    _daily_spearman,
    _parse_int_list,
    _topk_metrics,
)


def _now() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def _select_indices(
    dataset: SequencePathPackDataset,
    *,
    samples_per_date: int,
    max_samples: int,
    seed: int,
) -> np.ndarray:
    frame = dataset.sample_index[["date_idx"]].copy()
    frame["dataset_idx"] = np.arange(len(frame), dtype=np.int64)
    rng = np.random.default_rng(int(seed))
    selected: list[np.ndarray] = []
    if int(samples_per_date) > 0:
        for _date_idx, group in frame.groupby("date_idx", sort=True):
            values = group["dataset_idx"].to_numpy(dtype=np.int64, copy=False)
            if len(values) > int(samples_per_date):
                values = rng.choice(values, size=int(samples_per_date), replace=False)
            selected.append(np.sort(values))
        idx = np.concatenate(selected) if selected else np.asarray([], dtype=np.int64)
    else:
        idx = frame["dataset_idx"].to_numpy(dtype=np.int64, copy=True)
    if int(max_samples) > 0 and len(idx) > int(max_samples):
        idx = rng.choice(idx, size=int(max_samples), replace=False)
    return np.sort(idx.astype(np.int64, copy=False))


def _feature_names(dataset: SequencePathPackDataset) -> list[str]:
    names: list[str] = []
    for t in range(int(dataset.lookback_days)):
        rel = t - int(dataset.lookback_days) + 1
        prefix = "t0" if rel == 0 else f"t{rel}"
        for channel in dataset.channel_order:
            for col in dataset.feature_columns[channel]:
                names.append(f"{prefix}__{channel}__{col}")
    return names


def _estimate_dense_gb(row_count: int, feature_count: int) -> float:
    return float(int(row_count) * int(feature_count) * np.dtype("float32").itemsize) / (1024.0**3)


def _materialize_flat_features(
    dataset: SequencePathPackDataset,
    indices: np.ndarray,
    *,
    batch_size: int,
    progress_path: Path,
    stage: str,
) -> tuple[np.ndarray, np.ndarray]:
    indices = np.asarray(indices, dtype=np.int64)
    rows = int(len(indices))
    feature_count = int(dataset.lookback_days) * int(dataset.input_dim)
    x = np.empty((rows, feature_count), dtype=np.float32)
    y = np.empty(rows, dtype=np.float32)
    cursor = 0
    for start in range(0, rows, int(batch_size)):
        chunk_idx = indices[start : start + int(batch_size)]
        batch = dataset.get_batch(chunk_idx)
        batch_x = batch["x"].numpy().reshape(len(chunk_idx), feature_count)
        batch_y = batch["y_summary"].numpy()[:, int(dataset.value_index)]
        size = int(len(chunk_idx))
        x[cursor : cursor + size, :] = batch_x
        y[cursor : cursor + size] = batch_y.astype(np.float32, copy=False)
        cursor += size
        if cursor == rows or cursor % max(int(batch_size) * 20, 1) == 0:
            _write_json(
                progress_path,
                {
                    "status": stage,
                    "rows_done": int(cursor),
                    "rows_total": int(rows),
                    "updated_at": _now(),
                },
            )
        del batch, batch_x, batch_y
        gc.collect()
    return x, y


def _predict_split(
    *,
    dataset: SequencePathPackDataset,
    model: Any,
    feature_names: list[str],
    split: str,
    output_dir: Path,
    batch_size: int,
    top_k: tuple[int, ...],
    write_predictions: bool,
    max_eval_samples: int,
    progress_path: Path,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    pred_dir = output_dir / "predictions"
    if write_predictions:
        pred_dir.mkdir(parents=True, exist_ok=True)
    pred_path = pred_dir / f"{split}_flat_lgbm_predictions.csv"
    if write_predictions and pred_path.exists():
        pred_path.unlink()
    all_indices = np.arange(len(dataset), dtype=np.int64)
    if int(max_eval_samples) > 0 and len(all_indices) > int(max_eval_samples):
        rng = np.random.default_rng(DEFAULT_SEED)
        all_indices = np.sort(rng.choice(all_indices, size=int(max_eval_samples), replace=False))
    metric_chunks: list[pd.DataFrame] = []
    first_write = True
    feature_count = int(dataset.lookback_days) * int(dataset.input_dim)
    for start in range(0, len(all_indices), int(batch_size)):
        idx = all_indices[start : start + int(batch_size)]
        batch = dataset.get_batch(idx)
        x = batch["x"].numpy().reshape(len(idx), feature_count)
        score = model.predict(x, num_iteration=model.best_iteration_).astype("float32")
        y_summary = batch["y_summary"].numpy()
        rows: dict[str, Any] = {
            "trade_date": list(batch["trade_date"]),
            "symbol": list(batch["symbol"]),
            "score": score,
        }
        for col_idx, col in enumerate(dataset.path_summary_columns):
            rows[col] = y_summary[:, col_idx]
        chunk = pd.DataFrame(rows)
        metric_chunks.append(chunk)
        if write_predictions:
            chunk.to_csv(pred_path, index=False, mode="w" if first_write else "a", header=first_write, encoding="utf-8-sig")
            first_write = False
        if start == 0 or (start // max(int(batch_size), 1)) % 100 == 0:
            _write_json(
                progress_path,
                {
                    "status": "predicting",
                    "split": split,
                    "rows_done": int(min(start + len(idx), len(all_indices))),
                    "rows_total": int(len(all_indices)),
                    "updated_at": _now(),
                },
            )
        del batch, x, score, y_summary, chunk
        gc.collect()
    frame = pd.concat(metric_chunks, ignore_index=True)
    ic = _daily_spearman(frame, score_col="score", target_col=dataset.value_column)
    topk = _topk_metrics(frame, top_k_values=top_k, forward_days=dataset.forward_days, value_column=dataset.value_column)
    metrics = {
        "split": split,
        "row_count": int(len(frame)),
        "date_count": int(frame["trade_date"].nunique()),
        "rank_ic_mean": float(ic["rank_ic"].mean()) if not ic.empty else math.nan,
        "rank_ic_median": float(ic["rank_ic"].median()) if not ic.empty else math.nan,
        "rank_ic_positive_day_rate": float((ic["rank_ic"] > 0).mean()) if not ic.empty else math.nan,
        "target_mean": float(frame[dataset.value_column].mean()),
        "prediction_mean": float(frame["score"].mean()),
        "prediction_csv": str(pred_path.resolve()) if write_predictions else "",
        "max_eval_samples": int(max_eval_samples),
    }
    return ic, topk, metrics


@dataclass(frozen=True)
class FlatLgbmConfig:
    pack_manifest: Path
    output_root: Path
    run_tag: str
    train_samples_per_date: int
    max_train_samples: int
    validation_fit_samples_per_date: int
    max_validation_fit_samples: int
    max_eval_samples_per_split: int
    batch_size: int
    top_k: tuple[int, ...]
    seed: int
    n_estimators: int
    learning_rate: float
    num_leaves: int
    n_jobs: int
    max_dense_matrix_gb: float
    allow_full_flat_train: bool
    write_predictions: bool


def train_sequence_flat_lgbm(config: FlatLgbmConfig) -> dict[str, Any]:
    manifest = json.loads(Path(config.pack_manifest).read_text(encoding="utf-8"))
    output_dir = config.output_root / f"{config.run_tag}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    output_dir.mkdir(parents=True, exist_ok=True)
    progress_path = output_dir / "progress.json"
    _write_json(progress_path, {"status": "loading_datasets", "updated_at": _now()})
    train_ds = SequencePathPackDataset(manifest, split="train")
    val_ds = SequencePathPackDataset(manifest, split="validation")
    test_ds = SequencePathPackDataset(manifest, split="test")
    names = _feature_names(train_ds)
    train_idx = _select_indices(
        train_ds,
        samples_per_date=int(config.train_samples_per_date),
        max_samples=int(config.max_train_samples),
        seed=int(config.seed),
    )
    val_fit_idx = _select_indices(
        val_ds,
        samples_per_date=int(config.validation_fit_samples_per_date),
        max_samples=int(config.max_validation_fit_samples),
        seed=int(config.seed) + 17,
    )
    full_train_requested = (
        int(config.train_samples_per_date) <= 0
        and int(config.max_train_samples) <= 0
        and len(train_idx) == len(train_ds)
    )
    dense_gb = _estimate_dense_gb(len(train_idx), len(names))
    if full_train_requested and not bool(config.allow_full_flat_train):
        raise RuntimeError(
            f"full flat train would materialize about {dense_gb:.2f}GB before LightGBM binning; "
            "pass --allow-full-flat-train explicitly or use date-balanced sampling"
        )
    if dense_gb > float(config.max_dense_matrix_gb):
        raise RuntimeError(
            f"selected flat train matrix is {dense_gb:.2f}GB, above --max-dense-matrix-gb={config.max_dense_matrix_gb}; "
            "lower --train-samples-per-date or --max-train-samples"
        )
    _write_json(
        progress_path,
        {
            "status": "materializing_train",
            "train_rows": int(len(train_idx)),
            "validation_fit_rows": int(len(val_fit_idx)),
            "feature_count": int(len(names)),
            "estimated_train_dense_gb": float(dense_gb),
            "updated_at": _now(),
        },
    )
    x_train, y_train = _materialize_flat_features(
        train_ds,
        train_idx,
        batch_size=int(config.batch_size),
        progress_path=progress_path,
        stage="materializing_train",
    )
    x_val_fit, y_val_fit = _materialize_flat_features(
        val_ds,
        val_fit_idx,
        batch_size=int(config.batch_size),
        progress_path=progress_path,
        stage="materializing_validation_fit",
    )
    from lightgbm import LGBMRegressor, early_stopping, log_evaluation

    model = LGBMRegressor(
        objective="regression_l2",
        n_estimators=int(config.n_estimators),
        learning_rate=float(config.learning_rate),
        num_leaves=int(config.num_leaves),
        min_child_samples=100,
        subsample=0.85,
        colsample_bytree=0.65,
        reg_lambda=3.0,
        random_state=int(config.seed),
        n_jobs=int(config.n_jobs),
        verbosity=-1,
    )
    _write_json(progress_path, {"status": "training", "updated_at": _now()})
    model.fit(
        x_train,
        y_train,
        eval_set=[(x_val_fit, y_val_fit)],
        eval_metric="l2",
        callbacks=[early_stopping(50, verbose=False), log_evaluation(100)],
    )
    del x_train, y_train, x_val_fit, y_val_fit
    gc.collect()
    val_ic, val_topk, val_metrics = _predict_split(
        dataset=val_ds,
        model=model,
        feature_names=names,
        split="validation",
        output_dir=output_dir,
        batch_size=int(config.batch_size),
        top_k=config.top_k,
        write_predictions=bool(config.write_predictions),
        max_eval_samples=int(config.max_eval_samples_per_split),
        progress_path=progress_path,
    )
    test_ic, test_topk, test_metrics = _predict_split(
        dataset=test_ds,
        model=model,
        feature_names=names,
        split="test",
        output_dir=output_dir,
        batch_size=int(config.batch_size),
        top_k=config.top_k,
        write_predictions=bool(config.write_predictions),
        max_eval_samples=int(config.max_eval_samples_per_split),
        progress_path=progress_path,
    )
    split_metrics = pd.DataFrame([val_metrics, test_metrics])
    topk = pd.concat([val_topk.assign(split="validation"), test_topk.assign(split="test")], ignore_index=True)
    daily_ic = pd.concat([val_ic.assign(split="validation"), test_ic.assign(split="test")], ignore_index=True)
    feature_importance = pd.DataFrame(
        {
            "feature": names,
            "importance_gain": model.booster_.feature_importance(importance_type="gain"),
            "importance_split": model.booster_.feature_importance(importance_type="split"),
        }
    ).sort_values(["importance_gain", "importance_split"], ascending=False, kind="mergesort")
    outputs = {
        "split_metrics_csv": str((output_dir / "split_metrics.csv").resolve()),
        "topk_metrics_csv": str((output_dir / "topk_metrics.csv").resolve()),
        "daily_rank_ic_csv": str((output_dir / "daily_rank_ic.csv").resolve()),
        "feature_importance_csv": str((output_dir / "feature_importance.csv").resolve()),
    }
    split_metrics.to_csv(outputs["split_metrics_csv"], index=False, encoding="utf-8-sig")
    topk.to_csv(outputs["topk_metrics_csv"], index=False, encoding="utf-8-sig")
    daily_ic.to_csv(outputs["daily_rank_ic_csv"], index=False, encoding="utf-8-sig")
    feature_importance.to_csv(outputs["feature_importance_csv"], index=False, encoding="utf-8-sig")
    summary = {
        "artifact_type": "qdp_v2_sequence_flat_lgbm",
        "generated_at": _now(),
        "pack_manifest": str(Path(config.pack_manifest).resolve()),
        "output_dir": str(output_dir.resolve()),
        "lookback_days": int(train_ds.lookback_days),
        "forward_days": int(train_ds.forward_days),
        "value_column": str(train_ds.value_column),
        "input_dim_per_day": int(train_ds.input_dim),
        "flat_feature_count": int(len(names)),
        "train_rows": int(len(train_idx)),
        "validation_fit_rows": int(len(val_fit_idx)),
        "estimated_train_dense_gb": float(dense_gb),
        "model": {
            "type": "sequence_flat_LGBMRegressor",
            "best_iteration": int(model.best_iteration_ or config.n_estimators),
            "n_estimators": int(config.n_estimators),
            "learning_rate": float(config.learning_rate),
            "num_leaves": int(config.num_leaves),
            "n_jobs": int(config.n_jobs),
            "train_samples_per_date": int(config.train_samples_per_date),
            "max_train_samples": int(config.max_train_samples),
        },
        "split_metrics": split_metrics.to_dict("records"),
        "outputs": outputs,
    }
    summary_path = output_dir / "sequence_flat_lgbm_summary.json"
    _write_json(summary_path, summary)
    _write_json(progress_path, {"status": "completed", "summary_json": str(summary_path.resolve()), "updated_at": _now()})
    return summary


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train a flat LightGBM baseline from QDP v2 sequence pack inputs.")
    parser.add_argument("--pack-manifest", type=Path, default=None)
    parser.add_argument("--store-view", type=Path, default=None, help="Alias for a lightweight research_store view manifest.")
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--run-tag", default="qdp_v2_sequence_flat_lgbm")
    parser.add_argument("--train-samples-per-date", type=int, default=64)
    parser.add_argument("--max-train-samples", type=int, default=0)
    parser.add_argument("--validation-fit-samples-per-date", type=int, default=128)
    parser.add_argument("--max-validation-fit-samples", type=int, default=200000)
    parser.add_argument("--max-eval-samples-per-split", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--top-k", default="5,10,20,50,100")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--n-estimators", type=int, default=500)
    parser.add_argument("--learning-rate", type=float, default=0.035)
    parser.add_argument("--num-leaves", type=int, default=63)
    parser.add_argument("--n-jobs", type=int, default=4)
    parser.add_argument("--max-dense-matrix-gb", type=float, default=8.0)
    parser.add_argument("--allow-full-flat-train", action="store_true")
    parser.add_argument("--write-predictions", dest="write_predictions", action="store_true", default=False)
    parser.add_argument("--no-predictions", dest="write_predictions", action="store_false")
    parser.add_argument("--json", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    manifest_path = Path(args.store_view or args.pack_manifest) if (args.store_view or args.pack_manifest) else None
    if manifest_path is None:
        raise SystemExit("requires --pack-manifest or --store-view")
    summary = train_sequence_flat_lgbm(
        FlatLgbmConfig(
            pack_manifest=manifest_path,
            output_root=Path(args.output_root),
            run_tag=str(args.run_tag),
            train_samples_per_date=int(args.train_samples_per_date),
            max_train_samples=int(args.max_train_samples),
            validation_fit_samples_per_date=int(args.validation_fit_samples_per_date),
            max_validation_fit_samples=int(args.max_validation_fit_samples),
            max_eval_samples_per_split=int(args.max_eval_samples_per_split),
            batch_size=int(args.batch_size),
            top_k=_parse_int_list(args.top_k, default=DEFAULT_TOP_K),
            seed=int(args.seed),
            n_estimators=int(args.n_estimators),
            learning_rate=float(args.learning_rate),
            num_leaves=int(args.num_leaves),
            n_jobs=int(args.n_jobs),
            max_dense_matrix_gb=float(args.max_dense_matrix_gb),
            allow_full_flat_train=bool(args.allow_full_flat_train),
            write_predictions=bool(args.write_predictions),
        )
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2, default=_json_default) if bool(args.json) else summary)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
