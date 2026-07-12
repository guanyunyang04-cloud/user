from __future__ import annotations

import argparse
import json
import math
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping, Sequence

import lightgbm as lgb
import numpy as np

from daily_research.path_policy.seq100_qcurve import QCURVE_ENTRY_HORIZONS, QCURVE_HOLD_HORIZONS
from daily_research.path_policy.seq100_qcurve_backtest import evaluate_qcurve_fold
from daily_research.path_policy.seq100_qcurve_data import QCurvePack, _read_json, verify_qcurve_development_fold
from daily_research.path_policy.seq100_qcurve_training import LOSS_WEIGHT_MAP, _soft_reclaim


ENTER_ANCHORS = (2, 5, 10, 20, 40, 60)
HOLD_ANCHORS = (1, 2, 5, 10, 20, 40, 60)
HEADS = ("mean", "q20", "q50", "q80", "p_positive")
DEFAULT_SEED = 7
DEFAULT_MAX_EPOCHS = 10
DEFAULT_PATIENCE = 2
RANK_BLEND_WEIGHT = 0.25


def _now() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def _write_json(path: str | Path, payload: Mapping[str, Any]) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(target.suffix + ".tmp")
    temporary.write_text(
        json.dumps(dict(payload), ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, target)
    return target


def _contiguous_split(spans: Sequence[Mapping[str, Any]]) -> tuple[int, int, int]:
    start = int(spans[0]["candidate_start"])
    stop = int(spans[-1]["candidate_stop"])
    count = sum(int(item["candidate_stop"]) - int(item["candidate_start"]) for item in spans)
    if stop - start != count:
        raise ValueError("LightGBM split candidate ranges must be contiguous")
    return start, stop, count


def _feature_matrix(
    *,
    pack: QCurvePack,
    fold: Mapping[str, Any],
    spans: Sequence[Mapping[str, Any]],
    path: Path,
) -> np.memmap:
    source_start, source_stop, count = _contiguous_split(spans)
    feature_count = (int(pack.daily_state.shape[2]) + int(pack.ma_state.shape[2])) * 2
    expected_size = count * feature_count * np.dtype("float32").itemsize
    marker = path.with_suffix(path.suffix + ".complete.json")
    if path.is_file() and path.stat().st_size == expected_size and marker.is_file():
        return np.memmap(path, dtype="float32", mode="r", shape=(count, feature_count))
    path.parent.mkdir(parents=True, exist_ok=True)
    output = np.memmap(path, dtype="float32", mode="w+", shape=(count, feature_count))
    offset = 0
    for span in spans:
        start, stop, date_idx = int(span["candidate_start"]), int(span["candidate_stop"]), int(span["date_idx"])
        symbols = pack.candidate_symbols(start, stop)
        values = pack.current_lgbm_features(
            date_idx=date_idx,
            symbols=symbols,
            normalization=fold["normalization"]["inputs"],
        )
        output[offset : offset + symbols.size] = values
        offset += int(symbols.size)
        if offset % 500_000 < int(symbols.size):
            output.flush()
    if offset != count or source_stop - source_start != count:
        raise ValueError("LightGBM feature materialization row-count mismatch")
    output.flush()
    _write_json(marker, {"status": "completed", "shape": [count, feature_count], "completed_at": _now()})
    return np.memmap(path, dtype="float32", mode="r", shape=(count, feature_count))


def _model_key(action: str, horizon: int, head: str) -> str:
    return f"{action}_h{int(horizon)}_{head}"


def _target_vector(
    pack: QCurvePack,
    *,
    action: str,
    horizon: int,
    source_start: int,
    source_stop: int,
) -> np.ndarray:
    base = 2 if action == "enter" else 1
    return np.asarray(
        pack.target_arrays[f"base_{action}_net_log_return"][int(horizon) - base, source_start:source_stop],
        dtype=np.float32,
    )


def _rank_relevance(target: np.ndarray, groups: Sequence[int]) -> np.ndarray:
    result = np.zeros(target.size, dtype=np.int32)
    offset = 0
    for count in groups:
        values = target[offset : offset + int(count)]
        order = np.argsort(values, kind="mergesort")
        rank = np.empty(int(count), dtype=np.int32)
        rank[order] = np.floor(np.arange(int(count)) * 10.0 / max(int(count), 1)).astype(np.int32)
        descending = order[::-1]
        rank[descending[: min(32, int(count))]] = 12
        rank[descending[: min(3, int(count))]] = 15
        result[offset : offset + int(count)] = rank
        offset += int(count)
    return result


def _objective(head: str) -> dict[str, Any]:
    if head == "mean":
        return {"objective": "huber", "metric": "huber"}
    if head == "p_positive":
        return {"objective": "binary", "metric": "binary_logloss"}
    quantile = {"q20": 0.20, "q50": 0.50, "q80": 0.80}[head]
    return {"objective": "quantile", "alpha": quantile, "metric": "quantile"}


def _interpolate(values: np.ndarray, anchors: Sequence[int], horizons: Sequence[int]) -> np.ndarray:
    source = np.asarray(values, dtype=np.float32)
    anchor_values = np.asarray(anchors, dtype=np.float32)
    result = np.empty((source.shape[0], len(horizons)), dtype=np.float32)
    for column, horizon in enumerate(horizons):
        value = float(horizon)
        if value <= anchor_values[0]:
            result[:, column] = source[:, 0]
            continue
        if value >= anchor_values[-1]:
            result[:, column] = source[:, -1]
            continue
        upper = int(np.searchsorted(anchor_values, value, side="right"))
        lower = upper - 1
        weight = (value - float(anchor_values[lower])) / float(anchor_values[upper] - anchor_values[lower])
        result[:, column] = source[:, lower] * (1.0 - weight) + source[:, upper] * weight
    return result


def _anchor_predictions(
    *,
    boosters: Mapping[str, lgb.Booster],
    features: np.ndarray,
    iteration: int,
    rank_calibration: Mapping[str, tuple[float, float]],
    target_stats: Mapping[str, Any],
) -> dict[str, np.ndarray]:
    output: dict[str, np.ndarray] = {}
    for action, anchors in (("enter", ENTER_ANCHORS), ("hold", HOLD_ANCHORS)):
        for head in HEADS:
            columns = [
                np.asarray(
                    boosters[_model_key(action, horizon, head)].predict(features, num_iteration=int(iteration)),
                    dtype=np.float32,
                )
                for horizon in anchors
            ]
            values = np.column_stack(columns)
            if head != "p_positive":
                base = 2 if action == "enter" else 1
                locations = np.asarray(
                    [target_stats[action]["median"][int(horizon) - base] for horizon in anchors],
                    dtype=np.float32,
                )
                scales = np.asarray(
                    [target_stats[action]["scale"][int(horizon) - base] for horizon in anchors],
                    dtype=np.float32,
                )
                values = values * scales + locations
            if action == "enter" and head == "mean":
                ranked: list[np.ndarray] = []
                for column, horizon in enumerate(anchors):
                    rank_score = np.asarray(
                        boosters[_model_key("enter", horizon, "rank")].predict(
                            features,
                            num_iteration=int(iteration),
                        ),
                        dtype=np.float32,
                    )
                    intercept, slope = rank_calibration[str(int(horizon))]
                    ranked.append(intercept + slope * rank_score)
                calibrated = np.column_stack(ranked)
                values = (1.0 - RANK_BLEND_WEIGHT) * values + RANK_BLEND_WEIGHT * calibrated
            output[f"{action}_{head}"] = values
    return output


def _curves_from_anchors(anchor_predictions: Mapping[str, np.ndarray]) -> dict[str, np.ndarray]:
    result: dict[str, np.ndarray] = {}
    for action, anchors, horizons in (
        ("enter", ENTER_ANCHORS, QCURVE_ENTRY_HORIZONS),
        ("hold", HOLD_ANCHORS, QCURVE_HOLD_HORIZONS),
    ):
        for head in HEADS:
            result[f"{action}_{head}"] = _interpolate(anchor_predictions[f"{action}_{head}"], anchors, horizons)
        q50 = result[f"{action}_q50"]
        result[f"{action}_q20"] = np.minimum(result[f"{action}_q20"], q50)
        result[f"{action}_q80"] = np.maximum(result[f"{action}_q80"], q50)
        result[f"{action}_p_positive"] = np.clip(result[f"{action}_p_positive"], 1.0e-6, 1.0 - 1.0e-6)
    return result


def _huber(values: np.ndarray) -> np.ndarray:
    absolute = np.abs(values)
    return np.where(absolute < 1.0, 0.5 * np.square(values), absolute - 0.5)


def _pinball(prediction: np.ndarray, target: np.ndarray, quantile: float) -> np.ndarray:
    error = target - prediction
    return np.maximum(float(quantile) * error, (float(quantile) - 1.0) * error)


def _numpy_rank_loss(score: np.ndarray, target: np.ndarray) -> float:
    losses: list[float] = []
    for horizon in range(score.shape[1]):
        order = np.argsort(target[:, horizon])[::-1]
        count = min(32, order.size - 1)
        if count <= 0:
            continue
        top, rest = order[:count], order[count:]
        gap = target[top, horizon, None] - target[rest, horizon][None, :]
        positive = gap > 0.0
        if not bool(positive.any()):
            continue
        gap_scale = max(float(gap[positive].mean()), 1.0e-6)
        gap_weight = np.clip(gap / gap_scale, 0.25, 4.0)
        rank_weight = np.ones((count, 1), dtype=np.float64)
        rank_weight[: min(3, count)] = 4.0
        difference = score[top, horizon, None] - score[rest, horizon][None, :]
        raw = np.logaddexp(0.0, -difference) * gap_weight * rank_weight
        losses.append(float(raw[positive].mean()))
    return float(np.mean(losses)) if losses else 0.0


def _numpy_day_components(
    curves: Mapping[str, np.ndarray],
    *,
    raw_enter: np.ndarray,
    raw_hold: np.ndarray,
    target_stats: Mapping[str, Any],
) -> dict[str, float]:
    components = {name: [] for name in ("mean_loss", "quantile_loss", "positive_loss", "soft_exit_loss")}
    for action, raw_target in (("enter", raw_enter), ("hold", raw_hold)):
        location = np.asarray(target_stats[action]["median"], dtype=np.float64)
        scale = np.asarray(target_stats[action]["scale"], dtype=np.float64)
        target = (raw_target - location) / scale
        mean = (curves[f"{action}_mean"] - location) / scale
        q20 = (curves[f"{action}_q20"] - location) / scale
        q50 = (curves[f"{action}_q50"] - location) / scale
        q80 = (curves[f"{action}_q80"] - location) / scale
        probability = curves[f"{action}_p_positive"]
        components["mean_loss"].append(float(_huber(mean - target).mean()))
        components["quantile_loss"].append(
            float((_pinball(q20, target, 0.2) + _pinball(q50, target, 0.5) + _pinball(q80, target, 0.8)).mean() / 3.0)
        )
        positive = (raw_target > 0.0).astype(np.float64)
        components["positive_loss"].append(
            float(-(positive * np.log(probability) + (1.0 - positive) * np.log1p(-probability)).mean())
        )
        logits = curves[f"{action}_mean"] / 0.03
        logits -= np.max(logits, axis=1, keepdims=True)
        policy = np.exp(logits)
        policy /= np.maximum(policy.sum(axis=1, keepdims=True), 1.0e-12)
        components["soft_exit_loss"].append(float(-np.mean(np.sum(policy * raw_target, axis=1))))
    anchors = np.asarray([value - 2 for value in ENTER_ANCHORS], dtype=np.int64)
    return {
        **{name: float(np.mean(values)) for name, values in components.items()},
        "rank_loss": _numpy_rank_loss(curves["enter_mean"][:, anchors], raw_enter[:, anchors]),
        "structured_price_trend_aux_loss": 0.0,
        "va_vwap_aux_loss": 0.0,
    }


def _rank_calibration(
    *,
    boosters: Mapping[str, lgb.Booster],
    train_features: np.ndarray,
    pack: QCurvePack,
    train_source_start: int,
    train_source_stop: int,
    iteration: int,
) -> dict[str, tuple[float, float]]:
    count = train_source_stop - train_source_start
    sample = np.linspace(0, count - 1, min(count, 100_000), dtype=np.int64)
    features = train_features[sample]
    result: dict[str, tuple[float, float]] = {}
    for horizon in ENTER_ANCHORS:
        score = np.asarray(
            boosters[_model_key("enter", horizon, "rank")].predict(features, num_iteration=int(iteration)),
            dtype=np.float64,
        )
        target = _target_vector(
            pack,
            action="enter",
            horizon=horizon,
            source_start=train_source_start,
            source_stop=train_source_stop,
        )[sample].astype(np.float64)
        variance = float(np.var(score))
        slope = float(np.cov(score, target, ddof=0)[0, 1] / variance) if variance > 1.0e-12 else 0.0
        intercept = float(target.mean() - slope * score.mean())
        result[str(int(horizon))] = (intercept, slope)
    return result


def _development_loss(
    *,
    pack: QCurvePack,
    fold: Mapping[str, Any],
    dev_features: np.ndarray,
    dev_source_start: int,
    boosters: Mapping[str, lgb.Booster],
    iteration: int,
    rank_calibration: Mapping[str, tuple[float, float]],
) -> tuple[dict[str, float], dict[str, np.ndarray]]:
    anchors = _anchor_predictions(
        boosters=boosters,
        features=dev_features,
        iteration=iteration,
        rank_calibration=rank_calibration,
        target_stats=fold["normalization"]["targets"],
    )
    values: dict[str, list[float]] = {name: [] for name in LOSS_WEIGHT_MAP}
    offset = 0
    for span in fold["development_spans"]:
        count = int(span["candidate_stop"]) - int(span["candidate_start"])
        subset = {name: array[offset : offset + count] for name, array in anchors.items()}
        curves = _curves_from_anchors(subset)
        raw_enter, raw_hold = pack.q_targets(
            int(span["candidate_start"]),
            int(span["candidate_stop"]),
            cost="base",
        )
        components = _numpy_day_components(
            curves,
            raw_enter=raw_enter,
            raw_hold=raw_hold,
            target_stats=fold["normalization"]["targets"],
        )
        for name, value in components.items():
            values[name].append(float(value))
        offset += count
    return {name: float(np.mean(items)) for name, items in values.items()}, anchors


class LGBMQCurveProvider:
    def __init__(
        self,
        *,
        pack: QCurvePack,
        fold: Mapping[str, Any],
        boosters: Mapping[str, lgb.Booster],
        iteration: int,
        rank_calibration: Mapping[str, tuple[float, float]],
    ) -> None:
        self.pack = pack
        self.fold = fold
        self.boosters = boosters
        self.iteration = int(iteration)
        self.rank_calibration = rank_calibration

    def predict_symbols(self, date_idx: int, symbols: np.ndarray) -> Mapping[str, np.ndarray]:
        features = self.pack.current_lgbm_features(
            date_idx=int(date_idx),
            symbols=symbols,
            normalization=self.fold["normalization"]["inputs"],
        )
        anchors = _anchor_predictions(
            boosters=self.boosters,
            features=features,
            iteration=self.iteration,
            rank_calibration=self.rank_calibration,
            target_stats=self.fold["normalization"]["targets"],
        )
        return _curves_from_anchors(anchors)


def train_lgbm_qcurve_fold(
    *,
    fold_path: str | Path,
    output_dir: str | Path,
    seed: int = DEFAULT_SEED,
    max_epochs: int = DEFAULT_MAX_EPOCHS,
    patience: int = DEFAULT_PATIENCE,
) -> dict[str, Any]:
    verification = verify_qcurve_development_fold(fold_path)
    if verification["status"] != "ok":
        raise ValueError(f"invalid Q-curve fold: {verification['blockers']}")
    if int(seed) != 7 or int(patience) != 2 or not 1 <= int(max_epochs) <= 10:
        raise ValueError("frozen LGBM contract requires seed7, patience2, and 1..10 epochs")
    fold = _read_json(fold_path)
    pack = QCurvePack(fold["pack_manifest"])
    output = Path(output_dir)
    model_root = output / "models"
    model_root.mkdir(parents=True, exist_ok=True)
    progress_path = output / "progress.json"
    train_start, train_stop, train_count = _contiguous_split(fold["train_spans"])
    dev_start, dev_stop, dev_count = _contiguous_split(fold["development_spans"])
    train_features = _feature_matrix(
        pack=pack,
        fold=fold,
        spans=fold["train_spans"],
        path=output / "train_features.float32.dat",
    )
    dev_features = _feature_matrix(
        pack=pack,
        fold=fold,
        spans=fold["development_spans"],
        path=output / "development_features.float32.dat",
    )
    groups = [int(item["candidate_stop"]) - int(item["candidate_start"]) for item in fold["train_spans"]]
    initial_label = np.zeros(train_count, dtype=np.float32)
    train_set = lgb.Dataset(
        train_features,
        label=initial_label,
        group=groups,
        free_raw_data=False,
        params={"feature_pre_filter": False},
    )
    train_set.construct()
    base_params = {
        "verbosity": -1,
        "seed": int(seed),
        "feature_fraction_seed": int(seed),
        "bagging_seed": int(seed),
        "data_random_seed": int(seed),
        "deterministic": True,
        "force_col_wise": True,
        "num_leaves": 31,
        "learning_rate": 0.05,
        "min_data_in_leaf": 500,
        "max_bin": 63,
        "feature_fraction": 0.9,
        "bagging_fraction": 1.0,
        "num_threads": max((os.cpu_count() or 4) - 1, 1),
    }
    boosters: dict[str, lgb.Booster] = {}
    jobs: list[tuple[str, int, str]] = []
    for action, anchors in (("enter", ENTER_ANCHORS), ("hold", HOLD_ANCHORS)):
        jobs.extend((action, horizon, head) for horizon in anchors for head in HEADS)
    jobs.extend(("enter", horizon, "rank") for horizon in ENTER_ANCHORS)
    for job_idx, (action, horizon, head) in enumerate(jobs, start=1):
        key = _model_key(action, horizon, head)
        model_path = model_root / f"{key}.txt"
        if model_path.is_file():
            boosters[key] = lgb.Booster(model_file=str(model_path))
            continue
        target = _target_vector(
            pack,
            action=action,
            horizon=horizon,
            source_start=train_start,
            source_stop=train_stop,
        )
        if head == "p_positive":
            label = (target > 0.0).astype(np.float32)
            params = {**base_params, **_objective(head)}
        elif head == "rank":
            label = _rank_relevance(target, groups)
            params = {
                **base_params,
                "objective": "lambdarank",
                "metric": "ndcg",
                "label_gain": [float(2**value - 1) for value in range(16)],
                "lambdarank_truncation_level": 32,
            }
        else:
            stats = fold["normalization"]["targets"][action]
            base = 2 if action == "enter" else 1
            index = int(horizon) - base
            label = ((target - float(stats["median"][index])) / float(stats["scale"][index])).astype(np.float32)
            params = {**base_params, **_objective(head)}
        train_set.set_label(label)
        booster = lgb.train(params, train_set, num_boost_round=int(max_epochs))
        booster.save_model(str(model_path), num_iteration=int(max_epochs))
        boosters[key] = booster
        _write_json(
            progress_path,
            {
                "status": "training_models",
                "completed_model_count": job_idx,
                "total_model_count": len(jobs),
                "last_model": key,
                "resource": _soft_reclaim(),
                "updated_at": _now(),
            },
        )

    history: list[dict[str, Any]] = []
    best_loss = float("inf")
    best_epoch = 0
    best_calibration: dict[str, tuple[float, float]] = {}
    wait = 0
    loss_scales: dict[str, float] = {name: 1.0 for name in LOSS_WEIGHT_MAP}
    loss_scale_reference: dict[str, float] = {name: 1.0 for name in LOSS_WEIGHT_MAP}
    for epoch in range(1, int(max_epochs) + 1):
        calibration = _rank_calibration(
            boosters=boosters,
            train_features=train_features,
            pack=pack,
            train_source_start=train_start,
            train_source_stop=train_stop,
            iteration=epoch,
        )
        components, _ = _development_loss(
            pack=pack,
            fold=fold,
            dev_features=dev_features,
            dev_source_start=dev_start,
            boosters=boosters,
            iteration=epoch,
            rank_calibration=calibration,
        )
        total = float(
            sum(float(LOSS_WEIGHT_MAP[name]) * float(loss_scales[name]) * value for name, value in components.items())
        )
        improved = total < best_loss
        if improved:
            best_loss = total
            best_epoch = epoch
            best_calibration = calibration
            wait = 0
        else:
            wait += 1
        history.append(
            {
                "epoch": epoch,
                "development_total_loss": total,
                "development_date_equal_components": components,
                "improved": improved,
                "early_stopping_wait": wait,
            }
        )
        if wait >= int(patience):
            break
    provider = LGBMQCurveProvider(
        pack=pack,
        fold=fold,
        boosters=boosters,
        iteration=best_epoch,
        rank_calibration=best_calibration,
    )
    evaluation = evaluate_qcurve_fold(pack=pack, fold=fold, provider=provider)
    evaluation_path = _write_json(output / "evaluation.json", evaluation)
    history_path = _write_json(output / "training_history.json", {"history": history})
    summary = {
        "schema_version": 1,
        "status": "completed",
        "profile": "qcurve_lgbm",
        "development_year": int(fold["development_year"]),
        "seed": int(seed),
        "fold_path": str(Path(fold_path).resolve()),
        "fold_contract_sha256": fold["fold_contract_sha256"],
        "best_epoch": int(best_epoch),
        "best_development_total_loss": float(best_loss),
        "complete_epochs": len(history),
        "stopped_early": len(history) < int(max_epochs),
        "early_stopping": {
            "metric": "development_total_loss",
            "mode": "min",
            "patience": int(patience),
            "topk_selects_checkpoint": False,
            "restored_best_iteration": True,
        },
        "model_contract": {
            "fixed_enter_anchors": list(ENTER_ANCHORS),
            "fixed_hold_anchors": list(HOLD_ANCHORS),
            "ordered_quantiles": "post-prediction structural projection",
            "full_day_rank": "date-grouped LambdaRank, Top32 truncation, Top3 maximum relevance",
            "rank_mean_blend_weight": RANK_BLEND_WEIGHT,
        },
        "loss_weights": LOSS_WEIGHT_MAP,
        "loss_scales": loss_scales,
        "loss_scale_reference": loss_scale_reference,
        "rank_calibration": {key: list(value) for key, value in best_calibration.items()},
        "model_root": str(model_root.resolve()),
        "evaluation_path": str(evaluation_path.resolve()),
        "history_path": str(history_path.resolve()),
        "completed_at": _now(),
    }
    summary_path = _write_json(output / "summary.json", summary)
    _write_json(progress_path, {"status": "completed", "summary_path": str(summary_path.resolve()), "updated_at": _now()})
    return summary


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train the Seq100 Q-curve LightGBM anchor baseline.")
    parser.add_argument("--fold", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--max-epochs", type=int, default=DEFAULT_MAX_EPOCHS)
    parser.add_argument("--patience", type=int, default=DEFAULT_PATIENCE)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    result = train_lgbm_qcurve_fold(
        fold_path=args.fold,
        output_dir=args.output_dir,
        seed=int(args.seed),
        max_epochs=int(args.max_epochs),
        patience=int(args.patience),
    )
    print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
