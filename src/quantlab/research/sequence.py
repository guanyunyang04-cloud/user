from __future__ import annotations

import gc
import json
import math
import random
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime
from itertools import pairwise
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.nn import functional as F

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
    date_relevance_labels,
    fold_rows,
    load_data,
    open_array,
    write_json,
)
from .tree import evaluate_predictions

LOOKBACK = 60
RAW_CHANNEL_COUNT = 7  # relative OHLC, normalized volume/amount, valid-bar mask
MARKET_FEATURE_SLICE = slice(104, 118)
TARGET_NAME = "exact_net_return_d10_base"
SEED = 20260809
MAX_EPOCHS = 10
PATIENCE = 3
STEPS_PER_EPOCH = 160
DATES_PER_BATCH = 32
STOCKS_PER_DATE = 64


class RawSequenceError(RuntimeError):
    pass


def lookback_indices(date_idx: np.ndarray, lookback: int = LOOKBACK) -> np.ndarray:
    dates = np.asarray(date_idx, dtype=np.int32)
    offsets = np.arange(int(lookback) - 1, -1, -1, dtype=np.int32)
    grid = dates[:, None] - offsets[None, :]
    if grid.size and int(grid.min()) < 0:
        raise RawSequenceError("insufficient history for raw sequence")
    return grid


class RawSequenceBuilder:
    def __init__(
        self,
        data: ResearchData,
        *,
        market_mean: np.ndarray,
        market_std: np.ndarray,
    ) -> None:
        self.data = data
        self.daily_raw = open_array(data.execution["daily_raw"])
        self.has_bar = open_array(data.execution["has_bar"])
        self.row_dates = data.row_index["date_idx"].to_numpy(dtype=np.int32, copy=False)
        self.row_symbols = data.row_index["symbol_idx"].to_numpy(dtype=np.int32, copy=False)
        self.market_mean = np.asarray(market_mean, dtype=np.float32)
        self.market_std = np.asarray(market_std, dtype=np.float32)
        if self.market_mean.shape != (14,) or self.market_std.shape != (14,):
            raise RawSequenceError("market normalization must contain 14 fields")

    def build(self, rows: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        positions = np.asarray(rows, dtype=np.int64)
        dates = self.row_dates[positions]
        symbols = self.row_symbols[positions]
        grid = lookback_indices(dates)
        symbol_grid = np.broadcast_to(symbols[:, None], grid.shape)
        raw = np.asarray(self.daily_raw[grid, symbol_grid, :6], dtype=np.float32)
        has_bar = np.asarray(self.has_bar[grid, symbol_grid], dtype=bool)

        prices = raw[:, :, :4]
        current_close = prices[:, -1, 3]
        price_valid = (
            has_bar[:, :, None]
            & np.isfinite(prices)
            & (prices > 0.0)
            & np.isfinite(current_close[:, None, None])
            & (current_close[:, None, None] > 0.0)
        )
        denominator = np.where(np.isfinite(current_close) & (current_close > 0.0), current_close, 1.0)
        relative_price = np.zeros_like(prices, dtype=np.float32)
        np.log(
            np.clip(prices / denominator[:, None, None], 1.0e-4, 1.0e4),
            out=relative_price,
            where=price_valid,
        )
        relative_price = np.clip(relative_price, -2.0, 2.0)

        activity = raw[:, :, 4:6]
        activity_valid = has_bar[:, :, None] & np.isfinite(activity) & (activity > 0.0)
        logged = np.zeros_like(activity, dtype=np.float32)
        np.log1p(np.maximum(activity, 0.0), out=logged, where=activity_valid)
        counts = activity_valid.sum(axis=1, keepdims=True).clip(min=1)
        means = (logged * activity_valid).sum(axis=1, keepdims=True) / counts
        variance = (np.square(logged - means) * activity_valid).sum(axis=1, keepdims=True) / counts
        scales = np.sqrt(np.maximum(variance, 1.0e-4))
        normalized_activity = np.where(activity_valid, (logged - means) / scales, 0.0).astype(np.float32)
        normalized_activity = np.clip(normalized_activity, -5.0, 5.0)
        bar_mask = has_bar.astype(np.float32)[:, :, None]
        sequence = np.concatenate([relative_price, normalized_activity, bar_mask], axis=2).astype(np.float32)
        if sequence.shape != (len(positions), LOOKBACK, RAW_CHANNEL_COUNT):
            raise RawSequenceError("raw sequence shape changed")
        if not np.isfinite(sequence).all():
            raise RawSequenceError("raw sequence contains non-finite values")

        market = np.asarray(self.data.matrix[positions, MARKET_FEATURE_SLICE], dtype=np.float32)
        market = (market - self.market_mean) / self.market_std
        market = np.nan_to_num(market, nan=0.0, posinf=0.0, neginf=0.0)
        market = np.clip(market, -8.0, 8.0).astype(np.float32)
        return sequence, market


class ResidualTemporalBlock(nn.Module):
    def __init__(self, channels: int, dilation: int) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Conv1d(
                channels,
                channels,
                kernel_size=3,
                padding=dilation,
                dilation=dilation,
            ),
            nn.GroupNorm(4, channels),
            nn.GELU(),
            nn.Conv1d(channels, channels, kernel_size=1),
            nn.GroupNorm(4, channels),
        )

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return F.gelu(values + self.network(values))


class RawSequenceModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.input = nn.Conv1d(RAW_CHANNEL_COUNT, 32, kernel_size=3, padding=1)
        self.temporal = nn.Sequential(
            ResidualTemporalBlock(32, 1),
            ResidualTemporalBlock(32, 2),
            ResidualTemporalBlock(32, 4),
        )
        self.market = nn.Sequential(nn.Linear(14, 16), nn.GELU())
        self.head = nn.Sequential(
            nn.Linear(32 * 3 + 16, 64),
            nn.GELU(),
            nn.Dropout(0.10),
            nn.Linear(64, 1),
        )

    def forward(self, sequence: torch.Tensor, market: torch.Tensor) -> torch.Tensor:
        encoded = self.temporal(self.input(sequence.transpose(1, 2)))
        pooled = torch.cat([encoded[:, :, -1], encoded.mean(dim=2), encoded.amax(dim=2)], dim=1)
        return self.head(torch.cat([pooled, self.market(market)], dim=1)).squeeze(1)


def ranking_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    *,
    dates_per_batch: int,
    stocks_per_date: int,
) -> torch.Tensor:
    predicted = prediction.reshape(int(dates_per_batch), int(stocks_per_date))
    actual = target.reshape(int(dates_per_batch), int(stocks_per_date))
    point = F.smooth_l1_loss(predicted, actual)
    predicted_centered = predicted - predicted.mean(dim=1, keepdim=True)
    actual_centered = actual - actual.mean(dim=1, keepdim=True)
    numerator = (predicted_centered * actual_centered).sum(dim=1)
    denominator = torch.sqrt(predicted_centered.square().sum(dim=1) * actual_centered.square().sum(dim=1) + 1.0e-8)
    correlation = (numerator / denominator).mean()
    return point + 0.25 * (1.0 - correlation)


def _folds(data: ResearchData) -> list[dict[str, Any]]:
    return build_forward_folds(
        date_idx=data.dates,
        trade_date=data.row_index["trade_date"].astype(str).to_numpy(),
        validation_start_date=DEFAULT_VALIDATION_START_DATE,
        validation_end_date=data.maximum_outcome_date,
        fold_count=5,
        purge_days=30,
    )


def _market_normalization(data: ResearchData, train_rows: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    dates = data.dates[train_rows]
    _, first = np.unique(dates, return_index=True)
    representatives = train_rows[first]
    values = np.asarray(data.matrix[representatives, MARKET_FEATURE_SLICE], dtype=np.float32)
    mean = np.nanmean(values, axis=0).astype(np.float32)
    std = np.nanstd(values, axis=0).astype(np.float32)
    std = np.where(np.isfinite(std) & (std > 1.0e-6), std, 1.0).astype(np.float32)
    mean = np.nan_to_num(mean, nan=0.0)
    return mean, std


def _date_row_groups(rows: np.ndarray, dates: np.ndarray, valid: np.ndarray) -> list[np.ndarray]:
    usable = np.asarray(rows, dtype=np.int64)[np.asarray(valid, dtype=bool)]
    usable_dates = np.asarray(dates, dtype=np.int32)[np.asarray(valid, dtype=bool)]
    if not len(usable):
        raise RawSequenceError("no valid rows for sequence training")
    boundaries = np.flatnonzero(np.r_[True, usable_dates[1:] != usable_dates[:-1], True])
    return [usable[left:right] for left, right in pairwise(boundaries)]


def _sample_rows(groups: Sequence[np.ndarray], rng: np.random.Generator) -> np.ndarray:
    selected_dates = rng.choice(len(groups), size=DATES_PER_BATCH, replace=len(groups) < DATES_PER_BATCH)
    output = []
    for date_position in selected_dates:
        group = groups[int(date_position)]
        output.append(
            rng.choice(
                group,
                size=STOCKS_PER_DATE,
                replace=len(group) < STOCKS_PER_DATE,
            )
        )
    return np.concatenate(output).astype(np.int64)


def _validation_sample(
    rows: np.ndarray,
    dates: np.ndarray,
    valid: np.ndarray,
    *,
    stocks_per_date: int = 96,
) -> np.ndarray:
    groups = _date_row_groups(rows, dates, valid)
    rng = np.random.default_rng(SEED + 991)
    selected = [rng.choice(group, size=min(stocks_per_date, len(group)), replace=False) for group in groups]
    return np.concatenate(selected).astype(np.int64)


def _predict_rows(
    model: nn.Module,
    builder: RawSequenceBuilder,
    rows: np.ndarray,
    device: torch.device,
    *,
    batch_size: int = 8192,
) -> np.ndarray:
    model.eval()
    output = np.empty(len(rows), dtype=np.float32)
    with torch.inference_mode():
        for left in range(0, len(rows), int(batch_size)):
            right = min(left + int(batch_size), len(rows))
            sequence, market = builder.build(rows[left:right])
            sequence_tensor = torch.from_numpy(sequence).to(device, non_blocking=True)
            market_tensor = torch.from_numpy(market).to(device, non_blocking=True)
            with torch.autocast(device_type=device.type, enabled=device.type == "cuda"):
                prediction = model(sequence_tensor, market_tensor)
            output[left:right] = prediction.float().cpu().numpy()
    return output


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


def _root(fold_number: int) -> Path:
    return OUTPUT_ROOT / "sequence_raw60" / f"fold_{int(fold_number):02d}"


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False


def _contract(data: ResearchData, fold: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "pipeline": "raw_ohlcva_sequence",
        "lookback": LOOKBACK,
        "channels": [
            "open_log_relative_to_signal_close",
            "high_log_relative_to_signal_close",
            "low_log_relative_to_signal_close",
            "close_log_relative_to_signal_close",
            "volume_window_zscore",
            "amount_window_zscore",
            "has_bar",
        ],
        "market_features": list(data.feature_names[MARKET_FEATURE_SLICE]),
        "target": TARGET_NAME,
        "fold": fold,
        "seed": SEED,
        "max_epochs": MAX_EPOCHS,
        "patience": PATIENCE,
        "steps_per_epoch": STEPS_PER_EPOCH,
        "dates_per_batch": DATES_PER_BATCH,
        "stocks_per_date": STOCKS_PER_DATE,
    }


def _task_fingerprint(data: ResearchData, fold: Mapping[str, Any]) -> str:
    return stable_hash(
        {
            **_contract(data, fold),
            "input_fingerprint": data.fingerprints["input"],
            "view_fingerprint": data.fingerprints["features"],
            "target_fingerprint": data.fingerprints["exact_targets"],
        }
    )


@dataclass(frozen=True)
class _SequenceFoldInputs:
    data: ResearchData
    fold: dict[str, Any]
    contract: dict[str, Any]
    fingerprint: str
    root: Path
    train_rows: np.ndarray
    validation_rows: np.ndarray
    values: np.ndarray
    target_column: int
    validation_raw: np.ndarray
    validation_valid: np.ndarray
    target_by_row: np.ndarray
    groups: list[np.ndarray]
    validation_sample: np.ndarray
    market_mean: np.ndarray
    market_std: np.ndarray


@dataclass(frozen=True)
class _SequenceFit:
    model: RawSequenceModel
    builder: RawSequenceBuilder
    device: torch.device
    best_state: dict[str, torch.Tensor]
    best_epoch: int
    best_ic: float
    history: list[dict[str, Any]]


def _sequence_fold_inputs(fold_number: int) -> _SequenceFoldInputs:
    data = load_data()
    folds = _folds(data)
    if not 1 <= int(fold_number) <= len(folds):
        raise ResearchDataError("fold number is outside 1..5")
    fold = folds[int(fold_number) - 1]
    contract = _contract(data, fold)
    train_rows = fold_rows(data.dates, fold, "train")
    validation_rows = fold_rows(data.dates, fold, "validation")
    values, valid_panel, target_column = data.target(TARGET_NAME)
    train_raw = np.asarray(values[train_rows, target_column], dtype=np.float32)
    train_valid = np.asarray(valid_panel[train_rows, target_column], dtype=bool) & np.isfinite(train_raw)
    validation_raw = np.asarray(values[validation_rows, target_column], dtype=np.float32)
    validation_valid = np.asarray(valid_panel[validation_rows, target_column], dtype=bool) & np.isfinite(validation_raw)
    relevance = date_relevance_labels(data.dates[train_rows], train_raw, train_valid)
    target_by_row = np.zeros(data.row_count, dtype=np.float32)
    target_by_row[train_rows] = (relevance - 4.5) / 4.5
    groups = _date_row_groups(train_rows, data.dates[train_rows], train_valid)
    validation_sample = _validation_sample(validation_rows, data.dates[validation_rows], validation_valid)
    market_mean, market_std = _market_normalization(data, train_rows)
    return _SequenceFoldInputs(
        data=data,
        fold=dict(fold),
        contract=contract,
        fingerprint=_task_fingerprint(data, fold),
        root=_root(fold_number),
        train_rows=train_rows,
        validation_rows=validation_rows,
        values=values,
        target_column=target_column,
        validation_raw=validation_raw,
        validation_valid=validation_valid,
        target_by_row=target_by_row,
        groups=groups,
        validation_sample=validation_sample,
        market_mean=market_mean,
        market_std=market_std,
    )


def _cached_sequence_result(
    result_path: Path,
    fingerprint: str,
) -> dict[str, Any] | None:
    if not result_path.is_file():
        return None
    result = dict(json.loads(result_path.read_text(encoding="utf-8")))
    if (
        result.get("status") == "completed"
        and result.get("fingerprint") == fingerprint
        and cutoff_violation_count(result) == 0
        and all(Path(item["path"]).is_file() for item in result["files"].values())
    ):
        return result
    return None


def _sequence_training_step(
    *,
    model: RawSequenceModel,
    builder: RawSequenceBuilder,
    optimizer: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler,
    device: torch.device,
    rows: np.ndarray,
    target_by_row: np.ndarray,
) -> float:
    sequence, market = builder.build(rows)
    sequence_tensor = torch.from_numpy(sequence).to(device, non_blocking=True)
    market_tensor = torch.from_numpy(market).to(device, non_blocking=True)
    target_tensor = torch.from_numpy(target_by_row[rows]).to(device, non_blocking=True)
    optimizer.zero_grad(set_to_none=True)
    with torch.autocast(device_type=device.type, enabled=device.type == "cuda"):
        prediction = model(sequence_tensor, market_tensor)
        loss = ranking_loss(
            prediction,
            target_tensor,
            dates_per_batch=DATES_PER_BATCH,
            stocks_per_date=STOCKS_PER_DATE,
        )
    scaler.scale(loss).backward()
    scaler.unscale_(optimizer)
    torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
    scaler.step(optimizer)
    scaler.update()
    return float(loss.detach().cpu())


def _sample_validation_ic(
    *,
    model: RawSequenceModel,
    builder: RawSequenceBuilder,
    inputs: _SequenceFoldInputs,
    device: torch.device,
) -> float:
    prediction = _predict_rows(
        model,
        builder,
        inputs.validation_sample,
        device,
        batch_size=8192,
    )
    actual = np.asarray(
        inputs.values[inputs.validation_sample, inputs.target_column],
        dtype=np.float32,
    )
    _, metrics = daily_rank_metrics(
        dates=inputs.data.dates[inputs.validation_sample],
        actual=actual,
        prediction=prediction,
    )
    return float(metrics["daily_rank_ic_mean"])


def _fit_sequence(inputs: _SequenceFoldInputs, fold_number: int) -> _SequenceFit:
    builder = RawSequenceBuilder(
        inputs.data,
        market_mean=inputs.market_mean,
        market_std=inputs.market_std,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = RawSequenceModel().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1.0e-3, weight_decay=1.0e-4)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    rng = np.random.default_rng(SEED + int(fold_number) * 1009)
    best_ic = -math.inf
    best_epoch = 0
    best_state: dict[str, torch.Tensor] | None = None
    patience_used = 0
    history: list[dict[str, Any]] = []
    print(
        json.dumps(
            {
                "event": "raw60_training_started",
                "fold": fold_number,
                "train_rows": len(inputs.train_rows),
                "validation_rows": len(inputs.validation_rows),
                "device": str(device),
            }
        ),
        flush=True,
    )
    for epoch in range(1, MAX_EPOCHS + 1):
        model.train()
        losses: list[float] = []
        for _ in range(STEPS_PER_EPOCH):
            rows = _sample_rows(inputs.groups, rng)
            losses.append(
                _sequence_training_step(
                    model=model,
                    builder=builder,
                    optimizer=optimizer,
                    scaler=scaler,
                    device=device,
                    rows=rows,
                    target_by_row=inputs.target_by_row,
                )
            )
        validation_ic = _sample_validation_ic(
            model=model,
            builder=builder,
            inputs=inputs,
            device=device,
        )
        history.append(
            {
                "epoch": epoch,
                "training_loss": float(np.mean(losses)),
                "sample_validation_rank_ic": validation_ic,
            }
        )
        print(
            json.dumps(
                {
                    "event": "raw60_epoch_completed",
                    "fold": fold_number,
                    **history[-1],
                }
            ),
            flush=True,
        )
        if validation_ic > best_ic + 1.0e-5:
            best_ic = validation_ic
            best_epoch = epoch
            best_state = {name: tensor.detach().cpu().clone() for name, tensor in model.state_dict().items()}
            patience_used = 0
        else:
            patience_used += 1
        if epoch >= 3 and patience_used >= PATIENCE:
            break

    if best_state is None:
        raise RawSequenceError("sequence training produced no checkpoint")
    model.load_state_dict(best_state)
    return _SequenceFit(
        model=model,
        builder=builder,
        device=device,
        best_state=best_state,
        best_epoch=best_epoch,
        best_ic=best_ic,
        history=history,
    )


def _write_sequence_outputs(
    *,
    inputs: _SequenceFoldInputs,
    fit: _SequenceFit,
    prediction: np.ndarray,
    daily: pd.DataFrame,
    fold_number: int,
) -> tuple[dict[str, dict[str, str]], pd.DataFrame]:
    inputs.root.mkdir(parents=True, exist_ok=True)
    model_path = inputs.root / "model.pt"
    partial_model = model_path.with_suffix(".pt.partial")
    torch.save(
        {
            "state_dict": fit.best_state,
            "contract": inputs.contract,
            "market_mean": inputs.market_mean,
            "market_std": inputs.market_std,
        },
        partial_model,
    )
    partial_model.replace(model_path)
    frame = inputs.data.row_index.iloc[inputs.validation_rows][
        ["candidate_id", "date_idx", "trade_date", "symbol", "symbol_idx"]
    ].copy()
    frame.insert(0, "row_position", inputs.validation_rows)
    frame["fold"] = int(fold_number)
    frame["score"] = prediction
    frame["actual"] = inputs.validation_raw
    frame["target_valid"] = inputs.validation_valid
    prediction_path = inputs.root / "predictions.parquet"
    daily_path = inputs.root / "daily_metrics.parquet"
    history_path = inputs.root / "history.parquet"
    frame.to_parquet(prediction_path, index=False)
    daily.to_parquet(daily_path, index=False)
    pd.DataFrame(fit.history).to_parquet(history_path, index=False)
    return {
        "model": {"path": str(model_path)},
        "predictions": {"path": str(prediction_path)},
        "daily_metrics": {"path": str(daily_path)},
        "history": {"path": str(history_path)},
    }, frame


def train_fold(fold_number: int) -> dict[str, Any]:
    inputs = _sequence_fold_inputs(fold_number)
    result_path = inputs.root / "result.json"
    cached = _cached_sequence_result(result_path, inputs.fingerprint)
    if cached is not None:
        return cached
    _seed_everything(SEED + int(fold_number))
    started = time.perf_counter()
    fit = _fit_sequence(inputs, fold_number)
    prediction = _predict_rows(
        fit.model,
        fit.builder,
        inputs.validation_rows,
        fit.device,
    )
    elapsed = time.perf_counter() - started
    daily, metrics = daily_rank_metrics(
        dates=inputs.data.dates[inputs.validation_rows][inputs.validation_valid],
        actual=inputs.validation_raw[inputs.validation_valid],
        prediction=prediction[inputs.validation_valid],
    )
    files, prediction_frame = _write_sequence_outputs(
        inputs=inputs,
        fit=fit,
        prediction=prediction,
        daily=daily,
        fold_number=fold_number,
    )
    result = {
        "status": "completed",
        "completed_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "fingerprint": inputs.fingerprint,
        "model_family": "raw60_temporal_cnn",
        "fold": inputs.fold,
        "contract": inputs.contract,
        "best_epoch": fit.best_epoch,
        "best_sample_validation_rank_ic": fit.best_ic,
        "metrics": metrics,
        "elapsed_seconds": elapsed,
        "history": fit.history,
        **cutoff_audit_fields(),
        "files": files,
    }
    write_json(result_path, _safe_json(result))
    print(
        json.dumps(
            {
                "event": "raw60_training_completed",
                "fold": fold_number,
                "best_epoch": fit.best_epoch,
                "rank_ic": metrics["daily_rank_ic_mean"],
                "elapsed_seconds": elapsed,
            }
        ),
        flush=True,
    )
    del fit, prediction_frame
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return result


def train_all() -> list[dict[str, Any]]:
    return [train_fold(fold) for fold in range(1, 6)]


def load_fold_results() -> tuple[list[dict[str, Any]], pd.DataFrame]:
    data = load_data()
    folds = _folds(data)
    results: list[dict[str, Any]] = []
    frames: list[pd.DataFrame] = []
    for fold in range(1, 6):
        path = _root(fold) / "result.json"
        if not path.is_file():
            raise ResearchDataError(f"raw60 fold is missing: {path}")
        result = dict(json.loads(path.read_text(encoding="utf-8")))
        if (
            result.get("status") != "completed"
            or result.get("fingerprint") != _task_fingerprint(data, folds[fold - 1])
            or cutoff_violation_count(result) != 0
            or not all(Path(record["path"]).is_file() for record in result.get("files", {}).values())
        ):
            raise ResearchDataError(f"raw60 fold is stale or invalid: {path}")
        results.append(result)
        frames.append(pd.read_parquet(Path(result["files"]["predictions"]["path"])))
    oof = pd.concat(frames, ignore_index=True).sort_values(["date_idx", "symbol_idx"], kind="stable")
    if oof.duplicated("row_position").any():
        raise ResearchDataError("raw60 OOF row positions overlap")
    return results, oof


def evaluate() -> dict[str, Any]:
    results, oof = load_fold_results()
    return evaluate_predictions(
        run_name="sequence_raw60",
        fold_results=results,
        oof=oof,
        metadata={
            "model_family": "raw60_temporal_cnn",
            "feature_count": 7,
            "lookback": LOOKBACK,
            "market_feature_count": 14,
        },
    )
