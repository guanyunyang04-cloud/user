"""Reusable Seq100 feature, label, and model math.

The invalidated study lifecycle is retained in Git history, not in the runtime module.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import random
import time
from collections import deque
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Iterator, Mapping, Sequence

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as torch_functional
from scipy import stats

from daily_research.path_policy import (
    qdp_v2_sequence_path_training as sequence_training,
)

WORKSPACE_ROOT = Path(__file__).resolve().parents[2]

TARGET_CANDIDATE_IDS = (
    "raw_path_distribution_v1",
    "pareto_ordinal_v1",
    "competing_risk_path_v1",
    "direct_listwise_utility_v1",
)

EVENT_CENSORED = 0

EVENT_TERMINAL_FAILURE = 1

EVENT_MAJOR_DRAWDOWN = 2

EVENT_POST_PEAK_FADE = 3

EVENT_NO_LEGAL_SELL = 4

EVENT_SUSTAINED_UPSIDE = 5

TARGET_FLOAT_FIELDS = tuple(
    [f"close_net_log_d{day:02d}" for day in range(1, 21)]
    + [
        "r5_net",
        "r10_net",
        "r20_net",
        "r40_net",
        "r60_net",
        "speed5_log_per_day",
        "speed10_log_per_day",
        "speed20_log_per_day",
        "min_speed_5_10_20",
        "auc20_net_log",
        "time_above_break_even20",
        "mdd20",
        "post_peak_fade20",
        "mfe20",
        "mae20",
        "peak_day20",
        "sellable_ratio20",
        "trend_scan_t",
        "trend_scan_horizon",
        "trend_scan_bonferroni_p",
        "trend_consistency_v1",
        "trend_blend_5_10_20_dd100",
        "max_speed_negative_control",
        "common_sustained_action_utility",
        "raw_path_distribution_quality",
        "pareto_ordinal_quality",
        "competing_risk_quality",
        "direct_listwise_utility_quality",
        "competing_event_code",
        "competing_event_time",
        "competing_event_score",
        "dominance_margin",
        "pareto_relevance",
        "conditional_relevance",
        "action_relevance",
    ]
)

GRADE_THRESHOLDS = (0.50, 0.80, 0.95, 0.99)

MODEL_IDS = (
    "lgbm_lambdarank_multioutput",
    "tabm_multioutput",
    "patchtst_student_t_path",
    "market_industry_deepsets",
    "deephit_competing_risk",
    "neuralndcg_listwise_mlp",
)

NEURAL_MODEL_IDS = MODEL_IDS[1:]

TRAINING_CHECKPOINT_VERSION = "seq100_resumable_training_v1"

TRAINING_PROGRESS_VERSION = "seq100_training_progress_v1"

PROGRESS_HEARTBEAT_SECONDS = 30.0

CONSOLE_EVENT_SECONDS = 300.0

CHECKPOINT_INTERVAL_SECONDS = 600.0

LOW_MEMORY_PAUSE_AVAILABLE_GB = 1.0


@dataclass(frozen=True)
class FeatureViewPaths:
    root: Path
    continuous: Path
    categorical: Path
    candidate_date_idx: Path
    candidate_symbol_idx: Path
    manifest: Path


def _now() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def _json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        converted = float(value)
        return converted if math.isfinite(converted) else None
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, pd.Timestamp):
        return value.strftime("%Y-%m-%d")
    if isinstance(value, float) and not math.isfinite(value):
        return None
    raise TypeError(f"unsupported JSON value: {type(value)!r}")


def _atomic_write_json(path: Path, payload: Any) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, default=_json_default) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)
    return str(path.resolve())


def _append_jsonl(path: Path, payload: Mapping[str, Any]) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = json.dumps(
        dict(payload),
        ensure_ascii=False,
        separators=(",", ":"),
        default=_json_default,
    )
    with path.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(encoded + "\n")
        handle.flush()
    return str(path.resolve())


class TrainingPaused(RuntimeError):
    """Raised only after a safe optimizer-boundary checkpoint is durable."""


def _resolve_path(value: str | Path, *, base: Path = WORKSPACE_ROOT) -> Path:
    path = Path(value)
    if not path.is_absolute():
        path = base / path
    return path.resolve()


def reanchor_to_actual_next_open(
    future_ohlc: np.ndarray,
    *,
    source_price_anchor: str,
    required_mask: np.ndarray | None = None,
) -> np.ndarray:
    """Return future OHLC paths relative to the actual D1 open.

    The frozen PIT pack stores paths relative to signal-day close.  The study
    target is conditional on a successful next-open fill, so all future prices
    must share the executed D1 open denominator before any return descriptor is
    computed.
    """

    values = np.asarray(future_ohlc, dtype=np.float32)
    if values.ndim != 3 or values.shape[2] < 4:
        raise ValueError("future_ohlc must have shape [rows, horizon, >=4]")
    anchor = str(source_price_anchor or "").strip()
    if anchor not in {"today_close", "next_open"}:
        raise ValueError("source_price_anchor must be today_close or next_open")
    converted = sequence_training._legacy_entry_relative_path_numpy(
        values,
        price_anchor=anchor,
    )
    d1_open = np.asarray(converted[:, 0, 0], dtype=np.float32)
    normalized = np.isfinite(d1_open) & (np.abs(d1_open) <= 2.0e-6)
    required = (
        np.ones(len(converted), dtype=bool)
        if required_mask is None
        else np.asarray(required_mask, dtype=bool).reshape(-1)
    )
    if required.size != len(converted):
        raise ValueError("required_mask must align with future_ohlc rows")
    if bool(np.any(required & ~normalized)):
        raise ValueError(
            "a filled entry could not be normalized to the executed D1 open"
        )
    converted = np.asarray(converted, dtype=np.float32)
    converted[~normalized, :, :] = np.nan
    return converted


def apply_terminal_zero_recovery(
    entry_relative_path: np.ndarray,
    delisted_path: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Apply zero terminal recovery from the first delisted path day onward."""

    values = np.asarray(entry_relative_path, dtype=np.float32).copy()
    delisted = np.asarray(delisted_path, dtype=bool)
    if values.ndim != 3 or values.shape[2] < 4:
        raise ValueError("entry_relative_path must have shape [rows, horizon, >=4]")
    if delisted.shape != values.shape[:2]:
        raise ValueError("delisted_path must match the first two path dimensions")
    terminal_failure = delisted.any(axis=1)
    for row in np.flatnonzero(terminal_failure):
        first = int(np.argmax(delisted[row]))
        values[row, first:, :4] = -1.0
    return values, terminal_failure


def label_endpoint_is_allowed(
    *,
    signal_date_idx: int,
    date_values: Sequence[str],
    horizon_days: int,
    cutoff: str,
) -> bool:
    """Return whether the complete D1..DH label ends on or before cutoff."""

    signal_idx = int(signal_date_idx)
    horizon = int(horizon_days)
    if signal_idx < 0 or horizon <= 0:
        raise ValueError(
            "signal_date_idx must be non-negative and horizon_days positive"
        )
    endpoint = signal_idx + horizon
    if endpoint >= len(date_values):
        return False
    normalized_cutoff = pd.Timestamp(str(cutoff)).strftime("%Y-%m-%d")
    if normalized_cutoff >= "2026-01-01":
        raise ValueError("2026 must remain excluded from path target construction")
    endpoint_date = pd.Timestamp(str(date_values[endpoint])).strftime("%Y-%m-%d")
    return endpoint_date <= normalized_cutoff


def _rank_percentile(values: np.ndarray) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    out = np.full(array.shape, np.nan, dtype=np.float64)
    finite = np.isfinite(array)
    count = int(finite.sum())
    if count == 0:
        return out.astype(np.float32)
    if count == 1:
        out[finite] = 1.0
        return out.astype(np.float32)
    ranks = stats.rankdata(array[finite], method="average")
    out[finite] = (ranks - 1.0) / float(count - 1)
    return out.astype(np.float32)


def relevance_grades(relevance: np.ndarray) -> np.ndarray:
    values = np.asarray(relevance, dtype=np.float64)
    grades = np.full(values.shape, 255, dtype=np.uint8)
    finite = np.isfinite(values)
    if not bool(finite.any()):
        return grades
    current = np.zeros(int(finite.sum()), dtype=np.uint8)
    finite_values = values[finite]
    for threshold in GRADE_THRESHOLDS:
        current += (finite_values >= float(threshold)).astype(np.uint8)
    grades[finite] = current
    return grades


def pareto_dominance_counts(
    values: np.ndarray,
    *,
    block_size: int = 256,
    device: str = "auto",
) -> tuple[np.ndarray, np.ndarray]:
    """Return exact dominated/dominating counts for maximize-all dimensions."""

    array = np.asarray(values, dtype=np.float32)
    if array.ndim != 2:
        raise ValueError("Pareto values must have shape [rows, dimensions]")
    if not bool(np.isfinite(array).all()):
        raise ValueError("Pareto values must be finite")
    count = int(array.shape[0])
    if count == 0:
        return np.empty(0, dtype=np.int32), np.empty(0, dtype=np.int32)
    if count == 1:
        return np.zeros(1, dtype=np.int32), np.zeros(1, dtype=np.int32)

    normalized_device = str(device or "auto").lower()
    if normalized_device not in {"auto", "cpu", "cuda"}:
        raise ValueError("Pareto device must be auto, cpu, or cuda")
    use_cuda = False
    if normalized_device in {"auto", "cuda"}:
        try:
            import torch

            use_cuda = bool(torch.cuda.is_available())
        except Exception:
            use_cuda = False
        if normalized_device == "cuda" and not use_cuda:
            raise RuntimeError(
                "CUDA Pareto computation was requested but CUDA is unavailable"
            )

    dominated = np.zeros(count, dtype=np.int64)
    dominating = np.zeros(count, dtype=np.int64)
    block = max(int(block_size), 1)
    if use_cuda:
        import torch

        target = torch.from_numpy(array).to(device="cuda")
        for start in range(0, count, block):
            stop = min(start + block, count)
            current = target[start:stop]
            ge = current[:, None, :] >= target[None, :, :]
            gt = current[:, None, :] > target[None, :, :]
            matrix = torch.all(ge, dim=2) & torch.any(gt, dim=2)
            dominated[start:stop] = matrix.sum(dim=1).cpu().numpy()
            dominating += matrix.sum(dim=0).cpu().numpy()
            del current, ge, gt, matrix
        return dominated.astype(np.int32), dominating.astype(np.int32)

    for start in range(0, count, block):
        stop = min(start + block, count)
        current = array[start:stop]
        ge = current[:, None, :] >= array[None, :, :]
        gt = current[:, None, :] > array[None, :, :]
        matrix = np.all(ge, axis=2) & np.any(gt, axis=2)
        dominated[start:stop] = matrix.sum(axis=1, dtype=np.int64)
        dominating += matrix.sum(axis=0, dtype=np.int64)
    return dominated.astype(np.int32), dominating.astype(np.int32)


def _trend_scan(close_net_log: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    values = np.asarray(close_net_log, dtype=np.float64)
    rows, available_horizon = values.shape
    if available_horizon < 20:
        raise ValueError("trend scanning requires 20 path days")
    best_t = np.full(rows, np.nan, dtype=np.float64)
    best_horizon = np.full(rows, np.nan, dtype=np.float64)
    best_p = np.full(rows, np.nan, dtype=np.float64)
    tests = 16
    for horizon in range(5, 21):
        y = values[:, :horizon]
        finite = np.isfinite(y).all(axis=1)
        if not bool(finite.any()):
            continue
        x = np.arange(1, horizon + 1, dtype=np.float64)
        x_centered = x - x.mean()
        denominator = float(np.square(x_centered).sum())
        current = y[finite]
        centered = current - current.mean(axis=1, keepdims=True)
        slope = np.sum(centered * x_centered.reshape(1, -1), axis=1) / denominator
        fitted = current.mean(axis=1, keepdims=True) + slope.reshape(
            -1, 1
        ) * x_centered.reshape(1, -1)
        residual = current - fitted
        residual_variance = np.sum(np.square(residual), axis=1) / max(horizon - 2, 1)
        standard_error = np.sqrt(np.maximum(residual_variance / denominator, 0.0))
        t_value = np.divide(
            slope,
            standard_error,
            out=np.zeros_like(slope),
            where=standard_error > 1.0e-12,
        )
        p_value = np.minimum(
            1.0,
            2.0 * stats.t.sf(np.abs(t_value), df=max(horizon - 2, 1)) * tests,
        )
        positions = np.flatnonzero(finite)
        replace = ~np.isfinite(best_t[positions]) | (t_value > best_t[positions])
        chosen = positions[replace]
        best_t[chosen] = t_value[replace]
        best_horizon[chosen] = float(horizon)
        best_p[chosen] = p_value[replace]
    return (
        best_t.astype(np.float32),
        best_horizon.astype(np.float32),
        best_p.astype(np.float32),
    )


def compute_path_descriptors(
    entry_relative_path: np.ndarray,
    growth_multiplier: np.ndarray,
    *,
    exit_sellable: np.ndarray,
    terminal_failure: np.ndarray,
    log_wealth_floor: float = 1.0e-6,
) -> dict[str, np.ndarray]:
    """Compute the exact D1-D20 target vector from next-open-relative OHLC."""

    values = np.asarray(entry_relative_path, dtype=np.float64)
    multiplier = np.asarray(growth_multiplier, dtype=np.float64)
    sellable = np.asarray(exit_sellable, dtype=bool)
    terminal = np.asarray(terminal_failure, dtype=bool).reshape(-1)
    if values.ndim != 3 or values.shape[1:] != (20, 4):
        raise ValueError("entry_relative_path must have shape [rows, 20, 4]")
    if multiplier.shape != values.shape[:2]:
        raise ValueError("growth_multiplier must have shape [rows, 20]")
    if sellable.shape != values.shape[:2]:
        raise ValueError("exit_sellable must have shape [rows, 20]")
    if terminal.shape[0] != values.shape[0]:
        raise ValueError("terminal_failure length mismatch")

    high = values[:, :, 1]
    low = values[:, :, 2]
    close = values[:, :, 3]
    net_growth = (1.0 + close) * multiplier
    close_net_log = np.log(np.maximum(net_growth, float(log_wealth_floor)))
    path_available = (
        np.isfinite(values).all(axis=(1, 2))
        & np.isfinite(multiplier).all(axis=1)
        & (multiplier > 0.0).all(axis=1)
    )
    close_net_log[~path_available] = np.nan

    speed5 = close_net_log[:, 4] / 5.0
    speed10 = close_net_log[:, 9] / 10.0
    speed20 = close_net_log[:, 19] / 20.0
    minimum_speed = np.minimum(np.minimum(speed5, speed10), speed20)
    auc20 = np.nanmean(close_net_log, axis=1)
    time_above = np.nanmean(close_net_log > 0.0, axis=1)

    high_wealth = np.maximum(1.0 + high, 0.0)
    low_wealth = np.maximum(1.0 + low, 0.0)
    close_wealth = np.maximum(1.0 + close, 0.0)
    running_peak = np.maximum.accumulate(np.maximum(high_wealth, 1.0), axis=1)
    drawdown = 1.0 - np.divide(
        low_wealth,
        np.maximum(running_peak, float(log_wealth_floor)),
    )
    mdd20 = np.nanmax(drawdown, axis=1)
    peak_wealth = np.nanmax(np.maximum(high_wealth, 1.0), axis=1)
    fade20 = np.maximum(
        np.divide(
            peak_wealth - close_wealth[:, 19],
            np.maximum(peak_wealth, float(log_wealth_floor)),
        ),
        0.0,
    )
    mfe20 = np.nanmax(high, axis=1)
    mae20 = np.nanmin(low, axis=1)
    peak_day = np.argmax(np.where(np.isfinite(high), high, -np.inf), axis=1) + 1
    sellable_ratio = np.mean(sellable[:, 1:20], axis=1)
    no_legal_sell = ~sellable[:, 1:20].any(axis=1)

    trend_t, trend_horizon, trend_p = _trend_scan(close_net_log)
    r5 = np.expm1(close_net_log[:, 4])
    r10 = np.expm1(close_net_log[:, 9])
    r20 = np.expm1(close_net_log[:, 19])
    max_speed_negative = np.nanmax(
        np.column_stack(
            [
                close_net_log[:, 4] / math.sqrt(5.0),
                close_net_log[:, 9] / math.sqrt(10.0),
                close_net_log[:, 19] / math.sqrt(20.0),
            ]
        ),
        axis=1,
    )

    trend_components: list[np.ndarray] = []
    for horizon, weight in ((5, 0.50), (10, 0.30), (20, 0.20)):
        levels = close_net_log[:, :horizon]
        mean_level = np.nanmean(levels, axis=1)
        q25 = np.nanquantile(levels, 0.25, axis=1)
        horizon_mdd = np.nanmax(drawdown[:, :horizon], axis=1)
        score = (
            0.50 * close_net_log[:, horizon - 1]
            + 0.30 * mean_level
            + 0.20 * q25
            - horizon_mdd
        ) / math.sqrt(float(horizon))
        trend_components.append(float(weight) * score)
    trend_blend = np.maximum(np.sum(np.column_stack(trend_components), axis=1), 0.0)

    output: dict[str, np.ndarray] = {
        "close_net_log": close_net_log.astype(np.float32),
        "r5_net": r5.astype(np.float32),
        "r10_net": r10.astype(np.float32),
        "r20_net": r20.astype(np.float32),
        "speed5_log_per_day": speed5.astype(np.float32),
        "speed10_log_per_day": speed10.astype(np.float32),
        "speed20_log_per_day": speed20.astype(np.float32),
        "min_speed_5_10_20": minimum_speed.astype(np.float32),
        "auc20_net_log": auc20.astype(np.float32),
        "time_above_break_even20": time_above.astype(np.float32),
        "mdd20": mdd20.astype(np.float32),
        "post_peak_fade20": fade20.astype(np.float32),
        "mfe20": mfe20.astype(np.float32),
        "mae20": mae20.astype(np.float32),
        "peak_day20": peak_day.astype(np.float32),
        "sellable_ratio20": sellable_ratio.astype(np.float32),
        "trend_scan_t": trend_t,
        "trend_scan_horizon": trend_horizon,
        "trend_scan_bonferroni_p": trend_p,
        "trend_blend_5_10_20_dd100": trend_blend.astype(np.float32),
        "max_speed_negative_control": max_speed_negative.astype(np.float32),
        "path_available": path_available,
        "no_legal_sell": no_legal_sell,
        "terminal_failure": terminal,
    }
    for name, array in list(output.items()):
        if (
            isinstance(array, np.ndarray)
            and array.dtype.kind == "f"
            and name != "close_net_log"
        ):
            array[~path_available] = np.nan
    return output


def common_sustained_action_utility(
    descriptors: Mapping[str, np.ndarray],
    *,
    entry_filled: np.ndarray,
) -> np.ndarray:
    """Return ``min(g5/5,g10/10,g20/20)`` with cash for no fill."""

    filled = np.asarray(entry_filled, dtype=bool).reshape(-1)
    minimum_speed = np.asarray(
        descriptors["min_speed_5_10_20"], dtype=np.float32
    ).reshape(-1)
    if minimum_speed.size != filled.size:
        raise ValueError("entry_filled and path descriptor length mismatch")
    utility = np.full(filled.shape, np.nan, dtype=np.float32)
    utility[~filled] = 0.0
    utility[filled] = minimum_speed[filled]
    return utility


def _stable_daily_percentile(
    score: np.ndarray,
    *,
    eligible: np.ndarray,
    symbol_idx: np.ndarray,
    secondary: np.ndarray | None = None,
) -> np.ndarray:
    """Rank eligible rows from worst to best with deterministic tie breaks."""

    values = np.asarray(score, dtype=np.float64).reshape(-1)
    requested = np.asarray(eligible, dtype=bool).reshape(-1)
    symbols = np.asarray(symbol_idx, dtype=np.int64).reshape(-1)
    if values.size != requested.size or values.size != symbols.size:
        raise ValueError("daily percentile inputs must have equal length")
    tie = (
        np.zeros(values.size, dtype=np.float64)
        if secondary is None
        else np.asarray(secondary, dtype=np.float64).reshape(-1)
    )
    if tie.size != values.size:
        raise ValueError("secondary tie-break length mismatch")
    tie = np.where(np.isfinite(tie), tie, -np.inf)
    mask = requested & np.isfinite(values)
    out = np.full(values.shape, np.nan, dtype=np.float32)
    positions = np.flatnonzero(mask)
    if positions.size == 0:
        return out
    order = np.lexsort(
        (
            symbols[positions],
            tie[positions],
            values[positions],
        )
    )
    if positions.size == 1:
        out[positions[order]] = 1.0
    else:
        out[positions[order]] = np.arange(positions.size, dtype=np.float32) / float(
            positions.size - 1
        )
    return out


def compute_competing_risk_events(
    entry_relative_path: np.ndarray,
    growth_multiplier: np.ndarray,
    *,
    exit_sellable: np.ndarray,
    delisted_path: np.ndarray,
    entry_filled: np.ndarray,
    path_available: np.ndarray,
    upside_wealth: float = 1.05,
    time_above_threshold: float = 0.80,
    drawdown_threshold: float = 0.08,
    fade_threshold: float = 0.10,
) -> dict[str, np.ndarray]:
    """Construct the frozen D1-D20 competing-risk label."""

    values = np.asarray(entry_relative_path, dtype=np.float64)
    multiplier = np.asarray(growth_multiplier, dtype=np.float64)
    sellable = np.asarray(exit_sellable, dtype=bool)
    delisted = np.asarray(delisted_path, dtype=bool)
    filled = np.asarray(entry_filled, dtype=bool).reshape(-1)
    available = np.asarray(path_available, dtype=bool).reshape(-1)
    if values.ndim != 3 or values.shape[1:] != (20, 4):
        raise ValueError("competing-risk path must have shape [rows,20,4]")
    if multiplier.shape != values.shape[:2]:
        raise ValueError("competing-risk multiplier shape mismatch")
    if sellable.shape != values.shape[:2] or delisted.shape != values.shape[:2]:
        raise ValueError("competing-risk mask shape mismatch")
    if filled.size != values.shape[0] or available.size != values.shape[0]:
        raise ValueError("competing-risk row mask length mismatch")

    high_wealth = np.maximum(1.0 + values[:, :, 1], 0.0)
    low_wealth = np.maximum(1.0 + values[:, :, 2], 0.0)
    close_wealth = np.maximum(1.0 + values[:, :, 3], 0.0)
    net_close_wealth = close_wealth * multiplier
    net_close_log = np.log(np.maximum(net_close_wealth, 1.0e-6))
    running_high = np.maximum.accumulate(np.maximum(high_wealth, 1.0), axis=1)
    drawdown = 1.0 - np.divide(low_wealth, np.maximum(running_high, 1.0e-6))
    close_fade = np.maximum(
        1.0 - np.divide(close_wealth, np.maximum(running_high, 1.0e-6)),
        0.0,
    )
    time_above = np.cumsum(net_close_log > 0.0, axis=1) / np.arange(
        1, 21, dtype=np.float64
    ).reshape(1, -1)

    rows = values.shape[0]
    sentinel = np.int16(127)
    cause_days = np.full((rows, 5), sentinel, dtype=np.int16)

    def _first_true(mask: np.ndarray) -> np.ndarray:
        any_true = mask.any(axis=1)
        first = np.argmax(mask, axis=1).astype(np.int16) + 1
        return np.where(any_true, first, sentinel).astype(np.int16)

    cause_days[:, 0] = _first_true(delisted)
    cause_days[:, 1] = _first_true(drawdown >= float(drawdown_threshold))
    cause_days[:, 2] = _first_true(close_fade >= float(fade_threshold))
    no_sell = ~sellable[:, 1:20].any(axis=1)
    cause_days[no_sell, 3] = np.int16(20)
    running_mdd = np.maximum.accumulate(drawdown, axis=1)
    upside_mask = (
        (net_close_wealth >= float(upside_wealth))
        & (time_above >= float(time_above_threshold))
        & (running_mdd <= float(drawdown_threshold))
    )
    upside_mask[:, :4] = False
    cause_days[:, 4] = _first_true(upside_mask)

    valid = filled & available
    event_code = np.zeros(rows, dtype=np.uint8)
    event_time = np.full(rows, np.nan, dtype=np.float32)
    event_score = np.full(rows, np.nan, dtype=np.float32)
    codes = (
        EVENT_TERMINAL_FAILURE,
        EVENT_MAJOR_DRAWDOWN,
        EVENT_POST_PEAK_FADE,
        EVENT_NO_LEGAL_SELL,
        EVENT_SUSTAINED_UPSIDE,
    )
    for row in np.flatnonzero(valid):
        days = cause_days[row]
        best_day = int(days.min())
        if best_day >= int(sentinel):
            event_code[row] = np.uint8(EVENT_CENSORED)
            event_time[row] = 20.0
            event_score[row] = 0.0
            continue
        cause_position = int(np.flatnonzero(days == best_day)[0])
        code = codes[cause_position]
        weight = 1.0 if code == EVENT_SUSTAINED_UPSIDE else -1.0
        event_code[row] = np.uint8(code)
        event_time[row] = float(best_day)
        event_score[row] = float(weight * (21 - best_day))
    return {
        "event_code": event_code,
        "event_time": event_time,
        "event_score": event_score,
        "no_legal_sell": no_sell,
        "cause_days": cause_days,
    }


def compute_daily_relevance(
    descriptors: Mapping[str, np.ndarray],
    *,
    entry_filled: np.ndarray,
    path_available: np.ndarray,
    forced_low: np.ndarray,
    symbol_idx: np.ndarray | None = None,
    pareto_device: str = "auto",
    pareto_block_size: int = 256,
) -> dict[str, np.ndarray]:
    filled = np.asarray(entry_filled, dtype=bool)
    available = np.asarray(path_available, dtype=bool)
    low = np.asarray(forced_low, dtype=bool)
    count = int(filled.size)
    if available.size != count or low.size != count:
        raise ValueError("daily relevance mask length mismatch")
    normal = filled & available & ~low
    terminal_low = filled & low

    trend_inputs = np.column_stack(
        [
            np.asarray(descriptors["speed5_log_per_day"], dtype=np.float64),
            np.asarray(descriptors["speed10_log_per_day"], dtype=np.float64),
            np.asarray(descriptors["speed20_log_per_day"], dtype=np.float64),
            np.asarray(descriptors["auc20_net_log"], dtype=np.float64),
            np.asarray(descriptors["time_above_break_even20"], dtype=np.float64),
            -np.asarray(descriptors["mdd20"], dtype=np.float64),
            -np.asarray(descriptors["post_peak_fade20"], dtype=np.float64),
        ]
    )
    trend_consistency = np.full(count, np.nan, dtype=np.float32)
    if bool(normal.any()):
        ranked_components = np.column_stack(
            [
                _rank_percentile(trend_inputs[normal, idx])
                for idx in range(trend_inputs.shape[1])
            ]
        )
        trend_consistency[normal] = np.nanmean(ranked_components, axis=1).astype(
            np.float32
        )
    trend_consistency[terminal_low] = 0.0

    pareto = np.column_stack(
        [
            np.asarray(descriptors["min_speed_5_10_20"], dtype=np.float32),
            np.asarray(descriptors["auc20_net_log"], dtype=np.float32),
            np.asarray(descriptors["time_above_break_even20"], dtype=np.float32),
            -np.asarray(descriptors["mdd20"], dtype=np.float32),
            -np.asarray(descriptors["post_peak_fade20"], dtype=np.float32),
        ]
    )
    dominated = np.full(count, -1, dtype=np.int32)
    dominating = np.full(count, -1, dtype=np.int32)
    dominance_margin = np.full(count, np.nan, dtype=np.float32)
    relevance = np.full(count, np.nan, dtype=np.float32)
    normal_positions = np.flatnonzero(normal)
    if normal_positions.size:
        current_dominated, current_dominating = pareto_dominance_counts(
            pareto[normal_positions],
            block_size=pareto_block_size,
            device=pareto_device,
        )
        dominated[normal_positions] = current_dominated
        dominating[normal_positions] = current_dominating
        denominator = max(int(normal_positions.size) - 1, 1)
        current_margin = (
            current_dominated.astype(np.float64) - current_dominating.astype(np.float64)
        ) / float(denominator)
        dominance_margin[normal_positions] = current_margin.astype(np.float32)
        symbols = (
            np.asarray(symbol_idx, dtype=np.int64)[normal_positions]
            if symbol_idx is not None
            else normal_positions.astype(np.int64)
        )
        order = np.lexsort(
            (
                symbols,
                trend_consistency[normal_positions].astype(np.float64),
                current_margin,
            )
        )
        percentile = np.empty(int(normal_positions.size), dtype=np.float32)
        if normal_positions.size == 1:
            percentile[order] = 1.0
        else:
            percentile[order] = np.arange(
                normal_positions.size, dtype=np.float32
            ) / float(normal_positions.size - 1)
        relevance[normal_positions] = percentile
    if bool(terminal_low.any()):
        dominated[terminal_low] = 0
        dominating[terminal_low] = int(normal_positions.size)
        dominance_margin[terminal_low] = -1.0
        relevance[terminal_low] = 0.0

    conditional = relevance.copy()
    action = np.full(count, np.nan, dtype=np.float32)
    action[~filled] = 0.0
    action[filled] = conditional[filled]
    return {
        "dominated_count": dominated,
        "dominating_count": dominating,
        "dominance_margin": dominance_margin,
        "pareto_relevance": relevance,
        "trend_consistency_v1": trend_consistency,
        "conditional_relevance": conditional,
        "action_relevance": action,
        "relevance_grade": relevance_grades(conditional),
    }


def compute_target_candidate_qualities(
    descriptors: Mapping[str, np.ndarray],
    *,
    entry_relative_path: np.ndarray,
    growth_multiplier: np.ndarray,
    exit_sellable: np.ndarray,
    delisted_path: np.ndarray,
    entry_filled: np.ndarray,
    path_available: np.ndarray,
    forced_low: np.ndarray,
    symbol_idx: np.ndarray,
    pareto_device: str = "auto",
    pareto_block_size: int = 256,
    competing_upside_wealth: float = 1.05,
    competing_time_above: float = 0.80,
    competing_drawdown: float = 0.08,
    competing_fade: float = 0.10,
) -> dict[str, np.ndarray]:
    """Compute every pre-registered target candidate for one signal date."""

    filled = np.asarray(entry_filled, dtype=bool).reshape(-1)
    available = np.asarray(path_available, dtype=bool).reshape(-1)
    low = np.asarray(forced_low, dtype=bool).reshape(-1)
    symbols = np.asarray(symbol_idx, dtype=np.int64).reshape(-1)
    common_utility = common_sustained_action_utility(
        descriptors,
        entry_filled=filled,
    )
    conditional_utility = common_utility.copy()
    conditional_utility[~filled] = np.nan
    raw_score = conditional_utility.astype(np.float64)
    finite_raw = np.isfinite(raw_score) & filled & available
    if bool(finite_raw.any()):
        floor = float(np.nanmin(raw_score[finite_raw])) - 1.0
        raw_score[low & filled & available] = floor
    raw_quality = _stable_daily_percentile(
        raw_score,
        eligible=filled & available,
        symbol_idx=symbols,
    )

    pareto = compute_daily_relevance(
        descriptors,
        entry_filled=filled,
        path_available=available,
        forced_low=low,
        symbol_idx=symbols,
        pareto_device=pareto_device,
        pareto_block_size=pareto_block_size,
    )
    events = compute_competing_risk_events(
        entry_relative_path,
        growth_multiplier,
        exit_sellable=exit_sellable,
        delisted_path=delisted_path,
        entry_filled=filled,
        path_available=available,
        upside_wealth=competing_upside_wealth,
        time_above_threshold=competing_time_above,
        drawdown_threshold=competing_drawdown,
        fade_threshold=competing_fade,
    )
    competing_quality = _stable_daily_percentile(
        np.asarray(events["event_score"], dtype=np.float64),
        eligible=filled & available,
        symbol_idx=symbols,
        secondary=conditional_utility,
    )
    direct_quality = _stable_daily_percentile(
        common_utility,
        eligible=np.isfinite(common_utility),
        symbol_idx=symbols,
    )
    return {
        "common_sustained_action_utility": common_utility,
        "raw_path_distribution_quality": raw_quality,
        "pareto_ordinal_quality": np.asarray(
            pareto["conditional_relevance"], dtype=np.float32
        ),
        "competing_risk_quality": competing_quality,
        "direct_listwise_utility_quality": direct_quality,
        "competing_event_code": np.asarray(events["event_code"], dtype=np.float32),
        "competing_event_time": np.asarray(events["event_time"], dtype=np.float32),
        "competing_event_score": np.asarray(events["event_score"], dtype=np.float32),
        **pareto,
    }


def _feature_view_paths(root: Path) -> FeatureViewPaths:
    return FeatureViewPaths(
        root=root,
        continuous=root / "snapshot_continuous.float32.dat",
        categorical=root / "snapshot_categorical.int64.dat",
        candidate_date_idx=root / "candidate_date_idx.int32.dat",
        candidate_symbol_idx=root / "candidate_symbol_idx.int32.dat",
        manifest=root / "feature_view_manifest.json",
    )


def _close_memmap(array: np.memmap) -> None:
    array.flush()
    mapping = getattr(array, "_mmap", None)
    if mapping is not None and not mapping.closed:
        mapping.close()


def _recoverable_feature_publish_attempt(
    output_root: Path,
    *,
    candidate_count: int,
    continuous_feature_count: int,
    categorical_feature_count: int,
) -> Path | None:
    task_root = output_root / "feature_view"
    attempts = sorted(task_root.glob("attempt_*"), reverse=True)
    if not attempts:
        return None
    attempt = attempts[0]
    progress_path = attempt / "progress.json"
    if not progress_path.exists():
        return None
    try:
        progress = json.loads(progress_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    error = str(progress.get("error", ""))
    if (
        str(progress.get("status", "")) != "failed"
        or str(progress.get("error_type", "")) != "PermissionError"
        or "partial" not in error
    ):
        return None
    partial_paths = _feature_view_paths(attempt / "partial")
    final_paths = _feature_view_paths(attempt)
    expected_sizes = {
        "continuous": int(candidate_count) * int(continuous_feature_count) * 4,
        "categorical": int(candidate_count) * int(categorical_feature_count) * 8,
        "candidate_date_idx": int(candidate_count) * 4,
        "candidate_symbol_idx": int(candidate_count) * 4,
    }
    for name, expected_size in expected_sizes.items():
        partial_path = getattr(partial_paths, name)
        final_path = getattr(final_paths, name)
        existing = [path for path in (partial_path, final_path) if path.exists()]
        if len(existing) != 1 or int(existing[0].stat().st_size) != int(expected_size):
            return None
    if final_paths.manifest.exists() or (attempt / "folds").exists():
        return None
    return attempt


def _safe_divide(
    numerator: np.ndarray,
    denominator: np.ndarray,
    *,
    minimum_denominator: float = 1.0e-12,
) -> np.ndarray:
    top = np.asarray(numerator, dtype=np.float64)
    bottom = np.asarray(denominator, dtype=np.float64)
    valid = (
        np.isfinite(top)
        & np.isfinite(bottom)
        & (np.abs(bottom) > float(minimum_denominator))
    )
    out = np.full(
        np.broadcast_shapes(top.shape, bottom.shape), np.nan, dtype=np.float64
    )
    np.divide(top, bottom, out=out, where=valid)
    return out.astype(np.float32)


def _lagged_return(values: np.ndarray, horizon: int) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    lag = int(horizon)
    if array.ndim != 2 or lag <= 0:
        raise ValueError("lagged return requires a 2D array and positive horizon")
    out = np.full(array.shape, np.nan, dtype=np.float32)
    if lag >= array.shape[0]:
        return out
    current = array[lag:]
    previous = array[:-lag]
    valid = (
        np.isfinite(current)
        & np.isfinite(previous)
        & (current > 0.0)
        & (previous > 0.0)
    )
    result = np.full(current.shape, np.nan, dtype=np.float64)
    np.divide(current, previous, out=result, where=valid)
    result[valid] -= 1.0
    out[lag:] = result.astype(np.float32)
    return out


def _rolling_sums(
    values: np.ndarray, window: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    array = np.asarray(values, dtype=np.float64)
    width = int(window)
    if array.ndim != 2 or width <= 0:
        raise ValueError("rolling sums require a 2D array and positive window")
    finite = np.isfinite(array)
    clean = np.where(finite, array, 0.0)
    count_cumsum = np.cumsum(finite.astype(np.int32), axis=0, dtype=np.int32)
    sum_cumsum = np.cumsum(clean, axis=0, dtype=np.float64)
    square_cumsum = np.cumsum(np.square(clean), axis=0, dtype=np.float64)

    def difference(cumulative: np.ndarray) -> np.ndarray:
        out = cumulative.copy()
        if width < array.shape[0]:
            out[width:] -= cumulative[:-width]
        return out

    return difference(sum_cumsum), difference(square_cumsum), difference(count_cumsum)


def _rolling_mean_std(values: np.ndarray, window: int) -> tuple[np.ndarray, np.ndarray]:
    sums, squares, counts = _rolling_sums(values, window)
    width = int(window)
    valid = counts == width
    mean = np.full(sums.shape, np.nan, dtype=np.float64)
    mean[valid] = sums[valid] / float(width)
    variance = np.full(sums.shape, np.nan, dtype=np.float64)
    variance[valid] = np.maximum(
        squares[valid] / float(width) - np.square(mean[valid]),
        0.0,
    )
    mean[: width - 1] = np.nan
    variance[: width - 1] = np.nan
    return mean.astype(np.float32), np.sqrt(variance).astype(np.float32)


def _rolling_mean(values: np.ndarray, window: int) -> np.ndarray:
    return _rolling_mean_std(values, window)[0]


def _rolling_min_max(values: np.ndarray, window: int) -> tuple[np.ndarray, np.ndarray]:
    frame = pd.DataFrame(np.asarray(values, dtype=np.float32), copy=False)
    rolling = frame.rolling(window=int(window), min_periods=int(window))
    return (
        rolling.min().to_numpy(dtype=np.float32, copy=False),
        rolling.max().to_numpy(dtype=np.float32, copy=False),
    )


def _rolling_corr(x: np.ndarray, y: np.ndarray, window: int) -> np.ndarray:
    left = np.asarray(x, dtype=np.float64)
    right = np.asarray(y, dtype=np.float64)
    if left.shape != right.shape or left.ndim != 2:
        raise ValueError("rolling correlation inputs must have the same 2D shape")
    valid = np.isfinite(left) & np.isfinite(right)
    clean_left = np.where(valid, left, 0.0)
    clean_right = np.where(valid, right, 0.0)
    width = int(window)

    def window_sum(values: np.ndarray) -> np.ndarray:
        cumulative = np.cumsum(values, axis=0, dtype=np.float64)
        out = cumulative.copy()
        if width < values.shape[0]:
            out[width:] -= cumulative[:-width]
        return out

    count = window_sum(valid.astype(np.float64))
    sx = window_sum(clean_left)
    sy = window_sum(clean_right)
    sxx = window_sum(np.square(clean_left))
    syy = window_sum(np.square(clean_right))
    sxy = window_sum(clean_left * clean_right)
    numerator = sxy - sx * sy / float(width)
    denominator = np.sqrt(
        np.maximum(sxx - np.square(sx) / float(width), 0.0)
        * np.maximum(syy - np.square(sy) / float(width), 0.0)
    )
    out = np.full(left.shape, np.nan, dtype=np.float64)
    usable = (count == width) & (denominator > 1.0e-12)
    out[usable] = numerator[usable] / denominator[usable]
    out[: width - 1] = np.nan
    return np.clip(out, -1.0, 1.0).astype(np.float32)


def _rolling_linear_stats(
    values: np.ndarray,
    window: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Causal rolling OLS slope, t-statistic, R2 and last-point residual."""

    array = np.asarray(values, dtype=np.float64)
    width = int(window)
    if array.ndim != 2 or width < 3:
        raise ValueError("rolling linear stats require a 2D array and window >= 3")
    finite = np.isfinite(array)
    clean = np.where(finite, array, 0.0)
    index = np.arange(array.shape[0], dtype=np.float64).reshape(-1, 1)

    def window_sum(values_: np.ndarray) -> np.ndarray:
        cumulative = np.cumsum(values_, axis=0, dtype=np.float64)
        out_ = cumulative.copy()
        if width < values_.shape[0]:
            out_[width:] -= cumulative[:-width]
        return out_

    count = window_sum(finite.astype(np.float64))
    sy = window_sum(clean)
    sy2 = window_sum(np.square(clean))
    global_xy = window_sum(clean * index)
    start_index = index - float(width - 1)
    sxy = global_xy - start_index * sy
    mean_x = float(width - 1) / 2.0
    sxx = float(width * (width * width - 1) / 12.0)
    covariance = sxy - mean_x * sy
    slope = covariance / max(sxx, 1.0e-12)
    intercept = sy / float(width) - slope * mean_x
    sst = np.maximum(sy2 - np.square(sy) / float(width), 0.0)
    ssr = np.square(slope) * sxx
    sse = np.maximum(sst - ssr, 0.0)
    r2 = np.divide(ssr, sst, out=np.zeros_like(ssr), where=sst > 1.0e-12)
    standard_error = np.sqrt(sse / float(max(width - 2, 1)) / max(sxx, 1.0e-12))
    t_value = np.divide(
        slope,
        standard_error,
        out=np.zeros_like(slope),
        where=standard_error > 1.0e-12,
    )
    fitted_last = intercept + slope * float(width - 1)
    residual = array - fitted_last
    valid = count == width
    outputs = []
    for current in (slope, t_value, r2, residual):
        result = np.where(valid, current, np.nan)
        result[: width - 1] = np.nan
        outputs.append(result.astype(np.float32))
    return tuple(outputs)  # type: ignore[return-value]


def _rolling_extreme_age(
    values: np.ndarray,
    window: int,
    *,
    mode: str,
) -> np.ndarray:
    """Return causal age of the most recent rolling high/low, normalized later."""

    array = np.asarray(values, dtype=np.float32)
    width = int(window)
    if array.ndim != 2 or width <= 0 or mode not in {"max", "min"}:
        raise ValueError(
            "rolling extreme age requires 2D values, positive window, max/min"
        )
    out = np.full(array.shape, np.nan, dtype=np.float32)
    for symbol in range(array.shape[1]):
        queue: deque[int] = deque()
        missing: deque[int] = deque()
        series = array[:, symbol]
        for date in range(array.shape[0]):
            value = float(series[date])
            if not math.isfinite(value):
                missing.append(date)
            else:
                if mode == "max":
                    while queue and float(series[queue[-1]]) <= value:
                        queue.pop()
                else:
                    while queue and float(series[queue[-1]]) >= value:
                        queue.pop()
                queue.append(date)
            oldest = date - width + 1
            while queue and queue[0] < oldest:
                queue.popleft()
            while missing and missing[0] < oldest:
                missing.popleft()
            if date >= width - 1 and not missing and queue:
                out[date, symbol] = float(date - queue[0])
    return out


def _candidate_cross_section_percentile(
    values: np.ndarray,
    date_idx: np.ndarray,
) -> np.ndarray:
    scores = np.asarray(values, dtype=np.float32)
    dates = np.asarray(date_idx, dtype=np.int32)
    if scores.ndim != 1 or dates.shape != scores.shape:
        raise ValueError(
            "candidate values and date_idx must be aligned one-dimensional arrays"
        )
    if scores.size and bool(np.any(dates[1:] < dates[:-1])):
        raise ValueError("candidate coordinates must be ordered by signal date")
    out = np.full(scores.shape, np.nan, dtype=np.float32)
    if not scores.size:
        return out
    boundaries = np.flatnonzero(np.r_[True, dates[1:] != dates[:-1], True])
    for start, stop in zip(boundaries[:-1], boundaries[1:]):
        out[start:stop] = _rank_percentile(scores[start:stop])
    return out


def _stable_category_hash(value: Any) -> np.int64:
    text_value = str(value or "").strip()
    if not text_value or text_value.lower() in {
        "nan",
        "none",
        "unknown",
        "unclassified",
    }:
        return np.int64(0)
    digest = hashlib.blake2b(text_value.encode("utf-8"), digest_size=8).digest()
    integer = int.from_bytes(digest, byteorder="big", signed=False) & ((1 << 63) - 1)
    return np.int64(integer or 1)


def _open_reference_f0_material(
    manifest: Mapping[str, Any],
) -> tuple[list[np.memmap], list[np.memmap]]:
    channels = dict(manifest.get("feature_channels", {}) or {})
    continuous = [
        sequence_training._open_memmap(channels[name], dtype="float32")
        for name in ("daily_raw", "daily_state", "turnover")
    ]
    masks = dict(manifest.get("masks", {}) or {})
    binary = []
    if "turnover_valid" in masks:
        binary.append(
            sequence_training._open_memmap(masks["turnover_valid"], dtype="bool")
        )
    return continuous, binary


def _stratified_block_bootstrap_delta(
    left: pd.DataFrame,
    right: pd.DataFrame,
    *,
    iterations: int = 2000,
    block_days: int = 20,
    seed: int = 20260725,
) -> dict[str, float]:
    joined = left[["trade_date", "year", "top1pct_u_mean"]].merge(
        right[["trade_date", "year", "top1pct_u_mean"]],
        on=["trade_date", "year"],
        suffixes=("_left", "_right"),
        how="inner",
    )
    joined = joined.replace([np.inf, -np.inf], np.nan).dropna()
    if joined.empty:
        return {"estimate": np.nan, "lower": np.nan, "upper": np.nan}
    observed = float(
        np.mean(joined["top1pct_u_mean_left"] - joined["top1pct_u_mean_right"])
    )
    rng = np.random.default_rng(int(seed))
    draws = np.empty(int(iterations), dtype=np.float64)
    year_arrays = {
        int(year): (
            group["top1pct_u_mean_left"].to_numpy(dtype=np.float64)
            - group["top1pct_u_mean_right"].to_numpy(dtype=np.float64)
        )
        for year, group in joined.groupby("year", sort=True)
    }
    for draw in range(int(iterations)):
        sampled_years: list[float] = []
        for values in year_arrays.values():
            count = int(values.size)
            blocks: list[np.ndarray] = []
            while sum(len(block) for block in blocks) < count:
                start = int(rng.integers(0, count))
                indices = (start + np.arange(int(block_days))) % count
                blocks.append(values[indices])
            sampled_years.append(float(np.mean(np.concatenate(blocks)[:count])))
        draws[draw] = float(np.mean(sampled_years))
    return {
        "estimate": observed,
        "lower": float(np.quantile(draws, 0.025)),
        "upper": float(np.quantile(draws, 0.975)),
    }


def _fit_category_vocabularies(
    categorical: np.ndarray,
    train_row_ids: np.ndarray,
    categorical_columns: np.ndarray,
    *,
    chunk_size: int = 500_000,
) -> list[np.ndarray]:
    vocabularies: list[set[int]] = [set() for _ in categorical_columns]
    for start in range(0, len(train_row_ids), int(chunk_size)):
        rows = train_row_ids[start : start + int(chunk_size)]
        block = np.asarray(
            categorical[np.ix_(rows, categorical_columns)], dtype=np.int64
        )
        for column in range(block.shape[1]):
            values = np.unique(block[:, column])
            vocabularies[column].update(
                int(value) for value in values if int(value) != 0
            )
    return [np.asarray(sorted(values), dtype=np.int64) for values in vocabularies]


def _map_categories(values: np.ndarray, vocabulary: np.ndarray) -> np.ndarray:
    raw = np.asarray(values, dtype=np.int64)
    vocab = np.asarray(vocabulary, dtype=np.int64)
    out = np.zeros(raw.shape, dtype=np.int32)
    if not vocab.size:
        return out
    positions = np.searchsorted(vocab, raw)
    valid = (raw != 0) & (positions < vocab.size)
    valid &= vocab[np.minimum(positions, vocab.size - 1)] == raw
    out[valid] = positions[valid].astype(np.int32) + 1
    return out


def _make_lgb_sequence(
    *,
    continuous: np.ndarray,
    categorical: np.ndarray,
    row_ids: np.ndarray,
    continuous_columns: np.ndarray,
    categorical_columns: np.ndarray,
    category_vocabularies: Sequence[np.ndarray],
    batch_size: int = 65_536,
) -> Any:
    import lightgbm as lgb

    class ResearchSequence(lgb.Sequence):
        def __init__(self) -> None:
            self.batch_size = int(batch_size)

        def __len__(self) -> int:
            return int(len(row_ids))

        def __getitem__(self, idx: Any) -> np.ndarray:
            if isinstance(idx, slice):
                local = np.arange(
                    0 if idx.start is None else int(idx.start),
                    len(row_ids) if idx.stop is None else int(idx.stop),
                    1 if idx.step is None else int(idx.step),
                    dtype=np.int64,
                )
            elif isinstance(idx, (list, tuple, np.ndarray)):
                local = np.asarray(idx, dtype=np.int64)
            elif isinstance(idx, (int, np.integer)):
                global_row = int(row_ids[int(idx)])
                continuous_values = np.asarray(
                    continuous[global_row, continuous_columns], dtype=np.float64
                )
                category_values = [
                    _map_categories(
                        np.asarray(
                            [categorical[global_row, int(column)]], dtype=np.int64
                        ),
                        category_vocabularies[position],
                    )[0]
                    for position, column in enumerate(categorical_columns)
                ]
                return np.concatenate(
                    [continuous_values, np.asarray(category_values, dtype=np.float64)]
                )
            else:
                raise TypeError(
                    f"unsupported LightGBM sequence index: {type(idx).__name__}"
                )
            global_rows = row_ids[local]
            continuous_values = np.asarray(
                continuous[np.ix_(global_rows, continuous_columns)], dtype=np.float64
            )
            if not categorical_columns.size:
                return continuous_values
            category_parts = []
            for position, column in enumerate(categorical_columns):
                category_parts.append(
                    _map_categories(
                        categorical[global_rows, int(column)],
                        category_vocabularies[position],
                    ).reshape(-1, 1)
                )
            return np.concatenate([continuous_values, *category_parts], axis=1).astype(
                np.float64, copy=False
            )

    return ResearchSequence()


def _lgb_dataset(
    *,
    sequence: Any,
    label: np.ndarray,
    feature_names: Sequence[str],
    categorical_count: int,
    group: np.ndarray | None = None,
    reference: Any | None = None,
) -> Any:
    import lightgbm as lgb

    continuous_count = len(feature_names) - int(categorical_count)
    categorical_positions = list(
        range(continuous_count, continuous_count + int(categorical_count))
    )
    return lgb.Dataset(
        sequence,
        label=np.asarray(label),
        group=None if group is None else np.asarray(group, dtype=np.int32),
        feature_name=list(feature_names),
        categorical_feature=categorical_positions,
        reference=reference,
        free_raw_data=False,
    )


@dataclass(frozen=True)
class ModelDataContext:
    study: dict[str, Any]
    pack_manifest: dict[str, Any]
    feature_manifest: dict[str, Any]
    target_manifest: dict[str, Any]
    continuous: np.memmap
    categorical: np.memmap
    candidate_date_idx: np.memmap
    candidate_symbol_idx: np.memmap
    target: np.memmap
    grades: np.memmap
    flags: np.memmap
    candidate_index_path: Path
    training_budget_amendment: dict[str, Any] | None = None


def _row_groups(
    row_ids: np.ndarray,
    candidate_date_idx: np.ndarray,
) -> list[np.ndarray]:
    rows = np.asarray(row_ids, dtype=np.int64)
    if not rows.size:
        return []
    dates = np.asarray(candidate_date_idx[rows], dtype=np.int32)
    if bool(np.any(dates[1:] < dates[:-1])):
        raise ValueError("candidate rows must remain date ordered")
    boundaries = np.flatnonzero(np.r_[True, dates[1:] != dates[:-1], True])
    return [rows[start:stop] for start, stop in zip(boundaries[:-1], boundaries[1:])]


def deterministic_epoch_rows(
    row_ids: np.ndarray,
    candidate_date_idx: np.ndarray,
    *,
    seed: int,
) -> np.ndarray:
    """Freeze one epoch's row order independently of any resource batch size."""

    groups = _row_groups(row_ids, candidate_date_idx)
    rng = np.random.default_rng(int(seed))
    rng.shuffle(groups)
    if not groups:
        return np.empty(0, dtype=np.int64)
    return np.concatenate(
        [rng.permutation(group).astype(np.int64, copy=False) for group in groups]
    )


def effective_batch_plan(
    row_ids: np.ndarray,
    candidate_date_idx: np.ndarray,
    *,
    effective_batch: int,
    seed: int,
) -> list[np.ndarray]:
    """Build exact cross-date optimizer batches; only the final batch may be short."""

    if int(effective_batch) <= 0:
        raise ValueError("effective_batch must be positive")
    ordered = deterministic_epoch_rows(
        row_ids,
        candidate_date_idx,
        seed=int(seed),
    )
    return [
        ordered[start : start + int(effective_batch)]
        for start in range(0, len(ordered), int(effective_batch))
    ]


def complete_date_step_plan(
    row_ids: np.ndarray,
    candidate_date_idx: np.ndarray,
    *,
    target_candidates: int,
    seed: int | None,
) -> list[list[np.ndarray]]:
    """Pack complete signal dates without ever truncating a relational set."""

    if int(target_candidates) <= 0:
        raise ValueError("target_candidates must be positive")
    groups = _row_groups(row_ids, candidate_date_idx)
    if seed is not None:
        np.random.default_rng(int(seed)).shuffle(groups)
    steps: list[list[np.ndarray]] = []
    pending: list[np.ndarray] = []
    count = 0
    for group in groups:
        if pending and count + len(group) > int(target_candidates):
            steps.append(pending)
            pending = []
            count = 0
        pending.append(group)
        count += len(group)
        if count >= int(target_candidates):
            steps.append(pending)
            pending = []
            count = 0
    if pending:
        steps.append(pending)
    return steps


def _micro_batches(rows: np.ndarray, micro_batch: int) -> Iterator[np.ndarray]:
    if int(micro_batch) <= 0:
        raise ValueError("micro_batch must be positive")
    values = np.asarray(rows, dtype=np.int64)
    for start in range(0, len(values), int(micro_batch)):
        yield values[start : start + int(micro_batch)]


def global_normalized_component(
    numerator: torch.Tensor,
    denominator: float | int,
) -> torch.Tensor:
    """Normalize a micro-batch numerator by its complete optimizer-batch divisor."""

    return numerator / max(float(denominator), 1.0)


def _f0_batch(
    context: ModelDataContext,
    row_ids: np.ndarray,
    preprocessor: Mapping[str, Any],
    *,
    material: tuple[Sequence[np.ndarray], Sequence[np.ndarray]] | None = None,
) -> np.ndarray:
    """Load arbitrary candidate rows while preserving their requested order."""

    rows = np.asarray(row_ids, dtype=np.int64)
    continuous, binary = (
        material
        if material is not None
        else _open_reference_f0_material(context.pack_manifest)
    )
    mean = np.asarray(preprocessor["mean"], dtype=np.float32)
    std = np.asarray(preprocessor["std"], dtype=np.float32)
    output = np.empty((len(rows), 180, len(mean)), dtype=np.float32)
    dates = np.asarray(context.candidate_date_idx[rows], dtype=np.int32)
    for date_idx in np.unique(dates):
        positions = np.flatnonzero(dates == int(date_idx))
        current = rows[positions]
        history_start = int(date_idx) - 179
        if history_start < 0:
            raise ValueError("F0 row lacks the frozen 180-day history")
        symbols = np.asarray(context.candidate_symbol_idx[current], dtype=np.int64)
        parts: list[np.ndarray] = []
        for array in continuous:
            block = np.asarray(
                array[history_start : int(date_idx) + 1, symbols, :],
                dtype=np.float32,
            )
            parts.append(np.transpose(block, (1, 0, 2)))
        for array in binary:
            block = np.asarray(
                array[history_start : int(date_idx) + 1, symbols],
                dtype=np.float32,
            )
            parts.append(np.transpose(block, (1, 0))[:, :, None])
        output[positions] = np.concatenate(parts, axis=2).astype(np.float32, copy=False)
    output = (output - mean.reshape(1, 1, -1)) / std.reshape(1, 1, -1)
    return np.nan_to_num(
        output,
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    ).astype(np.float32, copy=False)


class _SnapshotEncoder(nn.Module):
    def __init__(
        self,
        *,
        n_num_features: int,
        cat_cardinalities: Sequence[int],
        hidden_width: int,
        layers: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.embeddings = nn.ModuleList()
        embedding_dim = 0
        for cardinality in cat_cardinalities:
            dim = min(16, max(4, int(math.ceil(math.sqrt(max(cardinality, 1))))))
            self.embeddings.append(nn.Embedding(int(cardinality), dim))
            embedding_dim += dim
        modules: list[nn.Module] = []
        input_dim = int(n_num_features) + int(embedding_dim)
        for layer in range(int(layers)):
            modules.extend(
                [
                    nn.Linear(input_dim if layer == 0 else hidden_width, hidden_width),
                    nn.ReLU(),
                    nn.Dropout(float(dropout)),
                ]
            )
        self.network = nn.Sequential(*modules)
        self.output_dim = int(hidden_width)

    def forward(self, x_num: torch.Tensor, x_cat: torch.Tensor) -> torch.Tensor:
        parts = [x_num]
        for idx, embedding in enumerate(self.embeddings):
            parts.append(embedding(x_cat[:, idx]))
        return self.network(torch.cat(parts, dim=1))


class _TabMModel(nn.Module):
    def __init__(
        self,
        *,
        n_num_features: int,
        cat_cardinalities: Sequence[int],
        config: Mapping[str, Any],
    ) -> None:
        super().__init__()
        import tabm

        self.model = tabm.TabM.make(
            n_num_features=int(n_num_features),
            cat_cardinalities=[int(item) for item in cat_cardinalities] or None,
            d_out=2,
            n_blocks=int(config["blocks"]),
            d_block=int(config["block_width"]),
            dropout=float(config["dropout"]),
            k=int(config["ensemble_size"]),
        )

    def forward(
        self, x_num: torch.Tensor, x_cat: torch.Tensor
    ) -> dict[str, torch.Tensor]:
        output = self.model(x_num, x_cat if x_cat.shape[1] else None)
        return {
            "quality": torch.sigmoid(output[..., 0]),
            "fill_logit": output[..., 1],
        }


class _PatchTSTStudentT(nn.Module):
    def __init__(self, *, input_channels: int, config: Mapping[str, Any]) -> None:
        super().__init__()
        patch_length = int(config["patch_length"])
        stride = int(config["stride"])
        patch_count = (180 - patch_length) // stride + 1
        d_model = int(config["d_model"])
        self.patch_length = patch_length
        self.stride = stride
        self.input_channels = int(input_channels)
        self.patch_projection = nn.Linear(patch_length, d_model)
        self.position = nn.Parameter(torch.zeros(1, patch_count, d_model))
        self.channel = nn.Parameter(torch.zeros(1, input_channels, d_model))
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=int(config["heads"]),
            dim_feedforward=int(config["d_ff"]),
            dropout=float(config["dropout"]),
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=int(config["layers"])
        )
        flattened = input_channels * d_model
        self.head = nn.Sequential(
            nn.LayerNorm(flattened),
            nn.Linear(flattened, 256),
            nn.GELU(),
            nn.Dropout(float(config["dropout"])),
        )
        self.location = nn.Linear(256, 20)
        self.scale = nn.Linear(256, 20)
        self.terminal = nn.Linear(256, 1)
        self.fill = nn.Linear(256, 1)

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        batch, length, channels = x.shape
        if length != 180 or channels != self.input_channels:
            raise ValueError("PatchTST input must be [batch,180,input_channels]")
        patches = x.transpose(1, 2).unfold(2, self.patch_length, self.stride)
        projected = self.patch_projection(
            patches.reshape(batch * channels, patches.shape[2], self.patch_length)
        )
        channel = self.channel.expand(batch, -1, -1).reshape(batch * channels, 1, -1)
        encoded = self.encoder(projected + self.position + channel)
        pooled = encoded.mean(dim=1).reshape(batch, channels * encoded.shape[-1])
        hidden = self.head(pooled)
        return {
            "location": self.location(hidden),
            "scale": torch_functional.softplus(self.scale(hidden)) + 1.0e-4,
            "terminal_logit": self.terminal(hidden).squeeze(-1),
            "fill_logit": self.fill(hidden).squeeze(-1),
        }


class _MarketIndustryDeepSets(nn.Module):
    def __init__(
        self,
        *,
        n_num_features: int,
        cat_cardinalities: Sequence[int],
        industry_position: int,
        config: Mapping[str, Any],
    ) -> None:
        super().__init__()
        width = int(config["item_width"])
        self.industry_position = int(industry_position)
        self.encoder = _SnapshotEncoder(
            n_num_features=n_num_features,
            cat_cardinalities=cat_cardinalities,
            hidden_width=width,
            layers=int(config["hidden_layers"]),
            dropout=float(config["dropout"]),
        )
        context_width = int(config["context_width"])
        self.market = nn.Sequential(nn.Linear(width * 2, context_width), nn.ReLU())
        self.industry = nn.Sequential(nn.Linear(width, context_width), nn.ReLU())
        self.residual = nn.Sequential(nn.Linear(width, context_width), nn.ReLU())
        self.gate = nn.Sequential(nn.Linear(context_width, context_width), nn.Sigmoid())
        self.quality = nn.Linear(context_width, 1)
        self.fill = nn.Linear(context_width, 1)

    def forward(
        self, x_num: torch.Tensor, x_cat: torch.Tensor
    ) -> dict[str, torch.Tensor]:
        item = self.encoder(x_num, x_cat)
        market_mean = item.mean(dim=0, keepdim=True)
        market_max = item.max(dim=0, keepdim=True).values
        market_context = self.market(torch.cat([market_mean, market_max], dim=1))
        industry_id = x_cat[:, self.industry_position]
        _unique, inverse = torch.unique(industry_id, sorted=True, return_inverse=True)
        industry_sum = torch.zeros(
            int(inverse.max().item()) + 1,
            item.shape[1],
            device=item.device,
            dtype=item.dtype,
        )
        industry_sum.index_add_(0, inverse, item)
        counts = torch.bincount(inverse, minlength=industry_sum.shape[0]).clamp_min(1)
        industry_mean = industry_sum / counts.to(item.dtype).unsqueeze(1)
        industry_context = self.industry(industry_mean[inverse])
        global_context = market_context.expand(len(item), -1)
        hidden = (
            self.residual(item)
            + industry_context
            + self.gate(global_context) * global_context
        )
        return {
            "quality": torch.sigmoid(self.quality(hidden).squeeze(-1)),
            "fill_logit": self.fill(hidden).squeeze(-1),
        }


class _DeepHitCompetingRisk(nn.Module):
    def __init__(
        self,
        *,
        n_num_features: int,
        cat_cardinalities: Sequence[int],
        config: Mapping[str, Any],
    ) -> None:
        super().__init__()
        self.encoder = _SnapshotEncoder(
            n_num_features=n_num_features,
            cat_cardinalities=cat_cardinalities,
            hidden_width=int(config["hidden_width"]),
            layers=int(config["layers"]),
            dropout=float(config["dropout"]),
        )
        self.joint = nn.Linear(self.encoder.output_dim, 101)
        self.fill = nn.Linear(self.encoder.output_dim, 1)

    def forward(
        self, x_num: torch.Tensor, x_cat: torch.Tensor
    ) -> dict[str, torch.Tensor]:
        hidden = self.encoder(x_num, x_cat)
        return {
            "joint_logits": self.joint(hidden),
            "fill_logit": self.fill(hidden).squeeze(-1),
        }


class _NeuralNDCGMLP(nn.Module):
    def __init__(
        self,
        *,
        n_num_features: int,
        cat_cardinalities: Sequence[int],
        config: Mapping[str, Any],
    ) -> None:
        super().__init__()
        self.encoder = _SnapshotEncoder(
            n_num_features=n_num_features,
            cat_cardinalities=cat_cardinalities,
            hidden_width=int(config["hidden_width"]),
            layers=int(config["layers"]),
            dropout=float(config["dropout"]),
        )
        self.score = nn.Linear(self.encoder.output_dim, 1)
        self.fill = nn.Linear(self.encoder.output_dim, 1)

    def forward(
        self, x_num: torch.Tensor, x_cat: torch.Tensor
    ) -> dict[str, torch.Tensor]:
        hidden = self.encoder(x_num, x_cat)
        return {
            "raw_score": self.score(hidden).squeeze(-1),
            "fill_logit": self.fill(hidden).squeeze(-1),
        }


def neural_sort_matrix(scores: torch.Tensor, *, temperature: float) -> torch.Tensor:
    values = scores.reshape(-1)
    count = int(values.numel())
    if count == 0:
        return values.reshape(0, 0)
    absolute = torch.abs(values[:, None] - values[None, :])
    penalties = absolute.sum(dim=1)
    scaling = (
        count
        + 1
        - 2 * torch.arange(1, count + 1, device=values.device, dtype=values.dtype)
    )
    logits = scaling[:, None] * values[None, :] - penalties[None, :]
    return torch.softmax(logits / float(temperature), dim=1)


def neural_ndcg_loss(
    scores: torch.Tensor,
    grades: torch.Tensor,
    *,
    temperature: float,
    gain: Sequence[float],
) -> torch.Tensor:
    probability = neural_sort_matrix(scores, temperature=temperature)
    gains = torch.as_tensor(gain, device=scores.device, dtype=scores.dtype)
    true_gain = gains[grades.to(torch.long).clamp(0, len(gain) - 1)]
    expected_gain = probability @ true_gain
    discount = 1.0 / torch.log2(
        torch.arange(2, len(scores) + 2, device=scores.device, dtype=scores.dtype)
    )
    dcg = torch.sum(expected_gain * discount)
    ideal = torch.sum(torch.sort(true_gain, descending=True).values * discount)
    return -(dcg / ideal.clamp_min(1.0e-8))


def deephit_joint_targets(
    event_code: torch.Tensor,
    event_time: torch.Tensor,
) -> torch.Tensor:
    code = event_code.to(torch.long)
    time_index = event_time.to(torch.long).clamp(1, 20) - 1
    target = torch.full_like(code, 100)
    observed = code > 0
    target[observed] = (code[observed] - 1) * 20 + time_index[observed]
    return target


def deephit_expected_event_score(probability: torch.Tensor) -> torch.Tensor:
    event = probability[:, :100].reshape(-1, 5, 20)
    weights = torch.as_tensor(
        [-1.0, -1.0, -1.0, -1.0, 1.0],
        dtype=event.dtype,
        device=event.device,
    ).reshape(1, 5, 1)
    time_value = torch.arange(
        20, 0, -1, dtype=event.dtype, device=event.device
    ).reshape(1, 1, 20)
    return torch.sum(event * weights * time_value, dim=(1, 2))


def deephit_ranking_loss(
    probability: torch.Tensor,
    event_code: torch.Tensor,
    event_time: torch.Tensor,
) -> torch.Tensor:
    """Bounded deterministic DeepHit-style cause-specific CIF ranking loss."""

    event = probability[:, :100].reshape(-1, 5, 20)
    cif = torch.cumsum(event, dim=2)
    code = event_code.to(torch.long)
    time = event_time.to(torch.long).clamp(1, 20)
    losses: list[torch.Tensor] = []
    for cause in range(1, 6):
        anchors = torch.nonzero(code == cause, as_tuple=False).flatten()
        if int(anchors.numel()) < 1 or int(probability.shape[0]) < 2:
            continue
        partners = torch.roll(
            torch.arange(probability.shape[0], device=probability.device),
            shifts=cause,
        )[anchors]
        comparable = (code[partners] == 0) | (time[partners] > time[anchors])
        if not bool(comparable.any()):
            continue
        left = anchors[comparable]
        right = partners[comparable]
        horizon = time[left] - 1
        left_cif = cif[left, cause - 1, horizon]
        right_cif = cif[right, cause - 1, horizon]
        losses.append(torch_functional.softplus(-(left_cif - right_cif)).mean())
    if not losses:
        return probability.sum() * 0.0
    return torch.stack(losses).mean()


def student_t_nll(
    target: torch.Tensor,
    location: torch.Tensor,
    scale: torch.Tensor,
    *,
    df: float,
) -> torch.Tensor:
    """Elementwise Student-t NLL with an auditable, distribution-free formula."""

    nu = torch.as_tensor(float(df), device=location.device, dtype=torch.float32)
    loc = location.float()
    sigma = scale.float().clamp_min(1.0e-6)
    value = target.float()
    log_normalizer = (
        torch.lgamma(nu / 2.0)
        + 0.5 * (torch.log(nu) + math.log(math.pi))
        + torch.log(sigma)
        - torch.lgamma((nu + 1.0) / 2.0)
    )
    standardized = (value - loc) / sigma
    return log_normalizer + ((nu + 1.0) / 2.0) * torch.log1p(standardized.square() / nu)


def _fixed_student_t_sobol(df: float = 5.0) -> np.ndarray:
    sampler = stats.qmc.Sobol(d=3, scramble=True, seed=1729)
    uniforms = np.clip(sampler.random_base2(m=6), 1.0e-7, 1.0 - 1.0e-7)
    return stats.t.ppf(uniforms, df=float(df)).astype(np.float32)


STUDENT_T_SOBOL_64X3 = _fixed_student_t_sobol()


def student_t_path_quality(
    location: np.ndarray,
    scale: np.ndarray,
    terminal_probability: np.ndarray,
) -> np.ndarray:
    mu = np.asarray(location, dtype=np.float32)
    sigma = np.asarray(scale, dtype=np.float32)
    terminal = np.asarray(terminal_probability, dtype=np.float32).reshape(-1)
    if mu.ndim != 2 or mu.shape[1] != 20 or sigma.shape != mu.shape:
        raise ValueError("Student-t path outputs must have shape [rows,20]")
    selected_mu = mu[:, [4, 9, 19]] / np.asarray([5.0, 10.0, 20.0])
    selected_scale = sigma[:, [4, 9, 19]] / np.asarray([5.0, 10.0, 20.0])
    draws = (
        selected_mu[:, None, :]
        + selected_scale[:, None, :] * STUDENT_T_SOBOL_64X3[None, :, :]
    )
    expected_minimum = np.mean(np.min(draws, axis=2), axis=1)
    return (expected_minimum * (1.0 - terminal)).astype(np.float32)


def _atomic_torch_save(path: Path, payload: Mapping[str, Any]) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    torch.save(dict(payload), temporary)
    os.replace(temporary, path)
    return str(path.resolve())


def _resource_snapshot() -> dict[str, Any]:
    payload: dict[str, Any] = {}
    try:
        import psutil

        process = psutil.Process(os.getpid())
        payload.update(
            {
                "process_rss_bytes": int(process.memory_info().rss),
                "process_cpu_percent": float(process.cpu_percent(interval=None)),
                "system_memory_percent": float(psutil.virtual_memory().percent),
            }
        )
    except Exception:
        pass
    if torch.cuda.is_available():
        payload.update(
            {
                "gpu_allocated_bytes": int(torch.cuda.memory_allocated()),
                "gpu_reserved_bytes": int(torch.cuda.memory_reserved()),
                "gpu_peak_allocated_bytes": int(torch.cuda.max_memory_allocated()),
                "gpu_peak_reserved_bytes": int(torch.cuda.max_memory_reserved()),
            }
        )
        try:
            import pynvml

            pynvml.nvmlInit()
            handle = pynvml.nvmlDeviceGetHandleByIndex(0)
            utilization = pynvml.nvmlDeviceGetUtilizationRates(handle)
            memory = pynvml.nvmlDeviceGetMemoryInfo(handle)
            payload.update(
                {
                    "gpu_utilization_percent": int(utilization.gpu),
                    "gpu_memory_utilization_percent": int(utilization.memory),
                    "gpu_device_used_bytes": int(memory.used),
                    "gpu_temperature_c": int(
                        pynvml.nvmlDeviceGetTemperature(
                            handle, pynvml.NVML_TEMPERATURE_GPU
                        )
                    ),
                }
            )
        except Exception:
            pass
    return payload


class _TrainingMonitor:
    def __init__(
        self,
        *,
        output_dir: Path,
        base: Mapping[str, Any],
        heartbeat_seconds: float = PROGRESS_HEARTBEAT_SECONDS,
        console_seconds: float = CONSOLE_EVENT_SECONDS,
        checkpoint_seconds: float = CHECKPOINT_INTERVAL_SECONDS,
    ) -> None:
        self.output_dir = output_dir
        self.progress_path = output_dir / "progress.json"
        self.events_path = output_dir / "training_events.jsonl"
        self.base = {
            "progress_schema": TRAINING_PROGRESS_VERSION,
            **dict(base),
        }
        self.heartbeat_seconds = float(heartbeat_seconds)
        self.console_seconds = float(console_seconds)
        self.checkpoint_seconds = float(checkpoint_seconds)
        self.started = time.perf_counter()
        self.last_write = 0.0
        self.last_console = 0.0
        self.last_checkpoint = time.perf_counter()
        self.last_trim_batch = 0
        self.trim_step = 0
        self.phase = ""
        self.rate_window: deque[tuple[float, int]] = deque(maxlen=32)
        self.latest: dict[str, Any] = {}

    @property
    def pause_path(self) -> Path:
        return self.output_dir / "pause.request"

    def pause_requested(self) -> bool:
        return self.pause_path.exists()

    def checkpoint_due(self) -> bool:
        return time.perf_counter() - self.last_checkpoint >= self.checkpoint_seconds

    def heartbeat_due(self) -> bool:
        return time.perf_counter() - self.last_write >= self.heartbeat_seconds

    def checkpoint_saved(self) -> None:
        self.last_checkpoint = time.perf_counter()

    def relieve_memory_pressure(self) -> dict[str, Any] | None:
        """Trim the mapped working set on a fixed cadence and report the event."""
        self.trim_step += 1
        self.last_trim_batch, event = (
            sequence_training._maybe_trim_training_working_set(
                batch_count=self.trim_step,
                last_trim_batch=self.last_trim_batch,
            )
        )
        if event is not None:
            _append_jsonl(self.events_path, {"event": "working_set_trim", **event})
        return event

    def memory_exhausted(self) -> dict[str, Any] | None:
        """Return diagnostics when memory stays critically low after a forced trim."""
        available_before = sequence_training._available_physical_memory_gb()
        if (
            available_before is None
            or available_before >= LOW_MEMORY_PAUSE_AVAILABLE_GB
        ):
            return None
        trim_succeeded = bool(sequence_training._trim_working_set())
        available_after = sequence_training._available_physical_memory_gb()
        if available_after is None or available_after >= LOW_MEMORY_PAUSE_AVAILABLE_GB:
            return None
        process_memory = sequence_training._current_process_memory_gb()
        return {
            "available_before_gb": float(available_before),
            "available_after_gb": float(available_after),
            "trim_succeeded": trim_succeeded,
            "working_set_gb": process_memory["working_set_gb"],
            "private_gb": process_memory["private_gb"],
            "pause_threshold_gb": float(LOW_MEMORY_PAUSE_AVAILABLE_GB),
        }

    def report(
        self,
        *,
        status: str,
        phase: str,
        samples_completed: int,
        samples_total: int,
        epoch: int | None = None,
        max_epochs: int | None = None,
        step: int | None = None,
        steps_total: int | None = None,
        loss: float | None = None,
        best_development_loss: float | None = None,
        patience_used: int | None = None,
        force: bool = False,
        event: str | None = None,
        extra: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        now = time.perf_counter()
        if str(phase) != self.phase:
            self.phase = str(phase)
            self.rate_window.clear()
        self.rate_window.append((now, int(samples_completed)))
        while len(self.rate_window) > 2 and now - self.rate_window[0][0] > 180.0:
            self.rate_window.popleft()
        rate = 0.0
        if len(self.rate_window) >= 2:
            elapsed = self.rate_window[-1][0] - self.rate_window[0][0]
            advanced = self.rate_window[-1][1] - self.rate_window[0][1]
            if elapsed > 0.0 and advanced >= 0:
                rate = float(advanced / elapsed)
        remaining = max(int(samples_total) - int(samples_completed), 0)
        eta = float(remaining / rate) if rate > 0.0 else None
        should_write = bool(force or now - self.last_write >= self.heartbeat_seconds)
        payload: dict[str, Any] = {
            **self.base,
            "status": str(status),
            "phase": str(phase),
            "updated_at": _now(),
            "elapsed_seconds": float(now - self.started),
            "samples_completed": int(samples_completed),
            "samples_total": int(samples_total),
            "samples_per_second": float(rate),
            "eta_seconds": eta,
            "epoch": int(epoch) if epoch is not None else None,
            "max_epochs": int(max_epochs) if max_epochs is not None else None,
            "step": int(step) if step is not None else None,
            "steps_total": int(steps_total) if steps_total is not None else None,
            "loss": float(loss) if loss is not None and math.isfinite(loss) else None,
            "best_development_loss": (
                float(best_development_loss)
                if best_development_loss is not None
                and math.isfinite(best_development_loss)
                else None
            ),
            "patience_used": (
                int(patience_used) if patience_used is not None else None
            ),
            **(_resource_snapshot() if should_write else {}),
            **dict(extra or {}),
        }
        self.latest = payload
        if should_write:
            _atomic_write_json(self.progress_path, payload)
            _append_jsonl(
                self.events_path,
                {
                    "event": str(event or "heartbeat"),
                    **payload,
                },
            )
            self.last_write = now
        should_console = bool(
            event
            in {
                "training_started",
                "epoch_end",
                "checkpoint",
                "paused",
                "failed",
                "completed",
            }
            or now - self.last_console >= self.console_seconds
        )
        if should_console and should_write:
            print(
                json.dumps(
                    {
                        "event": str(event or "training_progress"),
                        "model_id": payload.get("model_id"),
                        "fold_id": payload.get("fold_id"),
                        "phase": payload["phase"],
                        "epoch": payload["epoch"],
                        "step": payload["step"],
                        "samples_per_second": payload["samples_per_second"],
                        "eta_seconds": payload["eta_seconds"],
                        "loss": payload["loss"],
                        "gpu_reserved_bytes": payload.get("gpu_reserved_bytes", 0),
                    },
                    ensure_ascii=False,
                    default=_json_default,
                ),
                flush=True,
            )
            self.last_console = now
        return payload


def _capture_rng_state() -> dict[str, Any]:
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch_cpu": torch.get_rng_state(),
        "torch_cuda": torch.cuda.get_rng_state_all()
        if torch.cuda.is_available()
        else [],
    }


def _restore_rng_state(payload: Mapping[str, Any]) -> None:
    if "python" in payload:
        random.setstate(payload["python"])
    if "numpy" in payload:
        np.random.set_state(payload["numpy"])
    if "torch_cpu" in payload:
        torch.set_rng_state(payload["torch_cpu"])
    if torch.cuda.is_available() and payload.get("torch_cuda"):
        torch.cuda.set_rng_state_all(list(payload["torch_cuda"]))


def _save_last_training_checkpoint(
    *,
    output_dir: Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler: Any,
    resolved_config: Mapping[str, Any],
    epoch: int,
    next_step_index: int,
    best_epoch: int,
    best_development_loss: float,
    best_state: Mapping[str, torch.Tensor] | None,
    epochs_without_improvement: int,
    training_log: Sequence[Mapping[str, Any]],
    epoch_loss_sum: float = 0.0,
    epoch_sample_count: int = 0,
) -> Path:
    path = output_dir / "last_checkpoint.pt"
    _atomic_torch_save(
        path,
        {
            "schema": TRAINING_CHECKPOINT_VERSION,
            "saved_at": _now(),
            "training_parameters": dict(resolved_config),
            "execution_semantics_version": resolved_config.get(
                "execution_semantics_version"
            ),
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scaler_state_dict": scaler.state_dict(),
            "epoch": int(epoch),
            "next_step_index": int(next_step_index),
            "best_epoch": int(best_epoch),
            "best_development_loss": float(best_development_loss),
            "best_state": dict(best_state) if best_state is not None else None,
            "epochs_without_improvement": int(epochs_without_improvement),
            "training_log": [dict(item) for item in training_log],
            "epoch_loss_sum": float(epoch_loss_sum),
            "epoch_sample_count": int(epoch_sample_count),
            "rng_state": _capture_rng_state(),
        },
    )
    return path


def _load_last_training_checkpoint(
    *,
    output_dir: Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler: Any,
    resolved_config: Mapping[str, Any],
) -> dict[str, Any] | None:
    path = output_dir / "last_checkpoint.pt"
    if not path.exists():
        return None
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if str(payload.get("schema", "")) != TRAINING_CHECKPOINT_VERSION:
        raise ValueError("training checkpoint schema mismatch")
    if dict(payload.get("training_parameters", {}) or {}) != dict(resolved_config):
        raise ValueError("training checkpoint parameters changed")
    if payload.get("execution_semantics_version") != resolved_config.get(
        "execution_semantics_version"
    ):
        raise ValueError("training checkpoint execution semantics mismatch")
    model.load_state_dict(payload["model_state_dict"])
    optimizer.load_state_dict(payload["optimizer_state_dict"])
    scaler.load_state_dict(payload["scaler_state_dict"])
    _restore_rng_state(dict(payload.get("rng_state", {}) or {}))
    return dict(payload)


def _checkpoint_or_pause(
    *,
    monitor: _TrainingMonitor | None,
    force_checkpoint: bool,
    checkpoint_callback: Any,
    progress: Mapping[str, Any],
) -> None:
    memory_pause = None
    requested_pause = False
    if monitor is not None:
        monitor.relieve_memory_pressure()
        if (
            monitor.trim_step
            % sequence_training.WORKING_SET_MEMORY_CHECK_INTERVAL_BATCHES
            == 0
        ):
            memory_pause = monitor.memory_exhausted()
        requested_pause = monitor.pause_requested()
    pause = bool(requested_pause or memory_pause is not None)
    due = bool(monitor is not None and monitor.checkpoint_due())
    if not (force_checkpoint or pause or due):
        return
    checkpoint_path = Path(checkpoint_callback())
    if monitor is not None:
        monitor.checkpoint_saved()
        extra = {"last_checkpoint": str(checkpoint_path.resolve())}
        if memory_pause is not None:
            extra["memory_pause"] = memory_pause
        monitor.report(
            status="paused" if pause else "training",
            phase=str(progress.get("phase", "training")),
            samples_completed=int(progress.get("samples_completed", 0)),
            samples_total=int(progress.get("samples_total", 0)),
            epoch=int(progress.get("epoch", 0)),
            max_epochs=int(progress.get("max_epochs", 0)),
            step=int(progress.get("step", 0)),
            steps_total=int(progress.get("steps_total", 0)),
            loss=progress.get("loss"),
            best_development_loss=progress.get("best_development_loss"),
            patience_used=progress.get("patience_used"),
            force=True,
            event="paused" if pause else "checkpoint",
            extra=extra,
        )
    if pause:
        if requested_pause:
            try:
                assert monitor is not None
                monitor.pause_path.unlink()
            except FileNotFoundError:
                pass
        if memory_pause is not None:
            raise TrainingPaused(
                f"training paused on low available memory at {checkpoint_path.resolve()}"
            )
        raise TrainingPaused(f"training paused safely at {checkpoint_path.resolve()}")
