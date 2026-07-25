from __future__ import annotations

import argparse
import gc
import hashlib
import heapq
import json
import math
import os
import time
from collections import deque
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable, Iterator, Mapping, Sequence

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from scipy import stats

from daily_research.path_policy import qdp_v2_sequence_path_training as sequence_training


WORKSPACE_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_STUDY_PATH = WORKSPACE_ROOT / "daily_research/studies/seq100_pit_path_relevance_v1.json"
DEFAULT_OUTPUT_ROOT = (
    WORKSPACE_ROOT
    / "daily_research/output/path_policy/studies/seq100_pit_path_relevance_v1"
)

PATH_TARGET_PROFILE = "consistent_pareto5_v1"
TREND_FALLBACK_PROFILE = "trend_consistency_v1"
TARGET_ARTIFACT_TYPE = "seq100_path_relevance_target_view"
TARGET_FLAG_ENTRY_FILLED = 1 << 0
TARGET_FLAG_PATH_AVAILABLE = 1 << 1
TARGET_FLAG_FORCED_LOW = 1 << 2
TARGET_FLAG_TERMINAL_FAILURE = 1 << 3
TARGET_FLAG_NO_LEGAL_SELL = 1 << 4
TARGET_FLAG_PRICE_LABEL_VALID = 1 << 5

TARGET_FLOAT_FIELDS = tuple(
    [f"close_net_log_d{day:02d}" for day in range(1, 21)]
    + [
        "r5_net",
        "r10_net",
        "r20_net",
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
        "dominance_margin",
        "pareto_relevance",
        "conditional_relevance",
        "action_relevance",
    ]
)
TARGET_FIELD_INDEX = {name: idx for idx, name in enumerate(TARGET_FLOAT_FIELDS)}

GRADE_THRESHOLDS = (0.50, 0.80, 0.95, 0.99)
MODEL_IDS = (
    "lgbm_multi_reg",
    "lgbm_lambdarank",
    "tabm_multioutput",
    "gru_student_t_f0",
    "temporal_set_hybrid",
)
FORMAL_FOLD_YEARS = (2023, 2024, 2025)
FEATURE_WINDOWS = (5, 10, 20, 40, 60)
FEATURE_RETURN_HORIZONS = (1, 2, 5, 10, 20, 40, 60)
MARKET_GROUPS = ("all", "csi300", "csi500", "sse50")
MARKET_METRICS = (
    "ret1_mean",
    "ret5_mean",
    "ret20_mean",
    "vol20_mean",
    "drawdown60_mean",
    "breadth_ret1_positive",
    "breadth_ret5_positive",
    "breadth_above_ma20",
    "ret1_dispersion",
    "log_total_amount",
    "suspended_rate",
    "st_rate",
    "up_limit_rate",
    "down_limit_rate",
)
F4_CONTINUOUS_FEATURES = (
    "industry_ret1_mean",
    "industry_ret5_mean",
    "industry_breadth_ret1_positive",
    "industry_ret1_dispersion",
    "industry_relative_ret5",
    "industry_relative_ret20",
    "industry_member_count_log",
    "industry_source_age_days",
)
F5_CONTINUOUS_FEATURES = (
    "log_total_market_value",
    "log_circulating_market_value",
    "circulating_market_value_ratio",
    "signed_log_pe",
    "signed_log_pb",
    "log_turnover_rate",
    "log_total_share",
    "log_float_share",
    "float_share_ratio",
    "share_source_age_days",
    "listing_age_days",
    "is_csi300_member",
    "is_csi500_member",
    "is_sse50_member",
    "is_st_today",
    "is_suspended_today",
    "is_delisted_today",
    "st_rate_20d",
    "suspended_rate_20d",
    "valuation_missing",
    "share_capital_missing",
    "industry_missing",
)
CATEGORICAL_FEATURES = (
    ("industry_hash", "F4"),
    ("industry_broad_hash", "F4"),
    ("board_hash", "F5"),
    ("exchange_hash", "F5"),
    ("list_status_hash", "F5"),
)


@dataclass(frozen=True)
class PathTargetSpec:
    profile: str = PATH_TARGET_PROFILE
    entry_anchor: str = "actual_next_open_if_filled"
    primary_horizon_days: int = 20
    continuation_diagnostic_days: tuple[int, ...] = (40, 60)
    cost_profile: str = "pack_manifest_bound_double_slippage_proportional_cost"
    conditional_relevance_scope: str = "entry_filled_candidates"
    unfilled_action_relevance: float = 0.0
    terminal_unrecoverable_value: float = 0.0
    daily_grouping: str = "signal_trade_date"
    pareto_dimensions: tuple[str, ...] = (
        "min_speed_5_10_20",
        "auc20_net_log",
        "time_above_break_even20",
        "negative_mdd20",
        "negative_post_peak_fade20",
    )
    relevance_grade_thresholds: tuple[float, ...] = GRADE_THRESHOLDS
    trend_fallback_profile: str = TREND_FALLBACK_PROFILE
    log_wealth_floor: float = 1.0e-6
    pareto_block_size: int = 256


@dataclass(frozen=True)
class FeatureViewSpec:
    profile: str = "alpha_market_industry_pit_v1"
    input_sequence_profile: str = "f0_legacy_180x35"
    families: tuple[str, ...] = ("F1", "F2", "F3", "F4", "F5")
    rolling_windows: tuple[int, ...] = (5, 10, 20, 40, 60)
    return_horizons: tuple[int, ...] = (1, 2, 5, 10, 20, 40, 60)
    cross_sectional_transform: str = "same_day_pit_percentile"
    neural_normalization: str = "train_median_iqr_clip10_with_missing_mask"
    categorical_unknown_policy: str = "train_vocabulary_else_unknown_zero"
    feature_screen_year: int = 2022
    feature_screen_tolerance: float = 0.002


@dataclass(frozen=True)
class TargetPaths:
    root: Path
    float_values: Path
    dominance_counts: Path
    relevance_grade: Path
    flags: Path
    manifest: Path


@dataclass(frozen=True)
class FeatureViewPaths:
    root: Path
    continuous: Path
    categorical: Path
    candidate_date_idx: Path
    candidate_symbol_idx: Path
    manifest: Path


class _DeterministicReservoir:
    """Fixed-size deterministic reservoir keyed by (year, relevance grade)."""

    def __init__(self, capacity_per_bucket: int, seed: int) -> None:
        self.capacity_per_bucket = int(capacity_per_bucket)
        self.seed = int(seed)
        self._heaps: dict[tuple[int, int], list[tuple[int, int, dict[str, Any], np.ndarray]]] = {}

    def add(
        self,
        *,
        year: int,
        grade: int,
        candidate_id: int,
        metadata: Mapping[str, Any],
        morphology: np.ndarray,
    ) -> None:
        key = (int(year), int(grade))
        digest = hashlib.blake2b(
            f"{self.seed}:{int(candidate_id)}".encode("utf-8"), digest_size=8
        ).digest()
        priority = int.from_bytes(digest, byteorder="big", signed=False)
        item = (-priority, int(candidate_id), dict(metadata), np.asarray(morphology, dtype=np.float32).copy())
        heap = self._heaps.setdefault(key, [])
        if len(heap) < self.capacity_per_bucket:
            heapq.heappush(heap, item)
            return
        if item[0] > heap[0][0]:
            heapq.heapreplace(heap, item)

    def records(self) -> list[tuple[dict[str, Any], np.ndarray]]:
        out: list[tuple[dict[str, Any], np.ndarray]] = []
        for key in sorted(self._heaps):
            for _priority, _candidate_id, metadata, morphology in sorted(
                self._heaps[key], key=lambda item: item[1]
            ):
                out.append((metadata, morphology))
        return out


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


def _canonical_json_sha256(payload: Any) -> str:
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        default=_json_default,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _file_sha256(path: Path, *, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_write_json(path: Path, payload: Any) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, default=_json_default) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)
    return str(path.resolve())


def _atomic_write_parquet(path: Path, frame: pd.DataFrame) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    frame.to_parquet(temporary, index=False, compression="zstd")
    os.replace(temporary, path)
    return str(path.resolve())


def _next_attempt_dir(root: Path, task: str) -> Path:
    task_root = root / task
    task_root.mkdir(parents=True, exist_ok=True)
    existing: list[int] = []
    for path in task_root.glob("attempt_*"):
        try:
            existing.append(int(path.name.split("_")[-1]))
        except ValueError:
            continue
    attempt = task_root / f"attempt_{max(existing, default=0) + 1:03d}"
    attempt.mkdir(parents=False, exist_ok=False)
    return attempt


def _resolve_path(value: str | Path, *, base: Path = WORKSPACE_ROOT) -> Path:
    path = Path(value)
    if not path.is_absolute():
        path = base / path
    return path.resolve()


def load_study(path: Path = DEFAULT_STUDY_PATH) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload.get("contract"), Mapping):
        raise ValueError("study is missing contract")
    declared = str(payload.get("contract_sha256", "") or "").lower()
    computed = _canonical_json_sha256(payload["contract"])
    if declared != computed:
        raise ValueError(
            f"study contract SHA-256 mismatch: declared={declared} computed={computed}"
        )
    target_spec = dict(payload["contract"].get("target", {}) or {})
    feature_spec = dict(payload["contract"].get("features", {}) or {})
    if str(payload["contract"].get("target_spec_sha256", "")) != _canonical_json_sha256(
        target_spec
    ):
        raise ValueError("target_spec_sha256 mismatch")
    if str(payload["contract"].get("feature_spec_sha256", "")) != _canonical_json_sha256(
        feature_spec
    ):
        raise ValueError("feature_spec_sha256 mismatch")
    return payload


def _validate_source_bindings(study: Mapping[str, Any]) -> tuple[Path, dict[str, Any]]:
    data = dict(study["contract"].get("data", {}) or {})
    pack_path = _resolve_path(str(data.get("pack_manifest", "")))
    declared_pack_sha = str(data.get("pack_manifest_sha256", "") or "")
    actual_pack_sha = _file_sha256(pack_path)
    if actual_pack_sha != declared_pack_sha:
        raise ValueError(
            f"protected pack manifest changed: declared={declared_pack_sha} actual={actual_pack_sha}"
        )
    manifest = json.loads(pack_path.read_text(encoding="utf-8"))
    if int(manifest.get("forward_days", 0) or 0) < 60:
        raise ValueError("path relevance study requires at least 60 future path days")
    if int(manifest.get("lookback_days", 0) or 0) != 180:
        raise ValueError("path relevance study requires the frozen 180-day PIT pack")
    if str(dict(manifest.get("label_semantics", {}) or {}).get("price_anchor", "")) != "today_close":
        raise ValueError("source pack price anchor must be today_close")
    for domain, binding in dict(data.get("qdp_datasets", {}) or {}).items():
        manifest_path = _resolve_path(str(binding.get("manifest_path", "")))
        actual = _file_sha256(manifest_path)
        if actual != str(binding.get("manifest_sha256", "")):
            raise ValueError(f"QDP source manifest changed for {domain}")
        dataset = json.loads(manifest_path.read_text(encoding="utf-8"))
        if str(dataset.get("dataset_id", "")) != str(binding.get("dataset_id", "")):
            raise ValueError(f"QDP dataset ID mismatch for {domain}")
        if str(dataset.get("schema_hash", "")) != str(binding.get("schema_hash", "")):
            raise ValueError(f"QDP schema hash mismatch for {domain}")
    return pack_path, manifest


def _open_bool_panel(manifest: Mapping[str, Any], name: str) -> np.memmap:
    masks = dict(manifest.get("masks", {}) or {})
    if name not in masks:
        raise KeyError(f"source pack is missing mask {name}")
    return sequence_training._open_memmap(masks[name], dtype="bool")


def _open_future_path(manifest: Mapping[str, Any]) -> Any:
    labels = dict(manifest.get("label_arrays", {}) or {})
    meta = labels.get("future_ohlcva_path") or labels.get("future_path")
    if not isinstance(meta, Mapping):
        raise KeyError("source pack is missing future OHLC path")
    return sequence_training._open_label_array(meta, dtype="float32")


def _take_future_ohlc(
    reader: Any, date_idx: np.ndarray, symbol_idx: np.ndarray, *, horizon: int
) -> np.ndarray:
    if hasattr(reader, "take"):
        values = reader.take(date_idx, symbol_idx, field_slice=slice(0, 4))
    else:
        values = reader[date_idx, symbol_idx, :, :4]
    return np.asarray(values[:, : int(horizon), :4], dtype=np.float32)


def _future_panel_view(
    panel: np.ndarray,
    *,
    signal_date_idx: int,
    symbol_idx: np.ndarray,
    horizon: int,
    first_offset: int = 1,
) -> np.ndarray:
    offsets = np.arange(
        int(signal_date_idx) + int(first_offset),
        int(signal_date_idx) + int(first_offset) + int(horizon),
        dtype=np.int64,
    )
    return np.asarray(panel[offsets[:, None], symbol_idx[None, :]].T)


def reanchor_to_actual_next_open(
    future_ohlc: np.ndarray,
    *,
    source_price_anchor: str,
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
    if not np.allclose(
        converted[:, 0, 0],
        0.0,
        rtol=0.0,
        atol=2.0e-6,
        equal_nan=False,
    ):
        raise ValueError("D1 open could not be normalized to the executed entry price")
    return np.asarray(converted, dtype=np.float32)


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
        raise ValueError("signal_date_idx must be non-negative and horizon_days positive")
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
            raise RuntimeError("CUDA Pareto computation was requested but CUDA is unavailable")

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
        fitted = current.mean(axis=1, keepdims=True) + slope.reshape(-1, 1) * x_centered.reshape(1, -1)
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
        if isinstance(array, np.ndarray) and array.dtype.kind == "f" and name != "close_net_log":
            array[~path_available] = np.nan
    return output


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
            [_rank_percentile(trend_inputs[normal, idx]) for idx in range(trend_inputs.shape[1])]
        )
        trend_consistency[normal] = np.nanmean(ranked_components, axis=1).astype(np.float32)
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
            percentile[order] = np.arange(normal_positions.size, dtype=np.float32) / float(
                normal_positions.size - 1
            )
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


def _iter_date_groups(
    parquet_path: Path,
    *,
    columns: Sequence[str],
    batch_size: int = 250_000,
) -> Iterator[pd.DataFrame]:
    parquet_file = pq.ParquetFile(parquet_path)
    carry = pd.DataFrame(columns=list(columns))
    previous_date = ""
    for batch in parquet_file.iter_batches(batch_size=int(batch_size), columns=list(columns)):
        frame = batch.to_pandas()
        if not carry.empty:
            frame = pd.concat([carry, frame], ignore_index=True)
            carry = pd.DataFrame(columns=list(columns))
        if frame.empty:
            continue
        dates = frame["trade_date"].astype(str)
        last_date = str(dates.iloc[-1])
        final_mask = dates.eq(last_date)
        ready = frame.loc[~final_mask]
        carry = frame.loc[final_mask].copy()
        for trade_date, group in ready.groupby("trade_date", sort=False):
            current = str(trade_date)
            if previous_date and current < previous_date:
                raise ValueError("candidate index is not date ordered")
            previous_date = current
            yield group.reset_index(drop=True)
    if not carry.empty:
        current = str(carry["trade_date"].iloc[0])
        if previous_date and current < previous_date:
            raise ValueError("candidate index is not date ordered")
        yield carry.reset_index(drop=True)


def _target_paths(root: Path) -> TargetPaths:
    return TargetPaths(
        root=root,
        float_values=root / "target_values.float32.dat",
        dominance_counts=root / "dominance_counts.int32.dat",
        relevance_grade=root / "relevance_grade.uint8.dat",
        flags=root / "target_flags.uint8.dat",
        manifest=root / "target_manifest.json",
    )


def _feature_view_paths(root: Path) -> FeatureViewPaths:
    return FeatureViewPaths(
        root=root,
        continuous=root / "snapshot_continuous.float32.dat",
        categorical=root / "snapshot_categorical.int64.dat",
        candidate_date_idx=root / "candidate_date_idx.int32.dat",
        candidate_symbol_idx=root / "candidate_symbol_idx.int32.dat",
        manifest=root / "feature_view_manifest.json",
    )


def f1_feature_names() -> tuple[str, ...]:
    names = [
        "open_gap_1d",
        "high_ret_prev_close_1d",
        "low_ret_prev_close_1d",
        "close_ret_1d",
        "intraday_range_1d",
        "body_to_range_1d",
        "upper_shadow_to_range_1d",
        "lower_shadow_to_range_1d",
        "close_location_1d",
        "log_amount_1d",
        "log_volume_1d",
        "log_turnover_pct_1d",
        "relative_turnover_20d",
    ]
    names.extend(f"return_{h}d" for h in FEATURE_RETURN_HORIZONS)
    per_window = (
        "ma_distance",
        "volatility",
        "downside_volatility",
        "atr",
        "trend_slope",
        "trend_t",
        "trend_r2",
        "trend_residual",
        "efficiency_ratio",
        "price_range_position",
        "high_age_fraction",
        "low_age_fraction",
        "up_day_ratio",
        "down_day_ratio",
        "price_amount_correlation",
        "amount_ratio",
        "volume_ratio",
    )
    for window in FEATURE_WINDOWS:
        names.extend(f"{name}_{window}d" for name in per_window)
    return tuple(names)


def f2_feature_names() -> tuple[str, ...]:
    return tuple(f"cs_percentile__{name}" for name in f1_feature_names())


def f3_feature_names() -> tuple[str, ...]:
    return tuple(
        f"market_{group}__{metric}"
        for group in MARKET_GROUPS
        for metric in MARKET_METRICS
    )


def continuous_feature_catalog() -> list[dict[str, Any]]:
    catalog: list[dict[str, Any]] = []
    for family, names in (
        ("F1", f1_feature_names()),
        ("F2", f2_feature_names()),
        ("F3", f3_feature_names()),
        ("F4", F4_CONTINUOUS_FEATURES),
        ("F5", F5_CONTINUOUS_FEATURES),
    ):
        for name in names:
            catalog.append(
                {
                    "column_index": len(catalog),
                    "name": str(name),
                    "family": str(family),
                    "kind": "continuous",
                    "causal_as_of": "signal_date_close_or_earlier",
                }
            )
    return catalog


def categorical_feature_catalog() -> list[dict[str, Any]]:
    return [
        {
            "column_index": idx,
            "name": name,
            "family": family,
            "kind": "categorical_hash",
            "fold_mapping": "train_observed_hashes_else_unknown_zero",
        }
        for idx, (name, family) in enumerate(CATEGORICAL_FEATURES)
    ]


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
    out = np.full(np.broadcast_shapes(top.shape, bottom.shape), np.nan, dtype=np.float64)
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


def _rolling_sums(values: np.ndarray, window: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
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
        raise ValueError("rolling extreme age requires 2D values, positive window, max/min")
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
        raise ValueError("candidate values and date_idx must be aligned one-dimensional arrays")
    if scores.size and bool(np.any(dates[1:] < dates[:-1])):
        raise ValueError("candidate coordinates must be ordered by signal date")
    out = np.full(scores.shape, np.nan, dtype=np.float32)
    if not scores.size:
        return out
    boundaries = np.flatnonzero(np.r_[True, dates[1:] != dates[:-1], True])
    for start, stop in zip(boundaries[:-1], boundaries[1:]):
        out[start:stop] = _rank_percentile(scores[start:stop])
    return out


def _write_candidate_coordinates(
    candidate_path: Path,
    paths: FeatureViewPaths,
    *,
    candidate_count: int,
) -> tuple[np.memmap, np.memmap]:
    date_idx = np.memmap(
        paths.candidate_date_idx,
        dtype="int32",
        mode="w+",
        shape=(int(candidate_count),),
    )
    symbol_idx = np.memmap(
        paths.candidate_symbol_idx,
        dtype="int32",
        mode="w+",
        shape=(int(candidate_count),),
    )
    expected = 0
    parquet = pq.ParquetFile(candidate_path)
    for batch in parquet.iter_batches(
        batch_size=500_000,
        columns=["candidate_id", "date_idx", "symbol_idx"],
    ):
        frame = batch.to_pandas()
        ids = frame["candidate_id"].to_numpy(dtype=np.int64, copy=False)
        if ids.size:
            required = np.arange(expected, expected + ids.size, dtype=np.int64)
            if not np.array_equal(ids, required):
                raise ValueError("candidate_id must be contiguous and ordered from zero")
            date_idx[ids] = frame["date_idx"].to_numpy(dtype=np.int32, copy=False)
            symbol_idx[ids] = frame["symbol_idx"].to_numpy(dtype=np.int32, copy=False)
            expected += int(ids.size)
    if expected != int(candidate_count):
        raise ValueError(
            f"candidate coordinate row mismatch: expected={candidate_count} observed={expected}"
        )
    date_idx.flush()
    symbol_idx.flush()
    if candidate_count and bool(np.any(date_idx[1:] < date_idx[:-1])):
        raise ValueError("candidate index must remain date ordered")
    return date_idx, symbol_idx


class _FeatureWriter:
    def __init__(
        self,
        *,
        continuous: np.memmap,
        categorical: np.memmap,
        continuous_catalog: Sequence[Mapping[str, Any]],
        categorical_catalog: Sequence[Mapping[str, Any]],
        candidate_date_idx: np.ndarray,
        candidate_symbol_idx: np.ndarray,
        candidate_allowed: np.ndarray | None = None,
    ) -> None:
        self.continuous = continuous
        self.categorical = categorical
        self.continuous_index = {
            str(item["name"]): int(item["column_index"]) for item in continuous_catalog
        }
        self.categorical_index = {
            str(item["name"]): int(item["column_index"]) for item in categorical_catalog
        }
        self.date_idx = np.asarray(candidate_date_idx)
        self.symbol_idx = np.asarray(candidate_symbol_idx)
        self.allowed = (
            np.ones(self.date_idx.shape, dtype=bool)
            if candidate_allowed is None
            else np.asarray(candidate_allowed, dtype=bool)
        )
        if self.allowed.shape != self.date_idx.shape:
            raise ValueError("candidate_allowed must align with candidate coordinates")
        self.written_continuous: set[str] = set()
        self.written_categorical: set[str] = set()

    def continuous_candidate(self, name: str, values: np.ndarray) -> None:
        key = str(name)
        if key not in self.continuous_index:
            raise KeyError(f"unknown continuous feature: {key}")
        array = np.asarray(values, dtype=np.float32)
        if array.shape != self.date_idx.shape:
            raise ValueError(f"candidate feature {key} has shape {array.shape}")
        if not bool(self.allowed.all()):
            array = array.copy()
            array[~self.allowed] = np.nan
        self.continuous[:, self.continuous_index[key]] = array
        self.written_continuous.add(key)

    def continuous_dense(self, name: str, panel: np.ndarray) -> np.ndarray:
        values = np.asarray(panel[self.date_idx, self.symbol_idx], dtype=np.float32)
        self.continuous_candidate(name, values)
        return values

    def f1_with_rank(self, name: str, panel: np.ndarray) -> None:
        values = self.continuous_dense(name, panel)
        if not bool(self.allowed.all()):
            values = values.copy()
            values[~self.allowed] = np.nan
        rank_name = f"cs_percentile__{name}"
        self.continuous_candidate(
            rank_name,
            _candidate_cross_section_percentile(values, self.date_idx),
        )

    def categorical_candidate(self, name: str, values: np.ndarray) -> None:
        key = str(name)
        if key not in self.categorical_index:
            raise KeyError(f"unknown categorical feature: {key}")
        array = np.asarray(values, dtype=np.int64)
        if array.shape != self.date_idx.shape:
            raise ValueError(f"categorical feature {key} has shape {array.shape}")
        if not bool(self.allowed.all()):
            array = array.copy()
            array[~self.allowed] = 0
        self.categorical[:, self.categorical_index[key]] = array
        self.written_categorical.add(key)

    def categorical_dense(self, name: str, panel: np.ndarray) -> None:
        self.categorical_candidate(
            name,
            np.asarray(panel[self.date_idx, self.symbol_idx], dtype=np.int64),
        )

    def assert_complete(self) -> None:
        missing_continuous = sorted(set(self.continuous_index).difference(self.written_continuous))
        missing_categorical = sorted(set(self.categorical_index).difference(self.written_categorical))
        if missing_continuous or missing_categorical:
            raise AssertionError(
                "feature view is incomplete: "
                f"continuous={missing_continuous} categorical={missing_categorical}"
            )


def _stable_category_hash(value: Any) -> np.int64:
    text_value = str(value or "").strip()
    if not text_value or text_value.lower() in {"nan", "none", "unknown", "unclassified"}:
        return np.int64(0)
    digest = hashlib.blake2b(text_value.encode("utf-8"), digest_size=8).digest()
    integer = int.from_bytes(digest, byteorder="big", signed=False) & ((1 << 63) - 1)
    return np.int64(integer or 1)


def _qdp_shard_paths(study: Mapping[str, Any], domain: str) -> list[Path]:
    binding = dict(study["contract"]["data"]["qdp_datasets"].get(domain, {}) or {})
    if not binding:
        raise KeyError(f"study does not bind QDP domain {domain}")
    manifest_path = _resolve_path(str(binding["manifest_path"]))
    dataset = json.loads(manifest_path.read_text(encoding="utf-8"))
    qdp_root = WORKSPACE_ROOT / "quant_data_platform/data/qdp_v2"
    paths: list[Path] = []
    for shard in list(dataset.get("shards", []) or []):
        path = Path(str(shard.get("path", "") or ""))
        if not path.is_absolute():
            path = qdp_root / path
        path = path.resolve()
        if not path.exists():
            raise FileNotFoundError(path)
        paths.append(path)
    if not paths:
        raise ValueError(f"QDP domain {domain} has no shards")
    return paths


def _load_qdp_dense_domain(
    study: Mapping[str, Any],
    manifest: Mapping[str, Any],
    *,
    domain: str,
    numeric_fields: Sequence[str] = (),
    boolean_fields: Sequence[str] = (),
    categorical_fields: Sequence[str] = (),
    source_guards: Mapping[str, str] | None = None,
) -> dict[str, np.ndarray]:
    """Load one pinned QDP domain into dense PIT-aligned panels."""

    dates = [str(item) for item in list(manifest.get("date_values", []) or [])]
    symbols = [str(item) for item in list(manifest.get("symbol_values", []) or [])]
    date_map = {value: idx for idx, value in enumerate(dates)}
    symbol_map = {value: idx for idx, value in enumerate(symbols)}
    shape = (len(dates), len(symbols))
    output: dict[str, np.ndarray] = {
        name: np.full(shape, np.nan, dtype=np.float32) for name in numeric_fields
    }
    output.update({name: np.zeros(shape, dtype=bool) for name in boolean_fields})
    output.update({name: np.zeros(shape, dtype=np.int64) for name in categorical_fields})
    guard_map = dict(source_guards or {})
    required = {
        "trade_date",
        "symbol",
        *numeric_fields,
        *boolean_fields,
        *categorical_fields,
        *guard_map.values(),
    }
    category_cache: dict[str, np.int64] = {}
    observed_rows = 0
    for path in _qdp_shard_paths(study, domain):
        parquet = pq.ParquetFile(path)
        available = set(parquet.schema_arrow.names)
        missing = sorted(required.difference(available))
        if missing:
            raise ValueError(f"QDP {domain} shard is missing columns {missing}: {path}")
        for batch in parquet.iter_batches(batch_size=250_000, columns=sorted(required)):
            frame = batch.to_pandas()
            date_idx = frame["trade_date"].astype(str).map(date_map)
            symbol_idx = frame["symbol"].astype(str).map(symbol_map)
            usable = date_idx.notna() & symbol_idx.notna()
            if not bool(usable.any()):
                continue
            rows = frame.loc[usable].reset_index(drop=True)
            d = date_idx.loc[usable].to_numpy(dtype=np.int64)
            s = symbol_idx.loc[usable].to_numpy(dtype=np.int64)
            observed_rows += int(len(rows))
            guard_valid: dict[str, np.ndarray] = {}
            for field, source_field in guard_map.items():
                source = rows[source_field].fillna("").astype(str).to_numpy()
                trade = rows["trade_date"].astype(str).to_numpy()
                future = (source != "") & (source > trade)
                if bool(future.any()):
                    raise ValueError(
                        f"QDP {domain}.{field} has source_date after signal date"
                    )
                guard_valid[field] = (source != "") & (source <= trade)
            for field in numeric_fields:
                values = pd.to_numeric(rows[field], errors="coerce").to_numpy(dtype=np.float64)
                if field in guard_valid:
                    values[~guard_valid[field]] = np.nan
                output[field][d, s] = values.astype(np.float32)
            for field in boolean_fields:
                values = rows[field].astype("boolean").fillna(False).to_numpy(dtype=bool)
                if field in guard_valid:
                    values[~guard_valid[field]] = False
                output[field][d, s] = values
            for field in categorical_fields:
                raw = rows[field].fillna("").astype(str).to_numpy()
                unique = np.unique(raw)
                for value in unique:
                    if value not in category_cache:
                        category_cache[value] = _stable_category_hash(value)
                values = np.fromiter(
                    (category_cache[value] for value in raw),
                    dtype=np.int64,
                    count=len(raw),
                )
                if field in guard_valid:
                    values[~guard_valid[field]] = 0
                output[field][d, s] = values
    if observed_rows == 0:
        raise ValueError(f"QDP domain {domain} did not overlap the frozen pack")
    return output


def _load_index_membership(
    study: Mapping[str, Any],
    manifest: Mapping[str, Any],
) -> dict[str, np.ndarray]:
    dates = [str(item) for item in list(manifest.get("date_values", []) or [])]
    symbols = [str(item) for item in list(manifest.get("symbol_values", []) or [])]
    date_map = {value: idx for idx, value in enumerate(dates)}
    symbol_map = {value: idx for idx, value in enumerate(symbols)}
    shape = (len(dates), len(symbols))
    index_map = {
        "000300.SH": "csi300",
        "000905.SH": "csi500",
        "000016.SH": "sse50",
    }
    output = {name: np.zeros(shape, dtype=bool) for name in index_map.values()}
    for path in _qdp_shard_paths(study, "index_constituents"):
        parquet = pq.ParquetFile(path)
        for batch in parquet.iter_batches(
            batch_size=250_000,
            columns=[
                "trade_date",
                "symbol",
                "index_symbol",
                "source_snapshot_date",
            ],
        ):
            frame = batch.to_pandas()
            source = frame["source_snapshot_date"].fillna("").astype(str)
            trade = frame["trade_date"].astype(str)
            if bool(((source == "") | (source > trade)).any()):
                raise ValueError("index membership violates source_snapshot_date <= trade_date")
            d = trade.map(date_map)
            s = frame["symbol"].astype(str).map(symbol_map)
            usable = d.notna() & s.notna() & frame["index_symbol"].isin(index_map)
            if not bool(usable.any()):
                continue
            current = frame.loc[usable]
            date_values = d.loc[usable].to_numpy(dtype=np.int64)
            symbol_values = s.loc[usable].to_numpy(dtype=np.int64)
            index_values = current["index_symbol"].astype(str).to_numpy()
            for index_symbol, name in index_map.items():
                mask = index_values == index_symbol
                output[name][date_values[mask], symbol_values[mask]] = True
    return output


def _load_event_age_panel(
    study: Mapping[str, Any],
    manifest: Mapping[str, Any],
    *,
    domain: str,
    event_date_field: str,
) -> np.ndarray:
    dates = [str(item) for item in list(manifest.get("date_values", []) or [])]
    symbols = [str(item) for item in list(manifest.get("symbol_values", []) or [])]
    date_map = {value: idx for idx, value in enumerate(dates)}
    symbol_map = {value: idx for idx, value in enumerate(symbols)}
    output = np.full((len(dates), len(symbols)), np.nan, dtype=np.float32)
    for path in _qdp_shard_paths(study, domain):
        parquet = pq.ParquetFile(path)
        required = ["trade_date", "symbol", str(event_date_field)]
        missing = sorted(set(required).difference(parquet.schema_arrow.names))
        if missing:
            raise ValueError(f"QDP {domain} is missing age field columns {missing}")
        for batch in parquet.iter_batches(batch_size=250_000, columns=required):
            frame = batch.to_pandas()
            trade = frame["trade_date"].fillna("").astype(str)
            event = frame[event_date_field].fillna("").astype(str)
            future = (event != "") & (event > trade)
            if bool(future.any()):
                raise ValueError(
                    f"QDP {domain}.{event_date_field} is after the signal date"
                )
            d = trade.map(date_map)
            s = frame["symbol"].astype(str).map(symbol_map)
            usable = d.notna() & s.notna() & (event != "")
            if not bool(usable.any()):
                continue
            trade_dates = trade.loc[usable].to_numpy(dtype="datetime64[D]")
            event_dates = event.loc[usable].to_numpy(dtype="datetime64[D]")
            age = (trade_dates - event_dates).astype("timedelta64[D]").astype(np.float64)
            output[
                d.loc[usable].to_numpy(dtype=np.int64),
                s.loc[usable].to_numpy(dtype=np.int64),
            ] = age.astype(np.float32)
    return output


def _load_industry_panels(
    study: Mapping[str, Any],
    manifest: Mapping[str, Any],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    dates = [str(item) for item in list(manifest.get("date_values", []) or [])]
    symbols = [str(item) for item in list(manifest.get("symbol_values", []) or [])]
    date_map = {value: idx for idx, value in enumerate(dates)}
    symbol_map = {value: idx for idx, value in enumerate(symbols)}
    shape = (len(dates), len(symbols))
    industry = np.zeros(shape, dtype=np.int64)
    broad = np.zeros(shape, dtype=np.int64)
    source_age = np.full(shape, np.nan, dtype=np.float32)
    cache: dict[str, np.int64] = {}
    broad_cache: dict[str, np.int64] = {}
    for path in _qdp_shard_paths(study, "industry_concept"):
        parquet = pq.ParquetFile(path)
        for batch in parquet.iter_batches(
            batch_size=250_000,
            columns=["trade_date", "symbol", "industry", "industry_source_date"],
        ):
            frame = batch.to_pandas()
            trade = frame["trade_date"].fillna("").astype(str)
            source = frame["industry_source_date"].fillna("").astype(str)
            if bool(((source == "") | (source > trade)).any()):
                raise ValueError("industry source_date must be present and <= signal date")
            d = trade.map(date_map)
            s = frame["symbol"].astype(str).map(symbol_map)
            usable = d.notna() & s.notna()
            if not bool(usable.any()):
                continue
            rows = frame.loc[usable]
            raw = rows["industry"].fillna("").astype(str).to_numpy()
            for value in np.unique(raw):
                cache.setdefault(value, _stable_category_hash(value))
                stripped = str(value).strip()
                broad_value = stripped[:1] if stripped and stripped.lower() != "unknown" else ""
                broad_cache.setdefault(value, _stable_category_hash(broad_value))
            industry_values = np.fromiter(
                (cache[value] for value in raw), dtype=np.int64, count=len(raw)
            )
            broad_values = np.fromiter(
                (broad_cache[value] for value in raw), dtype=np.int64, count=len(raw)
            )
            date_values = d.loc[usable].to_numpy(dtype=np.int64)
            symbol_values = s.loc[usable].to_numpy(dtype=np.int64)
            industry[date_values, symbol_values] = industry_values
            broad[date_values, symbol_values] = broad_values
            trade_dates = trade.loc[usable].to_numpy(dtype="datetime64[D]")
            source_dates = source.loc[usable].to_numpy(dtype="datetime64[D]")
            source_age[date_values, symbol_values] = (
                (trade_dates - source_dates)
                .astype("timedelta64[D]")
                .astype(np.float32)
            )
    return industry, broad, source_age


def _trim_after_feature_stage() -> None:
    gc.collect()
    try:
        sequence_training._trim_working_set()
    except Exception:
        pass


def _build_f1_f2_features(
    writer: _FeatureWriter,
    manifest: Mapping[str, Any],
) -> dict[str, Any]:
    channels = dict(manifest.get("feature_channels", {}) or {})
    raw_meta = dict(channels["daily_raw"])
    turnover_meta = dict(channels["turnover"])
    raw = sequence_training._open_memmap(raw_meta, dtype="float32")
    turnover = sequence_training._open_memmap(turnover_meta, dtype="float32")
    raw_columns = {name: idx for idx, name in enumerate(raw_meta["columns"])}
    turnover_columns = {name: idx for idx, name in enumerate(turnover_meta["columns"])}
    open_price = raw[:, :, raw_columns["open"]]
    high = raw[:, :, raw_columns["high"]]
    low = raw[:, :, raw_columns["low"]]
    close = raw[:, :, raw_columns["close"]]
    volume = raw[:, :, raw_columns["volume"]]
    amount = raw[:, :, raw_columns["amount"]]
    previous_close = np.full(close.shape, np.nan, dtype=np.float32)
    previous_close[1:] = close[:-1]
    price_range = np.asarray(high - low, dtype=np.float32)

    direct = {
        "open_gap_1d": _safe_divide(open_price, previous_close) - 1.0,
        "high_ret_prev_close_1d": _safe_divide(high, previous_close) - 1.0,
        "low_ret_prev_close_1d": _safe_divide(low, previous_close) - 1.0,
        "close_ret_1d": _safe_divide(close, previous_close) - 1.0,
        "intraday_range_1d": _safe_divide(price_range, previous_close),
        "body_to_range_1d": _safe_divide(np.abs(close - open_price), price_range),
        "upper_shadow_to_range_1d": _safe_divide(
            high - np.maximum(open_price, close), price_range
        ),
        "lower_shadow_to_range_1d": _safe_divide(
            np.minimum(open_price, close) - low, price_range
        ),
        "close_location_1d": _safe_divide(close - low, price_range),
        "log_amount_1d": np.log1p(np.maximum(amount, 0.0)).astype(np.float32),
        "log_volume_1d": np.log1p(np.maximum(volume, 0.0)).astype(np.float32),
        "log_turnover_pct_1d": np.asarray(
            turnover[:, :, turnover_columns["log_turnover_pct"]], dtype=np.float32
        ),
        "relative_turnover_20d": np.asarray(
            turnover[:, :, turnover_columns["relative_turnover_20"]], dtype=np.float32
        ),
    }
    for name in f1_feature_names()[:13]:
        writer.f1_with_rank(name, direct[name])
        del direct[name]
    del direct

    for horizon in FEATURE_RETURN_HORIZONS:
        current = _lagged_return(close, horizon)
        writer.f1_with_rank(f"return_{horizon}d", current)
        del current

    log_close = np.where(close > 0.0, np.log(close), np.nan).astype(np.float32)
    log_return = np.full(close.shape, np.nan, dtype=np.float32)
    log_return[1:] = log_close[1:] - log_close[:-1]
    true_range = np.maximum.reduce(
        [
            np.asarray(high - low, dtype=np.float32),
            np.asarray(np.abs(high - previous_close), dtype=np.float32),
            np.asarray(np.abs(low - previous_close), dtype=np.float32),
        ]
    )
    true_range = _safe_divide(true_range, previous_close)
    log_amount = np.log1p(np.maximum(amount, 0.0)).astype(np.float32)
    abs_log_return = np.abs(log_return).astype(np.float32)
    for window in FEATURE_WINDOWS:
        ma = _rolling_mean(close, window)
        writer.f1_with_rank(f"ma_distance_{window}d", _safe_divide(close, ma) - 1.0)
        del ma
        _mean_return, volatility = _rolling_mean_std(log_return, window)
        writer.f1_with_rank(f"volatility_{window}d", volatility)
        del _mean_return, volatility
        downside = np.where(
            np.isfinite(log_return), np.minimum(log_return, 0.0), np.nan
        ).astype(np.float32)
        downside_square_mean = _rolling_mean(np.square(downside), window)
        writer.f1_with_rank(
            f"downside_volatility_{window}d",
            np.sqrt(np.maximum(downside_square_mean, 0.0)).astype(np.float32),
        )
        del downside, downside_square_mean
        writer.f1_with_rank(f"atr_{window}d", _rolling_mean(true_range, window))
        slope, t_value, r2, residual = _rolling_linear_stats(log_close, window)
        writer.f1_with_rank(f"trend_slope_{window}d", slope)
        writer.f1_with_rank(f"trend_t_{window}d", t_value)
        writer.f1_with_rank(f"trend_r2_{window}d", r2)
        writer.f1_with_rank(f"trend_residual_{window}d", residual)
        del slope, t_value, r2, residual
        path_length = _rolling_mean(abs_log_return, window) * float(window)
        displacement = np.full(log_close.shape, np.nan, dtype=np.float32)
        displacement[window:] = np.abs(log_close[window:] - log_close[:-window])
        writer.f1_with_rank(
            f"efficiency_ratio_{window}d",
            _safe_divide(displacement, path_length),
        )
        del path_length, displacement
        rolling_low, rolling_high = _rolling_min_max(close, window)
        writer.f1_with_rank(
            f"price_range_position_{window}d",
            _safe_divide(close - rolling_low, rolling_high - rolling_low),
        )
        del rolling_low, rolling_high
        high_age = _rolling_extreme_age(high, window, mode="max") / float(
            max(window - 1, 1)
        )
        low_age = _rolling_extreme_age(low, window, mode="min") / float(
            max(window - 1, 1)
        )
        writer.f1_with_rank(f"high_age_fraction_{window}d", high_age)
        writer.f1_with_rank(f"low_age_fraction_{window}d", low_age)
        del high_age, low_age
        up = np.where(np.isfinite(log_return), log_return > 0.0, np.nan).astype(np.float32)
        down = np.where(np.isfinite(log_return), log_return < 0.0, np.nan).astype(np.float32)
        writer.f1_with_rank(f"up_day_ratio_{window}d", _rolling_mean(up, window))
        writer.f1_with_rank(f"down_day_ratio_{window}d", _rolling_mean(down, window))
        del up, down
        writer.f1_with_rank(
            f"price_amount_correlation_{window}d",
            _rolling_corr(log_return, log_amount, window),
        )
        amount_mean = _rolling_mean(amount, window)
        volume_mean = _rolling_mean(volume, window)
        writer.f1_with_rank(f"amount_ratio_{window}d", _safe_divide(amount, amount_mean))
        writer.f1_with_rank(f"volume_ratio_{window}d", _safe_divide(volume, volume_mean))
        del amount_mean, volume_mean
        writer.continuous.flush()
        _trim_after_feature_stage()
    return {
        "f1_feature_count": len(f1_feature_names()),
        "f2_feature_count": len(f2_feature_names()),
        "causal_max_lookback_days": max(FEATURE_WINDOWS),
    }


def _build_f3_features(
    writer: _FeatureWriter,
    manifest: Mapping[str, Any],
    *,
    index_membership: Mapping[str, np.ndarray],
) -> dict[str, Any]:
    channels = dict(manifest.get("feature_channels", {}) or {})
    raw_meta = dict(channels["daily_raw"])
    raw = sequence_training._open_memmap(raw_meta, dtype="float32")
    columns = {name: idx for idx, name in enumerate(raw_meta["columns"])}
    close = raw[:, :, columns["close"]]
    amount = raw[:, :, columns["amount"]]
    ret1 = _lagged_return(close, 1)
    ret5 = _lagged_return(close, 5)
    ret20 = _lagged_return(close, 20)
    log_close = np.where(close > 0.0, np.log(close), np.nan).astype(np.float32)
    log_return = np.full(close.shape, np.nan, dtype=np.float32)
    log_return[1:] = log_close[1:] - log_close[:-1]
    _mean, vol20 = _rolling_mean_std(log_return, 20)
    del _mean
    ma20 = _rolling_mean(close, 20)
    _low60, high60 = _rolling_min_max(close, 60)
    del _low60
    drawdown60 = 1.0 - _safe_divide(close, high60)
    above_ma20 = close > ma20
    masks = dict(manifest.get("masks", {}) or {})
    universe = sequence_training._open_memmap(
        masks.get("pit_universe_has_bar", masks["has_bar"]), dtype="bool"
    )
    suspended = sequence_training._open_memmap(masks["is_suspended"], dtype="bool")
    st = sequence_training._open_memmap(masks["is_st"], dtype="bool")
    date_features = np.full(
        (int(manifest["date_count"]), len(f3_feature_names())),
        np.nan,
        dtype=np.float32,
    )
    feature_index = {name: idx for idx, name in enumerate(f3_feature_names())}
    group_masks: dict[str, np.ndarray] = {"all": np.asarray(universe, dtype=bool)}
    for name in ("csi300", "csi500", "sse50"):
        group_masks[name] = np.asarray(universe, dtype=bool) & np.asarray(
            index_membership[name], dtype=bool
        )
    for date in range(int(manifest["date_count"])):
        for group in MARKET_GROUPS:
            current_mask = group_masks[group][date]
            count = int(current_mask.sum())
            if count == 0:
                continue

            def finite_mean(values: np.ndarray) -> float:
                selected = np.asarray(values[date, current_mask], dtype=np.float64)
                selected = selected[np.isfinite(selected)]
                return float(selected.mean()) if selected.size else np.nan

            current_ret1 = np.asarray(ret1[date, current_mask], dtype=np.float64)
            current_ret5 = np.asarray(ret5[date, current_mask], dtype=np.float64)
            current_amount = np.asarray(amount[date, current_mask], dtype=np.float64)
            metric_values = {
                "ret1_mean": finite_mean(ret1),
                "ret5_mean": finite_mean(ret5),
                "ret20_mean": finite_mean(ret20),
                "vol20_mean": finite_mean(vol20),
                "drawdown60_mean": finite_mean(drawdown60),
                "breadth_ret1_positive": float(np.nanmean(current_ret1 > 0.0)),
                "breadth_ret5_positive": float(np.nanmean(current_ret5 > 0.0)),
                "breadth_above_ma20": float(
                    np.mean(np.asarray(above_ma20[date, current_mask], dtype=bool))
                ),
                "ret1_dispersion": float(np.nanstd(current_ret1)),
                "log_total_amount": float(
                    np.log1p(np.nansum(np.maximum(current_amount, 0.0)))
                ),
                "suspended_rate": float(
                    np.mean(np.asarray(suspended[date, current_mask], dtype=bool))
                ),
                "st_rate": float(np.mean(np.asarray(st[date, current_mask], dtype=bool))),
                "up_limit_rate": float(np.nanmean(current_ret1 >= 0.095)),
                "down_limit_rate": float(np.nanmean(current_ret1 <= -0.095)),
            }
            for metric, value in metric_values.items():
                date_features[date, feature_index[f"market_{group}__{metric}"]] = np.float32(
                    value
                )
    for name in f3_feature_names():
        writer.continuous_candidate(
            name,
            date_features[writer.date_idx, feature_index[name]],
        )
    writer.continuous.flush()
    del date_features, ret1, ret5, ret20, vol20, ma20, high60, drawdown60
    _trim_after_feature_stage()
    return {
        "groups": list(MARKET_GROUPS),
        "metrics": list(MARKET_METRICS),
        "cross_section_mask": "pit_universe_has_bar_and_optional_same_day_index_membership",
    }


def _industry_candidate_features(
    *,
    writer: _FeatureWriter,
    manifest: Mapping[str, Any],
    industry: np.ndarray,
    industry_source_age: np.ndarray,
) -> dict[str, Any]:
    channels = dict(manifest.get("feature_channels", {}) or {})
    raw_meta = dict(channels["daily_raw"])
    raw = sequence_training._open_memmap(raw_meta, dtype="float32")
    columns = {name: idx for idx, name in enumerate(raw_meta["columns"])}
    close = raw[:, :, columns["close"]]
    ret1 = _lagged_return(close, 1)
    ret5 = _lagged_return(close, 5)
    ret20 = _lagged_return(close, 20)
    masks = dict(manifest.get("masks", {}) or {})
    universe = sequence_training._open_memmap(
        masks.get("pit_universe_has_bar", masks["has_bar"]), dtype="bool"
    )
    values = np.full(
        (writer.date_idx.size, len(F4_CONTINUOUS_FEATURES)),
        np.nan,
        dtype=np.float32,
    )
    feature_index = {name: idx for idx, name in enumerate(F4_CONTINUOUS_FEATURES)}
    boundaries = np.flatnonzero(
        np.r_[True, writer.date_idx[1:] != writer.date_idx[:-1], True]
    )
    for start, stop in zip(boundaries[:-1], boundaries[1:]):
        date = int(writer.date_idx[start])
        base_mask = np.asarray(universe[date], dtype=bool) & (industry[date] != 0)
        if not bool(base_mask.any()):
            continue
        codes = industry[date, base_mask]
        unique, inverse = np.unique(codes, return_inverse=True)
        group_count = np.bincount(inverse).astype(np.float64)

        def aggregate(series: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
            current = np.asarray(series[date, base_mask], dtype=np.float64)
            finite = np.isfinite(current)
            count = np.bincount(inverse[finite], minlength=len(unique)).astype(np.float64)
            total = np.bincount(
                inverse[finite], weights=current[finite], minlength=len(unique)
            ).astype(np.float64)
            mean = np.divide(total, count, out=np.full(len(unique), np.nan), where=count > 0)
            centered_square = np.bincount(
                inverse[finite],
                weights=np.square(current[finite]),
                minlength=len(unique),
            ).astype(np.float64)
            variance = np.divide(
                centered_square,
                count,
                out=np.full(len(unique), np.nan),
                where=count > 0,
            ) - np.square(mean)
            return mean, np.sqrt(np.maximum(variance, 0.0))

        mean1, dispersion1 = aggregate(ret1)
        mean5, _dispersion5 = aggregate(ret5)
        mean20, _dispersion20 = aggregate(ret20)
        current_ret1 = np.asarray(ret1[date, base_mask], dtype=np.float64)
        finite_ret1 = np.isfinite(current_ret1)
        positive = np.bincount(
            inverse[finite_ret1],
            weights=(current_ret1[finite_ret1] > 0.0).astype(np.float64),
            minlength=len(unique),
        )
        valid_count = np.bincount(inverse[finite_ret1], minlength=len(unique)).astype(np.float64)
        breadth = np.divide(
            positive,
            valid_count,
            out=np.full(len(unique), np.nan),
            where=valid_count > 0,
        )
        candidate_symbols = writer.symbol_idx[start:stop]
        candidate_codes = industry[date, candidate_symbols]
        positions = np.searchsorted(unique, candidate_codes)
        known = (candidate_codes != 0) & (positions < len(unique))
        known &= unique[np.minimum(positions, len(unique) - 1)] == candidate_codes
        mapped = np.minimum(positions, len(unique) - 1)
        values[start:stop, feature_index["industry_ret1_mean"]][known] = mean1[mapped[known]]
        values[start:stop, feature_index["industry_ret5_mean"]][known] = mean5[mapped[known]]
        values[start:stop, feature_index["industry_breadth_ret1_positive"]][known] = breadth[
            mapped[known]
        ]
        values[start:stop, feature_index["industry_ret1_dispersion"]][known] = dispersion1[
            mapped[known]
        ]
        candidate_ret5 = ret5[date, candidate_symbols]
        candidate_ret20 = ret20[date, candidate_symbols]
        values[start:stop, feature_index["industry_relative_ret5"]][known] = (
            candidate_ret5[known] - mean5[mapped[known]]
        )
        values[start:stop, feature_index["industry_relative_ret20"]][known] = (
            candidate_ret20[known] - mean20[mapped[known]]
        )
        values[start:stop, feature_index["industry_member_count_log"]][known] = np.log1p(
            group_count[mapped[known]]
        )
        values[start:stop, feature_index["industry_source_age_days"]] = industry_source_age[
            date, candidate_symbols
        ]
    for name in F4_CONTINUOUS_FEATURES:
        writer.continuous_candidate(name, values[:, feature_index[name]])
    writer.continuous.flush()
    del values, ret1, ret5, ret20
    _trim_after_feature_stage()
    return {
        "aggregation_universe": "same_day_pit_universe_has_bar",
        "unknown_industry_hash": 0,
    }


def _signed_log1p(values: np.ndarray) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    return (np.sign(array) * np.log1p(np.abs(array))).astype(np.float32)


def _build_f5_features(
    writer: _FeatureWriter,
    study: Mapping[str, Any],
    manifest: Mapping[str, Any],
    *,
    index_membership: Mapping[str, np.ndarray],
    industry: np.ndarray,
) -> dict[str, Any]:
    valuation = _load_qdp_dense_domain(
        study,
        manifest,
        domain="valuation",
        numeric_fields=("total_mv", "circ_mv", "pe", "pb", "turnover_rate"),
    )
    total_mv = valuation["total_mv"]
    circ_mv = valuation["circ_mv"]
    writer.continuous_dense(
        "log_total_market_value", np.log1p(np.maximum(total_mv, 0.0)).astype(np.float32)
    )
    writer.continuous_dense(
        "log_circulating_market_value",
        np.log1p(np.maximum(circ_mv, 0.0)).astype(np.float32),
    )
    writer.continuous_dense(
        "circulating_market_value_ratio", _safe_divide(circ_mv, total_mv)
    )
    writer.continuous_dense("signed_log_pe", _signed_log1p(valuation["pe"]))
    writer.continuous_dense("signed_log_pb", _signed_log1p(valuation["pb"]))
    writer.continuous_dense(
        "log_turnover_rate",
        np.log1p(np.maximum(valuation["turnover_rate"], 0.0)).astype(np.float32),
    )
    valuation_missing = ~(
        np.isfinite(total_mv)
        & np.isfinite(circ_mv)
        & np.isfinite(valuation["pe"])
        & np.isfinite(valuation["pb"])
    )
    writer.continuous_dense("valuation_missing", valuation_missing.astype(np.float32))
    del valuation, total_mv, circ_mv
    _trim_after_feature_stage()

    share = _load_qdp_dense_domain(
        study,
        manifest,
        domain="share_capital",
        numeric_fields=("total_share", "float_share"),
        source_guards={
            "total_share": "total_share_source_date",
            "float_share": "float_share_source_date",
        },
    )
    total_share = share["total_share"]
    float_share = share["float_share"]
    writer.continuous_dense(
        "log_total_share", np.log1p(np.maximum(total_share, 0.0)).astype(np.float32)
    )
    writer.continuous_dense(
        "log_float_share", np.log1p(np.maximum(float_share, 0.0)).astype(np.float32)
    )
    writer.continuous_dense("float_share_ratio", _safe_divide(float_share, total_share))
    share_source_age = _load_event_age_panel(
        study,
        manifest,
        domain="share_capital",
        event_date_field="float_share_source_date",
    )
    writer.continuous_dense("share_source_age_days", share_source_age)
    writer.continuous_dense(
        "share_capital_missing",
        (~(np.isfinite(total_share) & np.isfinite(float_share))).astype(np.float32),
    )
    del share, total_share, float_share, share_source_age
    _trim_after_feature_stage()

    universe = _load_qdp_dense_domain(
        study,
        manifest,
        domain="universe_snapshot",
        categorical_fields=("board", "exchange", "list_status"),
    )
    listing_age = _load_event_age_panel(
        study,
        manifest,
        domain="universe_snapshot",
        event_date_field="list_date",
    )
    writer.continuous_dense("listing_age_days", listing_age)
    writer.categorical_dense("board_hash", universe["board"])
    writer.categorical_dense("exchange_hash", universe["exchange"])
    writer.categorical_dense("list_status_hash", universe["list_status"])
    del universe, listing_age

    for name in ("csi300", "csi500", "sse50"):
        writer.continuous_dense(
            f"is_{name}_member",
            np.asarray(index_membership[name], dtype=np.float32),
        )
    masks = dict(manifest.get("masks", {}) or {})
    st = sequence_training._open_memmap(masks["is_st"], dtype="bool")
    suspended = sequence_training._open_memmap(masks["is_suspended"], dtype="bool")
    delisted = sequence_training._open_memmap(masks["is_delisted"], dtype="bool")
    status_valid = sequence_training._open_memmap(masks["status_valid"], dtype="bool")
    writer.continuous_dense("is_st_today", np.asarray(st, dtype=np.float32))
    writer.continuous_dense(
        "is_suspended_today", np.asarray(suspended, dtype=np.float32)
    )
    writer.continuous_dense("is_delisted_today", np.asarray(delisted, dtype=np.float32))
    st_history = np.where(status_valid, st.astype(np.float32), np.nan).astype(np.float32)
    suspended_history = np.where(
        status_valid, suspended.astype(np.float32), np.nan
    ).astype(np.float32)
    writer.continuous_dense("st_rate_20d", _rolling_mean(st_history, 20))
    writer.continuous_dense(
        "suspended_rate_20d", _rolling_mean(suspended_history, 20)
    )
    writer.continuous_dense("industry_missing", (industry == 0).astype(np.float32))
    writer.continuous.flush()
    writer.categorical.flush()
    _trim_after_feature_stage()
    return {
        "valuation_source": "pinned_qdp_valuation_same_day",
        "share_source_guard": "source_date_lte_signal_date",
        "category_storage": "stable_hash_then_fold_train_vocabulary",
        "neural_missing_mask": "generated_from_nonfinite_continuous_values_at_batch_load",
    }


def _date_indices_for_year(date_values: Sequence[str], year: int) -> np.ndarray:
    return np.asarray(
        [idx for idx, value in enumerate(date_values) if int(str(value)[:4]) == int(year)],
        dtype=np.int32,
    )


def _candidate_range_count(date_idx: np.ndarray, start: int, end: int) -> int:
    if int(end) < int(start):
        return 0
    left = int(np.searchsorted(date_idx, int(start), side="left"))
    right = int(np.searchsorted(date_idx, int(end), side="right"))
    return max(right - left, 0)


def _build_fold_views(
    *,
    output_dir: Path,
    study: Mapping[str, Any],
    manifest: Mapping[str, Any],
    target_manifest: Mapping[str, Any],
    candidate_date_idx: np.ndarray,
) -> dict[str, Any]:
    folds_dir = output_dir / "folds"
    folds_dir.mkdir(parents=True, exist_ok=False)
    dates = [str(item) for item in list(manifest.get("date_values", []) or [])]
    purge = int(study["contract"]["folds"]["purge_trading_days"])
    source_root = _resolve_path(str(manifest["sample_index_path"])).parent.parent / "folds"
    fold_records: dict[str, Any] = {}
    for test_year in FORMAL_FOLD_YEARS:
        dev_year = int(test_year) - 1
        dev_dates = _date_indices_for_year(dates, dev_year)
        test_dates = _date_indices_for_year(dates, test_year)
        if not dev_dates.size or not test_dates.size:
            raise ValueError(f"pack lacks dates for fold {test_year}")
        dev_start = int(dev_dates.min())
        test_start = int(test_dates.min())
        train_end = dev_start - purge - 1
        dev_end = test_start - purge - 1
        train_start = int(candidate_date_idx.min())
        if train_end < train_start or dev_end < dev_start:
            raise ValueError(f"fold {test_year} purge leaves an empty train/development split")
        test_end = int(test_dates.max())
        if int(test_year) == 2025:
            allowed = [
                idx
                for idx in test_dates.tolist()
                if label_endpoint_is_allowed(
                    signal_date_idx=int(idx),
                    date_values=dates,
                    horizon_days=20,
                    cutoff=str(target_manifest["label_endpoint_cutoff"]),
                )
            ]
            if not allowed:
                raise ValueError("2025 fold has no D20-complete test dates")
            test_end = int(max(allowed))
        original_index = source_root / "indexes" / f"development_{dev_year}_purge60.parquet"
        original_candidates = source_root / "indexes" / f"candidates_{test_year}.parquet"
        if not original_index.exists() or not original_candidates.exists():
            raise FileNotFoundError(
                f"immutable source fold indexes are missing for fold {test_year}"
            )
        f0_normalization_view = source_root / "views" / f"l35v2_pit_{dev_year}.json"
        if not f0_normalization_view.exists():
            raise FileNotFoundError(f0_normalization_view)
        record = {
            "artifact_type": "seq100_path_relevance_fold_view",
            "study_contract_sha256": study["contract_sha256"],
            "target_manifest_sha256": _file_sha256(Path(str(target_manifest["manifest_path"]))),
            "fold_year": int(test_year),
            "purge_trading_days": purge,
            "ranges": {
                "train": {
                    "date_idx_start": train_start,
                    "date_idx_end": train_end,
                    "trade_date_start": dates[train_start],
                    "trade_date_end": dates[train_end],
                    "candidate_count": _candidate_range_count(
                        candidate_date_idx, train_start, train_end
                    ),
                },
                "development": {
                    "date_idx_start": dev_start,
                    "date_idx_end": dev_end,
                    "trade_date_start": dates[dev_start],
                    "trade_date_end": dates[dev_end],
                    "candidate_count": _candidate_range_count(
                        candidate_date_idx, dev_start, dev_end
                    ),
                },
                "test": {
                    "date_idx_start": test_start,
                    "date_idx_end": test_end,
                    "trade_date_start": dates[test_start],
                    "trade_date_end": dates[test_end],
                    "candidate_count": _candidate_range_count(
                        candidate_date_idx, test_start, test_end
                    ),
                },
            },
            "normalization_cutoff": dates[dev_start - 1],
            "category_vocabulary": "train_range_only",
            "f0_normalization_view": str(f0_normalization_view.resolve()),
            "immutable_source_indexes": {
                "development": {
                    "path": str(original_index.resolve()),
                    "sha256": _file_sha256(original_index),
                },
                "candidates": {
                    "path": str(original_candidates.resolve()),
                    "sha256": _file_sha256(original_candidates),
                },
            },
            "oos_2026_forbidden": True,
        }
        record["view_sha256"] = _canonical_json_sha256(record)
        path = folds_dir / f"fold_{test_year}.json"
        _atomic_write_json(path, record)
        fold_records[str(test_year)] = {
            "path": str(path.resolve()),
            "sha256": _file_sha256(path),
            "ranges": record["ranges"],
        }
    return fold_records


def _assign_target_fields(
    target: np.memmap,
    candidate_ids: np.ndarray,
    descriptors: Mapping[str, np.ndarray],
    relevance: Mapping[str, np.ndarray],
) -> None:
    close_net_log = np.asarray(descriptors["close_net_log"], dtype=np.float32)
    for day in range(1, 21):
        target[candidate_ids, TARGET_FIELD_INDEX[f"close_net_log_d{day:02d}"]] = close_net_log[
            :, day - 1
        ]
    for name in (
        "r5_net",
        "r10_net",
        "r20_net",
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
        "trend_blend_5_10_20_dd100",
        "max_speed_negative_control",
    ):
        target[candidate_ids, TARGET_FIELD_INDEX[name]] = np.asarray(
            descriptors[name], dtype=np.float32
        )
    for name in (
        "trend_consistency_v1",
        "dominance_margin",
        "pareto_relevance",
        "conditional_relevance",
        "action_relevance",
    ):
        target[candidate_ids, TARGET_FIELD_INDEX[name]] = np.asarray(
            relevance[name], dtype=np.float32
        )


def _morphology_matrix(entry_path: np.ndarray) -> np.ndarray:
    path = np.asarray(entry_path, dtype=np.float32)
    close = path[:, :, 3]
    high_close = path[:, :, 1] - close
    close_low = close - path[:, :, 2]
    return np.concatenate([close, high_close, close_low], axis=1).astype(np.float32)


def _append_daily_diagnostics(
    rows: list[dict[str, Any]],
    *,
    group: pd.DataFrame,
    descriptors: Mapping[str, np.ndarray],
    relevance: Mapping[str, np.ndarray],
    selection_profile: str = PATH_TARGET_PROFILE,
) -> None:
    action = np.asarray(relevance["action_relevance"], dtype=np.float64)
    conditional = np.asarray(relevance["conditional_relevance"], dtype=np.float64)
    grade = np.asarray(relevance["relevance_grade"], dtype=np.uint8)
    r20 = np.asarray(descriptors["r20_net"], dtype=np.float64)
    auc = np.asarray(descriptors["auc20_net_log"], dtype=np.float64)
    mdd = np.asarray(descriptors["mdd20"], dtype=np.float64)
    fade = np.asarray(descriptors["post_peak_fade20"], dtype=np.float64)
    dominating = np.asarray(relevance["dominating_count"], dtype=np.int32)
    available = np.isfinite(conditional)
    normal = available & (dominating >= 0)
    universe_mask = np.isfinite(conditional) & np.isfinite(r20) & np.isfinite(auc)
    universe_r20 = r20[universe_mask]
    universe_auc = auc[universe_mask]
    record: dict[str, Any] = {
        "selection_profile": str(selection_profile),
        "trade_date": str(group["trade_date"].iloc[0]),
        "year": int(group["year"].iloc[0]),
        "date_idx": int(group["date_idx"].iloc[0]),
        "candidate_count": int(len(group)),
        "target_count": int(available.sum()),
        "frontier_share": (
            float(np.mean(dominating[normal] == 0))
            if str(selection_profile) == PATH_TARGET_PROFILE and bool(normal.any())
            else np.nan
        ),
        "universe_r20_median": (
            float(np.median(universe_r20)) if universe_r20.size else np.nan
        ),
        "universe_auc_median": (
            float(np.median(universe_auc)) if universe_auc.size else np.nan
        ),
    }
    finite_action = np.isfinite(action)
    if bool(finite_action.any()):
        positions = np.flatnonzero(finite_action)
        order = positions[np.argsort(-action[positions], kind="mergesort")]
        top_count = max(1, int(math.ceil(len(group) * 0.01)))
        top = order[:top_count]
        record.update(
            {
                "top1pct_count": int(top.size),
                "top1pct_r20_median": float(np.nanmedian(r20[top])),
                "top1pct_auc_median": float(np.nanmedian(auc[top])),
                "top1pct_mdd_median": float(np.nanmedian(mdd[top])),
                "top1pct_fade_median": float(np.nanmedian(fade[top])),
                "top1pct_negative_r20_rate": float(np.nanmean(r20[top] < 0.0)),
                "top1pct_fade_gt10_rate": float(np.nanmean(fade[top] > 0.10)),
            }
        )
    for current_grade in range(5):
        mask = grade == current_grade
        record[f"grade{current_grade}_count"] = int(mask.sum())
        record[f"grade{current_grade}_r20_median"] = (
            float(np.nanmedian(r20[mask])) if bool(mask.any()) else np.nan
        )
        record[f"grade{current_grade}_auc_median"] = (
            float(np.nanmedian(auc[mask])) if bool(mask.any()) else np.nan
        )
        record[f"grade{current_grade}_mdd_median"] = (
            float(np.nanmedian(mdd[mask])) if bool(mask.any()) else np.nan
        )
    rows.append(record)


def _annual_target_diagnostics(
    daily: pd.DataFrame,
    *,
    require_frontier_gate: bool = True,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    if daily.empty:
        return pd.DataFrame(), {
            "passed": False,
            "reason": "no_target_dates",
            "frontier_gate_required": bool(require_frontier_gate),
        }
    annual_rows: list[dict[str, Any]] = []
    for year, group in daily.groupby("year", sort=True):
        row: dict[str, Any] = {
            "year": int(year),
            "date_count": int(len(group)),
            "frontier_share_median": float(group["frontier_share"].median()),
            "top1pct_r20_median": float(group["top1pct_r20_median"].median()),
            "top1pct_auc_median": float(group["top1pct_auc_median"].median()),
            "top1pct_mdd_median": float(group["top1pct_mdd_median"].median()),
            "top1pct_fade_median": float(group["top1pct_fade_median"].median()),
            "top1pct_negative_r20_rate": float(
                np.average(
                    group["top1pct_negative_r20_rate"].fillna(0.0),
                    weights=group["top1pct_count"].clip(lower=1),
                )
            ),
            "top1pct_fade_gt10_rate": float(
                np.average(
                    group["top1pct_fade_gt10_rate"].fillna(0.0),
                    weights=group["top1pct_count"].clip(lower=1),
                )
            ),
            "universe_r20_median": float(group["universe_r20_median"].median()),
            "universe_auc_median": float(group["universe_auc_median"].median()),
        }
        grade_r20 = np.asarray(
            [group[f"grade{grade}_r20_median"].median() for grade in range(5)],
            dtype=np.float64,
        )
        grade_mdd = np.asarray(
            [group[f"grade{grade}_mdd_median"].median() for grade in range(5)],
            dtype=np.float64,
        )
        valid_r20 = np.isfinite(grade_r20)
        valid_mdd = np.isfinite(grade_mdd)
        row["grade_r20_spearman"] = (
            float(stats.spearmanr(np.arange(5)[valid_r20], grade_r20[valid_r20]).statistic)
            if int(valid_r20.sum()) >= 3
            else np.nan
        )
        row["grade_mdd_spearman"] = (
            float(stats.spearmanr(np.arange(5)[valid_mdd], grade_mdd[valid_mdd]).statistic)
            if int(valid_mdd.sum()) >= 3
            else np.nan
        )
        row["high_relevance_better_than_universe"] = bool(
            row["top1pct_r20_median"] > row["universe_r20_median"]
            and row["top1pct_auc_median"] > row["universe_auc_median"]
        )
        row["grade_direction_consistent"] = bool(
            float(row["grade_r20_spearman"]) > 0.0
            and float(row["grade_mdd_spearman"]) < 0.0
        )
        annual_rows.append(row)
    annual = pd.DataFrame(annual_rows)
    weighted_top1_count = daily["top1pct_count"].clip(lower=1)
    frontier_median = float(daily["frontier_share"].median())
    gates = {
        "frontier_gate_required": bool(require_frontier_gate),
        "frontier_share_median": frontier_median,
        "frontier_share_in_range": bool(
            (not bool(require_frontier_gate))
            or (math.isfinite(frontier_median) and 0.0025 <= frontier_median <= 0.10)
        ),
        "top1pct_negative_r20_rate": float(
            np.average(
                daily["top1pct_negative_r20_rate"].fillna(0.0),
                weights=weighted_top1_count,
            )
        ),
        "top1pct_negative_r20_pass": bool(
            np.average(
                daily["top1pct_negative_r20_rate"].fillna(0.0),
                weights=weighted_top1_count,
            )
            <= 0.05
        ),
        "top1pct_mdd_median": float(daily["top1pct_mdd_median"].median()),
        "top1pct_mdd_pass": bool(float(daily["top1pct_mdd_median"].median()) <= 0.08),
        "top1pct_fade_gt10_rate": float(
            np.average(
                daily["top1pct_fade_gt10_rate"].fillna(0.0),
                weights=weighted_top1_count,
            )
        ),
        "top1pct_fade_pass": bool(
            np.average(
                daily["top1pct_fade_gt10_rate"].fillna(0.0),
                weights=weighted_top1_count,
            )
            <= 0.10
        ),
        "high_relevance_better_year_count": int(
            annual["high_relevance_better_than_universe"].sum()
        ),
        "high_relevance_better_year_pass": bool(
            int(annual["high_relevance_better_than_universe"].sum()) >= 13
        ),
        "grade_direction_consistent_year_count": int(
            annual["grade_direction_consistent"].sum()
        ),
        "grade_direction_consistent_year_pass": bool(
            int(annual["grade_direction_consistent"].sum()) >= 13
        ),
    }
    gates["passed"] = bool(
        gates["frontier_share_in_range"]
        and gates["top1pct_negative_r20_pass"]
        and gates["top1pct_mdd_pass"]
        and gates["top1pct_fade_pass"]
        and gates["high_relevance_better_year_pass"]
        and gates["grade_direction_consistent_year_pass"]
    )
    return annual, gates


def _fallback_relevance_view(
    relevance: Mapping[str, np.ndarray],
    *,
    entry_filled: np.ndarray,
) -> dict[str, np.ndarray]:
    """Expose the pre-registered trend score through the unified target schema."""

    filled = np.asarray(entry_filled, dtype=bool)
    conditional = np.asarray(relevance["trend_consistency_v1"], dtype=np.float32).copy()
    action = np.zeros(filled.shape, dtype=np.float32)
    action[filled] = conditional[filled]
    return {
        **dict(relevance),
        "conditional_relevance": conditional,
        "action_relevance": action,
        "relevance_grade": relevance_grades(conditional),
    }


def _rewrite_frozen_relevance_fields(
    paths: TargetPaths,
    *,
    candidate_count: int,
    frozen_profile: str,
    chunk_size: int = 1_000_000,
) -> None:
    if str(frozen_profile) == PATH_TARGET_PROFILE:
        return
    if str(frozen_profile) != TREND_FALLBACK_PROFILE:
        raise ValueError(f"unsupported frozen target profile: {frozen_profile}")
    target = np.memmap(
        paths.float_values,
        dtype="float32",
        mode="r+",
        shape=(int(candidate_count), len(TARGET_FLOAT_FIELDS)),
    )
    grades = np.memmap(
        paths.relevance_grade,
        dtype="uint8",
        mode="r+",
        shape=(int(candidate_count),),
    )
    flags = np.memmap(
        paths.flags,
        dtype="uint8",
        mode="r",
        shape=(int(candidate_count),),
    )
    trend_idx = TARGET_FIELD_INDEX["trend_consistency_v1"]
    conditional_idx = TARGET_FIELD_INDEX["conditional_relevance"]
    action_idx = TARGET_FIELD_INDEX["action_relevance"]
    for start in range(0, int(candidate_count), int(chunk_size)):
        stop = min(start + int(chunk_size), int(candidate_count))
        trend = np.asarray(target[start:stop, trend_idx], dtype=np.float32)
        filled = (np.asarray(flags[start:stop], dtype=np.uint8) & TARGET_FLAG_ENTRY_FILLED) != 0
        target[start:stop, conditional_idx] = trend
        action = np.zeros(stop - start, dtype=np.float32)
        action[filled] = trend[filled]
        target[start:stop, action_idx] = action
        grades[start:stop] = relevance_grades(trend)
    target.flush()
    grades.flush()
    del target, grades, flags


def _fit_path_atlas(
    records: list[tuple[dict[str, Any], np.ndarray]], output_dir: Path, *, seed: int
) -> dict[str, Any]:
    if not records:
        return {"sample_count": 0, "status": "empty"}
    from sklearn.cluster import MiniBatchKMeans
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import RobustScaler

    metadata = pd.DataFrame([record[0] for record in records])
    matrix = np.stack([record[1] for record in records]).astype(np.float32)
    finite = np.isfinite(matrix).all(axis=1)
    metadata = metadata.loc[finite].reset_index(drop=True)
    matrix = matrix[finite]
    scaler = RobustScaler(quantile_range=(10.0, 90.0))
    scaled = scaler.fit_transform(matrix)
    full_pca = PCA(random_state=int(seed))
    transformed_full = full_pca.fit_transform(scaled)
    cumulative = np.cumsum(full_pca.explained_variance_ratio_)
    components = int(np.searchsorted(cumulative, 0.95) + 1)
    transformed = transformed_full[:, :components]
    model = MiniBatchKMeans(
        n_clusters=6,
        random_state=int(seed),
        batch_size=min(max(1024, len(matrix) // 10), 8192),
        n_init=10,
    )
    metadata["cluster"] = model.fit_predict(transformed).astype(np.int16)
    _atomic_write_parquet(output_dir / "atlas_sample_metadata.parquet", metadata)
    np.savez_compressed(
        output_dir / "atlas_sample_paths.npz",
        morphology=matrix,
        pca=transformed.astype(np.float32),
        cluster=metadata["cluster"].to_numpy(dtype=np.int16),
    )

    cluster_rows: list[dict[str, Any]] = []
    fig, axes = plt.subplots(2, 3, figsize=(16, 8), sharex=True)
    close = matrix[:, :20]
    for cluster in range(6):
        mask = metadata["cluster"].to_numpy() == cluster
        current = close[mask]
        center = np.nanmedian(current, axis=0)
        lower = np.nanquantile(current, 0.10, axis=0)
        upper = np.nanquantile(current, 0.90, axis=0)
        axis = axes.reshape(-1)[cluster]
        days = np.arange(1, 21)
        axis.plot(days, center, color="#0B6E4F", linewidth=2)
        axis.fill_between(days, lower, upper, color="#8FC0A9", alpha=0.35)
        axis.axhline(0.0, color="black", linewidth=0.7)
        axis.set_title(f"Cluster {cluster} (n={int(mask.sum()):,})")
        cluster_rows.append(
            {
                "cluster": int(cluster),
                "sample_count": int(mask.sum()),
                "median_close_d5": float(np.nanmedian(current[:, 4])),
                "median_close_d10": float(np.nanmedian(current[:, 9])),
                "median_close_d20": float(np.nanmedian(current[:, 19])),
            }
        )
    fig.suptitle("PIT D1-D20 path morphology (P10/P50/P90)")
    fig.tight_layout()
    chart_path = output_dir / "path_clusters_k6.png"
    fig.savefig(chart_path, dpi=160)
    plt.close(fig)
    cluster_frame = pd.DataFrame(cluster_rows)
    _atomic_write_parquet(output_dir / "path_cluster_summary.parquet", cluster_frame)
    return {
        "sample_count": int(len(matrix)),
        "pca_components_95pct": components,
        "pca_explained_variance_95pct": float(cumulative[components - 1]),
        "cluster_count": 6,
        "cluster_summary": str((output_dir / "path_cluster_summary.parquet").resolve()),
        "chart": str(chart_path.resolve()),
    }


def build_atlas(
    *,
    study_path: Path = DEFAULT_STUDY_PATH,
    output_root: Path = DEFAULT_OUTPUT_ROOT,
    pareto_device: str = "auto",
) -> dict[str, Any]:
    study = load_study(study_path)
    pack_path, manifest = _validate_source_bindings(study)
    target_spec_payload = dict(study["contract"].get("target", {}) or {})
    spec = PathTargetSpec(
        profile=str(target_spec_payload.get("profile", PATH_TARGET_PROFILE)),
        entry_anchor=str(
            target_spec_payload.get("entry_anchor", "actual_next_open_if_filled")
        ),
        primary_horizon_days=int(target_spec_payload.get("primary_horizon_days", 20)),
        continuation_diagnostic_days=tuple(
            int(item)
            for item in target_spec_payload.get("continuation_diagnostic_days", [40, 60])
        ),
        cost_profile=str(
            target_spec_payload.get(
                "cost_profile", "pack_manifest_bound_double_slippage_proportional_cost"
            )
        ),
        conditional_relevance_scope=str(
            target_spec_payload.get(
                "conditional_relevance_scope", "entry_filled_candidates"
            )
        ),
        unfilled_action_relevance=float(
            target_spec_payload.get("unfilled_action_relevance", 0.0)
        ),
        terminal_unrecoverable_value=float(
            target_spec_payload.get("terminal_unrecoverable_value", 0.0)
        ),
        daily_grouping=str(target_spec_payload.get("daily_grouping", "signal_trade_date")),
        pareto_dimensions=tuple(
            target_spec_payload.get("pareto_dimensions", PathTargetSpec().pareto_dimensions)
        ),
        relevance_grade_thresholds=tuple(
            float(item)
            for item in target_spec_payload.get(
                "relevance_grade_thresholds", GRADE_THRESHOLDS
            )
        ),
        trend_fallback_profile=str(
            target_spec_payload.get("trend_fallback_profile", TREND_FALLBACK_PROFILE)
        ),
        log_wealth_floor=float(target_spec_payload.get("log_wealth_floor", 1.0e-6)),
        pareto_block_size=int(target_spec_payload.get("pareto_block_size", 256)),
    )
    if spec.primary_horizon_days != 20:
        raise ValueError("v1 target implementation is frozen to a 20-day primary horizon")
    attempt_dir = _next_attempt_dir(output_root, "atlas")
    progress_path = attempt_dir / "progress.json"
    _atomic_write_json(
        progress_path,
        {
            "status": "initializing",
            "started_at": _now(),
            "study_contract_sha256": study["contract_sha256"],
            "pack_manifest": str(pack_path),
        },
    )

    candidate_path = _resolve_path(str(manifest.get("candidate_index_path", "")))
    candidate_metadata = pq.ParquetFile(candidate_path).metadata
    candidate_count = int(candidate_metadata.num_rows)
    date_values = [str(item) for item in list(manifest.get("date_values", []) or [])]
    date_count = len(date_values)
    symbol_count = int(manifest.get("symbol_count", 0) or 0)
    cutoff = str(dict(study["contract"].get("data", {}) or {}).get("label_endpoint_cutoff", "2025-12-31"))
    if cutoff >= "2026-01-01":
        raise ValueError("2026 must remain excluded from target construction")

    future_reader = _open_future_path(manifest)
    exit_sellable_panel = _open_bool_panel(manifest, "exit_sellable")
    delisted_panel = _open_bool_panel(manifest, "is_delisted")
    growth_by_date = sequence_training._v3_growth_multiplier_for_dates(
        manifest,
        np.arange(date_count, dtype=np.int64),
        int(manifest.get("forward_days", 60) or 60),
        slippage_multiplier=2.0,
    )

    partial = attempt_dir / "partial"
    partial.mkdir(parents=True, exist_ok=False)
    target_partial = partial / "target_values.float32.dat"
    counts_partial = partial / "dominance_counts.int32.dat"
    grade_partial = partial / "relevance_grade.uint8.dat"
    flags_partial = partial / "target_flags.uint8.dat"
    target = np.memmap(
        target_partial,
        dtype="float32",
        mode="w+",
        shape=(candidate_count, len(TARGET_FLOAT_FIELDS)),
    )
    target[:] = np.nan
    counts = np.memmap(
        counts_partial,
        dtype="int32",
        mode="w+",
        shape=(candidate_count, 2),
    )
    counts[:] = -1
    grades = np.memmap(
        grade_partial,
        dtype="uint8",
        mode="w+",
        shape=(candidate_count,),
    )
    grades[:] = np.uint8(255)
    flags = np.memmap(
        flags_partial,
        dtype="uint8",
        mode="w+",
        shape=(candidate_count,),
    )
    flags[:] = np.uint8(0)

    reservoir = _DeterministicReservoir(capacity_per_bucket=500, seed=7)
    daily_rows: list[dict[str, Any]] = []
    fallback_daily_rows: list[dict[str, Any]] = []
    processed_candidates = 0
    processed_dates = 0
    skipped_after_cutoff = 0
    started = time.monotonic()
    columns = (
        "candidate_id",
        "year",
        "trade_date",
        "date_idx",
        "symbol_idx",
        "symbol",
        "entry_filled",
        "price_label_valid",
    )
    try:
        for group in _iter_date_groups(candidate_path, columns=columns):
            date_idx = int(group["date_idx"].iloc[0])
            if not label_endpoint_is_allowed(
                signal_date_idx=date_idx,
                date_values=date_values,
                horizon_days=20,
                cutoff=cutoff,
            ):
                skipped_after_cutoff += int(len(group))
                continue
            candidate_ids = group["candidate_id"].to_numpy(dtype=np.int64, copy=False)
            symbol_idx = group["symbol_idx"].to_numpy(dtype=np.int64, copy=False)
            date_idx_values = np.full(len(group), date_idx, dtype=np.int64)
            path = _take_future_ohlc(
                future_reader,
                date_idx_values,
                symbol_idx,
                horizon=20,
            )
            entry_path = reanchor_to_actual_next_open(
                path,
                source_price_anchor="today_close",
            )
            sellable = _future_panel_view(
                exit_sellable_panel,
                signal_date_idx=date_idx,
                symbol_idx=symbol_idx,
                horizon=20,
            ).astype(bool, copy=False)
            delisted = _future_panel_view(
                delisted_panel,
                signal_date_idx=date_idx,
                symbol_idx=symbol_idx,
                horizon=20,
            ).astype(bool, copy=False)
            entry_path, terminal_failure = apply_terminal_zero_recovery(
                entry_path,
                delisted,
            )
            multiplier = np.broadcast_to(
                growth_by_date[date_idx, :20].reshape(1, 20),
                (len(group), 20),
            ).copy()
            descriptors = compute_path_descriptors(
                entry_path,
                multiplier,
                exit_sellable=sellable,
                terminal_failure=terminal_failure,
                log_wealth_floor=spec.log_wealth_floor,
            )
            entry_filled = group["entry_filled"].astype("boolean").fillna(False).to_numpy(
                dtype=bool
            )
            price_label_valid = (
                group["price_label_valid"]
                .astype("boolean")
                .fillna(False)
                .to_numpy(dtype=bool)
            )
            path_available = np.asarray(descriptors["path_available"], dtype=bool)
            no_legal_sell = np.asarray(descriptors["no_legal_sell"], dtype=bool)
            forced_low = entry_filled & path_available & (terminal_failure | no_legal_sell)
            relevance = compute_daily_relevance(
                descriptors,
                entry_filled=entry_filled,
                path_available=path_available,
                forced_low=forced_low,
                symbol_idx=symbol_idx,
                pareto_device=pareto_device,
                pareto_block_size=spec.pareto_block_size,
            )
            _assign_target_fields(target, candidate_ids, descriptors, relevance)
            counts[candidate_ids, 0] = np.asarray(
                relevance["dominated_count"], dtype=np.int32
            )
            counts[candidate_ids, 1] = np.asarray(
                relevance["dominating_count"], dtype=np.int32
            )
            grades[candidate_ids] = np.asarray(relevance["relevance_grade"], dtype=np.uint8)
            current_flags = np.zeros(len(group), dtype=np.uint8)
            current_flags |= entry_filled.astype(np.uint8) * TARGET_FLAG_ENTRY_FILLED
            current_flags |= path_available.astype(np.uint8) * TARGET_FLAG_PATH_AVAILABLE
            current_flags |= forced_low.astype(np.uint8) * TARGET_FLAG_FORCED_LOW
            current_flags |= terminal_failure.astype(np.uint8) * TARGET_FLAG_TERMINAL_FAILURE
            current_flags |= no_legal_sell.astype(np.uint8) * TARGET_FLAG_NO_LEGAL_SELL
            current_flags |= price_label_valid.astype(np.uint8) * TARGET_FLAG_PRICE_LABEL_VALID
            flags[candidate_ids] = current_flags
            _append_daily_diagnostics(
                daily_rows,
                group=group,
                descriptors=descriptors,
                relevance=relevance,
                selection_profile=PATH_TARGET_PROFILE,
            )
            _append_daily_diagnostics(
                fallback_daily_rows,
                group=group,
                descriptors=descriptors,
                relevance=_fallback_relevance_view(
                    relevance,
                    entry_filled=entry_filled,
                ),
                selection_profile=TREND_FALLBACK_PROFILE,
            )

            morphology = _morphology_matrix(entry_path)
            current_grades = np.asarray(relevance["relevance_grade"], dtype=np.uint8)
            conditional = np.asarray(relevance["conditional_relevance"], dtype=np.float32)
            for row in np.flatnonzero(np.isfinite(conditional)):
                reservoir.add(
                    year=int(group["year"].iloc[row]),
                    grade=int(current_grades[row]),
                    candidate_id=int(candidate_ids[row]),
                    metadata={
                        "candidate_id": int(candidate_ids[row]),
                        "trade_date": str(group["trade_date"].iloc[row]),
                        "year": int(group["year"].iloc[row]),
                        "symbol": str(group["symbol"].iloc[row]),
                        "grade": int(current_grades[row]),
                        "conditional_relevance": float(conditional[row]),
                        "r20_net": float(np.asarray(descriptors["r20_net"])[row]),
                        "mdd20": float(np.asarray(descriptors["mdd20"])[row]),
                    },
                    morphology=morphology[row],
                )
            processed_candidates += int(len(group))
            processed_dates += 1
            if processed_dates % 25 == 0:
                target.flush()
                counts.flush()
                grades.flush()
                flags.flush()
                _atomic_write_json(
                    progress_path,
                    {
                        "status": "building_targets",
                        "updated_at": _now(),
                        "processed_dates": processed_dates,
                        "processed_candidates": processed_candidates,
                        "skipped_after_cutoff": skipped_after_cutoff,
                        "elapsed_seconds": time.monotonic() - started,
                    },
                )
    except Exception as exc:
        target.flush()
        counts.flush()
        grades.flush()
        flags.flush()
        _atomic_write_json(
            progress_path,
            {
                "status": "failed",
                "failed_at": _now(),
                "error_type": type(exc).__name__,
                "error": str(exc),
                "processed_dates": processed_dates,
                "processed_candidates": processed_candidates,
            },
        )
        raise

    target.flush()
    counts.flush()
    grades.flush()
    flags.flush()
    del target, counts, grades, flags
    paths = _target_paths(attempt_dir)
    os.replace(target_partial, paths.float_values)
    os.replace(counts_partial, paths.dominance_counts)
    os.replace(grade_partial, paths.relevance_grade)
    os.replace(flags_partial, paths.flags)
    partial.rmdir()

    daily = pd.DataFrame(daily_rows)
    fallback_daily = pd.DataFrame(fallback_daily_rows)
    annual, gates = _annual_target_diagnostics(daily, require_frontier_gate=True)
    fallback_annual, fallback_gates = _annual_target_diagnostics(
        fallback_daily,
        require_frontier_gate=False,
    )
    _atomic_write_parquet(attempt_dir / "daily_target_diagnostics.parquet", daily)
    _atomic_write_parquet(attempt_dir / "annual_target_diagnostics.parquet", annual)
    _atomic_write_parquet(
        attempt_dir / "daily_trend_consistency_diagnostics.parquet",
        fallback_daily,
    )
    _atomic_write_parquet(
        attempt_dir / "annual_trend_consistency_diagnostics.parquet",
        fallback_annual,
    )
    atlas = _fit_path_atlas(reservoir.records(), attempt_dir, seed=7)
    if bool(gates["passed"]):
        frozen_profile = PATH_TARGET_PROFILE
    elif bool(fallback_gates["passed"]):
        frozen_profile = TREND_FALLBACK_PROFILE
    else:
        frozen_profile = "target_invalid"
    gates["fallback_profile"] = TREND_FALLBACK_PROFILE
    gates["fallback_freeze_gates"] = fallback_gates
    if frozen_profile != "target_invalid":
        _rewrite_frozen_relevance_fields(
            paths,
            candidate_count=candidate_count,
            frozen_profile=frozen_profile,
        )

    file_metadata = {}
    for name, path in {
        "float_values": paths.float_values,
        "dominance_counts": paths.dominance_counts,
        "relevance_grade": paths.relevance_grade,
        "flags": paths.flags,
    }.items():
        file_metadata[name] = {
            "path": str(path.resolve()),
            "size": int(path.stat().st_size),
            "sha256": _file_sha256(path),
        }
    target_manifest = {
        "artifact_type": TARGET_ARTIFACT_TYPE,
        "schema_version": 1,
        "created_at": _now(),
        "study_contract": str(study_path.resolve()),
        "study_contract_sha256": study["contract_sha256"],
        "target_spec": asdict(spec),
        "target_spec_sha256": _canonical_json_sha256(target_spec_payload),
        "source_pack_manifest": str(pack_path.resolve()),
        "source_pack_manifest_sha256": _file_sha256(pack_path),
        "candidate_index": str(candidate_path.resolve()),
        "candidate_index_sha256": _file_sha256(candidate_path),
        "candidate_count": candidate_count,
        "processed_candidate_count": processed_candidates,
        "skipped_after_cutoff": skipped_after_cutoff,
        "date_count": date_count,
        "symbol_count": symbol_count,
        "label_endpoint_cutoff": cutoff,
        "float_fields": list(TARGET_FLOAT_FIELDS),
        "float_shape": [candidate_count, len(TARGET_FLOAT_FIELDS)],
        "dominance_shape": [candidate_count, 2],
        "grade_shape": [candidate_count],
        "flag_shape": [candidate_count],
        "flag_bits": {
            "entry_filled": TARGET_FLAG_ENTRY_FILLED,
            "path_available": TARGET_FLAG_PATH_AVAILABLE,
            "forced_low": TARGET_FLAG_FORCED_LOW,
            "terminal_failure": TARGET_FLAG_TERMINAL_FAILURE,
            "no_legal_sell": TARGET_FLAG_NO_LEGAL_SELL,
            "price_label_valid": TARGET_FLAG_PRICE_LABEL_VALID,
        },
        "files": file_metadata,
        "frozen_profile": frozen_profile,
        "freeze_gates": gates,
        "atlas": atlas,
        "status": "completed" if frozen_profile != "target_invalid" else "target_invalid",
    }
    _atomic_write_json(paths.manifest, target_manifest)
    summary = {
        "status": target_manifest["status"],
        "output_dir": str(attempt_dir.resolve()),
        "target_manifest": str(paths.manifest.resolve()),
        "frozen_profile": frozen_profile,
        "freeze_gates": gates,
        "processed_candidate_count": processed_candidates,
        "skipped_after_cutoff": skipped_after_cutoff,
        "atlas": atlas,
    }
    summary_path = attempt_dir / "atlas_summary.json"
    _atomic_write_json(summary_path, summary)
    _atomic_write_json(
        progress_path,
        {
            "status": target_manifest["status"],
            "completed_at": _now(),
            "summary_json": str(summary_path.resolve()),
            "target_manifest": str(paths.manifest.resolve()),
        },
    )
    if target_manifest["status"] == "completed":
        _atomic_write_json(
            output_root / "atlas/current.json",
            {
                "attempt": str(attempt_dir.resolve()),
                "target_manifest": str(paths.manifest.resolve()),
                "study_contract_sha256": study["contract_sha256"],
                "updated_at": _now(),
            },
        )
    return summary


def _load_current_artifact(
    output_root: Path,
    *,
    task: str,
    pointer_name: str,
) -> tuple[Path, dict[str, Any]]:
    pointer_path = output_root / task / "current.json"
    if not pointer_path.exists():
        raise FileNotFoundError(
            f"{task} has no completed current artifact; run the prerequisite command first"
        )
    pointer = json.loads(pointer_path.read_text(encoding="utf-8"))
    artifact_path = Path(str(pointer.get(pointer_name, "") or "")).resolve()
    if not artifact_path.exists():
        raise FileNotFoundError(artifact_path)
    return artifact_path, json.loads(artifact_path.read_text(encoding="utf-8"))


def _verify_protected_bindings(study: Mapping[str, Any]) -> dict[str, str]:
    _pack_path, _manifest = _validate_source_bindings(study)
    data = dict(study["contract"]["data"])
    active = _resolve_path(str(data["qdp_active_manifest"]))
    active_hash = _file_sha256(active)
    if active_hash != str(data["qdp_active_manifest_sha256"]):
        raise ValueError("QDP active manifest changed during the frozen study")
    registry = WORKSPACE_ROOT / "daily_research/models/registry.json"
    registry_hash = _file_sha256(registry)
    expected_registry_hash = str(
        study["contract"]["protection"]["baseline_model_registry_sha256"]
    )
    if registry_hash != expected_registry_hash:
        raise ValueError("registered model registry changed during the frozen study")
    return {
        "qdp_active_manifest_sha256": active_hash,
        "registered_model_registry_sha256": registry_hash,
    }


def _load_feature_view_bundle(
    output_root: Path,
    study: Mapping[str, Any],
) -> tuple[Path, dict[str, Any]]:
    path, manifest = _load_current_artifact(
        output_root,
        task="feature_view",
        pointer_name="feature_view_manifest",
    )
    if str(manifest.get("status", "")) != "completed":
        raise ValueError("feature view is not complete")
    if str(manifest.get("study_contract_sha256", "")) != str(
        study["contract_sha256"]
    ):
        raise ValueError("feature view belongs to a different study contract")
    declared = str(manifest.get("resolved_feature_view_sha256", ""))
    payload = dict(manifest)
    payload.pop("resolved_feature_view_sha256", None)
    if declared != _canonical_json_sha256(payload):
        raise ValueError("feature view resolved hash mismatch")
    for metadata in dict(manifest.get("files", {}) or {}).values():
        file_path = Path(str(metadata["path"])).resolve()
        if int(file_path.stat().st_size) != int(metadata["size"]):
            raise ValueError(f"feature view file size changed: {file_path}")
        if _file_sha256(file_path) != str(metadata["sha256"]):
            raise ValueError(f"feature view file hash changed: {file_path}")
    return path, manifest


def _open_feature_arrays(
    feature_manifest: Mapping[str, Any],
) -> tuple[np.memmap, np.memmap, np.memmap, np.memmap]:
    files = dict(feature_manifest["files"])
    continuous_shape = tuple(int(item) for item in feature_manifest["continuous_shape"])
    categorical_shape = tuple(int(item) for item in feature_manifest["categorical_shape"])
    candidate_count = int(feature_manifest["candidate_count"])
    return (
        np.memmap(
            files["continuous"]["path"],
            dtype="float32",
            mode="r",
            shape=continuous_shape,
        ),
        np.memmap(
            files["categorical"]["path"],
            dtype="int64",
            mode="r",
            shape=categorical_shape,
        ),
        np.memmap(
            files["candidate_date_idx"]["path"],
            dtype="int32",
            mode="r",
            shape=(candidate_count,),
        ),
        np.memmap(
            files["candidate_symbol_idx"]["path"],
            dtype="int32",
            mode="r",
            shape=(candidate_count,),
        ),
    )


def _load_target_bundle(
    output_root: Path,
    study: Mapping[str, Any],
) -> tuple[dict[str, Any], np.memmap, np.memmap, np.memmap, np.memmap]:
    path, manifest = _load_current_artifact(
        output_root,
        task="atlas",
        pointer_name="target_manifest",
    )
    if str(manifest.get("status", "")) != "completed":
        raise ValueError("target is not frozen")
    if str(manifest.get("study_contract_sha256", "")) != str(
        study["contract_sha256"]
    ):
        raise ValueError("target belongs to a different study contract")
    candidate_count = int(manifest["candidate_count"])
    files = dict(manifest["files"])
    for metadata in files.values():
        artifact = Path(str(metadata["path"])).resolve()
        if _file_sha256(artifact) != str(metadata["sha256"]):
            raise ValueError(f"target file hash changed: {artifact}")
    manifest = dict(manifest)
    manifest["manifest_path"] = str(path.resolve())
    return (
        manifest,
        np.memmap(
            files["float_values"]["path"],
            dtype="float32",
            mode="r",
            shape=(candidate_count, len(TARGET_FLOAT_FIELDS)),
        ),
        np.memmap(
            files["dominance_counts"]["path"],
            dtype="int32",
            mode="r",
            shape=(candidate_count, 2),
        ),
        np.memmap(
            files["relevance_grade"]["path"],
            dtype="uint8",
            mode="r",
            shape=(candidate_count,),
        ),
        np.memmap(
            files["flags"]["path"],
            dtype="uint8",
            mode="r",
            shape=(candidate_count,),
        ),
    )


def _load_fold_view(feature_manifest: Mapping[str, Any], fold_year: int) -> dict[str, Any]:
    metadata = dict(feature_manifest["fold_views"].get(str(int(fold_year)), {}) or {})
    if not metadata:
        raise KeyError(f"feature view lacks fold {fold_year}")
    path = Path(str(metadata["path"])).resolve()
    if _file_sha256(path) != str(metadata["sha256"]):
        raise ValueError(f"fold view changed: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    declared = str(payload.get("view_sha256", ""))
    check = dict(payload)
    check.pop("view_sha256", None)
    if declared != _canonical_json_sha256(check):
        raise ValueError("fold view hash mismatch")
    return payload


def _row_ids_for_date_range(
    candidate_date_idx: np.ndarray,
    start: int,
    end: int,
) -> np.ndarray:
    dates = np.asarray(candidate_date_idx, dtype=np.int32)
    left = int(np.searchsorted(dates, int(start), side="left"))
    right = int(np.searchsorted(dates, int(end), side="right"))
    return np.arange(left, right, dtype=np.int64)


def _profile_families(profile: str) -> tuple[str, ...]:
    families = tuple(part.strip() for part in str(profile).split("+") if part.strip())
    if not families or families[0] != "F1":
        raise ValueError(f"feature profile must be cumulative from F1: {profile}")
    expected = ("F1", "F2", "F3", "F4", "F5")[: len(families)]
    if families != expected:
        raise ValueError(f"feature profile is not a registered cumulative profile: {profile}")
    return families


def _feature_columns_for_profile(
    feature_manifest: Mapping[str, Any],
    profile: str,
) -> tuple[np.ndarray, np.ndarray, list[str], list[str]]:
    families = set(_profile_families(profile))
    continuous_catalog = list(feature_manifest["continuous_catalog"])
    categorical_catalog = list(feature_manifest["categorical_catalog"])
    continuous_items = [item for item in continuous_catalog if item["family"] in families]
    categorical_items = [item for item in categorical_catalog if item["family"] in families]
    return (
        np.asarray([item["column_index"] for item in continuous_items], dtype=np.int32),
        np.asarray([item["column_index"] for item in categorical_items], dtype=np.int32),
        [str(item["name"]) for item in continuous_items],
        [str(item["name"]) for item in categorical_items],
    )


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
        block = np.asarray(categorical[np.ix_(rows, categorical_columns)], dtype=np.int64)
        for column in range(block.shape[1]):
            values = np.unique(block[:, column])
            vocabularies[column].update(int(value) for value in values if int(value) != 0)
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
                    continuous[global_row, continuous_columns], dtype=np.float32
                )
                category_values = [
                    _map_categories(
                        np.asarray([categorical[global_row, int(column)]], dtype=np.int64),
                        category_vocabularies[position],
                    )[0]
                    for position, column in enumerate(categorical_columns)
                ]
                return np.concatenate(
                    [continuous_values, np.asarray(category_values, dtype=np.float32)]
                )
            else:
                raise TypeError(f"unsupported LightGBM sequence index: {type(idx).__name__}")
            global_rows = row_ids[local]
            continuous_values = np.asarray(
                continuous[np.ix_(global_rows, continuous_columns)], dtype=np.float32
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
            return np.concatenate(
                [continuous_values, *category_parts], axis=1
            ).astype(np.float32, copy=False)

    return ResearchSequence()


def _predict_lgb_sequence(model: Any, sequence: Any, *, chunk_size: int = 250_000) -> np.ndarray:
    output = np.empty(len(sequence), dtype=np.float32)
    for start in range(0, len(sequence), int(chunk_size)):
        stop = min(start + int(chunk_size), len(sequence))
        output[start:stop] = np.asarray(
            model.predict(sequence[start:stop], num_iteration=model.best_iteration),
            dtype=np.float32,
        )
    return output


def _ndcg_for_group(labels: np.ndarray, scores: np.ndarray, k: int) -> float:
    truth = np.asarray(labels, dtype=np.int64)
    predicted = np.asarray(scores, dtype=np.float64)
    if truth.size == 0:
        return np.nan
    cutoff = min(max(int(k), 1), int(truth.size))
    gains = np.asarray([0.0, 1.0, 3.0, 7.0, 15.0], dtype=np.float64)
    predicted_order = np.argsort(-predicted, kind="mergesort")[:cutoff]
    ideal_order = np.argsort(-truth, kind="mergesort")[:cutoff]
    discount = 1.0 / np.log2(np.arange(2, cutoff + 2, dtype=np.float64))
    dcg = float(np.sum(gains[np.clip(truth[predicted_order], 0, 4)] * discount))
    ideal = float(np.sum(gains[np.clip(truth[ideal_order], 0, 4)] * discount))
    return dcg / ideal if ideal > 0.0 else 1.0


def daily_ranking_metrics(
    *,
    date_idx: np.ndarray,
    action_relevance: np.ndarray,
    selection_score: np.ndarray,
) -> tuple[pd.DataFrame, dict[str, float]]:
    dates = np.asarray(date_idx, dtype=np.int32)
    truth = np.asarray(action_relevance, dtype=np.float32)
    score = np.asarray(selection_score, dtype=np.float32)
    if dates.shape != truth.shape or dates.shape != score.shape:
        raise ValueError("ranking metric arrays must be aligned")
    valid = np.isfinite(truth) & np.isfinite(score)
    grades = relevance_grades(truth)
    rows: list[dict[str, Any]] = []
    boundaries = np.flatnonzero(np.r_[True, dates[1:] != dates[:-1], True])
    for start, stop in zip(boundaries[:-1], boundaries[1:]):
        mask = valid[start:stop]
        if not bool(mask.any()):
            continue
        current_truth = truth[start:stop][mask]
        current_score = score[start:stop][mask]
        current_grades = grades[start:stop][mask]
        count = int(mask.sum())
        k1 = max(1, int(math.ceil(0.01 * count)))
        k5 = max(1, int(math.ceil(0.05 * count)))
        spearman = stats.spearmanr(current_truth, current_score).statistic
        rows.append(
            {
                "date_idx": int(dates[start]),
                "candidate_count": count,
                "ndcg_at_1pct": _ndcg_for_group(current_grades, current_score, k1),
                "ndcg_at_5pct": _ndcg_for_group(current_grades, current_score, k5),
                "rank_ic": float(spearman) if math.isfinite(float(spearman)) else np.nan,
            }
        )
    frame = pd.DataFrame(rows)
    summary = {
        "daily_ndcg_at_1pct": float(frame["ndcg_at_1pct"].mean()),
        "daily_ndcg_at_5pct": float(frame["ndcg_at_5pct"].mean()),
        "daily_spearman_rank_ic": float(frame["rank_ic"].mean()),
        "date_count": int(len(frame)),
    }
    return frame, summary


def _date_group_sizes(date_idx: np.ndarray) -> np.ndarray:
    dates = np.asarray(date_idx, dtype=np.int32)
    if dates.ndim != 1:
        raise ValueError("date group values must be one-dimensional")
    if not dates.size:
        return np.empty(0, dtype=np.int32)
    if bool(np.any(dates[1:] < dates[:-1])):
        raise ValueError("date groups must be ordered")
    boundaries = np.flatnonzero(np.r_[True, dates[1:] != dates[:-1], True])
    return np.diff(boundaries).astype(np.int32)


def _lgb_parameters(
    study: Mapping[str, Any],
    *,
    model_id: str,
    objective: str,
) -> tuple[dict[str, Any], int, int]:
    config = dict(study["contract"]["models"][model_id])
    parameters: dict[str, Any] = {
        "objective": objective,
        "num_leaves": int(config["num_leaves"]),
        "min_data_in_leaf": int(config["min_data_in_leaf"]),
        "learning_rate": float(config["learning_rate"]),
        "feature_fraction": float(config["feature_fraction"]),
        "bagging_fraction": float(config["bagging_fraction"]),
        "bagging_freq": int(config["bagging_freq"]),
        "lambda_l2": float(config["lambda_l2"]),
        "seed": int(study["contract"]["models"]["common"]["seed"]),
        "feature_fraction_seed": int(study["contract"]["models"]["common"]["seed"]),
        "bagging_seed": int(study["contract"]["models"]["common"]["seed"]),
        "data_random_seed": int(study["contract"]["models"]["common"]["seed"]),
        "verbosity": -1,
        "deterministic": True,
        "force_col_wise": True,
        "num_threads": max(min(int(os.cpu_count() or 4), 12), 1),
    }
    if objective == "lambdarank":
        parameters.update(
            {
                "metric": "ndcg",
                "label_gain": list(config["label_gain"]),
                "ndcg_eval_at": [1, 3, 10, 50],
            }
        )
    elif objective == "binary":
        parameters.update({"metric": ["binary_logloss", "binary_error"]})
    else:
        parameters.update({"metric": "huber", "alpha": 0.9})
    return (
        parameters,
        int(config["maximum_rounds"]),
        int(config["early_stopping_rounds"]),
    )


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


def _train_lgb_model(
    *,
    study: Mapping[str, Any],
    model_id: str,
    objective: str,
    train_sequence: Any,
    train_label: np.ndarray,
    development_sequence: Any,
    development_label: np.ndarray,
    feature_names: Sequence[str],
    categorical_count: int,
    output_path: Path,
    train_group: np.ndarray | None = None,
    development_group: np.ndarray | None = None,
) -> tuple[Any, dict[str, Any]]:
    import lightgbm as lgb

    parameters, maximum_rounds, patience = _lgb_parameters(
        study,
        model_id=model_id,
        objective=objective,
    )
    train_set = _lgb_dataset(
        sequence=train_sequence,
        label=train_label,
        feature_names=feature_names,
        categorical_count=categorical_count,
        group=train_group,
    )
    development_set = _lgb_dataset(
        sequence=development_sequence,
        label=development_label,
        feature_names=feature_names,
        categorical_count=categorical_count,
        group=development_group,
        reference=train_set,
    )
    evaluation: dict[str, Any] = {}
    started = time.perf_counter()
    model = lgb.train(
        parameters,
        train_set,
        num_boost_round=maximum_rounds,
        valid_sets=[development_set],
        valid_names=["development"],
        callbacks=[
            lgb.early_stopping(stopping_rounds=patience, first_metric_only=True, verbose=False),
            lgb.record_evaluation(evaluation),
            lgb.log_evaluation(period=50),
        ],
    )
    elapsed = time.perf_counter() - started
    output_path.parent.mkdir(parents=True, exist_ok=True)
    model.save_model(str(output_path), num_iteration=model.best_iteration)
    return model, {
        "model_path": str(output_path.resolve()),
        "model_sha256": _file_sha256(output_path),
        "best_iteration": int(model.best_iteration),
        "training_seconds": float(elapsed),
        "parameters": parameters,
        "evaluation": evaluation,
    }


def _fit_isotonic_mapping(
    raw_prediction: np.ndarray,
    conditional_relevance: np.ndarray,
) -> tuple[Any, dict[str, Any]]:
    from sklearn.isotonic import IsotonicRegression

    raw = np.asarray(raw_prediction, dtype=np.float64)
    target = np.asarray(conditional_relevance, dtype=np.float64)
    valid = np.isfinite(raw) & np.isfinite(target)
    if int(valid.sum()) < 2:
        raise ValueError("isotonic calibration requires at least two finite development rows")
    model = IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds="clip")
    model.fit(raw[valid], target[valid])
    return model, {
        "sample_count": int(valid.sum()),
        "x_thresholds": np.asarray(model.X_thresholds_, dtype=np.float64).tolist(),
        "y_thresholds": np.asarray(model.y_thresholds_, dtype=np.float64).tolist(),
    }


def _restore_isotonic_mapping(payload: Mapping[str, Any]) -> Any:
    from sklearn.isotonic import IsotonicRegression

    model = IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds="clip")
    model.X_thresholds_ = np.asarray(payload["x_thresholds"], dtype=np.float64)
    model.y_thresholds_ = np.asarray(payload["y_thresholds"], dtype=np.float64)
    model.f_ = lambda x: np.interp(
        x,
        model.X_thresholds_,
        model.y_thresholds_,
    )
    return model


def feature_screen(
    *,
    study_path: Path = DEFAULT_STUDY_PATH,
    output_root: Path = DEFAULT_OUTPUT_ROOT,
) -> dict[str, Any]:
    study = load_study(study_path)
    _view_path, feature_manifest = _load_feature_view_bundle(output_root, study)
    target_manifest, target, _counts, _grades, flags = _load_target_bundle(
        output_root, study
    )
    continuous, categorical, candidate_date_idx, _candidate_symbol_idx = _open_feature_arrays(
        feature_manifest
    )
    fold = _load_fold_view(feature_manifest, 2023)
    train_range = dict(fold["ranges"]["train"])
    development_range = dict(fold["ranges"]["development"])
    train_rows = _row_ids_for_date_range(
        candidate_date_idx,
        train_range["date_idx_start"],
        train_range["date_idx_end"],
    )
    development_rows = _row_ids_for_date_range(
        candidate_date_idx,
        development_range["date_idx_start"],
        development_range["date_idx_end"],
    )
    conditional = np.asarray(
        target[:, TARGET_FIELD_INDEX["conditional_relevance"]], dtype=np.float32
    )
    action = np.asarray(target[:, TARGET_FIELD_INDEX["action_relevance"]], dtype=np.float32)
    entry_filled = (np.asarray(flags, dtype=np.uint8) & TARGET_FLAG_ENTRY_FILLED) != 0
    rank_train_rows = train_rows[entry_filled[train_rows] & np.isfinite(conditional[train_rows])]
    rank_development_rows = development_rows[
        entry_filled[development_rows] & np.isfinite(conditional[development_rows])
    ]
    if not rank_train_rows.size or not rank_development_rows.size:
        raise ValueError("feature screen has no conditional relevance rows")

    attempt_dir = _next_attempt_dir(output_root, "feature_screen")
    progress_path = attempt_dir / "progress.json"
    _atomic_write_json(
        progress_path,
        {
            "status": "running",
            "started_at": _now(),
            "study_contract_sha256": study["contract_sha256"],
            "feature_view_sha256": feature_manifest["resolved_feature_view_sha256"],
            "target_manifest_sha256": _file_sha256(
                Path(str(target_manifest["manifest_path"]))
            ),
        },
    )
    profiles = list(study["contract"]["feature_screen"]["profiles_in_order"])
    records: list[dict[str, Any]] = []
    for profile in profiles:
        profile_dir = attempt_dir / str(profile).replace("+", "_")
        profile_dir.mkdir(parents=True, exist_ok=False)
        continuous_columns, categorical_columns, continuous_names, categorical_names = (
            _feature_columns_for_profile(feature_manifest, profile)
        )
        vocabularies = _fit_category_vocabularies(
            categorical,
            train_rows,
            categorical_columns,
        )
        feature_names = [*continuous_names, *categorical_names]
        train_all_sequence = _make_lgb_sequence(
            continuous=continuous,
            categorical=categorical,
            row_ids=train_rows,
            continuous_columns=continuous_columns,
            categorical_columns=categorical_columns,
            category_vocabularies=vocabularies,
        )
        development_all_sequence = _make_lgb_sequence(
            continuous=continuous,
            categorical=categorical,
            row_ids=development_rows,
            continuous_columns=continuous_columns,
            categorical_columns=categorical_columns,
            category_vocabularies=vocabularies,
        )
        fill_model, fill_evidence = _train_lgb_model(
            study=study,
            model_id="lgbm_lambdarank",
            objective="binary",
            train_sequence=train_all_sequence,
            train_label=entry_filled[train_rows].astype(np.uint8),
            development_sequence=development_all_sequence,
            development_label=entry_filled[development_rows].astype(np.uint8),
            feature_names=feature_names,
            categorical_count=len(categorical_names),
            output_path=profile_dir / "fill_model.txt",
        )
        rank_train_sequence = _make_lgb_sequence(
            continuous=continuous,
            categorical=categorical,
            row_ids=rank_train_rows,
            continuous_columns=continuous_columns,
            categorical_columns=categorical_columns,
            category_vocabularies=vocabularies,
        )
        rank_development_sequence = _make_lgb_sequence(
            continuous=continuous,
            categorical=categorical,
            row_ids=rank_development_rows,
            continuous_columns=continuous_columns,
            categorical_columns=categorical_columns,
            category_vocabularies=vocabularies,
        )
        rank_model, rank_evidence = _train_lgb_model(
            study=study,
            model_id="lgbm_lambdarank",
            objective="lambdarank",
            train_sequence=rank_train_sequence,
            train_label=np.asarray(_grades[rank_train_rows], dtype=np.uint8),
            development_sequence=rank_development_sequence,
            development_label=np.asarray(_grades[rank_development_rows], dtype=np.uint8),
            feature_names=feature_names,
            categorical_count=len(categorical_names),
            output_path=profile_dir / "rank_model.txt",
            train_group=_date_group_sizes(candidate_date_idx[rank_train_rows]),
            development_group=_date_group_sizes(
                candidate_date_idx[rank_development_rows]
            ),
        )
        raw_rank_development = _predict_lgb_sequence(
            rank_model, rank_development_sequence
        )
        isotonic, isotonic_evidence = _fit_isotonic_mapping(
            raw_rank_development,
            conditional[rank_development_rows],
        )
        _atomic_write_json(profile_dir / "isotonic.json", isotonic_evidence)
        fill_probability = _predict_lgb_sequence(
            fill_model, development_all_sequence
        )
        raw_all_rank = _predict_lgb_sequence(rank_model, development_all_sequence)
        conditional_prediction = np.asarray(
            isotonic.predict(raw_all_rank), dtype=np.float32
        )
        selection_score = fill_probability * conditional_prediction
        daily, metrics = daily_ranking_metrics(
            date_idx=candidate_date_idx[development_rows],
            action_relevance=action[development_rows],
            selection_score=selection_score,
        )
        _atomic_write_parquet(profile_dir / "daily_metrics.parquet", daily)
        vocabulary_payload = {
            name: values.tolist() for name, values in zip(categorical_names, vocabularies)
        }
        _atomic_write_json(profile_dir / "category_vocabularies.json", vocabulary_payload)
        record = {
            "profile": profile,
            "families": list(_profile_families(profile)),
            "continuous_feature_count": len(continuous_names),
            "categorical_feature_count": len(categorical_names),
            "train_candidate_count": int(len(train_rows)),
            "development_candidate_count": int(len(development_rows)),
            "rank_train_count": int(len(rank_train_rows)),
            "rank_development_count": int(len(rank_development_rows)),
            "metrics": metrics,
            "fill_model": fill_evidence,
            "rank_model": rank_evidence,
            "isotonic": isotonic_evidence,
            "feature_names": feature_names,
            "resolved_config_sha256": _canonical_json_sha256(
                {
                    "profile": profile,
                    "feature_names": feature_names,
                    "fill_parameters": fill_evidence["parameters"],
                    "rank_parameters": rank_evidence["parameters"],
                }
            ),
        }
        _atomic_write_json(profile_dir / "summary.json", record)
        records.append(record)
        _atomic_write_json(
            progress_path,
            {
                "status": "running",
                "updated_at": _now(),
                "completed_profiles": [item["profile"] for item in records],
            },
        )
        del fill_model, rank_model, isotonic
        _trim_after_feature_stage()
    best_metric = max(float(item["metrics"]["daily_ndcg_at_1pct"]) for item in records)
    tolerance = float(study["contract"]["feature_screen"]["tie_tolerance"])
    eligible = [
        item
        for item in records
        if float(item["metrics"]["daily_ndcg_at_1pct"]) >= best_metric - tolerance
    ]
    selected = min(eligible, key=lambda item: profiles.index(item["profile"]))
    freeze = {
        "artifact_type": "seq100_path_relevance_feature_freeze",
        "status": "completed",
        "created_at": _now(),
        "study_contract_sha256": study["contract_sha256"],
        "feature_view_sha256": feature_manifest["resolved_feature_view_sha256"],
        "target_manifest_sha256": _file_sha256(
            Path(str(target_manifest["manifest_path"]))
        ),
        "development_year": 2022,
        "profiles": records,
        "best_metric": best_metric,
        "tie_tolerance": tolerance,
        "selected_profile": selected["profile"],
        "selected_feature_names": selected["feature_names"],
        "selection_reason": "highest_ndcg_at_1pct_within_tolerance_choose_fewer_families",
    }
    freeze["feature_freeze_sha256"] = _canonical_json_sha256(freeze)
    freeze_path = attempt_dir / "feature_freeze.json"
    _atomic_write_json(freeze_path, freeze)
    summary = {
        "status": "completed",
        "output_dir": str(attempt_dir.resolve()),
        "feature_freeze": str(freeze_path.resolve()),
        "feature_freeze_sha256": freeze["feature_freeze_sha256"],
        "selected_profile": selected["profile"],
        "selected_metrics": selected["metrics"],
    }
    _atomic_write_json(attempt_dir / "feature_screen_summary.json", summary)
    _atomic_write_json(
        progress_path,
        {"status": "completed", "completed_at": _now(), **summary},
    )
    _atomic_write_json(
        output_root / "feature_screen/current.json",
        {
            "attempt": str(attempt_dir.resolve()),
            "feature_freeze": str(freeze_path.resolve()),
            "study_contract_sha256": study["contract_sha256"],
            "updated_at": _now(),
        },
    )
    return summary


def build_view(
    *,
    study_path: Path = DEFAULT_STUDY_PATH,
    output_root: Path = DEFAULT_OUTPUT_ROOT,
) -> dict[str, Any]:
    study = load_study(study_path)
    pack_path, manifest = _validate_source_bindings(study)
    protection_before = _verify_protected_bindings(study)
    target_manifest_path, target_manifest = _load_current_artifact(
        output_root,
        task="atlas",
        pointer_name="target_manifest",
    )
    if str(target_manifest.get("status", "")) != "completed":
        raise ValueError("feature construction requires a frozen valid target")
    if str(target_manifest.get("study_contract_sha256", "")) != str(
        study["contract_sha256"]
    ):
        raise ValueError("target artifact belongs to a different study contract")
    candidate_path = _resolve_path(str(manifest["candidate_index_path"]))
    candidate_count = int(pq.ParquetFile(candidate_path).metadata.num_rows)
    if int(target_manifest.get("candidate_count", -1)) != candidate_count:
        raise ValueError("target and candidate index row counts differ")

    attempt_dir = _next_attempt_dir(output_root, "feature_view")
    progress_path = attempt_dir / "progress.json"
    partial = attempt_dir / "partial"
    partial.mkdir(parents=True, exist_ok=False)
    paths = _feature_view_paths(partial)
    continuous_catalog = continuous_feature_catalog()
    categorical_catalog = categorical_feature_catalog()
    _atomic_write_json(
        progress_path,
        {
            "status": "initializing",
            "started_at": _now(),
            "study_contract_sha256": study["contract_sha256"],
            "target_manifest": str(target_manifest_path),
            "candidate_count": candidate_count,
            "continuous_feature_count": len(continuous_catalog),
            "categorical_feature_count": len(categorical_catalog),
        },
    )
    continuous: np.memmap | None = None
    categorical: np.memmap | None = None
    try:
        candidate_date_idx, candidate_symbol_idx = _write_candidate_coordinates(
            candidate_path,
            paths,
            candidate_count=candidate_count,
        )
        date_values = np.asarray(
            [str(item) for item in list(manifest.get("date_values", []) or [])],
            dtype=object,
        )
        candidate_dates = date_values[np.asarray(candidate_date_idx, dtype=np.int64)]
        candidate_allowed = candidate_dates <= "2025-12-31"
        if bool(np.any(candidate_dates >= "2026-01-01")) and bool(
            candidate_allowed[candidate_dates >= "2026-01-01"].any()
        ):
            raise AssertionError("2026 feature exclusion mask failed")
        continuous = np.memmap(
            paths.continuous,
            dtype="float32",
            mode="w+",
            shape=(candidate_count, len(continuous_catalog)),
        )
        categorical = np.memmap(
            paths.categorical,
            dtype="int64",
            mode="w+",
            shape=(candidate_count, len(categorical_catalog)),
        )
        writer = _FeatureWriter(
            continuous=continuous,
            categorical=categorical,
            continuous_catalog=continuous_catalog,
            categorical_catalog=categorical_catalog,
            candidate_date_idx=candidate_date_idx,
            candidate_symbol_idx=candidate_symbol_idx,
            candidate_allowed=candidate_allowed,
        )
        _atomic_write_json(
            progress_path,
            {
                "status": "building_f1_f2",
                "updated_at": _now(),
                "candidate_count": candidate_count,
            },
        )
        f1_f2 = _build_f1_f2_features(writer, manifest)
        _atomic_write_json(
            progress_path,
            {"status": "loading_pit_auxiliary_domains", "updated_at": _now()},
        )
        index_membership = _load_index_membership(study, manifest)
        industry, industry_broad, industry_source_age = _load_industry_panels(
            study, manifest
        )
        writer.categorical_dense("industry_hash", industry)
        writer.categorical_dense("industry_broad_hash", industry_broad)
        _atomic_write_json(
            progress_path,
            {"status": "building_f3", "updated_at": _now()},
        )
        f3 = _build_f3_features(
            writer,
            manifest,
            index_membership=index_membership,
        )
        _atomic_write_json(
            progress_path,
            {"status": "building_f4", "updated_at": _now()},
        )
        f4 = _industry_candidate_features(
            writer=writer,
            manifest=manifest,
            industry=industry,
            industry_source_age=industry_source_age,
        )
        _atomic_write_json(
            progress_path,
            {"status": "building_f5", "updated_at": _now()},
        )
        f5 = _build_f5_features(
            writer,
            study,
            manifest,
            index_membership=index_membership,
            industry=industry,
        )
        writer.assert_complete()
        continuous.flush()
        categorical.flush()
        candidate_date_idx.flush()
        candidate_symbol_idx.flush()
        del writer, continuous, categorical
        continuous = None
        categorical = None
        del index_membership, industry, industry_broad, industry_source_age
        _trim_after_feature_stage()

        final_paths = _feature_view_paths(attempt_dir)
        for source, destination in (
            (paths.continuous, final_paths.continuous),
            (paths.categorical, final_paths.categorical),
            (paths.candidate_date_idx, final_paths.candidate_date_idx),
            (paths.candidate_symbol_idx, final_paths.candidate_symbol_idx),
        ):
            os.replace(source, destination)
        partial.rmdir()
        target_with_path = dict(target_manifest)
        target_with_path["manifest_path"] = str(target_manifest_path.resolve())
        fold_views = _build_fold_views(
            output_dir=attempt_dir,
            study=study,
            manifest=manifest,
            target_manifest=target_with_path,
            candidate_date_idx=np.memmap(
                final_paths.candidate_date_idx,
                dtype="int32",
                mode="r",
                shape=(candidate_count,),
            ),
        )
        files = {}
        for name, path in (
            ("continuous", final_paths.continuous),
            ("categorical", final_paths.categorical),
            ("candidate_date_idx", final_paths.candidate_date_idx),
            ("candidate_symbol_idx", final_paths.candidate_symbol_idx),
        ):
            files[name] = {
                "path": str(path.resolve()),
                "size": int(path.stat().st_size),
                "sha256": _file_sha256(path),
            }
        protection_after = _verify_protected_bindings(study)
        if protection_after != protection_before:
            raise AssertionError("protected binding hashes changed during build-view")
        feature_manifest = {
            "artifact_type": "seq100_path_relevance_feature_view",
            "schema_version": 1,
            "status": "completed",
            "created_at": _now(),
            "study_contract": str(study_path.resolve()),
            "study_contract_sha256": study["contract_sha256"],
            "feature_spec": dict(study["contract"]["features"]),
            "feature_spec_sha256": study["contract"]["feature_spec_sha256"],
            "source_pack_manifest": str(pack_path.resolve()),
            "source_pack_manifest_sha256": _file_sha256(pack_path),
            "source_candidate_index": str(candidate_path.resolve()),
            "source_candidate_index_sha256": _file_sha256(candidate_path),
            "target_manifest": str(target_manifest_path.resolve()),
            "target_manifest_sha256": _file_sha256(target_manifest_path),
            "candidate_count": candidate_count,
            "feature_candidate_count_through_2025": int(candidate_allowed.sum()),
            "post_2025_rows_forced_missing": int((~candidate_allowed).sum()),
            "continuous_shape": [candidate_count, len(continuous_catalog)],
            "categorical_shape": [candidate_count, len(categorical_catalog)],
            "continuous_catalog": continuous_catalog,
            "categorical_catalog": categorical_catalog,
            "families": {
                "F0": {
                    "source": "protected_pack_180x35",
                    "normalization": "existing_fold_normalization",
                },
                "F1_F2": f1_f2,
                "F3": f3,
                "F4": f4,
                "F5": f5,
            },
            "cross_section_membership": "full_candidate_index_signal_day_known_eligibility_only",
            "entry_buyable_used_for_features": False,
            "future_labels_used_for_features": False,
            "fold_views": fold_views,
            "files": files,
            "protection_before": protection_before,
            "protection_after": protection_after,
        }
        feature_manifest["resolved_feature_view_sha256"] = _canonical_json_sha256(
            feature_manifest
        )
        _atomic_write_json(final_paths.manifest, feature_manifest)
        summary = {
            "status": "completed",
            "output_dir": str(attempt_dir.resolve()),
            "feature_view_manifest": str(final_paths.manifest.resolve()),
            "resolved_feature_view_sha256": feature_manifest[
                "resolved_feature_view_sha256"
            ],
            "candidate_count": candidate_count,
            "continuous_feature_count": len(continuous_catalog),
            "categorical_feature_count": len(categorical_catalog),
            "fold_views": fold_views,
        }
        _atomic_write_json(attempt_dir / "build_view_summary.json", summary)
        _atomic_write_json(
            progress_path,
            {
                "status": "completed",
                "completed_at": _now(),
                "feature_view_manifest": str(final_paths.manifest.resolve()),
            },
        )
        _atomic_write_json(
            output_root / "feature_view/current.json",
            {
                "attempt": str(attempt_dir.resolve()),
                "feature_view_manifest": str(final_paths.manifest.resolve()),
                "study_contract_sha256": study["contract_sha256"],
                "updated_at": _now(),
            },
        )
        return summary
    except Exception as exc:
        if continuous is not None:
            continuous.flush()
        if categorical is not None:
            categorical.flush()
        _atomic_write_json(
            progress_path,
            {
                "status": "failed",
                "failed_at": _now(),
                "error_type": type(exc).__name__,
                "error": str(exc),
            },
        )
        raise


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build and evaluate Seq100 PIT path-relevance research artifacts."
    )
    parser.add_argument("--study-contract", type=Path, default=DEFAULT_STUDY_PATH)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    sub = parser.add_subparsers(dest="command", required=True)

    atlas = sub.add_parser("build-atlas")
    atlas.add_argument("--pareto-device", choices=("auto", "cpu", "cuda"), default="auto")

    sub.add_parser("build-view")
    sub.add_parser("feature-screen")
    train = sub.add_parser("train")
    train.add_argument("--model-id", choices=MODEL_IDS, required=True)
    train.add_argument("--fold-year", type=int, choices=FORMAL_FOLD_YEARS, required=True)
    sub.add_parser("evaluate")
    sub.add_parser("closeout")
    parser.add_argument("--json", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> dict[str, Any]:
    args = build_parser().parse_args(argv)
    if args.command == "build-atlas":
        result = build_atlas(
            study_path=args.study_contract,
            output_root=args.output_root,
            pareto_device=str(args.pareto_device),
        )
    elif args.command == "build-view":
        result = build_view(
            study_path=args.study_contract,
            output_root=args.output_root,
        )
    elif args.command == "feature-screen":
        result = feature_screen(
            study_path=args.study_contract,
            output_root=args.output_root,
        )
    else:
        raise SystemExit(f"{args.command} is not implemented yet")
    if args.json:
        print(json.dumps(result, ensure_ascii=False, indent=2, default=_json_default))
    else:
        print(result.get("output_dir", ""))
    return result


if __name__ == "__main__":
    main()
