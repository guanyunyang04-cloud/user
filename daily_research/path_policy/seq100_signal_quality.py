from __future__ import annotations

import argparse
import gc
import hashlib
import heapq
import json
import math
import os
import platform
import random
import shutil
import subprocess
import threading
import time
from collections import deque
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from queue import Empty, Queue
from typing import Any, Iterable, Iterator, Mapping, Sequence

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from scipy import stats
import torch
import torch.nn as nn
import torch.nn.functional as torch_functional

from daily_research.path_policy import qdp_v2_sequence_path_training as sequence_training


WORKSPACE_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_STUDY_PATH = WORKSPACE_ROOT / "daily_research/studies/seq100_pit_signal_quality_v1.json"
DEFAULT_OUTPUT_ROOT = (
    WORKSPACE_ROOT
    / "daily_research/output/path_policy/studies/seq100_pit_signal_quality_v1"
)

STUDY_ID = "seq100_pit_signal_quality_v1"
TARGET_CANDIDATE_IDS = (
    "raw_path_distribution_v1",
    "pareto_ordinal_v1",
    "competing_risk_path_v1",
    "direct_listwise_utility_v1",
)
PATH_TARGET_PROFILE = "pareto_ordinal_v1"
TREND_FALLBACK_PROFILE = "trend_consistency_diagnostic_v1"
TARGET_ARTIFACT_TYPE = "seq100_signal_quality_target_view"
TARGET_FLAG_ENTRY_FILLED = 1 << 0
TARGET_FLAG_PATH_AVAILABLE = 1 << 1
TARGET_FLAG_FORCED_LOW = 1 << 2
TARGET_FLAG_TERMINAL_FAILURE = 1 << 3
TARGET_FLAG_NO_LEGAL_SELL = 1 << 4
TARGET_FLAG_PRICE_LABEL_VALID = 1 << 5

EVENT_CENSORED = 0
EVENT_TERMINAL_FAILURE = 1
EVENT_MAJOR_DRAWDOWN = 2
EVENT_POST_PEAK_FADE = 3
EVENT_NO_LEGAL_SELL = 4
EVENT_SUSTAINED_UPSIDE = 5
EVENT_NAMES = (
    "censored",
    "terminal_failure",
    "major_drawdown",
    "post_peak_fade",
    "no_legal_sell",
    "sustained_upside",
)

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
TARGET_FIELD_INDEX = {name: idx for idx, name in enumerate(TARGET_FLOAT_FIELDS)}

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
MODEL_EXECUTION_SEMANTICS = {
    "tabm_multioutput": "exact_effective_batch_global_loss_v2",
    "patchtst_student_t_path": "exact_effective_batch_global_loss_v2",
    "market_industry_deepsets": "complete_date_atomic_set_v2",
    "deephit_competing_risk": "exact_effective_batch_global_ranking_v2",
    "neuralndcg_listwise_mlp": "deterministic_complete_list_v2",
}
RUNTIME_AUTOTUNE_VERSION = "seq100_cuda_runtime_autotune_v1"
TRAINING_CHECKPOINT_VERSION = "seq100_resumable_training_v1"
TRAINING_PROGRESS_VERSION = "seq100_training_progress_v1"
TRAINING_BUDGET_AMENDMENT_VERSION = "seq100_training_budget_amendment_v1"
TRAINING_BUDGET_AMENDMENT_FILENAME = "training_budget_amendment.json"
TRAINING_BUDGET_MUTABLE_FIELDS = ("max_epochs", "patience")
PROGRESS_HEARTBEAT_SECONDS = 30.0
CONSOLE_EVENT_SECONDS = 300.0
CHECKPOINT_INTERVAL_SECONDS = 600.0
# Keep this strictly below the ordinary trim trigger so that trim gets a chance first.
LOW_MEMORY_PAUSE_AVAILABLE_GB = 1.0
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
    profile: str = "multi_candidate_v1"
    target_candidates: tuple[str, ...] = TARGET_CANDIDATE_IDS
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
    competing_upside_wealth: float = 1.05
    competing_time_above: float = 0.80
    competing_drawdown: float = 0.08
    competing_fade: float = 0.10


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
    if str(payload.get("study_id", "")) != STUDY_ID:
        raise ValueError(f"study_id must be {STUDY_ID}")
    if not isinstance(payload.get("contract"), Mapping):
        raise ValueError("study is missing contract")
    declared = str(payload.get("contract_sha256", "") or "").lower()
    computed = _canonical_json_sha256(payload["contract"])
    if declared != computed:
        raise ValueError(
            f"study contract SHA-256 mismatch: declared={declared} computed={computed}"
        )
    if str(payload["contract"].get("contract_id", "")) != STUDY_ID:
        raise ValueError(f"contract_id must be {STUDY_ID}")
    load_research_freeze(payload)
    return payload


def load_research_freeze(study: Mapping[str, Any]) -> dict[str, Any]:
    binding = dict(study["contract"].get("research_freeze", {}) or {})
    freeze_path = _resolve_path(str(binding.get("path", "")))
    if not freeze_path.is_file():
        raise FileNotFoundError(f"research freeze does not exist: {freeze_path}")
    actual_file_sha = _file_sha256(freeze_path)
    declared_file_sha = str(binding.get("file_sha256", "") or "").lower()
    if actual_file_sha != declared_file_sha:
        raise ValueError(
            "research freeze file changed: "
            f"declared={declared_file_sha} actual={actual_file_sha}"
        )
    payload = json.loads(freeze_path.read_text(encoding="utf-8"))
    if str(payload.get("study_id", "")) != STUDY_ID:
        raise ValueError("research freeze study_id mismatch")
    freeze = payload.get("freeze")
    if not isinstance(freeze, Mapping):
        raise ValueError("research freeze is missing freeze payload")
    computed = _canonical_json_sha256(freeze)
    declared = str(payload.get("freeze_sha256", "") or "").lower()
    bound = str(binding.get("freeze_sha256", "") or "").lower()
    if computed != declared or computed != bound:
        raise ValueError(
            "research freeze SHA-256 mismatch: "
            f"computed={computed} declared={declared} bound={bound}"
        )
    targets = list(freeze.get("target_candidates", []) or [])
    models = list(freeze.get("model_candidates", []) or [])
    target_ids = tuple(str(item.get("target_id", "")) for item in targets)
    if target_ids != TARGET_CANDIDATE_IDS:
        raise ValueError(f"target candidate order mismatch: {target_ids}")
    model_ids = tuple(str(item.get("model_id", "")) for item in models)
    if model_ids != MODEL_IDS:
        raise ValueError(f"model candidate order mismatch: {model_ids}")
    forbidden_years = set(
        int(item)
        for item in list(
            dict(freeze.get("scientific_firewall", {}) or {}).get("forbidden_years", [])
            or []
        )
    )
    if 2026 not in forbidden_years:
        raise ValueError("research freeze must explicitly forbid 2026")
    return payload


def _training_budget_amendment_path(output_root: Path) -> Path:
    return output_root / "model_screen" / TRAINING_BUDGET_AMENDMENT_FILENAME


def _load_training_budget_amendment(
    study: Mapping[str, Any],
    output_root: Path,
    *,
    research_freeze: Mapping[str, Any] | None = None,
) -> dict[str, Any] | None:
    path = _training_budget_amendment_path(output_root)
    if not path.exists():
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    if str(payload.get("schema", "")) != TRAINING_BUDGET_AMENDMENT_VERSION:
        raise ValueError("training budget amendment schema mismatch")
    if str(payload.get("study_id", "")) != STUDY_ID:
        raise ValueError("training budget amendment study_id mismatch")
    if str(payload.get("status", "")) != "frozen":
        raise ValueError("training budget amendment is not frozen")
    if str(payload.get("base_study_contract_sha256", "")) != str(
        study["contract_sha256"]
    ):
        raise ValueError("training budget amendment belongs to a different contract")
    declared = str(payload.get("amendment_sha256", ""))
    check = dict(payload)
    check.pop("amendment_sha256", None)
    if declared != _canonical_json_sha256(check):
        raise ValueError("training budget amendment hash mismatch")
    model_ids = tuple(str(item) for item in payload.get("model_ids", []))
    if model_ids != NEURAL_MODEL_IDS:
        raise ValueError("training budget amendment must apply to every neural model")
    changes = dict(payload.get("config_changes", {}) or {})
    if set(changes) != set(TRAINING_BUDGET_MUTABLE_FIELDS):
        raise ValueError("training budget amendment changes unsupported fields")
    expected_changes = {
        "max_epochs": {"from": 3, "to": 10},
        "patience": {"from": 1, "to": 2},
    }
    if changes != expected_changes:
        raise ValueError("training budget amendment values do not match the decision")
    freeze = dict(
        research_freeze
        if research_freeze is not None
        else load_research_freeze(study)["freeze"]
    )
    specs = {
        str(item["model_id"]): dict(item)
        for item in list(freeze.get("model_candidates", []) or [])
    }
    for model_id in model_ids:
        config = dict(specs[model_id].get("config", {}) or {})
        for field, change in changes.items():
            if config.get(field) != change["from"]:
                raise ValueError(
                    f"training budget amendment base mismatch for {model_id}.{field}"
                )
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
        raise ValueError("signal-quality study requires at least 60 future path days")
    if int(manifest.get("lookback_days", 0) or 0) != 180:
        raise ValueError("signal-quality study requires the frozen 180-day PIT pack")
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
    for key in ("candidate_index", "sample_index"):
        index_path = _resolve_path(str(data.get(key, "")))
        declared = str(data.get(f"{key}_sha256", "") or "").lower()
        actual = _file_sha256(index_path)
        if actual != declared:
            raise ValueError(
                f"protected {key} changed: declared={declared} actual={actual}"
            )
    active_path = _resolve_path(str(data.get("qdp_active_manifest", "")))
    if _file_sha256(active_path) != str(data.get("qdp_active_manifest_sha256", "")):
        raise ValueError("QDP active manifest changed")
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
        out[positions[order]] = np.arange(
            positions.size, dtype=np.float32
        ) / float(positions.size - 1)
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
    freeze = dict(load_research_freeze(study)["freeze"])
    firewall = dict(freeze.get("scientific_firewall", {}) or {})
    purge = int(firewall["purge_trading_days"])
    source_root = _resolve_path(str(manifest["sample_index_path"])).parent.parent / "folds"
    fold_records: dict[str, Any] = {}
    registered_folds: list[tuple[str, int, int, Sequence[int]]] = []
    screen = dict(firewall.get("model_prescreen", {}) or {})
    registered_folds.append(
        (
            "screen",
            int(screen["development_year"]),
            int(screen["test_year"]),
            tuple(int(item) for item in screen["train_years"]),
        )
    )
    formal = dict(firewall.get("formal_folds", {}) or {})
    if tuple(int(item) for item in formal) != FORMAL_FOLD_YEARS:
        raise ValueError("research freeze formal fold years changed")
    for test_year in FORMAL_FOLD_YEARS:
        current = dict(formal[str(test_year)])
        registered_folds.append(
            (
                str(test_year),
                int(current["development_year"]),
                int(current["test_year"]),
                tuple(int(item) for item in current["train_years"]),
            )
        )
    for fold_id, dev_year, test_year, train_years in registered_folds:
        if int(test_year) != int(dev_year) + 1:
            raise ValueError(f"fold {fold_id} development/test years are not adjacent")
        if tuple(train_years) != tuple(range(2010, int(dev_year))):
            raise ValueError(f"fold {fold_id} train years are not the frozen expanding window")
        dev_dates = _date_indices_for_year(dates, dev_year)
        test_dates = _date_indices_for_year(dates, test_year)
        if not dev_dates.size or not test_dates.size:
            raise ValueError(f"pack lacks dates for fold {fold_id}")
        dev_start = int(dev_dates.min())
        test_start = int(test_dates.min())
        train_end = dev_start - purge - 1
        dev_end = test_start - purge - 1
        train_start = int(candidate_date_idx.min())
        if train_end < train_start or dev_end < dev_start:
            raise ValueError(f"fold {fold_id} purge leaves an empty train/development split")
        if int(dates[train_end][:4]) > max(train_years):
            raise ValueError(f"fold {fold_id} train range crosses its frozen year boundary")
        if int(dates[dev_end][:4]) > int(dev_year):
            raise ValueError(f"fold {fold_id} development range crosses its frozen year boundary")
        test_end = int(test_dates.max())
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
            raise ValueError(f"fold {fold_id} has no D20-complete test dates")
        test_end = int(max(allowed))
        original_index = source_root / "indexes" / f"development_{dev_year}_purge60.parquet"
        original_candidates = source_root / "indexes" / f"candidates_{test_year}.parquet"
        if not original_index.exists() or not original_candidates.exists():
            raise FileNotFoundError(
                f"immutable source fold indexes are missing for fold {fold_id}"
            )
        f0_normalization_view = source_root / "views" / f"l35v2_pit_{dev_year}.json"
        if not f0_normalization_view.exists():
            raise FileNotFoundError(f0_normalization_view)
        record = {
            "artifact_type": "seq100_signal_quality_fold_view",
            "study_contract_sha256": study["contract_sha256"],
            "target_manifest_sha256": _file_sha256(Path(str(target_manifest["manifest_path"]))),
            "fold_id": str(fold_id),
            "fold_year": int(test_year),
            "train_years": list(train_years),
            "development_year": int(dev_year),
            "test_year": int(test_year),
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
        path = folds_dir / f"fold_{fold_id}.json"
        _atomic_write_json(path, record)
        fold_records[str(fold_id)] = {
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
        "common_sustained_action_utility",
        "raw_path_distribution_quality",
        "pareto_ordinal_quality",
        "competing_risk_quality",
        "direct_listwise_utility_quality",
        "competing_event_code",
        "competing_event_time",
        "competing_event_score",
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
    field_by_target = {
        "raw_path_distribution_v1": "raw_path_distribution_quality",
        "pareto_ordinal_v1": "pareto_ordinal_quality",
        "competing_risk_path_v1": "competing_risk_quality",
        "direct_listwise_utility_v1": "direct_listwise_utility_quality",
    }
    if str(frozen_profile) not in field_by_target:
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
    source_idx = TARGET_FIELD_INDEX[field_by_target[str(frozen_profile)]]
    conditional_idx = TARGET_FIELD_INDEX["conditional_relevance"]
    action_idx = TARGET_FIELD_INDEX["action_relevance"]
    for start in range(0, int(candidate_count), int(chunk_size)):
        stop = min(start + int(chunk_size), int(candidate_count))
        quality = np.asarray(target[start:stop, source_idx], dtype=np.float32)
        filled = (np.asarray(flags[start:stop], dtype=np.uint8) & TARGET_FLAG_ENTRY_FILLED) != 0
        conditional = np.full(stop - start, np.nan, dtype=np.float32)
        conditional[filled] = quality[filled]
        target[start:stop, conditional_idx] = conditional
        action = np.zeros(stop - start, dtype=np.float32)
        action[filled] = quality[filled]
        target[start:stop, action_idx] = action
        grades[start:stop] = relevance_grades(action)
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


def validate_research(
    *,
    study_path: Path = DEFAULT_STUDY_PATH,
    output_root: Path = DEFAULT_OUTPUT_ROOT,
) -> dict[str, Any]:
    study = load_study(study_path)
    freeze_payload = load_research_freeze(study)
    freeze = dict(freeze_payload["freeze"])
    pack_path, manifest = _validate_source_bindings(study)
    protection_before = _verify_protected_bindings(study)
    evidence = dict(freeze.get("evidence_bindings", {}) or {})
    for name, binding in evidence.items():
        path = _resolve_path(str(binding.get("path", "")))
        actual = _file_sha256(path)
        if actual != str(binding.get("sha256", "")):
            raise ValueError(f"research evidence changed for {name}")
    review_path = _resolve_path(str(evidence["external_review"]["path"]))
    review = json.loads(review_path.read_text(encoding="utf-8"))
    coverage = dict(review.get("coverage", {}) or {})
    if int(coverage.get("source_count", 0)) < 24:
        raise ValueError("external review has fewer than 24 primary sources")
    if int(coverage.get("original_paper_or_book_count", 0)) < 12:
        raise ValueError("external review has fewer than 12 original papers/books")
    if int(coverage.get("official_or_author_repository_count", 0)) < 8:
        raise ValueError("external review has fewer than 8 official repositories")
    if len(list(coverage.get("method_families", []) or [])) < 8:
        raise ValueError("external review has fewer than 8 method families")
    if tuple(
        str(item.get("target_id", ""))
        for item in list(freeze.get("target_candidates", []) or [])
    ) != TARGET_CANDIDATE_IDS:
        raise ValueError("research target shortlist changed")
    if tuple(
        str(item.get("model_id", ""))
        for item in list(freeze.get("model_candidates", []) or [])
    ) != MODEL_IDS:
        raise ValueError("research model shortlist changed")
    if len(dict(study["contract"]["data"].get("qdp_datasets", {}) or {})) != 14:
        raise ValueError("study contract must bind all 14 active QDP datasets")
    if any(str(value).startswith("2026") for value in manifest.get("date_values", [])):
        # Coordinates may exist in the pack, but no result stage may authorize them.
        forbidden = set(
            int(item)
            for item in dict(freeze.get("scientific_firewall", {}) or {}).get(
                "forbidden_years", []
            )
        )
        if 2026 not in forbidden:
            raise ValueError("2026 pack coordinates are not covered by a hard firewall")
    attempt_dir = _next_attempt_dir(output_root, "validate_research")
    protection_after = _verify_protected_bindings(study)
    if protection_after != protection_before:
        raise RuntimeError("protected-object hashes changed during research validation")
    summary = {
        "status": "validated",
        "study_id": STUDY_ID,
        "output_dir": str(attempt_dir.resolve()),
        "study_contract_sha256": study["contract_sha256"],
        "research_freeze_sha256": freeze_payload["freeze_sha256"],
        "pack_manifest": str(pack_path.resolve()),
        "coverage": coverage,
        "target_candidate_ids": list(TARGET_CANDIDATE_IDS),
        "model_candidate_ids": list(MODEL_IDS),
        "protection_before": protection_before,
        "protection_after": protection_after,
    }
    _atomic_write_json(attempt_dir / "validation_summary.json", summary)
    _atomic_write_json(
        output_root / "validate_research/current.json",
        {
            "attempt": str(attempt_dir.resolve()),
            "summary": str((attempt_dir / "validation_summary.json").resolve()),
            "study_contract_sha256": study["contract_sha256"],
            "updated_at": _now(),
        },
    )
    return summary


def _append_target_candidate_diagnostic(
    rows: list[dict[str, Any]],
    *,
    target_id: str,
    group: pd.DataFrame,
    descriptors: Mapping[str, np.ndarray],
    qualities: Mapping[str, np.ndarray],
    entry_filled: np.ndarray,
    terminal_failure: np.ndarray,
) -> None:
    filled = np.asarray(entry_filled, dtype=bool)
    terminal = np.asarray(terminal_failure, dtype=bool)
    field = {
        "raw_path_distribution_v1": "raw_path_distribution_quality",
        "pareto_ordinal_v1": "pareto_ordinal_quality",
        "competing_risk_path_v1": "competing_risk_quality",
        "direct_listwise_utility_v1": "direct_listwise_utility_quality",
    }[str(target_id)]
    conditional = np.asarray(qualities[field], dtype=np.float64)
    if str(target_id) == "direct_listwise_utility_v1":
        action_quality = conditional.copy()
    else:
        action_quality = np.full(conditional.shape, np.nan, dtype=np.float64)
        action_quality[~filled] = 0.0
        action_quality[filled] = conditional[filled]
    grade = relevance_grades(
        np.where(filled, conditional, np.nan)
    )
    utility = np.asarray(
        qualities["common_sustained_action_utility"], dtype=np.float64
    )
    r20 = np.asarray(descriptors["r20_net"], dtype=np.float64)
    mdd = np.asarray(descriptors["mdd20"], dtype=np.float64)
    fade = np.asarray(descriptors["post_peak_fade20"], dtype=np.float64)
    action_r20 = np.where(filled, r20, 0.0)
    action_mdd = np.where(filled, mdd, 0.0)
    action_fade = np.where(filled, fade, 0.0)
    action_terminal = np.where(filled, terminal.astype(np.float64), 0.0)
    valid = (
        np.isfinite(action_quality)
        & np.isfinite(utility)
        & np.isfinite(action_r20)
        & np.isfinite(action_mdd)
        & np.isfinite(action_fade)
    )
    record: dict[str, Any] = {
        "target_id": str(target_id),
        "trade_date": str(group["trade_date"].iloc[0]),
        "year": int(group["year"].iloc[0]),
        "date_idx": int(group["date_idx"].iloc[0]),
        "candidate_count": int(len(group)),
        "label_count": int(valid.sum()),
    }
    positions = np.flatnonzero(valid)
    top_count = min(
        int(positions.size),
        max(1, int(math.ceil(len(group) * 0.01))),
    )
    if top_count:
        order = positions[
            np.lexsort(
                (
                    group["symbol_idx"].to_numpy(dtype=np.int64, copy=False)[positions],
                    -action_quality[positions],
                )
            )
        ]
        top = order[:top_count]
        record.update(
            {
                "top1pct_count": int(top_count),
                "top1pct_u_mean": float(np.mean(utility[top])),
                "top1pct_d20_median": float(np.median(action_r20[top])),
                "top1pct_negative_d20_rate": float(np.mean(action_r20[top] < 0.0)),
                "top1pct_mdd_median": float(np.median(action_mdd[top])),
                "top1pct_fade_median": float(np.median(action_fade[top])),
                "top1pct_fade_gt10_rate": float(np.mean(action_fade[top] > 0.10)),
                "top1pct_terminal_failure_rate": float(
                    np.mean(action_terminal[top] > 0.5)
                ),
            }
        )
    else:
        record.update(
            {
                "top1pct_count": 0,
                "top1pct_u_mean": np.nan,
                "top1pct_d20_median": np.nan,
                "top1pct_negative_d20_rate": np.nan,
                "top1pct_mdd_median": np.nan,
                "top1pct_fade_median": np.nan,
                "top1pct_fade_gt10_rate": np.nan,
                "top1pct_terminal_failure_rate": np.nan,
            }
        )
    for current_grade in range(5):
        mask = valid & (grade == current_grade)
        record[f"grade{current_grade}_count"] = int(mask.sum())
        for name, values in (
            ("u", utility),
            ("d20", action_r20),
            ("mdd", action_mdd),
            ("fade", action_fade),
            ("terminal_failure", action_terminal),
        ):
            record[f"grade{current_grade}_{name}"] = (
                float(np.median(values[mask]))
                if name != "terminal_failure" and bool(mask.any())
                else (
                    float(np.mean(values[mask]))
                    if name == "terminal_failure" and bool(mask.any())
                    else np.nan
                )
            )
    rows.append(record)


def build_targets(
    *,
    study_path: Path = DEFAULT_STUDY_PATH,
    output_root: Path = DEFAULT_OUTPUT_ROOT,
    pareto_device: str = "auto",
) -> dict[str, Any]:
    study = load_study(study_path)
    freeze_payload = load_research_freeze(study)
    freeze = dict(freeze_payload["freeze"])
    pack_path, manifest = _validate_source_bindings(study)
    target_candidates = list(freeze.get("target_candidates", []) or [])
    competing = next(
        item
        for item in target_candidates
        if str(item.get("target_id", "")) == "competing_risk_path_v1"
    )
    competing_parameters = dict(competing.get("free_parameters", {}) or {})
    spec = PathTargetSpec(
        competing_upside_wealth=float(competing_parameters.get("upside_wealth", 1.05)),
        competing_time_above=float(competing_parameters.get("time_above", 0.80)),
        competing_drawdown=float(competing_parameters.get("drawdown", 0.08)),
        competing_fade=float(competing_parameters.get("fade", 0.10)),
    )
    if tuple(spec.target_candidates) != TARGET_CANDIDATE_IDS:
        raise ValueError("target candidate implementation order changed")
    attempt_dir = _next_attempt_dir(output_root, "targets")
    progress_path = attempt_dir / "progress.json"
    protection_before = _verify_protected_bindings(study)
    _atomic_write_json(
        progress_path,
        {
            "status": "initializing",
            "started_at": _now(),
            "study_contract_sha256": study["contract_sha256"],
            "research_freeze_sha256": freeze_payload["freeze_sha256"],
            "pack_manifest": str(pack_path),
            "protection_before": protection_before,
        },
    )

    candidate_path = _resolve_path(str(manifest.get("candidate_index_path", "")))
    candidate_count = int(pq.ParquetFile(candidate_path).metadata.num_rows)
    date_values = [str(item) for item in list(manifest.get("date_values", []) or [])]
    date_count = len(date_values)
    symbol_count = int(manifest.get("symbol_count", 0) or 0)
    data_contract = dict(study["contract"].get("data", {}) or {})
    cutoff = str(data_contract.get("label_endpoint_cutoff", "2025-12-31"))
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

    daily_rows: list[dict[str, Any]] = []
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
            entry_filled = (
                group["entry_filled"]
                .astype("boolean")
                .fillna(False)
                .to_numpy(dtype=bool)
            )
            price_label_valid = (
                group["price_label_valid"]
                .astype("boolean")
                .fillna(False)
                .to_numpy(dtype=bool)
            )
            date_idx_values = np.full(len(group), date_idx, dtype=np.int64)
            source_path = _take_future_ohlc(
                future_reader,
                date_idx_values,
                symbol_idx,
                horizon=60,
            )
            entry_path = reanchor_to_actual_next_open(
                source_path,
                source_price_anchor="today_close",
                required_mask=entry_filled & price_label_valid,
            )
            sellable = _future_panel_view(
                exit_sellable_panel,
                signal_date_idx=date_idx,
                symbol_idx=symbol_idx,
                horizon=60,
            ).astype(bool, copy=False)
            delisted = _future_panel_view(
                delisted_panel,
                signal_date_idx=date_idx,
                symbol_idx=symbol_idx,
                horizon=60,
            ).astype(bool, copy=False)
            entry_path, _terminal_any60 = apply_terminal_zero_recovery(
                entry_path,
                delisted,
            )
            multiplier = np.broadcast_to(
                growth_by_date[date_idx, :60].reshape(1, 60),
                (len(group), 60),
            ).copy()
            terminal_failure20 = delisted[:, :20].any(axis=1)
            descriptors = compute_path_descriptors(
                entry_path[:, :20, :],
                multiplier[:, :20],
                exit_sellable=sellable[:, :20],
                terminal_failure=terminal_failure20,
                log_wealth_floor=spec.log_wealth_floor,
            )
            continuation_growth = (
                np.maximum(1.0 + entry_path[:, :, 3], 0.0) * multiplier
            )
            descriptors["r40_net"] = np.asarray(
                continuation_growth[:, 39] - 1.0, dtype=np.float32
            )
            descriptors["r60_net"] = np.asarray(
                continuation_growth[:, 59] - 1.0, dtype=np.float32
            )
            path_available = np.asarray(descriptors["path_available"], dtype=bool)
            no_legal_sell = np.asarray(descriptors["no_legal_sell"], dtype=bool)
            forced_low = (
                entry_filled
                & path_available
                & (terminal_failure20 | no_legal_sell)
            )
            qualities = compute_target_candidate_qualities(
                descriptors,
                entry_relative_path=entry_path[:, :20, :],
                growth_multiplier=multiplier[:, :20],
                exit_sellable=sellable[:, :20],
                delisted_path=delisted[:, :20],
                entry_filled=entry_filled,
                path_available=path_available,
                forced_low=forced_low,
                symbol_idx=symbol_idx,
                pareto_device=pareto_device,
                pareto_block_size=spec.pareto_block_size,
                competing_upside_wealth=spec.competing_upside_wealth,
                competing_time_above=spec.competing_time_above,
                competing_drawdown=spec.competing_drawdown,
                competing_fade=spec.competing_fade,
            )
            _assign_target_fields(target, candidate_ids, descriptors, qualities)
            counts[candidate_ids, 0] = np.asarray(
                qualities["dominated_count"], dtype=np.int32
            )
            counts[candidate_ids, 1] = np.asarray(
                qualities["dominating_count"], dtype=np.int32
            )
            current_flags = np.zeros(len(group), dtype=np.uint8)
            current_flags |= entry_filled.astype(np.uint8) * TARGET_FLAG_ENTRY_FILLED
            current_flags |= path_available.astype(np.uint8) * TARGET_FLAG_PATH_AVAILABLE
            current_flags |= forced_low.astype(np.uint8) * TARGET_FLAG_FORCED_LOW
            current_flags |= (
                terminal_failure20.astype(np.uint8) * TARGET_FLAG_TERMINAL_FAILURE
            )
            current_flags |= (
                no_legal_sell.astype(np.uint8) * TARGET_FLAG_NO_LEGAL_SELL
            )
            current_flags |= (
                price_label_valid.astype(np.uint8) * TARGET_FLAG_PRICE_LABEL_VALID
            )
            flags[candidate_ids] = current_flags
            for target_id in TARGET_CANDIDATE_IDS:
                _append_target_candidate_diagnostic(
                    daily_rows,
                    target_id=target_id,
                    group=group,
                    descriptors=descriptors,
                    qualities=qualities,
                    entry_filled=entry_filled,
                    terminal_failure=terminal_failure20,
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
    _atomic_write_parquet(attempt_dir / "daily_target_diagnostics.parquet", daily)
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
    protection_after = _verify_protected_bindings(study)
    if protection_after != protection_before:
        raise RuntimeError("protected-object hashes changed while building targets")
    target_spec_payload = {
        "common_trade_semantics": freeze.get("common_trade_semantics"),
        "shared_descriptors": freeze.get("shared_descriptors"),
        "target_candidates": target_candidates,
    }
    target_manifest = {
        "artifact_type": TARGET_ARTIFACT_TYPE,
        "schema_version": 1,
        "created_at": _now(),
        "status": "targets_built",
        "study_contract": str(study_path.resolve()),
        "study_contract_sha256": study["contract_sha256"],
        "research_freeze_sha256": freeze_payload["freeze_sha256"],
        "target_spec": target_spec_payload,
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
        "target_candidate_ids": list(TARGET_CANDIDATE_IDS),
        "frozen_target_id": None,
        "diagnostics": {
            "path": str(
                (attempt_dir / "daily_target_diagnostics.parquet").resolve()
            ),
            "sha256": _file_sha256(
                attempt_dir / "daily_target_diagnostics.parquet"
            ),
        },
        "files": file_metadata,
        "protection_before": protection_before,
        "protection_after": protection_after,
    }
    target_manifest["target_artifact_sha256"] = _canonical_json_sha256(
        target_manifest
    )
    _atomic_write_json(paths.manifest, target_manifest)
    summary = {
        "status": "targets_built",
        "output_dir": str(attempt_dir.resolve()),
        "target_manifest": str(paths.manifest.resolve()),
        "target_artifact_sha256": target_manifest["target_artifact_sha256"],
        "processed_candidate_count": processed_candidates,
        "skipped_after_cutoff": skipped_after_cutoff,
    }
    _atomic_write_json(attempt_dir / "build_targets_summary.json", summary)
    _atomic_write_json(
        progress_path,
        {
            "status": "targets_built",
            "completed_at": _now(),
            "target_manifest": str(paths.manifest.resolve()),
        },
    )
    _atomic_write_json(
        output_root / "targets/current.json",
        {
            "attempt": str(attempt_dir.resolve()),
            "target_manifest": str(paths.manifest.resolve()),
            "study_contract_sha256": study["contract_sha256"],
            "updated_at": _now(),
        },
    )
    return summary


def _reference_complete_dates(
    candidate_path: Path,
    *,
    years: Sequence[int],
    max_rows: int,
    seed: int = 1729,
) -> set[int]:
    requested = {int(year) for year in years}
    records: list[tuple[int, int, int]] = []
    for group in _iter_date_groups(
        candidate_path,
        columns=("trade_date", "year", "date_idx"),
    ):
        year = int(group["year"].iloc[0])
        if year not in requested:
            continue
        date_idx = int(group["date_idx"].iloc[0])
        digest = hashlib.blake2b(
            f"{int(seed)}:{date_idx}".encode("utf-8"), digest_size=8
        ).digest()
        priority = int.from_bytes(digest, byteorder="big", signed=False)
        records.append((priority, date_idx, int(len(group))))
    if int(max_rows) <= 0:
        return {date_idx for _priority, date_idx, _count in records}
    selected: set[int] = set()
    used = 0
    for _priority, date_idx, count in sorted(records):
        if selected and used + int(count) > int(max_rows):
            continue
        selected.add(int(date_idx))
        used += int(count)
        if used >= int(max_rows):
            break
    if not selected and records:
        selected.add(int(min(records)[1]))
    return selected


def _reference_candidate_rows(
    candidate_path: Path,
    *,
    date_indices: set[int],
) -> pd.DataFrame:
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
    groups: list[pd.DataFrame] = []
    for group in _iter_date_groups(candidate_path, columns=columns):
        if int(group["date_idx"].iloc[0]) in date_indices:
            groups.append(group)
    if not groups:
        return pd.DataFrame(columns=list(columns))
    return pd.concat(groups, ignore_index=True)


def _reference_snapshot_features(
    manifest: Mapping[str, Any],
    rows: pd.DataFrame,
) -> tuple[np.ndarray, np.ndarray]:
    channels = dict(manifest.get("feature_channels", {}) or {})
    date_idx = rows["date_idx"].to_numpy(dtype=np.int64, copy=False)
    symbol_idx = rows["symbol_idx"].to_numpy(dtype=np.int64, copy=False)
    parts: list[np.ndarray] = []
    momentum20: np.ndarray | None = None
    for name in ("daily_raw", "daily_state", "turnover"):
        metadata = dict(channels[name])
        values = sequence_training._open_memmap(metadata, dtype="float32")
        current = np.asarray(values[date_idx, symbol_idx, :], dtype=np.float32)
        parts.append(current)
        if name == "daily_state":
            columns = list(metadata.get("columns", []) or [])
            momentum20 = current[:, columns.index("ret_20d")].astype(
                np.float32, copy=True
            )
    masks = dict(manifest.get("masks", {}) or {})
    if "turnover_valid" in masks:
        panel = sequence_training._open_memmap(masks["turnover_valid"], dtype="bool")
        parts.append(
            np.asarray(panel[date_idx, symbol_idx], dtype=np.float32).reshape(-1, 1)
        )
    output = np.concatenate(parts, axis=1).astype(np.float32, copy=False)
    output[~np.isfinite(output)] = np.nan
    if momentum20 is None:
        raise KeyError("daily_state is missing ret_20d")
    return output, momentum20


def _reference_target_material(
    target_manifest: Mapping[str, Any],
    rows: pd.DataFrame,
) -> tuple[np.ndarray, np.ndarray]:
    candidate_count = int(target_manifest["candidate_count"])
    float_fields = list(target_manifest["float_fields"])
    files = dict(target_manifest["files"])
    values = np.memmap(
        files["float_values"]["path"],
        dtype="float32",
        mode="r",
        shape=(candidate_count, len(float_fields)),
    )
    flags = np.memmap(
        files["flags"]["path"],
        dtype="uint8",
        mode="r",
        shape=(candidate_count,),
    )
    candidate_ids = rows["candidate_id"].to_numpy(dtype=np.int64, copy=False)
    field_by_target = {
        "raw_path_distribution_v1": "raw_path_distribution_quality",
        "pareto_ordinal_v1": "pareto_ordinal_quality",
        "competing_risk_path_v1": "competing_risk_quality",
        "direct_listwise_utility_v1": "direct_listwise_utility_quality",
    }
    quality_columns = [float_fields.index(field_by_target[item]) for item in TARGET_CANDIDATE_IDS]
    quality = np.asarray(
        values[np.ix_(candidate_ids, np.asarray(quality_columns, dtype=np.int64))],
        dtype=np.float32,
    )
    filled = (
        np.asarray(flags[candidate_ids], dtype=np.uint8) & TARGET_FLAG_ENTRY_FILLED
    ) != 0
    action = quality.copy()
    action[~filled, :] = 0.0
    utility = np.asarray(
        values[candidate_ids, float_fields.index("common_sustained_action_utility")],
        dtype=np.float32,
    )
    del values, flags
    return action, utility


def _date_group_sizes_for_mask(rows: pd.DataFrame, mask: np.ndarray) -> np.ndarray:
    selected_dates = rows.loc[np.asarray(mask, dtype=bool), "date_idx"].to_numpy(
        dtype=np.int64, copy=False
    )
    if selected_dates.size == 0:
        return np.empty(0, dtype=np.int32)
    changes = np.flatnonzero(np.diff(selected_dates) != 0) + 1
    boundaries = np.concatenate(([0], changes, [selected_dates.size]))
    return np.diff(boundaries).astype(np.int32)


def _daily_top_fraction_utility(
    rows: pd.DataFrame,
    scores: np.ndarray,
    utility: np.ndarray,
    *,
    fraction: float = 0.01,
) -> pd.DataFrame:
    score_values = np.asarray(scores, dtype=np.float64).reshape(-1)
    utility_values = np.asarray(utility, dtype=np.float64).reshape(-1)
    if len(rows) != score_values.size or len(rows) != utility_values.size:
        raise ValueError("daily metric input length mismatch")
    records: list[dict[str, Any]] = []
    for trade_date, indices in rows.groupby("trade_date", sort=False).groups.items():
        positions = np.asarray(list(indices), dtype=np.int64)
        valid = np.isfinite(score_values[positions]) & np.isfinite(
            utility_values[positions]
        )
        eligible = positions[valid]
        k = min(
            int(eligible.size),
            max(1, int(math.ceil(len(positions) * float(fraction)))),
        )
        if k:
            symbols = rows.iloc[eligible]["symbol_idx"].to_numpy(
                dtype=np.int64, copy=False
            )
            order = np.lexsort((symbols, -score_values[eligible]))
            top = eligible[order[:k]]
            mean_u = float(np.mean(utility_values[top]))
        else:
            mean_u = np.nan
        records.append(
            {
                "trade_date": str(trade_date),
                "year": int(str(trade_date)[:4]),
                "date_idx": (
                    int(rows.iloc[positions[0]]["date_idx"])
                    if "date_idx" in rows.columns and positions.size
                    else -1
                ),
                "candidate_count": int(len(positions)),
                "label_coverage": float(eligible.size / max(len(positions), 1)),
                "top_count": int(k),
                "top1pct_u_mean": mean_u,
            }
        )
    return pd.DataFrame(records)


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


def _iter_reference_f0_batches(
    rows: pd.DataFrame,
    *,
    continuous: Sequence[np.ndarray],
    binary: Sequence[np.ndarray],
    batch_size: int,
    normalization_mean: np.ndarray | None,
    normalization_std: np.ndarray | None,
    shuffle: bool,
    seed: int,
) -> Iterator[tuple[np.ndarray, np.ndarray]]:
    if rows.empty:
        return
    rng = np.random.default_rng(int(seed))
    grouped = [
        (int(date_idx), np.asarray(list(indices), dtype=np.int64))
        for date_idx, indices in rows.groupby("date_idx", sort=True).groups.items()
    ]
    if shuffle:
        rng.shuffle(grouped)
    for date_idx, positions in grouped:
        if shuffle:
            positions = rng.permutation(positions)
        for start_pos in range(0, int(positions.size), int(batch_size)):
            current_positions = positions[start_pos : start_pos + int(batch_size)]
            symbols = rows.iloc[current_positions]["symbol_idx"].to_numpy(
                dtype=np.int64, copy=False
            )
            history_start = int(date_idx) - 179
            history_end = int(date_idx) + 1
            if history_start < 0:
                raise ValueError("F0 reference row lacks the complete 180-day lookback")
            parts: list[np.ndarray] = []
            for array in continuous:
                block = np.asarray(
                    array[history_start:history_end, symbols, :], dtype=np.float32
                )
                parts.append(np.transpose(block, (1, 0, 2)))
            for array in binary:
                block = np.asarray(
                    array[history_start:history_end, symbols], dtype=np.float32
                )
                parts.append(np.transpose(block, (1, 0))[:, :, None])
            x = np.concatenate(parts, axis=2).astype(np.float32, copy=False)
            if normalization_mean is not None and normalization_std is not None:
                x = (
                    x - normalization_mean.reshape(1, 1, -1)
                ) / normalization_std.reshape(1, 1, -1)
                x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0).astype(
                    np.float32, copy=False
                )
            yield current_positions, x


def _fit_reference_f0_normalization(
    rows: pd.DataFrame,
    *,
    continuous: Sequence[np.ndarray],
    binary: Sequence[np.ndarray],
    batch_size: int = 512,
) -> tuple[np.ndarray, np.ndarray]:
    sums: np.ndarray | None = None
    squares: np.ndarray | None = None
    counts: np.ndarray | None = None
    for _positions, x in _iter_reference_f0_batches(
        rows,
        continuous=continuous,
        binary=binary,
        batch_size=batch_size,
        normalization_mean=None,
        normalization_std=None,
        shuffle=False,
        seed=0,
    ):
        finite = np.isfinite(x)
        current = np.where(finite, x, 0.0).astype(np.float64, copy=False)
        batch_sums = current.sum(axis=(0, 1), dtype=np.float64)
        batch_squares = np.square(current).sum(axis=(0, 1), dtype=np.float64)
        batch_counts = finite.sum(axis=(0, 1), dtype=np.int64)
        if sums is None:
            sums = batch_sums
            squares = batch_squares
            counts = batch_counts
        else:
            sums += batch_sums
            squares += batch_squares
            counts += batch_counts
    if sums is None or squares is None or counts is None:
        raise ValueError("cannot fit F0 normalization on an empty reference set")
    denominator = np.maximum(counts.astype(np.float64), 1.0)
    mean = sums / denominator
    variance = np.maximum(squares / denominator - np.square(mean), 1.0e-6)
    std = np.sqrt(variance)
    if binary:
        mean[-len(binary) :] = 0.0
        std[-len(binary) :] = 1.0
    return mean.astype(np.float32), std.astype(np.float32)


def _run_lgbm_target_reference(
    *,
    train_rows: pd.DataFrame,
    development_rows: pd.DataFrame,
    evaluation_rows: pd.DataFrame,
    train_features: np.ndarray,
    development_features: np.ndarray,
    evaluation_features: np.ndarray,
    train_targets: np.ndarray,
    development_targets: np.ndarray,
    evaluation_utility: np.ndarray,
    baseline_daily: pd.DataFrame,
    config: Mapping[str, Any],
    output_dir: Path,
) -> dict[str, Any]:
    import lightgbm as lgb

    output_dir.mkdir(parents=True, exist_ok=True)
    result: dict[str, Any] = {"targets": {}}
    baseline_mean = float(baseline_daily["top1pct_u_mean"].mean())
    for target_idx, target_id in enumerate(TARGET_CANDIDATE_IDS):
        train_grade = relevance_grades(train_targets[:, target_idx])
        development_grade = relevance_grades(development_targets[:, target_idx])
        train_mask = train_grade != np.uint8(255)
        development_mask = development_grade != np.uint8(255)
        train_groups = _date_group_sizes_for_mask(train_rows, train_mask)
        development_groups = _date_group_sizes_for_mask(
            development_rows, development_mask
        )
        if train_groups.size == 0 or development_groups.size == 0:
            raise ValueError(f"reference LightGBM has no rows for {target_id}")
        train_set = lgb.Dataset(
            train_features[train_mask],
            label=train_grade[train_mask].astype(np.int32),
            group=train_groups,
            free_raw_data=False,
        )
        development_set = lgb.Dataset(
            development_features[development_mask],
            label=development_grade[development_mask].astype(np.int32),
            group=development_groups,
            reference=train_set,
            free_raw_data=False,
        )
        params = {
            "objective": "lambdarank",
            "metric": "ndcg",
            "ndcg_eval_at": [20],
            "label_gain": [0, 1, 3, 7, 15],
            "num_leaves": int(config.get("num_leaves", 31)),
            "min_data_in_leaf": int(config.get("min_data_in_leaf", 1000)),
            "learning_rate": float(config.get("learning_rate", 0.05)),
            "feature_fraction": float(config.get("feature_fraction", 0.8)),
            "bagging_fraction": float(config.get("bagging_fraction", 0.8)),
            "bagging_freq": int(config.get("bagging_freq", 1)),
            "lambda_l2": float(config.get("lambda_l2", 1.0)),
            "verbosity": -1,
            "seed": 7,
            "num_threads": 8,
        }
        booster = lgb.train(
            params,
            train_set,
            num_boost_round=int(config.get("max_rounds", 500)),
            valid_sets=[development_set],
            valid_names=["development"],
            callbacks=[
                lgb.early_stopping(
                    int(config.get("early_stopping_rounds", 50)), verbose=False
                ),
                lgb.log_evaluation(period=0),
            ],
        )
        model_path = output_dir / f"{target_id}.txt"
        booster.save_model(str(model_path))
        predictions = np.asarray(
            booster.predict(
                evaluation_features,
                num_iteration=booster.best_iteration,
            ),
            dtype=np.float32,
        )
        daily = _daily_top_fraction_utility(
            evaluation_rows,
            predictions,
            evaluation_utility,
        )
        daily_path = output_dir / f"{target_id}_daily.parquet"
        _atomic_write_parquet(daily_path, daily)
        mean_u = float(daily["top1pct_u_mean"].mean())
        result["targets"][target_id] = {
            "top1pct_u_mean": mean_u,
            "delta_vs_momentum": float(mean_u - baseline_mean),
            "best_iteration": int(booster.best_iteration or 0),
            "model_path": str(model_path.resolve()),
            "model_sha256": _file_sha256(model_path),
            "daily_path": str(daily_path.resolve()),
            "daily_sha256": _file_sha256(daily_path),
        }
    return result


def _run_gru_target_reference(
    *,
    manifest: Mapping[str, Any],
    train_rows: pd.DataFrame,
    development_rows: pd.DataFrame,
    evaluation_rows: pd.DataFrame,
    train_targets: np.ndarray,
    development_targets: np.ndarray,
    evaluation_utility: np.ndarray,
    baseline_daily: pd.DataFrame,
    config: Mapping[str, Any],
    output_dir: Path,
) -> dict[str, Any]:
    import torch
    import torch.nn as nn
    import torch.nn.functional as torch_functional

    output_dir.mkdir(parents=True, exist_ok=True)
    continuous, binary = _open_reference_f0_material(manifest)
    batch_size = int(config.get("effective_batch", 512))
    mean, std = _fit_reference_f0_normalization(
        train_rows,
        continuous=continuous,
        binary=binary,
        batch_size=batch_size,
    )
    _atomic_write_json(
        output_dir / "normalization.json",
        {"mean": mean.tolist(), "std": std.tolist(), "fit_years": list(range(2010, 2020))},
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(7)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(7)

    class ReferenceGRU(nn.Module):
        def __init__(self, input_dim: int) -> None:
            super().__init__()
            self.gru = nn.GRU(
                input_dim,
                int(config.get("hidden_dim", 64)),
                num_layers=int(config.get("layers", 2)),
                dropout=float(config.get("dropout", 0.1)),
                batch_first=True,
            )
            self.head = nn.Linear(int(config.get("hidden_dim", 64)), len(TARGET_CANDIDATE_IDS))

        def forward(self, x: Any) -> Any:
            output, _state = self.gru(x)
            return torch.sigmoid(self.head(output[:, -1, :]))

    model = ReferenceGRU(int(mean.size)).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(config.get("learning_rate", 1.0e-3)),
        weight_decay=float(config.get("weight_decay", 1.0e-4)),
    )
    amp_enabled = bool(config.get("amp", True) and device.type == "cuda")
    scaler = torch.cuda.amp.GradScaler(enabled=amp_enabled)

    def _development_loss() -> float:
        model.eval()
        total = 0.0
        weight = 0
        with torch.no_grad():
            for positions, x in _iter_reference_f0_batches(
                development_rows,
                continuous=continuous,
                binary=binary,
                batch_size=batch_size,
                normalization_mean=mean,
                normalization_std=std,
                shuffle=False,
                seed=0,
            ):
                y = development_targets[positions]
                mask = np.isfinite(y)
                if not bool(mask.any()):
                    continue
                tensor_x = torch.from_numpy(x).to(device)
                tensor_y = torch.from_numpy(np.nan_to_num(y, nan=0.0)).to(device)
                tensor_mask = torch.from_numpy(mask.astype(np.float32)).to(device)
                with torch.cuda.amp.autocast(enabled=amp_enabled):
                    prediction = model(tensor_x)
                    loss_values = torch_functional.smooth_l1_loss(
                        prediction, tensor_y, reduction="none"
                    )
                    loss = (loss_values * tensor_mask).sum() / tensor_mask.sum().clamp_min(1.0)
                count = int(mask.sum())
                total += float(loss.detach().cpu()) * count
                weight += count
        return float(total / max(weight, 1))

    best_loss = float("inf")
    best_state: dict[str, Any] | None = None
    epochs_without_improvement = 0
    epoch_log: list[dict[str, Any]] = []
    for epoch in range(int(config.get("max_epochs", 2))):
        model.train()
        running = 0.0
        weight = 0
        for positions, x in _iter_reference_f0_batches(
            train_rows,
            continuous=continuous,
            binary=binary,
            batch_size=batch_size,
            normalization_mean=mean,
            normalization_std=std,
            shuffle=True,
            seed=7 + epoch,
        ):
            y = train_targets[positions]
            mask = np.isfinite(y)
            if not bool(mask.any()):
                continue
            tensor_x = torch.from_numpy(x).to(device)
            tensor_y = torch.from_numpy(np.nan_to_num(y, nan=0.0)).to(device)
            tensor_mask = torch.from_numpy(mask.astype(np.float32)).to(device)
            optimizer.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=amp_enabled):
                prediction = model(tensor_x)
                loss_values = torch_functional.smooth_l1_loss(
                    prediction, tensor_y, reduction="none"
                )
                loss = (loss_values * tensor_mask).sum() / tensor_mask.sum().clamp_min(1.0)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            count = int(mask.sum())
            running += float(loss.detach().cpu()) * count
            weight += count
        development_loss = _development_loss()
        epoch_log.append(
            {
                "epoch": epoch + 1,
                "train_loss": float(running / max(weight, 1)),
                "development_loss": development_loss,
            }
        )
        if development_loss < best_loss - 1.0e-8:
            best_loss = development_loss
            best_state = {
                key: value.detach().cpu().clone()
                for key, value in model.state_dict().items()
            }
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement > int(config.get("patience", 1)):
                break
    if best_state is None:
        raise RuntimeError("reference GRU did not produce a checkpoint")
    model.load_state_dict(best_state)
    checkpoint_path = output_dir / "reference_gru.pt"
    torch.save(
        {
            "state_dict": best_state,
            "normalization_mean": mean,
            "normalization_std": std,
            "config": dict(config),
            "target_ids": list(TARGET_CANDIDATE_IDS),
        },
        checkpoint_path,
    )
    _atomic_write_json(output_dir / "training_log.json", epoch_log)

    predictions = np.full(
        (len(evaluation_rows), len(TARGET_CANDIDATE_IDS)),
        np.nan,
        dtype=np.float32,
    )
    model.eval()
    with torch.no_grad():
        for positions, x in _iter_reference_f0_batches(
            evaluation_rows,
            continuous=continuous,
            binary=binary,
            batch_size=batch_size,
            normalization_mean=mean,
            normalization_std=std,
            shuffle=False,
            seed=0,
        ):
            tensor_x = torch.from_numpy(x).to(device)
            with torch.cuda.amp.autocast(enabled=amp_enabled):
                current = model(tensor_x)
            predictions[positions] = current.detach().cpu().numpy().astype(np.float32)
    baseline_mean = float(baseline_daily["top1pct_u_mean"].mean())
    result: dict[str, Any] = {
        "device": str(device),
        "best_development_loss": float(best_loss),
        "checkpoint_path": str(checkpoint_path.resolve()),
        "checkpoint_sha256": _file_sha256(checkpoint_path),
        "targets": {},
    }
    for target_idx, target_id in enumerate(TARGET_CANDIDATE_IDS):
        daily = _daily_top_fraction_utility(
            evaluation_rows,
            predictions[:, target_idx],
            evaluation_utility,
        )
        daily_path = output_dir / f"{target_id}_daily.parquet"
        _atomic_write_parquet(daily_path, daily)
        mean_u = float(daily["top1pct_u_mean"].mean())
        result["targets"][target_id] = {
            "top1pct_u_mean": mean_u,
            "delta_vs_momentum": float(mean_u - baseline_mean),
            "daily_path": str(daily_path.resolve()),
            "daily_sha256": _file_sha256(daily_path),
        }
    return result


def _run_target_reference_learners(
    *,
    study: Mapping[str, Any],
    manifest: Mapping[str, Any],
    target_manifest: Mapping[str, Any],
    output_dir: Path,
) -> dict[str, Any]:
    freeze_payload = load_research_freeze(study)
    gate = dict(freeze_payload["freeze"].get("target_freeze_gate", {}) or {})
    learner_specs = {
        str(item["learner_id"]): dict(item.get("config", {}) or {})
        for item in list(gate.get("reference_learners", []) or [])
    }
    lgb_config = learner_specs["snapshot_lgbm_reference"]
    gru_config = learner_specs["f0_gru_reference"]
    candidate_path = _resolve_path(str(manifest["candidate_index_path"]))
    lgb_train_dates = _reference_complete_dates(
        candidate_path,
        years=range(2010, 2020),
        max_rows=int(lgb_config.get("max_train_rows", 500_000)),
    )
    gru_train_dates = _reference_complete_dates(
        candidate_path,
        years=range(2010, 2020),
        max_rows=int(gru_config.get("max_train_rows", 200_000)),
    )
    development_dates = _reference_complete_dates(
        candidate_path,
        years=[2020],
        max_rows=0,
    )
    evaluation_dates = _reference_complete_dates(
        candidate_path,
        years=[2021],
        max_rows=0,
    )
    lgb_train_rows = _reference_candidate_rows(
        candidate_path, date_indices=lgb_train_dates
    ).reset_index(drop=True)
    gru_train_rows = _reference_candidate_rows(
        candidate_path, date_indices=gru_train_dates
    ).reset_index(drop=True)
    development_rows = _reference_candidate_rows(
        candidate_path, date_indices=development_dates
    ).reset_index(drop=True)
    evaluation_rows = _reference_candidate_rows(
        candidate_path, date_indices=evaluation_dates
    ).reset_index(drop=True)
    if any(
        frame.empty
        for frame in (
            lgb_train_rows,
            gru_train_rows,
            development_rows,
            evaluation_rows,
        )
    ):
        raise ValueError("reference learner split contains no rows")
    if bool((evaluation_rows["year"].astype(int) > 2021).any()):
        raise RuntimeError("reference learner read beyond 2021")

    lgb_train_features, _train_momentum = _reference_snapshot_features(
        manifest, lgb_train_rows
    )
    development_features, _development_momentum = _reference_snapshot_features(
        manifest, development_rows
    )
    evaluation_features, evaluation_momentum = _reference_snapshot_features(
        manifest, evaluation_rows
    )
    lgb_train_targets, _lgb_train_utility = _reference_target_material(
        target_manifest, lgb_train_rows
    )
    gru_train_targets, _gru_train_utility = _reference_target_material(
        target_manifest, gru_train_rows
    )
    development_targets, _development_utility = _reference_target_material(
        target_manifest, development_rows
    )
    _evaluation_targets, evaluation_utility = _reference_target_material(
        target_manifest, evaluation_rows
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    baseline_daily = _daily_top_fraction_utility(
        evaluation_rows,
        evaluation_momentum,
        evaluation_utility,
    )
    baseline_path = output_dir / "past_20d_momentum_daily.parquet"
    _atomic_write_parquet(baseline_path, baseline_daily)
    lgbm = _run_lgbm_target_reference(
        train_rows=lgb_train_rows,
        development_rows=development_rows,
        evaluation_rows=evaluation_rows,
        train_features=lgb_train_features,
        development_features=development_features,
        evaluation_features=evaluation_features,
        train_targets=lgb_train_targets,
        development_targets=development_targets,
        evaluation_utility=evaluation_utility,
        baseline_daily=baseline_daily,
        config=lgb_config,
        output_dir=output_dir / "lightgbm",
    )
    gru = _run_gru_target_reference(
        manifest=manifest,
        train_rows=gru_train_rows,
        development_rows=development_rows,
        evaluation_rows=evaluation_rows,
        train_targets=gru_train_targets,
        development_targets=development_targets,
        evaluation_utility=evaluation_utility,
        baseline_daily=baseline_daily,
        config=gru_config,
        output_dir=output_dir / "gru",
    )
    baseline_mean = float(baseline_daily["top1pct_u_mean"].mean())
    targets: dict[str, Any] = {}
    for target_id in TARGET_CANDIDATE_IDS:
        lgb_metrics = dict(lgbm["targets"][target_id])
        gru_metrics = dict(gru["targets"][target_id])
        targets[target_id] = {
            "momentum_top1pct_u_mean": baseline_mean,
            "snapshot_lgbm_reference": lgb_metrics,
            "f0_gru_reference": gru_metrics,
            "best_delta_vs_momentum": float(
                max(
                    float(lgb_metrics["delta_vs_momentum"]),
                    float(gru_metrics["delta_vs_momentum"]),
                )
            ),
        }
    summary = {
        "artifact_type": "seq100_target_reference_learners",
        "created_at": _now(),
        "authorized_train_years": list(range(2010, 2020)),
        "authorized_development_year": 2020,
        "authorized_evaluation_year": 2021,
        "row_counts": {
            "lgbm_train": int(len(lgb_train_rows)),
            "gru_train": int(len(gru_train_rows)),
            "development": int(len(development_rows)),
            "evaluation": int(len(evaluation_rows)),
        },
        "baseline_daily_path": str(baseline_path.resolve()),
        "baseline_daily_sha256": _file_sha256(baseline_path),
        "lightgbm": lgbm,
        "gru": gru,
        "targets": targets,
    }
    summary["reference_sha256"] = _canonical_json_sha256(summary)
    _atomic_write_json(output_dir / "reference_summary.json", summary)
    return summary


def _target_gate_from_diagnostics(
    daily: pd.DataFrame,
    *,
    target_id: str,
    tolerance: float = 1.0e-8,
) -> dict[str, Any]:
    current = daily[daily["target_id"].astype(str).eq(str(target_id))].copy()
    current = current[current["year"].astype(int).isin([2020, 2021])]
    year_records: dict[str, Any] = {}
    passed = True
    for year in (2020, 2021):
        group = current[current["year"].astype(int).eq(year)].copy()
        if group.empty:
            year_records[str(year)] = {"passed": False, "reason": "missing_year"}
            passed = False
            continue
        weights = group["top1pct_count"].clip(lower=1).to_numpy(dtype=np.float64)
        grade_values: dict[str, list[float]] = {}
        for metric in ("u", "d20", "mdd", "fade", "terminal_failure"):
            grade_values[metric] = [
                float(group[f"grade{grade}_{metric}"].median())
                for grade in range(5)
            ]
        finite_grades = all(
            bool(np.isfinite(np.asarray(values, dtype=np.float64)).all())
            for values in grade_values.values()
        )
        monotone = bool(finite_grades)
        if monotone:
            monotone = bool(
                np.all(np.diff(grade_values["u"]) >= -float(tolerance))
                and np.all(np.diff(grade_values["d20"]) >= -float(tolerance))
                and np.all(np.diff(grade_values["mdd"]) <= float(tolerance))
                and np.all(np.diff(grade_values["fade"]) <= float(tolerance))
                and np.all(
                    np.diff(grade_values["terminal_failure"]) <= float(tolerance)
                )
            )
        negative_rate = float(
            np.average(
                group["top1pct_negative_d20_rate"].fillna(1.0),
                weights=weights,
            )
        )
        mdd_median = float(group["top1pct_mdd_median"].median())
        fade_rate = float(
            np.average(
                group["top1pct_fade_gt10_rate"].fillna(1.0),
                weights=weights,
            )
        )
        record = {
            "date_count": int(len(group)),
            "top1pct_u_mean": float(
                np.average(group["top1pct_u_mean"].fillna(-np.inf), weights=weights)
            ),
            "top1pct_d20_median": float(group["top1pct_d20_median"].median()),
            "top1pct_negative_d20_rate": negative_rate,
            "top1pct_mdd_median": mdd_median,
            "top1pct_fade_gt10_rate": fade_rate,
            "top1pct_terminal_failure_rate": float(
                np.average(
                    group["top1pct_terminal_failure_rate"].fillna(1.0),
                    weights=weights,
                )
            ),
            "grade_values": grade_values,
            "monotone": monotone,
            "negative_d20_pass": bool(negative_rate <= 0.05),
            "mdd_pass": bool(mdd_median <= 0.08),
            "fade_pass": bool(fade_rate <= 0.10),
        }
        record["passed"] = bool(
            record["monotone"]
            and record["negative_d20_pass"]
            and record["mdd_pass"]
            and record["fade_pass"]
        )
        passed = bool(passed and record["passed"])
        year_records[str(year)] = record
    yearly_u = [
        float(year_records[str(year)].get("top1pct_u_mean", -np.inf))
        for year in (2020, 2021)
    ]
    return {
        "target_id": str(target_id),
        "years": year_records,
        "semantic_gate_passed": bool(passed),
        "minimum_year_u": float(min(yearly_u)),
        "year_u_dispersion": float(np.std(yearly_u)),
        "pooled_year_u": float(np.mean(yearly_u)),
    }


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


def _count_numeric_parameters(value: Any) -> int:
    if isinstance(value, bool) or value is None:
        return 0
    if isinstance(value, (int, float, np.integer, np.floating)):
        return 1
    if isinstance(value, Mapping):
        return sum(_count_numeric_parameters(item) for item in value.values())
    if isinstance(value, (list, tuple)):
        return sum(_count_numeric_parameters(item) for item in value)
    return 0


def freeze_target(
    *,
    study_path: Path = DEFAULT_STUDY_PATH,
    output_root: Path = DEFAULT_OUTPUT_ROOT,
) -> dict[str, Any]:
    study = load_study(study_path)
    freeze_payload = load_research_freeze(study)
    freeze = dict(freeze_payload["freeze"])
    pack_path, manifest = _validate_source_bindings(study)
    protection_before = _verify_protected_bindings(study)
    target_manifest_path, target_manifest = _load_current_artifact(
        output_root,
        task="targets",
        pointer_name="target_manifest",
    )
    if str(target_manifest.get("status", "")) != "targets_built":
        raise ValueError("freeze-target requires an unfrozen targets_built artifact")
    if str(target_manifest.get("study_contract_sha256", "")) != str(
        study["contract_sha256"]
    ):
        raise ValueError("target artifact belongs to a different contract")
    diagnostics_meta = dict(target_manifest.get("diagnostics", {}) or {})
    diagnostics_path = Path(str(diagnostics_meta.get("path", ""))).resolve()
    if _file_sha256(diagnostics_path) != str(diagnostics_meta.get("sha256", "")):
        raise ValueError("target diagnostics changed")
    daily = pd.read_parquet(
        diagnostics_path,
        filters=[("year", "in", [2020, 2021])],
    )
    if bool((daily["year"].astype(int) > 2021).any()):
        raise RuntimeError("freeze-target read beyond the authorized 2021 boundary")
    attempt_dir = _next_attempt_dir(output_root, "target_freeze")
    semantic = {
        target_id: _target_gate_from_diagnostics(daily, target_id=target_id)
        for target_id in TARGET_CANDIDATE_IDS
    }
    reference = _run_target_reference_learners(
        study=study,
        manifest=manifest,
        target_manifest=target_manifest,
        output_dir=attempt_dir / "reference_learners",
    )
    target_specs = {
        str(item["target_id"]): item
        for item in list(freeze.get("target_candidates", []) or [])
    }
    eligible: list[str] = []
    for target_id in TARGET_CANDIDATE_IDS:
        learner = dict(reference.get("targets", {}).get(target_id, {}) or {})
        best_delta = float(learner.get("best_delta_vs_momentum", -np.inf))
        semantic[target_id]["reference_learners"] = learner
        semantic[target_id]["learnability_passed"] = bool(best_delta > 0.0)
        semantic[target_id]["free_parameter_count"] = _count_numeric_parameters(
            dict(target_specs[target_id].get("free_parameters", {}) or {})
        )
        semantic[target_id]["passed"] = bool(
            semantic[target_id]["semantic_gate_passed"]
            and semantic[target_id]["learnability_passed"]
        )
        if semantic[target_id]["passed"]:
            eligible.append(target_id)

    pairwise: list[dict[str, Any]] = []
    frozen_target_id: str | None = None
    if eligible:
        pooled_order = sorted(
            eligible,
            key=lambda target_id: (
                -float(semantic[target_id]["pooled_year_u"]),
                target_id,
            ),
        )
        leader = pooled_order[0]
        tied = [leader]
        leader_daily = daily[daily["target_id"].astype(str).eq(leader)]
        for challenger in pooled_order[1:]:
            comparison = _stratified_block_bootstrap_delta(
                leader_daily,
                daily[daily["target_id"].astype(str).eq(challenger)],
            )
            pairwise.append(
                {"left": leader, "right": challenger, **comparison}
            )
            if not (math.isfinite(comparison["lower"]) and comparison["lower"] > 0.0):
                tied.append(challenger)
        frozen_target_id = sorted(
            tied,
            key=lambda target_id: (
                -float(semantic[target_id]["minimum_year_u"]),
                float(semantic[target_id]["year_u_dispersion"]),
                -float(semantic[target_id]["pooled_year_u"]),
                int(semantic[target_id]["free_parameter_count"]),
                target_id,
            ),
        )[0]

    status = "target_frozen" if frozen_target_id is not None else "target_invalid"
    report = {
        "artifact_type": "seq100_target_freeze_report",
        "schema_version": 1,
        "created_at": _now(),
        "status": status,
        "study_contract_sha256": study["contract_sha256"],
        "research_freeze_sha256": freeze_payload["freeze_sha256"],
        "authorized_result_years": [2020, 2021],
        "semantic_gates": semantic,
        "reference_learners": reference,
        "pairwise_bootstrap": pairwise,
        "frozen_target_id": frozen_target_id,
    }
    report["target_freeze_sha256"] = _canonical_json_sha256(report)
    report_path = attempt_dir / "target_freeze_report.json"
    _atomic_write_json(report_path, report)

    target_manifest = dict(target_manifest)
    if frozen_target_id is not None:
        paths = TargetPaths(
            root=target_manifest_path.parent,
            float_values=Path(target_manifest["files"]["float_values"]["path"]),
            dominance_counts=Path(
                target_manifest["files"]["dominance_counts"]["path"]
            ),
            relevance_grade=Path(
                target_manifest["files"]["relevance_grade"]["path"]
            ),
            flags=Path(target_manifest["files"]["flags"]["path"]),
            manifest=target_manifest_path,
        )
        _rewrite_frozen_relevance_fields(
            paths,
            candidate_count=int(target_manifest["candidate_count"]),
            frozen_profile=frozen_target_id,
        )
        for name, path in {
            "float_values": paths.float_values,
            "dominance_counts": paths.dominance_counts,
            "relevance_grade": paths.relevance_grade,
            "flags": paths.flags,
        }.items():
            target_manifest["files"][name] = {
                "path": str(path.resolve()),
                "size": int(path.stat().st_size),
                "sha256": _file_sha256(path),
            }
        selected_spec = {
            "common_trade_semantics": freeze.get("common_trade_semantics"),
            "shared_descriptors": freeze.get("shared_descriptors"),
            "target": target_specs[frozen_target_id],
        }
        target_manifest["frozen_target_id"] = frozen_target_id
        target_manifest["frozen_target_sha256"] = _canonical_json_sha256(
            selected_spec
        )
    target_manifest["status"] = status
    target_manifest["target_freeze_report"] = {
        "path": str(report_path.resolve()),
        "sha256": _file_sha256(report_path),
        "target_freeze_sha256": report["target_freeze_sha256"],
    }
    target_manifest.pop("target_artifact_sha256", None)
    target_manifest["target_artifact_sha256"] = _canonical_json_sha256(
        target_manifest
    )
    _atomic_write_json(target_manifest_path, target_manifest)
    protection_after = _verify_protected_bindings(study)
    if protection_after != protection_before:
        raise RuntimeError("protected-object hashes changed while freezing target")
    summary = {
        "status": status,
        "output_dir": str(attempt_dir.resolve()),
        "target_freeze_report": str(report_path.resolve()),
        "target_freeze_sha256": report["target_freeze_sha256"],
        "frozen_target_id": frozen_target_id,
        "target_manifest": str(target_manifest_path.resolve()),
        "target_artifact_sha256": target_manifest["target_artifact_sha256"],
    }
    _atomic_write_json(attempt_dir / "freeze_target_summary.json", summary)
    _atomic_write_json(
        output_root / "target_freeze/current.json",
        {
            "attempt": str(attempt_dir.resolve()),
            **summary,
            "study_contract_sha256": study["contract_sha256"],
            "updated_at": _now(),
        },
    )
    _atomic_write_json(
        output_root / "targets/current.json",
        {
            "attempt": str(target_manifest_path.parent.resolve()),
            "target_manifest": str(target_manifest_path.resolve()),
            "target_artifact_sha256": target_manifest["target_artifact_sha256"],
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
        task="targets",
        pointer_name="target_manifest",
    )
    if str(manifest.get("status", "")) != "target_frozen":
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


def _load_fold_view(
    feature_manifest: Mapping[str, Any], fold_year: int | str
) -> dict[str, Any]:
    key = str(fold_year) if str(fold_year) == "screen" else str(int(fold_year))
    metadata = dict(feature_manifest["fold_views"].get(key, {}) or {})
    if not metadata:
        raise KeyError(f"feature view lacks fold {key}")
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
                    continuous[global_row, continuous_columns], dtype=np.float64
                )
                category_values = [
                    _map_categories(
                        np.asarray([categorical[global_row, int(column)]], dtype=np.int64),
                        category_vocabularies[position],
                    )[0]
                    for position, column in enumerate(categorical_columns)
                ]
                return np.concatenate(
                    [continuous_values, np.asarray(category_values, dtype=np.float64)]
                )
            else:
                raise TypeError(f"unsupported LightGBM sequence index: {type(idx).__name__}")
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
            return np.concatenate(
                [continuous_values, *category_parts], axis=1
            ).astype(np.float64, copy=False)

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
    seed: int = 7,
) -> tuple[dict[str, Any], int, int]:
    freeze = dict(load_research_freeze(study)["freeze"])
    model_specs = {
        str(item["model_id"]): dict(item)
        for item in list(freeze.get("model_candidates", []) or [])
    }
    if str(model_id) not in model_specs:
        raise KeyError(f"unregistered frozen model: {model_id}")
    config = dict(model_specs[str(model_id)].get("config", {}) or {})
    parameters: dict[str, Any] = {
        "objective": objective,
        "num_leaves": int(config["num_leaves"]),
        "min_data_in_leaf": int(config["min_data_in_leaf"]),
        "learning_rate": float(config["learning_rate"]),
        "feature_fraction": float(config["feature_fraction"]),
        "bagging_fraction": float(config["bagging_fraction"]),
        "bagging_freq": int(config["bagging_freq"]),
        "lambda_l2": float(config["lambda_l2"]),
        "seed": int(seed),
        "feature_fraction_seed": int(seed),
        "bagging_seed": int(seed),
        "data_random_seed": int(seed),
        "verbosity": -1,
        "deterministic": True,
        "force_col_wise": True,
        "num_threads": int(config.get("num_threads", 8)),
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
        int(config["max_rounds"]),
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
    seed: int = 7,
) -> tuple[Any, dict[str, Any]]:
    import lightgbm as lgb

    parameters, maximum_rounds, patience = _lgb_parameters(
        study,
        model_id=model_id,
        objective=objective,
        seed=seed,
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
    freeze_payload = load_research_freeze(study)
    freeze = dict(freeze_payload["freeze"])
    protection_before = _verify_protected_bindings(study)
    _pack_path, pack_manifest = _validate_source_bindings(study)
    _view_path, feature_manifest = _load_feature_view_bundle(output_root, study)
    target_manifest, target, _counts, grades, flags = _load_target_bundle(
        output_root, study
    )
    continuous, categorical, candidate_date_idx, candidate_symbol_idx = _open_feature_arrays(
        feature_manifest
    )
    fold = _load_fold_view(feature_manifest, "screen")
    train_range = dict(fold["ranges"]["train"])
    development_range = dict(fold["ranges"]["development"])
    test_range = dict(fold["ranges"]["test"])
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
    test_rows = _row_ids_for_date_range(
        candidate_date_idx,
        test_range["date_idx_start"],
        test_range["date_idx_end"],
    )
    if bool(np.any(np.asarray(candidate_date_idx[test_rows], dtype=np.int32) >= len(pack_manifest["date_values"]))):
        raise ValueError("feature screen test coordinates exceed the source calendar")
    test_years = {
        int(str(pack_manifest["date_values"][int(item)])[:4])
        for item in np.unique(np.asarray(candidate_date_idx[test_rows], dtype=np.int32))
    }
    if test_years != {2022}:
        raise RuntimeError(f"feature screen may read only 2022 test results, got {sorted(test_years)}")
    conditional = np.asarray(
        target[:, TARGET_FIELD_INDEX["conditional_relevance"]], dtype=np.float32
    )
    action = np.asarray(target[:, TARGET_FIELD_INDEX["action_relevance"]], dtype=np.float32)
    utility = np.asarray(
        target[:, TARGET_FIELD_INDEX["common_sustained_action_utility"]],
        dtype=np.float32,
    )
    entry_filled = (np.asarray(flags, dtype=np.uint8) & TARGET_FLAG_ENTRY_FILLED) != 0
    rank_train_rows = train_rows[entry_filled[train_rows] & np.isfinite(conditional[train_rows])]
    rank_development_rows = development_rows[
        entry_filled[development_rows] & np.isfinite(conditional[development_rows])
    ]
    if not rank_train_rows.size or not rank_development_rows.size or not test_rows.size:
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
            "authorized_train_years": list(range(2010, 2021)),
            "authorized_development_year": 2021,
            "authorized_test_year": 2022,
        },
    )
    screen_spec = dict(dict(freeze["features"])["snapshot_screen"])
    profiles = [str(item) for item in list(screen_spec["profiles"])]
    fixed_model_id = str(screen_spec["fixed_model"])
    if fixed_model_id != "lgbm_lambdarank_multioutput":
        raise ValueError("snapshot screen fixed model changed")
    bootstrap_spec = dict(screen_spec["block_bootstrap"])
    records: list[dict[str, Any]] = []
    daily_by_profile: dict[str, pd.DataFrame] = {}
    date_values = np.asarray(
        [str(item) for item in list(pack_manifest.get("date_values", []) or [])],
        dtype=object,
    )
    test_coordinates = pd.DataFrame(
        {
            "trade_date": date_values[np.asarray(candidate_date_idx[test_rows], dtype=np.int64)],
            "date_idx": np.asarray(candidate_date_idx[test_rows], dtype=np.int32),
            "symbol_idx": np.asarray(candidate_symbol_idx[test_rows], dtype=np.int32),
        }
    )
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
        test_all_sequence = _make_lgb_sequence(
            continuous=continuous,
            categorical=categorical,
            row_ids=test_rows,
            continuous_columns=continuous_columns,
            categorical_columns=categorical_columns,
            category_vocabularies=vocabularies,
        )
        fill_model, fill_evidence = _train_lgb_model(
            study=study,
            model_id=fixed_model_id,
            objective="binary",
            train_sequence=train_all_sequence,
            train_label=entry_filled[train_rows].astype(np.uint8),
            development_sequence=development_all_sequence,
            development_label=entry_filled[development_rows].astype(np.uint8),
            feature_names=feature_names,
            categorical_count=len(categorical_names),
            output_path=profile_dir / "fill_model.txt",
            seed=7,
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
            model_id=fixed_model_id,
            objective="lambdarank",
            train_sequence=rank_train_sequence,
            train_label=np.asarray(grades[rank_train_rows], dtype=np.uint8),
            development_sequence=rank_development_sequence,
            development_label=np.asarray(grades[rank_development_rows], dtype=np.uint8),
            feature_names=feature_names,
            categorical_count=len(categorical_names),
            output_path=profile_dir / "rank_model.txt",
            train_group=_date_group_sizes(candidate_date_idx[rank_train_rows]),
            development_group=_date_group_sizes(
                candidate_date_idx[rank_development_rows]
            ),
            seed=7,
        )
        raw_rank_development = _predict_lgb_sequence(
            rank_model, rank_development_sequence
        )
        isotonic, isotonic_evidence = _fit_isotonic_mapping(
            raw_rank_development,
            conditional[rank_development_rows],
        )
        _atomic_write_json(profile_dir / "isotonic.json", isotonic_evidence)
        fill_probability = _predict_lgb_sequence(fill_model, test_all_sequence)
        raw_all_rank = _predict_lgb_sequence(rank_model, test_all_sequence)
        conditional_prediction = np.asarray(
            isotonic.predict(raw_all_rank), dtype=np.float32
        )
        selection_score = fill_probability * conditional_prediction
        ranking_daily, ranking_metrics = daily_ranking_metrics(
            date_idx=candidate_date_idx[test_rows],
            action_relevance=action[test_rows],
            selection_score=selection_score,
        )
        utility_daily = _daily_top_fraction_utility(
            test_coordinates,
            selection_score,
            utility[test_rows],
        )
        daily = utility_daily.merge(
            ranking_daily.drop(columns=["year"], errors="ignore"),
            on="date_idx",
            how="left",
            validate="one_to_one",
        )
        daily_by_profile[profile] = daily
        _atomic_write_parquet(profile_dir / "daily_metrics.parquet", daily)
        metrics = {
            **ranking_metrics,
            "daily_top1pct_common_utility": float(daily["top1pct_u_mean"].mean()),
            "test_year": 2022,
        }
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
            "test_candidate_count": int(len(test_rows)),
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
    leader = max(
        records,
        key=lambda item: (
            float(item["metrics"]["daily_top1pct_common_utility"]),
            -profiles.index(str(item["profile"])),
        ),
    )
    comparisons: list[dict[str, Any]] = []
    statistically_tied: list[dict[str, Any]] = []
    for record in records:
        if str(record["profile"]) == str(leader["profile"]):
            comparison = {
                "estimate": 0.0,
                "lower": 0.0,
                "upper": 0.0,
            }
        else:
            comparison = _stratified_block_bootstrap_delta(
                daily_by_profile[str(leader["profile"])],
                daily_by_profile[str(record["profile"])],
                iterations=int(bootstrap_spec["iterations"]),
                block_days=int(bootstrap_spec["block_trading_days"]),
                seed=20260725,
            )
        comparisons.append(
            {
                "leader": str(leader["profile"]),
                "challenger": str(record["profile"]),
                **comparison,
            }
        )
        if str(record["profile"]) == str(leader["profile"]) or not (
            math.isfinite(float(comparison["lower"]))
            and float(comparison["lower"]) > 0.0
        ):
            statistically_tied.append(record)
    selected = min(
        statistically_tied,
        key=lambda item: profiles.index(str(item["profile"])),
    )
    best_metric = float(leader["metrics"]["daily_top1pct_common_utility"])
    freeze = {
        "artifact_type": "seq100_signal_quality_feature_freeze",
        "status": "completed",
        "created_at": _now(),
        "study_contract_sha256": study["contract_sha256"],
        "research_freeze_sha256": freeze_payload["freeze_sha256"],
        "feature_view_sha256": feature_manifest["resolved_feature_view_sha256"],
        "target_manifest_sha256": _file_sha256(
            Path(str(target_manifest["manifest_path"]))
        ),
        "train_years": list(range(2010, 2021)),
        "development_year": 2021,
        "test_year": 2022,
        "profiles": records,
        "primary_metric": "daily_equal_weight_top1pct_common_sustained_action_utility",
        "best_metric": float(best_metric),
        "point_estimate_leader": str(leader["profile"]),
        "pairwise_block_bootstrap": comparisons,
        "selected_profile": selected["profile"],
        "selected_feature_names": selected["feature_names"],
        "selection_reason": "highest_2022_top1pct_common_utility; when the 20-day block-bootstrap difference is not strictly positive choose fewer frozen feature families",
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
    protection_after = _verify_protected_bindings(study)
    if protection_after != protection_before:
        raise RuntimeError("protected-object hashes changed during feature screening")
    return summary


def build_view(
    *,
    study_path: Path = DEFAULT_STUDY_PATH,
    output_root: Path = DEFAULT_OUTPUT_ROOT,
) -> dict[str, Any]:
    study = load_study(study_path)
    freeze_payload = load_research_freeze(study)
    frozen_features = dict(freeze_payload["freeze"]["features"])
    pack_path, manifest = _validate_source_bindings(study)
    protection_before = _verify_protected_bindings(study)
    target_manifest_path, target_manifest = _load_current_artifact(
        output_root,
        task="targets",
        pointer_name="target_manifest",
    )
    if str(target_manifest.get("status", "")) != "target_frozen":
        raise ValueError("feature construction requires a frozen valid target")
    if str(target_manifest.get("study_contract_sha256", "")) != str(
        study["contract_sha256"]
    ):
        raise ValueError("target artifact belongs to a different study contract")
    candidate_path = _resolve_path(str(manifest["candidate_index_path"]))
    candidate_count = int(pq.ParquetFile(candidate_path).metadata.num_rows)
    if int(target_manifest.get("candidate_count", -1)) != candidate_count:
        raise ValueError("target and candidate index row counts differ")

    continuous_catalog = continuous_feature_catalog()
    categorical_catalog = categorical_feature_catalog()
    recovered_attempt = _recoverable_feature_publish_attempt(
        output_root,
        candidate_count=candidate_count,
        continuous_feature_count=len(continuous_catalog),
        categorical_feature_count=len(categorical_catalog),
    )
    attempt_dir = (
        recovered_attempt
        if recovered_attempt is not None
        else _next_attempt_dir(output_root, "feature_view")
    )
    progress_path = attempt_dir / "progress.json"
    partial = attempt_dir / "partial"
    if recovered_attempt is None:
        partial.mkdir(parents=True, exist_ok=False)
    paths = _feature_view_paths(partial)
    final_paths = _feature_view_paths(attempt_dir)
    continuous: np.memmap | None = None
    categorical: np.memmap | None = None
    candidate_date_idx: np.memmap | None = None
    candidate_symbol_idx: np.memmap | None = None
    try:
        date_values = np.asarray(
            [str(item) for item in list(manifest.get("date_values", []) or [])],
            dtype=object,
        )
        if recovered_attempt is None:
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
            candidate_date_idx, candidate_symbol_idx = _write_candidate_coordinates(
                candidate_path,
                paths,
                candidate_count=candidate_count,
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
            del writer
            _close_memmap(continuous)
            _close_memmap(categorical)
            _close_memmap(candidate_date_idx)
            _close_memmap(candidate_symbol_idx)
            continuous = None
            categorical = None
            candidate_date_idx = None
            candidate_symbol_idx = None
            del index_membership, industry, industry_broad, industry_source_age
            _trim_after_feature_stage()
        else:
            coordinate_path = (
                final_paths.candidate_date_idx
                if final_paths.candidate_date_idx.exists()
                else paths.candidate_date_idx
            )
            candidate_date_idx = np.memmap(
                coordinate_path,
                dtype="int32",
                mode="r",
                shape=(candidate_count,),
            )
            candidate_dates = date_values[np.asarray(candidate_date_idx, dtype=np.int64)]
            candidate_allowed = candidate_dates <= "2025-12-31"
            _close_memmap(candidate_date_idx)
            candidate_date_idx = None
            f1_f2 = {
                "f1_feature_count": len(f1_feature_names()),
                "f2_feature_count": len(f2_feature_names()),
                "causal_max_lookback_days": max(FEATURE_WINDOWS),
            }
            f3 = {
                "groups": list(MARKET_GROUPS),
                "metrics": list(MARKET_METRICS),
                "cross_section_mask": "pit_universe_has_bar_and_optional_same_day_index_membership",
            }
            f4 = {
                "aggregation_universe": "same_day_pit_universe_has_bar",
                "unknown_industry_hash": 0,
            }
            f5 = {
                "valuation_source": "pinned_qdp_valuation_same_day",
                "share_source_guard": "source_date_lte_signal_date",
                "category_storage": "stable_hash_then_fold_train_vocabulary",
                "neural_missing_mask": "generated_from_nonfinite_continuous_values_at_batch_load",
            }
            _atomic_write_json(
                progress_path,
                {
                    "status": "recovering_publish",
                    "updated_at": _now(),
                    "study_contract_sha256": study["contract_sha256"],
                    "candidate_count": candidate_count,
                },
            )

        for source, destination in (
            (paths.continuous, final_paths.continuous),
            (paths.categorical, final_paths.categorical),
            (paths.candidate_date_idx, final_paths.candidate_date_idx),
            (paths.candidate_symbol_idx, final_paths.candidate_symbol_idx),
        ):
            if source.exists():
                if destination.exists():
                    raise FileExistsError(
                        f"feature file exists in both partial and final locations: {source.name}"
                    )
                os.replace(source, destination)
            elif not destination.exists():
                raise FileNotFoundError(source)
        if partial.exists():
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
            "artifact_type": "seq100_signal_quality_feature_view",
            "schema_version": 1,
            "status": "completed",
            "created_at": _now(),
            "study_contract": str(study_path.resolve()),
            "study_contract_sha256": study["contract_sha256"],
            "research_freeze_sha256": freeze_payload["freeze_sha256"],
            "feature_spec": frozen_features,
            "feature_spec_sha256": _canonical_json_sha256(frozen_features),
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
            _close_memmap(continuous)
        if categorical is not None:
            _close_memmap(categorical)
        if candidate_date_idx is not None:
            _close_memmap(candidate_date_idx)
        if candidate_symbol_idx is not None:
            _close_memmap(candidate_symbol_idx)
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


@dataclass
class ModelDataContext:
    study: dict[str, Any]
    research_freeze: dict[str, Any]
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


def _load_feature_freeze(
    output_root: Path,
    study: Mapping[str, Any],
) -> tuple[Path, dict[str, Any]]:
    path, payload = _load_current_artifact(
        output_root,
        task="feature_screen",
        pointer_name="feature_freeze",
    )
    if str(payload.get("status", "")) != "completed":
        raise ValueError("feature screen is not complete")
    if str(payload.get("study_contract_sha256", "")) != str(
        study["contract_sha256"]
    ):
        raise ValueError("feature freeze belongs to a different study contract")
    declared = str(payload.get("feature_freeze_sha256", ""))
    check = dict(payload)
    check.pop("feature_freeze_sha256", None)
    if declared != _canonical_json_sha256(check):
        raise ValueError("feature freeze hash mismatch")
    return path, payload


def _load_formal_matrix(
    output_root: Path,
    study: Mapping[str, Any],
) -> tuple[Path, dict[str, Any]]:
    path, payload = _load_current_artifact(
        output_root,
        task="model_screen",
        pointer_name="formal_matrix",
    )
    if str(payload.get("status", "")) != "formal_matrix_frozen":
        raise ValueError("formal model matrix is not frozen")
    if str(payload.get("study_contract_sha256", "")) != str(
        study["contract_sha256"]
    ):
        raise ValueError("formal matrix belongs to a different study contract")
    declared = str(payload.get("formal_matrix_sha256", ""))
    check = dict(payload)
    check.pop("formal_matrix_sha256", None)
    if declared != _canonical_json_sha256(check):
        raise ValueError("formal matrix hash mismatch")
    return path, payload


def _resolve_model_data(
    *,
    study_path: Path,
    output_root: Path,
) -> ModelDataContext:
    study = load_study(study_path)
    freeze_payload = load_research_freeze(study)
    training_budget_amendment = _load_training_budget_amendment(
        study,
        output_root,
        research_freeze=freeze_payload["freeze"],
    )
    _pack_path, pack_manifest = _validate_source_bindings(study)
    _feature_path, feature_manifest = _load_feature_view_bundle(output_root, study)
    target_manifest, target, _counts, grades, flags = _load_target_bundle(
        output_root, study
    )
    continuous, categorical, candidate_date_idx, candidate_symbol_idx = (
        _open_feature_arrays(feature_manifest)
    )
    candidate_count = int(feature_manifest["candidate_count"])
    if int(target_manifest["candidate_count"]) != candidate_count:
        raise ValueError("model target/feature candidate counts differ")
    candidate_index_path = _resolve_path(str(pack_manifest["candidate_index_path"]))
    if int(pq.ParquetFile(candidate_index_path).metadata.num_rows) != candidate_count:
        raise ValueError("candidate index row count changed")
    return ModelDataContext(
        study=study,
        research_freeze=dict(freeze_payload["freeze"]),
        pack_manifest=pack_manifest,
        feature_manifest=feature_manifest,
        target_manifest=target_manifest,
        continuous=continuous,
        categorical=categorical,
        candidate_date_idx=candidate_date_idx,
        candidate_symbol_idx=candidate_symbol_idx,
        target=target,
        grades=grades,
        flags=flags,
        candidate_index_path=candidate_index_path,
        training_budget_amendment=training_budget_amendment,
    )


def _model_spec(context: ModelDataContext, model_id: str) -> dict[str, Any]:
    for item in list(context.research_freeze.get("model_candidates", []) or []):
        if str(item.get("model_id", "")) == str(model_id):
            spec = dict(item)
            amendment = context.training_budget_amendment
            if amendment is not None and str(model_id) in set(amendment["model_ids"]):
                config = dict(spec.get("config", {}) or {})
                for field, change in dict(amendment["config_changes"]).items():
                    if config.get(field) != change["from"]:
                        raise ValueError(
                            f"training budget base changed for {model_id}.{field}"
                        )
                    config[field] = change["to"]
                spec["config"] = config
                spec["training_budget_amendment_sha256"] = amendment[
                    "amendment_sha256"
                ]
            return spec
    raise KeyError(f"model is not in the frozen shortlist: {model_id}")


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


def _prefetch_iterator(
    source: Iterable[Any],
    *,
    depth: int,
) -> Iterator[Any]:
    """Bounded one-producer prefetch that propagates loader exceptions."""

    if int(depth) <= 0:
        yield from source
        return
    queue: Queue[tuple[str, Any]] = Queue(maxsize=int(depth))
    stop = threading.Event()

    def _produce() -> None:
        try:
            for item in source:
                if stop.is_set():
                    break
                while not stop.is_set():
                    try:
                        queue.put(("item", item), timeout=0.1)
                        break
                    except Exception:
                        continue
        except BaseException as exc:  # propagated in the consumer thread
            while not stop.is_set():
                try:
                    queue.put(("error", exc), timeout=0.1)
                    break
                except Exception:
                    continue
        finally:
            while not stop.is_set():
                try:
                    queue.put(("end", None), timeout=0.1)
                    break
                except Exception:
                    continue

    worker = threading.Thread(target=_produce, name="seq100-batch-prefetch", daemon=True)
    worker.start()
    try:
        while True:
            try:
                kind, value = queue.get(timeout=0.2)
            except Empty:
                if not worker.is_alive():
                    break
                continue
            if kind == "item":
                yield value
            elif kind == "error":
                raise value
            else:
                break
    finally:
        stop.set()
        worker.join(timeout=5.0)


def _deterministic_complete_date_sample(
    row_ids: np.ndarray,
    candidate_date_idx: np.ndarray,
    *,
    max_rows: int,
    seed: int = 1729,
) -> np.ndarray:
    groups = _row_groups(row_ids, candidate_date_idx)
    prioritized: list[tuple[int, np.ndarray]] = []
    for group in groups:
        date_idx = int(candidate_date_idx[int(group[0])])
        digest = hashlib.blake2b(
            f"{int(seed)}:{date_idx}".encode("utf-8"), digest_size=8
        ).digest()
        prioritized.append(
            (int.from_bytes(digest, byteorder="big", signed=False), group)
        )
    selected: list[np.ndarray] = []
    count = 0
    for _priority, group in sorted(prioritized, key=lambda item: item[0]):
        if selected and count + len(group) > int(max_rows):
            continue
        selected.append(group)
        count += int(len(group))
        if count >= int(max_rows):
            break
    if not selected and groups:
        selected.append(groups[0])
    if not selected:
        return np.empty(0, dtype=np.int64)
    return np.sort(np.concatenate(selected).astype(np.int64, copy=False))


def _fit_snapshot_preprocessor(
    context: ModelDataContext,
    train_rows: np.ndarray,
    *,
    profile: str,
    output_dir: Path,
    normalization_max_rows: int = 250_000,
    required_categorical_names: Sequence[str] = (),
) -> dict[str, Any]:
    (
        continuous_columns,
        categorical_columns,
        continuous_names,
        categorical_names,
    ) = _feature_columns_for_profile(context.feature_manifest, profile)
    categorical_columns = np.asarray(categorical_columns, dtype=np.int32)
    categorical_names = list(categorical_names)
    categorical_catalog = {
        str(item["name"]): int(item["column_index"])
        for item in list(context.feature_manifest["categorical_catalog"])
    }
    for name in required_categorical_names:
        key = str(name)
        if key not in categorical_catalog:
            raise KeyError(f"required categorical feature is unavailable: {key}")
        if key not in categorical_names:
            categorical_names.append(key)
            categorical_columns = np.append(
                categorical_columns, np.int32(categorical_catalog[key])
            )
    vocabularies = _fit_category_vocabularies(
        context.categorical,
        np.asarray(train_rows, dtype=np.int64),
        categorical_columns,
    )
    sample_rows = _deterministic_complete_date_sample(
        train_rows,
        context.candidate_date_idx,
        max_rows=int(normalization_max_rows),
    )
    if not sample_rows.size:
        raise ValueError("snapshot normalization has no training rows")
    sample = np.asarray(
        context.continuous[np.ix_(sample_rows, continuous_columns)],
        dtype=np.float32,
    )
    with np.errstate(all="ignore"):
        median = np.nanmedian(sample, axis=0).astype(np.float32)
        q25 = np.nanquantile(sample, 0.25, axis=0).astype(np.float32)
        q75 = np.nanquantile(sample, 0.75, axis=0).astype(np.float32)
    iqr = q75 - q25
    median[~np.isfinite(median)] = 0.0
    iqr[~np.isfinite(iqr) | (iqr < 1.0e-6)] = 1.0
    payload = {
        "profile": str(profile),
        "continuous_columns": continuous_columns.tolist(),
        "categorical_columns": categorical_columns.tolist(),
        "continuous_names": continuous_names,
        "categorical_names": categorical_names,
        "median": median.tolist(),
        "iqr": iqr.tolist(),
        "clip": 10.0,
        "missing_mask_appended": True,
        "normalization_sample_rule": "deterministic_complete_signal_dates_blake2b_seed1729",
        "normalization_sample_count": int(len(sample_rows)),
        "normalization_max_rows": int(normalization_max_rows),
        "category_vocabularies": [values.tolist() for values in vocabularies],
        "fit_candidate_id_min": int(np.min(train_rows)),
        "fit_candidate_id_max": int(np.max(train_rows)),
    }
    payload["preprocessor_sha256"] = _canonical_json_sha256(payload)
    _atomic_write_json(output_dir / "snapshot_preprocessor.json", payload)
    del sample
    return payload


def _load_or_fit_snapshot_preprocessor(
    context: ModelDataContext,
    train_rows: np.ndarray,
    *,
    profile: str,
    output_dir: Path,
    normalization_max_rows: int = 250_000,
    required_categorical_names: Sequence[str] = (),
) -> dict[str, Any]:
    path = output_dir / "snapshot_preprocessor.json"
    if path.exists():
        payload = json.loads(path.read_text(encoding="utf-8"))
        declared = str(payload.get("preprocessor_sha256", ""))
        check = dict(payload)
        check.pop("preprocessor_sha256", None)
        if declared != _canonical_json_sha256(check):
            raise ValueError("snapshot preprocessor hash mismatch")
        if str(payload.get("profile", "")) != str(profile):
            raise ValueError("snapshot preprocessor profile changed on resume")
        missing = set(str(item) for item in required_categorical_names) - set(
            str(item) for item in payload.get("categorical_names", [])
        )
        if missing:
            raise ValueError(
                f"snapshot preprocessor lacks required categories: {sorted(missing)}"
            )
        return payload
    return _fit_snapshot_preprocessor(
        context,
        train_rows,
        profile=str(profile),
        output_dir=output_dir,
        normalization_max_rows=int(normalization_max_rows),
        required_categorical_names=required_categorical_names,
    )


def _snapshot_batch(
    context: ModelDataContext,
    row_ids: np.ndarray,
    preprocessor: Mapping[str, Any],
) -> tuple[np.ndarray, np.ndarray]:
    rows = np.asarray(row_ids, dtype=np.int64)
    continuous_columns = np.asarray(
        preprocessor["continuous_columns"], dtype=np.int32
    )
    categorical_columns = np.asarray(
        preprocessor["categorical_columns"], dtype=np.int32
    )
    values = np.asarray(
        context.continuous[np.ix_(rows, continuous_columns)], dtype=np.float32
    )
    missing = ~np.isfinite(values)
    median = np.asarray(preprocessor["median"], dtype=np.float32)
    iqr = np.asarray(preprocessor["iqr"], dtype=np.float32)
    normalized = np.clip((values - median) / iqr, -10.0, 10.0)
    normalized[missing] = 0.0
    x_num = np.concatenate(
        [normalized, missing.astype(np.float32)], axis=1
    ).astype(np.float32, copy=False)
    if categorical_columns.size:
        raw_categories = np.asarray(
            context.categorical[np.ix_(rows, categorical_columns)],
            dtype=np.int64,
        )
        mapped = [
            _map_categories(
                raw_categories[:, idx],
                np.asarray(vocabulary, dtype=np.int64),
            ).reshape(-1, 1)
            for idx, vocabulary in enumerate(preprocessor["category_vocabularies"])
        ]
        x_cat = np.concatenate(mapped, axis=1).astype(np.int64, copy=False)
    else:
        x_cat = np.empty((len(rows), 0), dtype=np.int64)
    return x_num, x_cat


def _load_f0_preprocessor(fold: Mapping[str, Any]) -> dict[str, Any]:
    path = Path(str(fold["f0_normalization_view"])).resolve()
    payload = json.loads(path.read_text(encoding="utf-8"))
    normalization = dict(payload["normalization"])
    means: list[float] = []
    stds: list[float] = []
    for name in ("daily_raw", "daily_state", "turnover"):
        means.extend(float(item) for item in normalization[name]["mean"])
        stds.extend(float(item) for item in normalization[name]["std"])
    means.append(0.0)
    stds.append(1.0)
    result = {
        "normalization_view": str(path),
        "normalization_view_sha256": _file_sha256(path),
        "fit_date_start": normalization["fit_date_start"],
        "fit_date_end": normalization["fit_date_end"],
        "mean": means,
        "std": stds,
        "input_dim": len(means),
    }
    result["preprocessor_sha256"] = _canonical_json_sha256(result)
    return result


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
        output[positions] = np.concatenate(parts, axis=2).astype(
            np.float32, copy=False
        )
    output = (output - mean.reshape(1, 1, -1)) / std.reshape(1, 1, -1)
    return np.nan_to_num(
        output,
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    ).astype(np.float32, copy=False)


def _iter_f0_batches(
    context: ModelDataContext,
    row_ids: np.ndarray,
    preprocessor: Mapping[str, Any],
    *,
    batch_size: int,
    shuffle_dates: bool,
    seed: int,
    cross_date: bool = False,
) -> Iterator[tuple[np.ndarray, np.ndarray]]:
    material = _open_reference_f0_material(context.pack_manifest)
    if cross_date:
        if shuffle_dates:
            raise ValueError("cross-date F0 batches cannot use date shuffling")
        rows = np.asarray(row_ids, dtype=np.int64)
        for start in range(0, len(rows), int(batch_size)):
            current = rows[start : start + int(batch_size)]
            yield current, _f0_batch(
                context,
                current,
                preprocessor,
                material=material,
            )
        return
    groups = _row_groups(row_ids, context.candidate_date_idx)
    rng = np.random.default_rng(int(seed))
    if shuffle_dates:
        rng.shuffle(groups)
    for group in groups:
        rows = rng.permutation(group) if shuffle_dates else group
        for start in range(0, len(rows), int(batch_size)):
            current = np.asarray(rows[start : start + int(batch_size)], dtype=np.int64)
            yield current, _f0_batch(
                context,
                current,
                preprocessor,
                material=material,
            )


def _iter_snapshot_batches(
    context: ModelDataContext,
    row_ids: np.ndarray,
    preprocessor: Mapping[str, Any],
    *,
    batch_size: int,
    shuffle_dates: bool,
    seed: int,
    cross_date: bool = False,
) -> Iterator[tuple[np.ndarray, np.ndarray, np.ndarray]]:
    if cross_date:
        if shuffle_dates:
            raise ValueError("cross-date snapshot batches cannot use date shuffling")
        rows = np.asarray(row_ids, dtype=np.int64)
        for start in range(0, len(rows), int(batch_size)):
            current = rows[start : start + int(batch_size)]
            x_num, x_cat = _snapshot_batch(context, current, preprocessor)
            yield current, x_num, x_cat
        return
    groups = _row_groups(row_ids, context.candidate_date_idx)
    rng = np.random.default_rng(int(seed))
    if shuffle_dates:
        rng.shuffle(groups)
    for group in groups:
        rows = rng.permutation(group) if shuffle_dates else group
        for start in range(0, len(rows), int(batch_size)):
            current = np.asarray(rows[start : start + int(batch_size)], dtype=np.int64)
            x_num, x_cat = _snapshot_batch(context, current, preprocessor)
            yield current, x_num, x_cat


def _target_material(
    context: ModelDataContext,
    row_ids: np.ndarray,
) -> dict[str, np.ndarray]:
    rows = np.asarray(row_ids, dtype=np.int64)
    flags = np.asarray(context.flags[rows], dtype=np.uint8)
    filled = (flags & TARGET_FLAG_ENTRY_FILLED) != 0
    path_available = (flags & TARGET_FLAG_PATH_AVAILABLE) != 0
    terminal = (flags & TARGET_FLAG_TERMINAL_FAILURE) != 0
    close_columns = np.asarray(
        [TARGET_FIELD_INDEX[f"close_net_log_d{day:02d}"] for day in range(1, 21)],
        dtype=np.int64,
    )
    path = np.asarray(
        context.target[np.ix_(rows, close_columns)], dtype=np.float32
    )
    return {
        "filled": filled.astype(np.float32),
        "path_available": path_available,
        "terminal": terminal.astype(np.float32),
        "conditional_quality": np.asarray(
            context.target[rows, TARGET_FIELD_INDEX["conditional_relevance"]],
            dtype=np.float32,
        ),
        "action_quality": np.asarray(
            context.target[rows, TARGET_FIELD_INDEX["action_relevance"]],
            dtype=np.float32,
        ),
        "grade": np.asarray(context.grades[rows], dtype=np.uint8),
        "path": path,
        "event_code": np.asarray(
            context.target[rows, TARGET_FIELD_INDEX["competing_event_code"]],
            dtype=np.float32,
        ),
        "event_time": np.asarray(
            context.target[rows, TARGET_FIELD_INDEX["competing_event_time"]],
            dtype=np.float32,
        ),
    }


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

    def forward(self, x_num: torch.Tensor, x_cat: torch.Tensor) -> dict[str, torch.Tensor]:
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
        channel = self.channel.expand(batch, -1, -1).reshape(
            batch * channels, 1, -1
        )
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
        self.market = nn.Sequential(
            nn.Linear(width * 2, context_width), nn.ReLU()
        )
        self.industry = nn.Sequential(nn.Linear(width, context_width), nn.ReLU())
        self.residual = nn.Sequential(nn.Linear(width, context_width), nn.ReLU())
        self.gate = nn.Sequential(nn.Linear(context_width, context_width), nn.Sigmoid())
        self.quality = nn.Linear(context_width, 1)
        self.fill = nn.Linear(context_width, 1)

    def forward(self, x_num: torch.Tensor, x_cat: torch.Tensor) -> dict[str, torch.Tensor]:
        item = self.encoder(x_num, x_cat)
        market_mean = item.mean(dim=0, keepdim=True)
        market_max = item.max(dim=0, keepdim=True).values
        market_context = self.market(torch.cat([market_mean, market_max], dim=1))
        industry_id = x_cat[:, self.industry_position]
        _unique, inverse = torch.unique(
            industry_id, sorted=True, return_inverse=True
        )
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

    def forward(self, x_num: torch.Tensor, x_cat: torch.Tensor) -> dict[str, torch.Tensor]:
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

    def forward(self, x_num: torch.Tensor, x_cat: torch.Tensor) -> dict[str, torch.Tensor]:
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
    return log_normalizer + ((nu + 1.0) / 2.0) * torch.log1p(
        standardized.square() / nu
    )


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


def _category_cardinalities(preprocessor: Mapping[str, Any]) -> list[int]:
    return [
        int(len(values)) + 1
        for values in list(preprocessor.get("category_vocabularies", []) or [])
    ]


def _torch_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _clone_torch_state(model: nn.Module) -> dict[str, torch.Tensor]:
    return {
        name: value.detach().cpu().clone()
        for name, value in model.state_dict().items()
    }


def _atomic_torch_save(path: Path, payload: Mapping[str, Any]) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    torch.save(dict(payload), temporary)
    os.replace(temporary, path)
    return str(path.resolve())


def _torch_parameter_count(model: nn.Module) -> int:
    return int(sum(parameter.numel() for parameter in model.parameters()))


def _hardware_fingerprint() -> dict[str, Any]:
    gpu: dict[str, Any] | None = None
    if torch.cuda.is_available():
        properties = torch.cuda.get_device_properties(0)
        gpu = {
            "name": str(properties.name),
            "total_memory_bytes": int(properties.total_memory),
            "capability": [int(properties.major), int(properties.minor)],
        }
    payload = {
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "logical_cpu_count": int(os.cpu_count() or 1),
        "torch_version": str(torch.__version__),
        "torch_cuda_version": str(torch.version.cuda or ""),
        "cuda_available": bool(torch.cuda.is_available()),
        "gpu": gpu,
    }
    payload["hardware_sha256"] = _canonical_json_sha256(payload)
    return payload


def _adamw_optimizer(
    model: nn.Module,
    config: Mapping[str, Any],
    *,
    device: torch.device,
) -> tuple[torch.optim.Optimizer, bool]:
    kwargs = {
        "lr": float(config["learning_rate"]),
        "weight_decay": float(config["weight_decay"]),
    }
    if device.type == "cuda":
        try:
            return torch.optim.AdamW(model.parameters(), fused=True, **kwargs), True
        except (RuntimeError, TypeError):
            pass
    return torch.optim.AdamW(model.parameters(), **kwargs), False


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
        if (
            available_after is None
            or available_after >= LOW_MEMORY_PAUSE_AVAILABLE_GB
        ):
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
        while (
            len(self.rate_window) > 2
            and now - self.rate_window[0][0] > 180.0
        ):
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
        "torch_cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
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
            "resume_key_sha256": str(resolved_config["resume_key_sha256"]),
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
    if str(payload.get("resume_key_sha256", "")) != str(
        resolved_config["resume_key_sha256"]
    ):
        raise ValueError("training checkpoint resume key mismatch")
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
        raise TrainingPaused(
            f"training paused safely at {checkpoint_path.resolve()}"
        )


def _torch_runtime(config: Mapping[str, Any]) -> tuple[torch.device, bool, Any]:
    device = _torch_device()
    amp_enabled = bool(config.get("amp", True) and device.type == "cuda")
    scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)
    return device, amp_enabled, scaler


def _save_torch_adapter(
    *,
    output_dir: Path,
    model: nn.Module,
    model_id: str,
    model_config: Mapping[str, Any],
    resolved_config: Mapping[str, Any],
    preprocessor: Mapping[str, Any],
    best_epoch: int,
    best_development_loss: float,
    training_log: Sequence[Mapping[str, Any]],
    extra: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    checkpoint = output_dir / "best_model.pt"
    _atomic_torch_save(
        checkpoint,
        {
            "model_id": str(model_id),
            "model_state_dict": model.state_dict(),
            "model_config": dict(model_config),
            "resolved_config": dict(resolved_config),
            "preprocessor": dict(preprocessor),
            "best_epoch": int(best_epoch),
            "best_development_loss": float(best_development_loss),
            "extra": dict(extra or {}),
        },
    )
    _atomic_write_json(output_dir / "training_log.json", list(training_log))
    evidence = {
        "checkpoint_path": str(checkpoint.resolve()),
        "checkpoint_sha256": _file_sha256(checkpoint),
        "parameter_count": _torch_parameter_count(model),
        "checkpoint_size_bytes": int(checkpoint.stat().st_size),
        "best_epoch": int(best_epoch),
        "best_development_loss": float(best_development_loss),
        "training_log": str((output_dir / "training_log.json").resolve()),
    }
    last_checkpoint = output_dir / "last_checkpoint.pt"
    if last_checkpoint.exists():
        evidence["last_checkpoint_path"] = str(last_checkpoint.resolve())
        evidence["last_checkpoint_sha256"] = _file_sha256(last_checkpoint)
    events = output_dir / "training_events.jsonl"
    if events.exists():
        evidence["training_events_path"] = str(events.resolve())
    return evidence


def _fit_lgbm_adapter(
    *,
    context: ModelDataContext,
    train_rows: np.ndarray,
    development_rows: np.ndarray,
    resolved_config: Mapping[str, Any],
    output_dir: Path,
    seed: int,
) -> dict[str, Any]:
    profile = str(resolved_config["feature_profile"])
    (
        continuous_columns,
        categorical_columns,
        continuous_names,
        categorical_names,
    ) = _feature_columns_for_profile(context.feature_manifest, profile)
    vocabularies = _fit_category_vocabularies(
        context.categorical,
        train_rows,
        categorical_columns,
    )
    feature_names = [*continuous_names, *categorical_names]
    filled = (np.asarray(context.flags, dtype=np.uint8) & TARGET_FLAG_ENTRY_FILLED) != 0
    conditional = np.asarray(
        context.target[:, TARGET_FIELD_INDEX["conditional_relevance"]],
        dtype=np.float32,
    )
    rank_train_rows = train_rows[
        filled[train_rows]
        & np.isfinite(conditional[train_rows])
        & (np.asarray(context.grades[train_rows], dtype=np.uint8) < 5)
    ]
    rank_development_rows = development_rows[
        filled[development_rows]
        & np.isfinite(conditional[development_rows])
        & (np.asarray(context.grades[development_rows], dtype=np.uint8) < 5)
    ]
    if not rank_train_rows.size or not rank_development_rows.size:
        raise ValueError("LightGBM adapter lacks conditional-quality rows")

    def _sequence(rows: np.ndarray) -> Any:
        return _make_lgb_sequence(
            continuous=context.continuous,
            categorical=context.categorical,
            row_ids=rows,
            continuous_columns=continuous_columns,
            categorical_columns=categorical_columns,
            category_vocabularies=vocabularies,
        )

    fill_model, fill_evidence = _train_lgb_model(
        study=context.study,
        model_id="lgbm_lambdarank_multioutput",
        objective="binary",
        train_sequence=_sequence(train_rows),
        train_label=filled[train_rows].astype(np.uint8),
        development_sequence=_sequence(development_rows),
        development_label=filled[development_rows].astype(np.uint8),
        feature_names=feature_names,
        categorical_count=len(categorical_names),
        output_path=output_dir / "fill_model.txt",
        seed=int(seed),
    )
    rank_model, rank_evidence = _train_lgb_model(
        study=context.study,
        model_id="lgbm_lambdarank_multioutput",
        objective="lambdarank",
        train_sequence=_sequence(rank_train_rows),
        train_label=np.asarray(context.grades[rank_train_rows], dtype=np.uint8),
        development_sequence=_sequence(rank_development_rows),
        development_label=np.asarray(
            context.grades[rank_development_rows], dtype=np.uint8
        ),
        feature_names=feature_names,
        categorical_count=len(categorical_names),
        output_path=output_dir / "quality_model.txt",
        train_group=_date_group_sizes(context.candidate_date_idx[rank_train_rows]),
        development_group=_date_group_sizes(
            context.candidate_date_idx[rank_development_rows]
        ),
        seed=int(seed),
    )
    raw_development = _predict_lgb_sequence(
        rank_model, _sequence(rank_development_rows)
    )
    isotonic, isotonic_payload = _fit_isotonic_mapping(
        raw_development,
        conditional[rank_development_rows],
    )
    _atomic_write_json(output_dir / "isotonic.json", isotonic_payload)
    preprocessor = {
        "profile": profile,
        "continuous_columns": continuous_columns.tolist(),
        "categorical_columns": categorical_columns.tolist(),
        "continuous_names": continuous_names,
        "categorical_names": categorical_names,
        "category_vocabularies": [values.tolist() for values in vocabularies],
    }
    preprocessor["preprocessor_sha256"] = _canonical_json_sha256(preprocessor)
    _atomic_write_json(output_dir / "tree_preprocessor.json", preprocessor)
    artifact = {
        "adapter_type": "lightgbm_fill_plus_lambdarank",
        "model_id": "lgbm_lambdarank_multioutput",
        "capabilities": list(LightGBMAdapter.capabilities),
        "fill_model": fill_evidence,
        "quality_model": rank_evidence,
        "isotonic": isotonic_payload,
        "preprocessor": preprocessor,
        "parameter_count": int(
            fill_model.num_trees() + rank_model.num_trees()
        ),
        "checkpoint_size_bytes": int(
            Path(fill_evidence["model_path"]).stat().st_size
            + Path(rank_evidence["model_path"]).stat().st_size
        ),
        "resolved_config_sha256": str(resolved_config["resolved_config_sha256"]),
    }
    _atomic_write_json(output_dir / "adapter_manifest.json", artifact)
    return {
        "artifact": artifact,
        "handle": {
            "fill_model": fill_model,
            "quality_model": rank_model,
            "isotonic": isotonic,
            "preprocessor": preprocessor,
        },
    }


def _predict_lgbm_adapter(
    *,
    context: ModelDataContext,
    row_ids: np.ndarray,
    fitted: Mapping[str, Any],
) -> dict[str, np.ndarray]:
    handle = dict(fitted["handle"])
    preprocessor = dict(handle["preprocessor"])
    rows = np.asarray(row_ids, dtype=np.int64)
    sequence = _make_lgb_sequence(
        continuous=context.continuous,
        categorical=context.categorical,
        row_ids=rows,
        continuous_columns=np.asarray(preprocessor["continuous_columns"], dtype=np.int32),
        categorical_columns=np.asarray(preprocessor["categorical_columns"], dtype=np.int32),
        category_vocabularies=[
            np.asarray(values, dtype=np.int64)
            for values in preprocessor["category_vocabularies"]
        ],
    )
    fill = np.clip(
        _predict_lgb_sequence(handle["fill_model"], sequence), 0.0, 1.0
    )
    raw = _predict_lgb_sequence(handle["quality_model"], sequence)
    quality = np.asarray(handle["isotonic"].predict(raw), dtype=np.float32)
    return {
        "fill_probability": fill.astype(np.float32),
        "conditional_quality_prediction": quality,
        "selection_score": (fill * quality).astype(np.float32),
    }


def _fit_tabm_adapter(
    *,
    context: ModelDataContext,
    train_rows: np.ndarray,
    development_rows: np.ndarray,
    resolved_config: Mapping[str, Any],
    output_dir: Path,
    seed: int,
    monitor: _TrainingMonitor | None = None,
) -> dict[str, Any]:
    config = dict(_model_spec(context, "tabm_multioutput")["config"])
    preprocessor = _load_or_fit_snapshot_preprocessor(
        context,
        train_rows,
        profile=str(resolved_config["feature_profile"]),
        output_dir=output_dir,
    )
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))
    device, amp_enabled, scaler = _torch_runtime(config)
    model = _TabMModel(
        n_num_features=2 * len(preprocessor["continuous_columns"]),
        cat_cardinalities=_category_cardinalities(preprocessor),
        config=config,
    ).to(device)
    optimizer, fused_optimizer = _adamw_optimizer(
        model,
        config,
        device=device,
    )
    micro_batch = int(resolved_config["micro_batch"])
    effective_batch = int(resolved_config["effective_batch"])
    validation_batch = int(resolved_config.get("validation_batch", effective_batch))
    prefetch_depth = int(resolved_config.get("prefetch_depth", 0))
    quality_weight = float(config["loss_weights"]["quality_huber"])
    fill_weight = float(config["loss_weights"]["fill_bce"])
    members = int(config["ensemble_size"])

    def _micro_loss(
        rows: np.ndarray,
        x_num: np.ndarray,
        x_cat: np.ndarray,
        *,
        quality_denominator: float,
        fill_denominator: float,
    ) -> torch.Tensor:
        material = _target_material(context, rows)
        tensor_num = torch.from_numpy(x_num).to(device)
        tensor_cat = torch.from_numpy(x_cat).to(device)
        quality = torch.from_numpy(
            np.nan_to_num(material["conditional_quality"], nan=0.0)
        ).to(device)
        quality_mask = torch.from_numpy(
            (np.isfinite(material["conditional_quality"]) & (material["filled"] > 0.5)).astype(np.float32)
        ).to(device)
        fill = torch.from_numpy(material["filled"]).to(device)
        with torch.autocast(device.type, dtype=torch.float16, enabled=amp_enabled):
            output = model(tensor_num, tensor_cat)
            qloss = torch_functional.smooth_l1_loss(
                output["quality"], quality[:, None].expand(-1, members), reduction="none"
            )
            qloss = global_normalized_component(
                (qloss * quality_mask[:, None]).sum(),
                quality_denominator,
            )
            floss = global_normalized_component(
                torch_functional.binary_cross_entropy_with_logits(
                    output["fill_logit"],
                    fill[:, None].expand(-1, members),
                    reduction="sum",
                ),
                fill_denominator,
            )
            return quality_weight * qloss + fill_weight * floss

    def _effective_loss(rows: np.ndarray, *, backward: bool) -> torch.Tensor:
        material = _target_material(context, rows)
        valid_quality = np.isfinite(material["conditional_quality"]) & (
            material["filled"] > 0.5
        )
        quality_denominator = float(valid_quality.sum() * members)
        fill_denominator = float(len(rows) * members)
        total = torch.zeros((), dtype=torch.float32, device=device)
        current_micro_batch = micro_batch if backward else validation_batch

        def _source() -> Iterator[tuple[np.ndarray, np.ndarray, np.ndarray]]:
            for current in _micro_batches(rows, current_micro_batch):
                x_num, x_cat = _snapshot_batch(context, current, preprocessor)
                yield current, x_num, x_cat

        for current, x_num, x_cat in _prefetch_iterator(
            _source(), depth=prefetch_depth if backward else 0
        ):
            loss = _micro_loss(
                current,
                x_num,
                x_cat,
                quality_denominator=quality_denominator,
                fill_denominator=fill_denominator,
            )
            if backward:
                scaler.scale(loss).backward()
            total += loss.detach()
        return total

    def _development_loss(epoch: int) -> float:
        model.eval()
        total = torch.zeros((), dtype=torch.float32, device=device)
        count = 0
        batches = [
            development_rows[start : start + validation_batch]
            for start in range(0, len(development_rows), validation_batch)
        ]
        with torch.no_grad():
            for step_index, rows in enumerate(batches):
                loss = _effective_loss(rows, backward=False)
                total += loss * len(rows)
                count += len(rows)
                if monitor is not None:
                    monitor.report(
                        status="training",
                        phase="validation",
                        samples_completed=count,
                        samples_total=len(development_rows),
                        epoch=epoch,
                        max_epochs=int(config["max_epochs"]),
                        step=step_index + 1,
                        steps_total=len(batches),
                    )
        return float((total / max(count, 1)).cpu())

    best_loss = float("inf")
    best_epoch = 0
    best_state: dict[str, torch.Tensor] | None = None
    epochs_without_improvement = 0
    training_log: list[dict[str, Any]] = []
    start_epoch = 1
    resume_step = 0
    resumed_epoch_loss_sum = 0.0
    resumed_epoch_sample_count = 0
    resume = _load_last_training_checkpoint(
        output_dir=output_dir,
        model=model,
        optimizer=optimizer,
        scaler=scaler,
        resolved_config=resolved_config,
    )
    if resume is not None:
        start_epoch = int(resume["epoch"])
        resume_step = int(resume["next_step_index"])
        best_epoch = int(resume["best_epoch"])
        best_loss = float(resume["best_development_loss"])
        best_state = resume.get("best_state")
        epochs_without_improvement = int(resume["epochs_without_improvement"])
        training_log = [dict(item) for item in resume.get("training_log", [])]
        resumed_epoch_loss_sum = float(resume.get("epoch_loss_sum", 0.0))
        resumed_epoch_sample_count = int(resume.get("epoch_sample_count", 0))
    started = time.perf_counter()
    for epoch in range(start_epoch, int(config["max_epochs"]) + 1):
        batches = effective_batch_plan(
            train_rows,
            context.candidate_date_idx,
            effective_batch=effective_batch,
            seed=int(seed) + epoch,
        )
        first_step = resume_step if epoch == start_epoch else 0
        if first_step > len(batches):
            raise ValueError("TabM resume cursor exceeds the deterministic epoch plan")
        model.train()
        train_total = torch.as_tensor(
            resumed_epoch_loss_sum if epoch == start_epoch else 0.0,
            dtype=torch.float32,
            device=device,
        )
        train_count = resumed_epoch_sample_count if epoch == start_epoch else 0
        for step_index in range(first_step, len(batches)):
            rows = batches[step_index]
            optimizer.zero_grad(set_to_none=True)
            step_loss = _effective_loss(rows, backward=True)
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), float(config["gradient_clip_norm"])
            )
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
            train_total += step_loss * len(rows)
            train_count += len(rows)
            current_loss = (
                float((train_total / max(train_count, 1)).cpu())
                if monitor is not None and monitor.heartbeat_due()
                else None
            )
            progress = {
                "phase": "training",
                "samples_completed": train_count,
                "samples_total": len(train_rows),
                "epoch": epoch,
                "max_epochs": int(config["max_epochs"]),
                "step": step_index + 1,
                "steps_total": len(batches),
                "loss": current_loss,
                "best_development_loss": best_loss,
                "patience_used": epochs_without_improvement,
            }
            if monitor is not None:
                monitor.report(status="training", **progress)

            def _save_step() -> Path:
                return _save_last_training_checkpoint(
                    output_dir=output_dir,
                    model=model,
                    optimizer=optimizer,
                    scaler=scaler,
                    resolved_config=resolved_config,
                    epoch=epoch,
                    next_step_index=step_index + 1,
                    best_epoch=best_epoch,
                    best_development_loss=best_loss,
                    best_state=best_state,
                    epochs_without_improvement=epochs_without_improvement,
                    training_log=training_log,
                    epoch_loss_sum=float(train_total.cpu()),
                    epoch_sample_count=train_count,
                )

            _checkpoint_or_pause(
                monitor=monitor,
                force_checkpoint=False,
                checkpoint_callback=_save_step,
                progress=progress,
            )
        train_loss = float((train_total / max(train_count, 1)).cpu())
        development_loss = _development_loss(epoch)
        training_log.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "development_loss": development_loss,
            }
        )
        if development_loss < best_loss - 1.0e-8:
            best_loss = development_loss
            best_epoch = epoch
            best_state = _clone_torch_state(model)
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
        epoch_progress = {
            "phase": "validation",
            "samples_completed": len(development_rows),
            "samples_total": len(development_rows),
            "epoch": epoch,
            "max_epochs": int(config["max_epochs"]),
            "step": 1,
            "steps_total": 1,
            "loss": development_loss,
            "best_development_loss": best_loss,
            "patience_used": epochs_without_improvement,
        }
        if monitor is not None:
            monitor.report(
                status="training",
                force=True,
                event="epoch_end",
                **epoch_progress,
            )

        def _save_epoch() -> Path:
            return _save_last_training_checkpoint(
                output_dir=output_dir,
                model=model,
                optimizer=optimizer,
                scaler=scaler,
                resolved_config=resolved_config,
                epoch=epoch + 1,
                next_step_index=0,
                best_epoch=best_epoch,
                best_development_loss=best_loss,
                best_state=best_state,
                epochs_without_improvement=epochs_without_improvement,
                training_log=training_log,
            )

        _checkpoint_or_pause(
            monitor=monitor,
            force_checkpoint=True,
            checkpoint_callback=_save_epoch,
            progress=epoch_progress,
        )
        resume_step = 0
        resumed_epoch_loss_sum = 0.0
        resumed_epoch_sample_count = 0
        if (
            epoch >= int(config["min_epochs"])
            and epochs_without_improvement >= int(config["patience"])
        ):
            break
    if best_state is None:
        raise RuntimeError("TabM training did not produce a finite checkpoint")
    model.load_state_dict(best_state)
    evidence = _save_torch_adapter(
        output_dir=output_dir,
        model=model,
        model_id="tabm_multioutput",
        model_config=config,
        resolved_config=resolved_config,
        preprocessor=preprocessor,
        best_epoch=best_epoch,
        best_development_loss=best_loss,
        training_log=training_log,
        extra={"training_seconds": time.perf_counter() - started},
    )
    artifact = {
        "adapter_type": "official_tabm_multioutput",
        "model_id": "tabm_multioutput",
        "capabilities": list(TabMAdapter.capabilities),
        "fused_adamw": bool(fused_optimizer),
        "execution_semantics_version": resolved_config.get(
            "execution_semantics_version"
        ),
        "resolved_config_sha256": str(resolved_config["resolved_config_sha256"]),
        **evidence,
    }
    _atomic_write_json(output_dir / "adapter_manifest.json", artifact)
    return {
        "artifact": artifact,
        "handle": {
            "model": model,
            "preprocessor": preprocessor,
            "prediction_batch": int(resolved_config.get("prediction_batch", 4096)),
        },
    }


def _predict_tabm_adapter(
    *,
    context: ModelDataContext,
    row_ids: np.ndarray,
    fitted: Mapping[str, Any],
    batch_size: int | None = None,
    monitor: _TrainingMonitor | None = None,
) -> dict[str, np.ndarray]:
    model = fitted["handle"]["model"]
    preprocessor = fitted["handle"]["preprocessor"]
    device = next(model.parameters()).device
    rows_all = np.asarray(row_ids, dtype=np.int64)
    fill = np.empty(len(rows_all), dtype=np.float32)
    quality = np.empty(len(rows_all), dtype=np.float32)
    cursor = 0
    model.eval()
    with torch.no_grad():
        for rows, x_num, x_cat in _iter_snapshot_batches(
            context,
            rows_all,
            preprocessor,
            batch_size=int(batch_size or fitted["handle"].get("prediction_batch", 4096)),
            shuffle_dates=False,
            seed=0,
            cross_date=True,
        ):
            output = model(
                torch.from_numpy(x_num).to(device),
                torch.from_numpy(x_cat).to(device),
            )
            if not np.array_equal(rows, rows_all[cursor : cursor + len(rows)]):
                raise AssertionError("TabM prediction order changed")
            current = slice(cursor, cursor + len(rows))
            quality[current] = output["quality"].mean(dim=1).float().cpu().numpy()
            fill[current] = torch.sigmoid(output["fill_logit"]).mean(dim=1).float().cpu().numpy()
            cursor += len(rows)
            if monitor is not None:
                monitor.report(
                    status="predicting",
                    phase="prediction",
                    samples_completed=cursor,
                    samples_total=len(rows_all),
                )
    return {
        "fill_probability": fill,
        "conditional_quality_prediction": quality,
        "selection_score": (fill * quality).astype(np.float32),
    }


def _fit_patchtst_adapter(
    *,
    context: ModelDataContext,
    train_rows: np.ndarray,
    development_rows: np.ndarray,
    resolved_config: Mapping[str, Any],
    output_dir: Path,
    seed: int,
    monitor: _TrainingMonitor | None = None,
) -> dict[str, Any]:
    config = dict(_model_spec(context, "patchtst_student_t_path")["config"])
    fold = _load_fold_view(context.feature_manifest, resolved_config["fold_id"])
    preprocessor = _load_f0_preprocessor(fold)
    _atomic_write_json(output_dir / "f0_preprocessor.json", preprocessor)
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))
    device, amp_enabled, scaler = _torch_runtime(config)
    model = _PatchTSTStudentT(
        input_channels=int(preprocessor["input_dim"]), config=config
    ).to(device)
    optimizer, fused_optimizer = _adamw_optimizer(
        model,
        config,
        device=device,
    )
    micro_batch = int(resolved_config["micro_batch"])
    effective_batch = int(resolved_config["effective_batch"])
    validation_batch = int(resolved_config.get("validation_batch", effective_batch))
    prefetch_depth = int(resolved_config.get("prefetch_depth", 0))
    df = float(config["student_t_df"])
    path_weight = float(config["loss_weights"]["path_nll"])
    terminal_weight = float(config["loss_weights"]["terminal_bce"])
    fill_weight = float(config["loss_weights"]["fill_bce"])
    f0_material = _open_reference_f0_material(context.pack_manifest)

    def _micro_loss(
        rows: np.ndarray,
        x: np.ndarray,
        *,
        path_denominator: float,
        fill_denominator: float,
    ) -> torch.Tensor:
        material = _target_material(context, rows)
        tensor_x = torch.from_numpy(x).to(device)
        target_path = torch.from_numpy(
            np.nan_to_num(material["path"], nan=0.0)
        ).to(device)
        valid_path = torch.from_numpy(
            (
                material["path_available"]
                & (material["filled"] > 0.5)
                & np.isfinite(material["path"]).all(axis=1)
            ).astype(np.float32)
        ).to(device)
        terminal = torch.from_numpy(material["terminal"]).to(device)
        fill = torch.from_numpy(material["filled"]).to(device)
        with torch.autocast(device.type, dtype=torch.float16, enabled=amp_enabled):
            output = model(tensor_x)
        nll_by_row = student_t_nll(
            target_path,
            output["location"],
            output["scale"],
            df=df,
        ).mean(dim=1)
        path_loss = global_normalized_component(
            (nll_by_row * valid_path).sum(),
            path_denominator,
        )
        terminal_loss = global_normalized_component(
            (
                torch_functional.binary_cross_entropy_with_logits(
                    output["terminal_logit"].float(),
                    terminal.float(),
                    reduction="none",
                )
                * valid_path
            ).sum(),
            path_denominator,
        )
        fill_loss = global_normalized_component(
            torch_functional.binary_cross_entropy_with_logits(
                output["fill_logit"].float(), fill.float(), reduction="sum"
            ),
            fill_denominator,
        )
        return path_weight * path_loss + terminal_weight * terminal_loss + fill_weight * fill_loss

    def _effective_loss(rows: np.ndarray, *, backward: bool) -> torch.Tensor:
        material = _target_material(context, rows)
        valid_path = (
            material["path_available"]
            & (material["filled"] > 0.5)
            & np.isfinite(material["path"]).all(axis=1)
        )
        path_denominator = float(valid_path.sum())
        fill_denominator = float(len(rows))
        total = torch.zeros((), dtype=torch.float32, device=device)
        current_micro_batch = micro_batch if backward else validation_batch

        def _source() -> Iterator[tuple[np.ndarray, np.ndarray]]:
            for current in _micro_batches(rows, current_micro_batch):
                yield current, _f0_batch(
                    context,
                    current,
                    preprocessor,
                    material=f0_material,
                )

        for current, x in _prefetch_iterator(
            _source(), depth=prefetch_depth if backward else 0
        ):
            loss = _micro_loss(
                current,
                x,
                path_denominator=path_denominator,
                fill_denominator=fill_denominator,
            )
            if backward:
                scaler.scale(loss).backward()
            total += loss.detach()
        return total

    def _development_loss(epoch: int) -> float:
        model.eval()
        total = torch.zeros((), dtype=torch.float32, device=device)
        count = 0
        batches = [
            development_rows[start : start + validation_batch]
            for start in range(0, len(development_rows), validation_batch)
        ]
        with torch.no_grad():
            for step_index, rows in enumerate(batches):
                loss = _effective_loss(rows, backward=False)
                total += loss * len(rows)
                count += len(rows)
                if monitor is not None:
                    monitor.report(
                        status="training",
                        phase="validation",
                        samples_completed=count,
                        samples_total=len(development_rows),
                        epoch=epoch,
                        max_epochs=int(config["max_epochs"]),
                        step=step_index + 1,
                        steps_total=len(batches),
                    )
        return float((total / max(count, 1)).cpu())

    best_loss = float("inf")
    best_epoch = 0
    best_state: dict[str, torch.Tensor] | None = None
    epochs_without_improvement = 0
    training_log: list[dict[str, Any]] = []
    start_epoch = 1
    resume_step = 0
    resumed_epoch_loss_sum = 0.0
    resumed_epoch_sample_count = 0
    resume = _load_last_training_checkpoint(
        output_dir=output_dir,
        model=model,
        optimizer=optimizer,
        scaler=scaler,
        resolved_config=resolved_config,
    )
    if resume is not None:
        start_epoch = int(resume["epoch"])
        resume_step = int(resume["next_step_index"])
        best_epoch = int(resume["best_epoch"])
        best_loss = float(resume["best_development_loss"])
        best_state = resume.get("best_state")
        epochs_without_improvement = int(resume["epochs_without_improvement"])
        training_log = [dict(item) for item in resume.get("training_log", [])]
        resumed_epoch_loss_sum = float(resume.get("epoch_loss_sum", 0.0))
        resumed_epoch_sample_count = int(resume.get("epoch_sample_count", 0))
    started = time.perf_counter()
    for epoch in range(start_epoch, int(config["max_epochs"]) + 1):
        batches = effective_batch_plan(
            train_rows,
            context.candidate_date_idx,
            effective_batch=effective_batch,
            seed=int(seed) + epoch,
        )
        first_step = resume_step if epoch == start_epoch else 0
        if first_step > len(batches):
            raise ValueError("PatchTST resume cursor exceeds the deterministic epoch plan")
        model.train()
        train_total = torch.as_tensor(
            resumed_epoch_loss_sum if epoch == start_epoch else 0.0,
            dtype=torch.float32,
            device=device,
        )
        train_count = resumed_epoch_sample_count if epoch == start_epoch else 0
        for step_index in range(first_step, len(batches)):
            rows = batches[step_index]
            optimizer.zero_grad(set_to_none=True)
            step_loss = _effective_loss(rows, backward=True)
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), float(config["gradient_clip_norm"])
            )
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
            train_total += step_loss * len(rows)
            train_count += len(rows)
            current_loss = (
                float((train_total / max(train_count, 1)).cpu())
                if monitor is not None and monitor.heartbeat_due()
                else None
            )
            progress = {
                "phase": "training",
                "samples_completed": train_count,
                "samples_total": len(train_rows),
                "epoch": epoch,
                "max_epochs": int(config["max_epochs"]),
                "step": step_index + 1,
                "steps_total": len(batches),
                "loss": current_loss,
                "best_development_loss": best_loss,
                "patience_used": epochs_without_improvement,
            }
            if monitor is not None:
                monitor.report(status="training", **progress)

            def _save_step() -> Path:
                return _save_last_training_checkpoint(
                    output_dir=output_dir,
                    model=model,
                    optimizer=optimizer,
                    scaler=scaler,
                    resolved_config=resolved_config,
                    epoch=epoch,
                    next_step_index=step_index + 1,
                    best_epoch=best_epoch,
                    best_development_loss=best_loss,
                    best_state=best_state,
                    epochs_without_improvement=epochs_without_improvement,
                    training_log=training_log,
                    epoch_loss_sum=float(train_total.cpu()),
                    epoch_sample_count=train_count,
                )

            _checkpoint_or_pause(
                monitor=monitor,
                force_checkpoint=False,
                checkpoint_callback=_save_step,
                progress=progress,
            )
        train_loss = float((train_total / max(train_count, 1)).cpu())
        development_loss = _development_loss(epoch)
        training_log.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "development_loss": development_loss,
            }
        )
        if development_loss < best_loss - 1.0e-8:
            best_loss = development_loss
            best_epoch = epoch
            best_state = _clone_torch_state(model)
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
        epoch_progress = {
            "phase": "validation",
            "samples_completed": len(development_rows),
            "samples_total": len(development_rows),
            "epoch": epoch,
            "max_epochs": int(config["max_epochs"]),
            "step": 1,
            "steps_total": 1,
            "loss": development_loss,
            "best_development_loss": best_loss,
            "patience_used": epochs_without_improvement,
        }
        if monitor is not None:
            monitor.report(
                status="training",
                force=True,
                event="epoch_end",
                **epoch_progress,
            )

        def _save_epoch() -> Path:
            return _save_last_training_checkpoint(
                output_dir=output_dir,
                model=model,
                optimizer=optimizer,
                scaler=scaler,
                resolved_config=resolved_config,
                epoch=epoch + 1,
                next_step_index=0,
                best_epoch=best_epoch,
                best_development_loss=best_loss,
                best_state=best_state,
                epochs_without_improvement=epochs_without_improvement,
                training_log=training_log,
            )

        _checkpoint_or_pause(
            monitor=monitor,
            force_checkpoint=True,
            checkpoint_callback=_save_epoch,
            progress=epoch_progress,
        )
        resume_step = 0
        resumed_epoch_loss_sum = 0.0
        resumed_epoch_sample_count = 0
        if (
            epoch >= int(config["min_epochs"])
            and epochs_without_improvement >= int(config["patience"])
        ):
            break
    if best_state is None:
        raise RuntimeError("PatchTST training did not produce a finite checkpoint")
    model.load_state_dict(best_state)
    evidence = _save_torch_adapter(
        output_dir=output_dir,
        model=model,
        model_id="patchtst_student_t_path",
        model_config=config,
        resolved_config=resolved_config,
        preprocessor=preprocessor,
        best_epoch=best_epoch,
        best_development_loss=best_loss,
        training_log=training_log,
        extra={
            "training_seconds": time.perf_counter() - started,
            "student_t_df": df,
            "quality_mapping": "64_fixed_sobol_draws_seed1729_expected_min_g5_g10_g20_times_nonterminal",
            "student_t_nll": "explicit_audited_formula",
        },
    )
    artifact = {
        "adapter_type": "patchtst_student_t_path_distribution",
        "model_id": "patchtst_student_t_path",
        "capabilities": list(PatchTSTAdapter.capabilities),
        "student_t_df": df,
        "fused_adamw": bool(fused_optimizer),
        "execution_semantics_version": resolved_config.get(
            "execution_semantics_version"
        ),
        "resolved_config_sha256": str(resolved_config["resolved_config_sha256"]),
        **evidence,
    }
    _atomic_write_json(output_dir / "adapter_manifest.json", artifact)
    return {
        "artifact": artifact,
        "handle": {
            "model": model,
            "preprocessor": preprocessor,
            "df": df,
            "prediction_batch": int(resolved_config.get("prediction_batch", 1024)),
        },
    }


def _predict_patchtst_adapter(
    *,
    context: ModelDataContext,
    row_ids: np.ndarray,
    fitted: Mapping[str, Any],
    batch_size: int | None = None,
    monitor: _TrainingMonitor | None = None,
) -> dict[str, np.ndarray]:
    model = fitted["handle"]["model"]
    preprocessor = fitted["handle"]["preprocessor"]
    df = float(fitted["handle"]["df"])
    device = next(model.parameters()).device
    rows_all = np.asarray(row_ids, dtype=np.int64)
    count = len(rows_all)
    location = np.empty((count, 20), dtype=np.float32)
    scale = np.empty((count, 20), dtype=np.float32)
    terminal = np.empty(count, dtype=np.float32)
    fill = np.empty(count, dtype=np.float32)
    cursor = 0
    model.eval()
    with torch.no_grad():
        for rows, x in _iter_f0_batches(
            context,
            rows_all,
            preprocessor,
            batch_size=int(
                batch_size or fitted["handle"].get("prediction_batch", 1024)
            ),
            shuffle_dates=False,
            seed=0,
            cross_date=True,
        ):
            if not np.array_equal(rows, rows_all[cursor : cursor + len(rows)]):
                raise AssertionError("PatchTST prediction order changed")
            output = model(torch.from_numpy(x).to(device))
            current = slice(cursor, cursor + len(rows))
            location[current] = output["location"].float().cpu().numpy()
            scale[current] = output["scale"].float().cpu().numpy()
            terminal[current] = torch.sigmoid(output["terminal_logit"]).float().cpu().numpy()
            fill[current] = torch.sigmoid(output["fill_logit"]).float().cpu().numpy()
            cursor += len(rows)
            if monitor is not None:
                monitor.report(
                    status="predicting",
                    phase="prediction",
                    samples_completed=cursor,
                    samples_total=len(rows_all),
                )
    quality = student_t_path_quality(location, scale, terminal)
    quantile_values = stats.t.ppf([0.10, 0.50, 0.90], df=df).astype(np.float32)
    return {
        "fill_probability": fill,
        "conditional_quality_prediction": quality,
        "selection_score": (fill * quality).astype(np.float32),
        "path_location": location,
        "path_scale": scale,
        "path_q10": location + scale * quantile_values[0],
        "path_q50": location + scale * quantile_values[1],
        "path_q90": location + scale * quantile_values[2],
        "terminal_probability": terminal,
    }


def _fit_deepsets_adapter(
    *,
    context: ModelDataContext,
    train_rows: np.ndarray,
    development_rows: np.ndarray,
    resolved_config: Mapping[str, Any],
    output_dir: Path,
    seed: int,
    monitor: _TrainingMonitor | None = None,
) -> dict[str, Any]:
    config = dict(_model_spec(context, "market_industry_deepsets")["config"])
    preprocessor = _load_or_fit_snapshot_preprocessor(
        context,
        train_rows,
        profile=str(resolved_config["feature_profile"]),
        output_dir=output_dir,
        required_categorical_names=("industry_hash",),
    )
    categorical_names = list(preprocessor["categorical_names"])
    if "industry_hash" not in categorical_names:
        raise ValueError("Deep Sets requires the frozen PIT industry category")
    industry_position = categorical_names.index("industry_hash")
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))
    device, amp_enabled, scaler = _torch_runtime(config)
    model = _MarketIndustryDeepSets(
        n_num_features=2 * len(preprocessor["continuous_columns"]),
        cat_cardinalities=_category_cardinalities(preprocessor),
        industry_position=industry_position,
        config=config,
    ).to(device)
    optimizer, fused_optimizer = _adamw_optimizer(
        model,
        config,
        device=device,
    )
    target_candidates = int(config["effective_candidates_per_step"])
    quality_weight = float(config["loss_weights"]["quality_huber"])
    fill_weight = float(config["loss_weights"]["fill_bce"])

    def _date_loss(
        rows: np.ndarray,
        *,
        quality_denominator: float,
        fill_denominator: float,
    ) -> torch.Tensor:
        x_num, x_cat = _snapshot_batch(context, rows, preprocessor)
        material = _target_material(context, rows)
        quality = torch.from_numpy(
            np.nan_to_num(material["conditional_quality"], nan=0.0)
        ).to(device)
        quality_mask = torch.from_numpy(
            (np.isfinite(material["conditional_quality"]) & (material["filled"] > 0.5)).astype(np.float32)
        ).to(device)
        fill = torch.from_numpy(material["filled"]).to(device)
        with torch.autocast(device.type, dtype=torch.float16, enabled=amp_enabled):
            output = model(
                torch.from_numpy(x_num).to(device),
                torch.from_numpy(x_cat).to(device),
            )
            qloss = torch_functional.smooth_l1_loss(
                output["quality"], quality, reduction="none"
            )
            qloss = global_normalized_component(
                (qloss * quality_mask).sum(),
                quality_denominator,
            )
            floss = global_normalized_component(
                torch_functional.binary_cross_entropy_with_logits(
                    output["fill_logit"], fill, reduction="sum"
                ),
                fill_denominator,
            )
            return quality_weight * qloss + fill_weight * floss

    def _step_loss(groups: Sequence[np.ndarray], *, backward: bool) -> torch.Tensor:
        rows_all = np.concatenate(groups).astype(np.int64, copy=False)
        material = _target_material(context, rows_all)
        valid_quality = np.isfinite(material["conditional_quality"]) & (
            material["filled"] > 0.5
        )
        quality_denominator = float(valid_quality.sum())
        fill_denominator = float(len(rows_all))
        total = torch.zeros((), dtype=torch.float32, device=device)
        for rows in groups:
            loss = _date_loss(
                rows,
                quality_denominator=quality_denominator,
                fill_denominator=fill_denominator,
            )
            if backward:
                scaler.scale(loss).backward()
            total += loss.detach()
        return total

    def _development_loss(epoch: int) -> float:
        model.eval()
        total = torch.zeros((), dtype=torch.float32, device=device)
        count = 0
        steps = complete_date_step_plan(
            development_rows,
            context.candidate_date_idx,
            target_candidates=target_candidates,
            seed=None,
        )
        with torch.no_grad():
            for step_index, groups in enumerate(steps):
                current_count = int(sum(len(rows) for rows in groups))
                loss = _step_loss(groups, backward=False)
                total += loss * current_count
                count += current_count
                if monitor is not None:
                    monitor.report(
                        status="training",
                        phase="validation",
                        samples_completed=count,
                        samples_total=len(development_rows),
                        epoch=epoch,
                        max_epochs=int(config["max_epochs"]),
                        step=step_index + 1,
                        steps_total=len(steps),
                    )
        return float((total / max(count, 1)).cpu())

    best_loss = float("inf")
    best_epoch = 0
    best_state: dict[str, torch.Tensor] | None = None
    epochs_without_improvement = 0
    training_log: list[dict[str, Any]] = []
    start_epoch = 1
    resume_step = 0
    resumed_epoch_loss_sum = 0.0
    resumed_epoch_sample_count = 0
    resume = _load_last_training_checkpoint(
        output_dir=output_dir,
        model=model,
        optimizer=optimizer,
        scaler=scaler,
        resolved_config=resolved_config,
    )
    if resume is not None:
        start_epoch = int(resume["epoch"])
        resume_step = int(resume["next_step_index"])
        best_epoch = int(resume["best_epoch"])
        best_loss = float(resume["best_development_loss"])
        best_state = resume.get("best_state")
        epochs_without_improvement = int(resume["epochs_without_improvement"])
        training_log = [dict(item) for item in resume.get("training_log", [])]
        resumed_epoch_loss_sum = float(resume.get("epoch_loss_sum", 0.0))
        resumed_epoch_sample_count = int(resume.get("epoch_sample_count", 0))
    started = time.perf_counter()
    for epoch in range(start_epoch, int(config["max_epochs"]) + 1):
        steps = complete_date_step_plan(
            train_rows,
            context.candidate_date_idx,
            target_candidates=target_candidates,
            seed=int(seed) + epoch,
        )
        first_step = resume_step if epoch == start_epoch else 0
        if first_step > len(steps):
            raise ValueError("Deep Sets resume cursor exceeds the deterministic epoch plan")
        model.train()
        train_total = torch.as_tensor(
            resumed_epoch_loss_sum if epoch == start_epoch else 0.0,
            dtype=torch.float32,
            device=device,
        )
        train_count = resumed_epoch_sample_count if epoch == start_epoch else 0
        for step_index in range(first_step, len(steps)):
            groups = steps[step_index]
            current_count = int(sum(len(rows) for rows in groups))
            optimizer.zero_grad(set_to_none=True)
            step_loss = _step_loss(groups, backward=True)
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), float(config["gradient_clip_norm"])
            )
            scaler.step(optimizer)
            scaler.update()
            train_total += step_loss * current_count
            train_count += current_count
            current_loss = (
                float((train_total / max(train_count, 1)).cpu())
                if monitor is not None and monitor.heartbeat_due()
                else None
            )
            progress = {
                "phase": "training",
                "samples_completed": train_count,
                "samples_total": len(train_rows),
                "epoch": epoch,
                "max_epochs": int(config["max_epochs"]),
                "step": step_index + 1,
                "steps_total": len(steps),
                "loss": current_loss,
                "best_development_loss": best_loss,
                "patience_used": epochs_without_improvement,
            }
            if monitor is not None:
                monitor.report(status="training", **progress)

            def _save_step() -> Path:
                return _save_last_training_checkpoint(
                    output_dir=output_dir,
                    model=model,
                    optimizer=optimizer,
                    scaler=scaler,
                    resolved_config=resolved_config,
                    epoch=epoch,
                    next_step_index=step_index + 1,
                    best_epoch=best_epoch,
                    best_development_loss=best_loss,
                    best_state=best_state,
                    epochs_without_improvement=epochs_without_improvement,
                    training_log=training_log,
                    epoch_loss_sum=float(train_total.cpu()),
                    epoch_sample_count=train_count,
                )

            _checkpoint_or_pause(
                monitor=monitor,
                force_checkpoint=False,
                checkpoint_callback=_save_step,
                progress=progress,
            )
        train_loss = float((train_total / max(train_count, 1)).cpu())
        development_loss = _development_loss(epoch)
        training_log.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "development_loss": development_loss,
            }
        )
        if development_loss < best_loss - 1.0e-8:
            best_loss = development_loss
            best_epoch = epoch
            best_state = _clone_torch_state(model)
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
        epoch_progress = {
            "phase": "validation",
            "samples_completed": len(development_rows),
            "samples_total": len(development_rows),
            "epoch": epoch,
            "max_epochs": int(config["max_epochs"]),
            "step": 1,
            "steps_total": 1,
            "loss": development_loss,
            "best_development_loss": best_loss,
            "patience_used": epochs_without_improvement,
        }
        if monitor is not None:
            monitor.report(
                status="training",
                force=True,
                event="epoch_end",
                **epoch_progress,
            )

        def _save_epoch() -> Path:
            return _save_last_training_checkpoint(
                output_dir=output_dir,
                model=model,
                optimizer=optimizer,
                scaler=scaler,
                resolved_config=resolved_config,
                epoch=epoch + 1,
                next_step_index=0,
                best_epoch=best_epoch,
                best_development_loss=best_loss,
                best_state=best_state,
                epochs_without_improvement=epochs_without_improvement,
                training_log=training_log,
            )

        _checkpoint_or_pause(
            monitor=monitor,
            force_checkpoint=True,
            checkpoint_callback=_save_epoch,
            progress=epoch_progress,
        )
        resume_step = 0
        resumed_epoch_loss_sum = 0.0
        resumed_epoch_sample_count = 0
        if (
            epoch >= int(config["min_epochs"])
            and epochs_without_improvement >= int(config["patience"])
        ):
            break
    if best_state is None:
        raise RuntimeError("Deep Sets training did not produce a finite checkpoint")
    model.load_state_dict(best_state)
    evidence = _save_torch_adapter(
        output_dir=output_dir,
        model=model,
        model_id="market_industry_deepsets",
        model_config=config,
        resolved_config=resolved_config,
        preprocessor=preprocessor,
        best_epoch=best_epoch,
        best_development_loss=best_loss,
        training_log=training_log,
        extra={
            "training_seconds": time.perf_counter() - started,
            "set_batching": "complete_signal_dates_atomic; optimizer steps pack whole dates toward frozen candidate target",
            "industry_position": industry_position,
        },
    )
    artifact = {
        "adapter_type": "permutation_equivariant_market_industry_deepsets",
        "model_id": "market_industry_deepsets",
        "capabilities": list(DeepSetsAdapter.capabilities),
        "industry_position": industry_position,
        "fused_adamw": bool(fused_optimizer),
        "execution_semantics_version": resolved_config.get(
            "execution_semantics_version"
        ),
        "resolved_config_sha256": str(resolved_config["resolved_config_sha256"]),
        **evidence,
    }
    _atomic_write_json(output_dir / "adapter_manifest.json", artifact)
    return {
        "artifact": artifact,
        "handle": {
            "model": model,
            "preprocessor": preprocessor,
            "prediction_batch": int(resolved_config.get("prediction_batch", 4096)),
        },
    }


def _predict_deepsets_adapter(
    *,
    context: ModelDataContext,
    row_ids: np.ndarray,
    fitted: Mapping[str, Any],
    monitor: _TrainingMonitor | None = None,
) -> dict[str, np.ndarray]:
    model = fitted["handle"]["model"]
    preprocessor = fitted["handle"]["preprocessor"]
    device = next(model.parameters()).device
    rows_all = np.asarray(row_ids, dtype=np.int64)
    fill = np.empty(len(rows_all), dtype=np.float32)
    quality = np.empty(len(rows_all), dtype=np.float32)
    cursor = 0
    model.eval()
    with torch.no_grad():
        for rows in _row_groups(rows_all, context.candidate_date_idx):
            if not np.array_equal(rows, rows_all[cursor : cursor + len(rows)]):
                raise AssertionError("Deep Sets prediction order changed")
            x_num, x_cat = _snapshot_batch(context, rows, preprocessor)
            output = model(
                torch.from_numpy(x_num).to(device),
                torch.from_numpy(x_cat).to(device),
            )
            current = slice(cursor, cursor + len(rows))
            quality[current] = output["quality"].float().cpu().numpy()
            fill[current] = torch.sigmoid(output["fill_logit"]).float().cpu().numpy()
            cursor += len(rows)
            if monitor is not None:
                monitor.report(
                    status="predicting",
                    phase="prediction",
                    samples_completed=cursor,
                    samples_total=len(rows_all),
                )
    return {
        "fill_probability": fill,
        "conditional_quality_prediction": quality,
        "selection_score": (fill * quality).astype(np.float32),
    }


def _fit_deephit_adapter(
    *,
    context: ModelDataContext,
    train_rows: np.ndarray,
    development_rows: np.ndarray,
    resolved_config: Mapping[str, Any],
    output_dir: Path,
    seed: int,
    monitor: _TrainingMonitor | None = None,
) -> dict[str, Any]:
    config = dict(_model_spec(context, "deephit_competing_risk")["config"])
    preprocessor = _load_or_fit_snapshot_preprocessor(
        context,
        train_rows,
        profile=str(resolved_config["feature_profile"]),
        output_dir=output_dir,
    )
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))
    device, amp_enabled, scaler = _torch_runtime(config)
    model = _DeepHitCompetingRisk(
        n_num_features=2 * len(preprocessor["continuous_columns"]),
        cat_cardinalities=_category_cardinalities(preprocessor),
        config=config,
    ).to(device)
    optimizer, fused_optimizer = _adamw_optimizer(
        model,
        config,
        device=device,
    )
    micro_batch = int(resolved_config["micro_batch"])
    effective_batch = int(resolved_config["effective_batch"])
    validation_batch = int(resolved_config.get("validation_batch", effective_batch))
    prefetch_depth = int(resolved_config.get("prefetch_depth", 0))
    likelihood_weight = float(config["loss_weights"]["likelihood"])
    ranking_weight = float(config["loss_weights"]["cause_ranking"])
    fill_weight = float(config["loss_weights"]["fill_bce"])

    def _effective_loss(rows: np.ndarray, *, backward: bool) -> torch.Tensor:
        material = _target_material(context, rows)
        valid = (
            material["path_available"]
            & (material["filled"] > 0.5)
            & np.isfinite(material["event_code"])
            & np.isfinite(material["event_time"])
        )
        event_code = torch.from_numpy(
            np.nan_to_num(material["event_code"], nan=0.0).astype(np.int64)
        ).to(device)
        event_time = torch.from_numpy(
            np.nan_to_num(material["event_time"], nan=20.0).astype(np.int64)
        ).to(device)
        valid_tensor = torch.from_numpy(valid).to(device)
        fill = torch.from_numpy(material["filled"]).to(device)
        joint_parts: list[torch.Tensor] = []
        fill_parts: list[torch.Tensor] = []
        current_micro_batch = micro_batch if backward else validation_batch

        def _source() -> Iterator[tuple[np.ndarray, np.ndarray, np.ndarray]]:
            for current in _micro_batches(rows, current_micro_batch):
                x_num, x_cat = _snapshot_batch(context, current, preprocessor)
                yield current, x_num, x_cat

        for _current, x_num, x_cat in _prefetch_iterator(
            _source(), depth=prefetch_depth if backward else 0
        ):
            with torch.autocast(
                device.type, dtype=torch.float16, enabled=amp_enabled
            ):
                output = model(
                    torch.from_numpy(x_num).to(device),
                    torch.from_numpy(x_cat).to(device),
                )
            joint_parts.append(output["joint_logits"].float())
            fill_parts.append(output["fill_logit"].float())
        joint_logits = torch.cat(joint_parts, dim=0)
        fill_logit = torch.cat(fill_parts, dim=0)
        if bool(valid.any()):
            joint_target = deephit_joint_targets(event_code, event_time)
            likelihood = torch_functional.cross_entropy(
                joint_logits[valid_tensor],
                joint_target[valid_tensor],
            )
            probability = torch.softmax(
                joint_logits[valid_tensor], dim=1
            )
            ranking = deephit_ranking_loss(
                probability,
                event_code[valid_tensor],
                event_time[valid_tensor],
            )
        else:
            likelihood = joint_logits.sum() * 0.0
            ranking = likelihood
        fill_loss = torch_functional.binary_cross_entropy_with_logits(
            fill_logit, fill.float()
        )
        loss = (
            likelihood_weight * likelihood
            + ranking_weight * ranking
            + fill_weight * fill_loss
        )
        if backward:
            scaler.scale(loss).backward()
        return loss.detach()

    def _development_loss(epoch: int) -> float:
        model.eval()
        total = torch.zeros((), dtype=torch.float32, device=device)
        count = 0
        batches = [
            development_rows[start : start + validation_batch]
            for start in range(0, len(development_rows), validation_batch)
        ]
        with torch.no_grad():
            for step_index, rows in enumerate(batches):
                loss = _effective_loss(rows, backward=False)
                total += loss * len(rows)
                count += len(rows)
                if monitor is not None:
                    monitor.report(
                        status="training",
                        phase="validation",
                        samples_completed=count,
                        samples_total=len(development_rows),
                        epoch=epoch,
                        max_epochs=int(config["max_epochs"]),
                        step=step_index + 1,
                        steps_total=len(batches),
                    )
        return float((total / max(count, 1)).cpu())

    best_loss = float("inf")
    best_epoch = 0
    best_state: dict[str, torch.Tensor] | None = None
    epochs_without_improvement = 0
    training_log: list[dict[str, Any]] = []
    start_epoch = 1
    resume_step = 0
    resumed_epoch_loss_sum = 0.0
    resumed_epoch_sample_count = 0
    resume = _load_last_training_checkpoint(
        output_dir=output_dir,
        model=model,
        optimizer=optimizer,
        scaler=scaler,
        resolved_config=resolved_config,
    )
    if resume is not None:
        start_epoch = int(resume["epoch"])
        resume_step = int(resume["next_step_index"])
        best_epoch = int(resume["best_epoch"])
        best_loss = float(resume["best_development_loss"])
        best_state = resume.get("best_state")
        epochs_without_improvement = int(resume["epochs_without_improvement"])
        training_log = [dict(item) for item in resume.get("training_log", [])]
        resumed_epoch_loss_sum = float(resume.get("epoch_loss_sum", 0.0))
        resumed_epoch_sample_count = int(resume.get("epoch_sample_count", 0))
    started = time.perf_counter()
    for epoch in range(start_epoch, int(config["max_epochs"]) + 1):
        batches = effective_batch_plan(
            train_rows,
            context.candidate_date_idx,
            effective_batch=effective_batch,
            seed=int(seed) + epoch,
        )
        first_step = resume_step if epoch == start_epoch else 0
        if first_step > len(batches):
            raise ValueError("DeepHit resume cursor exceeds the deterministic epoch plan")
        model.train()
        train_total = torch.as_tensor(
            resumed_epoch_loss_sum if epoch == start_epoch else 0.0,
            dtype=torch.float32,
            device=device,
        )
        train_count = resumed_epoch_sample_count if epoch == start_epoch else 0
        for step_index in range(first_step, len(batches)):
            rows = batches[step_index]
            optimizer.zero_grad(set_to_none=True)
            step_loss = _effective_loss(rows, backward=True)
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), float(config["gradient_clip_norm"])
            )
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
            train_total += step_loss * len(rows)
            train_count += len(rows)
            current_loss = (
                float((train_total / max(train_count, 1)).cpu())
                if monitor is not None and monitor.heartbeat_due()
                else None
            )
            progress = {
                "phase": "training",
                "samples_completed": train_count,
                "samples_total": len(train_rows),
                "epoch": epoch,
                "max_epochs": int(config["max_epochs"]),
                "step": step_index + 1,
                "steps_total": len(batches),
                "loss": current_loss,
                "best_development_loss": best_loss,
                "patience_used": epochs_without_improvement,
            }
            if monitor is not None:
                monitor.report(status="training", **progress)

            def _save_step() -> Path:
                return _save_last_training_checkpoint(
                    output_dir=output_dir,
                    model=model,
                    optimizer=optimizer,
                    scaler=scaler,
                    resolved_config=resolved_config,
                    epoch=epoch,
                    next_step_index=step_index + 1,
                    best_epoch=best_epoch,
                    best_development_loss=best_loss,
                    best_state=best_state,
                    epochs_without_improvement=epochs_without_improvement,
                    training_log=training_log,
                    epoch_loss_sum=float(train_total.cpu()),
                    epoch_sample_count=train_count,
                )

            _checkpoint_or_pause(
                monitor=monitor,
                force_checkpoint=False,
                checkpoint_callback=_save_step,
                progress=progress,
            )
        train_loss = float((train_total / max(train_count, 1)).cpu())
        development_loss = _development_loss(epoch)
        training_log.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "development_loss": development_loss,
            }
        )
        if development_loss < best_loss - 1.0e-8:
            best_loss = development_loss
            best_epoch = epoch
            best_state = _clone_torch_state(model)
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
        epoch_progress = {
            "phase": "validation",
            "samples_completed": len(development_rows),
            "samples_total": len(development_rows),
            "epoch": epoch,
            "max_epochs": int(config["max_epochs"]),
            "step": 1,
            "steps_total": 1,
            "loss": development_loss,
            "best_development_loss": best_loss,
            "patience_used": epochs_without_improvement,
        }
        if monitor is not None:
            monitor.report(
                status="training",
                force=True,
                event="epoch_end",
                **epoch_progress,
            )

        def _save_epoch() -> Path:
            return _save_last_training_checkpoint(
                output_dir=output_dir,
                model=model,
                optimizer=optimizer,
                scaler=scaler,
                resolved_config=resolved_config,
                epoch=epoch + 1,
                next_step_index=0,
                best_epoch=best_epoch,
                best_development_loss=best_loss,
                best_state=best_state,
                epochs_without_improvement=epochs_without_improvement,
                training_log=training_log,
            )

        _checkpoint_or_pause(
            monitor=monitor,
            force_checkpoint=True,
            checkpoint_callback=_save_epoch,
            progress=epoch_progress,
        )
        resume_step = 0
        resumed_epoch_loss_sum = 0.0
        resumed_epoch_sample_count = 0
        if (
            epoch >= int(config["min_epochs"])
            and epochs_without_improvement >= int(config["patience"])
        ):
            break
    if best_state is None:
        raise RuntimeError("DeepHit training did not produce a finite checkpoint")
    model.load_state_dict(best_state)
    evidence = _save_torch_adapter(
        output_dir=output_dir,
        model=model,
        model_id="deephit_competing_risk",
        model_config=config,
        resolved_config=resolved_config,
        preprocessor=preprocessor,
        best_epoch=best_epoch,
        best_development_loss=best_loss,
        training_log=training_log,
        extra={
            "training_seconds": time.perf_counter() - started,
            "joint_support": "five_causes_x_20_days_plus_d20_censor_atom",
        },
    )
    artifact = {
        "adapter_type": "deephit_joint_competing_risk",
        "model_id": "deephit_competing_risk",
        "capabilities": list(DeepHitAdapter.capabilities),
        "fused_adamw": bool(fused_optimizer),
        "execution_semantics_version": resolved_config.get(
            "execution_semantics_version"
        ),
        "resolved_config_sha256": str(resolved_config["resolved_config_sha256"]),
        **evidence,
    }
    _atomic_write_json(output_dir / "adapter_manifest.json", artifact)
    return {
        "artifact": artifact,
        "handle": {
            "model": model,
            "preprocessor": preprocessor,
            "prediction_batch": int(resolved_config.get("prediction_batch", 4096)),
        },
    }


def _predict_deephit_adapter(
    *,
    context: ModelDataContext,
    row_ids: np.ndarray,
    fitted: Mapping[str, Any],
    batch_size: int | None = None,
    monitor: _TrainingMonitor | None = None,
) -> dict[str, np.ndarray]:
    model = fitted["handle"]["model"]
    preprocessor = fitted["handle"]["preprocessor"]
    device = next(model.parameters()).device
    rows_all = np.asarray(row_ids, dtype=np.int64)
    probability = np.empty((len(rows_all), 101), dtype=np.float32)
    fill = np.empty(len(rows_all), dtype=np.float32)
    cursor = 0
    model.eval()
    with torch.no_grad():
        for rows, x_num, x_cat in _iter_snapshot_batches(
            context,
            rows_all,
            preprocessor,
            batch_size=int(batch_size or fitted["handle"].get("prediction_batch", 4096)),
            shuffle_dates=False,
            seed=0,
            cross_date=True,
        ):
            if not np.array_equal(rows, rows_all[cursor : cursor + len(rows)]):
                raise AssertionError("DeepHit prediction order changed")
            output = model(
                torch.from_numpy(x_num).to(device),
                torch.from_numpy(x_cat).to(device),
            )
            current = slice(cursor, cursor + len(rows))
            probability[current] = torch.softmax(
                output["joint_logits"].float(), dim=1
            ).cpu().numpy()
            fill[current] = torch.sigmoid(output["fill_logit"]).float().cpu().numpy()
            cursor += len(rows)
            if monitor is not None:
                monitor.report(
                    status="predicting",
                    phase="prediction",
                    samples_completed=cursor,
                    samples_total=len(rows_all),
                )
    quality = deephit_expected_event_score(
        torch.from_numpy(probability)
    ).numpy().astype(np.float32)
    return {
        "fill_probability": fill,
        "conditional_quality_prediction": quality,
        "selection_score": (fill * quality).astype(np.float32),
        "joint_event_probability": probability,
    }


def _listwise_partitions(
    context: ModelDataContext,
    rows: np.ndarray,
    *,
    list_size: int,
    epoch_offset: int,
) -> Iterator[np.ndarray]:
    symbols = np.asarray(context.candidate_symbol_idx[rows], dtype=np.int64)
    bucket = (symbols + int(epoch_offset)) % int(list_size)
    order = np.lexsort((symbols, bucket))
    ordered = np.asarray(rows, dtype=np.int64)[order]
    for start in range(0, len(ordered), int(list_size)):
        yield ordered[start : start + int(list_size)]


def _listwise_epoch_plan(
    context: ModelDataContext,
    row_ids: np.ndarray,
    *,
    list_size: int,
    epoch: int,
    seed: int | None,
) -> list[np.ndarray]:
    groups = _row_groups(row_ids, context.candidate_date_idx)
    if seed is not None:
        np.random.default_rng(int(seed)).shuffle(groups)
    return [
        rows
        for group in groups
        for rows in _listwise_partitions(
            context,
            group,
            list_size=int(list_size),
            epoch_offset=int(epoch),
        )
    ]


def _fit_neuralndcg_adapter(
    *,
    context: ModelDataContext,
    train_rows: np.ndarray,
    development_rows: np.ndarray,
    resolved_config: Mapping[str, Any],
    output_dir: Path,
    seed: int,
    monitor: _TrainingMonitor | None = None,
) -> dict[str, Any]:
    config = dict(_model_spec(context, "neuralndcg_listwise_mlp")["config"])
    preprocessor = _load_or_fit_snapshot_preprocessor(
        context,
        train_rows,
        profile=str(resolved_config["feature_profile"]),
        output_dir=output_dir,
    )
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))
    device, amp_enabled, scaler = _torch_runtime(config)
    model = _NeuralNDCGMLP(
        n_num_features=2 * len(preprocessor["continuous_columns"]),
        cat_cardinalities=_category_cardinalities(preprocessor),
        config=config,
    ).to(device)
    optimizer, fused_optimizer = _adamw_optimizer(
        model,
        config,
        device=device,
    )
    list_size = int(config["training_list_size"])
    ndcg_weight = float(config["loss_weights"]["neural_ndcg"])
    fill_weight = float(config["loss_weights"]["fill_bce"])

    def _loss(rows: np.ndarray) -> torch.Tensor:
        x_num, x_cat = _snapshot_batch(context, rows, preprocessor)
        material = _target_material(context, rows)
        grades = torch.from_numpy(material["grade"].astype(np.int64)).to(device)
        valid_grade = grades < 5
        fill = torch.from_numpy(material["filled"]).to(device)
        with torch.autocast(device.type, dtype=torch.float16, enabled=amp_enabled):
            output = model(
                torch.from_numpy(x_num).to(device),
                torch.from_numpy(x_cat).to(device),
            )
        if int(valid_grade.sum().item()) >= 2:
            ndcg = neural_ndcg_loss(
                output["raw_score"].float()[valid_grade],
                grades[valid_grade],
                temperature=float(config["temperature"]),
                gain=config["gain"],
            )
        else:
            ndcg = output["raw_score"].float().sum() * 0.0
        fill_loss = torch_functional.binary_cross_entropy_with_logits(
            output["fill_logit"].float(), fill.float()
        )
        return ndcg_weight * ndcg + fill_weight * fill_loss

    def _development_loss(epoch: int) -> float:
        model.eval()
        total = torch.zeros((), dtype=torch.float32, device=device)
        count = 0
        steps = _listwise_epoch_plan(
            context,
            development_rows,
            list_size=list_size,
            epoch=0,
            seed=None,
        )
        with torch.no_grad():
            for step_index, rows in enumerate(steps):
                loss = _loss(rows)
                total += loss.detach() * len(rows)
                count += len(rows)
                if monitor is not None:
                    monitor.report(
                        status="training",
                        phase="validation",
                        samples_completed=count,
                        samples_total=len(development_rows),
                        epoch=epoch,
                        max_epochs=int(config["max_epochs"]),
                        step=step_index + 1,
                        steps_total=len(steps),
                    )
        return float((total / max(count, 1)).cpu())

    best_loss = float("inf")
    best_epoch = 0
    best_state: dict[str, torch.Tensor] | None = None
    epochs_without_improvement = 0
    training_log: list[dict[str, Any]] = []
    start_epoch = 1
    resume_step = 0
    resumed_epoch_loss_sum = 0.0
    resumed_epoch_sample_count = 0
    resume = _load_last_training_checkpoint(
        output_dir=output_dir,
        model=model,
        optimizer=optimizer,
        scaler=scaler,
        resolved_config=resolved_config,
    )
    if resume is not None:
        start_epoch = int(resume["epoch"])
        resume_step = int(resume["next_step_index"])
        best_epoch = int(resume["best_epoch"])
        best_loss = float(resume["best_development_loss"])
        best_state = resume.get("best_state")
        epochs_without_improvement = int(resume["epochs_without_improvement"])
        training_log = [dict(item) for item in resume.get("training_log", [])]
        resumed_epoch_loss_sum = float(resume.get("epoch_loss_sum", 0.0))
        resumed_epoch_sample_count = int(resume.get("epoch_sample_count", 0))
    started = time.perf_counter()
    for epoch in range(start_epoch, int(config["max_epochs"]) + 1):
        steps = _listwise_epoch_plan(
            context,
            train_rows,
            list_size=list_size,
            epoch=epoch,
            seed=int(seed) + epoch,
        )
        first_step = resume_step if epoch == start_epoch else 0
        if first_step > len(steps):
            raise ValueError("NeuralNDCG resume cursor exceeds the deterministic list plan")
        model.train()
        train_total = torch.as_tensor(
            resumed_epoch_loss_sum if epoch == start_epoch else 0.0,
            dtype=torch.float32,
            device=device,
        )
        train_count = resumed_epoch_sample_count if epoch == start_epoch else 0
        for step_index in range(first_step, len(steps)):
            rows = steps[step_index]
            optimizer.zero_grad(set_to_none=True)
            loss = _loss(rows)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), float(config["gradient_clip_norm"])
            )
            scaler.step(optimizer)
            scaler.update()
            train_total += loss.detach() * len(rows)
            train_count += len(rows)
            current_loss = (
                float((train_total / max(train_count, 1)).cpu())
                if monitor is not None and monitor.heartbeat_due()
                else None
            )
            progress = {
                "phase": "training",
                "samples_completed": train_count,
                "samples_total": len(train_rows),
                "epoch": epoch,
                "max_epochs": int(config["max_epochs"]),
                "step": step_index + 1,
                "steps_total": len(steps),
                "loss": current_loss,
                "best_development_loss": best_loss,
                "patience_used": epochs_without_improvement,
            }
            if monitor is not None:
                monitor.report(status="training", **progress)

            def _save_step() -> Path:
                return _save_last_training_checkpoint(
                    output_dir=output_dir,
                    model=model,
                    optimizer=optimizer,
                    scaler=scaler,
                    resolved_config=resolved_config,
                    epoch=epoch,
                    next_step_index=step_index + 1,
                    best_epoch=best_epoch,
                    best_development_loss=best_loss,
                    best_state=best_state,
                    epochs_without_improvement=epochs_without_improvement,
                    training_log=training_log,
                    epoch_loss_sum=float(train_total.cpu()),
                    epoch_sample_count=train_count,
                )

            _checkpoint_or_pause(
                monitor=monitor,
                force_checkpoint=False,
                checkpoint_callback=_save_step,
                progress=progress,
            )
        train_loss = float((train_total / max(train_count, 1)).cpu())
        development_loss = _development_loss(epoch)
        training_log.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "development_loss": development_loss,
            }
        )
        if development_loss < best_loss - 1.0e-8:
            best_loss = development_loss
            best_epoch = epoch
            best_state = _clone_torch_state(model)
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
        epoch_progress = {
            "phase": "validation",
            "samples_completed": len(development_rows),
            "samples_total": len(development_rows),
            "epoch": epoch,
            "max_epochs": int(config["max_epochs"]),
            "step": 1,
            "steps_total": 1,
            "loss": development_loss,
            "best_development_loss": best_loss,
            "patience_used": epochs_without_improvement,
        }
        if monitor is not None:
            monitor.report(
                status="training",
                force=True,
                event="epoch_end",
                **epoch_progress,
            )

        def _save_epoch() -> Path:
            return _save_last_training_checkpoint(
                output_dir=output_dir,
                model=model,
                optimizer=optimizer,
                scaler=scaler,
                resolved_config=resolved_config,
                epoch=epoch + 1,
                next_step_index=0,
                best_epoch=best_epoch,
                best_development_loss=best_loss,
                best_state=best_state,
                epochs_without_improvement=epochs_without_improvement,
                training_log=training_log,
            )

        _checkpoint_or_pause(
            monitor=monitor,
            force_checkpoint=True,
            checkpoint_callback=_save_epoch,
            progress=epoch_progress,
        )
        resume_step = 0
        resumed_epoch_loss_sum = 0.0
        resumed_epoch_sample_count = 0
        if (
            epoch >= int(config["min_epochs"])
            and epochs_without_improvement >= int(config["patience"])
        ):
            break
    if best_state is None:
        raise RuntimeError("NeuralNDCG training did not produce a finite checkpoint")
    model.load_state_dict(best_state)

    raw_development = np.empty(len(development_rows), dtype=np.float32)
    cursor = 0
    model.eval()
    with torch.no_grad():
        for rows, x_num, x_cat in _iter_snapshot_batches(
            context,
            development_rows,
            preprocessor,
            batch_size=int(resolved_config.get("validation_batch", 4096)),
            shuffle_dates=False,
            seed=0,
            cross_date=True,
        ):
            if not np.array_equal(
                rows, development_rows[cursor : cursor + len(rows)]
            ):
                raise AssertionError("NeuralNDCG calibration order changed")
            output = model(
                torch.from_numpy(x_num).to(device),
                torch.from_numpy(x_cat).to(device),
            )
            raw_development[cursor : cursor + len(rows)] = (
                output["raw_score"].float().cpu().numpy()
            )
            cursor += len(rows)
    development_material = _target_material(context, development_rows)
    valid_calibration = (
        (development_material["filled"] > 0.5)
        & np.isfinite(development_material["conditional_quality"])
    )
    isotonic, isotonic_payload = _fit_isotonic_mapping(
        raw_development[valid_calibration],
        development_material["conditional_quality"][valid_calibration],
    )
    _atomic_write_json(output_dir / "isotonic.json", isotonic_payload)
    evidence = _save_torch_adapter(
        output_dir=output_dir,
        model=model,
        model_id="neuralndcg_listwise_mlp",
        model_config=config,
        resolved_config=resolved_config,
        preprocessor=preprocessor,
        best_epoch=best_epoch,
        best_development_loss=best_loss,
        training_log=training_log,
        extra={
            "training_seconds": time.perf_counter() - started,
            "isotonic": isotonic_payload,
            "list_partition": config["list_partition"],
        },
    )
    artifact = {
        "adapter_type": "neural_sort_ndcg_listwise_mlp",
        "model_id": "neuralndcg_listwise_mlp",
        "capabilities": list(NeuralNDCGAdapter.capabilities),
        "fused_adamw": bool(fused_optimizer),
        "execution_semantics_version": resolved_config.get(
            "execution_semantics_version"
        ),
        "isotonic": isotonic_payload,
        "resolved_config_sha256": str(resolved_config["resolved_config_sha256"]),
        **evidence,
    }
    _atomic_write_json(output_dir / "adapter_manifest.json", artifact)
    return {
        "artifact": artifact,
        "handle": {
            "model": model,
            "preprocessor": preprocessor,
            "isotonic": isotonic,
            "prediction_batch": int(resolved_config.get("prediction_batch", 4096)),
        },
    }


def _predict_neuralndcg_adapter(
    *,
    context: ModelDataContext,
    row_ids: np.ndarray,
    fitted: Mapping[str, Any],
    batch_size: int | None = None,
    monitor: _TrainingMonitor | None = None,
) -> dict[str, np.ndarray]:
    model = fitted["handle"]["model"]
    preprocessor = fitted["handle"]["preprocessor"]
    isotonic = fitted["handle"]["isotonic"]
    device = next(model.parameters()).device
    rows_all = np.asarray(row_ids, dtype=np.int64)
    raw = np.empty(len(rows_all), dtype=np.float32)
    fill = np.empty(len(rows_all), dtype=np.float32)
    cursor = 0
    model.eval()
    with torch.no_grad():
        for rows, x_num, x_cat in _iter_snapshot_batches(
            context,
            rows_all,
            preprocessor,
            batch_size=int(batch_size or fitted["handle"].get("prediction_batch", 4096)),
            shuffle_dates=False,
            seed=0,
            cross_date=True,
        ):
            if not np.array_equal(rows, rows_all[cursor : cursor + len(rows)]):
                raise AssertionError("NeuralNDCG prediction order changed")
            output = model(
                torch.from_numpy(x_num).to(device),
                torch.from_numpy(x_cat).to(device),
            )
            current = slice(cursor, cursor + len(rows))
            raw[current] = output["raw_score"].float().cpu().numpy()
            fill[current] = torch.sigmoid(output["fill_logit"]).float().cpu().numpy()
            cursor += len(rows)
            if monitor is not None:
                monitor.report(
                    status="predicting",
                    phase="prediction",
                    samples_completed=cursor,
                    samples_total=len(rows_all),
                )
    quality = np.asarray(isotonic.predict(raw), dtype=np.float32)
    return {
        "fill_probability": fill,
        "conditional_quality_prediction": quality,
        "selection_score": (fill * quality).astype(np.float32),
        "raw_listwise_score": raw,
    }


def _fit_model_dispatch(
    *,
    model_id: str,
    train_rows: np.ndarray,
    development_rows: np.ndarray,
    resolved_config: Mapping[str, Any],
    context: ModelDataContext,
    output_dir: Path,
    seed: int,
    monitor: _TrainingMonitor | None = None,
    **_unused: Any,
) -> dict[str, Any]:
    common = {
        "context": context,
        "train_rows": train_rows,
        "development_rows": development_rows,
        "resolved_config": resolved_config,
        "output_dir": output_dir,
        "seed": int(seed),
    }
    if str(model_id) == "lgbm_lambdarank_multioutput":
        return _fit_lgbm_adapter(**common)
    if str(model_id) == "tabm_multioutput":
        return _fit_tabm_adapter(**common, monitor=monitor)
    if str(model_id) == "patchtst_student_t_path":
        return _fit_patchtst_adapter(**common, monitor=monitor)
    if str(model_id) == "market_industry_deepsets":
        return _fit_deepsets_adapter(**common, monitor=monitor)
    if str(model_id) == "deephit_competing_risk":
        return _fit_deephit_adapter(**common, monitor=monitor)
    if str(model_id) == "neuralndcg_listwise_mlp":
        return _fit_neuralndcg_adapter(**common, monitor=monitor)
    raise KeyError(model_id)


def _predict_model_dispatch(
    *,
    model_id: str,
    row_ids: np.ndarray,
    context: ModelDataContext,
    fitted: Mapping[str, Any],
    **kwargs: Any,
) -> dict[str, np.ndarray]:
    common = {
        "context": context,
        "row_ids": row_ids,
        "fitted": fitted,
    }
    if str(model_id) == "lgbm_lambdarank_multioutput":
        return _predict_lgbm_adapter(**common)
    if str(model_id) == "tabm_multioutput":
        return _predict_tabm_adapter(**common, **kwargs)
    if str(model_id) == "patchtst_student_t_path":
        return _predict_patchtst_adapter(**common, **kwargs)
    if str(model_id) == "market_industry_deepsets":
        return _predict_deepsets_adapter(**common, **kwargs)
    if str(model_id) == "deephit_competing_risk":
        return _predict_deephit_adapter(**common, **kwargs)
    if str(model_id) == "neuralndcg_listwise_mlp":
        return _predict_neuralndcg_adapter(**common, **kwargs)
    raise KeyError(model_id)


class FrozenModelAdapter:
    model_id: str = ""
    capabilities: tuple[str, ...] = ()

    def fit(
        self,
        train: np.ndarray,
        development: np.ndarray,
        resolved_config: Mapping[str, Any],
        **kwargs: Any,
    ) -> dict[str, Any]:
        return _fit_model_dispatch(
            model_id=self.model_id,
            train_rows=np.asarray(train, dtype=np.int64),
            development_rows=np.asarray(development, dtype=np.int64),
            resolved_config=resolved_config,
            **kwargs,
        )

    def predict(
        self,
        full_candidate_test_set: np.ndarray,
        **kwargs: Any,
    ) -> dict[str, np.ndarray]:
        return _predict_model_dispatch(
            model_id=self.model_id,
            row_ids=np.asarray(full_candidate_test_set, dtype=np.int64),
            **kwargs,
        )


class LightGBMAdapter(FrozenModelAdapter):
    model_id = "lgbm_lambdarank_multioutput"
    capabilities = ("fill", "scalar_quality")


class TabMAdapter(FrozenModelAdapter):
    model_id = "tabm_multioutput"
    capabilities = ("fill", "scalar_quality")


class PatchTSTAdapter(FrozenModelAdapter):
    model_id = "patchtst_student_t_path"
    capabilities = ("fill", "distribution", "quantile", "scalar_quality")


class DeepSetsAdapter(FrozenModelAdapter):
    model_id = "market_industry_deepsets"
    capabilities = ("fill", "scalar_quality")


class DeepHitAdapter(FrozenModelAdapter):
    model_id = "deephit_competing_risk"
    capabilities = ("fill", "hazard", "scalar_quality")


class NeuralNDCGAdapter(FrozenModelAdapter):
    model_id = "neuralndcg_listwise_mlp"
    capabilities = ("fill", "scalar_quality")


MODEL_ADAPTERS: dict[str, type[FrozenModelAdapter]] = {
    adapter.model_id: adapter
    for adapter in (
        LightGBMAdapter,
        TabMAdapter,
        PatchTSTAdapter,
        DeepSetsAdapter,
        DeepHitAdapter,
        NeuralNDCGAdapter,
    )
}


def get_model_adapter(model_id: str) -> FrozenModelAdapter:
    if str(model_id) not in MODEL_ADAPTERS:
        raise KeyError(model_id)
    return MODEL_ADAPTERS[str(model_id)]()


DEFAULT_MICRO_BATCH = {
    "lgbm_lambdarank_multioutput": 0,
    "tabm_multioutput": 512,
    "patchtst_student_t_path": 256,
    "market_industry_deepsets": 2048,
    "deephit_competing_risk": 512,
    "neuralndcg_listwise_mlp": 256,
}

DEFAULT_VALIDATION_BATCH = {
    "tabm_multioutput": 4096,
    "patchtst_student_t_path": 1024,
    "market_industry_deepsets": 0,
    "deephit_competing_risk": 4096,
    "neuralndcg_listwise_mlp": 4096,
}

DEFAULT_PREDICTION_BATCH = dict(DEFAULT_VALIDATION_BATCH)


def _resolved_model_config(
    context: ModelDataContext,
    *,
    model_id: str,
    fold_id: int | str,
    seed: int,
    feature_freeze: Mapping[str, Any],
    resource_retry_index: int = 0,
    runtime_profile: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    model = _model_spec(context, model_id)
    fold = _load_fold_view(context.feature_manifest, fold_id)
    micro_batch = int(DEFAULT_MICRO_BATCH[str(model_id)])
    config = dict(model["config"])
    profile = dict(runtime_profile or {})
    if profile:
        declared = str(profile.get("runtime_profile_sha256", ""))
        check = dict(profile)
        check.pop("runtime_profile_sha256", None)
        if declared != _canonical_json_sha256(check):
            raise ValueError("runtime profile hash mismatch")
        if str(profile.get("model_id", "")) != str(model_id):
            raise ValueError("runtime profile belongs to a different model")
        if str(profile.get("execution_semantics_version", "")) != str(
            MODEL_EXECUTION_SEMANTICS.get(str(model_id), "")
        ):
            raise ValueError("runtime profile execution semantics mismatch")
        if str(dict(profile.get("hardware", {}) or {}).get("hardware_sha256", "")) != str(
            _hardware_fingerprint()["hardware_sha256"]
        ):
            raise ValueError("runtime profile belongs to different hardware")
        micro_batch = int(profile.get("train_micro_batch", micro_batch))
    if str(model_id) in {
        "tabm_multioutput",
        "patchtst_student_t_path",
        "deephit_competing_risk",
    }:
        effective_batch = int(config["effective_batch"])
        if int(resource_retry_index):
            micro_batch = max(1, micro_batch // (2 ** int(resource_retry_index)))
        if effective_batch % max(micro_batch, 1):
            raise ValueError("micro batch must divide the frozen effective batch")
        accumulation = effective_batch // micro_batch
    else:
        effective_batch = int(
            config.get(
                "effective_candidates_per_step",
                config.get("training_list_size", 0),
            )
        )
        accumulation = 1
    payload = {
        "model_id": str(model_id),
        "slot": str(model["slot"]),
        "capabilities": list(model["capabilities"]),
        "frozen_model_config": config,
        "feature_profile": str(feature_freeze["selected_profile"]),
        "feature_freeze_sha256": str(feature_freeze["feature_freeze_sha256"]),
        "fold_id": str(fold_id),
        "fold_view_sha256": str(fold["view_sha256"]),
        "seed": int(seed),
        "micro_batch": int(micro_batch),
        "gradient_accumulation": int(accumulation),
        "effective_batch": int(effective_batch),
        "resource_retry_index": int(resource_retry_index),
        "resource_only_change": bool(resource_retry_index),
        "study_contract_sha256": str(context.study["contract_sha256"]),
        "target_sha256": str(context.target_manifest["frozen_target_sha256"]),
        "feature_sha256": str(
            context.feature_manifest["resolved_feature_view_sha256"]
        ),
    }
    if str(model_id) in NEURAL_MODEL_IDS:
        payload.update(
            {
                "execution_semantics_version": MODEL_EXECUTION_SEMANTICS[
                    str(model_id)
                ],
                "runtime_autotune_version": RUNTIME_AUTOTUNE_VERSION,
                "runtime_profile_sha256": str(
                    profile.get("runtime_profile_sha256", "static_fallback")
                ),
                "validation_batch": int(
                    profile.get(
                        "validation_batch",
                        DEFAULT_VALIDATION_BATCH[str(model_id)],
                    )
                ),
                "prediction_batch": int(
                    profile.get(
                        "prediction_batch",
                        DEFAULT_PREDICTION_BATCH[str(model_id)],
                    )
                ),
                "prefetch_depth": int(profile.get("prefetch_depth", 2)),
                "pinned_memory": bool(profile.get("pinned_memory", False)),
            }
        )
    if "training_budget_amendment_sha256" in model:
        payload.update(
            {
                "training_budget_amendment_sha256": str(
                    model["training_budget_amendment_sha256"]
                ),
                "training_budget_amended_fields": list(
                    TRAINING_BUDGET_MUTABLE_FIELDS
                ),
            }
        )
    payload["resolved_config_sha256"] = _canonical_json_sha256(payload)
    payload["resume_key_sha256"] = _canonical_json_sha256(
        {
            "contract_sha256": payload["study_contract_sha256"],
            "target_sha256": payload["target_sha256"],
            "feature_sha256": payload["feature_sha256"],
            "resolved_config_sha256": payload["resolved_config_sha256"],
            "fold_id": payload["fold_id"],
            "seed": payload["seed"],
        }
    )
    return payload


def _candidate_frame_for_rows(
    context: ModelDataContext,
    row_ids: np.ndarray,
) -> pd.DataFrame:
    rows = np.asarray(row_ids, dtype=np.int64)
    if not rows.size:
        return pd.DataFrame()
    if bool(np.any(rows[1:] != rows[:-1] + 1)):
        raise ValueError("prediction candidate rows must be one contiguous fold range")
    frame = pq.read_table(
        context.candidate_index_path,
        columns=[
            "candidate_id",
            "trade_date",
            "year",
            "date_idx",
            "symbol_idx",
            "symbol",
            "entry_filled",
            "price_label_valid",
        ],
        filters=[
            ("candidate_id", ">=", int(rows[0])),
            ("candidate_id", "<=", int(rows[-1])),
        ],
    ).to_pandas()
    frame = frame.sort_values("candidate_id", kind="stable").reset_index(drop=True)
    if not np.array_equal(
        frame["candidate_id"].to_numpy(dtype=np.int64, copy=False), rows
    ):
        raise ValueError("candidate index filter did not reproduce the requested fold rows")
    if bool((frame["year"].astype(int) >= 2026).any()):
        raise RuntimeError("2026 candidates are forbidden in model prediction")
    return frame


def _fixed_size_list_array(values: np.ndarray) -> pa.Array:
    array = np.asarray(values, dtype=np.float32)
    if array.ndim != 2:
        raise ValueError("fixed-size prediction output must be two-dimensional")
    return pa.FixedSizeListArray.from_arrays(
        pa.array(array.reshape(-1), type=pa.float32()),
        int(array.shape[1]),
    )


def _write_prediction_artifact(
    *,
    context: ModelDataContext,
    row_ids: np.ndarray,
    predictions: Mapping[str, np.ndarray],
    model_id: str,
    fold_year: int,
    seed: int,
    resolved_config: Mapping[str, Any],
    output_path: Path,
    chunk_size: int = 100_000,
) -> dict[str, Any]:
    rows = np.asarray(row_ids, dtype=np.int64)
    coordinates = _candidate_frame_for_rows(context, rows)
    required_prediction = {
        "fill_probability",
        "conditional_quality_prediction",
        "selection_score",
    }
    if not required_prediction.issubset(predictions):
        raise ValueError(
            f"model prediction lacks {sorted(required_prediction.difference(predictions))}"
        )
    for name, values in predictions.items():
        if int(np.asarray(values).shape[0]) != len(rows):
            raise ValueError(f"prediction output {name} does not cover every candidate")
        if not bool(np.isfinite(np.asarray(values)).all()):
            raise ValueError(f"prediction output {name} contains non-finite values")
    target = context.target
    flags = np.asarray(context.flags[rows], dtype=np.uint8)
    filled = (flags & TARGET_FLAG_ENTRY_FILLED) != 0
    conditional_truth = np.asarray(
        target[rows, TARGET_FIELD_INDEX["conditional_relevance"]],
        dtype=np.float32,
    )
    truth_columns = {
        "frozen_target_quality": np.asarray(
            target[rows, TARGET_FIELD_INDEX["action_relevance"]], dtype=np.float32
        ),
        "common_sustained_action_utility": np.asarray(
            target[
                rows,
                TARGET_FIELD_INDEX["common_sustained_action_utility"],
            ],
            dtype=np.float32,
        ),
        "d5_net_return": np.asarray(
            target[rows, TARGET_FIELD_INDEX["r5_net"]], dtype=np.float32
        ),
        "d10_net_return": np.asarray(
            target[rows, TARGET_FIELD_INDEX["r10_net"]], dtype=np.float32
        ),
        "d20_net_return": np.asarray(
            target[rows, TARGET_FIELD_INDEX["r20_net"]], dtype=np.float32
        ),
        "mdd20": np.asarray(
            target[rows, TARGET_FIELD_INDEX["mdd20"]], dtype=np.float32
        ),
        "fade20": np.asarray(
            target[rows, TARGET_FIELD_INDEX["post_peak_fade20"]],
            dtype=np.float32,
        ),
        "mfe20": np.asarray(
            target[rows, TARGET_FIELD_INDEX["mfe20"]], dtype=np.float32
        ),
        "mae20": np.asarray(
            target[rows, TARGET_FIELD_INDEX["mae20"]], dtype=np.float32
        ),
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_name(output_path.name + ".tmp")
    writer: pq.ParquetWriter | None = None
    try:
        for start in range(0, len(rows), int(chunk_size)):
            stop = min(start + int(chunk_size), len(rows))
            frame = coordinates.iloc[start:stop]
            arrays: dict[str, pa.Array] = {
                "trade_date": pa.array(frame["trade_date"].astype(str)),
                "date_idx": pa.array(frame["date_idx"].astype(np.int32)),
                "symbol": pa.array(frame["symbol"].astype(str)),
                "symbol_idx": pa.array(frame["symbol_idx"].astype(np.int32)),
                "candidate_id": pa.array(frame["candidate_id"].astype(np.int64)),
                "candidate_mask": pa.array(np.ones(stop - start, dtype=bool)),
                "fill_label_mask": pa.array(np.ones(stop - start, dtype=bool)),
                "quality_label_mask": pa.array(
                    filled[start:stop] & np.isfinite(conditional_truth[start:stop])
                ),
                "entry_filled": pa.array(filled[start:stop]),
                "fill_probability": pa.array(
                    np.asarray(predictions["fill_probability"])[start:stop],
                    type=pa.float32(),
                ),
                "conditional_quality_prediction": pa.array(
                    np.asarray(predictions["conditional_quality_prediction"])[start:stop],
                    type=pa.float32(),
                ),
                "selection_score": pa.array(
                    np.asarray(predictions["selection_score"])[start:stop],
                    type=pa.float32(),
                ),
            }
            for name, values in truth_columns.items():
                arrays[name] = pa.array(values[start:stop], type=pa.float32())
            arrays["terminal_failure"] = pa.array(
                (flags[start:stop] & TARGET_FLAG_TERMINAL_FAILURE) != 0
            )
            if "path_location" in predictions:
                path_columns = np.asarray(
                    [
                        TARGET_FIELD_INDEX[f"close_net_log_d{day:02d}"]
                        for day in range(1, 21)
                    ],
                    dtype=np.int64,
                )
                true_path = np.asarray(
                    target[np.ix_(rows[start:stop], path_columns)],
                    dtype=np.float32,
                )
                arrays["true_path_net_log"] = _fixed_size_list_array(true_path)
            if "joint_event_probability" in predictions:
                arrays["competing_event_code"] = pa.array(
                    np.asarray(
                        target[
                            rows[start:stop],
                            TARGET_FIELD_INDEX["competing_event_code"],
                        ],
                        dtype=np.float32,
                    ),
                    type=pa.float32(),
                )
                arrays["competing_event_time"] = pa.array(
                    np.asarray(
                        target[
                            rows[start:stop],
                            TARGET_FIELD_INDEX["competing_event_time"],
                        ],
                        dtype=np.float32,
                    ),
                    type=pa.float32(),
                )
            arrays.update(
                {
                    "model_id": pa.array([str(model_id)] * (stop - start)),
                    "fold_year": pa.array(
                        np.full(stop - start, int(fold_year), dtype=np.int16)
                    ),
                    "seed": pa.array(
                        np.full(stop - start, int(seed), dtype=np.int32)
                    ),
                    "contract_sha256": pa.array(
                        [str(context.study["contract_sha256"])] * (stop - start)
                    ),
                    "target_sha256": pa.array(
                        [str(context.target_manifest["frozen_target_sha256"])]
                        * (stop - start)
                    ),
                    "feature_sha256": pa.array(
                        [str(context.feature_manifest["resolved_feature_view_sha256"])]
                        * (stop - start)
                    ),
                    "resolved_config_sha256": pa.array(
                        [str(resolved_config["resolved_config_sha256"])]
                        * (stop - start)
                    ),
                }
            )
            for name, values in predictions.items():
                if name in required_prediction:
                    continue
                current = np.asarray(values)[start:stop]
                arrays[name] = (
                    _fixed_size_list_array(current)
                    if current.ndim == 2
                    else pa.array(current)
                )
            table = pa.table(arrays)
            if writer is None:
                writer = pq.ParquetWriter(
                    temporary,
                    table.schema,
                    compression="zstd",
                    use_dictionary=True,
                )
            writer.write_table(table)
    finally:
        if writer is not None:
            writer.close()
    if writer is None:
        raise ValueError("cannot write an empty prediction artifact")
    os.replace(temporary, output_path)
    return {
        "path": str(output_path.resolve()),
        "sha256": _file_sha256(output_path),
        "size": int(output_path.stat().st_size),
        "candidate_count": int(len(rows)),
        "schema": pq.ParquetFile(output_path).schema_arrow.names,
    }


def _expected_calibration_error(
    labels: np.ndarray,
    probabilities: np.ndarray,
    *,
    bins: int = 20,
) -> float:
    truth = np.asarray(labels, dtype=np.float64)
    score = np.asarray(probabilities, dtype=np.float64)
    edges = np.linspace(0.0, 1.0, int(bins) + 1)
    total = max(len(truth), 1)
    error = 0.0
    for index in range(int(bins)):
        mask = (score >= edges[index]) & (
            score <= edges[index + 1]
            if index == int(bins) - 1
            else score < edges[index + 1]
        )
        if bool(mask.any()):
            error += float(mask.sum() / total) * abs(
                float(score[mask].mean()) - float(truth[mask].mean())
            )
    return float(error)


def _quick_prediction_metrics(
    *,
    context: ModelDataContext,
    row_ids: np.ndarray,
    predictions: Mapping[str, np.ndarray],
) -> tuple[pd.DataFrame, dict[str, Any]]:
    from sklearn.metrics import (
        average_precision_score,
        brier_score_loss,
        roc_auc_score,
    )

    rows = np.asarray(row_ids, dtype=np.int64)
    score = np.asarray(predictions["selection_score"], dtype=np.float32)
    utility = np.asarray(
        context.target[
            rows, TARGET_FIELD_INDEX["common_sustained_action_utility"]
        ],
        dtype=np.float32,
    )
    action = np.asarray(
        context.target[rows, TARGET_FIELD_INDEX["action_relevance"]],
        dtype=np.float32,
    )
    dates = np.asarray(context.candidate_date_idx[rows], dtype=np.int32)
    symbols = np.asarray(context.candidate_symbol_idx[rows], dtype=np.int32)
    date_values = np.asarray(context.pack_manifest["date_values"], dtype=object)
    coordinates = pd.DataFrame(
        {
            "trade_date": date_values[dates],
            "date_idx": dates,
            "symbol_idx": symbols,
        }
    )
    utility_daily = _daily_top_fraction_utility(
        coordinates,
        score,
        utility,
    )
    ranking_daily, ranking = daily_ranking_metrics(
        date_idx=dates,
        action_relevance=action,
        selection_score=score,
    )
    daily = utility_daily.merge(
        ranking_daily,
        on="date_idx",
        how="left",
        validate="one_to_one",
    )
    filled = (
        np.asarray(context.flags[rows], dtype=np.uint8) & TARGET_FLAG_ENTRY_FILLED
    ) != 0
    probability = np.clip(
        np.asarray(predictions["fill_probability"], dtype=np.float64), 0.0, 1.0
    )
    metrics = {
        "daily_top1pct_common_utility": float(daily["top1pct_u_mean"].mean()),
        **ranking,
        "fill_roc_auc": float(roc_auc_score(filled, probability)),
        "fill_pr_auc": float(average_precision_score(filled, probability)),
        "fill_brier": float(brier_score_loss(filled, probability)),
        "fill_ece": _expected_calibration_error(filled, probability),
        "candidate_count": int(len(rows)),
        "date_count": int(len(daily)),
    }
    return daily, metrics


def _run_model_task(
    *,
    context: ModelDataContext,
    model_id: str,
    fold_id: int | str,
    seed: int,
    feature_freeze: Mapping[str, Any],
    output_dir: Path,
    resource_retry_index: int = 0,
    runtime_profile: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    protection_before = _verify_protected_bindings(context.study)
    fold = _load_fold_view(context.feature_manifest, fold_id)
    ranges = dict(fold["ranges"])
    train_rows = _row_ids_for_date_range(
        context.candidate_date_idx,
        int(ranges["train"]["date_idx_start"]),
        int(ranges["train"]["date_idx_end"]),
    )
    development_rows = _row_ids_for_date_range(
        context.candidate_date_idx,
        int(ranges["development"]["date_idx_start"]),
        int(ranges["development"]["date_idx_end"]),
    )
    test_rows = _row_ids_for_date_range(
        context.candidate_date_idx,
        int(ranges["test"]["date_idx_start"]),
        int(ranges["test"]["date_idx_end"]),
    )
    test_years = {
        int(str(context.pack_manifest["date_values"][int(item)])[:4])
        for item in np.unique(context.candidate_date_idx[test_rows])
    }
    expected_test_year = int(fold["test_year"])
    if test_years != {expected_test_year} or expected_test_year >= 2026:
        raise RuntimeError(
            f"model task test boundary violation: expected={expected_test_year} got={sorted(test_years)}"
        )
    resolved_config = _resolved_model_config(
        context,
        model_id=model_id,
        fold_id=fold_id,
        seed=int(seed),
        feature_freeze=feature_freeze,
        resource_retry_index=int(resource_retry_index),
        runtime_profile=runtime_profile,
    )
    _atomic_write_json(output_dir / "resolved_config.json", resolved_config)
    budget_extension: dict[str, Any] | None = None
    budget_extension_path = output_dir / "budget_extension_provenance.json"
    if budget_extension_path.exists():
        budget_extension = json.loads(budget_extension_path.read_text(encoding="utf-8"))
        declared = str(budget_extension.get("provenance_sha256", ""))
        check = dict(budget_extension)
        check.pop("provenance_sha256", None)
        if declared != _canonical_json_sha256(check):
            raise ValueError("budget extension provenance hash mismatch")
        if str(budget_extension.get("destination_resume_key_sha256", "")) != str(
            resolved_config["resume_key_sha256"]
        ):
            raise ValueError("budget extension destination resume key mismatch")
        if str(budget_extension.get("amendment_sha256", "")) != str(
            resolved_config.get("training_budget_amendment_sha256", "")
        ):
            raise ValueError("budget extension amendment mismatch")
    monitor = _TrainingMonitor(
        output_dir=output_dir,
        base={
            "started_at": _now(),
            "model_id": model_id,
            "fold_id": str(fold_id),
            "seed": int(seed),
            "resume_key_sha256": resolved_config["resume_key_sha256"],
            "resolved_config_sha256": resolved_config["resolved_config_sha256"],
            "execution_semantics_version": resolved_config.get(
                "execution_semantics_version"
            ),
            "train_candidate_count": int(len(train_rows)),
            "development_candidate_count": int(len(development_rows)),
            "test_candidate_count": int(len(test_rows)),
            "budget_extension_provenance_sha256": (
                budget_extension.get("provenance_sha256")
                if budget_extension is not None
                else None
            ),
        },
    )
    monitor.report(
        status="training",
        phase="training",
        samples_completed=0,
        samples_total=len(train_rows),
        epoch=(
            int(budget_extension["start_epoch"])
            if budget_extension is not None
            else 1
        ),
        max_epochs=int(
            dict(_model_spec(context, model_id)["config"]).get("max_epochs", 1)
        ),
        step=0,
        steps_total=max(
            1,
            math.ceil(
                len(train_rows) / max(int(resolved_config.get("effective_batch", 1)), 1)
            ),
        ),
        force=True,
        event="training_started",
        extra={
            "continued_from_checkpoint": bool(budget_extension is not None),
        },
    )
    adapter = get_model_adapter(model_id)
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
    try:
        fitted = adapter.fit(
            train_rows,
            development_rows,
            resolved_config,
            context=context,
            output_dir=output_dir,
            seed=int(seed),
            monitor=monitor,
        )
    except TrainingPaused:
        raise
    except Exception as exc:
        monitor.report(
            status="failed",
            phase=str(monitor.latest.get("phase", "training")),
            samples_completed=int(monitor.latest.get("samples_completed", 0)),
            samples_total=int(monitor.latest.get("samples_total", len(train_rows))),
            epoch=monitor.latest.get("epoch"),
            max_epochs=monitor.latest.get("max_epochs"),
            step=monitor.latest.get("step"),
            steps_total=monitor.latest.get("steps_total"),
            force=True,
            event="failed",
            extra={"error_type": type(exc).__name__, "error": str(exc)},
        )
        raise
    monitor.report(
        status="predicting",
        phase="prediction",
        samples_completed=0,
        samples_total=len(test_rows),
        force=True,
        event="prediction_started",
    )
    inference_started = time.perf_counter()
    predictions = adapter.predict(
        test_rows,
        context=context,
        fitted=fitted,
        monitor=monitor,
    )
    inference_seconds = time.perf_counter() - inference_started
    prediction_meta = _write_prediction_artifact(
        context=context,
        row_ids=test_rows,
        predictions=predictions,
        model_id=model_id,
        fold_year=expected_test_year,
        seed=int(seed),
        resolved_config=resolved_config,
        output_path=output_dir / "predictions.parquet",
    )
    daily, metrics = _quick_prediction_metrics(
        context=context,
        row_ids=test_rows,
        predictions=predictions,
    )
    _atomic_write_parquet(output_dir / "daily_metrics.parquet", daily)
    gpu_peak = (
        int(torch.cuda.max_memory_allocated()) if torch.cuda.is_available() else 0
    )
    protection_after = _verify_protected_bindings(context.study)
    if protection_after != protection_before:
        raise RuntimeError("protected-object hashes changed during model task")
    summary = {
        "status": "completed",
        "completed_at": _now(),
        "model_id": model_id,
        "fold_id": str(fold_id),
        "fold_year": expected_test_year,
        "seed": int(seed),
        "resume_key_sha256": resolved_config["resume_key_sha256"],
        "resolved_config_sha256": resolved_config["resolved_config_sha256"],
        "adapter": fitted["artifact"],
        "prediction": prediction_meta,
        "daily_metrics": {
            "path": str((output_dir / "daily_metrics.parquet").resolve()),
            "sha256": _file_sha256(output_dir / "daily_metrics.parquet"),
        },
        "metrics": metrics,
        "inference_seconds": float(inference_seconds),
        "inference_microseconds_per_candidate": float(
            inference_seconds * 1.0e6 / max(len(test_rows), 1)
        ),
        "gpu_peak_memory_bytes": gpu_peak,
        "protection_before": protection_before,
        "protection_after": protection_after,
    }
    if budget_extension is not None:
        summary["budget_extension"] = budget_extension
    _atomic_write_json(output_dir / "task_summary.json", summary)
    monitor.report(
        status="completed",
        phase="completed",
        samples_completed=len(test_rows),
        samples_total=len(test_rows),
        force=True,
        event="completed",
        extra={"task_summary": str((output_dir / "task_summary.json").resolve())},
    )
    return summary


def _model_research_audit(context: ModelDataContext) -> dict[str, Any]:
    binding = dict(context.research_freeze["evidence_bindings"]["external_review"])
    path = _resolve_path(str(binding["path"]))
    if _file_sha256(path) != str(binding["sha256"]):
        raise ValueError("external method review changed before model screening")
    review = json.loads(path.read_text(encoding="utf-8"))
    shortlist = list(review.get("formal_model_shortlist", []) or [])
    if tuple(str(item["model_id"]) for item in shortlist) != MODEL_IDS:
        raise ValueError("formal model shortlist no longer matches the research freeze")
    methods = {
        str(item["method_id"]): dict(item)
        for item in list(review.get("methods", []) or [])
    }
    sources = {
        str(item["source_id"]): dict(item)
        for item in list(review.get("sources", []) or [])
    }
    records: list[dict[str, Any]] = []
    for item in shortlist:
        basis = [str(value) for value in list(item.get("basis", []) or [])]
        if not basis or any(name not in methods for name in basis):
            raise ValueError(f"model {item['model_id']} has an unresolved research basis")
        source_ids = sorted(
            {
                str(source_id)
                for name in basis
                for source_id in list(methods[name].get("source_ids", []) or [])
            }
        )
        if not source_ids or any(source_id not in sources for source_id in source_ids):
            raise ValueError(f"model {item['model_id']} has unresolved primary sources")
        source_records = [sources[source_id] for source_id in source_ids]
        if not any(str(source.get("kind", "")).startswith("paper") for source in source_records):
            raise ValueError(f"model {item['model_id']} lacks a primary paper")
        repository_sources = [
            source
            for source in source_records
            if str(source.get("kind", "")).startswith("repository")
        ]
        for source in repository_sources:
            if not str(source.get("commit", "") or ""):
                raise ValueError(
                    f"repository source {source['source_id']} lacks an exact commit/tag"
                )
            if not str(source.get("license", "") or ""):
                raise ValueError(
                    f"repository source {source['source_id']} lacks a license audit"
                )
        records.append(
            {
                "model_id": str(item["model_id"]),
                "slot": str(item["slot"]),
                "basis": basis,
                "source_ids": source_ids,
                "repository_count": len(repository_sources),
                "implementation": (
                    "installed_official_library"
                    if str(item["model_id"])
                    in {"lgbm_lambdarank_multioutput", "tabm_multioutput"}
                    else "independent_from_audited_paper_formulas"
                ),
                "passed": True,
            }
        )
    return {
        "status": "passed",
        "review_path": str(path.resolve()),
        "review_sha256": _file_sha256(path),
        "models": records,
    }


def _autotune_train_candidates(model_id: str, effective_batch: int) -> list[int]:
    if str(model_id) == "patchtst_student_t_path":
        preferred = [64, 128, 256, 512]
    elif str(model_id) in {"tabm_multioutput", "deephit_competing_risk"}:
        preferred = [256, 512, 1024, 2048, 4096]
    else:
        return [int(DEFAULT_MICRO_BATCH[str(model_id)])]
    return [
        value
        for value in preferred
        if value <= int(effective_batch) and int(effective_batch) % value == 0
    ]


def _autotune_inference_candidates(model_id: str) -> list[int]:
    if str(model_id) == "patchtst_student_t_path":
        return [256, 512, 1024, 2048]
    if str(model_id) in {
        "tabm_multioutput",
        "deephit_competing_risk",
        "neuralndcg_listwise_mlp",
    }:
        return [512, 1024, 2048, 4096, 8192]
    return [0]


def _autotune_model_runtime(
    *,
    context: ModelDataContext,
    model_id: str,
    model: nn.Module,
    config: Mapping[str, Any],
    preprocessor: Mapping[str, Any],
    train_rows: np.ndarray,
    device: torch.device,
) -> dict[str, Any]:
    hardware = _hardware_fingerprint()
    semantics = MODEL_EXECUTION_SEMANTICS[str(model_id)]
    amp_enabled = bool(config.get("amp", True) and device.type == "cuda")
    if str(model_id) in {
        "tabm_multioutput",
        "patchtst_student_t_path",
        "deephit_competing_risk",
    }:
        effective_batch = int(config["effective_batch"])
    elif str(model_id) == "market_industry_deepsets":
        effective_batch = int(config["effective_candidates_per_step"])
    else:
        effective_batch = int(config["training_list_size"])
    fallback_micro = int(DEFAULT_MICRO_BATCH[str(model_id)])
    fallback_validation = int(DEFAULT_VALIDATION_BATCH[str(model_id)])
    fallback_prediction = int(DEFAULT_PREDICTION_BATCH[str(model_id)])
    profile: dict[str, Any] = {
        "schema": RUNTIME_AUTOTUNE_VERSION,
        "created_at": _now(),
        "model_id": str(model_id),
        "execution_semantics_version": semantics,
        "hardware": hardware,
        "effective_batch": effective_batch,
        "train_micro_batch": fallback_micro,
        "validation_batch": fallback_validation,
        "prediction_batch": fallback_prediction,
        "prefetch_depth": 2 if device.type == "cuda" else 0,
        "pinned_memory": False,
        "train_benchmarks": [],
        "inference_benchmarks": [],
        "selection_rule": "highest measured samples_per_second with at least 15pct CUDA memory headroom; deterministic smaller-batch tie break",
    }
    if str(model_id) == "market_industry_deepsets":
        profile["train_micro_batch"] = 0
    if device.type != "cuda":
        profile["runtime_profile_sha256"] = _canonical_json_sha256(profile)
        return profile

    initial_state = _clone_torch_state(model)
    total_memory = int(torch.cuda.get_device_properties(0).total_memory)
    f0_material = (
        _open_reference_f0_material(context.pack_manifest)
        if str(model_id) == "patchtst_student_t_path"
        else None
    )

    def _load_inputs(rows: np.ndarray) -> tuple[torch.Tensor, ...]:
        if str(model_id) == "patchtst_student_t_path":
            x = _f0_batch(
                context,
                rows,
                preprocessor,
                material=f0_material,
            )
            return (torch.from_numpy(x).to(device),)
        x_num, x_cat = _snapshot_batch(context, rows, preprocessor)
        return (
            torch.from_numpy(x_num).to(device),
            torch.from_numpy(x_cat).to(device),
        )

    def _forward(inputs: tuple[torch.Tensor, ...]) -> dict[str, torch.Tensor]:
        with torch.autocast(device.type, dtype=torch.float16, enabled=amp_enabled):
            return model(*inputs)

    def _tabm_micro_objective(
        rows: np.ndarray,
        output: Mapping[str, torch.Tensor],
        *,
        quality_denominator: float,
        fill_denominator: float,
    ) -> torch.Tensor:
        material = _target_material(context, rows)
        quality = torch.from_numpy(
            np.nan_to_num(material["conditional_quality"], nan=0.0)
        ).to(device)
        quality_mask = torch.from_numpy(
            (
                np.isfinite(material["conditional_quality"])
                & (material["filled"] > 0.5)
            ).astype(np.float32)
        ).to(device)
        fill = torch.from_numpy(material["filled"]).to(device)
        members = int(config["ensemble_size"])
        qloss = torch_functional.smooth_l1_loss(
            output["quality"],
            quality[:, None].expand(-1, members),
            reduction="none",
        )
        qloss = global_normalized_component(
            (qloss * quality_mask[:, None]).sum(), quality_denominator
        )
        floss = global_normalized_component(
            torch_functional.binary_cross_entropy_with_logits(
                output["fill_logit"],
                fill[:, None].expand(-1, members),
                reduction="sum",
            ),
            fill_denominator,
        )
        weights = dict(config["loss_weights"])
        return float(weights["quality_huber"]) * qloss + float(
            weights["fill_bce"]
        ) * floss

    def _patch_micro_objective(
        rows: np.ndarray,
        output: Mapping[str, torch.Tensor],
        *,
        path_denominator: float,
        fill_denominator: float,
    ) -> torch.Tensor:
        material = _target_material(context, rows)
        target_path = torch.from_numpy(
            np.nan_to_num(material["path"], nan=0.0)
        ).to(device)
        valid_path = torch.from_numpy(
            (
                material["path_available"]
                & (material["filled"] > 0.5)
                & np.isfinite(material["path"]).all(axis=1)
            ).astype(np.float32)
        ).to(device)
        terminal = torch.from_numpy(material["terminal"]).to(device)
        fill = torch.from_numpy(material["filled"]).to(device)
        nll_by_row = student_t_nll(
            target_path,
            output["location"],
            output["scale"],
            df=float(config["student_t_df"]),
        ).mean(dim=1)
        path_loss = global_normalized_component(
            (nll_by_row * valid_path).sum(), path_denominator
        )
        terminal_loss = global_normalized_component(
            (
                torch_functional.binary_cross_entropy_with_logits(
                    output["terminal_logit"].float(),
                    terminal.float(),
                    reduction="none",
                )
                * valid_path
            ).sum(),
            path_denominator,
        )
        fill_loss = global_normalized_component(
            torch_functional.binary_cross_entropy_with_logits(
                output["fill_logit"].float(), fill.float(), reduction="sum"
            ),
            fill_denominator,
        )
        weights = dict(config["loss_weights"])
        return (
            float(weights["path_nll"]) * path_loss
            + float(weights["terminal_bce"]) * terminal_loss
            + float(weights["fill_bce"]) * fill_loss
        )

    def _deephit_objective(
        rows: np.ndarray,
        joint_logits: torch.Tensor,
        fill_logit: torch.Tensor,
    ) -> torch.Tensor:
        material = _target_material(context, rows)
        valid = (
            material["path_available"]
            & (material["filled"] > 0.5)
            & np.isfinite(material["event_code"])
            & np.isfinite(material["event_time"])
        )
        event_code = torch.from_numpy(
            np.nan_to_num(material["event_code"], nan=0.0).astype(np.int64)
        ).to(device)
        event_time = torch.from_numpy(
            np.nan_to_num(material["event_time"], nan=20.0).astype(np.int64)
        ).to(device)
        valid_tensor = torch.from_numpy(valid).to(device)
        fill = torch.from_numpy(material["filled"]).to(device)
        if bool(valid.any()):
            joint_target = deephit_joint_targets(event_code, event_time)
            likelihood = torch_functional.cross_entropy(
                joint_logits[valid_tensor], joint_target[valid_tensor]
            )
            probability = torch.softmax(joint_logits[valid_tensor], dim=1)
            ranking = deephit_ranking_loss(
                probability,
                event_code[valid_tensor],
                event_time[valid_tensor],
            )
        else:
            likelihood = joint_logits.sum() * 0.0
            ranking = likelihood
        fill_loss = torch_functional.binary_cross_entropy_with_logits(
            fill_logit, fill.float()
        )
        weights = dict(config["loss_weights"])
        return (
            float(weights["likelihood"]) * likelihood
            + float(weights["cause_ranking"]) * ranking
            + float(weights["fill_bce"]) * fill_loss
        )

    if str(model_id) in {
        "tabm_multioutput",
        "patchtst_student_t_path",
        "deephit_competing_risk",
    }:
        benchmark_step_count = 6
        warmup_step_count = 2
        benchmark_rows = deterministic_epoch_rows(
            train_rows,
            context.candidate_date_idx,
            seed=90210,
        )[: effective_batch * benchmark_step_count]
        for candidate in _autotune_train_candidates(model_id, effective_batch):
            record: dict[str, Any] = {"micro_batch": int(candidate)}
            optimizer = None
            scaler = None
            loss = None
            inputs = None
            output = None
            joint_parts = None
            fill_parts = None
            full_material = None
            try:
                model.load_state_dict(initial_state)
                model.train()
                optimizer, fused = _adamw_optimizer(model, config, device=device)
                scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)
                torch.cuda.empty_cache()
                torch.cuda.reset_peak_memory_stats()
                measured_seconds = 0.0
                measured_samples = 0
                for benchmark_step in range(benchmark_step_count):
                    effective_rows = benchmark_rows[
                        benchmark_step * effective_batch : (benchmark_step + 1)
                        * effective_batch
                    ]
                    optimizer.zero_grad(set_to_none=True)
                    started = time.perf_counter()
                    if str(model_id) == "deephit_competing_risk":
                        joint_parts: list[torch.Tensor] = []
                        fill_parts: list[torch.Tensor] = []
                        for rows in _micro_batches(effective_rows, candidate):
                            output = _forward(_load_inputs(rows))
                            joint_parts.append(output["joint_logits"].float())
                            fill_parts.append(output["fill_logit"].float())
                        loss = _deephit_objective(
                            effective_rows,
                            torch.cat(joint_parts, dim=0),
                            torch.cat(fill_parts, dim=0),
                        )
                        scaler.scale(loss).backward()
                    else:
                        full_material = _target_material(context, effective_rows)
                        if str(model_id) == "tabm_multioutput":
                            quality_denominator = float(
                                (
                                    np.isfinite(full_material["conditional_quality"])
                                    & (full_material["filled"] > 0.5)
                                ).sum()
                                * int(config["ensemble_size"])
                            )
                            fill_denominator = float(
                                len(effective_rows) * int(config["ensemble_size"])
                            )
                        else:
                            path_denominator = float(
                                (
                                    full_material["path_available"]
                                    & (full_material["filled"] > 0.5)
                                    & np.isfinite(full_material["path"]).all(axis=1)
                                ).sum()
                            )
                            fill_denominator = float(len(effective_rows))
                        for rows in _micro_batches(effective_rows, candidate):
                            output = _forward(_load_inputs(rows))
                            loss = (
                                _tabm_micro_objective(
                                    rows,
                                    output,
                                    quality_denominator=quality_denominator,
                                    fill_denominator=fill_denominator,
                                )
                                if str(model_id) == "tabm_multioutput"
                                else _patch_micro_objective(
                                    rows,
                                    output,
                                    path_denominator=path_denominator,
                                    fill_denominator=fill_denominator,
                                )
                            )
                            scaler.scale(loss).backward()
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        model.parameters(), float(config["gradient_clip_norm"])
                    )
                    scaler.step(optimizer)
                    scaler.update()
                    torch.cuda.synchronize()
                    elapsed = time.perf_counter() - started
                    if benchmark_step >= warmup_step_count:
                        measured_seconds += elapsed
                        measured_samples += len(effective_rows)
                peak_reserved = int(torch.cuda.max_memory_reserved())
                record.update(
                    {
                        "status": "passed",
                        "samples": int(measured_samples),
                        "elapsed_seconds": float(measured_seconds),
                        "samples_per_second": float(
                            measured_samples / measured_seconds
                        ),
                        "gpu_peak_reserved_bytes": peak_reserved,
                        "memory_headroom_ratio": float(
                            max(total_memory - peak_reserved, 0) / total_memory
                        ),
                        "fused_adamw": bool(fused),
                    }
                )
            except RuntimeError as exc:
                record.update(
                    {
                        "status": "failed",
                        "error": str(exc),
                    }
                )
                model.load_state_dict(initial_state)
                model.zero_grad(set_to_none=True)
            finally:
                del (
                    loss,
                    inputs,
                    output,
                    joint_parts,
                    fill_parts,
                    full_material,
                    optimizer,
                    scaler,
                )
                gc.collect()
                torch.cuda.empty_cache()
            profile["train_benchmarks"].append(record)
        eligible = [
            item
            for item in profile["train_benchmarks"]
            if item.get("status") == "passed"
            and float(item.get("memory_headroom_ratio", 0.0)) >= 0.15
        ]
        if eligible:
            best_throughput = max(float(item["samples_per_second"]) for item in eligible)
            near_best = [
                item
                for item in eligible
                if float(item["samples_per_second"]) >= 0.98 * best_throughput
            ]
            selected = min(near_best, key=lambda item: int(item["micro_batch"]))
            profile["train_micro_batch"] = int(selected["micro_batch"])

    if str(model_id) in {
        "market_industry_deepsets",
        "neuralndcg_listwise_mlp",
    }:
        record: dict[str, Any] = {
            "fixed_semantic_batch": (
                "complete_signal_date"
                if str(model_id) == "market_industry_deepsets"
                else "deterministic_list_256"
            )
        }
        optimizer = None
        scaler = None
        loss = None
        output = None
        try:
            model.load_state_dict(initial_state)
            model.train()
            optimizer, fused = _adamw_optimizer(model, config, device=device)
            scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)
            if str(model_id) == "market_industry_deepsets":
                semantic_steps = _row_groups(
                    train_rows, context.candidate_date_idx
                )[:6]
            else:
                semantic_steps = _listwise_epoch_plan(
                    context,
                    train_rows,
                    list_size=int(config["training_list_size"]),
                    epoch=1,
                    seed=90212,
                )[:6]
            measured_seconds = 0.0
            measured_samples = 0
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            for step_index, rows in enumerate(semantic_steps):
                optimizer.zero_grad(set_to_none=True)
                started = time.perf_counter()
                output = _forward(_load_inputs(rows))
                material = _target_material(context, rows)
                fill = torch.from_numpy(material["filled"]).to(device)
                if str(model_id) == "market_industry_deepsets":
                    quality = torch.from_numpy(
                        np.nan_to_num(material["conditional_quality"], nan=0.0)
                    ).to(device)
                    quality_mask = torch.from_numpy(
                        (
                            np.isfinite(material["conditional_quality"])
                            & (material["filled"] > 0.5)
                        ).astype(np.float32)
                    ).to(device)
                    qloss = global_normalized_component(
                        (
                            torch_functional.smooth_l1_loss(
                                output["quality"],
                                quality,
                                reduction="none",
                            )
                            * quality_mask
                        ).sum(),
                        float(quality_mask.sum().item()),
                    )
                    fill_loss = torch_functional.binary_cross_entropy_with_logits(
                        output["fill_logit"], fill
                    )
                    weights = dict(config["loss_weights"])
                    loss = float(weights["quality_huber"]) * qloss + float(
                        weights["fill_bce"]
                    ) * fill_loss
                else:
                    grades = torch.from_numpy(
                        material["grade"].astype(np.int64)
                    ).to(device)
                    valid_grade = grades < 5
                    if int(valid_grade.sum().item()) >= 2:
                        ndcg = neural_ndcg_loss(
                            output["raw_score"].float()[valid_grade],
                            grades[valid_grade],
                            temperature=float(config["temperature"]),
                            gain=config["gain"],
                        )
                    else:
                        ndcg = output["raw_score"].float().sum() * 0.0
                    fill_loss = torch_functional.binary_cross_entropy_with_logits(
                        output["fill_logit"].float(), fill.float()
                    )
                    weights = dict(config["loss_weights"])
                    loss = float(weights["neural_ndcg"]) * ndcg + float(
                        weights["fill_bce"]
                    ) * fill_loss
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), float(config["gradient_clip_norm"])
                )
                scaler.step(optimizer)
                scaler.update()
                torch.cuda.synchronize()
                elapsed = time.perf_counter() - started
                if step_index >= 2:
                    measured_seconds += elapsed
                    measured_samples += len(rows)
            peak_reserved = int(torch.cuda.max_memory_reserved())
            record.update(
                {
                    "status": "passed",
                    "samples": int(measured_samples),
                    "elapsed_seconds": float(measured_seconds),
                    "samples_per_second": float(
                        measured_samples / measured_seconds
                    ),
                    "gpu_peak_reserved_bytes": peak_reserved,
                    "memory_headroom_ratio": float(
                        max(total_memory - peak_reserved, 0) / total_memory
                    ),
                    "fused_adamw": bool(fused),
                }
            )
        except RuntimeError as exc:
            record.update({"status": "failed", "error": str(exc)})
        finally:
            del loss, output, optimizer, scaler
            gc.collect()
            torch.cuda.empty_cache()
        profile["train_benchmarks"].append(record)

    inference_rows_source = deterministic_epoch_rows(
        train_rows,
        context.candidate_date_idx,
        seed=90211,
    )
    if str(model_id) == "market_industry_deepsets":
        first_group = _row_groups(train_rows, context.candidate_date_idx)[0]
        candidates = [len(first_group)]
    else:
        candidates = _autotune_inference_candidates(model_id)
    model.load_state_dict(initial_state)
    model.eval()
    with torch.no_grad():
        for candidate in candidates:
            if int(candidate) <= 0:
                continue
            rows = (
                first_group
                if str(model_id) == "market_industry_deepsets"
                else inference_rows_source[: int(candidate)]
            )
            record = {"batch_size": int(len(rows))}
            inputs = None
            output = None
            try:
                torch.cuda.empty_cache()
                torch.cuda.reset_peak_memory_stats()
                elapsed_values: list[float] = []
                for iteration in range(3):
                    started = time.perf_counter()
                    inputs = _load_inputs(rows)
                    with torch.autocast(
                        device.type, dtype=torch.float16, enabled=amp_enabled
                    ):
                        output = model(*inputs)
                    _ = [value.float().cpu() for value in output.values()]
                    torch.cuda.synchronize()
                    if iteration:
                        elapsed_values.append(time.perf_counter() - started)
                elapsed = float(np.median(elapsed_values))
                peak_reserved = int(torch.cuda.max_memory_reserved())
                record.update(
                    {
                        "status": "passed",
                        "elapsed_seconds": elapsed,
                        "samples_per_second": float(len(rows) / elapsed),
                        "gpu_peak_reserved_bytes": peak_reserved,
                        "memory_headroom_ratio": float(
                            max(total_memory - peak_reserved, 0) / total_memory
                        ),
                    }
                )
            except RuntimeError as exc:
                record.update({"status": "failed", "error": str(exc)})
            finally:
                del inputs, output
                gc.collect()
                torch.cuda.empty_cache()
            profile["inference_benchmarks"].append(record)
    eligible_inference = [
        item
        for item in profile["inference_benchmarks"]
        if item.get("status") == "passed"
        and float(item.get("memory_headroom_ratio", 0.0)) >= 0.15
    ]
    if eligible_inference:
        selected = max(
            eligible_inference,
            key=lambda item: (
                float(item["samples_per_second"]),
                -int(item["batch_size"]),
            ),
        )
        profile["prediction_batch"] = int(selected["batch_size"])
        profile["validation_batch"] = (
            effective_batch
            if str(model_id) == "deephit_competing_risk"
            else int(selected["batch_size"])
        )
    if str(model_id) == "market_industry_deepsets":
        profile["validation_batch"] = 0
        profile["prediction_batch"] = 0
    if str(model_id) == "neuralndcg_listwise_mlp":
        profile["train_micro_batch"] = int(config["training_list_size"])
    model.load_state_dict(initial_state)
    model.zero_grad(set_to_none=True)
    torch.cuda.empty_cache()
    profile["runtime_profile_sha256"] = _canonical_json_sha256(profile)
    return profile


def _run_model_preflight(
    *,
    context: ModelDataContext,
    model_id: str,
    feature_freeze: Mapping[str, Any],
    output_dir: Path,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=False)
    fold = _load_fold_view(context.feature_manifest, "screen")
    train_range = dict(fold["ranges"]["train"])
    train_rows = _row_ids_for_date_range(
        context.candidate_date_idx,
        int(train_range["date_idx_start"]),
        int(train_range["date_idx_end"]),
    )
    groups = _row_groups(train_rows, context.candidate_date_idx)
    if not groups:
        raise ValueError("real-pack preflight has no training date")
    first_group = groups[0]
    config = dict(_model_spec(context, model_id)["config"])
    device = _torch_device()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    finite = True
    details: dict[str, Any] = {}
    runtime_profile: dict[str, Any] | None = None
    model: nn.Module | None = None
    preprocessor: dict[str, Any] | None = None
    if model_id == "lgbm_lambdarank_multioutput":
        import lightgbm as lgb

        profile = str(feature_freeze["selected_profile"])
        continuous_columns, categorical_columns, *_names = _feature_columns_for_profile(
            context.feature_manifest, profile
        )
        rows = first_group[: min(512, len(first_group))]
        vocabulary = _fit_category_vocabularies(
            context.categorical, rows, categorical_columns
        )
        sequence = _make_lgb_sequence(
            continuous=context.continuous,
            categorical=context.categorical,
            row_ids=rows,
            continuous_columns=continuous_columns,
            categorical_columns=categorical_columns,
            category_vocabularies=vocabulary,
            batch_size=128,
        )
        matrix = np.asarray(sequence[: len(rows)], dtype=np.float32)
        synthetic = np.random.default_rng(7).normal(
            size=(96, matrix.shape[1])
        ).astype(np.float32)
        labels = np.tile(np.arange(5, dtype=np.int32), 20)[:96]
        groups_tiny = np.asarray([32, 32, 32], dtype=np.int32)
        booster = lgb.train(
            {
                "objective": "lambdarank",
                "metric": "ndcg",
                "label_gain": [0, 1, 3, 7, 15],
                "verbosity": -1,
                "num_threads": 2,
                "seed": 7,
            },
            lgb.Dataset(synthetic, label=labels, group=groups_tiny),
            num_boost_round=2,
        )
        prediction = np.asarray(booster.predict(matrix), dtype=np.float32)
        finite = bool(np.isfinite(prediction).all())
        details = {
            "tiny_synthetic_rows": 96,
            "real_batch_rows": int(len(rows)),
            "real_feature_count": int(matrix.shape[1]),
        }
    elif model_id == "patchtst_student_t_path":
        preprocessor = _load_f0_preprocessor(fold)
        rows = first_group[: min(4, len(first_group))]
        _rows, x = next(
            _iter_f0_batches(
                context,
                rows,
                preprocessor,
                batch_size=len(rows),
                shuffle_dates=False,
                seed=0,
            )
        )
        model = _PatchTSTStudentT(
            input_channels=int(preprocessor["input_dim"]), config=config
        ).to(device)
        tiny = model(torch.randn(2, 180, int(preprocessor["input_dim"]), device=device))
        tiny_loss = sum(value.float().square().mean() for value in tiny.values())
        tiny_loss.backward()
        model.zero_grad(set_to_none=True)
        real = model(torch.from_numpy(x).to(device))
        real_loss = sum(value.float().square().mean() for value in real.values())
        real_loss.backward()
        finite = bool(
            math.isfinite(float(real_loss.detach().cpu()))
            and all(torch.isfinite(value).all().item() for value in real.values())
        )
        details = {"tiny_synthetic_rows": 2, "real_batch_rows": int(len(rows))}
    else:
        required_categories = (
            ("industry_hash",)
            if model_id == "market_industry_deepsets"
            else ()
        )
        sample_rows = first_group[: min(2048, len(first_group))]
        preprocessor = _fit_snapshot_preprocessor(
            context,
            sample_rows,
            profile=str(feature_freeze["selected_profile"]),
            output_dir=output_dir,
            normalization_max_rows=int(len(sample_rows)),
            required_categorical_names=required_categories,
        )
        real_rows = (
            first_group
            if model_id == "market_industry_deepsets"
            else first_group[: min(256, len(first_group))]
        )
        x_num, x_cat = _snapshot_batch(context, real_rows, preprocessor)
        cardinalities = _category_cardinalities(preprocessor)
        if model_id == "tabm_multioutput":
            model = _TabMModel(
                n_num_features=x_num.shape[1],
                cat_cardinalities=cardinalities,
                config=config,
            ).to(device)
        elif model_id == "market_industry_deepsets":
            industry_position = list(preprocessor["categorical_names"]).index(
                "industry_hash"
            )
            model = _MarketIndustryDeepSets(
                n_num_features=x_num.shape[1],
                cat_cardinalities=cardinalities,
                industry_position=industry_position,
                config=config,
            ).to(device)
        elif model_id == "deephit_competing_risk":
            model = _DeepHitCompetingRisk(
                n_num_features=x_num.shape[1],
                cat_cardinalities=cardinalities,
                config=config,
            ).to(device)
        elif model_id == "neuralndcg_listwise_mlp":
            model = _NeuralNDCGMLP(
                n_num_features=x_num.shape[1],
                cat_cardinalities=cardinalities,
                config=config,
            ).to(device)
        else:
            raise KeyError(model_id)
        tiny_num = torch.randn(16, x_num.shape[1], device=device)
        tiny_cat = torch.stack(
            [
                torch.randint(0, cardinality, (16,), device=device)
                for cardinality in cardinalities
            ],
            dim=1,
        ) if cardinalities else torch.empty((16, 0), dtype=torch.long, device=device)
        tiny = model(tiny_num, tiny_cat)
        tiny_loss = sum(value.float().square().mean() for value in tiny.values())
        tiny_loss.backward()
        model.zero_grad(set_to_none=True)
        real = model(
            torch.from_numpy(x_num).to(device),
            torch.from_numpy(x_cat).to(device),
        )
        if model_id == "deephit_competing_risk":
            probability = torch.softmax(real["joint_logits"].float(), dim=1)
            real_loss = -torch.log(probability[:, 100].clamp_min(1.0e-8)).mean()
        elif model_id == "neuralndcg_listwise_mlp":
            grades = torch.from_numpy(
                np.asarray(context.grades[real_rows], dtype=np.int64).clip(0, 4)
            ).to(device)
            real_loss = neural_ndcg_loss(
                real["raw_score"].float()[: min(64, len(real_rows))],
                grades[: min(64, len(real_rows))],
                temperature=float(config["temperature"]),
                gain=config["gain"],
            )
        else:
            real_loss = sum(value.float().square().mean() for value in real.values())
        real_loss.backward()
        finite = bool(
            math.isfinite(float(real_loss.detach().cpu()))
            and all(torch.isfinite(value).all().item() for value in real.values())
        )
        details = {
            "tiny_synthetic_rows": 16,
            "real_batch_rows": int(len(real_rows)),
            "real_numeric_feature_count": int(x_num.shape[1]),
            "real_categorical_feature_count": int(x_cat.shape[1]),
        }
    if not finite:
        raise FloatingPointError(f"{model_id} preflight produced non-finite values")
    if str(model_id) in NEURAL_MODEL_IDS:
        if model is None or preprocessor is None:
            raise RuntimeError("neural preflight did not construct runtime material")
        runtime_profile = _autotune_model_runtime(
            context=context,
            model_id=str(model_id),
            model=model,
            config=config,
            preprocessor=preprocessor,
            train_rows=train_rows,
            device=device,
        )
        runtime_profile_path = output_dir / "runtime_profile.json"
        _atomic_write_json(runtime_profile_path, runtime_profile)
    record = {
        "status": "passed",
        "model_id": model_id,
        "device": str(device),
        "elapsed_seconds": float(time.perf_counter() - started),
        "gpu_peak_memory_bytes": (
            int(torch.cuda.max_memory_allocated()) if torch.cuda.is_available() else 0
        ),
        "finite": True,
        **details,
    }
    if runtime_profile is not None:
        record["execution_semantics_version"] = MODEL_EXECUTION_SEMANTICS[
            str(model_id)
        ]
        record["runtime_profile"] = runtime_profile
        record["runtime_profile_artifact"] = {
            "path": str(runtime_profile_path.resolve()),
            "sha256": _file_sha256(runtime_profile_path),
        }
    _atomic_write_json(output_dir / "preflight.json", record)
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return record


def _artifact_hashes_match(payload: Any) -> bool:
    if isinstance(payload, Mapping):
        if "path" in payload and "sha256" in payload:
            try:
                path = _resolve_path(str(payload["path"]))
                if not path.is_file() or _file_sha256(path) != str(payload["sha256"]):
                    return False
            except (OSError, ValueError):
                return False
        for key, value in payload.items():
            if str(key).endswith("_path"):
                sha_key = f"{str(key)[:-5]}_sha256"
                if sha_key in payload:
                    try:
                        path = _resolve_path(str(value))
                        if not path.is_file() or _file_sha256(path) != str(payload[sha_key]):
                            return False
                    except (OSError, ValueError):
                        return False
            if not _artifact_hashes_match(value):
                return False
        return True
    if isinstance(payload, (list, tuple)):
        return all(_artifact_hashes_match(value) for value in payload)
    return True


def _screen_stage_directories(model_dir: Path, stage: str) -> list[Path]:
    base = model_dir / str(stage)
    paths = ([base] if base.exists() else []) + sorted(
        model_dir.glob(f"{stage}_retry_*"),
        key=lambda path: path.name,
    )
    return paths


def _next_screen_retry_dir(model_dir: Path, stage: str) -> Path:
    base = model_dir / str(stage)
    if not base.exists():
        return base
    existing: list[int] = []
    for path in model_dir.glob(f"{stage}_retry_*"):
        try:
            existing.append(int(path.name.split("_")[-1]))
        except ValueError:
            continue
    return model_dir / f"{stage}_retry_{max(existing, default=0) + 1:03d}"


def _prior_screen_model_directories(
    output_root: Path,
    *,
    current_attempt: Path,
    model_id: str,
) -> list[Path]:
    task_root = output_root / "model_screen"
    return [
        attempt / str(model_id)
        for attempt in sorted(task_root.glob("attempt_*"), reverse=True)
        if attempt.resolve() != current_attempt.resolve()
        and (attempt / str(model_id)).is_dir()
    ]


def _budget_extension_compatibility(
    source_config: Mapping[str, Any],
    expected_config: Mapping[str, Any],
    amendment: Mapping[str, Any] | None,
) -> dict[str, Any] | None:
    if amendment is None:
        return None
    model_id = str(expected_config.get("model_id", ""))
    if (
        model_id not in set(str(item) for item in amendment.get("model_ids", []))
        or str(source_config.get("model_id", "")) != model_id
        or str(source_config.get("fold_id", "")) != "screen"
        or str(expected_config.get("fold_id", "")) != "screen"
        or int(source_config.get("seed", -1)) != 7
        or int(expected_config.get("seed", -1)) != 7
        or str(expected_config.get("training_budget_amendment_sha256", ""))
        != str(amendment.get("amendment_sha256", ""))
    ):
        return None
    source_model_config = dict(source_config.get("frozen_model_config", {}) or {})
    expected_model_config = dict(expected_config.get("frozen_model_config", {}) or {})
    if set(source_model_config) != set(expected_model_config):
        return None
    changes = dict(amendment.get("config_changes", {}) or {})
    changed_fields = {
        key
        for key in source_model_config
        if source_model_config[key] != expected_model_config[key]
    }
    if changed_fields != set(TRAINING_BUDGET_MUTABLE_FIELDS):
        return None
    for field in TRAINING_BUDGET_MUTABLE_FIELDS:
        change = dict(changes.get(field, {}) or {})
        if (
            source_model_config.get(field) != change.get("from")
            or expected_model_config.get(field) != change.get("to")
        ):
            return None
    source_core = dict(source_config)
    expected_core = dict(expected_config)
    for payload in (source_core, expected_core):
        payload.pop("frozen_model_config", None)
        payload.pop("resolved_config_sha256", None)
        payload.pop("resume_key_sha256", None)
        payload.pop("training_budget_amendment_sha256", None)
        payload.pop("training_budget_amended_fields", None)
    if source_core != expected_core:
        return None
    return {
        "model_id": model_id,
        "fold_id": "screen",
        "seed": 7,
        "config_changes": {
            field: {
                "from": source_model_config[field],
                "to": expected_model_config[field],
            }
            for field in TRAINING_BUDGET_MUTABLE_FIELDS
        },
    }


def _find_budget_extension_source(
    model_directories: Sequence[Path],
    *,
    model_id: str,
    expected_config: Mapping[str, Any],
    amendment: Mapping[str, Any] | None,
) -> dict[str, Any] | None:
    for model_dir in model_directories:
        for output_dir in reversed(_screen_stage_directories(model_dir, "prescreen")):
            config_path = output_dir / "resolved_config.json"
            checkpoint_path = output_dir / "last_checkpoint.pt"
            summary_path = output_dir / "task_summary.json"
            if not (config_path.exists() and checkpoint_path.exists() and summary_path.exists()):
                continue
            try:
                source_config = json.loads(config_path.read_text(encoding="utf-8"))
                summary = json.loads(summary_path.read_text(encoding="utf-8"))
                checkpoint = torch.load(
                    checkpoint_path,
                    map_location="cpu",
                    weights_only=False,
                )
            except (OSError, RuntimeError, ValueError, json.JSONDecodeError):
                continue
            compatibility = _budget_extension_compatibility(
                source_config,
                expected_config,
                amendment,
            )
            if compatibility is None:
                continue
            if (
                str(summary.get("status", "")) != "completed"
                or str(summary.get("model_id", "")) != str(model_id)
                or str(summary.get("fold_id", "")) != "screen"
                or int(summary.get("seed", -1)) != 7
                or str(summary.get("resolved_config_sha256", ""))
                != str(source_config.get("resolved_config_sha256", ""))
                or str(summary.get("resume_key_sha256", ""))
                != str(source_config.get("resume_key_sha256", ""))
                or summary.get("protection_before") != summary.get("protection_after")
                or not _artifact_hashes_match(summary)
                or str(checkpoint.get("schema", "")) != TRAINING_CHECKPOINT_VERSION
                or str(checkpoint.get("resume_key_sha256", ""))
                != str(source_config.get("resume_key_sha256", ""))
                or checkpoint.get("execution_semantics_version")
                != expected_config.get("execution_semantics_version")
                or int(checkpoint.get("next_step_index", -1)) != 0
                or int(checkpoint.get("epoch", 0))
                > int(dict(expected_config["frozen_model_config"])["max_epochs"])
                or int(checkpoint.get("epochs_without_improvement", 0))
                >= int(dict(expected_config["frozen_model_config"])["patience"])
            ):
                continue
            return {
                "source_dir": output_dir,
                "source_config": source_config,
                "source_checkpoint": checkpoint,
                "compatibility": compatibility,
            }
    return None


def _prepare_budget_extension_directory(
    *,
    output_dir: Path,
    source: Mapping[str, Any],
    expected_config: Mapping[str, Any],
    amendment: Mapping[str, Any],
) -> dict[str, Any]:
    source_dir = Path(source["source_dir"]).resolve()
    source_checkpoint_path = source_dir / "last_checkpoint.pt"
    output_dir.mkdir(parents=True, exist_ok=False)
    migrated = dict(source["source_checkpoint"])
    migrated["resume_key_sha256"] = str(expected_config["resume_key_sha256"])
    migrated["budget_extension_amendment_sha256"] = str(
        amendment["amendment_sha256"]
    )
    migrated["budget_extension_source_checkpoint_sha256"] = _file_sha256(
        source_checkpoint_path
    )
    migrated["budget_extension_migrated_at"] = _now()
    _atomic_torch_save(output_dir / "last_checkpoint.pt", migrated)
    for name in ("snapshot_preprocessor.json", "f0_preprocessor.json"):
        source_path = source_dir / name
        if source_path.exists():
            temporary = (output_dir / name).with_name(name + ".tmp")
            shutil.copyfile(source_path, temporary)
            os.replace(temporary, output_dir / name)
    provenance = {
        "schema": TRAINING_BUDGET_AMENDMENT_VERSION,
        "created_at": _now(),
        "amendment_sha256": str(amendment["amendment_sha256"]),
        "source_dir": str(source_dir),
        "source_checkpoint": str(source_checkpoint_path),
        "source_checkpoint_sha256": _file_sha256(source_checkpoint_path),
        "source_resolved_config_sha256": str(
            dict(source["source_config"])["resolved_config_sha256"]
        ),
        "source_resume_key_sha256": str(
            dict(source["source_config"])["resume_key_sha256"]
        ),
        "destination_resume_key_sha256": str(expected_config["resume_key_sha256"]),
        "start_epoch": int(migrated["epoch"]),
        "best_epoch": int(migrated["best_epoch"]),
        "epochs_without_improvement": int(
            migrated["epochs_without_improvement"]
        ),
        **dict(source["compatibility"]),
    }
    provenance["provenance_sha256"] = _canonical_json_sha256(provenance)
    _atomic_write_json(output_dir / "budget_extension_provenance.json", provenance)
    return provenance


def _find_resumable_task_directory(
    directories: Sequence[Path],
    *,
    expected_config: Mapping[str, Any],
) -> Path | None:
    for output_dir in reversed(list(directories)):
        config_path = output_dir / "resolved_config.json"
        checkpoint_path = output_dir / "last_checkpoint.pt"
        progress_path = output_dir / "progress.json"
        if not (config_path.exists() and checkpoint_path.exists() and progress_path.exists()):
            continue
        try:
            config = json.loads(config_path.read_text(encoding="utf-8"))
            progress = json.loads(progress_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if config != dict(expected_config):
            continue
        if str(progress.get("status", "")) not in {"training", "paused"}:
            continue
        try:
            checkpoint = torch.load(
                checkpoint_path,
                map_location="cpu",
                weights_only=False,
            )
        except (OSError, RuntimeError, ValueError):
            continue
        if (
            str(checkpoint.get("schema", "")) == TRAINING_CHECKPOINT_VERSION
            and str(checkpoint.get("resume_key_sha256", ""))
            == str(expected_config["resume_key_sha256"])
        ):
            return output_dir
    return None


def _load_completed_screen_task(
    model_dir: Path,
    *,
    model_id: str,
    expected_config: Mapping[str, Any],
    expected_test_year: int,
    expected_candidate_count: int,
) -> tuple[Path, dict[str, Any]] | None:
    for output_dir in reversed(_screen_stage_directories(model_dir, "prescreen")):
        summary_path = output_dir / "task_summary.json"
        config_path = output_dir / "resolved_config.json"
        if not summary_path.exists() or not config_path.exists():
            continue
        try:
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            resolved_config = json.loads(config_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if resolved_config != dict(expected_config):
            continue
        if (
            str(summary.get("status", "")) != "completed"
            or str(summary.get("model_id", "")) != str(model_id)
            or str(summary.get("fold_id", "")) != "screen"
            or int(summary.get("fold_year", -1)) != int(expected_test_year)
            or int(summary.get("seed", -1)) != 7
            or str(summary.get("resume_key_sha256", ""))
            != str(expected_config["resume_key_sha256"])
            or str(summary.get("resolved_config_sha256", ""))
            != str(expected_config["resolved_config_sha256"])
            or int(dict(summary.get("prediction", {}) or {}).get("candidate_count", -1))
            != int(expected_candidate_count)
            or int(dict(summary.get("metrics", {}) or {}).get("candidate_count", -1))
            != int(expected_candidate_count)
            or summary.get("protection_before") != summary.get("protection_after")
            or not _artifact_hashes_match(summary)
        ):
            continue
        return output_dir, summary
    return None


def _load_passed_screen_preflight(
    model_dir: Path,
    *,
    model_id: str,
) -> tuple[Path, dict[str, Any]] | None:
    for output_dir in reversed(_screen_stage_directories(model_dir, "preflight")):
        path = output_dir / "preflight.json"
        if not path.exists():
            continue
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        runtime_valid = True
        if str(model_id) in NEURAL_MODEL_IDS:
            profile = dict(payload.get("runtime_profile", {}) or {})
            declared = str(profile.get("runtime_profile_sha256", ""))
            check = dict(profile)
            check.pop("runtime_profile_sha256", None)
            runtime_valid = bool(
                str(payload.get("execution_semantics_version", ""))
                == MODEL_EXECUTION_SEMANTICS[str(model_id)]
                and declared == _canonical_json_sha256(check)
                and str(dict(profile.get("hardware", {}) or {}).get("hardware_sha256", ""))
                == str(_hardware_fingerprint()["hardware_sha256"])
            )
        if (
            str(payload.get("status", "")) == "passed"
            and str(payload.get("model_id", "")) == str(model_id)
            and bool(payload.get("finite", False))
            and runtime_valid
            and _artifact_hashes_match(payload)
        ):
            return output_dir, payload
    return None


def _model_screen_binding(
    context: ModelDataContext,
    feature_freeze: Mapping[str, Any],
    audit: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "study_contract_sha256": str(context.study["contract_sha256"]),
        "target_sha256": str(context.target_manifest["frozen_target_sha256"]),
        "feature_sha256": str(
            context.feature_manifest["resolved_feature_view_sha256"]
        ),
        "feature_freeze_sha256": str(feature_freeze["feature_freeze_sha256"]),
        "research_audit_sha256": _canonical_json_sha256(audit),
        "training_budget_amendment_sha256": str(
            dict(context.training_budget_amendment or {}).get(
                "amendment_sha256", ""
            )
        ),
    }


def _find_resumable_model_screen_attempt(
    *,
    output_root: Path,
    context: ModelDataContext,
    feature_freeze: Mapping[str, Any],
    audit: Mapping[str, Any],
) -> Path | None:
    task_root = output_root / "model_screen"
    binding = _model_screen_binding(context, feature_freeze, audit)
    fold = _load_fold_view(context.feature_manifest, "screen")
    expected_year = int(fold["test_year"])
    expected_count = int(fold["ranges"]["test"]["candidate_count"])
    for attempt in sorted(task_root.glob("attempt_*"), reverse=True):
        if any(
            (attempt / name).exists()
            for name in ("formal_matrix.json", "research_design_insufficient.json")
        ):
            continue
        audit_path = attempt / "formula_source_license_audit.json"
        if not audit_path.exists():
            continue
        try:
            recorded_audit = json.loads(audit_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if recorded_audit != dict(audit):
            continue
        progress_path = attempt / "progress.json"
        if progress_path.exists():
            try:
                progress = json.loads(progress_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                progress = {}
            if all(str(progress.get(key, "")) == str(value) for key, value in binding.items()):
                return attempt
        for model_id in MODEL_IDS:
            preflight = _load_passed_screen_preflight(
                attempt / model_id,
                model_id=model_id,
            )
            runtime_profile = (
                dict(preflight[1].get("runtime_profile", {}) or {})
                if preflight is not None
                else None
            )
            expected_config = _resolved_model_config(
                context,
                model_id=model_id,
                fold_id="screen",
                seed=7,
                feature_freeze=feature_freeze,
                runtime_profile=runtime_profile,
            )
            if _load_completed_screen_task(
                attempt / model_id,
                model_id=model_id,
                expected_config=expected_config,
                expected_test_year=expected_year,
                expected_candidate_count=expected_count,
            ) is not None:
                return attempt
    return None


def screen_models(
    *,
    study_path: Path = DEFAULT_STUDY_PATH,
    output_root: Path = DEFAULT_OUTPUT_ROOT,
) -> dict[str, Any]:
    study = load_study(study_path)
    amendment = _load_training_budget_amendment(study, output_root)
    amendment_sha256 = str(dict(amendment or {}).get("amendment_sha256", ""))
    existing = output_root / "model_screen/current.json"
    if existing.exists():
        current = json.loads(existing.read_text(encoding="utf-8"))
        if str(current.get("status", "")) == "research_design_insufficient":
            if str(current.get("training_budget_amendment_sha256", "")) == amendment_sha256:
                result_path = Path(str(current["result"])).resolve()
                if _file_sha256(result_path) != str(current["result_sha256"]):
                    raise ValueError("model-screen failure record changed")
                return json.loads(result_path.read_text(encoding="utf-8"))
        else:
            _path, matrix = _load_formal_matrix(output_root, study)
            if str(matrix.get("training_budget_amendment_sha256", "")) == amendment_sha256:
                return {
                    "status": "formal_matrix_frozen",
                    "output_dir": str(Path(matrix["output_dir"]).resolve()),
                    "formal_matrix": str(_path.resolve()),
                    "formal_matrix_sha256": matrix["formal_matrix_sha256"],
                    "model_ids": list(matrix["model_ids"]),
                }
    try:
        _feature_path, feature_freeze = _load_feature_freeze(output_root, study)
    except FileNotFoundError:
        feature_screen(study_path=study_path, output_root=output_root)
        _feature_path, feature_freeze = _load_feature_freeze(output_root, study)
    context = _resolve_model_data(study_path=study_path, output_root=output_root)
    protection_before = _verify_protected_bindings(study)
    audit = _model_research_audit(context)
    resumed_attempt = _find_resumable_model_screen_attempt(
        output_root=output_root,
        context=context,
        feature_freeze=feature_freeze,
        audit=audit,
    )
    attempt_dir = (
        resumed_attempt
        if resumed_attempt is not None
        else _next_attempt_dir(output_root, "model_screen")
    )
    audit_path = attempt_dir / "formula_source_license_audit.json"
    if resumed_attempt is None:
        _atomic_write_json(audit_path, audit)
    elif json.loads(audit_path.read_text(encoding="utf-8")) != audit:
        raise ValueError("resumable model-screen audit no longer matches the frozen review")
    binding = _model_screen_binding(context, feature_freeze, audit)
    progress_path = attempt_dir / "progress.json"
    fold = _load_fold_view(context.feature_manifest, "screen")
    expected_test_year = int(fold["test_year"])
    expected_candidate_count = int(fold["ranges"]["test"]["candidate_count"])
    preflights: dict[str, Any] = {}
    prescreens: dict[str, Any] = {}
    passed: list[str] = []
    reused: list[str] = []
    continued: list[str] = []
    reused_preflights: list[str] = []
    for model_id in MODEL_IDS:
        model_dir = attempt_dir / model_id
        model_dir.mkdir(parents=True, exist_ok=True)
        prior_model_dirs = _prior_screen_model_directories(
            output_root,
            current_attempt=attempt_dir,
            model_id=model_id,
        )
        existing_preflight = _load_passed_screen_preflight(
            model_dir,
            model_id=model_id,
        )
        if existing_preflight is None:
            for prior_model_dir in prior_model_dirs:
                existing_preflight = _load_passed_screen_preflight(
                    prior_model_dir,
                    model_id=model_id,
                )
                if existing_preflight is not None:
                    reused_preflights.append(model_id)
                    break
        _atomic_write_json(
            progress_path,
            {
                "status": "screening",
                "updated_at": _now(),
                **binding,
                "resumed_attempt": resumed_attempt is not None,
                "completed_model_ids": list(passed),
                "reused_model_ids": list(reused),
                "continued_model_ids": list(continued),
                "reused_preflight_model_ids": list(reused_preflights),
                "current_model_id": model_id,
            },
        )
        preserve_paused_progress = False
        try:
            if existing_preflight is None:
                preflights[model_id] = _run_model_preflight(
                    context=context,
                    model_id=model_id,
                    feature_freeze=feature_freeze,
                    output_dir=_next_screen_retry_dir(model_dir, "preflight"),
                )
            else:
                _preflight_dir, preflights[model_id] = existing_preflight
            runtime_profile = dict(
                preflights[model_id].get("runtime_profile", {}) or {}
            ) or None
            expected_config = _resolved_model_config(
                context,
                model_id=model_id,
                fold_id="screen",
                seed=7,
                feature_freeze=feature_freeze,
                runtime_profile=runtime_profile,
            )
            completed = _load_completed_screen_task(
                model_dir,
                model_id=model_id,
                expected_config=expected_config,
                expected_test_year=expected_test_year,
                expected_candidate_count=expected_candidate_count,
            )
            if completed is None:
                for prior_model_dir in prior_model_dirs:
                    completed = _load_completed_screen_task(
                        prior_model_dir,
                        model_id=model_id,
                        expected_config=expected_config,
                        expected_test_year=expected_test_year,
                        expected_candidate_count=expected_candidate_count,
                    )
                    if completed is not None:
                        break
            if completed is None:
                task_output_dir = _find_resumable_task_directory(
                    _screen_stage_directories(model_dir, "prescreen"),
                    expected_config=expected_config,
                )
                if task_output_dir is None:
                    task_output_dir = _next_screen_retry_dir(model_dir, "prescreen")
                    extension_source = _find_budget_extension_source(
                        prior_model_dirs,
                        model_id=model_id,
                        expected_config=expected_config,
                        amendment=context.training_budget_amendment,
                    )
                    if extension_source is not None:
                        _prepare_budget_extension_directory(
                            output_dir=task_output_dir,
                            source=extension_source,
                            expected_config=expected_config,
                            amendment=dict(context.training_budget_amendment or {}),
                        )
                        continued.append(model_id)
                _atomic_write_json(
                    progress_path,
                    {
                        "status": "screening",
                        "updated_at": _now(),
                        **binding,
                        "resumed_attempt": resumed_attempt is not None,
                        "completed_model_ids": list(passed),
                        "reused_model_ids": list(reused),
                        "continued_model_ids": list(continued),
                        "reused_preflight_model_ids": list(reused_preflights),
                        "current_model_id": model_id,
                        "current_task_progress": str(
                            (task_output_dir / "progress.json").resolve()
                        ),
                    },
                )
                prescreens[model_id] = _run_model_task(
                    context=context,
                    model_id=model_id,
                    fold_id="screen",
                    seed=7,
                    feature_freeze=feature_freeze,
                    output_dir=task_output_dir,
                    runtime_profile=runtime_profile,
                )
            else:
                _task_dir, prescreens[model_id] = completed
                reused.append(model_id)
            passed.append(model_id)
        except TrainingPaused as exc:
            preserve_paused_progress = True
            paused = {
                "status": "paused",
                "updated_at": _now(),
                **binding,
                "current_model_id": model_id,
                "completed_model_ids": list(passed),
                "reused_model_ids": list(reused),
                "continued_model_ids": list(continued),
                "reused_preflight_model_ids": list(reused_preflights),
                "reason": str(exc),
            }
            _atomic_write_json(progress_path, paused)
            return {
                "status": "paused",
                "output_dir": str(attempt_dir.resolve()),
                "model_id": model_id,
                "reason": str(exc),
            }
        except Exception as exc:
            preflights.setdefault(
                model_id,
                {
                    "status": "failed",
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                },
            )
            prescreens[model_id] = {
                "status": "failed",
                "error_type": type(exc).__name__,
                "error": str(exc),
            }
            _atomic_write_json(model_dir / "failure.json", prescreens[model_id])
        finally:
            if not preserve_paused_progress:
                _atomic_write_json(
                    progress_path,
                    {
                        "status": "screening",
                        "updated_at": _now(),
                        **binding,
                        "resumed_attempt": resumed_attempt is not None,
                        "completed_model_ids": list(passed),
                        "reused_model_ids": list(reused),
                        "continued_model_ids": list(continued),
                        "reused_preflight_model_ids": list(reused_preflights),
                        "current_model_id": None,
                    },
                )
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    minimum = int(context.research_freeze["model_funnel"]["minimum_formal_families"])
    maximum = int(context.research_freeze["model_funnel"]["maximum_formal_families"])
    slots = [str(_model_spec(context, model_id)["slot"]) for model_id in passed]
    sufficient = bool(
        minimum <= len(passed) <= maximum and len(slots) == len(set(slots))
    )
    protection_after = _verify_protected_bindings(study)
    if protection_after != protection_before:
        raise RuntimeError("protected-object hashes changed during model screening")
    if not sufficient:
        result = {
            "status": "research_design_insufficient",
            "created_at": _now(),
            "output_dir": str(attempt_dir.resolve()),
            "passed_model_ids": passed,
            "minimum_required": minimum,
            "preflights": preflights,
            "prescreens": prescreens,
            "protection_before": protection_before,
            "protection_after": protection_after,
        }
        result_path = attempt_dir / "research_design_insufficient.json"
        _atomic_write_json(result_path, result)
        _atomic_write_json(
            output_root / "model_screen/current.json",
            {
                "attempt": str(attempt_dir.resolve()),
                "status": "research_design_insufficient",
                "result": str(result_path.resolve()),
                "result_sha256": _file_sha256(result_path),
                "study_contract_sha256": study["contract_sha256"],
                "training_budget_amendment_sha256": binding[
                    "training_budget_amendment_sha256"
                ],
                "updated_at": _now(),
            },
        )
        _atomic_write_json(
            progress_path,
            {
                "status": "research_design_insufficient",
                "completed_at": _now(),
                **binding,
                "completed_model_ids": list(passed),
                "reused_model_ids": list(reused),
            },
        )
        return result
    continued_final = [
        model_id
        for model_id in passed
        if isinstance(prescreens[model_id].get("budget_extension"), Mapping)
    ]
    matrix = {
        "artifact_type": "seq100_signal_quality_formal_model_matrix",
        "status": "formal_matrix_frozen",
        "created_at": _now(),
        "output_dir": str(attempt_dir.resolve()),
        "study_contract_sha256": study["contract_sha256"],
        "research_freeze_sha256": context.target_manifest[
            "research_freeze_sha256"
        ],
        "target_sha256": context.target_manifest["frozen_target_sha256"],
        "feature_sha256": context.feature_manifest[
            "resolved_feature_view_sha256"
        ],
        "feature_freeze_sha256": feature_freeze["feature_freeze_sha256"],
        "selected_feature_profile": feature_freeze["selected_profile"],
        "model_ids": passed,
        "models": [
            {
                "model_id": model_id,
                "slot": _model_spec(context, model_id)["slot"],
                "capabilities": _model_spec(context, model_id)["capabilities"],
                "prescreen_metrics": prescreens[model_id]["metrics"],
                "prescreen_resolved_config_sha256": prescreens[model_id][
                    "resolved_config_sha256"
                ],
                "preflight": preflights[model_id],
            }
            for model_id in passed
        ],
        "replacement_allowed": False,
        "formal_seed": 7,
        "formal_fold_years": list(FORMAL_FOLD_YEARS),
        "protection_before": protection_before,
        "protection_after": protection_after,
        "training_budget_amendment_sha256": str(
            dict(context.training_budget_amendment or {}).get(
                "amendment_sha256", ""
            )
        ),
        "resource_reuse": {
            "reused_preflight_model_ids": list(dict.fromkeys(reused_preflights)),
            "reused_completed_model_ids": list(dict.fromkeys(reused)),
            "continued_checkpoint_model_ids": continued_final,
        },
    }
    if context.training_budget_amendment is not None:
        amendment_path = _training_budget_amendment_path(output_root)
        matrix["training_budget_amendment"] = {
            "path": str(amendment_path.resolve()),
            "sha256": _file_sha256(amendment_path),
            "amendment_sha256": context.training_budget_amendment[
                "amendment_sha256"
            ],
            "config_changes": context.training_budget_amendment[
                "config_changes"
            ],
        }
    matrix["formal_matrix_sha256"] = _canonical_json_sha256(matrix)
    matrix_path = attempt_dir / "formal_matrix.json"
    _atomic_write_json(matrix_path, matrix)
    _atomic_write_json(
        output_root / "model_screen/current.json",
        {
            "attempt": str(attempt_dir.resolve()),
            "formal_matrix": str(matrix_path.resolve()),
            "formal_matrix_sha256": matrix["formal_matrix_sha256"],
            "study_contract_sha256": study["contract_sha256"],
            "training_budget_amendment_sha256": matrix[
                "training_budget_amendment_sha256"
            ],
            "updated_at": _now(),
        },
    )
    summary = {
        "status": "formal_matrix_frozen",
        "output_dir": str(attempt_dir.resolve()),
        "formal_matrix": str(matrix_path.resolve()),
        "formal_matrix_sha256": matrix["formal_matrix_sha256"],
        "model_ids": passed,
        "resource_reuse": matrix["resource_reuse"],
    }
    _atomic_write_json(attempt_dir / "screen_models_summary.json", summary)
    _atomic_write_json(
        progress_path,
        {
            "status": "formal_matrix_frozen",
            "completed_at": _now(),
            **binding,
            "completed_model_ids": list(passed),
            "reused_model_ids": list(reused),
            "continued_model_ids": continued_final,
            "reused_preflight_model_ids": list(dict.fromkeys(reused_preflights)),
            "formal_matrix": str(matrix_path.resolve()),
            "formal_matrix_sha256": matrix["formal_matrix_sha256"],
        },
    )
    return summary


def _robustness_authorization(
    output_root: Path,
    *,
    model_id: str,
) -> bool:
    pointer = output_root / "evaluation/current.json"
    if not pointer.exists():
        return False
    payload = json.loads(pointer.read_text(encoding="utf-8"))
    return str(model_id) in set(
        str(item) for item in list(payload.get("robustness_model_ids", []) or [])
    )


def train_model(
    *,
    model_id: str,
    fold_year: int,
    seed: int,
    study_path: Path = DEFAULT_STUDY_PATH,
    output_root: Path = DEFAULT_OUTPUT_ROOT,
) -> dict[str, Any]:
    context = _resolve_model_data(study_path=study_path, output_root=output_root)
    _matrix_path, matrix = _load_formal_matrix(output_root, context.study)
    if str(model_id) not in set(str(item) for item in matrix["model_ids"]):
        raise ValueError("model is not in the frozen formal matrix")
    matrix_model = next(
        dict(item)
        for item in list(matrix.get("models", []) or [])
        if str(item.get("model_id", "")) == str(model_id)
    )
    runtime_profile = dict(
        dict(matrix_model.get("preflight", {}) or {}).get("runtime_profile", {})
        or {}
    ) or None
    if int(fold_year) not in FORMAL_FOLD_YEARS:
        raise ValueError("formal fold year is not registered")
    training_spec = dict(context.research_freeze["training"])
    first_seed = int(training_spec["first_seed"])
    robustness_seeds = set(int(item) for item in training_spec["robustness_seeds"])
    if int(seed) != first_seed:
        if int(seed) not in robustness_seeds:
            raise ValueError("seed is not registered by the frozen study")
        if not _robustness_authorization(output_root, model_id=model_id):
            raise ValueError("robustness seed is not authorized for this model")
    _feature_path, feature_freeze = _load_feature_freeze(output_root, context.study)
    task_root = (
        output_root
        / "training"
        / str(model_id)
        / f"fold_{int(fold_year)}"
        / f"seed_{int(seed)}"
    )
    task_root.mkdir(parents=True, exist_ok=True)
    current_path = task_root / "current.json"
    if current_path.exists():
        current = json.loads(current_path.read_text(encoding="utf-8"))
        summary_path = Path(str(current["task_summary"])).resolve()
        if _file_sha256(summary_path) != str(current["task_summary_sha256"]):
            raise ValueError("completed task summary changed")
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        if str(summary.get("status", "")) == "completed":
            return summary
    retry_limit = int(training_spec["resource_retry"]["limit"])
    last_error: Exception | None = None
    for retry_index in range(retry_limit + 1):
        expected_config = _resolved_model_config(
            context,
            model_id=str(model_id),
            fold_id=int(fold_year),
            seed=int(seed),
            feature_freeze=feature_freeze,
            resource_retry_index=retry_index,
            runtime_profile=runtime_profile,
        )
        attempts_root = task_root / "attempts"
        attempt_dir = _find_resumable_task_directory(
            sorted(attempts_root.glob("attempt_*")),
            expected_config=expected_config,
        ) or _next_attempt_dir(task_root, "attempts")
        active_path = task_root / "active.json"
        _atomic_write_json(
            active_path,
            {
                "status": "training",
                "attempt": str(attempt_dir.resolve()),
                "progress": str((attempt_dir / "progress.json").resolve()),
                "resume_key_sha256": expected_config["resume_key_sha256"],
                "updated_at": _now(),
            },
        )
        try:
            summary = _run_model_task(
                context=context,
                model_id=str(model_id),
                fold_id=int(fold_year),
                seed=int(seed),
                feature_freeze=feature_freeze,
                output_dir=attempt_dir,
                resource_retry_index=retry_index,
                runtime_profile=runtime_profile,
            )
            summary_path = attempt_dir / "task_summary.json"
            _atomic_write_json(
                current_path,
                {
                    "attempt": str(attempt_dir.resolve()),
                    "task_summary": str(summary_path.resolve()),
                    "task_summary_sha256": _file_sha256(summary_path),
                    "resume_key_sha256": summary["resume_key_sha256"],
                    "updated_at": _now(),
                },
            )
            _atomic_write_json(
                active_path,
                {
                    "status": "completed",
                    "attempt": str(attempt_dir.resolve()),
                    "progress": str((attempt_dir / "progress.json").resolve()),
                    "task_summary": str(summary_path.resolve()),
                    "updated_at": _now(),
                },
            )
            return summary
        except TrainingPaused as exc:
            _atomic_write_json(
                active_path,
                {
                    "status": "paused",
                    "attempt": str(attempt_dir.resolve()),
                    "progress": str((attempt_dir / "progress.json").resolve()),
                    "reason": str(exc),
                    "updated_at": _now(),
                },
            )
            raise
        except RuntimeError as exc:
            last_error = exc
            message = str(exc).lower()
            resource_failure = "out of memory" in message or "cuda error" in message
            _atomic_write_json(
                attempt_dir / "failure.json",
                {
                    "status": "failed",
                    "failed_at": _now(),
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                    "resource_failure": resource_failure,
                    "retry_index": retry_index,
                },
            )
            _atomic_write_json(
                active_path,
                {
                    "status": "failed",
                    "attempt": str(attempt_dir.resolve()),
                    "progress": str((attempt_dir / "progress.json").resolve()),
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                    "updated_at": _now(),
                },
            )
            if not resource_failure or retry_index >= retry_limit:
                raise
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception as exc:
            _atomic_write_json(
                active_path,
                {
                    "status": "failed",
                    "attempt": str(attempt_dir.resolve()),
                    "progress": str((attempt_dir / "progress.json").resolve()),
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                    "updated_at": _now(),
                },
            )
            raise
    assert last_error is not None
    raise last_error


EVALUATION_COLUMNS = [
    "trade_date",
    "date_idx",
    "symbol",
    "symbol_idx",
    "candidate_id",
    "entry_filled",
    "fill_probability",
    "conditional_quality_prediction",
    "selection_score",
    "frozen_target_quality",
    "common_sustained_action_utility",
    "d5_net_return",
    "d10_net_return",
    "d20_net_return",
    "mdd20",
    "fade20",
    "mfe20",
    "mae20",
    "terminal_failure",
]


def _load_training_task_summary(
    output_root: Path,
    *,
    model_id: str,
    fold_year: int,
    seed: int,
) -> dict[str, Any] | None:
    current_path = (
        output_root
        / "training"
        / str(model_id)
        / f"fold_{int(fold_year)}"
        / f"seed_{int(seed)}"
        / "current.json"
    )
    if not current_path.exists():
        return None
    current = json.loads(current_path.read_text(encoding="utf-8"))
    summary_path = Path(str(current["task_summary"])).resolve()
    if not summary_path.exists():
        raise FileNotFoundError(summary_path)
    if _file_sha256(summary_path) != str(current["task_summary_sha256"]):
        raise ValueError(f"training task summary changed: {summary_path}")
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    if str(summary.get("status", "")) != "completed":
        return None
    prediction = dict(summary["prediction"])
    prediction_path = Path(str(prediction["path"])).resolve()
    if _file_sha256(prediction_path) != str(prediction["sha256"]):
        raise ValueError(f"training prediction changed: {prediction_path}")
    summary["task_summary_path"] = str(summary_path)
    return summary


def _load_ensemble_frame(
    task_summaries: Sequence[Mapping[str, Any]],
) -> pd.DataFrame:
    if not task_summaries:
        raise ValueError("cannot construct an ensemble without task summaries")
    frames: list[pd.DataFrame] = []
    for summary in task_summaries:
        frame = pd.read_parquet(
            summary["prediction"]["path"], columns=EVALUATION_COLUMNS
        )
        frames.append(frame)
    base = frames[0].copy()
    identity = base["candidate_id"].to_numpy(dtype=np.int64, copy=False)
    for frame in frames[1:]:
        if not np.array_equal(
            identity, frame["candidate_id"].to_numpy(dtype=np.int64, copy=False)
        ):
            raise ValueError("seed prediction candidate order differs")
        for name in (
            "trade_date",
            "date_idx",
            "symbol",
            "symbol_idx",
            "entry_filled",
            "frozen_target_quality",
            "common_sustained_action_utility",
            "d20_net_return",
            "terminal_failure",
        ):
            left = base[name].to_numpy()
            right = frame[name].to_numpy()
            if left.dtype.kind == "f":
                equal = np.allclose(left, right, equal_nan=True, rtol=0.0, atol=0.0)
            else:
                equal = np.array_equal(left, right)
            if not equal:
                raise ValueError(f"seed truth column differs: {name}")
    for name in (
        "fill_probability",
        "conditional_quality_prediction",
        "selection_score",
    ):
        base[name] = np.mean(
            np.stack(
                [frame[name].to_numpy(dtype=np.float64) for frame in frames],
                axis=0,
            ),
            axis=0,
        ).astype(np.float32)
    return base


def _daily_selection_evaluation(
    frame: pd.DataFrame,
    *,
    score_column: str = "selection_score",
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for trade_date, group in frame.groupby("trade_date", sort=False):
        score = group[score_column].to_numpy(dtype=np.float64, copy=False)
        symbol = group["symbol_idx"].to_numpy(dtype=np.int64, copy=False)
        valid = np.isfinite(score)
        positions = np.flatnonzero(valid)
        if not positions.size:
            continue
        order = positions[np.lexsort((symbol[positions], -score[positions]))]
        count = int(len(group))
        selection_sizes = {
            "top1": 1,
            "top3": 3,
            "top10": 10,
            "top1pct": max(1, int(math.ceil(0.01 * count))),
            "top5pct": max(1, int(math.ceil(0.05 * count))),
        }
        frozen_truth = group["frozen_target_quality"].to_numpy(
            dtype=np.float64, copy=False
        )
        grades = relevance_grades(frozen_truth)
        k1 = min(selection_sizes["top1pct"], len(order))
        k5 = min(selection_sizes["top5pct"], len(order))
        rank_truth = frozen_truth[positions]
        rank_score = score[positions]
        spearman = stats.spearmanr(rank_truth, rank_score).statistic
        record: dict[str, Any] = {
            "trade_date": str(trade_date),
            "year": int(str(trade_date)[:4]),
            "date_idx": int(group["date_idx"].iloc[0]),
            "candidate_count": count,
            "score_coverage": float(len(positions) / max(count, 1)),
            "ndcg_at_1pct": _ndcg_for_group(
                grades[positions], rank_score, k1
            ),
            "ndcg_at_5pct": _ndcg_for_group(
                grades[positions], rank_score, k5
            ),
            "rank_ic": (
                float(spearman) if math.isfinite(float(spearman)) else np.nan
            ),
        }
        filled = group["entry_filled"].to_numpy(dtype=bool, copy=False)
        action_values = {
            "u": group["common_sustained_action_utility"].to_numpy(
                dtype=np.float64, copy=False
            ),
            "d5": np.where(
                filled,
                group["d5_net_return"].to_numpy(dtype=np.float64, copy=False),
                0.0,
            ),
            "d10": np.where(
                filled,
                group["d10_net_return"].to_numpy(dtype=np.float64, copy=False),
                0.0,
            ),
            "d20": np.where(
                filled,
                group["d20_net_return"].to_numpy(dtype=np.float64, copy=False),
                0.0,
            ),
            "mdd": np.where(
                filled,
                group["mdd20"].to_numpy(dtype=np.float64, copy=False),
                0.0,
            ),
            "fade": np.where(
                filled,
                group["fade20"].to_numpy(dtype=np.float64, copy=False),
                0.0,
            ),
            "mfe": np.where(
                filled,
                group["mfe20"].to_numpy(dtype=np.float64, copy=False),
                0.0,
            ),
            "mae": np.where(
                filled,
                group["mae20"].to_numpy(dtype=np.float64, copy=False),
                0.0,
            ),
            "terminal": (
                filled
                & group["terminal_failure"].to_numpy(dtype=bool, copy=False)
            ).astype(np.float64),
        }
        for label, requested in selection_sizes.items():
            top = order[: min(int(requested), len(order))]
            record[f"{label}_count"] = int(len(top))
            record[f"{label}_u_mean"] = float(np.mean(action_values["u"][top]))
            record[f"{label}_d20_mean"] = float(
                np.mean(action_values["d20"][top])
            )
        top = order[:k1]
        record.update(
            {
                "top1pct_d5_mean": float(np.mean(action_values["d5"][top])),
                "top1pct_d10_mean": float(np.mean(action_values["d10"][top])),
                "top1pct_d20_median": float(np.median(action_values["d20"][top])),
                "top1pct_mdd_mean": float(np.mean(action_values["mdd"][top])),
                "top1pct_fade_mean": float(np.mean(action_values["fade"][top])),
                "top1pct_mfe_mean": float(np.mean(action_values["mfe"][top])),
                "top1pct_mae_mean": float(np.mean(action_values["mae"][top])),
                "top1pct_negative_d20_rate": float(
                    np.mean(action_values["d20"][top] < 0.0)
                ),
                "top1pct_terminal_failure_rate": float(
                    np.mean(action_values["terminal"][top] > 0.5)
                ),
                "top1pct_fill_coverage": float(np.mean(filled[top])),
            }
        )
        rows.append(record)
    return pd.DataFrame(rows)


def _summarize_daily_selection(daily: pd.DataFrame) -> dict[str, Any]:
    if daily.empty:
        raise ValueError("selection evaluation produced no daily rows")
    yearly = {
        str(int(year)): float(group["top1pct_u_mean"].mean())
        for year, group in daily.groupby("year", sort=True)
    }
    mean_columns = [
        "top1_u_mean",
        "top3_u_mean",
        "top10_u_mean",
        "top1pct_u_mean",
        "top5pct_u_mean",
        "top1pct_d5_mean",
        "top1pct_d10_mean",
        "top1pct_d20_mean",
        "top1pct_d20_median",
        "top1pct_mdd_mean",
        "top1pct_fade_mean",
        "top1pct_mfe_mean",
        "top1pct_mae_mean",
        "top1pct_negative_d20_rate",
        "top1pct_terminal_failure_rate",
        "top1pct_fill_coverage",
        "ndcg_at_1pct",
        "ndcg_at_5pct",
        "rank_ic",
    ]
    return {
        "date_count": int(len(daily)),
        "yearly_top1pct_common_utility": yearly,
        "worst_year_top1pct_common_utility": float(min(yearly.values())),
        **{
            name: float(daily[name].mean())
            for name in mean_columns
        },
    }


def _stratified_block_bootstrap_metric_delta(
    left: pd.DataFrame,
    right: pd.DataFrame,
    *,
    metric: str,
    iterations: int,
    block_days: int,
    seed: int,
) -> dict[str, float]:
    joined = left[["trade_date", "year", metric]].merge(
        right[["trade_date", "year", metric]],
        on=["trade_date", "year"],
        suffixes=("_left", "_right"),
        how="inner",
    )
    joined = joined.replace([np.inf, -np.inf], np.nan).dropna()
    if joined.empty:
        return {"estimate": np.nan, "lower": np.nan, "upper": np.nan}
    differences = {
        int(year): (
            group[f"{metric}_left"].to_numpy(dtype=np.float64)
            - group[f"{metric}_right"].to_numpy(dtype=np.float64)
        )
        for year, group in joined.groupby("year", sort=True)
    }
    observed = float(np.mean([values.mean() for values in differences.values()]))
    rng = np.random.default_rng(int(seed))
    draws = np.empty(int(iterations), dtype=np.float64)
    for draw in range(int(iterations)):
        yearly: list[float] = []
        for values in differences.values():
            count = len(values)
            sampled: list[np.ndarray] = []
            sampled_count = 0
            while sampled_count < count:
                start = int(rng.integers(0, count))
                indices = (start + np.arange(int(block_days))) % count
                block = values[indices]
                sampled.append(block)
                sampled_count += len(block)
            yearly.append(float(np.mean(np.concatenate(sampled)[:count])))
        draws[draw] = float(np.mean(yearly))
    alpha = 0.025
    return {
        "estimate": observed,
        "lower": float(np.quantile(draws, alpha)),
        "upper": float(np.quantile(draws, 1.0 - alpha)),
    }


def _causal_baseline_scores(
    context: ModelDataContext,
    frame: pd.DataFrame,
) -> dict[str, np.ndarray]:
    candidate_ids = frame["candidate_id"].to_numpy(dtype=np.int64, copy=False)
    catalog = {
        str(item["name"]): int(item["column_index"])
        for item in list(context.feature_manifest["continuous_catalog"])
    }
    momentum = np.asarray(
        context.continuous[candidate_ids, catalog["return_20d"]],
        dtype=np.float32,
    )
    trend_names = (
        "cs_percentile__return_5d",
        "cs_percentile__return_10d",
        "cs_percentile__return_20d",
        "cs_percentile__trend_t_20d",
        "cs_percentile__up_day_ratio_20d",
    )
    trend_parts = np.column_stack(
        [
            np.asarray(
                context.continuous[candidate_ids, catalog[name]], dtype=np.float32
            )
            for name in trend_names
        ]
    )
    with np.errstate(all="ignore"):
        trend = np.nanmean(trend_parts, axis=1).astype(np.float32)
    floor = np.finfo(np.float32).min
    return {
        "past_20d_momentum": np.nan_to_num(
            momentum, nan=floor, neginf=floor, posinf=np.finfo(np.float32).max
        ),
        "past_path_trend_consistency": np.nan_to_num(
            trend, nan=floor, neginf=floor, posinf=np.finfo(np.float32).max
        ),
    }


def _v2c_p0_scores(
    frame: pd.DataFrame,
    *,
    fold_year: int,
) -> np.ndarray:
    task_result = (
        WORKSPACE_ROOT
        / "daily_research/output/path_policy/studies/seq100_signal_close_path_value_2x2_v1"
        / "tasks"
        / f"V2C-P0_{int(fold_year)}"
        / "task_result.json"
    )
    if not task_result.exists():
        raise FileNotFoundError(task_result)
    result = json.loads(task_result.read_text(encoding="utf-8"))
    prediction_path = Path(str(result["prediction_path"])).resolve()
    if _file_sha256(prediction_path) != str(result["prediction_sha256"]):
        raise ValueError(f"frozen V2C-P0 prediction changed: {prediction_path}")
    baseline = pd.read_csv(
        prediction_path,
        usecols=["trade_date", "symbol", "score"],
        dtype={"trade_date": str, "symbol": str, "score": np.float32},
    )
    if bool(baseline.duplicated(["trade_date", "symbol"]).any()):
        raise ValueError("frozen V2C-P0 predictions contain duplicate coordinates")
    joined = frame[["trade_date", "symbol"]].merge(
        baseline,
        on=["trade_date", "symbol"],
        how="left",
        validate="one_to_one",
        sort=False,
    )
    if bool(joined["score"].isna().any()):
        missing = int(joined["score"].isna().sum())
        raise ValueError(f"frozen V2C-P0 lacks {missing} PIT candidates in {fold_year}")
    return joined["score"].to_numpy(dtype=np.float32, copy=False)


def _baseline_evaluations(
    *,
    context: ModelDataContext,
    base_by_fold: Mapping[int, pd.DataFrame],
    tree_by_fold: Mapping[int, pd.DataFrame],
    output_dir: Path,
) -> tuple[dict[str, pd.DataFrame], dict[str, dict[str, Any]]]:
    pieces: dict[str, list[pd.DataFrame]] = {
        "past_20d_momentum": [],
        "past_path_trend_consistency": [],
        "frozen_v2c_p0_predictions": [],
        "formal_tree_model": [],
    }
    for fold_year in FORMAL_FOLD_YEARS:
        frame = base_by_fold[int(fold_year)].copy()
        scores = _causal_baseline_scores(context, frame)
        scores["frozen_v2c_p0_predictions"] = _v2c_p0_scores(
            frame, fold_year=int(fold_year)
        )
        tree = tree_by_fold[int(fold_year)]
        if not np.array_equal(
            frame["candidate_id"].to_numpy(dtype=np.int64),
            tree["candidate_id"].to_numpy(dtype=np.int64),
        ):
            raise ValueError("formal tree baseline candidate order differs")
        scores["formal_tree_model"] = tree["selection_score"].to_numpy(
            dtype=np.float32, copy=False
        )
        for baseline_id, score in scores.items():
            current = frame.copy()
            current["selection_score"] = score
            pieces[baseline_id].append(_daily_selection_evaluation(current))
    daily: dict[str, pd.DataFrame] = {}
    summaries: dict[str, dict[str, Any]] = {}
    output_dir.mkdir(parents=True, exist_ok=True)
    for baseline_id, frames in pieces.items():
        current = pd.concat(frames, ignore_index=True)
        daily[baseline_id] = current
        summaries[baseline_id] = _summarize_daily_selection(current)
        _atomic_write_parquet(output_dir / f"{baseline_id}.parquet", current)
    return daily, summaries


def _fixed_list_batch_to_numpy(array: pa.Array) -> np.ndarray:
    if not pa.types.is_fixed_size_list(array.type):
        raise TypeError(f"expected fixed-size list, got {array.type}")
    width = int(array.type.list_size)
    return np.asarray(array.values.to_numpy(zero_copy_only=False), dtype=np.float32).reshape(
        len(array), width
    )


def _distribution_metrics_for_task(summary: Mapping[str, Any]) -> dict[str, Any]:
    model_id = str(summary["model_id"])
    path = Path(str(summary["prediction"]["path"])).resolve()
    parquet = pq.ParquetFile(path)
    if model_id == "patchtst_student_t_path":
        columns = [
            "entry_filled",
            "path_location",
            "path_scale",
            "path_q10",
            "path_q50",
            "path_q90",
            "true_path_net_log",
        ]
        nll_sum = 0.0
        crps_sum = 0.0
        valid_count = 0
        cover = np.zeros(3, dtype=np.float64)
        quantiles = np.linspace(0.05, 0.95, 19, dtype=np.float64)
        quantile_z = stats.t.ppf(quantiles, df=5.0).astype(np.float32)
        for batch in parquet.iter_batches(batch_size=8192, columns=columns):
            filled = np.asarray(batch.column(0).to_numpy(), dtype=bool)
            location = _fixed_list_batch_to_numpy(batch.column(1))
            scale = _fixed_list_batch_to_numpy(batch.column(2))
            q10 = _fixed_list_batch_to_numpy(batch.column(3))
            q50 = _fixed_list_batch_to_numpy(batch.column(4))
            q90 = _fixed_list_batch_to_numpy(batch.column(5))
            truth = _fixed_list_batch_to_numpy(batch.column(6))
            valid_rows = filled & np.isfinite(truth).all(axis=1)
            if not bool(valid_rows.any()):
                continue
            y = truth[valid_rows]
            mu = location[valid_rows]
            sigma = np.maximum(scale[valid_rows], 1.0e-6)
            standardized = (y - mu) / sigma
            nll = -stats.t.logpdf(standardized, df=5.0) + np.log(sigma)
            nll_sum += float(np.sum(nll))
            predicted_quantiles = (
                mu[:, :, None] + sigma[:, :, None] * quantile_z[None, None, :]
            )
            error = y[:, :, None] - predicted_quantiles
            pinball = np.maximum(
                quantiles[None, None, :] * error,
                (quantiles[None, None, :] - 1.0) * error,
            )
            crps_sum += float(np.sum(2.0 * np.mean(pinball, axis=2)))
            cover += np.asarray(
                [
                    np.mean(y <= q10[valid_rows]),
                    np.mean(y <= q50[valid_rows]),
                    np.mean(y <= q90[valid_rows]),
                ],
                dtype=np.float64,
            ) * int(y.size)
            valid_count += int(y.size)
        if not valid_count:
            return {"status": "no_valid_distribution_labels"}
        return {
            "status": "completed",
            "student_t_nll": float(nll_sum / valid_count),
            "crps_19_quantile_approximation": float(crps_sum / valid_count),
            "quantile_coverage_10": float(cover[0] / valid_count),
            "quantile_coverage_50": float(cover[1] / valid_count),
            "quantile_coverage_90": float(cover[2] / valid_count),
            "path_element_count": int(valid_count),
        }
    if model_id == "deephit_competing_risk":
        columns = [
            "entry_filled",
            "joint_event_probability",
            "competing_event_code",
            "competing_event_time",
        ]
        nll_sum = 0.0
        count = 0
        for batch in parquet.iter_batches(batch_size=32768, columns=columns):
            filled = np.asarray(batch.column(0).to_numpy(), dtype=bool)
            probability = _fixed_list_batch_to_numpy(batch.column(1))
            code = np.asarray(batch.column(2).to_numpy(), dtype=np.float64)
            event_time = np.asarray(batch.column(3).to_numpy(), dtype=np.float64)
            valid = filled & np.isfinite(code) & np.isfinite(event_time)
            if not bool(valid.any()):
                continue
            current_code = code[valid].astype(np.int64)
            current_time = np.clip(event_time[valid].astype(np.int64), 1, 20)
            target = np.full(len(current_code), 100, dtype=np.int64)
            observed = current_code > 0
            target[observed] = (
                (current_code[observed] - 1) * 20 + current_time[observed] - 1
            )
            selected = probability[valid, target]
            nll_sum += float(np.sum(-np.log(np.maximum(selected, 1.0e-12))))
            count += len(target)
        return {
            "status": "completed" if count else "no_valid_hazard_labels",
            "joint_event_nll": float(nll_sum / count) if count else np.nan,
            "event_count": int(count),
        }
    return {"status": "not_applicable"}


def _fill_metrics(frame: pd.DataFrame) -> dict[str, float]:
    from sklearn.metrics import (
        average_precision_score,
        brier_score_loss,
        roc_auc_score,
    )

    truth = frame["entry_filled"].to_numpy(dtype=bool, copy=False)
    probability = np.clip(
        frame["fill_probability"].to_numpy(dtype=np.float64, copy=False),
        0.0,
        1.0,
    )
    climatology = np.full(len(truth), float(np.mean(truth)), dtype=np.float64)
    return {
        "fill_roc_auc": float(roc_auc_score(truth, probability)),
        "fill_pr_auc": float(average_precision_score(truth, probability)),
        "fill_brier": float(brier_score_loss(truth, probability)),
        "fill_climatology_brier": float(brier_score_loss(truth, climatology)),
        "fill_ece": _expected_calibration_error(truth, probability),
    }


def _evaluate_model_tasks(
    *,
    model_id: str,
    tasks_by_fold: Mapping[int, Sequence[Mapping[str, Any]]],
    output_dir: Path,
) -> tuple[dict[int, pd.DataFrame], pd.DataFrame, dict[str, Any]]:
    frames: dict[int, pd.DataFrame] = {}
    daily_parts: list[pd.DataFrame] = []
    all_frames: list[pd.DataFrame] = []
    all_tasks: list[Mapping[str, Any]] = []
    for fold_year in FORMAL_FOLD_YEARS:
        tasks = list(tasks_by_fold[int(fold_year)])
        frame = _load_ensemble_frame(tasks)
        frames[int(fold_year)] = frame
        all_frames.append(frame)
        daily_parts.append(_daily_selection_evaluation(frame))
        all_tasks.extend(tasks)
    daily = pd.concat(daily_parts, ignore_index=True)
    combined = pd.concat(all_frames, ignore_index=True)
    summary = {
        **_summarize_daily_selection(daily),
        **_fill_metrics(combined),
        "seed_count": int(
            len({int(task["seed"]) for task in all_tasks})
        ),
        "seeds": sorted({int(task["seed"]) for task in all_tasks}),
        "fold_years": list(FORMAL_FOLD_YEARS),
        "inference_microseconds_per_candidate": float(
            np.mean(
                [
                    float(task["inference_microseconds_per_candidate"])
                    for task in all_tasks
                ]
            )
        ),
        "parameter_count": int(
            max(int(task["adapter"].get("parameter_count", 0)) for task in all_tasks)
        ),
        "checkpoint_size_bytes": int(
            max(
                int(task["adapter"].get("checkpoint_size_bytes", 0))
                for task in all_tasks
            )
        ),
    }
    distribution_records = [
        _distribution_metrics_for_task(task) for task in all_tasks
    ]
    applicable = [
        record
        for record in distribution_records
        if str(record.get("status", "")) == "completed"
    ]
    if applicable:
        numeric_keys = sorted(
            set.intersection(
                *[
                    {
                        key
                        for key, value in record.items()
                        if isinstance(value, (int, float)) and key not in {"event_count", "path_element_count"}
                    }
                    for record in applicable
                ]
            )
        )
        summary["distribution_metrics"] = {
            "status": "completed",
            **{
                key: float(np.mean([float(record[key]) for record in applicable]))
                for key in numeric_keys
            },
            "task_count": int(len(applicable)),
        }
    else:
        summary["distribution_metrics"] = {"status": "not_applicable"}
    output_dir.mkdir(parents=True, exist_ok=True)
    _atomic_write_parquet(output_dir / f"{model_id}_daily.parquet", daily)
    _atomic_write_json(output_dir / f"{model_id}_summary.json", summary)
    return frames, daily, summary


def _strongest_baseline_id(
    baseline_summaries: Mapping[str, Mapping[str, Any]],
    *,
    model_id: str,
) -> str:
    candidates = list(baseline_summaries)
    if str(model_id) == "lgbm_lambdarank_multioutput":
        candidates = [item for item in candidates if item != "formal_tree_model"]
    return max(
        candidates,
        key=lambda item: (
            float(baseline_summaries[item]["top1pct_u_mean"]),
            item,
        ),
    )


def _qualify_model(
    *,
    model_id: str,
    model_daily: pd.DataFrame,
    model_summary: Mapping[str, Any],
    baseline_daily: Mapping[str, pd.DataFrame],
    baseline_summaries: Mapping[str, Mapping[str, Any]],
    bootstrap_spec: Mapping[str, Any],
    three_seed_complete: bool,
) -> dict[str, Any]:
    baseline_id = _strongest_baseline_id(
        baseline_summaries, model_id=model_id
    )
    base_daily = baseline_daily[baseline_id]
    base = baseline_summaries[baseline_id]
    iterations = int(bootstrap_spec["iterations"])
    block_days = int(bootstrap_spec["block_trading_days"])
    seed = int(bootstrap_spec["seed"])
    primary = _stratified_block_bootstrap_metric_delta(
        model_daily,
        base_daily,
        metric="top1pct_u_mean",
        iterations=iterations,
        block_days=block_days,
        seed=seed,
    )
    negative = _stratified_block_bootstrap_metric_delta(
        model_daily,
        base_daily,
        metric="top1pct_negative_d20_rate",
        iterations=iterations,
        block_days=block_days,
        seed=seed + 1,
    )
    terminal = _stratified_block_bootstrap_metric_delta(
        model_daily,
        base_daily,
        metric="top1pct_terminal_failure_rate",
        iterations=iterations,
        block_days=block_days,
        seed=seed + 2,
    )
    improved_years = sum(
        float(value)
        > float(base["yearly_top1pct_common_utility"].get(year, np.inf))
        for year, value in dict(
            model_summary["yearly_top1pct_common_utility"]
        ).items()
    )
    gates = {
        "pooled_primary_delta_positive": bool(primary["estimate"] > 0.0),
        "pooled_bootstrap_lower_positive": bool(primary["lower"] > 0.0),
        "improved_years_min_2": bool(improved_years >= 2),
        "top1pct_d20_median_degradation_within_0_5pct": bool(
            float(model_summary["top1pct_d20_median"])
            >= float(base["top1pct_d20_median"]) - 0.005
        ),
        "mdd_degradation_within_1pct": bool(
            float(model_summary["top1pct_mdd_mean"])
            <= float(base["top1pct_mdd_mean"]) + 0.01
        ),
        "fade_degradation_within_1pct": bool(
            float(model_summary["top1pct_fade_mean"])
            <= float(base["top1pct_fade_mean"]) + 0.01
        ),
        "negative_d20_not_significantly_worse": bool(
            not (math.isfinite(negative["lower"]) and negative["lower"] > 0.0)
        ),
        "terminal_failure_not_significantly_worse": bool(
            not (math.isfinite(terminal["lower"]) and terminal["lower"] > 0.0)
        ),
        "fill_brier_beats_climatology": bool(
            float(model_summary["fill_brier"])
            < float(model_summary["fill_climatology_brier"])
        ),
        "three_seed_complete": bool(three_seed_complete),
    }
    return {
        "model_id": model_id,
        "strongest_baseline_id": baseline_id,
        "primary_bootstrap": primary,
        "negative_d20_bootstrap": negative,
        "terminal_failure_bootstrap": terminal,
        "improved_year_count": int(improved_years),
        "gates": gates,
        "passed_without_seed_gate": bool(
            all(value for key, value in gates.items() if key != "three_seed_complete")
        ),
        "passed": bool(all(gates.values())),
    }


def _winner_sort_key(
    model_id: str,
    summary: Mapping[str, Any],
) -> tuple[Any, ...]:
    return (
        -float(summary["top1pct_u_mean"]),
        -float(summary["worst_year_top1pct_common_utility"]),
        -float(summary["ndcg_at_1pct"]),
        -float(summary["top1pct_d20_mean"]),
        float(summary["top1pct_mdd_mean"]),
        float(summary["top1pct_terminal_failure_rate"]),
        float(summary["inference_microseconds_per_candidate"]),
        int(summary["checkpoint_size_bytes"]),
        str(model_id),
    )


def _write_evaluation_result(
    *,
    output_root: Path,
    attempt_dir: Path,
    study: Mapping[str, Any],
    report: dict[str, Any],
) -> dict[str, Any]:
    report["evaluation_sha256"] = _canonical_json_sha256(report)
    report_path = attempt_dir / "evaluation_report.json"
    _atomic_write_json(report_path, report)
    pointer = {
        "attempt": str(attempt_dir.resolve()),
        "evaluation_report": str(report_path.resolve()),
        "evaluation_report_sha256": _file_sha256(report_path),
        "evaluation_sha256": report["evaluation_sha256"],
        "status": report["status"],
        "winner": report.get("winner"),
        "robustness_model_ids": list(report.get("robustness_model_ids", []) or []),
        "study_contract_sha256": study["contract_sha256"],
        "updated_at": _now(),
    }
    _atomic_write_json(output_root / "evaluation/current.json", pointer)
    return {
        "status": report["status"],
        "output_dir": str(attempt_dir.resolve()),
        "evaluation_report": str(report_path.resolve()),
        "evaluation_sha256": report["evaluation_sha256"],
        "winner": report.get("winner"),
        "robustness_model_ids": pointer["robustness_model_ids"],
    }


def evaluate_study(
    *,
    study_path: Path = DEFAULT_STUDY_PATH,
    output_root: Path = DEFAULT_OUTPUT_ROOT,
) -> dict[str, Any]:
    context = _resolve_model_data(study_path=study_path, output_root=output_root)
    _matrix_path, matrix = _load_formal_matrix(output_root, context.study)
    protection_before = _verify_protected_bindings(context.study)
    attempt_dir = _next_attempt_dir(output_root, "evaluation")
    seed7_tasks: dict[str, dict[int, list[Mapping[str, Any]]]] = {}
    missing: list[str] = []
    for model_id in list(matrix["model_ids"]):
        seed7_tasks[str(model_id)] = {}
        for fold_year in FORMAL_FOLD_YEARS:
            summary = _load_training_task_summary(
                output_root,
                model_id=str(model_id),
                fold_year=int(fold_year),
                seed=7,
            )
            if summary is None:
                missing.append(f"{model_id}/fold_{fold_year}/seed_7")
            else:
                seed7_tasks[str(model_id)][int(fold_year)] = [summary]
    if missing:
        protection_after = _verify_protected_bindings(context.study)
        report = {
            "artifact_type": "seq100_signal_quality_evaluation",
            "status": "study_incomplete",
            "created_at": _now(),
            "study_contract_sha256": context.study["contract_sha256"],
            "formal_matrix_sha256": matrix["formal_matrix_sha256"],
            "missing_formal_tasks": missing,
            "winner": None,
            "robustness_model_ids": [],
            "protection_before": protection_before,
            "protection_after": protection_after,
        }
        return _write_evaluation_result(
            output_root=output_root,
            attempt_dir=attempt_dir,
            study=context.study,
            report=report,
        )

    seed7_frames: dict[str, dict[int, pd.DataFrame]] = {}
    seed7_daily: dict[str, pd.DataFrame] = {}
    seed7_summaries: dict[str, dict[str, Any]] = {}
    for model_id in list(matrix["model_ids"]):
        frames, daily, summary = _evaluate_model_tasks(
            model_id=str(model_id),
            tasks_by_fold=seed7_tasks[str(model_id)],
            output_dir=attempt_dir / "seed7_models",
        )
        seed7_frames[str(model_id)] = frames
        seed7_daily[str(model_id)] = daily
        seed7_summaries[str(model_id)] = summary
    tree_id = "lgbm_lambdarank_multioutput"
    base_by_fold = seed7_frames[tree_id]
    baseline_daily, baseline_summaries = _baseline_evaluations(
        context=context,
        base_by_fold=base_by_fold,
        tree_by_fold=seed7_frames[tree_id],
        output_dir=attempt_dir / "baselines",
    )
    bootstrap_spec = dict(context.research_freeze["evaluation"]["bootstrap"])
    preliminary_qualification = {
        model_id: _qualify_model(
            model_id=model_id,
            model_daily=seed7_daily[model_id],
            model_summary=seed7_summaries[model_id],
            baseline_daily=baseline_daily,
            baseline_summaries=baseline_summaries,
            bootstrap_spec=bootstrap_spec,
            three_seed_complete=False,
        )
        for model_id in list(matrix["model_ids"])
    }
    preliminary_eligible = [
        model_id
        for model_id in list(matrix["model_ids"])
        if preliminary_qualification[model_id]["passed_without_seed_gate"]
    ]
    robustness_model_ids = sorted(
        preliminary_eligible,
        key=lambda model_id: _winner_sort_key(
            model_id, seed7_summaries[model_id]
        ),
    )[:2]
    path_candidates = [
        model_id
        for model_id, summary in seed7_summaries.items()
        if model_id == "patchtst_student_t_path"
        and str(summary.get("distribution_metrics", {}).get("status", ""))
        == "completed"
    ]
    best_path_forecaster = (
        min(
            path_candidates,
            key=lambda model_id: float(
                seed7_summaries[model_id]["distribution_metrics"]["student_t_nll"]
            ),
        )
        if path_candidates
        else None
    )
    if not robustness_model_ids:
        protection_after = _verify_protected_bindings(context.study)
        report = {
            "artifact_type": "seq100_signal_quality_evaluation",
            "status": "winner_null",
            "created_at": _now(),
            "study_contract_sha256": context.study["contract_sha256"],
            "formal_matrix_sha256": matrix["formal_matrix_sha256"],
            "target_sha256": context.target_manifest["frozen_target_sha256"],
            "feature_sha256": context.feature_manifest[
                "resolved_feature_view_sha256"
            ],
            "baseline_summaries": baseline_summaries,
            "seed7_model_summaries": seed7_summaries,
            "preliminary_qualification": preliminary_qualification,
            "robustness_model_ids": [],
            "selector_winner": None,
            "best_path_forecaster": best_path_forecaster,
            "winner": None,
            "reason": "no model passed the frozen first-seed qualification gates",
            "protection_before": protection_before,
            "protection_after": protection_after,
        }
        return _write_evaluation_result(
            output_root=output_root,
            attempt_dir=attempt_dir,
            study=context.study,
            report=report,
        )

    robustness_tasks: dict[str, dict[int, list[Mapping[str, Any]]]] = {}
    missing_robustness: list[str] = []
    for model_id in robustness_model_ids:
        robustness_tasks[model_id] = {}
        for fold_year in FORMAL_FOLD_YEARS:
            tasks = list(seed7_tasks[model_id][int(fold_year)])
            for seed in (17, 29):
                summary = _load_training_task_summary(
                    output_root,
                    model_id=model_id,
                    fold_year=int(fold_year),
                    seed=int(seed),
                )
                if summary is None:
                    missing_robustness.append(
                        f"{model_id}/fold_{fold_year}/seed_{seed}"
                    )
                else:
                    tasks.append(summary)
            robustness_tasks[model_id][int(fold_year)] = tasks
    if missing_robustness:
        protection_after = _verify_protected_bindings(context.study)
        report = {
            "artifact_type": "seq100_signal_quality_evaluation",
            "status": "robustness_required",
            "created_at": _now(),
            "study_contract_sha256": context.study["contract_sha256"],
            "formal_matrix_sha256": matrix["formal_matrix_sha256"],
            "target_sha256": context.target_manifest["frozen_target_sha256"],
            "feature_sha256": context.feature_manifest[
                "resolved_feature_view_sha256"
            ],
            "baseline_summaries": baseline_summaries,
            "seed7_model_summaries": seed7_summaries,
            "preliminary_qualification": preliminary_qualification,
            "robustness_model_ids": robustness_model_ids,
            "missing_robustness_tasks": missing_robustness,
            "selector_winner": None,
            "best_path_forecaster": best_path_forecaster,
            "winner": None,
            "protection_before": protection_before,
            "protection_after": protection_after,
        }
        return _write_evaluation_result(
            output_root=output_root,
            attempt_dir=attempt_dir,
            study=context.study,
            report=report,
        )

    robust_frames: dict[str, dict[int, pd.DataFrame]] = {}
    robust_daily: dict[str, pd.DataFrame] = {}
    robust_summaries: dict[str, dict[str, Any]] = {}
    final_qualification: dict[str, Any] = {}
    for model_id in robustness_model_ids:
        frames, daily, summary = _evaluate_model_tasks(
            model_id=model_id,
            tasks_by_fold=robustness_tasks[model_id],
            output_dir=attempt_dir / "three_seed_models",
        )
        robust_frames[model_id] = frames
        robust_daily[model_id] = daily
        robust_summaries[model_id] = summary
        final_qualification[model_id] = _qualify_model(
            model_id=model_id,
            model_daily=daily,
            model_summary=summary,
            baseline_daily=baseline_daily,
            baseline_summaries=baseline_summaries,
            bootstrap_spec=bootstrap_spec,
            three_seed_complete=True,
        )
    qualified = [
        model_id
        for model_id in robustness_model_ids
        if final_qualification[model_id]["passed"]
    ]
    winner = (
        min(
            qualified,
            key=lambda model_id: _winner_sort_key(
                model_id, robust_summaries[model_id]
            ),
        )
        if qualified
        else None
    )
    complexity_fallback: dict[str, Any] | None = None
    if winner is not None and winner != tree_id and tree_id in qualified:
        comparison = _stratified_block_bootstrap_metric_delta(
            robust_daily[winner],
            robust_daily[tree_id],
            metric="top1pct_u_mean",
            iterations=int(bootstrap_spec["iterations"]),
            block_days=int(bootstrap_spec["block_trading_days"]),
            seed=int(bootstrap_spec["seed"]) + 99,
        )
        complexity_fallback = {
            "advanced_model": winner,
            "tree_model": tree_id,
            **comparison,
        }
        if comparison["lower"] <= 0.0 <= comparison["upper"]:
            winner = tree_id
            complexity_fallback["applied"] = True
        else:
            complexity_fallback["applied"] = False
    protection_after = _verify_protected_bindings(context.study)
    report = {
        "artifact_type": "seq100_signal_quality_evaluation",
        "status": "completed" if winner is not None else "winner_null",
        "created_at": _now(),
        "study_contract_sha256": context.study["contract_sha256"],
        "formal_matrix_sha256": matrix["formal_matrix_sha256"],
        "target_sha256": context.target_manifest["frozen_target_sha256"],
        "feature_sha256": context.feature_manifest[
            "resolved_feature_view_sha256"
        ],
        "bootstrap": bootstrap_spec,
        "baseline_summaries": baseline_summaries,
        "seed7_model_summaries": seed7_summaries,
        "preliminary_qualification": preliminary_qualification,
        "robustness_model_ids": robustness_model_ids,
        "three_seed_model_summaries": robust_summaries,
        "final_qualification": final_qualification,
        "complexity_fallback": complexity_fallback,
        "selector_winner": winner,
        "best_path_forecaster": best_path_forecaster,
        "winner": winner,
        "protection_before": protection_before,
        "protection_after": protection_after,
    }
    return _write_evaluation_result(
        output_root=output_root,
        attempt_dir=attempt_dir,
        study=context.study,
        report=report,
    )


def _terminal_study_evidence(
    *,
    study: Mapping[str, Any],
    output_root: Path,
) -> tuple[str, Path, dict[str, Any]]:
    evaluation_pointer = output_root / "evaluation/current.json"
    if evaluation_pointer.exists():
        pointer = json.loads(evaluation_pointer.read_text(encoding="utf-8"))
        report_path = Path(str(pointer["evaluation_report"])).resolve()
        if _file_sha256(report_path) != str(pointer["evaluation_report_sha256"]):
            raise ValueError("evaluation report changed before closeout")
        report = json.loads(report_path.read_text(encoding="utf-8"))
        status = str(report["status"])
        if status == "robustness_required":
            raise ValueError("study cannot close while robustness tasks are pending")
        if status in {"completed", "winner_null", "study_incomplete"}:
            return status, report_path, report
    target_path, target = _load_current_artifact(
        output_root, task="targets", pointer_name="target_manifest"
    )
    if str(target.get("status", "")) == "target_invalid":
        return "target_invalid", target_path, target
    screen_pointer = output_root / "model_screen/current.json"
    if screen_pointer.exists():
        pointer = json.loads(screen_pointer.read_text(encoding="utf-8"))
        if str(pointer.get("status", "")) == "research_design_insufficient":
            result_path = Path(str(pointer["result"])).resolve()
            if _file_sha256(result_path) != str(pointer["result_sha256"]):
                raise ValueError("model-screen terminal record changed")
            return (
                "research_design_insufficient",
                result_path,
                json.loads(result_path.read_text(encoding="utf-8")),
            )
    raise ValueError("study has no terminal evidence to close")


def _upsert_marked_state_section(
    path: Path,
    *,
    marker: str,
    body: str,
) -> None:
    text = path.read_text(encoding="utf-8")
    start = f"<!-- {marker}:start -->"
    end = f"<!-- {marker}:end -->"
    section = f"{start}\n{body.rstrip()}\n{end}"
    if start in text and end in text:
        prefix, remainder = text.split(start, 1)
        _old, suffix = remainder.split(end, 1)
        updated = prefix.rstrip() + "\n\n" + section + suffix
    else:
        updated = text.rstrip() + "\n\n" + section + "\n"
    current_date = datetime.now().astimezone().strftime("%Y-%m-%d")
    lines = updated.splitlines()
    for index, line in enumerate(lines):
        if line.startswith("Updated:"):
            lines[index] = f"Updated: `{current_date}`"
            break
    path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


def _run_closeout_command(command: Sequence[str]) -> dict[str, Any]:
    completed = subprocess.run(
        list(command),
        cwd=WORKSPACE_ROOT,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        check=False,
    )
    return {
        "command": list(command),
        "returncode": int(completed.returncode),
        "stdout": completed.stdout[-20_000:],
        "stderr": completed.stderr[-20_000:],
        "passed": bool(completed.returncode == 0),
    }


def closeout_study(
    *,
    study_path: Path = DEFAULT_STUDY_PATH,
    output_root: Path = DEFAULT_OUTPUT_ROOT,
) -> dict[str, Any]:
    study = load_study(study_path)
    freeze_payload = load_research_freeze(study)
    protection_before = _verify_protected_bindings(study)
    terminal_status, evidence_path, evidence = _terminal_study_evidence(
        study=study, output_root=output_root
    )
    record_root = (
        WORKSPACE_ROOT
        / "daily_research/research_records/seq100/seq100_pit_signal_quality_v1"
    )
    if record_root.exists():
        artifact_path = record_root / "artifact.json"
        if artifact_path.exists():
            artifact = json.loads(artifact_path.read_text(encoding="utf-8"))
            return {
                "status": str(artifact["status"]),
                "research_record": str(artifact_path.resolve()),
                "already_closed": True,
            }
        raise FileExistsError(record_root)
    record_root.mkdir(parents=True, exist_ok=False)
    contract_copy = record_root / "contract.json"
    _atomic_write_json(contract_copy, study)
    target_pointer = output_root / "targets/current.json"
    target_summary: dict[str, Any] | None = None
    if target_pointer.exists():
        target_path, target = _load_current_artifact(
            output_root, task="targets", pointer_name="target_manifest"
        )
        target_summary = {
            "status": target.get("status"),
            "frozen_target_id": target.get("frozen_target_id"),
            "frozen_target_sha256": target.get("frozen_target_sha256"),
            "target_manifest": str(target_path.resolve()),
            "target_manifest_sha256": _file_sha256(target_path),
        }
    feature_summary: dict[str, Any] | None = None
    feature_pointer = output_root / "feature_screen/current.json"
    if feature_pointer.exists():
        feature_path, feature = _load_feature_freeze(output_root, study)
        feature_summary = {
            "selected_profile": feature.get("selected_profile"),
            "feature_freeze_sha256": feature.get("feature_freeze_sha256"),
            "feature_freeze": str(feature_path.resolve()),
            "feature_freeze_file_sha256": _file_sha256(feature_path),
        }
    matrix_summary: dict[str, Any] | None = None
    matrix_pointer = output_root / "model_screen/current.json"
    if matrix_pointer.exists():
        pointer = json.loads(matrix_pointer.read_text(encoding="utf-8"))
        if "formal_matrix" in pointer:
            matrix_path, matrix = _load_formal_matrix(output_root, study)
            matrix_summary = {
                "model_ids": list(matrix["model_ids"]),
                "formal_matrix_sha256": matrix["formal_matrix_sha256"],
                "formal_matrix": str(matrix_path.resolve()),
                "formal_matrix_file_sha256": _file_sha256(matrix_path),
            }
    artifact = {
        "schema_version": 1,
        "artifact_type": "seq100_pit_signal_quality_research_record",
        "status": terminal_status,
        "completed_at": _now(),
        "study_id": STUDY_ID,
        "selection_years": [2023, 2024, 2025],
        "confirmation_year": 2026,
        "contract": {
            "path": str(contract_copy.relative_to(WORKSPACE_ROOT)).replace("\\", "/"),
            "contract_sha256": study["contract_sha256"],
            "contract_file_sha256": _file_sha256(contract_copy),
        },
        "external_research": {
            "review_path": freeze_payload["freeze"]["evidence_bindings"][
                "external_review"
            ]["path"],
            "review_sha256": freeze_payload["freeze"]["evidence_bindings"][
                "external_review"
            ]["sha256"],
            "source_count": 38,
            "paper_or_book_count": 23,
            "official_or_author_repository_count": 14,
            "research_freeze_path": study["contract"]["research_freeze"]["path"],
            "research_freeze_sha256": freeze_payload["freeze_sha256"],
            "research_freeze_file_sha256": study["contract"]["research_freeze"][
                "file_sha256"
            ],
        },
        "target": target_summary,
        "features": feature_summary,
        "formal_matrix": matrix_summary,
        "terminal_evidence": {
            "path": str(evidence_path.resolve()),
            "sha256": _file_sha256(evidence_path),
        },
        "winner": evidence.get("winner"),
        "selector_winner": evidence.get("selector_winner"),
        "best_path_forecaster": evidence.get("best_path_forecaster"),
        "key_statistics": {
            "baseline_summaries": evidence.get("baseline_summaries"),
            "three_seed_model_summaries": evidence.get(
                "three_seed_model_summaries"
            ),
            "final_qualification": evidence.get("final_qualification"),
            "reason": evidence.get("reason"),
        },
        "retained_evidence": {
            "root": str(output_root.relative_to(WORKSPACE_ROOT)).replace("\\", "/"),
            "checkpoints_retained": True,
            "predictions_retained": True,
            "logs_retained": True,
            "source_review_retained": True,
        },
        "mutations": {
            "qdp": False,
            "source_pack": False,
            "formal_model_registry": False,
            "registered_checkpoints": False,
            "active_execution": False,
            "commit_or_push": False,
        },
        "protection_before": protection_before,
    }
    artifact_path = record_root / "artifact.json"
    _atomic_write_json(artifact_path, artifact)

    index_path = WORKSPACE_ROOT / "daily_research/research_records/seq100/index.json"
    index = json.loads(index_path.read_text(encoding="utf-8"))
    record_entry = {
        "id": STUDY_ID,
        "status": terminal_status,
        "path": str(artifact_path.relative_to(WORKSPACE_ROOT)).replace("\\", "/"),
    }
    records = [
        item for item in list(index.get("records", []) or [])
        if str(item.get("id", "")) != STUDY_ID
    ]
    records.append(record_entry)
    index["records"] = records
    index["active_studies"] = [
        item for item in list(index.get("active_studies", []) or [])
        if str(item) != STUDY_ID
    ]
    index["updated_at"] = datetime.now().astimezone().strftime("%Y-%m-%d")
    _atomic_write_json(index_path, index)

    winner_text = str(evidence.get("winner") or "null")
    state_body = (
        f"- `{STUDY_ID}` 已以 `{terminal_status}` 收口；selector winner 为 "
        f"`{winner_text}`。\n"
        "- 权威 compact record："
        "`daily_research/research_records/seq100/seq100_pit_signal_quality_v1/artifact.json`。\n"
        "- 正式模型 registry、active execution、QDP 与完整 PIT pack 均未修改；2026 仍只允许新合同确认。"
    )
    _upsert_marked_state_section(
        WORKSPACE_ROOT / "daily_research/brain/state.md",
        marker="seq100-signal-quality",
        body=state_body,
    )
    _upsert_marked_state_section(
        WORKSPACE_ROOT / "brain/state.md",
        marker="seq100-signal-quality",
        body=state_body,
    )

    focused_test = _run_closeout_command(
        [
            "C:/Users/ASUS/miniconda3/envs/yolos/python.exe",
            "-m",
            "pytest",
            "daily_research/path_policy/tests/test_seq100_signal_quality.py",
            "-q",
        ]
    )
    if not focused_test["passed"]:
        raise RuntimeError("focused closeout regression failed")
    doctor_script = Path(
        "C:/Users/ASUS/.agents/skills/workspace-brain/scripts/workspace_brain.py"
    )
    validations = {
        "focused_tests": focused_test,
        "brain_doctor": _run_closeout_command(
            [
                "C:/Users/ASUS/miniconda3/envs/yolos/python.exe",
                str(doctor_script),
                "doctor",
                "--root",
                str(WORKSPACE_ROOT),
                "--json",
            ]
        ),
        "brain_integrity": _run_closeout_command(
            [
                "C:/Users/ASUS/miniconda3/envs/yolos/python.exe",
                "-m",
                "tools.brain.integrity_check",
                "--json",
            ]
        ),
        "git_diff_check": _run_closeout_command(["git", "diff", "--check"]),
    }
    if not all(record["passed"] for record in validations.values()):
        raise RuntimeError("one or more closeout validation gates failed")
    if study_path.resolve() != DEFAULT_STUDY_PATH.resolve():
        raise ValueError("closeout may remove only the canonical active study contract")
    study_path.unlink()
    protection_after = _verify_protected_bindings(study)
    if protection_after != protection_before:
        raise RuntimeError("protected-object hashes changed during closeout")
    artifact["protection_after"] = protection_after
    artifact["validations"] = validations
    artifact["active_contract_removed"] = True
    artifact["artifact_sha256"] = _canonical_json_sha256(artifact)
    _atomic_write_json(artifact_path, artifact)
    summary = {
        "status": terminal_status,
        "research_record": str(artifact_path.resolve()),
        "artifact_sha256": artifact["artifact_sha256"],
        "active_contract_removed": True,
        "winner": evidence.get("winner"),
    }
    _atomic_write_json(output_root / "closeout_summary.json", summary)
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build and evaluate the frozen Seq100 PIT signal-quality study."
    )
    parser.add_argument("--study-contract", type=Path, default=DEFAULT_STUDY_PATH)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("validate-research")
    targets = sub.add_parser("build-targets")
    targets.add_argument(
        "--pareto-device", choices=("auto", "cpu", "cuda"), default="auto"
    )
    sub.add_parser("freeze-target")
    sub.add_parser("build-view")
    sub.add_parser("screen-models")
    train = sub.add_parser("train")
    train.add_argument("--model-id", choices=MODEL_IDS, required=True)
    train.add_argument("--fold-year", type=int, choices=FORMAL_FOLD_YEARS, required=True)
    train.add_argument("--seed", type=int, default=7)
    sub.add_parser("evaluate")
    sub.add_parser("closeout")
    parser.add_argument("--json", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> dict[str, Any]:
    args = build_parser().parse_args(argv)
    if args.command == "validate-research":
        result = validate_research(
            study_path=args.study_contract,
            output_root=args.output_root,
        )
    elif args.command == "build-targets":
        result = build_targets(
            study_path=args.study_contract,
            output_root=args.output_root,
            pareto_device=str(args.pareto_device),
        )
    elif args.command == "freeze-target":
        result = freeze_target(
            study_path=args.study_contract,
            output_root=args.output_root,
        )
    elif args.command == "build-view":
        result = build_view(
            study_path=args.study_contract,
            output_root=args.output_root,
        )
    elif args.command == "screen-models":
        result = screen_models(
            study_path=args.study_contract,
            output_root=args.output_root,
        )
    elif args.command == "train":
        result = train_model(
            study_path=args.study_contract,
            output_root=args.output_root,
            model_id=str(args.model_id),
            fold_year=int(args.fold_year),
            seed=int(args.seed),
        )
    elif args.command == "evaluate":
        result = evaluate_study(
            study_path=args.study_contract,
            output_root=args.output_root,
        )
    elif args.command == "closeout":
        result = closeout_study(
            study_path=args.study_contract,
            output_root=args.output_root,
        )
    else:
        raise AssertionError(args.command)
    if args.json:
        print(json.dumps(result, ensure_ascii=False, indent=2, default=_json_default))
    else:
        print(result.get("output_dir", ""))
    return result


if __name__ == "__main__":
    main()
