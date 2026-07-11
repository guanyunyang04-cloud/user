from __future__ import annotations

import argparse
import ctypes
import gc
import hashlib
import json
import math
import random
import time
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Iterator, Mapping

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import BatchSampler, Dataset

from daily_research.path_policy.qdp_v2_sequence_path_pack import (
    DEFAULT_FORWARD_DAYS,
    DEFAULT_LOOKBACK_DAYS,
    PATH_OHLCVA_FIELDS,
    PATH_OHLC_FIELDS,
    PATH_SUMMARY_COLUMNS,
    _json_default,
    _resolve_deferred_exit_days,
    _write_json,
    path_summary_columns,
    path_value_column,
)


DEFAULT_OUTPUT_ROOT = Path("daily_research/output/path_policy/sequence_path_training")
DEFAULT_TOP_K = (5, 10, 20, 50, 100)
DEFAULT_SEED = 7
EVALUATION_MODE_STANDARD = "standard"
EVALUATION_MODE_FIXED_OOS = "fixed_oos"
EVALUATION_MODE_DEVELOPMENT = "development"
EVALUATION_MODES = (
    EVALUATION_MODE_STANDARD,
    EVALUATION_MODE_FIXED_OOS,
    EVALUATION_MODE_DEVELOPMENT,
)
EARLY_STOPPING_METRIC_VALIDATION_RANK_IC = "validation_rank_ic_mean"
EARLY_STOPPING_METRIC_DEVELOPMENT_TOTAL_LOSS = "development_total_loss"
EARLY_STOPPING_MODE_MIN = "min"
EARLY_STOPPING_MODE_MAX = "max"
EARLY_STOPPING_MODES = (EARLY_STOPPING_MODE_MIN, EARLY_STOPPING_MODE_MAX)
PATH_VALUE_V2_WAITING_PENALTY = 0.04
PATH_VALUE_V2_DRAWDOWN_PENALTY = 0.60
PATH_VALUE_V2_TRANSACTION_COST = 0.002
PATH_VALUE_V2_TEMPERATURE = 0.03
PATH_VALUE_GRADIENT_PROFILE_SMOOTH = "smooth_current"
PATH_VALUE_GRADIENT_PROFILE_HARD_ST = "hard_st"
PATH_VALUE_GRADIENT_PROFILES = (
    PATH_VALUE_GRADIENT_PROFILE_SMOOTH,
    PATH_VALUE_GRADIENT_PROFILE_HARD_ST,
)
RANK_TRAINING_PROFILE_LOCAL_CHUNK = "local_chunk"
RANK_TRAINING_PROFILE_GLOBAL_TAIL_512 = "global_tail_512"
RANK_TRAINING_PROFILES = (
    RANK_TRAINING_PROFILE_LOCAL_CHUNK,
    RANK_TRAINING_PROFILE_GLOBAL_TAIL_512,
)
GLOBAL_TAIL_GROUP_COUNTS = {
    "true_top": 32,
    "true_rank_33_256": 128,
    "middle": 128,
    "bottom": 96,
    "prior_epoch_false_positive": 128,
}
SUMMARY_LOSS_PROFILE_BASE = "base"
SUMMARY_LOSS_PROFILE_MULTI_HORIZON_OHLC = "multi_horizon_ohlc"
SUMMARY_LOSS_PROFILE_MULTI_HORIZON_OHLC_NO60 = "multi_horizon_ohlc_no60"
SUMMARY_LOSS_PROFILES = (
    SUMMARY_LOSS_PROFILE_BASE,
    SUMMARY_LOSS_PROFILE_MULTI_HORIZON_OHLC,
    SUMMARY_LOSS_PROFILE_MULTI_HORIZON_OHLC_NO60,
)
PATH_LOSS_PROFILE_DEFAULT = "default"
PATH_LOSS_PROFILE_OHLCVA_EQUAL = "ohlcva_equal"
PATH_LOSS_PROFILES = (
    PATH_LOSS_PROFILE_DEFAULT,
    PATH_LOSS_PROFILE_OHLCVA_EQUAL,
)
MULTI_HORIZON_OHLC_WINDOWS = (5, 10, 20, 40, 60)
INPUT_CHANNEL_PROFILE_ALL = "all"
INPUT_CHANNEL_PROFILE_DAILY_ONLY = "daily_only"
INPUT_CHANNEL_PROFILE_NO_INTRADAY_SUMMARY = "no_intraday_summary"
INPUT_CHANNEL_PROFILE_NO_LIMIT_STRUCTURE = "no_limit_structure"
INPUT_CHANNEL_PROFILES = (
    INPUT_CHANNEL_PROFILE_ALL,
    INPUT_CHANNEL_PROFILE_DAILY_ONLY,
    INPUT_CHANNEL_PROFILE_NO_INTRADAY_SUMMARY,
    INPUT_CHANNEL_PROFILE_NO_LIMIT_STRUCTURE,
)
INPUT_CHANNEL_PROFILE_ORDERS = {
    INPUT_CHANNEL_PROFILE_ALL: ("daily_raw", "daily_state", "intraday_summary", "limit_structure"),
    INPUT_CHANNEL_PROFILE_DAILY_ONLY: ("daily_raw", "daily_state"),
    INPUT_CHANNEL_PROFILE_NO_INTRADAY_SUMMARY: ("daily_raw", "daily_state", "limit_structure"),
    INPUT_CHANNEL_PROFILE_NO_LIMIT_STRUCTURE: ("daily_raw", "daily_state", "intraday_summary"),
}
BASE_INPUT_MASK_FEATURES = ("has_bar", "is_suspended", "previous_close_valid", "zero_range")
INTRADAY_INPUT_MASK_FEATURES = ("corr_valid",)
UNIFIED_VALUE_WAIT_PENALTY = 0.015
UNIFIED_VALUE_HOLD_PENALTY = 0.025
UNIFIED_VALUE_DRAWDOWN_PENALTY = 0.60
UNIFIED_VALUE_LIQUIDITY_PENALTY = 0.02
UNIFIED_VALUE_TRANSACTION_COST = 0.002
UNIFIED_VALUE_TEMPERATURE = 0.03
RICHER_MODEL_TYPES = {"gru_richer_path_value", "gru_richer_path_value_symbol"}
OHLCVA_MODEL_TYPES = {"gru_ohlcva_path_value"}
OHLCVA_AUX_MODEL_TYPES = {"gru_ohlcva_aux_path_value"}
DIRECT_VALUE_MODEL_TYPES = {"gru_direct_value"}
PATH_VALUE_MODEL_TYPES = {
    "gru_path_value",
    "gru_path_value_symbol",
    "gru_path_value_residual",
    *OHLCVA_MODEL_TYPES,
    *OHLCVA_AUX_MODEL_TYPES,
    *RICHER_MODEL_TYPES,
}
RESIDUAL_MODEL_TYPES = {"gru_path_value_residual"}
RICHER_DAILY_RAW_TARGETS = [
    "open_ret_prev_close",
    "high_ret_prev_close",
    "low_ret_prev_close",
    "close_ret_prev_close",
    "volume_log",
    "amount_log",
    "intraday_range_raw",
]
RICHER_DAILY_STATE_TARGETS = [
    "ret_1d",
    "ret_3d",
    "ret_5d",
    "ret_10d",
    "ret_20d",
    "volatility_20d",
    "amount_ratio_5_20",
    "volume_ratio_5_20",
    "distance_to_20d_high",
    "distance_to_60d_high",
    "drawdown_from_20d_high",
    "range_1d",
    "body_to_range_1d",
    "upper_shadow_to_range_1d",
    "lower_shadow_to_range_1d",
    "open_gap_1d",
    "close_to_open_1d",
]
RICHER_INTRADAY_TARGETS = [
    "first_5m_ret",
    "opening_auction_ret",
    "opening_auction_amount_share",
    "first_30m_ret",
    "first_30m_amount_share",
    "last_5m_ret",
    "closing_auction_ret",
    "closing_auction_amount_share",
    "last_30m_ret",
    "last_30m_amount_share",
    "intraday_ret",
    "close_to_vwap",
    "intraday_range",
    "close_position",
    "intraday_realized_vol",
    "intraday_price_volume_corr",
    "high_time_frac",
    "low_time_frac",
    "high_before_low",
    "open_to_high_ret",
    "open_to_low_ret",
    "high_to_close_ret",
    "low_to_close_ret",
    "intraday_max_drawdown",
    "intraday_max_runup",
    "price_above_vwap_share",
    "cum_vwap_slope",
    "amount_top_bar_share",
    "amount_concentration_hhi",
    "am_ret",
    "pm_ret",
    "am_pm_ret_spread",
    "am_pm_vol_spread",
    "am_amount_share",
    "am_pm_amount_spread",
    "early_strength_late_weak",
    "close_pressure_30m",
]
RICHER_LIMIT_TARGETS = [
    "is_open_limit_up",
    "is_close_limit_up",
    "touch_limit_up",
    "is_one_word_limit_up",
    "opened_after_limit_up",
    "close_sealed_up",
    "limit_up_touch_minutes",
    "limit_up_close_minutes",
    "break_limit_up_count",
    "sealed_up_minutes_to_close",
    "limit_up_strength_score",
]


def _trim_working_set() -> None:
    if not hasattr(ctypes, "WinDLL"):
        return
    try:
        kernel32 = ctypes.WinDLL("kernel32.dll", use_last_error=True)
        psapi = ctypes.WinDLL("psapi.dll", use_last_error=True)
        kernel32.GetCurrentProcess.restype = ctypes.c_void_p
        psapi.EmptyWorkingSet.argtypes = [ctypes.c_void_p]
        psapi.EmptyWorkingSet.restype = ctypes.c_bool
        psapi.EmptyWorkingSet(kernel32.GetCurrentProcess())
    except Exception:
        return


def _now() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def _file_sha256(path: Path, *, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while chunk := handle.read(int(chunk_size)):
            digest.update(chunk)
    return digest.hexdigest()


def _parse_int_list(raw: str | None, *, default: tuple[int, ...]) -> tuple[int, ...]:
    if raw is None:
        return default
    values = [int(item.strip()) for item in str(raw).split(",") if item.strip()]
    return tuple(sorted(set(values))) or default


def _set_seed(seed: int) -> None:
    random.seed(int(seed))
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))


def _resolve_device(value: str) -> torch.device:
    raw = str(value or "auto").strip().lower()
    if raw == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if raw == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but torch.cuda.is_available() is False")
    if raw not in {"cpu", "cuda"}:
        raise ValueError("--device must be auto, cpu, or cuda")
    return torch.device(raw)


def _open_memmap(meta: Mapping[str, Any], *, dtype: str) -> np.memmap:
    path = Path(str(meta.get("path", "") or ""))
    shape = tuple(int(item) for item in list(meta.get("shape", []) or []))
    if not path.exists():
        raise FileNotFoundError(path)
    return np.memmap(path, dtype=dtype, mode="r", shape=shape)


class ShardedLabelReader:
    def __init__(self, meta: Mapping[str, Any], *, dtype: str) -> None:
        self.meta = dict(meta)
        self.shape = tuple(int(item) for item in list(meta.get("shape", []) or []))
        self.shards: list[dict[str, Any]] = []
        for shard in list(meta.get("shards", []) or []):
            path = Path(str(shard.get("path", "") or ""))
            shape = tuple(int(item) for item in list(shard.get("shape", []) or []))
            self.shards.append(
                {
                    "start": int(shard.get("date_start_idx", 0)),
                    "end": int(shard.get("date_end_idx", -1)),
                    "array": np.memmap(path, dtype=dtype, mode="r", shape=shape),
                }
            )
        self.shards.sort(key=lambda item: int(item["start"]))

    def take(self, date_idx: np.ndarray, symbol_idx: np.ndarray, *, field_slice: slice | None = None) -> np.ndarray:
        date_idx = np.asarray(date_idx, dtype=np.int64)
        symbol_idx = np.asarray(symbol_idx, dtype=np.int64)
        tail_shape = self.shape[2:] if field_slice is None else (*self.shape[2:-1], len(range(*field_slice.indices(self.shape[-1]))))
        out = np.empty((int(date_idx.size), *tail_shape), dtype=np.float32)
        for shard in self.shards:
            mask = (date_idx >= int(shard["start"])) & (date_idx <= int(shard["end"]))
            if not bool(mask.any()):
                continue
            local_date = date_idx[mask] - int(shard["start"])
            values = np.asarray(shard["array"][local_date, symbol_idx[mask], :, :], dtype=np.float32)
            if field_slice is not None:
                values = values[:, :, field_slice]
            out[np.flatnonzero(mask)] = values
        return out

    def __getitem__(self, key: Any) -> np.ndarray:
        date_idx, symbol_idx = key[0], key[1]
        field_slice = key[3] if len(key) > 3 and isinstance(key[3], slice) else None
        if np.isscalar(date_idx):
            return self.take(np.asarray([date_idx]), np.asarray([symbol_idx]), field_slice=field_slice)[0]
        return self.take(np.asarray(date_idx), np.asarray(symbol_idx), field_slice=field_slice)


def _open_label_array(meta: Mapping[str, Any], *, dtype: str) -> np.memmap | ShardedLabelReader:
    if list(meta.get("shards", []) or []):
        return ShardedLabelReader(meta, dtype=dtype)
    return _open_memmap(meta, dtype=dtype)


def _normalize_price_anchor(meta: Mapping[str, Any] | None) -> str:
    payload = dict(meta or {})
    explicit = str(payload.get("price_anchor", "") or "").strip()
    if explicit in {"next_open", "today_close"}:
        return explicit
    anchor = str(payload.get("anchor", "") or "").strip()
    if anchor in {"signal_day_close", "today_close"}:
        return "today_close"
    return "next_open"


def _input_channel_order(profile: str) -> list[str]:
    value = str(profile or INPUT_CHANNEL_PROFILE_ALL).strip().lower()
    if value not in INPUT_CHANNEL_PROFILE_ORDERS:
        raise ValueError(f"input_channel_profile must be one of {INPUT_CHANNEL_PROFILES}")
    return list(INPUT_CHANNEL_PROFILE_ORDERS[value])


def _validated_sample_date_indices(
    sample_index: pd.DataFrame,
    *,
    context: str,
    allow_empty: bool = False,
) -> np.ndarray:
    if not isinstance(sample_index, pd.DataFrame):
        raise ValueError(f"{context} must be a pandas DataFrame")
    if "date_idx" not in sample_index.columns:
        raise ValueError(f"{context} is missing required date_idx column")
    if sample_index.empty:
        if allow_empty:
            return np.empty(0, dtype=np.int64)
        raise ValueError(f"{context} is empty")
    numeric = pd.to_numeric(sample_index["date_idx"], errors="coerce").to_numpy(
        dtype=np.float64,
        na_value=np.nan,
    )
    if not bool(np.isfinite(numeric).all()) or not bool(np.equal(numeric, np.trunc(numeric)).all()):
        raise ValueError(f"{context} date_idx must contain only finite integers")
    if bool((numeric < 0).any()):
        raise ValueError(f"{context} date_idx must be non-negative")
    if bool((numeric > np.iinfo(np.int32).max).any()):
        raise ValueError(f"{context} date_idx exceeds the supported int32 range")
    return numeric.astype(np.int64)


DATE_COMPLETE_SPREAD_LIMIT_POLICY = "date_complete_even_spread_v1"


def _limit_sample_index_by_complete_dates(
    sample_index: pd.DataFrame,
    *,
    max_samples: int,
    context: str,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Limit training work without truncating a daily cross-section.

    Ranking metrics and global-tail slates require complete signal-date
    universes.  A row-level ``head(N)`` limit violates that invariant and also
    concentrates expanding-window folds in the earliest years.  This limiter
    therefore keeps whole dates selected at deterministic, evenly spaced
    positions over the available history.
    """

    frame = sample_index.reset_index(drop=True)
    requested = int(max_samples)
    original_rows = int(len(frame))
    original_dates = int(frame["date_idx"].nunique()) if original_rows else 0
    audit: dict[str, Any] = {
        "policy": "all_rows",
        "requested_max_samples": requested,
        "original_row_count": original_rows,
        "selected_row_count": original_rows,
        "original_date_count": original_dates,
        "selected_date_count": original_dates,
        "date_complete": True,
    }
    if original_rows:
        audit.update(
            {
                "selected_trade_date_start": str(frame["trade_date"].astype(str).min()),
                "selected_trade_date_end": str(frame["trade_date"].astype(str).max()),
            }
        )
    if requested <= 0 or original_rows <= requested:
        return frame, audit

    group_sizes = frame.groupby("date_idx", sort=True).size()
    if group_sizes.empty:
        return frame, audit
    largest_date_rows = int(group_sizes.max())
    if requested < largest_date_rows:
        raise ValueError(
            f"{context} max_samples={requested} is smaller than the largest complete date "
            f"cross-section ({largest_date_rows}); increase the limit or disable it"
        )

    date_values = group_sizes.index.to_numpy(dtype=np.int64, copy=True)
    sizes = group_sizes.to_numpy(dtype=np.int64, copy=True)
    upper_date_count = min(int(date_values.size), max(int(requested // max(int(sizes.min()), 1)), 1))
    selected_positions: np.ndarray | None = None
    for date_count in range(upper_date_count, 0, -1):
        if date_count == 1:
            positions = np.asarray([int(date_values.size // 2)], dtype=np.int64)
        else:
            positions = np.rint(np.linspace(0, int(date_values.size) - 1, num=date_count)).astype(np.int64)
            positions = np.unique(positions)
        if int(sizes[positions].sum()) <= requested:
            selected_positions = positions
            break
    if selected_positions is None or int(selected_positions.size) == 0:
        raise ValueError(f"{context} could not select a complete-date sample within max_samples={requested}")

    selected_dates = date_values[selected_positions]
    limited = frame.loc[frame["date_idx"].astype(np.int64).isin(selected_dates)].reset_index(drop=True)
    selected_group_sizes = limited.groupby("date_idx", sort=True).size()
    expected_group_sizes = group_sizes.loc[selected_group_sizes.index]
    if not selected_group_sizes.equals(expected_group_sizes):
        raise AssertionError(f"{context} date-complete sampling invariant failed")
    if int(len(limited)) > requested:
        raise AssertionError(f"{context} sample limit exceeded")

    audit.update(
        {
            "policy": DATE_COMPLETE_SPREAD_LIMIT_POLICY,
            "selected_row_count": int(len(limited)),
            "selected_date_count": int(selected_group_sizes.size),
            "selected_trade_date_start": str(limited["trade_date"].astype(str).min()),
            "selected_trade_date_end": str(limited["trade_date"].astype(str).max()),
            "selected_date_idx_start": int(selected_dates.min()),
            "selected_date_idx_end": int(selected_dates.max()),
            "largest_complete_date_row_count": largest_date_rows,
        }
    )
    return limited, audit


class SequencePathPackDataset(Dataset):
    def __init__(
        self,
        manifest: Mapping[str, Any],
        *,
        split: str,
        max_samples: int = 0,
        input_channel_profile: str = INPUT_CHANNEL_PROFILE_ALL,
        index_role: str = "supervised",
    ) -> None:
        self.manifest = dict(manifest)
        self.index_role = str(index_role or "supervised").strip().lower()
        if self.index_role not in {"supervised", "candidate"}:
            raise ValueError("index_role must be supervised or candidate")
        if self.index_role == "candidate" and int(max_samples) != 0:
            raise ValueError("candidate scoring requires max_samples=0")
        self.input_channel_profile = str(input_channel_profile or INPUT_CHANNEL_PROFILE_ALL).strip().lower()
        self.lookback_days = int(self.manifest.get("lookback_days", DEFAULT_LOOKBACK_DAYS) or DEFAULT_LOOKBACK_DAYS)
        self.forward_days = int(self.manifest.get("forward_days", DEFAULT_FORWARD_DAYS) or DEFAULT_FORWARD_DAYS)
        index_key = "candidate_index_path" if self.index_role == "candidate" else "sample_index_path"
        if not str(self.manifest.get(index_key, "") or ""):
            raise ValueError(f"manifest is missing required {index_key} for index_role={self.index_role}")
        self.index_path = Path(str(self.manifest[index_key])).resolve()
        sample_index = pd.read_parquet(str(self.index_path))
        required_index_columns = {"split", "date_idx", "symbol_idx", "trade_date", "symbol"}
        if self.index_role == "candidate":
            required_index_columns.update(
                {
                    "candidate_id",
                    "entry_filled",
                    "label_valid",
                    "price_label_valid",
                    "va_aux_valid",
                }
            )
        missing_index_columns = sorted(required_index_columns.difference(sample_index.columns))
        if missing_index_columns:
            raise ValueError(f"pack {self.index_role} index is missing required columns: {missing_index_columns}")
        split_index = sample_index[sample_index["split"].astype(str).eq(str(split))].reset_index(drop=True)
        self.sample_index, self.sample_selection = _limit_sample_index_by_complete_dates(
            split_index,
            max_samples=int(max_samples),
            context=f"{split} {self.index_role} index",
        )
        self.sample_selection.update(
            {
                "policy": "all_candidates" if self.index_role == "candidate" else self.sample_selection["policy"],
                "index_role": self.index_role,
                "index_path": str(self.index_path),
                "index_sha256": _file_sha256(self.index_path) if self.index_role == "candidate" else "",
            }
        )
        self.date_idx_values = _validated_sample_date_indices(
            self.sample_index,
            context=f"{split} sample index",
            allow_empty=True,
        ).astype(np.int32, copy=False)
        self.symbol_idx_values = self.sample_index["symbol_idx"].astype(np.int32).to_numpy(copy=True)
        self.label_symbol_idx_values = (
            self.sample_index["label_symbol_idx"].astype(np.int32).to_numpy(copy=True)
            if "label_symbol_idx" in self.sample_index.columns
            else self.symbol_idx_values.copy()
        )
        self.trade_date_values = self.sample_index["trade_date"].astype(str).to_numpy(copy=True)
        self.symbol_values = self.sample_index["symbol"].astype(str).to_numpy(copy=True)
        self.split = str(split)
        channels = dict(self.manifest.get("feature_channels", {}) or {})
        self.channel_order = _input_channel_order(self.input_channel_profile)
        self.feature_arrays = {name: _open_memmap(channels[name], dtype="float32") for name in self.channel_order}
        self.feature_columns = {name: list(channels[name].get("columns", []) or []) for name in self.channel_order}
        self.normalization = dict(self.manifest.get("normalization", {}) or {})
        self.normalization_mean: dict[str, np.ndarray] = {}
        self.normalization_std: dict[str, np.ndarray] = {}
        for name in self.channel_order:
            stats = dict(self.normalization.get(name, {}) or {})
            feature_count = len(self.feature_columns[name])
            self.normalization_mean[name] = np.asarray(
                stats.get("mean", [0.0] * feature_count),
                dtype=np.float32,
            )
            self.normalization_std[name] = np.maximum(
                np.asarray(stats.get("std", [1.0] * feature_count), dtype=np.float32),
                np.float32(1.0e-6),
            )
        labels = dict(self.manifest.get("label_arrays", {}) or {})
        masks = dict(self.manifest.get("masks", {}) or {})
        requested_input_masks = [*BASE_INPUT_MASK_FEATURES]
        if "intraday_summary" in self.channel_order:
            requested_input_masks.extend(INTRADAY_INPUT_MASK_FEATURES)
        self.input_mask_features = [name for name in requested_input_masks if name in masks]
        self.input_mask_arrays = {
            name: _open_memmap(masks[name], dtype="bool") for name in self.input_mask_features
        }
        self.tradable_panel = _open_memmap(masks["tradable"], dtype="bool") if "tradable" in masks else None
        self.exit_sellable_panel = (
            _open_memmap(masks["exit_sellable"], dtype="bool") if "exit_sellable" in masks else None
        )
        self.observed_price_panel = (
            _open_memmap(masks.get("price_observed", masks.get("has_bar")), dtype="bool")
            if ("price_observed" in masks or "has_bar" in masks)
            else None
        )
        self.future_ohlcva_path = _open_label_array(labels["future_ohlcva_path"], dtype="float32") if "future_ohlcva_path" in labels else None
        self.future_path = _open_label_array(labels["future_ohlc_path"], dtype="float32") if "future_ohlc_path" in labels else None
        if self.future_path is None and self.future_ohlcva_path is None:
            raise KeyError("pack label_arrays must include future_ohlc_path or future_ohlcva_path")
        label_meta = labels.get("future_ohlc_path") or labels.get("future_ohlcva_path") or {}
        self.price_anchor = _normalize_price_anchor(label_meta)
        self.has_ohlcva_path = self.future_ohlcva_path is not None
        self.ohlcva_path_fields = list(labels.get("future_ohlcva_path", {}).get("fields", []) or PATH_OHLCVA_FIELDS)
        output_path_dim = 6 if self.future_ohlcva_path is not None else 4
        self.path_summary = _open_memmap(labels["path_summary"], dtype="float32") if "path_summary" in labels else None
        self.path_summary_columns = (
            list(labels["path_summary"].get("columns", []) or path_summary_columns(self.forward_days))
            if "path_summary" in labels
            else derived_path_summary_columns(self.forward_days, path_dim=output_path_dim)
        )
        self.value_column = (
            path_value_column(self.forward_days)
            if "path_summary" in labels
            else value_column_for_path(self.forward_days, path_dim=output_path_dim)
        )
        self.value_index = self.path_summary_columns.index(self.value_column)
        self.input_dim = int(
            sum(len(self.feature_columns[name]) for name in self.channel_order) + len(self.input_mask_features)
        )
        requested_richer_targets = {
            "daily_raw": RICHER_DAILY_RAW_TARGETS,
            "daily_state": RICHER_DAILY_STATE_TARGETS,
            "intraday_summary": RICHER_INTRADAY_TARGETS,
            "limit_structure": RICHER_LIMIT_TARGETS,
        }
        self.richer_target_columns = {
            name: [col for col in requested if col in self.feature_columns.get(name, [])]
            for name, requested in requested_richer_targets.items()
        }
        self.richer_target_indices = {
            name: [self.feature_columns[name].index(col) for col in columns]
            for name, columns in self.richer_target_columns.items()
        }
        self.richer_path_fields = [
            "open_ret_from_entry_open",
            "high_ret_from_entry_open",
            "low_ret_from_entry_open",
            "close_ret_from_entry_open",
            *[
                f"{channel}:{column}"
                for channel in self.channel_order
                for column in self.richer_target_columns.get(channel, [])
            ],
        ]
        self.richer_path_dim = int(len(self.richer_path_fields))
        manifest_symbols = list(self.manifest.get("symbol_values", []) or [])
        self.symbol_count = int(len(manifest_symbols)) or int(self.sample_index["symbol_idx"].astype(int).max() + 1)
        execution_arrays = dict(self.manifest.get("execution_arrays", {}) or {})
        self.entry_open_raw_panel = (
            _open_memmap(execution_arrays["entry_open_raw"], dtype="float32")
            if "entry_open_raw" in execution_arrays
            else None
        )
        self.exit_close_raw_panel = (
            _open_memmap(execution_arrays["exit_close_raw"], dtype="float32")
            if "exit_close_raw" in execution_arrays
            else None
        )
        self.execution_tail_days = int(self.manifest.get("execution_tail_days", 0) or 0)
        terminal_contract = dict(self.manifest.get("terminal_execution_contract", {}) or {})
        self.terminal_recovery_fraction = float(
            terminal_contract.get("recovery_fraction_of_entry_notional", 0.0) or 0.0
        )
        self.has_deterministic_execution = bool(
            self.entry_open_raw_panel is not None
            and self.exit_close_raw_panel is not None
            and self.exit_sellable_panel is not None
            and self.execution_tail_days >= 0
        )
        self._path_value_targets_cache: np.ndarray | None = None

    def _index_flag(self, name: str, indices: np.ndarray, *, default: bool) -> np.ndarray:
        if name not in self.sample_index.columns:
            return np.full(int(indices.size), bool(default), dtype=bool)
        values = self.sample_index[name].iloc[indices].astype("boolean").fillna(False)
        return values.to_numpy(dtype=bool, copy=True)

    def __len__(self) -> int:
        return int(len(self.sample_index))

    def path_value_targets(self, *, chunk_size: int = 16384) -> np.ndarray:
        """Return the hard path-value-v2 target for every loaded sample.

        Corrected packs may persist this scalar in ``sample_index``.  Legacy packs
        fall back to a one-time chunked derivation from OHLC labels; the result is
        cached for all subsequent global-tail epochs.
        """

        if self._path_value_targets_cache is not None:
            return self._path_value_targets_cache
        for column in ("path_trade_value_v2_target", path_value_v2_column(self.forward_days)):
            if column in self.sample_index.columns:
                target = pd.to_numeric(self.sample_index[column], errors="coerce").to_numpy(dtype=np.float32, copy=True)
                self._path_value_targets_cache = target
                return target
        target = np.full(len(self), np.nan, dtype=np.float32)
        step = max(int(chunk_size), 1)
        for start in range(0, len(self), step):
            stop = min(start + step, len(self))
            date_idx = self.date_idx_values[start:stop].astype(np.int64, copy=False)
            label_symbol_idx = self.label_symbol_idx_values[start:stop].astype(np.int64, copy=False)
            path = (
                np.asarray(
                    self.future_path[date_idx, label_symbol_idx, : self.forward_days, :4],
                    dtype=np.float32,
                ).copy()
                if self.future_path is not None
                else np.asarray(
                    self.future_ohlcva_path[date_idx, label_symbol_idx, : self.forward_days, :4],
                    dtype=np.float32,
                ).copy()
            )
            tradable_path = self._future_mask_batch(
                self.tradable_panel,
                date_idx,
                self.symbol_idx_values[start:stop].astype(np.int64, copy=False),
            )
            target[start:stop] = _derive_path_summary_numpy(
                path,
                price_anchor=self.price_anchor,
                tradable_path=tradable_path,
            )[:, -1]
        self._path_value_targets_cache = target
        return target

    def _normalize(self, name: str, values: np.ndarray) -> np.ndarray:
        mean = self.normalization_mean[name]
        std = self.normalization_std[name]
        out = (values.astype(np.float32, copy=False) - mean.reshape(1, -1)) / std.reshape(1, -1)
        return np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32, copy=False)

    def _normalize_batch(self, name: str, values: np.ndarray) -> np.ndarray:
        mean = self.normalization_mean[name]
        std = self.normalization_std[name]
        out = (values.astype(np.float32, copy=False) - mean.reshape(1, 1, -1)) / std.reshape(1, 1, -1)
        return np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32, copy=False)

    def _normalize_selected_future_batch(self, name: str, values: np.ndarray, indices: list[int]) -> np.ndarray:
        mean_all = self.normalization_mean[name]
        std_all = self.normalization_std[name]
        mean = mean_all[np.asarray(indices, dtype=np.int64)]
        std = np.maximum(std_all[np.asarray(indices, dtype=np.int64)], 1.0e-6)
        return (values.astype(np.float32, copy=False) - mean.reshape(1, 1, -1)) / std.reshape(1, 1, -1)

    def _future_channel_values(self, name: str, date_idx: np.ndarray, symbol_idx: np.ndarray, columns: list[str]) -> np.ndarray:
        indices = self.richer_target_indices.get(name, [])
        values = np.full((int(date_idx.size), self.forward_days, len(columns)), np.nan, dtype=np.float32)
        if not indices:
            return values
        for current_date in np.unique(date_idx):
            mask = date_idx == int(current_date)
            symbols = symbol_idx[mask]
            start = int(current_date) + 1
            end = start + int(self.forward_days)
            available_end = min(end, int(self.feature_arrays[name].shape[0]))
            if start >= available_end:
                continue
            block = np.asarray(self.feature_arrays[name][start:available_end, symbols, :], dtype=np.float32)
            selected = np.transpose(block[:, :, indices], (1, 0, 2))
            values[np.flatnonzero(mask), : selected.shape[1], :] = selected
        return self._normalize_selected_future_batch(name, values, indices)

    def _future_richer_path_batch(self, y_path: np.ndarray, date_idx: np.ndarray, symbol_idx: np.ndarray) -> np.ndarray:
        parts = [np.asarray(y_path, dtype=np.float32)]
        for name in self.channel_order:
            columns = self.richer_target_columns.get(name, [])
            if columns:
                parts.append(self._future_channel_values(name, date_idx, symbol_idx, columns))
        out = np.concatenate(parts, axis=2).astype(np.float32, copy=False)
        return out

    def _future_mask_batch(
        self,
        panel: np.ndarray | None,
        date_idx: np.ndarray,
        symbol_idx: np.ndarray,
        *,
        days: int | None = None,
    ) -> np.ndarray | None:
        if panel is None:
            return None
        requested_days = int(self.forward_days if days is None else days)
        values = np.zeros((int(date_idx.size), requested_days), dtype=bool)
        for current_date in np.unique(date_idx):
            mask = date_idx == int(current_date)
            symbols = symbol_idx[mask]
            start = int(current_date) + 1
            end = start + requested_days
            block = np.asarray(panel[start:end, symbols], dtype=bool)
            values[np.flatnonzero(mask), : block.shape[0]] = np.transpose(block, (1, 0))
        return values

    def _future_float_panel_batch(
        self,
        panel: np.ndarray | None,
        date_idx: np.ndarray,
        symbol_idx: np.ndarray,
        *,
        days: int,
    ) -> np.ndarray | None:
        if panel is None:
            return None
        values = np.full((int(date_idx.size), int(days)), np.nan, dtype=np.float32)
        for current_date in np.unique(date_idx):
            mask = date_idx == int(current_date)
            symbols = symbol_idx[mask]
            start = int(current_date) + 1
            end = start + int(days)
            block = np.asarray(panel[start:end, symbols], dtype=np.float32)
            values[np.flatnonzero(mask), : block.shape[0]] = np.transpose(block, (1, 0))
        return values

    def __getitem__(self, idx: int) -> dict[str, Any]:
        row = self.sample_index.iloc[int(idx)]
        date_idx = int(row["date_idx"])
        symbol_idx = int(row["symbol_idx"])
        label_symbol_idx = int(row["label_symbol_idx"]) if "label_symbol_idx" in row.index else symbol_idx
        start = date_idx - self.lookback_days + 1
        end = date_idx + 1
        parts = [
            self._normalize(name, np.asarray(self.feature_arrays[name][start:end, symbol_idx, :], dtype=np.float32))
            for name in self.channel_order
        ]
        if self.input_mask_features:
            mask_values = np.stack(
                [
                    np.asarray(self.input_mask_arrays[name][start:end, symbol_idx], dtype=np.float32)
                    for name in self.input_mask_features
                ],
                axis=1,
            )
            parts.append(mask_values)
        x = np.concatenate(parts, axis=1).astype(np.float32, copy=False)
        y_path = (
            np.asarray(self.future_path[date_idx, label_symbol_idx, : self.forward_days, :4], dtype=np.float32).copy()
            if self.future_path is not None
            else np.asarray(self.future_ohlcva_path[date_idx, label_symbol_idx, : self.forward_days, :4], dtype=np.float32).copy()
        )
        y_ohlcva_path = (
            np.asarray(self.future_ohlcva_path[date_idx, label_symbol_idx, : self.forward_days, :], dtype=np.float32).copy()
            if self.future_ohlcva_path is not None
            else y_path.copy()
        )
        y_summary = (
            np.asarray(self.path_summary[date_idx, label_symbol_idx, :], dtype=np.float32).copy()
            if self.path_summary is not None
            else _derive_path_summary_numpy(y_ohlcva_path.reshape(1, self.forward_days, y_ohlcva_path.shape[-1]), price_anchor=self.price_anchor)[0]
        )
        y_richer_path = self._future_richer_path_batch(
            y_path.reshape(1, self.forward_days, 4),
            np.asarray([date_idx], dtype=np.int64),
            np.asarray([symbol_idx], dtype=np.int64),
        )[0]
        return {
            "x": torch.from_numpy(x),
            "y_path": torch.from_numpy(y_path),
            "y_ohlcva_path": torch.from_numpy(y_ohlcva_path),
            "y_richer_path": torch.from_numpy(y_richer_path),
            "y_summary": torch.from_numpy(y_summary),
            "date_idx": int(date_idx),
            "symbol_idx": int(symbol_idx),
            "trade_date": str(row["trade_date"]),
            "symbol": str(row["symbol"]),
        }

    def get_batch(
        self,
        indices: list[int] | np.ndarray,
        *,
        include_ohlcva_path: bool = True,
        include_richer_path: bool = True,
        include_summary: bool = True,
        include_metadata: bool = True,
    ) -> dict[str, Any]:
        idx = np.asarray(indices, dtype=np.int64)
        if idx.ndim != 1 or idx.size == 0:
            raise ValueError("batch indices must be a non-empty 1D array")
        date_idx = self.date_idx_values[idx].astype(np.int64, copy=False)
        symbol_idx = self.symbol_idx_values[idx].astype(np.int64, copy=False)
        label_symbol_idx = self.label_symbol_idx_values[idx].astype(np.int64, copy=False)
        batch_size = int(idx.size)
        channel_parts: list[np.ndarray] = []
        for name in self.channel_order:
            feature_count = len(self.feature_columns[name])
            values = np.empty((batch_size, self.lookback_days, feature_count), dtype=np.float32)
            for current_date in np.unique(date_idx):
                mask = date_idx == int(current_date)
                symbols = symbol_idx[mask]
                start = int(current_date) - self.lookback_days + 1
                end = int(current_date) + 1
                block = np.asarray(self.feature_arrays[name][start:end, symbols, :], dtype=np.float32)
                values[mask, :, :] = np.transpose(block, (1, 0, 2))
            channel_parts.append(self._normalize_batch(name, values))
        if self.input_mask_features:
            mask_values = np.empty(
                (batch_size, self.lookback_days, len(self.input_mask_features)),
                dtype=np.float32,
            )
            for current_date in np.unique(date_idx):
                mask = date_idx == int(current_date)
                symbols = symbol_idx[mask]
                start = int(current_date) - self.lookback_days + 1
                end = int(current_date) + 1
                for field_idx, name in enumerate(self.input_mask_features):
                    block = np.asarray(self.input_mask_arrays[name][start:end, symbols], dtype=np.float32)
                    mask_values[mask, :, field_idx] = np.transpose(block, (1, 0))
            channel_parts.append(mask_values)
        x = np.concatenate(channel_parts, axis=2).astype(np.float32, copy=False)
        y_path = (
            np.asarray(self.future_path[date_idx, label_symbol_idx, : self.forward_days, :4], dtype=np.float32).copy()
            if self.future_path is not None
            else np.asarray(self.future_ohlcva_path[date_idx, label_symbol_idx, : self.forward_days, :4], dtype=np.float32).copy()
        )
        y_ohlcva_path = None
        if bool(include_ohlcva_path):
            y_ohlcva_path = (
                np.asarray(self.future_ohlcva_path[date_idx, label_symbol_idx, : self.forward_days, :], dtype=np.float32).copy()
                if self.future_ohlcva_path is not None
                else y_path.copy()
            )
        label_valid = self._index_flag("label_valid", idx, default=True)
        price_label_valid = self._index_flag("price_label_valid", idx, default=True) & label_valid
        va_aux_valid = self._index_flag("va_aux_valid", idx, default=True) & label_valid
        if bool((~price_label_valid).any()):
            y_path[~price_label_valid, :, :] = np.nan
            if y_ohlcva_path is not None:
                y_ohlcva_path[~price_label_valid, :, :4] = np.nan
        if y_ohlcva_path is not None and int(y_ohlcva_path.shape[-1]) > 4 and bool((~va_aux_valid).any()):
            y_ohlcva_path[~va_aux_valid, :, 4:] = np.nan
        y_richer_path = self._future_richer_path_batch(y_path, date_idx, symbol_idx) if bool(include_richer_path) else None
        y_tradable_path = self._future_mask_batch(self.tradable_panel, date_idx, symbol_idx)
        y_observed_price_path = self._future_mask_batch(self.observed_price_panel, date_idx, symbol_idx)
        execution_days = int(self.forward_days + self.execution_tail_days)
        exit_sellable_path = self._future_mask_batch(
            self.exit_sellable_panel,
            date_idx,
            symbol_idx,
            days=execution_days,
        )
        exit_close_raw_path = self._future_float_panel_batch(
            self.exit_close_raw_panel,
            date_idx,
            symbol_idx,
            days=execution_days,
        )
        entry_open_raw = (
            np.asarray(self.entry_open_raw_panel[date_idx + 1, symbol_idx], dtype=np.float32).copy()
            if self.entry_open_raw_panel is not None
            else None
        )
        y_summary = None
        if bool(include_summary):
            y_summary = (
                np.asarray(self.path_summary[date_idx, label_symbol_idx, :], dtype=np.float32).copy()
                if self.path_summary is not None
                else _derive_path_summary_numpy(
                    y_ohlcva_path if y_ohlcva_path is not None else y_path,
                    price_anchor=self.price_anchor,
                )
            )
            if bool((~label_valid).any()):
                y_summary[~label_valid, :] = np.nan
        batch = {
            "x": torch.from_numpy(x),
            "y_path": torch.from_numpy(y_path),
            "date_idx": torch.from_numpy(date_idx.astype(np.int64, copy=False)),
            "symbol_idx": torch.from_numpy(symbol_idx.astype(np.int64, copy=False)),
        }
        if bool(include_metadata):
            batch["trade_date"] = [str(item) for item in self.trade_date_values[idx]]
            batch["symbol"] = [str(item) for item in self.symbol_values[idx]]
        batch["y_ohlcva_path"] = torch.from_numpy(y_ohlcva_path) if y_ohlcva_path is not None else None
        batch["y_richer_path"] = torch.from_numpy(y_richer_path) if y_richer_path is not None else None
        batch["y_summary"] = torch.from_numpy(y_summary) if y_summary is not None else None
        batch["y_tradable_path"] = torch.from_numpy(y_tradable_path) if y_tradable_path is not None else None
        batch["y_observed_price_path"] = (
            torch.from_numpy(y_observed_price_path) if y_observed_price_path is not None else None
        )
        batch["entry_open_raw"] = (
            torch.from_numpy(entry_open_raw) if entry_open_raw is not None else None
        )
        batch["exit_close_raw_path"] = (
            torch.from_numpy(exit_close_raw_path) if exit_close_raw_path is not None else None
        )
        batch["exit_sellable_path"] = (
            torch.from_numpy(exit_sellable_path) if exit_sellable_path is not None else None
        )
        entry_filled = (
            self.sample_index["entry_filled"].astype(bool).to_numpy(copy=False)[idx]
            if "entry_filled" in self.sample_index.columns
            else np.ones(int(idx.size), dtype=bool)
        )
        batch["entry_filled"] = torch.from_numpy(np.asarray(entry_filled, dtype=bool))
        batch["label_valid"] = torch.from_numpy(label_valid)
        batch["price_label_valid"] = torch.from_numpy(price_label_valid)
        batch["va_aux_valid"] = torch.from_numpy(va_aux_valid)
        return batch


class DateGroupedBatchSampler(BatchSampler):
    def __init__(self, sample_index: pd.DataFrame, *, batch_size: int, shuffle: bool, seed: int) -> None:
        self.batch_size = max(int(batch_size), 1)
        self.shuffle = bool(shuffle)
        self.seed = int(seed)
        self.epoch = 0
        groups: dict[int, list[int]] = {}
        for idx, date_idx in enumerate(sample_index["date_idx"].astype(int).to_numpy()):
            groups.setdefault(int(date_idx), []).append(int(idx))
        self.groups = groups
        self.date_indices = list(groups.keys())

    def _date_order(self, epoch: int) -> list[int]:
        dates = list(self.date_indices)
        if self.shuffle:
            rng = np.random.default_rng(self.seed + int(epoch))
            rng.shuffle(dates)
        return dates

    def _count_batches(self, dates: list[int]) -> int:
        count = 0
        pending_count = 0
        for date_idx in dates:
            item_count = len(self.groups[date_idx])
            full_chunks, remainder = divmod(item_count, self.batch_size)
            if full_chunks:
                if pending_count:
                    count += 1
                    pending_count = 0
                count += int(full_chunks)
            if remainder:
                if pending_count + remainder > self.batch_size:
                    count += 1
                    pending_count = 0
                pending_count += int(remainder)
                if pending_count == self.batch_size:
                    count += 1
                    pending_count = 0
        if pending_count:
            count += 1
        return int(count)

    def __iter__(self) -> Iterator[list[int]]:
        rng = np.random.default_rng(self.seed + self.epoch)
        dates = self._date_order(self.epoch)
        self.epoch += 1
        pending: list[int] = []
        for date_idx in dates:
            items = list(self.groups[date_idx])
            if self.shuffle:
                rng.shuffle(items)
            for start in range(0, len(items), self.batch_size):
                chunk = items[start : start + self.batch_size]
                if len(chunk) == self.batch_size:
                    if pending:
                        yield pending
                        pending = []
                    yield chunk
                    continue
                if pending and len(pending) + len(chunk) > self.batch_size:
                    yield pending
                    pending = []
                pending.extend(chunk)
        if pending:
            yield pending

    def __len__(self) -> int:
        return self._count_batches(self._date_order(self.epoch))


class ShuffledBatchSampler(BatchSampler):
    """Full-sample path batches without date-boundary remainder waste."""

    def __init__(self, sample_count: int, *, batch_size: int, shuffle: bool, seed: int) -> None:
        self.sample_count = max(int(sample_count), 0)
        self.batch_size = max(int(batch_size), 1)
        self.shuffle = bool(shuffle)
        self.seed = int(seed)
        self.epoch = 0

    def __iter__(self) -> Iterator[list[int]]:
        indices = np.arange(self.sample_count, dtype=np.int64)
        if self.shuffle:
            np.random.default_rng(self.seed + self.epoch).shuffle(indices)
        self.epoch += 1
        for start in range(0, self.sample_count, self.batch_size):
            yield indices[start : start + self.batch_size].tolist()

    def __len__(self) -> int:
        return int(math.ceil(self.sample_count / self.batch_size)) if self.sample_count else 0


class GlobalTailBatchSampler(BatchSampler):
    """One auditable target-stratified ranking slate per signal date."""

    def __init__(
        self,
        sample_index: pd.DataFrame,
        target_values: np.ndarray,
        *,
        batch_size: int = 512,
        shuffle: bool,
        seed: int,
        prior_epoch_scores: np.ndarray | None = None,
    ) -> None:
        if int(batch_size) != sum(GLOBAL_TAIL_GROUP_COUNTS.values()):
            raise ValueError("global_tail_512 requires batch_size=512 and the frozen 32/128/128/96/128 allocation")
        target = np.asarray(target_values, dtype=np.float32).reshape(-1)
        if len(sample_index) != int(target.size):
            raise ValueError("sample_index and target_values must have equal length")
        prior = None if prior_epoch_scores is None else np.asarray(prior_epoch_scores, dtype=np.float32).reshape(-1)
        if prior is not None and int(prior.size) != int(target.size):
            raise ValueError("prior_epoch_scores must match target_values")
        self.sample_index = sample_index
        self.target_values = target
        self.prior_epoch_scores = prior
        self.batch_size = int(batch_size)
        self.shuffle = bool(shuffle)
        self.seed = int(seed)
        self.epoch = 0
        groups: dict[int, np.ndarray] = {}
        for date_idx, raw_group in sample_index.groupby("date_idx", sort=True).groups.items():
            indices = np.asarray(list(raw_group), dtype=np.int64)
            finite = np.isfinite(target[indices])
            if bool(finite.any()):
                groups[int(date_idx)] = indices[finite]
        self.groups = groups
        self.date_indices = list(groups)

    @staticmethod
    def _sample(pool: np.ndarray, count: int, rng: np.random.Generator) -> np.ndarray:
        values = np.asarray(pool, dtype=np.int64)
        if int(values.size) <= int(count):
            return values.copy()
        return rng.choice(values, size=int(count), replace=False).astype(np.int64, copy=False)

    def build_slate(self, date_idx: int, *, epoch: int | None = None) -> tuple[list[int], dict[str, list[int]]]:
        current_epoch = self.epoch if epoch is None else int(epoch)
        rng = np.random.default_rng(self.seed + current_epoch * 1000003 + int(date_idx))
        indices = self.groups[int(date_idx)]
        ranked = indices[np.argsort(-self.target_values[indices], kind="mergesort")]
        if int(ranked.size) <= self.batch_size:
            values = ranked.tolist()
            return values, {"all_available": values}

        top = ranked[: GLOBAL_TAIL_GROUP_COUNTS["true_top"]]
        near_pool = ranked[GLOBAL_TAIL_GROUP_COUNTS["true_top"] : min(256, int(ranked.size))]
        bottom_pool = ranked[-GLOBAL_TAIL_GROUP_COUNTS["bottom"] :]
        body_end = max(256, int(ranked.size) - GLOBAL_TAIL_GROUP_COUNTS["bottom"])
        hard_pool = ranked[256:body_end]
        if self.prior_epoch_scores is not None:
            prior = self.prior_epoch_scores[hard_pool]
            finite_prior = np.isfinite(prior)
            hard_pool = hard_pool[finite_prior]
            prior = prior[finite_prior]
            hard = hard_pool[np.argsort(-prior, kind="mergesort")[: GLOBAL_TAIL_GROUP_COUNTS["prior_epoch_false_positive"]]]
        else:
            hard = self._sample(hard_pool, GLOBAL_TAIL_GROUP_COUNTS["prior_epoch_false_positive"], rng)
        near = self._sample(near_pool, GLOBAL_TAIL_GROUP_COUNTS["true_rank_33_256"], rng)
        bottom = self._sample(bottom_pool, GLOBAL_TAIL_GROUP_COUNTS["bottom"], rng)
        selected = set(np.concatenate([top, near, bottom, hard]).tolist())
        middle_pool = np.asarray([item for item in ranked[256:body_end] if int(item) not in selected], dtype=np.int64)
        middle = self._sample(middle_pool, GLOBAL_TAIL_GROUP_COUNTS["middle"], rng)
        categories = {
            "true_top": top.tolist(),
            "true_rank_33_256": near.tolist(),
            "middle": middle.tolist(),
            "bottom": bottom.tolist(),
            "prior_epoch_false_positive": hard.tolist(),
        }
        slate = [item for values in categories.values() for item in values]
        if len(slate) < self.batch_size:
            used = set(slate)
            fill_pool = np.asarray([item for item in ranked if int(item) not in used], dtype=np.int64)
            fill = self._sample(fill_pool, self.batch_size - len(slate), rng).tolist()
            categories["fallback_fill"] = fill
            slate.extend(fill)
        if len(slate) != self.batch_size or len(set(slate)) != len(slate):
            raise RuntimeError("global-tail slate must contain 512 unique samples")
        rng.shuffle(slate)
        return slate, categories

    def __iter__(self) -> Iterator[list[int]]:
        dates = list(self.date_indices)
        if self.shuffle:
            np.random.default_rng(self.seed + self.epoch).shuffle(dates)
        current_epoch = self.epoch
        self.epoch += 1
        for date_idx in dates:
            slate, _ = self.build_slate(date_idx, epoch=current_epoch)
            yield slate

    def __len__(self) -> int:
        return len(self.date_indices)


class SequencePathModel(nn.Module):
    def __init__(
        self,
        *,
        input_dim: int,
        hidden_dim: int,
        layers: int,
        forward_days: int,
        summary_dim: int,
        dropout: float,
        model_type: str = "gru_last",
        symbol_count: int = 0,
        symbol_embedding_dim: int = 16,
        richer_path_dim: int = 4,
    ) -> None:
        super().__init__()
        normalized_model_type = str(model_type or "gru_last").strip().lower()
        allowed_model_types = {
            "gru_last",
            "gru_attention",
            "gru_path_value",
            "gru_path_value_symbol",
            "gru_path_value_residual",
            "gru_ohlcva_path_value",
            "gru_ohlcva_aux_path_value",
            "gru_richer_path_value",
            "gru_richer_path_value_symbol",
            "gru_direct_value",
        }
        if normalized_model_type not in allowed_model_types:
            raise ValueError(
                "model_type must be gru_last, gru_attention, gru_path_value, gru_path_value_symbol, "
                "gru_path_value_residual, gru_ohlcva_path_value, gru_ohlcva_aux_path_value, gru_richer_path_value, "
                "gru_richer_path_value_symbol, or gru_direct_value"
            )
        self.model_type = normalized_model_type
        self.uses_derived_path_value = normalized_model_type in PATH_VALUE_MODEL_TYPES
        self.uses_direct_value = normalized_model_type in DIRECT_VALUE_MODEL_TYPES
        self.uses_symbol_embedding = normalized_model_type in {"gru_path_value_symbol", "gru_richer_path_value_symbol"}
        self.uses_residual_score = normalized_model_type in RESIDUAL_MODEL_TYPES
        self.uses_richer_path = normalized_model_type in RICHER_MODEL_TYPES
        self.uses_ohlcva_path = normalized_model_type in OHLCVA_MODEL_TYPES
        self.uses_ohlcva_aux_path = normalized_model_type in OHLCVA_AUX_MODEL_TYPES
        self.input_norm = nn.LayerNorm(input_dim)
        self.proj = nn.Linear(input_dim, hidden_dim)
        self.encoder = nn.GRU(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=max(int(layers), 1),
            batch_first=True,
            dropout=float(dropout) if int(layers) > 1 else 0.0,
        )
        self.dropout = nn.Dropout(float(dropout))
        self.attention = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1),
        )
        self.symbol_embedding_dim = int(symbol_embedding_dim) if self.uses_symbol_embedding else 0
        if self.uses_symbol_embedding:
            if int(symbol_count) <= 0:
                raise ValueError("symbol_count must be positive when model_type=gru_path_value_symbol")
            self.symbol_embedding = nn.Embedding(int(symbol_count), self.symbol_embedding_dim)
        else:
            self.symbol_embedding = None
        head_dim = int(hidden_dim) + self.symbol_embedding_dim
        self.path_dim = 6 if self.uses_ohlcva_path else 4
        self.richer_path_dim = (
            int(max(4, richer_path_dim))
            if self.uses_richer_path
            else 6
            if self.uses_ohlcva_aux_path
            else self.path_dim
        )
        self.path_head = None if self.uses_direct_value else nn.Linear(head_dim, int(forward_days) * self.richer_path_dim)
        if self.uses_derived_path_value:
            self.summary_head = None
            self.score_head = None
        elif self.uses_direct_value:
            self.summary_head = None
            self.score_head = nn.Linear(head_dim, 1)
        else:
            self.summary_head = nn.Linear(head_dim, int(summary_dim))
            self.score_head = nn.Linear(head_dim, 1)
        self.residual_score_head = nn.Linear(head_dim, 1) if self.uses_residual_score else None
        self.forward_days = int(forward_days)
        self.summary_dim = int(summary_dim)

    def forward(self, x: torch.Tensor, symbol_idx: torch.Tensor | None = None) -> dict[str, torch.Tensor]:
        z = self.input_norm(x)
        z = F.gelu(self.proj(z))
        encoded, _ = self.encoder(z)
        if self.model_type == "gru_attention":
            weights = torch.softmax(self.attention(encoded).squeeze(-1), dim=1)
            pooled = torch.sum(encoded * weights.unsqueeze(-1), dim=1)
        else:
            pooled = encoded[:, -1, :]
        pooled = self.dropout(pooled)
        if self.uses_symbol_embedding:
            if symbol_idx is None:
                raise ValueError("symbol_idx is required when model_type=gru_path_value_symbol")
            embedded = self.symbol_embedding(symbol_idx.to(device=pooled.device, dtype=torch.long))
            pooled = torch.cat([pooled, embedded], dim=1)
        if self.uses_direct_value:
            assert self.score_head is not None
            return {"score": self.score_head(pooled).squeeze(-1)}
        assert self.path_head is not None
        path_output = self.path_head(pooled).view(-1, self.forward_days, self.richer_path_dim)
        future_path = path_output[:, :, : self.path_dim] if (self.uses_richer_path or self.uses_ohlcva_aux_path) else path_output
        if self.uses_derived_path_value:
            outputs = {"future_path": future_path}
            if self.uses_richer_path:
                outputs["future_richer_path"] = path_output
            if self.uses_ohlcva_aux_path:
                outputs["future_ohlcva_aux_path"] = path_output
            if self.uses_residual_score:
                assert self.residual_score_head is not None
                outputs["residual_score"] = self.residual_score_head(pooled).squeeze(-1)
            return outputs
        assert self.summary_head is not None
        assert self.score_head is not None
        return {
            "future_path": future_path,
            "path_summary": self.summary_head(pooled).view(-1, self.summary_dim),
            "score": self.score_head(pooled).squeeze(-1),
        }


def _finite_smooth_l1(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    mask = torch.isfinite(target)
    safe_target = torch.where(mask, target, pred.detach())
    raw_loss = F.smooth_l1_loss(pred, safe_target, reduction="none")
    masked_loss = torch.where(mask, raw_loss, torch.zeros_like(raw_loss))
    count = mask.sum().to(dtype=pred.dtype)
    return masked_loss.sum() / torch.clamp(count, min=1.0)


def _finite_smooth_l1_fields_equal(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    if int(pred.shape[-1]) != int(target.shape[-1]):
        raise ValueError(f"field dim mismatch: predicted={pred.shape[-1]} target={target.shape[-1]}")
    mask = torch.isfinite(target)
    safe_target = torch.where(mask, target, pred.detach())
    raw_loss = F.smooth_l1_loss(pred, safe_target, reduction="none")
    masked_loss = torch.where(mask, raw_loss, torch.zeros_like(raw_loss))
    reduce_dims = tuple(range(int(target.ndim) - 1))
    counts = mask.sum(dim=reduce_dims).to(dtype=pred.dtype)
    field_losses = masked_loss.sum(dim=reduce_dims) / torch.clamp(counts, min=1.0)
    valid_fields = counts > 0
    return torch.where(valid_fields, field_losses, torch.zeros_like(field_losses)).sum() / torch.clamp(
        valid_fields.sum().to(dtype=pred.dtype),
        min=1.0,
    )


def _finite_smooth_l1_columns(pred: torch.Tensor, target: torch.Tensor, columns: list[int]) -> torch.Tensor:
    if not columns:
        return pred.sum() * 0.0
    index = torch.as_tensor(columns, device=pred.device, dtype=torch.long)
    return _finite_smooth_l1(pred.index_select(1, index), target.index_select(1, index))


def _normalize_path_value_gradient_profile(value: str) -> str:
    profile = str(value or PATH_VALUE_GRADIENT_PROFILE_SMOOTH).strip().lower()
    if profile not in PATH_VALUE_GRADIENT_PROFILES:
        raise ValueError(f"path_value_gradient_profile must be one of {PATH_VALUE_GRADIENT_PROFILES}")
    return profile


def _normalize_rank_training_profile(value: str) -> str:
    profile = str(value or RANK_TRAINING_PROFILE_LOCAL_CHUNK).strip().lower()
    if profile not in RANK_TRAINING_PROFILES:
        raise ValueError(f"rank_training_profile must be one of {RANK_TRAINING_PROFILES}")
    return profile


def _path_value_with_gradient_profile(
    hard_value: torch.Tensor,
    candidates: torch.Tensor,
    *,
    temperature: float,
    profile: str,
    reduction_dim: int,
) -> torch.Tensor:
    """Return the configured differentiable value without changing hard inference semantics.

    ``hard_st`` has the exact hard-max forward value used by NumPy inference, while its
    backward pass follows the temperature-smoothed log-sum-exp surrogate.  The legacy
    ``smooth_current`` profile remains available so existing evidence stays reproducible.
    """

    normalized = _normalize_path_value_gradient_profile(profile)
    scale = max(float(temperature), 1.0e-6)
    smooth_value = scale * torch.logsumexp(candidates / scale, dim=int(reduction_dim))
    if normalized == PATH_VALUE_GRADIENT_PROFILE_HARD_ST:
        return hard_value.detach() + smooth_value - smooth_value.detach()
    return smooth_value


def _loss_parts_payload(
    *,
    total: torch.Tensor,
    path_loss: torch.Tensor,
    summary_loss: torch.Tensor,
    richer_loss: torch.Tensor,
    price_delta_loss: torch.Tensor,
    va_level_loss: torch.Tensor,
    va_delta_loss: torch.Tensor,
    value_loss: torch.Tensor,
    rank_loss: torch.Tensor,
    residual_penalty: torch.Tensor,
    return_tensors: bool,
) -> dict[str, float] | dict[str, torch.Tensor]:
    tensors = {
        "loss": total,
        "path_loss": path_loss,
        "summary_loss": summary_loss,
        "richer_loss": richer_loss,
        "price_delta_loss": price_delta_loss,
        "va_level_loss": va_level_loss,
        "va_delta_loss": va_delta_loss,
        "value_loss": value_loss,
        "rank_loss": rank_loss,
        "residual_penalty": residual_penalty,
    }
    if bool(return_tensors):
        return {key: value.detach() for key, value in tensors.items()}
    return {key: float(value.detach().cpu().item()) for key, value in tensors.items()}


def _rank_loss_by_date(score: torch.Tensor, target: torch.Tensor, date_idx: torch.Tensor, *, max_per_side: int = 64) -> torch.Tensor:
    losses: list[torch.Tensor] = []
    for date in torch.unique(date_idx):
        mask = (date_idx == date) & torch.isfinite(target)
        if int(mask.sum().item()) < 4:
            continue
        s = score[mask]
        y = target[mask]
        order = torch.argsort(y)
        n = int(order.numel())
        side = min(int(max_per_side), n // 2)
        low = order[:side]
        high = order[-side:]
        diff = s[high].view(-1, 1) - s[low].view(1, -1)
        losses.append(F.softplus(-diff).mean())
    if not losses:
        return score.sum() * 0.0
    return torch.stack(losses).mean()


def _global_tail_rank_loss_by_date(
    score: torch.Tensor,
    target: torch.Tensor,
    date_idx: torch.Tensor,
    *,
    top_count: int = 32,
) -> torch.Tensor:
    """Target-gap weighted pairwise loss over each global-tail daily slate."""

    losses: list[torch.Tensor] = []
    for date in torch.unique(date_idx):
        mask = (date_idx == date) & torch.isfinite(target) & torch.isfinite(score)
        valid_indices = torch.where(mask)[0]
        if int(valid_indices.numel()) < 4:
            continue
        s = score.index_select(0, valid_indices)
        y = target.index_select(0, valid_indices)
        order = torch.argsort(y, descending=True)
        s = s.index_select(0, order)
        y = y.index_select(0, order)
        target_gap = y.view(-1, 1) - y.view(1, -1)
        score_gap = s.view(-1, 1) - s.view(1, -1)
        ordered_pair = target_gap > 0.0
        abs_gap = torch.where(ordered_pair, target_gap, torch.zeros_like(target_gap))
        positive_count = torch.clamp(ordered_pair.sum().to(dtype=s.dtype), min=1.0)
        gap_scale = torch.clamp(abs_gap.sum() / positive_count, min=1.0e-6)
        gap_weight = torch.clamp(abs_gap / gap_scale, min=0.25, max=4.0)
        top = min(int(top_count), int(s.numel()))
        top_pair = torch.zeros_like(ordered_pair)
        top_pair[:top, :] = True
        pair_weight = gap_weight * torch.where(
            top_pair,
            torch.full_like(target_gap, 2.0),
            torch.ones_like(target_gap),
        )
        raw = F.softplus(-score_gap) * pair_weight
        losses.append(torch.where(ordered_pair, raw, torch.zeros_like(raw)).sum() / positive_count)
    if not losses:
        return score.sum() * 0.0
    return torch.stack(losses).mean()


def _rank_loss_for_profile(
    score: torch.Tensor,
    target: torch.Tensor,
    date_idx: torch.Tensor,
    *,
    profile: str,
    max_per_side: int,
) -> torch.Tensor:
    normalized = _normalize_rank_training_profile(profile)
    if normalized == RANK_TRAINING_PROFILE_GLOBAL_TAIL_512:
        return _global_tail_rank_loss_by_date(score, target, date_idx)
    return _rank_loss_by_date(score, target, date_idx, max_per_side=max_per_side)


def _derived_path_rank_score_and_target(
    outputs: Mapping[str, torch.Tensor],
    target_path: torch.Tensor,
    *,
    price_anchor: str,
    path_value_gradient_profile: str,
    target_tradable_path: torch.Tensor | None = None,
    residual_weight: float = 0.25,
) -> tuple[torch.Tensor, torch.Tensor]:
    if "future_path" not in outputs:
        raise ValueError("separated path ranking requires future_path output")
    if int(outputs["future_path"].shape[-1]) != 4:
        raise ValueError("global-tail ranking requires price-only OHLC path value, not unified OHLCVA value semantics")
    target_summary = _derive_path_summary_torch(
        target_path[:, :, :4],
        smooth_value=False,
        price_anchor=price_anchor,
        tradable_path=target_tradable_path,
    ).detach()
    predicted_summary = _derive_path_summary_torch(
        outputs["future_path"],
        smooth_value=True,
        price_anchor=price_anchor,
        path_value_gradient_profile=path_value_gradient_profile,
    )
    score = predicted_summary[:, -1]
    if "residual_score" in outputs:
        score = score + float(residual_weight) * outputs["residual_score"]
    return score, target_summary[:, -1]


def _delta_along_days(path: torch.Tensor) -> torch.Tensor:
    if int(path.shape[1]) < 2:
        return path[:, :0]
    return path[:, 1:] - path[:, :-1]


def _close_log_delta_from_anchor(path: torch.Tensor) -> torch.Tensor:
    close_ret = path[:, :, 3]
    close_log_level = torch.log(torch.clamp(1.0 + close_ret, min=1.0e-6))
    anchor = torch.zeros((int(close_log_level.shape[0]), 1), device=path.device, dtype=path.dtype)
    return torch.cat([close_log_level[:, :1] - anchor, close_log_level[:, 1:] - close_log_level[:, :-1]], dim=1)


def _multi_horizon_ohlc_summary_loss_loop(pred_path: torch.Tensor, target_path: torch.Tensor, *, price_anchor: str) -> torch.Tensor:
    losses: list[torch.Tensor] = []
    for window in _summary_loss_windows(int(pred_path.shape[1])):
        target_summary = _derive_path_summary_torch(
            target_path[:, :window, :4],
            smooth_value=False,
            price_anchor=price_anchor,
        ).detach()
        pred_summary = _derive_path_summary_torch(
            pred_path[:, :window, :4],
            smooth_value=True,
            price_anchor=price_anchor,
        )
        losses.append(
            _finite_smooth_l1_columns(
                pred_summary,
                target_summary,
                _derived_summary_loss_indices(int(window), path_dim=4),
            )
        )
    if not losses:
        return pred_path.sum() * 0.0
    return torch.stack(losses).mean()


def _multi_horizon_ohlc_summary_features(
    path: torch.Tensor,
    *,
    price_anchor: str,
    include_full_horizon: bool = True,
    tradable_path: torch.Tensor | None = None,
) -> torch.Tensor:
    path = _legacy_entry_relative_path_torch(path[:, :, :4], price_anchor=price_anchor)
    batch_size = int(path.shape[0])
    forward_days = int(path.shape[1])
    windows = _summary_loss_windows(forward_days, include_full_horizon=bool(include_full_horizon))
    if not windows:
        return path.new_empty((batch_size, 0, 6))
    horizon_idx = torch.as_tensor([window - 1 for window in windows], device=path.device, dtype=torch.long)
    horizon_values = horizon_idx.to(dtype=path.dtype) + 1.0
    day_idx = torch.arange(forward_days, device=path.device, dtype=torch.long)
    day_float = day_idx.to(dtype=path.dtype) + 1.0

    high_ret = path[:, :, 1]
    low_ret = path[:, :, 2]
    close_ret = path[:, :, 3]

    max_ret = torch.cummax(high_ret, dim=1).values.index_select(1, horizon_idx)
    min_ret = torch.cummin(low_ret, dim=1).values.index_select(1, horizon_idx)
    final_ret = close_ret.index_select(1, horizon_idx)

    valid_by_horizon = day_idx.view(1, forward_days, 1) <= horizon_idx.view(1, 1, -1)
    high_for_peak = torch.where(
        valid_by_horizon,
        high_ret.unsqueeze(2),
        torch.full((batch_size, forward_days, len(windows)), float("-inf"), device=path.device, dtype=path.dtype),
    )
    peak_idx = torch.argmax(high_for_peak, dim=1)
    after_peak = day_idx.view(1, forward_days, 1) >= peak_idx.view(batch_size, 1, len(windows))
    low_after_peak = torch.where(
        valid_by_horizon & after_peak,
        low_ret.unsqueeze(2),
        torch.full((batch_size, forward_days, len(windows)), float("inf"), device=path.device, dtype=path.dtype),
    )
    min_after_peak = torch.min(low_after_peak, dim=1).values
    drawdown_after_peak = (1.0 + min_after_peak) / torch.clamp(1.0 + max_ret, min=1.0e-6) - 1.0

    pre_exit_drawdown = torch.clamp(-torch.cummin(low_ret, dim=1).values, min=0.0)
    waiting = torch.sqrt(day_float.view(forward_days, 1) / torch.clamp(horizon_values.view(1, -1), min=1.0))
    candidate = (
        close_ret.unsqueeze(2)
        - float(PATH_VALUE_V2_DRAWDOWN_PENALTY) * pre_exit_drawdown.unsqueeze(2)
        - float(PATH_VALUE_V2_WAITING_PENALTY) * waiting.unsqueeze(0)
        - float(PATH_VALUE_V2_TRANSACTION_COST)
    )
    candidate_valid = valid_by_horizon
    if tradable_path is not None:
        if tuple(tradable_path.shape) != (batch_size, forward_days):
            raise ValueError(f"tradable_path must have shape {(batch_size, forward_days)}")
        candidate_valid = candidate_valid & tradable_path.to(dtype=torch.bool).unsqueeze(2)
    candidate = torch.where(candidate_valid, candidate, torch.full_like(candidate, float("-inf")))
    best_idx = torch.argmax(candidate, dim=1)
    best_exit_close = torch.gather(close_ret, 1, best_idx)
    best_pre_exit_drawdown = torch.gather(pre_exit_drawdown, 1, best_idx)
    has_tradable_exit = candidate_valid.any(dim=1)
    best_exit_close = torch.where(has_tradable_exit, best_exit_close, torch.full_like(best_exit_close, float("nan")))
    best_pre_exit_drawdown = torch.where(
        has_tradable_exit,
        best_pre_exit_drawdown,
        torch.full_like(best_pre_exit_drawdown, float("nan")),
    )

    return torch.stack(
        [
            max_ret,
            min_ret,
            final_ret,
            drawdown_after_peak,
            best_exit_close,
            best_pre_exit_drawdown,
        ],
        dim=2,
    )


def _multi_horizon_ohlc_summary_loss_vectorized(
    pred_path: torch.Tensor,
    target_path: torch.Tensor,
    *,
    price_anchor: str,
    include_full_horizon: bool = True,
    target_tradable_path: torch.Tensor | None = None,
) -> torch.Tensor:
    target_features = _multi_horizon_ohlc_summary_features(
        target_path,
        price_anchor=price_anchor,
        include_full_horizon=bool(include_full_horizon),
        tradable_path=target_tradable_path,
    ).detach()
    pred_features = _multi_horizon_ohlc_summary_features(
        pred_path,
        price_anchor=price_anchor,
        include_full_horizon=bool(include_full_horizon),
    )
    if int(pred_features.shape[1]) == 0:
        return pred_path.sum() * 0.0
    finite_target = torch.isfinite(target_features)
    # Feeding NaN targets into smooth_l1_loss and masking the resulting loss is
    # not sufficient: autograd can still propagate NaN through the zeroed branch.
    # Substitute detached predictions before evaluating the loss so unavailable
    # exit fields contribute exactly zero value and zero (finite) gradient.
    safe_target = torch.where(finite_target, target_features, pred_features.detach())
    raw_loss = F.smooth_l1_loss(pred_features, safe_target, reduction="none")
    masked_loss = torch.where(finite_target, raw_loss, torch.zeros_like(raw_loss))
    horizon_counts = finite_target.sum(dim=(0, 2))
    horizon_loss = torch.where(
        horizon_counts > 0,
        masked_loss.sum(dim=(0, 2)) / torch.clamp(horizon_counts.to(dtype=pred_features.dtype), min=1.0),
        torch.zeros_like(horizon_counts, dtype=pred_features.dtype),
    )
    return horizon_loss.mean()


def _derived_summary_loss(
    pred_path: torch.Tensor,
    target_path: torch.Tensor,
    *,
    summary_loss_profile: str,
    price_anchor: str,
    path_value_gradient_profile: str = PATH_VALUE_GRADIENT_PROFILE_SMOOTH,
    target_tradable_path: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    profile = str(summary_loss_profile or SUMMARY_LOSS_PROFILE_BASE).strip().lower()
    if profile not in SUMMARY_LOSS_PROFILES:
        raise ValueError(f"summary_loss_profile must be one of {SUMMARY_LOSS_PROFILES}")
    path_dim = int(pred_path.shape[2])

    if profile == SUMMARY_LOSS_PROFILE_MULTI_HORIZON_OHLC and path_dim == 4:
        summary_loss = _multi_horizon_ohlc_summary_loss_vectorized(
            pred_path,
            target_path,
            price_anchor=price_anchor,
            target_tradable_path=target_tradable_path,
        )
        target_summary = _derive_path_summary_torch(
            target_path,
            smooth_value=False,
            price_anchor=price_anchor,
            tradable_path=target_tradable_path,
        ).detach()
        pred_summary = _derive_path_summary_torch(
            pred_path,
            smooth_value=True,
            price_anchor=price_anchor,
            path_value_gradient_profile=path_value_gradient_profile,
        )
        return summary_loss, target_summary, pred_summary

    if profile == SUMMARY_LOSS_PROFILE_MULTI_HORIZON_OHLC_NO60 and path_dim == 4:
        summary_loss = _multi_horizon_ohlc_summary_loss_vectorized(
            pred_path,
            target_path,
            price_anchor=price_anchor,
            include_full_horizon=False,
            target_tradable_path=target_tradable_path,
        )
        target_summary = _derive_path_summary_torch(
            target_path,
            smooth_value=False,
            price_anchor=price_anchor,
            tradable_path=target_tradable_path,
        ).detach()
        pred_summary = _derive_path_summary_torch(
            pred_path,
            smooth_value=True,
            price_anchor=price_anchor,
            path_value_gradient_profile=path_value_gradient_profile,
        )
        return summary_loss, target_summary, pred_summary

    target_summary = _derive_path_summary_torch(
        target_path,
        smooth_value=False,
        price_anchor=price_anchor,
        tradable_path=target_tradable_path,
    ).detach()
    pred_summary = _derive_path_summary_torch(
        pred_path,
        smooth_value=True,
        price_anchor=price_anchor,
        path_value_gradient_profile=path_value_gradient_profile,
    )
    summary_loss = _finite_smooth_l1_columns(
        pred_summary,
        target_summary,
        _derived_summary_loss_indices(int(pred_path.shape[1]), path_dim=path_dim),
    )
    return summary_loss, target_summary, pred_summary


def _compute_loss(
    outputs: Mapping[str, torch.Tensor],
    y_path: torch.Tensor,
    y_summary: torch.Tensor,
    date_idx: torch.Tensor,
    *,
    y_ohlcva_path: torch.Tensor | None = None,
    y_richer_path: torch.Tensor | None = None,
    value_index: int,
    path_weight: float = 0.35,
    summary_weight: float = 0.35,
    value_weight: float = 0.15,
    rank_weight: float = 0.15,
    richer_weight: float = 0.0,
    rank_max_per_side: int = 64,
    price_delta_weight: float = 0.0,
    va_level_weight: float = 0.0,
    va_delta_weight: float = 0.0,
    residual_weight: float = 0.25,
    residual_penalty_weight: float = 0.01,
    price_anchor: str = "next_open",
    summary_loss_profile: str = SUMMARY_LOSS_PROFILE_BASE,
    path_loss_profile: str = PATH_LOSS_PROFILE_DEFAULT,
    direct_value_horizon: int = 0,
    path_value_gradient_profile: str = PATH_VALUE_GRADIENT_PROFILE_SMOOTH,
    rank_training_profile: str = RANK_TRAINING_PROFILE_LOCAL_CHUNK,
    y_tradable_path: torch.Tensor | None = None,
    return_tensor_parts: bool = False,
) -> tuple[torch.Tensor, dict[str, float] | dict[str, torch.Tensor]]:
    normalized_path_loss_profile = str(path_loss_profile or PATH_LOSS_PROFILE_DEFAULT).strip().lower()
    if normalized_path_loss_profile not in PATH_LOSS_PROFILES:
        raise ValueError(f"path_loss_profile must be one of {PATH_LOSS_PROFILES}")
    if "future_path" not in outputs and "score" in outputs:
        score = outputs["score"]
        horizon = int(direct_value_horizon) if int(direct_value_horizon) > 0 else int(y_path.shape[1])
        horizon = max(1, min(horizon, int(y_path.shape[1])))
        target_summary = _derive_path_summary_torch(
            y_path[:, :horizon, :4],
            smooth_value=False,
            price_anchor=price_anchor,
            tradable_path=(y_tradable_path[:, :horizon] if y_tradable_path is not None else None),
        ).detach()
        value_column = value_column_for_path(horizon, path_dim=4)
        value_index = derived_path_summary_columns(horizon, path_dim=4).index(value_column)
        value_target = target_summary[:, int(value_index)]
        path_loss = score.sum() * 0.0
        summary_loss = score.sum() * 0.0
        richer_loss = score.sum() * 0.0
        price_delta_loss = score.sum() * 0.0
        va_level_loss = score.sum() * 0.0
        va_delta_loss = score.sum() * 0.0
        residual_penalty = score.sum() * 0.0
        value_loss = _finite_smooth_l1(score, value_target)
        rank_loss = (
            _rank_loss_for_profile(
                score,
                value_target,
                date_idx,
                profile=rank_training_profile,
                max_per_side=int(rank_max_per_side),
            )
            if float(rank_weight) != 0.0
            else score.sum() * 0.0
        )
        total = (
            float(value_weight) * value_loss
            + float(rank_weight) * rank_loss
            + float(path_weight) * path_loss
            + float(summary_weight) * summary_loss
            + float(richer_weight) * richer_loss
            + float(price_delta_weight) * price_delta_loss
            + float(va_level_weight) * va_level_loss
            + float(va_delta_weight) * va_delta_loss
            + float(residual_penalty_weight) * residual_penalty
        )
        parts = _loss_parts_payload(
            total=total,
            path_loss=path_loss,
            summary_loss=summary_loss,
            richer_loss=richer_loss,
            price_delta_loss=price_delta_loss,
            va_level_loss=va_level_loss,
            va_delta_loss=va_delta_loss,
            value_loss=value_loss,
            rank_loss=rank_loss,
            residual_penalty=residual_penalty,
            return_tensors=return_tensor_parts,
        )
        if bool(return_tensor_parts):
            parts["_ranking_score"] = score.detach()
        return total, parts

    if normalized_path_loss_profile == PATH_LOSS_PROFILE_OHLCVA_EQUAL:
        if y_ohlcva_path is None:
            raise ValueError("path_loss_profile=ohlcva_equal requires y_ohlcva_path")
        predicted_ohlcva = outputs.get("future_ohlcva_aux_path", outputs["future_path"])
        if int(predicted_ohlcva.shape[-1]) < 6 or int(y_ohlcva_path.shape[-1]) < 6:
            raise ValueError("path_loss_profile=ohlcva_equal requires 6-dimensional OHLCVA predictions and targets")
        path_loss = _finite_smooth_l1_fields_equal(predicted_ohlcva[:, :, :6], y_ohlcva_path[:, :, :6])
    else:
        path_loss = _finite_smooth_l1(outputs["future_path"], y_path)
    price_delta_loss = outputs["future_path"].sum() * 0.0
    if float(price_delta_weight) != 0.0:
        price_delta_loss = _finite_smooth_l1(
            _close_log_delta_from_anchor(outputs["future_path"]),
            _close_log_delta_from_anchor(y_path),
        )
    va_level_loss = outputs["future_path"].sum() * 0.0
    va_delta_loss = outputs["future_path"].sum() * 0.0
    need_va_diagnostics = bool(
        float(va_level_weight) != 0.0
        or float(va_delta_weight) != 0.0
        or normalized_path_loss_profile == PATH_LOSS_PROFILE_OHLCVA_EQUAL
    )
    if need_va_diagnostics and y_ohlcva_path is not None and "future_ohlcva_aux_path" in outputs:
        predicted_aux = outputs["future_ohlcva_aux_path"]
        if int(predicted_aux.shape[-1]) < 6 or int(y_ohlcva_path.shape[-1]) < 6:
            raise ValueError("future_ohlcva_aux_path requires 6-dimensional OHLCVA targets")
        va_level_loss = _finite_smooth_l1(predicted_aux[:, :, 4:6], y_ohlcva_path[:, :, 4:6])
        va_delta_loss = _finite_smooth_l1(
            _delta_along_days(predicted_aux[:, :, 4:6]),
            _delta_along_days(y_ohlcva_path[:, :, 4:6]),
        )
    richer_loss = outputs["future_path"].sum() * 0.0
    if float(richer_weight) != 0.0 and y_richer_path is not None and "future_richer_path" in outputs:
        predicted_richer = outputs["future_richer_path"]
        if int(predicted_richer.shape[-1]) != int(y_richer_path.shape[-1]):
            raise ValueError(f"richer path dim mismatch: predicted={predicted_richer.shape[-1]} target={y_richer_path.shape[-1]}")
        if int(predicted_richer.shape[-1]) > 4:
            richer_loss = _finite_smooth_l1(predicted_richer[:, :, 4:], y_richer_path[:, :, 4:])
        else:
            richer_loss = _finite_smooth_l1(predicted_richer, y_richer_path)
    residual_penalty = outputs["future_path"].sum() * 0.0
    if "score" in outputs:
        summary_loss = _finite_smooth_l1(outputs["path_summary"], y_summary)
        value_target = y_summary[:, int(value_index)]
        score = outputs["score"]
    else:
        summary_loss, target_summary, pred_summary = _derived_summary_loss(
            outputs["future_path"],
            y_path,
            summary_loss_profile=summary_loss_profile,
            price_anchor=price_anchor,
            path_value_gradient_profile=path_value_gradient_profile,
            target_tradable_path=y_tradable_path,
        )
        value_target = target_summary[:, -1]
        path_value_score = pred_summary[:, -1]
        path_value_loss = _finite_smooth_l1(path_value_score, value_target)
        if "residual_score" in outputs:
            residual_score = outputs["residual_score"]
            score = path_value_score + float(residual_weight) * residual_score
            residual_penalty = torch.mean(torch.square(residual_score))
            final_value_loss = _finite_smooth_l1(score, value_target)
            value_loss = 0.5 * (path_value_loss + final_value_loss)
        else:
            score = path_value_score
            value_loss = path_value_loss
    if "score" in outputs:
        value_loss = _finite_smooth_l1(score, value_target)
    rank_loss = (
        _rank_loss_for_profile(
            score,
            value_target,
            date_idx,
            profile=rank_training_profile,
            max_per_side=int(rank_max_per_side),
        )
        if float(rank_weight) != 0.0
        else score.sum() * 0.0
    )
    total = (
        float(path_weight) * path_loss
        + float(summary_weight) * summary_loss
        + float(value_weight) * value_loss
        + float(rank_weight) * rank_loss
        + float(richer_weight) * richer_loss
        + float(price_delta_weight) * price_delta_loss
        + float(va_level_weight) * va_level_loss
        + float(va_delta_weight) * va_delta_loss
        + float(residual_penalty_weight) * residual_penalty
    )
    parts = _loss_parts_payload(
        total=total,
        path_loss=path_loss,
        summary_loss=summary_loss,
        richer_loss=richer_loss,
        price_delta_loss=price_delta_loss,
        va_level_loss=va_level_loss,
        va_delta_loss=va_delta_loss,
        value_loss=value_loss,
        rank_loss=rank_loss,
        residual_penalty=residual_penalty,
        return_tensors=return_tensor_parts,
    )
    if bool(return_tensor_parts):
        parts["_ranking_score"] = score.detach()
    return total, parts


def _batch_to_device(
    batch: Mapping[str, Any], device: torch.device
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor | None,
    torch.Tensor | None,
    torch.Tensor | None,
    torch.Tensor,
    torch.Tensor,
]:
    x = batch["x"].to(device, non_blocking=device.type == "cuda")
    y_path = batch["y_path"].to(device, non_blocking=device.type == "cuda")
    y_ohlcva_path = (
        batch["y_ohlcva_path"].to(device, non_blocking=device.type == "cuda")
        if batch.get("y_ohlcva_path") is not None
        else None
    )
    y_richer_path = (
        batch["y_richer_path"].to(device, non_blocking=device.type == "cuda")
        if batch.get("y_richer_path") is not None
        else None
    )
    y_summary = (
        batch["y_summary"].to(device, non_blocking=device.type == "cuda")
        if batch.get("y_summary") is not None
        else None
    )
    date_idx = batch["date_idx"].to(device, non_blocking=device.type == "cuda")
    symbol_idx = batch["symbol_idx"].to(device, non_blocking=device.type == "cuda")
    return x, y_path, y_ohlcva_path, y_richer_path, y_summary, date_idx, symbol_idx


def _iter_index_batches(dataset: SequencePathPackDataset, *, batch_size: int, shuffle: bool, seed: int) -> Iterator[list[int]]:
    sampler = DateGroupedBatchSampler(dataset.sample_index, batch_size=int(batch_size), shuffle=bool(shuffle), seed=int(seed))
    yield from sampler


def _pin_tensor_batch(batch: dict[str, Any]) -> dict[str, Any]:
    for key, value in batch.items():
        if isinstance(value, torch.Tensor) and value.device.type == "cpu":
            batch[key] = value.pin_memory()
    return batch


def _iter_prefetched_batches(
    dataset: SequencePathPackDataset,
    index_batches: Iterator[list[int]] | BatchSampler,
    *,
    get_batch_kwargs: Mapping[str, Any],
    prefetch_batches: int,
    pin_memory: bool,
) -> Iterator[tuple[list[int], dict[str, Any]]]:
    """Overlap one CPU memmap/normalization gather with the active GPU step."""

    iterator = iter(index_batches)

    def load(indices: list[int]) -> tuple[list[int], dict[str, Any]]:
        batch = dataset.get_batch(indices, **dict(get_batch_kwargs))
        if bool(pin_memory):
            batch = _pin_tensor_batch(batch)
        return indices, batch

    if int(prefetch_batches) <= 0:
        for indices in iterator:
            yield load(indices)
        return
    workers = 1
    with ThreadPoolExecutor(max_workers=workers, thread_name_prefix="seq100-batch-prefetch") as executor:
        try:
            first_indices = next(iterator)
        except StopIteration:
            return
        future: Future[tuple[list[int], dict[str, Any]]] = executor.submit(load, first_indices)
        while True:
            current = future.result()
            try:
                next_indices = next(iterator)
            except StopIteration:
                yield current
                break
            future = executor.submit(load, next_indices)
            yield current


def _run_global_tail_rank_step(
    *,
    model: nn.Module,
    dataset: SequencePathPackDataset,
    indices: list[int],
    device: torch.device,
    optimizer: torch.optim.Optimizer,
    scaler: Any,
    amp_enabled: bool,
    path_value_gradient_profile: str,
    rank_loss_weight: float,
    residual_score_weight: float,
) -> tuple[torch.Tensor, int]:
    """Run one rank-only optimizer step for a single-date global-tail slate."""

    batch = dataset.get_batch(
        indices,
        include_ohlcva_path=False,
        include_richer_path=False,
        include_summary=False,
        include_metadata=False,
    )
    tradable_path = (
        batch["y_tradable_path"].to(device, non_blocking=device.type == "cuda")
        if batch.get("y_tradable_path") is not None
        else None
    )
    x, y_path, _y_ohlcva, _y_richer, _y_summary, date_idx, symbol_idx = _batch_to_device(batch, device)
    optimizer.zero_grad(set_to_none=True)
    with torch.amp.autocast(device_type=device.type, enabled=amp_enabled):
        outputs = model(x, symbol_idx=symbol_idx)
        score, target = _derived_path_rank_score_and_target(
            outputs,
            y_path,
            price_anchor=dataset.price_anchor,
            path_value_gradient_profile=path_value_gradient_profile,
            target_tradable_path=tradable_path,
            residual_weight=float(residual_score_weight),
        )
        rank_loss = _global_tail_rank_loss_by_date(score, target, date_idx)
        weighted_loss = float(rank_loss_weight) * rank_loss
    scaler.scale(weighted_loss).backward()
    scaler.unscale_(optimizer)
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    scaler.step(optimizer)
    scaler.update()
    return rank_loss.detach(), int(x.shape[0])


@torch.no_grad()
def _mine_global_tail_scores(
    *,
    model: nn.Module,
    dataset: SequencePathPackDataset,
    device: torch.device,
    batch_size: int,
    amp_enabled: bool,
    path_value_gradient_profile: str,
    residual_score_weight: float,
) -> np.ndarray:
    """Score the full train split from one frozen epoch-end model state."""

    scores = np.full(len(dataset), np.nan, dtype=np.float32)
    was_training = bool(model.training)
    model.eval()
    try:
        sampler = ShuffledBatchSampler(
            len(dataset),
            batch_size=int(batch_size),
            shuffle=False,
            seed=0,
        )
        for indices in sampler:
            batch = dataset.get_batch(
                indices,
                include_ohlcva_path=False,
                include_richer_path=False,
                include_summary=False,
                include_metadata=False,
            )
            x = batch["x"].to(device, non_blocking=device.type == "cuda")
            symbol_idx = batch["symbol_idx"].to(device, non_blocking=device.type == "cuda")
            with torch.amp.autocast(device_type=device.type, enabled=amp_enabled):
                outputs = model(x, symbol_idx=symbol_idx)
                if "future_path" not in outputs or int(outputs["future_path"].shape[-1]) != 4:
                    raise ValueError("global-tail mining requires price-only OHLC path predictions")
                predicted_summary = _derive_path_summary_torch(
                    outputs["future_path"],
                    smooth_value=True,
                    price_anchor=dataset.price_anchor,
                    path_value_gradient_profile=path_value_gradient_profile,
                )
                score = predicted_summary[:, -1]
                if "residual_score" in outputs:
                    score = score + float(residual_score_weight) * outputs["residual_score"]
            scores[np.asarray(indices, dtype=np.int64)] = score.detach().float().cpu().numpy()
    finally:
        model.train(was_training)
    return scores


def _daily_spearman(frame: pd.DataFrame, *, score_col: str, target_col: str) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for trade_date, group in frame.groupby("trade_date", sort=True):
        valid = group[[score_col, target_col]].apply(pd.to_numeric, errors="coerce").dropna()
        if len(valid) < 5:
            continue
        corr = valid.corr(method="spearman").iloc[0, 1]
        rows.append(
            {
                "trade_date": str(trade_date),
                "rank_ic": float(corr) if pd.notna(corr) else np.nan,
                "count": int(len(valid)),
            }
        )
    return pd.DataFrame(rows)


def _finite_threshold_rate(values: pd.Series, *, threshold: float, comparison: str) -> float:
    numeric = pd.to_numeric(values, errors="coerce").dropna()
    if numeric.empty:
        return np.nan
    if comparison == "ge":
        return float((numeric >= float(threshold)).mean())
    if comparison == "le":
        return float((numeric <= float(threshold)).mean())
    raise ValueError("comparison must be ge or le")


def _topk_daily_rows(frame: pd.DataFrame, *, top_k_values: tuple[int, ...], forward_days: int, value_column: str) -> list[dict[str, Any]]:
    suffix = f"{int(forward_days)}d"
    rows: list[dict[str, Any]] = []
    metric_cols = [
        f"future_max_return_{suffix}",
        f"future_min_return_{suffix}",
        f"future_final_return_{suffix}",
        f"drawdown_after_peak_{suffix}",
        value_column,
    ]
    optional_metric_cols = [
        f"best_exit_close_return_{suffix}",
        f"pre_exit_max_drawdown_{suffix}",
        f"volume_expansion_peak_{suffix}",
        f"amount_expansion_peak_{suffix}",
        f"price_volume_confirmation_{suffix}",
        f"path_volatility_{suffix}",
        f"best_entry_day_{suffix}",
        f"best_entry_price_{suffix}",
        f"best_exit_price_{suffix}",
        f"best_holding_days_{suffix}",
        f"pre_entry_wait_days_{suffix}",
        f"in_trade_max_drawdown_{suffix}",
        f"entry_amount_condition_{suffix}",
        "realized_fill",
        "realized_trade_return",
        "realized_in_trade_drawdown",
        "missed_opportunity",
        "realized_entry_day",
        "realized_exit_day",
        "predicted_exit_day",
        "realized_plan_entry_filled",
        "realized_plan_exit_day",
        "realized_plan_exit_tradable",
        "realized_plan_covered",
        "realized_plan_terminal_recovery",
        "realized_plan_return",
        "realized_plan_gross_return",
        "realized_plan_drawdown",
        "opportunity_value",
        "realized_plan_value",
        "oracle_regret",
    ]
    for trade_date, raw_group in frame.groupby("trade_date", sort=True):
        # Candidate completeness is a scoring invariant.  A finite score enters
        # the daily ranking even when future supervision is unavailable; metric
        # means below ignore NaN labels instead of deleting the candidate before
        # Top-K selection.
        group = raw_group.dropna(subset=["score"]).copy()
        if group.empty:
            continue
        sort_columns = ["score", *( ["symbol"] if "symbol" in group.columns else [])]
        ascending = [False, *( [True] if "symbol" in group.columns else [])]
        group = group.sort_values(sort_columns, ascending=ascending, kind="mergesort")
        universe_symbols = (
            sorted(group["symbol"].astype(str).tolist())
            if "symbol" in group.columns
            else [str(item) for item in sorted(group.index.tolist())]
        )
        universe_hash = hashlib.sha256("\n".join(universe_symbols).encode("utf-8")).hexdigest()
        for top_k in top_k_values:
            top = group.head(int(top_k))
            row: dict[str, Any] = {
                "trade_date": str(trade_date),
                "top_k": int(top_k),
                "universe_count": int(len(group)),
                "selected_count": int(len(top)),
                "universe_hash": universe_hash,
                "universe_labeled_count": int(pd.to_numeric(group[value_column], errors="coerce").notna().sum()),
                "selected_labeled_count": int(pd.to_numeric(top[value_column], errors="coerce").notna().sum()),
            }
            row["universe_label_coverage"] = float(row["universe_labeled_count"] / max(len(group), 1))
            row["selected_label_coverage"] = float(row["selected_labeled_count"] / max(len(top), 1))
            for col in [*metric_cols, *[col for col in optional_metric_cols if col in group.columns]]:
                universe_values = pd.to_numeric(group[col], errors="coerce")
                selected_values = pd.to_numeric(top[col], errors="coerce")
                universe_count = int(universe_values.notna().sum())
                selected_count = int(selected_values.notna().sum())
                row[f"universe_{col}_coverage"] = float(universe_count / max(len(group), 1))
                row[f"selected_{col}_coverage"] = float(selected_count / max(len(top), 1))
                execution_primary = col in {"realized_plan_return", "realized_plan_value"}
                universe_mean = (
                    universe_values.mean()
                    if not execution_primary or universe_count == len(group)
                    else np.nan
                )
                selected_mean = (
                    selected_values.mean()
                    if not execution_primary or selected_count == len(top)
                    else np.nan
                )
                row[f"selected_{col}"] = float(selected_mean)
                row[f"universe_{col}"] = float(universe_mean)
                row[f"alpha_{col}"] = float(selected_mean - universe_mean)
            row["selected_hit_5pct_rate"] = _finite_threshold_rate(
                top[f"future_max_return_{suffix}"], threshold=0.05, comparison="ge"
            )
            row["selected_hit_10pct_rate"] = _finite_threshold_rate(
                top[f"future_max_return_{suffix}"], threshold=0.10, comparison="ge"
            )
            row["selected_hit_20pct_rate"] = _finite_threshold_rate(
                top[f"future_max_return_{suffix}"], threshold=0.20, comparison="ge"
            )
            row["selected_loss_3pct_rate"] = _finite_threshold_rate(
                top[f"future_min_return_{suffix}"], threshold=-0.03, comparison="le"
            )
            row["selected_loss_5pct_rate"] = _finite_threshold_rate(
                top[f"future_min_return_{suffix}"], threshold=-0.05, comparison="le"
            )
            row["selected_loss_10pct_rate"] = _finite_threshold_rate(
                top[f"future_min_return_{suffix}"], threshold=-0.10, comparison="le"
            )
            row["selected_peak_day_mean"] = float(pd.to_numeric(top[f"future_peak_day_{suffix}"], errors="coerce").mean())
            if f"best_exit_day_{suffix}" in top.columns:
                row["selected_best_exit_day_mean"] = float(pd.to_numeric(top[f"best_exit_day_{suffix}"], errors="coerce").mean())
            if f"pre_exit_max_drawdown_{suffix}" in top.columns:
                row["selected_pre_exit_max_drawdown_mean"] = float(
                    pd.to_numeric(top[f"pre_exit_max_drawdown_{suffix}"], errors="coerce").mean()
                )
            if f"best_entry_day_{suffix}" in top.columns:
                row["selected_best_entry_day_mean"] = float(pd.to_numeric(top[f"best_entry_day_{suffix}"], errors="coerce").mean())
            if f"best_holding_days_{suffix}" in top.columns:
                row["selected_best_holding_days_mean"] = float(pd.to_numeric(top[f"best_holding_days_{suffix}"], errors="coerce").mean())
            if f"in_trade_max_drawdown_{suffix}" in top.columns:
                row["selected_in_trade_max_drawdown_mean"] = float(pd.to_numeric(top[f"in_trade_max_drawdown_{suffix}"], errors="coerce").mean())
            if "realized_fill" in top.columns:
                row["selected_realized_fill_rate"] = float(pd.to_numeric(top["realized_fill"], errors="coerce").mean())
            if "realized_plan_entry_filled" in top.columns:
                row["selected_entry_fill_rate"] = float(
                    pd.to_numeric(top["realized_plan_entry_filled"], errors="coerce").mean()
                )
            if "realized_plan_covered" in top.columns:
                row["selected_realized_plan_coverage"] = float(
                    pd.to_numeric(top["realized_plan_covered"], errors="coerce").mean()
                )
            if "missed_opportunity" in top.columns:
                row["selected_missed_opportunity_rate"] = float(pd.to_numeric(top["missed_opportunity"], errors="coerce").mean())
            rows.append(row)
    return rows


def _topk_candidate_rows(
    frame: pd.DataFrame,
    *,
    max_top_k: int,
    forward_days: int,
    value_column: str,
) -> list[dict[str, Any]]:
    suffix = f"{int(forward_days)}d"
    metric_columns = [
        value_column,
        f"future_max_return_{suffix}",
        f"future_min_return_{suffix}",
        f"future_final_return_{suffix}",
        f"drawdown_after_peak_{suffix}",
        f"best_exit_day_{suffix}",
        f"best_exit_close_return_{suffix}",
        f"pre_exit_max_drawdown_{suffix}",
        f"in_trade_max_drawdown_{suffix}",
        "predicted_exit_day",
        "realized_plan_entry_filled",
        "realized_plan_covered",
        "realized_plan_terminal_recovery",
        "realized_plan_exit_day",
        "realized_plan_return",
        "realized_plan_gross_return",
        "opportunity_value",
        "realized_plan_value",
        "oracle_regret",
    ]
    rows: list[dict[str, Any]] = []
    for trade_date, raw_group in frame.groupby("trade_date", sort=True):
        group = raw_group.dropna(subset=["score"]).copy()
        if group.empty:
            continue
        sort_columns = ["score", *( ["symbol"] if "symbol" in group.columns else [])]
        ascending = [False, *( [True] if "symbol" in group.columns else [])]
        group = group.sort_values(sort_columns, ascending=ascending, kind="mergesort")
        universe_symbols = (
            sorted(group["symbol"].astype(str).tolist())
            if "symbol" in group.columns
            else [str(item) for item in sorted(group.index.tolist())]
        )
        universe_hash = hashlib.sha256("\n".join(universe_symbols).encode("utf-8")).hexdigest()
        top = group.head(max(int(max_top_k), 0))
        for rank, (_, item) in enumerate(top.iterrows(), start=1):
            row: dict[str, Any] = {
                "trade_date": str(trade_date),
                "score_rank": int(rank),
                "symbol": str(item.get("symbol", "")),
                "score": float(item["score"]),
                "universe_count": int(len(group)),
                "universe_hash": universe_hash,
                "label_available": bool(pd.notna(pd.to_numeric(pd.Series([item.get(value_column)]), errors="coerce").iloc[0])),
            }
            for column in metric_columns:
                if column in item.index:
                    value = pd.to_numeric(pd.Series([item[column]]), errors="coerce").iloc[0]
                    row[column] = float(value) if pd.notna(value) else np.nan
            rows.append(row)
    return rows


def _aggregate_topk_daily_rows(rows: list[dict[str, Any]]) -> pd.DataFrame:
    daily = pd.DataFrame(rows)
    if daily.empty:
        return pd.DataFrame()
    out_rows: list[dict[str, Any]] = []
    for top_k, group in daily.groupby("top_k", sort=True):
        out: dict[str, Any] = {"top_k": int(top_k), "day_count": int(len(group))}
        for col in [c for c in group.columns if c not in {"trade_date", "top_k", "universe_hash", "score_column"}]:
            values = pd.to_numeric(group[col], errors="coerce")
            execution_primary = any(
                token in col
                for token in (
                    "selected_realized_plan_return",
                    "universe_realized_plan_return",
                    "alpha_realized_plan_return",
                    "selected_realized_plan_value",
                    "universe_realized_plan_value",
                    "alpha_realized_plan_value",
                )
            ) and not col.endswith("_coverage")
            out[col] = float(values.mean()) if not execution_primary or bool(values.notna().all()) else np.nan
        out_rows.append(out)
    return pd.DataFrame(out_rows)


def _topk_metrics(frame: pd.DataFrame, *, top_k_values: tuple[int, ...], forward_days: int, value_column: str) -> pd.DataFrame:
    rows = _topk_daily_rows(frame, top_k_values=top_k_values, forward_days=forward_days, value_column=value_column)
    return _aggregate_topk_daily_rows(rows)


def _validate_development_topk_execution_coverage(topk: pd.DataFrame) -> None:
    required = {
        "selected_realized_plan_return_coverage",
        "selected_realized_plan_value_coverage",
        "selected_realized_plan_coverage",
    }
    missing = sorted(required.difference(topk.columns))
    if missing:
        raise ValueError(f"development TopK is missing deterministic execution coverage columns: {missing}")
    for column in sorted(required):
        values = pd.to_numeric(topk[column], errors="coerce")
        if values.empty or not bool(np.isfinite(values).all()) or not bool((values >= 1.0 - 1.0e-12).all()):
            raise ValueError(
                f"development TopK cannot use partial execution outcomes: {column} must be 100%"
            )


def _split_metrics_for_score(
    frame: pd.DataFrame,
    *,
    split: str,
    score_col: str,
    value_column: str,
    prediction_csv: str,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    ic = _daily_spearman(frame, score_col=score_col, target_col=value_column)
    target_values = pd.to_numeric(frame[value_column], errors="coerce").to_numpy(dtype=np.float64, copy=False)
    prediction_values = pd.to_numeric(frame[score_col], errors="coerce").to_numpy(dtype=np.float64, copy=False)
    target_mask = np.isfinite(target_values)
    prediction_mask = np.isfinite(prediction_values)
    metrics = {
        "split": split,
        "row_count": int(len(frame)),
        "date_count": int(frame["trade_date"].nunique()),
        "rank_ic_mean": float(ic["rank_ic"].mean()) if not ic.empty else np.nan,
        "rank_ic_median": float(ic["rank_ic"].median()) if not ic.empty else np.nan,
        "rank_ic_positive_day_rate": float((ic["rank_ic"] > 0).mean()) if not ic.empty else np.nan,
        "value_column": str(value_column),
        "score_column": str(score_col),
        "target_mean": float(np.nansum(target_values[target_mask]) / max(int(target_mask.sum()), 1)),
        "prediction_mean": float(np.nansum(prediction_values[prediction_mask]) / max(int(prediction_mask.sum()), 1)),
        "prediction_csv": str(prediction_csv),
    }
    return ic, metrics


def _find_latest_baseline_summary(forward_days: int) -> dict[str, Any]:
    root = Path("daily_research/output/path_policy/path_value_predictability")
    matches: list[tuple[float, Path, dict[str, Any]]] = []
    for path in sorted(root.glob("*/path_value_predictability_summary.json")):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        detected = payload.get("forward_days")
        if detected is None:
            text = json.dumps(payload, ensure_ascii=False)
            detected = 60 if "60-day" in text or "_60d" in text or "path60" in str(path) else None
        if detected is not None and int(detected) == int(forward_days):
            matches.append((path.stat().st_mtime, path, payload))
    if not matches:
        return {}
    return sorted(matches, key=lambda item: item[0])[-1][2]


def path_value_v2_column(forward_days: int) -> str:
    return f"path_trade_value_v2_{int(forward_days)}d"


def unified_path_value_column(forward_days: int) -> str:
    return f"unified_path_trade_value_{int(forward_days)}d"


def legacy_derived_path_summary_columns(forward_days: int) -> list[str]:
    suffix = f"{int(forward_days)}d"
    return [
        f"future_max_return_{suffix}",
        f"future_min_return_{suffix}",
        f"future_final_return_{suffix}",
        f"future_peak_day_{suffix}",
        f"future_trough_day_{suffix}",
        f"drawdown_after_peak_{suffix}",
        f"time_above_zero_{suffix}",
        f"time_below_zero_{suffix}",
        f"best_exit_day_{suffix}",
        f"best_exit_close_return_{suffix}",
        f"pre_exit_max_drawdown_{suffix}",
        path_value_v2_column(forward_days),
    ]


def unified_path_summary_columns(forward_days: int) -> list[str]:
    suffix = f"{int(forward_days)}d"
    return [
        f"future_max_return_{suffix}",
        f"future_min_return_{suffix}",
        f"future_final_return_{suffix}",
        f"future_peak_day_{suffix}",
        f"future_trough_day_{suffix}",
        f"drawdown_after_peak_{suffix}",
        f"time_above_zero_{suffix}",
        f"time_below_zero_{suffix}",
        f"volume_expansion_peak_{suffix}",
        f"amount_expansion_peak_{suffix}",
        f"price_volume_confirmation_{suffix}",
        f"path_volatility_{suffix}",
        f"best_entry_day_{suffix}",
        f"best_entry_price_{suffix}",
        f"best_exit_day_{suffix}",
        f"best_exit_price_{suffix}",
        f"best_holding_days_{suffix}",
        f"pre_entry_wait_days_{suffix}",
        f"in_trade_max_drawdown_{suffix}",
        f"entry_amount_condition_{suffix}",
        unified_path_value_column(forward_days),
    ]


def derived_path_summary_columns(forward_days: int, *, path_dim: int = 4) -> list[str]:
    return unified_path_summary_columns(forward_days) if int(path_dim) >= 6 else legacy_derived_path_summary_columns(forward_days)


def value_column_for_path(forward_days: int, *, path_dim: int = 4) -> str:
    return unified_path_value_column(forward_days) if int(path_dim) >= 6 else path_value_v2_column(forward_days)


def _derived_summary_loss_indices(forward_days: int, *, path_dim: int = 4) -> list[int]:
    columns = derived_path_summary_columns(forward_days, path_dim=path_dim)
    keep = {
        f"future_max_return_{int(forward_days)}d",
        f"future_min_return_{int(forward_days)}d",
        f"future_final_return_{int(forward_days)}d",
        f"drawdown_after_peak_{int(forward_days)}d",
        f"best_exit_close_return_{int(forward_days)}d",
        f"pre_exit_max_drawdown_{int(forward_days)}d",
        f"volume_expansion_peak_{int(forward_days)}d",
        f"amount_expansion_peak_{int(forward_days)}d",
        f"price_volume_confirmation_{int(forward_days)}d",
        f"path_volatility_{int(forward_days)}d",
        f"in_trade_max_drawdown_{int(forward_days)}d",
        f"entry_amount_condition_{int(forward_days)}d",
    }
    return [idx for idx, col in enumerate(columns) if col in keep]


def _summary_loss_windows(forward_days: int, *, include_full_horizon: bool = True) -> tuple[int, ...]:
    horizon = int(forward_days)
    windows = [int(window) for window in MULTI_HORIZON_OHLC_WINDOWS if int(window) <= horizon]
    if bool(include_full_horizon) and horizon not in windows:
        windows.append(horizon)
    if not bool(include_full_horizon):
        windows = [int(window) for window in windows if int(window) < horizon]
    return tuple(sorted(set(windows)))


def _legacy_entry_relative_path_torch(path: torch.Tensor, *, price_anchor: str) -> torch.Tensor:
    if str(price_anchor) != "today_close":
        return path
    entry_open = path[:, :1, 0:1]
    return (1.0 + path) / torch.clamp(1.0 + entry_open, min=1.0e-6) - 1.0


def _legacy_entry_relative_path_numpy(path: np.ndarray, *, price_anchor: str) -> np.ndarray:
    values = np.asarray(path, dtype=np.float32)
    if str(price_anchor) != "today_close":
        return values
    entry_open = values[:, :1, 0:1].astype(np.float64, copy=False)
    converted = (1.0 + values.astype(np.float64, copy=False)) / np.maximum(1.0 + entry_open, 1.0e-6) - 1.0
    return converted.astype(np.float32, copy=False)


def _candidate_path_values_torch(
    path: torch.Tensor,
    *,
    tradable_path: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    forward_days = int(path.shape[1])
    close_ret = path[:, :, 3]
    low_ret = path[:, :, 2]
    worst_low_so_far = torch.cummin(low_ret, dim=1).values
    pre_exit_drawdown = torch.clamp(-worst_low_so_far, min=0.0)
    day = torch.arange(forward_days, device=path.device, dtype=path.dtype)
    waiting = torch.sqrt((day + 1.0) / max(float(forward_days), 1.0)).view(1, -1)
    candidate = (
        close_ret
        - float(PATH_VALUE_V2_DRAWDOWN_PENALTY) * pre_exit_drawdown
        - float(PATH_VALUE_V2_WAITING_PENALTY) * waiting
        - float(PATH_VALUE_V2_TRANSACTION_COST)
    )
    if tradable_path is not None:
        if tuple(tradable_path.shape) != tuple(candidate.shape):
            raise ValueError(f"tradable_path must have shape {tuple(candidate.shape)}")
        candidate = torch.where(tradable_path.to(dtype=torch.bool), candidate, torch.full_like(candidate, float("-inf")))
    return candidate, pre_exit_drawdown


def _entry_exit_ret_torch(path: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    open_ret = path[:, :, 0]
    high_ret = path[:, :, 1]
    low_ret = path[:, :, 2]
    close_ret = path[:, :, 3]
    entry_ret = 0.50 * low_ret + 0.25 * open_ret + 0.25 * close_ret
    exit_ret = 0.50 * high_ret + 0.25 * open_ret + 0.25 * close_ret
    return entry_ret, exit_ret


def _unified_trade_candidates_torch(path: torch.Tensor) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    forward_days = int(path.shape[1])
    low_ret = path[:, :, 2]
    amount_rel = path[:, :, 5]
    entry_ret, exit_ret = _entry_exit_ret_torch(path)
    batch_size = int(path.shape[0])
    entry_matrix = entry_ret.unsqueeze(2)
    exit_matrix = exit_ret.unsqueeze(1)
    trade_return = (1.0 + exit_matrix) / torch.clamp(1.0 + entry_matrix, min=1.0e-6) - 1.0
    low_between = torch.full((batch_size, forward_days, forward_days), float("inf"), device=path.device, dtype=path.dtype)
    amount_mean = torch.full_like(low_between, float("nan"))
    for entry_day in range(forward_days):
        low_cum = torch.cummin(low_ret[:, entry_day:], dim=1).values
        low_between[:, entry_day, entry_day:] = low_cum
        amount_cum = torch.cumsum(amount_rel[:, entry_day:], dim=1)
        denom = torch.arange(1, forward_days - entry_day + 1, device=path.device, dtype=path.dtype).view(1, -1)
        amount_mean[:, entry_day, entry_day:] = amount_cum / denom
    in_trade_drawdown = torch.clamp(1.0 - (1.0 + low_between) / torch.clamp(1.0 + entry_matrix, min=1.0e-6), min=0.0)
    day = torch.arange(forward_days, device=path.device, dtype=path.dtype)
    entry_day = day.view(1, forward_days, 1)
    exit_day = day.view(1, 1, forward_days)
    holding_days = exit_day - entry_day
    wait_penalty = float(UNIFIED_VALUE_WAIT_PENALTY) * torch.sqrt((entry_day + 1.0) / max(float(forward_days), 1.0))
    hold_penalty = float(UNIFIED_VALUE_HOLD_PENALTY) * torch.sqrt((holding_days + 1.0) / max(float(forward_days), 1.0))
    liquidity_penalty = float(UNIFIED_VALUE_LIQUIDITY_PENALTY) * torch.relu(-amount_mean)
    candidate = (
        trade_return
        - wait_penalty
        - hold_penalty
        - float(UNIFIED_VALUE_DRAWDOWN_PENALTY) * in_trade_drawdown
        - liquidity_penalty
        - float(UNIFIED_VALUE_TRANSACTION_COST)
    )
    valid = holding_days > 0.0
    candidate = torch.where(valid, candidate, torch.full_like(candidate, float("-inf")))
    return candidate, {
        "entry_ret": entry_ret,
        "exit_ret": exit_ret,
        "trade_return": trade_return,
        "in_trade_drawdown": in_trade_drawdown,
        "amount_mean": amount_mean,
    }


def _derive_unified_path_summary_torch(
    path: torch.Tensor,
    *,
    smooth_value: bool,
    path_value_gradient_profile: str = PATH_VALUE_GRADIENT_PROFILE_SMOOTH,
) -> torch.Tensor:
    forward_days = int(path.shape[1])
    high_ret = path[:, :, 1]
    low_ret = path[:, :, 2]
    close_ret = path[:, :, 3]
    volume_rel = path[:, :, 4]
    amount_rel = path[:, :, 5]
    max_ret, peak_idx = torch.max(high_ret, dim=1)
    min_ret, trough_idx = torch.min(low_ret, dim=1)
    final_ret = close_ret[:, -1]
    day_idx = torch.arange(forward_days, device=path.device).view(1, -1)
    after_peak = day_idx >= peak_idx.view(-1, 1)
    low_after_peak = torch.where(after_peak, low_ret, torch.full_like(low_ret, float("inf")))
    min_after_peak = torch.min(low_after_peak, dim=1).values
    drawdown_after_peak = (1.0 + min_after_peak) / torch.clamp(1.0 + max_ret, min=1.0e-6) - 1.0
    time_above = (close_ret > 0.0).to(path.dtype).mean(dim=1)
    time_below = (close_ret < 0.0).to(path.dtype).mean(dim=1)
    volume_peak = torch.max(volume_rel, dim=1).values
    amount_peak = torch.max(amount_rel, dim=1).values
    price_volume_confirmation = torch.mean(torch.relu(close_ret) * torch.relu(amount_rel), dim=1)
    path_volatility = torch.std(close_ret, dim=1, unbiased=False)
    candidate, aux = _unified_trade_candidates_torch(path)
    flat = candidate.view(candidate.shape[0], -1)
    hard_value, flat_idx = torch.max(flat, dim=1)
    if smooth_value:
        best_value = _path_value_with_gradient_profile(
            hard_value,
            flat,
            temperature=UNIFIED_VALUE_TEMPERATURE,
            profile=path_value_gradient_profile,
            reduction_dim=1,
        )
    else:
        best_value = hard_value
    best_entry_idx = torch.div(flat_idx, forward_days, rounding_mode="floor")
    best_exit_idx = flat_idx % forward_days
    row = torch.arange(path.shape[0], device=path.device)
    best_entry_price = aux["entry_ret"][row, best_entry_idx]
    best_exit_price = aux["exit_ret"][row, best_exit_idx]
    best_holding_days = (best_exit_idx - best_entry_idx).to(path.dtype)
    best_drawdown = aux["in_trade_drawdown"][row, best_entry_idx, best_exit_idx]
    entry_amount_condition = amount_rel[row, best_entry_idx]
    return torch.stack(
        [
            max_ret,
            min_ret,
            final_ret,
            peak_idx.to(path.dtype) + 1.0,
            trough_idx.to(path.dtype) + 1.0,
            drawdown_after_peak,
            time_above,
            time_below,
            volume_peak,
            amount_peak,
            price_volume_confirmation,
            path_volatility,
            best_entry_idx.to(path.dtype) + 1.0,
            best_entry_price,
            best_exit_idx.to(path.dtype) + 1.0,
            best_exit_price,
            best_holding_days,
            best_entry_idx.to(path.dtype),
            best_drawdown,
            entry_amount_condition,
            best_value,
        ],
        dim=1,
    )


def _derive_path_summary_torch(
    path: torch.Tensor,
    *,
    smooth_value: bool,
    price_anchor: str = "next_open",
    path_value_gradient_profile: str = PATH_VALUE_GRADIENT_PROFILE_SMOOTH,
    tradable_path: torch.Tensor | None = None,
) -> torch.Tensor:
    if int(path.shape[2]) >= 6:
        if tradable_path is not None:
            raise ValueError("tradable_path-aware unified OHLCVA value is not implemented")
        return _derive_unified_path_summary_torch(
            path,
            smooth_value=smooth_value,
            path_value_gradient_profile=path_value_gradient_profile,
        )
    path = _legacy_entry_relative_path_torch(path, price_anchor=price_anchor)
    forward_days = int(path.shape[1])
    high_ret = path[:, :, 1]
    low_ret = path[:, :, 2]
    close_ret = path[:, :, 3]
    max_ret, peak_idx = torch.max(high_ret, dim=1)
    min_ret, trough_idx = torch.min(low_ret, dim=1)
    final_ret = close_ret[:, -1]
    day_idx = torch.arange(forward_days, device=path.device).view(1, -1)
    after_peak = day_idx >= peak_idx.view(-1, 1)
    low_after_peak = torch.where(after_peak, low_ret, torch.full_like(low_ret, float("inf")))
    min_after_peak = torch.min(low_after_peak, dim=1).values
    drawdown_after_peak = (1.0 + min_after_peak) / torch.clamp(1.0 + max_ret, min=1.0e-6) - 1.0
    time_above = (close_ret > 0.0).to(path.dtype).mean(dim=1)
    time_below = (close_ret < 0.0).to(path.dtype).mean(dim=1)
    candidate, pre_exit_drawdown = _candidate_path_values_torch(path, tradable_path=tradable_path)
    best_value_hard, best_idx = torch.max(candidate, dim=1)
    if smooth_value:
        best_value = _path_value_with_gradient_profile(
            best_value_hard,
            candidate,
            temperature=PATH_VALUE_V2_TEMPERATURE,
            profile=path_value_gradient_profile,
            reduction_dim=1,
        )
    else:
        best_value = best_value_hard
    best_exit_close = close_ret.gather(1, best_idx.view(-1, 1)).squeeze(1)
    best_pre_exit_drawdown = pre_exit_drawdown.gather(1, best_idx.view(-1, 1)).squeeze(1)
    has_tradable_exit = torch.isfinite(best_value_hard)
    best_exit_day = torch.where(
        has_tradable_exit,
        best_idx.to(path.dtype) + 1.0,
        torch.full_like(best_value_hard, float("nan")),
    )
    best_exit_close = torch.where(has_tradable_exit, best_exit_close, torch.full_like(best_exit_close, float("nan")))
    best_pre_exit_drawdown = torch.where(
        has_tradable_exit,
        best_pre_exit_drawdown,
        torch.full_like(best_pre_exit_drawdown, float("nan")),
    )
    best_value = torch.where(has_tradable_exit, best_value, torch.full_like(best_value, float("nan")))
    return torch.stack(
        [
            max_ret,
            min_ret,
            final_ret,
            peak_idx.to(path.dtype) + 1.0,
            trough_idx.to(path.dtype) + 1.0,
            drawdown_after_peak,
            time_above,
            time_below,
            best_exit_day,
            best_exit_close,
            best_pre_exit_drawdown,
            best_value,
        ],
        dim=1,
    )


def _derive_path_summary_numpy(
    path: np.ndarray,
    *,
    price_anchor: str = "next_open",
    tradable_path: np.ndarray | None = None,
) -> np.ndarray:
    values = np.asarray(path, dtype=np.float32)
    if values.ndim != 3 or values.shape[2] not in {4, 6}:
        raise ValueError("path must have shape [batch, forward_days, 4 or 6]")
    if values.shape[2] >= 6:
        if tradable_path is not None:
            raise ValueError("tradable_path-aware unified OHLCVA value is not implemented")
        return _derive_unified_path_summary_numpy(values)
    values = _legacy_entry_relative_path_numpy(values, price_anchor=price_anchor)
    forward_days = int(values.shape[1])
    high_ret = values[:, :, 1].astype(np.float64, copy=False)
    low_ret = values[:, :, 2].astype(np.float64, copy=False)
    close_ret = values[:, :, 3].astype(np.float64, copy=False)
    max_ret = np.nanmax(high_ret, axis=1)
    min_ret = np.nanmin(low_ret, axis=1)
    final_ret = close_ret[:, -1]
    peak_idx = np.nanargmax(np.where(np.isfinite(high_ret), high_ret, -np.inf), axis=1)
    trough_idx = np.nanargmin(np.where(np.isfinite(low_ret), low_ret, np.inf), axis=1)
    min_after_peak = np.full(values.shape[0], np.nan, dtype=np.float64)
    for row_idx, pidx in enumerate(peak_idx):
        min_after_peak[row_idx] = np.nanmin(low_ret[row_idx, int(pidx) :])
    drawdown_after_peak = (1.0 + min_after_peak) / np.maximum(1.0 + max_ret, 1.0e-6) - 1.0
    time_above = np.nanmean(close_ret > 0.0, axis=1)
    time_below = np.nanmean(close_ret < 0.0, axis=1)
    worst_low_so_far = np.minimum.accumulate(low_ret, axis=1)
    pre_exit_drawdown = np.maximum(-worst_low_so_far, 0.0)
    day = np.arange(forward_days, dtype=np.float64)
    waiting = np.sqrt((day + 1.0) / max(float(forward_days), 1.0))
    candidate = (
        close_ret
        - float(PATH_VALUE_V2_DRAWDOWN_PENALTY) * pre_exit_drawdown
        - float(PATH_VALUE_V2_WAITING_PENALTY) * waiting.reshape(1, -1)
        - float(PATH_VALUE_V2_TRANSACTION_COST)
    )
    tradable = None if tradable_path is None else np.asarray(tradable_path, dtype=bool)
    if tradable is not None:
        if tradable.shape != candidate.shape:
            raise ValueError(f"tradable_path must have shape {candidate.shape}, got {tradable.shape}")
        candidate = np.where(tradable, candidate, -np.inf)
    best_idx = np.nanargmax(np.where(np.isfinite(candidate), candidate, -np.inf), axis=1)
    row = np.arange(values.shape[0])
    best_exit_close = close_ret[row, best_idx]
    best_pre_exit_drawdown = pre_exit_drawdown[row, best_idx]
    best_value = candidate[row, best_idx]
    summary = np.column_stack(
        [
            max_ret,
            min_ret,
            final_ret,
            peak_idx + 1,
            trough_idx + 1,
            drawdown_after_peak,
            time_above,
            time_below,
            best_idx + 1,
            best_exit_close,
            best_pre_exit_drawdown,
            best_value,
        ]
    )
    invalid_value = ~np.isfinite(best_value)
    summary[invalid_value, 8:] = np.nan
    return summary.astype(np.float32, copy=False)


def _realize_path_value_v2_plan_numpy(
    pred_summary: np.ndarray,
    true_path: np.ndarray,
    *,
    forward_days: int,
    price_anchor: str = "next_open",
    tradable_path: np.ndarray | None = None,
    entry_filled: np.ndarray | None = None,
    entry_open_raw: np.ndarray | None = None,
    exit_close_raw_path: np.ndarray | None = None,
    exit_sellable_path: np.ndarray | None = None,
    execution_tail_days: int = 0,
    terminal_recovery_fraction: float = 0.0,
) -> dict[str, np.ndarray]:
    """Evaluate the exit day selected from a predicted four-field OHLC path.

    The opportunity target remains the hard maximum over the realized path.  The
    realized-plan value instead evaluates the same deterministic candidate function
    at the predicted exit day, making exit-timing error and oracle regret observable.
    If a tradability mask is supplied, a suspended planned exit is deferred to the
    first later tradable day; no later tradable day leaves the plan unrealized.
    """

    values = _legacy_entry_relative_path_numpy(np.asarray(true_path, dtype=np.float32), price_anchor=price_anchor)
    summary = np.asarray(pred_summary, dtype=np.float64)
    horizon = int(forward_days)
    if values.ndim != 3 or int(values.shape[1]) != horizon or int(values.shape[2]) < 4:
        raise ValueError("true_path must have shape [batch, forward_days, >=4]")
    columns = legacy_derived_path_summary_columns(horizon)
    exit_day_idx = columns.index(f"best_exit_day_{horizon}d")
    value_idx = columns.index(path_value_v2_column(horizon))
    n = int(values.shape[0])
    predicted_exit_day = np.full(n, np.nan, dtype=np.float32)
    realized_exit_day = np.full(n, np.nan, dtype=np.float32)
    realized_return = np.full(n, np.nan, dtype=np.float32)
    realized_drawdown = np.full(n, np.nan, dtype=np.float32)
    realized_value = np.full(n, np.nan, dtype=np.float32)
    exit_tradable = np.zeros(n, dtype=np.float32)
    realized_covered = np.zeros(n, dtype=np.float32)
    terminal_recovery_applied = np.zeros(n, dtype=np.float32)
    filled = (
        np.asarray(entry_filled, dtype=bool).reshape(-1)
        if entry_filled is not None
        else np.ones(n, dtype=bool)
    )
    if int(filled.size) != n:
        raise ValueError("entry_filled must match the path batch size")
    raw_execution_supplied = any(
        item is not None for item in (entry_open_raw, exit_close_raw_path, exit_sellable_path)
    )
    if raw_execution_supplied and not all(
        item is not None for item in (entry_open_raw, exit_close_raw_path, exit_sellable_path)
    ):
        raise ValueError("raw execution evaluation requires entry open, exit close path, and exit sellable path")
    execution_days = int(horizon + int(execution_tail_days))
    raw_entry = None
    raw_exit = None
    sellable = None
    resolved_exit_days = None
    if raw_execution_supplied:
        raw_entry = np.asarray(entry_open_raw, dtype=np.float64).reshape(-1)
        raw_exit = np.asarray(exit_close_raw_path, dtype=np.float64)
        sellable = np.asarray(exit_sellable_path, dtype=bool)
        if int(raw_entry.size) != n or raw_exit.shape != (n, execution_days) or sellable.shape != (n, execution_days):
            raise ValueError("raw execution arrays do not match batch and forward+tail dimensions")
        recovery = float(terminal_recovery_fraction)
        if not math.isfinite(recovery) or recovery < 0.0 or recovery > 1.0:
            raise ValueError("terminal_recovery_fraction must be within [0, 1]")

    true_summary = _derive_path_summary_numpy(
        values,
        price_anchor="next_open",
        tradable_path=tradable_path,
    )
    opportunity_value = true_summary[:, value_idx].astype(np.float32, copy=False)
    close_ret = values[:, :, 3].astype(np.float64, copy=False)
    low_ret = values[:, :, 2].astype(np.float64, copy=False)
    pre_exit_drawdown = np.maximum(-np.minimum.accumulate(low_ret, axis=1), 0.0)
    day = np.arange(horizon, dtype=np.float64)
    waiting = np.sqrt((day + 1.0) / max(float(horizon), 1.0))
    candidate = (
        close_ret
        - float(PATH_VALUE_V2_DRAWDOWN_PENALTY) * pre_exit_drawdown
        - float(PATH_VALUE_V2_WAITING_PENALTY) * waiting.reshape(1, -1)
        - float(PATH_VALUE_V2_TRANSACTION_COST)
    )
    tradable = (
        np.asarray(tradable_path, dtype=bool)
        if tradable_path is not None
        else np.ones((n, horizon), dtype=bool)
    )
    if tradable.shape != (n, horizon):
        raise ValueError(f"tradable_path must have shape {(n, horizon)}, got {tradable.shape}")

    if raw_execution_supplied:
        resolved_exit_days = _resolve_deferred_exit_days(
            summary[:, exit_day_idx],
            sellable,
            forward_days=horizon,
            execution_tail_days=int(execution_tail_days),
        )

    for row in range(n):
        raw_day = summary[row, exit_day_idx]
        if not np.isfinite(raw_day):
            continue
        # The entry occurs on path day 1.  A-share T+1 makes path day 2 the
        # earliest legal exit even if the predicted path peaks immediately.
        planned = max(1, min(int(round(float(raw_day))) - 1, horizon - 1))
        predicted_exit_day[row] = np.float32(planned + 1)
        if not bool(filled[row]):
            realized_return[row] = 0.0
            realized_drawdown[row] = 0.0
            realized_value[row] = 0.0
            realized_covered[row] = 1.0
            continue
        if raw_execution_supplied:
            assert raw_entry is not None and raw_exit is not None and resolved_exit_days is not None
            resolved_day = resolved_exit_days[row]
            if np.isfinite(resolved_day):
                actual = int(round(float(resolved_day))) - 1
                entry_price = float(raw_entry[row])
                exit_price = float(raw_exit[row, actual])
                if not math.isfinite(entry_price) or entry_price <= 0.0:
                    continue
                if not math.isfinite(exit_price) or exit_price < 0.0:
                    continue
                gross_return = exit_price / entry_price - 1.0
                exit_tradable[row] = 1.0
                realized_covered[row] = 1.0
                realized_exit_day[row] = np.float32(actual + 1)
                realized_return[row] = np.float32(gross_return)
                observed_day = min(actual, horizon - 1)
                observed_low = low_ret[row, : observed_day + 1]
                realized_drawdown[row] = np.float32(
                    max(-float(np.nanmin(observed_low)), 0.0)
                    if bool(np.isfinite(observed_low).any())
                    else 0.0
                )
                wait_penalty = float(PATH_VALUE_V2_WAITING_PENALTY) * math.sqrt(
                    float(actual + 1) / max(float(execution_days), 1.0)
                )
                realized_value[row] = np.float32(
                    gross_return
                    - float(PATH_VALUE_V2_DRAWDOWN_PENALTY) * float(realized_drawdown[row])
                    - wait_penalty
                    - float(PATH_VALUE_V2_TRANSACTION_COST)
                )
            else:
                # A filled position that cannot exit through the complete retry
                # window is settled by the manifest-bound terminal rule.  It is
                # an observed execution outcome, not a missing label.
                recovery = float(terminal_recovery_fraction)
                terminal_recovery_applied[row] = 1.0
                realized_covered[row] = 1.0
                realized_exit_day[row] = np.float32(execution_days)
                realized_return[row] = np.float32(recovery - 1.0)
                realized_drawdown[row] = np.float32(1.0 - recovery)
                realized_value[row] = np.float32(recovery - 1.0)
            continue
        later = np.flatnonzero(tradable[row, planned:])
        if not len(later):
            continue
        actual = planned + int(later[0])
        if not np.isfinite(candidate[row, actual]):
            continue
        exit_tradable[row] = 1.0
        realized_covered[row] = 1.0
        realized_exit_day[row] = np.float32(actual + 1)
        realized_return[row] = np.float32(close_ret[row, actual])
        realized_drawdown[row] = np.float32(pre_exit_drawdown[row, actual])
        realized_value[row] = np.float32(candidate[row, actual])

    return {
        "predicted_exit_day": predicted_exit_day,
        "realized_plan_entry_filled": filled.astype(np.float32, copy=False),
        "realized_plan_exit_day": realized_exit_day,
        "realized_plan_exit_tradable": exit_tradable,
        "realized_plan_covered": realized_covered,
        "realized_plan_terminal_recovery": terminal_recovery_applied,
        "realized_plan_return": realized_return,
        "realized_plan_gross_return": realized_return.copy(),
        "realized_plan_drawdown": realized_drawdown,
        "opportunity_value": opportunity_value,
        "realized_plan_value": realized_value,
        "oracle_regret": opportunity_value - realized_value,
    }


def _entry_exit_ret_numpy(path: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    open_ret = path[:, :, 0].astype(np.float64, copy=False)
    high_ret = path[:, :, 1].astype(np.float64, copy=False)
    low_ret = path[:, :, 2].astype(np.float64, copy=False)
    close_ret = path[:, :, 3].astype(np.float64, copy=False)
    entry_ret = 0.50 * low_ret + 0.25 * open_ret + 0.25 * close_ret
    exit_ret = 0.50 * high_ret + 0.25 * open_ret + 0.25 * close_ret
    return entry_ret, exit_ret


def _unified_trade_candidates_numpy(path: np.ndarray) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    values = np.asarray(path, dtype=np.float64)
    batch_size, forward_days, _ = values.shape
    low_ret = values[:, :, 2]
    amount_rel = values[:, :, 5]
    entry_ret, exit_ret = _entry_exit_ret_numpy(values)
    candidate = np.full((batch_size, forward_days, forward_days), -np.inf, dtype=np.float64)
    trade_return = np.full_like(candidate, np.nan)
    in_trade_drawdown = np.full_like(candidate, np.nan)
    amount_mean = np.full_like(candidate, np.nan)
    for entry_day in range(forward_days):
        entry = entry_ret[:, entry_day]
        for exit_day in range(entry_day + 1, forward_days):
            exit_value = exit_ret[:, exit_day]
            ret = (1.0 + exit_value) / np.maximum(1.0 + entry, 1.0e-6) - 1.0
            low_between = np.nanmin(low_ret[:, entry_day : exit_day + 1], axis=1)
            drawdown = np.maximum(1.0 - (1.0 + low_between) / np.maximum(1.0 + entry, 1.0e-6), 0.0)
            amount_avg = np.nanmean(amount_rel[:, entry_day : exit_day + 1], axis=1)
            wait_penalty = float(UNIFIED_VALUE_WAIT_PENALTY) * math.sqrt((entry_day + 1.0) / max(float(forward_days), 1.0))
            hold_penalty = float(UNIFIED_VALUE_HOLD_PENALTY) * math.sqrt((exit_day - entry_day + 1.0) / max(float(forward_days), 1.0))
            liquidity_penalty = float(UNIFIED_VALUE_LIQUIDITY_PENALTY) * np.maximum(-amount_avg, 0.0)
            value = (
                ret
                - wait_penalty
                - hold_penalty
                - float(UNIFIED_VALUE_DRAWDOWN_PENALTY) * drawdown
                - liquidity_penalty
                - float(UNIFIED_VALUE_TRANSACTION_COST)
            )
            trade_return[:, entry_day, exit_day] = ret
            in_trade_drawdown[:, entry_day, exit_day] = drawdown
            amount_mean[:, entry_day, exit_day] = amount_avg
            candidate[:, entry_day, exit_day] = value
    return candidate, {
        "entry_ret": entry_ret,
        "exit_ret": exit_ret,
        "trade_return": trade_return,
        "in_trade_drawdown": in_trade_drawdown,
        "amount_mean": amount_mean,
    }


def _derive_unified_path_summary_numpy(path: np.ndarray) -> np.ndarray:
    values = np.asarray(path, dtype=np.float64)
    if values.ndim != 3 or values.shape[2] < 6:
        raise ValueError("OHLCVA path must have shape [batch, forward_days, 6]")
    forward_days = int(values.shape[1])
    high_ret = values[:, :, 1]
    low_ret = values[:, :, 2]
    close_ret = values[:, :, 3]
    volume_rel = values[:, :, 4]
    amount_rel = values[:, :, 5]
    max_ret = np.nanmax(high_ret, axis=1)
    min_ret = np.nanmin(low_ret, axis=1)
    final_ret = close_ret[:, -1]
    peak_idx = np.nanargmax(np.where(np.isfinite(high_ret), high_ret, -np.inf), axis=1)
    trough_idx = np.nanargmin(np.where(np.isfinite(low_ret), low_ret, np.inf), axis=1)
    min_after_peak = np.full(values.shape[0], np.nan, dtype=np.float64)
    for row_idx, pidx in enumerate(peak_idx):
        min_after_peak[row_idx] = np.nanmin(low_ret[row_idx, int(pidx) :])
    drawdown_after_peak = (1.0 + min_after_peak) / np.maximum(1.0 + max_ret, 1.0e-6) - 1.0
    time_above = np.nanmean(close_ret > 0.0, axis=1)
    time_below = np.nanmean(close_ret < 0.0, axis=1)
    volume_peak = np.nanmax(volume_rel, axis=1)
    amount_peak = np.nanmax(amount_rel, axis=1)
    price_volume_confirmation = np.nanmean(np.maximum(close_ret, 0.0) * np.maximum(amount_rel, 0.0), axis=1)
    path_volatility = np.nanstd(close_ret, axis=1)
    candidate, aux = _unified_trade_candidates_numpy(values)
    flat = candidate.reshape(candidate.shape[0], -1)
    best_flat = np.nanargmax(np.where(np.isfinite(flat), flat, -np.inf), axis=1)
    best_entry_idx = best_flat // forward_days
    best_exit_idx = best_flat % forward_days
    row = np.arange(values.shape[0])
    best_entry_price = aux["entry_ret"][row, best_entry_idx]
    best_exit_price = aux["exit_ret"][row, best_exit_idx]
    best_holding_days = best_exit_idx - best_entry_idx
    best_drawdown = aux["in_trade_drawdown"][row, best_entry_idx, best_exit_idx]
    entry_amount_condition = amount_rel[row, best_entry_idx]
    best_value = candidate[row, best_entry_idx, best_exit_idx]
    summary = np.column_stack(
        [
            max_ret,
            min_ret,
            final_ret,
            peak_idx + 1,
            trough_idx + 1,
            drawdown_after_peak,
            time_above,
            time_below,
            volume_peak,
            amount_peak,
            price_volume_confirmation,
            path_volatility,
            best_entry_idx + 1,
            best_entry_price,
            best_exit_idx + 1,
            best_exit_price,
            best_holding_days,
            best_entry_idx,
            best_drawdown,
            entry_amount_condition,
            best_value,
        ]
    )
    return summary.astype(np.float32, copy=False)


def _realize_predicted_plan_numpy(pred_summary: np.ndarray, true_path: np.ndarray, *, forward_days: int) -> dict[str, np.ndarray]:
    values = np.asarray(true_path, dtype=np.float64)
    summary = np.asarray(pred_summary, dtype=np.float64)
    columns = unified_path_summary_columns(forward_days)
    entry_day_idx = columns.index(f"best_entry_day_{int(forward_days)}d")
    entry_price_idx = columns.index(f"best_entry_price_{int(forward_days)}d")
    exit_day_idx = columns.index(f"best_exit_day_{int(forward_days)}d")
    exit_price_idx = columns.index(f"best_exit_price_{int(forward_days)}d")
    value_idx = columns.index(unified_path_value_column(forward_days))
    n = int(values.shape[0])
    filled = np.zeros(n, dtype=np.float32)
    realized_return = np.full(n, np.nan, dtype=np.float32)
    realized_drawdown = np.full(n, np.nan, dtype=np.float32)
    missed = np.zeros(n, dtype=np.float32)
    realized_entry_day = np.full(n, np.nan, dtype=np.float32)
    realized_exit_day = np.full(n, np.nan, dtype=np.float32)
    true_oracle = _derive_unified_path_summary_numpy(values)[:, value_idx]
    for row in range(n):
        entry_day = int(round(summary[row, entry_day_idx])) - 1
        exit_day = int(round(summary[row, exit_day_idx])) - 1
        if entry_day < 0 or exit_day <= entry_day or exit_day >= int(forward_days):
            missed[row] = 1.0 if true_oracle[row] > 0.0 else 0.0
            continue
        entry_limit = float(summary[row, entry_price_idx])
        exit_limit = float(summary[row, exit_price_idx])
        true_open = values[row, :, 0]
        true_high = values[row, :, 1]
        true_low = values[row, :, 2]
        true_close = values[row, :, 3]
        if not np.isfinite(entry_limit) or not np.isfinite(exit_limit):
            missed[row] = 1.0 if true_oracle[row] > 0.0 else 0.0
            continue
        if not np.isfinite(true_low[entry_day]) or true_low[entry_day] > entry_limit:
            missed[row] = 1.0 if true_oracle[row] > 0.0 else 0.0
            continue
        entry_fill = min(float(true_open[entry_day]), entry_limit) if np.isfinite(true_open[entry_day]) and true_open[entry_day] <= entry_limit else entry_limit
        if np.isfinite(true_open[exit_day]) and true_open[exit_day] >= exit_limit:
            exit_fill = float(true_open[exit_day])
        elif np.isfinite(true_high[exit_day]) and true_high[exit_day] >= exit_limit:
            exit_fill = exit_limit
        else:
            exit_fill = float(true_close[exit_day])
        if not np.isfinite(entry_fill) or not np.isfinite(exit_fill):
            missed[row] = 1.0 if true_oracle[row] > 0.0 else 0.0
            continue
        filled[row] = 1.0
        realized_entry_day[row] = float(entry_day + 1)
        realized_exit_day[row] = float(exit_day + 1)
        realized_return[row] = np.float32((1.0 + exit_fill) / max(1.0 + entry_fill, 1.0e-6) - 1.0 - float(UNIFIED_VALUE_TRANSACTION_COST))
        low_between = np.nanmin(true_low[entry_day : exit_day + 1])
        realized_drawdown[row] = np.float32(max(1.0 - (1.0 + low_between) / max(1.0 + entry_fill, 1.0e-6), 0.0))
    return {
        "realized_fill": filled,
        "realized_trade_return": realized_return,
        "realized_in_trade_drawdown": realized_drawdown,
        "missed_opportunity": missed,
        "realized_entry_day": realized_entry_day,
        "realized_exit_day": realized_exit_day,
    }


@torch.no_grad()
def _predict_split(
    *,
    model: nn.Module,
    dataset: SequencePathPackDataset,
    device: torch.device,
    output_dir: Path,
    split: str,
    batch_size: int,
    amp_enabled: bool,
    top_k: tuple[int, ...],
    write_predictions: bool,
    write_path_predictions: bool = True,
    direct_value_horizon: int = 0,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    pred_dir = output_dir / "predictions"
    if write_predictions:
        pred_dir.mkdir(parents=True, exist_ok=True)
    pred_path = pred_dir / f"{split}_predictions.csv"
    if write_predictions and pred_path.exists():
        pred_path.unlink()
    uses_derived_path_value = bool(getattr(model, "uses_derived_path_value", False))
    uses_direct_value = bool(getattr(model, "uses_direct_value", False))
    eval_forward_days = (
        max(1, min(int(direct_value_horizon), int(dataset.forward_days)))
        if uses_direct_value and int(direct_value_horizon) > 0
        else int(dataset.forward_days)
    )
    output_path_dim = 6 if bool(getattr(model, "uses_ohlcva_path", False)) else 4
    uses_ohlcva_aux_path = bool(getattr(model, "uses_ohlcva_aux_path", False))
    output_path_fields = PATH_OHLCVA_FIELDS if output_path_dim >= 6 else PATH_OHLC_FIELDS
    if uses_direct_value:
        summary_columns = derived_path_summary_columns(eval_forward_days, path_dim=4)
        value_column = value_column_for_path(eval_forward_days, path_dim=4)
    else:
        summary_columns = (
            derived_path_summary_columns(dataset.forward_days, path_dim=output_path_dim)
            if uses_derived_path_value
            else list(dataset.path_summary_columns)
        )
        value_column = value_column_for_path(dataset.forward_days, path_dim=output_path_dim) if uses_derived_path_value else str(dataset.value_column)
    pending_date: str | None = None
    pending_frames: list[pd.DataFrame] = []
    ic_rows_by_score: dict[str, list[dict[str, Any]]] = {"score": []}
    topk_daily_rows_by_score: dict[str, list[dict[str, Any]]] = {"score": []}
    topk_candidate_rows: list[dict[str, Any]] = []
    row_count = 0
    date_values: set[str] = set()
    target_sum = 0.0
    target_count = 0
    prediction_sums: dict[str, float] = {"score": 0.0}
    prediction_counts: dict[str, int] = {"score": 0}
    va_abs_sum = 0.0
    va_abs_count = 0
    va_delta_abs_sum = 0.0
    va_delta_abs_count = 0
    price_path_abs_sum = np.zeros(4, dtype=np.float64)
    price_path_abs_count = np.zeros(4, dtype=np.int64)
    observed_path_count = 0
    observed_path_total = 0
    tradable_path_count = 0
    tradable_path_total = 0
    entry_filled_count = 0
    entry_candidate_count = 0
    first_write = True
    model.eval()

    def flush_pending_date() -> None:
        nonlocal pending_date, pending_frames, ic_rows_by_score, topk_daily_rows_by_score, topk_candidate_rows
        if not pending_frames:
            pending_date = None
            return
        date_frame = pd.concat(pending_frames, ignore_index=True, copy=False)
        score_columns = ["score"]
        for optional_score in ["path_value_score", "residual_score"]:
            if optional_score in date_frame.columns:
                score_columns.append(optional_score)
        for score_col in score_columns:
            ic_rows_by_score.setdefault(score_col, []).extend(
                _daily_spearman(date_frame, score_col=score_col, target_col=value_column).to_dict("records")
            )
            score_frame = date_frame
            if score_col != "score":
                score_frame = date_frame.copy()
                score_frame["score"] = score_frame[score_col].to_numpy(copy=False)
            daily_rows = _topk_daily_rows(
                score_frame,
                top_k_values=top_k,
                forward_days=eval_forward_days,
                value_column=value_column,
            )
            for row in daily_rows:
                row["score_column"] = str(score_col)
            topk_daily_rows_by_score.setdefault(score_col, []).extend(daily_rows)
            if score_col == "score":
                candidate_rows = _topk_candidate_rows(
                    score_frame,
                    max_top_k=max(top_k, default=0),
                    forward_days=eval_forward_days,
                    value_column=value_column,
                )
                for row in candidate_rows:
                    row["score_column"] = "score"
                topk_candidate_rows.extend(candidate_rows)
        pending_frames = []
        pending_date = None

    for predict_batch_count, batch_indices in enumerate(
        _iter_index_batches(dataset, batch_size=int(batch_size), shuffle=False, seed=0),
        start=1,
    ):
        batch = dataset.get_batch(
            batch_indices,
            include_ohlcva_path=bool(output_path_dim >= 6 or uses_ohlcva_aux_path),
            include_richer_path=False,
            include_summary=not bool(uses_derived_path_value or uses_direct_value),
        )
        true_tradable_np = (
            batch["y_tradable_path"].numpy().astype(bool, copy=False)
            if batch.get("y_tradable_path") is not None
            else None
        )
        true_observed_np = (
            batch["y_observed_price_path"].numpy().astype(bool, copy=False)
            if batch.get("y_observed_price_path") is not None
            else None
        )
        entry_filled_np = batch["entry_filled"].numpy().astype(bool, copy=False)
        entry_open_raw_np = (
            batch["entry_open_raw"].numpy().astype(np.float32, copy=False)
            if batch.get("entry_open_raw") is not None
            else None
        )
        exit_close_raw_path_np = (
            batch["exit_close_raw_path"].numpy().astype(np.float32, copy=False)
            if batch.get("exit_close_raw_path") is not None
            else None
        )
        exit_sellable_path_np = (
            batch["exit_sellable_path"].numpy().astype(bool, copy=False)
            if batch.get("exit_sellable_path") is not None
            else None
        )
        entry_filled_count += int(entry_filled_np.sum())
        entry_candidate_count += int(entry_filled_np.size)
        if true_observed_np is not None:
            observed_path_count += int(true_observed_np.sum())
            observed_path_total += int(true_observed_np.size)
        if true_tradable_np is not None:
            tradable_path_count += int(true_tradable_np.sum())
            tradable_path_total += int(true_tradable_np.size)
        x, y_path, y_ohlcva_path, _y_richer_path, y_summary, _date_idx, symbol_idx = _batch_to_device(batch, device)
        target_path = y_ohlcva_path if output_path_dim >= 6 and y_ohlcva_path is not None else y_path
        trade_dates = list(batch["trade_date"])
        symbols = list(batch["symbol"])
        del batch
        pred_aux_np = None
        with torch.amp.autocast(device_type=device.type, enabled=amp_enabled):
            out = model(x, symbol_idx=symbol_idx)
        true_path_np = target_path.detach().float().cpu().numpy()
        if uses_direct_value:
            score_np = out["score"].detach().float().cpu().numpy()
            eval_true_path_np = true_path_np[:, :eval_forward_days, :4]
            true_summary_np = _derive_path_summary_numpy(
                eval_true_path_np,
                price_anchor=dataset.price_anchor,
                tradable_path=(true_tradable_np[:, :eval_forward_days] if true_tradable_np is not None else None),
            )
            pred_summary_np = np.full_like(true_summary_np, np.nan)
            pred_summary_np[:, summary_columns.index(value_column)] = score_np
            pred_path_np = np.empty((int(score_np.shape[0]), 0, 4), dtype=np.float32)
            residual_np = None
            path_value_score_np = score_np
        elif "score" in out:
            pred_path_np = out["future_path"].detach().float().cpu().numpy()
            pred_summary_np = out["path_summary"].detach().float().cpu().numpy()
            score_np = out["score"].detach().float().cpu().numpy()
            true_summary_np = y_summary.detach().float().cpu().numpy()
        else:
            pred_path_np = out["future_path"].detach().float().cpu().numpy()
            pred_summary_np = _derive_path_summary_numpy(pred_path_np, price_anchor=dataset.price_anchor)
            true_summary_np = _derive_path_summary_numpy(
                true_path_np,
                price_anchor=dataset.price_anchor,
                tradable_path=true_tradable_np,
            )
            path_value_score_np = pred_summary_np[:, summary_columns.index(value_column)]
            if "residual_score" in out:
                residual_np = out["residual_score"].detach().float().cpu().numpy()
                score_np = path_value_score_np + float(getattr(model, "residual_weight", 0.25)) * residual_np
            else:
                residual_np = None
                score_np = path_value_score_np
        if uses_ohlcva_aux_path and y_ohlcva_path is not None and "future_ohlcva_aux_path" in out:
            pred_aux_np = out["future_ohlcva_aux_path"].detach().float().cpu().numpy()
            true_ohlcva_np = y_ohlcva_path.detach().float().cpu().numpy()
            pred_va = pred_aux_np[:, :, 4:6]
            true_va = true_ohlcva_np[:, :, 4:6]
            finite_va = np.isfinite(true_va)
            if bool(finite_va.any()):
                va_abs_sum += float(np.abs(pred_va[finite_va] - true_va[finite_va]).sum())
                va_abs_count += int(finite_va.sum())
            pred_va_delta = np.diff(pred_va, axis=1)
            true_va_delta = np.diff(true_va, axis=1)
            finite_va_delta = np.isfinite(true_va_delta)
            if bool(finite_va_delta.any()):
                va_delta_abs_sum += float(np.abs(pred_va_delta[finite_va_delta] - true_va_delta[finite_va_delta]).sum())
                va_delta_abs_count += int(finite_va_delta.sum())
        if not uses_direct_value and int(pred_path_np.shape[1]) > 0:
            for field_idx in range(4):
                true_field = true_path_np[:, :, field_idx]
                pred_field = pred_path_np[:, :, field_idx]
                finite_field = np.isfinite(true_field)
                if bool(finite_field.any()):
                    price_path_abs_sum[field_idx] += float(
                        np.abs(pred_field[finite_field] - true_field[finite_field]).sum()
                    )
                    price_path_abs_count[field_idx] += int(finite_field.sum())
        rows: dict[str, Any] = {
            "trade_date": trade_dates,
            "symbol": symbols,
            "score": score_np,
        }
        if "score" not in out and residual_np is not None:
            rows["path_value_score"] = path_value_score_np
            rows["residual_score"] = residual_np
        for idx, col in enumerate(summary_columns):
            rows[f"true_{col}"] = true_summary_np[:, idx]
            rows[f"pred_{col}"] = pred_summary_np[:, idx]
        realized_cols: list[str] = []
        if output_path_dim >= 6:
            realized = _realize_predicted_plan_numpy(pred_summary_np, true_path_np, forward_days=dataset.forward_days)
            for col, values in realized.items():
                rows[col] = values
                realized_cols.append(col)
        elif uses_derived_path_value and not uses_direct_value:
            realized = _realize_path_value_v2_plan_numpy(
                pred_summary_np,
                true_path_np,
                forward_days=dataset.forward_days,
                price_anchor=dataset.price_anchor,
                tradable_path=true_tradable_np,
                entry_filled=entry_filled_np,
                entry_open_raw=entry_open_raw_np,
                exit_close_raw_path=exit_close_raw_path_np,
                exit_sellable_path=exit_sellable_path_np,
                execution_tail_days=int(dataset.execution_tail_days),
                terminal_recovery_fraction=float(dataset.terminal_recovery_fraction),
            )
            for col, values in realized.items():
                rows[col] = values
                realized_cols.append(col)
        chunk = pd.DataFrame(rows)
        if write_predictions:
            if bool(write_path_predictions) and not uses_direct_value:
                path_rows: dict[str, Any] = {}
                for day in range(dataset.forward_days):
                    for field_idx, field in enumerate(output_path_fields):
                        path_rows[f"true_{field}_ret_d{day + 1}"] = true_path_np[:, day, field_idx]
                        path_rows[f"pred_{field}_ret_d{day + 1}"] = pred_path_np[:, day, field_idx]
                chunk = pd.concat([chunk, pd.DataFrame(path_rows)], axis=1, copy=False)
            chunk.to_csv(pred_path, index=False, mode="w" if first_write else "a", header=first_write, encoding="utf-8-sig")
            first_write = False
        metric_frame = chunk[["trade_date", "symbol", "score", *[f"true_{c}" for c in summary_columns], *realized_cols]].copy()
        for optional_score in ["path_value_score", "residual_score"]:
            if optional_score in chunk.columns:
                metric_frame[optional_score] = chunk[optional_score].to_numpy(copy=False)
        metric_frame = metric_frame.rename(columns={f"true_{c}": c for c in summary_columns})
        row_count += int(len(metric_frame))
        date_values.update(str(item) for item in metric_frame["trade_date"].unique())
        target_values = pd.to_numeric(metric_frame[value_column], errors="coerce").to_numpy(dtype=np.float64, copy=False)
        target_mask = np.isfinite(target_values)
        target_sum += float(np.nansum(target_values[target_mask]))
        target_count += int(target_mask.sum())
        for score_col in [col for col in ["score", "path_value_score", "residual_score"] if col in metric_frame.columns]:
            prediction_values = pd.to_numeric(metric_frame[score_col], errors="coerce").to_numpy(dtype=np.float64, copy=False)
            prediction_mask = np.isfinite(prediction_values)
            prediction_sums[score_col] = prediction_sums.get(score_col, 0.0) + float(np.nansum(prediction_values[prediction_mask]))
            prediction_counts[score_col] = prediction_counts.get(score_col, 0) + int(prediction_mask.sum())
        for trade_date, date_frame in metric_frame.groupby("trade_date", sort=False):
            current_date = str(trade_date)
            if pending_date is not None and current_date != pending_date:
                flush_pending_date()
            pending_date = current_date
            pending_frames.append(date_frame.reset_index(drop=True))
        if "residual_np" in locals():
            del residual_np
        if "path_value_score_np" in locals():
            del path_value_score_np
        if pred_aux_np is not None:
            del pred_aux_np, true_ohlcva_np, pred_va, true_va, finite_va, pred_va_delta, true_va_delta, finite_va_delta
        del x, y_path, y_ohlcva_path, _y_richer_path, y_summary, target_path, symbol_idx, out, pred_path_np, pred_summary_np, score_np, true_path_np, true_summary_np, chunk, metric_frame
        del trade_dates, symbols, true_tradable_np, true_observed_np, entry_filled_np
        del entry_open_raw_np, exit_close_raw_path_np, exit_sellable_path_np
        if predict_batch_count % 100 == 0:
            gc.collect()
            _trim_working_set()
    flush_pending_date()
    ic = pd.DataFrame(ic_rows_by_score.get("score", []))
    daily_topk = pd.DataFrame(topk_daily_rows_by_score.get("score", []))
    topk_candidates = pd.DataFrame(topk_candidate_rows)
    topk = _aggregate_topk_daily_rows(topk_daily_rows_by_score.get("score", []))
    prediction_csv = str(pred_path.resolve()) if write_predictions else ""
    metrics = {
        "split": split,
        "row_count": int(row_count),
        "date_count": int(len(date_values)),
        "rank_ic_mean": float(ic["rank_ic"].mean()) if not ic.empty else np.nan,
        "rank_ic_median": float(ic["rank_ic"].median()) if not ic.empty else np.nan,
        "rank_ic_positive_day_rate": float((ic["rank_ic"] > 0).mean()) if not ic.empty else np.nan,
        "value_column": str(value_column),
        "score_column": "score",
        "target_mean": float(target_sum / max(target_count, 1)),
        "prediction_mean": float(prediction_sums.get("score", 0.0) / max(prediction_counts.get("score", 0), 1)),
        "va_level_mae": float(va_abs_sum / va_abs_count) if va_abs_count > 0 else np.nan,
        "va_delta_mae": float(va_delta_abs_sum / va_delta_abs_count) if va_delta_abs_count > 0 else np.nan,
        "path_mae": float(price_path_abs_sum.sum() / price_path_abs_count.sum())
        if int(price_path_abs_count.sum()) > 0
        else np.nan,
        **{
            f"path_{field}_mae": (
                float(price_path_abs_sum[field_idx] / price_path_abs_count[field_idx])
                if int(price_path_abs_count[field_idx]) > 0
                else np.nan
            )
            for field_idx, field in enumerate(PATH_OHLC_FIELDS)
        },
        "entry_fill_rate": float(entry_filled_count / entry_candidate_count) if entry_candidate_count else np.nan,
        "observed_price_path_rate": float(observed_path_count / observed_path_total) if observed_path_total else np.nan,
        "tradable_path_rate": float(tradable_path_count / tradable_path_total) if tradable_path_total else np.nan,
        "prediction_csv": prediction_csv,
    }
    diagnostics: dict[str, dict[str, Any]] = {}
    for score_col, rows_for_score in ic_rows_by_score.items():
        score_ic = pd.DataFrame(rows_for_score)
        score_topk = _aggregate_topk_daily_rows(topk_daily_rows_by_score.get(score_col, []))
        diagnostics[score_col] = {
            "rank_ic_mean": float(score_ic["rank_ic"].mean()) if not score_ic.empty else np.nan,
            "rank_ic_median": float(score_ic["rank_ic"].median()) if not score_ic.empty else np.nan,
            "rank_ic_positive_day_rate": float((score_ic["rank_ic"] > 0).mean()) if not score_ic.empty else np.nan,
            "prediction_mean": float(prediction_sums.get(score_col, 0.0) / max(prediction_counts.get(score_col, 0), 1)),
            "topk": score_topk.to_dict("records"),
        }
    metrics["score_diagnostics"] = diagnostics
    return ic, topk, daily_topk, topk_candidates, metrics


def _write_report(
    output_dir: Path,
    *,
    summary: Mapping[str, Any],
    split_metrics: pd.DataFrame,
    topk: pd.DataFrame,
    baseline_summary: Mapping[str, Any],
) -> str:
    forward_days = int(summary.get("forward_days", DEFAULT_FORWARD_DAYS) or DEFAULT_FORWARD_DAYS)
    direct_value_horizon = int(summary.get("direct_value_horizon", 0) or 0)
    report_days = direct_value_horizon if direct_value_horizon > 0 else forward_days
    suffix = f"{report_days}d"
    value_col = str(summary.get("value_column", "") or path_value_column(report_days))
    max_col = f"future_max_return_{suffix}"
    final_col = f"future_final_return_{suffix}"
    best_exit_col = f"best_exit_close_return_{suffix}"
    uses_direct_value = bool(dict(summary.get("model", {}) or {}).get("uses_direct_value", False))
    lines: list[str] = []
    lines.append("# QDP v2 Sequence Path Model")
    lines.append("")
    lines.append("## Method")
    lines.append("")
    if uses_direct_value:
        lines.append(
            f"This model reads past 100-day multi-channel sequences from QDP v2 and directly predicts the "
            f"{report_days}-day path-value ranking score. It does not emit a future OHLC path."
        )
        lines.append("The supervised target and Top-K diagnostics are derived from true future OHLC labels.")
    else:
        lines.append(
            f"This model reads past 100-day multi-channel sequences from QDP v2 and predicts future {forward_days}-day OHLC paths. "
            "Path summaries and ranking values are derived from the predicted path when using a path-value model."
        )
    summary_profile = str(summary.get("loss_weights", {}).get("summary_profile", ""))
    if summary_profile == SUMMARY_LOSS_PROFILE_MULTI_HORIZON_OHLC:
        lines.append("Summary loss uses multi-horizon OHLC-derived constraints while the model output remains the future OHLC path.")
    elif summary_profile == SUMMARY_LOSS_PROFILE_MULTI_HORIZON_OHLC_NO60:
        lines.append(
            "Summary loss uses the same OHLC-derived constraints as multi-horizon summary v2, excluding the full-horizon window."
        )
    early = dict(summary.get("early_stopping", {}) or {})
    if early.get("metric"):
        lines.append(
            f"Early stopping monitors {early.get('metric', 'validation_rank_ic_mean')} with "
            f"mode={early.get('mode', '')}, patience={int(early.get('patience', 0))}, "
            f"min_complete_epochs={int(early.get('minimum_complete_epochs', 1))}, "
            f"min_delta={float(early.get('min_delta', 0.0)):.4g}."
        )
    elif str(summary.get("checkpoint_policy", "")) == "final_epoch":
        lines.append("The epoch count is fixed in advance; OOS is evaluated once after the final checkpoint is saved.")
    lines.append("")
    lines.append("## Split Metrics")
    lines.append("")
    for row in split_metrics.to_dict("records"):
        lines.append(
            f"- {row['split']}: rows={int(row['row_count']):,}, "
            f"rank_ic_mean={row['rank_ic_mean']:.4f}, "
            f"rank_ic_positive_day_rate={row['rank_ic_positive_day_rate']:.2%}, "
            f"target_mean={row['target_mean'] * 100:.2f}%"
        )
    lines.append("")
    lines.append("## Top-K")
    lines.append("")
    if not topk.empty:
        for row in topk.sort_values(["split", "top_k"]).to_dict("records"):
            detail = (
                f"- {row['split']} top{int(row['top_k'])}: "
                f"path_value_alpha={row[f'alpha_{value_col}'] * 100:.2f}%, "
                f"max_alpha={row[f'alpha_{max_col}'] * 100:.2f}%, "
                f"final_alpha={row[f'alpha_{final_col}'] * 100:.2f}%, "
            )
            if f"alpha_{best_exit_col}" in row:
                detail += f"best_exit_alpha={row[f'alpha_{best_exit_col}'] * 100:.2f}%, "
            if "selected_best_exit_day_mean" in row:
                detail += f"best_exit_day={row['selected_best_exit_day_mean']:.1f}, "
            detail += f"hit10={row['selected_hit_10pct_rate']:.2%}, loss5={row['selected_loss_5pct_rate']:.2%}"
            lines.append(detail)
    if baseline_summary:
        lines.append("")
        lines.append("## Baseline")
        lines.append("")
        lines.append(f"- matching feature baseline summary: `{baseline_summary.get('output_dir', '')}`")
        for item in list(baseline_summary.get("split_metrics", []) or []):
            if str(item.get("split", "")) in {"validation", "test"}:
                lines.append(
                    f"- baseline {item.get('split')}: rank_ic_mean={float(item.get('rank_ic_mean', float('nan'))):.4f}, "
                    f"positive_day_rate={float(item.get('rank_ic_positive_day_rate', float('nan'))):.2%}"
                )
    lines.append("")
    lines.append("## Boundaries")
    lines.append("")
    if str(summary.get("model", {}).get("uses_symbol_embedding", "")).lower() == "true":
        lines.append("- Symbol embedding is used as a controlled identity-memory experiment.")
    else:
        lines.append("- No symbol id or symbol embedding is used.")
    lines.append("- Path types are explanation labels; the model trains on numeric paths and path value.")
    lines.append(
        f"- Path-value gradient profile: `{summary.get('path_value_gradient_profile', PATH_VALUE_GRADIENT_PROFILE_SMOOTH)}`."
    )
    ranking_contract = dict(summary.get("ranking_contract", {}) or {})
    lines.append(
        f"- Rank training profile: `{ranking_contract.get('profile', RANK_TRAINING_PROFILE_LOCAL_CHUNK)}`; "
        f"separate path/rank batches={bool(ranking_contract.get('path_and_rank_batches_separate', False))}."
    )
    if str(summary.get("evaluation_mode", "")) == EVALUATION_MODE_FIXED_OOS:
        lines.append("- OOS was not read during training and did not select the checkpoint.")
    elif str(summary.get("evaluation_mode", "")) == EVALUATION_MODE_DEVELOPMENT:
        lines.append(
            "- Development total loss alone selects the checkpoint; Top-K diagnostics score the complete candidate index and do not select within-run epochs."
        )
    else:
        lines.append("- This is a first sequence model; failed improvement over baseline should be diagnosed, not hidden by test-set tuning.")
    path = output_dir / "sequence_path_training_report.md"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return str(path.resolve())


@dataclass(frozen=True)
class TrainConfig:
    pack_manifest: Path
    output_root: Path
    run_tag: str
    epochs: int
    batch_size: int
    model_type: str
    hidden_dim: int
    layers: int
    dropout: float
    symbol_embedding_dim: int
    learning_rate: float
    weight_decay: float
    path_loss_weight: float
    path_loss_profile: str
    summary_loss_weight: float
    richer_loss_weight: float
    price_delta_loss_weight: float
    va_level_loss_weight: float
    va_delta_loss_weight: float
    value_loss_weight: float
    rank_loss_weight: float
    residual_score_weight: float
    residual_penalty_weight: float
    summary_loss_profile: str
    input_channel_profile: str
    direct_value_horizon: int
    rank_max_per_side: int
    device: str
    amp: bool
    seed: int
    top_k: tuple[int, ...]
    max_samples_per_split: int
    prediction_mode: str
    early_stopping_patience: int
    early_stopping_min_delta: float
    evaluation_mode: str = EVALUATION_MODE_STANDARD
    path_value_gradient_profile: str = PATH_VALUE_GRADIENT_PROFILE_SMOOTH
    rank_training_profile: str = RANK_TRAINING_PROFILE_LOCAL_CHUNK
    rank_batch_size: int = 512
    rank_interval: int = 4
    prefetch_batches: int = 1
    early_stopping_metric: str = ""
    early_stopping_mode: str = ""
    min_complete_epochs: int = 1
    development_contract: Path | None = None


def _resolved_training_config(config: TrainConfig, *, evaluation_mode: str) -> dict[str, Any]:
    excluded = {"pack_manifest", "output_root", "run_tag"}
    payload = {key: value for key, value in config.__dict__.items() if key not in excluded}
    payload["evaluation_mode"] = str(evaluation_mode)
    return json.loads(json.dumps(payload, default=_json_default))


def _validate_evaluation_mode(config: TrainConfig) -> str:
    if int(getattr(config, "epochs", 1)) <= 0:
        raise ValueError("epochs must be positive")
    if int(getattr(config, "max_samples_per_split", 0)) < 0:
        raise ValueError("max_samples_per_split must be non-negative")
    if int(getattr(config, "early_stopping_patience", 0)) < 0:
        raise ValueError("early_stopping_patience must be non-negative")
    if float(getattr(config, "early_stopping_min_delta", 0.0)) < 0.0:
        raise ValueError("early_stopping_min_delta must be non-negative")
    if int(getattr(config, "min_complete_epochs", 1)) < 1:
        raise ValueError("min_complete_epochs must be at least 1")
    if int(getattr(config, "min_complete_epochs", 1)) > int(getattr(config, "epochs", 1)):
        raise ValueError("min_complete_epochs cannot exceed epochs")
    mode = str(config.evaluation_mode or EVALUATION_MODE_STANDARD).strip().lower()
    if mode not in EVALUATION_MODES:
        raise ValueError(f"evaluation_mode must be one of {EVALUATION_MODES}")
    if mode == EVALUATION_MODE_FIXED_OOS and int(config.early_stopping_patience) != 0:
        raise ValueError("fixed_oos requires early_stopping_patience=0 because OOS cannot select checkpoints")
    metric = str(getattr(config, "early_stopping_metric", "") or "").strip().lower()
    stopping_mode = str(getattr(config, "early_stopping_mode", "") or "").strip().lower()
    if stopping_mode and stopping_mode not in EARLY_STOPPING_MODES:
        raise ValueError(f"early_stopping_mode must be one of {EARLY_STOPPING_MODES}")
    if mode == EVALUATION_MODE_DEVELOPMENT:
        if int(config.max_samples_per_split) != 0:
            raise ValueError("development requires max_samples_per_split=0 so every fold uses full training data")
        if int(config.early_stopping_patience) <= 0:
            raise ValueError("development requires early_stopping_patience > 0")
        if metric != EARLY_STOPPING_METRIC_DEVELOPMENT_TOTAL_LOSS:
            raise ValueError(
                "development requires early_stopping_metric=development_total_loss"
            )
        if stopping_mode != EARLY_STOPPING_MODE_MIN:
            raise ValueError("development requires early_stopping_mode=min")
    elif mode == EVALUATION_MODE_STANDARD:
        if metric and metric != EARLY_STOPPING_METRIC_VALIDATION_RANK_IC:
            raise ValueError("standard early stopping metric must be validation_rank_ic_mean")
        if stopping_mode and stopping_mode != EARLY_STOPPING_MODE_MAX:
            raise ValueError("standard early_stopping_mode must be max")
    elif metric or stopping_mode:
        raise ValueError("fixed_oos does not accept an early-stopping metric or mode")
    return mode


def _canonical_json_sha256(payload: Any) -> str:
    encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=_json_default)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _validated_development_contract(path: Path) -> dict[str, Any]:
    contract_path = Path(path).resolve()
    payload = json.loads(contract_path.read_text(encoding="utf-8"))
    declared = str(payload.get("contract_sha256", "") or "")
    semantic_payload = dict(payload)
    semantic_payload.pop("contract_sha256", None)
    computed = _canonical_json_sha256(semantic_payload)
    if not declared or computed != declared:
        raise ValueError("development contract semantic sha256 does not match contract_sha256")
    protocol = dict(payload.get("development_protocol", {}) or {})
    if dict(protocol.get("split_roles", {}) or {}) != {"fit": "train", "evaluation": "development"}:
        raise ValueError("development contract must declare train/development split roles")
    early = dict(payload.get("early_stopping", {}) or {})
    if (
        str(early.get("metric", "")) != EARLY_STOPPING_METRIC_DEVELOPMENT_TOTAL_LOSS
        or str(early.get("mode", "")) != EARLY_STOPPING_MODE_MIN
        or int(early.get("minimum_complete_epochs", 0)) < 1
        or not bool(early.get("restore_best_checkpoint", False))
    ):
        raise ValueError("development contract early-stopping semantics are incompatible")
    return {
        "contract_id": str(payload.get("contract_id", "")),
        "path": str(contract_path),
        "contract_sha256": declared,
        "contract_file_sha256": _file_sha256(contract_path),
    }


def _development_split_storage_name(manifest: Mapping[str, Any]) -> str:
    contract = dict(manifest.get("development_walkforward", {}) or {})
    split_roles = dict(contract.get("split_roles", {}) or {})
    if split_roles != {"fit": "train", "evaluation": "development"}:
        raise ValueError("development fold must declare split_roles fit=train and evaluation=development")
    index_paths = [Path(str(manifest.get("sample_index_path", "") or ""))]
    if str(manifest.get("candidate_index_path", "") or ""):
        index_paths.append(Path(str(manifest["candidate_index_path"])))
    split_sets = [
        set(pd.read_parquet(str(path), columns=["split"])["split"].astype(str).unique())
        for path in index_paths
    ]
    if split_sets and all("development" in values for values in split_sets):
        return "development"
    # Explicit compatibility only: the manifest's canonical evaluation role is
    # development, while both stored indexes use the retired physical name oos.
    if split_sets and all("oos" in values for values in split_sets):
        return "oos"
    raise ValueError("development fold indexes must contain the canonical development split")


def _validate_fixed_oos_split_contract(
    *,
    manifest: Mapping[str, Any],
    train_ds: SequencePathPackDataset,
    oos_ds: SequencePathPackDataset,
) -> None:
    train_forward_days = int(train_ds.forward_days)
    if train_forward_days <= 0:
        raise ValueError("fixed_oos forward_days must be positive")
    if int(oos_ds.forward_days) != train_forward_days:
        raise ValueError("fixed_oos train and OOS forward_days must match")
    train_date_idx = _validated_sample_date_indices(
        train_ds.sample_index,
        context="fixed_oos train sample index",
    )
    oos_date_idx = _validated_sample_date_indices(
        oos_ds.sample_index,
        context="fixed_oos oos sample index",
    )
    oos_start_idx = int(oos_date_idx.min())
    label_end_idx = train_date_idx + train_forward_days
    overlap_count = int((label_end_idx >= oos_start_idx).sum())
    if overlap_count:
        raise ValueError(
            "fixed_oos requires every training label to end before OOS: "
            f"label_overlap_count={overlap_count}"
        )
    contract = dict(manifest.get("purged_walkforward", {}) or {})
    if not contract:
        raise ValueError("fixed_oos requires a purged_walkforward contract")
    declared_start = int(contract.get("oos_start_date_idx", -1))
    if declared_start != oos_start_idx:
        raise ValueError("fixed_oos manifest oos_start_date_idx does not match the sample index")
    if int(contract.get("label_overlap_count", -1)) != 0:
        raise ValueError("fixed_oos manifest does not declare label_overlap_count=0")
    normalization = dict(manifest.get("normalization", {}) or {})
    if normalization.get("fit_scope") != "feature_dates_before_oos_start":
        raise ValueError("fixed_oos normalization must use feature_dates_before_oos_start")
    if str(normalization.get("fit_date_end_exclusive", "")) != str(contract.get("oos_start", "")):
        raise ValueError("fixed_oos normalization cutoff does not match OOS start")

    # The walk-forward builder binds the sample index, normalization, panels,
    # masks, and split/purge metadata into this immutable digest.  Recompute it
    # here as well so a direct training CLI call cannot bypass orchestration QA.
    from daily_research.path_policy.seq100_walkforward import (
        _validated_fold_training_contract,
        _validated_source_view_provenance,
    )

    _validated_fold_training_contract(manifest)
    if str(contract.get("method", "")) == "expanding_train_fixed_oos":
        _validated_source_view_provenance(manifest)


def _validate_development_split_contract(
    *,
    manifest: Mapping[str, Any],
    train_ds: SequencePathPackDataset,
    development_ds: SequencePathPackDataset,
) -> None:
    dependency_days = int(
        dict(manifest.get("development_walkforward", {}) or {}).get(
            "max_label_dependency_days",
            manifest.get("max_label_dependency_days", train_ds.forward_days),
        )
    )
    if dependency_days <= 0:
        raise ValueError("development max_label_dependency_days must be positive")
    if int(development_ds.forward_days) != int(train_ds.forward_days):
        raise ValueError("development train and evaluation forward_days must match")
    train_date_idx = _validated_sample_date_indices(
        train_ds.sample_index,
        context="development train sample index",
    )
    development_date_idx = _validated_sample_date_indices(
        development_ds.sample_index,
        context="development supervised sample index",
    )
    development_start_idx = int(development_date_idx.min())
    dependency_column = next(
        (
            name
            for name in (
                "dependency_end_date_idx",
                "max_label_dependency_date_idx",
                "label_dependency_end_date_idx",
            )
            if name in train_ds.sample_index.columns
        ),
        "",
    )
    if dependency_column:
        dependency_end = pd.to_numeric(
            train_ds.sample_index[dependency_column], errors="coerce"
        ).to_numpy(dtype=np.float64, na_value=np.nan)
        if not bool(np.isfinite(dependency_end).all()):
            raise ValueError(f"development train {dependency_column} must contain finite values")
    else:
        dependency_end = train_date_idx + dependency_days
    overlap_count = int((dependency_end >= development_start_idx).sum())
    if overlap_count:
        raise ValueError(
            "development requires every training label dependency to end before development: "
            f"label_overlap_count={overlap_count}"
        )
    contract = dict(manifest.get("development_walkforward", {}) or {})
    declared_start = int(
        contract.get("development_start_date_idx", contract.get("oos_start_date_idx", -1))
    )
    if declared_start != development_start_idx:
        raise ValueError("development manifest start date index does not match supervised index")
    if int(
        contract.get(
            "label_dependency_overlap_count",
            contract.get("label_overlap_count", contract.get("dependency_overlap_count", -1)),
        )
    ) != 0:
        raise ValueError("development manifest does not declare label dependency overlap count=0")
    normalization = dict(manifest.get("normalization", {}) or {})
    expected_start = str(contract.get("development_start", contract.get("oos_start", "")) or "")
    if normalization.get("fit_scope") not in {
        "feature_dates_before_development_start",
        "feature_dates_before_oos_start",
    }:
        raise ValueError("development normalization must use only feature dates before development start")
    if str(normalization.get("fit_date_end_exclusive", "")) != expected_start:
        raise ValueError("development normalization cutoff does not match development start")
    from daily_research.path_policy.seq100_walkforward import (
        _validated_development_fold_training_contract,
    )

    _validated_development_fold_training_contract(manifest)


VALIDATION_LOSS_KEYS = (
    "loss",
    "path_loss",
    "summary_loss",
    "richer_loss",
    "price_delta_loss",
    "va_level_loss",
    "va_delta_loss",
    "value_loss",
    "rank_loss",
    "residual_penalty",
)


@torch.no_grad()
def _evaluate_development_loss(
    *,
    model: nn.Module,
    dataset: SequencePathPackDataset,
    device: torch.device,
    config: TrainConfig,
    amp_enabled: bool,
    path_value_gradient_profile: str,
    rank_training_profile: str,
) -> dict[str, Any]:
    """Evaluate the configured mathematical objective over every supervised row."""

    uses_direct_value = bool(getattr(model, "uses_direct_value", False))
    uses_derived_path_value = bool(getattr(model, "uses_derived_path_value", False))
    uses_ohlcva_path = bool(getattr(model, "uses_ohlcva_path", False))
    uses_ohlcva_aux_path = bool(getattr(model, "uses_ohlcva_aux_path", False))
    uses_richer_path = bool(getattr(model, "uses_richer_path", False))
    totals = torch.zeros(len(VALIDATION_LOSS_KEYS), device=device, dtype=torch.float64)
    sample_count = 0
    batch_count = 0
    started_at = time.perf_counter()
    was_training = bool(model.training)
    model.eval()
    try:
        batches = DateGroupedBatchSampler(
            dataset.sample_index,
            batch_size=int(config.batch_size),
            shuffle=False,
            seed=0,
        )
        for indices in batches:
            batch = dataset.get_batch(
                indices,
                include_ohlcva_path=bool(uses_ohlcva_path or uses_ohlcva_aux_path),
                include_richer_path=uses_richer_path,
                include_summary=not bool(uses_derived_path_value or uses_direct_value),
                include_metadata=False,
            )
            y_tradable_path = (
                batch["y_tradable_path"].to(device, non_blocking=device.type == "cuda")
                if batch.get("y_tradable_path") is not None
                else None
            )
            x, y_path, y_ohlcva_path, y_richer_path, y_summary, date_idx, symbol_idx = _batch_to_device(
                batch, device
            )
            target_path = y_ohlcva_path if uses_ohlcva_path and y_ohlcva_path is not None else y_path
            with torch.amp.autocast(device_type=device.type, enabled=amp_enabled):
                outputs = model(x, symbol_idx=symbol_idx)
                _loss, parts = _compute_loss(
                    outputs,
                    target_path,
                    y_summary,
                    date_idx,
                    y_ohlcva_path=y_ohlcva_path,
                    y_richer_path=y_richer_path,
                    value_index=dataset.value_index,
                    path_weight=float(config.path_loss_weight),
                    path_loss_profile=str(config.path_loss_profile),
                    summary_weight=float(config.summary_loss_weight),
                    value_weight=float(config.value_loss_weight),
                    rank_weight=float(config.rank_loss_weight),
                    richer_weight=float(config.richer_loss_weight),
                    rank_max_per_side=int(config.rank_max_per_side),
                    price_delta_weight=float(config.price_delta_loss_weight),
                    va_level_weight=float(config.va_level_loss_weight),
                    va_delta_weight=float(config.va_delta_loss_weight),
                    residual_weight=float(config.residual_score_weight),
                    residual_penalty_weight=float(config.residual_penalty_weight),
                    price_anchor=dataset.price_anchor,
                    summary_loss_profile=str(config.summary_loss_profile),
                    direct_value_horizon=int(config.direct_value_horizon),
                    path_value_gradient_profile=path_value_gradient_profile,
                    rank_training_profile=rank_training_profile,
                    y_tradable_path=y_tradable_path,
                    return_tensor_parts=True,
                )
            current_count = int(x.shape[0])
            totals.add_(
                torch.stack([parts[key].to(dtype=torch.float64) for key in VALIDATION_LOSS_KEYS]),
                alpha=float(current_count),
            )
            sample_count += current_count
            batch_count += 1
            del batch, x, y_path, y_ohlcva_path, y_richer_path, y_summary, y_tradable_path
            del date_idx, symbol_idx, target_path, outputs, _loss, parts
    finally:
        model.train(was_training)
    if sample_count != len(dataset):
        raise RuntimeError(
            f"development loss evaluation did not cover all supervised rows: {sample_count} != {len(dataset)}"
        )
    means = totals.detach().cpu().numpy() / max(sample_count, 1)
    result = {
        key: float(value)
        for key, value in zip(VALIDATION_LOSS_KEYS, means.tolist(), strict=True)
    }
    if not np.isfinite(result["loss"]):
        raise ValueError("development_total_loss must be finite")
    result.update(
        {
            "sample_count": int(sample_count),
            "batch_count": int(batch_count),
            "seconds": float(time.perf_counter() - started_at),
            "aggregation": "sample_weighted_batch_mean_all_supervised_rows",
        }
    )
    return result


def train_sequence_path_model(config: TrainConfig) -> dict[str, Any]:
    evaluation_mode = _validate_evaluation_mode(config)
    path_value_gradient_profile = _normalize_path_value_gradient_profile(config.path_value_gradient_profile)
    rank_training_profile = _normalize_rank_training_profile(config.rank_training_profile)
    if rank_training_profile == RANK_TRAINING_PROFILE_GLOBAL_TAIL_512:
        if int(config.rank_batch_size) != 512:
            raise ValueError("global_tail_512 requires rank_batch_size=512")
        if int(config.rank_interval) <= 0:
            raise ValueError("global_tail_512 requires rank_interval > 0")
    _set_seed(config.seed)
    manifest = json.loads(Path(config.pack_manifest).read_text(encoding="utf-8"))
    development_contract_binding: dict[str, Any] = {}
    if evaluation_mode == EVALUATION_MODE_DEVELOPMENT:
        manifest_contract = dict(manifest.get("research_contract", {}) or {})
        contract_path_raw = (
            config.development_contract
            or manifest_contract.get("path")
        )
        if not contract_path_raw:
            raise ValueError("development requires an approved development_contract path")
        development_contract_binding = _validated_development_contract(Path(contract_path_raw))
        for field in ("contract_id", "contract_sha256", "contract_file_sha256"):
            if str(manifest_contract.get(field, "") or "") != str(
                development_contract_binding[field]
            ):
                raise ValueError(f"manifest research_contract {field} does not match approved contract")
    fold_training_contract = (
        manifest.get("development_fold_training_contract")
        if evaluation_mode == EVALUATION_MODE_DEVELOPMENT
        else manifest.get("fold_training_contract")
    )
    if fold_training_contract is not None and not isinstance(fold_training_contract, Mapping):
        raise ValueError("fold_training_contract must be a JSON object")
    device = _resolve_device(config.device)
    amp_enabled = bool(config.amp and device.type == "cuda")
    output_dir = config.output_root / f"{config.run_tag}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    output_dir.mkdir(parents=True, exist_ok=True)
    progress_path = output_dir / "progress.json"
    _write_json(progress_path, {"status": "loading_datasets", "updated_at": _now()})
    train_ds = SequencePathPackDataset(
        manifest,
        split="train",
        max_samples=int(config.max_samples_per_split),
        input_channel_profile=str(config.input_channel_profile),
    )
    if len(train_ds) == 0:
        raise ValueError("training split is empty")
    development_ds = None
    development_scoring_ds = None
    if evaluation_mode == EVALUATION_MODE_FIXED_OOS:
        val_ds = None
        test_ds = None
        oos_ds = SequencePathPackDataset(
            manifest,
            split="oos",
            max_samples=0,
            input_channel_profile=str(config.input_channel_profile),
        )
        if len(oos_ds) == 0:
            raise ValueError("oos split is empty")
        _validate_fixed_oos_split_contract(manifest=manifest, train_ds=train_ds, oos_ds=oos_ds)
    elif evaluation_mode == EVALUATION_MODE_DEVELOPMENT:
        val_ds = None
        test_ds = None
        oos_ds = None
        development_storage_split = _development_split_storage_name(manifest)
        development_ds = SequencePathPackDataset(
            manifest,
            split=development_storage_split,
            max_samples=0,
            input_channel_profile=str(config.input_channel_profile),
        )
        development_scoring_ds = SequencePathPackDataset(
            manifest,
            split=development_storage_split,
            max_samples=0,
            input_channel_profile=str(config.input_channel_profile),
            index_role="candidate",
        )
        if len(development_ds) == 0:
            raise ValueError("development supervised split is empty")
        if len(development_scoring_ds) == 0:
            raise ValueError("development candidate split is empty")
        if not bool(manifest.get("dependency_padding_complete", False)):
            raise ValueError("development requires complete forward+execution-tail date padding")
        if not bool(development_scoring_ds.has_deterministic_execution):
            raise ValueError(
                "development execution evaluator requires entry_open_raw, exit_close_raw, and exit_sellable"
            )
        if int(development_scoring_ds.execution_tail_days) <= 0:
            raise ValueError("development requires a positive execution_tail_days retry window")
        _validate_development_split_contract(
            manifest=manifest,
            train_ds=train_ds,
            development_ds=development_ds,
        )
    else:
        oos_ds = None
        val_ds = SequencePathPackDataset(
            manifest,
            split="validation",
            max_samples=0,
            input_channel_profile=str(config.input_channel_profile),
        )
        test_ds = SequencePathPackDataset(
            manifest,
            split="test",
            max_samples=0,
            input_channel_profile=str(config.input_channel_profile),
        )
        if len(val_ds) == 0 or len(test_ds) == 0:
            raise ValueError("standard evaluation requires non-empty validation and test splits")
    if str(config.model_type) in (OHLCVA_MODEL_TYPES | OHLCVA_AUX_MODEL_TYPES) and not bool(train_ds.has_ohlcva_path):
        raise ValueError(f"model_type={config.model_type} requires a pack with label_arrays.future_ohlcva_path")
    model = SequencePathModel(
        input_dim=train_ds.input_dim,
        hidden_dim=int(config.hidden_dim),
        layers=int(config.layers),
        forward_days=train_ds.forward_days,
        summary_dim=len(train_ds.path_summary_columns),
        dropout=float(config.dropout),
        model_type=str(config.model_type),
        symbol_count=int(train_ds.symbol_count),
        symbol_embedding_dim=int(config.symbol_embedding_dim),
        richer_path_dim=int(train_ds.richer_path_dim),
    ).to(device)
    model.residual_weight = float(config.residual_score_weight)
    uses_direct_value = bool(getattr(model, "uses_direct_value", False))
    uses_derived_path_value = bool(getattr(model, "uses_derived_path_value", False))
    uses_ohlcva_path = bool(getattr(model, "uses_ohlcva_path", False))
    uses_ohlcva_aux_path = bool(getattr(model, "uses_ohlcva_aux_path", False))
    uses_richer_path = bool(getattr(model, "uses_richer_path", False))
    if rank_training_profile == RANK_TRAINING_PROFILE_GLOBAL_TAIL_512 and (
        not uses_derived_path_value or uses_direct_value or uses_ohlcva_path
    ):
        raise ValueError("global_tail_512 currently requires a price-only derived path-value model")
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(config.learning_rate), weight_decay=float(config.weight_decay))
    scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)
    best_val_ic = -1e9
    best_development_loss = float("inf")
    best_epoch = 0
    epochs_without_improvement = 0
    best_path = output_dir / (
        "final_model.pt" if evaluation_mode == EVALUATION_MODE_FIXED_OOS else "best_model.pt"
    )
    history: list[dict[str, Any]] = []
    global_tail_targets: np.ndarray | None = None
    prior_epoch_scores: np.ndarray | None = None
    if rank_training_profile == RANK_TRAINING_PROFILE_GLOBAL_TAIL_512:
        _write_json(progress_path, {"status": "loading_global_tail_targets", "updated_at": _now()})
        global_tail_targets = train_ds.path_value_targets()
        if not bool(np.isfinite(global_tail_targets).any()):
            raise ValueError("global_tail_512 requires finite path-value targets")
    for epoch in range(1, int(config.epochs) + 1):
        epoch_started_at = time.perf_counter()
        _write_json(progress_path, {"status": "training", "epoch": epoch, "updated_at": _now()})
        model.train()
        loss_total_keys = (
            "loss",
            "path_loss",
            "summary_loss",
            "richer_loss",
            "price_delta_loss",
            "va_level_loss",
            "va_delta_loss",
            "value_loss",
            "rank_loss",
            "residual_penalty",
        )
        # Keep scalar diagnostics on-device for the whole epoch.  The former ten
        # ``.cpu().item()`` calls per batch serialized the CUDA stream.
        loss_totals_tensor = torch.zeros(len(loss_total_keys), device=device, dtype=torch.float64)
        rank_loss_total_tensor = torch.zeros((), device=device, dtype=torch.float64)
        batch_count = 0
        sample_count = 0
        rank_batch_count = 0
        rank_sample_count = 0
        if rank_training_profile == RANK_TRAINING_PROFILE_GLOBAL_TAIL_512:
            train_batches = ShuffledBatchSampler(
                len(train_ds),
                batch_size=int(config.batch_size),
                shuffle=True,
                seed=int(config.seed) + int(epoch) * 1009,
            )
            assert global_tail_targets is not None
            rank_batches = GlobalTailBatchSampler(
                train_ds.sample_index,
                global_tail_targets,
                batch_size=int(config.rank_batch_size),
                shuffle=True,
                seed=int(config.seed) + int(epoch) * 2027,
                prior_epoch_scores=prior_epoch_scores,
            )
            rank_batch_iterator: Iterator[list[int]] | None = iter(rank_batches)
        else:
            train_batches = DateGroupedBatchSampler(
                train_ds.sample_index,
                batch_size=int(config.batch_size),
                shuffle=True,
                seed=int(config.seed) + int(epoch) * 1009,
            )
            rank_batch_iterator = None
        total_batches = len(train_batches)
        loaded_train_batches = _iter_prefetched_batches(
            train_ds,
            train_batches,
            get_batch_kwargs={
                "include_ohlcva_path": bool(uses_ohlcva_path or uses_ohlcva_aux_path),
                "include_richer_path": uses_richer_path,
                "include_summary": not bool(uses_derived_path_value or uses_direct_value),
                "include_metadata": False,
            },
            prefetch_batches=int(config.prefetch_batches) if device.type == "cuda" else 0,
            pin_memory=bool(device.type == "cuda" and int(config.prefetch_batches) > 0),
        )
        for batch_indices, batch in loaded_train_batches:
            batch_start = time.perf_counter()
            y_tradable_path = (
                batch["y_tradable_path"].to(device, non_blocking=device.type == "cuda")
                if batch.get("y_tradable_path") is not None
                else None
            )
            x, y_path, y_ohlcva_path, y_richer_path, y_summary, date_idx, symbol_idx = _batch_to_device(batch, device)
            target_path = y_ohlcva_path if uses_ohlcva_path and y_ohlcva_path is not None else y_path
            del batch
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast(device_type=device.type, enabled=amp_enabled):
                out = model(x, symbol_idx=symbol_idx)
                loss, parts = _compute_loss(
                    out,
                    target_path,
                    y_summary,
                    date_idx,
                    y_ohlcva_path=y_ohlcva_path,
                    y_richer_path=y_richer_path,
                    value_index=train_ds.value_index,
                    path_weight=float(config.path_loss_weight),
                    path_loss_profile=str(config.path_loss_profile),
                    summary_weight=float(config.summary_loss_weight),
                    value_weight=float(config.value_loss_weight),
                    rank_weight=(
                        0.0
                        if rank_training_profile == RANK_TRAINING_PROFILE_GLOBAL_TAIL_512
                        else float(config.rank_loss_weight)
                    ),
                    richer_weight=float(config.richer_loss_weight),
                    rank_max_per_side=int(config.rank_max_per_side),
                    price_delta_weight=float(config.price_delta_loss_weight),
                    va_level_weight=float(config.va_level_loss_weight),
                    va_delta_weight=float(config.va_delta_loss_weight),
                    residual_weight=float(config.residual_score_weight),
                    residual_penalty_weight=float(config.residual_penalty_weight),
                    price_anchor=train_ds.price_anchor,
                    summary_loss_profile=str(config.summary_loss_profile),
                    direct_value_horizon=int(config.direct_value_horizon),
                    path_value_gradient_profile=path_value_gradient_profile,
                    rank_training_profile=RANK_TRAINING_PROFILE_LOCAL_CHUNK,
                    y_tradable_path=y_tradable_path,
                    return_tensor_parts=True,
                )
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            loss_totals_tensor.add_(
                torch.stack([parts[key].to(dtype=torch.float64) for key in loss_total_keys])
            )
            batch_count += 1
            sample_count += int(x.shape[0])
            if rank_batch_iterator is not None and batch_count % int(config.rank_interval) == 0:
                try:
                    rank_indices = next(rank_batch_iterator)
                except StopIteration:
                    rank_batch_iterator = None
                else:
                    rank_loss, rank_samples = _run_global_tail_rank_step(
                        model=model,
                        dataset=train_ds,
                        indices=rank_indices,
                        device=device,
                        optimizer=optimizer,
                        scaler=scaler,
                        amp_enabled=amp_enabled,
                        path_value_gradient_profile=path_value_gradient_profile,
                        rank_loss_weight=float(config.rank_loss_weight),
                        residual_score_weight=float(config.residual_score_weight),
                    )
                    rank_loss_total_tensor.add_(rank_loss.to(dtype=torch.float64))
                    rank_batch_count += 1
                    rank_sample_count += int(rank_samples)
            batch_seconds = float(time.perf_counter() - batch_start)
            if batch_count == 1 or batch_count % 25 == 0:
                _write_json(
                    progress_path,
                    {
                        "status": "training",
                        "epoch": int(epoch),
                        "batch": int(batch_count),
                        "total_batches": int(total_batches),
                        "sample_count": int(sample_count),
                        "rank_batch_count": int(rank_batch_count),
                        "last_batch_seconds": batch_seconds,
                        "updated_at": _now(),
                    },
                )
            del x, y_path, y_ohlcva_path, y_richer_path, target_path, y_summary, y_tradable_path, date_idx, symbol_idx, out, loss, parts
        # ``rank_interval`` controls interleaving, not date sampling.  Drain any
        # remaining slates so every included date contributes exactly one rank
        # step even for small debug subsets or larger path batch sizes.
        if rank_batch_iterator is not None:
            for rank_indices in rank_batch_iterator:
                rank_loss, rank_samples = _run_global_tail_rank_step(
                    model=model,
                    dataset=train_ds,
                    indices=rank_indices,
                    device=device,
                    optimizer=optimizer,
                    scaler=scaler,
                    amp_enabled=amp_enabled,
                    path_value_gradient_profile=path_value_gradient_profile,
                    rank_loss_weight=float(config.rank_loss_weight),
                    residual_score_weight=float(config.residual_score_weight),
                )
                rank_loss_total_tensor.add_(rank_loss.to(dtype=torch.float64))
                rank_batch_count += 1
                rank_sample_count += int(rank_samples)
        mining_seconds = 0.0
        mining_sample_count = 0
        if (
            rank_training_profile == RANK_TRAINING_PROFILE_GLOBAL_TAIL_512
            and epoch < int(config.epochs)
        ):
            mining_started_at = time.perf_counter()
            prior_epoch_scores = _mine_global_tail_scores(
                model=model,
                dataset=train_ds,
                device=device,
                batch_size=int(config.batch_size),
                amp_enabled=amp_enabled,
                path_value_gradient_profile=path_value_gradient_profile,
                residual_score_weight=float(config.residual_score_weight),
            )
            mining_seconds = float(time.perf_counter() - mining_started_at)
            mining_sample_count = int(np.isfinite(prior_epoch_scores).sum())
        loss_totals = {
            key: float(value)
            for key, value in zip(loss_total_keys, loss_totals_tensor.detach().cpu().tolist(), strict=True)
        }
        training_seconds = float(time.perf_counter() - epoch_started_at)
        train_row = {
            "epoch": int(epoch),
            "train_sample_count": int(sample_count),
            "rank_batch_count": int(rank_batch_count),
            "rank_sample_count": int(rank_sample_count),
            "global_tail_rank_loss": (
                float(rank_loss_total_tensor.detach().cpu().item()) / max(rank_batch_count, 1)
                if rank_batch_count
                else 0.0
            ),
            "optimizer_step_count": int(batch_count + rank_batch_count),
            "training_seconds": training_seconds,
            "path_samples_per_second": float(sample_count / max(training_seconds, 1.0e-9)),
            "optimizer_samples_per_second": float(
                (sample_count + rank_sample_count) / max(training_seconds, 1.0e-9)
            ),
            "hard_negative_mining_seconds": float(mining_seconds),
            "hard_negative_mining_sample_count": int(mining_sample_count),
            "hard_negative_mining_samples_per_second": float(
                mining_sample_count / max(mining_seconds, 1.0e-9)
            ) if mining_sample_count else 0.0,
            **{key: float(value / max(batch_count, 1)) for key, value in loss_totals.items()},
        }
        if evaluation_mode == EVALUATION_MODE_STANDARD:
            assert val_ds is not None
            _write_json(progress_path, {"status": "validating_epoch", "epoch": int(epoch), "updated_at": _now()})
            val_ic, val_topk, val_daily_topk, val_candidates, val_metrics = _predict_split(
                model=model,
                dataset=val_ds,
                device=device,
                output_dir=output_dir / f"epoch_{epoch:03d}",
                split="validation",
                batch_size=int(config.batch_size),
                amp_enabled=amp_enabled,
                top_k=config.top_k,
                write_predictions=False,
                write_path_predictions=False,
                direct_value_horizon=int(config.direct_value_horizon),
            )
            current_val_ic = float(val_metrics["rank_ic_mean"])
            improved = bool(np.isfinite(current_val_ic) and current_val_ic > best_val_ic + float(config.early_stopping_min_delta))
            train_row["validation_rank_ic_mean"] = current_val_ic
            train_row["is_best"] = bool(improved)
            if improved:
                best_val_ic = current_val_ic
                best_epoch = int(epoch)
                epochs_without_improvement = 0
                torch.save(
                    {
                        "model_state_dict": model.state_dict(),
                        "config": config.__dict__,
                        "input_dim": train_ds.input_dim,
                        "summary_columns": train_ds.path_summary_columns,
                        "feature_channels": manifest.get("feature_channels", {}),
                        "best_epoch": int(epoch),
                        "best_validation_rank_ic_mean": best_val_ic,
                        "checkpoint_policy": "best_validation_rank_ic",
                    },
                    best_path,
                )
            else:
                epochs_without_improvement += 1
            train_row["early_stopping_wait"] = int(epochs_without_improvement)
            history.append(train_row)
            progress_payload = {
                "status": "epoch_completed",
                "epoch": int(epoch),
                "validation_rank_ic_mean": float(val_metrics["rank_ic_mean"]),
                "best_validation_rank_ic_mean": float(best_val_ic),
                "best_epoch": int(best_epoch),
                "early_stopping_wait": int(epochs_without_improvement),
                "early_stopping_patience": int(config.early_stopping_patience),
                "updated_at": _now(),
            }
            del val_ic, val_topk, val_daily_topk, val_candidates
        elif evaluation_mode == EVALUATION_MODE_DEVELOPMENT:
            assert development_ds is not None and development_scoring_ds is not None
            epoch_dir = output_dir / f"epoch_{epoch:03d}"
            _write_json(
                progress_path,
                {"status": "evaluating_development_loss", "epoch": int(epoch), "updated_at": _now()},
            )
            development_loss = _evaluate_development_loss(
                model=model,
                dataset=development_ds,
                device=device,
                config=config,
                amp_enabled=amp_enabled,
                path_value_gradient_profile=path_value_gradient_profile,
                rank_training_profile=rank_training_profile,
            )
            _write_json(
                progress_path,
                {"status": "diagnosing_development_topk", "epoch": int(epoch), "updated_at": _now()},
            )
            (
                development_ic,
                development_topk,
                development_daily_topk,
                development_candidates,
                development_metrics,
            ) = _predict_split(
                model=model,
                dataset=development_scoring_ds,
                device=device,
                output_dir=epoch_dir,
                split="development",
                batch_size=int(config.batch_size),
                amp_enabled=amp_enabled,
                top_k=config.top_k,
                write_predictions=False,
                write_path_predictions=False,
                direct_value_horizon=int(config.direct_value_horizon),
            )
            if uses_derived_path_value:
                _validate_development_topk_execution_coverage(development_topk)
            diagnostics_path = epoch_dir / "development_epoch_diagnostics.json"
            _write_json(
                diagnostics_path,
                {
                    "epoch": int(epoch),
                    "checkpoint_selector": EARLY_STOPPING_METRIC_DEVELOPMENT_TOTAL_LOSS,
                    "topk_selects_checkpoint": False,
                    "loss": development_loss,
                    "candidate_metrics": development_metrics,
                    "topk": development_topk.to_dict("records"),
                },
            )
            current_development_loss = float(development_loss["loss"])
            improved = bool(
                current_development_loss
                < best_development_loss - float(config.early_stopping_min_delta)
            )
            for key in VALIDATION_LOSS_KEYS:
                history_key = "development_total_loss" if key == "loss" else f"development_{key}"
                train_row[history_key] = float(development_loss[key])
            train_row.update(
                {
                    "development_supervised_sample_count": int(development_loss["sample_count"]),
                    "development_candidate_count": int(len(development_scoring_ds)),
                    "development_loss_seconds": float(development_loss["seconds"]),
                    "development_rank_ic_mean": float(development_metrics["rank_ic_mean"]),
                    "development_diagnostics_json": str(diagnostics_path.resolve()),
                    "checkpoint_policy": "best_development_total_loss",
                    "is_best": bool(improved),
                }
            )
            if improved:
                best_development_loss = current_development_loss
                best_epoch = int(epoch)
                epochs_without_improvement = 0
                checkpoint_payload = {
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "scaler_state_dict": scaler.state_dict(),
                    "python_random_state": random.getstate(),
                    "numpy_random_state": np.random.get_state(),
                    "torch_random_state": torch.get_rng_state(),
                    "cuda_random_states": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
                    "config": config.__dict__,
                    "resolved_training_config": _resolved_training_config(
                        config, evaluation_mode=evaluation_mode
                    ),
                    "input_dim": train_ds.input_dim,
                    "summary_columns": train_ds.path_summary_columns,
                    "feature_channels": manifest.get("feature_channels", {}),
                    "fold_training_contract": fold_training_contract,
                    "research_contract": development_contract_binding,
                    "development_candidate_index": {
                        "path": str(development_scoring_ds.index_path),
                        "sha256": str(development_scoring_ds.sample_selection["index_sha256"]),
                        "row_count": int(len(development_scoring_ds)),
                    },
                    "best_epoch": int(epoch),
                    "best_development_total_loss": best_development_loss,
                    "development_loss_components": development_loss,
                    "checkpoint_policy": "best_development_total_loss",
                }
                torch.save(checkpoint_payload, best_path)
            else:
                epochs_without_improvement += 1
            train_row["early_stopping_wait"] = int(epochs_without_improvement)
            history.append(train_row)
            progress_payload = {
                "status": "epoch_completed",
                "epoch": int(epoch),
                "development_total_loss": current_development_loss,
                "best_development_total_loss": float(best_development_loss),
                "best_epoch": int(best_epoch),
                "early_stopping_wait": int(epochs_without_improvement),
                "early_stopping_patience": int(config.early_stopping_patience),
                "minimum_complete_epochs": int(config.min_complete_epochs),
                "topk_selects_checkpoint": False,
                "updated_at": _now(),
            }
            del development_ic, development_topk, development_daily_topk, development_candidates
        else:
            train_row["checkpoint_policy"] = "final_epoch"
            history.append(train_row)
            progress_payload = {
                "status": "epoch_completed",
                "epoch": int(epoch),
                "checkpoint_policy": "final_epoch",
                "oos_evaluated": False,
                "updated_at": _now(),
            }
        pd.DataFrame(history).to_csv(output_dir / "training_history_partial.csv", index=False, encoding="utf-8-sig")
        _write_json(progress_path, progress_payload)
        if device.type == "cuda":
            torch.cuda.empty_cache()
        gc.collect()
        _trim_working_set()
        standard_should_stop = bool(
            evaluation_mode == EVALUATION_MODE_STANDARD
            and int(config.early_stopping_patience) > 0
            and epochs_without_improvement >= int(config.early_stopping_patience)
        )
        development_should_stop = bool(
            evaluation_mode == EVALUATION_MODE_DEVELOPMENT
            and int(epoch) >= int(config.min_complete_epochs)
            and epochs_without_improvement >= int(config.early_stopping_patience)
        )
        if standard_should_stop or development_should_stop:
            stopping_metric_payload = (
                {"best_development_total_loss": float(best_development_loss)}
                if evaluation_mode == EVALUATION_MODE_DEVELOPMENT
                else {"best_validation_rank_ic_mean": float(best_val_ic)}
            )
            _write_json(
                progress_path,
                {
                    "status": "early_stopped",
                    "epoch": int(epoch),
                    "best_epoch": int(best_epoch),
                    **stopping_metric_payload,
                    "early_stopping_wait": int(epochs_without_improvement),
                    "updated_at": _now(),
                },
            )
            break
    if evaluation_mode == EVALUATION_MODE_FIXED_OOS:
        best_epoch = int(len(history))
        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "config": config.__dict__,
                "input_dim": train_ds.input_dim,
                "summary_columns": train_ds.path_summary_columns,
                "feature_channels": manifest.get("feature_channels", {}),
                "best_epoch": int(best_epoch),
                "checkpoint_policy": "final_epoch",
            },
            best_path,
        )
    elif best_path.exists():
        payload = torch.load(best_path, map_location=device, weights_only=False)
        model.load_state_dict(payload["model_state_dict"])
    elif evaluation_mode == EVALUATION_MODE_DEVELOPMENT:
        raise RuntimeError("development training completed without a finite best-loss checkpoint")
    _write_json(progress_path, {"status": "final_evaluation", "updated_at": _now()})
    evaluation_datasets: list[tuple[str, SequencePathPackDataset]]
    if evaluation_mode == EVALUATION_MODE_FIXED_OOS:
        assert oos_ds is not None
        evaluation_datasets = [("oos", oos_ds)]
    elif evaluation_mode == EVALUATION_MODE_DEVELOPMENT:
        assert development_scoring_ds is not None
        evaluation_datasets = [("development", development_scoring_ds)]
    else:
        assert val_ds is not None and test_ds is not None
        evaluation_datasets = [("validation", val_ds), ("test", test_ds)]
    metric_rows: list[dict[str, Any]] = []
    topk_frames: list[pd.DataFrame] = []
    daily_ic_frames: list[pd.DataFrame] = []
    daily_topk_frames: list[pd.DataFrame] = []
    candidate_frames: list[pd.DataFrame] = []
    for split_name, evaluation_ds in evaluation_datasets:
        split_ic, split_topk, split_daily_topk, split_candidates, split_metric = _predict_split(
            model=model,
            dataset=evaluation_ds,
            device=device,
            output_dir=output_dir,
            split=split_name,
            batch_size=int(config.batch_size),
            amp_enabled=amp_enabled,
            top_k=config.top_k,
            write_predictions=str(config.prediction_mode) != "none",
            write_path_predictions=str(config.prediction_mode) == "full",
            direct_value_horizon=int(config.direct_value_horizon),
        )
        metric_rows.append(split_metric)
        topk_frames.append(split_topk.assign(split=split_name))
        daily_ic_frames.append(split_ic.assign(split=split_name))
        daily_topk_frames.append(split_daily_topk.assign(split=split_name))
        candidate_frames.append(split_candidates.assign(split=split_name))
        if evaluation_mode == EVALUATION_MODE_DEVELOPMENT and uses_derived_path_value:
            _validate_development_topk_execution_coverage(split_topk)
    split_metrics = pd.DataFrame(metric_rows)
    topk = pd.concat(topk_frames, ignore_index=True)
    daily_ic = pd.concat(daily_ic_frames, ignore_index=True)
    daily_topk = pd.concat(daily_topk_frames, ignore_index=True)
    topk_candidates = pd.concat(candidate_frames, ignore_index=True)
    split_metrics_path = output_dir / "split_metrics.csv"
    topk_path = output_dir / "topk_metrics.csv"
    daily_ic_path = output_dir / "daily_rank_ic.csv"
    daily_topk_path = output_dir / "daily_topk_metrics.csv"
    topk_candidates_path = output_dir / "topk_candidates.parquet"
    history_path = output_dir / "training_history.csv"
    split_metrics.to_csv(split_metrics_path, index=False, encoding="utf-8-sig")
    topk.to_csv(topk_path, index=False, encoding="utf-8-sig")
    daily_ic.to_csv(daily_ic_path, index=False, encoding="utf-8-sig")
    daily_topk.to_csv(daily_topk_path, index=False, encoding="utf-8-sig")
    topk_candidates.to_parquet(topk_candidates_path, index=False)
    pd.DataFrame(history).to_csv(history_path, index=False, encoding="utf-8-sig")
    uses_direct_value = bool(getattr(model, "uses_direct_value", False))
    baseline_forward_days = int(config.direct_value_horizon) if uses_direct_value else int(train_ds.forward_days)
    baseline_summary = _find_latest_baseline_summary(baseline_forward_days)
    active_path_dim = 6 if bool(getattr(model, "uses_ohlcva_path", False)) else 4
    active_value_column = (
        value_column_for_path(int(config.direct_value_horizon), path_dim=4)
        if uses_direct_value
        else value_column_for_path(train_ds.forward_days, path_dim=active_path_dim)
        if bool(getattr(model, "uses_derived_path_value", False))
        else train_ds.value_column
    )
    active_summary_columns = (
        derived_path_summary_columns(int(config.direct_value_horizon), path_dim=4)
        if uses_direct_value
        else derived_path_summary_columns(train_ds.forward_days, path_dim=active_path_dim)
        if bool(getattr(model, "uses_derived_path_value", False))
        else list(train_ds.path_summary_columns)
    )
    walkforward_contract = dict(
        manifest.get(
            "development_walkforward"
            if evaluation_mode == EVALUATION_MODE_DEVELOPMENT
            else "purged_walkforward",
            {},
        )
        or {}
    )
    evaluation_splits = [name for name, _ in evaluation_datasets]
    checkpoint_policy = (
        "final_epoch"
        if evaluation_mode == EVALUATION_MODE_FIXED_OOS
        else "best_development_total_loss"
        if evaluation_mode == EVALUATION_MODE_DEVELOPMENT
        else "best_validation_rank_ic"
    )
    prediction_outputs = {
        f"{split_name}_predictions_csv": (
            str((output_dir / "predictions" / f"{split_name}_predictions.csv").resolve())
            if str(config.prediction_mode) != "none"
            else ""
        )
        for split_name in evaluation_splits
    }
    summary = {
        "artifact_type": "qdp_v2_sequence_path_training",
        "generated_at": _now(),
        "run_tag": str(config.run_tag),
        "seed": int(config.seed),
        "top_k": [int(item) for item in config.top_k],
        "pack_manifest": str(Path(config.pack_manifest).resolve()),
        "output_dir": str(output_dir.resolve()),
        "lookback_days": int(train_ds.lookback_days),
        "forward_days": int(train_ds.forward_days),
        "price_anchor": str(train_ds.price_anchor),
        "value_column": str(active_value_column),
        "path_value_gradient_profile": path_value_gradient_profile,
        "rank_training_profile": rank_training_profile,
        "path_summary_columns": list(active_summary_columns),
        "direct_value_horizon": int(config.direct_value_horizon) if uses_direct_value else 0,
        "device": str(device),
        "amp_enabled": bool(amp_enabled),
        "prediction_mode": str(config.prediction_mode),
        "evaluation_mode": evaluation_mode,
        "evaluation_splits": evaluation_splits,
        "evaluation_split": evaluation_splits[0] if len(evaluation_splits) == 1 else "",
        "checkpoint_policy": checkpoint_policy,
        "resolved_training_config": _resolved_training_config(config, evaluation_mode=evaluation_mode),
        "fold_year": int(
            walkforward_contract.get(
                "development_year", walkforward_contract.get("oos_year", 0)
            )
            or 0
        ),
        "development_year": int(walkforward_contract.get("development_year", 0) or 0),
        "train_label_end_before": str(
            walkforward_contract.get(
                "development_start", walkforward_contract.get("oos_start", "")
            )
            or ""
        ),
        "normalization_cutoff": str(walkforward_contract.get("normalization_cutoff_exclusive", "") or ""),
        "epochs": int(config.epochs),
        "completed_epochs": int(len(history)),
        "best_epoch": int(best_epoch),
        "batch_size": int(config.batch_size),
        "max_samples_per_split": int(config.max_samples_per_split),
        "sample_selection": {
            "train": dict(train_ds.sample_selection),
            **({"oos": dict(oos_ds.sample_selection)} if oos_ds is not None else {}),
            **({"validation": dict(val_ds.sample_selection)} if val_ds is not None else {}),
            **({"test": dict(test_ds.sample_selection)} if test_ds is not None else {}),
            **(
                {"development_supervised": dict(development_ds.sample_selection)}
                if development_ds is not None
                else {}
            ),
            **(
                {"development_candidates": dict(development_scoring_ds.sample_selection)}
                if development_scoring_ds is not None
                else {}
            ),
        },
        "input_channel_profile": str(train_ds.input_channel_profile),
        "input_channels": list(train_ds.channel_order),
        "input_mask_features": list(train_ds.input_mask_features),
        "model": {
            "type": f"SequencePathModel_{str(config.model_type)}",
            "input_dim": int(train_ds.input_dim),
            "hidden_dim": int(config.hidden_dim),
            "layers": int(config.layers),
            "dropout": float(config.dropout),
            "uses_derived_path_value": bool(getattr(model, "uses_derived_path_value", False)),
            "uses_direct_value": bool(getattr(model, "uses_direct_value", False)),
            "uses_symbol_embedding": bool(getattr(model, "uses_symbol_embedding", False)),
            "uses_residual_score": bool(getattr(model, "uses_residual_score", False)),
            "uses_richer_path": bool(getattr(model, "uses_richer_path", False)),
            "uses_ohlcva_path": bool(getattr(model, "uses_ohlcva_path", False)),
            "uses_ohlcva_aux_path": bool(getattr(model, "uses_ohlcva_aux_path", False)),
            "path_dim": int(getattr(model, "path_dim", 4)),
            "richer_path_dim": int(getattr(model, "richer_path_dim", 4)),
            "richer_path_fields": list(train_ds.richer_path_fields) if bool(getattr(model, "uses_richer_path", False)) else [],
            "symbol_embedding_dim": int(config.symbol_embedding_dim) if bool(getattr(model, "uses_symbol_embedding", False)) else 0,
            "residual_score_weight": float(config.residual_score_weight) if bool(getattr(model, "uses_residual_score", False)) else 0.0,
        },
        "loss_weights": {
            "path": float(config.path_loss_weight),
            "path_profile": str(config.path_loss_profile),
            "summary": float(config.summary_loss_weight),
            "richer": float(config.richer_loss_weight) if bool(getattr(model, "uses_richer_path", False)) else 0.0,
            "price_delta": float(config.price_delta_loss_weight),
            "value": float(config.value_loss_weight),
            "path_value_gradient_profile": path_value_gradient_profile,
            "rank": float(config.rank_loss_weight),
            "rank_training_profile": rank_training_profile,
            "rank_batch_size": int(config.rank_batch_size),
            "rank_interval": int(config.rank_interval),
            "rank_max_per_side": int(config.rank_max_per_side),
            "summary_profile": str(config.summary_loss_profile),
            "va_level": float(config.va_level_loss_weight),
            "va_delta": float(config.va_delta_loss_weight),
        },
        "ranking_contract": {
            "profile": rank_training_profile,
            "path_and_rank_batches_separate": bool(
                rank_training_profile == RANK_TRAINING_PROFILE_GLOBAL_TAIL_512
            ),
            "date_weighting": "equal" if rank_training_profile == RANK_TRAINING_PROFILE_GLOBAL_TAIL_512 else "batch_local",
            "tail_group_counts": (
                dict(GLOBAL_TAIL_GROUP_COUNTS)
                if rank_training_profile == RANK_TRAINING_PROFILE_GLOBAL_TAIL_512
                else {}
            ),
            "hard_negative_source": (
                "prior_epoch_frozen_eval_full_train_scores"
                if rank_training_profile == RANK_TRAINING_PROFILE_GLOBAL_TAIL_512
                else "none"
            ),
        },
        "early_stopping": {
            "metric": (
                EARLY_STOPPING_METRIC_DEVELOPMENT_TOTAL_LOSS
                if evaluation_mode == EVALUATION_MODE_DEVELOPMENT
                else EARLY_STOPPING_METRIC_VALIDATION_RANK_IC
                if evaluation_mode == EVALUATION_MODE_STANDARD
                else ""
            ),
            "mode": (
                EARLY_STOPPING_MODE_MIN
                if evaluation_mode == EVALUATION_MODE_DEVELOPMENT
                else EARLY_STOPPING_MODE_MAX
                if evaluation_mode == EVALUATION_MODE_STANDARD
                else ""
            ),
            "patience": int(config.early_stopping_patience),
            "min_delta": float(config.early_stopping_min_delta),
            "minimum_complete_epochs": int(config.min_complete_epochs),
            "restore_best_checkpoint": bool(evaluation_mode != EVALUATION_MODE_FIXED_OOS),
            "topk_selects_checkpoint": False if evaluation_mode == EVALUATION_MODE_DEVELOPMENT else None,
            "best_value": (
                float(best_development_loss)
                if evaluation_mode == EVALUATION_MODE_DEVELOPMENT
                else float(best_val_ic)
                if evaluation_mode == EVALUATION_MODE_STANDARD
                else None
            ),
            "stopped_early": bool(
                evaluation_mode in {EVALUATION_MODE_STANDARD, EVALUATION_MODE_DEVELOPMENT}
                and int(config.early_stopping_patience) > 0
                and len(history) < int(config.epochs)
            ),
        },
        "best_checkpoint": str(best_path.resolve()),
        "best_checkpoint_sha256": _file_sha256(best_path),
        "history": history,
        "split_metrics": split_metrics.to_dict("records"),
        "outputs": {
            "split_metrics_csv": str(split_metrics_path.resolve()),
            "topk_metrics_csv": str(topk_path.resolve()),
            "daily_rank_ic_csv": str(daily_ic_path.resolve()),
            "daily_topk_metrics_csv": str(daily_topk_path.resolve()),
            "topk_candidates_parquet": str(topk_candidates_path.resolve()),
            "training_history_csv": str(history_path.resolve()),
            **prediction_outputs,
        },
        "baseline_feature_summary": baseline_summary.get("output_dir", ""),
    }
    if evaluation_mode == EVALUATION_MODE_DEVELOPMENT:
        assert development_scoring_ds is not None
        summary["research_contract"] = development_contract_binding
        summary["development_candidate_index"] = {
            "path": str(development_scoring_ds.index_path),
            "sha256": str(development_scoring_ds.sample_selection["index_sha256"]),
            "row_count": int(len(development_scoring_ds)),
            "date_count": int(development_scoring_ds.sample_index["date_idx"].nunique()),
            "policy": "all_candidates",
        }
        summary["candidate_index_path"] = str(development_scoring_ds.index_path)
        summary["candidate_index_sha256"] = str(
            development_scoring_ds.sample_selection["index_sha256"]
        )
        summary["development_fold_training_contract"] = fold_training_contract
        summary["development_execution_evaluation"] = {
            "entry": "raw_next_open",
            "earliest_exit_day": 2,
            "exit_retry_days": int(development_scoring_ds.execution_tail_days),
            "exit_prices": "raw_close",
            "terminal_recovery_fraction": float(
                development_scoring_ds.terminal_recovery_fraction
            ),
            "realized_plan_return_semantics": "gross_raw_execution_return",
            "realized_plan_value_cost_semantics": "path_value_v2_fixed_transaction_cost_surrogate",
            "full_account_net_costs_required_for_deployment": True,
        }
    if fold_training_contract is not None:
        summary["fold_training_contract"] = fold_training_contract
    if bool(getattr(model, "uses_residual_score", False)):
        summary["loss_weights"]["residual_score_weight"] = float(config.residual_score_weight)
        summary["loss_weights"]["residual_penalty"] = float(config.residual_penalty_weight)
    report_path = _write_report(output_dir, summary=summary, split_metrics=split_metrics, topk=topk, baseline_summary=baseline_summary)
    summary["outputs"]["report_md"] = report_path
    summary_path = output_dir / "sequence_path_training_summary.json"
    _write_json(summary_path, summary)
    _write_json(progress_path, {"status": "completed", "summary_json": str(summary_path.resolve()), "updated_at": _now()})
    return summary


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train QDP v2 sequence path models.")
    sub = parser.add_subparsers(dest="command", required=True)
    train = sub.add_parser("train")
    train.add_argument("--pack-manifest", type=Path, default=None)
    train.add_argument("--store-view", type=Path, default=None, help="Alias for a lightweight research_store view manifest.")
    train.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    train.add_argument("--run-tag", default="qdp_v2_sequence_path_gru_path_value")
    train.add_argument("--epochs", type=int, default=10)
    train.add_argument("--batch-size", type=int, default=512)
    train.add_argument(
        "--model-type",
        default="",
        choices=(
            "gru_last",
            "gru_attention",
            "gru_path_value",
            "gru_path_value_symbol",
            "gru_path_value_residual",
            "gru_ohlcva_path_value",
            "gru_ohlcva_aux_path_value",
            "gru_richer_path_value",
            "gru_richer_path_value_symbol",
            "gru_direct_value",
        ),
        help=argparse.SUPPRESS,
    )
    train.add_argument(
        "--experimental-model-type",
        default="",
        choices=("gru_last", "gru_attention", "gru_path_value_residual"),
        help=argparse.SUPPRESS,
    )
    train.add_argument("--with-symbol", action="store_true", help="Use symbol identity embedding with the path-value model.")
    train.add_argument("--richer-path", action="store_true", help="Predict price, volume, intraday, and limit-structure future paths.")
    train.add_argument("--hidden-dim", type=int, default=128)
    train.add_argument("--layers", type=int, default=2)
    train.add_argument("--dropout", type=float, default=0.10)
    train.add_argument("--symbol-embedding-dim", type=int, default=16)
    train.add_argument("--learning-rate", type=float, default=1.0e-3)
    train.add_argument("--weight-decay", type=float, default=1.0e-4)
    train.add_argument("--path-loss-weight", type=float, default=0.40)
    train.add_argument(
        "--path-loss-profile",
        default=PATH_LOSS_PROFILE_DEFAULT,
        choices=PATH_LOSS_PROFILES,
        help="Path reconstruction loss profile: default tensor mean or equal-weight OHLCVA field loss.",
    )
    train.add_argument("--summary-loss-weight", type=float, default=0.20)
    train.add_argument("--richer-loss-weight", type=float, default=0.10)
    train.add_argument("--price-delta-loss-weight", type=float, default=0.0)
    train.add_argument("--va-level-loss-weight", type=float, default=0.0)
    train.add_argument("--va-delta-loss-weight", type=float, default=0.0)
    train.add_argument("--value-loss-weight", type=float, default=0.25)
    train.add_argument("--rank-loss-weight", type=float, default=0.15)
    train.add_argument("--direct-value-horizon", type=int, default=0)
    train.add_argument("--residual-score-weight", type=float, default=0.25, help=argparse.SUPPRESS)
    train.add_argument("--residual-penalty-weight", type=float, default=0.01, help=argparse.SUPPRESS)
    train.add_argument(
        "--summary-loss-profile",
        default=SUMMARY_LOSS_PROFILE_BASE,
        choices=SUMMARY_LOSS_PROFILES,
        help="Summary loss profile: base full-horizon constraints or multi-horizon OHLC-derived constraints.",
    )
    train.add_argument(
        "--input-channel-profile",
        default=INPUT_CHANNEL_PROFILE_ALL,
        choices=INPUT_CHANNEL_PROFILES,
        help="Input channel profile: all channels or daily_only without minute-derived intraday/limit channels.",
    )
    train.add_argument("--rank-max-per-side", type=int, default=64)
    train.add_argument(
        "--path-value-gradient-profile",
        default=PATH_VALUE_GRADIENT_PROFILE_SMOOTH,
        choices=PATH_VALUE_GRADIENT_PROFILES,
        help="Value surrogate: legacy smooth forward value or hard-max forward with straight-through smooth gradients.",
    )
    train.add_argument(
        "--rank-training-profile",
        default=RANK_TRAINING_PROFILE_LOCAL_CHUNK,
        choices=RANK_TRAINING_PROFILES,
        help="Legacy random date chunks or separated full-day global-tail ranking slates.",
    )
    train.add_argument("--rank-batch-size", type=int, default=512)
    train.add_argument("--rank-interval", type=int, default=4, help="Run one separated rank step after this many path steps.")
    train.add_argument("--prefetch-batches", type=int, default=1, choices=(0, 1))
    train.add_argument("--device", default="auto", choices=("auto", "cpu", "cuda"))
    train.add_argument("--amp", dest="amp", action="store_true", default=True)
    train.add_argument("--no-amp", dest="amp", action="store_false")
    train.add_argument("--seed", type=int, default=DEFAULT_SEED)
    train.add_argument("--top-k", default="5,10,20,50,100")
    train.add_argument(
        "--max-samples-per-split",
        type=int,
        default=0,
        help=(
            "Training-only screening cap. Keeps complete dates spread across the full train history; "
            "validation/test/OOS evaluation remains full-universe."
        ),
    )
    train.add_argument("--prediction-mode", default="compact", choices=("full", "compact", "none"))
    train.add_argument(
        "--evaluation-mode",
        default=EVALUATION_MODE_STANDARD,
        choices=EVALUATION_MODES,
        help=(
            "standard selects by validation rank IC; fixed_oos evaluates OOS only after its final epoch; "
            "development selects by full supervised development loss and diagnoses TopK on all candidates."
        ),
    )
    train.add_argument(
        "--allow-large-predictions",
        action="store_true",
        help="Allow prediction-mode=full to write wide per-day path prediction CSVs.",
    )
    train.add_argument("--early-stopping-patience", type=int, default=0)
    train.add_argument("--early-stopping-min-delta", type=float, default=0.0)
    train.add_argument("--early-stopping-metric", default="")
    train.add_argument("--early-stopping-mode", default="", choices=("", *EARLY_STOPPING_MODES))
    train.add_argument("--min-complete-epochs", type=int, default=1)
    train.add_argument("--development-contract", type=Path, default=None)
    train.add_argument("--json", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    if str(args.prediction_mode) == "full" and not bool(args.allow_large_predictions):
        raise SystemExit("--prediction-mode full writes large path-level prediction CSVs; add --allow-large-predictions to opt in.")
    manifest_path = Path(args.store_view or args.pack_manifest) if (args.store_view or args.pack_manifest) else None
    if manifest_path is None:
        raise SystemExit("train requires --pack-manifest or --store-view")
    model_type = str(args.experimental_model_type or args.model_type or "").strip()
    if not model_type:
        if bool(args.richer_path):
            model_type = "gru_richer_path_value_symbol" if bool(args.with_symbol) else "gru_richer_path_value"
        else:
            model_type = "gru_path_value_symbol" if bool(args.with_symbol) else "gru_path_value"
    elif bool(args.with_symbol):
        if model_type not in {"gru_path_value", "gru_path_value_symbol", "gru_richer_path_value", "gru_richer_path_value_symbol"}:
            raise SystemExit("--with-symbol can only be combined with the path-value model")
        model_type = "gru_richer_path_value_symbol" if model_type.startswith("gru_richer_") else "gru_path_value_symbol"
    if model_type in DIRECT_VALUE_MODEL_TYPES and int(args.direct_value_horizon) <= 0:
        raise SystemExit("--direct-value-horizon must be positive when model-type=gru_direct_value")
    cfg = TrainConfig(
        pack_manifest=manifest_path,
        output_root=Path(args.output_root),
        run_tag=str(args.run_tag),
        epochs=int(args.epochs),
        batch_size=int(args.batch_size),
        model_type=model_type,
        hidden_dim=int(args.hidden_dim),
        layers=int(args.layers),
        dropout=float(args.dropout),
        symbol_embedding_dim=int(args.symbol_embedding_dim),
        learning_rate=float(args.learning_rate),
        weight_decay=float(args.weight_decay),
        path_loss_weight=float(args.path_loss_weight),
        path_loss_profile=str(args.path_loss_profile),
        summary_loss_weight=float(args.summary_loss_weight),
        richer_loss_weight=float(args.richer_loss_weight),
        price_delta_loss_weight=float(args.price_delta_loss_weight),
        va_level_loss_weight=float(args.va_level_loss_weight),
        va_delta_loss_weight=float(args.va_delta_loss_weight),
        value_loss_weight=float(args.value_loss_weight),
        rank_loss_weight=float(args.rank_loss_weight),
        residual_score_weight=float(args.residual_score_weight),
        residual_penalty_weight=float(args.residual_penalty_weight),
        summary_loss_profile=str(args.summary_loss_profile),
        input_channel_profile=str(args.input_channel_profile),
        direct_value_horizon=int(args.direct_value_horizon),
        rank_max_per_side=int(args.rank_max_per_side),
        device=str(args.device),
        amp=bool(args.amp),
        seed=int(args.seed),
        top_k=_parse_int_list(args.top_k, default=DEFAULT_TOP_K),
        max_samples_per_split=int(args.max_samples_per_split),
        prediction_mode=str(args.prediction_mode),
        early_stopping_patience=int(args.early_stopping_patience),
        early_stopping_min_delta=float(args.early_stopping_min_delta),
        evaluation_mode=str(args.evaluation_mode),
        path_value_gradient_profile=str(args.path_value_gradient_profile),
        rank_training_profile=str(args.rank_training_profile),
        rank_batch_size=int(args.rank_batch_size),
        rank_interval=int(args.rank_interval),
        prefetch_batches=int(args.prefetch_batches),
        early_stopping_metric=str(args.early_stopping_metric),
        early_stopping_mode=str(args.early_stopping_mode),
        min_complete_epochs=int(args.min_complete_epochs),
        development_contract=(Path(args.development_contract) if args.development_contract else None),
    )
    result = train_sequence_path_model(cfg)
    print(json.dumps(result, ensure_ascii=False, indent=2, default=_json_default) if bool(args.json) else result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
